# AnyTile 图像切分放大插件 (bofengttp)
# 节点组:
#   AnyTileSplit       - 图片等比放大到目标分辨率后, 切分为带重叠的 tile 批次 (image + latent + layout)
#   AnyTileBatchSampler - 把 tile latent 按"每批N个"分批采样并VAE解码, 避免大批次OOM
#   AnyTileMerge       - 把采样完成的 tile 批次按像素层线性羽化拼接回完整大图
import math
import logging

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("anytile")

# 自定义中间类型: 切分布局信息, 仅在 AnyTileSplit -> AnyTileMerge 之间传递
ANYTILE_LAYOUT = "ANYTILE_LAYOUT"

_PIL_RESAMPLE = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


# ---------------------------------------------------------------- 基础工具
def _snap_up(v, m):
    """向上取整对齐到 m 的倍数 (至少 m)"""
    return max(int(m), int(math.ceil(int(v) / int(m))) * int(m))


def _snap_down(v, m):
    """向下取整对齐到 m 的倍数 (至少 m)"""
    return max(int(m), (int(v) // int(m)) * int(m))


def _get_downscale_ratio(vae):
    """VAE 空间下采样倍率。

    未接 VAE 时按 16 对齐: 16 是 8 的倍数, 同时满足 /8 VAE (SD/SDXL/Flux) 与
    /16 VAE (Qwen-2.1 等) —— tile/目标尺寸对齐到 16 后, 经任何主流 VAE 编解码
    尺寸都不会漂移; 接了 VAE 则按其实际倍率。"""
    if vae is None:
        return 16
    dr = getattr(vae, "downscale_ratio", 8)
    if not isinstance(dr, int) or dr <= 0:
        dr = 8
    return dr


def _resize_image(image, target_w, target_h, interp="lanczos"):
    """(B,H,W,C) float 0-1 -> PIL 缩放到目标尺寸 -> (B,h,w,C) float 0-1"""
    resample = _PIL_RESAMPLE.get(interp, Image.LANCZOS)
    outs = []
    for img in image:
        arr = (img.clamp(0.0, 1.0).cpu().contiguous().numpy() * 255.0).round().astype("uint8")
        pil = Image.fromarray(arr)
        pil = pil.resize((int(target_w), int(target_h)), resample)
        outs.append(torch.from_numpy(np.array(pil, copy=True)).to(torch.float32) / 255.0)
    return torch.stack(outs, dim=0)


def _slice_tensor_maybe(t, start, length, total):
    """batch==total的张量切片, 其余(batch=1广播等)保持原样"""
    if isinstance(t, torch.Tensor) and int(t.shape[0]) == total:
        return t[start : start + length]
    return t


def _slice_cond_obj(v, start, length, total):
    """切片 model_conds 里的 COND 对象 (CONDRegular/CONDList 等), 其他类型原样保留"""
    cond = getattr(v, "cond", None)
    if isinstance(cond, torch.Tensor):
        if int(cond.shape[0]) == total:
            return v._copy_with(cond[start : start + length])
        return v
    if isinstance(cond, list):  # CONDList: 逐个切片
        return v._copy_with([_slice_tensor_maybe(t, start, length, total) for t in cond])
    return v


def _slice_conditioning(cond_list, start, length, total):
    """分批采样专用: 把batch=total的conditioning切出 [start:start+length]。

    ComfyUI 对 cond批次>latent批次 只会取前几条(repeat_to_batch_size的narrow),
    导致后续批次错用第1批的逐tile条件/参考图, 所以每批采样前必须先切好。
    支持两种格式:
    - 原始conditioning列表: [[tensor, opts], ...]
    - guider内转换后格式: [{cross_attn: tensor, reference_latents: [...], model_conds: {...}, ...}, ...]
    batch不等于total的字段(如batch=1的广播条件)保持原样。
    """
    if cond_list is None or len(cond_list) == 0:
        return cond_list

    if isinstance(cond_list[0], dict):
        # guider内 convert_cond 之后的格式: 条目是字典
        out = []
        for entry in cond_list:
            e = dict(entry)
            for k, v in entry.items():
                if k in ("uuid", "model_conds"):
                    continue
                if k == "reference_latents" and isinstance(v, list):
                    e[k] = [_slice_tensor_maybe(t, start, length, total) for t in v]
                elif isinstance(v, torch.Tensor):
                    e[k] = _slice_tensor_maybe(v, start, length, total)
                else:
                    e[k] = v
            mc = entry.get("model_conds")
            if isinstance(mc, dict):
                e["model_conds"] = {mk: _slice_cond_obj(mv, start, length, total) for mk, mv in mc.items()}
            out.append(e)
        return out

    # 原始格式: [tensor, opts] 条目
    if int(cond_list[0][0].shape[0]) != total:
        return cond_list
    out = []
    for tensor, opts in cond_list:
        new_opts = {}
        for k, v in opts.items():
            if k == "reference_latents" and isinstance(v, list):
                new_opts[k] = [_slice_tensor_maybe(t, start, length, total) for t in v]
            elif isinstance(v, torch.Tensor):
                new_opts[k] = _slice_tensor_maybe(v, start, length, total)
            else:
                new_opts[k] = v
        out.append([tensor[start : start + length], new_opts])
    return out


def _upscale_with_model(upscale_model, image):
    """ESRGAN 类模型放大, 逻辑与核心节点 ImageUpscaleWithModel 一致 (tiled + OOM 自动减半)"""
    import comfy.model_management
    import comfy.utils

    device = upscale_model.patcher.load_device
    if image.shape[-1] == 4:
        image = image[..., :3]

    memory_required = (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    comfy.model_management.load_models_gpu(
        [upscale_model.patcher], memory_required=memory_required, force_full_load=True
    )

    in_img = image.movedim(-1, -3).to(device)
    tile = 512
    overlap = 32
    output_device = comfy.model_management.intermediate_device()

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
            )
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a.float()),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
                output_device=output_device,
            )
            oom = False
        except comfy.model_management.OOM_EXCEPTION as e:
            comfy.model_management.raise_non_oom(e)
            tile //= 2
            if tile < 128:
                raise e
    return torch.clamp(s.movedim(-3, -1), min=0.0, max=1.0).to(output_device)


# ---------------------------------------------------------------- 切分布局计算
def _compute_positions(total, tile, count):
    """沿一条轴计算 tile 起始位置: 均匀分摊步长, 最后一块自然贴边 (tile 尺寸不变)"""
    if count <= 1:
        return [0]
    step = (total - tile) / (count - 1)
    pos = [int(round(i * step)) for i in range(count)]
    pos[-1] = total - tile  # 消除舍入误差, 保证最后一块严格贴边
    return pos


def _compute_layout(target_w, target_h, tile_size, overlap, dr):
    """计算切分布局 (方形tile, 档位尺寸)。

    核心策略——块数只由"无重叠覆盖"决定, 重叠与尾块浪费都通过缩小画布吸收,
    且全程严格保持宽高比 (宁可不省块, 也不不等比变形):
    - 每轴块数 n = ceil(L / tile), 与overlap无关 (避免"1px重叠多出一整块tile"的浪费)
    - n块tile带重叠的覆盖能力 cap = tile*n - ov*(n-1), 画布缩放到 min(L, cap)
      (overlap 吸收: 拼接后由 AnyTileMerge 自动缩回 target 尺寸)
    - 等比网格约束: 搜索 tile 整数倍网格 (a×tile, b×tile) 替代原始目标, 三重门槛:
      ①块数 < 原网格 (确实省块) ②吸收重叠后的画布比例偏差 <=10% (等比不变形)
      ③候选目标面积 >=90% 原目标 (信息损失可控); 取面积最大者改写目标分辨率,
      两轴统一缩放, 拼接后由 merge 拉回原始目标恢复精确比例;
      无合格候选则不约束 (保比例, 尾块浪费保留)
      例: 7296x1072/tile1024 -> 原网格 8x2=16块, 约束到 8x1=8块 (画布7296x1024)
    - 图某轴 <= 单块tile: 该轴只切1块, tile尺寸取图尺寸 (整图一块)
    - 重叠向上对齐 VAE 下采样倍率 (dr): tile与步长均为dr倍数, latent切分无舍入
    - 重叠像素钳制 (不超过 tile 的一半); 画布 <= 覆盖能力, 实际重叠恒 >= ov
    """
    tw = _snap_down(tile_size, dr)
    th = tw
    ov = max(0, int(overlap))
    ov = _snap_up(ov, dr) if ov > 0 else 0
    ovx = min(ov, tw // 2)
    ovy = min(ov, th // 2)
    orig_w, orig_h = int(target_w), int(target_h)  # 原始目标 (merge 拉回宽高比用)

    def _canvas(total, n, tile, ov_a):
        # n块tile吸收重叠后的画布尺寸 (n<=1 单块: 画布=target)
        return total if n <= 1 else min(total, tile * n - ov_a * (n - 1))

    # 等比网格搜索: 两轴统一决策, 比例偏差用吸收重叠后的画布度量 (画布才是被采样的内容)
    best = None
    a0 = math.ceil(orig_w / tw)
    b0 = math.ceil(orig_h / th)
    if a0 * b0 > 1:
        for a in range(1, a0 + 1):
            for b in range(1, b0 + 1):
                if a * b >= a0 * b0:
                    continue  # 门槛①: 必须省块
                w1, h1 = a * tw, b * th          # 候选目标分辨率 (tile 整数倍, 两轴同网格)
                cw = _canvas(w1, a, tw, ovx)
                ch = _canvas(h1, b, th, ovy)
                if abs(cw / ch - orig_w / orig_h) / (orig_w / orig_h) > 0.10:
                    continue  # 门槛②: 画布比例偏差 <=10%, 不等比变形一票否决
                if w1 * h1 < 0.90 * orig_w * orig_h:
                    continue  # 门槛③: 分辨率损失 <=10%
                key = (w1 * h1, -(a * b))        # 面积最大优先, 同面积块数少优先
                if best is None or key > best[0]:
                    best = (key, a, b, w1, h1)

    if best is not None:
        _, cols, rows, target_w, target_h = best
        grid_constrained = True
        canvas_w = _canvas(target_w, cols, tw, ovx)
        canvas_h = _canvas(target_h, rows, th, ovy)
    else:
        grid_constrained = False

        def _axis(total, tile, ov_a):
            if total <= tile:
                return 1, total, total  # 单块: tile取图尺寸 (整图一块)
            n = math.ceil(total / tile)
            cap = tile * n - ov_a * (n - 1)
            return n, tile, min(total, cap)

        cols, tw, canvas_w = _axis(orig_w, tw, ovx)
        rows, th, canvas_h = _axis(orig_h, th, ovy)

    xs = _compute_positions(canvas_w, tw, cols)
    ys = _compute_positions(canvas_h, th, rows)
    positions = [[x, y] for y in ys for x in xs]

    return {
        "type": "anytile_layout",
        "version": 2,
        "width": int(canvas_w),
        "height": int(canvas_h),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "tile_width": int(tw),
        "tile_height": int(th),
        "cols": int(cols),
        "rows": int(rows),
        "count": int(cols * rows),
        "overlap": int(ovx),
        "grid_constrained": grid_constrained,  # 等比网格约束触发 (target 已改写为 tile 整数倍网格)
        "original_width": orig_w,   # 约束前原始目标: merge 拼接后拉回, 恢复宽高比
        "original_height": orig_h,
        "positions": positions,  # 行优先 [[x, y], ...]
    }


# ---------------------------------------------------------------- 利用率优先布局 (双轴 overlap 反解)
def _tile_rows_rel(tile, dr):
    """单块前向行数相对权重 (tile/dr/2)² —— 仅用于成本比较, 不参与几何"""
    q = max(1, int(tile) // max(1, int(dr)) // 2)
    return q * q


def _axis_candidates(axis_len, tile, ov_user, dr, tol, n_hi=96):
    """单轴候选: (n, ov, cap, err)，cap 为真实画布长度，ov 已对齐 dr。

    两条生成路径 (画布都 <= 覆盖能力, 绝不缩 tile):
    A. 用用户设定的重叠: cap0 = tile*n - ov_user*(n-1);
       cap0 超过目标时把实际重叠加大到 (tile*n - target)/(n-1) 把画布收回到目标
       (重叠加大是安全方向, 只是浪费算力, 由 η 排序自然淘汰);
       cap0 不足目标时画布只能取 cap0 -> 分辨率降 (err<0), 由 tol 决定能否接受。
    B. 反解: ov = (tile*n - target)/(n-1) 取上下两个 dr 对齐点 -> cap≈目标 (err≈0)。
    只接受 ov >= ov_user (重叠不降到用户设定以下) 且 ov <= tile/2。
    """
    target = int(axis_len)
    lo_cap, hi_cap = target * (1.0 - tol), target * (1.0 + tol)
    out = {}
    n_cap = min(int(n_hi), int(math.ceil(target / float(tile))) + 3)
    for n in range(1, n_cap + 1):
        if n == 1:
            if lo_cap <= tile <= hi_cap:
                out[(1, 0)] = (1, 0, int(tile), abs(tile - target) / float(target))
            continue
        # ---- 路径 A: 用户重叠 / 收到目标 ----
        cap0 = tile * n - int(ov_user) * (n - 1)
        want = min(cap0, target)
        if want >= lo_cap and want >= tile:
            ov = int(round((tile * n - want) / float(n - 1) / int(dr))) * int(dr)
            cap = tile * n - ov * (n - 1)
            if cap >= lo_cap and cap >= tile and cap <= hi_cap:
                out[(n, ov)] = (n, ov, cap, abs(cap - target) / float(target))
        # ---- 路径 B: 反解两点 ----
        raw = (tile * n - target) / float(n - 1)
        for ov in (int(math.floor(raw / dr)) * dr, int(math.ceil(raw / dr)) * dr):
            if ov < 0 or ov < int(ov_user) or ov > tile // 2:
                continue
            cap = tile * n - ov * (n - 1)
            if cap >= tile and lo_cap <= cap <= hi_cap:
                out[(n, ov)] = (n, ov, cap, abs(cap - target) / float(target))
    return list(out.values())


def _compute_layout_util(target_w, target_h, tile_size, ovx_user, ovy_user, dr,
                         k_tol=0.05, eps=0.10, pull_back=False):
    """利用率优先布局: 两轴 overlap 各自反解/收纳, 让画布正好等于覆盖能力。

    与 _compute_layout 的差异:
    - overlap 拆成 x/y 两轴独立: 每轴在 [用户设定, tile/2] 内找 cap 最接近目标且 η 最高的解
    - 覆盖能力大于目标时用"加大实际重叠"把画布收回到目标 (不浪费分辨率、不拉伸)
    - 覆盖能力不足目标时画布只能取覆盖能力 -> 成片略小 (err<0, 由 目标容差/非等比预算 决定能否接受)
    - pull_back=True 时把画布回拉到原目标 (吸收 err, 表现为 <=eps 的非等比); False 则成片=画布, 零拉伸
    - 全程不缩 tile(不掉档), 不越界
    返回与 _compute_layout 同结构的 layout; 无解返回 None。
    """
    tw = _snap_down(tile_size, dr)
    th = tw
    target_w, target_h = int(target_w), int(target_h)
    ux = min(max(0, int(ovx_user)), tw // 4)   # 用户重叠下限钳到 tile/4 (再大纯浪费)
    uy = min(max(0, int(ovy_user)), tw // 4)

    best = None
    for tol in (max(float(k_tol), float(eps), 0.02), 0.25):  # 第二轮放开到 1/4
        cx = _axis_candidates(target_w, tw, ux, dr, tol)
        cy = _axis_candidates(target_h, th, uy, dr, tol)
        for nx, ovx, cw, ex in cx:
            for ny, ovy, ch, ey in cy:
                # 门槛③: 拉回目标后两轴缩放差(非等比变形) <= 非等比预算
                # (逐轴容差只保证各轴偏差, 两轴一正一负时宽高比变形可达 (1+tol)/(1-tol)-1,
                #  必须在组合级用 eps 显式约束, 才是真正的"非等比缩放预算")
                if abs((1.0 + ex) / (1.0 + ey) - 1.0) > float(eps):
                    continue
                cnt = nx * ny
                eta = (cw * ch) / float(cnt * tw * th)
                cost = cnt * _tile_rows_rel(tw, dr)
                key = (-round(eta, 6), round(ex + ey, 6), cost)
                if best is None or key < best[0]:
                    best = (key, nx, ovx, cw, ex, ny, ovy, ch, ey, eta, tol)
        if best is not None:
            break
    if best is None:
        return None

    _, nx, ovx, cw, ex, ny, ovy, ch, ey, eta, tol_used = best
    xs = _compute_positions(cw, tw, nx)
    ys = _compute_positions(ch, th, ny)
    positions = [[x, y] for y in ys for x in xs]
    return {
        "type": "anytile_layout",
        "version": 3,
        "width": int(cw),
        "height": int(ch),
        "target_width": int(cw),
        "target_height": int(ch),
        "tile_width": int(tw),
        "tile_height": int(th),
        "cols": int(nx),
        "rows": int(ny),
        "count": int(nx * ny),
        "overlap": int(max(ovx, ovy)),      # 兼容旧字段/旧日志
        "overlap_x": int(ovx),
        "overlap_y": int(ovy),
        "grid_constrained": False,
        "util_mode": True,
        "pull_back": bool(pull_back),
        "tol_used": round(float(tol_used), 3),
        "eta": round(float(eta), 4),        # tile 利用率 = 画布面积/(块数*tile²)
        "err_w": round(float(ex), 4),       # 画布与目标的相对偏差(负=成片略小)
        "err_h": round(float(ey), 4),
        "cost": int(nx * ny * _tile_rows_rel(tw, dr)),
        # 画布==目标时不缩放; 否则 pull_back 决定是否回拉(回拉即 <=eps 的非等比)
        "original_width": int(target_w if pull_back else cw),
        "original_height": int(target_h if pull_back else ch),
        "positions": positions,
    }


def _coverage_ok(layout):
    """校验 tile 完全覆盖画布且不越界"""
    w, h = layout["width"], layout["height"]
    tw, th = layout["tile_width"], layout["tile_height"]
    covered = torch.zeros((h, w), dtype=torch.bool)
    for x, y in layout["positions"]:
        if x < 0 or y < 0 or x + tw > w or y + th > h:
            return False
        covered[y : y + th, x : x + tw] = True
    return bool(covered.all())


# ---------------------------------------------------------------- 节点 1: 切分放大
class AnyTileSplit:
    TILE_SIZE_TIERS = ["512", "768", "1024", "1536", "2048", "2560"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "输入图片 (单张)"}),
                "scale_mode": (
                    ["放大倍数", "目标分辨率"],
                    {"tooltip": "放大倍数=按倍数等比放大; 目标分辨率=长边放大到该像素 (等比)"},
                ),
                "放大倍数": (
                    "INT",
                    {"default": 2, "min": 1, "max": 8, "step": 1,
                     "tooltip": "仅'放大倍数'模式生效 (整数倍)"},
                ),
                "目标分辨率": (
                    "INT",
                    {"default": 2048, "min": 512, "max": 16384, "step": 512,
                     "tooltip": "仅'目标分辨率'模式生效: 长边像素, 需为512的倍数 (自动就近吸附)"},
                ),
                "tile_size": (
                    cls.TILE_SIZE_TIERS,
                    {"default": "1024",
                     "tooltip": "方形tile档位 (Qwen-2.1特性: 这些尺寸无像素偏移, 其他尺寸会有偏移)"},
                ),
                "overlap": (
                    "INT",
                    {"default": 128, "min": 0, "max": 4096, "step": 8,
                     "tooltip": "相邻tile的重叠像素下限 (自动向上对齐16的倍数)。利用率优先模式下作为该轴重叠的下限, "
                                "画布由 tile×n−ov×(n−1) 反解得出 -> 画布正好铺满覆盖能力, 不再缩画布; "
                                "旧算法下块数只由无重叠覆盖决定, 重叠通过把画布缩小吸收, 拼接后拉回; "
                                "过大时钳到tile/4"},
                ),
                "interp": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {"tooltip": "纯插值放大算法 (未接放大模型时使用)"},
                ),
                # ---- 以下为本轮新增, 追加在末尾以保证旧工作流的 widgets_values 不错位 ----
                "overlap_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 4096, "step": 8,
                     "tooltip": "纵向重叠像素下限; 0=与 overlap 相同。利用率优先模式下两轴各自反解, 精确命中目标长度"},
                ),
                "利用率优先": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "开启=按目标长度反解两轴 overlap, 画布正好铺满覆盖能力(利用率 70%→90%+), 不再为省块缩画布拉伸; "
                                "关闭=旧算法(块数=n=ceil(边长/tile), 重叠通过缩画布吸收)"},
                ),
                "目标容差": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 0.25, "step": 0.01,
                     "tooltip": "利用率优先: 成片与目标允许的最大相对偏差(0.05=5%)。实测 5% 即覆盖全部比例, 再放宽只降分辨率不提利用率"},
                ),
                "非等比预算": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 0.30, "step": 0.01,
                     "tooltip": "利用率优先: 拼接拉回目标后两轴缩放的最大相对差——即非等比变形预算(0.10=10%); "
                                "0=只接受等比解(可能回退旧算法)"},
                ),
                "拉回目标": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "关=成片尺寸=实际画布(全是模型真像素, 零拉伸); 开=拼接后拉回你填的目标分辨率(会引入<=非等比预算的缩放)"},
                ),
                "接缝贴片比例": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05,
                     "tooltip": "接缝延续时只把重叠区的这一比例贴给下一块(黄区), 拼接时整块丢弃(绿块/色偏区被切掉); "
                                "0=关闭(旧行为, 全重叠贴且不丢弃); 建议 >=0.5, 使融合带(重叠-贴片)不超过贴片宽度"},
                ),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "接上=输出VAE编码后的tile latent (img2img); 不接=输出空latent (配denoise=1.0做txt2img)"}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "接上=先ESRGAN类模型放大再对齐目标尺寸; 不接=纯插值放大"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", ANYTILE_LAYOUT)
    RETURN_NAMES = ("tile_images", "tiles_latent", "upscaled_image", "layout")
    OUTPUT_TOOLTIPS = (
        "切分出的tile批次 (统一尺寸, 按行优先排列)",
        "tile latent批次: 接了VAE=编码latent, 没接=空latent(4通道, 采样器会自动校正通道)",
        "放大后的完整图片 (画布尺寸: 为吸收重叠可能略小于目标分辨率; 拼接输出会自动缩回)",
        "切分布局, 连接到 AnyTileMerge",
    )
    FUNCTION = "split"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = (
        "图片先等比放大到目标分辨率 (纯插值或ESRGAN模型), 再切分为带重叠的tile批次输出。\n"
        "块数只由无重叠覆盖决定 n=ceil(边长/tile), 重叠需求通过把画布缩小到 tile×块数-重叠×(块数-1) 吸收——\n"
        "绝不因微小overlap多出一整块tile; 等比网格约束: 搜索省块且画布比例偏差≤10%、分辨率损失≤10% 的\n"
        "tile整数倍网格替代原始目标 (两轴统一缩放, 全程保持宽高比), 例: 7296x1072/tile1024 → 8x1=8块\n"
        "(原8x2=16块); 无合格候选则不约束——宁可不省块也不不等比变形; 拼接后自动拉回原始目标尺寸。\n"
        "放大等比无偏移: 同一倍率作用于两轴, 尺寸对齐到16的倍数 (16是8的倍数, 同时兼容8/16倍下采样VAE)。\n"
        "tile档位 (方形): 512/768/1024/1536/2048/2560 — Qwen-2.1特性: 这些尺寸无像素偏移。\n"
        "切分规则: 画布内等步长排布、最后一块贴边, tile尺寸不变。\n"
        "【利用率优先】(默认开): 两轴 overlap 各自反解 ov=(tile×n−目标)/(n−1), 让画布正好等于覆盖能力,\n"
        "消除\"算了两遍没产生新像素\"的重叠冗余(tile 利用率 70%→90%+), 且不再缩画布再拉伸;\n"
        "反解越界时换块数/换档位; 两轴画布残差 ≤ 目标容差/非等比预算, 拉回后非等比变形 ≤ 非等比预算,\n"
        "无解回退旧算法 (等比网格约束)。"
    )

    def split(self, image, scale_mode, 放大倍数, 目标分辨率, tile_size,
              overlap, interp, overlap_y=0, 利用率优先=True, 目标容差=0.05,
              非等比预算=0.10, 拉回目标=True, 接缝贴片比例=0.5, vae=None, upscale_model=None):
        if image.shape[0] != 1:
            raise Exception("AnyTile 切分放大仅支持单张图片 (batch=1), 请先拆分批次")
        H, W = int(image.shape[1]), int(image.shape[2])
        tile_size = int(tile_size)

        dr = _get_downscale_ratio(vae)

        # ---- 1. 目标分辨率 (等比, 对齐 dr) ----
        if scale_mode == "放大倍数":
            f = float(int(放大倍数))
        else:  # 目标分辨率: 长边像素, 吸附到512的倍数
            target_long = max(512, int(round(int(目标分辨率) / 512.0)) * 512)
            f = target_long / max(W, H)
        target_w = _snap_up(max(dr, round(W * f)), dr)
        target_h = _snap_up(max(dr, round(H * f)), dr)

        # ---- 2. 布局 ----
        # 利用率优先: 两轴 overlap 各自反解(=offset=(tile*n-目标)/(n-1)), 画布正好铺满覆盖能力;
        # 旧算法: 块数=n=ceil(边长/tile), 重叠通过把画布缩小 cap 吸收 (会带来 merge 拉伸)
        ov_y = int(overlap_y) if int(overlap_y) > 0 else int(overlap)
        layout = None
        if 利用率优先:
            layout = _compute_layout_util(
                target_w, target_h, tile_size, overlap, ov_y, dr,
                k_tol=float(目标容差), eps=float(非等比预算), pull_back=bool(拉回目标),
            )
            if layout is None:
                log.warning(
                    "[AnyTile] 利用率优先无解: tile %d 覆盖不了目标 %dx%d (反解 overlap 越界或偏差超限), "
                    "已回退旧算法; 建议换 tile 档位 / 提高 目标容差 / 提高 非等比预算",
                    tile_size, target_w, target_h,
                )
        if layout is None:
            layout = _compute_layout(target_w, target_h, tile_size, overlap, dr)
        canvas_w, canvas_h = layout["width"], layout["height"]

        # ---- 2.5 接缝贴片宽度 seam_strip ----
        # 采样器把"邻居的重叠条带"贴进本tile外缘(接缝延续), 这块区域在解码后必然带污染
        # (绿块/色偏), 无法根治。对策: 只贴重叠区的一部分(黄区), 拼接时整块丢弃,
        # 融合带 = 重叠 - 贴片 >= 贴片, 相邻tile仍有两块内容可混合, 覆盖不会出现空洞。
        ovx_real = int(layout.get("overlap_x", layout["overlap"]))
        ovy_real = int(layout.get("overlap_y", layout["overlap"]))
        tw_l, th_l = int(layout["tile_width"]), int(layout["tile_height"])
        seam_ratio = min(0.9, max(0.0, float(接缝贴片比例)))

        def _strip(ov, n, dim):
            if n <= 1 or ov <= 0 or seam_ratio <= 0:
                return 0
            # 向上对齐: 保证 融合带(=重叠-贴片) <= 贴片宽度, 即"拼接重叠不超过黄区"
            s = int(math.ceil(ov * seam_ratio / float(dr))) * int(dr)
            return max(0, min(s, ov - int(dr), dim // 2))

        layout["seam_ratio"] = seam_ratio
        if seam_ratio > 0:
            layout["seam_strip_x"] = _strip(ovx_real, int(layout["cols"]), tw_l)
            layout["seam_strip_y"] = _strip(ovy_real, int(layout["rows"]), th_l)
        # 比例=0 时不写 strip 字段: 采样器/拼接回落到旧行为(整条重叠贴 + 60/40阶梯, 不丢弃)

        # ---- 3. 放大到画布尺寸 ----
        if upscale_model is not None:
            base = _upscale_with_model(upscale_model, image)
            if int(base.shape[2]) != canvas_w or int(base.shape[1]) != canvas_h:
                base = _resize_image(base, canvas_w, canvas_h, interp)
        else:
            base = _resize_image(image, canvas_w, canvas_h, interp)

        # ---- 4. 切分 (float 张量直接裁切, 无二次量化) ----
        tw, th = layout["tile_width"], layout["tile_height"]
        tiles = torch.stack(
            [base[0, y : y + th, x : x + tw, :] for x, y in layout["positions"]], dim=0
        )

        # ---- 5. latent: 接了VAE=编码, 没接=空latent ----
        if vae is not None:
            chunks = []
            step = 4  # 分块编码限制显存
            for i0 in range(0, layout["count"], step):
                chunks.append(vae.encode(tiles[i0 : i0 + step, :, :, :3]))
            samples = torch.cat(chunks, dim=0)
            latent = {"samples": samples}
        else:
            # 空latent按SD系/8惯例输出(通道数由采样器自动校正), 文生图tile用;
            # Qwen-2.1 等编辑模型请接VAE走编码latent
            latent = {
                "samples": torch.zeros(
                    (layout["count"], 4, th // 8, tw // 8), dtype=torch.float32
                )
            }

        if layout.get("util_mode"):
            log.info(
                "[AnyTile] 输入 %dx%d -> 目标 %dx%d | 画布 %dx%d (偏差 %.2f%%/%.2f%%, %s) | "
                "tile %dx%d 网格 %dx%d=%d块 重叠 %d/%d | 利用率 %.1f%% | latent=%s",
                W, H, target_w, target_h, canvas_w, canvas_h,
                layout["err_w"] * 100.0, layout["err_h"] * 100.0,
                "拼接拉回目标" if layout.get("pull_back") else "成片=画布, 零拉伸",
                tw, th, layout["cols"], layout["rows"], layout["count"],
                layout["overlap_x"], layout["overlap_y"], layout["eta"] * 100.0,
                "vae编码" if vae is not None else "空latent",
            )
            if float(layout.get("tol_used", 0)) > max(float(目标容差), float(非等比预算)) + 1e-6:
                log.warning(
                    "[AnyTile] 利用率优先: 目标 %dx%d 在 tile %d 下贴不齐, 已放宽到 ±%.0f%%; "
                    "成片 %dx%d (差 %.1f%%/%.1f%%)",
                    target_w, target_h, int(tile_size), float(layout["tol_used"]) * 100.0,
                    canvas_w, canvas_h, layout["err_w"] * 100.0, layout["err_h"] * 100.0,
                )
            # 调参提示: 重叠减半能否少切几块 (求解成本极低, 只为产线省时间)
            if int(overlap) >= 64:
                alt = _compute_layout_util(
                    target_w, target_h, tile_size, max(16, int(overlap) // 2),
                    max(16, ov_y // 2), dr,
                    k_tol=float(目标容差), eps=float(非等比预算), pull_back=bool(拉回目标),
                )
                if alt is not None and int(alt["count"]) < int(layout["count"]):
                    log.info(
                        "[AnyTile] 调参提示: overlap %d -> %d 可把 %d 块降到 %d 块 (画布 %dx%d, 利用率 %.0f%%); "
                        "接缝冗余会减少, 接缝延续/遮罩羽化需相应确认",
                        int(overlap), max(16, int(overlap) // 2),
                        int(layout["count"]), int(alt["count"]),
                        int(alt["width"]), int(alt["height"]), float(alt["eta"]) * 100.0,
                    )
        else:
            log.info(
                "[AnyTile] 输入 %dx%d -> 目标 %dx%d (画布 %dx%d%s) | tile %dx%d 网格 %dx%d=%d块 重叠%d | latent=%s",
                W, H, target_w, target_h, canvas_w, canvas_h,
                ", 等比网格约束" if layout["grid_constrained"] else ", 重叠吸收部分拼接后缩回",
                tw, th, layout["cols"], layout["rows"], layout["count"], layout["overlap"],
                "vae编码" if vae is not None else "空latent",
            )
        return (tiles, latent, base, layout)


SEAM_RAMP_RATIO = 0.6  # mask 台阶位置: 重叠条带60%处


def _strip_mask_pattern(wl):
    """条带noise_mask分布 (阶梯): 分块边界起60%的区域=0 (锁定邻居内容, 不重绘),
    到60%处直接跳变为1 (自由生成), 无线性过渡。
    过渡台阶藏在重叠区内部、远离tile核心, 拼接时被重叠内容吸收。"""
    step = max(1, int(round(wl * SEAM_RAMP_RATIO)))
    m = torch.zeros(wl)
    m[step:] = 1.0
    return m


def _seam_constraint(samples_in, idx, layout, sampled, lh, lw):
    """接缝延续: 为第 idx 块tile构建"携带已采样邻居重叠内容"的初始latent与noise_mask。

    行优先采样, 左邻(idx-1)/上邻(idx-cols)已生成时, 把邻居对应区域的 latent 贴进本tile外缘。

    两种模式 (由 layout["seam_strip_x/y"] 决定):
    - 贴片模式(strip>=0, 默认): 只贴重叠区的 strip 像素宽(黄区), 且整块 mask=0(完全钉死)。
      这块区域解码后会带污染(绿块/色偏), 由 AnyTileMerge 在拼接时整块丢弃, 不进成片。
      邻居 latent 的取片起点固定为"重叠起点", 与画布坐标严格对齐(与贴片宽度无关)。
    - 旧模式(旧 layout 无 strip 字段): 贴整个重叠条带, 60%钉死+40%自由 的阶梯 mask。
    """
    positions = layout["positions"]
    cols = layout["cols"]
    tw, th = layout["tile_width"], layout["tile_height"]
    # 像素 -> 本tile局部latent 的换算: 用 tile 尺寸 (不是画布尺寸!)
    sy = lh / max(1, th)
    sx = lw / max(1, tw)
    x, y = positions[idx][0], positions[idx][1]
    init = samples_in[idx : idx + 1].clone()
    mask = torch.ones((1, 1, lh, lw), dtype=init.dtype, device=init.device)
    r, c = divmod(idx, cols)
    strip_x = int(layout.get("seam_strip_x", -1))   # -1 = 旧 layout
    strip_y = int(layout.get("seam_strip_y", -1))

    # 左邻 (同行, 已采样): 本tile左侧重叠条带
    if c > 0 and (idx - 1) in sampled:
        ov_px = tw - (x - positions[idx - 1][0])
        ov_lat = min(lw, max(0, int(round(ov_px * sx))))     # 全重叠宽度(latent列)
        wl = ov_lat if strip_x < 0 else min(ov_lat, max(0, int(round(strip_x * sx))))
        if ov_lat > 0 and wl > 0:
            # 邻居latent的"重叠起点"列 = lw-ov_lat, 与本tile第0列同处画布同一位置
            nl = sampled[idx - 1][:, :, :, lw - ov_lat : lw - ov_lat + wl]
            if strip_x < 0:
                m = _strip_mask_pattern(wl).to(dtype=init.dtype, device=init.device).view(1, 1, 1, wl)
                init[:, :, :, :wl] = nl * (1.0 - m) + init[:, :, :, :wl] * m
                mask[:, :, :, :wl] = torch.minimum(mask[:, :, :, :wl], m)
            else:
                init[:, :, :, :wl] = nl                       # 整块替换
                mask[:, :, :, :wl] = 0.0                      # 全部钉死(反正拼接时丢弃)
    # 上邻 (同列, 已采样): 本tile顶部重叠条带
    if r > 0 and (idx - cols) in sampled:
        ov_px = th - (y - positions[idx - cols][1])
        ov_lat = min(lh, max(0, int(round(ov_px * sy))))
        hl = ov_lat if strip_y < 0 else min(ov_lat, max(0, int(round(strip_y * sy))))
        if ov_lat > 0 and hl > 0:
            nt = sampled[idx - cols][:, :, lh - ov_lat : lh - ov_lat + hl, :]
            if strip_y < 0:
                m = _strip_mask_pattern(hl).to(dtype=init.dtype, device=init.device).view(1, 1, hl, 1)
                init[:, :, :hl, :] = nt * (1.0 - m) + init[:, :, :hl, :] * m
                mask[:, :, :hl, :] = torch.minimum(mask[:, :, :hl, :], m)
            else:
                init[:, :, :hl, :] = nt
                mask[:, :, :hl, :] = 0.0
    return init, mask


# ---------------------------------------------------------------- 节点 2: 分批采样
class AnyTileBatchSampler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "采样模型"}),
                "latent_image": ("LATENT", {"tooltip": "AnyTileSplit 输出的 tiles_latent"}),
                "vae": ("VAE", {"tooltip": "用于解码采样后的tile latent"}),
                "positive": ("CONDITIONING", {"tooltip": "正向条件 (作用于所有tile)"}),
                "negative": ("CONDITIONING", {"tooltip": "负向条件"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "每批tile噪声种子依次 +1, 保证各tile噪声不同"}),
                "tile_batch": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1,
                                       "tooltip": "每次采样/解码的tile数量, 显存不足就调小"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                  "tooltip": "Qwen/Flux 等流模型常用 1.0, SD/SDXL 常用 6-8"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "空latent用1.0; VAE编码latent建议0.3-0.5"}),
                "接缝延续": ("BOOLEAN", {"default": False,
                                       "tooltip": "采样级接缝融合: 每块tile采样时把邻居重叠区的一部分(宽度=切分节点的 接缝贴片比例)贴进本tile外缘并完全钉死, 生成时延续邻居结构; 该贴片区解码后带绿块/色偏, 由拼接节点整块丢弃 (需接layout; tile_batch强制为1)"}),
            },
            "optional": {
                "layout": (ANYTILE_LAYOUT, {"tooltip": "AnyTileSplit 输出的切分布局; 接缝延续需要"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("tiles_image", "samples")
    OUTPUT_TOOLTIPS = ("解码后的tile图像批次 (行优先, 与切分顺序一致)", "采样完成的完整latent批次")
    FUNCTION = "sample"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = "把 tile latent 按'每批N个'循环采样并VAE解码, 避免一次性大批次采样OOM。接缝延续开启后, 每块tile采样时携带已采样邻居的重叠latent与羽化噪声遮罩, 接缝在采样阶段即融合。"

    def sample(self, model, latent_image, vae, positive, negative, seed,
               tile_batch, steps, cfg, sampler_name, scheduler, denoise,
               接缝延续=False, layout=None):
        from nodes import common_ksampler
        import comfy.model_management

        samples_in = latent_image.get("samples")
        if samples_in is None:
            raise Exception("latent_image 中没有 samples")
        n = int(samples_in.shape[0])
        seed = int(seed) % (0xFFFFFFFFFFFFFFFF + 1)

        seam = bool(接缝延续) and layout is not None
        if bool(接缝延续) and layout is None:
            log.warning("[AnyTile] 接缝延续已开启但未接layout, 本次不启用")
        tb = 1 if seam else max(1, int(tile_batch))
        if seam:
            layout["seam_applied"] = True   # 通知拼接节点: 外缘被贴过邻居内容, 需要丢弃绿块区
            log.info("[AnyTile] 接缝延续启用: 逐tile依赖采样 (tile_batch 强制为1)")

        lh, lw = int(samples_in.shape[2]), int(samples_in.shape[3])
        sampled = {}  # tile序号 -> 已采样latent (供后续tile接缝延续)
        outs = []
        done = 0
        for i0 in range(0, n, tb):
            m = min(tb, n - i0)
            chunk = {"samples": samples_in[i0 : i0 + m].clone()}
            if "noise_mask" in latent_image:
                chunk["noise_mask"] = latent_image["noise_mask"][i0 : i0 + m]
            if seam:
                for k in range(i0, i0 + m):
                    init_k, smask = _seam_constraint(samples_in, k, layout, sampled, lh, lw)
                    chunk["samples"][k - i0 : k - i0 + 1] = init_k  # 局部索引!
                    if float(smask.min()) < 1.0:  # 有实际约束才设置noise_mask
                        if "noise_mask" in chunk:
                            chunk["noise_mask"][k - i0 : k - i0 + 1] = torch.minimum(
                                chunk["noise_mask"][k - i0 : k - i0 + 1], smask)
                        else:
                            chunk["noise_mask"] = smask
            # 逐tile条件/参考按批切片, 避免ComfyUI截断导致后续批次错用第1批条件
            pos_c = _slice_conditioning(positive, i0, m, n)
            neg_c = _slice_conditioning(negative, i0, m, n)
            out = common_ksampler(
                model, (seed + i0) % (0xFFFFFFFFFFFFFFFF + 1), steps, cfg,
                sampler_name, scheduler, pos_c, neg_c, chunk, denoise=denoise,
            )[0]["samples"]
            if seam:
                for k in range(i0, i0 + m):
                    sampled[k] = out[k - i0 : k - i0 + 1]
            outs.append(out)
            done += int(out.shape[0])
            log.info("[AnyTile] 采样进度 %d/%d tile", done, n)
            comfy.model_management.soft_empty_cache()
        samples_out = torch.cat(outs, dim=0)

        # ---- 分批解码 ----
        imgs = []
        for i0 in range(0, n, max(1, tb)):
            imgs.append(vae.decode(samples_out[i0 : i0 + max(1, tb)]))
        image = torch.cat(imgs, dim=0)
        return (image, {"samples": samples_out})


# ---------------------------------------------------------------- 拼接辅助: 颜色还原 + 遮罩拓展/羽化
def _match_tile_color(tile_np, ref_np, strength):
    """孤海-裁剪恢复同款颜色还原: 鲁棒增益+偏移统一校色 (输入/输出 HxWx3 float32 0-1)。

    只采样"轻微偏色"的同位置像素(排除明显重绘区), 6轮中位数回归估计每通道
    增益(钳制0.9~1.1)与偏移(钳制±24级), 按strength混合, 避免放大噪声。"""
    import cv2
    if strength <= 0:
        return tile_np
    # 通道防御: 统一按RGB校色 (RGBA/RGB混用时报广播错)
    if tile_np.shape[-1] > 3:
        tile_np = tile_np[..., :3]
    if ref_np.shape[-1] > 3:
        ref_np = ref_np[..., :3]
    高, 宽 = tile_np.shape[:2]
    比例 = min(1.0, 256.0 / max(高, 宽))
    工作尺寸 = (max(1, round(宽 * 比例)), max(1, round(高 * 比例)))
    源 = cv2.resize(tile_np, 工作尺寸, interpolation=cv2.INTER_AREA)
    目标 = cv2.resize(ref_np, 工作尺寸, interpolation=cv2.INTER_AREA)

    # 局部明暗结构基本不受整体偏色影响, 降低新增/移除轮廓的采样权重
    源灰度 = cv2.cvtColor(源, cv2.COLOR_RGB2GRAY)
    目标灰度 = cv2.cvtColor(目标, cv2.COLOR_RGB2GRAY)
    源细节 = 源灰度 - cv2.GaussianBlur(源灰度, (0, 0), 1.5)
    目标细节 = 目标灰度 - cv2.GaussianBlur(目标灰度, (0, 0), 1.5)
    结构权重 = 1.0 / (1.0 + ((源细节 - 目标细节) / 0.02) ** 2)
    # 仅接受轻微偏色: RGB各通道最多差24级, 明度最多差18级(按0~255计)
    候选区域 = (np.max(np.abs(目标 - 源), axis=2) <= 24.0 / 255.0)
    候选区域 &= (np.abs(目标灰度 - 源灰度) <= 18.0 / 255.0) & (结构权重 > 0.8)
    候选区域 = cv2.erode(候选区域.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    最少样本 = max(16, 候选区域.size * 0.05)
    if np.count_nonzero(候选区域) < 最少样本:
        return tile_np
    源样本 = 源[候选区域].astype(np.float64)
    目标样本 = 目标[候选区域].astype(np.float64)
    结构权重 = 结构权重[候选区域]
    增益 = np.ones(3, dtype=np.float64)
    偏移 = np.median(目标样本 - 源样本, axis=0)
    for _ in range(6):
        残差 = 目标样本 - (源样本 * 增益 + 偏移)
        中心 = np.median(残差, axis=0)
        尺度 = np.maximum(np.median(np.abs(残差 - 中心), axis=0) * 1.4826, 0.004)
        偏差 = np.max(np.abs(残差 - 中心) / (4.685 * 尺度), axis=1)
        权重 = np.maximum(1.0 - 偏差 ** 2, 0.0) ** 2 * 结构权重
        权重和 = 权重.sum()
        if 权重和 < max(16, len(源样本) * 0.1):
            return tile_np
        权重 = 权重[:, None] / 权重和
        源均值 = (源样本 * 权重).sum(axis=0)
        目标均值 = (目标样本 * 权重).sum(axis=0)
        方差 = ((源样本 - 源均值) ** 2 * 权重).sum(axis=0)
        协方差 = ((源样本 - 源均值) * (目标样本 - 目标均值) * 权重).sum(axis=0)
        # 平坦背景缺少对比度信息, 增益回归1, 主要用偏移校正以免放大噪声
        增益 = np.clip((协方差 + 1e-4) / (方差 + 1e-4), 0.9, 1.1)
        偏移 = np.clip(目标均值 - 源均值 * 增益, -24.0 / 255.0, 24.0 / 255.0)

    稳定区域 = (np.max(np.abs(目标样本 - (源样本 * 增益 + 偏移)), axis=1) < 0.035) & (结构权重 > 0.5)
    if np.count_nonzero(稳定区域) < max(最少样本, len(源样本) * 0.6):
        return tile_np
    混合强度 = min(float(strength), 1.0)
    增益 = (1.0 + (增益 - 1.0) * 混合强度).astype(np.float32)
    偏移 = (偏移 * 混合强度).astype(np.float32)
    校正 = np.empty_like(tile_np)
    for 起点 in range(0, 高, 256):
        校正[起点:起点 + 256] = np.clip(tile_np[起点:起点 + 256] * 增益 + 偏移, 0.0, 1.0)
    return 校正


def _build_tile_mask(th, tw, margin_px, feather_px,
                     edge_left=False, edge_right=False, edge_top=False, edge_bottom=False):
    """tile贴回遮罩: 全权重核心向四边收缩margin_px(线性), 再高斯羽化feather_px。

    edge_*: 该侧是画布外边界(整图边缘, 无邻居补偿)时不收缩不羽化——
    若收缩, 外圈权重跌破1, 接背景图时 alpha=weight 会把低分辨率背景
    混进整图边缘一圈 (表现为成片边缘模糊)。边界侧应100%使用tile原像素。
    下限0.02防止单tile覆盖区归一化除零/接缝黑线; float全程不量化。"""
    import cv2
    mask = np.ones((th, tw), dtype=np.float32)
    if margin_px > 0:
        BIG = float(th + tw + margin_px)
        行上 = np.full(th, BIG, np.float32) if edge_top else np.arange(th, dtype=np.float32)
        行下 = np.full(th, BIG, np.float32) if edge_bottom else np.arange(th - 1, -1, -1, dtype=np.float32)
        列左 = np.full(tw, BIG, np.float32) if edge_left else np.arange(tw, dtype=np.float32)
        列右 = np.full(tw, BIG, np.float32) if edge_right else np.arange(tw - 1, -1, -1, dtype=np.float32)
        距边 = np.minimum(np.minimum(行上, 行下)[:, None], np.minimum(列左, 列右)[None, :])
        mask = np.clip(距边 / float(margin_px), 0.0, 1.0)
    if feather_px > 0:
        sigma = max(0.5, feather_px / 2.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REPLICATE)
    return np.clip(mask, 0.02, 1.0).astype(np.float32)


# ---------------------------------------------------------------- 节点 3: 拼接
class AnyTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE", {"tooltip": "采样完成并解码后的tile图像批次 (顺序需与切分时一致; 尺寸与布局不一致时自动缩放对齐)"}),
                "layout": (ANYTILE_LAYOUT, {"tooltip": "AnyTileSplit 输出的切分布局"}),
                "颜色还原强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,
                                        "tooltip": "孤海裁剪恢复同款: 每块tile与背景图对应区域比较, 统一修正明度/颜色; 0=关闭; 需接背景图"}),
                "遮罩拓展百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1,
                                        "tooltip": "tile贴回遮罩向内收缩, 像素=tile平均边长×%/100; 接缝过渡带会自动扩展到重叠区的60%以锚定原图, 此参数为下限; 0=全重叠线性羽化(旧行为)"}),
                "遮罩羽化百分比": ("INT", {"default": 5, "min": 0, "max": 30, "step": 1,
                                        "tooltip": "拓展后的遮罩边界再做高斯羽化, 像素=tile平均边长×%/100; 0=不加"}),
                "丢弃接缝贴片": ("BOOLEAN", {"default": True,
                                        "tooltip": "开=把采样时被贴过邻居内容的外缘条带(绿块/色偏区)整块丢弃, 由邻居补上; "
                                                   "仅当采样器开了接缝延续时需要; 走其它采样器时请关掉"}),
            },
            "optional": {
                "背景图": ("IMAGE", {"tooltip": "两种接法: ①切分节点的upscaled_image(画布尺寸)→颜色还原+tile边缘融回原图; ②切分节点的tile_images(原始tile批次)→仅作逐tile颜色还原参考"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    OUTPUT_TOOLTIPS = ("拼接完成的完整大图 (尺寸=目标分辨率)",)
    FUNCTION = "merge"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = (
        "把tile批次按像素层拼接回完整大图。\n"
        "切分时若 利用率优先 开启, 画布=覆盖能力(tile×n−ov×(n−1)), 本节点按 layout 记录的\n"
        "original尺寸输出: 拉回目标=开则等比/非等比缩回你填的目标分辨率, 关则直接输出画布。\n"
        "旧算法下画布被缩小以吸收重叠 (避免为微小overlap多出tile), 拼接完成后自动缩回目标分辨率。\n"
        "接背景图后: 每块tile先做颜色还原(孤海裁剪恢复同款鲁棒校色), 再按拓展/羽化百分比"
        "生成的贴回遮罩平滑融回原图; 不接背景图时为tile间线性羽化拼接。"
    )

    def merge(self, tiles, layout, 背景图=None, 颜色还原强度=1.0, 遮罩拓展百分比=5, 遮罩羽化百分比=5,
              丢弃接缝贴片=True):
        import cv2

        if not isinstance(layout, dict) or layout.get("type") != "anytile_layout":
            raise Exception("layout 不是 AnyTileSplit 的输出")
        count = layout["count"]
        if int(tiles.shape[0]) != count:
            raise Exception(f"tile 数量不匹配: 切分时 {count} 块, 当前 {int(tiles.shape[0])} 块 (检查切分与拼接参数是否一致)")
        tw, th = layout["tile_width"], layout["tile_height"]

        W, H = layout["width"], layout["height"]
        cols = layout["cols"]
        rows = int(layout["rows"])
        positions = layout["positions"]
        tiles_cpu = tiles.cpu().float()
        bg_raw = 背景图.cpu().float() if 背景图 is not None else None

        # 通道统一: 取 tiles/背景图 的公共通道数 (RGBA图源 与 RGB解码 混用等情形)
        common_c = int(tiles_cpu.shape[-1])
        if bg_raw is not None and int(bg_raw.shape[-1]) < common_c:
            common_c = int(bg_raw.shape[-1])
        if int(tiles_cpu.shape[-1]) != common_c:
            tiles_cpu = tiles_cpu[..., :common_c]
        C = common_c

        # tile 解码尺寸容错: 与布局不一致(如VAE下采样倍率未对齐)时缩放回布局尺寸
        a_th, a_tw = int(tiles_cpu.shape[1]), int(tiles_cpu.shape[2])
        if (a_th, a_tw) != (th, tw):
            tiles_cpu = _resize_image(tiles_cpu, tw, th, "lanczos")
            log.warning(
                "[AnyTile] tile解码尺寸 %dx%d 与布局 %dx%d 不一致(VAE下采样倍率未对齐), 已自动缩放对齐;"
                " 建议切分节点也接同一个VAE以从源头对齐尺寸", a_tw, a_th, tw, th,
            )

        # 背景图双模式: 画布(W×H) 或 原始tile批次(数量×tile尺寸)
        bg = None        # 画布模式: 颜色还原参考 + 边缘融回
        bg_tiles = None  # tile批次模式: 仅逐tile颜色还原参考
        if bg_raw is not None:
            b = bg_raw
            if int(b.shape[0]) == count and int(b.shape[1]) == th and int(b.shape[2]) == tw:
                bg_tiles = b[:, :, :, :C]
            elif int(b.shape[1]) == H and int(b.shape[2]) == W:
                bg = b[0, :, :, :C]
            else:
                raise Exception(
                    f"背景图尺寸不符: 需要 画布{W}x{H} 或 tile批次{count}块{tw}x{th}, "
                    f"实际 {int(b.shape[2])}x{int(b.shape[1])}×{int(b.shape[0])}"
                )
        has_bg_ref = bg is not None or bg_tiles is not None
        if 颜色还原强度 > 0 and not has_bg_ref:
            log.warning("[AnyTile] 颜色还原需要接背景图(upscaled_image或tile_images), 本次跳过颜色还原")

        # 遮罩拓展/羽化像素值: tile平均边长 × 百分比 / 100 (孤海同款换算)
        avg_side = (tw + th) / 2.0
        margin_px = max(0, int(round(avg_side * max(0, int(遮罩拓展百分比)) / 100.0)))
        feather_px = max(0, int(round(avg_side * max(0, int(遮罩羽化百分比)) / 100.0)))
        use_mask = margin_px > 0 or feather_px > 0

        canvas = torch.zeros((H, W, C), dtype=torch.float32)
        weight = torch.zeros((H, W, 1), dtype=torch.float32)

        # 接缝贴片丢弃宽度: 采样阶段被贴过邻居内容的外缘(黄区), 解码后带绿块/色偏,
        # 这里整块不给权重 -> 该区域完全由邻居tile覆盖(融合带=重叠-贴片, 仍有两块内容可混)
        want_strip = bool(丢弃接缝贴片) and bool(layout.get("seam_applied", True))
        strip_x = max(0, min(int(layout.get("seam_strip_x", 0) or 0), tw // 2)) if want_strip else 0
        strip_y = max(0, min(int(layout.get("seam_strip_y", 0) or 0), th // 2)) if want_strip else 0
        if strip_x or strip_y:
            log.info(
                "[AnyTile] 拼接舍弃接缝贴片: 左右 %dpx / 上下 %dpx (重叠 %d/%d, 贴片比例 %.2f)",
                strip_x, strip_y, int(layout.get("overlap_x", 0) or 0),
                int(layout.get("overlap_y", 0) or 0), float(layout.get("seam_ratio", 0) or 0),
            )

        for i, (x, y) in enumerate(positions):
            t = tiles_cpu[i]
            r, c = divmod(i, cols)
            sx0 = strip_x if c > 0 else 0      # 左侧被贴过(有左邻)
            sy0 = strip_y if r > 0 else 0      # 顶部被贴过(有上邻)
            # 1) 颜色还原: 与背景对应区域鲁棒校色 (孤海裁剪恢复同款; 背景可为画布或原始tile)
            if has_bg_ref and 颜色还原强度 > 0:
                tile_np = np.array(t.numpy(), copy=True)
                ref_np = (np.ascontiguousarray(bg_tiles[i].numpy()) if bg_tiles is not None
                          else np.ascontiguousarray(bg[y : y + th, x : x + tw, :].numpy()))
                if sx0 or sy0:
                    # 污染区先换成参考内容, 避免绿块把整块校色统计带偏
                    ref_c = ref_np[..., :tile_np.shape[-1]]
                    tile_np[:, :sx0] = ref_c[:, :sx0]
                    tile_np[:sy0, :] = ref_c[:sy0, :]
                t = torch.from_numpy(_match_tile_color(tile_np, ref_np, 颜色还原强度))

            # 2) 贴回遮罩: 拓展(核心收缩)+羽化; 未启用时保留与左/上邻实际重叠的线性羽化
            if use_mask:
                # 画布边界侧 (整图外缘, 无邻居补偿) 不收缩: 保持全权重, 避免外圈融回背景图变模糊
                w2 = torch.from_numpy(_build_tile_mask(
                    th, tw, margin_px, feather_px,
                    edge_left=(c == 0), edge_right=(c == cols - 1),
                    edge_top=(r == 0), edge_bottom=(r == rows - 1),
                )).unsqueeze(-1)
            else:
                w2 = torch.ones((th, tw, 1), dtype=torch.float32)
                if c > 0:
                    ovx = tw - (x - positions[i - 1][0])
                    if ovx > 0:
                        w2[:, :ovx, :] *= torch.linspace(0.0, 1.0, ovx).view(1, ovx, 1)
                if r > 0:
                    ovy = th - (y - positions[i - cols][1])
                    if ovy > 0:
                        w2[:ovy, :, :] *= torch.linspace(0.0, 1.0, ovy).view(ovy, 1, 1)

            # 3) 丢弃含绿块的接缝贴片区 (必须在羽化之后, 否则羽化会把权重抬起来)
            if sx0 > 0:
                w2[:, :sx0, :] = 0.0
            if sy0 > 0:
                w2[:sy0, :, :] = 0.0

            canvas[y : y + th, x : x + tw, :] += t * w2
            weight[y : y + th, x : x + tw, :] += w2

        out = canvas / weight.clamp_min(1e-6)
        if bg is not None:
            # 画布模式背景: 未被tile全权重覆盖的区域平滑融回背景图
            alpha = weight.clamp(0.0, 1.0)
            out = out * alpha + bg * (1.0 - alpha)
        # 恢复目标: 尾块约束把 target 改写为画布值(切分时两轴缩幅不同会破坏宽高比),
        # 此处拉回约束前的原始目标尺寸——比例与分辨率一并恢复;
        # 未约束场景 original==target, 等价于 overlap 吸收缩回; 老 layout 无记录不缩放
        ow = int(layout.get("original_width", layout.get("target_width", W)))
        oh = int(layout.get("original_height", layout.get("target_height", H)))
        if (ow, oh) != (W, H):
            out = _resize_image(out.unsqueeze(0), ow, oh, "lanczos")[0]
            log.info("[AnyTile] 拼接完成 %dx%d -> 恢复目标 %dx%d", W, H, ow, oh)
        return (out.unsqueeze(0),)


# ---------------------------------------------------------------- 节点 4: 逐Tile文本编码 (Qwen Image 2.1)
_REF_SIZE_OPTIONS = ["自动", "768", "1024", "1344", "1536", "2048"]


def _area_limited_size(width, height, max_area):
    """面积上限内取长宽比保持、向下对齐32的尺寸, 不放大小图"""
    width = max(1, int(width))
    height = max(1, int(height))
    max_area = max(32 * 32, min(int(max_area), 2048 * 2048))
    if width * height <= max_area:
        tw = max(32, (width // 32) * 32)
        th = max(32, (height // 32) * 32)
    else:
        scale = math.sqrt(max_area / (width * height))
        tw = max(32, min(width, round(width * scale / 32) * 32))
        th = max(32, min(height, round(height * scale / 32) * 32))
    while tw * th > max_area:
        if tw >= th and tw > 32:
            tw -= 32
        elif th > 32:
            th -= 32
        else:
            break
    return tw, th


def _prepare_vision_image(image):
    """取RGB; 带alpha时白底合成"""
    rgb = image[:, :, :, :3]
    if image.shape[-1] > 3:
        rgb = rgb * image[:, :, :, 3:4] + (1.0 - image[:, :, :, 3:4])
    return rgb


def _batch_conditionings(per_tile):
    """把N个tile各自的conditioning列表合并成batch=N的conditioning列表。

    encode_from_tokens_scheduled 返回 [[cond(1,T,D), dict], ...] (dict含pooled_output/image_slots等);
    合并后为 [[cond(N,T,D), dict], ...], 张量按batch拼接, 非张量同值项(如image_slots)保留一份。
    """
    n_entries = len(per_tile[0])
    merged = []
    for j in range(n_entries):
        cond = torch.cat([e[j][0] for e in per_tile], dim=0)
        opts = {}
        keys = []
        for e in per_tile:
            for k in e[j][1].keys():
                if k not in keys:
                    keys.append(k)
        for k in keys:
            vals = [e[j][1].get(k) for e in per_tile]
            first = vals[0]
            if isinstance(first, torch.Tensor):
                if all(v is not None and v.shape == first.shape for v in vals):
                    opts[k] = torch.cat(vals, dim=0)
                else:
                    log.warning("[AnyTile] 条件字段 %s 各tile形状不一致, 保留第1块tile的值", k)
                    opts[k] = first
            else:
                opts[k] = first
        merged.append([cond, opts])
    return merged


class AnyTilePerTileTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "Qwen Image 2.1 的 CLIP (Qwen2.5-VL 文本编码器)"}),
                "tile_images": ("IMAGE", {"tooltip": "AnyTile切分放大输出的 tile_images 批次, 每块tile用自己的图做视觉参考"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                                      "tooltip": "正向提示词, 作用于所有tile (如 8K 高清细节修复 类指令)"}),
                "ref_image_size": (_REF_SIZE_OPTIONS, {"default": "自动",
                                                       "tooltip": "每块tile参考图面积上限; 自动=1024²; 越大VL/参考token越多越吃显存"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "接上=每块tile附带自己的reference_latents (Qwen-2.1编辑模型标准用法, 视觉token由参考latent顶替); 不接=图像仅通过文本编码器视觉通道条件"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    OUTPUT_TOOLTIPS = (
        "逐tile条件批次 (batch=tile数, 每块tile自己的视觉参考)",
        "与positive相同, 供CFG=1使用 (Qwen-2.1编辑流惯例)",
        "接VAE=每块tile全分辨率编码latent(可作采样latent); 不接VAE=Qwen-2.1格式空latent(对齐官方节点行为)",
    )
    FUNCTION = "encode"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = (
        "把tile批次逐块做Qwen-2.1文本编码: 每块tile = 自己的图 + 同一条prompt, 输出batch=tile数的conditioning。\n"
        "采样器条件batch=latent批次时逐元素对应, 每块tile只看到自己的参考 (接VAE时还带各自reference_latents)。\n"
        "negative输出=positive, 请配合CFG=1使用。latent输出与官方'文本编码 Qwen Image 2.1'节点对齐, 便于工作流替换。"
    )

    def encode(self, clip, tile_images, prompt, ref_image_size, vae=None):
        import comfy.utils
        import comfy.model_management
        import node_helpers

        n = int(tile_images.shape[0])
        if ref_image_size not in _REF_SIZE_OPTIONS:
            ref_image_size = "自动"
        max_area = 1024 * 1024 if ref_image_size == "自动" else min(int(ref_image_size) ** 2, 2048 * 2048)
        keep_vision = vae is None

        # 每块tile: 面积限制缩放 + VL视觉预处理
        prepared = []
        for i in range(n):
            img = tile_images[i : i + 1]
            w0, h0 = int(img.shape[2]), int(img.shape[1])
            tw, th = _area_limited_size(w0, h0, max_area)
            if (tw, th) != (w0, h0):
                img = comfy.utils.common_upscale(img.movedim(-1, 1), tw, th, "lanczos", "disabled").movedim(1, -1)
            prepared.append(img)

        # 逐tile编码 (视觉参考各自独立, 文本相同)
        conds = []
        ref_lats = []
        latents = []
        for i in range(n):
            tokens = clip.tokenize(
                prompt, images=[_prepare_vision_image(prepared[i])],
                keep_vision=keep_vision, prevent_empty_text=True,
            )
            conds.append(clip.encode_from_tokens_scheduled(tokens))
            if vae is not None:
                full = tile_images[i : i + 1, :, :, :3]
                lat_i = vae.encode(full)  # 全分辨率tile latent (采样起点)
                latents.append(lat_i)
                if int(prepared[i].shape[1]) == int(full.shape[1]) and int(prepared[i].shape[2]) == int(full.shape[2]):
                    ref_lats.append(lat_i)  # 参考图未被缩放, 复用同一latent
                else:
                    ref_lats.append(vae.encode(prepared[i][:, :, :, :3]))
            if (i + 1) % 8 == 0 or i + 1 == n:
                log.info("[AnyTile] 逐tile编码进度 %d/%d", i + 1, n)

        positive = _batch_conditionings(conds)  # batch=N, 每块tile一条
        if vae is not None:
            refs = torch.cat(ref_lats, dim=0)  # (N, C, h, w) 每块tile自己的参考latent
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [refs]}, append=True
            )
            latent = {"samples": torch.cat(latents, dim=0)}
        else:
            H, W = int(tile_images.shape[1]), int(tile_images.shape[2])
            latent = {"samples": torch.zeros(
                (n, 64, H // 16, W // 16),
                device=comfy.model_management.intermediate_device(),
            )}
        negative = positive

        log.info(
            "[AnyTile] 逐tile编码完成: %d块 | cond=%s | 参考=%s | latent=%s",
            n, tuple(positive[0][0].shape),
            "reference_latents" if vae is not None else "仅视觉通道",
            tuple(latent["samples"].shape),
        )
        return (positive, negative, latent)


def _slice_guider(guider, start, length, total):
    """分批采样专用: 浅拷贝guider并把其original_conds切片到当前批。

    CFGGuider系列(BasicGuider/CFGGuider/DualCFGGuider)的条件存放在 original_conds 字典,
    每次调用 sample() 时由 prepare_sampling 处理; 浅拷贝共享 model_patcher/模型选项,
    仅替换条件, 不重复加载模型。"""
    import copy
    original = getattr(guider, "original_conds", None)
    if original is None or getattr(guider, "model_patcher", None) is None:
        raise Exception("AnyTile 分批采样: 不支持的guider类型, 需要 CFGGuider 系列 (BasicGuider/CFGGuider/DualCFGGuider)")
    new = copy.copy(guider)
    new.original_conds = {}
    for k, v in original.items():
        new.original_conds[k] = _slice_conditioning(v, start, length, total)
    return new


# ---------------------------------------------------------------- 节点 5: 分批采样 (SamplerCustomAdvanced 兼容)
class AnyTileBatchSamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "噪声源 (RandomNoise 等), 与 SamplerCustomAdvanced 相同"}),
                "guider": ("GUIDER", {"tooltip": "引导器 (BasicGuider/CFGGuider/DualCFGGuider), 与 SamplerCustomAdvanced 相同"}),
                "sampler": ("SAMPLER", {"tooltip": "采样器选择, 与 SamplerCustomAdvanced 相同"}),
                "sigmas": ("SIGMAS", {"tooltip": "sigma表 (ManualSigmas/KarrasScheduler 等), 与 SamplerCustomAdvanced 相同"}),
                "latent_image": ("LATENT", {"tooltip": "AnyTile输出的tile latent批次"}),
                "tile_batch": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1,
                                       "tooltip": "每批采样的tile数量; 1=一次一张(16GB卡推荐), 原SamplerCustomAdvanced是整批一次前向"}),
                "接缝延续": ("BOOLEAN", {"default": False,
                                       "tooltip": "采样级接缝融合: 每块tile采样时把邻居重叠区的一部分(宽度=切分节点的 接缝贴片比例)贴进本tile外缘并完全钉死, 生成时延续邻居结构; 该贴片区解码后带绿块/色偏, 由拼接节点整块丢弃 (需接layout; tile_batch强制为1)"}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "接上=分批VAE解码, 直接输出tile图像批次(接AnyTile拼接); 不接=tiles_image输出占位图"}),
                "layout": (ANYTILE_LAYOUT, {"tooltip": "AnyTileSplit 输出的切分布局; 接缝延续需要"}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("output", "tiles_image")
    OUTPUT_TOOLTIPS = (
        "采样完成的完整latent批次 (等价SamplerCustomAdvanced的output)",
        "分批解码后的tile图像批次, 接AnyTile拼接 (未接vae时为占位图)",
    )
    FUNCTION = "sample"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = (
        "SamplerCustomAdvanced的分批版: noise/guider/sampler/sigmas 全套组件原样接入, 内部按'每批N个tile'循环采样。\n"
        "接缝延续开启后(接layout), 每块tile采样时携带已采样邻居的重叠latent与羽化噪声遮罩, 接缝在采样阶段即融合。"
    )

    def sample(self, noise, guider, sampler, sigmas, latent_image, tile_batch, 接缝延续=False, vae=None, layout=None):
        import comfy.sample
        import comfy.model_management
        import latent_preview

        latent = latent_image
        samples_in = latent["samples"]
        n = int(samples_in.shape[0])
        seed = int(noise.seed)

        seam = bool(接缝延续) and layout is not None
        if bool(接缝延续) and layout is None:
            log.warning("[AnyTile] 接缝延续已开启但未接layout, 本次不启用")
        tb = 1 if seam else max(1, int(tile_batch))
        if seam:
            layout["seam_applied"] = True   # 通知拼接节点: 外缘被贴过邻居内容, 需要丢弃绿块区
            log.info("[AnyTile] 接缝延续启用: 逐tile依赖采样 (tile_batch 强制为1)")

        # 通道校正 + 全量噪声一次生成(确定性), 再按批切片 —— 与整批一次跑噪声完全一致
        full_latent = dict(latent)
        fixed = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, samples_in,
            latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None),
        )
        full_latent["samples"] = fixed
        full_noise = noise.generate_noise(full_latent)

        lh, lw = int(fixed.shape[2]), int(fixed.shape[3])
        sig_len = int(sigmas.shape[-1]) - 1
        sampled = {}
        outs = []
        done = 0
        for i0 in range(0, n, tb):
            m = min(tb, n - i0)
            g = _slice_guider(guider, i0, m, n)
            callback = latent_preview.prepare_callback(g.model_patcher, sig_len, {})
            chunk_latent = fixed[i0 : i0 + m].clone()
            chunk_mask = latent["noise_mask"][i0 : i0 + m].clone() if "noise_mask" in latent else None
            if seam:
                for k in range(i0, i0 + m):
                    init_k, smask = _seam_constraint(fixed, k, layout, sampled, lh, lw)
                    chunk_latent[k - i0 : k - i0 + 1] = init_k  # 局部索引!
                    if float(smask.min()) < 1.0:  # 有实际约束才设置noise_mask
                        if chunk_mask is not None:
                            chunk_mask[k - i0 : k - i0 + 1] = torch.minimum(
                                chunk_mask[k - i0 : k - i0 + 1], smask)
                        else:
                            chunk_mask = smask
            s = g.sample(
                full_noise[i0 : i0 + m], chunk_latent, sampler, sigmas,
                denoise_mask=chunk_mask, callback=callback, disable_pbar=False, seed=seed,
            )
            s = s.to(comfy.model_management.intermediate_device())
            if seam:
                for k in range(i0, i0 + m):
                    sampled[k] = s[k - i0 : k - i0 + 1]
            outs.append(s)
            done += m
            log.info("[AnyTile] 分批采样进度 %d/%d tile", done, n)
            comfy.model_management.soft_empty_cache()

        samples = torch.cat(outs, dim=0)
        out = dict(latent)
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        if vae is not None:
            imgs = []
            for i0 in range(0, n, max(1, tb)):
                imgs.append(vae.decode(samples[i0 : i0 + max(1, tb)]))
                comfy.model_management.soft_empty_cache()
            image = torch.cat(imgs, dim=0)
        else:
            log.warning("[AnyTile] 未接vae, tiles_image输出为占位图, 请自行分块解码latent")
            image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        log.info("[AnyTile] 分批采样完成: %d块, 每批%d | latent=%s", n, tb, tuple(samples.shape))
        return (out, image)


# ---------------------------------------------------------------- 调试节点: 接缝延续预览 (不采样)
class AnyTileSeamDebugPreview:
    """调试: 模拟采样器接缝延续的初始内容 (不走采样器)。

    用 tile 原图代替"邻居已采样latent", 按 _seam_constraint 同款几何/阶梯mask规则,
    在像素空间合成每块tile的模拟图 (锁定区=邻居原图内容, 自由区=自身内容),
    并输出对应 noise_mask (latent级阶梯上采样到像素), 供肉眼检查锁定区/台阶位置。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_images": ("IMAGE", {"tooltip": "AnyTile切分放大输出的 tile_images 原始tile批次"}),
                "layout": (ANYTILE_LAYOUT, {"tooltip": "AnyTileSplit 输出的切分布局"}),
                "接缝延续": ("BOOLEAN", {"default": True,
                                       "tooltip": "开=按接缝延续规则合成(锁定区放邻居内容); 关=输出原图(对照组)"}),
                "对齐倍率": ("INT", {"default": 16, "min": 4, "max": 64, "step": 4,
                                   "tooltip": "latent下采样倍率 (Qwen-2.1=16, SD/Flux=8); 决定阶梯mask的像素级形状, 与采样器实际latent一致"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("sim_tiles", "noise_mask")
    OUTPUT_TOOLTIPS = (
        "模拟采样器初始内容的tile图 (锁定区=邻居原图内容, 自由区=自身内容)",
        "对应的noise_mask (latent级阶梯上采样到像素; 黑=锁定, 白=自由)",
    )
    FUNCTION = "preview"
    CATEGORY = "bofengttp/AnyTile"
    DESCRIPTION = "调试工具: 不走采样器, 在像素空间复现采样器接缝延续的初始内容与noise_mask, 用于检查锁定区/台阶位置是否正确。"

    def preview(self, tile_images, layout, 接缝延续, 对齐倍率):
        if not isinstance(layout, dict) or layout.get("type") != "anytile_layout":
            raise Exception("layout 不是 AnyTileSplit 的输出")
        count = layout["count"]
        imgs = tile_images.cpu().float()
        if int(imgs.shape[0]) != count:
            raise Exception(f"tile 数量不匹配: 布局 {count} 块, 实际 {int(imgs.shape[0])} 块")
        th, tw = layout["tile_height"], layout["tile_width"]
        if int(imgs.shape[1]) != th or int(imgs.shape[2]) != tw:
            raise Exception(f"tile 尺寸 {int(imgs.shape[2])}x{int(imgs.shape[1])} 与布局 {tw}x{th} 不符")

        positions = layout["positions"]
        cols = layout["cols"]
        dr = max(1, int(对齐倍率))
        sim = imgs.clone()
        masks = torch.ones((count, 1, th, tw), dtype=torch.float32)
        sampled = {}  # 模拟"已采样" (像素空间)
        strip_x = int(layout.get("seam_strip_x", -1))
        strip_y = int(layout.get("seam_strip_y", -1))

        for idx in range(count):
            if 接缝延续:
                r, c = divmod(idx, cols)
                x, y = positions[idx][0], positions[idx][1]
                # 左邻: 左侧重叠条带
                if c > 0 and (idx - 1) in sampled:
                    ov_px = min(tw, max(0, int(round(tw - (x - positions[idx - 1][0])))))
                    wl = ov_px if strip_x < 0 else min(ov_px, max(0, strip_x))
                    if ov_px > 0 and wl > 0:
                        nl = sampled[idx - 1][:, :, tw - ov_px : tw - ov_px + wl, :]  # 重叠起点对齐
                        if strip_x < 0:
                            ov_lat = max(1, int(round(ov_px / dr)))
                            m_px = torch.repeat_interleave(_strip_mask_pattern(ov_lat), dr)[:wl].to(sim.dtype)
                            sim[idx : idx + 1, :, :wl, :] = nl * (1.0 - m_px.view(1, 1, wl, 1)) + sim[idx : idx + 1, :, :wl, :] * m_px.view(1, 1, wl, 1)
                            masks[idx, :, :, :wl] = torch.minimum(masks[idx, :, :, :wl], m_px.view(1, wl))
                        else:
                            sim[idx : idx + 1, :, :wl, :] = nl      # 贴片: 整块继承
                            masks[idx, :, :, :wl] = 0.0             # 全钉死(拼接时丢弃)
                # 上邻: 顶部重叠条带
                if r > 0 and (idx - cols) in sampled:
                    ov_px = min(th, max(0, int(round(th - (y - positions[idx - cols][1])))))
                    hl = ov_px if strip_y < 0 else min(ov_px, max(0, strip_y))
                    if ov_px > 0 and hl > 0:
                        nt = sampled[idx - cols][:, th - ov_px : th - ov_px + hl, :, :]
                        if strip_y < 0:
                            ov_lat = max(1, int(round(ov_px / dr)))
                            m_px = torch.repeat_interleave(_strip_mask_pattern(ov_lat), dr)[:hl].to(sim.dtype)
                            sim[idx : idx + 1, :hl, :, :] = nt * (1.0 - m_px.view(1, hl, 1, 1)) + sim[idx : idx + 1, :hl, :, :] * m_px.view(1, hl, 1, 1)
                            masks[idx, :hl, :, :] = torch.minimum(masks[idx, :hl, :, :], m_px.view(1, 1, hl))
                        else:
                            sim[idx : idx + 1, :hl, :, :] = nt
                            masks[idx, :hl, :, :] = 0.0
            sampled[idx] = sim[idx : idx + 1].clone()

        log.info("[AnyTile-Debug] 接缝延续预览: %d块 | 对齐倍率%d | 接缝延续=%s | 贴片 %d/%d px(0=整条重叠)",
                 count, dr, 接缝延续, max(0, strip_x), max(0, strip_y))
        return (sim, masks.squeeze(1).clamp(0.0, 1.0))


NODE_CLASS_MAPPINGS = {
    "AnyTileSplit": AnyTileSplit,
    "AnyTileBatchSampler": AnyTileBatchSampler,
    "AnyTileBatchSamplerCustom": AnyTileBatchSamplerCustom,
    "AnyTileMerge": AnyTileMerge,
    "AnyTilePerTileTextEncode": AnyTilePerTileTextEncode,
    "AnyTileSeamDebugPreview": AnyTileSeamDebugPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTileSplit": "AnyTile 图像切分放大",
    "AnyTileBatchSampler": "AnyTile 分批采样器",
    "AnyTileBatchSamplerCustom": "AnyTile 分批采样 (Custom)",
    "AnyTileMerge": "AnyTile 拼接",
    "AnyTilePerTileTextEncode": "AnyTile 逐Tile文本编码 (Qwen)",
    "AnyTileSeamDebugPreview": "AnyTile 接缝延续调试预览",
}

__all__ = ["AnyTileSplit", "AnyTileBatchSampler", "AnyTileBatchSamplerCustom", "AnyTileMerge",
           "AnyTilePerTileTextEncode",
           "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]




