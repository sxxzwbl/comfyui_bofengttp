# comfyui_bofengttp — AnyTile 切分放大

把图切成带重叠的方形 tile，分批采样，再羽化拼回完整大图。为高分辨率放大设计。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sxxzwbl/comfyui_bofengttp.git
```

重启 ComfyUI，节点出现在 `bofengttp/AnyTile` 分类下。无额外依赖（只用 ComfyUI 自带的 numpy / torch / PIL / cv2）。

## 使用

示例工作流：[`examples/qwen-image-2.1-tile-sampling.json`](examples/qwen-image-2.1-tile-sampling.json)，拖进画布即可（用到 rgthree 的 Image Comparer，没装可删掉那个节点）。

```
LoadImage ─► AnyTileSplit ─┬─ tile_images ─────────────────┬─► AnyTileMerge ─► 成片
                           │                              │        ▲
                           ├─ upscaled_image ─────────────┘        │
                           │                                       │
                           └─ tiles_latent ─► AnyTileBatchSamplerCustom ─► tiles_image
                                                     ▲
                            AnyTilePerTileTextEncode ─┘（positive/negative 接 guider）
```

| 节点 | 干什么 | 关键输出 |
|---|---|---|
| **AnyTile 图像切分放大** | 等比放大到目标分辨率 → 按重叠切块 | `tile_images` 未采样分块 / `tiles_latent` 采样起点 / `upscaled_image` 放大整图 / `layout` 几何 |
| **AnyTile 分批采样 (Custom)** | 按批采样 + 解码（也可用非 Custom 版走 KSampler 流程） | `tiles_image` 采样后的分块 |
| **AnyTile 逐Tile文本编码** | 每块 tile 用自己的图做视觉参考，输出 batch=N 的条件 | `positive` / `negative` / `latent` |
| **AnyTile 拼接** | 颜色还原 + 遮罩羽化 + 拼回大图 | `merged_image` |
| **AnyTile 接缝延续调试预览** | 不采样，直接看贴片几何 | `sim_tiles` / `noise_mask` |

`layout` 是几何载体，采样器与拼接节点都从它读参数，不要重复填。

## 数值意义

### AnyTileSplit

| 参数 | 默认 | 数值含义 |
|---|---|---|
| `scale_mode` | 放大倍数 | `放大倍数`=按整数倍等比放大；`目标分辨率`=把长边放到指定像素 |
| `放大倍数` | 2 | 1~8 的整数倍，仅 `放大倍数` 模式生效 |
| `目标分辨率` | 2048 | **长边**像素，自动吸附到 512 的倍数；短边按原图比例算。填 4096 意味着成片长边 ≈4096 |
| `tile_size` | 1024 | 每块边长（像素）。档位 512/768/1024/1280/1536/2048/2560 —— 这些是模型的无偏移尺寸，**不要自造**。块数 ≈ 目标长边 ÷ tile_size |
| `overlap` | 128 | 横向相邻两块的**重叠像素**（16 对齐）。越大接缝越稳，但这圈像素会被算两遍，利用率下降 |
| `overlap_y` | 0 | 纵向重叠像素；`0` = 与 `overlap` 相同。两轴目标长度不同，才需要分开给 |
| `interp` | lanczos | 纯插值放大算法（未接 upscale_model 时使用） |
| `利用率优先` | true | 开 = 自动解两轴 overlap 让画布正好铺满覆盖能力（利用率 90%+）；关 = 旧算法（块数 = `ceil(目标/tile)`，重叠靠缩画布吸收） |
| `目标容差` | 0.05 | 成片尺寸允许偏离目标的比例，0.05 = ±5% |
| `非等比预算` | 0.10 | 允许画布偏离目标的容差，与 `目标容差` 取较大者；0.10 = ±10% |
| `拉回目标` | true | 开 = 成片严格等于目标分辨率（差额靠缩放吸收）；关 = 成片 = 实际画布，零缩放，尺寸可能小 5~10% |
| `接缝贴片比例` | 0.5 | 接缝延续时贴给下一块的条带宽度，占重叠的比例；0 = 关闭该优化，0.5 = 贴一半、融合一半 |
| `vae`（可选） | — | 接上才输出真正的编码 latent；不接则给空 latent（配 denoise=1 用） |
| `upscale_model`（可选） | — | 接上先用 ESRGAN 类模型放大，再对齐到画布尺寸 |

### AnyTileMerge

| 参数 | 默认 | 数值含义 |
|---|---|---|
| `颜色还原强度` | 1.0 | 每块与参考图做逐通道增益 + 偏移还原，1.0 = 全部应用，0 = 关闭 |
| `遮罩拓展百分比` | 5 | 贴回遮罩向内收缩，像素 = tile 平均边长 × 5%（tile 1024 → 约 51px）；0 = 只做重叠线性羽化 |
| `遮罩羽化百分比` | 5 | 收缩后再做高斯羽化，同样的换算尺度；接缝可见就加大 |
| `丢弃接缝贴片` | true | 丢弃采样时继承邻居内容的外缘条带；仅当采样器开了 `接缝延续` 时需要 |
| `背景图`（可选） | — | 只接 `AnyTileSplit` 的 `tile_images`（第 i 块对第 i 块）或 `upscaled_image`（取同位置区域）。不接则不校色 |

### 采样节点

| 参数 | 默认 | 数值含义 |
|---|---|---|
| `tile_batch` | 1 | 每次前向吃几块，1 = 最省显存 |
| `接缝延续` | false | 开 = 每块携带已采样邻居的内容做锚定（强制 `tile_batch=1`） |
| `denoise` | 1.0 | 1.0 = 全新生成；0.3~0.5 = 精修/放大（配合接 `vae` 的编码 latent） |
| `ref_image_size` | 自动 | 逐 tile 编码时每块视觉参考的面积上限，自动 = 1024² |

### 日志怎么看

```
[AnyTile] 输入 6364x1462 -> 目标 7168x1648 | 画布 7184x1648 (偏差 +0.22%/+0.00%, 拼接拉回目标) |
          tile 1024x1024 网格 8x2=16块 重叠 144/400 | 利用率 70.6%
[AnyTile] 颜色还原: 16块 | 中位增益 0.965/0.964/0.982 中位偏移 -6.1/-6.0/-8.8 级
```

- **利用率** = 画布面积 ÷ (块数 × tile²)。低于 85% 说明目标比例与方形 tile 不匹配，换 `tile_size` 比调别的有效
- **重叠 144/400**：两轴实际生效的重叠。两数差得离谱（如 144 vs 400）就是同一个信号——短边不是 tile 的整数倍
- **偏差**：画布与目标的差，`拉回目标` 开着时由拼接缩放吸收
- **中位增益/偏移**：校色实际做了什么。增益恒 1.000、偏移 0.0 = 参考图与采样结果一致，没有整体偏色可校
