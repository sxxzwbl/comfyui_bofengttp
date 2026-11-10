# comfyui_bofengttp — AnyTile 切分放大节点组

把一张图按方形 tile 切分、分批采样、再羽化拼回完整大图。为高分辨率放大与 Qwen-Image 类编辑模型设计，重点解决两件事：

1. **tile 利用率**——画布正好铺满覆盖能力，不为省一块 tile 而缩小画布再拉伸；
2. **接缝污染**——采样器贴进 tile 外缘的邻居内容会解码成绿块/色偏，拼接时整块丢弃，由邻居补上。

适用于 ComfyUI（`custom_nodes` 规范，无外部依赖，仅用 numpy / torch / PIL / cv2，均为 ComfyUI 自带）。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sxxzwbl/comfyui_bofengttp.git
```

重启 ComfyUI，节点出现在 `bofengttp/AnyTile` 分类下。

---

## 节点清单

| 节点 | 作用 | 输入 | 输出 |
|---|---|---|---|
| **AnyTile 图像切分放大** `AnyTileSplit` | 等比放大到目标分辨率 → 按重叠切块 | `image`、放大与切分参数、`vae`/`upscale_model`(可选) | `tile_images`、`tiles_latent`、`upscaled_image`、`layout` |
| **AnyTile 分批采样器** `AnyTileBatchSampler` | 经典 KSampler 流程，按批采样并解码 | `model`/`latent_image`/`vae`/`positive`/`negative`/采样参数/`接缝延续` | `tiles_image`、`samples` |
| **AnyTile 分批采样 (Custom)** `AnyTileBatchSamplerCustom` | 兼容 `SamplerCustomAdvanced` 组件式流程 | `noise`/`guider`/`sampler`/`sigmas`/`latent_image`/`tile_batch`/`接缝延续` | `output`、`tiles_image` |
| **AnyTile 拼接** `AnyTileMerge` | 颜色还原 + 遮罩羽化 + 丢弃接缝污染区，拼回大图 | `tiles`、`layout`、遮罩参数、`丢弃接缝贴片`、`背景图`(可选) | `merged_image` |
| **AnyTile 逐Tile文本编码 (Qwen)** `AnyTilePerTileTextEncode` | 每块 tile 用自己的图做视觉参考，输出 batch=N 的条件 | `clip`、`tile_images`、`prompt`、`ref_image_size`、`vae`(可选) | `positive`、`negative`、`latent` |
| **AnyTile 接缝延续调试预览** `AnyTileSeamDebugPreview` | 不采样，直接看贴片几何与 noise_mask | `tile_images`、`layout`、`接缝延续`、`对齐倍率` | `sim_tiles`、`noise_mask` |

`layout` 是节点间传递的切分布局字典（`ANYTILE_LAYOUT`），采样器与拼接节点都从它读取几何，**不需要重复填参数**。

---

## AnyTileSplit 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `scale_mode` | 放大倍数 | `放大倍数` = 按整数倍等比放大；`目标分辨率` = 长边放到指定像素（等比） |
| `目标分辨率` | 2048 | 长边像素，自动吸附到 512 的倍数 |
| `tile_size` | 1024 | 方形 tile 档位：512 / 768 / 1024 / 1280 / 1536 / 2048 / 2560。这些尺寸在 Qwen 系模型上无像素偏移，**不要自造尺寸** |
| `overlap` | 128 | 横向重叠下限（16 对齐）。利用率优先模式下是**下限**，算法可加大以把画布收回到目标 |
| `overlap_y` | 0 | 纵向重叠下限；`0` = 与 `overlap` 相同。两轴目标长度不同，必须能分开解 |
| `interp` | lanczos | 纯插值放大的算法 |
| `利用率优先` | true | 开启 = 双轴 overlap 反解，画布 = 覆盖能力；关闭 = 旧算法（块数 = `ceil(边长/tile)`，重叠靠缩画布吸收） |
| `目标容差` | 0.05 | 允许成片偏离目标的比例 |
| `非等比预算` | 0.10 | 与 `目标容差` 取较大值，作为画布偏离目标的容差 |
| `拉回目标` | true | 开 = 拼接后严格输出目标分辨率（差额靠缩放吸收）；关 = 成片 = 实际画布（零拉伸，但尺寸可能小 5~10%） |
| `接缝贴片比例` | 0.5 | 见下节。0 = 回旧行为（整条重叠贴、不丢弃） |

---

## 布局算法（利用率优先）

单轴覆盖长度：

```
覆盖长度 = tile × n − overlap × (n − 1)
```

对固定 `n` 这是关于 `overlap` 的一次函数，所以能反解出让画布**正好等于目标长度**的 overlap：

```
overlap = (tile × n − 目标) / (n − 1)      # 再对齐到 16 的倍数
```

由此得到 tile 利用率：

```
η = 画布面积 / (块数 × tile²)
```

算法枚举两轴各自的 `n` 与 `overlap`，按 **η 最高 → 偏差最小 → 块数最少** 排序。三条硬约束：

- `overlap ≥ 用户设定值`（不擅自降低接缝冗余）且 `≤ tile/2`；
- 画布不低于 `tile`，**绝不把 tile 缩成图尺寸**（会掉出模型的无偏移档位）；
- 反解越界时**换块数或换档位**，不采用"缩小画布再让拼接拉回"的旧做法。

旧做法的问题：`块数 = ceil(目标/tile)` 而 `n` 块最多覆盖 `tile×n − ov×(n−1)`，不足部分被"缩画布 + 拼接拉回"补齐，表现为 **10~12% 的隐藏横向拉伸**（画布 6400 被拉回 7168）。利用率优先模式直接去掉了这一步。

实测（4.35:1 → 长边 7168，tile 1024，overlap 128）：

| | 块数 | 画布 | 利用率 | 几何 |
|---|---|---|---|---|
| 旧算法 | 14 | 6400×1648 | 72% | 横向拉伸 +12.0% |
| 利用率优先 | 16 | **7184×1648** | 71% | 零拉伸，有效像素 +12% |
| 利用率优先 + tile 1536 | **5** | 7168×1536 | **93%** | 纵向 6.8% 非等比（在预算内） |

**要点：想真正提高利用率，优先换 `tile_size` 而不是调分辨率。** 目标短边是否接近 `tile × n` 才是浪费的根源——4.35:1 的短边 1648 配 1024 的 tile，纵向必然浪费 20%；换成 1536 档，一行就铺满。

---

## 接缝贴片（绿块处理）

`接缝延续` 开启时，采样器把邻居已采样 latent 的一段贴进当前 tile 的外缘做锚定。这段 latent 与本地 latent 不连续，VAE 解码后会在 tile 外缘出现**绿块/色偏**（调整 mask 阶梯无效，问题在解码端而非去噪端）。

对策：**只贴重叠区的一部分，拼接时整块丢弃**。

```
重叠区 overlap
├──── 融合带 = overlap − strip ────┤
├──── 贴片区 strip ────┤
     ↑ 采样阶段贴给下一块，mask 全 0 钉死
     ↑ 拼接阶段权重归零，由邻居覆盖
```

- `strip = ceil(overlap × 接缝贴片比例)` 并对齐到 16，且钳制到 `overlap − 16`，保证 **融合带 ≤ 贴片宽度**；
- 融合带仍有 `overlap − strip ≥ strip` 的宽度供两块内容羽化混合，相邻 tile 不会留缝；
- 采样器贴片时，邻居 latent 的取片起点固定为"重叠起点"（`lw − ov_lat`），与画布坐标严格对齐——这点在缩窄贴片后必须成立，否则内容会左右错位；
- 拼接节点在羽化**之后**把贴片区权重置零，并在颜色还原前用参考图替换该区，避免绿块污染整块 tile 的校色统计。

实测（灰底 + 接缝条带涂纯绿模拟污染）：

| 接缝贴片比例 | 贴片宽度 | 融合带 | 输出绿像素 | 空洞 |
|---|---|---|---|---|
| 0（旧行为） | 144/400 px | — | 1.757% | 有暗斑 |
| 0.5（默认） | 80/208 px | 64 px | **0.000%** | 无 |
| 0.7 | 112/288 px | 32 px | **0.000%** | 无 |
| 0.9 | 128/368 px | 16 px | **0.000%** | 无 |

另跑 54 组（6 个目标尺寸 × 3 个比例 × 3 组遮罩参数），覆盖率全部完整。

---

## 典型接线

**A. 纯切分放大（不重绘）**

```
LoadImage → AnyTileSplit(upscale_model 可选) → tile_images / upscaled_image
                                               ↓
                                     AnyTileMerge(tiles=tile_images, layout=layout)
```

**B. Qwen-Image 编辑式放大**

```
LoadImage → AnyTileSplit(接 vae) → tiles_latent → AnyTileBatchSamplerCustom(接缝延续=on, layout)
                       ↓                                    ↓
              tile_images ── AnyTilePerTileTextEncode ── guider
                                                            ↓
                                              AnyTileMerge(tiles, layout, 背景图=upscaled_image)
```

**C. 排查接缝**

```
AnyTileSplit → AnyTileSeamDebugPreview(接缝延续=on) → 看 sim_tiles / noise_mask
```

---

## 调参建议

| 目标 | 做法 |
|---|---|
| 要最高清晰度（≤9s 段） | `tile_size=1536`，`overlap` 保持 128；单块前向行数是 1024 档的 2.25 倍，仍在线性预算内 |
| 15s 长段 | tile 只能用 ≤1024（1536 档行数超标会花屏），降低 `overlap` 换取更少的块 |
| 接缝明显 | 提高 `接缝贴片比例`（融合带更窄）、或降低 `tile_size`、或开 `接缝延续` |
| 成片尺寸必须精确 | `拉回目标=true`（默认） |
| 只要真像素、不介意尺寸浮动 | `拉回目标=false` |
| 完全回旧行为 | `利用率优先=false` + `接缝贴片比例=0` |
| 快出图 | 看日志里的 `调参提示`：它会直接告诉你 overlap 减半能少切几块 |

日志每次都会打印：`输入/目标 → 画布(偏差%) | tile 网格 块数 重叠 x/y | 利用率%`。两个重叠值差距很大（如 `144/400`）说明比例与方形 tile 不匹配，**换档位比调参数有效**。

---

## 兼容性

- 旧 layout（v2，无 `seam_strip_*` 字段）自动回落到旧逻辑：整条重叠贴 + 60/40 阶梯 mask + 拼接不丢弃；
- `利用率优先` 关闭时走的是原 `_compute_layout`，行为与历史版本一致；
- 跳过 `AnyTileBatchSamplerCustom`、直接用 layout 接拼接时，把 `丢弃接缝贴片` 关掉，否则会白丢一圈真实像素。

## 已知限制

- 仅支持单张图（`batch=1`），多张请先拆批次；
- 方形 tile 的档位是模型无像素偏移的前提，非档位尺寸会有偏移/质量下降；
- 纵向块数 >1 且短边不是 tile 整数倍时，纵向重叠会被动加大（算法会如实记录在 `overlap_y`），这部分算力是结构性代价，只能靠换档位消掉。

## 版本

- `layout` v3：`overlap_x/overlap_y/eta/err_w/err_h/seam_strip_x/seam_strip_y/seam_applied`
- 节点参数向后兼容：新增项一律追加在 `required` 末尾，旧工作流的 `widgets_values` 不会错位
