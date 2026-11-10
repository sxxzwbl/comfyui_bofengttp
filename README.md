# comfyui_bofengttp — AnyTile 切分放大

把图按方形 tile 切分、分批采样、羽化拼回大图。核心是**画布正好铺满 tile 覆盖能力**：不为省一块 tile 缩小画布再拉伸，避免旧做法 10~12% 的隐藏失真。适用于 ComfyUI，仅依赖自带的 numpy / torch / PIL / cv2。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sxxzwbl/comfyui_bofengttp.git
```

重启后节点在 `bofengttp/AnyTile` 分类下。

## 节点

| 节点 | 作用 | 输出 |
|---|---|---|
| **AnyTile 图像切分放大** | 等比放大到目标分辨率 → 按重叠切块 | `tile_images` / `tiles_latent` / `upscaled_image` / `layout` |
| **AnyTile 分批采样器** | KSampler 流程，按批采样 + 解码 | `tiles_image` / `samples` |
| **AnyTile 分批采样 (Custom)** | 兼容 SamplerCustomAdvanced 组件式流程 | `output` / `tiles_image` |
| **AnyTile 拼接** | 颜色还原 + 遮罩羽化 + 拼回大图 | `merged_image` |
| **AnyTile 逐Tile文本编码 (Qwen)** | 每块 tile 用自己的图做视觉参考 | `positive` / `negative` / `latent` |
| **AnyTile 接缝延续调试预览** | 不采样，直接看贴片几何与 mask | `sim_tiles` / `noise_mask` |

`layout` 是切分几何的载体，采样器与拼接节点都从它读参数，不必重复填。

## AnyTileSplit 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `scale_mode` | 放大倍数 | `放大倍数`=整数倍等比；`目标分辨率`=长边放到指定像素 |
| `目标分辨率` | 2048 | 长边像素，吸附到 512 倍数 |
| `tile_size` | 1024 | 档位：512/768/1024/1280/1536/2048/2560。这些尺寸在 Qwen 系无像素偏移，不要自造 |
| `overlap` | 128 | 横向重叠下限（16 对齐），算法可加大以收回画布 |
| `overlap_y` | 0 | 纵向重叠下限；`0`=与 `overlap` 相同。两轴目标长度不同，必须分开解 |
| `interp` | lanczos | 纯插值算法 |
| `利用率优先` | true | 开=双轴 overlap 反解，画布=覆盖能力；关=旧算法 |
| `目标容差` | 0.05 | 允许成片偏离目标的比例 |
| `非等比预算` | 0.10 | 与 `目标容差` 取较大者作为画布容差 |
| `拉回目标` | true | 开=输出严格等于目标分辨率；关=输出=实际画布（零拉伸，尺寸可能小 5~10%） |
| `接缝贴片比例` | 0.5 | 接缝延续时贴给邻居的条带宽度；0=关闭该优化 |

可选输入：`vae`（接上则输出编码 latent）、`upscale_model`（ESRGAN 类先放大再对齐）。

## 布局：画布 = 覆盖能力

单轴覆盖长度 `= tile × n − overlap × (n − 1)`，对固定 `n` 是关于 overlap 的一次函数，因此反解可得

```
overlap = (tile × n − 目标) / (n − 1)    # 再对齐 16
利用率 η = 画布面积 / (块数 × tile²)
```

算法枚举两轴各自的 `n` 与 `overlap`，按 **η 最高 → 偏差最小 → 块数最少** 排序。三条硬约束：`overlap ≥` 用户设定且 `≤ tile/2`；画布不低于 `tile`（绝不缩 tile，会掉出无偏移档位）；反解越界时换块数/换档位，不采用"缩画布再拉回"。

实测（4.35:1 → 长边 7168，overlap 128）：

| | 块数 | 画布 | 利用率 | 几何 |
|---|---|---|---|---|
| 旧算法 | 14 | 6400×1648 | 72% | 横向拉伸 +12.0% |
| 利用率优先，tile 1024 | 16 | **7184×1648** | 71% | 零拉伸，有效像素 +12% |
| 利用率优先，tile 1536 | **5** | 7168×1536 | **93%** | 纵向 6.8% 非等比（在预算内） |

**想真正提高利用率，优先换 `tile_size` 而不是调分辨率**：目标短边是否接近 `tile × n` 才是浪费根源。上例短边 1648 配 1024 的 tile，纵向必然浪费 20%；换 1536 档一行就铺满。

## 接线

**纯切分放大**

```
LoadImage → AnyTileSplit → tile_images / upscaled_image → AnyTileMerge(tiles, layout)
```

**Qwen 编辑式放大**

```
LoadImage → AnyTileSplit(接 vae) → tiles_latent → AnyTileBatchSamplerCustom(接缝延续=on, layout)
                       ↓                                    ↑
              tile_images ── AnyTilePerTileTextEncode ── guider
                                                            ↓
                                     AnyTileMerge(tiles, layout, 背景图=upscaled_image)
```

## 示例工作流

[`examples/qwen-image-2.1-tile-sampling.json`](examples/qwen-image-2.1-tile-sampling.json) —— Qwen-Image 2.1 turbo 的分块放大 / 修图流程，拖进 ComfyUI 画布即可。

- 切分：目标长边 4096 / tile 1024 / overlap 128，`利用率优先` 与 `拉回目标` 全开
- 采样：`tile_batch=1`、接缝延续开（euler_ancestral + 12 档 ManualSigmas）
- 拼接：颜色还原 1.0、遮罩拓展/羽化 3%、`丢弃接缝贴片` 开

需自备模型：`qwen_image_2.1_turbo_int8_convrot`(UNET)、`qwen_image_2.1_vae_bf16`(VAE)、`qwen3vl_8b_int8_convrot`(CLIP)、`High quality`(LoRA, 0.4)。

## 调参

| 目标 | 做法 |
|---|---|
| 最高清晰度（≤9s 段） | `tile_size=1536`，overlap 保持 128 |
| 15s 长段 | tile 用 ≤1024，降低 `overlap` 换更少块 |
| 成片尺寸必须精确 | `拉回目标=true`（默认） |
| 只要真像素、可接受尺寸浮动 | `拉回目标=false` |
| 完全回旧行为 | `利用率优先=false` |

日志每次打印 `画布(偏差%) / 网格 / 重叠 x/y / 利用率%`。两轴重叠差得离谱（如 `144/400`）说明比例与方形 tile 不匹配，**换档位比调参数有效**；日志还会提示 overlap 减半能少切几块。

## 兼容与限制

- 旧 layout（v2）自动回落到旧逻辑；`利用率优先=false` 时行为与历史版本一致；
- 不走 `AnyTileBatchSamplerCustom` 而直接 layout 接拼接时，把 `丢弃接缝贴片` 关掉；
- 仅支持单张图（`batch=1`）；非档位 tile 尺寸会有像素偏移；
- 纵向块数 >1 且短边不是 tile 整数倍时，纵向重叠会被动加大（如实记录在 `overlap_y`），这是结构性代价，只能靠换档位消掉。
