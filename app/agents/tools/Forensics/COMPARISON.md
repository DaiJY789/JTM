# ForenAgent 与现有 image 工具对比

本目录是 ForenAgent `low_level_tool` 的原样快照；`../ifd` 是本项目当前的深度伪造
检测工具实现。二者定位不同：前者提供可组合的低层图像算子，后者提供可直接接入智能体的
结构化取证工具。

## 能力对应

| ForenAgent 工具 | 主要用途 | `image` 中的对应或互补实现 |
| --- | --- | --- |
| `bayar_filter` | 固定预测残差，突出局部篡改痕迹 | `SRM_Filter` 的残差滤波器组覆盖相近目标，但没有同一 Bayar 核 |
| `jpeg_ghost_map` | 多质量重压缩差异图 | 当前没有直接对应实现，是有价值的补充 |
| `srm_residual_bank` | 4 个 SRM 残差核 | `SRM_Filter` 使用更完整的滤波器组，并输出统计量、区域和证据结构 |
| `fft_highpass_residual` | FFT 高频残差 | `Spectrum_Grid_Analyzer` 做加窗频谱、共轭峰和周期网格分析，语义更丰富 |
| `dwt_hh_map` | DWT 对角高频子带 | `Noise_Analyzer` 使用 Haar/DWT 估计噪声异常并定位区域 |
| `local_correlation_map` | 局部相关性异常 | 没有直接对应实现，可作为合成纹理辅助证据 |
| `prnu_residual` | 单图小波去噪残差 | `Noise_Analyzer` 有近似噪声残差；真正 PRNU 设备归因仍需要参考图像集合 |
| `median_filter_trace_score` | 中值滤波痕迹启发式分数 | 当前没有直接对应实现 |
| `resampling_periodicity_score` | 重采样周期性启发式分数 | `Spectrum_Grid_Analyzer` 可发现部分周期频谱，但不是同一检测方法 |
| `dct_highpass_filter` | DCT 高频残差 | 当前没有直接对应实现 |
| `sobel_filter` | 一阶梯度边缘图 | 现有工具内部会使用边缘信息，但没有单独暴露 Sobel 工具 |
| `highpass_filter` | Laplacian 高频图 | 与 `SRM_Filter`、`Noise_Analyzer` 的部分残差信息重叠 |

`ifd` 目录还独有以下能力：`PatchDense_Scanner` 的疑似复制移动区域检测、
`CFA_Validator` 的 CFA/去马赛克一致性核查，以及 `ROI_Extractor`、`Image_Resizer`、
`Visual_Enhancer` 三个工作流辅助工具。

## 优劣结论

### ForenAgent 的优势

- 12 个函数短小、输入输出接近 NumPy 算子，适合研究、消融和快速组合。
- 覆盖 JPEG ghost、中值滤波痕迹、显式重采样周期、DCT 高频等现有目录缺少的信号。
- 大部分单个算子计算轻量，不需要模型权重或外部 API。

### ForenAgent 的不足与已核实风险

- 返回裸数组或单个启发式分数，没有统一的成功状态、置信度、异常区域、可视化产物和
  运行元数据，不能直接作为当前 Agent 的工具返回值。
- 文档/Schema 与运行行为不完全一致：`bayar_filter`、`jpeg_ghost_map`、
  `srm_residual_bank`、`local_correlation_map`、`prnu_residual`、
  `median_filter_trace_score`、`resampling_periodicity_score` 声明支持字符串路径，实测会失败。
- `bayar_filter`、`jpeg_ghost_map`、`local_correlation_map` 未执行文档声明的枚举校验；
  `srm_residual_bank` 的非法索引抛出 `IndexError`，而不是文档中的 `ValueError`。
- `dct_highpass_filter` 的注释说明 DCT 直流分量位于左上角，但实现却以图像中心构造
  圆形低频掩模，实际没有按描述去除左上角低频分量。
- `dwt_hh_map` 使用 `coeffs[level]`。按 `pywt.wavedec2` 的系数顺序，多层分解时这会
  取到较细尺度的细节，而不是文档所称的指定粗尺度；层级语义需要修正和回归测试。
- `srm_residual_bank`、`local_correlation_map`、`resampling_periodicity_score` 含 Python
  像素循环或多角度旋转，对大图缺少尺寸上限，延迟会随分辨率明显增长。
- 上游阈值是未经本项目数据集校准的启发式值。随机噪声样本也可能超过重采样阈值，
  因此不能把单一分数直接转换成“伪造”结论。
- 同步提交没有许可证文件；在获得作者明确授权前，不应默认具备再分发或商业使用权。

### 现有 `image` 工具的优势

- 已封装成 `BaseTool`/OpenAI Schema，具有可信路径检查、参数清洗、错误处理和统一 JSON
  证据协议，可被当前视觉智能体直接调用。
- 输出包括统计量、适用性、候选区域、可选热力图和耗时信息，比裸残差图更适合最终解释。
- 频谱、SRM、噪声、CFA、复制移动五类信号互补，并设置了下采样或最大边长，运行边界更明确。
- 对证据能力的表述更保守：启发式分数标记为未校准，频谱阳性不单独下最终结论。

### 现有 `image` 工具的不足

- 代码和输出协议更复杂，维护成本高；开启产物导出时还会产生磁盘 I/O。
- `PatchDense_Scanner` 明显比其他工具慢，应继续只在疑似复制移动时选择性调用。
- 阈值同样尚未在目标评测集上完成系统校准，不能只靠规则分数代替最终模型判断。
- 对 JPEG 重压缩历史、中值滤波痕迹和显式重采样周期的覆盖弱于 ForenAgent。

## 建议采用方式

不要用 ForenAgent 整体替换 `image`。较稳妥的路径是保持本目录为可追溯的上游快照，
优先挑选 `jpeg_ghost_map`、`median_filter_trace_score`、
`resampling_periodicity_score` 三项增量能力，在修复接口和算法问题后另建适配层，使其输出
`image/common.py` 定义的证据协议。Bayar/SRM/FFT/Sobel/Laplacian 等重叠算子只作为内部
特征，不宜全部暴露给 Agent，以免增加工具选择成本并重复计算。

当前同步仅完成代码落盘和包导出，**尚未把这些函数注册进 Deepfake Agent 的工具列表**。
在完成许可证确认、算法修正和目标数据集校准前，这是有意保留的隔离边界。
