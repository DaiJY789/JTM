# 分类演示样例

本次新增 **MFC 10 个、IFD 10 个**，按任务和标签分目录。已有 `MMD-010/`、`OpenFake-54/` 保留，未计入本次新增的 20 个。

```text
MFC/
  True/          2 个
  Mostly True/   2 个
  Mixture/       2 个
  Mostly False/  2 个
  False/         2 个
IFD/
  Real_Image/       4 个
  Forged_Image/     3 个
  Synthetic_Image/  3 个
```

MFC 的 `.txt` 是原始正文，按各例 README 顺序上传图片；IFD 每例只上传一张图片。`question.txt` 是中性检测问题；`README.md` 和 `sample.json` 是给演示者看的参考资料，不作为模型输入。

## MFC 样例

| 标签 | 样例 | 内容 | 演示重点 |
|---|---|---|---|
| True | [MFC-136](MFC/True/MFC-136/README.md) | 沃尔玛收购 Vizio | 商业报道与品牌配图一致 |
| True | [MFC-101](MFC/True/MFC-101/README.md) | 北海道暴风雪警报 | 简短天气消息与警报说明图 |
| Mostly True | [MFC-201](MFC/Mostly%20True/MFC-201/README.md) | JAPAN JAM 公演取消 | 核心事件真实，配图时间语境需要区分 |
| Mostly True | [MFC-469](MFC/Mostly%20True/MFC-469/README.md) | 月全食与红月画面 | 基本事实与视觉强化并存 |
| Mixture | [MFC-493](MFC/Mixture/MFC-493/README.md) | AMD 营收指引与行情图 | 真实财报信息与编辑行情图混合 |
| Mixture | [MFC-306](MFC/Mixture/MFC-306/README.md) | 花滑组合退役与新闻字幕 | 真实退役消息、字幕编辑与场景错置 |
| Mostly False | [MFC-296](MFC/Mostly%20False/MFC-296/README.md) | MacBook 新品发布海报 | 促销线索被包装为官方发布 |
| Mostly False | [MFC-006](MFC/Mostly%20False/MFC-006/README.md) | 邮轮病例与专利、品牌拼图 | 真实监测背景与错误关联及拼接图 |
| False | [MFC-178](MFC/False/MFC-178/README.md) | 霍尔木兹海峡冲突现场 | 未经证实的事件叙事与生成图片 |
| False | [MFC-496](MFC/False/MFC-496/README.md) | 巴黎圣日耳曼夺冠与比分图 | 虚构夺冠画面、比分与文字冲突 |

所选 MFC 样例合计覆盖 T1–T3、V1–V2、C1–C3 八种标注维度。

## IFD 样例

| 标签 | 样例 | 内容 | 来源与分类依据 |
|---|---|---|---|
| Real_Image | [IFD-01](IFD/Real_Image/IFD-01/README.md) | 餐食近景 | OpenFake 原始 real 标注 |
| Real_Image | [IFD-02](IFD/Real_Image/IFD-02/README.md) | 林地中的鸟类 | OpenFake 原始 real 标注 |
| Real_Image | [IFD-03](IFD/Real_Image/IFD-03/README.md) | 道路、车辆与晚霞 | OpenFake 原始 real 标注 |
| Real_Image | [IFD-04](IFD/Real_Image/IFD-04/README.md) | 微距与浅景深 | OpenFake 原始 real 标注 |
| Forged_Image | [IFD-05](IFD/Forged_Image/IFD-05/README.md) | 专利、品牌与新闻画面拼接 | MFC 图片编辑标注派生 |
| Forged_Image | [IFD-06](IFD/Forged_Image/IFD-06/README.md) | 花滑新闻字幕被编辑 | MFC 图片编辑标注派生 |
| Forged_Image | [IFD-07](IFD/Forged_Image/IFD-07/README.md) | 船只画面添加爆炸与烟雾 | MFC 图片编辑标注派生 |
| Synthetic_Image | [IFD-08](IFD/Synthetic_Image/IFD-08/README.md) | 生成的海上冲突画面 | MFC 图片生成标注派生 |
| Synthetic_Image | [IFD-09](IFD/Synthetic_Image/IFD-09/README.md) | 机器兔与野生动物场景 | MFC 图片生成标注派生 |
| Synthetic_Image | [IFD-10](IFD/Synthetic_Image/IFD-10/README.md) | 生成的足球夺冠庆典 | MFC 图片生成标注派生 |

## 标签和来源

- MFC 保留本地 `data/mmd/dataset_metadata.json` 的原始 `verdict`，不把原始素材目录改名。
- IFD 的真实类使用 OpenFake 的 `real` 标注。OpenFake 的 `fake` 没有直接作为“合成”使用；篡改、合成类均选自有 V2/V1 且说明明确指向所选图片的 MFC 记录，三分类属于派生的演示参考标签。
- 素材的原始文件路径、标签依据、原始说明及 SHA-256 保存在 `manifest.json` 与每例 `sample.json`。正文和图片原样复制；素材整理阶段没有重新生成图片或运行检测；最新实测见下方记录。
- 参考标签用于组织演示素材，不表示当前模型必然输出该标签，也不替代对新闻事实或图片来源的独立核验。

## 实测结果 · 20261111-203718

20 个样例已按当前默认问题完成真实检测，详见 [本轮完整报告](runs/20261111-203718/report.md)。参考标签与实际预测分开保留。
