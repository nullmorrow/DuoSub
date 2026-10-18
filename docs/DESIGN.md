# SubKit-Opus 技术设计（r1）

## 1. 技术选型

| 部分 | 选择 | 理由 | 否决的方案 |
|---|---|---|---|
| 语言 | Python 3.12 | 基线核心（ASS 无损解析、Sudachi 读音、LaBSE 对齐、libass 测量排版、拼音）都是 Python，经过 Utena 金标准评测；整体重写成 C++/C# 会丢掉这些积累 | C++/Qt（重写量大）、C#/WPF（核心需要跨进程） |
| 界面 | **PySide6 6.11（Qt Widgets）**，本机已安装 | 真正的原生桌面控件（不是网页）：停靠窗口、高 DPI、快捷键、撤销框架、自绘控件（波形、带振假名的文本）都成熟；LGPLv3 | Tk（自绘振假名和复杂列表太弱）、Electron/Tauri/WebView（任务书禁止）、QML（可以，但 Widgets 对编辑器类软件更直接） |
| 播放 | **libmpv**（官方推荐的 shinchiro 构建，用户同意后下载）+ 自写 ctypes 绑定 | 视频/音频同步播放、逐帧、音轨选择，并且**用 libass 实时渲染正在编辑的 ASS**——和发布后用户在播放器里看到的一致；嵌入 Qt 窗口（`wid`） | Qt Multimedia（不能渲染 ASS，要自己叠图层，修改后刷新慢） |
| 波形 | ffmpeg 解码所选音轨 → numpy 峰值金字塔 → QPainter 自绘 | 不依赖额外库；缓存到工程 `cache/` | — |
| 排版测量 | 基线的 libass 实测（ffmpeg `subtitles` 滤镜批量渲染）+ sqlite 缓存 | 与 mpv 同一渲染器，测量结果就是播放器的结果 | 用 Qt 字体度量（和 libass 不一致） |
| 打包（第四阶段） | PyInstaller 单目录 → `DuoSub.exe` | 双击即用；在那之前用 `DuoSub.pyw`（双击运行，不需要 PYTHONPATH） | — |

模型：不自动启动 Ollama/Qwen，不自动下载模型。LaBSE（已在本机缓存）和 whisper 都是**用户明确开启**的可选增强；
没有它们、断网时，导入、合并（按时间）、编辑、播放、导出全部可用。

## 2. 模块

```
src/subkit/            核心（无界面依赖，可单元测试）
  ass/ encoding.py textutil.py      基线：ASS 无损解析/写出、编码检测（沿用）
  reading/ ruby/ pinyin.py gloss.py karaoke.py songs.py   基线：读音、排版、拼音、旁注、卡拉 OK（逐步改成“只给建议”）
  align/               对齐：基线特征 + v2 动态规划（保序、无 swap、整句/多行组、切分代价）
  model/               新：工程格式 v1（ids、来源、文档、run、注音锚点、序列化、校验：顺序/台账/引用）
  core/                新：导入（只读来源 + 结构化元数据）、合并（来源 → 初始文档）、编辑操作（带撤销）、导出 ASS
src/subkit_app/        界面（PySide6）
  player/mpv.py        libmpv ctypes 绑定 + 嵌入控件
  waveform.py          波形与时间条
  views/               Group 列表、Group 编辑区、原文面板、读音弹窗
  dialogs/             新建工程、选择原料、合并设置、导出
tests/                 单元测试 + 验收测试（REQUIREMENTS 第十九节）
tools/                 基线校验、评测脚本
third_party/mpv/       libmpv（来源与许可证见 SOURCE.md）
```

基线的 `workfile.py`（work.ass + state.json 流程）、`cli.py`、`release.py` 只作为旧工程导入和行为对照保留，不再是主流程。

## 3. 编辑模型

- 所有修改都是**操作**（命令模式）：执行、撤销、重做；每个操作带 `op_` ID，写入 `journal/ops.jsonl`（崩溃恢复）。
- 文档有修订号；每个成员也有修订号。后台任务（读音分析、排版测量、波形、对齐建议）的结果带着它所基于的修订号，
  **只在修订号没变时才应用**，否则丢弃重算——不会用过期结果覆盖刚改的内容。
- 读音、注音、K 轴挂在字符身份上（格式 6.3），移动文字时跟着走；底字被改写时标记失效并提示。

## 4. 预览管线

文档 → 预览 ASS（成员文本 + 用缓存的测量结果生成的振假名/拼音/旁注）→ 临时文件 → mpv `sub-reload`。
- 修改后 300 ms 去抖；只重新测量改动过的成员（sqlite 缓存按“样式 + 文本”命中）。
- 测量未完成时预览先用上一次的排版，状态栏显示“注音排版更新中”。
- 发布导出和预览用同一个生成函数，区别只在“去掉工程数据”的那一步。

## 5. 播放器

- libmpv 在 Qt 的原生子窗口里渲染（`wid` = 控件的窗口句柄）；事件通过 `mpv_set_wakeup_callback` 转成 Qt 信号，在主线程处理。
- 观察属性：`time-pos`、`pause`、`duration`、`track-list`、`estimated-vf-fps`；命令：`loadfile`、`seek … absolute+exact`、
  `frame-step`/`frame-back-step`、`ab-loop`（重播本组）、`sub-add`/`sub-reload`、`set aid`。
- 时间以 mpv 的 `time-pos`（秒，双精度）为准，换算到毫秒；帧率来自容器，用于逐帧微调，避免累计漂移。

## 6. 线程

主线程只做界面和应用结果。QThreadPool 里跑：ffmpeg 解码波形、Sudachi 分析（每个线程一个分词器实例）、libass 测量、
LaBSE 向量（第一次使用时才导入 torch）。每个任务都能取消，并在状态栏显示进度。

## 7. 许可证（详见 THIRD_PARTY.md）

PySide6/Qt：LGPLv3（动态链接，可替换）。libmpv：GPLv2+ 构建。ffmpeg：外部进程调用。SudachiPy、Sudachi 词典：Apache-2.0。
sentence-transformers/LaBSE：Apache-2.0。whisper：MIT。pypinyin、rapidfuzz：MIT。numpy：BSD。**不使用 Aegisub 的代码**，
只参考它的交互习惯（快捷键、布局）。SubKit 自身的许可证待用户决定；若与 libmpv 一起分发，需与 GPL 兼容。

## 8. 测试

- 单元测试：格式校验（顺序、台账、引用）、导入无损（字节哈希不变、run 文本与来源一致）、对齐不变式、编辑操作的撤销。
- 验收测试：REQUIREMENTS 第十九节 20 条，能自动化的写成测试，其余写成人工操作清单（`docs/ACCEPTANCE.md`）。
- 真实回归：Soma、Utena、Gundam、NANA 第一话（只读）；Utena 金标准评测改为同时报告“逐行一致率”和“多行组一致率”，
  并始终报告顺序违规数（必须为 0）和隐藏内容数。
