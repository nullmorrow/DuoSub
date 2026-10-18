# 第三方组件与许可证

| 组件 | 用途 | 许可证 | 获取方式 | 分发注意 |
|---|---|---|---|---|
| Python 3.12 | 运行时 | PSF | 本机已装 | — |
| PySide6 6.11 / Qt 6.11 | 原生界面 | LGPLv3（也有 GPL/商业） | 本机已装（pip） | 动态链接、允许用户替换 Qt 库；随附许可证 |
| libmpv（mpv git 3186d369f9） | 视频/音频播放、libass 实时字幕 | GPLv2+（此构建按 GPL 对待） | 用户同意后从 shinchiro/mpv-winbuild-cmake 下载，见 `third_party/mpv/SOURCE.md` | 分发时需遵守 GPL：随附许可证、提供源码获取方式；SubKit 自身许可证需与 GPL 兼容 |
| ffmpeg（chocolatey 构建） | 音频解码（波形）、libass 测量 | GPL 构建（外部进程调用，不链接） | 本机已装 | 不随程序分发；用户自装或另行说明 |
| SudachiPy 0.6 + sudachidict_full | 日文分词与读音 | Apache-2.0 | 本机已装 | 随附许可证 |
| fugashi + unidic-lite | 读音回退 | MIT / BSD | 本机已装 | |
| sentence-transformers + LaBSE | 可选：中日语义相似度 | Apache-2.0 | 本机已有缓存；**不自动下载** | 用户明确开启才使用 |
| openai-whisper（medium） | 可选：配音辅助读音建议 | MIT | 本机已有模型；**不自动下载** | 用户明确开启才使用 |
| 本机 Qwen3.8-27B-UD-IQ4_XS GGUF + llama.cpp b11146 | AI听译：根据中文生成日语语义候选 | 模型 Apache-2.0（本机已有下载元数据）；llama.cpp MIT | 复用用户配置的本地文件，不随程序分发、不自动下载；独立 CLI 进程 | 不将翻译候选当作音频听写；不开启通用 Agent／模型路由或常驻服务；分发时另核对对应模型与量化版本许可证 |
| pypinyin | 中文拼音 | MIT | 本机已装 | |
| opencc-python-reimplemented 0.1.7 | 简体 → 繁体（Bangumi 导入的人名） | Apache-2.0（OpenCC 词典）/ MIT | 用户 2026-10-04 同意后 pip 安装 | 随附许可证 |
| Bangumi API（api.bgm.tv） | 可选：按作品导入人物的日文名、中文名 | 在线服务，数据按 Bangumi 的使用条款 | 只在用户点「从 Bangumi 导入」时联网 | 不缓存图片；请求带程序名和版本的 User-Agent |
| rapidfuzz | 文本相似度 | MIT | 本机已装 | |
| Pyphen（可选，en_US 词典） | 英文歌词按音节拆拍 | Pyphen：GPL 2.0+ / LGPL 2.1+ / MPL 1.1；词典按各自许可证 | 不默认安装、不自动下载；仅本机可导入时使用，否则用元音规则 | 如随程序分发，随附所选代码许可证及 en_US 词典许可证 |
| numpy / scipy | 数值计算 | BSD | 本机已装 | |

**Aegisub**：不使用其代码或二进制（包括它附带的 VSFilter.dll），只参考交互习惯。

SubKit 本身的许可证：**待用户决定**。如果将来要连同 libmpv 一起分发，建议选 GPLv3 或与之兼容的许可证。
