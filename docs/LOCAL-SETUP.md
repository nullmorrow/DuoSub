# 本地组件与路径

公开代码使用通用默认路径，不包含某台电脑的部署目录。
将根目录的 `duosub.local.example.json` 复制为 `duosub.local.json`，只填写本机需要覆盖的项目。
此文件被忽略，禁止强制加入 Git；已有工具界面中的路径设置仍然有效。

| 配置键 | 用途 | 通用默认值 |
|---|---|---|
| `SUBKIT_ASSFONTS` | 批量字体嵌入工具 | `assfonts` |
| `SUBKIT_ASR_MODEL_DIR` | 已下载的 Whisper 模型目录 | 用户缓存下的 `whisper` |
| `SUBKIT_ASR_PACKAGES` | 可选独立 Python 包目录 | 留空，使用当前 Python 环境 |
| `TORCH_HOME` | 已下载的人声分离模型缓存 | 用户缓存下的 `torch` |
| `SUBKIT_TRANSLATION_GGUF` | 本地翻译模型 | 用户缓存下的 `duosub` 目录 |
| `SUBKIT_LLAMA_CLI` | llama.cpp CLI | `llama-cli` |

同名环境变量优先于本地 JSON；缺失／无效的 JSON 使用默认值。相对路径相对于启动时的工作目录，建议本地配置填实际绝对路径。
本地配置不修改全局环境或 Python 安装；听译子进程读取相同配置。
字体库默认从系统的 LocalAppData 下 `assfonts/fonts.json` 获取，也可在程序中选择。
FFmpeg/FFprobe 仍使用 PATH 或已有的 `SUBKIT_FFMPEG`／`SUBKIT_FFPROBE`。

配置文件只保存路径，不应保存密码、令牌或其他凭据。模型兼容性与首次安装需自行验证；程序不自动下载模型。

## 真实素材测试

少数测试和 `tools/eval` 评测脚本需要真实字幕与媒体，统一从根目录的 `sample-media/` 读取，例如 `sample-media/少女革命ウテナ/`。
该目录被忽略，不会进入 Git；可以用目录联接指向本机的素材目录（例如在命令提示符中 `mklink /J sample-media <素材目录>`）。
缺少素材时这些测试会自动跳过，不算失败；测试只读取素材，不会修改原文件。
