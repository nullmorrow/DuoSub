# 双语字幕工坊 · DuoSub

面向 Windows 的本地双语字幕编辑器，使用 Python 3.12、PySide6、libmpv 和 FFmpeg/libass。
将原声字幕与翻译字幕组织成组，编辑时间、样式、日文振假名、中文拼音和歌词 K 轴，导出 ASS。
原字幕保留只读来源副本，人工编辑、读音、配对和时间调整可撤销。

## 启动

1. 安装 Python 3.12，并按 `requirements.txt` 安装依赖。
2. 将 FFmpeg/FFprobe 放入 PATH，或设置 `SUBKIT_FFMPEG`／`SUBKIT_FFPROBE`。
3. 按 [libmpv 获取说明](third_party/mpv/SOURCE.md)准备播放器组件。
4. 双击 `DuoSub.pyw`，新建工程并导入视频及字幕。

本仓库不分发视频、模型、播放器 DLL 或个人工程；可选 AI 功能不自动下载模型。
本机工具和模型路径可写进不入 Git 的 `duosub.local.json`，参见 [本地配置](docs/LOCAL-SETUP.md)。

## 主要功能

- 原声／译文成组导入、保序合并、改字、配对、调轴和撤销。
- F10 看片打轴、F11 观剧编辑、波形和频谱、字幕位置与 ASS 标签编辑。
- 日文分词、振假名、人物读音和 Project 词库；同一分词按整词居中和限宽。
- 拼音、旁注、样式模板，显示与导出时的标点、半宽假名及批量替换规则。
- 歌词 K 轴与模块化特效，单集／整季 ASS 导出、参考简繁字幕及可读报告。
- 审议包导出／导回；可选本地 AI 听译和独立标注为“未经音频确认”的翻译候选。

完整操作说明见 [用户手册](docs/manual/index.html)，程序内帮助菜单和 F1 也可打开。
核心代码在 `src/subkit`，界面在 `src/subkit_app`；更改记录见 [CHANGELOG](CHANGELOG.md)。

## 开发与检查

```powershell
python -m unittest discover -s tests -t .
git config core.hooksPath .githooks
python tools/privacy_check.py --cached
```

部分测试需要本机 FFmpeg、libass、字体或可选依赖；真实媒体验收另行配置，不随仓库发布。
提交与发布约定见 [CONTRIBUTING](CONTRIBUTING.md)。

## 第三方

依赖和播放器构建的许可证／来源见 [第三方说明](docs/THIRD_PARTY.md)。
本项目许可证尚未指定；公开源码不代表自动授予第三方组件的再分发许可。
