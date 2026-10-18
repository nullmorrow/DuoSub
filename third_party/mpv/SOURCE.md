# libmpv（第三方组件来源记录）

用途：编辑器内的视频/音频播放，以及用 libass 实时渲染正在编辑的字幕（与 mpv、MPC-HC+libass、Aegisub 的 libass 预览同一渲染器）。
用户在 2026-10-02 明确同意下载（选项“下载官方 libmpv（推荐）”）。

| 项目 | 内容 |
|---|---|
| 来源 | https://github.com/shinchiro/mpv-winbuild-cmake/releases/tag/20261001 （mpv 官网 Windows 构建页推荐的自动构建） |
| 文件 | `mpv-dev-x86_64-20261001-git-3186d369f9.7z`（31 514 910 字节） |
| 压缩包 SHA-256 | `de0aa24a39e27b07d9f663616f8d09c91586c7328d8c33fbfbc8e3888c35d89c`（与 GitHub release 页面公布的 digest 一致） |
| 解压后 | `dev/libmpv-2.dll`（120 886 784 字节，SHA-256 `ec77593d23735f36eb08e486154e89cf1cc6a5650327dab125736c1bb5d3af40`）、`dev/include/mpv/*.h`、`dev/libmpv.dll.a` |
| mpv 版本 | git 3186d369f9，client API 2.5 |
| 下载时间 | 2026-10-02 08:17 (UTC+10) |

## 许可证

- mpv 源码默认按 **GPLv2 或更高版本** 授权（用 `-Dgpl=false` 构建时为 LGPLv2.1+）。这个自动构建没有声明是 LGPL 构建，
  并且静态链接了 FFmpeg、libass（ISC）、HarfBuzz、FreeType 等，**整体按 GPLv2+ 对待**。
- SubKit 通过 ctypes 在同一进程里调用 libmpv。本机自用没有问题；**如果以后要把 SubKit 连同这个 DLL 一起分发**，
  需要遵守 GPL（随附许可证文本、提供对应源码的获取方式：上面的 release 与 mpv/FFmpeg 的对应提交），
  并让 SubKit 自身采用与 GPL 兼容的许可证。SubKit 的许可证尚未决定（待用户决定），见 `docs/THIRD_PARTY.md`。
- 不使用任何 Aegisub 代码。

## 替换/升级

把新的 `libmpv-2.dll` 放到 `dev/` 下（或在设置里指定路径），更新本文件的版本和哈希。程序启动时会检查 client API 主版本为 2。
公开仓库不包含 DLL 或下载压缩包。按上述来源取得组件，并在本地核对版本与哈希。
