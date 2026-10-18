# 开发与发布

公开仓库为 `https://github.com/nullmorrow/DuoSub`。
提交身份统一使用 `nullmorrow <344519723+nullmorrow@users.noreply.github.com>`。
请保留必要的第三方许可证和署名，不将第三方作者误当成本项目的个人身份删除。

首次克隆后启用检查：

```powershell
git config core.hooksPath .githooks
```

提交前检查索引内容；推送前检查可达历史和目标仓库。检查不会打印疑似凭据的具体值。
禁止跳过检查或强制添加本机配置、原片、模型、用户工程、运行日志、内部交接和回退归档。
历史身份的额外阻止词可放进不入 Git 的 `.privacy-blocklist.local.json`。

新功能同步用户手册、快捷键说明和更改记录，保留原字幕、人工编辑及撤销语义。
测试须如实区分自动测试、真实渲染与原生界面／听觉验收；不要把旧版本结果当作新版本全套通过。
