# 与 Astra 的协作和交换（r1）

## 共同基线

- Baseline ID：`SubKit-Claude-20261002-e85c0d7221354442`（Astra 于 2026-10-01T22:12:04Z 固定）。
- Opus 侧已核对：`baseline/manifest.json` 列出的 133 个文件，原工程、Astra 快照、Opus 快照三方 SHA-256 完全一致；
  汇总哈希按“manifest 顺序的 `path\tsha256` 行用 `\n` 连接（无结尾换行）”复现一致。复核命令：
  `本地配置路径`，结果在 `reports/baseline-verification.json`。
- 基线不含 `sources/jp/*.srt`（日文原料，由 jp-extract 从压缩包提取）。测试工程直接只读引用它们，哈希记在测试夹具清单里。

## 交换方式

1. 工程格式以 `docs/FORMAT-v1.md` 为提案。Astra 如果有自己的提案，把差异逐条写进下面的兼容性清单；确认前双方都不要把分歧固化进保存格式。
2. 交换单位是整个工程目录（不含 `journal/`）。读取方先运行校验器；认不出的字段原样保留。
3. 共用测试工程：`test_projects/` 下的小工程（只读引用原料），附带期望的不变式（顺序、台账、哈希）。双方都应该能读取并通过校验。
4. 合并主干时：比较两边对同一测试工程的输出和用户操作次数（REQUIREMENTS 十九节末尾的人工评价重点），择优迁移模块，不长期维护两套后端。

## 兼容性清单（待双方填写）

| 项目 | Opus 提案 | Astra | 状态 |
|---|---|---|---|
| 时间单位 | 毫秒整数 | | 待确认 |
| 来源事件键 | `<来源ID>:<文件内序号>` | | 待确认 |
| 字符身份 | 来源 run（src,e,a,b）/ 插入 run（ins,a,b） | | 待确认 |
| 文档顺序 | 显式有序的 `groups` 列表 | | 待确认 |
| 未配对的表示 | 派生（Group 内只有一种语言） | | 待确认 |
| 语气词 Comment | `hidden_filler`，`position` 指明位置 | | 待确认 |
| 删除 | 墓碑 | | 待确认 |
| 读音优先级 | FORMAT 第 10 节 | | 待确认 |
| 样式优先级 | FORMAT 第 9 节 | | 待确认 |
| 台词 ↔ 歌曲转换 | Group / Member 的 `before_convert`、Member 的 `karaoke_off`、Karaoke 的 `dormant`（FORMAT §6.1 / §6.2 / §6.7） | | 待确认 |
| K 轴人工拆合 | Karaoke 的 `edits`：`split` / `merge`，以底字身份或片段底字身份 + 读音偏移/校验定位边界（FORMAT §6.7 r3） | | 待确认 |
| K 轴译文吸附 | 拍的 `link.start` / `link.end` 引用同组原声拍的 `core/k/sung` 键与 `edge`；同事务跟随、区间内比例插值，失效删除并写轨道 `issues`（FORMAT §6.7） | | 待确认 |
