# SubKit 工程格式 v1（Opus 提案，草案 r1）

状态：**提案**，给 Astra 评审。双方确认前，字段可能改名；确认后按第 12 节的规则升级版本。
目的：同一个测试工程能被两边读取，或通过双方确认的无损迁移读取（REQUIREMENTS 十七.5）。
文中“必须/不得”是规范性要求，校验器 `subkit.model.validate` 逐条检查。

## 1. 原则

1. **原料只读**：来源字幕的字节不改。工程里保存路径、SHA-256，并在工程内放一份字节完全相同的副本。
2. **一切可追溯**：工作字幕里的每一个字符都知道自己从哪来（来源事件 + 偏移，或用户输入）。来源的每一个非空白字符都有可解释的去向（第 8 节）。
3. **顺序是数据，不是时间**：文档顺序是一个显式的 Group 列表。时间可以重叠，顺序检查按来源索引做（第 7 节）。
4. **派生物不入正文**：振假名、拼音、旁注、K 轴、特效是挂在字符上的结构化数据，导出 ASS 时才生成。
5. **人工决定优先保存**：删除、隐藏、确认、例外都是记录，任何刷新/重算都不能复活或覆盖它们。

## 2. 目录

```
<工程目录>/                         （与程序分离，用户自选位置）
  subkit-project.json                工程清单
  templates/<模板ID>/v<版本>.json    样式模板（版本化，第 9 节）
  dict/readings.json                 项目词库（读音规则）
  dict/pinyin.json                   项目拼音规则（逐字音节，§6.6）
  dict/characters.json               人物
  dict/passages.json                 重复段落库（含歌曲记忆，§6.8）
  dict/passages/<段落ID>.npz          段落音频特征
  dict/songs.json                    旧声明，未使用
  episodes/<集ID>/
    episode.json                     本集的原料、媒体、设置
    sources/<来源ID>.<扩展名>        原料字节副本（只读，哈希与 episode.json 一致）
    sources/<来源ID>.events.json     解析结果：事件 + 结构化元数据片段（可由字节重建）
    document.json                    工作文档（Group、成员、注音……）
    journal/                         自动保存与崩溃恢复（实现自定，不参与交换）
    exports/                         导出结果与报告
    review/answers.json              审议意见的人工处理记录（§2.1）
```

所有 JSON：UTF-8（无 BOM），`format` + `format_version` 字段必有。路径用 `/`。时间单位一律**毫秒整数**（ASS 的厘秒 ×10 精确表示；SRT 的毫秒原样）。

### 2.1 `review/answers.json`

```json
{"format": "subkit-review-answers", "format_version": 1,
 "items": [{"fingerprint": "SHA-256 十六进制", "decision": "accepted", "at": "2026-10-05T00:00:00+00:00"}]}
```

这是本集审议意见的人工决定，`decision` 为 `accepted`（采用）、`rejected`（不采用）或 `handled`（已处理）。
`at` 是带时区的 ISO 8601 时间。文件缺失表示尚无决定；按指纹保存最新决定，再次导入标为已处理。
指纹为规范化后的 `[group, lang, kind, target, old, new]` JSON 数组（UTF-8、不转义 Unicode、无多余空白）的 SHA-256；
理由不参与指纹。未知种类规范化为 `other`，语言按注册表规范化（泛指中文的 `zh` 保留），正文片段及读音不改写。
解析和匹配只产生建议；应用与决定记录由调用方在用户选择后显式执行。记录不自动改字幕，字幕采用由文档撤销事务保存。

## 3. ID

- 工程内唯一的字符串，带类型前缀：`prj_` `g_` `m_` `c_` `a_` `k_` `i_`（用户插入）`op_`（操作）`chr_`（人物）`rule_` `psg_`（重复段落）。
  推荐格式：前缀 + 12 位十六进制毫秒时间 + 8 位十六进制随机数。实现可以不同，但**只能当作不透明字符串使用**。
- 来源 ID 是集内的短名：`t1`（翻译）、`o1`（原声）……；来源事件键 `<来源ID>:<事件序号>`，序号是该事件在文件中的 0 起顺序
  （ASS/SSA：`[Events]` 里所有 Dialogue 和 Comment 行按出现顺序；SRT：块的出现顺序，不是块号）。
- ID 一经写入不得复用；拆分产生新 ID，被合并的 ID 记入操作历史，不能悬空引用。

## 4. `subkit-project.json`

```json
{
  "format": "subkit-project", "format_version": 1,
  "id": "prj_0192abcd1234_5f3e9a01",
  "name": "食戟のソーマ",
  "writer": {"app": "SubKit-Opus", "version": "0.2.0"},
  "media_roots": ["sample-media/example"],
  "languages": {"original": "ja", "translation": "zh-Hant"},
  "seasons": [{"id": "S01", "title": "第一季", "settings": {}}],
  "episodes": [{"id": "S01E001", "season": "S01", "number": 1, "path": "episodes/S01E001"}],
  "settings": {"merge": {}, "ruby": {}, "release": {}},
  "template": {"id": "tpl_main", "version": 2, "path": "templates/tpl_main/v2.ass",
               "scheme": {"fonts": {"cn_hans": "…", "cn_size": 68, "cn_mv": 65, "…": "…"},
                          "palette": {"text": "A81F5F", "outline": "FFFFFF", "…": "…"}}},
  "dictionaries": {"readings": "dict/readings.json", "characters": "dict/characters.tsv",
                   "characters_json": "dict/characters.json", "songs": "dict/songs.json"}
}
```

设置继承：工程 `settings` → 季 `settings` → 集 `settings`（后者覆盖前者的同名键，键级合并）。

### 显示规则：标点和空格

`settings.display.punctuation` 按 `translation` / `original` 保存字符标准化和显示字体两层规则，各自按行类别判断作用范围；不修改文档。
每套规范字段如下（未列出的值按默认处理）：

- `comma`: `off` / `spaces` / `ideographic`；`dunhao`、`period`、`spaces`、`separator`：布尔值，默认 false。
- `ellipsis`: `off` / `single` / `double` / `keep_count`；`exclaim`、`question`: `off` / `full` / `half`。
- `quotes`: `off` / `corner` / `white_corner` / `nested` / `nested_inverse`。
- `tilde`: `off` / `full`（～）/ `half`（~）/ `wave`（〜）；处理 ~、～、〜、∼，保留个数。
- `dash`: `off` / `half`（-）/ `em`（—）/ `choon`（ー）/ `bar`（―）/ `fullwidth_hyphen`（－）；连续破折号保留个数。
- `scope_chars`、`scope_fonts`: `dialogue` / `song` / `other` 的数组，空数组表示该层停用；默认分别为 `["dialogue"]`、`["dialogue", "song", "other"]`，缺少字段的旧工程也使用此默认值。
  `dialogue` 包括 dialogue/top/alt，`song` 包括 song/song_jp/song_cn，其余为 `other`，与多行合并分类相同。
- `fonts`: 对象，键为 `comma`、`dunhao`、`period`、`colon`、`ellipsis`、`exclaim`、`question`、`quotes`、`brackets`、`separator`、`tilde`、`dash`、`long_vowel`；值为字体名或 null（沿用行字体）。

字体覆盖转换后的该类所有标点，包含未换字形的标点；相邻同字体标点合并为一个 ASS `\fn` 区段，随后恢复来源、字符格式或样式字体。
小数点、英文缩写/英文之间的点、数字千位分隔号、英文单词/数字间的连字符及 Re- 的末尾连字符不归入相应标点类。
假名后「ー」「ｰ」及延续它们的长音仅归入 `long_vowel`，不参与字符标准化；独立或汉字后的「ー」仍归入 `dash`；`{@Raw=…}` 和 ASS 绘图不受规则或字体影响。
歌曲的 K 标签保留并按显示位置映射，字体在振假名排版测量前插入，振假名事件不套用标点字体；竖排歌词同样适用。
完整字符集、类别优先级和选项见 `SPEC-2026-10-13-PUNCT.md` §1、§2、§5。

兼容读取：旧 `comma=true` 为 `comma=spaces`，未设 `dunhao` 时同时开启；旧 `quotes=true` 为 `nested`。
只有缺少 `fonts` 字段时，旧 `font` 迁移为省略号、感叹号、问号、引号四类字体；作用范围扩大为这四类全部标点。旧设置（有内容，但没有 `fonts`、`scope_fonts`、`scope_chars`、`tilde`、`dash` 中任何一个）的 `scope_fonts` 默认为只有 `dialogue`，旧工程的歌词和屏幕字导出不变；空设置或新写入的设置默认三类都有。
`PunctuationRules.to_settings()` 写入全部规范字段、两层作用范围及十三类 `fonts`（沿用为 null），不写旧 `font`；此兼容读取不自动改写工程文件。

`settings.pinyin = {"mode": "off" | "PY" | "ZY" | "WG"}` 是工程级拼音开关，缺省为 `off`。
`PY` 显示声调符号、`ZY` 显示注音符号、`WG` 显示威妥玛；关闭只停止自动生成和显示，已存注音保留。
开启在当前集给所有 `lang` 以 `zh` 开头的成员生成自动拼音，与 `lane` 无关；模式和生成合为一步撤销。

`template.scheme`：可视化模板编辑器的全部数值（`subkit.templates.FontSet` / `Palette` 的字段），重新打开编辑器时原样恢复；
没有它的工程从模板文件读回。换模板写新版本文件，旧版本文件保留。
`template.scheme.fonts.ruby_offset` / `pinyin_offset` 是全工程的振假名 / 拼音上下微调（1080p 像素，浮点数，−40 到 +40，缺省 0）；正数离正文更远，负数更近，在正文自身的坐标方向上移动，随分辨率换算，换字体预设或读入 ASS 样式时保留。预览、导出、点选和自动排位置使用同一位置；0 保持旧导出字节不变。

### 4.1 工程人物表 `dict/characters.json`

路径取 `dictionaries.characters_json`，缺省为 `dict/characters.json`。`characters` 的旧 TSV 声明只作为转换来源。
首次打开时，有 TSV 而没有 JSON 才转换；TSV 字节不变，之后以 JSON 保存人物数据。

```json
{"format": "subkit-characters", "format_version": 1,
 "characters": [{"id": "chr_…", "names": {"ja": "幸平創真", "ja-kana": "ゆきひらそうま", "zh-Hans": "幸平创真"},
   "aliases": ["創真", "ソーマ"], "name_parts": ["幸平", "創真"], "kana_parts": ["ゆきひら", "そうま"],
   "self_refs": [{"surface": "私", "reading": "わたし", "scope": {}}],
   "persona": {"self": "", "address": "", "catchphrases": "", "tone": "", "note_ja": "", "note_tr": ""}}],
 "crowd": ["生徒A"], "not_person": ["効果音"],
 "pending": [{"label": "未知", "count": 2, "examples": ["例句"], "episodes": ["S01E001"],
   "by_episode": {"S01E001": {"count": 2, "examples": ["例句"]}}}]}
```

人物 ID 稳定、删除后不复用；`style` 是可选默认样式。`names` 按语言码保存，`ja-kana` 为姓名假名；
`name_parts` / `kana_parts` 保留 TSV 姓名分段，供固定词读音使用。旧中文名放在工程翻译语言键下，备注转换为 `persona.note_ja`。
`self_refs.scope` 可为 `{}`（全部）、`{"season": "S01"}`、`{"episodes": ["S01E001"]}` 或 `{"line": "m_…"}`。

标签按 `cc.normalize_speaker` 规范化后，仅与名字/别名完全相等才关联；重名歧义仍为待确认，不猜测、不自动合并别名。
`crowd` / `not_person` 为人工决定并优先于名字匹配，后者不进入待确认。待确认记录每集的成员出现次数，例句最多三条；
`by_episode` 用来替换同一集的统计，避免重新合并/关联重复累加。统计只存在人物表，不进入工作文档。
人物合并/删除影响打开文档时，使用同一个撤销事务恢复人物 JSON 原字节和成员关联。

## 5. 来源

### 5.1 `episode.json` 里的来源条目

```json
{"id": "o1", "lang": "ja", "lane": "original", "format": "srt",
 "path": "sample-media/example", "copy": "sources/o1.srt",
 "sha256": "…", "bytes": 41234, "encoding": "utf-8-sig", "newline": "crlf",
 "imported_at": "2026-10-02T09:00:00+10:00"}
```

`lane`：双语字幕 = **原声 + 翻译**。`original`（原声的语言：日文、英文……，通常需要振假名等注音）/ `translation`（翻译：中文，拼音可选）/
`reference`（只参考，不进入工作文本）。合并时默认以翻译的时间轴为基准（可选），原声字幕的原时间始终保留在来源里。
语言标签用 BCP 47：`zh-Hans` `zh-Hant` `ja` `en`。数据结构不得把原声语言写死为日文。

### 5.2 `sources/<id>.events.json`

```json
{
  "format": "subkit-source-events", "format_version": 1,
  "source": "o1", "sha256": "…",
  "events": [
    {"i": 0, "kind": "Dialogue", "start": 9050, "end": 11880, "text": "（城一郎）おあがりよ",
     "style": null, "name": null, "layer": null, "margins": null, "effect": null, "line": 3, "cue": "1"}
  ],
  "spans": [
    {"e": 0, "a": 0, "b": 5, "kind": "speaker_label", "value": "城一郎"},
    {"e": 7, "a": 0, "b": 1, "kind": "marker", "value": "≪"},
    {"e": 9, "a": 2, "b": 8, "kind": "reading_hint", "base": [0, 2], "value": "とおつき"},
    {"e": 11, "a": 0, "b": 7, "kind": "sfx"},
    {"e": 12, "a": 0, "b": 2, "kind": "music"}
  ],
  "analyzer": {"name": "ja-cc", "version": 1}
}
```

- `text`：事件文本的**原样字符串**。ASS/SSA 是 Text 字段原文（`\N`、`{…}` 都不展开）；SRT 是块内各行用 `\n` 连接（只统一行尾，不改字）。
- `spans`：来源分析器认出的**结构化元数据片段**（偏移指向 `text`）。种类：`speaker_label` `marker`（≪ ≫ ➡ [TEL] 等）
  `reading_hint`（行内注音的括号部分；`base` 是被注音的字的范围）`sfx`（效果音描述）`music`（♪ 记号行）
  `flashback_mark`（`((` `))`）`format`（SRT 里的格式标记 `<font …>` `<i>` `{\an8}` 等，`data` 记下标签名和属性；
  分析器 ja-cc v2 起）`other`。两人对话的 cue 每一行都以破折号开头时（`-A` / `-B`），破折号是 `marker`，`data.turn = true`。片段不删除任何东西，只说明“这几个字不是对白正文”。
  不确定时**不标**（按对白保留，并在 `doubts` 里给出提示）。
- 这个文件可以由原料字节和分析器版本重新生成；分析器升级时旧文件保留，迁移见第 12 节。

## 6. `document.json`

```json
{
  "format": "subkit-document", "format_version": 1,
  "episode": "S01E001", "revision": 12,
  "groups":      [ /* 有序！这就是文档顺序 */ ],
  "members":     {"m_…": {}},
  "comments":    {"c_…": {}},
  "annotations": {"a_…": {}},
  "karaoke":     {"k_…": {}},
  "tombstones":  [],
  "rules":       [],
  "history":     {"ops": "journal/ops.jsonl"}
}
```

### 6.1 Group（编辑单元）

```json
{"id": "g_…", "kind": "dialogue", "members": ["m_cn…", "m_ja…"], "comments": ["c_…"],
 "state": "auto", "origin": {"by": "aligner", "method": "subkit-align-v2", "score": 3.1, "doubts": ["cut_inside_cue"]},
 "flags": []}
```

- `kind`：`dialogue` / `song` / `sign`（屏幕字、从来源原样保留的排版行）/ `info`（制作信息）。
- 配对状态是**派生的**：只有一种语言的成员 = 未配对。没有专门的“未配对”种类，避免两处状态不一致。
- `state`：`auto`（算法给的，待确认）/ `confirmed`（用户确认，重新对齐时锁定）。
- `flags`：`order_exception`（用户接受了这里的来源顺序冲突）、`source_time_inversion`（来源本身时间倒序）等。
- 同一时刻的两句话、背景字、歌词**不会因为时间相同就合组**；组关系只来自对齐结果和用户的 Link/Unlink。
- `before_convert`（内存中为 `group.extra["before_convert"]`）：`{"kind": "dialogue"}`，记录最近一次普通台词 ↔ 歌曲转换前的种类；
  每次转换覆盖，转回普通台词时恢复。转换只改种类、成员角色和样式，不改文字、时间、位置或来源。

### 6.2 Member（一条文本对象）

```json
{"id": "m_…", "group": "g_…", "lang": "ja", "lane": "original", "role": "dialogue",
 "start": 905000, "end": 911880, "layer": 0, "style": "JP", "margins": [0, 0, 0], "effect": "",
 "speaker": {"character": "chr_…", "label": "城一郎", "by": "user"},
 "runs": [ /* 见 6.3 */ ],
 "overrides": {}, "format": []}
```

- `speaker.character`：人物 ID、`"crowd"`（群众）或 `null`（原标签/未关联）；`label` 始终保留来源标签。
  `by: "user"` 表示人工指定（包括明确选回原标签），自动关联/刷新跳过；旧字段缺省表示自动。
  重新合并按来源字符区间保留人工选择；若不同人工说话人被合到同一行，或人工行没有可对应的来源，则停止合并并保留旧文档。
  编辑器按整组或逐行指定说话人，每次一步撤销。重新分析来源沿用原先的“分析后重新合并”流程；
  调用者传入当前编辑器时，可只刷新人物关联并纳入该编辑器的撤销历史。
- `overrides`：用户对整行的设置，导出时写成行首标签（原字幕不变）。位置：`an` `pos` `move`；格式：`fn` `fs` `b` `i` `c1`；
  `raw`：任意 ASS 覆盖标签（如 `\frz10\fscx120`，不含 `\pos` `\move` `\an` `\k` `\p`）。原字幕里同名的标签不进入输出（`\t` 除外）。
- `format`：部分文字的格式，`[{"anchor": [段], "props": {"fs": 80, "raw": "\frz15", …}}]`，锚在字符身份上（跟着字走）；
  导出时在开头写标签、结尾恢复原来的值；`raw` 在选中文字里不能用整行才有意义的标签（`\org` `\clip` `\fad` 等）。
- `time_manual`（内存中为 `member.extra["time_manual"]`）：人工设时间、波形打轴/拖线、边界微调或整组平移时在同一事务内设为 `true`，撤销恢复原值。
  自动调轴不设置或清除这个标记；默认保护确认过的组以及任意成员有此标记的整组。旧工程没有此字段时，仅按组的确认状态保护。
- `before_convert`（内存中为 `member.extra["before_convert"]`）：`{"role": "top", "style": "CN.Top"}`，记录最近一次转换前的角色和样式，
  每次转换覆盖。转回台词恢复 `dialogue` / `top` / `alt`；没有记录的歌曲用普通台词角色和 `CN` / `JP`（按工程样式映射）。
  转回歌曲时，记住的歌曲样式属于所选 OP / ED / IN 家族就原样恢复，否则使用工程的歌曲样式映射。
- `karaoke_off`（内存中为 `member.extra["karaoke_off"]`）：转为普通台词时为 `true`，转回歌曲时移除；阻止 K 轴按字符身份套用到本行。
  来源中的 `\k` / `\kf` / `\ko` / `\K` 只从预览和导出文本里去掉，原字幕和 run 不变；导出报告 `karaoke_stripped` 列出去掉标签的行。

`role`：`dialogue` `top` `alt` `song` `sign` `info`。`start`/`end` 是这个成员自己的时间；Group 没有自己的时间。

复制组（0.22.0）使用新的 Group/Member/Annotation/Karaoke ID，所有非 sep Run 使用独立 insertion ID，读音、局部格式和拍锚点同步重映射；来源位置、来源台账及墓碑不复制。组未确认并去掉 hidden/marked/mark_note，其他显示内容与时间保留，翻译拍 link 去掉。

可选 Run 字段 `render_src` / `render_e`（0.22.0）仅保存副本的原 ASS 渲染背景：来源样式/分辨率、制作署名及原事件位置判断。副本的 `src` / `e` 为空，`ins` 是全新 ID，`a` / `b` 只定位新插入的字；这两个渲染字段不参与字符身份、顺序、去向核对、来源注音或自动配对。复制前属于原 ASS 的标签仍按来源坐标缩放，后来手打的新标签按工程坐标；字段随 Run 的复制、切分、搬句及 Comment 恢复移动，不依赖原 Member 继续存在。

含这两个字段的文档保存为 **document format_version 2**（Project/episode 仍为 1），没有复制来源上下文的文档继续写版本 1。读取器支持两版；旧版程序会拒绝版本 2，避免旧 `run_from_json` 忽略未知字段后保存、丢掉副本样式/坐标。回退到 0.21.0 时应打开修改前工程备份，不用旧版重存含新复制项的文档。版本 1 不需要批量迁移。

精确设置时间同时保护选中行的两个 `time_src` 边，`time_manual=true`；逐字读音每字一个 manual Annotation，非空为 confirmed，空框为 hidden，以防自动刷新补回。两者都使用正常撤销前后像，不要求文档版本升级。

### 6.3 Run（字符身份）

成员文本 = 各 run 的 `text` 依次相连（就是导出到 ASS 的 Text 字段）。

```json
{"src": "o1", "e": 57, "a": 3, "b": 10, "text": "七実様 落ち"}
{"ins": "i_…", "a": 0, "b": 2, "text": "さま"}
{"kind": "tag", "src": "t1", "e": 12, "a": 0, "b": 13, "text": "{\\fad(200,0)}"}
{"kind": "sep", "text": "  "}
```

- 来源 run（`kind` 省略 = `text`，或 `tag`）：`text` **必须**等于来源事件 `text[a:b]`，一个字都不能不同。
- `sep`：派生的分隔——同一成员里来自不同行、不同 cue 的原文之间的空白或 `\N`。它不是内容，不带来源，只能是空白或换行，
  校验器拒绝任何别的文字。来源里的空白字符不计入台账（第 8 节）。
- 插入 run：用户输入的文字。`i_…` 是那一次输入的 ID，`a`/`b` 是在那次输入原文里的偏移。
- 字符身份 = (`src`,`e`,偏移) 或 (`ins`,偏移)。编辑时没动到的字符身份不变；注音、K 轴都锚定在字符身份上，所以**文字移到别的成员时，注音跟着走**。

### 6.4 Comment（挂在 Group 上）

```json
{"id": "c_…", "group": "g_…", "kind": "hidden_filler", "lang": "ja",
 "runs": [{"src": "o1", "e": 60, "a": 0, "b": 3, "text": "あっ…"}],
 "position": {"after": "m_ja…"}, "by": "tool", "restorable": true}
```

- `kind`：`hidden_filler`（简洁模式下未渲染的语气词；Group 上必须显示数量，可一键恢复成显示文本）、
  `user_hidden`（用户隐藏的文字）、`note`（备注，`text` 字段）。
- 位置：带来源文字的 Comment（被隐藏的语气词、用户隐藏的文字）**按来源位置**放进本组的文字流——它在哪里被说出来，就排在哪里。
  顺序检查和“恢复显示”都按来源位置插入，不依赖成员边界（成员的文字会被编辑、移动）。
  `position`（`{"before": 成员}` / `{"after": 成员}` / `{"only": true}`）只是界面显示的提示；没有来源文字的备注才用它定位。
- 移动分界、拆分时，落在移动一侧的隐藏文字随之移到相邻的组（否则会造成乱序）。
- 来源原句（REQUIREMENTS 二.4 “原始日文始终带入”）**不复制成 Comment**：界面由成员 run 的来源事件直接显示完整原句、原时间、cue 编号；
  原句被分到多个 Group 时共用同一个来源事件键，各自的片段范围就是各自 run 的 `a`/`b`。

### 6.5 Tombstone（用户删除）

```json
{"src": "o1", "e": 60, "a": 0, "b": 3, "text": "あっ…", "op": "op_…", "member": "m_…", "by": "user"}
```

用户删掉来源文字时写入。重新对齐、刷新、模板更新都必须先读墓碑：被删的文字**不得复活**。用户撤销删除时墓碑移除。

### 6.6 Annotation（振假名 / 拼音 / 旁注）

```json
{"id": "a_…", "kind": "ruby", "member": "m_…",
 "anchor": [{"src": "o1", "e": 57, "a": 0, "b": 2}],
 "base": "七実", "text": "ななみ",
 "status": "confirmed", "origin": "manual",
 "rule": null, "candidates": ["ななみ"], "mono": false,
 "layout": {"scale": null, "dx": 0, "dy": 0}}
```

- `kind`：`ruby`（日文振假名）、`pinyin`（中文拼音/注音/威妥玛）、`gloss`（旁注，CalUP/English 合并）。
- `anchor`：字符身份范围（可以跨多个 run）。`base` 是冗余的底字，用于校验：底字被改写时注音标记为失效，提示用户，不硬猜。
- `text` 是整个 `base` 的读音（含送假名，便于审核：`合わ`/`あわ`）；`parts` 是振假名实际画在哪里：
  `[[a, b, 读音], …]`，相对 `base` 的偏移，**只覆盖汉字块，送假名和其他假名不加注音**（旧工具的规则，也是用户成品的做法：
  `合わ`→`合(あ)わ`，`突き落とさ`→`突(つ)き落(お)とさ`，`目覚め`→`目覚(めざ)め`）。含数字/字母的词整体注音（`A級`、`５日`）；
  无法按假名对齐的特殊读法注在汉字范围上。旧文档缺 `parts` 时，读取方按同一规则计算。
- `status`：`auto`（自动，可被普通刷新更新）/ `uncertain`（有候选待确认；`?` 只存在于这里，**不进入导出文本**）/
  `confirmed`（用户确认）/ `hidden`（用户选择不标注）。
- `origin`：`exception`（本处例外）`dict`（项目词库）`character`（人物规则）`episode_rule` `manual` `cc_hint`（原料自带）
  `rule` `analyzer` `voice`（配音建议，只能是 `uncertain` 或用户采用后的 `confirmed`）`legacy`（旧工程迁移）`passage`（重复段落库）。
- 读音优先级见第 10 节。

拼音 `kind: "pinyin"` 每个汉字一个 Annotation，`base` 为一个汉字，`text` 为带声调音节（轻声不带符号），
`parts` 为 `[[0, 1, "háng"]]`；`candidates` 含所选音节和多音字候选。自动生成用整句上下文，
`status: "auto"`，`origin: "analyzer"` 或 `"rule"`；多音字不因有多个候选而成为 `uncertain`。
确认、隐藏以及人工/例外/旧版来源的拼音不被自动刷新覆盖，改字或搬移仍按字符身份跟随。
输入接受 `hang2`、`háng`、`lv4`（v = ü）；5 或无数字表示轻声，存储统一为声调符号。

本集拼音规则存 `document.rules`：`{"kind": "pinyin", "surface": "行长", "reading": "háng zhǎng"}`。
项目词库 `dict/pinyin.json`（可由 `dictionaries.pinyin` 指定）格式：
`{"format": "subkit-pinyin", "format_version": 1, "words": {"行长": "háng zhǎng"}}`。
词必须全为汉字，读音用空格分隔，每字一个音节；本集同词规则覆盖项目规则，重叠时长词优先。
只改这里写逐字人工注音；本集/项目规则有自动注音影响预览，规则和当前集刷新合为一步撤销。
项目规则供其他集以后合并或刷新时使用。审议包开拼音时只在多音字及已确认字后显示 `行(háng)`；关时保持原输出。

### 6.7 Karaoke（r3，已实现：`subkit.core.karaoke`）

```json
{"id": "k_…", "member": "m_…", "state": "timed", "unit": "syllable",
 "beats": [{"core": [{"src": "t1", "e": 363, "a": 0, "b": 1}], "k": 0, "sung": "う",
            "t0": 241560, "t1": 241935, "by": "tap"}],
 "effect": {"preset": "kf", "params": {}}}
```

- **存时间和人工边界**。拍的结构（哪几个字、哪段读音、送假名/っ/标点跟哪一拍）每次都从当前文字和读音推出来，再应用 `edits`：
  汉字块按读音分拍；假名一拍；小写假名总跟前一拍，`unit = syllable` 时 `っ ー ん` 也跟前一拍（`mora` 时分开）；
  拉丁字母/数字一个词一拍；标点、空格跟前一拍（行首的跟后一拍）。
- `core`：这一拍“自己的”字（汉字块、假名、词）的字符身份；`k`：在块里的第几拍；`sung`：唱的假名（平假名）或文字。
  三者一起是这一拍时间的键：字和读音没变，时间就一直对得上；字移到别的成员里，时间跟着字走。
- `edits`（可选，缺省为 `[]`）：人工拆合列表 `[{"op": "split" | "merge", "at": 边界键}]`。
  `split` 添加边界，`merge` 去掉边界；同一键最后一项生效。边界表示新拍从哪里开始，不使用成员内的临时文字位置：
  假名/词用新拍第一个底字的身份，例如 `ガー` 的 `ー`：`{"core": [{"ins": "i_…", "a": 1, "b": 2}]}`；
  振假名用片段底字的完整身份、读音字符偏移 `r` 和完整片段读音 `reading`（平假名校验），例如 `学(がっ)` 的 `っ`：
  `{"core": [{"src": "o1", "e": 57, "a": 0, "b": 1}], "r": 1, "reading": "がっ"}`。
  `r` 为 0 起偏移，跨底字块合并时可指向下一片段的 `r=0`；行首不能拆合。底字/读音校验失效的项保留在数据中但不应用，
  提示“拆分/合并的位置找不到了（字或读音改过）”；切换音节/拍粒度仍应用有效项。文字移到其他成员时边界随字符身份走。
  编辑后的拍用所唱各字的身份（去重）作为 `core`，`k` 按导出块内的拍顺序确定；边界布局相同，键就相同。
- 拆拍/合拍同时写边界和新拍时间，一次操作一步撤销，同句光晕层按其自身字符身份同步。
  拆拍默认按底字/读音字符比例分配原时间段，也可给出绝对拆分时刻；有时间的拍两侧至少各 10 毫秒，原拍不足 20 毫秒时拒绝拆分。
  合拍时间为前拍开始至后拍结束；含未打拍的拆合结果仍为占位，不能算完成。只有 `edits`、没有 `beats` 时间的轨道照常导出占位拍。
- `t0` `t1`：**绝对时间**（毫秒）。改行的开始/结束不动拍；平移 Group 时拍一起平移。
- `beats[].link`（可选，译文拍）：按译文自己的边界保存原声引用，`start` / `end` 可各自存在，例如：
  `"link": {"start": {"core": [{"src": "o1", "e": 57, "a": 0, "b": 1}], "k": 0, "sung": "が", "edge": "end"}}`。
  外层 `start` / `end` 指这一译文拍的起止；引用内的 `core`、`k`、`sung` 是同组原声拍的完整存时键（含人工拆合后的键），
  `edge` 指原声拍的 `start` / `end`。不使用临时拍序号或文字位置。相邻译文拍共用的边界会在前拍 `end` 和后拍 `start` 存同一引用。
  只有拖译文边界在原声边界 6 像素内吸附时建立链接；Alt 临时不吸附，拖离吸附范围解除该边界的链接。
  原声打拍、拖拍、读入时间、拆合、平移及恢复歌曲时，链接边界在同一事务中跟随，两个有效链接间的未链接边界按原时间比例移动，
  四舍五入到整数毫秒（正好半毫秒取偶数）；没有两端链接的边界保持原时间，整组平移仍整体移动。译文同句图层使用各自底字身份同步。
  找不到原声拍键的链接删除，轨道的 `issues` 保留“吸附的原声拍找不到了（字、读音或拆合改过）”，显示于 K 轴和发布检查；撤销同时恢复链接和时间。
  休眠轨道不跟随，恢复为歌曲后重新对齐；不同组的原声拍不作为跟随目标。
- `by`：`tap`（按 K 打的）`drag`（波形上拖的）`import`（来源 `\k` 读入）`memory`（歌曲记忆）`onset`（起音建议）`manual`、`align`（按歌声自动对齐，计入已打拍）。
- 拍的可选 `check: true` 表示自动对齐需要复查；读取到拍的 `flags`，拖动该拍边缘、重新打拍或确认轨道时清除，未改动拍的标记保留。
- 翻译拍的可选 `auto_split: true` 表示「重新分配翻译拍」把一个原声拍按比例拆给了几个翻译字，这一拍至少有一条边不在原声拍边界上（没有链接）。它是占位拍时，两边最近的链接所覆盖的原声拍都打完后自动算打完；英文按音节拆开的拍没有这个标记，仍要自己打。
  没有存时间的拍在读取时得到“占位”时间（平均分在相邻已打的拍之间），读音改过的汉字块保留原时间段并平均分，都会被标出来。
- `state`：`suggested`（自动建议，还需确认）/ `timed` / `confirmed`（用户确认）；读取时若有占位拍则视为未完成。发布检查列出未完成的 K 轴。
- `effect.preset`：`kf`（默认）/ `k` / `ko` / `none`，以及以后的特效模块名；换特效不改 `beats`。
- `dormant`：转为普通台词时为 `true`，轨道和 `beats` 原样保存在工程中；休眠轨道不参与 K 轴索引、有效时间、导出或未完成检查。
  转回歌曲时移除该键，恢复原轨道 ID、拍和特效，不新建重复轨道。缺省或 `false` 表示活动轨道。
- `before_convert`、`karaoke_off` 在 JSON 中直接写在 Group / Member 对象上，轨道的 `dormant` 直接写在轨道对象上；保存重开保留这些字段。
- 导出（与用户成品一致）：底字每个块一个标签，汉字块拿到它所有拍的总长；每个振假名片段先空拍等待，再逐拍（`{\kf199}{\kf24}い`）；
  拍之间的休止是空拍；来源行原有的 `\k` 在输出里被替换（来源不变）。同一句歌词的另一个图层（字幕组的光晕层）打轴时同步。
  拆开的底字拍各有标签，跨底字块合拍用一个标签覆盖相关块；汉字块内部拆合仍在底字上汇总时长，振假名按编辑后的拍写标签。
  合拍跨越多个振假名片段或底字时，各振假名片段按所唱字符数分得合拍时间，后面的片段先等待；翻译行沿用自己的拍。
  无 `edits` 的行沿用原有导出规则。

### 6.8 重复段落库（记录与建议：`subkit.core.passages`）

清单路径使用工程 `dictionaries.passages`；没有声明时明确使用 `dict/passages.json`，不修改工程清单。
`dict/songs.json` 从未写入，保留旧声明但不再使用。音频模板固定在 `dict/passages/<id>.npz`，
沿用 `audio_match.Template` 的频谱、起音、hop 和特征版本字段，无 pickle。

```json
{"format": "subkit-passages", "format_version": 1, "passages": [
 {"id": "psg_…", "name": "…", "category": "OP", "version": "TV",
  "episode_id": "S01E001", "created_at": "…", "start": 1000, "duration": 4000,
  "groups": [{"kind": "song", "members": [
   {"lane": "original", "lang": "ja", "role": "song", "layer": 0, "style": "OP.JP",
    "margins": [0, 0, 0], "start": 0, "end": 4000, "text": "学校", "visible": "学校",
    "inline_tags": "", "overrides": {}, "format": [],
    "annotations": [{"kind": "ruby", "a": 0, "b": 2, "text": "がっこう",
                     "parts": [[0, 2, "がっこう"]], "status": "confirmed"}], "karaoke": null}
  ]}], "summary": {"groups": 1, "lines": 1, "readings": 1, "readings_skipped": 0,
                    "unconfirmed_lines": 0, "karaoke": {"none": 1, "done": 0, "partial": 0}}}
]}
```

- `category`：`OP` / `ED` / `IN` / `NA`（旁白或固定台词）。保留文档组、成员顺序，忽略 comment 角色和无可见文字的成员。
- `start` 是来源段落最早成员开始，`duration` 到最晚结束；成员和拍时间均为相对段落开始的毫秒。
  `text` 去掉 tag run，保留换行；`visible` 使用统一 `visible_map` 的可见文字，`inline_tags` 仅作原标签参考。
- 注音只保存 `confirmed` / `hidden`，`a/b` 为可见字符的半开区间，`parts` 仍相对注音底字。
  格式用 `a/b` 代替 anchor，其他字段深复制；整行 overrides 原样深复制。不保存来源字符身份。
- `karaoke` 有 `unit/state/effect/edits/beats`。`core` 是有序可见区间数组，如 `[[0, 2], [3, 4]]`；
  拆合的 `at.core` 同样转换，`r/reading` 保留。仅保存已打拍；占位拍不保存，suggested 行不保存任何拍。
  译文拍链接的 `core` 使用同组原声成员偏移，并携带 `member_index`（该原声成员在保存组内的索引），其余 `k/sung/edge` 不变。
- `summary.karaoke` 的 none/done/partial 是记录对话框统计；partial 包含建议和未打完的行。
- 纯计划使用 `audio_match.Match`，默认平移，选缩放才按段落原点乘比例，整数毫秒半值取偶数。
  动作携带目标成员/组 ID 和独立数据；相同缺组的多行共享一个稳定插入动作 ID，数据只包含覆盖范围内的行。
  K 轴动作携带组内保存成员、匹配目标 ID 映射及 `readings_differ` 标记，供应用时复核唱词序列。
- 应用标记存于 `group.extra.passage = {id, at, score, applied_at}`，序列化后直接是 Group 的 `passage` 字段；
  不使用 Document.extra，确保后续事务能够撤销。`at` 距匹配起点不超过 2000 ms 时视为已套用。

## 7. 顺序规则（规范）

对每个来源 S 的每条**流**（主语言、第二语言都要检查）：按 `groups` 顺序遍历；在每个 Group 内按 `members` 顺序，
把挂在成员前后的 Comment 按 `position` 插进去；依次读出所有 `src == S` 且属于这条流的 run，得到序列 `(e, a)`。

- **流**：SRT 来源整个文件是一条流；ASS/SSA 来源**每个（Style, 角色）是一条流**（角色在导入时确定并保存在来源里：对白、顶部对白、
  原声歌词、翻译歌词、屏幕字、注释）。理由（实测 Soma VCB 字幕）：OP/ED 的日文歌词、中文歌词各自成块放在文件末尾；
  带 `\pos` 的屏幕字有时混在对白 Style 里、而且不按时间顺序。按（Style, 角色）分流后，每条流自己的事件保持文件顺序可以严格检查，
  又不会把字幕组的分块习惯误报成乱序。日文 SRT 是全文件严格保序。
- 这个序列**必须**按字典序不减。包括未配对的成员、`hidden_filler`、`user_hidden`。墓碑不参与。
- 用户手动移配造成的违反，界面必须显示；用户接受后在相关 Group 打 `order_exception`，校验器把它列为“已接受的例外”，不再算错误。
- 时间不参与这条规则：同时对白可以重叠，Group 顺序与开始时间不一致也不算错误（另有“时间倒序”提示，只是提示）。
- 来源本身时间倒序的事件（实测：Soma 第二季 E02/E03 共约 8 处，其他番组没有）按来源顺序排列，打 `source_time_inversion`。

## 8. 台账规则（规范）

对每个参与工程的来源事件（含 Comment 事件）的每个**非空白字符**，必须恰好落在下列一类：

| 类别 | 判定 |
|---|---|
| 正文（已配对） | 被某个成员的 run 覆盖，该组有两种语言 |
| 正文（未配对） | 被某个成员的 run 覆盖，该组只有一种语言 |
| 同组 Comment | 被 `hidden_filler` / `user_hidden` Comment 覆盖 |
| 结构化元数据 | 落在 `spans` 里（说话人标签、标记、行内注音、效果音、音乐记号）；被 `rules` 里 `{"kind": "span_text", "src", "e", "a", "b"}` 放回正文的不算 |
| 用户删除 | 被墓碑覆盖 |

- 被覆盖 0 次 = **缺失**（错误）；被覆盖 2 次以上 = **重复**（错误）。按每个出现位置计数，不按字符集合或总字数比较。
- 导出报告逐类列出数量和位置；“缺失”不能通过忽略警告变成合法结果。

## 9. 样式（规范：优先级）

成员的最终样式 = 模板基准样式（按 `lane` + `role`）← 类别规则 ← 人物规则 ← 成员自身 `overrides`（越往右优先级越高）。
- 模板有 ID 和版本；`episode.json` 记录本集用的模板版本。
- 规则有作用范围（工程 / 季 / 集）和“只对本次命中的成员”或“今后的默认”两种；批量修改默认只写成员 `overrides`，不升级成规则。
- 振假名、拼音、旁注的样式从**底字成员的最终样式**派生（字号比例、描边、间距在模板里定义），不需要单独批量修改。

## 10. 读音（规范：优先级）

同一位置有多个来源时，按下列顺序取第一个：

1. `exception`：用户明确选择的“本处例外”（界面会显示它覆盖了哪条规则）。
2. 项目词库 / 人物规则（已确认的权威规则；人物规则可带季、集、情境条件）。
3. 当前字幕（本集）规则。
4. `manual`：本处人工记录。与 1–3 冲突时**不静默失效**：显示“被某规则覆盖”，让用户选择改成例外或接受规则。
5. `cc_hint`：原料自带的注音。
6. 自动：上下文规则、分词器、配音建议。只能产生 `auto` / `uncertain`，不得覆盖 1–5。

- 已确认（`confirmed`）的读音在普通刷新中不再送去自动消歧。
- 词库锁定的词同时锁定分词边界；单字人名只在分词结果里单独成词时生效。

## 11. 导出（发布 ASS）

- 输出：模板样式 + 派生样式、各成员的 Dialogue、生成的振假名/拼音/旁注行、编译后的卡拉 OK 特效、来源里的 sign/info 事件。
- 不输出：工程 ID、Comment 类工程数据、未确认的 `?`、`placeholder` K 轴（检查拦截，用户可以选择仍然导出并写进报告）、特效槽名。
- Actor/Name：选项（默认保留说话人名，按语言输出对应名字）。来源原句参考 Comment：选项（默认不带）。
- **显示规则**（工程设置 `settings.display`，由用户主动开启，只作用于预览和导出，**永远不改文档**）：
  `hide_final_period` — 原声文本里句末的「。」「｡」（后面是空白、换行/分隔或行尾；「…。」引号里的不算）不显示。
  导出报告写明被规则隐藏的字数。
- 不得用“删除所有 Comment / 所有含 @ 或 # 的文本”的办法清理；正文里合法的 `@` `#` 和 ASS 标签原样保留。
- 没有显示时间的成员（结束不晚于开始，或整行在 0 之前）写成 Comment，不写 Dialogue（时长为 0 的 Dialogue 会让 mpv 把整条字幕当成
  时长未知、截短别的行），导出报告列出。
- 振假名 / 拼音行跟随底字：底字处的旋转（`\frz` `\frx` `\fry`，配底字行的 `\org` 或锚点）、斜切（`\fax` `\fay`）、缩放、颜色和透明度、
  其中的 `\t` 动画、`\move` `\clip` `\fad`；位置在未旋转的版面里计算（与 libass 一致）。
- 默认另存新文件；同名文件存在时询问或加版本号；不得覆盖原料。

## 12. 版本与迁移

- `format_version` 是整数。读取到更高版本：只读打开并提示。升级：先备份（`*.v<旧版本>.bak.json`），再写新版本，迁移步骤可重复执行。
- 旧 SubKit 工作文件（`work.ass` + `state.json`，基线格式）的导入：
  按 `#编号` 恢复配对 → 用来源原文重新定位每段文字（找不到的记为 `ins` run 并列入迁移报告）→ `state.json` 的手改读音变成 `manual` 注音。
  无法可靠迁移的部分列清单，不做静默猜测。

## 13. 与另一实现交换

- 交换单位：整个工程目录（不含 `journal/`）。读取方必须先跑校验器（第 7、8 节 + ID/引用完整性 + run 文本与来源一致）。
- 认不出的字段：读取方**原样保留**（写回时不丢）；认不出的 `kind`：只读显示，不删除。
- 分歧写进 `docs/INTEROP.md` 的兼容性清单，不在各自实现里悄悄分叉。
