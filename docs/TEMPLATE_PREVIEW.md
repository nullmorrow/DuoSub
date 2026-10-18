# 样式模板预览：中性例句、按语言渲染、背景、注音上下微调

状态：Claude 2026-10-05 定（用户：预览词更中性；选别的语言时预览仍是中日文；背景是灰色；振假名/拼音要能整体上下微调，可正可负，全工程通用）。
界面文字以本文件为准（「」里照抄）。

## 1. 中性例句（不来自任何作品）
四个预览场景：对白 / 两人同时说话（第二个人显示在上面）/ 屏幕顶部 / 歌词。每种语言一套，按工程选的**原声语言**和**翻译语言**各取一套；
日语例句带固定读音（下面括号里的就是振假名，预览按这里画，不依赖分词器），中文的拼音由程序生成。

| 语言 | 对白 | 第二个人 | 屏幕顶部 | 歌词 |
|---|---|---|---|---|
| 日语 ja | 駅(えき)の前(まえ)で待(ま)っています。 | わかった、気(き)をつけてね。 | 三日後(みっかご) | 空(そら)に向(む)かって歌(うた)おう |
| 简中 zh-Hans | 我在车站前面等你。 | 知道了，路上小心。 | 三天后 | 向着天空歌唱吧 |
| 繁中 zh-Hant | 我在車站前面等你。 | 知道了，路上小心。 | 三天後 | 向著天空歌唱吧 |
| 粤语 yue | 我喺車站前面等你。 | 知道喇，路上小心。 | 三日後 | 向住天空唱歌啦 |
| 韩语 ko | 역 앞에서 기다릴게요. | 알겠어, 조심해. | 사흘 후 | 하늘을 향해 노래하자 |
| 英语 en | I'll wait for you in front of the station. | Got it. Take care. | Three days later | Let's sing to the sky |
| 越南语 vi | Mình sẽ đợi bạn trước nhà ga. | Biết rồi, đi cẩn thận nhé. | Ba ngày sau | Hãy hát về phía bầu trời |
| 法语 fr | Je t'attends devant la gare. | D'accord, fais attention. | Trois jours plus tard | Chantons vers le ciel |
| 德语 de | Ich warte vor dem Bahnhof auf dich. | Alles klar, pass auf dich auf. | Drei Tage später | Lass uns zum Himmel singen |
| 西班牙语 es | Te espero frente a la estación. | Entendido, cuídate. | Tres días después | Cantemos hacia el cielo |
| 葡萄牙语 pt | Vou te esperar em frente à estação. | Entendi, tome cuidado. | Três dias depois | Vamos cantar para o céu |
| 意大利语 it | Ti aspetto davanti alla stazione. | Va bene, stai attento. | Tre giorni dopo | Cantiamo verso il cielo |
| 俄语 ru | Я жду тебя у вокзала. | Понял, будь осторожен. | Три дня спустя | Давай споём небу |
| 其他语言 | 用英语那一套 | | | |

- 选的语言一变（新建工程对话框的「原声」「翻译」下拉框，或工程语言），预览**马上**换成那种语言的例句并重新渲染；振假名只在日语那一边、拼音只在中文那一边（开着拼音时）出现。
- 两边选成同一种语言时照常显示（例句相同）。

## 2. 背景
- 背景选项不变（渐变 / 黑 / 白 / 图片 / 这一集的视频画面），新加「深色渐变（接近画面）」并作为没有视频时的默认；工程里有带视频的集时，默认用「这一集的视频画面」。
- **选的背景铺满整个预览区**，「放大字幕区」时也一样（不再出现中间一条灰、上下黑边）。原来的「渐变灰」改名「浅灰渐变」，还能选。

## 3. 振假名 / 拼音上下微调（全工程通用）
- 模板编辑器「振假名 / 拼音」一栏加两项：「振假名上下微调」「拼音上下微调」，单位像素（1080p），可正可负，范围 −40 到 +40，步长 0.5，默认 0；
  提示文字「正数离正文更远，负数更近」。
- 存在模板里（随工程走，换模板时保留数值）；预览、观剧模式、导出、点选位置、「按发布版的间距自动排位置」都按微调后的位置算。为 0 时导出和以前逐字节一样。

## 4. 标准按钮的中文
- 对话框里 Qt 自带的按钮（Cancel、OK、Yes、No、Apply、Close……）显示为中文（取消、确定、是、否……）：程序启动时加载 Qt 自带的简体中文翻译。
