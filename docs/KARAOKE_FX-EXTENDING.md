# DuoSub 特效模块扩展

2026-10-16，适用于 0.24.0。特效代码在 `src/subkit/karaoke_fx/`，模型仍只保存稳定的效果代号和参数；预览、单集导出、整季导出复用同一生成器。

## 添加一个效果

1. 在该包里新增 `effects_<name>.py`，实现 `Effect.generate(ctx)`。
2. 文件导出 `EFFECTS = [效果实例, ...]`。代号必须独立且稳定，不得重用 `kf/k/ko/none`，不得使用工程预设的 `preset:` 前缀。
3. 重启程序。注册表自动发现这些模块，保留旧特效顺序，再按模块名追加新模块。无需修改主窗口、工具栏、参数窗口、预设读写或导出器中的特效菜单清单。
4. 同批补使用说明、版本记录和必要验证。新参数使用中文标签，F1 沿用 `karaoke-fx`。

这里只加载随应用维护的包内 Python 模块。工程里的 `dict/karaoke_fx.json` 是参数预设，不存代码；当前没有运行时安装器或热重载。

```python
from .base import Effect, Parameter, paint


class MyEffect(Effect):
    def generate(self, ctx):
        events = []
        for glyph in ctx.glyphs:
            if not glyph.beats:
                continue
            # 示意：这里实现实际的入场／唱到／退场逻辑。
            ctx.emit(events, glyph, ctx.start, ctx.end,
                     paint(ctx, glyph, ctx.start))
        return events


EFFECTS = [MyEffect("my_effect", "我的特效", {
    "height": Parameter("int", 12, "高度（像素）", 0, 200),
})]
```

`Parameter` 支持 `int/float/bool/choice/color/colors`。`schema` 自动合并共同参数（提前出现、唱完后保留、淡入淡出、振假名跟着、翻译行跟着）；`resolve()` 校验并补默认值。示例只说明接线，实际效果应落实这些共同参数的行为。不要从其他特效复制不适用的淡入淡出或时间原点。

## 生成器契约

- `FxContext` 和 `FxGlyph.beats` 的时间单位都是绝对毫秒；ASS `Event.start/end` 是厘秒。通过 `rounded_ms()` 和 `ctx.emit()` 统一对齐，使用连续的 `[start,end)` 分段，不生成零长度行。
- `glyph.x/y`、`box` 是当前 ASS 画面坐标，`glyph.tags` 保留字号、缩放和其他有效文字状态。原行有移动时，叠加 `position_at(ctx, glyph, time)`，不要覆盖成固定原点。复杂旋转／透视需要单独做画面验收。
- `glyph.owner` 保留命中与选择信息；所有新行都用 `ctx.emit()` 生成，保证 `events` 与 `ctx.owners` 一一对应。不要把运行时附加信息写进用户的原文或原料。
- `glyph.anchor_ids`：正文是该字的身份；振假名是**这块注音实际覆盖的全部本字身份**。它来自 Ruby 排版块的实际范围，支持一字多假名、一块多汉字、同 annotation 的不同本字块。需要跟随的动画按这些身份建立共享单元，不能只按相同文字、相近坐标或 beat index 猜配。
- 需要随机时使用 `ctx.rng`；种子按成员和已解析参数固定。关联单元每个时间格只取一次位移，并给本字和全部注音复用。同一工程重新生成、预览与导出应一致。
- 分段后用 `paint()` 保留此前的已唱状态，不能在每段重新播放已经发生的变色。淡入／淡出保留整行的绝对时间原点，不能在每个短段重新开始。
- 单成员包含本字、假名及附加层的总计上限为 `ctx.event_cap`（默认 2000）。先估算生成量；可以在有说明和报告提示的前提下自适应降低密度。最终 `EventLimit` 会走原有整行退回 `kf` 的保护，不得静默截掉后半行。
- 非致命的实际调整写入 `ctx.warnings`，由导出器加入报告。不要在生成器里弹 Qt 窗口、访问网络或直接写文件。
- 工程保存的是代号而非模块名。之后改变文件结构也应保留代号；新参数提供默认值，未知代号／参数按现有规则明确报错。

## 惊恐颤抖的实现与范围

`effects_fear.py` 独立实现 `tremble`。默认每 40 ms 给各本字块选一个随机整数像素位移，左右／上下范围各为 ±3；原有 `shake` 仍是唱到时的小角度旋转。参考《見える子ちゃん》第 7 集约 2:43–2:50 的分段 XY 位置变化，未复制或执行来源字幕中的 Lua 模板代码。

勾选“振假名跟着”时，以 `anchor_ids` 合并实际本字块，再共享位移表；关闭则保留原有静态振假名路径。“整行一起抖”让全部字共用位移。未唱透明度默认 208，唱到后恢复，颜色取模板。中文译文按自己的 K 轴生成。

生成前根据本字加假名的总数调整时间格，保证普通长句不直接触发 2000 行保护；若需要放慢到超过 200 ms，仍退回渐变扫过并报告。此策略作用于该新特效，不改变其他效果的密度或保护规则。

## 验证

新增效果至少覆盖确定性、参数校验、时间连续性、所有者映射、预览缓存、静态／移动位置、振假名跟随开关、单字多假名及多汉字一块注音、典型长句与上限保护。涉及几何移动时，实际 libass 渲染应按 ASS 分辨率验收，避免把缩小画面后的字形栅格变化误判为坐标不同步。

GUI 验证菜单完整性（与 `REGISTRY` 一致）、预设保留、混合选择只改歌曲、一次撤销和 F1。具体回归和本次证据见 `tests/test_fear_shake.py`、`tests/test_karaoke_fx.py`、`tests/test_karaoke_fx_ui.py` 及本次 Claude 交接文档。
