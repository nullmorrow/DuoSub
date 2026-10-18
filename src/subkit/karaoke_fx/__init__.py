"""Discover trusted in-package effects_*.py modules; every UI uses REGISTRY."""
from importlib import import_module
from pkgutil import iter_modules

from .base import Effect, EventLimit, ExportError, FxContext, FxGlyph, Parameter

INLINE = {"kf": "渐变扫过", "k": "瞬间变色", "ko": "描边先出", "none": "无"}


def _register(registry, effects):
    for fx in effects:
        if not isinstance(fx, Effect) or not isinstance(fx.name, str) or not fx.name.strip() or fx.name.startswith("preset:"):
            raise ValueError("卡拉 OK 特效必须提供有效的 Effect 和独立代号")
        if fx.name in registry:
            raise ValueError(f"卡拉 OK 特效代号重复：{fx.name}")
        registry[fx.name] = fx


REGISTRY = {name: Effect(name, label) for name, label in INLINE.items()}
# Keep the original menu order; new independent modules follow alphabetically.
_modules = sorted((m.name for m in iter_modules(__path__) if m.name.startswith("effects_")),
                  key=lambda name: ({"effects_basic": 0, "effects_motion": 1}.get(name, 2), name))
for _module_name in _modules:
    _module = import_module("." + _module_name, __name__)
    _register(REGISTRY, _module.EFFECTS)
EFFECTS = [fx for name, fx in REGISTRY.items() if name not in INLINE]


def get_effect(name):
    if name not in REGISTRY:
        raise ExportError(f"未知的卡拉 OK 特效：{name}")
    return REGISTRY[name]
