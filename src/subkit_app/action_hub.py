"""Shared Qt actions and scene-aware window shortcuts."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeySequence

from .actions import ActionRegistry


class ActionHub(QObject):
    """Keep one action per id, including native dock toggle actions."""

    def __init__(self, registry: ActionRegistry, parent: QObject) -> None:
        super().__init__(parent)
        self.registry = registry
        self._parent = parent
        self._actions: dict[str, QAction] = {}
        self._scene = "main"
        self._shortcuts: dict[str, Callable[[QKeySequence], None] | None] = {}
        self._fallbacks: dict[int, tuple[str, QKeySequence]] = {}
        parent.installEventFilter(self)

    def action(self, action_id: str, *, factory: Callable[[], QAction] | None = None) -> QAction:
        """Get/create an action; factories retain Qt's native dock state wiring."""
        spec = self.registry.lookup(action_id)
        if action_id not in self._actions:
            action = factory() if factory is not None else QAction(spec.text, self._parent)
            action.setProperty("action_id", action_id)
            self._actions[action_id] = action
        return self._actions[action_id]

    def bind_shortcuts(self, action_id: str, handler: Callable[[QKeySequence], None] | None = None,
                       *, connect_triggered: bool = False) -> None:
        """Migrate a command's keys, optionally retaining its legacy keyboard handler.

        A keyboard handler also stays reachable when menu state disables/hides
        the action, just as the former independent QShortcut did.
        """
        action = self.action(action_id)
        if action_id not in self._shortcuts:
            action.installEventFilter(self)
            action.changed.connect(lambda: self._sync_fallbacks(action_id))
            self._parent.addAction(action)
        if connect_triggered and action_id not in self._shortcuts:
            action.triggered.connect(lambda _=False: self._shortcuts[action_id](action.shortcut()))
        self._shortcuts[action_id] = handler
        action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self._apply_keys(action_id)

    def set_scene(self, scene: str) -> None:
        """Reapply registry keys for the commands migrated to window actions."""
        if scene not in ("main", "watch"):
            raise ValueError(f"Unknown window scene: {scene}")
        self._scene = scene
        for action_id in self._shortcuts:
            self._apply_keys(action_id)

    def refresh(self, scene: str) -> None:
        """Include newly assigned menu shortcuts and resync all fallback grabs."""
        for action_id, action in tuple(self._actions.items()):
            spec = self.registry.lookup(action_id)
            if (action_id not in self._shortcuts and spec.customizable
                    and "main" in spec.keys):
                self.bind_shortcuts(action_id)
        self.set_scene(scene)

    def _apply_keys(self, action_id: str) -> None:
        action = self._actions[action_id]
        spec = self.registry.lookup(action_id)
        if "watch" in spec.keys and "main" not in spec.keys:
            action.setEnabled(self._scene == "watch")
        action.setShortcuts([QKeySequence(key) for key in self.registry.effective_keys(action_id, self._scene)])
        self._sync_fallbacks(action_id)

    def _sync_fallbacks(self, action_id: str) -> None:
        for shortcut_id, (owner, _key) in tuple(self._fallbacks.items()):
            if owner == action_id:
                self._parent.releaseShortcut(shortcut_id)
                del self._fallbacks[shortcut_id]
        action = self._actions[action_id]
        if self._shortcuts[action_id] is None or (action.isEnabled() and action.isVisible()):
            return
        # QWidget's shortcut map still honours ShortcutOverride in text fields,
        # modal windows and popups. No duplicate QAction/QShortcut is needed.
        for key in action.shortcuts():
            shortcut_id = self._parent.grabShortcut(key, Qt.ShortcutContext.WindowShortcut)
            self._fallbacks[shortcut_id] = (action_id, key)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Shortcut:
            if obj is self._parent:
                binding = self._fallbacks.get(event.shortcutId())
                if binding is not None:
                    if not event.isAmbiguous():
                        self._shortcuts[binding[0]](binding[1])
                    return True
            elif isinstance(obj, QAction):
                handler = self._shortcuts.get(obj.property("action_id"))
                if handler is not None:
                    if not event.isAmbiguous():
                        handler(event.key())
                    return True
        return super().eventFilter(obj, event)
