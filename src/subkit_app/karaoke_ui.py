"""按住 K 跟唱 — karaoke timing by ear (docs/KARAOKE.md §2.2).

* With a song group selected and the video paused: **K** plays from just before the line (or from the beat you
  clicked); then **hold K while each beat is sung** — key down = the beat starts, key up = it ends.  A release
  shorter than ``GAP_MS`` before the next press counts as legato (no rest).  The last release finishes the line.
* While the episode plays (刷剧连打): just hold K along with the song — each press goes to the next beat of
  the song line under the playhead; when playback leaves the line it is finished and written.
* Esc cancels (nothing is written).  Every finished line is one undo step.

Times are read straight from mpv (``time-pos``) at the key event, so a slowed-down playback (0.5×, 0.75×)
still records real song time.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit

from .actions import ActionRegistry

GAP_MS = 150          # shorter releases are legato
MIN_BEAT_MS = 30


class KaraokeTapper(QObject):
    finished = Signal(str, object)        # member id, beats (timed ones by "tap")
    activeBeat = Signal(str, int)         # lane, beat index (-1 = none)
    status = Signal(str)

    def __init__(self, player, find_line: Callable[[float], Optional[tuple]], parent=None,
                 on_idle_k: Optional[Callable[[], None]] = None,
                 on_selected_k: Optional[Callable[[], bool]] = None,
                 registry: ActionRegistry | None = None,
                 accepts_input: Optional[Callable[[], bool]] = None) -> None:
        super().__init__(parent)
        self.registry = registry if registry is not None else ActionRegistry()
        self.accepts_input = accepts_input
        self.player = player
        self.find_line = find_line       # t_ms → (member id, lane, beats, line start, line end) for 刷剧连打
        self.on_idle_k = on_idle_k       # K while paused and idle: arm the selected song line
        self.on_selected_k = on_selected_k  # a selected beat takes priority over continuous tapping
        self.run: Optional[dict] = None
        self.offset_ms = 0               # tap latency compensation (added to every tap)
        self.enabled = True
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------ control
    def arm(self, mid: str, lane: str, beats: list, line: tuple[int, int], from_index: int = 0) -> None:
        """Start timing one line: play from just before it (or before the chosen beat) and wait for K."""
        if not beats or not 0 <= from_index < len(beats):
            return
        beats = [copy.copy(b) for b in beats]
        for b in beats[from_index:]:
            b.flags = [f for f in b.flags if f not in ("placeholder",)]
        self.run = {"mid": mid, "lane": lane, "beats": beats, "i": from_index, "start": line[0], "end": line[1],
                    "down": None, "last_up": None, "auto": False, "tapped": 0}
        t = (beats[from_index].t0 if from_index > 0 and beats[from_index].t0 is not None else line[0]) - 600
        self.player.seek(max(0.0, t / 1000.0))
        self.player.play()
        self.activeBeat.emit(lane, from_index)
        self.status.emit(f"K 轴：按住 K 跟唱（{len(beats) - from_index} 拍），Esc 取消")

    def cancel(self) -> None:
        if self.run is not None:
            lane = self.run["lane"]
            self.run = None
            self.activeBeat.emit(lane, -1)
            self.status.emit("K 轴：已取消，没有改动")

    @property
    def active(self) -> bool:
        return self.run is not None

    # ------------------------------------------------------------ keys
    def eventFilter(self, obj, e) -> bool:  # noqa: N802
        t = e.type()
        if t not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) or not self.enabled:
            return False
        tap_key = any(e.key() == QKeySequence(key)[0].key()
                      for key in self.registry.effective_keys("k.tap", "k"))
        if tap_key and e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            return False
        if e.isAutoRepeat():
            return tap_key and self._accepts_k()
        if e.key() == Qt.Key.Key_Escape and t == QEvent.Type.KeyPress and self.run is not None:
            self.cancel()
            return True
        if not tap_key or e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        if not self._accepts_k():
            return False
        if t == QEvent.Type.KeyPress:
            self._down()
        else:
            self._up()
        return True

    def _accepts_k(self) -> bool:
        if self.accepts_input is not None and not self.accepts_input():
            return False
        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QPlainTextEdit)):
            return False
        if QApplication.activePopupWidget() is not None or QApplication.activeModalWidget() is not None:
            return False
        return True

    def _now(self) -> int:
        return int(round(self.player.time_now() * 1000)) + self.offset_ms

    def _down(self) -> None:
        r = self.run
        now = self._now()
        if r is None:
            if self.on_selected_k is not None and self.on_selected_k():
                return
            if self.player.paused:
                if self.on_idle_k is not None:
                    self.on_idle_k()
                return
            found = self.find_line(now)          # 刷剧连打: the song line under the playhead
            if not found:
                return
            mid, lane, beats, s, e = found
            self.run = r = {"mid": mid, "lane": lane, "beats": [copy.copy(b) for b in beats], "i": 0, "start": s, "end": e,
                            "down": None, "last_up": None, "auto": True, "tapped": 0}
        if r["i"] >= len(r["beats"]):
            return
        b = r["beats"][r["i"]]
        if r["i"] > 0 and r["last_up"] is not None and now - r["last_up"] <= GAP_MS:
            r["beats"][r["i"] - 1].t1 = now                       # legato: no rest between the two beats
        b.t0, b.t1, b.by = now, None, "tap"
        b.flags = [f for f in b.flags if f != "placeholder"]
        r["down"] = now
        self.activeBeat.emit(r["lane"], r["i"])

    def _up(self) -> None:
        r = self.run
        if r is None or r["down"] is None:
            return
        now = self._now()
        b = r["beats"][r["i"]]
        b.t1 = max(now, b.t0 + MIN_BEAT_MS)
        r["last_up"], r["down"] = now, None
        r["i"] += 1
        r["tapped"] += 1
        if r["i"] >= len(r["beats"]):
            self._finish()
        else:
            self.activeBeat.emit(r["lane"], r["i"])

    def on_time(self, t_sec: float) -> None:
        """Playback left the line: finish it (刷剧连打) or stop waiting (single line)."""
        r = self.run
        if r is None or r["down"] is not None:
            return
        t = t_sec * 1000
        if r["auto"] and t > r["end"] + 250:
            self._finish()
        elif not r["auto"] and t > r["end"] + 1500:
            self._finish()
            self.player.pause()

    def _finish(self) -> None:
        r, self.run = self.run, None
        if r is None:
            return
        self.activeBeat.emit(r["lane"], -1)
        if r["tapped"] == 0:
            self.status.emit("K 轴：这一行没有打拍")
            return
        beats = r["beats"]
        last = r["i"]
        # beats that were not reached keep no time (they become placeholders after this line's taps)
        for b in beats[last:]:
            if b.t0 is not None and beats[last - 1].t1 is not None and b.t0 < beats[last - 1].t1:
                b.t0 = b.t1 = None
        n = len(beats)
        msg = f"K 轴：{r['tapped']} 拍已打" + ("" if last >= n else f"，还差 {n - last} 拍（显示为虚线）") + "（Ctrl+Z 撤销）"
        self.status.emit(msg)
        if not r["auto"]:
            self.player.pause()
        self.finished.emit(r["mid"], beats)
