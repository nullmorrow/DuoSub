"""Decide ambiguous readings from the voice track (optional, slow on CPU).

For a JP line and a word with candidate readings, each candidate is written
into the line in kana and scored with Whisper by forced decoding:

    score = log P(text | line audio) − log P(text | silence)

Subtracting the silence score cancels Whisper's language-model preference
(わたしたち is simply more common than わたくしたち), leaving the acoustic
evidence.  Only clear margins are applied; the rest stay marked ``?``.
Results are cached in .cache/voice.sqlite3.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..audio import ffmpeg_audio_index


class WhisperScorer:
    def __init__(self, model: str = "medium", download_root: Optional[str] = None) -> None:
        import torch
        import whisper

        self.torch = torch
        self.whisper = whisper
        kw = {"download_root": download_root} if download_root else {}
        self.model = whisper.load_model(model, device="cuda" if torch.cuda.is_available() else "cpu", **kw)
        self.tok = whisper.tokenizer.get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages,
                                                   language="ja", task="transcribe")
        self._silence = None
        self.name = model

    def encode(self, audio: np.ndarray):
        w = self.whisper
        mel = w.log_mel_spectrogram(w.pad_or_trim(audio), n_mels=self.model.dims.n_mels).unsqueeze(0).to(self.model.device)
        with self.torch.no_grad():
            return self.model.encoder(mel)

    def logp(self, feats, text: str) -> float:
        sot = list(self.tok.sot_sequence_including_notimestamps)
        toks = sot + self.tok.encode(text) + [self.tok.eot]
        x = self.torch.tensor([toks[:-1]], device=self.model.device)
        with self.torch.no_grad():
            logits = self.model.decoder(x, feats)
        lp = self.torch.log_softmax(logits[0].float(), dim=-1)
        target = toks[1:]
        return float(sum(lp[i, target[i]] for i in range(len(sot) - 1, len(target))))

    def scores(self, audio: np.ndarray, texts: list[str]) -> list[float]:
        feats = self.encode(audio)
        if self._silence is None:
            self._silence = self.encode(np.zeros(16000, dtype=np.float32))
        return [self.logp(feats, t) - self.logp(self._silence, t) for t in texts]


def load_audio(video: Path, start: float, dur: float, ffmpeg: str = "ffmpeg", *,
               audio_track: Optional[int] = None) -> np.ndarray:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, start):.2f}", "-t", f"{dur:.2f}", "-i", str(video),
           "-map", f"0:a:{ffmpeg_audio_index(audio_track)}", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    from ..proc import run as _run_hidden

    raw = _run_hidden(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class VoiceCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, scores TEXT)")

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, k: str) -> Optional[list[float]]:
        row = self.db.execute("SELECT scores FROM v WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, scores: list[float]) -> None:
        self.db.execute("INSERT OR REPLACE INTO v VALUES (?,?)", (k, json.dumps(scores)))
        self.db.commit()
