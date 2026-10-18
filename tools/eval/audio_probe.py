"""Experiment: decide heteronym readings from the voice track with Whisper.

For a line and candidate readings, score each candidate transcription by
teacher-forced decoding: score = logP(text | audio) − logP(text | silence).
Subtracting the silence prior removes Whisper's language-model preference, so
only acoustic evidence decides.

Test set: Utena final lines with 私 (user-corrected わたくし vs わたし).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import plain_text  # noqa: E402

from build_utena import UTENA, final_path  # noqa: E402


def video_path(ep: int) -> Path:
    return UTENA / f"origin/S01E{ep:03d} - [LoliHouse] Shoujo Kakumei Utena - {ep:02d} [BDRip 1440x1080 HEVC-10bit AACx2].mkv"


def load_audio(video: Path, start: float, dur: float) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(video),
           "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperScorer:
    def __init__(self, name: str = "medium") -> None:
        import torch
        import whisper

        self.torch = torch
        self.whisper = whisper
        self.model = whisper.load_model(name, device="cpu", download_root='./sample-media/whisper')
        self.tok = whisper.tokenizer.get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages,
                                                   language="ja", task="transcribe")
        self._silence = None

    def encode(self, audio: np.ndarray):
        w = self.whisper
        audio = w.pad_or_trim(audio)
        mel = w.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).unsqueeze(0)
        with self.torch.no_grad():
            return self.model.encoder(mel)

    def logp(self, feats, text: str) -> float:
        toks = list(self.tok.sot_sequence_including_notimestamps) + self.tok.encode(text) + [self.tok.eot]
        x = self.torch.tensor([toks[:-1]])
        with self.torch.no_grad():
            logits = self.model.decoder(x, feats)
        lp = self.torch.log_softmax(logits[0].float(), dim=-1)
        n0 = len(self.tok.sot_sequence_including_notimestamps)
        target = toks[1:]
        return float(sum(lp[i, target[i]] for i in range(n0 - 1, len(target))))

    def silence(self):
        if self._silence is None:
            self._silence = self.encode(np.zeros(16000, dtype=np.float32))
        return self._silence

    def choose(self, audio: np.ndarray, candidates: list[str]) -> list[tuple[str, float]]:
        feats = self.encode(audio)
        sil = self.silence()
        out = [(c, self.logp(feats, c) - self.logp(sil, c)) for c in candidates]
        return sorted(out, key=lambda x: -x[1])


def items(eps):
    for ep in eps:
        doc = AssDocument.load(final_path(ep))
        jp = {(e.start, e.end): plain_text(e.text, breaks="") for e in doc.events if e.is_dialogue and e.style in ("JP", "JPSecondary", "JPTop")}
        for e in doc.events:
            if e.is_comment and e.style == "振假名" and "私（" in e.text:
                m = re.search(r"私（([^,）]+)", e.text)
                line = jp.get((e.start, e.end))
                if m and line and "私" in line:
                    yield ep, e.start / 100, e.end / 100, line, m.group(1)


def main() -> None:
    scorer = WhisperScorer(sys.argv[1] if len(sys.argv) > 1 else "medium")
    ok = total = 0
    t0 = time.time()
    rows = list(items(range(1, 11)))
    # all わたくし cases + an equal number of わたし controls
    wk = [r for r in rows if r[4] == "わたくし"]
    wt = [r for r in rows if r[4] != "わたくし"][: max(10, len(wk))]
    for ep, st, en, line, gold in wk + wt:
        audio = load_audio(video_path(ep), max(0, st - 0.3), (en - st) + 0.6)
        cands = [line.replace("私", r, 1) for r in ("わたし", "わたくし", "あたし")]
        res = scorer.choose(audio, cands)
        pick = ["わたし", "わたくし", "あたし"][cands.index(res[0][0])]
        total += 1
        ok += pick == gold
        print(f"E{ep:02d} {st:7.2f} gold={gold:4} pick={pick:4} {'OK ' if pick == gold else 'BAD'} "
              f"margin={res[0][1] - res[1][1]:5.2f}  {line}", flush=True)
    print(f"accuracy {ok}/{total}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
