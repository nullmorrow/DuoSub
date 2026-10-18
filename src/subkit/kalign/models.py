"""Lazy model inference for the isolated worker; no torch imports at startup."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .song import ModelEmission

MMS_ID = "torchaudio.pipelines.MMS_FA:with_star=True"
KANA_ID = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
KANA_REVISION = "017225bb128a6b1c6de9d58391f891908d69bc7b"
DEMUCS_ID = "htdemucs:955717e8"


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


def checkpoint_download_needed(torch, filename, download=None):
    """torch.hub uses the URL basename under <hub>/checkpoints."""
    missing = not (Path(torch.hub.get_dir()) / "checkpoints" / filename).is_file()
    if missing and download:
        download()
    return missing


def load_kana(processor_type, model_type, download=None):
    """Prefer cached pinned weights; notify before permitting network access."""
    kwargs = {"revision": KANA_REVISION, "local_files_only": True}
    try:
        processor = processor_type.from_pretrained(KANA_ID, **kwargs)
        model = model_type.from_pretrained(KANA_ID, **kwargs)
    except OSError:
        if download:
            download()
        kwargs["local_files_only"] = False
        processor = processor_type.from_pretrained(KANA_ID, **kwargs)
        model = model_type.from_pretrained(KANA_ID, **kwargs)
    return processor, model


def separate_vocals(mix_stereo_44k, download=None):
    """Return float32 stereo vocals at 44.1 kHz, using prototype normalization."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    mix = np.asarray(mix_stereo_44k, dtype=np.float32)
    if mix.ndim != 2 or mix.shape[1] != 2 or len(mix) < 2 or not np.isfinite(mix).all():
        raise ValueError("expected finite stereo audio at 44.1 kHz")
    device = _device(torch)
    if download:
        checkpoint_download_needed(torch, "955717e8-8726e21a.th", download)
    model = get_model("htdemucs").eval().to(device)
    x = torch.from_numpy(mix.T.copy())
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std()
    if float(std) == 0:
        return np.zeros_like(mix)
    x = (x - mean) / std
    with torch.inference_mode():
        out = apply_model(model, x[None], device=device, shifts=1, split=True,
                          overlap=0.25, progress=False)[0]
    vocals = (out[model.sources.index("vocals")].cpu() * std + mean).numpy().T
    return vocals.astype(np.float32, copy=False)


def chunk_ranges(samples: int, seconds: int = 30):
    """Core sample intervals and 1-second context; every core sample occurs once."""
    step, context = seconds * 16000, 16000
    for a in range(0, samples, step):
        b = min(samples, a + step)
        yield a, b, max(0, a - context), min(samples, b + context)


def chunk_crop(a, b, lo, hi, frames):
    """Prototype nearest-frame context removal, including the short final chunk."""
    fps = frames / (hi - lo)
    start = int(round((a - lo) * fps))
    return start, min(frames, start + int(round((b - a) * fps)))


def _mono(audio):
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1 or len(x) < 400 or not np.isfinite(x).all():
        raise ValueError("expected at least 25 ms of finite mono 16 kHz audio")
    return x


def mms_emission(vocals16k, progress=None, download=None):
    import torch
    import torchaudio

    audio = _mono(vocals16k)
    device = _device(torch)
    bundle = torchaudio.pipelines.MMS_FA
    vocab = bundle.get_dict(star="*")
    if download:
        checkpoint_download_needed(torch, "model.pt", download)
    model = bundle.get_model(with_star=True).eval().to(device)
    # Normalize the whole clip, not each context independently (prototype).
    x = torch.from_numpy(audio)
    x = (x - x.mean()) / (x.std() + 1e-7) * 0.1
    ranges = list(chunk_ranges(len(audio)))
    chunks = []
    with torch.inference_mode():
        for i, (a, b, lo, hi) in enumerate(ranges):
            logits, _ = model(x[None, lo:hi].to(device))
            logits = logits[0]
            s, e = chunk_crop(a, b, lo, hi, len(logits))
            chunks.append(torch.log_softmax(logits[s:e].float(), dim=-1).cpu().numpy())
            if progress:
                progress(i + 1, len(ranges))
    lp = np.concatenate(chunks)
    return ModelEmission("mms", lp, len(audio) / 16 / len(lp), vocab, 0, vocab["*"])


def kana_emission(vocals16k, progress=None, download=None):
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    audio = _mono(vocals16k)
    device = _device(torch)
    processor, model = load_kana(AutoProcessor, AutoModelForCTC, download)
    model = model.eval().to(device)
    vocab = processor.tokenizer.get_vocab()
    # Keep the measured prototype's 20-second kana chunks.
    ranges = list(chunk_ranges(len(audio), seconds=20))
    chunks = []
    with torch.inference_mode():
        for i, (a, b, lo, hi) in enumerate(ranges):
            x = processor(audio[lo:hi], sampling_rate=16000, return_tensors="pt").input_values
            logits = model(x.to(device)).logits[0]
            s, e = chunk_crop(a, b, lo, hi, len(logits))
            chunks.append(torch.log_softmax(logits[s:e].float(), dim=-1).cpu().numpy())
            if progress:
                progress(i + 1, len(ranges))
    lp = np.concatenate(chunks)
    return ModelEmission("kana", lp, len(audio) / 16 / len(lp), vocab,
                         vocab[processor.tokenizer.pad_token])


def rms_db_10ms(vocals, rate=44100):
    """Prototype centred 50 ms RMS windows sampled every 10 ms."""
    audio = np.asarray(vocals, dtype=np.float32)
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    if mono.ndim != 1 or not np.isfinite(mono).all() or rate < 100 or rate % 100:
        raise ValueError("expected finite vocals and sample rate divisible by 100")
    hop = rate // 100
    padded = np.pad(mono, (hop * 2, hop * 3))
    frames = np.lib.stride_tricks.sliding_window_view(padded, hop * 5)[::hop][:len(mono) // hop + 1]
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
