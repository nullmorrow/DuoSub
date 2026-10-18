"""Singing alignment: beat tokens, windowed CTC, then song-level timing.

This Qt-free package keeps optional separation/recognition dependencies lazy.
The GUI may probe availability without importing models or loading weights.
"""

from importlib.util import find_spec


def available() -> tuple[bool, list[str]]:
    """Return whether model dependencies are installed, and missing module names.

    This is a discovery probe only; native-library usability is checked by the
    worker when it actually imports the dependencies.
    """
    missing = []
    for name in ("torch", "torchaudio", "transformers", "demucs", "soundfile"):
        try:
            found = find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return not missing, missing
