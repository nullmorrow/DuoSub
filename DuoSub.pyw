#! python3.12
"""Double-click to start 双语字幕工坊 / DuoSub (no PYTHONPATH, no command line needed).

Until the packaged DuoSub.exe exists (stage 4), this launcher uses the installed
Python (pythonw.exe is associated with .pyw files on Windows).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from subkit_app.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
