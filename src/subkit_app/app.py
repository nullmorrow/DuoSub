"""DuoSub (双语字幕工坊, formerly SubKit) desktop application entry point."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def install_chinese_translator(app) -> bool:
    """Keep Qt's bundled translator alive for the application's lifetime; missing files are harmless."""
    from PySide6.QtCore import QLibraryInfo, QTranslator

    translator = QTranslator(app)
    loaded = translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
    if loaded:
        app.installTranslator(translator)
    app._chinese_translator = translator
    return loaded


ICON = Path(__file__).resolve().parents[2] / "assets" / "duosub.ico"


def set_app_identity() -> None:
    """Windows groups taskbar buttons by process: give DuoSub its own identity so the taskbar shows the
    window icon instead of pythonw's."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SubKit.DuoSub")
    except (AttributeError, OSError):
        pass


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never download models implicitly
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication

    from .mainwindow import MainWindow
    from .help import critical_message

    set_app_identity()
    app = QApplication(argv if argv is not None else sys.argv)
    if ICON.is_file():
        app.setWindowIcon(QIcon(str(ICON)))
    install_chinese_translator(app)
    from .gc_guard import install as install_gc_guard

    install_gc_guard(app)          # cyclic garbage is collected on the GUI thread only
    app.setApplicationName("DuoSub")
    app.setOrganizationName("SubKit")                     # pre-rename settings scope (see MainWindow.settings)
    f = QFont()
    f.setFamilies(["Microsoft YaHei UI", "Yu Gothic UI", "Segoe UI"])
    f.setPointSizeF(10)
    app.setFont(f)
    app.setStyle("Fusion")
    from . import APP_VERSION
    from .crash_log import install as install_crash_log

    crash_log = install_crash_log(app, APP_VERSION)

    def excepthook(t, v, tb):
        msg = "".join(traceback.format_exception(t, v, tb))
        crash_log.exception(t, v, tb)
        if sys.stderr is not None:            # pythonw has no console
            sys.stderr.write(msg)
        try:
            # Parent to the active window so the box is never hidden behind the main window.
            parent = QApplication.activeModalWidget() or QApplication.activeWindow()
            critical_message(parent, "双语字幕工坊 出错", msg[-2000:])
        except Exception:
            pass

    sys.excepthook = excepthook
    from .single_instance import activate_window, prepare_instance

    argv_all = list(argv if argv is not None else sys.argv)[1:]
    try:
        instance, primary = prepare_instance(argv_all, app)
    except RuntimeError as exc:
        critical_message(None, "双语字幕工坊", str(exc))
        return 1
    if not primary:
        instance.close()
        return 0
    w = MainWindow()
    if instance is not None:
        app.aboutToQuit.connect(instance.close)
    w.show()
    args = [a for a in argv_all if not a.startswith("-")]
    if args and os.path.isdir(args[0]):
        w.open_project(__import__("pathlib").Path(args[0]))
    if instance is not None:
        instance.set_handler(lambda arguments: activate_window(w, arguments))
    if "--smoke-test" in argv_all:
        # start, open the window (and a project if given), then quit: used to verify the launcher
        from PySide6.QtCore import QTimer

        def done() -> None:
            ok = w.isVisible() and (not args or w.session is not None or w.project is not None)
            log = os.environ.get("SUBKIT_SMOKE_LOG")
            if log:
                with open(log, "w", encoding="utf-8") as fh:
                    fh.write(f"ok={ok} project={w.project.name if w.project else None} "
                             f"groups={len(w.session.doc.groups) if w.session else 0}\n")
            w.close()             # the normal close path stops and waits for background measuring
            app.exit(0 if ok else 3)

        QTimer.singleShot(4000, done)
    try:
        return app.exec()
    finally:
        if instance is not None:
            instance.close()


if __name__ == "__main__":
    sys.exit(main())
