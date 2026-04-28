# run_qt.py

from __future__ import annotations

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon
from pathlib import Path
import os

APP_NAME = "CepstralVox"
APP_VERSION = "2.0.0"


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Fonotech Academy")
    icon_path = Path(__file__).resolve().parent / "logo.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        from main import CepstralVoxQt

        window = CepstralVoxQt()
        window.show()

        sys.exit(app.exec())

    except Exception as exc:
        traceback.print_exc()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("CepstralVox startup error")
        msg.setText("CepstralVox could not start.")
        msg.setDetailedText(f"{exc}\n\n{traceback.format_exc()}")
        msg.exec()

        sys.exit(1)


if __name__ == "__main__":
    main()