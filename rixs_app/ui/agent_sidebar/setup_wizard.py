"""Setup wizard for configuring the CBORG API Key."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRunnable, QThreadPool, Signal, QObject, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QMessageBox
)

from rixs_app.agent.auth import verify_connection, save_api_key
from rixs_app.ui.theme import set_accent_btn, set_success_btn, set_cancel_btn, PALETTE


class _WorkerSignals(QObject):
    finished = Signal(bool, str)


class _VerifyWorker(QRunnable):
    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.signals = _WorkerSignals()

    def run(self):
        success, msg = verify_connection(self.key)
        self.signals.finished.emit(success, msg)


class CBORGSetupWizard(QDialog):
    """Dialog for entering and validating the CBORG API key."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CBORG API Setup")
        self.setFixedSize(450, 400)
        self.setStyleSheet(f"background-color: {PALETTE['bg_panel']}; color: {PALETTE['text']};")
        self._api_key: str | None = None
        
        layout = QVBoxLayout(self)
        
        title_label = QLabel("🔑 CBORG API Key Setup")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        
        inst_label = QLabel(
            "The mRIXS Co-Pilot uses LBNL's CBORG API for AI-powered assistance.\n"
            "You need a CBORG API key to use this feature."
        )
        inst_label.setWordWrap(True)
        layout.addWidget(inst_label)
        
        link_btn = QPushButton("Open CBORG Key Manager ↗")
        link_btn.setStyleSheet(f"color: {PALETTE['accent_blue']}; text-align: left; background: transparent; border: none; text-decoration: underline;")
        link_btn.setCursor(Qt.PointingHandCursor)
        link_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://api.cborg.lbl.gov/key/manage")))
        layout.addWidget(link_btn)
        
        note_label = QLabel("Note: CBORG allows one key per account. If you've forgotten yours, delete and recreate it.")
        note_label.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px;")
        note_label.setWordWrap(True)
        layout.addWidget(note_label)
        
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setPlaceholderText("Paste your API key here")
        layout.addWidget(self.key_input)
        
        self.test_btn = QPushButton("Test Connection")
        set_accent_btn(self.test_btn)
        self.test_btn.clicked.connect(self._on_test_clicked)
        layout.addWidget(self.test_btn)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        set_cancel_btn(self.cancel_btn)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.save_btn = QPushButton("Save & Continue")
        set_success_btn(self.save_btn)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_clicked)
        
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        
    @property
    def api_key(self) -> str | None:
        return self._api_key
        
    def _on_test_clicked(self):
        key = self.key_input.text().strip()
        if not key:
            self.status_label.setText("Please enter an API key.")
            self.status_label.setStyleSheet(f"color: {PALETTE['text_error']};")
            return
            
        self.test_btn.setEnabled(False)
        self.status_label.setText("Testing connection...")
        self.status_label.setStyleSheet(f"color: {PALETTE['text_dim']};")
        
        worker = _VerifyWorker(key)
        worker.signals.finished.connect(self._on_test_finished)
        QThreadPool.globalInstance().start(worker)
        
    def _on_test_finished(self, success: bool, msg: str):
        self.test_btn.setEnabled(True)
        if success:
            self.status_label.setText(f"✓ {msg}")
            self.status_label.setStyleSheet(f"color: {PALETTE['accent_green']}; font-weight: bold;")
            self.save_btn.setEnabled(True)
        else:
            self.status_label.setText(f"❌ {msg}")
            self.status_label.setStyleSheet(f"color: {PALETTE['text_error']};")
            self.save_btn.setEnabled(False)
            
    def _on_save_clicked(self):
        key = self.key_input.text().strip()
        if key:
            save_api_key(key)
            self._api_key = key
            self.accept()
