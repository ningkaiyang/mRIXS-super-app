"""Sidebar widget for the mRIXS Co-Pilot agent."""
from __future__ import annotations

from typing import TYPE_CHECKING
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QCheckBox, QMessageBox, QSpacerItem, QSizePolicy
)

from rixs_app.ui.theme import PALETTE, set_tool_btn, set_danger_btn, set_accent_btn, set_danger_secondary_btn
from rixs_app.ui.agent_sidebar.chat_web_view import ChatWebView

if TYPE_CHECKING:
    from rixs_app.agent.bridge import GuiAgentBridge

class ChatInput(QTextEdit):
    send_requested = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(60)
        self.setPlaceholderText("Ask mRIXS Co-Pilot...")
        self.setStyleSheet(f"background-color: {PALETTE['bg_widget']}; border: 1px solid {PALETTE['border']}; border-radius: 4px; font-size: 13px; color: {PALETTE['text']};")
        
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                text = self.toPlainText().strip()
                if text:
                    self.send_requested.emit(text)
                event.accept()
        else:
            super().keyPressEvent(event)


class AgentSidebarWidget(QWidget):
    sidebar_close_requested = Signal()
    
    def __init__(self, bridge: GuiAgentBridge, main_window_ref: QWidget, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.main_window_ref = main_window_ref
        
        self.setStyleSheet(f"background-color: {PALETTE['bg_panel']};")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header bar
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 8)
        header_frame = QWidget()
        header_frame.setStyleSheet(f"border-bottom: 1px solid {PALETTE['border']};")
        header_frame.setLayout(header_layout)
        
        title = QLabel("🤖 mRIXS Co-Pilot")
        title.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.clear_btn = QPushButton("🗑")
        self.clear_btn.setFixedSize(24, 24)
        self.clear_btn.setToolTip("Clear conversation")
        set_tool_btn(self.clear_btn)
        self.clear_btn.clicked.connect(self._clear_chat)
        header_layout.addWidget(self.clear_btn)
        
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setToolTip("Close sidebar")
        set_danger_btn(self.close_btn)
        self.close_btn.clicked.connect(self.sidebar_close_requested.emit)
        header_layout.addWidget(self.close_btn)
        
        layout.addWidget(header_frame)
        
        # Chat area
        self.chat_view = ChatWebView()
        layout.addWidget(self.chat_view)
        
        # Bottom control panel
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        
        self.input_field = ChatInput()
        self.input_field.send_requested.connect(self._send_text)
        bottom_layout.addWidget(self.input_field)
        
        toolbar_layout = QHBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.bridge.set_model)
        toolbar_layout.addWidget(self.model_combo)
        
        self.terminal_cb = QCheckBox("Full Terminal Access")
        self.terminal_cb.toggled.connect(self._on_terminal_toggled)
        toolbar_layout.addWidget(self.terminal_cb)
        
        toolbar_layout.addStretch()
        
        self.send_btn = QPushButton("Send")
        set_accent_btn(self.send_btn)
        self.send_btn.clicked.connect(self._send_message)
        toolbar_layout.addWidget(self.send_btn)
        
        bottom_layout.addLayout(toolbar_layout)
        layout.addLayout(bottom_layout)
        
        # Connect bridge signals
        self.bridge.token_received.connect(self._on_token)
        self.bridge.tool_started.connect(self._on_tool_started)
        self.bridge.tool_output_received.connect(self._on_tool_output)
        self.bridge.approval_requested.connect(self._on_approval_requested)
        self.bridge.error_raised.connect(self._on_error)
        self.bridge.generation_finished.connect(self._on_finished)
        self.bridge.cli_stdout_line.connect(self._on_cli_line)
        
        self.chat_view.approval_given.connect(self.bridge.approve_tool)
        self.chat_view.approval_rejected.connect(self.bridge.reject_tool)
        
        self._current_msg_id = None

    def populate_models(self, model_list: list[str]) -> None:
        self.model_combo.clear()
        self.model_combo.addItems(model_list)

    def get_minimal_gui_context(self) -> str:
        return "GUI Context: User is viewing mRIXS app."

    def _send_text(self, text: str):
        self.input_field.clear()
        self.chat_view.append_user_message(text)
        gui_context = self.get_minimal_gui_context()
        self.bridge.send_message(text, gui_context)
        
        self.send_btn.setText("Stop ■")
        set_danger_btn(self.send_btn)
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self.bridge.cancel)
        
    def _send_message(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self._send_text(text)
            
    def _on_token(self, text: str):
        if not self._current_msg_id:
            self._current_msg_id = self.chat_view.start_assistant_message()
        self.chat_view.append_token(self._current_msg_id, text)
        
    def _on_tool_started(self, name: str, args: str):
        self.chat_view.add_tool_card(name, args, 'running')
        
    def _on_tool_output(self, name: str, output: str):
        self.chat_view.update_tool_card(name, output, 'done')
        
    def _on_approval_requested(self, callback_id: str, name: str, args: str):
        self.chat_view.add_approval_card(callback_id, name, args)
        
    def _on_error(self, msg: str):
        self.chat_view.add_error_card(msg)
        self._on_finished()
        
    def _on_finished(self):
        if self._current_msg_id:
            self.chat_view.finalize_message(self._current_msg_id)
            self._current_msg_id = None
            
        self.send_btn.setText("Send")
        set_accent_btn(self.send_btn)
        self.send_btn.clicked.disconnect()
        self.send_btn.clicked.connect(self._send_message)
        
    def _on_cli_line(self, line: str):
        self.chat_view.add_cli_line(line)
        
    def _clear_chat(self):
        self.bridge.reset()
        self.chat_view.clear_all()
        self._current_msg_id = None
        
    def _on_terminal_toggled(self, checked: bool):
        if checked:
            reply = QMessageBox.warning(
                self,
                "Warning: Full Terminal Access",
                "Granting full terminal access allows the AI to execute arbitrary commands on your system. Proceed with caution.",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                self.terminal_cb.setChecked(False)
                return
        self.bridge.set_terminal_access(checked)
