"""Sidebar widget for the mRIXS Co-Pilot agent."""
from __future__ import annotations

from typing import TYPE_CHECKING
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QMessageBox,
)

from rixs_app.ui.theme import (
    PALETTE, set_play_btn, set_danger_btn, set_accent_btn, set_danger_secondary_btn,
)
from rixs_app.ui.agent_sidebar.chat_web_view import ChatWebView

if TYPE_CHECKING:
    from rixs_app.agent.bridge import GuiAgentBridge


class ChatInput(QTextEdit):
    """Auto-sending text input: Enter sends, Shift+Enter inserts newline."""

    send_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setMaximumHeight(60)
        self.setPlaceholderText("Ask mRIXS Co-Pilot...")
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {PALETTE['bg_widget']};
                border: 1px solid {PALETTE['border']};
                border-radius: 6px;
                font-size: 13px;
                color: {PALETTE['text']};
                padding: 6px 8px;
            }}
            QTextEdit:focus {{
                border: 1px solid {PALETTE['border_focus']};
            }}
        """)

    def insertFromMimeData(self, source):
        """Strictly paste plain text to prevent rich-text / colored styling injection."""
        self.insertPlainText(source.text())

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
    """Collapsible sidebar hosting the LLM chat interface."""

    sidebar_close_requested = Signal()

    def __init__(
        self,
        bridge: GuiAgentBridge,
        main_window_ref: QWidget,
        parent=None,
        *,
        chat_view: ChatWebView | None = None,
    ):
        super().__init__(parent)
        self.bridge = bridge
        self.main_window_ref = main_window_ref

        # Scope the background to *this* widget only so child button styles
        # from the global QSS (objectName-based) are not clobbered.
        self.setObjectName("agent_sidebar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QWidget#agent_sidebar {{
                background-color: {PALETTE['bg_panel']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(12, 0, 10, 0)
        header_frame = QWidget()
        header_frame.setFixedHeight(38)
        header_frame.setObjectName("sidebar_header")
        header_frame.setStyleSheet(f"""
            QWidget#sidebar_header {{
                background-color: {PALETTE['bg_panel']};
                border-bottom: 1px solid {PALETTE['border']};
            }}
        """)
        header_frame.setLayout(header_layout)

        title = QLabel("🤖 mRIXS Co-Pilot")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.setFixedHeight(26)
        self.clear_btn.setToolTip("Clear conversation history")
        set_danger_secondary_btn(self.clear_btn)
        self.clear_btn.clicked.connect(self._clear_chat)
        header_layout.addWidget(self.clear_btn)

        layout.addWidget(header_frame)

        # ── Chat area ─────────────────────────────────────────────────
        self.chat_view = chat_view if chat_view is not None else ChatWebView()
        layout.addWidget(self.chat_view, stretch=1)

        # ── Bottom control panel ──────────────────────────────────────
        bottom_frame = QWidget()
        bottom_frame.setObjectName("sidebar_bottom")
        bottom_frame.setStyleSheet(f"""
            QWidget#sidebar_bottom {{
                background-color: {PALETTE['bg_panel']};
                border-top: 1px solid {PALETTE['border']};
            }}
        """)
        bottom_layout = QVBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setSpacing(8)

        self.input_field = ChatInput()
        self.input_field.send_requested.connect(self._send_text)
        bottom_layout.addWidget(self.input_field)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(28)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.model_combo.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self.model_combo.currentTextChanged.connect(self.bridge.set_model)
        toolbar_layout.addWidget(self.model_combo)

        self._terminal_access: bool = False
        self.terminal_btn = QPushButton("Full Terminal: OFF")
        self.terminal_btn.setFixedHeight(28)
        self.terminal_btn.setToolTip("Toggle Full Terminal Access (allows AI to execute terminal commands)")
        set_danger_btn(self.terminal_btn)
        self.terminal_btn.clicked.connect(self._toggle_terminal_access)
        toolbar_layout.addWidget(self.terminal_btn)

        toolbar_layout.addStretch()

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(72, 30)
        set_accent_btn(self.send_btn)
        self.send_btn.clicked.connect(self._send_message)
        toolbar_layout.addWidget(self.send_btn)

        bottom_layout.addLayout(toolbar_layout)
        layout.addWidget(bottom_frame)

        # ── Connect bridge signals ────────────────────────────────────
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
        if model_list:
            fm = self.model_combo.fontMetrics()
            max_w = max((fm.horizontalAdvance(m) for m in model_list), default=180) + 40
            self.model_combo.view().setMinimumWidth(max(max_w, 200))

    def get_minimal_gui_context(self) -> str:
        return "GUI Context: User is viewing mRIXS app."

    def _finalize_active_message(self) -> None:
        if self._current_msg_id:
            self.chat_view.finalize_message(self._current_msg_id)
            self._current_msg_id = None

    def _send_text(self, text: str):
        self._finalize_active_message()
        self.input_field.clear()
        self.chat_view.append_user_message(text)
        gui_context = self.get_minimal_gui_context()
        self.bridge.send_message(text, gui_context)

        self.send_btn.setText("Stop ■")
        set_danger_btn(self.send_btn)
        try:
            self.send_btn.clicked.disconnect()
        except Exception:
            pass
        self.send_btn.clicked.connect(self.bridge.cancel)

    def _send_message(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self._send_text(text)

    def _on_token(self, text: str):
        if not self._current_msg_id:
            self._current_msg_id = self.chat_view.start_assistant_message()
        self.chat_view.append_token(self._current_msg_id, text)

    def _on_tool_started(self, call_id: str, name: str, args: str):
        self._finalize_active_message()
        self.chat_view.add_tool_card(call_id, name, args, 'running')

    def _on_tool_output(self, call_id: str, name: str, output: str):
        self._finalize_active_message()
        self.chat_view.update_tool_card(call_id, name, output, 'done')

    def _on_approval_requested(self, callback_id: str, name: str, args: str):
        self._finalize_active_message()
        self.chat_view.add_approval_card(callback_id, name, args)

    def _on_error(self, msg: str):
        self._finalize_active_message()
        self.chat_view.add_error_card(msg)
        self._on_finished()

    def _on_finished(self):
        self._finalize_active_message()
        self.send_btn.setText("Send")
        set_accent_btn(self.send_btn)
        try:
            self.send_btn.clicked.disconnect()
        except Exception:
            pass
        self.send_btn.clicked.connect(self._send_message)

    def _on_cli_line(self, line: str):
        self._finalize_active_message()
        self.chat_view.add_cli_line(line)

    def _clear_chat(self):
        self.bridge.reset()
        self.chat_view.clear_all()
        self._current_msg_id = None

    def _update_terminal_btn(self) -> None:
        """Update button text and color based on full terminal access state."""
        if self._terminal_access:
            self.terminal_btn.setText("Full Terminal: ON")
            set_play_btn(self.terminal_btn)
        else:
            self.terminal_btn.setText("Full Terminal: OFF")
            set_danger_btn(self.terminal_btn)

    def _toggle_terminal_access(self) -> None:
        """Toggle full terminal access state with confirmation when enabling."""
        if not self._terminal_access:
            reply = QMessageBox.warning(
                self,
                "Warning: Full Terminal Access",
                "Granting full terminal access allows the AI to execute "
                "arbitrary commands on your system. Proceed with caution.",
                QMessageBox.Ok | QMessageBox.Cancel
            )
            if reply == QMessageBox.Cancel:
                return
            self._terminal_access = True
        else:
            self._terminal_access = False

        self._update_terminal_btn()
        self.bridge.set_terminal_access(self._terminal_access)

    @property
    def terminal_access(self) -> bool:
        """Current full terminal access state."""
        return self._terminal_access

    @terminal_access.setter
    def terminal_access(self, enabled: bool) -> None:
        """Set terminal access state programmatically."""
        self._terminal_access = enabled
        self._update_terminal_btn()
        self.bridge.set_terminal_access(enabled)

