"""Web engine view for rendering chat messages.

Uses QWebChannel for reliable JavaScript → Python communication,
replacing the fragile ``document.title`` hack that failed on
PySide6/macOS when the page was loaded via ``setHtml()``.

Includes a JS call queue that buffers ``runJavaScript`` calls until
the page is fully loaded — prevents silent failures when the widget
is pre-created in hidden state for instant sidebar opening.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal, Slot, QUrl, QObject
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView


class _ChatBridge(QObject):
    """QObject exposed to JavaScript via QWebChannel.

    JavaScript calls ``bridge.approve(id)``, ``bridge.reject(id, fb)``, or
    ``bridge.quick_prompt(text)`` and these slots emit Qt signals back to the Python side.
    """

    approval_given = Signal(str)
    approval_rejected = Signal(str, str)
    quick_prompt_requested = Signal(str)

    @Slot(str)
    def approve(self, callback_id: str) -> None:
        """Called from JS when the user clicks Approve."""
        self.approval_given.emit(callback_id)

    @Slot(str, str)
    def reject(self, callback_id: str, feedback: str) -> None:
        """Called from JS when the user clicks Reject."""
        self.approval_rejected.emit(callback_id, feedback)

    @Slot(str)
    def quick_prompt(self, text: str) -> None:
        """Called from JS when the user clicks a quick prompt chip."""
        self.quick_prompt_requested.emit(text)


class ChatWebView(QWebEngineView):
    """View component that renders the chat interface via HTML/JS.

    Safe to create before the widget is visible — JS calls are queued
    until ``loadFinished`` fires.
    """

    approval_given = Signal(str)
    approval_rejected = Signal(str, str)
    quick_prompt_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().setBackgroundColor(QColor('#1a1a2e'))

        self._page_ready = False
        self._js_queue: list[str] = []

        # ── QWebChannel bridge (must be set up BEFORE loading HTML) ───
        self._chat_bridge = _ChatBridge(self)
        self._channel = QWebChannel(self.page())
        self._channel.registerObject("bridge", self._chat_bridge)
        self.page().setWebChannel(self._channel)

        # Wire bridge signals → our own public signals
        self._chat_bridge.approval_given.connect(self.approval_given.emit)
        self._chat_bridge.approval_rejected.connect(
            lambda cid, fb: self.approval_rejected.emit(cid, fb)
        )
        self._chat_bridge.quick_prompt_requested.connect(self.quick_prompt_requested.emit)

        # Track page readiness
        self.loadFinished.connect(self._on_load_finished)

        # ── Load HTML template ────────────────────────────────────────
        template_dir = Path(__file__).parent / 'templates'
        template_path = template_dir / 'chat.html'
        html_content = template_path.read_text(encoding='utf-8')
        self.setHtml(html_content, QUrl.fromLocalFile(str(template_dir) + '/'))

    # ── Page readiness ────────────────────────────────────────────────

    def _on_load_finished(self, ok: bool) -> None:
        """Flush any queued JS calls once the page is ready."""
        self._page_ready = True
        for code in self._js_queue:
            self.page().runJavaScript(code)
        self._js_queue.clear()

    def _run_js(self, code: str) -> None:
        """Run JavaScript, queuing if the page hasn't finished loading."""
        if self._page_ready:
            self.page().runJavaScript(code)
        else:
            self._js_queue.append(code)

    # ── Public API (called from SidebarWidget) ────────────────────────

    def append_user_message(self, text: str) -> None:
        self._run_js(f'appendUserMessage({json.dumps(text)})')

    def start_assistant_message(self) -> str:
        import uuid
        msg_id = uuid.uuid4().hex[:8]
        self._run_js(f'startAssistantMessage("{msg_id}")')
        return msg_id

    def append_token(self, msg_id: str, token: str) -> None:
        self._run_js(f'appendToken("{msg_id}", {json.dumps(token)})')

    def finalize_message(self, msg_id: str) -> None:
        self._run_js(f'finalizeMessage("{msg_id}")')

    def add_tool_card(self, call_id: str, tool_name: str, args_json: str, status: str = 'running') -> None:
        self._run_js(f'addToolCard({json.dumps(call_id)}, {json.dumps(tool_name)}, {json.dumps(args_json)}, {json.dumps(status)})')

    def update_tool_card(self, call_id: str, tool_name: str, output: str, status: str = 'done') -> None:
        self._run_js(f'updateToolCard({json.dumps(call_id)}, {json.dumps(tool_name)}, {json.dumps(output)}, {json.dumps(status)})')

    def add_approval_card(self, callback_id: str, tool_name: str, args_json: str) -> None:
        self._run_js(f'addApprovalCard({json.dumps(callback_id)}, {json.dumps(tool_name)}, {json.dumps(args_json)})')

    def add_error_card(self, error: str) -> None:
        self._run_js(f'addErrorCard({json.dumps(error)})')

    def add_cli_line(self, line: str) -> None:
        self._run_js(f'addCLILine({json.dumps(line)})')

    def finalize_cli(self) -> None:
        """Mark the terminal panel as complete (stops the pulsing dot)."""
        self._run_js('finalizeCLI()')

    def clear_all(self) -> None:
        self._run_js('clearAll()')
