"""Web engine view for rendering chat messages."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QColor
from PySide6.QtWebEngineWidgets import QWebEngineView


class ChatWebView(QWebEngineView):
    """View component that renders the chat interface via HTML/JS."""
    
    approval_given = Signal(str)
    approval_rejected = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.page().setBackgroundColor(QColor('#1a1a2e'))
        
        template_dir = Path(__file__).parent / 'templates'
        template_path = template_dir / 'chat.html'
        html_content = template_path.read_text(encoding='utf-8')
        self.setHtml(html_content, QUrl.fromLocalFile(str(template_dir) + '/'))
        
        self.page().titleChanged.connect(self._on_title_changed)
        
    def _on_title_changed(self, title: str):
        try:
            cmd = json.loads(title)
            if cmd.get('action') == 'approve':
                self.approval_given.emit(cmd['callbackId'])
            elif cmd.get('action') == 'reject':
                self.approval_rejected.emit(cmd['callbackId'], cmd.get('feedback', ''))
        except (json.JSONDecodeError, KeyError):
            pass

    def _run_js(self, code: str) -> None:
        self.page().runJavaScript(code)
    
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
    
    def add_tool_card(self, tool_name: str, args_json: str, status: str = 'running') -> None:
        self._run_js(f'addToolCard({json.dumps(tool_name)}, {json.dumps(args_json)}, {json.dumps(status)})')
    
    def update_tool_card(self, tool_name: str, output: str, status: str = 'done') -> None:
        self._run_js(f'updateToolCard({json.dumps(tool_name)}, {json.dumps(output)}, {json.dumps(status)})')
    
    def add_approval_card(self, callback_id: str, tool_name: str, args_json: str) -> None:
        self._run_js(f'addApprovalCard({json.dumps(callback_id)}, {json.dumps(tool_name)}, {json.dumps(args_json)})')
    
    def add_error_card(self, error: str) -> None:
        self._run_js(f'addErrorCard({json.dumps(error)})')
    
    def add_cli_line(self, line: str) -> None:
        self._run_js(f'addCLILine({json.dumps(line)})')
    
    def clear_all(self) -> None:
        self._run_js('clearAll()')
