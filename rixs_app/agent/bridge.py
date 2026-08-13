from __future__ import annotations

import asyncio
from typing import Any
from PySide6.QtCore import QObject, Signal, QThread

from rixs_app.agent.engine import CborgAgentEngine, AgentEvent


class _AsyncLoopThread(QThread):
    """Worker thread running an asyncio event loop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.wait(3000)


class GuiAgentBridge(QObject):
    """Bridge between async agent engine and Qt main thread."""
    
    token_received = Signal(str)
    tool_started = Signal(str, str)
    tool_output_received = Signal(str, str)
    approval_requested = Signal(str, str, str)
    error_raised = Signal(str)
    generation_finished = Signal()
    cli_stdout_line = Signal(str)

    def __init__(self, engine: CborgAgentEngine, parent=None):
        """Initializes the Agent Bridge.
        
        Args:
            engine (CborgAgentEngine): The initialized agent engine.
            parent: The QObject parent.
        """
        super().__init__(parent)
        self.engine = engine
        self._thread = _AsyncLoopThread(self)
        self._is_generating = False

    def start_worker(self) -> None:
        """Starts the background worker thread."""
        self._thread.start()

    def stop_worker(self) -> None:
        """Stops the background worker thread."""
        self._thread.stop()

    def is_generating(self) -> bool:
        """Checks if generation is active."""
        return self._is_generating

    def send_message(self, text: str, gui_context: str) -> None:
        """Sends a message to the agent engine.
        
        Args:
            text (str): The user's message.
            gui_context (str): The current GUI context.
        """
        if self._is_generating:
            return
            
        self._is_generating = True
        asyncio.run_coroutine_threadsafe(self._run_chat(text, gui_context), self._thread.loop)

    async def _run_chat(self, text: str, gui_context: str) -> None:
        """Internal coroutine handling the chat stream on the worker thread."""
        try:
            async for event in self.engine.stream_chat(text, gui_context):
                if event.type == 'token':
                    self.token_received.emit(event.content)
                elif event.type == 'tool_start':
                    args = event.metadata.get('args', '') if event.metadata else ''
                    self.tool_started.emit(event.content, args)
                elif event.type == 'tool_output':
                    output = event.metadata.get('output', '') if event.metadata else ''
                    self.tool_output_received.emit(event.content, output)
                elif event.type == 'approval_required':
                    callback_id = event.metadata.get('callback_id', '') if event.metadata else ''
                    args = event.metadata.get('args', '') if event.metadata else ''
                    self.approval_requested.emit(callback_id, event.content, args)
                elif event.type == 'done':
                    pass
        except Exception as e:
            self.error_raised.emit(str(e))
        finally:
            self._is_generating = False
            self.generation_finished.emit()

    def cancel(self) -> None:
        """Cancels generation."""
        self.engine.cancel()

    def approve_tool(self, callback_id: str) -> None:
        """Approves a tool execution."""
        self.engine.approve_tool(callback_id)

    def reject_tool(self, callback_id: str, feedback: str) -> None:
        """Rejects a tool execution."""
        self.engine.reject_tool(callback_id, feedback)

    def set_model(self, model_id: str) -> None:
        """Updates the engine model."""
        self.engine.set_model(model_id)

    def set_terminal_access(self, enabled: bool) -> None:
        """Updates terminal access for the engine."""
        self.engine.set_terminal_access(enabled)

    def reset(self) -> None:
        """Resets the conversation history."""
        self.engine.reset()
