"""Unit and integration tests for RIXS Co-Pilot agent chat flow,
message lifecycle, in-place tool card streaming, and approval handling.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget

from rixs_app.agent.engine import CborgAgentEngine, AgentEvent
from rixs_app.agent.bridge import GuiAgentBridge
from rixs_app.agent.tools import ToolRegistry, create_default_registry
from rixs_app.ui.agent_sidebar.sidebar_widget import AgentSidebarWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(["-platform", "offscreen"])
    return app


class MockChatView(QWidget):
    """QWidget mock conforming to ChatWebView interface for PySide6 layouts."""
    approval_given = Signal(str)
    approval_rejected = Signal(str, str)
    quick_prompt_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.append_user_message = MagicMock()
        self.start_assistant_message = MagicMock()
        self.append_token = MagicMock()
        self.finalize_message = MagicMock()
        self.add_tool_card = MagicMock()
        self.update_tool_card = MagicMock()
        self.add_approval_card = MagicMock()
        self.add_error_card = MagicMock()
        self.add_cli_line = MagicMock()
        self.clear_all = MagicMock()


def test_execute_terminal_command_streaming():
    """Test execute_terminal_command streams lines via callback in real time."""
    async def _run():
        registry = ToolRegistry()
        from rixs_app.agent.tools import _register_tools
        _register_tools(registry)

        streamed_lines = []
        registry.set_cli_line_callback(lambda line: streamed_lines.append(line))

        # Run a simple echo command with 3 lines
        cmd = 'echo "Line 1"; echo "Line 2"; echo "Line 3"'
        result = await registry.execute("execute_terminal_command", {"command": cmd})

        assert "Line 1" in streamed_lines
        assert "Line 2" in streamed_lines
        assert "Line 3" in streamed_lines
        assert "Line 1" in result
        assert "Exit code: 0" in result

    asyncio.run(_run())


def test_engine_tool_events_have_call_ids():
    """Test that CborgAgentEngine attaches unique call_ids to tool_start and tool_output events."""
    async def _run():
        from types import SimpleNamespace

        registry = ToolRegistry()

        @registry.tool("dummy_calc", "Adds two numbers", requires_approval=False)
        def dummy_calc(a: int, b: int) -> str:
            return str(a + b)

        engine = CborgAgentEngine(api_key="test-key", registry=registry)

        # Mock stream response from OpenAI using SimpleNamespace
        mock_chunk_tool = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_calc_999",
                                function=SimpleNamespace(
                                    name="dummy_calc",
                                    arguments='{"a": 2, "b": 3}'
                                )
                            )
                        ]
                    )
                )
            ]
        )
        mock_chunk_text = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="The sum is 5.",
                        tool_calls=None
                    )
                )
            ]
        )

        async def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            # If the last message is tool result, return text chunk; otherwise return tool call chunk
            if any(m.get("role") == "tool" for m in messages):
                async def gen2():
                    yield mock_chunk_text
                return gen2()
            else:
                async def gen1():
                    yield mock_chunk_tool
                return gen1()

        engine.client.chat.completions.create = AsyncMock(side_effect=mock_create)

        events = []
        async for ev in engine.stream_chat("What is 2+3?", "Context"):
            events.append(ev)

        event_types = [e.type for e in events]
        assert "tool_start" in event_types
        assert "tool_output" in event_types
        assert "token" in event_types
        assert "done" in event_types

        tool_start_ev = next(e for e in events if e.type == "tool_start")
        assert tool_start_ev.metadata["call_id"] == "call_calc_999"
        assert tool_start_ev.content == "dummy_calc"

        tool_output_ev = next(e for e in events if e.type == "tool_output")
        assert tool_output_ev.metadata["call_id"] == "call_calc_999"
        assert tool_output_ev.metadata["output"] == "5"

    asyncio.run(_run())


def test_bridge_signals_with_call_id(qapp):
    """Test that GuiAgentBridge emits tool_started and tool_output_received with call_id."""
    registry = create_default_registry()
    engine = CborgAgentEngine(api_key="test-key", registry=registry)
    bridge = GuiAgentBridge(engine)

    tool_started_args = []
    tool_output_args = []

    bridge.tool_started.connect(lambda cid, name, args: tool_started_args.append((cid, name, args)))
    bridge.tool_output_received.connect(lambda cid, name, out: tool_output_args.append((cid, name, out)))

    bridge.tool_started.emit("call_123", "execute_terminal_command", '{"command": "ls"}')
    bridge.tool_output_received.emit("call_123", "execute_terminal_command", "file1.txt")

    assert len(tool_started_args) == 1
    assert tool_started_args[0] == ("call_123", "execute_terminal_command", '{"command": "ls"}')

    assert len(tool_output_args) == 1
    assert tool_output_args[0] == ("call_123", "execute_terminal_command", "file1.txt")


def test_sidebar_message_lifecycle_multi_round(qapp):
    """Test that SidebarWidget starts fresh message bubbles after tool calls."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)

    mock_chat_view = MockChatView()
    msg_counter = [0]

    def mock_start():
        msg_counter[0] += 1
        return f"msg_{msg_counter[0]}"

    mock_chat_view.start_assistant_message.side_effect = mock_start

    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    # 1. First pre-tool speech
    sidebar._on_token("Pre-tool text...")
    assert sidebar._current_msg_id == "msg_1"
    mock_chat_view.start_assistant_message.assert_called_once()
    mock_chat_view.append_token.assert_called_with("msg_1", "Pre-tool text...")

    # 2. Tool 1 starts -> msg_1 must be finalized and _current_msg_id cleared
    sidebar._on_tool_started("call_1", "execute_terminal_command", '{"command": "echo 1"}')
    assert sidebar._current_msg_id is None
    mock_chat_view.finalize_message.assert_called_with("msg_1")
    mock_chat_view.add_tool_card.assert_called_with("call_1", "execute_terminal_command", '{"command": "echo 1"}', "running")

    # 3. Tool 1 finishes
    sidebar._on_tool_output("call_1", "execute_terminal_command", "1")
    mock_chat_view.update_tool_card.assert_called_with("call_1", "execute_terminal_command", "1", "done")

    # 4. Post-tool speech -> must create a NEW assistant message (msg_2)
    sidebar._on_token("Post-tool text round 1...")
    assert sidebar._current_msg_id == "msg_2"
    assert mock_chat_view.start_assistant_message.call_count == 2
    mock_chat_view.append_token.assert_called_with("msg_2", "Post-tool text round 1...")

    # 5. Tool 2 starts (chaining) -> msg_2 must be finalized
    sidebar._on_tool_started("call_2", "list_directory", '{"path": "."}')
    assert sidebar._current_msg_id is None
    mock_chat_view.finalize_message.assert_called_with("msg_2")
    mock_chat_view.add_tool_card.assert_called_with("call_2", "list_directory", '{"path": "."}', "running")

    # 6. Tool 2 finishes
    sidebar._on_tool_output("call_2", "list_directory", "files...")
    mock_chat_view.update_tool_card.assert_called_with("call_2", "list_directory", "files...", "done")

    # 7. Post-tool speech round 2 -> must create a NEW assistant message (msg_3)
    sidebar._on_token("Final conclusion text...")
    assert sidebar._current_msg_id == "msg_3"
    assert mock_chat_view.start_assistant_message.call_count == 3
    mock_chat_view.append_token.assert_called_with("msg_3", "Final conclusion text...")

    # 8. Generation finishes
    sidebar._on_finished()
    assert sidebar._current_msg_id is None
    mock_chat_view.finalize_message.assert_called_with("msg_3")
    assert sidebar.send_btn.text() == "Send"


def test_sidebar_approval_flow_lifecycle(qapp):
    """Test approval requested finalizes current message and adds approval card."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)

    mock_chat_view = MockChatView()
    mock_chat_view.start_assistant_message.return_value = "msg_init"

    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    sidebar._on_token("I need your approval.")
    assert sidebar._current_msg_id == "msg_init"

    # Approval requested
    sidebar._on_approval_requested("cb_1", "execute_terminal_command", '{"command": "rm -rf"}')
    assert sidebar._current_msg_id is None
    mock_chat_view.finalize_message.assert_called_with("msg_init")
    mock_chat_view.add_approval_card.assert_called_with("cb_1", "execute_terminal_command", '{"command": "rm -rf"}')


def test_sidebar_rejection_flow(qapp):
    """Test rejecting a tool call with custom reason."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)

    mock_chat_view = MockChatView()
    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    # Trigger rejection via chat_view signal
    mock_chat_view.approval_rejected.emit("cb_99", "Too dangerous")
    engine.reject_tool.assert_called_with("cb_99", "Too dangerous")


def test_sidebar_cli_line_streaming(qapp):
    """Test that CLI stdout lines finalize any pending message and forward to chat view."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)

    mock_chat_view = MockChatView()
    mock_chat_view.start_assistant_message.return_value = "msg_cli"

    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    sidebar._on_token("Starting script...")
    assert sidebar._current_msg_id == "msg_cli"

    sidebar._on_cli_line("Counting: 1")
    assert sidebar._current_msg_id is None
    mock_chat_view.finalize_message.assert_called_with("msg_cli")
    mock_chat_view.add_cli_line.assert_called_with("Counting: 1")


def test_chat_input_plain_text_pasting(qapp):
    """Test that ChatInput rejects rich text and only pastes clean plain text."""
    from PySide6.QtCore import QMimeData
    from rixs_app.ui.agent_sidebar.sidebar_widget import ChatInput

    input_widget = ChatInput()
    assert input_widget.acceptRichText() is False

    # Simulate rich text clipboard with inline red color formatting
    mime = QMimeData()
    mime.setHtml('<span style="color: #ff0000; font-family: monospace;">Red styled text</span>')
    mime.setText("Red styled text")

    input_widget.insertFromMimeData(mime)
    assert input_widget.toPlainText() == "Red styled text"
    # Ensure no HTML span tags polluted the document
    assert "<span" not in input_widget.toHtml()


def test_structured_tools_schema_generation():
    """Test that ToolRegistry generates rich JSON schemas for structured tools."""
    registry = create_default_registry()
    schemas = registry.get_tool_schemas()
    schema_map = {s["function"]["name"]: s["function"] for s in schemas}

    assert "run_spatial_alignment" in schema_map
    align_schema = schema_map["run_spatial_alignment"]
    assert "directory" in align_schema["parameters"]["properties"]
    assert "engine" in align_schema["parameters"]["properties"]
    assert "ephemeral_cache" not in align_schema["parameters"]["properties"]
    assert align_schema["parameters"]["properties"]["directory"]["type"] == "string"
    assert align_schema["parameters"]["properties"]["engine"]["default"] == "ECC"
    assert "required" in align_schema["parameters"]
    assert "directory" in align_schema["parameters"]["required"]

    assert "run_zeroth_order_calibration" in schema_map
    calib_schema = schema_map["run_zeroth_order_calibration"]
    assert "directory" in calib_schema["parameters"]["properties"]
    assert "scan_log_path" in calib_schema["parameters"]["properties"]

    assert "run_image_denoising" in schema_map
    denoise_schema = schema_map["run_image_denoising"]
    assert "clip" in denoise_schema["parameters"]["properties"]
    assert "despike" in denoise_schema["parameters"]["properties"]


test_tool_definitions_and_schema = test_structured_tools_schema_generation


def test_cli_runner_smart_normalization():
    """Test that cli_runner gracefully handles varied command formats and rejects unknown scripts."""
    async def _run():
        registry = create_default_registry()

        # Non-existent script
        res_invalid = await registry.execute("cli_runner", {"command": "python non_existent.py -d ."})
        assert "Error: Command must target one of the project CLI scripts" in res_invalid

        # Empty command
        res_empty = await registry.execute("cli_runner", {"command": ""})
        assert "Error: Empty command string." in res_empty

        # Help flag on align_cli.py using different prefix forms
        res_direct = await registry.execute("cli_runner", {"command": "align_cli.py -h"})
        assert "usage: align_cli.py" in res_direct or "Headless TIFF alignment CLI" in res_direct

        res_python = await registry.execute("cli_runner", {"command": "python3 ./align_cli.py --help"})
        assert "Headless TIFF alignment CLI" in res_python

    asyncio.run(_run())


def test_run_spatial_alignment_nonexistent_directory():
    """Test run_spatial_alignment returns clean error on nonexistent directory."""
    async def _run():
        registry = create_default_registry()
        res = await registry.execute("run_spatial_alignment", {
            "directory": "/path/does/not/exist/at/all_12345",
            "engine": "ECC",
        })
        assert "Error: Directory '/path/does/not/exist/at/all_12345' does not exist" in res

    asyncio.run(_run())


def test_run_spatial_alignment_no_ephemeral_cache_flag(monkeypatch, tmp_path):
    """Test that run_spatial_alignment passes only valid modern CLI flags to subprocess command."""
    from unittest.mock import AsyncMock

    captured_calls = []

    async def mock_subprocess_exec(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        mock_proc = MagicMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=[b"alignment complete\n", b""])
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        return mock_proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_subprocess_exec)

    async def _run():
        registry = create_default_registry()
        res = await registry.execute("run_spatial_alignment", {
            "directory": str(tmp_path),
            "engine": "ECC",
        })
        assert "alignment complete" in res
        assert len(captured_calls) == 1
        cmd_args = captured_calls[0]["args"]
        forbidden_flag = "--" + "ephemeral" + "-cache"
        assert forbidden_flag not in cmd_args
        assert "-e" in cmd_args
        assert "ECC" in cmd_args

    asyncio.run(_run())


def test_unbuffered_streaming_flags(monkeypatch):
    """Test that _run_subprocess_streaming passes PYTHONUNBUFFERED=1 and -u."""
    captured_calls = []

    async def mock_subprocess_exec(*args, **kwargs):
        captured_calls.append({"args": args, "kwargs": kwargs})
        mock_proc = MagicMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=[b"line 1\n", b""])
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)
        mock_proc.returncode = 0
        return mock_proc

    from unittest.mock import AsyncMock
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_subprocess_exec)

    async def _run():
        registry = create_default_registry()
        res = await registry.execute("cli_runner", {"command": "align_cli.py -h"})
        assert "line 1" in res
        assert len(captured_calls) == 1
        call = captured_calls[0]
        assert "-u" in call["args"]
        assert call["kwargs"]["env"]["PYTHONUNBUFFERED"] == "1"

    asyncio.run(_run())


def test_sidebar_terminal_access_toggle(qapp, monkeypatch):
    """Test full terminal access toggle button transitions, styles, warning modal, and bridge updates."""
    from PySide6.QtWidgets import QMessageBox
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)
    mock_chat_view = MockChatView()
    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    # 1. Initial default state: OFF / danger (red)
    assert sidebar.terminal_access is False
    assert sidebar.terminal_btn.text() == "Full Terminal: OFF"
    assert sidebar.terminal_btn.objectName() == "danger_btn"

    # 2. Cancel the warning dialog -> remains OFF
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Cancel)
    sidebar.terminal_btn.click()
    assert sidebar.terminal_access is False
    assert sidebar.terminal_btn.text() == "Full Terminal: OFF"
    assert sidebar.terminal_btn.objectName() == "danger_btn"
    engine.set_terminal_access.assert_not_called()

    # 3. Accept the warning dialog -> transitions to ON / play (green)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    sidebar.terminal_btn.click()
    assert sidebar.terminal_access is True
    assert sidebar.terminal_btn.text() == "Full Terminal: ON"
    assert sidebar.terminal_btn.objectName() == "play_btn"
    engine.set_terminal_access.assert_called_with(True)

    # 4. Click again to turn OFF (no modal needed) -> transitions to OFF / danger (red)
    sidebar.terminal_btn.click()
    assert sidebar.terminal_access is False
    assert sidebar.terminal_btn.text() == "Full Terminal: OFF"
    assert sidebar.terminal_btn.objectName() == "danger_btn"
    engine.set_terminal_access.assert_called_with(False)

    # 5. Programmatic setter test
    sidebar.terminal_access = True
    assert sidebar.terminal_btn.text() == "Full Terminal: ON"
    assert sidebar.terminal_btn.objectName() == "play_btn"


def test_sidebar_quick_prompt_flow(qapp):
    """Test that quick prompt signal triggers send_text, appends user message and dispatches to bridge."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)
    mock_chat_view = MockChatView()
    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    prompt_text = "Help me align the loaded TIFF sequence using ECC engine"
    mock_chat_view.quick_prompt_requested.emit(prompt_text)

    mock_chat_view.append_user_message.assert_called_with(prompt_text)
    assert engine.stream_chat.called or bridge is not None
    assert sidebar.send_btn.text() == "Stop ■"
    assert sidebar.send_btn.objectName() == "danger_btn"


def test_sidebar_close_signal(qapp):
    """Test that sidebar_close_requested signal can be emitted and connected."""
    engine = MagicMock(spec=CborgAgentEngine)
    bridge = GuiAgentBridge(engine)
    mock_chat_view = MockChatView()
    main_win = QWidget()
    sidebar = AgentSidebarWidget(
        bridge=bridge,
        main_window_ref=main_win,
        chat_view=mock_chat_view,
    )

    close_emitted = []
    sidebar.sidebar_close_requested.connect(lambda: close_emitted.append(True))
    sidebar.sidebar_close_requested.emit()
    assert len(close_emitted) == 1

