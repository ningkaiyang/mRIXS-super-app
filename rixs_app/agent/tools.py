"""Tool registry and implementations for the mRIXS Co-Pilot agent.

Provides a decorator-based tool registration system that auto-generates
OpenAI-compatible JSON schemas from Python function signatures, and
implements the V1 tool set for dataset inspection, CLI execution, and
GUI parameter control.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints


# Type mapping for JSON schema generation
_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


@dataclass
class ToolDef:
    """Internal representation of a registered tool."""

    name: str
    description: str
    func: Callable
    requires_approval: bool
    schema: dict


class ToolRegistry:
    """Registry for agent tools with automatic OpenAI schema generation.

    Usage::

        registry = ToolRegistry()

        @registry.tool("my_tool", "Does something", requires_approval=False)
        def my_tool(param: str) -> str:
            return "result"
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._gui_context: Any = None
        self._cli_line_callback: Callable[[str], None] | None = None

    def set_gui_context(self, main_window: Any) -> None:
        """Provide a reference to the main window for GUI tools."""
        self._gui_context = main_window

    def set_cli_line_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback invoked for each CLI output line (for live UI updates)."""
        self._cli_line_callback = callback

    def tool(
        self, name: str, description: str, *, requires_approval: bool = False
    ):
        """Decorator to register a function as an agent tool."""

        def decorator(func: Callable) -> Callable:
            schema = self._generate_schema(func, name, description)
            self._tools[name] = ToolDef(
                name=name,
                description=description,
                func=func,
                requires_approval=requires_approval,
                schema=schema,
            )
            return func

        return decorator

    def _generate_schema(
        self, func: Callable, name: str, description: str
    ) -> dict:
        """Generate OpenAI tool schema from function signature and type hints."""
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_type = hints.get(param_name, str)
            json_type = _PYTHON_TO_JSON_TYPE.get(param_type, "string")
            properties[param_name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def get_tool_schemas(self) -> list[dict]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.schema for t in self._tools.values()]

    def get_tool_def(self, name: str) -> ToolDef | None:
        """Look up a tool definition by name."""
        return self._tools.get(name)

    def requires_approval(self, name: str) -> bool:
        """Check if a tool requires user approval before execution."""
        tool = self._tools.get(name)
        return tool.requires_approval if tool else True

    async def execute(self, name: str, kwargs: dict[str, Any]) -> str:
        """Execute a registered tool and return its string result."""
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        try:
            func = tool.func
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: func(**kwargs))
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"


# ------------------------------------------------------------------
# V1 Tool implementations
# ------------------------------------------------------------------

def _find_project_root() -> Path:
    """Locate the project root directory."""
    from rixs_app.agent.auth import _find_project_root as _find_root

    return _find_root()


def _register_tools(registry: ToolRegistry) -> None:
    """Register all V1 tools on the given registry."""

    @registry.tool(
        "list_directory",
        "List files and subdirectories at the given path. Returns a formatted listing with sizes.",
        requires_approval=False,
    )
    def list_directory(path: str) -> str:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"Error: Path '{path}' does not exist."
        if not target.is_dir():
            return f"Error: '{path}' is not a directory."

        entries = sorted(
            target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
        )
        lines = [f"📁 {target}\n"]
        for entry in entries[:100]:
            if entry.is_dir():
                try:
                    n_children = sum(1 for _ in entry.iterdir())
                except PermissionError:
                    n_children = "?"
                lines.append(f"  📂 {entry.name}/ ({n_children} items)")
            else:
                size = entry.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  📄 {entry.name} ({size_str})")

        total = sum(1 for _ in target.iterdir())
        if total > 100:
            lines.append(f"  ... and {total - 100} more entries")

        return "\n".join(lines)

    @registry.tool(
        "check_alignment_readiness",
        "Check if a directory contains enough TIFF files for spatial drift alignment. Reports file count and readiness.",
        requires_approval=False,
    )
    def check_alignment_readiness(directory: str) -> str:
        from rixs_app.core.cli_utils import glob_tifs

        target = Path(directory).expanduser().resolve()
        if not target.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not target.is_dir():
            return f"Error: '{directory}' is not a directory."

        tif_files = glob_tifs(str(target))
        n = len(tif_files)

        if n == 0:
            return f"❌ Not ready: No .tif files found in {target}"
        elif n == 1:
            return (
                f"⚠️ Not ready: Only 1 .tif file found in {target}. "
                f"Need ≥2 for alignment.\n  📄 {Path(tif_files[0]).name}"
            )
        else:
            names = "\n".join(f"  📄 {Path(f).name}" for f in tif_files[:20])
            if n > 20:
                names += f"\n  ... and {n - 20} more"
            return f"✅ Ready: {n} .tif files found in {target}\n{names}"

    @registry.tool(
        "check_calibration_readiness",
        "Check if a directory is ready for zeroth-order calibration. Verifies TIFF files and scan log metadata.",
        requires_approval=False,
    )
    def check_calibration_readiness(directory: str) -> str:
        from rixs_app.core.cli_utils import glob_tifs

        target = Path(directory).expanduser().resolve()
        if not target.exists():
            return f"Error: Directory '{directory}' does not exist."

        tif_files = glob_tifs(str(target))
        n = len(tif_files)

        lines = []
        if n < 2:
            lines.append(f"❌ Not ready: {n} .tif file(s). Need ≥2 for calibration.")
        else:
            lines.append(f"✅ {n} .tif files found.")

        txt_files = list(target.glob("*.txt"))
        if txt_files:
            txt_path = txt_files[0]
            lines.append(f"📋 Scan log found: {txt_path.name}")
            try:
                from rixs_app.core.txt_metadata_parser import parse_scan_log

                metadata = parse_scan_log(str(txt_path))
                lines.append(f"  ✅ Parsed {len(metadata)} motor position entries.")
            except Exception as e:
                lines.append(f"  ⚠️ Could not parse scan log: {e}")
        else:
            lines.append(
                "⚠️ No .txt scan log found. Calibration will work, "
                "but motor positions won't be available."
            )

        ready = n >= 2
        status = "✅ Ready for calibration" if ready else "❌ Not ready for calibration"
        lines.insert(0, status)
        return "\n".join(lines)

    @registry.tool(
        "read_file_contents",
        "Read and return the contents of a text file (.txt, .csv, .json, .log, .py, .md). Truncates at max_lines.",
        requires_approval=False,
    )
    def read_file_contents(path: str, max_lines: int = 200) -> str:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"Error: File '{path}' does not exist."
        if not target.is_file():
            return f"Error: '{path}' is not a file."

        allowed = {".txt", ".csv", ".json", ".log", ".py", ".md", ".ini", ".cfg", ".yaml", ".yml", ".toml"}
        if target.suffix.lower() not in allowed:
            return f"Error: Cannot read '{target.suffix}' files. Supported: {', '.join(sorted(allowed))}"

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            truncated = len(lines) > max_lines
            content = "\n".join(lines[:max_lines])
            header = f"📄 {target.name} ({len(lines)} lines)"
            if truncated:
                header += f" — showing first {max_lines}"
            return f"{header}\n{'─' * 40}\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"

    @registry.tool(
        "get_cli_help",
        "Get the --help output for a project CLI tool. Always call this before constructing CLI commands.",
        requires_approval=False,
    )
    def get_cli_help(cli_script: str) -> str:
        allowed_scripts = {"align_cli.py", "zeroth_order_cli.py", "denoise_cli.py"}
        if cli_script not in allowed_scripts:
            return f"Error: Unknown script '{cli_script}'. Allowed: {', '.join(sorted(allowed_scripts))}"

        project_root = _find_project_root()
        script_path = project_root / cli_script
        if not script_path.exists():
            return f"Error: Script '{cli_script}' not found at {script_path}"

        try:
            result = subprocess.run(
                ["python3", str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=str(project_root),
            )
            return result.stdout or result.stderr or "No output"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out."
        except Exception as e:
            return f"Error: {e}"

    @registry.tool(
        "cli_runner",
        "Execute a project CLI command (align_cli.py, zeroth_order_cli.py, or denoise_cli.py). Streams stdout live and returns full output.",
        requires_approval=True,
    )
    async def cli_runner(command: str) -> str:
        allowed_prefixes = [
            "python align_cli.py", "python3 align_cli.py",
            "python zeroth_order_cli.py", "python3 zeroth_order_cli.py",
            "python denoise_cli.py", "python3 denoise_cli.py",
        ]
        if not any(command.strip().startswith(p) for p in allowed_prefixes):
            return f"Error: Command must start with a valid CLI prefix."

        project_root = _find_project_root()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
                cwd=str(project_root),
            )

            lines: list[str] = []
            cb = registry._cli_line_callback

            async def _read_with_timeout():
                while True:
                    line_bytes = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=300
                    )
                    if not line_bytes:
                        break
                    line = line_bytes.decode('utf-8', errors='replace').rstrip()
                    lines.append(line)
                    if cb:
                        cb(line)

            await _read_with_timeout()
            await proc.wait()

            output = "\n".join(lines)
            result = f"{output}\n\nExit code: {proc.returncode}" if output else f"Command completed with no output.\nExit code: {proc.returncode}"
            return result
        except asyncio.TimeoutError:
            return "Error: Command timed out (no output for 300 seconds)."
        except Exception as e:
            return f"Error: {e}"

    @registry.tool(
        "get_active_gui_state",
        "Query the current GUI state: active view, loaded directory, file count, frame index, and view-specific details.",
        requires_approval=False,
    )
    def get_active_gui_state() -> str:
        main_window = registry._gui_context
        if main_window is None:
            return json.dumps({"error": "GUI context not available"})

        try:
            state: dict[str, Any] = {"active_view": "unknown", "details": {}}
            stack = main_window._stack
            current_idx = stack.currentIndex()

            view_names = {
                0: "SortingView",
                1: "SlideshowView",
                2: "ExportComparisonView",
                3: "ZerothOrderSlideshowView",
            }
            state["active_view"] = view_names.get(current_idx, f"unknown_{current_idx}")

            if current_idx == 0:  # SortingView
                sv = main_window.sorting_view
                if hasattr(sv, "file_list"):
                    state["details"]["file_count"] = len(sv.file_list)
                    if sv.file_list:
                        state["details"]["first_file"] = sv.file_list[0]
                        state["details"]["directory"] = str(Path(sv.file_list[0]).parent)

            elif current_idx == 1:  # SlideshowView
                sv = main_window.slideshow_view
                if hasattr(sv, "_manager") and sv._manager:
                    mgr = sv._manager
                    state["details"]["file_count"] = getattr(mgr, "_n_frames", 0)
                    state["details"]["current_frame"] = getattr(mgr, "_current_idx", 0)
                    directory = getattr(mgr, "_directory", "")
                    if directory:
                        state["details"]["directory"] = str(directory)

            elif current_idx == 3:  # ZerothOrderSlideshowView
                zv = main_window.zeroth_order_view
                if hasattr(zv, "_manager") and zv._manager:
                    mgr = zv._manager
                    state["details"]["file_count"] = getattr(mgr, "_n_frames", 0)
                    state["details"]["current_frame"] = getattr(mgr, "_current_idx", 0)
                    directory = getattr(mgr, "_directory", "")
                    if directory:
                        state["details"]["directory"] = str(directory)
                    state["details"]["has_scan_log"] = getattr(mgr, "_txt_path", None) is not None

            return json.dumps(state, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @registry.tool(
        "update_gui_parameter",
        "Update a GUI parameter such as frame_index.",
        requires_approval=True,
    )
    def update_gui_parameter(parameter: str, value: str) -> str:
        main_window = registry._gui_context
        if main_window is None:
            return "Error: GUI context not available."

        try:
            param_lower = parameter.lower()

            if param_lower == "frame_index":
                idx = int(value)
                current = main_window._stack.currentIndex()
                if current == 1 and hasattr(main_window.slideshow_view, "next_frame"):
                    # Navigate to specific frame by going to first and then stepping
                    return f"Frame navigation to index {idx} requested (manual stepping required)."
                elif current == 3 and hasattr(main_window.zeroth_order_view, "next_frame"):
                    return f"Frame navigation to index {idx} requested (manual stepping required)."
                else:
                    return "Error: No active slideshow view."
            else:
                return f"Error: Unknown parameter '{parameter}'. Supported: frame_index."
        except Exception as e:
            return f"Error updating parameter: {e}"

    @registry.tool(
        "execute_terminal_command",
        "Execute an arbitrary shell command. Requires 'Full Terminal Access' to be enabled and always requires approval.",
        requires_approval=True,
    )
    async def execute_terminal_command(command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_find_project_root()),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            parts = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                parts.append(f"STDERR:\n{stderr.decode('utf-8', errors='replace')}")
            parts.append(f"\nExit code: {proc.returncode}")
            return "\n".join(parts) or "Command completed with no output."
        except asyncio.TimeoutError:
            return "Error: Command timed out after 120 seconds."
        except Exception as e:
            return f"Error: {e}"


def create_default_registry() -> ToolRegistry:
    """Create and return a ToolRegistry with all V1 tools registered."""
    registry = ToolRegistry()
    _register_tools(registry)
    return registry
