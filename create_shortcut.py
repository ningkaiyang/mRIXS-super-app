#!/usr/bin/env python3
"""
create_shortcut.py — Automated Desktop Shortcut Creator for RIXS Super-App.

Uses `pyshortcuts` to create a cross-platform Desktop shortcut (Windows .lnk, 
macOS .app bundle, Linux .desktop) that launches `run.py` directly without 
requiring lab staff to type terminal commands or specify paths.

NOTE FOR MACOS VS WINDOWS:
  • macOS: Run with `--terminal` flag (`python3 create_shortcut.py --terminal`)
    to launch via Terminal.app, avoiding macOS Gatekeeper Automator stub crashes.
  • Windows: Standard execution (`python create_shortcut.py`) works out of the box
    and launches silently via pythonw.exe without opening a command prompt.

Usage:
    python3 create_shortcut.py --terminal            # Recommended on macOS
    python create_shortcut.py                         # Standard usage on Windows
    python3 create_shortcut.py --icon rixs_app/assets/icon.png --name "RIXS Super-App"
"""

import sys
import os
import argparse
import platform
import subprocess

def fix_mac_app_icon(app_path, icon_png_path):
    """
    Ensures macOS Finder correctly renders the custom icon for the .app bundle by:
    1. Fixing `CFBundleIconFile` in Info.plist to include `.icns` extension.
    2. Converting icon PNG to ICNS if needed.
    3. Updating timestamp/touching the .app bundle to flush macOS Finder icon cache.
    """
    if not os.path.exists(app_path):
        return

    info_plist = os.path.join(app_path, "Contents", "Info.plist")
    resources_dir = os.path.join(app_path, "Contents", "Resources")
    os.makedirs(resources_dir, exist_ok=True)

    icns_path = os.path.join(resources_dir, "app_icon.icns")

    # Convert PNG to ICNS using native macOS sips and iconutil if sips available
    if icon_png_path and os.path.exists(icon_png_path):
        try:
            iconset_dir = os.path.join(resources_dir, "app.iconset")
            os.makedirs(iconset_dir, exist_ok=True)

            # Generate icon sizes for mac
            sizes = [16, 32, 64, 128, 256, 512]
            for s in sizes:
                subprocess.run(
                    ["sips", "-z", str(s), str(s), icon_png_path, "--out", os.path.join(iconset_dir, f"icon_{s}x{s}.png")],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                subprocess.run(
                    ["sips", "-z", str(s*2), str(s*2), icon_png_path, "--out", os.path.join(iconset_dir, f"icon_{s}x{s}@2x.png")],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )

            # Compile into ICNS file using iconutil
            subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
            subprocess.run(["rm", "-rf", iconset_dir])
        except Exception:
            pass

    # Update Info.plist if present
    if os.path.exists(info_plist):
        try:
            with open(info_plist, "r") as f:
                content = f.read()

            # Fix missing extension in CFBundleIconFile
            import re
            content = re.sub(
                r"<key>CFBundleIconFile</key>\s*<string>[^<]*</string>",
                f"<key>CFBundleIconFile</key>\n    <string>{os.path.basename(icns_path if os.path.exists(icns_path) else 'RIXS_Super-App.icns')}</string>",
                content
            )

            with open(info_plist, "w") as f:
                f.write(content)
        except Exception:
            pass

    # Flush Finder icon cache by touching app bundle and running SetFile/touch
    try:
        subprocess.run(["touch", app_path])
        subprocess.run(["killall", "Finder"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def main():
    project_dir = os.path.abspath(os.path.dirname(__file__))
    default_script = os.path.join(project_dir, "run.py")
    # Check rixs_app/assets/icon.png first, fallback to root assets/icon.png
    primary_icon = os.path.join(project_dir, "rixs_app", "assets", "icon.png")
    fallback_icon = os.path.join(project_dir, "assets", "icon.png")
    default_icon = primary_icon if os.path.exists(primary_icon) else fallback_icon

    parser = argparse.ArgumentParser(
        description="Create Desktop shortcut for RIXS Super-App using pyshortcuts.",
        epilog="Note for macOS: Use --terminal flag to avoid macOS Gatekeeper Automator crashes. Windows does not need this flag."
    )
    parser.add_argument("--name", type=str, default="RIXS Super-App", help="Shortcut display name")
    parser.add_argument("--script", type=str, default=default_script, help="Path to run.py entry point")
    parser.add_argument("--icon", type=str, default=default_icon if os.path.exists(default_icon) else None, help="Path to custom icon (.png, .ico, .icns)")
    parser.add_argument("--terminal", action="store_true", help="Launch app inside a terminal window (Recommended on macOS to bypass Gatekeeper crashes)")
    args = parser.parse_args()

    script_path = os.path.abspath(args.script)
    if not os.path.exists(script_path):
        print(f"❌ Error: Target script not found at '{script_path}'")
        sys.exit(1)

    icon_path = os.path.abspath(args.icon) if args.icon and os.path.exists(args.icon) else None

    print("==================================================")
    print(f"🚀 Creating Desktop Shortcut for: {args.name}")
    print("==================================================")
    print(f"  • Script:       {script_path}")
    print(f"  • Working Dir:  {project_dir}")
    print(f"  • Python Exec:  {sys.executable}")
    print(f"  • Icon Path:    {icon_path or 'Default Python Icon'}")
    print(f"  • Terminal:     {'Yes' if args.terminal else 'No (GUI App)'}")
    print(f"  • Operating OS: {platform.system()}")

    try:
        from pyshortcuts import make_shortcut

        # Create shortcut using pyshortcuts
        make_shortcut(
            script=script_path,
            name=args.name,
            icon=icon_path,
            working_dir=project_dir,
            executable=sys.executable,
            terminal=args.terminal,
            desktop=True,
            startmenu=True
        )

        # On macOS, fix icon in Info.plist & flush Finder cache
        if platform.system() == "Darwin":
            app_path = os.path.expanduser(f"~/Desktop/{args.name.replace(' ', '_')}.app")
            fix_mac_app_icon(app_path, icon_path)

        print("\n✅ Success! Desktop shortcut successfully created!")
        print("   Lab staff can now double-click the shortcut on their Desktop to launch the RIXS Super-App.")
    
    except Exception as e:
        print(f"\n❌ Error creating shortcut: {e}")

if __name__ == "__main__":
    main()
