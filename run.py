#!/usr/bin/env python3
import signal
import sys
from rixs_app.main import RixsApp

if __name__ == "__main__":
    app = RixsApp()

    def handle_sigint(sig, frame):
        try:
            app.on_close()
        except Exception:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    app.mainloop()
