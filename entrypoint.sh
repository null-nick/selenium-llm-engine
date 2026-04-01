#!/usr/bin/env bash
set -e

# Start X virtual framebuffer
Xvfb :1 -screen 0 1280x720x24 &
XVFB_PID=$!

# Start VNC server on display :1 (best effort for noVNC)
x11vnc -display :1 -nopw -forever -shared -noxdamage -noxfixes -noscr -bg

# Start websockify for noVNC (browser UI)
websockify --web=/usr/share/novnc/ 3000 localhost:5900 &
WEBSOCKIFY_PID=$!

# Start a window manager (optional)
fluxbox &
FLUXBOX_PID=$!

# Per login manuale, lancia Chromium SOLO con profilo SEPARATO, NON quello usato da Selenium.
# Esempio (decommenta solo se serve login manuale):
# chromium --no-sandbox --user-data-dir=/config/.config/chromium-manual --display=:1 --start-maximized &
# CHROME_PID=$!

# Start API (in foreground, so container stays up only if API is running)
cd /app
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

# cleanup (non verrà mai eseguito, ma lasciato per debug)
# kill $CHROME_PID $FLUXBOX_PID $WEBSOCKIFY_PID $XVFB_PID || true
