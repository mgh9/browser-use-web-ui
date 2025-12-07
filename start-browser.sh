#!/usr/bin/env sh
set -e

: "${DISPLAY:=:99}"

start_browser() {
  CMD=""
  if command -v chromium >/dev/null 2>&1; then
    CMD="chromium"
  elif command -v google-chrome >/dev/null 2>&1; then
    CMD="google-chrome"
  elif command -v google-chrome-stable >/dev/null 2>&1; then
    CMD="google-chrome-stable"
  else
    # Try Playwright-installed Chromium (PLAYWRIGHT_BROWSERS_PATH default: /ms-browsers)
    PW_CHROME="$(find /ms-browsers -type f -name chrome -path '*chromium*' 2>/dev/null | head -n 1)"
    if [ -n "$PW_CHROME" ] && [ -x "$PW_CHROME" ]; then
      CMD="$PW_CHROME"
    fi
  fi

  if [ -n "$CMD" ]; then
    echo "Starting browser ($CMD) on DISPLAY=${DISPLAY}..."
    "$CMD" --no-sandbox --start-maximized about:blank >/tmp/chrome.log 2>&1 &
  else
    echo "No Chromium/Chrome binary found; skipping visible browser startup." >&2
  fi
}

start_browser

echo "Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
