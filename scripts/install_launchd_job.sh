#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.alphaforge.quant.incremental"
HOUR="8"
MINUTE="10"
POOL_SIZE="100"
MODE="install"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hour) HOUR="$2"; shift 2 ;;
    --minute) MINUTE="$2"; shift 2 ;;
    --pool-size) POOL_SIZE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --uninstall) MODE="uninstall"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/$LABEL.plist"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$PLIST_DIR" "$LOG_DIR"

if [[ "$MODE" == "uninstall" ]]; then
  launchctl unload "$PLIST_PATH" 2>/dev/null || true
  rm -f "$PLIST_PATH"
  echo "uninstalled launchd job: $LABEL"
  exit 0
fi

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>cd '$ROOT_DIR' && export POOL_SIZE='$POOL_SIZE' && ./scripts/run_incremental_update.sh</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>$HOUR</integer>
    <key>Minute</key>
    <integer>$MINUTE</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>$LOG_DIR/incremental_update.out.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/incremental_update.err.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "installed launchd job: $LABEL"
echo "schedule: daily $(printf '%02d:%02d' "$HOUR" "$MINUTE")"
echo "pool-size: $POOL_SIZE"
echo "plist: $PLIST_PATH"
