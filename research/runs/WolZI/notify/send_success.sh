#!/usr/bin/env bash
set -e
SUMMARY="${1:-WINNER/SUMMARY.md}"
TO="viktormashalov@gmail.com"
SUBJECT="[QUANT WINNER] Strategy passed all gates — $(date -u +%Y-%m-%dT%H:%MZ)"

# Try msmtp first
if command -v msmtp >/dev/null 2>&1; then
  { printf "To: %s\nSubject: %s\nContent-Type: text/markdown\n\n" "$TO" "$SUBJECT"; cat "$SUMMARY"; } \
    | msmtp "$TO" && echo "Emailed via msmtp."
elif command -v mail >/dev/null 2>&1; then
  mail -s "$SUBJECT" "$TO" < "$SUMMARY" && echo "Emailed via mail."
fi

# macOS notification (best-effort)
osascript -e 'display notification "Strategy passed all gates. See WINNER/SUMMARY.md." with title "Quant Research" sound name "Glass"' 2>/dev/null || true

touch ./WINNER_FLAG
echo "WINNER_FLAG written. Done."
