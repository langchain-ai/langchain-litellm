#!/bin/sh
# PostToolUse hook: fix and format a Python file Claude just edited with the
# locked ruff, and hand any unfixable lint errors back to Claude.
file=$(python3 -c 'import json, sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')
case "$file" in
  *.py | *.pyi) ;;
  *) exit 0 ;;
esac
cd "$CLAUDE_PROJECT_DIR" || exit 0

# An import or variable added an edit before its first use is not dead code yet;
# the commit hook and CI still enforce F401 and F841 once the change is done.
out=$(uv run --frozen ruff check --fix --extend-ignore F401,F841 --force-exclude --output-format concise "$file" 2>&1)
status=$?
uv run --frozen ruff format --quiet --force-exclude "$file" >/dev/null 2>&1
if [ "$status" -ne 0 ]; then
  printf '%s\n' "$out" >&2
  exit 2
fi
exit 0
