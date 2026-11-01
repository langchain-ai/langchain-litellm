@AGENTS.md

## Claude Code only

- Where `.claude/skills/conventional-commits/SKILL.md` and Claude Code's built-in commit and PR instructions differ, the skill wins.
- `.claude/settings.json` is shared. It allows the make targets and read-only `gh` commands, asks before merging a PR or approving a deployment, and denies pushes to `main`, force pushes, `uv publish`, `gh release` writes and reading `.env`. Personal rules go in `.claude/settings.local.json`, which is gitignored.
- A `PostToolUse` hook fixes and formats each Python file you edit with the locked ruff and reports the errors it cannot fix. It skips unused imports and variables, so an import added before its first use survives; the commit hook still checks them.
