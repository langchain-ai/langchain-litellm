# Contributing to langchain-litellm

The LangChain [contributing guide](https://docs.langchain.com/oss/python/contributing/overview) and the [org guidelines](https://github.com/langchain-ai/.github/blob/main/CONTRIBUTING.md) apply here: start from an issue a maintainer has approved, fill in the PR template, and say when you used AI. This page covers what is specific to this repo.

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.22 or later, git and make. Then run:

```bash
make setup
```

It installs every dependency group into `.venv` on the Python `.python-version` names (uv downloads it if needed), installs the pre-commit and pre-push hooks, and makes `git blame` skip the bulk reformat commits.

To skip local setup, open the repo in its devcontainer, or [in a Codespace](https://codespaces.new/langchain-ai/langchain-litellm). The container keeps its own `.venv`. If you commit from inside it, run `make setup` there instead of on your machine.

Every command lives in one list, the [Commands section of AGENTS.md](AGENTS.md#commands).

## Editors

Every editor below reads `.editorconfig`, and each committed config runs the ruff version pinned in `uv.lock`, so your editor, the git hooks and CI agree.

- **VS Code, Cursor, Windsurf:** accept the recommended extensions. `.vscode/settings.json` uses `.venv`, formats with ruff on save and runs the unit tests from the Testing panel.
- **Zed:** `.zed/settings.json`. Zed finds `.venv` and its ruff on its own.
- **Helix:** `.helix/languages.toml`.
- **Neovim 0.11+:** `.nvim.lua`. Set `vim.o.exrc = true` and trust the file when Neovim asks.
- **Emacs 29+:** `.dir-locals.el`. Accept the prompt to apply its `eval` form.
- **JetBrains IDEs:** nothing is committed, because `.idea/` mixes shared and personal state. Set the interpreter to `.venv/bin/python` and point the IDE's Ruff integration at `.venv/bin/ruff`.

## Tests

- `make test` runs the unit tests offline. CI runs them on every Python version the classifiers in `pyproject.toml` list, and `make test_matrix` does the same locally.
- Integration tests call real providers. Run `cp .env.example .env`, fill it in, then `make integration_tests`. `.env` is gitignored.

## Pull requests

- Branch name: `<type>/<kebab-slug>`, for example `fix/router-retries`.
- Title: [Conventional Commits](https://www.conventionalcommits.org/), checked on every PR. Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore and revert. The scope is optional, and `!` marks a breaking change, as in `feat!: ...`.
- Merges are squash-only, so the PR title and body become the commit on `main`. Keep the body accurate.
- Every review thread, Copilot's included, must be resolved before the PR can merge.

## Automation

- `label-pr.yml` sets the PR's type label from its title, adds `breaking` for `!`, and removes type labels it did not set.
- Copilot reviews every push.
- CodeQL blocks a merge on a security alert of high severity or above, or an error-level alert.
- CI checks the lockfile, runs the pre-commit hooks, lint and type checks, spell check, and unit tests on every supported Python version, once against `uv.lock` and once against the lowest versions `pyproject.toml` allows. A PR that changes `pyproject.toml` also runs the unit tests against the newest litellm release.
- A pre-commit hook requires the comment on every pinned action to name its full release, such as `# v6.1.1`. Dependabot then updates the comment with the SHA.
- Dependabot opens weekly grouped `build(deps)` PRs for uv, GitHub Actions, the pre-commit hooks and the devcontainer images, a week after each release. The uv PR also bumps the pytest plugins that only `uv.lock` names; no other lock-only package moves without a security update or a relock. Security updates skip that wait and come in their own PR.

## Releases

release-please keeps a release PR open and updates it on every push to `main`. Only feat, fix, perf, revert and breaking changes cut a release. Below 1.0, feat and breaking changes bump the minor version and the rest bump the patch. Merging the release PR tags the release. The Release workflow then publishes to PyPI through trusted publishing once a maintainer approves the `pypi` environment.

## Claude Code

`.claude/settings.json` is shared by everyone. It allows the make targets and read-only `gh` commands, asks before merging a PR, and denies pushes to `main`, force pushes, publishing and reading `.env`. Personal rules go in `.claude/settings.local.json`, which is gitignored.

Those rules match command text, so a command worded differently can get past them. To block `.env` reads at the OS level too, turn on Claude Code's sandbox in your `.claude/settings.local.json`:

```json
{ "sandbox": { "enabled": true } }
```
