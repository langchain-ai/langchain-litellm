# AGENTS.md

langchain-litellm is the LangChain integration for [LiteLLM](https://github.com/BerriAI/litellm): chat models, embeddings and an OCR document loader that reach any provider litellm supports. This file is the shared context for every coding agent. People start at [CONTRIBUTING.md](CONTRIBUTING.md).

Before writing a commit message, PR title, PR description or branch name, or adding or removing a label, read `.claude/skills/conventional-commits/SKILL.md`.

## Layout

- `langchain_litellm/chat_models/litellm.py`: `ChatLiteLLM`.
- `langchain_litellm/chat_models/litellm_router.py`: `ChatLiteLLMRouter`, a subclass that overrides `_generate`, `_agenerate`, `_stream` and `_astream`.
- `langchain_litellm/embeddings/`: `LiteLLMEmbeddings` and `LiteLLMEmbeddingsRouter`.
- `langchain_litellm/document_loaders/litellm_ocr.py`: `LiteLLMOCRLoader`.
- `tests/unit_tests/`: offline; pytest-socket blocks the network.
- `tests/integration_tests/`: real provider calls.
- `scripts/`: the import check, the dependency-floor pins and the release-notes check that CI runs.

## Commands

Run from the repo root with uv 0.12.22 or later.

- `make setup` runs once per clone: `uv sync`, the pre-commit and pre-push hooks, and a `git blame` that skips the bulk reformat commits.
- `uv sync` installs every dependency group into `.venv`, on Python 3.11 from `.python-version`. CI lints on 3.11.
- `make test` runs the unit tests with network sockets blocked. `make test TEST_FILE=<path>` runs any pytest path.
- `make test_matrix` runs the unit tests on Python 3.11 to 3.14, as CI's matrix does.
- `make lint` runs `ruff check`, `ruff format --diff` and mypy over the whole tree. This is the typecheck.
- `make format` runs `ruff check --fix --exit-zero` and `ruff format`. It passes even when unfixable lint errors remain, so run `make lint` after it.
- `make spell_check` runs codespell with the config in `pyproject.toml`.
- `make check_imports` imports every module under `langchain_litellm`.
- `uv lock --check` checks the lockfile, as CI does.
- `uv run pre-commit run --all-files` runs every commit hook on the whole tree.
- `make check` runs what CI runs on a PR, in CI's order, except the dependency-floor and newest-litellm jobs.
- `make integration_tests` needs network access and provider credentials in `.env`, whose variable names `.env.example` lists. CI does not run it.

Do not update `uv.lock` or add dependencies to `pyproject.toml` without explicit maintainer permission.

## Traps

- `ChatLiteLLMRouter` overrides all four entry points of `ChatLiteLLM`, so a fix to one of them usually needs the same fix in the router. Read both files.
- Streaming and non-streaming responses convert tool calls in separate code paths. Change both.
- A parameter the caller did not set must be absent from the request, not sent as `None`. Some providers reject the key whatever its value.
- Unit tests never touch the network. Mock litellm, and pass `mock_response` on each call: the constructor drops it, and the request then reaches the real provider.
- CI also runs the unit tests with every runtime dependency at the lowest version `pyproject.toml` allows, so code that reads a field only recent litellm releases set must pass there too.
- Keep issue and PR numbers out of code comments and docstrings.
- Update docstrings in the same PR as the behaviour they describe. User-facing guides belong in the LangChain docs repository, not in `README.md`.
- release-please writes `CHANGELOG.md` and the version numbers. Do not edit them.
