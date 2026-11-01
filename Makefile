.PHONY: all help setup check format format_diff lint lint_diff lint_package lint_tests \
	test tests test_matrix test_watch integration_test integration_tests \
	spell_check spell_fix check_imports

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/

# One-time setup: every dependency group, the pre-commit and pre-push hooks,
# and a local blame that skips the commits in .git-blame-ignore-revs.
setup:
	uv sync
	uv run pre-commit install
	git config blame.ignoreRevsFile .git-blame-ignore-revs

# Everything CI runs on a PR, in CI's order, except the dependency-floor and
# newest-litellm jobs, which install other versions over the environment.
check:
	uv lock --check
	uv run pre-commit run --all-files
	$(MAKE) lint
	$(MAKE) spell_check
	$(MAKE) test

# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	uv run --group test pytest --disable-socket --allow-unix-socket $(TEST_FILE)

# unit tests on every Python CI tests, each in a fresh environment
PYTHON_VERSIONS ?= 3.11 3.12 3.13 3.14
test_matrix:
	for v in $(PYTHON_VERSIONS); do \
		echo "== Python $$v"; \
		uv run --isolated --python $$v --group test pytest --disable-socket --allow-unix-socket -q $(TEST_FILE) || exit 1; \
	done

test_watch:
	uv run --group test ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	uv run --group test pytest $(TEST_FILE)

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_litellm
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || { mkdir -p $(MYPY_CACHE) && uv run --group typing mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE); }

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff check --fix --exit-zero $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || uv run --group lint ruff format $(PYTHON_FILES)

spell_check:
	uv run --group codespell codespell --toml pyproject.toml

spell_fix:
	uv run --group codespell codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_litellm -name '*.py')
	uv run python ./scripts/check_imports.py $^

######################
# HELP
######################

help:
	@echo '----'
	@echo 'setup                        - install dependencies and git hooks (run once)'
	@echo 'check                        - run what CI runs, except the floor and newest-litellm tests'
	@echo 'format                       - fix lint and format code in place'
	@echo 'format_diff                  - format only Python files changed from main'
	@echo 'lint                         - ruff check, ruff format --diff and mypy'
	@echo 'lint_diff                    - lint only Python files changed from main'
	@echo 'lint_package                 - lint langchain_litellm/ only'
	@echo 'lint_tests                   - lint tests/ only'
	@echo 'test                         - run unit tests (network disabled)'
	@echo 'test TEST_FILE=<test_file>   - run the unit tests in one file'
	@echo 'test_matrix                  - run unit tests on Python 3.11 to 3.14'
	@echo 'test_watch                   - rerun unit tests on every change'
	@echo 'integration_tests            - run integration tests (needs .env)'
	@echo 'spell_check                  - check spelling with codespell'
	@echo 'spell_fix                    - fix spelling in place'
	@echo 'check_imports                - check each module imports on its own'
