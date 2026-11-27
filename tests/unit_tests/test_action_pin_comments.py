"""Test the pre-commit hook that checks the comment on a pinned action."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
PIN = "      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"

pytestmark = pytest.mark.skipif(
    not CONFIG.exists(), reason="needs the repository checkout"
)


def _hook() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    return next(hook for hook in hooks if hook["id"] == "action-pin-comments")


def _reported(line: str) -> bool:
    """Whether the hook reports the line; pygrep searches each line for its entry."""
    return re.search(_hook()["entry"], line) is not None


@pytest.mark.parametrize("comment", ["# v7.0.1", "# 7.0.1", "#v7.0.1", "# v10.3.3  "])
def test_a_pin_commented_with_its_full_release_passes(comment: str) -> None:
    assert not _reported(f"{PIN} {comment}")


@pytest.mark.parametrize(
    "tail",
    ["", " # v7", " # v7.0", " # main", " # v7.0.1 until the next release"],
)
def test_a_pin_without_its_full_release_is_reported(tail: str) -> None:
    assert _reported(f"{PIN}{tail}")


@pytest.mark.parametrize(
    "line",
    ["      - uses: ./.github/actions/setup", "      - uses: actions/checkout@v7"],
)
def test_a_reference_that_is_not_a_sha_pin_is_not_this_hooks_concern(line: str) -> None:
    assert not _reported(line)


def test_the_hook_reads_every_workflow_and_each_one_passes() -> None:
    hook = _hook()
    workflows = sorted(WORKFLOWS.glob("*.y*ml"))
    assert workflows
    unread = [
        path.name
        for path in workflows
        if not re.search(hook["files"], path.relative_to(REPO_ROOT).as_posix())
    ]
    assert unread == []
    reported = [
        f"{path.name}:{number}"
        for path in workflows
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(hook["entry"], line)
    ]
    assert reported == []
