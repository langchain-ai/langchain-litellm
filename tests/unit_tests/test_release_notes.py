"""Test the reconciliation of releasable commits against the pending changelog."""

# stdlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# third-party
import pytest

# first-party
from release_notes import Commit, missing_entries, pending_section

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_release_notes.py"
COMMIT_URL = "https://github.com/langchain-ai/langchain-litellm/commit"

# Throwaway repos: none of the developer's git config, and no inherited GIT_DIR
# or GIT_INDEX_FILE, from a hook say, to aim these commands at the real repo.
GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _commit(subject: str) -> Commit:
    """A commit whose sha is stable and distinct, derived from its subject."""
    return Commit(hashlib.sha1(subject.encode()).hexdigest(), subject)


def _entry(commit: Commit) -> str:
    """One changelog bullet, linked the way release-please links it."""
    return f"* {commit.subject} ([{commit.sha[:7]}]({COMMIT_URL}/{commit.sha}))\n"


def _changelog(*releases: tuple[str, list[Commit]]) -> str:
    """A CHANGELOG.md with one section per (version, commits), newest first."""
    text = "# Changelog\n"
    for version, commits in releases:
        text += f"\n## [{version}](https://example.com) (2026-09-23)\n\n"
        text += "### Bug Fixes\n\n" + "".join(_entry(commit) for commit in commits)
    return text


# ── which commits the pending section must list ──────────────────────────────


def test_a_section_listing_every_releasable_commit_passes() -> None:
    commits = [
        _commit("feat(chat_models): surface the response cost (#281)"),
        _commit("fix(chat_models): name a streamed cost once (#284)"),
        _commit("perf: reuse the client"),
        _commit("revert: reuse the client"),
    ]
    changelog = _changelog(("0.8.0", commits))

    assert missing_entries(commits, pending_section(changelog, "0.8.0")) == []


def test_an_unlisted_releasable_commit_is_returned_with_its_subject() -> None:
    """How a fix went missing from 0.8.0: release-please could not parse its body."""
    listed = _commit("fix: accept base_url for LiteLLM embeddings (#203)")
    dropped = _commit(
        "fix: accept base_url as an alias for api_base in ChatLiteLLM (#200)"
    )
    changelog = _changelog(("0.8.0", [listed]))

    missing = missing_entries([listed, dropped], pending_section(changelog, "0.8.0"))

    assert missing == [dropped]


@pytest.mark.parametrize(
    "subject",
    [
        "docs(readme): point activity badges at the canonical repo (#274)",
        "chore(main): release langchain-litellm 0.8.0 (#270)",
        "ci: pin the release action",
        "test(chat_models): pin that top_p and top_k default to None (#240)",
        "build(deps): bump anyio from 4.12.1 to 4.14.2 (#273)",
        "style: reformat",
        "refactor(router): split the retry loop",
    ],
)
def test_a_hidden_type_is_never_required(subject: str) -> None:
    """release-please never renders these, so no section would ever list them."""
    assert missing_entries([_commit(subject)], "") == []


@pytest.mark.parametrize(
    "subject",
    [
        "feat!: make .stream() stream (#272)",
        "fix(chat_models)!: stop republishing litellm's router bookkeeping (#283)",
        "refactor(router)!: split the retry loop",
        "build(deps)!: require litellm 2",
        "chore!: drop python 3.9",
    ],
)
def test_a_breaking_change_is_required(subject: str) -> None:
    """release-please renders a breaking change and bumps for it, hidden type or not."""
    commit = _commit(subject)

    assert missing_entries([commit], "") == [commit]


@pytest.mark.parametrize("subject", ["Fix: a", "FEAT(router): b", "Perf: c"])
def test_the_type_is_matched_case_insensitively(subject: str) -> None:
    """release-please renders a `Fix:` commit under Bug Fixes like any other."""
    commit = _commit(subject)

    assert missing_entries([commit], "") == [commit]


@pytest.mark.parametrize(
    "subject",
    [
        "fix(): a",
        "fix : a",
        "feat (router): b",
        "feat!(router): c",
        'Revert "fix(chat_models): name a streamed cost once" (#290)',
    ],
)
def test_a_header_release_please_rejects_is_still_required(subject: str) -> None:
    """release-please's grammar rejects these and drops them: losses to catch."""
    commit = _commit(subject)

    assert missing_entries([commit], "") == [commit]


@pytest.mark.parametrize(
    "subject",
    [
        "Update README.md",
        "Merge pull request #1 from fork/branch",
        "fix the flaky test",
        "fixup! fix: a",
        "squash! feat: b",
        "Fixes: a",
        "feature: b",
        'Revert "docs: c"',
    ],
)
def test_a_non_conventional_subject_is_not_required(subject: str) -> None:
    assert missing_entries([_commit(subject)], "") == []


# ── what counts as listed ────────────────────────────────────────────────────


def test_the_short_sha_alone_counts() -> None:
    """A hand-written entry may cite the short sha without release-please's link."""
    commit = _commit("fix: a")

    assert missing_entries([commit], f"* a ({commit.sha[:7]})\n") == []


def test_a_sha_inside_a_longer_hex_run_does_not_count() -> None:
    """Another commit's full sha can contain these seven characters mid-string."""
    commit = _commit("fix: a")
    other = "0" * 20 + commit.sha[:7] + "0" * 13
    section = f"* b ([{other[:7]}]({COMMIT_URL}/{other}))\n"

    assert missing_entries([commit], section) == [commit]


# ── isolating the pending section ────────────────────────────────────────────


def test_an_entry_in_an_older_section_does_not_count() -> None:
    commit = _commit("fix: a")
    changelog = _changelog(("0.8.1", []), ("0.8.0", [commit]))

    assert missing_entries([commit], pending_section(changelog, "0.8.1")) == [commit]


def test_a_longer_version_sharing_the_prefix_is_not_the_pending_section() -> None:
    commit = _commit("fix: a")
    changelog = _changelog(("0.8.10", [commit]), ("0.8.1", []))

    assert missing_entries([commit], pending_section(changelog, "0.8.1")) == [commit]


def test_a_changelog_without_the_pending_section_is_an_error() -> None:
    """Returning nothing would report every commit as missing, hiding the cause."""
    with pytest.raises(ValueError, match=r"0\.8\.1"):
        pending_section(_changelog(("0.8.0", [])), "0.8.1")


# ── the entry point, against a real repository ───────────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        cwd=repo,
        env=GIT_ENV,
        text=True,
        check=True,
    ).stdout.strip()


def _record(repo: Path, message: str, files: dict[str, str] | None = None) -> str:
    """Commit `files` (possibly none) with `message` and return the new sha."""
    for name, text in (files or {}).items():
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "--allow-empty", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _main_since_release(tmp_path: Path, *subjects: str) -> tuple[Path, list[str]]:
    """A repo whose main has `subjects` committed since the 0.1.0 release tag.

    The `fix:` released in 0.1.0 checks that the range starts at the tag.
    Returns the repo and the full shas of `subjects`, in order.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=main")
    _record(repo, "fix: released in 0.1.0 (#1)")
    _record(
        repo,
        "chore(main): release langchain-litellm 0.1.0",
        {".release-please-manifest.json": json.dumps({".": "0.1.0"})},
    )
    _git(repo, "tag", "langchain-litellm==0.1.0")
    return repo, [_record(repo, subject) for subject in subjects]


def _release_branch(repo: Path, version: str, listed: list[str]) -> None:
    """Branch `release` off main, with notes for `version` citing `listed`."""
    _git(repo, "checkout", "--quiet", "-b", "release")
    _record(
        repo,
        f"chore(main): release langchain-litellm {version}",
        {
            ".release-please-manifest.json": json.dumps({".": version}),
            "CHANGELOG.md": _changelog(
                (version, [Commit(sha, "fix") for sha in listed])
            ),
        },
    )
    _git(repo, "checkout", "--quiet", "main")


def _check(repo: Path, main: str = "main") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--main", main, "--release", "release"],
        capture_output=True,
        cwd=repo,
        env=GIT_ENV,
        text=True,
        check=False,
    )


def test_the_script_names_each_missing_commit_and_fails(tmp_path: Path) -> None:
    repo, (listed, dropped, _) = _main_since_release(
        tmp_path, "fix: listed (#2)", "fix: dropped (#3)", "docs: not releasable (#4)"
    )
    _release_branch(repo, "0.1.1", [listed])

    result = _check(repo)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"missing: {dropped[:7]} fix: dropped (#3)" in result.stdout
    assert listed[:7] not in result.stdout
    assert "released in 0.1.0" not in result.stdout
    assert "BEGIN_COMMIT_OVERRIDE" in result.stdout


def test_the_script_passes_when_every_commit_is_listed(tmp_path: Path) -> None:
    repo, shas = _main_since_release(tmp_path, "fix: listed (#2)", "feat: too (#3)")
    _release_branch(repo, "0.2.0", shas)

    result = _check(repo)

    assert result.returncode == 0, result.stdout + result.stderr


def test_without_a_release_branch_every_releasable_commit_is_missing(
    tmp_path: Path,
) -> None:
    """release-please opens no release PR when it parsed nothing releasable."""
    repo, (dropped,) = _main_since_release(tmp_path, "fix: dropped (#2)")

    result = _check(repo)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "No release notes pending" in result.stdout
    assert f"missing: {dropped[:7]} fix: dropped (#2)" in result.stdout


def test_a_release_branch_left_at_the_released_version_lists_nothing(
    tmp_path: Path,
) -> None:
    """A merged release PR's branch can outlive it, holding only released notes."""
    repo, (dropped,) = _main_since_release(tmp_path, "fix: dropped (#2)")
    _release_branch(repo, "0.1.0", [_git(repo, "rev-parse", "main~2")])

    result = _check(repo)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "No release notes pending" in result.stdout
    assert f"missing: {dropped[:7]} fix: dropped (#2)" in result.stdout


def test_nothing_releasable_since_the_tag_passes(tmp_path: Path) -> None:
    repo, _ = _main_since_release(tmp_path, "docs: not releasable (#2)")

    result = _check(repo)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_release_cut_after_the_checked_push_leaves_nothing_missing(
    tmp_path: Path,
) -> None:
    """A release PR merged before this push's check runs has shipped its commits."""
    repo, (shipped,) = _main_since_release(tmp_path, "fix: shipped (#2)")
    _record(
        repo,
        "chore(main): release langchain-litellm 0.1.1",
        {".release-please-manifest.json": json.dumps({".": "0.1.1"})},
    )
    _git(repo, "tag", "langchain-litellm==0.1.1")

    result = _check(repo, main=shipped)

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_merged_release_that_was_never_tagged_is_named(tmp_path: Path) -> None:
    repo, _ = _main_since_release(tmp_path)
    _record(
        repo,
        "chore(main): release langchain-litellm 0.1.1",
        {".release-please-manifest.json": json.dumps({".": "0.1.1"})},
    )

    result = _check(repo)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "langchain-litellm==0.1.1 does not exist" in result.stdout
