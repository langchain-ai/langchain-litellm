---
name: conventional-commits
description: Use before writing a commit message, PR title, PR description or branch name, or adding or removing a label in langchain-litellm. Covers the Conventional Commits title rules that pr-title-lint.yml checks, the PR description rules that keep release-please from dropping or duplicating a commit, branch names, dependency commits that must ship in a release, and which labels bots own.
---

# Conventional commits for langchain-litellm

`main` takes squash merges only. The squash commit's subject is the PR title and its body is the PR description; branch commits never reach `main`. release-please computes the version and changelog from those squash commits, and `label-pr.yml` labels the PR from its title. A malformed title means a wrong release, a missing changelog entry and a wrong label.

## Titles and commit messages

1. Every commit message and every PR title follows Conventional Commits 1.0.0: `<type>[(<scope>)][!]: <description>`.
   - The type is lowercase.
   - The scope is optional, names one subsystem such as `chat_models`, `router`, `deps` or `release`, and is never an empty `()`.
   - The description is imperative, starts with a lowercase verb, keeps identifiers and proper nouns in their own case, and has no trailing period.
   - CI checks the type, the header shape and the empty scope. The rest is on the author.
2. The type is one of the `types:` in `.github/workflows/pr-title-lint.yml`, which is the source of truth: check it. Today: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.
3. A breaking change puts `!` before the colon, and only a breaking change carries it. `label-pr.yml` detects only `!`, so a `BREAKING CHANGE:` footer alone gets no `breaking` label.

## PR descriptions

4. The PR description must parse as a commit body. It becomes the squash commit body, whoever merges can rewrap it, and release-please tries every body line as a footer.
   - No line starts with a word joined to `(`, such as `foo(` or `` `foo( ``, unless that `(` closes before any other `(` and before the line ends. Code blocks count, and indenting their lines does not help: the squash rewraps the description at about 72 columns and drops the indent.
   - So a call with a space or a nested paren inside, such as `foo(a, b)`, stays out of the description, code blocks included, or the PR gets a `BEGIN_COMMIT_OVERRIDE` block before the merge. A call like `foo(x)` is safe anywhere.
   - No paragraph starts with `<type>:` or `<type>(<scope>):`, which release-please reads as a second commit. Do not repeat the title as a line of the description.
   - Close an open issue with `Fixes #N`. Link an issue that is already closed with `Follow-up to #N`, never `Fixes` or `Refs`, because release-please prints every footer reference as "closes #N".
   - release-please drops a commit it cannot parse, and the release-please job still succeeds. The `release-notes` job then fails on the push to `main`. The fix is a `BEGIN_COMMIT_OVERRIDE` block in the PR description, added before the merge when the description is at risk, or afterwards followed by a re-run of Release.

## Dependencies and branches

5. A dependency change that users need released uses `fix(deps)`. `build` has no entry in `changelog-sections` in `release-please-config.json`, so it cuts no release.
6. Branch names are `<type>/<kebab-case-slug>` with the same type as the PR title, for example `ci/label-dependabot-security-prs`. Bot branches are exempt.
7. Never push to `main`. Every change goes through a PR, and the `main-protection` ruleset blocks direct pushes for everyone.

## Labels

8. Respect label ownership.
   - PR type labels `feat`, `fix`, `docs`, `refactor`, `chore`, `ci`, `test`, `perf`, `build` and `breaking`: `label-pr.yml` applies them from the title when a PR is opened or edited, and removes any the title no longer implies. Never set them by hand on PRs. `style` and `revert` have no label.
   - Bot-owned: `autorelease: pending` and `autorelease: tagged` (release-please); `dependencies`, `python`, `python:uv` and `github_actions` (Dependabot). Never add or remove them.
   - Issue labels: the issue templates apply `bug`, `feature request` or `task`. Everything else is manual triage by a maintainer: `duplicate`, `invalid`, `wontfix`, `question`, `good first issue`, `help wanted`, `upstream`, `security`, `regression`, and type labels such as `docs`, `chore` or `ci` on issues.
   - `bug` and `feature request` are for issues, `fix` and `feat` for PRs. Never swap them.

## Actions

9. Pin any new or updated GitHub Action to a full commit SHA with a comment naming its full release, such as `# v6.1.1`, as every existing `uses:` line is. A pre-commit hook rejects a shorter comment such as `# v6`.
