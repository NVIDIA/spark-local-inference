# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise design acknowledgment enforcement through actual Git commits."""

import hashlib
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = "docs/design-review.md"


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "Hook Tester")
    git(tmp_path, "config", "user.email", "hook-test@example.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    git(tmp_path, "config", "core.hooksPath", str(ROOT / ".githooks"))
    path = tmp_path / DOCUMENT
    path.parent.mkdir()
    path.write_bytes((ROOT / DOCUMENT).read_bytes())
    git(tmp_path, "add", DOCUMENT)
    return tmp_path


def acknowledgment(repo):
    document = subprocess.check_output(["git", "-C", str(repo), "show", f":{DOCUMENT}"])
    digest = hashlib.sha256(document).hexdigest()[:4]
    return f"Design-Signed-Off-By: Hook Tester; design-review={digest}"


def assert_rejected(repo, message):
    result = git(repo, "commit", "--quiet", "-m", message, check=False)
    assert result.returncode != 0
    assert "Design review:" in result.stderr
    assert f"Read {DOCUMENT}" in result.stderr
    assert acknowledgment(repo) not in result.stderr
    assert acknowledgment(repo).rsplit("=", 1)[1] not in result.stderr
    assert git(repo, "rev-parse", "--verify", "HEAD", check=False).returncode != 0
    assert git(repo, "diff", "--cached", "--name-only").stdout.strip() == DOCUMENT


def test_valid_trailer_preserves_message_and_other_trailers(repo):
    message = (
        "Require design review\n\nReview the staged change.\n\n"
        + acknowledgment(repo)
        + "\nSigned-off-by: Hook Tester <hook-test@example.com>"
    )
    git(repo, "commit", "--quiet", "-m", message)
    assert git(repo, "log", "-1", "--format=%B").stdout.rstrip() == message


@pytest.mark.parametrize("value", [
    "", "Hook Tester", "Hook Tester; design-review=", "Hook Tester; design-review=xyz!",
    "Hook Tester; design-review=123", "Hook Tester; design-review=12345",
    "Hook Tester; design-review=ABCD",
    " ; design-review=1234", "Hook Tester; core-design=1234",
])
def test_missing_or_malformed_acknowledgment_is_rejected(repo, value):
    assert_rejected(repo, f"Require design review\n\nDesign-Signed-Off-By: {value}")


def test_absent_trailer_is_rejected(repo):
    assert_rejected(repo, "Require design review\n\nReview the staged change.")


def test_editor_comments_after_trailers_are_ignored(repo):
    git(repo, "config", "core.commentChar", "#")
    message = "Require design review\n\n" + acknowledgment(repo)
    path = repo / "message.txt"
    path.write_text(message + "\n\n# Editor status comments.\n# On branch main\n")
    git(repo, "commit", "--quiet", "--cleanup=strip", "-F", str(path))
    assert git(repo, "log", "-1", "--format=%B").stdout.rstrip() == message


def test_body_occurrence_is_not_an_acknowledgment(repo):
    assert_rejected(
        repo,
        "Require design review\n\n" + acknowledgment(repo) + "\n\nAdditional body text.",
    )


def test_duplicate_acknowledgments_are_rejected(repo):
    trailer = acknowledgment(repo)
    assert_rejected(repo, f"Require design review\n\n{trailer}\n{trailer}")


def test_changed_staged_checklist_rejects_prior_acknowledgment(repo):
    trailer = acknowledgment(repo)
    path = repo / DOCUMENT
    path.write_bytes(path.read_bytes() + b"\nReview cache boundary transitions.\n")
    git(repo, "add", DOCUMENT)
    assert acknowledgment(repo) != trailer
    assert_rejected(repo, "Require design review\n\n" + trailer)
    git(repo, "commit", "--quiet", "-m", "Require design review\n\n" + acknowledgment(repo))


def test_unstaged_checklist_changes_do_not_change_required_digest(repo):
    staged = (repo / DOCUMENT).read_bytes()
    trailer = acknowledgment(repo)
    (repo / DOCUMENT).write_bytes(staged + b"\nUnstaged checklist changes.\n")
    git(repo, "commit", "--quiet", "-m", "Require design review\n\n" + trailer)
    committed = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{DOCUMENT}"])
    assert committed == staged
    assert git(repo, "status", "--porcelain").stdout == f" M {DOCUMENT}\n"


def test_missing_staged_document_is_rejected(repo):
    trailer = acknowledgment(repo)
    git(repo, "rm", "--cached", DOCUMENT)
    (repo / "change.txt").write_text("A staged change.\n")
    git(repo, "add", "change.txt")
    result = git(repo, "commit", "--quiet", "-m", "Require design review\n\n" + trailer, check=False)
    assert result.returncode != 0
    assert f"cannot read the staged {DOCUMENT}" in result.stderr
    assert f"Read {DOCUMENT}" in result.stderr
    assert git(repo, "rev-parse", "--verify", "HEAD", check=False).returncode != 0


def test_linked_worktree_uses_shared_hook_configuration(repo, tmp_path):
    git(repo, "commit", "--quiet", "-m", "Require design review\n\n" + acknowledgment(repo))
    worktree = tmp_path / "linked"
    git(repo, "worktree", "add", "--quiet", "-b", "linked", str(worktree))
    (worktree / "change.txt").write_text("A staged worktree change.\n")
    git(worktree, "add", "change.txt")
    previous = git(worktree, "rev-parse", "HEAD").stdout
    result = git(worktree, "commit", "--quiet", "-m", "Change in a linked worktree", check=False)
    assert result.returncode != 0
    assert "Design review:" in result.stderr
    assert git(worktree, "rev-parse", "HEAD").stdout == previous
    git(worktree, "commit", "--quiet", "-m", "Change in a linked worktree\n\n" + acknowledgment(worktree))
