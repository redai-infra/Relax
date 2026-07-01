#!/usr/bin/env python3
"""Plan GitHub PR commits to cherry-pick into GitLab dev.

The script is intentionally read-only: it prints a sync plan and never changes
the working tree.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass


PR_RE = re.compile(r"(\(#\d+\)\s*$|^Merge pull request #\d+\b)")


@dataclass(frozen=True)
class Commit:
    sha: str
    timestamp: int
    subject: str


@dataclass(frozen=True)
class Match:
    method: str
    sha: str
    score: str


def git(args: list[str], *, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def git_lines(args: list[str]) -> list[str]:
    output = git(args)
    return output.splitlines() if output else []


def first_parent_commits(ref: str, limit: int) -> list[str]:
    args = ["rev-list", "--first-parent", "--reverse"]
    if limit > 0:
        args.extend(["--max-count", str(limit)])
    args.append(ref)
    return git_lines(args)


def commit_info(sha: str) -> Commit:
    raw = git(["show", "-s", "--format=%H%x00%ct%x00%s", sha])
    full_sha, timestamp, subject = raw.split("\x00", 2)
    return Commit(full_sha, int(timestamp), subject)


def normalize_subject(subject: str) -> str:
    subject = re.sub(r"\s*\(#\d+\)\s*$", "", subject)
    subject = re.sub(r"^Merge pull request #\d+ from \S+\s*", "", subject)
    return re.sub(r"\s+", " ", subject).strip().lower()


def patch_id(sha: str) -> str | None:
    diff = git(["show", "--first-parent", "--format=", "--find-renames", sha])
    if not diff:
        return None
    output = git(["patch-id", "--stable"], input_text=diff)
    return output.split()[0] if output else None


def is_ancestor(sha: str, ref: str) -> bool:
    result = subprocess.run(["git", "merge-base", "--is-ancestor", sha, ref])
    return result.returncode == 0


def build_dev_indexes(dev_ref: str, limit: int) -> tuple[dict[str, str], dict[str, Commit]]:
    patch_index: dict[str, str] = {}
    subject_index: dict[str, Commit] = {}
    for sha in reversed(first_parent_commits(dev_ref, limit)):
        info = commit_info(sha)
        subject_index.setdefault(normalize_subject(info.subject), info)
        pid = patch_id(sha)
        if pid:
            patch_index.setdefault(pid, sha)
    return patch_index, subject_index


def match_in_dev(
    commit: Commit, dev_ref: str, patch_index: dict[str, str], subject_index: dict[str, Commit]
) -> Match | None:
    if is_ancestor(commit.sha, dev_ref):
        return Match("exact-sha", commit.sha, "high")

    pid = patch_id(commit.sha)
    if pid and pid in patch_index:
        return Match("patch-id", patch_index[pid], "high")

    normalized = normalize_subject(commit.subject)
    candidate = subject_index.get(normalized)
    if candidate:
        days = abs(candidate.timestamp - commit.timestamp) // 86400
        score = "medium" if days <= 14 else "low"
        return Match(f"subject-date({days}d)", candidate.sha, score)

    return None


def is_pr_commit(commit: Commit) -> bool:
    return PR_RE.search(commit.subject) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github", default="github/main", help="GitHub ref to scan")
    parser.add_argument("--dev", default="gitlab/dev", help="GitLab dev ref")
    parser.add_argument("--limit", type=int, default=3000, help="first-parent commits to inspect per ref; 0 means all")
    args = parser.parse_args()

    github_shas = first_parent_commits(args.github, args.limit)
    github_infos = [commit_info(sha) for sha in github_shas]
    patch_index, subject_index = build_dev_indexes(args.dev, args.limit)

    matches: dict[str, Match] = {}
    for commit in github_infos:
        match = match_in_dev(commit, args.dev, patch_index, subject_index)
        if match:
            matches[commit.sha] = match

    pr_commits = [commit for commit in github_infos if is_pr_commit(commit)]
    missing_prs = [
        commit
        for commit in pr_commits
        if commit.sha not in matches or matches[commit.sha].method.startswith("subject-date")
    ]

    print(f"github_ref: {args.github}")
    print(f"dev_ref: {args.dev}")
    print(f"github_head: {git(['rev-parse', args.github])}")
    print(f"dev_head: {git(['rev-parse', args.dev])}")

    if not missing_prs:
        print("status: no-unabsorbed-github-pr-commits")
        print("external_commits_after_A: <none>")
        return 0

    first_missing = missing_prs[0]
    first_missing_index = github_infos.index(first_missing)

    anchor: Commit | None = None
    anchor_match: Match | None = None
    for commit in reversed(github_infos[:first_missing_index]):
        match = matches.get(commit.sha)
        if match and match.score in {"high", "medium"}:
            anchor = commit
            anchor_match = match
            break

    if anchor is None or anchor_match is None:
        print("status: commitA-not-found")
        print(f"first_missing_external_pr: {first_missing.sha} {first_missing.subject}")
        print("action: stop-and-ask-user")
        return 2

    external_after_anchor = [
        commit for commit in pr_commits if github_infos.index(commit) > github_infos.index(anchor)
    ]
    commit_a = anchor_match.sha

    print("status: plan-ready")
    print(f"commitA: {commit_a}")
    print(f"commitA_match: {anchor_match.method} confidence={anchor_match.score}")
    print(f"github_anchor: {anchor.sha}")
    print(f"github_anchor_subject: {anchor.subject}")
    print(f"recommended_branch: sync/github-main-to-dev-{commit_a[:12]}")
    print("external_commits_after_A:")
    for commit in external_after_anchor:
        match = matches.get(commit.sha)
        state = "not-in-dev" if match is None else f"in-dev:{match.method}:{match.score}"
        print(f"  {commit.sha} {state} {commit.subject}")

    print("recommended_cherry_pick_order:")
    for commit in external_after_anchor:
        print(f"  {commit.sha}")

    low_confidence = anchor_match.score != "high"
    if low_confidence:
        print("warning: commitA is not high-confidence; stop and ask the user to confirm before cherry-pick")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
