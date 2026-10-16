# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Daily digest of open PRs targeting `main`: what needs review, and what is stale.

Outputs (all optional):
  --full-report-file     every tracked PR
  --summary-report-file  non-draft stale PRs + PRs with status "Needs Review"
  --slack-message-file   the summary as a Slack chat.postMessage body

GROUP. A PR goes in the first group that fits:
  1. Stale                   no PR activity for 14+ days
  2. Onboarding - GenAI      has that label
  3. Onboarding - Non-GenAI  has that label
  4. Onboarding - Robotics   has that label
  5. Other                   no onboarding label, author not in tetra-developers
                             or tetra-nexa-developers-china

STATUS. First match wins:
  1. Draft                      the PR is a draft
  2. Approved                   ai-hub-models-reviewers approved the PR; there are no requested changes
  3. WIP                        the Issues column has anything other than "CI running" or "changes requested", and
                                  the PR author has not re-requested a review from everyone
  4. Needs Review (India Team)  (Non-GenAI group only) no India reviewer has approved
  5. Needs Review               everything else

ISSUES.
  review comments pending    a comment thread is open & the PR author did not write last reply
  changes requested          a reviewer selected "request changes"
  merge conflicts            merge conflicts
  failing CI                 CI failed
  CI running                 CI still running
  required checks not run    a CI check required to merge never ran

REVIEWER(S). "AIHM reviewers" are the members of ai-hub-models-reviewers and
ai-hub-models-non-genai-reviewers (India Reviewer Team); everyone else is ignored.
  Approved PR      the AIHM reviewers who approved
  Any other PR     the AIHM reviewers who commented; if none have, the ones who
                   were asked to review or are assigned to the PR
Non-GenAI PRs adjust that first:
  an India reviewer approved         leave the India reviewers out
  no India approval yet              if an India reviewer was asked to review,
                                     show only the India reviewers

LAST UPDATE is the author's last activity (open, commit, force-push,
ready-for-review, comment, edit).

LAST REVIEW is a reviewer's last review or
comment, blank if none. Both show hours under a day (5h), otherwise days (3.2d).

Requires the `gh` CLI, authenticated locally via `gh auth login` or in Actions via
GH_TOKEN. The token must be able to read org teams (`read:org`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# =====================================================================
# 1. RULES
# =====================================================================

REVIEWERS_TEAM = "ai-hub-models-reviewers"
NON_GENAI_REVIEWERS_TEAM = "ai-hub-models-non-genai-reviewers"
DEVELOPER_TEAMS = ["tetra-developers", "tetra-nexa-developers-china"]
BOT_AUTHORS = {"qaihm-bot_QCOM"}
BASE_BRANCH = "main"
STALE_DAYS = 14

# Stale is checked first (see pick_group) but listed last.
STALE = "Stale"
GENAI = "Onboarding - GenAI"
NON_GENAI = "Onboarding - Non-GenAI"
ROBOTICS = "Onboarding - Robotics"
OTHER = "Other"
LABEL_GROUPS = [GENAI, NON_GENAI, ROBOTICS]
GROUP_ORDER = [*LABEL_GROUPS, OTHER, STALE]

NEEDS_REVIEW = "Needs Review"
NEEDS_REVIEW_INDIA = "Needs Review (India Team)"
APPROVED = "Approved"
WIP = "WIP"
DRAFT = "Draft"
STATUS_ORDER = [NEEDS_REVIEW, NEEDS_REVIEW_INDIA, APPROVED, WIP, DRAFT]

ISSUE_COMMENTS = "review comments pending"
ISSUE_CHANGES_REQUESTED = "changes requested"
ISSUE_CONFLICTS = "merge conflicts"
ISSUE_CI_FAILING = "failing CI"
ISSUE_CI_RUNNING = "CI running"
ISSUE_CHECKS_MISSING = "required checks not run"
# Shown in Issues, but alone they leave an unapproved PR at Needs Review.
NOT_WIP_ISSUES = {ISSUE_CI_RUNNING, ISSUE_CHANGES_REQUESTED}

PR_PAGE_SIZE = 30

# gh calls are tried MAX_ATTEMPTS times, waiting 5s, 10s, 20s between tries.
MAX_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 5
TRANSIENT_ERRORS = (
    "http 5",
    "502",
    "503",
    "504",
    "rate limit",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary failure",
    "eof",
    "unexpected end of json",
    "bad gateway",
    "service unavailable",
    "gateway time",
    "internal server error",
)

# Slack allows 100 rows per table and 10,000 characters of table text per message.
SLACK_TABLE_MAX_ROWS = 100
SLACK_TABLE_CHAR_BUDGET = 10_000
SLACK_PR_TITLE_MAX = 60

# =====================================================================
# 2. DATA TYPES
# =====================================================================


@dataclass
class PR:
    """One row in a report."""

    number: int
    title: str
    url: str
    author: str
    group: str
    status: str
    issues: list[str]
    reviewers: list[str]
    days_since_update: float
    days_since_review: float | None


@dataclass
class Context:
    """What is shared by every PR in a run."""

    aihm_reviewers: set[str]  # both reviewer teams; nobody else counts
    india_reviewers: set[str]  # non-GenAI team, minus anyone also in the main team
    developers: set[str]
    required_checks: set[str]
    now: datetime


@dataclass
class Activity:
    """When the author and AIHM reviewers last acted on a PR."""

    last_author_update: datetime
    last_review: datetime | None
    has_unanswered_thread: bool


# =====================================================================
# 3. FETCH (all the network code)
#
# We use GraphQL rather than the REST API because REST is far too slow here.
# Per PR we need each reviewer's latest review, comments, review threads,
# commits, labels, CI status and merge state. REST needs about 5 requests per PR,
# one after another (and has no flag for whether a review thread is resolved),
# while one GraphQL request returns all of that for 30 PRs at once.
# =====================================================================

# Threads fetch only the first and last comments, to stay well under GitHub's
# 500k-node limit.
THREAD_FRAGMENT = """
fragment ThreadFields on PullRequestReviewThread {
  isResolved
  firstComment: comments(first: 1) {
    nodes { author { login __typename } createdAt }
  }
  lastComments: comments(last: 2) {
    nodes { author { login } createdAt }
  }
}
"""

PR_QUERY = (
    """
query($owner: String!, $repo: String!, $base: String!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequests(states: OPEN, baseRefName: $base, first: $pageSize, after: $cursor) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        title
        url
        isDraft
        mergeable
        createdAt
        updatedAt
        lastEditedAt
        author { login }
        labels(first: 20) { nodes { name } }
        assignees(first: 10) { nodes { login } }
        reviewRequests(first: 20) {
          nodes { requestedReviewer { ... on User { login } } }
        }
        latestOpinionatedReviews(first: 100) {
          nodes { author { login } state submittedAt }
        }
        latestReviews(first: 100) {
          nodes { author { login } state }
        }
        comments(last: 100) {
          nodes { author { login } createdAt }
        }
        reviewThreads(first: 100) {
          pageInfo { hasNextPage endCursor }
          nodes { ...ThreadFields }
        }
        commits(last: 1) {
          nodes {
            commit {
              committedDate
              statusCheckRollup {
                state
                contexts(first: 100) {
                  totalCount
                  nodes {
                    __typename
                    ... on CheckRun { name }
                    ... on StatusContext { context }
                  }
                }
              }
            }
          }
        }
        timelineItems(
          last: 1
          itemTypes: [HEAD_REF_FORCE_PUSHED_EVENT, READY_FOR_REVIEW_EVENT]
        ) {
          nodes {
            ... on HeadRefForcePushedEvent { createdAt }
            ... on ReadyForReviewEvent { createdAt }
          }
        }
      }
    }
  }
}
"""
    + THREAD_FRAGMENT
)

THREADS_QUERY = (
    """
query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { ...ThreadFields }
      }
    }
  }
}
"""
    + THREAD_FRAGMENT
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def is_transient(stderr: str) -> bool:
    return any(marker in stderr.lower() for marker in TRANSIENT_ERRORS)


def gh(args: list[str]) -> str:
    """Run `gh <args>` and return stdout, retrying transient failures."""
    name = " ".join(args[:2])
    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(
            ["gh", *args], check=False, capture_output=True, text=True
        )
        if proc.returncode == 0:
            return proc.stdout
        error = proc.stderr.strip()
        if attempt == MAX_ATTEMPTS or not is_transient(error):
            raise RuntimeError(f"gh {name} failed: {error}")
        delay = RETRY_BASE_DELAY_S * 2 ** (attempt - 1)
        log(
            f"gh {name} failed ({error.splitlines()[0]}); "
            f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay}s"
        )
        time.sleep(delay)
    raise AssertionError("unreachable")


def gh_json(args: list[str]) -> Any:
    return json.loads(gh(args))


def graphql(query: str, **variables: Any) -> dict[str, Any]:
    """Run a GraphQL query and return its `repository` object."""
    args = ["api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        if value is not None:
            args += ["-F", f"{name}={value}"]
    return gh_json(args)["data"]["repository"]


def fetch_team_members(org: str, slug: str) -> set[str]:
    log(f"Fetching members of {org}/{slug}...")
    out = gh(
        ["api", "--paginate", f"orgs/{org}/teams/{slug}/members", "--jq", ".[].login"]
    )
    members = set(out.split())
    log(f"  {len(members)} members")
    return members


def fetch_required_checks(owner: str, repo: str) -> set[str]:
    """Names of the checks the base branch requires."""
    log(f"Fetching required checks for {BASE_BRANCH}...")
    try:
        rules = gh_json(["api", f"repos/{owner}/{repo}/rules/branches/{BASE_BRANCH}"])
    except RuntimeError as e:
        log(f"  could not read branch rules ({e}); skipping the required-check test")
        return set()
    required = {
        check["context"]
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    }
    log(f"  {len(required)} required checks")
    return required


def fetch_remaining_threads(owner: str, repo: str, gql_pr: dict[str, Any]) -> None:
    """Fetch a PR's review threads beyond the first 100, in place."""
    threads = gql_pr["reviewThreads"]
    while threads["pageInfo"]["hasNextPage"]:
        log(f"  PR #{gql_pr['number']}: {len(threads['nodes'])}+ review threads...")
        page = graphql(
            THREADS_QUERY,
            owner=owner,
            repo=repo,
            number=gql_pr["number"],
            cursor=threads["pageInfo"]["endCursor"],
        )["pullRequest"]["reviewThreads"]
        threads["nodes"] += page["nodes"]
        threads["pageInfo"] = page["pageInfo"]


def fetch_open_prs(owner: str, repo: str) -> list[dict[str, Any]]:
    """Every open PR, as the dicts that GraphQL returns (named `gql_pr` below).

    A `gql_pr` has the fields requested in PR_QUERY (latest reviews, comments, review
    threads, commits, labels, ...) exactly as GitHub sends them. The rest of the
    script turns each one into a `PR`.
    """
    prs: list[dict[str, Any]] = []
    cursor = None
    while True:
        log(f"Fetching open PRs (page {len(prs) // PR_PAGE_SIZE + 1})...")
        page = graphql(
            PR_QUERY,
            owner=owner,
            repo=repo,
            base=BASE_BRANCH,
            pageSize=PR_PAGE_SIZE,
            cursor=cursor,
        )["pullRequests"]
        for gql_pr in page["nodes"]:
            fetch_remaining_threads(owner, repo, gql_pr)
        prs += page["nodes"]
        log(f"  {len(prs)} PRs fetched so far")
        if not page["pageInfo"]["hasNextPage"]:
            return prs
        cursor = page["pageInfo"]["endCursor"]


# =====================================================================
# 4. FACTS ABOUT ONE PR
# =====================================================================


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def days_since(now: datetime, then: datetime) -> float:
    return (now - then).total_seconds() / 86400


def login(node: dict[str, Any]) -> str:
    return (node.get("author") or {}).get("login", "")


def is_bot(node: dict[str, Any]) -> bool:
    author = node.get("author") or {}
    return author.get("__typename") == "Bot" or author.get("login") in BOT_AUTHORS


def thread_comments(thread: dict[str, Any]) -> list[dict[str, Any]]:
    """The comments we fetched for a thread: its first and its last two."""
    return [*thread["firstComment"]["nodes"], *thread["lastComments"]["nodes"]]


def activity_events(gql_pr: dict[str, Any]) -> list[tuple[str, datetime]]:
    """(login, time) of each approval or change request, PR comment and thread comment."""
    events = [
        (login(review), parse_ts(review["submittedAt"]))
        for review in gql_pr["latestOpinionatedReviews"]["nodes"]
    ]
    events += [
        (login(comment), parse_ts(comment["createdAt"]))
        for comment in gql_pr["comments"]["nodes"]
    ]
    for thread in gql_pr["reviewThreads"]["nodes"]:
        events += [
            (login(comment), parse_ts(comment["createdAt"]))
            for comment in thread_comments(thread)
        ]
    return events


def awaits_author(thread: dict[str, Any], author: str) -> bool:
    """An open thread started by someone else (not a bot) that the author hasn't
    answered last.
    """
    first = thread["firstComment"]["nodes"]
    if thread["isResolved"] or not first:
        return False
    starter = first[0]
    last = thread["lastComments"]["nodes"][-1]
    return (
        login(starter) not in ("", author)
        and not is_bot(starter)
        and login(last) != author
    )


def collect_activity(
    gql_pr: dict[str, Any], author: str, listed_reviewers: set[str]
) -> Activity:
    events = activity_events(gql_pr)
    author_times = [
        parse_ts(gql_pr["createdAt"]),
        *([parse_ts(gql_pr["lastEditedAt"])] if gql_pr["lastEditedAt"] else []),
        *(parse_ts(c["commit"]["committedDate"]) for c in gql_pr["commits"]["nodes"]),
        *(parse_ts(t["createdAt"]) for t in gql_pr["timelineItems"]["nodes"]),
        *(when for who, when in events if who == author),
    ]
    reviewer_times = [when for who, when in events if who in listed_reviewers]
    return Activity(
        last_author_update=max(author_times),
        last_review=max(reviewer_times, default=None),
        has_unanswered_thread=any(
            awaits_author(t, author) for t in gql_pr["reviewThreads"]["nodes"]
        ),
    )


def latest_review_states(
    gql_pr: dict[str, Any], aihm_reviewers: set[str]
) -> dict[str, str]:
    """Each reviewer's current review state: approval, change request or comment.

    GitHub tracks this per reviewer, so PRs with hundreds of reviews don't need
    paging. A later comment doesn't override an earlier approval or change request.
    """
    states = {
        login(r): r["state"]
        for r in gql_pr["latestReviews"]["nodes"]
        if login(r) in aihm_reviewers
    }
    states.update(
        (login(r), r["state"])
        for r in gql_pr["latestOpinionatedReviews"]["nodes"]
        if login(r) in aihm_reviewers
    )
    return states


def pending_requests(gql_pr: dict[str, Any], aihm_reviewers: set[str]) -> list[str]:
    """Reviewers with a pending review request. Team requests aren't included."""
    requested = (n["requestedReviewer"] for n in gql_pr["reviewRequests"]["nodes"])
    return [r["login"] for r in requested if r and r["login"] in aihm_reviewers]


def requested_reviewers(gql_pr: dict[str, Any], aihm_reviewers: set[str]) -> list[str]:
    """AIHM reviewers with a pending review request or assigned to the PR."""
    assigned = [a["login"] for a in gql_pr["assignees"]["nodes"]]
    logins = [*pending_requests(gql_pr, aihm_reviewers), *assigned]
    return [who for who in dict.fromkeys(logins) if who in aihm_reviewers]


# =====================================================================
# 5. GROUP
# =====================================================================


def pick_group(
    gql_pr: dict[str, Any], labels: set[str], author: str, ctx: Context
) -> str:
    if (ctx.now - parse_ts(gql_pr["updatedAt"])).days >= STALE_DAYS:
        return STALE
    for group in LABEL_GROUPS:
        if group in labels:
            return group
    if author in ctx.developers or author in BOT_AUTHORS:
        return ""
    return OTHER


# =====================================================================
# 6. ISSUES AND STATUS
# =====================================================================


def ci_issues(gql_pr: dict[str, Any], required_checks: set[str]) -> list[str]:
    issues = []
    if gql_pr["mergeable"] == "CONFLICTING":
        issues.append(ISSUE_CONFLICTS)
    rollup = gql_pr["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
    if rollup is None:
        if required_checks:
            issues.append(ISSUE_CHECKS_MISSING)
        return issues
    if rollup["state"] in ("FAILURE", "ERROR"):
        issues.append(ISSUE_CI_FAILING)
    elif rollup["state"] in ("PENDING", "EXPECTED"):
        # A job that hasn't started yet looks like a missing check.
        return [*issues, ISSUE_CI_RUNNING]
    contexts = rollup["contexts"]
    # With more checks than we fetched, we can't tell which are missing.
    fetched_all = contexts["totalCount"] <= len(contexts["nodes"])
    seen = {c.get("name") or c.get("context") for c in contexts["nodes"]}
    if fetched_all and not required_checks <= seen:
        issues.append(ISSUE_CHECKS_MISSING)
    return issues


def find_issues(
    gql_pr: dict[str, Any],
    activity: Activity,
    changes_requested: bool,
    required_checks: set[str],
) -> list[str]:
    issues = []
    if activity.has_unanswered_thread:
        issues.append(ISSUE_COMMENTS)
    if changes_requested:
        issues.append(ISSUE_CHANGES_REQUESTED)
    return [*issues, *ci_issues(gql_pr, required_checks)]


def pick_status(
    is_draft: bool,
    approvers: set[str],
    issues: list[str],
    awaiting_india: bool,
    rerequested: bool,
) -> str:
    """`approvers` are the approvals that count; `awaiting_india` is Non-GenAI
    with no India approval yet. CI running and changes requested don't make a PR
    WIP, and neither do open threads once every reviewer was re-requested.
    """
    if is_draft:
        return DRAFT
    if approvers:
        return APPROVED
    ignored = NOT_WIP_ISSUES | ({ISSUE_COMMENTS} if rerequested else set())
    if set(issues) - ignored:
        return WIP
    return NEEDS_REVIEW_INDIA if awaiting_india else NEEDS_REVIEW


# =====================================================================
# 7. REVIEWERS
# =====================================================================


def pick_reviewers(
    submitted: list[str],
    requested: list[str],
    is_non_genai: bool,
    approvers: set[str],
    india: set[str],
) -> list[str]:
    listed = list(dict.fromkeys([*submitted, *requested]))
    if is_non_genai:
        if approvers & india:
            listed = [r for r in listed if r not in india]
        elif india & set(requested):
            listed = [r for r in listed if r in india]
    return listed


# =====================================================================
# 8. ANALYZE
# =====================================================================


def analyze(gql_pr: dict[str, Any], ctx: Context) -> PR:
    author = (gql_pr["author"] or {}).get("login", "ghost")
    labels = {label["name"] for label in gql_pr["labels"]["nodes"]}
    # Members of the two reviewer teams, excluding the PR author.
    aihm_reviewers = ctx.aihm_reviewers - {author}
    # Reviewers currently requested in the UI's Reviewers list (team requests excluded).
    pending = set(pending_requests(gql_pr, aihm_reviewers))
    # Each reviewer's latest review state: approved, changes requested or commented.
    states = latest_review_states(gql_pr, aihm_reviewers)
    # Reviewers who have submitted a review of any kind.
    submitted = set(states)
    # "Last review" only counts activity from pending or submitted reviewers.
    activity = collect_activity(gql_pr, author, pending | submitted)
    changes_requested = "CHANGES_REQUESTED" in states.values()
    issues = find_issues(gql_pr, activity, changes_requested, ctx.required_checks)

    # An approver who is asked to review again no longer counts as approved.
    approvers = {
        who
        for who, state in states.items()
        if state == "APPROVED" and who not in pending
    }
    # Every reviewer who left a non-approving review (change request or comment)
    # is asked again: the PR is back with the reviewers, so open threads stop
    # making it WIP (they still show in Issues).
    awaiting_rereview = submitted - {
        who for who, s in states.items() if s == "APPROVED"
    }
    rerequested = bool(pending) and awaiting_rereview <= pending

    is_non_genai = NON_GENAI in labels
    # On Non-GenAI an India approval only hands the PR on; it doesn't approve it.
    counted = approvers - ctx.india_reviewers if is_non_genai else approvers
    # Outstanding change requests cancel every approval, even if re-requested.
    final_approvers = set() if changes_requested else counted
    status = pick_status(
        gql_pr["isDraft"],
        final_approvers,
        issues,
        awaiting_india=is_non_genai and not approvers,
        rerequested=rerequested,
    )
    if status == APPROVED:
        shown = sorted(final_approvers)
    else:
        requested = requested_reviewers(gql_pr, aihm_reviewers)
        shown = pick_reviewers(
            sorted(submitted),
            requested,
            is_non_genai,
            approvers,
            ctx.india_reviewers,
        )

    return PR(
        number=gql_pr["number"],
        title=gql_pr["title"],
        url=gql_pr["url"],
        author=author,
        group=pick_group(gql_pr, labels, author, ctx),
        status=status,
        issues=issues,
        reviewers=shown,
        days_since_update=days_since(ctx.now, activity.last_author_update),
        days_since_review=(
            None
            if activity.last_review is None
            else days_since(ctx.now, activity.last_review)
        ),
    )


def build_groups(owner: str, repo: str) -> dict[str, list[PR]]:
    main_reviewers = fetch_team_members(owner, REVIEWERS_TEAM)
    non_genai_reviewers = fetch_team_members(owner, NON_GENAI_REVIEWERS_TEAM)
    ctx = Context(
        aihm_reviewers=main_reviewers | non_genai_reviewers,
        india_reviewers=non_genai_reviewers - main_reviewers,
        developers=set().union(
            *(fetch_team_members(owner, team) for team in DEVELOPER_TEAMS)
        ),
        required_checks=fetch_required_checks(owner, repo),
        now=datetime.now(timezone.utc),
    )
    gql_prs = fetch_open_prs(owner, repo)
    groups: dict[str, list[PR]] = {name: [] for name in GROUP_ORDER}
    for i, gql_pr in enumerate(gql_prs, 1):
        log(f"Analyzing PR {i}/{len(gql_prs)} (#{gql_pr['number']})")
        pr = analyze(gql_pr, ctx)
        if pr.group:
            groups[pr.group].append(pr)
    return groups


def summarize_groups(groups: dict[str, list[PR]]) -> dict[str, list[PR]]:
    """Non-draft stale PRs, plus PRs with status Needs Review."""
    summary = {}
    for name, prs in groups.items():
        if name == STALE:
            summary[name] = [p for p in prs if p.status != DRAFT]
        else:
            summary[name] = [p for p in prs if p.status == NEEDS_REVIEW]
    return summary


# =====================================================================
# 9. REPORTS
# =====================================================================

HEADERS = ["PR", "Status", "Issues", "Reviewer(s)", "Last update", "Last review"]
SLACK_COLUMN_SETTINGS: list[dict[str, Any]] = [
    {"is_wrapped": True},
    {},
    {"is_wrapped": True},
    {"is_wrapped": True},
    {"align": "right"},
    {"align": "right"},
]


def group_title(group: str) -> str:
    """The table title; label names are fixed by GitHub, so they're mapped here."""
    if group == OTHER:
        return "Other External PRs"
    return group.replace("Onboarding", "External Onboarding", 1)


def fmt_age(days: float | None) -> str:
    """5h under a day, otherwise 3.2d; empty if there is no date."""
    if days is None:
        return ""
    hours = round(days * 24)
    return f"{hours}h" if hours < 24 else f"{days:.1f}d"


def sort_prs(group: str, prs: list[PR]) -> list[PR]:
    """Longest since the last update first; non-stale groups sort by status first."""
    if group == STALE:
        return sorted(prs, key=lambda p: -p.days_since_update)
    return sorted(
        prs, key=lambda p: (STATUS_ORDER.index(p.status), -p.days_since_update)
    )


def pr_label(pr: PR, max_title: int | None = None) -> str:
    """The PR cell: #123 title (author)."""
    title = pr.title
    if max_title is not None and len(title) > max_title:
        title = title[: max_title - 1] + "…"
    return f"#{pr.number} {title} ({pr.author})"


def row_cells(pr: PR, max_title: int | None = None) -> list[str]:
    return [
        pr_label(pr, max_title),
        pr.status,
        ", ".join(pr.issues),
        ", ".join(pr.reviewers),
        fmt_age(pr.days_since_update),
        fmt_age(pr.days_since_review),
    ]


def escape_markdown(text: str) -> str:
    """Escape characters that break a markdown link or table."""
    for char in "\\[]|":
        text = text.replace(char, "\\" + char)
    return text


def render_markdown(groups: dict[str, list[PR]], title: str) -> str:
    lines = [f"# {title}", ""]
    for name in GROUP_ORDER:
        if not groups[name]:
            continue
        lines += [
            f"## {group_title(name)} ({len(groups[name])})",
            "",
            "| " + " | ".join(HEADERS) + " |",
            "|" + "---|" * len(HEADERS),
        ]
        for pr in sort_prs(name, groups[name]):
            cells = row_cells(pr)
            cells[0] = f"[{escape_markdown(cells[0])}]({pr.url})"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def slack_text_cell(text: str) -> dict[str, Any]:
    return {"type": "raw_text", "text": text or " "}


def slack_rich_cell(element: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "rich_text",
        "elements": [{"type": "rich_text_section", "elements": [element]}],
    }


def slack_bold_cell(text: str) -> dict[str, Any]:
    return slack_rich_cell({"type": "text", "text": text, "style": {"bold": True}})


def slack_link_cell(text: str, url: str) -> dict[str, Any]:
    return slack_rich_cell({"type": "link", "text": text, "url": url})


def slack_rows(prs: list[PR], budget: int) -> tuple[list[list[dict[str, Any]]], int]:
    """Rows for the PRs that fit in `budget` characters, and the characters used."""
    rows: list[list[dict[str, Any]]] = []
    used = sum(len(h) for h in HEADERS)
    for pr in prs:
        texts = row_cells(pr, SLACK_PR_TITLE_MAX)
        cost = sum(len(t) for t in texts)
        if used + cost > budget or len(rows) >= SLACK_TABLE_MAX_ROWS:
            continue
        used += cost
        rows.append(
            [slack_link_cell(texts[0], pr.url), *map(slack_text_cell, texts[1:])]
        )
    return rows, used


def slack_section(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def render_slack_payload(groups: dict[str, list[PR]]) -> dict[str, Any]:
    """A chat.postMessage body with one table per group.

    Rows past Slack's table limits are dropped and counted in a closing note.
    """
    blocks = [slack_section(":mag: *Open PR Digest*")]
    budget = SLACK_TABLE_CHAR_BUDGET
    hidden = 0
    for name in GROUP_ORDER:
        prs = sort_prs(name, groups[name])
        rows, used = slack_rows(prs, budget)
        hidden += len(prs) - len(rows)
        if not rows:
            continue
        budget -= used
        blocks += [
            slack_section(f"*{group_title(name)}* ({len(prs)})"),
            {
                "type": "table",
                "column_settings": SLACK_COLUMN_SETTINGS,
                "rows": [[slack_bold_cell(h) for h in HEADERS], *rows],
            },
        ]
    if hidden:
        note = f"{hidden} more PRs are not shown here. See the full report."
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": note}]}
        )
    return {"text": "Open PR Digest", "blocks": blocks}


# =====================================================================
# 10. MAIN
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name of the repository (default: $GITHUB_REPOSITORY, else the "
        "repo `gh` resolves for the current directory)",
    )
    parser.add_argument(
        "--full-report-file",
        help="Write the full markdown report (all PRs) to this file.",
    )
    parser.add_argument(
        "--summary-report-file",
        help="Write a markdown report with only stale PRs and PRs awaiting "
        "review to this file.",
    )
    parser.add_argument(
        "--slack-message-file",
        help="Write the summary as a Slack chat.postMessage JSON body (native "
        "table blocks) to this file.",
    )
    parser.add_argument(
        "--step-summary-file",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Append the full report here (defaults to the Actions step summary).",
    )
    return parser.parse_args()


def resolve_repo(repo: str | None) -> str:
    """--repo or $GITHUB_REPOSITORY, else the repo `gh` resolves locally."""
    if repo:
        return repo
    resolved = gh(
        ["repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    ).strip()
    log(f"Using repo {resolved} (from `gh`). If wrong, pass --repo.")
    return resolved


def main() -> None:
    args = parse_args()
    owner, repo = resolve_repo(args.repo).split("/")

    groups = build_groups(owner, repo)
    summary_groups = summarize_groups(groups)

    full_report = render_markdown(groups, "Open PR Digest - Full")
    print(full_report)
    if args.step_summary_file:
        with open(args.step_summary_file, "a") as f:
            f.write(full_report + "\n")
    if args.full_report_file:
        Path(args.full_report_file).write_text(full_report + "\n")
    if args.summary_report_file:
        summary = render_markdown(summary_groups, "Open PR Digest - Summary")
        Path(args.summary_report_file).write_text(summary + "\n")
    if args.slack_message_file:
        Path(args.slack_message_file).write_text(
            json.dumps(render_slack_payload(summary_groups))
        )


if __name__ == "__main__":
    main()
