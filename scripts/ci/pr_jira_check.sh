#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Main entrypoint for the PR JIRA title check CI job.
# Orchestrates team membership check, JIRA fetch, validation, labeling, and
# requesting the reviewers the labels call for (via pr_triage.py).
#
# Required env:
#   JIRA_TOKEN      - Bearer token for JIRA API
#   PR_AUTHOR       - GitHub username of the PR author
#   PR_TITLE        - PR title string
#   PR_NUMBER       - PR number
#
# Optional env:
#   GH_TOKEN        - GitHub token with org:read for team membership. Needed if not logged in to the GitHub CLI.
#   JIRA_PROJECT    - JIRA project key (default: TETRAAI)
#   REPO            - GitHub repository (default: qcom-ai-hub/ai-hub-models-internal)
#   EXEMPT_TEAMS    - Space-separated team slugs (default: ai-hub-models-reviewers tetra-developers)

set -euo pipefail

JIRA_PROJECT="${JIRA_PROJECT:-TETRAAI}"
REPO="${REPO:-qcom-ai-hub/ai-hub-models-internal}"
EXEMPT_TEAMS="${EXEMPT_TEAMS:-ai-hub-models-reviewers ai-hub-models-friends tetra-developers}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UTIL_DIR="$(cd "$SCRIPT_DIR/../util" && pwd)"

# 1. Check team membership
# shellcheck disable=SC2086
SKIP=$("$UTIL_DIR/check_github_team_membership.sh" "$PR_AUTHOR" $EXEMPT_TEAMS)

# 2. Extract ticket from title
TICKET=$(echo "$PR_TITLE" | grep -oP "${JIRA_PROJECT}-\d+" | head -1 || true)
if [ -z "$TICKET" ]; then
  if [ "$SKIP" != "true" ]; then
    echo "::error::PR title must contain a JIRA ticket (e.g., ${JIRA_PROJECT}-123). Current title: '$PR_TITLE'"
    exit 1
  fi
  echo "No JIRA found in PR title.  Current title: '$PR_TITLE'"
  exit 0
fi
echo "Found JIRA ticket: $TICKET"

# 3. Fetch JIRA issue
JIRA_JSON=$("$UTIL_DIR/fetch_jira_issue.sh" "$JIRA_TOKEN" "$TICKET")
echo "$JIRA_JSON" >&2

ERROR=$(echo "$JIRA_JSON" | jq -r '.error // empty')
if [ -n "$ERROR" ]; then
  MESSAGE=$(echo "$JIRA_JSON" | jq -r '.message')
  echo "::error::${TICKET}: ${MESSAGE}"
  exit 1
fi

ISSUE_TYPE=$(echo "$JIRA_JSON" | jq -r '.issue_type')
TYPE_OF_REQUEST=$(echo "$JIRA_JSON" | jq -r '.type_of_request')
STATUS_NAME=$(echo "$JIRA_JSON" | jq -r '.status_name')
STATUS_CATEGORY=$(echo "$JIRA_JSON" | jq -r '.status_category')
JIRA_LABELS=$(echo "$JIRA_JSON" | jq -r '.labels')

# 4. Validate ticket
if [ "$STATUS_CATEGORY" = "done" ]; then
  echo "::error::JIRA ticket '${TICKET}' must be open, not ${STATUS_NAME}."
  exit 1
fi

PR_LABEL=""
if [ "$ISSUE_TYPE" = "Onboarding" ]; then
  case "$TYPE_OF_REQUEST" in
    Non-GenAI) PR_LABEL="Onboarding - Non-GenAI" ;;
    GenAI)     PR_LABEL="Onboarding - GenAI" ;;
    Robotics)  PR_LABEL="Onboarding - Robotics" ;;
    *)
      echo "::error::Onboarding ticket '${TICKET}' must have 'Type of request' set to Non-GenAI, GenAI, or Robotics (got: '${TYPE_OF_REQUEST}')"
      exit 1
      ;;
  esac
fi

# 5. Apply labels
if [ -n "$PR_LABEL" ]; then
  gh pr edit "$PR_NUMBER" --add-label "$PR_LABEL" -R "$REPO"
  echo "Applied label: $PR_LABEL"
fi

if echo "$JIRA_LABELS" | grep -q "CE_BU_IOT"; then
  gh pr edit "$PR_NUMBER" --add-label "IoT" -R "$REPO"
  echo "Applied label: IoT"
fi

# 6. Request the reviewers the labels call for
if [ "$SKIP" != "true" ] && [ -n "$PR_LABEL" ]; then
  python3 "$SCRIPT_DIR/../../src/qai_hub_models/scripts/pr_triage.py" \
    --repo "$REPO" --pr "$PR_NUMBER" --apply \
    || echo "::warning::Requesting reviewers failed. The daily triage will retry."
fi
