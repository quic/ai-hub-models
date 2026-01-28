# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, TypedDict

from slack_sdk import WebClient
from slack_sdk.web.slack_response import SlackResponse

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Configuration: read token from environment. Do NOT hard-code secrets.
# -----------------------------------------------------------------------------
SLACK_BOT_TOKEN_VAR = "SLACK_BOT_TOKEN"
slack_token = os.getenv(SLACK_BOT_TOKEN_VAR)
if not slack_token:
    raise RuntimeError(
        f"Environment variable {SLACK_BOT_TOKEN_VAR} is not set. "
        "Export your Slack bot token before running, e.g.: "
        'export SLACK_BOT_TOKEN="xoxb-***"'
    )

client = WebClient(token=slack_token)


# -----------------------------------------------------------------------------
# Minimal types to keep mypy happy for fields we use
# -----------------------------------------------------------------------------
class UserInfoDict(TypedDict, total=False):
    user: Mapping[str, Any]


class ChannelDict(TypedDict, total=False):
    id: str
    name: str


class MessageDict(TypedDict, total=False):
    text: str
    ts: str
    reply_count: int
    user: str


# Cache for user names to avoid repeated API calls
user_cache: dict[str, str] = {}


def get_user_name(user_id: str) -> str:
    """Return a display name for a Slack user id."""
    if not user_id:
        return "Unknown"

    if user_id in user_cache:
        return user_cache[user_id]

    try:
        resp: SlackResponse = client.users_info(user=user_id)
        # SlackResponse is Mapping-like; guard access for mypy.
        data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
        if isinstance(data, Mapping):
            user = data.get("user")
            if isinstance(user, Mapping):
                name = user.get("real_name")
                if isinstance(name, str) and name:
                    user_cache[user_id] = name
                    return name
        return "Unknown"
    except Exception as e:
        # Keep broad catch for robustness here, but still log.
        logger.warning(
            "Failed to resolve user_id=%s; returning 'Unknown'. Error: %s", user_id, e
        )
        return "Unknown"


def main() -> None:
    # Fetch all channels
    channels: list[ChannelDict] = []
    excluded_channel_ids = {
        "C07DMDFJAJX",
        "C07H7569R7H",
        "C0868KDMDEJ",
        "C08RV59CESC",
        "C089L6VEM28",
        "C08BNQTJLTW",
        "C09A5SF7HJA",
        "C08R9DQJKKM",
        "C09EVG667M0",
        "C09LLDL69QV",
        "C089S1F862K",
        "C099Y1GBDKK",
        "C07A5PJ7M8B",
    }

    cursor: str | None = None
    while True:
        resp: SlackResponse = client.conversations_list(
            types="public_channel", cursor=cursor
        )
        data = resp.data if hasattr(resp, "data") else resp  # type: ignore[assignment]
        if not isinstance(data, Mapping):
            logger.error("Unexpected conversations_list response type")
            break

        raw_channels = data.get("channels", [])
        if isinstance(raw_channels, list):
            for ch in raw_channels:
                if isinstance(ch, Mapping):
                    channels.append(
                        ChannelDict(
                            id=str(ch.get("id", "")),
                            name=str(ch.get("name", "")),
                        )
                    )

        meta = data.get("response_metadata") if isinstance(data, Mapping) else None
        next_cursor = None
        if isinstance(meta, Mapping):
            nc = meta.get("next_cursor")
            next_cursor = str(nc) if nc else None

        if next_cursor:
            cursor = next_cursor
        else:
            break

    # Calculate timestamp for 7 days ago
    one_week_ago = datetime.now() - timedelta(days=7)
    oldest_timestamp = one_week_ago.timestamp()

    output_path = "/Users/meghan/Downloads/slack_messages_all_channels_this_week.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Text",
                "Date Submitted",
                "Channel Name",
                "Replies in Thread",
                "Submitted By",
            ]
        )

        # Iterate through channels
        for channel in channels:
            channel_id = channel.get("id", "")
            channel_name = channel.get("name", "")

            # Skip excluded channels by ID
            if channel_id in excluded_channel_ids:
                logger.info("Skipping channel ID: %s (%s)", channel_id, channel_name)
                continue

            logger.info("Processing channel: %s", channel_name)

            msg_cursor: str | None = None
            while True:
                hist_resp: SlackResponse = client.conversations_history(
                    channel=channel_id,
                    cursor=msg_cursor,
                    oldest=str(oldest_timestamp),
                )
                hdata = hist_resp.data if hasattr(hist_resp, "data") else hist_resp  # type: ignore[assignment]
                if not isinstance(hdata, Mapping):
                    logger.error("Unexpected conversations_history response type")
                    break

                msgs = hdata.get("messages", [])
                if isinstance(msgs, list):
                    for m in msgs:
                        if not isinstance(m, Mapping):
                            continue
                        text = str(m.get("text", ""))
                        ts_raw = m.get("ts")
                        # ts is a string like "1731349200.000100" — guard parse
                        try:
                            ts_float = float(ts_raw) if ts_raw is not None else 0.0
                        except (TypeError, ValueError):
                            ts_float = 0.0
                        date_submitted = (
                            datetime.fromtimestamp(ts_float).strftime("%Y-%m-%d")
                            if ts_float
                            else ""
                        )
                        reply_count = m.get("reply_count", 0)
                        replies = (
                            "Y"
                            if (isinstance(reply_count, int) and reply_count > 0)
                            else "N"
                        )
                        user_id = (
                            str(m.get("user", "")) if m.get("user") is not None else ""
                        )
                        user_name = get_user_name(user_id)
                        writer.writerow(
                            [text, date_submitted, channel_name, replies, user_name]
                        )

                hmeta = (
                    hdata.get("response_metadata")
                    if isinstance(hdata, Mapping)
                    else None
                )
                next_cursor = None
                if isinstance(hmeta, Mapping):
                    nc = hmeta.get("next_cursor")
                    next_cursor = str(nc) if nc else None

                if next_cursor:
                    msg_cursor = next_cursor
                else:
                    break

    logger.info("✅ Export complete: %s", output_path)


if __name__ == "__main__":
    main()
