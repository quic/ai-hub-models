#!/usr/bin/env bash

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -euo pipefail

LOCAL_AWS_ACCESS_KEY_ID="$1"
LOCAL_AWS_SECRET_ACCESS_KEY="$2"
LOCAL_AWS_DEFAULT_REGION="$3"
LOCAL_AWS_PROFILE="$4"

aws configure set aws_access_key_id "$LOCAL_AWS_ACCESS_KEY_ID" --profile "$LOCAL_AWS_PROFILE"
aws configure set aws_secret_access_key "$LOCAL_AWS_SECRET_ACCESS_KEY" --profile "$LOCAL_AWS_PROFILE"
aws configure set region "$LOCAL_AWS_DEFAULT_REGION" --profile "$LOCAL_AWS_PROFILE"

aws sts get-caller-identity --profile "$LOCAL_AWS_PROFILE"
