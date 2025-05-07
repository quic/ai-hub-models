#!/bin/bash

# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Arguments
deployments="$1" # List of deployments separated by commas (e.g., "prod,staging,dev")
prod_token=$2
staging_token=$3
dev_token=$4
sandbox_token=$5
user=${6:-"DEFAULT"}
venv_path=$7

user_upper=$(echo "$user" | tr '[:lower:]' '[:upper:]')
if [ "$user_upper" == "DEFAULT" ]; then
  TOKEN_PREFIX="HUB_USER_TOKEN_"
else
  TOKEN_PREFIX="HUB_${user_upper}_USER_TOKEN_"
fi

# Convert the deployments list to an array
IFS=',' read -r -a hub_deployments <<< "$deployments"

# Set env vars for accessing each deployment
for deployment in "${hub_deployments[@]}"; do
  # Bash env variables can't have {".", "-"} characters in the name, replace with "_" for valid naming
  deployment_upper=$(echo "$deployment" | tr '[:lower:]' '[:upper:]' | sed 's/-/_/g' | sed 's/\./_/g')

  case "$deployment_upper" in
    PROD)
      token=$prod_token
      deployment_url_name="app"
      ;;
    APP)
      token=$prod_token
      deployment_url_name="app"
      ;;
    STAGING)
      token=$staging_token
      deployment_url_name="staging"
      ;;
    DEV)
      token=$dev_token
      deployment_url_name="dev"
      ;;
    *)
      token=$sandbox_token
      deployment_url_name=$deployment
      ;;
  esac

  url="https://${deployment_url_name}.aihub.qualcomm.com/"
  export "${TOKEN_PREFIX}$deployment_upper"=$token
  echo "${TOKEN_PREFIX}$deployment_upper=$token" >> "$GITHUB_ENV"
  echo "Set ${TOKEN_PREFIX}$deployment_upper=$token"

  export "HUB_API_URL_$deployment_upper"=$url
  echo "Set HUB_API_URL_$deployment_upper=$url"
  echo "HUB_API_URL_$deployment_upper=$url" >> "$GITHUB_ENV"
done

# Configure AI Hub to use the first deployment in the list by default
if [ -n "$venv_path" ]; then
  # Bash env variables can't have {".", "-"} characters in the name, replace with "_" for valid naming
  deployment_upper=$(echo "${hub_deployments[0]}" | tr '[:lower:]' '[:upper:]' | sed 's/-/_/g' | sed 's/\./_/g')
  token_varname="${TOKEN_PREFIX}${deployment_upper}"
  url_varname="HUB_API_URL_${deployment_upper}"

  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate" && \
    qai-hub configure \
    --api_token "${!token_varname}" \
    --api_url "${!url_varname}" \
    --web_url "${!url_varname}" \
    --no-verbose
fi
