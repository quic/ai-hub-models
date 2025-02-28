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
venv_path=$6


# Convert the deployments list to an array
IFS=',' read -r -a hub_deployments <<< "$deployments"

# Set env vars for accessing each deployment
for deployment in "${hub_deployments[@]}"; do
  deployment_upper=$(echo "$deployment" | tr '[:lower:]' '[:upper:]')

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
  export "HUB_USER_TOKEN_$deployment_upper"=$token
  echo "HUB_USER_TOKEN_$deployment_upper=$token" >> "$GITHUB_ENV"
  echo "Set HUB_USER_TOKEN_$deployment_upper=$token"

  export "HUB_API_URL_$deployment_upper"=$url
  echo "Set HUB_API_URL_$deployment_upper=$url"
  echo "HUB_API_URL_$deployment_upper=$url" >> "$GITHUB_ENV"
done

# Configure AI Hub to use the first deployment in the list by default
if [ -n "$venv_path" ]; then
  deployment_upper=$(echo "${hub_deployments[0]}" | tr '[:lower:]' '[:upper:]')
  token_varname="HUB_USER_TOKEN_${deployment_upper}"
  url_varname="HUB_API_URL_${deployment_upper}"

  # shellcheck disable=SC1091
  source "${venv_path}/bin/activate" && \
    qai-hub configure \
    --api_token "${!token_varname}" \
    --api_url "${!url_varname}" \
    --web_url "${!url_varname}" \
    --no-verbose
fi
