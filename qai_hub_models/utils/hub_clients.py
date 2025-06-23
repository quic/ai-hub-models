# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import configparser
import os
import threading
from contextlib import contextmanager
from typing import Optional

import qai_hub as hub
from qai_hub.api_utils import str2bool
from qai_hub.client import Client as HubClient
from qai_hub.public_rest_api import ClientConfig as HubClientConfig

from qai_hub_models.utils.asset_loaders import EXECUTING_IN_CI_ENVIRONMENT

"""
A mapping of Hub clients for the given deployment names.
The mapping is Map<AI Hub deployment name (lower case), AI Hub client>

Deployment names are mapped to None in the dictionary if a client for them could not be created.
"""
# Dict: <user, dict<deployment, Client>>
_CACHED_CLIENTS: dict[str, dict[str, HubClient | None]] = {}
DEFAULT_CLIENT_USER = "DEFAULT"
PRIVATE_SCORECARD_CLIENT_USER = "PRIVATE"
HUB_GLOBAL_CLIENT_CONFIG_OVERRIDE_REENTRANT_LOCK = threading.RLock()


def deployment_is_prod(deployment: str):
    return deployment.lower() in ["app", "prod"]


def _get_global_client() -> Optional[tuple[str, HubClient]]:
    """
    Returns [global client Hub deployment name], [global client]
    """
    try:
        global_client = hub.hub._global_client
        # The deployment name is the subdomain of AIHub that is used by the config.
        # Transform https://blah.aihub.qualcomm.com/ -> blah
        deployment_name = global_client.config.api_url.split("/")[2].split(".")[0]

        #
        # Prod deployment is a special case where the deployment "name" does not match the URL
        # "prod" deployment is "app.aihub.qualcomm.com"
        #
        deployment_name = (
            "prod" if deployment_name == "app" else deployment_name.lower()
        )

        return deployment_name, global_client
    except FileNotFoundError:
        # no global config exists
        pass
    except hub.client.UserError:
        # no global config exists
        pass
    except IndexError:
        # Can't read the global config's api_url. Ignore it
        pass

    return None


DEFAULT_GLOBAL_CLIENT = _get_global_client()


def _read_hub_config(path: str) -> Optional[HubClient]:
    if not os.path.exists(path):
        return None

    config = configparser.ConfigParser()
    config.read(path)
    try:
        client_config = config["api"]
        api_config = HubClientConfig(
            api_url=client_config["api_url"],
            web_url=client_config["web_url"],
            api_token=client_config["api_token"],
            verbose=(
                str2bool(client_config["verbose"])
                if "verbose" in client_config
                else True
            ),
        )
        return HubClient(api_config)
    except KeyError:
        pass

    return None


def get_hub_client(
    deployment_name: str = "prod", user: str = DEFAULT_CLIENT_USER
) -> Optional[HubClient]:
    user = user.upper()
    token_envvar_prefix = (
        f"HUB_{user}_USER_TOKEN_" if user != DEFAULT_CLIENT_USER else "HUB_USER_TOKEN_"
    )
    deployment_name = deployment_name.lower()
    deployment_name = "prod" if deployment_name == "app" else deployment_name

    # Return Cached client if applicable
    if user in _CACHED_CLIENTS:
        if deployment_name in _CACHED_CLIENTS[user]:
            return _CACHED_CLIENTS[user][deployment_name]

    # Create client if their tokens are in the global environment.
    # Bash env variables can't have {".", "-"} characters in the name, replace with "_" for valid naming
    upper_deployment_name = deployment_name.upper().replace("-", "_").replace(".", "_")
    client = None
    if user_token := os.environ.get(
        f"{token_envvar_prefix}{upper_deployment_name}", None
    ):
        #
        # Prod deployment is a special case where the deployment "name" does not match the URL
        # "prod" deployment is "app.aihub.qualcomm.com"
        #
        deployment_name_url = "app" if deployment_name == "prod" else deployment_name
        #

        api_url = os.environ.get(
            f"HUB_API_URL_{upper_deployment_name}",
            f"https://{deployment_name_url}.aihub.qualcomm.com/",
        )

        client = HubClient(HubClientConfig(api_url, api_url, user_token, True))

    # If there is no environment variable set, check if there is a 'deploymentname_username.ini' file
    if client is None:
        config_name = (
            f"{deployment_name}_{user.lower()}"
            if user != DEFAULT_CLIENT_USER
            else f"{deployment_name}"
        )
        config_path = os.path.expanduser(f"~/.qai_hub/{config_name}.ini")
        client = _read_hub_config(config_path)

    # If there is no environment variable set and no special client file, check the default client
    if (
        client is None
        and user == DEFAULT_CLIENT_USER
        and DEFAULT_GLOBAL_CLIENT is not None
        and DEFAULT_GLOBAL_CLIENT[0] == deployment_name
    ):
        client = DEFAULT_GLOBAL_CLIENT[1]

    if user not in _CACHED_CLIENTS:
        _CACHED_CLIENTS[user] = {}
    _CACHED_CLIENTS[user][deployment_name] = client
    return client


def get_hub_client_or_raise(
    deployment_name: str = "prod", user: str = DEFAULT_CLIENT_USER
) -> HubClient:
    hub_client = get_hub_client(deployment_name, user)
    if not hub_client:
        raise ValueError(
            f"Unable to create a client for Hub deployment {deployment_name}"
        )
    return hub_client


def get_scorecard_client(
    deployment_name: str = "prod", restrict_access: bool = False
) -> Optional[HubClient]:
    user = DEFAULT_CLIENT_USER
    if EXECUTING_IN_CI_ENVIRONMENT and restrict_access:
        user = PRIVATE_SCORECARD_CLIENT_USER
    return get_hub_client(deployment_name, user=user)


def get_scorecard_client_or_raise(
    deployment_name: str = "prod", restrict_access: bool = False
) -> HubClient:
    user = DEFAULT_CLIENT_USER
    if EXECUTING_IN_CI_ENVIRONMENT and restrict_access:
        user = PRIVATE_SCORECARD_CLIENT_USER
    return get_hub_client_or_raise(deployment_name, user=user)


def set_default_hub_client(client: hub.client.Client):
    hub.hub._global_client = client
    for default_global_client_method in hub.hub.__all__:
        setattr(
            hub,
            default_global_client_method,
            getattr(client, default_global_client_method),
        )
        setattr(
            hub.hub,
            default_global_client_method,
            getattr(client, default_global_client_method),
        )


@contextmanager
def default_hub_client_as(client: hub.client.Client):
    """
    Within this context, the default Hub client is replaced by the given client.

    To prevent unexpected behavior, only 1 thread can use this context at a time.
    Contexts can be nested in the call stack so long as they live on the same thread.
    """
    prev_client = hub.hub._global_client
    with HUB_GLOBAL_CLIENT_CONFIG_OVERRIDE_REENTRANT_LOCK:
        try:
            set_default_hub_client(client)
            yield client
        finally:
            set_default_hub_client(prev_client)


def get_default_hub_deployment() -> str | None:
    if hub_client := _get_global_client():
        return hub_client[0]
    return None
