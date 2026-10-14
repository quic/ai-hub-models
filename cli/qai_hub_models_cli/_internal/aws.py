# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import configparser
import contextlib
import functools
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import requests
import tqdm

from qai_hub_models_cli.common import sample_command
from qai_hub_models_cli.envvars import AWS_SESSION_DURATION_ENVVAR, bool_envvar_value
from qai_hub_models_cli.errors import (
    CLIFileNotFoundError,
    CLIRuntimeError,
    CLIUserError,
)

try:
    import boto3
    import botocore.exceptions
    import botocore.session
    from botocore.credentials import (
        AssumeRoleWithWebIdentityCredentialFetcher,
        DeferredRefreshableCredentials,
    )
    from botocore.exceptions import ClientError, NoCredentialsError
    from mypy_boto3_s3.service_resource import Bucket
except ImportError as e:
    raise ImportError(
        'AWS packages are missing. `pip install "qai_hub_models_cli[internal]"` and try again.'
    ) from e

CallableRetT = TypeVar("CallableRetT")

QAIHM_PRIVATE_S3_BUCKET = "qai-hub-models-private-assets"
QAIHM_AWS_PROFILE = "qaihm"
REGION = "us-west-2"
DEFAULT_SESSION_DURATION = 3600
MIN_SESSION_DURATION = 3600
MAX_SESSION_DURATION = 28800
# Role behind the qaihm profile; set in CI steps that need S3 past the 12h session cap.
QAIHM_AWS_ROLE_ARN_ENVVAR = "AWS_ROLE_ARN"
_OIDC_TOKEN_TIMEOUT = (30, 60)


def _get_session_duration() -> int:
    """Resolve AWS session duration from the envvar, clamped to IT limits.

    Invalid or missing values fall back to the default; out-of-range values
    are clamped rather than rejected so a caller who asks for "as long as
    possible" gets the ceiling instead of an error.
    """
    raw = os.environ.get(AWS_SESSION_DURATION_ENVVAR)
    if not raw:
        return DEFAULT_SESSION_DURATION
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SESSION_DURATION
    return max(MIN_SESSION_DURATION, min(value, MAX_SESSION_DURATION))


SETUP_DOCS_URL = "https://qualcomm-confluence.atlassian.net/wiki/spaces/ML/pages/3188064594/Private+AWS+Access+Setup"


class NoAWSCredsError(CLIUserError):
    def __init__(self) -> None:
        super().__init__(
            f"S3 credentials not found or expired. "
            f"Run `{sample_command('validate_aws_credentials')}` to refresh credentials, "
            f"or see {SETUP_DOCS_URL} for setup instructions."
        )


def attempt_with_s3_credentials_warning(
    s3_call: Callable[[], CallableRetT],
) -> CallableRetT:
    """
    Attempt to call the given function. Wrap the failure with a helpful warning about missing credentials.

    Typically you would call this like so:
        attempt_with_s3_credentials_warning(lambda: s3_download(bucket, key, path))
    """
    try:
        return s3_call()
    except (ClientError, NoCredentialsError) as e:
        if isinstance(e, NoCredentialsError) or e.response.get("Error", {}).get(
            "Code", None
        ) in ["400", "ExpiredToken"]:
            raise NoAWSCredsError() from e
        raise


def github_oidc_available() -> bool:
    """Whether this is a GitHub Actions job with ``id-token: write``."""
    return bool(
        os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
        and os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    )


def fetch_github_oidc_token() -> str:
    resp = requests.get(
        f"{os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']}&audience=sts.amazonaws.com",
        headers={
            "Authorization": f"Bearer {os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']}"
        },
        timeout=_OIDC_TOKEN_TIMEOUT,
    )
    resp.raise_for_status()
    return str(resp.json()["value"])


def oidc_refreshing_session(
    role_arn: str, region: str, session_name: str
) -> boto3.Session:
    """Session whose credentials re-assume ``role_arn`` via GitHub OIDC as they near expiry."""
    bc_session = botocore.session.Session()
    bc_session.set_config_variable("region", region)
    fetcher = AssumeRoleWithWebIdentityCredentialFetcher(
        client_creator=bc_session.create_client,
        web_identity_token_loader=fetch_github_oidc_token,
        role_arn=role_arn,
        extra_args={"RoleSessionName": session_name},
    )
    # botocore has no public setter for refreshable credentials.
    bc_session._credentials = DeferredRefreshableCredentials(  # type: ignore[attr-defined]
        refresh_using=fetcher.fetch_credentials,
        method="assume-role-with-web-identity",
    )
    return boto3.Session(botocore_session=bc_session, region_name=region)


def qaihm_session() -> boto3.Session:
    """Session for the qaihm role.

    The static qaihm profile written by CI dies after 12h, which long LLM perf
    jobs outlive (run 37102182278), so under GitHub OIDC credentials self-refresh.
    """
    role_arn = os.environ.get(QAIHM_AWS_ROLE_ARN_ENVVAR)
    if role_arn and github_oidc_available():
        return oidc_refreshing_session(role_arn, REGION, "qaihm-ci")
    return boto3.Session(profile_name=QAIHM_AWS_PROFILE)


def _load_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise CLIUserError(
            f"Missing required environment variable {name}. "
            f"See {SETUP_DOCS_URL} for setup instructions."
        )
    return value


def _profile_exists() -> bool:
    try:
        boto3.Session(profile_name=QAIHM_AWS_PROFILE)
    except botocore.exceptions.ProfileNotFound:
        logging.warning(f"Profile not found: {QAIHM_AWS_PROFILE}")
        return False
    return True


def _add_profile() -> None:
    config = configparser.ConfigParser()
    config_file = os.path.expanduser(f"~{os.sep}.aws{os.sep}config")

    config_dir = os.path.dirname(config_file)
    os.makedirs(config_dir, exist_ok=True)

    config.read(config_file)
    with contextlib.suppress(configparser.DuplicateSectionError):
        config.add_section(f"profile {QAIHM_AWS_PROFILE}")

    config.set(f"profile {QAIHM_AWS_PROFILE}", "region", REGION)
    config.set(f"profile {QAIHM_AWS_PROFILE}", "sts_regional_endpoints", "regional")

    with open(config_file, "w") as f:
        config.write(f)


def _prune_default() -> None:
    """Remove bare [default] section from ~/.aws/config if it exists."""
    config = configparser.ConfigParser()
    config_file = os.path.expanduser(f"~{os.sep}.aws{os.sep}config")
    config.read(config_file)
    config.remove_section("default")
    with open(config_file, "w") as f:
        config.write(f)


def _create_saml2aws_config(account_id: str, role: str, idp_app_id: str) -> None:
    config_file = os.path.expanduser("~/.saml2aws")
    user_email = None
    if os.path.exists(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        with contextlib.suppress(configparser.NoSectionError):
            user_email = config.get("default", "username")

    user_email = user_email or input("Your qualcomm email address: ")

    with open(config_file, "w") as f:
        f.write(
            f"[default]\n"
            f"app_id                  = {idp_app_id}\n"
            f"url                     = https://account.activedirectory.windowsazure.com\n"
            f"username                = {user_email}\n"
            f"provider                = AzureAD\n"
            f"mfa                     = Auto\n"
            f"skip_verify             = false\n"
            f"timeout                 = 0\n"
            f"aws_urn                 = urn:amazon:webservices\n"
            f"aws_session_duration    = {_get_session_duration()}\n"
            f"aws_profile             = {QAIHM_AWS_PROFILE}\n"
            f"role_arn                = arn:aws:iam::{account_id}:role/{role}\n"
            f"region                  = {REGION}\n"
            f"saml_cache              = false\n"
            f"disable_remember_device = false\n"
            f"disable_sessions        = false\n"
            f"download_browser_driver = false\n"
            f"headless                = false\n"
        )


def _add_default_credentials_section() -> None:
    config_file = os.path.expanduser("~/.aws/credentials")
    config = configparser.ConfigParser()
    config.read(config_file)
    if not config.has_section("default"):
        config.add_section("default")

    for option, value in config.items(QAIHM_AWS_PROFILE):
        config.set("default", option, value)

    with open(config_file, "w") as f:
        config.write(f)


@functools.cache
def _pass_initialized() -> bool:
    """Check if pass (password-store) has a valid GPG key configured."""
    gpg_id_file = os.path.expanduser("~/.password-store/.gpg-id")
    if not os.path.exists(gpg_id_file):
        return False
    with open(gpg_id_file) as f:
        key_id = f.readline().strip()
    result = subprocess.run(
        ["gpg", "--list-keys", key_id],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


@functools.cache
def _is_password_saved() -> bool:
    if sys.platform == "darwin":
        result = subprocess.run(
            [
                "security",
                "find-internet-password",
                "-s",
                "account.activedirectory.windowsazure.com",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.find("saml2aws") != -1:
            return True
    elif sys.platform == "linux" and _pass_initialized():
        result = subprocess.run(
            [
                "pass",
                "show",
                "saml2aws/https:/account.activedirectory.windowsazure.com",
            ],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    return False


def _clear_saved_password() -> None:
    if sys.platform == "darwin":
        subprocess.run(
            [
                "security",
                "delete-internet-password",
                "-s",
                "account.activedirectory.windowsazure.com",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    elif sys.platform == "linux" and _pass_initialized():
        subprocess.run(
            [
                "pass",
                "delete",
                "-f",
                "saml2aws/https:/account.activedirectory.windowsazure.com",
            ],
            capture_output=True,
            check=False,
        )


@functools.cache
def credentials_valid() -> bool:
    """Check if the current AWS credentials for the qaihm profile are valid."""
    try:
        session = boto3.Session(profile_name=QAIHM_AWS_PROFILE)
        session.client("sts").get_caller_identity()
    except (
        botocore.exceptions.NoCredentialsError,
        botocore.exceptions.ClientError,
        botocore.exceptions.ProfileNotFound,
    ):
        return False
    return True


def validate_credentials() -> None:
    """Ensure AWS credentials are valid, refreshing via saml2aws if needed."""
    account_id = _load_env("QAIHM_AWS_ACCOUNT_ID")
    admin_role = os.environ.get("QAIHM_AWS_ADMIN_ROLE", "")
    role = admin_role or _load_env("QAIHM_AWS_ROLE")
    idp_app_id = _load_env("QAIHM_AWS_IDP_APP_ID")

    if not _profile_exists():
        logging.info("Creating AWS profile entry")
        _add_profile()

    _create_saml2aws_config(account_id, role, idp_app_id)
    _prune_default()

    if credentials_valid():
        print("AWS credentials are valid.")
        return

    if not sys.stdin.isatty():
        raise CLIRuntimeError(
            "This is not a TTY and hence this script cannot prompt you for your password. "
            "Please re-run this in a different, interactive terminal."
        )

    print(f"Getting AWS credentials for {QAIHM_AWS_PROFILE}")

    command = ["saml2aws", "login"]

    if _is_password_saved():
        command.append("--skip-prompt")

    env: dict[str, str] = os.environ.copy()

    if sys.platform == "linux" and not _pass_initialized():
        command.append("--disable-keychain")
        if shutil.which("pass"):
            print(
                "pass is installed but not configured. Disabling keychain for now.\n"
                "To set up pass so saml2aws can save your password:\n\n"
                "gpg --batch --passphrase '' --quick-gen-key saml2aws\n"
                "pass init $(gpg --list-keys --with-colons saml2aws"
                " | awk -F: '/^fpr/{print $10; exit}')\n"
            )

    if shutil.which("saml2aws") is None:
        raise CLIFileNotFoundError(
            "saml2aws is not installed. Install it from "
            "https://github.com/Versent/saml2aws#install and try again."
        )

    try:
        subprocess.run(command, check=True, env=env)
    except Exception:
        if _is_password_saved():
            print(
                "Failed to authenticate. If you updated your password recently, that's probably why."
            )
            should_clear = input(
                "Would you like me to erase your saved password? y/N: "
            )
            if should_clear.lower() in ["y", "yes"]:
                _clear_saved_password()
                print("Saved password erased. Please try again.")
        raise

    _add_default_credentials_section()
    print("AWS credentials refreshed successfully.")


@functools.cache
def get_bucket(bucket_name: str = QAIHM_PRIVATE_S3_BUCKET) -> Bucket:
    """Get a boto3 Bucket object using the qaihm AWS profile."""
    try:
        session = qaihm_session()
        session.client("sts").get_caller_identity()
        return session.resource("s3").Bucket(bucket_name)
    except (botocore.exceptions.BotoCoreError, ClientError, NoCredentialsError) as e:
        raise NoAWSCredsError() from e


def s3_download(
    key: str,
    local_path: str | os.PathLike,
    bucket_name: str = QAIHM_PRIVATE_S3_BUCKET,
    quiet: bool = False,
) -> None:
    """Download a file from the private S3 bucket to a local path."""
    bucket = get_bucket(bucket_name)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    is_ci = bool_envvar_value("QAIHM_CI")
    obj = bucket.Object(key)
    with tqdm.tqdm(
        total=obj.content_length,
        unit="B",
        unit_scale=True,
        disable=quiet,
        desc=f"Downloading {key.rsplit('/', 1)[-1]}",
    ) as bar:
        attempt_with_s3_credentials_warning(
            lambda: obj.download_file(
                str(local_path),
                Callback=bar.update if not is_ci else None,
            )
        )
        if is_ci:
            bar.update(obj.content_length)


def s3_file_exists(key: str, bucket_name: str = QAIHM_PRIVATE_S3_BUCKET) -> bool:
    """Check if a file exists in the private S3 bucket."""
    bucket = get_bucket(bucket_name)
    try:
        attempt_with_s3_credentials_warning(lambda: bucket.Object(key).load())
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        raise
