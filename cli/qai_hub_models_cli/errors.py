# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations


class CLIUserError(Exception):
    """An expected error the CLI reports as a message, without a stack trace."""


class CLIUnknownNameError(KeyError, CLIUserError):
    """A user-supplied name (model, device, runtime, ...) that does not exist."""


# Keep the original builtin as a base so existing `except` clauses still match.
class CLIFileNotFoundError(FileNotFoundError, CLIUserError):
    pass


class CLIFileExistsError(FileExistsError, CLIUserError):
    pass


class CLIRuntimeError(RuntimeError, CLIUserError):
    pass


class CLITimeoutError(TimeoutError, CLIUserError):
    pass


class CLITypeError(TypeError, CLIUserError):
    pass


class AssetNotFoundError(FileNotFoundError, CLIUserError):
    def __init__(self, *args: object, model_sharing_restricted: bool = False) -> None:
        self.model_sharing_restricted = model_sharing_restricted
        super().__init__(*args)


class UnsupportedVersionError(RuntimeError, CLIUserError):
    pass
