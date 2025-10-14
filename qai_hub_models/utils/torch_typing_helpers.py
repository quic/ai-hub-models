# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Iterator
from typing import Generic, TypeVar, overload

import torch
from torch import nn

T = TypeVar("T", bound=nn.Module)


class TypedModuleList(Generic[T], nn.ModuleList):
    """
    Identical to nn.ModuleList with valid typings for indexing.
    """

    def __iter__(self) -> Iterator[T]:
        return super().__iter__()  # type: ignore[return-value]

    def append(self, module: T) -> "TypedModuleList[T]":  # type: ignore[override]
        return super().append(module)  # type: ignore[return-value]

    @overload
    def __getitem__(self, idx: slice) -> "TypedModuleList[T]": ...

    @overload
    def __getitem__(self, idx: int) -> T: ...

    def __getitem__(self, idx):  # type: ignore[no-untyped-def]
        return super().__getitem__(idx)

    def __setitem__(self, idx: int, module: T) -> None:  # type: ignore[override]
        super().__setitem__(idx, module)


class Conv2DWithBias(nn.Conv2d):
    """
    Identical to nn.Conv2D, but bias is strongly typed as non-None.
    """

    def __init__(self, *args, **kwargs):
        if "bias" in kwargs:
            assert kwargs["bias"]
        else:
            kwargs["bias"] = True
        super().__init__(*args, **kwargs)
        assert self.bias is not None
        self.bias: torch.Tensor
