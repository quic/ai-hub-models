# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import re
import shutil
import tempfile
import textwrap
from pathlib import Path
from zipfile import ZipFile

import requests
from prettytable import PrettyTable
from tqdm import tqdm

from qai_hub_models_cli.errors import CLIFileExistsError


def get_next_free_path(path: str | os.PathLike, delim: str = "-") -> Path:
    """Adds an incrementing number at the end of the given path until a non-existent filename is found."""
    path_without_ext, ext = os.path.splitext(path)

    counter = 1
    while os.path.exists(path):
        path = f"{path_without_ext}{delim}{counter}{ext}"
        counter += 1

    return Path(path)


_RETRYABLE_ERRORS = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ContentDecodingError,
    requests.exceptions.Timeout,
)


def _validate_download_response(response: requests.Response, url: str) -> None:
    """Raise on non-200/206 status or unexpected HTML content-type."""
    if response.status_code not in (200, 206):
        response.close()
        raise requests.HTTPError(
            f"Unable to download file at {url} (status {response.status_code})",
            response=response,
        )
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        response.close()
        raise ValueError(
            f"Expected a downloadable file at {url}"
            f" but received an HTML response."
            " The URL may have been redirected to a login or error page."
        )


def _get_total_size(response: requests.Response, bytes_downloaded: int) -> int:
    """Parse total file size from Content-Range or Content-Length headers."""
    content_range = response.headers.get("content-range", "")
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    return bytes_downloaded + int(response.headers.get("content-length", 0))


def _download_to_file(
    url: str, tmp_filepath: str, num_retries: int, quiet: bool = False
) -> None:
    """Download *url* into *tmp_filepath* with resume and retry."""
    filename = url.rsplit("/", 1)[-1]

    for attempt in range(num_retries + 1):
        bytes_downloaded = (
            os.path.getsize(tmp_filepath) if os.path.exists(tmp_filepath) else 0
        )
        headers = {"Range": f"bytes={bytes_downloaded}-"} if bytes_downloaded else {}

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=60)
            _validate_download_response(response, url)

            # Server ignored range request — restart from scratch.
            if bytes_downloaded > 0 and response.status_code == 200:
                bytes_downloaded = 0

            mode = "ab" if bytes_downloaded > 0 else "wb"
            total_size = _get_total_size(response, bytes_downloaded)

            with tqdm(
                total=total_size,
                initial=bytes_downloaded,
                unit="B",
                unit_scale=True,
                disable=quiet,
            ) as bar:
                bar.set_description(f"Downloading {filename}")
                with open(tmp_filepath, mode) as f:
                    for chunk in response.iter_content(1024 * 1024):
                        bar.update(len(chunk))
                        f.write(chunk)
                bar.set_postfix_str("Done")

            actual_size = os.path.getsize(tmp_filepath)
            if not total_size:
                if not quiet:
                    print(
                        "Warning: server did not report file size; "
                        "cannot verify download completeness."
                    )
            elif actual_size < total_size:
                if attempt < num_retries:
                    if not quiet:
                        print(
                            f"Download incomplete ({actual_size}/{total_size} bytes), "
                            f"retrying ({attempt + 1}/{num_retries})..."
                        )
                    continue
                raise ConnectionError(
                    f"Download incomplete after {num_retries} retries: "
                    f"got {actual_size}/{total_size} bytes from {url}"
                )
            return

        except _RETRYABLE_ERRORS as e:
            if attempt < num_retries:
                if not quiet:
                    print(
                        f"Download interrupted ({e.__class__.__name__}), "
                        f"retrying ({attempt + 1}/{num_retries})..."
                    )
            else:
                raise


def download(
    url: str,
    dst_path: str | os.PathLike,
    num_retries: int = 4,
    extract: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Download a file from *url* to *dst_path* with a progress bar.

    Supports resuming partial downloads on connection failures.
    If *extract* is True, the downloaded zip is extracted directly from the
    temp directory and the zip file is not kept.

    Parameters
    ----------
    url
        URL to download.
    dst_path
        Destination file path.
    num_retries
        Number of retry attempts on connection failure or incomplete download.
    extract
        If True, extract the downloaded zip and return the extraction directory
        (``dst_path`` without the ``.zip`` extension).
    quiet
        If True, suppress all output (progress bar, warnings, retry messages).

    Returns
    -------
    Path
        Path to the downloaded file, or extraction directory if *extract* is True.

    Raises
    ------
    FileExistsError
        If *dst_path* already exists.
    requests.HTTPError
        If the server returns a non-200/206 status.
    ConnectionError
        If the download is incomplete after all retries.
    """
    if url.startswith("s3://"):
        from qai_hub_models_cli._internal.aws import s3_download

        without_scheme = url.removeprefix("s3://")
        bucket_name, s3_key = without_scheme.split("/", 1)
        dst = Path(dst_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_fp = Path(tmp_dir) / dst.name
            s3_download(s3_key, tmp_fp, bucket_name=bucket_name, quiet=quiet)

            if extract:
                return extract_zip_file(tmp_fp, dst.parent / dst.stem)

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_fp, dst)

        return dst

    dst = Path(dst_path)
    if dst.exists():
        raise CLIFileExistsError(f"Destination already exists: {dst}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_filepath = os.path.join(tmp_dir, dst.name)
        _download_to_file(url, tmp_filepath, num_retries, quiet=quiet)

        if extract:
            return extract_zip_file(tmp_filepath, dst.parent / dst.stem)

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(tmp_filepath, dst)

    return dst


def extract_zip_file(
    filepath: str | os.PathLike,
    out_path: str | os.PathLike | None = None,
) -> Path:
    """
    Extract a zip file's contents into *out_path*.

    If the archive contains a single top-level directory, its contents
    are unwrapped directly into *out_path*. If the archive contains multiple
    top-level entries, they are all placed into *out_path* as-is.

    Parameters
    ----------
    filepath
        Path to the zip file.
    out_path
        Destination directory. If None, defaults to the zip file path
        without the .zip extension. Must not already exist.

    Returns
    -------
    Path
        Path to the extracted directory.
    """
    filepath = Path(filepath)
    out_path = Path(out_path if out_path else filepath.parent / filepath.stem)

    if out_path.exists():
        raise CLIFileExistsError(f"Cannot extract to an existing directory: {out_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(filepath, "r") as zf:
            tmp_resolved = Path(tmp).resolve()
            for member in zf.namelist():
                if not (tmp_resolved / member).resolve().is_relative_to(tmp_resolved):
                    raise ValueError(f"Unsafe zip entry detected: {member!r}")
            zf.extractall(path=tmp)

        top_level_files = list(Path(tmp).iterdir())
        if len(top_level_files) == 1 and top_level_files[0].is_dir():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(top_level_files[0], out_path)
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            for file in top_level_files:
                shutil.move(file, out_path / file.name)

    return out_path


def _wrap_on_commas(text: str, budget: int) -> str:
    """Wrap a comma-separated string, breaking only at commas (never mid-item)."""
    items = [s.strip() for s in text.split(",") if s.strip()]
    lines: list[str] = []
    current: list[str] = []
    for item in items:
        if current and len(", ".join([*current, item])) > budget:
            lines.append(", ".join(current) + ",")
            current = [item]
        else:
            current.append(item)
    if current:
        lines.append(", ".join(current))
    return "\n".join(lines)


def wrap_table_column(
    table: PrettyTable, col_index: int, wrap_on_commas: bool = False
) -> None:
    """Wrap a single column of a PrettyTable to fit within the terminal width.

    When *wrap_on_commas* is True, the column is treated as a comma-separated
    list and only broken at commas, so individual items (e.g. chipset names
    with spaces) are never split across lines.
    """
    num_cols = len(table.field_names)
    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    fixed_cols_width = sum(
        max(len(h), max((len(r[i]) for r in table._rows), default=0))
        for i, h in enumerate(table.field_names)
        if i != col_index
    )
    overhead = num_cols * 4 + 1
    # Shrink the flexible column to whatever space remains, but never below the
    # width of its own header. The other columns keep their content intact, so a
    # narrow terminal is honored as closely as possible without truncating
    # headers.
    budget = max(
        term_width - overhead - fixed_cols_width,
        len(table.field_names[col_index]),
    )
    for row in table._rows:
        if wrap_on_commas:
            row[col_index] = _wrap_on_commas(row[col_index], budget)
        else:
            row[col_index] = "\n".join(
                textwrap.wrap(row[col_index], width=budget, break_on_hyphens=False)
            )


def build_table(
    field_names: list[str],
    rows: list[list[str]],
    wrap_column: str,
    title: str | None = None,
    wrap_on_commas: bool = False,
) -> str:
    """Build a left-aligned table, wrapping the named column to the terminal width.

    When *wrap_on_commas* is True, the wrapped column is treated as a
    comma-separated list and only broken at commas.
    """
    table = PrettyTable()
    if title:
        table.title = title
    table.field_names = field_names
    table.align = "l"
    for row in rows:
        table.add_row(row)
    wrap_table_column(
        table, field_names.index(wrap_column), wrap_on_commas=wrap_on_commas
    )
    return str(table)
