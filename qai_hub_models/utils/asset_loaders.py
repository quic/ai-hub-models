# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import fileinput
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from zipfile import ZipFile

import gdown
import numpy as np
import requests
import torch
import yaml
from git import Repo
from PIL import Image
from schema import And, Schema, SchemaError

ASSET_BASES_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "asset_bases.yaml"
)

QAIHM_STORE_ROOT = os.environ.get("QAIHM_STORE_ROOT", os.path.expanduser("~"))
LOCAL_STORE_DEFAULT_PATH = os.path.join(QAIHM_STORE_ROOT, ".qaihm")

SOURCE_AS_ROOT_LOCK = threading.Lock()

VersionType = Union[str, int]

# If non-None, always enter this for yes (True)/no (False) prompts
_always_answer = None


@contextmanager
def always_answer_prompts(answer):
    global _always_answer
    old_value = _always_answer
    _always_answer = answer
    try:
        yield
    finally:
        _always_answer = old_value


class QAIHM_WEB_ASSET(Enum):
    STATIC_IMG = 0
    ANIMATED_MOV = 1


class ModelZooAssetConfig:
    def __init__(
        self,
        asset_url: str,
        web_asset_folder: str,
        static_web_banner_filename: str,
        animated_web_banner_filename: str,
        model_asset_folder: str,
        dataset_asset_folder: str,
        local_store_path: str,
        qaihm_repo: str,
        example_use: str,
        huggingface_path: str,
        repo_url: str,
        models_website_url: str,
        models_website_relative_path: str,
    ) -> None:
        self.local_store_path = local_store_path
        self.asset_url = asset_url
        self.web_asset_folder = web_asset_folder
        self.static_web_banner_filename = static_web_banner_filename
        self.animated_web_banner_filename = animated_web_banner_filename
        self.model_asset_folder = model_asset_folder
        self.dataset_asset_folder = dataset_asset_folder
        self.qaihm_repo = qaihm_repo
        self.example_use = example_use
        self.huggingface_path = huggingface_path
        self.repo_url = repo_url
        self.models_website_url = models_website_url
        self.models_website_relative_path = models_website_relative_path

        # Validation
        for name in [
            self.asset_url,
            self.web_asset_folder,
            self.model_asset_folder,
            self.static_web_banner_filename,
            self.animated_web_banner_filename,
            self.local_store_path,
            self.qaihm_repo,
            self.example_use,
            self.huggingface_path,
            self.models_website_relative_path,
        ]:
            assert not name.endswith("/") and not name.endswith("\\")
        for name in [
            self.static_web_banner_filename,
            self.animated_web_banner_filename,
        ]:
            assert not name.startswith("/") and not name.startswith("\\")

        for name in [self.repo_url, self.models_website_url]:
            assert not name.endswith("/"), "URLs should not end with a slash"

    def get_hugging_face_url(self, model_name: str) -> str:
        return f"https://huggingface.co/{self.get_huggingface_path(model_name)}"

    def get_huggingface_path(self, model_name: str) -> str:
        return self.huggingface_path.replace("{model_name}", str(model_name))

    def get_web_asset_url(self, model_id: str, type: QAIHM_WEB_ASSET):
        if type == QAIHM_WEB_ASSET.STATIC_IMG:
            file = self.static_web_banner_filename
        elif type == QAIHM_WEB_ASSET.ANIMATED_MOV:
            file = self.animated_web_banner_filename
        else:
            raise NotImplementedError("unsupported web asset type")
        return f"{self.asset_url}/{ModelZooAssetConfig._replace_path_keywords(self.web_asset_folder, model_id=model_id)}/{file}"

    def get_local_store_model_path(
        self, model_name: str, version: VersionType, filename: str
    ) -> str:
        model_dir = os.path.join(
            self.local_store_path,
            self.get_relative_model_asset_path(model_name, version, filename),
        )
        return model_dir

    def get_local_store_dataset_path(
        self, dataset_name: str, version: VersionType, filename: str
    ) -> str:
        model_dir = os.path.join(
            self.local_store_path,
            self.get_relative_dataset_asset_path(dataset_name, version, filename),
        )
        return model_dir

    def get_relative_model_asset_path(
        self, model_id: str, version: Union[int, str], file_name: str
    ):
        assert not file_name.startswith("/") and not file_name.startswith("\\")
        return f"{ModelZooAssetConfig._replace_path_keywords(self.model_asset_folder, model_id=model_id, version=version)}/{file_name}"

    def get_relative_dataset_asset_path(
        self, dataset_id: str, version: Union[int, str], file_name: str
    ):
        assert not file_name.startswith("/") and not file_name.startswith("\\")
        return f"{ModelZooAssetConfig._replace_path_keywords(self.dataset_asset_folder, dataset_id=dataset_id, version=version)}/{file_name}"

    def get_model_asset_url(
        self, model_id: str, version: Union[int, str], file_name: str
    ):
        assert not file_name.startswith("/") and not file_name.startswith("\\")
        return f"{self.asset_url}/{self.get_relative_model_asset_path(model_id, version, file_name)}"

    def get_dataset_asset_url(
        self, dataset_id: str, version: Union[int, str], file_name: str
    ):
        assert not file_name.startswith("/") and not file_name.startswith("\\")
        return f"{self.asset_url}/{self.get_relative_dataset_asset_path(dataset_id, version, file_name)}"

    def get_qaihm_repo(self, model_id: str, relative=True):
        relative_path = f"{ModelZooAssetConfig._replace_path_keywords(self.qaihm_repo, model_id=model_id)}"
        if not relative:
            return self.repo_url + "/" + relative_path

        return relative_path

    def get_website_url(self, model_id: str, relative=False):
        relative_path = f"{ModelZooAssetConfig._replace_path_keywords(self.models_website_relative_path, model_id=model_id)}"
        if not relative:
            return self.models_website_url + "/" + relative_path
        return relative_path

    def get_example_use(self, model_id: str):
        return f"{ModelZooAssetConfig._replace_path_keywords(self.example_use, model_id=model_id)}"

    ###
    # Helpers
    ###
    @staticmethod
    def _replace_path_keywords(
        path: str,
        model_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        version: Optional[Union[int, str]] = None,
    ):
        if model_id:
            path = path.replace("{model_id}", model_id)
        if dataset_id:
            path = path.replace("{dataset_id}", dataset_id)
        if version:
            path = path.replace("{version}", str(version))
        return path

    ###
    # Load from CFG
    ###
    @staticmethod
    def from_cfg(
        asset_cfg_path: str = ASSET_BASES_DEFAULT_PATH,
        local_store_path: str = LOCAL_STORE_DEFAULT_PATH,
        verify_env_has_all_variables: bool = False,
    ):
        # Load CFG and params
        asset_cfg = ModelZooAssetConfig.load_asset_cfg(
            asset_cfg_path, verify_env_has_all_variables
        )

        return ModelZooAssetConfig(
            asset_cfg["store_url"],
            asset_cfg["web_asset_folder"],
            asset_cfg["static_web_banner_filename"],
            asset_cfg["animated_web_banner_filename"],
            asset_cfg["model_asset_folder"],
            asset_cfg["dataset_asset_folder"],
            local_store_path,
            asset_cfg["qaihm_repo"],
            asset_cfg["example_use"],
            asset_cfg["huggingface_path"],
            asset_cfg["repo_url"],
            asset_cfg["models_website_url"],
            asset_cfg["models_website_relative_path"],
        )

    ASSET_CFG_SCHEMA = Schema(
        And(
            {
                "store_url": str,
                "web_asset_folder": str,
                "dataset_asset_folder": str,
                "static_web_banner_filename": str,
                "animated_web_banner_filename": str,
                "model_asset_folder": str,
                "qaihm_repo": str,
                "example_use": str,
                "huggingface_path": str,
                "repo_url": str,
                "models_website_url": str,
                "models_website_relative_path": str,
            }
        )
    )

    @staticmethod
    def load_asset_cfg(path, verify_env_has_all_variables: bool = False):
        with open(path) as f:
            data = yaml.safe_load(f)
            try:
                # Validate high level-schema
                ModelZooAssetConfig.ASSET_CFG_SCHEMA.validate(data)
            except SchemaError as e:
                assert 0, f"{e.code} in {path}"

            for key, value in data.items():
                # Environment variable replacement
                if isinstance(value, str) and value.startswith("env::"):
                    values = value.split("::")
                    if len(values) == 2:
                        _, env_var_name = values
                        default = value
                    elif len(values) == 3:
                        _, env_var_name, default = values
                    else:
                        raise NotImplementedError(
                            "Environment vars should be specified in asset_bases "
                            "using format env::<var_name>::<default>"
                        )

                    data[key] = os.environ.get(env_var_name, default)
                    if (
                        verify_env_has_all_variables
                        and default == value
                        and env_var_name not in os.environ
                    ):
                        raise ValueError(
                            f"Environment variable '{env_var_name}' was specified in "
                            f"asset_bases.yaml for key '{key}', but is not defined."
                        )

            return data


ASSET_CONFIG = ModelZooAssetConfig.from_cfg()


def _query_yes_no(question, default="yes"):
    """
    Ask a yes/no question and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Sourced from https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    global _always_answer
    if _always_answer is not None:
        return _always_answer

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        print(question + prompt, end="")
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def maybe_clone_git_repo(
    git_file_path: str,
    commit_hash,
    model_name: str,
    model_version: VersionType,
    patches: List[str] = [],
) -> str:
    """Clone (or pull) a repository, save it to disk in a standard location,
    and return the absolute path to the cloned location. Patches can be applied
    by providing a list of paths to diff files."""

    # http://blah.come/author/name.git -> name, author
    repo_name = os.path.basename(git_file_path).split(".")[0]
    repo_author = os.path.basename(os.path.dirname(git_file_path))
    local_path = ASSET_CONFIG.get_local_store_model_path(
        model_name, model_version, f"{repo_author}_{repo_name}_git"
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(os.path.join(local_path, ".git")):
        # Clone repo
        should_clone = _query_yes_no(
            f"{model_name} requires repository {git_file_path} . Ok to clone?",
        )
        if should_clone:
            print(f"Cloning {git_file_path} to {local_path}...")
            repo = Repo.clone_from(git_file_path, local_path)
            repo.git.checkout(commit_hash)
            for patch in patches:
                repo.git.execute(["git", "apply", patch])
            print("Done")
        else:
            raise ValueError(
                f"Unable to load {model_name} without its required repository."
            )

    return local_path


def _load_file(
    file: PathType,
    loader_func: Callable[[str], Any],
    dst_folder_path: tempfile.TemporaryDirectory | str | None = None,
) -> Any:
    if isinstance(file, (str, Path)):
        file = str(file)
        if file.startswith("http"):
            if dst_folder_path is None:
                dst_folder_path = tempfile.TemporaryDirectory()
            if isinstance(dst_folder_path, tempfile.TemporaryDirectory):
                dst_folder_path_str = dst_folder_path.name
            else:
                dst_folder_path_str = dst_folder_path
            dst_path = os.path.join(dst_folder_path_str, os.path.basename(file))
            download_file(file, dst_path)
            return loader_func(dst_path)
        else:
            return loader_func(file)
    elif isinstance(file, CachedWebAsset):
        return loader_func(str(file.fetch()))
    else:
        raise NotImplementedError()


def load_image(image: PathType, verbose=False, desc="image") -> Image.Image:
    if verbose:
        print(f"Loading {desc} from {image}")
    return _load_file(image, Image.open)


def load_numpy(file: PathType) -> Any:
    return _load_file(file, np.load)


def load_torch(pt: PathType) -> Any:
    return _load_file(pt, partial(torch.load, map_location="cpu"))


def load_json(json_file: PathType) -> Dict:
    def _load_json_helper(file_path) -> Any:
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    return _load_file(json_file, _load_json_helper)


def load_path(file: PathType, tmpdir: tempfile.TemporaryDirectory | str) -> str | Path:
    """
    Get asset path on disk.
    If `file` is a string URL, downloads the file to tmpdir.name.
    """

    def return_path(path):
        return path

    return _load_file(file, return_path, tmpdir)


@contextmanager
def SourceAsRoot(
    source_repo_url: str,
    source_repo_commit_hash: str,
    source_repo_name: str,
    source_repo_version: int | str,
    source_repo_patches: List[str] = [],
    keep_sys_path: bool = False,
):
    """
    Context manager that runs code with:
     * the source repository added to the system path,
     * cwd set to the source repo's root directory.

    Only one of this class should be active per Python session.
    """

    repository_path = maybe_clone_git_repo(
        source_repo_url,
        source_repo_commit_hash,
        source_repo_name,
        source_repo_version,
        patches=source_repo_patches,
    )
    SOURCE_AS_ROOT_LOCK.acquire()
    cwd = os.getcwd()
    original_path = list(sys.path)
    try:
        # Patch path for this load only, since the model source
        # code references modules via a global scope.
        # Insert with highest priority (see #7666)
        sys.path.insert(0, repository_path)
        os.chdir(repository_path)

        yield repository_path
    finally:
        # Be careful editing these lines (failure means partial clean-up)
        os.chdir(cwd)
        if not keep_sys_path:
            sys.path = original_path
        SOURCE_AS_ROOT_LOCK.release()


def find_replace_in_repo(
    repo_path: str, filepaths: Union[str, List[str]], find_str: str, replace_str: str
):
    """
    When loading models from external repos, sometimes small modifications
    need to be made to the repo code to get it working in the zoo env.

    This does a simple find + replace within a single file.

    Parameters:
        repo_path: Local filepath to the repo of interest.
        filepath: Filepath within the repo to the file to change.
        find_str: The string that needs to be replaced.
        replace_str: The string with which to replace all instances of `find_str`.
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    for filepath in filepaths:
        with fileinput.FileInput(
            Path(repo_path) / filepath,
            inplace=True,
            backup=".bak",
        ) as file:
            for line in file:
                print(line.replace(find_str, replace_str), end="")


class CachedWebAsset:
    """
    Helper class for downloading files for storage in the QAIHM asset cache.
    """

    def __init__(
        self,
        url: str,
        local_cache_path: str,
        asset_config=ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries=4,
    ):
        self.url = url
        self.local_cache_path = local_cache_path
        self.asset_config: ModelZooAssetConfig = asset_config
        self._downloader: Callable = model_downloader or download_file
        self.downloader_num_retries = downloader_num_retries

        # Append file name to local path if no file name is present
        path, ext = os.path.splitext(self.local_cache_path)
        if not ext:
            file_name = self.url.rsplit("/", 1)[-1]
            self.local_cache_path = os.path.join(path, file_name)

        # Set is_extracted if already extracted on disk
        file, _ = os.path.splitext(self.local_cache_path)
        self.is_extracted = list(
            filter(local_cache_path.endswith, [".zip", ".tar", ".tar.gz", ".tgz"])
        ) != [] and os.path.isdir(file)

    def __repr__(self):
        return self.url

    @staticmethod
    def from_asset_store(
        relative_store_file_path: str, num_retries=4, asset_config=ASSET_CONFIG
    ):
        """
        File from the online qaihm asset store.

        Parameters:
            relative_store_file_path: Path relative to `qai_hub_models` cache root to store this asset.
                                      (also relative to the root of the online file store)

            num_retries: Number of retries when downloading thie file.

            asset_config: Asset config to use to save this file.
        """
        web_store_path = f"{asset_config.asset_url}/{relative_store_file_path}"
        return CachedWebAsset(
            web_store_path,
            relative_store_file_path,
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod
    def from_google_drive(
        gdrive_file_id: str,
        relative_store_file_path: str,
        num_retries=4,
        asset_config=ASSET_CONFIG,
    ):
        """
        File from google drive.

        Parameters:
            gdrive_file_id: Unique identifier of the file in Google Drive.
                Typically found in the URL.

            relative_store_file_path: Path relative to `qai_hub_models` cache root to store this asset.

            num_retries: Number of retries when downloading thie file.

            asset_config: Asset config to use to save this file.
        """
        return CachedWebAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            relative_store_file_path,
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )

    def path(self, extracted=None) -> Path:
        """
        Get the path of this asset on disk.

        By default, for archived (.zip, .tar, .etc) assets, path() will return the extracted path if the asset
        has been extracted, and the original archive file's path if it has not been extracted.

        Parameters:
            extracted: If true, return the path of the extracted asset on disk.
                       If false, return the path of the archive path on disk.
        """
        if (extracted is None and self.is_extracted) or extracted:
            file, _ = os.path.splitext(self.local_cache_path)
        else:
            file = self.local_cache_path

        return Path(self.asset_config.local_store_path) / file

    def fetch(self, force=False, extract=False) -> Path:
        """
        Fetch this file from the web if it does not exist on disk.

        Parameters:
            force: If the file exists on disk already, discard it and download it again.

            extract: Extract the asset after downloading it.
        """
        path = self.path()

        # Delete existing asset if requested
        if path.exists():
            if force:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                self.is_extracted = False
            else:
                return path
        elif self.is_extracted:
            # Someone deleted the extracted path. Fetch it again.
            self.is_extracted = False
            path = self.path()

        # Create dirs
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Downloader should return path we expect.
        p1 = self._downloader(self.url, self.local_cache_path)
        assert str(p1) == str(path)

        # Extract asset if requested
        if extract:
            self.extract(force)

        return self.path()

    def extract(self, force=True) -> Path:
        """
        Extract this asset if it is compressed. Updates the path of this asset to the folder to which the zip file was extracted.
        """
        if self.is_extracted:
            if force:
                os.remove(self.path())
                self.is_extracted = False
            else:
                return self.path()

        _, ext = os.path.splitext(self.local_cache_path)
        if ext == ".zip":
            # Update local cache path to pont to the extracted zip folder.
            extract_zip_file(str(self.path()))
            os.remove(self.path())  # Deletes zip file
            self.is_extracted = True  # Updates path() to return extracted path
        elif ext in [".tar", ".gz", ".tgz"]:
            with tarfile.open(self.path()) as f:
                f.extractall(os.path.dirname(self.path()))
            os.remove(self.path())  # Deletes tar file
            self.is_extracted = True  # Updates path() to return extracted path
        else:
            raise ValueError(f"Unsupported compressed file type: {ext}")

        return self.path()


class CachedWebModelAsset(CachedWebAsset):
    """
    Helper class for downloading files for storage in the QAIHM asset cache.
    """

    def __init__(
        self,
        url: str,
        model_id: str,
        model_asset_version: int | str,
        filename: str,
        asset_config=ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries=4,
    ):
        local_cache_path = asset_config.get_local_store_model_path(
            model_id, model_asset_version, filename
        )
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            model_downloader,
            downloader_num_retries,
        )
        self.model_id = model_id
        self.model_version = model_asset_version

    @staticmethod  # type: ignore
    def from_asset_store(
        model_id: str,
        model_asset_version: str | int,
        filename: str,
        num_retries=4,
        asset_config=ASSET_CONFIG,
    ):
        """
        File from the online qaihm asset store.

        Parameters:
            model_id: str
                Model ID

            model_asset_version: str | int
                Asset version for this model.

            num_retries: int
                Number of retries when downloading thie file.

            asset_config: ModelZooAssetConfig
                Asset config to use to save this file.
        """
        web_store_path = asset_config.get_model_asset_url(
            model_id, model_asset_version, filename
        )
        return CachedWebModelAsset(
            web_store_path,
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod  # type: ignore
    def from_google_drive(
        gdrive_file_id: str,
        model_id: str,
        model_asset_version: str | int,
        filename: str,
        num_retries=4,
        asset_config=ASSET_CONFIG,
    ):
        """
        File from google drive.

        Parameters:
            gdrive_file_id: Unique identifier of the file in Google Drive.
                Typically found in the URL.

            model_id: Model ID

            model_asset_version: Asset version for this model.

            filename: Filename for this asset on disk.

            num_retries: Number of retries when downloading thie file.

            asset_config: Asset config to use to save this file.
        """
        return CachedWebModelAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )


class CachedWebDatasetAsset(CachedWebAsset):
    """
    Class representing dataset-specific files that needs stored in the local cache once downloaded.

    These files should correspond to a single (or group) of datasets in `qai_hub_models/dataset`.
    """

    def __init__(
        self,
        url: str,
        dataset_id: str,
        dataset_version: int | str,
        filename: str,
        asset_config=ASSET_CONFIG,
        model_downloader: Callable[[str, str, int], str] | None = None,
        downloader_num_retries=4,
    ):
        local_cache_path = asset_config.get_local_store_dataset_path(
            dataset_id, dataset_version, filename
        )
        super().__init__(
            url,
            local_cache_path,
            asset_config,
            model_downloader,
            downloader_num_retries,
        )
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version

    @staticmethod  # type: ignore
    def from_asset_store(
        dataset_id: str,
        dataset_version: str | int,
        filename: str,
        num_retries=4,
        asset_config=ASSET_CONFIG,
    ):
        """
        File from the online qaihm asset store.

        Parameters:
            model_id: Model ID

            dataset_version: Asset version for this model.

            num_retries: Number of retries when downloading thie file.

            asset_config: Asset config to use to save this file.
        """
        web_store_path = asset_config.get_dataset_asset_url(
            dataset_id, dataset_version, filename
        )
        return CachedWebModelAsset(
            web_store_path,
            dataset_id,
            dataset_version,
            filename,
            asset_config,
            download_file,
            num_retries,
        )

    @staticmethod  # type: ignore
    def from_google_drive(
        gdrive_file_id: str,
        model_id: str,
        model_asset_version: str | int,
        filename: str,
        num_retries=4,
        asset_config=ASSET_CONFIG,
    ):
        """
        File from google drive.

        Parameters:
            gdrive_file_id: Unique identifier of the file in Google Drive.
                Typically found in the URL.

            model_id: Model ID

            model_asset_version: Asset version for this model.

            filename: Filename for this asset on disk.

            num_retries: Number of retries when downloading thie file.

            asset_config: Asset config to use to save this file.
        """
        return CachedWebModelAsset(
            f"https://drive.google.com/uc?id={gdrive_file_id}",
            model_id,
            model_asset_version,
            filename,
            asset_config,
            download_and_cache_google_drive,
            num_retries,
        )


def download_file(web_url: str, dst_path: str, num_retries: int = 4) -> str:
    """
    Downloads data from the internet and stores in `dst_folder`.
    `dst_folder` should be relative to the local cache root for qai_hub_models.
    """
    if not os.path.exists(dst_path):
        print(f"Downloading data at {web_url} to {dst_path}... ", end="")
        file_data = requests.get(web_url)
        if file_data.status_code != 200:
            raise ValueError(f"Unable to download file at {web_url}")
        with open(dst_path, "wb") as dst_file:
            dst_file.write(file_data.content)
        print("Done")
    return dst_path


def download_and_cache_google_drive(web_url: str, dst_path: str, num_retries: int = 4):
    """
    Download file from google drive to the local directory.

    Parameters:
        file_id: Unique identifier of the file in Google Drive.
            Typically found in the URL.
        model_name: Model for which this asset is being downloaded.
            Used to choose where in the local filesystem to put it.
        filename: Filename under which it will be saved locally.
        num_retries: Number of times to retry in case download fails.

    Returns:
        Filepath within the local filesystem.
    """
    for i in range(num_retries):
        print(f"Downloading data at {web_url} to {dst_path}... ")
        try:
            gdown.download(web_url, dst_path, quiet=False)
        except Exception:
            pass
        if os.path.exists(dst_path):
            print("Done")
            return dst_path
        else:
            print(f"Failed to download file at {web_url}")
            if i < num_retries - 1:
                print("Retrying in 3 seconds.")
                time.sleep(3)
    return dst_path


def copyfile(src: str, dst: str, num_retries: int = 4):
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copyfile(src, dst)
    return dst


def extract_zip_file(filepath_str: str) -> Path:
    """
    Given a local filepath to a zip file, extract its contents into a folder
    in the same directory. The directory with the contents will have the same
    name as the .zip file without the `.zip` extention.

    Parameters:
        filepath_str: String of the path to the zip file in the local directory.
    """
    filepath = Path(filepath_str)
    with ZipFile(filepath, "r") as zf:
        out_path = filepath.parent / filepath.stem
        zf.extractall(path=out_path)
    return out_path


def callback_with_retry(
    num_retries: int,
    callback: Callable,
    *args: Optional[Any],
    **kwargs: Optional[Any],
) -> Any:
    """Allow retries when running provided function."""
    if num_retries == 0:
        raise RuntimeError(f"Unable to run function {callback.__name__}")
    else:
        try:
            return callback(*args, **kwargs)
        except Exception as error:
            error_msg = (
                f"Error: {error.message}"  # type: ignore
                if hasattr(error, "message")
                else f"Error: {str(error)}"
            )
            print(error_msg)
            if hasattr(error, "status_code"):
                print(f"Status code: {error.status_code}")  # type: ignore
            time.sleep(10)
            return callback_with_retry(num_retries - 1, callback, *args, **kwargs)


PathType = Union[str, Path, CachedWebAsset]
