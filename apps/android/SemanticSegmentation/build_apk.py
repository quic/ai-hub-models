# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from enum import Enum


class MODELNAME(Enum):
    fcn_resnet50 = 1


def printmenu():
    print("*****************************")
    print("*       TYPE OF MODEL       *")
    print("*****************************")
    for m in MODELNAME:
        print(str(m.value) + ". " + m.name)
    print("*****************************")


## Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--qnnsdk", required=True, help="Give path of QNN SDK")
parser.add_argument("-m", "--model_name", type=str, help="Model Name")
parser.add_argument("-path", "--model_path", type=str, help="TFLITE model file")
args = parser.parse_args()


## FOLDER NAME
folder_name = "app"

##based on this pre-post can be decided
if not args.model_name:
    printmenu()
    inp_model_name = int(input("Please select one: "))
    args.model_name = MODELNAME(inp_model_name).name.lower()


destAsset = os.path.join(".", folder_name, "src", "main", "assets")
if not os.path.exists(destAsset):
    os.makedirs(destAsset)


## MODEL PATH NOT MENTIONED, add information into model_path
if not args.model_path:
    exportstatus = input("Do you want us to download the model from AI hub (y/n)")

    ##DOWNLAOD USING EXPORT.PY
    if exportstatus.lower().startswith("y"):
        print("EXPORT form path")
        subprocess.run(
            [
                "python",
                "-m",
                "qai_hub_models.models." + args.model_name + ".export",
                "--skip-inferencing",
                "--skip-profiling",
            ]
        )
        tflite_file = glob.glob(
            "build" + os.sep + args.model_name + os.sep + "*.tflite", recursive=True
        )
        args.model_path = tflite_file[0]

    ##GET USER TO GIVE PATH
    else:
        args.model_path = input("Give model File as input")

if args.model_path:
    print(args.model_path)
    if not os.path.exists(args.model_path):
        print("PATH DO NOT EXIST: " + args.model_path)
        exit()
    shutil.copy(args.model_path, destAsset + os.sep + "segmentationModel.tflite")


print("Copying resources")
## COPYING REQUIRED FILES FROM QNN SDK
destJNI = os.path.join(".", folder_name, "src", "main", "jniLibs", "arm64-v8a")
if not os.path.exists(destJNI):
    os.makedirs(destJNI)

# copy *.so from $qnn_sdk/libs/aarch64-android to $jni_lib_dir
qnnbasiclibs = os.path.join(args.qnnsdk, "lib", "aarch64-android")
shutil.copytree(qnnbasiclibs, destJNI, dirs_exist_ok=True)

# copy $qnn_sdk/lib/hexagon-v**/unsigned/libQnnHtpV**Skel.so to $jni_lib_dir
skelstubfiles = os.path.join(args.qnnsdk, "lib", "hexagon-v**", "unsigned", "*.so")
for file in glob.glob(skelstubfiles):
    shutil.copy(file, destJNI)

# copy qtld-release.aar to $test_app_root/Application/
destaar = os.path.join(".", folder_name, "libs")
if not os.path.exists(destaar):
    os.makedirs(destaar)
aarfile = os.path.join(args.qnnsdk, "lib", "android", "qtld-release.aar")
shutil.copy(aarfile, destaar)


print("Downloading Open CV")

url = "https://sourceforge.net/projects/opencvlibrary/files/4.5.5/opencv-4.5.5-android-sdk.zip/download"
local_file_path = "opencv-4.5.5-android-sdk.zip"


print("OpenCV is being downloaded, it could take some time.")


def progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent_complete = (downloaded / total_size) * 100
    sys.stdout.write(f"\rDownloading: {percent_complete:.2f}%")
    sys.stdout.flush()


try:
    urllib.request.urlretrieve(url, local_file_path, reporthook=progress_bar)
    print(f"File downloaded successfully to {local_file_path}")
except Exception as e:
    print(f"Error downloading the file: {e}")


opencvzipfile = "opencv-4.5.5-android-sdk.zip"
with zipfile.ZipFile(opencvzipfile, "r") as zip_ref:
    zip_ref.extractall()

os.remove(opencvzipfile)

if os.path.exists("sdk"):
    shutil.rmtree("sdk")

source_directory = "OpenCV-android-sdk/sdk/"
target_directory = "."

shutil.move(source_directory, target_directory)

shutil.rmtree("OpenCV-android-sdk")  # Remove the 'OpenCV-android-sdk' directory


print("BUILDING APK")
## BUILDING APK
if sys.platform.startswith("win"):
    print("Detected platform is windows")
    gradleoutput = subprocess.run(["gradlew.bat", "assembleDebug"], cwd=".")
elif sys.platform.startswith("darwin"):
    print("Detected platform is MAC")
    gradleoutput = subprocess.run(["./gradlew", "assembleDebug"], cwd=".")
else:
    print("Detected platform is Linux")
    gradleoutput = subprocess.run(["./gradlew", "assembleDebug"], cwd=".")


## COPYING APK TO CWD
ApkPath = os.path.join(
    os.getcwd(),
    folder_name,
    "build",
    "outputs",
    "apk",
    "debug",
    folder_name + "-debug.apk",
)
print("APK Is copied at current Working Directory")
shutil.copy(ApkPath, ".")


install_perm = input("Do you want to install this apk in connected device(y/n): ")
## INSTALLING AND RUNNING APK
if install_perm.lower().startswith("y"):
    command_to_install = ["adb", "install", folder_name + "-debug.apk"]
    subprocess.run(command_to_install, cwd=".")
    command_to_run = [
        "adb",
        "shell",
        "am",
        "start",
        "-a",
        "com.example.ACTION_NAME",
        "-n",
        "com.qcom.aihub_segmentation/com.qcom.aihub_segmentation.HomeScreenActivity",
    ]
    subprocess.run(command_to_run, cwd=".")
