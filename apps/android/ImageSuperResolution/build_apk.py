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
from enum import Enum


class MODELNAME(Enum):
    xlsr = 1
    esrgan = 2
    real_esrgan_general_x4v3 = 3
    # real_esrgan_x4plus = 4
    sesr_m5 = 5
    # quicksrnetsmall = 6
    # QuickSRNetMedium = 7
    # QuickSRNetLarge = 8
    # sesr_m5_quantized = 9
    # xlsr_quantized = 10


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


# group = parser.add_mutually_exclusive_group()
# group.add_argument('-stopdownload', '--stopdownload', action = "store_true", help = "Do NOT Download Model from AI HUB")
parser.add_argument("-path", "--model_path", type=str, help="TFLITE model file")

args = parser.parse_args()


##based on this pre-post can be decided
if not args.model_name:
    printmenu()
    inp_model_name = int(input("Please select one: "))
    args.model_name = MODELNAME(inp_model_name).name


destAsset = os.path.join(".", "superresolution", "src", "main", "assets")
if not os.path.exists(destAsset):
    os.makedirs(destAsset)


## MODEL PATH NOT MENTIONED, add information into model_path
if not args.model_path:
    exportstatus = input("Do you want us to download the model from AI hub (y/n)")

    ##DOWNLAOD USING EXPORT.PY
    if exportstatus.lower().startswith("y"):
        print("EXPORT form path")
        pathtomodel = os.path.join(
            "..",
            "..",
            "..",
            "",
            "qai_hub_models",
            "models",
            args.model_name,
            "export.py",
        )
        if not os.path.exists(pathtomodel):
            print("PATH DO NOT EXIST: " + pathtomodel)
            exit()
        subprocess.run(["python", pathtomodel, "--skip-inferencing"])
        tflite_file = glob.glob(
            "build" + os.sep + args.model_name + os.sep + "*.tflite", recursive=True
        )
        args.model_path = tflite_file[0]
        # shutil.copy(tflite_file[0], destAsset+os.sep+"superresmodel.tflite")

    ##GET USER TO GIVE PATH
    else:
        args.model_path = input("Give model File as input")
        # if not os.path.exists(tflite_file):
        # print("PATH DO NOT EXIST: "+tflite_file)
        # exit()
        # shutil.copy(tflite_file, destAsset+os.sep+"superresmodel.tflite")


if args.model_path:
    print(args.model_path)
    if not os.path.exists(args.model_path):
        print("PATH DO NOT EXIST: " + args.model_path)
        exit()
    shutil.copy(args.model_path, destAsset + os.sep + "superresmodel.tflite")


## GETTING PRE/POST PROCESSS BASED ON MODEL NAME
ProPostModel = "XLSR"
AppType = "SUPERRES"

if args.model_name:
    if "esrgan" in args.model_name.lower():
        print("MODEL IS ESRAGAN")
        ProPostModel = "ESRGAN"

with open("build.properties", "w") as f:
    f.write(f"MODELTYPE={ProPostModel}\n")
    f.write(f"APPTYPE={AppType}\n")

## COPYING REQUIRED FILES FROM QNN SDK
destJNI = os.path.join(".", "superresolution", "src", "main", "jniLibs", "arm64-v8a")
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
destaar = os.path.join(".", "superresolution", "libs")
if not os.path.exists(destaar):
    os.makedirs(destaar)
aarfile = os.path.join(args.qnnsdk, "lib", "android", "qtld-release.aar")
shutil.copy(aarfile, destaar)


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
    "superresolution",
    "build",
    "outputs",
    "apk",
    "debug",
    "superresolution-debug.apk",
)
print("APK Is copied at current Working Directory")
shutil.copy(ApkPath, ".")


install_perm = input("Do you want to install this apk in connected device")
## INSTALLING AND RUNNING APK
if install_perm.lower().startswith("y"):
    command_to_install = ["adb", "install", "superresolution-debug.apk"]
    subprocess.run(command_to_install, cwd=".")
    command_to_run = [
        "adb",
        "shell",
        "am",
        "start",
        "-a",
        "com.example.ACTION_NAME",
        "-n",
        "com.qcom.imagesuperres/com.qcom.imagesuperres.QNNActivity",
    ]
    subprocess.run(command_to_run, cwd=".")
