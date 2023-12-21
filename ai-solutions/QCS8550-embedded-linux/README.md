## Table of Contents

- [Table of Contents](#table-of-contents)
- [LE Build setup](#le-build-setup)
- [Generating ai-solutions binary](#generating-ai-solutions-binary)
- [Running ai-solutions application](#running-ai-solutions-application)

## LE Build setup

1. Follow "00067.1 Release Note for QCS8550.LE.1.0" to Setup "qti-distro-rb-debug" LE.1.0 build server for QCS8550
2. Make sure "bitbake qti-robotics-image" is successful 
3. Verify the "qti-distro-rb-debug" build by flashing on target using "fastboot". Commands to flash:

    ```
    cd build-qti-distro-rb-debug/tmp-glibc/deploy/images/kalama/qti-robotics-image/
    adb root
    adb reboot bootloader
        
    fastboot flash abl_a abl.elf
    fastboot flash abl_b abl.elf
    fastboot flash dtbo_a dtbo.img
    fastboot flash dtbo_b dtbo.img
    fastboot flash boot_a boot.img
    fastboot flash boot_b boot.img
    fastboot flash system_a system.img
    fastboot flash system_b system.img
    fastboot flash userdata userdata.img
    fastboot flash persist persist.img

    fastboot reboot
    ```

## Generating ai-solutions binary

1. Copy snpe-2.x folder to "<APPS_ROOT>/poky/meta-qti-ml-prop/recipes/snpe-sdk/files/snpe/". 
    ```
    cp -r <SNPE_ROOT>/* <APPS_ROOT>/poky/meta-qti-ml-prop/recipes/snpe-sdk/files/snpe/
    ```
2. Copy "meta-qti-ai-solutions" into "<APPS_ROOT>/poky/" folder
    ```
    cp -r meta-qti-ai-solutions <APPS_ROOT>/poky/
    ```
3. Copy SNPE,DiagLog,DlContainer,DlSystem and Wrapper.hpp
    ```
    cp -r <SNPE_ROOT>/include/SNPE/Dl* <APPS_ROOT>/poky/meta-qti-ai-solutions/recipes/ai-solutions/files/app/inc/
    cp -r <SNPE_ROOT>/include/SNPE/DiagLog/ <APPS_ROOT>/poky/meta-qti-ai-solutions/recipes/ai-solutions/files/app/inc/
    cp -r <SNPE_ROOT>/include/SNPE/Wrapper.hpp <APPS_ROOT>/poky/meta-qti-ai-solutions/recipes/ai-solutions/files/app/inc/
    cp -r <SNPE_ROOT>/include/SNPE/SNPE/ <APPS_ROOT>/poky/meta-qti-ai-solutions/recipes/ai-solutions/files/app/inc/
    ```
4. Update "snpe.bb" in "poky/meta-qti-ml-prop/recipes/snpe-sdk" folder
    1. Make sure platform "aarch64-oe-linux-gcc11.2" is selected
    2. Update DSP lib path
    ```
    -- install -m 0755 ${S}/lib/dsp/* ${D}/${libdir}/rfsa/adsp
    ++ install -m 0755 ${S}/lib/hexagon-v73/unsigned/lib* ${D}/${libdir}/rfsa/adsp
    ```
5. Run the following commands
    ```
    cd <APPS_ROOT>/LE.PRODUCT.2.1.r1/apps_proc/poky
    export MACHINE=kalama DISTRO=qti-distro-rb-debug
    source qti-conf/set_bb_env.sh
    export PREBUILT_SRC_DIR="<APPS_ROOT>/prebuilt_HY11"
    bitbake qti-robotics-image
    ```
6. Flash the latest build on target. (Note: Check if "ai-solutions" binary is generated in the "build-qti-distro-fullstack-debug/tmp-glibc/work/qrb5165_rb5-oe-linux/qti-robotics-image/1.0-r0/rootfs/usr/bin/" path)

## Running ai-solutions application
1. Execute the following commands to remount the target
    ```
    adb root
    adb disable-verity
    adb reboot
    adb root
    adb shell "mount -o remount,rw /"
    ```
2. Push "meta-qti-ai-solutions/recipes/ai-solutions/files/app/" and "SNPE-2.14" onto the device
    ```
    adb push <file> <path_on_target>
    ```
3. Execute the following commands to setup snpe on target
    ```
    adb shell
    cd <SNPE_ROOT>
    cp -r lib/aarch64-oe-linux-gcc11.2/lib* /usr/lib/
    cp bin/aarch64-oe-linux-gcc11.2/snpe-net-run /usr/bin/
    cp -r lib/hexagon-v73/unsigned/lib* /usr/lib/rfsa/adsp/
    chmod +x /usr/bin/snpe-net-run
    snpe-net-run --version
    ```
    Expected output: SNPE v2.14.2.230905160328_61726
4. Run ai-solutions application 
    ```
    adb shell
    cd <app_path>
    ai-solutions -c <path to config.json> -i <path to Input image> -o <path to Output image>
    ```
    Example:

    ```
    ai-solutions -c data/config.json -i Sample1.jpg -o output.jpg
    ```

    ### Details on Input arguments:

    #### Sample config.json

    ```json
    "model-configs":[
    
        "model-name":"QSrnet-medium",                            -> model name which is used while enabling solution
        "model-type":"superresolution",                          -> To specify the use case such superresolution or detection or segmentation etc..
        "model-path":"models/quicksrnet_medium_quantized.dlc",   -> Path at which model is located on target
        "runtime":"DSP",                                         -> Select Runtime either CPU or DSP
        "input-layers":[                                         -> Input layer of the model 
            "t.1"
        ],
        "output-layers":[
            "depth_to_space#1"                                   -> Output layer of the model
        ],
        "output-tensors":[
                "65"                                             -> Output node for post processing
            ]
    ]
    ```

    solution-config:
    ```json
    "solution-configs":[
        {
            "solution-name":"AI-Solutions",                   -> To identify usecase
            "model-name":"SESR",                                 -> Specify model name to be executed
            "input-config-name":"image",                       -> To read input from image
            "Enable":0                                          -> Enable specific solution
        },
        {
            "solution-name":"AI-Solutions",
            "model-name":"SRGAN",
            "input-config-name":"image",
            "Enable":1
        }
    ]
    ```
