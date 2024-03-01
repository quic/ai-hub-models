## Table of Contents

- [Table of Contents](#table-of-contents)
- [LE Build setup](#le-build-setup)
- [Generating ai-solutions binary](#generating-ai-solutions-binary)
- [Running ai-solutions application](#running-ai-solutions-application)
  * [Details on Input arguments:](#details-on-input-arguments)
    + [Sample config.json](#sample-configjson)

## LE Build setup

1. Follow "00023.4 Release Note for QRB5165.LE.1.0" to Setup "qti-distro-fullstack-debug" LE.1.0 build server for QRB5165
2. Make sure "bitbake qti-robotics-image" is successful  
3. Verify the "qti-distro-fullstack-debug" build by flashing on target using "QFIL"

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
    1. Make sure platform "aarch64-oe-linux-gcc9.3" is selected
    2. Update DSP lib path
    ```
    -- install -m 0755 ${S}/lib/dsp/* ${D}/${libdir}/rfsa/adsp
    ++ install -m 0755 ${S}/lib/hexagon-v66/unsigned/lib* ${D}/${libdir}/rfsa/adsp
    ```
5. Run the following commands
    ```bash
    cd <APPS_ROOT>/poky
    export MACHINE=qrb5165-rb5 DISTRO=qti-distro-fullstack-debug
    source qti-conf/set_bb_env.sh
    export PREBUILT_SRC_DIR="<APPS_ROOT>/prebuilt_HY11"
    bitbake qti-robotics-image
    ```
6. Flash the latest build on target. (Note: Check if "ai-solutions" binary is generated in the "build-qti-distro-fullstack-debug/tmp-glibc/work/qrb5165_rb5-oe-linux/qti-robotics-image/1.0-r0/rootfs/usr/bin/" path)

## Running ai-solutions application
1. Execute the following commands to remount the target
    
    ```bash
    adb root
    adb disable-verity
    adb reboot
    adb root
    adb shell "mount -o remount,rw /"
    ```
2. Push "meta-qti-ai-solutions/recipes/ai-solutions/files/app/" and "SNPE-2.12" onto the device
    ```bash
    adb push <file> <path_on_target>
    ```
3. Execute the following commands to setup snpe on target
    ```bash
    adb shell
    cd <SNPE_ROOT>
    cp -r lib/aarch64-oe-linux-gcc9.3/* /usr/lib/
    cp bin/aarch64-oe-linux-gcc9.3/snpe-net-run /usr/bin/
    cp -r lib/hexagon-v66/unsigned/lib* /usr/lib/rfsa/adsp/
    chmod +x /usr/bin/snpe-net-run
    snpe-net-run --version
    ```
    Expected output: SNPE v2.12.0.230626174329_59328
4. Run ai-solutions application 
    ```
    adb shell
    cd <app_path>

    ```bash
    export XDG_RUNTIME_DIR=/run/user/root
    ```
    #### To run inference on input image
    NOTE: Make sure "input-config-name":"image" in data/config.json
    ```bash
    ./out/ai-solutions -c ../data/config.json -i Sample1.jpg -o output.jpg
    ```
    #### To run inference on camera stream
    NOTE: Make sure "input-config-name":"camera" in data/config.json
    ```bash
    ./out/ai-solutions -c ../data/config.json
    ```

    ### Details on Input arguments:

    #### Sample config.json
    model-config:
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
            "solution-name":"AI-Solutions",                      -> To identify usecase
            "model-name":"mobilenet-ssd",                        -> Specify model name to be executed
            "input-config-name":"camera",                        -> To read input from camera stream
            "Enable":1,                                          -> Enable specific solution
            "output-type":"wayland"                              -> To display output on monitor
        },
        {
            "solution-name":"AI-Solutions",
            "model-name":"SRGAN",
            "input-config-name":"image",                        -> To read input from image
            "Enable":0,
            "output-type":"wayland"
        }
    ]
    ```