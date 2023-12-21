## Table of Contents

- [Table of Contents](#table-of-contents)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [Prepare Executable - Ubuntu](#prepare-executable---ubuntu)
  * [Prepare stand-alone executable - Ubuntu](#prepare-stand-alone-executable---ubuntu)
  * [Prepare Installer](#prepare-installer)
  * [Installing Package](#installing-package)
  * [Running application](#running-application)
  * [Uninstalling Package](#uninstalling-package)
  * [Supported Platforms](#supported-platforms)

## Environment Setup

<TODO_INTERNAL>Please copy all zips(Assests_ZIP.zip) from @\\upagrah\hyderabad_qidk\shubgoya\WoS_super_res\CPP_SNPE_EXE_multiple_SR_Model\V2

Prepare required assets to build the application 

* sr_dlc: Paste it in pyton_flask_server directory  
* SNPE LIBS: paste at any appropriate location and mention the path in CMakelists.txt
* SNPE INCLUDE: paste at any appropriate location and mention the path in CMakelists.txt
* SNPE_CPP_CODE: paste all files present in this directory to SNPE_CPP_CODE folder of this github repo.

## Dependencies
* Python 3.8 is installed.
* Models used in this solution need to be generated, using steps mentioned at <TODO_INTERNAL> https://github.qualcomm.com/qualcomm-model-zoo-public-mirror/models-for-solutions 


## Directory Structure
This Repo Contains 3 directories, which handle different aspects of this project.

1. Electron app UI: This directory contains UI code and handles the part to start UI and connecting it to flask server. Here, User provides input image, selects the AI model for their use. All this information is sent to python using ajax request.

2. Python Flask Server : Electron UI acts as foreground, and Flask server works in background to handle request from elecron UI. It takes all information given by elecron UI and pre-process the received image here, and then give the processed image to SNPE_CPP_CODE for running the selected model. SNPE_CPP_CODE returns the output of the model and then we process the data given by model into human understandable form and return that back to Electron UI for display.

3. SNPE_CPP_CODE: This works as a service for flask server. This runs the preprocessed image on network and return the output given by model back to Flask Server. 

## Prepare Executable - Ubuntu

### Prepare stand-alone executable - Ubuntu
* Pre-requisites

```bash
apt update
curl -sL https://deb.nodesource.com/setup_18.x -o /tmp/nodesource_setup.sh
bash /tmp/nodesource_setup.sh
apt install nodejs cmake libzmq3-dev python3-pip ruby-dev build-essential
gem i fpm -f
apt upgrade
```

* In python_flask_server:
  - Python pkg dependencies : 
```bash
pip install empatches flask opencv-python pillow icecream flask_cors zmq pyinstaller numpy==1.24.3
```
  - Create sr_dlc Directory and put DLC(s) there, Please follow relevant section for generating DLC.  <--TODO.
  - To start flask server, please run 
```bash
python server.py
```
  - It will start server at port : 9081
  - To view webpage on browser, please use this URL : http://localhost:9081

* In SNPE_CPP_Code:
  - Apply zmq_support.patch to the SNPE SampleCode present in SNPE sdk. 
  - After that please copy all the files in the examples/SNPE/NativeCpp/SampleCode/jni/ folder to SNPE_CPP_CODE folder in this github repo.
  
  - Change following paths in CmakeLists.txt of SNPE_CPP_Code according to your setup:
  ```bash
  set (SNPE_ROOT "/opt/qcom/aistack/snpe/2.12.0.230626/")
  set (SNPE_INCLUDE_DIR "${SNPE_ROOT}/include/SNPE")
  set (SNPE_LIB_PREFIX "${SNPE_ROOT}/lib")
  set (_dtuple_POSTFIX ubuntu-gcc7.5)
  ```
  - Create a build folder and build files.
  ```bash
  mkdir build
  cd build
  mkdir Release
  cmake --build ./
  ```

  - For running, please go to build/Release folder and run snpe-sample
 
 * In electron_app_ui:
   - Execute 
   ```bash
   npm install
   ```
   This will make node modules directory which will contain all necessary npm packages.
   - To start UI, please run : 
   ```bash 
   npm start
   ```
   - If you face any issues with,
   ```bash 
   npm start
   ```
   execute 
   ```bash 
   node package_snpe_cpp.js && node package_python.js
   ```
   before running 
   ```bash
   npm start
   ```

### Prepare Installer

Please execute following commands. These will generate "dist" directory which will contain all your packaged data.
```bash
npm install
export USE_SYSTEM_FPM=true
npm run package
```

### Installing Package
```bash
dpkg -i <package>.deb
```

### Running application
- Run the following commands to render UI on Host
- Open "Command Prompt" on host machine

```bash
setx  DISPLAY 127.0.0.1:0.0
ssh -X root@target_ip   NOTE:(default password:- oelinux123)
echo "X11Forwarding yes" >> /etc/ssh/sshd_config
export DISPLAY="localhost:10.0"
ai-solutions --no-sandbox
```

### Uninstalling Package
```bash
dpkg --remove <package name>
```

Note: Make sure that you have resolved all dependencies mentioned in [Prepare Installer](#prepare-installer) section, like setting SNPE and ZMQ libs, installing python packages etc.

### Supported Platforms

This solution is verified on following IOT platforms

- QRB5165
