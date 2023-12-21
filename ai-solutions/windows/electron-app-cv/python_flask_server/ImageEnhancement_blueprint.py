# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
from flask import Blueprint
from flask import request, jsonify, make_response, send_file
from PIL import Image
from empatches import EMPatches
import io, os
import cv2
import numpy as np
import time
import functools
import zmq
# from utils import pyinstaller_absolute_path

import globalvar
time_taken_model = ""
upscaled_img_dims = ""
# old_runtime = ""
# old_model_name = ""

imageEnhance_bp = Blueprint("ImageEnhance",__name__)

runtime_name_decoder={'DSP':b"DSP",'GPU':b"GPU", 'CPU':b"CPU"}
# dlc_name_decoder={'EnhancementGAN':'quant_enhancement_240_320_8350.dlc', 'MBLLEN':'quant_mbllen_214.dlc', 'RUAS':'quant_ruas_214.dlc','SCI':'quant_sci_214.dlc','StableLLVE':'quant_StableLLVE_214.dlc','Zero-DCE':'quant_zerodce_80_214.dlc','Zero-DCE++':'quant_zerodce++_214.dlc'}
# dlc_name_decoder={'MBLLEN':'quant_mbllen_214.dlc', 'RUAS':'quant_ruas_214.dlc','SCI':'quant_sci_214.dlc','StableLLVE':'quant_StableLLVE_214.dlc','Zero-DCE':'quant_zerodce_80_214.dlc'}
dlc_name_decoder={'MBLLEN':'quant_mbllen_480_640_8350_212.dlc', 'RUAS':'quant_ruas_480_640_8350_212.dlc','SCI':'quant_sci_480_640_8350_212.dlc','StableLLVE':'quant_stablellve_480_640_8350_212.dlc','Zero-DCE':'quant_zerodce_480_640_212_8350_80_out.dlc'}
    

@imageEnhance_bp.route('/image_enhancement/ie_checkdlc', methods=['POST'])
def checkdlc():
    print("checkdlc: ")
    from flask import jsonify
    import os
    model_name = request.form.get('model_name')
    
    print("MODEL NAME IN CHECKDLC: ", model_name)
    dlc_path = os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","enhancement", dlc_name_decoder.get(model_name))
    if(os.path.isfile(dlc_path)):
        print("found")
        output_new = {
                "dlc_available": "yes",
                "dlc_path" : dlc_path
            }
    else:
        print("not found")
        output_new = {
                "dlc_available": "no",
                "dlc_path" : dlc_path
            }
    return jsonify(output_new), 200


def buildnetwork_ie(socket, model_name, run_time):

    print("BUILDING NETWORK Low light")
    print("Model name: ",model_name)
    first_str = b"networkbuild"
    
    dlc_path = bytes(os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","enhancement", dlc_name_decoder.get(model_name)),'utf-8')
    
    socket.send_multipart([first_str,dlc_path, runtime_name_decoder.get(run_time)])   

    print("Messages sent for building network, waiting for reply")
    message_build = socket.recv()
    print(message_build)


def runmodel_ie(socket, patch, model_name, run_time, scaling_factor=4 ):
    
    try:
        print("LOW LIGHT MODEL")
        
        ## PREPROC ##
        if model_name=='MBLLEN':
            patch = cv2.resize(patch, (640,480))
            patch = patch/255
        elif model_name=='RUAS':
            patch = cv2.resize(patch, (640,480))
            patch = patch/255
        elif model_name=='SCI':
            patch = cv2.resize(patch, (640,480))
            patch = patch/255
        elif model_name=='StableLLVE':
            patch = cv2.resize(patch, (640,480))
            patch = patch/255
        elif model_name=='Zero-DCE':
            patch = cv2.resize(patch, (640,480))
            patch = patch/255
        else:
            print("Out of Context: Model Specified is wrong")
        
        
        img = np.array(patch)
        img = img.astype(np.float32)
        img = img.tobytes()

        print("Preproc done")

        socket.send_multipart([b"infer",img])

        print("Messages Image sent, waiting for reply")
        message_img_out = socket.recv()

        prediction = np.frombuffer(message_img_out, dtype=np.float32)
        print("Message received from server :: Shape: ", prediction.shape) #," data: ", prediction)

        socket.send(b"get_infer_time")
        message_infer_time = socket.recv()
        print("message_infer_time", message_infer_time.decode('UTF-8'))
        elapsed_time = 0.0
        elapsed_time = float(message_infer_time.decode('UTF-8'))/1000

        print("post start")
        
        if model_name=='MBLLEN':
            prediction = prediction.reshape(480,640,3)
            prediction = prediction*255
        elif model_name=='RUAS':
            prediction = prediction.reshape(480,640,3)
            prediction = prediction*255
        elif model_name=='SCI':
            prediction = prediction.reshape(480,640,3)
            prediction = prediction*255
        elif model_name=='StableLLVE':
            prediction = prediction.reshape(480,640,3)
            prediction = prediction*255
        elif model_name=='Zero-DCE':
            prediction = prediction.reshape(480,640,3)
            prediction = prediction*255
        else:
            print("Out of Context: Model Specified is wrong")
        
        # for all other models, post proc is same #
        # prediction = prediction*255

        upscaled_patch = np.clip(prediction, 0, 255).astype(np.uint8)

    except Exception as e:
        print("Exception",str(e))
    
    return upscaled_patch, elapsed_time


# Endpoint for super resolution
@imageEnhance_bp.route('/image_enhancement', methods=['POST'])
def image_enhancement():
    try:
        print("Image enhancement blueprint")
    
        ## GETTING DATA FROM ELECTRON ##
        print("Fetching image data from the POST request")
        image_data = request.files['imageData']
        
        model_name = request.form['model_name']
        print("MODEL NAME:",model_name)
        
        runtime = request.form['runtime']
        print("RUN TIME:",runtime)
        
        print("load as PIL IMG")
        image_data = Image.open(image_data)
        #image_data.save("input_img.png")
        width, height = image_data.size
        print(f"Received img height = {height} ; width = {width}")
        
        
        ## MAKING CONNECTION WITH SNPE EXE ##
        context = zmq.Context()
        
        # Create a REQ (request) socket
        socket = context.socket(zmq.REQ)
        server_address = "tcp://localhost:5555"  # Replace with your server's address
        socket.connect(server_address)

        
        ## BUILDING NETWORK ##
        # global old_model_name
        # global old_runtime
        
        if model_name != globalvar.old_model_name or runtime != globalvar.old_runtime:
            print("___________________BUILDINGNETWORK________________")
            print("old_model_name: ", globalvar.old_model_name, "::model_name: ",model_name)
            print("old_runtime: ", globalvar.old_runtime, "::runtime: ",runtime)
            buildnetwork_ie(socket, model_name, runtime)  ##build network when there is some change other than image
            globalvar.old_model_name = model_name
            globalvar.old_runtime = runtime 


        ## INFERENCING ON NETWORK ##        
        
        # Step 1: Read Image and Extract 128x128 patches from the image
        image_np = np.array(image_data)

        merged_img, time_taken = runmodel_ie(socket, image_np, model_name, runtime)
        
        print("Received Enhanced Image")
        
        global time_taken_model
        global upscaled_img_dims
        time_taken_model = str(f'{time_taken*1000:.2f}')+" ms"
        
        
        # Step 3: Getting image dimensions
        
        upscaled_img_dims = str(merged_img.shape[1]) + " x " +str(merged_img.shape[0]);
        print("upscaled_img_dims: ",upscaled_img_dims)
        merged_img = Image.fromarray(np.uint8(merged_img))
        # merged_img.save("upscaled_lowlight.png")
        
        # Convert the upscaled image to a binary response
        output_buffer = io.BytesIO()
        
        merged_img.save(output_buffer, format='PNG')
        
        print("Sending enhanced image as output to electron ...")
        output_buffer.seek(0)
        return send_file(output_buffer, mimetype='image/png')
 
    except Exception as e:
        print("#############EXCEPTION####################")
        print(str(e))
        return jsonify({'error': str(e)}), 400
        
# Endpoint for super resolution
@imageEnhance_bp.route('/low-light/timer_string', methods=['POST'])
def timer_string():
    output_new = {
            "infertime": time_taken_model,
            "outputdims": upscaled_img_dims,
        }
    return jsonify(output_new), 200