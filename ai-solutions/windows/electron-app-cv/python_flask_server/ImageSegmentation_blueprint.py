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
from flask import request, jsonify, make_response, send_file, render_template
from flask_cors import cross_origin
from PIL import Image
import io
import os
import cv2
import numpy as np
import zmq
from torch import from_numpy

from threading import Lock
from datetime import datetime

import globalvar
# from utils import pyinstaller_absolute_path


__sockets = []
time_taken_model = ""
upscaled_img_dims = ""
old_runtime = ""
old_model_name = ""

lockmut = Lock()

imageSegment_bp = Blueprint("ImageSegment", __name__)

runtime_name_decoder={'DSP':b"DSP",'GPU':b"GPU", 'CPU':b"CPU"}
dlc_name_decoder={'FCN_RESNET50':'fcn_resnet50_quant16_w8a16.dlc', 'FCN_RESNET101':'fcn_resnet101_quant16_w8a16.dlc',"LRASPP":"lraspp_mobilenet_v3_large_quant16_w8a16.dlc", "DEEPLABV3_RESNET50":"deeplabv3_resnet50_quant_w8a8.dlc", "DEEPLABV3_RESNET101":"deeplabv3_resnet101_quant_w8a8.dlc" }
    

from torchvision import transforms as T

transform = T.Compose([
                T.Resize((400,400)),
                
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

label_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus 
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 0.8 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def get_segment_labels(image, model, device):
    # transform the image to tensor and load into computation device
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = outputs.detach().cpu().numpy()
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

@imageSegment_bp.route('/is_checkdlc', methods=['POST'])
def checkdlc():
    from flask import jsonify
    import os
    model_name = request.form.get('model_name')
    dlc_path = os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","imageseg", dlc_name_decoder.get(model_name))

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
def buildnetwork_is(socket, model_name, run_time):

    print("BUILDING NETWORK imageSegment")
    print("Model name: ",model_name)
    first_str = b"networkbuild"
    
    dlc_path = bytes(os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","imageseg", dlc_name_decoder.get(model_name)),'utf-8')
    
    socket.send_multipart([first_str,dlc_path, runtime_name_decoder.get(run_time)])   

    print("Messages sent for building network, waiting for reply")
    message_build = socket.recv()
    print(message_build)

@imageSegment_bp.route('/is_stop_infer', methods=['POST'])
def stop_infer():
    print("***********************************************************************************************************************************************************************")
    print("Setting Gobal var 'Stop Infer'")
    print("***********************************************************************************************************************************************************************")
    lockmut.acquire()
    globalvar.stop_infer = 1
    lockmut.release()
    return jsonify({'Stopped Server': 'Done'}), 200
	
@imageSegment_bp.route('/is_build_model', methods=['POST'])
def build_model():

    try:
        ## GETTING DATA FROM ELECTRON ##
        model_name = request.form['model_name']
        runtime = request.form['runtime_name']
        
         
        # print(" flag_modelbuild_inprocess: ",globalvar.flag_modelbuild_inprocess)
        # print("model_name: ",model_name)
        # print("old_model_name: ",globalvar.old_model_name)
        # print("runtime: ",runtime)
        # print("old_runtime: ",globalvar.old_runtime)
        
        
        if (model_name != globalvar.old_model_name or runtime != globalvar.old_runtime) and (globalvar. flag_modelbuild_inprocess==0):
            # print("Building NEW Model")
            globalvar.flag_modelbuild_inprocess = 1   
            print("___________________BUILDINGNETWORK________________")
            print("old_model_name: ", globalvar.old_model_name, "::model_name: ",model_name)
            print("old_runtime: ", globalvar.old_runtime, "::runtime: ",runtime)
            buildnetwork_is(globalvar.__sockets[0]["socket"], model_name, runtime)  ##build network when there is some change other than image
            globalvar.flag_modelbuild_inprocess = 0
            globalvar.old_model_name = model_name
            globalvar.old_runtime = runtime
            globalvar.stop_infer = 0
            print("___________________DONE___________________________")
            return jsonify({'msg': 'model build successfully'}), 200

        print("Model already built")
        globalvar.stop_infer = 0
        return jsonify({'msg': 'model already build'}), 200
    except Exception as e:
        print("<<<<<<<<<<<<<<<Eexception at model build time>>>>>>>>>>>>>>>")
        print(str(e))
        return jsonify({'msg': str(e)}), 400
		
@imageSegment_bp.route('/image_segmentation', methods=['POST'])
def image_segment():
    try:
        ## GETTING DATA FROM ELECTRON ##
        lockmut.acquire()
        if globalvar.stop_infer == 1:
            return jsonify({'error': 'busy'}), 400
        lockmut.release()
        image_data = request.files['image_data']
        

        # MAKING CONNECTION WITH SNPE EXE ##

        

        
        model_name = globalvar.old_model_name

        
        ## INFERENCING ON NETWORK ##    
        print("INFERNCECING")
       
        start = datetime.now()
       
        original_image = Image.open(image_data)

        input_ = preprocess(image_data)
        print("preprocssed")
        data = input_.transpose(0, 2, 3, 1)
        
        end = datetime.now()
        print("preprocess Time: ", end-start)
		
        for inds, s in enumerate(globalvar.__sockets):
            if s["lock"].acquire(blocking=False):
                now = datetime.now()
                s["socket"].send_multipart([b"infer", data.tobytes()], zmq.NOBLOCK)
                print("data sent")
                message = s["socket"].recv()
                end = datetime.now()
                print("infer Time: ", end-now)
                # print("data received")
                s["lock"].release()

                # print("lock release")
                inf_result = np.frombuffer(message, dtype=np.float32)
                now = datetime.now()
                # print("resuly")
                # print("RSULT shape:",inf_result.shape)

                    

                ret = postprocess(original_image, inf_result)
                _, frame_encoded = cv2.imencode(".jpg", ret)
                output_buffer = io.BytesIO(frame_encoded.tobytes())
                end = datetime.now()
                print("postprocess time: ",end-now)
                return send_file(output_buffer, mimetype='image/jpeg')
        
        return jsonify({'error': 'busy'}), 400

    except Exception as e:
        print("<<<<<<<<<<<<<<<EXCEPTION>>>>>>>>>>>>>>>")
        print(str(e))
        return jsonify({'error': str(e)}), 400



def preprocess(input):  #TODO Preprocessing depends on model architecture and it can different for different models, Here all models have same pre and post processing 
    img = Image.open(input).convert('RGB')
    img = transform(img).unsqueeze(0) # To tensor of NCHW
    img = img.numpy()
    return img


def postprocess(input,output):
    res_reshape = output.reshape((1,400,400)).astype(np.float32)
    model_img = from_numpy(res_reshape)
    segmented_image = draw_segmentation_map(model_img[0])
    input  = input.resize((400,400))
    final_image = image_overlay(input, segmented_image)
    return final_image

    
