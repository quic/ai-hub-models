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

from utils import draw_box
from threading import Lock

from datetime import datetime

import MobileNetSSd
import YoloNas 
import YoloX
import SSDLite

import globalvar

# from utils import pyinstaller_absolute_path

runtime_name_decoder={'DSP':b"DSP",'GPU':b"GPU", 'CPU':b"CPU"}
# dlc_name_decoder={'yolonas':'Quant_yoloNas_s_320.dlc', 'ssdlite':'yolo_nas_s.dlc'}
dlc_name_decoder={'yolonas':'quant_yolo_nas_s.dlc', 'mobilenetssd':'ssd_mobilenetV2_without_ABP-NMS_Q.dlc', 'yolox':'yolox_x_212_Q.dlc'}
    
objectDetect_bp = Blueprint("ObjectDetect", __name__)

lockmut = Lock()



@objectDetect_bp.route('/')
def index():
    return render_template('index.html')



@objectDetect_bp.route('/od_checkdlc', methods=['POST'])
def checkdlc():
    from flask import jsonify
    import os
    model_name = request.form.get('model_name')
    dlc_path = os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","objectdetect", dlc_name_decoder.get(model_name))

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
    
def buildnetwork_od(socket, model_name, run_time):

    print("BUILDING NETWORK ObjectDetect")
    print("Model name: ",model_name)
    first_str = b"networkbuild"
    
    dlc_path = bytes(os.path.join("C:\Qualcomm\AIStack\AI_Solutions\DLC","objectdetect", dlc_name_decoder.get(model_name)),'utf-8')

    # global flag_modelbuild_inprocess
    
    # if flag_modelbuild_inprocess==0:
        # flag_modelbuild_inprocess = 1   
    socket.send_multipart([first_str,dlc_path, runtime_name_decoder.get(run_time)])   
    print("Messages sent for building network, waiting for reply")
    message_build = socket.recv()
    # flag_modelbuild_inprocess = 0
    print(message_build)
    



@objectDetect_bp.route('/od_stop_infer', methods=['POST'])
def stop_infer():
    print("***********************************************************************************************************************************************************************")
    print("Setting Gobal var 'Stop Infer'")
    print("***********************************************************************************************************************************************************************")
    lockmut.acquire()
    globalvar.stop_infer = 1
    lockmut.release()
    return jsonify({'Stopped Server': 'Done'}), 200

@objectDetect_bp.route('/od_build_model', methods=['POST'])
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
            buildnetwork_od(globalvar.__sockets[0]["socket"], model_name, runtime)  ##build network when there is some change other than image
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

@objectDetect_bp.route('/object_detection', methods=['POST'])
def object_detection():

    try:
        # print("Value of globalvar.stop_infer: ",globalvar.stop_infer)
        lockmut.acquire()
        if globalvar.stop_infer == 1:
            return jsonify({'error': 'busy'}), 400
        lockmut.release()
            
        image_data = request.files['image_data']
        
        ## INFERENCING ON NETWORK ##    
        # print("INFERNCECING")
        start = datetime.now()
        
        image_data = Image.open(image_data)        
        
        image_np = np.array(image_data)
        image_np = image_np.astype(np.float32)
        
        model_name = globalvar.old_model_name
        
        if model_name == 'yolonas':
            od_model = YoloNas.YoloNAS()
        elif model_name == 'ssdlite':
            od_model = SSDLite.SSDLITE()
        elif model_name == 'mobilenetssd':
            od_model = MobileNetSSd.MobileNetSSD()
        elif model_name == 'yolox':
            od_model = YoloX.YoloX()
        else:
            print("FATAL ISSUE")
            print('model_name: ',model_name)
            od_model =  YoloNas.YoloNAS()
            
        image_pre = od_model.preprocessing(image_np)
        end = datetime.now()
        print("preprocess Time: ", end-start)
       
        for inds, s in enumerate(globalvar.__sockets):
            if s["lock"].acquire(blocking=False):
                # socket.send_multipart([b"infer", image_np])
                # socket.send_multipart([b"infer", data.tobytes()])
                # message = socket.recv()
                now = datetime.now()
                # s["socket"].send_multipart([b"infer", data.tobytes()], zmq.NOBLOCK)
                print("index of socket:", inds)
                s["socket"].send_multipart([b"infer", image_pre.tobytes()], zmq.NOBLOCK)
                print("DATA sent to snpe")
                # if globalvar.stop_infer == 1:
                    # s["lock"].release()
                    # return jsonify({'error': 'busy'}), 400
                # print("data sent")
                message = s["socket"].recv()
                print("data received from snpe")
                end = datetime.now()
                print("infer Time: ", end-now)
                # print("data received")
                s["lock"].release()

                print("lock release")
                inf_result = np.frombuffer(message, dtype=np.float32)
                # print("inf_result.shape:: ",inf_result.shape)
                # print("First Value of vector in python: ",inf_result[:5])
                # print("Last 5 Value of vector in python: ",inf_result[-5:])
                now = datetime.now()
                # print("resuly")
                # print("RSULT shape:",inf_result.shape)
                
                img = od_model.postprocessing(image_data, inf_result)
                # print("postprocess done")
                _, frame_encoded = cv2.imencode(".jpeg", img)

                output_buffer = io.BytesIO(frame_encoded.tobytes())
                end = datetime.now()
                print("postprocess time: ",end-now)
                return send_file(output_buffer, mimetype='image/jpeg')
        
        return jsonify({'error': 'busy'}), 400

    except Exception as e:
        print("<<<<<<<<<<<<<<<EXCEPTION>>>>>>>>>>>>>>>")
        print(str(e))
        return jsonify({'error': str(e)}), 400