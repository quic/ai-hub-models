# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
import datetime
from sqlite3 import Date
from flask import Flask, render_template, request, jsonify, make_response, send_file
from flask_cors import CORS
from PIL import Image
from empatches import EMPatches
import io, os
import cv2
import numpy as np
import time
import functools
import zmq
import sys
import torch
import tensorflow as tf

model=None
app = Flask(__name__,
            static_url_path='', 
            static_folder='static')
CORS(app)

time_taken_model = ""
upscaled_img_dims = ""
old_runtime = ""
old_model_name = ""

def pyinstaller_absolute_path(relative_path):
    """ For PyInstaller, getting absolute path of resources"""
    base_path = getattr( sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(base_path, relative_path)
    return abs_path

def func(start_logits,end_logits,inputs,tokenizer):
    answer_start_index = int(tf.math.argmax(start_logits, axis=-1)[0])
    answer_end_index = int(tf.math.argmax(end_logits, axis=-1)[0])

    start=min(answer_start_index,answer_end_index)
    end=max(answer_start_index,answer_end_index)
    print("start_index:",answer_start_index,"end index:",answer_end_index)
    predict_answer_tokens = inputs.input_ids[0, start : end+ 1]
    return tokenizer.decode(predict_answer_tokens)


@app.route('/api/BuildModel', methods=['POST'])
def getModelName():

    data=request.json

    inp=data['input']
    global model
    model=inp['model']
    print(model,data)
    return model



def buildnetwork(socket, model_name, run_time):

    print("BUILDING NETWORK")
    first_str = b"networkbuild"
    
    runtime_name_decoder={'DSP':b"DSP",'GPU':b"GPU", 'CPU':b"CPU"}
    dlc_name_decoder={'distilbert':'distilbert2_float.dlc','alberta':'alberta_float.dlc','mobile_bert':'mobile_bert_updated_float.dlc','electrabase':'electrabase_float.dlc','bert_base':'bert_base2_float.dlc'}
    dlc_path = bytes(pyinstaller_absolute_path(os.path.join("dlc", dlc_name_decoder.get(model_name))),'utf-8')
    
    socket.send_multipart([first_str,dlc_path, runtime_name_decoder.get(run_time)])   

    print("Messages sent for building network, waiting for reply")
    message_build = socket.recv()
    print(message_build)


def preprocess(question,text,model_name,tokenizer,socket):
    inputs = tokenizer(question, text, return_tensors="np",
            padding='max_length',
            truncation="longest_first",
            max_length=384)
    if model_name!='distilbert':
        attention_mask =inputs['attention_mask'].tobytes()
        input_ids=inputs['input_ids'].tobytes()
        token_type_ids=inputs['token_type_ids'].tobytes()
        #Sending multiple messages to the snpe cpp server
        socket.send_multipart([b"infer",b"3",attention_mask,input_ids,token_type_ids])
        

    else:
        attention_mask =inputs['attention_mask'].tobytes()
        input_ids=inputs['input_ids'].tobytes()
        #Sending multiple messages to the snpe cpp server
        socket.send_multipart([b"infer",b"2",input_ids,attention_mask])

    return inputs
        


def predict(socket,question,text, model_name, run_time ):
    
    
    runtime_name_decoder={'DSP':"--use_dsp",'GPU':"--use_gpu", 'CPU':""}

    if model_name=='alberta':
        from transformers import AutoTokenizer, AlbertForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")

    elif model_name=='distilbert':
        from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        inputs=preprocess(question,text,'distilbert',tokenizer,socket)

    elif model_name=='electrabase':
        from transformers import AutoTokenizer, ElectraForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/electra-base-squad2")
        inputs=preprocess(question,text,'electrabase',tokenizer,socket)
    elif model_name=='mobile_bert':
        from transformers import AutoTokenizer, MobileBertForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("csarron/mobilebert-uncased-squad-v2")
        inputs=preprocess(question,text,'mobile_bert',tokenizer,socket)

    elif model_name=='bert_base':
        from transformers import AutoTokenizer, BertForQuestionAnswering
        tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        inputs=preprocess(question,text,'bert_base',tokenizer,socket)
    
    dlc_name_decoder={'distilbert':'distilbert2_float.dlc','alberta':'alberta_float.dlc','mobile_bert':'mobile_bert_updated_float.dlc','electrabase':'electrabase_float.dlc','bert_base':'bert_base2_float.dlc'}
    dlc_path = os.path.join("dlc", dlc_name_decoder.get(model_name))
    
    print("Messages sent, waiting for reply")
    message_img_out1 = socket.recv()
    message_img_out2 = socket.recv()
    
    end_logits= np.frombuffer(message_img_out1, dtype=np.float32)
    start_logits= np.frombuffer(message_img_out2, dtype=np.float32)
    
    start_logits = start_logits.reshape(1,384)
    end_logits = end_logits.reshape(1,384)
    
    result=func(start_logits,end_logits,inputs,tokenizer)
    print("Result is :",result)
    socket.send(b"get_infer_time")
    message_infer_time = socket.recv()
    print("message_infer_time", message_infer_time.decode('UTF-8'))
    return result,message_infer_time.decode('UTF-8')
        


# Serve INDEX HTML file
@app.route('/')
def index():
    return render_template('index.html')
    
# Endpoint for super resolution
@app.route('/timer_string', methods=['POST'])
def timer_string():
    print("Fetching image data from the POST request")
    

@app.route('/api/fetchPredictionResults', methods=['POST'])
def initialization():
    print("Fetching image data from the POST request")
    data=request.json

    inp=data['input']
    question=inp['question']
    paragraph=inp['paragraph']
    runtime=inp['runtime']
    print("question:",question)
    print("paragraph:",paragraph)
   

    print("Model Name",model)
    old_model_name='mobilebert'
    old_runtime="DSP"
    ## MAKING CONNECTION WITH SNPE EXE ##
    context = zmq.Context()
    # Create a REQ (request) socket
    socket = context.socket(zmq.REQ)
    server_address = "tcp://localhost:5555"  # Replace with your server's address
    socket.connect(server_address)
    if model != old_model_name or runtime != old_runtime:
            print("___________________BUILDINGNETWORK________________")
            print("old_model_name: ", old_model_name, "::model_name: ",model)
            print("old_runtime: ", old_runtime, "::runtime: ",runtime)
            buildnetwork(socket, model, runtime)  ##build network when there is some change other than image
            old_model_name = model
            old_runtime = runtime 

    result,infer_time=predict(socket,question,paragraph, model,runtime )

    current_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    response_data={
        'question':question,
        'answer':result,
        'time':current_time,
        'executionTime':infer_time,
        'error_message':None
    }
    print(response_data)
    response=make_response(jsonify(response_data),200)
    return response
    # #     image_data = request.files['imageData']
        
    #     model_name = request.form['model_name']
    #     print("MODEL NAME:",model_name)
        
    #     runtime = request.form['runtime']
    #     print("RUN TIME:",runtime)
        
    #     print("load as PIL IMG")
    #     image_data = Image.open(image_data)
    #     #image_data.save("input_img.png")
    #     width, height = image_data.size
    #     print(f"Received img height = {height} ; width = {width}")
        
        
    #     ## MAKING CONNECTION WITH SNPE EXE ##
    #     context = zmq.Context()
    #     # Create a REQ (request) socket
    #     socket = context.socket(zmq.REQ)
    #     server_address = "tcp://localhost:5555"  # Replace with your server's address
    #     socket.connect(server_address)

        
    #     ## BUILDING NETWORK ##
    #     global old_model_name
    #     global old_runtime
        
    #     if model_name != old_model_name or runtime != old_runtime:
    #         print("___________________BUILDINGNETWORK________________")
    #         print("old_model_name: ", old_model_name, "::model_name: ",model_name)
    #         print("old_runtime: ", old_runtime, "::runtime: ",runtime)
    #         buildnetwork(socket, model_name, runtime)  ##build network when there is some change other than image
    #         old_model_name = model_name
    #         old_runtime = runtime 


    #     ## INFERENCING ON NETWORK ##
        
        
    #     # Step 0: Set upscaling params
    #     patch_size = 128
    #     overlap_factor = 0.1
    #     scaling_factor= 4
        
        
    #     # Step 1: Read Image and Extract 128x128 patches from the image
    #     image_np = np.array(image_data)

    #     # Dividing image into small patches
    #     emp = EMPatches()
    #     img_patches, indices = emp.extract_patches(image_np, patchsize=patch_size, overlap=overlap_factor)
    #     print(f"Num of patches of 128 = {len(img_patches)}")
        
        
    #     # Step 2: Upscale each patch by a factor of 4
    #     upscaled_patches= []
    #     infer_time_list = []
    #     time_taken = 0
    #     for patch in img_patches:
    #         pt, single_infer_time = upscale_patch(socket, patch, model_name, runtime)
    #         upscaled_patches.append(pt)
    #         time_taken = time_taken + single_infer_time  ##Adding time for all patches
            
    #     print("Received upscaled patches")
        
    #     global time_taken_model
    #     global upscaled_img_dims
    #     time_taken_model = str(f'{time_taken*1000:.2f}')+" ms"
        
        
        
    #     # Step 3: Stitch back the upscaled patches into a single image
        
    #     # Calculate the upscaled stiching indices
    #     up_img = np.zeros((image_np.shape[0]*scaling_factor, image_np.shape[1]*scaling_factor, image_np.shape[2]), np.uint8)
    #     _, new_indices = emp.extract_patches(up_img, patchsize=patch_size*scaling_factor, overlap=overlap_factor)
        
    #     # merge with new_indices
    #     merged_img = emp.merge_patches(upscaled_patches, new_indices, mode='min')
    #     upscaled_img_dims = str(merged_img.shape[0]) + " x " +str(merged_img.shape[1]);
    #     print("upscaled_img_dims: ",upscaled_img_dims)
        
    #     merged_img = Image.fromarray(np.uint8(merged_img))
    #     #merged_img.save("upscaled.png")
        
    #     # Convert the upscaled image to a binary response
    #     output_buffer = io.BytesIO()
        
    #     merged_img.save(output_buffer, format='PNG')
        
    #     print("Sending upscaled image as output to electron ...")
    #     output_buffer.seek(0)
    #     return send_file(output_buffer, mimetype='image/png')
 
    # except Exception as e:
    #     print("#############EXCEPTION####################")
    #     print(str(e))
    #     return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9081, debug=True)
