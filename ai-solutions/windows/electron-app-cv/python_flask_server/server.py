# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
from flask import Flask
from flask_cors import CORS

from ImageEnhancement_blueprint import imageEnhance_bp
from SuperResolution_blueprint import superRes_bp
from ObjectDetection_blueprint import objectDetect_bp
from ImageSegmentation_blueprint import imageSegment_bp
from utils import objectDetect_init

from waitress import serve


app = Flask(__name__,
            static_url_path='', 
            static_folder='static')
CORS(app)

app.register_blueprint(objectDetect_bp)
app.register_blueprint(imageEnhance_bp)
app.register_blueprint(superRes_bp)
app.register_blueprint(imageSegment_bp)


if __name__ == '__main__':
    objectDetect_init()
    
    ## Debug/developer Mode
    # app.run(host='0.0.0.0', port=9081, debug=True, threaded=True)

    ##Production server
    serve(app, host='0.0.0.0', port=9081, threads=4)    
    
