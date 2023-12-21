# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
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

from ImageEnhancement_blueprint import imageEnhance_bp
from SuperResolution_blueprint import superRes_bp
app = Flask(__name__,
            static_url_path='', 
            static_folder='static')
CORS(app)

time_taken_model = ""
upscaled_img_dims = ""
old_runtime = ""
old_model_name = ""

app.register_blueprint(imageEnhance_bp)
app.register_blueprint(superRes_bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9081, debug=True)
