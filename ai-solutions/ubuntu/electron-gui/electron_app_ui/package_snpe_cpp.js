// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
const nodeCmd = require('node-cmd')

let command = String.raw `mkdir -p Release && cd ../SNPE_CPP_Code && mkdir -p build && cd build && mkdir -p Release && cmake ../ && cmake --build . && cp Release/snpe-sample ../../electron_app_ui/Release/snpe-sample`;
			 
nodeCmd.runSync(command, (err, data, stderr) => {
if(data) { 
 return res.json(data);
}
return err; 	
});