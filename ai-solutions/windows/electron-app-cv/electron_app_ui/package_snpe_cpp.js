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

let command = String.raw `cd ..\SNPE_CPP_Code && (if not exist build (mkdir build && cd build) else (cd build)) && cmake ../. -G "Visual Studio 17 2022" -A ARM64 -DCHISPSET SC8380 && cmake --build ./ --config Release`;

nodeCmd.runSync(command, (err, data, stderr) => {
if(data) { 
 return res.json(data);
}
return err; 	
});