// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
const path = require("path");

const spawn = require("child_process").spawn,
  ls = spawn(
    "pyinstaller",
    [
      "-w",
      "--onefile",
      `--add-data ../python_flask_server/templates${path.delimiter}templates`,
      `--add-data ../python_flask_server/static${path.delimiter}static`,
	  `--add-data ../python_flask_server/DLC${path.delimiter}DLC`,
      "--distpath dist-python",
      "../python_flask_server/server.py",
    ],
    {
      shell: true,
    }
  );

ls.stdout.on("data", function (data) {
  // stream output of build process
  console.log("INFO: ", data.toString());
});

ls.stderr.on("data", function (data) {
  console.log( data.toString());
});
ls.on("exit", function (code) {
  console.log("pyinstaller process exited with code " + code.toString());
});
