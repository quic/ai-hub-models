// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
// Modules to control application life and create native browser window
const { app, BrowserWindow } = require('electron')
const path = require('path')
var processes = [];  //global list to hold PID(s)

//Module to kill child processes
const killSubprocesses = (main_pid) => {
  let cleanup_completed = false;
  const psTree = require("ps-tree");
  console.log("killSubprocesses: ");
  psTree(main_pid, function (err, children) {
		let child_pids_array = [main_pid].concat(children.map(function (p){
		console.log("PID: ",p.PID);
		return p.PID}));
		child_pids_array.forEach(function (pid) {
			console.log("Killing PIDS: ", pid);
		    process.kill(pid);
		});
		cleanup_completed= true;
  });
	return new Promise(function (resolve, reject) {
    (function waitForSubProcessCleanup() {
      if (cleanup_completed) return resolve();
      setTimeout(waitForSubProcessCleanup, 30);
    })();
  });
};

function createWindow () {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })

  mainWindow.maximize()
  // and load the index.html of the app.
  mainWindow.loadFile('index_sr.html')
  console.log("Opened")
  // Open the DevTools.
  // mainWindow.webContents.openDevTools()
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  

 console.log("APP ready")
 
 server_exe_path = path.join(
	  __dirname,
      'dist-python',
      'server'
    );

 // console.log("EXE path:", server_exe_path)
 
 //Run Flask Server
 const execFile = require("child_process").spawn(server_exe_path);
 processes.push(execFile);
 execFile.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  execFile.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  execFile.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });
  
  execFile.on('exit', function(code, signal) {
	  console.log(`EXITING CHILD PROCESS ${code} ${signal} ${execFile.pid}`);
  });
  
  execFile.on('error', function(err) {
  console.log('Exe Not present at specified path (Use npm run package to make .exe) and paste it at ' + server_exe_path);
  processes = processes.filter(function (iter_el) {
        return iter_el != execFile;
    });
  });
  
  //Run SNPE exe
  cpp_exe_path = path.join(
	  __dirname,
      'Release',
      'snpe-sample'
    );
	
 // console.log("cpp_exe_path path:", cpp_exe_path)
 const cppexecFile = require("child_process").spawn(cpp_exe_path);
 processes.push(cppexecFile);
 cppexecFile.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  cppexecFile.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  cppexecFile.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
  });
  
  cppexecFile.on('exit', function(code, signal) {
	  console.log(`EXITING CHILD PROCESS ${code} ${signal} ${cppexecFile.pid}`);
  });
 
  cppexecFile.on('error', function(err) {
    console.log('Exe Not present at specified path (Use npm run package to make .exe) and paste Release folder from SNPE_CPP_CODE at ' + cpp_exe_path);
	processes = processes.filter(function (iter_el) {
        return iter_el != cppexecFile;
    });
  });
  createWindow()
  

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })

});

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', function () {
  if (process.platform !== 'darwin'){
	  console.log("Inside not darwin");
	  if(processes.length!=0){
		processes.forEach(function(proc) {
			killSubprocesses(proc.pid).then(()=>{app.quit();
			});
		});
	  }
	  else
	  {
		app.quit();
	  }
  }
}); 
