// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');
const request = require('request');
require('dotenv').config({path:path.join(__dirname, "/.env")});
const fs = require('fs');
const { execFile ,exec} = require('child_process');

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

const topicsPath = fs.existsSync(path.join(__dirname, "/dist/assets/")) ? path.join(__dirname, "/dist/assets/") : path.join(__dirname, "/src/assets/")

const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, '/dist')));


app.get('/api/getTopics', (req, res) => {
	try{
		fs.readFile(topicsPath+'QA_List.json', (err, data) => {
    		if(err) {
	        	console.log(err)
	        	return res.status(400).json(err)
	        }
    		let jsonData = JSON.parse(data);
    		res.status(200).json(jsonData);
		});
	}
	catch(err){
		res.status(400).json(err);
	}
})

app.get('/api/serverDetails', (req, res) => {
	let info;
	if (req.query.type === 'ip') {
		info = process.env.IP
	}
	if (req.query.type === 'port') {
		info = process.env.AGENT_PORT
	}
	let resp = {status:"successfull",info:info}
	res.status(200).json(resp);
})

app.post('/api/fetchPredictionResults', (req, res) => {
	const options = {
    	body: req.body.input,
    	json: true,
    	url: req.body.urlDetails.url,
    	method: req.body.urlDetails.method,
	};
	request(options, function(err, response, body) {
		if (err) {
			console.log(err)
			res.status(400).json(err);
			return
		}
		let inference = body
		let resp = {}
	    resp['question'] = req.body.input.question;
	    resp['answer'] = inference.answer;
	    resp['time'] = new Date();
	    resp['executionTime'] = inference.exec_time;
	    resp['error'] = inference.error;
	    res.status(200).json(resp);
	});
})

app.get('/*',(req, res) => {
  res.sendFile(path.join(__dirname, 'dist/index.html'));
});


app.listen(PORT,() => {
    console.log(`Running on port ${PORT}`)
})

module.exports = app;