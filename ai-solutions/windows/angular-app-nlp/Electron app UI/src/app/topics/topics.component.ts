// -*- mode: ts -*-
// =============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
// =============================================================================

import { Component, OnInit } from '@angular/core';
import { BackendService } from '../backend.service';
import { Router } from '@angular/router';
import { FormBuilder, FormControl, FormGroup, FormArray, Validators,AbstractControl} from '@angular/forms';

@Component({
  selector: 'app-topics',
  templateUrl: './topics.component.html',
  styleUrls: ['./topics.component.css']
})
export class TopicsComponent implements OnInit {
	[x: string]: any;

	public topics: any[];
	public ModelFormGroup:FormGroup;
	
  	constructor(public auth: BackendService,private router: Router,private fb: FormBuilder) {}

  	ngOnInit(): void {
  		this.fetchTopics();
		this.ModelFormGroup = this.ModelFormGroupFn();
  	}
	ModelFormGroupFn(){
	return this.fb.group({
		//htp: ['',Validators.required],
		htp:['mobile_bert']
	});
	}

  	fetchTopics(){
	  	this.auth.getTopics().subscribe((response:any) => {
	        this.auth.topicContetnt = response.topics;
	        this.topics =  [...new Set(this.auth.topicContetnt.map((element :any) => element.topic))];
	    },(err) => {
	      console.log(err);
	    });

		
  		}
	  async Start(){
		
	  }

  	Next(){
		let inputInfo = {model:this.ModelFormGroup.get('htp')!.value}
		try{
			console.log(inputInfo)
			let ipInfo:any =  this.auth.fetchServerDetails('ip').toPromise();
			let portInfo:any =  this.auth.fetchServerDetails('port').toPromise();
			let ip = ipInfo.info;
			let port = portInfo.info;
			let method = 'POST';
			let api = '/predict'
			let protocol = 'http://'
			let urlDetails ={
			  url: protocol+ip+':'+port+api,
			  method:method
			}
			let data = {
			  urlDetails:urlDetails,
			  input:inputInfo
			}
			console.log(urlDetails)
			this.auth.BuildModel(data).toPromise();
			
		  }
		  catch (err:any){
			console.log(err)
			alert(err.message)
			
		  }
  		this.router.navigateByUrl('/prediction');
  	}

}
