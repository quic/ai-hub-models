// -*- mode: ts -*-
// =============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
// =============================================================================

import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class BackendService {

	public results:any = [];
  	public topicContetnt:any;
  	public selectedTopic:any;

  constructor(private http: HttpClient, private router: Router) { }

  	private request(method: 'post'|'get'|'patch'|'delete', type: any,data:any,param:any) {
	    let base;
	    if (method === 'post') 
	    {
			base = this.http.post(`http://127.0.0.1:9081/api/${type}`,data);
		}
	    else if (method === 'patch') 
	    {
	      base = this.http.patch(`/api/${type}/`+param,data);
	    } 
      else if (method === 'delete') {
        base = this.http.delete(`/api/${type}`);
      }
		else 
		{
			
		base = this.http.get(`/api/${type}`);

		}
	    const request = base.pipe(map((data) => {return data}));
		console.log("request"+request);
	    return request;
  	}
	

  	public getTopics(){
      return this.request('get', 'getTopics',null,null);
    }

    public preprocess(data:any){
  		return this.request('post', 'preprocess',data,null);
  	}

  	public fetchPredictionResults(data:any){
    	return this.request('post', 'fetchPredictionResults',data,null);
  	}
	  public BuildModel(data:any){
    	return this.request('post', 'BuildModel',data,null);
  	}

  	public postProcess(data:any){
  		return this.request('post', 'postProcess',data,null);
  	}

  	public dummyAPI(data:any){
  		return this.request('post', 'dummyAPI',data,null);
  	}

  	public fetchServerDetails(type:any){
      return this.request('get', 'serverDetails?type='+type,null,null);
    }
}
