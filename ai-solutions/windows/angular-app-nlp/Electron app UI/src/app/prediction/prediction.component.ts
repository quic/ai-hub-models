// -*- mode: ts -*-
// =============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
// =============================================================================

import { Component, OnInit, ViewChild  } from '@angular/core';
import { ResultsComponent } from '../results/results.component';
import { FormBuilder, FormControl, FormGroup, FormArray, Validators,AbstractControl} from '@angular/forms';
import { BackendService } from '../backend.service';
import { MatTableDataSource } from "@angular/material/table";
import { Router } from '@angular/router';
import {LayoutModule} from '@angular/cdk/layout';


@Component({
  providers:[ResultsComponent],
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class PredictionComponent implements OnInit {

	  public PredictionFormGroup:FormGroup;
	  public showProgress:boolean = false;
    public inferOutput:any;
    public fetchedResults :any = [];
    public content:any;
    public questionList:any = [];
    results = new MatTableDataSource<any>(this.fetchedResults);

  	constructor(private fb: FormBuilder,public auth: BackendService,public resultInfo:ResultsComponent,private router: Router) { }

  	ngOnInit(): void {
  		this.PredictionFormGroup = this.PredictionFormGroupFn();
      let topicIndex = this.auth.topicContetnt.map( (x:any) => { return x.topic; }).indexOf(this.auth.selectedTopic[0]);
      this.content = this.auth.topicContetnt[topicIndex].content;
      this.questionList = this.auth.topicContetnt[topicIndex].sampleQuestions;
      this.auth.results = [];
  	}

  	PredictionFormGroupFn(){
	    return this.fb.group({
	      	question: ['',Validators.required],
          //htp:['Cloud AI 100']
          htp:['DSP']
	    });
  	}

    Back(){
      this.showProgress = false;
      this.resetContent()
      this.router.navigateByUrl('/topics');
    }

    async Start(){
      this.resetContent()
      this.showProgress = true;
      let inputInfo = {question:this.PredictionFormGroup.get('question')!.value,paragraph:this.content,runtime:this.PredictionFormGroup.get('htp')!.value}


      try{
        console.log(inputInfo)
        let ipInfo:any = await this.auth.fetchServerDetails('ip').toPromise();
        let portInfo:any = await this.auth.fetchServerDetails('port').toPromise();
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
        this.inferOutput = await this.auth.fetchPredictionResults(data).toPromise();
        console.log("******************* inferOutput ********************")
        console.log(this.inferOutput)
        
        console.log("Before Fetched Results",this.fetchedResults)
        this.fetchedResults.push(this.inferOutput)
        console.log("After Fetched Results")
        console.log(this.fetchedResults)
        this.auth.results = this.fetchedResults;
        this.fetchedResults = this.fetchedResults.sort(function compare(a:any, b:any) {
          var dateA:any = new Date(a.time);
          var dateB:any = new Date(b.time);
            return dateB - dateA;
          });
        console.log("After Sorting",this.fetchedResults)
        this.results = new MatTableDataSource<any>(this.fetchedResults);

        console.log("Results",this.results)

        console.log("this.auth")
        this.highlight(this.inferOutput.answer)
        this.showProgress = false;
        this.resultInfo.results=this.auth.results;

        console.log("ResultComponent",this.resultInfo.results);
      }
      catch (err:any){
        console.log(err)
        alert(err.message)
        this.showProgress = false;
      }
    }

    updateQuestion(question:any){
      this.PredictionFormGroup.patchValue({ question: question });
    }

    highlight(text:any) {
      var inputText = document.getElementById("style-4");
      var innerHTML = inputText!.innerHTML.toLowerCase();
      var index = innerHTML.indexOf(text);
      if (index >= 0) { 
        innerHTML = innerHTML.substring(0,index) + "<span style='background-color:yellow'>" + innerHTML.substring(index,index+text.length) + "</span>" + innerHTML.substring(index + text.length);
        inputText!.innerHTML = innerHTML;
      }
    }

    resetContent(){
      var inputText = document.getElementById("style-4");
      inputText!.innerHTML = this.content;
    }
}
