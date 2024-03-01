// -*- mode: ts -*-
// =============================================================================
//  @@-COPYRIGHT-START-@@
//
//  Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
//  SPDX-License-Identifier: BSD-3-Clause
//
//  @@-COPYRIGHT-END-@@
// =============================================================================

import { Component, OnInit,Input } from '@angular/core';
import { BackendService } from '../backend.service';
import { MatTableDataSource } from "@angular/material/table";

@Component({
  	selector: 'app-results',
  	templateUrl: './results.component.html',
  	styleUrls: ['./results.component.css']
})

export class ResultsComponent implements OnInit {
	@Input() results: any;

	displayedColumns: string[] = ['Date', 'Question', 'Answer','Execution Time (ms)'];

	dataSource:MatTableDataSource<any>;

  	constructor(public auth: BackendService) {
		this.dataSource=new MatTableDataSource(this.results);
		console.log("datasource",this.dataSource)
	 }

	ngOnInit(): void {

		console.log("Inside Results Components",this.auth.results,this.dataSource)
	}

}
