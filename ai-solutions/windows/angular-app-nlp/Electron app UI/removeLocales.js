// -*- mode: js -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
/*******************************************************************************
#
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
*******************************************************************************/

exports.default = async function(context) {
	var fs = require('fs');
	var localeDir = context.appOutDir+'/locales/';
	fs.readdir(localeDir, function(err, files){
		if(!(files && files.length)) return;
		for (var i = 0, len = files.length; i < len; i++) {
			var match = files[i].match(/en-US\.pak/);
			if(match === null){
				fs.unlinkSync(localeDir+files[i]);
			}
		}
	});
}