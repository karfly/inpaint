define({ "api": [  {    "type": "post",    "url": "/add_image",    "title": "Add image",    "version": "0.0.1",    "name": "add_image",    "group": "Inpaint",    "parameter": {      "fields": {        "Parameter": [          {            "group": "Parameter",            "type": "File",            "optional": false,            "field": "image",            "description": "<p>Png file 256x256</p>"          }        ]      }    },    "success": {      "fields": {        "Success 200": [          {            "group": "Success 200",            "type": "String",            "optional": false,            "field": "image_id",            "description": "<p>Image Id in DB</p>"          }        ]      }    },    "filename": "./app.py",    "groupTitle": "Inpaint"  },  {    "type": "post",    "url": "/apply_mask",    "title": "Apply mask",    "version": "0.0.1",    "name": "apply_mask",    "group": "Inpaint",    "parameter": {      "fields": {        "Parameter": [          {            "group": "Parameter",            "type": "String",            "optional": false,            "field": "image_id",            "description": "<p>Id of source image to apply</p>"          },          {            "group": "Parameter",            "type": "Integer",            "optional": false,            "field": "step_id",            "description": "<p>Stroke number in history</p>"          },          {            "group": "Parameter",            "type": "File",            "optional": false,            "field": "mask",            "description": "<p>Png file 256x256, where painted is white and remained is black/transparent</p>"          }        ]      }    },    "success": {      "fields": {        "Success 200": [          {            "group": "Success 200",            "type": "String",            "optional": false,            "field": "image_id",            "description": "<p>Id of applied source image (unchanged)</p>"          },          {            "group": "Success 200",            "type": "Integer",            "optional": false,            "field": "step_id",            "description": "<p>Stroke number in history (unchanged)</p>"          },          {            "group": "Success 200",            "type": "String",            "optional": false,            "field": "result",            "description": "<p>DataURI of result png</p>"          }        ]      }    },    "filename": "./app.py",    "groupTitle": "Inpaint"  },  {    "type": "get",    "url": "/pick_random",    "title": "Pick random photo",    "version": "0.0.1",    "name": "pick_random",    "group": "Inpaint",    "success": {      "fields": {        "Success 200": [          {            "group": "Success 200",            "type": "File",            "optional": false,            "field": "File",            "description": "<p>random image png</p>"          }        ]      }    },    "filename": "./app.py",    "groupTitle": "Inpaint"  },  {    "success": {      "fields": {        "Success 200": [          {            "group": "Success 200",            "optional": false,            "field": "varname1",            "description": "<p>No type.</p>"          },          {            "group": "Success 200",            "type": "String",            "optional": false,            "field": "varname2",            "description": "<p>With type.</p>"          }        ]      }    },    "type": "",    "url": "",    "version": "0.0.0",    "filename": "./static/docs/main.js",    "group": "_Users_vlivashkin_Documents_GitHub_illusionww_inpaint_app_static_docs_main_js",    "groupTitle": "_Users_vlivashkin_Documents_GitHub_illusionww_inpaint_app_static_docs_main_js",    "name": ""  }] });
