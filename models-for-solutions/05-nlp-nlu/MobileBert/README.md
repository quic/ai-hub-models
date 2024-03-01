# "Question Answering" using MobileBert

| Field | Description |
| --- | --- |
| Model Name | MobileBert |
| DNN Framwork | ONNX |
| Public Repo  | https://github.com/gemde001/MobileBERT |
| Paper        | https://arxiv.org/abs/2004.02984 |
| DLC Number of Inputs | 3 |
| DLC Input Ids Dimension | (1,384) |
| DLC Attention Mask Dimension | (1,384) |
| DLC Token Type Ids | (1,384) |
| Pre-Processing | Use Model Specific Tokenizer |
| Post-Processing | Again Use Model Specific Tokenizer to Post Process the Output |

## Pre-Requisites

- Qualcomm® Neural Processing SDK setup should be completed by following the guide [here](https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html)
- Install onnx v1.6.0. Installation instruction can be found [here](https://qdn-drekartst.qualcomm.com/hardware/qualcomm-innovators-development-kit/frameworks-qualcomm-neural-processing-sdk-for-ai)
- Install onnxsim ```pip install onnxsim``` and onnxruntime ```pip install onnxruntime```.
- Install optimum ```pip install optimum```.

## How to get the model ? 

For this demo, you can directly get the [onnx model](Models/onnx-model/model.onnx) from the directory onnx_model or you can generate it from this [jupyter_notebook](generating_model). 


## Convert model to DLC 

- Convert the onnx model to DLC with below command. Below, command will also fix the input dimension for the dlc. 

```python
snpe-onnx-to-dlc -i mobilebert-onnx/model.onnx -d input_ids 1,384 -d attention_mask 1,384 -d token_type_ids 1,384 -o mobile_bert.dlc
```

## Quantization of DLC
- Quantization can improve model performance in terms of latency and make the model light weight. 
- Before Running this command make sure you've created the raw file and the list.txt
```python
snpe-dlc-graph-prepare --input_dlc mobile_bert.dlc --input_list tf_raw_list.txt  --output_dlc mobile_bert_int.dlc 
```

# Accuracy analysis
- To check accuracy please run "accuracy_analyzer.ipynb" a jupyter notebook present in accuracy folder.
- To run any jupyter notebook, run below command. It will generate few links on the screen, pick the link with your machine name on it (host-name) and paste it in any browser.
- Navigate to the notebook ".ipynb" file and simply click that file.
```python
jupyter notebook --no-browser --port=8080 --ip 0.0.0.0 --allow-root
```



# References

1. https://arxiv.org/abs/2004.02984
2. https://github.com/gemde001/MobileBERT

    
###### *Qualcomm Neural Processing SDK and Snapdragon are products of Qualcomm Technologies, Inc. and/or its subsidiaries.*    
