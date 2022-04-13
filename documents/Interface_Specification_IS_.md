# Interface Specification (IS) <Algorithm>

Revision History: 

| Date   | Author   | Description                                |
| ------ | -------- | ------------------------------------------ |
| Apr 11 | DONG Jin | Converted the template & the first version |
|        |          |                                            |

## Introduction

<!--
Specify the aim of the document from the module’s point of view.
-->

## Services

<!--
Specify the services provided and/or required by the module.
-->

 

### Services Provided



| Service                                            | Provided  By   | Tested  By |
| -------------------------------------------------- | -------------- | ---------- |
| User can get the general model                     | Pretrain       | T1         |
| User can fine-tune the model fo each specific user | Fine-tuneModel | T2         |
| User can get the future motion state               | PredictResult  | T3         |



 

### Access Method 

| **Access   Method**                                 | **Parameter   name** | **Parameter   type** | **Description**                                              | **Exceptions** | **Map to services** |
| --------------------------------------------------- | -------------------- | -------------------- | ------------------------------------------------------------ | -------------- | ------------------- |
| getPretrainedModel() : pretrain_model               | path                 | string               | path is the location of the prepared data                    |                | 1                   |
| fineTuneModel(model_before, datapath) : model_after | model_before,path    | nn.Module,string     | model_before is the original model which need to be fine-tuned. datapath indicates the location of data file. |                | 2                   |
| predictResult(model, real_time_data) : state        | model, data          | nn.Module,tensor     | tensor is the real-time data collected from sensors.         |                | 3                   |
|                                                     |                      |                      |                                                              |                |                     |

 

### Access Method Effects

| **Access   Method** | **Description**                                              |
| ------------------- | ------------------------------------------------------------ |
| getPretrainedModel  | Server can train the general model for classification and prediction by using the prepared data in the database. |
| fineTuneModel       | For each specific person, the user can train their model by using their own data. |
| predictResult       | After the data sent to the server, the server can call predictResult method to get the next motion state of the user. |
|                     |                                                              |

 

### Services Required

<!--
Format similar to Sec. 2.1.
-->

| **Access   Method** | **Parameter   name** | **Parameter   type** | **Description** | **Exceptions** | **Map to services** |
| ------------------- | -------------------- | -------------------- | --------------- | -------------- | ------------------- |
|                     |                      |                      |                 |                |                     |



## Local Types

<!--
Specify the data formats inside/between the module(s).
-->

| **Type** | **Value Space** |
| -------- | --------------- |
|          |                 |
|          |                 |



## Interface Design Issues

<!--
Describe any design issues that arose during development. Describe the alternatives and the rationale for the alternative chosen.
-->

Eg. 1. Whether there is a database of email we can store all the emails there?



## Test Cases

<!--
Characterize the expected value of the outputs over sets calls to the module.
-->

#### T1

| **Step** |  **Description**   | **Input Type/Value** | **Expected Results**               | **Service** | **Preamble** |
| :------: | :----------------: | -------------------- | ---------------------------------- | ----------- | ------------ |
|    1     | getPretrainedModel | Path                 | the server get the original model. |             | 1            |
|          |                    |                      |                                    |             |              |
|          |                    |                      |                                    |             |              |

#### T2

| **Step** | **Description** | **Input Type/Value** | **Expected Results**               | **Service** | **Preamble** |
| -------- | --------------- | -------------------- | ---------------------------------- | ----------- | ------------ |
| 1        | fineTuneModel   | model_before,path    | the server get the fine-tune model |             | 2            |

####  T3

| **Step** | **Description** | **Input Type/Value** | **Expected Results**                             | **Service** | **Preamble** |
| -------- | --------------- | -------------------- | ------------------------------------------------ | ----------- | ------------ |
| 1        | predictResult   | model, data          | the server get the next motion state of the user |             | 3            |