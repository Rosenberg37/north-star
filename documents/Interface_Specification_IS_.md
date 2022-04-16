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



| Service                                                      | Provided  By | Tested  By |
| ------------------------------------------------------------ | ------------ | ---------- |
| 1.server can train the general model                         | train        | T1         |
| 2.server can train the personalized model for specific users | train        | T2         |
| 3.server can predict the next motion state                   | predict      | T3         |



 

### Access Method 

| **Access   Method**          | **Parameter   name**                     | **Parameter   type** | **Description**                                              | **Exceptions** | **Map to services** |
| ---------------------------- | ---------------------------------------- | -------------------- | ------------------------------------------------------------ | -------------- | ------------------- |
| get_model_instance_for_train | haper_params,[params]                    | Dict,[Dict]          | haper_params can control some data in the training process by changing values such as learning rate. |                | 1,2                 |
| get_data                     | path_data                                | String               | path_data provides the location of data and this method can get original data. |                | 1,2                 |
| get_real_time_data           | socket                                   | Socket               | socket continually provide real-time data                    |                | 3                   |
| process_data                 | original_data                            | Dataframe            | when training, data is [n,1800], n is uncertain; when predicting, data is [1,n], containg real-time data for 5 seconds. |                | 1,2,3               |
| get_model_params             | path_params                              | String               | path_params provide the location for one model's params.     |                | 2,3                 |
| save_model_params            | trained_model_params, path_to_save_model | Dict, String         | after the model has been trained, the params will be saved   |                | 1,2                 |
| save_predict_result          | next_motion_state                        | Integer              | next_motion_state will be one of 0,1,2,3,4,5                 |                | 3                   |

 

### Access Method Effects

| **Access   Method**          | **Description**                                              |
| ---------------------------- | ------------------------------------------------------------ |
| get_model_instance_for_train | when training or predicting, one model will be created.      |
| get_data                     | before training, original data will be achieved by file.     |
| get_real_time_data           | before predicting, original data will be achieved by socket. |
| process_data                 | after get original data, this method will prepocessing the data and then generate data for training or predicting, the type of data is tensor. |
| get_model_params             | before geting the instance of model, params may be needed.   |
| save_model_params            | after training process, the params need to be saved.         |
| save_predict_result          | after predicting process, the result need to be saved.       |

 

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

| **Step** | **Description** | **Input Type/Value** | **Expected Results** | **Service** | **Preamble** |
| :------: | :-------------: | -------------------- | -------------------- | ----------- | ------------ |
|          |                 |                      |                      |             |              |
|          |                 |                      |                      |             |              |
|          |                 |                      |                      |             |              |

#### T2

| **Step** | **Description** | **Input Type/Value** | **Expected Results** | **Service** | **Preamble** |
| -------- | --------------- | -------------------- | -------------------- | ----------- | ------------ |
|          |                 |                      |                      |             |              |

####  T3

| **Step** | **Description** | **Input Type/Value** | **Expected Results** | **Service** | **Preamble** |
| -------- | --------------- | -------------------- | -------------------- | ----------- | ------------ |
|          |                 |                      |                      |             |              |