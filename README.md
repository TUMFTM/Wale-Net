# Wale-Net Prediction Network for CommonRoad

<img src="images/network_architecture.png" alt="Network architecture" width="800"/>

This repository provides an Recurrent Neural Network (RNN) for vehicle trajectory prediction with uncertainties. It builds up on the work of [Convolutional Social Pooling](https://github.com/nachiket92/conv-social-pooling). It has been adapted to CommonRoad and extended by the ability of scene understanding and online learning.
## Requirements

- Linux Ubuntu (tested on versions 16.04, 18.04 and 20.04)
- Python >=3.6

## Installation

This repository can be installed as a package for the pure usage of the prediction.
Otherwise this repository can be cloned and requirements installed for further development.
### A. As a package

Install package:
* `pip install --extra-index-url https://__token__:Bo8ChTzNxFdJG37x6j3K@gitlab.lrz.de/api/v4/projects/54930/packages/pypi/simple mod_prediction`


### B. As repository

Clone repository:
* `git clone https://gitlab.lrz.de/motionplanning1/mod_prediction.git`

Install requirements:
* `pip install -r requirements.txt`


## Deployment in Motion Planning Framework

1. After packet installation import the Prediction class from  `mod_precition` e.g. with `from mod_prediction import WaleNet`. Available classes for predcition are:
    * `Prediction` for ground truth predictions, uncertainties are zero.
    * `WaleNet` for probability-based LSTM prediction.
2. Initialize the class with a CommonRoad scenario with `predictor = WaleNet(<CommonRoad Scenario object>)`. *Optionally* provide a dictionary of [online_args](mod_prediction/configs/online/README.md) for the prediction with different models or for online learning*
3. Call `predictor.step(time_step, obstacle_id_list)` in a loop, wehere time_step is the current time_step of an CommonRoad scenario object and obstacle_id_list is a list of all the IDs of the dynamic obstacle that should be predicted. 
It outputs a dictionary in the following format:
    ```
    prediction_result = {
        obstacle_id: {
            'pos_list': [list] list of x,y positions of the predicted trajectory in m
            'cov_list': [list] list of 2x2 covariance matrices for uncertainties
        }
        ...
    }
    ```


4. *Optionally* call `predictor.get_positions()` or `predictor.get_positions()` to get a list of x,y positions or covariances of *all* predicted vehicles.


## Training

1. Create your desired [configuration](configs/README.md) for the prediction network and training. Start by making a copy of the [default.json](configs/default.json). 
2. Make sure your dataset is available, either downloaded or self-created. (See Data)
3. Execute `python train.py`. This will train a model on the given dataset specified in the configs. The result will be saved in `trained_models` and the logs in `tb_logs`
    *  `--config <path to your config>` to use your config. Per default `default.json` is used.

## Files

| File | Description |
|----|----|
main.py | Used to deploy the prediction network on a CommonRoad scenario.
train.py   | Used to train the prediction network. 
evaluate.py | Used to evaluate a trained prediction model on the test set.
evaluate_online_learning.py | Ordered evaluation of an online configuration on all the scenarios.

## Data

* The full dataset for training can be downloaded [here](https://syncandshare.lrz.de/dl/fiRsZFzqZAxEkWZEBHYJnSAt/commonroad.zip). Alternatively a new dataset can be generated with the `tools/commonroad_dataset.py` script.
* CommonRoad scenes can be downloaded [here](https://gitlab.lrz.de/tum-cps/commonroad-scenarios).


## Qualitative Examples

Below is an example visualization of the prediction on a scenario that was not trained on. 

![Exemplary Result](images/exemplary_result.gif)

## Inference time

Time for the prediction of a single vehicle takes around **10 ms** on NVIDIA V100 GPU and **23 ms** on an average laptop CPU.

## References

* Maximilian Geisslinger, Phillip Karle, Johannes Betz and Markus Lienkamp "Watch-and-Learn-Net: Self-supervised Online Learning for Vehicle Trajectory Prediction". *submitted to IEEE 2021 Intelligent Vehicles Symposium*
* Nachiket Deo and Mohan M. Trivedi,"Convolutional Social Pooling for Vehicle Trajectory Prediction." CVPRW, 2018

