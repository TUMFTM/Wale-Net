## Online Configs
This config folder holds the configuration files that are needed for inference and online learning.
A description of all configuration parameters if given below.

| Parameter | Description | Example / Range|
|---|---|---|
|pred_config_path | Path to the config that was used for training, which holds relevant model informations |"mod_prediction/configs/best_config.json"|
|pred_model_path | Path to the model that should be used for prediction | "mod_prediction/trained_models/best_model.tar"|
|gpu |String that specifies the system's GPU to run the training on| "" or "0" |
|min_obs_length | First timestep a prediction step will be performed. Before this timestep the ground truth is predicted | 0 |
|on_pred_learn_method | Method for online learning | null, "switch_layer" or "parallel_head"|
|on_pred_horizon | List of integers that indicates when an online learning step is performed | [10, 20, 30, 40] |
|on_lr | Learning rate for online learning | 2e-4 |
|on_pred_learn_density | Only the i-th prediction will be used for online learning | 5 |
|online_layer | Number of online layers in the switch layer method | 10 |
|on_loss | Loss for online learning | e.g. "NLL" or "MSE"|
|on_optimizer | Optimizer for online learning | e.g. "SGD" or "Adam" |
|on_train_loss_threshold | Threshold for online learning | null or 0 |
