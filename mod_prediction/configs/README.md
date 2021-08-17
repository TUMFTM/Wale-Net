## Configs
This config folder holds the configuration files that are needed for training, testing and evaluation.
A description of all configuration parameters if given below.

| Parameter | Description | Example / Range|
|---|---|---|
|gpu |String that specifies the system's GPU to run the training on| "" or "0" |
|worker | Number of workers for data loading | 0 or 12 etc.|
|encoder size | Size of encoder LSTM| 64 |
|decoder size | Size of decoder LSTM| 128 |
|in_length | Length of past time steps as input | 31 |
|out_length | Length of predicted time steps as output | 50|
|grid_size | Grid size of array that keeps the neighbor vehicle's trajectories| [13,3] |
|soc_conv_depth| Number of convolutional filters on the social tensor | 64 |
|conv_3x1_depth| Number of convolutional filters in the conv_3x1 layer | 16 |
|dyn_embedding_size| Size of vehicle dynamics embedding | 32 |
|input_embedding_size| Size of input embedding | 32|
|train_flag | true for training | true/false|
|save_path | Path to save the trained model | "trained_models/"|
|dataset | Dataset used for training | "commonroad600" |
|img_path| Path for scene images within dataset | "data/sc_imgs600"|
|save_best | Save model with lowest validation loss | true/false|
|lr_rmse | Learning rate for training with RMSE loss | 1.5e-4 |
|lr_nll| Learning rate for training with NLL loss | 1.5e-4|
|scene_images | Use scene images for training | true/false |
|tb_logs | Path to save the tensorboard log files | "tb_logs" |
|pretrainEpochs | Number of epochs with RMSE loss | 5 |
|trainEpochs | Number of epochs with NLL loss | 3 |
|dec_img_size| Size of decoded image | 32 |
|batch_size | Number of samples per batch | 128 |