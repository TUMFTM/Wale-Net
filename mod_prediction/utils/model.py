from __future__ import division
import torch
import torch.nn as nn
from .neural_network import outputActivation


class predictionNet(nn.Module):

    # Initialization
    def __init__(self, args, **kwargs):
        super(predictionNet, self).__init__()

        # Unpack arguments
        self.args = args

        # Use gpu flag & set device for parallel one hot net
        self.use_cuda = args["use_cuda"]
        if self.use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Sizes of network layers
        self.encoder_size = args["encoder_size"]
        self.decoder_size = args["decoder_size"]
        self.out_length = args["out_length"]
        self.grid_size = args["grid_size"]
        self.soc_conv_depth = args["soc_conv_depth"]
        self.conv_3x1_depth = args["conv_3x1_depth"]
        self.dyn_embedding_size = args["dyn_embedding_size"]
        self.input_embedding_size = args["input_embedding_size"]
        self.soc_embedding_size = (
            ((args["grid_size"][0] - 4) + 1) // 2
        ) * self.conv_3x1_depth
        self.scene_images = args["scene_images"]
        self.dec_img_size = args["dec_img_size"]
        self.online_layer = args["online_layer"]
        self.num_img_filters = args["num_img_filters"]
        self.enc_dec_layer = args["enc_dec_layer"]

        # Define network weights

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM/GRU
        if "lstm" in self.enc_dec_layer:
            self.enc_lstm = torch.nn.LSTM(
                self.input_embedding_size, self.encoder_size, 1
            )
        elif "gru" in self.enc_dec_layer:
            self.enc_lstm = torch.nn.GRU(
                self.input_embedding_size, self.encoder_size, 1
            )

        # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)

        # Convolutional social pooling layer and social embedding layer
        self.soc_conv = torch.nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3)
        self.conv_3x1 = torch.nn.Conv2d(
            self.soc_conv_depth, self.conv_3x1_depth, (3, 1)
        )
        self.soc_maxpool = torch.nn.MaxPool2d((2, 1), padding=(1, 0))

        if self.scene_images:
            # Convolutional processing of scene representation
            self.sc_conv1 = torch.nn.Conv2d(
                1, self.num_img_filters, kernel_size=3, stride=1, padding=1
            )
            self.sc_conv2 = torch.nn.Conv2d(
                self.num_img_filters,
                self.num_img_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.sc_conv3 = torch.nn.Conv2d(
                self.num_img_filters,
                self.num_img_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.sc_conv4 = torch.nn.Conv2d(
                self.num_img_filters,
                self.num_img_filters,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.sc_conv5 = torch.nn.Conv2d(
                self.num_img_filters,
                self.dec_img_size,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.sc_conv6 = torch.nn.Conv2d(
                self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
            )
            self.sc_conv7 = torch.nn.Conv2d(
                self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
            )
            self.sc_conv8 = torch.nn.Conv2d(
                self.dec_img_size, self.dec_img_size, kernel_size=3, stride=1, padding=1
            )

            self.sc_maxpool = torch.nn.MaxPool2d((2, 2), padding=(0, 0))

        # Decoder LSTM/GRU
        if self.scene_images:
            if "lstm" in self.enc_dec_layer:
                self.dec_lstm = torch.nn.LSTM(
                    self.soc_embedding_size
                    + self.dyn_embedding_size
                    + self.dec_img_size,
                    self.decoder_size,
                )
            elif "gru" in self.enc_dec_layer:
                self.dec_lstm = torch.nn.GRU(
                    self.soc_embedding_size
                    + self.dyn_embedding_size
                    + self.dec_img_size,
                    self.decoder_size,
                )

        else:
            if "lstm" in self.enc_dec_layer:
                self.dec_lstm = torch.nn.GRU(
                    self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size
                )
            elif "gru" in self.enc_dec_layer:
                self.dec_lstm = torch.nn.GRU(
                    self.soc_embedding_size + self.dyn_embedding_size, self.decoder_size
                )

        # Output layers:
        # Base layer
        self.op = torch.nn.Linear(self.decoder_size, 5)

        # online learning layer
        self.on_pred_learn_method = args.get("on_pred_learn_method", "default_decoder")

        switch_pred_func = {
            "default_decoder": lambda: self.decode,
            "switch_layer": self.on_init_change_output,
            "parallel_head": self.on_init_parallel_head,
        }

        # init layer for prediction method and get prediction function
        torch.manual_seed(0)
        init_layers = switch_pred_func.get(
            self.on_pred_learn_method, lambda: self.decode
        )
        self.pred_func = init_layers()

        # stack for seen obstacle ids
        self.seen_obstacle_ids = []
        self.assignment_dict = {}

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def on_init_change_output(self):
        self.online_dict_layers = {}
        # create online layers according to the input
        for i in range(self.online_layer):
            self.online_dict_layers["op_{0}".format(i)] = torch.nn.Linear(
                self.decoder_size, 5
            )

        return self.on_change_output_layer

    def on_init_parallel_head(self):
        #  Merge one-hote vector with parallel LSTM on Output Fully
        self.online_dict_layers = {}
        on_linear = torch.nn.Linear(self.online_layer, self.input_embedding_size)
        torch.nn.init.zeros_(on_linear.weight)
        torch.nn.init.zeros_(on_linear.bias)

        on_lstm = torch.nn.LSTM(self.input_embedding_size, self.decoder_size)

        self.online_dict_layers["op_"] = torch.nn.Sequential(on_linear, on_lstm).to(
            torch.device(self.device)
        )

        return self.on_parallel_head

    # Forward Pass
    def forward(self, hist, nbrs, sc_img, obstacle_id=None):

        # Forward pass hist:
        if "lstm" in self.enc_dec_layer:
            _, (hist_hidden_state, _) = self.enc_lstm(
                self.leaky_relu(self.ip_emb(hist))
            )
        elif "gru" in self.enc_dec_layer:
            _, hist_hidden_state = self.enc_lstm(self.leaky_relu(self.ip_emb(hist)))

        hist_enc = self.leaky_relu(
            self.dyn_emb(
                hist_hidden_state.view(
                    hist_hidden_state.shape[1], hist_hidden_state.shape[2]
                )
            )
        )

        # Forward pass nbrs
        if "lstm" in self.enc_dec_layer:
            _, (nbrs_hidden_state, _) = self.enc_lstm(
                self.leaky_relu(self.ip_emb(nbrs))
            )
        elif "gru" in self.enc_dec_layer:
            _, nbrs_hidden_state = self.enc_lstm(self.leaky_relu(self.ip_emb(nbrs)))
        nbrs_enc = nbrs_hidden_state.view(
            nbrs_hidden_state.shape[1], nbrs_hidden_state.shape[2]
        )

        if self.scene_images:
            # Forward pass sc_img
            sc_img = self.sc_maxpool(self.sc_conv1(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv2(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv3(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv4(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv5(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv6(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv7(sc_img))
            sc_img = self.sc_maxpool(self.sc_conv8(sc_img))

            sc_img = torch.squeeze(sc_img, 2)
            sc_img = torch.squeeze(sc_img, 2)

            # sc_img_plt = torch.squeeze(sc_img, 0).cpu().detach().numpy()

        # Masked scatter alternative
        soc_enc = nbrs_enc.reshape(
            hist.shape[1], self.grid_size[1], self.grid_size[0], self.encoder_size
        )
        soc_enc = soc_enc.permute(0, 3, 2, 1)

        # Apply convolutional social pooling:
        soc_enc = self.soc_maxpool(
            self.leaky_relu(self.conv_3x1(self.leaky_relu(self.soc_conv(soc_enc))))
        )
        soc_enc = soc_enc.view(-1, self.soc_embedding_size)

        # Concatenate encodings:
        enc = torch.cat((soc_enc, hist_enc), 1)
        if self.scene_images:
            enc = torch.cat((enc, sc_img), 1)

        fut_pred1 = self.pred_func(enc, obstacle_id)
        # Output: [mean_x, mean_y, std_x, std_y, correlation_coeff]

        return fut_pred1

    def decode(self, enc, obstacle_id):
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def birth_death_memory(self, obstacle_id):
        if obstacle_id not in self.seen_obstacle_ids:
            # assign internal id
            if len(self.seen_obstacle_ids) < self.online_layer:
                self.assignment_dict[obstacle_id] = len(self.seen_obstacle_ids)
                # add obstacle to the seen obsacle list
                self.seen_obstacle_ids.append(obstacle_id)

    def on_change_output_layer(self, enc, obstacle_id):
        # Function changes linear output layer depending on observed obstacle ID
        self.birth_death_memory(obstacle_id)
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)

        fut_pred = self.online_dict_layers[
            "op_{0}".format(self.assignment_dict[obstacle_id])
        ](h_dec)

        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def on_parallel_head(self, enc, obstacle_id):
        # Function create one-hot vector depending on observed obstacle ID and feeds it as additional input to the net
        # variant: Merge additional LSTM on Output Fully
        self.birth_death_memory(obstacle_id)
        enc = enc.repeat(self.out_length, 1, 1)
        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)

        latent_expand = torch.zeros(self.online_layer, device=torch.device(self.device))
        latent_expand[self.assignment_dict[obstacle_id]] = 1.0

        latent_expand = latent_expand.repeat(self.out_length, 1, 1)
        latent_feedin, _ = self.online_dict_layers["op_"](latent_expand)
        latent_feedin = latent_feedin.permute(1, 0, 2)

        fut_pred = self.op(h_dec + latent_feedin)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred
