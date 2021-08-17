from __future__ import print_function, division
import torch


def outputActivation(x):
    """Custom activation for output layer (Graves, 2015)

    Arguments:
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    muX = x[:, :, 0:1]
    muY = x[:, :, 1:2]
    sigX = x[:, :, 2:3]
    sigY = x[:, :, 3:4]
    rho = x[:, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=2)
    return out


def MSE(y_pred, y_gt):
    """MSE Loss for single outputs.

    Arguments:
        y_pred {[type]} -- [description]
        y_gt {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    # If GT has not enough timesteps, shrink y_pred
    if y_gt.shape[0] < y_pred.shape[0]:
        y_pred = y_pred[: y_gt.shape[0], :, :]

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    mse_det = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    count = torch.sum(torch.ones(mse_det.shape))
    mse = torch.sum(mse_det) / count
    return mse, mse_det


def MSE2(y_pred1, y_pred2, probs, y_gt):
    """MSE loss for multiple outputs

    Arguments:
        y_pred1 {[type]} -- [description]
        y_pred2 {[type]} -- [description]
        probs {[type]} -- [description]
        y_gt {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    mse1, mse_det1 = MSE(y_pred1, y_gt)
    mse2, mse_det2 = MSE(y_pred2, y_gt)
    mse_det = probs[:, 0] * mse_det1 + probs[:, 1] * mse_det2
    count = torch.sum(torch.ones(mse_det.shape))
    mse = torch.sum(mse_det) / count
    return mse, mse_det


def NLL(y_pred, y_gt):
    """NLL loss for single output

    Arguments:
        y_pred {[type]} -- [description]
        y_gt {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    # If GT has not enough timesteps, shrink y_pred
    if y_gt.shape[0] < y_pred.shape[0]:
        y_pred = y_pred[: y_gt.shape[0], :, :]

    muX = y_pred[:, :, 0]
    muY = y_pred[:, :, 1]
    sigX = y_pred[:, :, 2]  # actually 1/ sigma_x
    sigY = y_pred[:, :, 3]  # actually 1/sigma_y
    rho = y_pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = y_gt[:, :, 0]
    y = y_gt[:, :, 1]
    out_det = torch.pow(ohr, 2) * (
        torch.pow(sigX, 2) * torch.pow(x - muX, 2)
        + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
        - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)
    ) - torch.log(sigX * sigY * ohr)
    count = torch.sum(torch.ones(out_det.shape))
    out = torch.sum(out_det) / count
    return out, out_det


def NLL2(y_pred1, y_pred2, probs, y_gt):
    """NLL loss for multiple trajectory outputs

    Arguments:
        y_pred1 {[type]} -- [description]
        y_pred2 {[type]} -- [description]
        probs {[type]} -- [description]
        y_gt {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    nll1, nll_det1 = NLL(y_pred1, y_gt)
    nll2, nll_det2 = NLL(y_pred2, y_gt)
    nll_det = probs[:, 0] * nll_det1 + probs[:, 1] * nll_det2
    count = torch.sum(torch.ones(nll_det2.shape))
    nll = torch.sum(nll_det) / count
    return nll, nll_det


def multi_loss(y_pred1, y_pred2, probs):
    """This loss should encourage the network to propose two different predictions without a probibility being close to 0.

    Arguments:
        y_pred1 {[torch tensor]} -- [description]
        y_pred2 {[torcj tensor]} -- [description]
        probs {[torch tensor]} -- [description]
    """

    # predicted trajectories should not match
    mse, _ = MSE(y_pred1, y_pred2)
    loss = 1 / mse * (-torch.log(probs[:, 0] * probs[:, 1]))
    loss = torch.sum(loss) / torch.sum(torch.ones(loss.shape))
    return loss


def logsumexp(inputs, dim=None, keepdim=False):
    """Helper function for log sum exp calculation

    Arguments:
        inputs {[type]} -- [description]

    Keyword Arguments:
        dim {[type]} -- [description] (default: {None})
        keepdim {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """

    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def add_histograms(writer, net, global_step=0):
    """Add histograms of weights to tensorboard

    Arguments:
        writer {[tensorboard writer]} -- [Tensorboard writer]
        net {[torch network]} -- [Prediction network]

    Keyword Arguments:
        global_step {int} -- [Training epoch] (default: {0})

    Returns:
        writer {[tensorboard writer]} -- [Tensorboard writer]
    """
    writer.add_histogram("conv_3x1", net.conv_3x1.weight, global_step=global_step)
    writer.add_histogram(
        "dec_lstm_0", net.dec_lstm.all_weights[0][0], global_step=global_step
    )
    writer.add_histogram(
        "dec_lstm_1", net.dec_lstm.all_weights[0][1], global_step=global_step
    )
    writer.add_histogram(
        "dec_lstm_2", net.dec_lstm.all_weights[0][2], global_step=global_step
    )
    writer.add_histogram(
        "dec_lstm_3", net.dec_lstm.all_weights[0][3], global_step=global_step
    )
    writer.add_histogram("dyn_emb", net.dyn_emb.weight, global_step=global_step)
    writer.add_histogram(
        "enc_lstm_0", net.enc_lstm.all_weights[0][0], global_step=global_step
    )
    writer.add_histogram(
        "enc_lstm_1", net.enc_lstm.all_weights[0][1], global_step=global_step
    )
    writer.add_histogram(
        "enc_lstm_2", net.enc_lstm.all_weights[0][2], global_step=global_step
    )
    writer.add_histogram(
        "enc_lstm_3", net.enc_lstm.all_weights[0][3], global_step=global_step
    )
    writer.add_histogram("ip_emb", net.ip_emb.weight, global_step=global_step)
    writer.add_histogram("soc_conv", net.soc_conv.weight, global_step=global_step)

    return writer
