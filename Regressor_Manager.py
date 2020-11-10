import torch.utils.data
from torch import nn, optim
import numpy as np
from Regressor import Regressor
from utils import Utils


class Regressor_Manager:
    def train(self, train_arguments, device):
        data_loader = train_arguments["data_loader"]
        saved_model_path = train_arguments["saved_model_path"]

        epochs = train_arguments["epochs"]
        lr = train_arguments["learning_rate"]
        weight_decay = train_arguments["weight_decay"]
        in_channel = train_arguments["in_channel"]
        hidden_channel = train_arguments["hidden_channel"]
        out_dims = train_arguments["out_dims"]
        loss_plot_path = train_arguments["loss_plot_path"]

        print("..Regressor training started..")
        model = Regressor(in_channel=in_channel,
                          hidden_channel=hidden_channel,
                          out_dims=out_dims,
                          train_mode="regressor").to(device)

        lossF = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
        loss_train = []

        # start training
        for epoch in range(epochs):
            total_loss = 0
            model.train()

            for batch in data_loader:
                l_channel, a_channel, b_channel = batch
                l_channel = l_channel.to(device)

                a_b_mean = Utils.get_ab_mean(a_channel, b_channel)
                a_b_mean_hat = model(l_channel)

                if torch.cuda.is_available():
                    loss = lossF(a_b_mean_hat.float().cuda(),
                                 a_b_mean.float().cuda()).to(device)
                else:
                    loss = lossF(a_b_mean_hat.float(),
                                 a_b_mean.float()).to(device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print("epoch: {0}, loss: {1}"
                  .format(epoch, total_loss))
            loss_train.append(total_loss)

        Utils.plot_loss_epoch(loss_train, loss_plot_path)
        torch.save(model.state_dict(), saved_model_path)

    def test(self, _arguments, device):
        data_loader = _arguments["data_loader"]
        saved_model_path = _arguments["saved_model_path"]

        in_channel = _arguments["in_channel"]
        hidden_channel = _arguments["hidden_channel"]
        out_dims = _arguments["out_dims"]

        print("..Regressor testing started..")

        model = Regressor(in_channel=in_channel,
                          hidden_channel=hidden_channel,
                          out_dims=out_dims,
                          train_mode="regressor").to(device)
        model.load_state_dict(torch.load(saved_model_path, map_location=device))

        a_list = []
        b_list = []
        lossF = nn.MSELoss()
        total_loss = 0
        loss_train = []
        for batch in data_loader:
            l_channel, a_channel, b_channel = batch
            l_channel = l_channel.to(device)

            a_b_mean = Utils.get_ab_mean(a_channel, b_channel)
            a_b_mean_hat = model(l_channel).detach()

            if torch.cuda.is_available():
                loss = lossF(a_b_mean_hat.float().cuda(),
                             a_b_mean.float().cuda()).to(device)
            else:
                loss = lossF(a_b_mean_hat.float(),
                             a_b_mean.float()).to(device)

            loss_train.append(loss.item())

            a_b_pred = a_b_mean_hat[0].cpu().numpy()
            a_list.append(a_b_pred[0])
            b_list.append(a_b_pred[1])

        print("MSE:", np.average(np.asarray(loss_train)))
        print("Image_num || Mean a || Mean b")
        for i in range(1, len(a_list)):
            print("Image: {0} mean_a: {1} mean_b:{2}".format(
                i, (a_list[i] * 255) - 128, (b_list[i] * 255) - 128
            ))