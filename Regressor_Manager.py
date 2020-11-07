import matplotlib.pyplot as plt
import torch.utils.data
from torch import nn, optim

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

        print("..Training started..")
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

        self.plot_loss_epoch(loss_train, loss_plot_path)
        torch.save(model.state_dict(), saved_model_path)

    @staticmethod
    def plot_loss_epoch(train_loss_avg, fig_name):
        plt.ion()
        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()
