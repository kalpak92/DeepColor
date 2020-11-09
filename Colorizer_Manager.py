import torch.utils.data
from torch import nn, optim

from Colorizer import Colorizer
from utils import Utils


class Colorizer_Manager:
    def train(self, train_arguments, device):
        data_loader = train_arguments["data_loader"]
        saved_model_path = train_arguments["saved_model_path"]
        print(saved_model_path)

        epochs = train_arguments["epochs"]
        lr = train_arguments["learning_rate"]
        weight_decay = train_arguments["weight_decay"]
        in_channel = train_arguments["in_channel"]
        hidden_channel = train_arguments["hidden_channel"]
        loss_plot_path = train_arguments["loss_plot_path"]

        print("..Colorizer Training started..")
        model = Colorizer(in_channel=3, hidden_channel=3, is_RELU=True).to(device)

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

                a_b_channel = torch.cat([a_channel, b_channel], dim=1)
                a_b_channel_hat = model(l_channel)

                if torch.cuda.is_available():
                    loss = lossF(a_b_channel_hat.float().cuda(),
                                 a_b_channel.float().cuda()).to(device)
                else:
                    loss = lossF(a_b_channel_hat.float(),
                                 a_b_channel.float()).to(device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print("epoch: {0}, loss: {1}"
                  .format(epoch, total_loss))
            loss_train.append(total_loss)

        Utils.plot_loss_epoch(loss_train, loss_plot_path)
        torch.save(model.state_dict(), saved_model_path)

    def test(self, test_arguments, device):
        data_loader = test_arguments["data_loader"]
        saved_model_path = test_arguments["saved_model_path"]

        in_channel = test_arguments["in_channel"]
        hidden_channel = test_arguments["hidden_channel"]
        loss_plot_path = test_arguments["loss_plot_path"]

        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        display_image_path_orig = {'grayscale': 'display/gray/', 'color': 'display/color/'}
        display_image_path_reconstructed = {'grayscale': 'display/gray/',
                                            'color': "display/reconstructed_color/"}

        print("..Colorizer Training started..")
        model = Colorizer(in_channel=in_channel, hidden_channel=hidden_channel, is_RELU=True).to(device)
        model.load_state_dict(torch.load(saved_model_path, map_location=device))

        lossF = nn.MSELoss()
        serial_num = 0
        for batch in data_loader:
            serial_num += 1
            l_channel, a_channel, b_channel = batch
            l_channel = l_channel.to(device)

            a_b_channel = torch.cat([a_channel, b_channel], dim=1)
            a_b_channel_hat = model(l_channel).detach()

            loss = lossF(a_b_channel, a_b_channel_hat)
            print("Image: {0}, loss: {1}".format(serial_num, loss.item()))
            # print("l_channel: ", l_channel.size())
            # print(a_b_channel_hat.size())

            # image_original = torch.cat([l_channel, a_b_channel], dim=1)
            # image_reconst = torch.cat([l_channel, a_b_channel_hat], dim=1)

            # print(image_original.size())
            # print(image_reconst.size())
            # print(image_reconst)

            # Utils.show_img_tensor(image_original[0])

            save_name_orig = 'Orig_img_{0}.jpg'.format(serial_num)
            save_name_recons = 'Recons_img_{0}.jpg'.format(serial_num)

            Utils.to_rgb(l_channel[0], a_b_channel[0],
                         save_path=save_path, save_name=save_name_orig)
            Utils.to_rgb(l_channel[0], a_b_channel_hat[0],
                         save_path=save_path, save_name=save_name_recons)

            if serial_num % 7 == 0:
                display_image_original = {'grayscale': 'Orig_img_{0}.jpg'.format(serial_num),
                                          'color': 'Orig_color_img_{0}.jpg'.format(serial_num)}
                display_image_name_reconstructed = {'grayscale': 'Orig_img_{0}.jpg'.format(serial_num),
                                                    'color': 'Reconstructed_color_img_{0}.jpg'.format(serial_num)}

                Utils.to_rgb(l_channel[0], a_b_channel[0],
                             display_path=display_image_path_orig, display_name=display_image_original)
                Utils.to_rgb(l_channel[0], a_b_channel_hat[0],
                             display_path=display_image_path_reconstructed,
                             display_name=display_image_name_reconstructed)

            # Utils.show_img(torchvision.utils.make_grid(image_original))
            # Utils.show_img(torchvision.utils.make_grid(l_channel))
            # Utils.show_img(torchvision.utils.make_grid(image_reconst))

            # Utils.show_img_tensor(image_original[0])
            # Utils.show_img_tensor(image_reconst[0])
