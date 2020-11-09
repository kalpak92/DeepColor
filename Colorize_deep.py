from Colorizer_Manager import Colorizer_Manager
from Constants import Constants
from Regressor_Manager import Regressor_Manager
from utils import Utils
from itertools import product

class Colorize_deep:
    def train_regressor(self, augmented_dataset_batch, device):
        regressor_train_arguments = {
            "data_loader": augmented_dataset_batch,
            "saved_model_path": Constants.REGRESSOR_SAVED_MODEL_PATH,
            "epochs": Constants.REGRESSOR_EPOCH,
            "learning_rate": Constants.REGRESSOR_LR,
            "weight_decay": Constants.REGRESSOR_WEIGHT_DECAY,
            "in_channel": Constants.REGRESSOR_IN_CHANNEL,
            "hidden_channel": Constants.REGRESSOR_HIDDEN_CHANNEL,
            "out_dims": Constants.REGRESSOR_OUT_DIMS,
            "loss_plot_path": Constants.REGRESSOR_LOSS_PLOT_PATH
        }

        regressor_manager = Regressor_Manager()
        regressor_manager.train(regressor_train_arguments, device)

    def test_regressor(self, augmented_dataset_batch, device):
        regressor_arguments = {
            "data_loader": augmented_dataset_batch,
            "saved_model_path": Constants.REGRESSOR_SAVED_MODEL_PATH,
            "in_channel": Constants.REGRESSOR_IN_CHANNEL,
            "hidden_channel": Constants.REGRESSOR_HIDDEN_CHANNEL,
            "out_dims": Constants.REGRESSOR_OUT_DIMS,
            "loss_plot_path": Constants.REGRESSOR_LOSS_PLOT_PATH
        }

        regressor_manager = Regressor_Manager()
        regressor_manager.test(regressor_arguments, device)

    def train_colorizer(self, augmented_dataset_batch,
                        activation_function, model_name,
                        device):
        parameters = Utils.get_hyperparameters()
        for lr, weight_decay, epoch in product(*parameters):
            print("Run - epoch: {0} lr: {1} weight_decay:{2}".format(epoch, lr, weight_decay))
            colorizer_train_arguments = {
                "train_data_loader": augmented_dataset_batch,
                # "val_data_loader": augmented_dataset_batch_val,
                "saved_model_path": model_name.format(epoch, lr, weight_decay),
                "epochs": epoch,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "in_channel": Constants.COLORIZER_IN_CHANNEL,
                "hidden_channel": Constants.COLORIZER_HIDDEN_CHANNEL,
                "loss_plot_path": Constants.COLORIZER_LOSS_PLOT_PATH
            }

            colorizer_manager = Colorizer_Manager()
            colorizer_manager.train(colorizer_train_arguments,
                                    activation_function, device)

    def test_colorizer(self, augmented_dataset_batch,
                       activation_function, save_path, model_name, device):
        parameters = Utils.get_hyperparameters()
        for lr, weight_decay, epoch in product(*parameters):
            print("----"*10)
            print("Run - epoch: {0} lr: {1} weight_decay:{2}".format(epoch, lr, weight_decay))
            colorizer_train_arguments = {
                "data_loader": augmented_dataset_batch,
                "saved_model_path": model_name.format(epoch, lr, weight_decay),
                "in_channel": Constants.COLORIZER_IN_CHANNEL,
                "hidden_channel": Constants.COLORIZER_HIDDEN_CHANNEL,
                "loss_plot_path": Constants.COLORIZER_LOSS_PLOT_PATH,
            }

            colorizer_manager = Colorizer_Manager()
            colorizer_manager.test(colorizer_train_arguments,
                                   activation_function, save_path,
                                   device, lr, weight_decay, epoch)
