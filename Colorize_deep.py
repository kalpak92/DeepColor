from Constants import Constants
from Colorizer_Manager import Colorizer_Manager
from Regressor_Manager import Regressor_Manager


class Colorize_deep:
    def execute_regressor(self, augmented_dataset_batch, device):
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

    def execute_colorizer(self, augmented_dataset_batch, device):
        colorizer_train_arguments = {
            "data_loader": augmented_dataset_batch,
            "saved_model_path": Constants.COLORIZER_SAVED_MODEL_PATH,
            "epochs": Constants.COLORIZER_EPOCH,
            "learning_rate": Constants.COLORIZER_LR,
            "weight_decay": Constants.COLORIZER_WEIGHT_DECAY,
            "in_channel": Constants.COLORIZER_IN_CHANNEL,
            "hidden_channel": Constants.COLORIZER_HIDDEN_CHANNEL,
            "loss_plot_path": Constants.COLORIZER_LOSS_PLOT_PATH
        }

        colorizer_manager = Colorizer_Manager()
        colorizer_manager.train(colorizer_train_arguments, device)
