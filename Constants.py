class Constants:
    REGRESSOR_EPOCH = 75
    REGRESSOR_SAVED_MODEL_PATH = "./Model/Regressor/Regressor.pth"
    REGRESSOR_LOSS_PLOT_PATH = "Plots/Regressor/Regressor_Loss_plot.jpeg"
    REGRESSOR_LR = 0.0001
    REGRESSOR_WEIGHT_DECAY = 1e-5
    REGRESSOR_IN_CHANNEL = 1
    REGRESSOR_HIDDEN_CHANNEL = 3
    REGRESSOR_OUT_DIMS = 2
    REGRESSOR_BATCH_SIZE_CPU = 32
    REGRESSOR_BATCH_SIZE_CUDA = 16

    COLORIZER_EPOCH = 100
    COLORIZER_SAVED_MODEL_PATH_TANH = "./Model/Colorizer/Colorizer_tanh_epoch_{0}_lr_{1}_weight_decay_{2}.pth"
    COLORIZER_SAVED_MODEL_PATH_RELU = "./Model/Colorizer/Colorizer_100_relu.pth"
    COLORIZER_SAVED_MODEL_PATH_SIGMOID = "./Model/Colorizer/Colorizer_sigmoid_epoch_{0}_lr_{1}_weight_decay_{2}.pth"
    COLORIZER_LOSS_PLOT_PATH = "Plots/Colorizer/Colorizer_Loss_plot.jpeg"
    COLORIZER_LR = 0.0001
    COLORIZER_WEIGHT_DECAY = 1e-5
    COLORIZER_IN_CHANNEL = 3
    COLORIZER_HIDDEN_CHANNEL = 3
    COLORIZER_OUT_DIMS = 2
    COLORIZER_BATCH_SIZE_CPU = 32
    COLORIZER_BATCH_SIZE_CUDA = 8

    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
