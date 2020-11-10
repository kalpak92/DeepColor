# Deep Color

## Program Execution

```shell
python main.py
```



## Folder Structure

```tex
DeepColor
   ├── Colorize_deep.py
   ├── Colorizer.py
   ├── Colorizer_Manager.py
   ├── Constants.py
   ├── Regressor.py
   ├── Regressor_Manager.py
   ├── buildDataset.py
   ├── utils.py
   ├── main.py
   ├── README.md
   ├── Model
   │   ├── Colorizer
   │   └── Regressor
   ├── Plots
   │   ├── Colorizer
   │   ├── Final
   │   └── Regressor
   ├── __pycache__
   ├── data
   │   ├── test
   │   └── train
   ├── face_images
   ├── outputs_sigmoid
   │   ├── color
   │   └── gray
   └── outputs_tanh
       ├── color
       └── gray
```



## Architecture

The Colorizer Network is defined in `Colorizer.py` which internally invokes the Regressor model written in `Regressor.py`, responsible for extracting the features from the batch of images.

`ColorizerManager.py` has the `train` and `test` method which is responsible to training and testing the models from the respective ***train*** and ***test*** datasets provided. 

First, it receives the data in the defined `batch-size` from the `DataLoader` , retreives the hyperparameters from `Constants.py` and then trains the networks for the given number of epochs.

Once the network is trained, we ***test*** the model by loading the test data and subsequently store the outputs under the `outputs_sigmoid` and `outputs_tanh` folders for the respective runs as per requirement.



For the Regressor, there is a separate `Regressor_Manager.py` that trains and tests the model to generate the required performance and plots the ***Loss function*** over the epochs. The plot of the same is stored under `Plots->Regressor` .



## Outputs

1. #### Regressor

   The regressor outputs two scalar values which are the average of the `a` and `b` channel for each of the images.

2. #### Colorizer

   The colorizer output color images and stores the same in the respective `output` folder based on the activation function. The grayscale images are stored under the **gray** folder and the colorized images goes into the **color** folder.

   

## GPU

The code extracts the `device` at the start of execution using ` torch.cuda.is_available()` and loads the tensors accordingly, leveraging the compute based on availability.



## Summary





## Contributors

- [Kalpak Seal](https://github.com/kalpak92)
- [Shantanu Ghosh](https://github.com/Shantanu48114860)

