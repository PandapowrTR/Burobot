
# Features 💡
- Best Image Classification Model Finder using Transfer Learning
- Paramater Domination tools for Image Classification Model Finder(update_data_path, update_train_folder_path, update_device_memory, split, ...)
- Data Management Tools (Image Data Splitter, Duplicate Image Finder, Duplicate Image Deleter, Similar Image Finder, Similar Image Deleter, Image Count Equalizer, RGB Converter, Image Resizer, Image Data Augmentation Tool, Label to txt splitter, Image to txt splitter, Image and Label to txt splitter, ....)
- Data Collection Tools (Simple Data Collector, Text Data Collector, Image Data Collector, ...)
- Developer Tools (Image Data Loader, Image Data Counter, ...)
# Image Classification Model Finder 
    
Burobot is a powerful Python library for developing and testing deep learning models. It helps you find the best image classification model using transfer learning.

In this section, we will examine a Python code example for finding an image classification model using Burobot.

# Image Classification Model Finder NEW MODEL 🆕

First, let's import the necessary libraries
```py
from Burobot.Training.ImageClassification import TransferLearning as TL
import tensorflow as tf
```

Then, let's define the model parameters using a TL.Params object:

```py
params = TL.Params(
    dense_units=[256, 512],
    dense_count=list(range(2,3)),
    conv_filters=[256, 512],
    conv_count=list(range(0, 2)),
    conv_layer_repeat_limit=2, 
    drop_outs=[0.3, 0.5, 0.8],
    activation_functions=TL.Params.get_activation_functions()[:1],
    loss_functions= [TL.Params.get_loss_functions()[1]],
    optimizers=[tf.keras.optimizers.Adam],
    output_activation_functions=TL.Params.get_output_activation_functions()[1:3],
    frozen_layers=[0.2, 0.5, 0.9],
    base_models=[tf.keras.applications.MobileNet, tf.keras.applications.MobileNetV2],
    epochs=60
)
```
This code defines the following parameters:

- dense_units: Number of units in the dense layers.
- dense_count: Number of dense layers.
- conv_filters: Number of filters in the convolutional layers.
- conv_count: Number of convolutional layers.
- conv_layer_repeat_limit: Maximum number of times convolutional layers are repeated.
- drop_outs: Dropout rates.
- activation_functions: Activation functions.
- loss_functions: Loss functions.
- optimizers: Optimization algorithms.
- output_activation_functions: Activation functions in the output layer.
- frozen_layers: Layers to be frozen.
- base_models: Base models to be used for transfer learning.
- epochs: Number of epochs.
  
Next, let's define the save_to_path and data_path variables:
```py
save_to_path = "save_to_path"
data_path = "data path"
split_to_data_path = "split_to_data_path"
```
These variables specify the path where the trained model will be saved, the path to the training data and the path to the data be splitted, respectively.

Finally, let's find the best image classification model using the TL.FindModel() function:
```py
best_model, best_acc, best_values, best_history = TL.FindModel(
    stop_massages=False,
    params = params,
    use_multiprocessing=True
    ,data_path=data_path,
    device_memory=6,
    model_name="Model",
    patience=3,
    save_to_path=save_to_path,
    gpu=True,
    split_to_data_path=split_to_data_path,
    data_split_ratio=(0.6, 0.2)
)
```
This code calls the TL.FindModel() function. This function finds and trains the best image classification model using the parameters given as arguments.
This code defines the following parameters:
- stop_massages:Stop option in messages. Default True 
- params:The params object or file
- use_multiprocessing:Option to use multiprocessing for tensorflow. Default is False
- data_path:Model dataset path
- device_memory:gpu/cpu memory (GB)
- model_name:To be trained model name
- patience:Number of rights to be granted for each base model. If a model fails, is less successful than the most successful model, and overfits, the patience value is reduced by 1. If the patience value is 0, that base model is skipped
- save_to_path:Path to Train folder be saved
- gpu:Option to use gpu. Default True
- split_to_data_path:Path to the data be splitted. Default is current dir
- data_split_ratio:Data split ratio for splitting data. Default is (0.7, 0.1) => 70% train, 10%test, 20%validation (calculated with: 1- (train+test))
  
The function returns the following values:

- best_model: The best image classification model.
- best_acc: The best validation accuracy.
- best_values: A dictionary containing the best values for each parameter.
- best_history: A history object containing the training and validation accuracy/loss values for each epoch.

# Image Classification Model Finder OLD MODEL 👴
In Burobot you can continue model training with *Unused Params File*.
First, let's import the necessary libraries
```py
from Burobot.Training.ImageClassification import TransferLearning as TL
import tensorflow as tf
```
Then, let's define the model parameters using a TL.Params object:

```py
params = "/your/params/file/path.txt"
```
This code defines the following parameters:

- params: The Unused Params File path

And thats it! You can continue traning!
```py
best_model, best_acc, best_values, best_history = TL.FindModel(
    params = params
)
```
This code defines the following parameters:
- params:The params object or file
  
And also you can dominate params file! Using TL.Params.dominate_params_file()
```py
from Burobot.Training.ImageClassification import TransferLearning as TL
TL.Params.dominate_params_file("/path/to/params/file.txt", "/path/to/Train/Folder/path").split(divide=2)
TL.Params.dominate_params_file("/path/to/params/file.txt", "/path/to/Train/Folder/path").update_data_path("path/to/new/data/path" or ["path/to/new/data/train", "path/to/new/data/test", "path/to/new/data/val"])
TL.Params.dominate_params_file("/path/to/params/file.txt", "/path/to/Train/Folder/path").update_device_memory(1)
TL.Params.dominate_params_file("/path/to/params/file.txt", "/path/to/OldTrain/Folder/path").update_train_folder_path("path/to/new/train/path")
```

You can dominate values with this way to
```py
best_model, best_acc, best_values, best_history = TL.FindModel(
    params = params,
    use_multiprocessing=True
    device_memory=1,
)
```
In this code device_memory updated to 1
