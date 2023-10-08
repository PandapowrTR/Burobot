
# Features 💡
- Best Image Classification Model Finder using Transfer Learning
- Paramater Domination tools for Image Classification Model Finder(update_data_path, update_train_folder_path, update_device_memory, split, ...)
- Data Management Tools (Image Data Splitter, Duplicate Image Finder, Duplicate Image Deleter, Similar Image Finder, Similar Image Deleter, Image Count Equalizer, RGB Converter, Image Resizer, Image Data Augmentation Tool, Label to txt splitter, Image to txt splitter, Image and Label to txt splitter, ....)
- Data Collection Tools (Simple Data Collector, Text Data Collector, Image Data Collector, ...)
- Developer Tools (Image Data Loader, Image Data Counter, ...)
    
Burobot is a powerful Python library for developing and testing deep learning models. It helps you find the Best Artifical intelegent models.

# Natural Language Processing 🆕
First import Burobot Best NLP Model Finder
```py
from Burobot.Training.NLP import Learning as NLP
```
Then, let's define the model parameters using a NLP.Params object:
```py
params = NLP.Params(
    embedding_outputs=[100, 200, 300, 500, 800],
    lstm_units=[256, 512],
    lstm_activation_functions=NLPL.Params.get_activation_functions()[:1],
    lstm_count_limit=2,
    dense_units=[256, 512],
    dense_count_limit=3,
    dense_activation_functions=NLPL.Params.get_activation_functions()[:1],
    output_activation_functions=NLPL.Params.get_output_activation_functions()[:1],
    loss_functions=NLPL.Params.get_loss_functions()[:1],
    optimizers=NLPL.Params.get_optimizers()[:1],
    epochs=500
)
```
This code defines the following parameters:

- embedding_outputs: The number of dimensions to use for the embedding layer.
- lstm_units: The number of units in the LSTM layer.
- lstm_activation_functions: The activation function to use in the LSTM layer.
- lstm_count_limit: The maximum number of LSTM layers to use in the model.
- dense_units: The number of units in the dense layer.
- dense_count_limit: Number of dense layers.
- dense_activation_functions: The activation function to use in the dense layer.
- output_activation_functions: The activation function to use in the output layer.
- loss_functions: The loss function to use to train the model.
- optimizers: The optimizer to use to train the model.
- epochs: The number of epochs to train the model for.

```py
Next, let's define the save_to_path , data_path and test_path variables:
```
save_to_path = "path/to/save"
data_path = "path/to/data.json"
test_data = "path/to/test.json"
```
These variables specify the path where the trained model will be saved, the path to the training data and the path to the test data, respectively.

Finally, let's find the best natural language processing model using the NLP.FindModel() function:
```py
best_model, best_acc, best_values, best_history, tokenizer = NLPL.FindModel(
    params, data_path, test_data, save_to_path, model_name, 3000, stop_massages=False
)
```
- params: The params object
- data_path: Model dataset path
- test_data: Model test dataset path
- save_to_path:Path to Train folder be saved
- model_name:To be trained model name
- tokenizer_num_words:num words param for tokenizer
- gpu:Option to use gpu. Default True
- stop_massages:Stop option in messages. Default True 

# Image Classification Model Finder 
# Image Classification Model Finder NEW MODEL 👨

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
