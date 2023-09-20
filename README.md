
# Features 💡
- Best Image Classification Model Finder using Transfer Learning
- Paramater Domination tools for Image Classification Model Finder(update_data_path, update_train_folder_path, update_device_memory, split, ...)
- Data Management Tools (Image Data Splitter, Duplicate Image Finder, Duplicate Image Deleter, Similar Image Finder, Similar Image Deleter, Image Count Equalizer, RGB Converter, Image Resizer, Image Data Augmentation Tool, Label to txt splitter, Image to txt splitter, Image and Label to txt splitter, ....)
- Data Collection Tools (Simple Data Collector, Text Data Collector, Image Data Collector, ...)
- Developer Tools (Image Data Loader, Image Data Counter, ...)

# Image Classification Model Finder
Burobot is a powerful Python library for developing and testing deep learning models. It helps you find the best image classification model using transfer learning.

In this section, we will examine a Python code example for finding an image classification model using Burobot.

First, let's import the necessary libraries:

Python
from Burobot.Training.ImageClassification import TransferLearning as TL
import tensorflow as tf

Then, let's define the model parameters using a TL.Params object:

Python
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

This code defines the following parameters:

dense_units: Number of units in the dense layers.
dense_count: Number of dense layers.
conv_filters: Number of filters in the convolutional layers.
conv_count: Number of convolutional layers.
conv_layer_repeat_limit: Maximum number of times convolutional layers are repeated.
drop_outs: Dropout rates.
activation_functions: Activation functions.
loss_functions: Loss functions.
optimizers: Optimization algorithms.
output_activation_functions: Activation functions in the output layer.
frozen_layers: Layers to be frozen.
base_models: Base models to be used for transfer learning.
epochs: Number of epochs.
Next, let's define the save_to_path and data_path variables:


save_to_path = "save_to_path"
data_path = "data path"

These variables specify the path where the trained model will be saved and the path to the training data, respectively.

Finally, let's find the best image classification model using the TL.FindModel() function:

best_model, best_acc, best_values, best_history = TL.FindModel(
    stop_massages=False,
    params = params,
    use_multiprocessing=True
    ,data_path=data_path,
    device_memory=6,
    model_name="Model",
    patience=3,
    save_to_path=save_to_path
)
This code calls the TL.FindModel() function. This function finds and trains the best image classification model using the parameters given as arguments.

The function returns the following values:

best_model: The best image classification model.
best_acc: The best validation accuracy.
best_values: A dictionary containing the best values for each parameter.
best_history: A history object containing the training and validation accuracy/loss values for each epoch.
