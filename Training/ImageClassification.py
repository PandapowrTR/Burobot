# BUROBOT
import os, gc, sys, itertools, time, psutil, threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except:
    pass
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput
from Burobot.tools import BurobotImageData
from Burobot.Data.Dominate import DominateImage


def test_model(model, test_data, device_memory: int, return_predictions: bool = False):
    """Test the model on test data and calculate accuracy.

    Args:
        model: The trained model.
        test_data: Test data in a suitable format.
        device_memory: Amount of device memory in GB.
        return_predictions: Whether to return prediction results.

    Returns:
        The test accuracy or -1 if memory threshold exceeded.
        If return_predictions is True, it also returns prediction results.
    """
    if device_memory <= 0:
        raise ValueError("device_memory must be larger than 0 🔢")
    test_class_counts = test_data.reduce(
        tf.zeros(len(test_data.class_names), dtype=tf.int32),
        lambda x, y: x + tf.reduce_sum(tf.cast(y[1], tf.int32), axis=0),
    ).numpy()
    test_class_counts = dict(zip(test_data.class_names, test_class_counts))
    test_size = 0

    test_acc = 0
    same_class_count = 0
    previous_prediction = 0
    predictions = []
    progress = 0
    for images, labels in test_data:
        progress += 1
        percentage = (progress / len(test_data)) * 100

        print("\r" + " " * 80, end="")
        print(f"\rTest Progress: %{percentage:.2f}", end="")
        for i in range(len(images)):
            test_size += 1
            img = np.expand_dims(images[i].numpy() / 255, 0)
            pred = np.argmax(model.predict(img, verbose=0))
            pred = test_data.class_names[pred]
            real = test_data.class_names[np.argmax(labels[i])]
            # if prediction is true
            if real == pred:
                test_acc += 1
                predictions.append(
                    {
                        "Real": real,
                        "Predicted": pred,
                        "result": True,
                    }
                )
            # if prediction is false
            else:
                predictions.append(
                    {
                        "Real": real,
                        "Predicted": pred,
                        "result": False,
                    }
                )
            # if previous prediction is equal to current prediction
            if previous_prediction == pred:
                # increase same class counter
                same_class_count += 1
            # if previous prediction is NOT equal to current prediction
            else:
                # reset the same class counter and previous prediction value
                previous_prediction = pred
                same_class_count = 0
            # if same class counter value is bigger or equal to 1.7 times more than previous prediction
            if same_class_count >= int(test_class_counts[previous_prediction] * 1.3):
                tf.keras.backend.clear_session()
                gc.collect()
                print("\r" + " " * 80, end="")
                print("\rTest Progress: %100", end="")
                print()
                del model
                if return_predictions:
                    return -1, predictions
                return -1
    class_predictions = {}
    class_counts = {}
    for p in predictions:
        if p["Predicted"] not in list(class_predictions.keys()):
            class_predictions[p["Predicted"]] = 1
        else:
            class_predictions[p["Predicted"]] += 1
        if p["Real"] not in list(class_counts.keys()):
            class_counts[p["Real"]] = 1
        else:
            class_counts[p["Real"]] += 1
    overfitting_threshold_big = 1.3
    overfitting_threshold_small = 0.1
    for key in class_predictions.keys():
        if class_counts[key] <= 20:
            overfitting_threshold_big = 1.5
            overfitting_threshold_small = 0.1
        elif class_counts[key] <= 100:
            overfitting_threshold_big = 1.4
            overfitting_threshold_small = 0.08
        elif class_counts[key] <= 200:
            overfitting_threshold_big = 1.3
            overfitting_threshold_small = 0.04
        elif class_counts[key] <= 500:
            overfitting_threshold_big = 1.2
            overfitting_threshold_small = 0.01
        elif class_counts[key] <= 1000:
            overfitting_threshold_big = 1.1
            overfitting_threshold_small = 0.01
        else:
            overfitting_threshold_big = 1.05
            overfitting_threshold_small = 0.01
        if (
            class_predictions[key] >= class_counts[key] * overfitting_threshold_big
            or class_predictions[key] <= class_counts[key] * overfitting_threshold_small
        ):
            if return_predictions:
                return -1, predictions
            return -1

    print()
    test_acc = test_acc / test_size * 100
    if return_predictions:
        return test_acc, predictions
    return test_acc


def _draw_model(
    history,
    predictions,
    model_name,
    model_output,
    acc,
    values,
    save_to_path,
    test_size=None,
):
    """Draw and save model training and validation history plots.

    Args:
        history: The training history of the model.
        model_name: Name of the model.
        model_output: Output file name for the plot.
        acc: Accuracy value or -1 for overfitting.
        values: Dictionary of additional values to display.
        save_to_path: Path to save the plot.
        test_size: Number of test items (optional).

    Returns:
        None
    """
    old_dir = os.getcwd()
    gs = GridSpec(2, 2)
    fig = plt.figure(figsize=(15, 12))

    # table 1
    ax1 = plt.subplot(gs[0])
    ax1.plot(history.history["val_accuracy"])
    ax1.plot(history.history["val_loss"])
    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["loss"])
    ax1.set_xlabel("epochs")
    ax1.legend(
        ["val_accuracy", "val_loss", "train_accuracy", "train_loss"], loc="upper left"
    )

    # table 2
    ax2 = plt.subplot(gs[1])
    vals = f"{model_name}\n"
    vals += "\n".join([f"{key}: {value}" for key, value in values.items()])
    vals += "\n" + "Accuracy: " + (("%" + str(acc)) if acc != -1 else "Overfitting")
    if test_size is not None:
        vals += "\nTest items: " + str(test_size)
    ax2.text(0.5, 0.4, vals, fontsize=16, ha="center")
    ax2.axis("off")

    # table 3
    ax3 = plt.subplot(gs[2])
    y_true = [item["Real"] for item in predictions]
    y_pred = [item["Predicted"] for item in predictions]
    cm = confusion_matrix(y_true, y_pred)
    class_labels = np.unique(np.concatenate((y_true, y_pred)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", ax=ax3, xticks_rotation="vertical")

    os.chdir(save_to_path)
    plt.savefig(model_output + ".jpg")

    if model_output + ".jpg" in os.listdir():
        os.remove(model_output + ".jpg")
    plt.savefig(model_output + ".jpg")
    plt.clf()
    os.chdir(old_dir)
    del old_dir


class TransferLearning:
    conv0 = False

    class Params:
        def __init__(
            self,
            dense_units=None,
            dense_count=None,
            conv_filters=None,
            conv_count=None,
            conv_layer_repeat_limit=None,
            drop_outs=None,
            activation_functions=None,
            loss_functions=None,
            optimizers=None,
            output_activation_functions=None,
            frozen_layers=None,
            base_models=None,
            epochs=None,
            **kwargs,
        ):
            if kwargs:
                dense_units = kwargs["kwargs"]["dense_units"]
                dense_count = kwargs["kwargs"]["dense_count"]
                conv_filters = kwargs["kwargs"]["conv_filters"]
                conv_count = kwargs["kwargs"]["conv_count"]
                conv_layer_repeat_limit = kwargs["kwargs"]["conv_layer_repeat_limit"]
                drop_outs = kwargs["kwargs"]["drop_outs"]
                frozen_layers = kwargs["kwargs"]["frozen_layers"]

                activation_functions = eval(
                    str(kwargs["kwargs"]["activation_functions"]).replace("'", "")
                )
                loss_functions = eval(
                    str(kwargs["kwargs"]["loss_functions"]).replace("'", "")
                )
                optimizers = eval(str(kwargs["kwargs"]["optimizers"]).replace("'", ""))
                output_activation_functions = eval(
                    str(kwargs["kwargs"]["output_activation_functions"]).replace(
                        "'", ""
                    )
                )
                base_models = eval(
                    str(kwargs["kwargs"]["base_models"]).replace("'", "")
                )

            def _check_base_models():
                if base_models is not None:
                    for base_model in base_models:
                        if base_model not in tf.keras.applications.__dict__.values():
                            raise Exception(
                                "Your base_models value is not true. Please use models from tensorflow.keras.applications 🎚️"
                            )

            def _check_functions():
                for items in [
                    optimizers,
                    activation_functions,
                    loss_functions,
                    output_activation_functions,
                ]:
                    if type(items) != list:
                        raise ValueError(
                            "Optimizers, activations functions, loss functions and outpu layer functions value(s) must be in a list 🔢"
                        )
                    for item in items:
                        if (
                            not item in tf.keras.optimizers.__dict__.values()
                            and not item in tf.keras.activations.__dict__.values()
                            and not item in tf.keras.losses.__dict__.values()
                        ):
                            raise ValueError(
                                "Invalid optimizer, activation function, loss function or output_activation_functions use tensorflow.keras.(optimizers, activations, losses).[your function] 🔢)"
                            )

            def _check_other():
                if None in [
                    dense_units,
                    dense_count,
                    conv_filters,
                    conv_count,
                    conv_layer_repeat_limit,
                    drop_outs,
                    activation_functions,
                    loss_functions,
                    optimizers,
                    output_activation_functions,
                    frozen_layers,
                    base_models,
                    epochs,
                ]:
                    raise ValueError(
                        "One or more mandatory parameters are missing. Please provide the missing parameter 🔢"
                    )
                for itmind, items in enumerate(
                    [dense_units, drop_outs, conv_filters, conv_count, dense_count]
                ):
                    if items is not None:
                        for item in items:
                            if (type(item) == int and itmind in [0, 2, 3, 4]) or (
                                type(item) == float and itmind == 1
                            ):
                                if item <= 0 and itmind in [0, 2, 4]:
                                    raise ValueError(
                                        "values must be larger than 0\nCheck this values: dense_units, conv_filters, dense_count 🔢"
                                    )
                                elif itmind in [1, 3] and item < 0:
                                    raise ValueError(
                                        "values must be larger or equal to 0.\nCheck this values: drop_outs, conv_count 🔢"
                                    )
                            else:
                                raise ValueError(
                                    "Input value types must be int/float.\nCheck this values: dense_units, drop_outs, conv_filters, conv_count, dense_count 🔢"
                                )
                if frozen_layers is not None:
                    for frozen_layer in frozen_layers:
                        if type(frozen_layer) == int or type(frozen_layer) == float:
                            if frozen_layer > 1.0 or frozen_layer <= 0:
                                raise Exception(
                                    "values must be larger than 0 and smaller than 1\nCheck this values: frozen_layers 🔢"
                                )
                        else:
                            raise Exception(
                                "Input value types must be int/float.\nCheck this values: frozen_layers 🔢"
                            )
                if not (
                    dense_units is None
                    or conv_layer_repeat_limit is None
                    or drop_outs is None
                    or activation_functions is None
                    or loss_functions is None
                    or optimizers is None
                    or base_models is None
                    or epochs is None
                ):
                    if not (
                        len(dense_units) > 0
                        and conv_layer_repeat_limit > 0
                        and len(drop_outs) > 0
                        and len(activation_functions) > 0
                        and len(loss_functions) > 0
                        and len(optimizers) > 0
                        and len(base_models) > 0
                        and epochs > 0
                    ):
                        raise Exception(
                            "Your dense_units, conv_layer_repeat_limit, drop_outs, activation_functions, loss_functions, optimizers, base_models and epochs value(s) are not true. Please check your values! 🔢"
                        )

            threads = []
            thread = threading.Thread(target=_check_base_models)
            threads.append(thread)
            thread.start()
            thread = threading.Thread(target=_check_functions)
            threads.append(thread)
            thread.start()
            thread = threading.Thread(target=_check_other)
            thread.start()
            threads.append(thread)

            for t in threads:
                t.join()

            self.dense_units = dense_units
            self.dense_count = dense_count
            self.conv_filters = conv_filters
            self.conv_count = conv_count
            self.conv_layer_repeat_limit = conv_layer_repeat_limit
            self.drop_outs = drop_outs
            self.activation_functions = activation_functions
            self.loss_functions = loss_functions
            self.optimizers = optimizers
            self.output_activation_functions = output_activation_functions
            self.frozen_layers = frozen_layers
            self.epochs = epochs
            self.base_models = base_models

        def get_activation_functions():
            """
            Returns a list of common activation functions for neural networks.

            - tf.keras.activations.relu: Rectified Linear Unit (ReLU) activation.
            - tf.keras.activations.sigmoid: Sigmoid activation.
            - tf.keras.activations.tanh: Hyperbolic tangent activation.
            - tf.keras.activations.softmax: Softmax activation (for multiclass classification).
            - tf.keras.activations.linear: Linear activation (default for regression tasks).
            - tf.keras.activations.elu: Exponential Linear Unit (ELU) activation.
            - tf.keras.activations.softplus: Softplus activation.
            - tf.keras.activations.selu: Scaled Exponential Linear Unit (SELU) activation.
            - tf.keras.activations.hard_sigmoid: Hard Sigmoid activation.
            - tf.keras.activations.swish: Swish activation.
            - tf.keras.activations.exponential: Exponential activation.
            - tf.keras.activations.softsign: Softsign activation.
            """
            return [
                tf.keras.activations.relu,
                tf.keras.activations.sigmoid,
                tf.keras.activations.tanh,
                tf.keras.activations.softmax,
                tf.keras.activations.linear,
                tf.keras.activations.elu,
                tf.keras.activations.softplus,
                tf.keras.activations.selu,
                tf.keras.activations.hard_sigmoid,
                tf.keras.activations.swish,
                tf.keras.activations.exponential,
                tf.keras.activations.softsign,
            ]

        def get_loss_functions():
            """
            Returns a list of common loss functions for neural networks.

            - tf.keras.losses.binary_crossentropy: Binary Cross-Entropy loss (for binary classification).
            - tf.keras.losses.categorical_crossentropy: Categorical Cross-Entropy loss (for multiclass classification).
            - tf.keras.losses.mean_squared_error: Mean Squared Error loss (for regression tasks).
            - tf.keras.losses.sparse_categorical_crossentropy: Sparse Categorical Cross-Entropy loss (for multiclass classification).
            - tf.keras.losses.mean_absolute_error: Mean Absolute Error loss.
            - tf.keras.losses.cosine_similarity: Cosine Similarity loss.
            - tf.keras.losses.kl_divergence: Kullback-Leibler Divergence loss.
            - tf.keras.losses.hinge: Hinge loss (for SVM-style classification).
            - tf.keras.losses.huber: Huber loss.
            - tf.keras.losses.log_cosh: Log-Cosh loss.
            - tf.keras.losses.mean_squared_logarithmic_error: Mean Squared Logarithmic Error loss.
            - tf.keras.losses.poisson: Poisson loss (for count data).
            """
            return [
                tf.keras.losses.binary_crossentropy,
                tf.keras.losses.categorical_crossentropy,
                tf.keras.losses.mean_squared_error,
                tf.keras.losses.sparse_categorical_crossentropy,
                tf.keras.losses.mean_absolute_error,
                tf.keras.losses.cosine_similarity,
                tf.keras.losses.kl_divergence,
                tf.keras.losses.hinge,
                tf.keras.losses.huber,
                tf.keras.losses.log_cosh,
                tf.keras.losses.mean_squared_logarithmic_error,
                tf.keras.losses.poisson,
            ]

        def get_optimizers():
            """
            Returns a list of common optimizers for neural networks.

            - tf.keras.optimizers.Adam: Adam optimizer (Efficient and widely used for deep learning).
            - tf.keras.optimizers.SGD: Stochastic Gradient Descent (SGD) optimizer (Simple and widely used for training neural networks).
            - tf.keras.optimizers.RMSprop: RMSprop optimizer (Adaptive optimizer with momentum).
            - tf.keras.optimizers.Adagrad: Adagrad optimizer (Adaptive optimizer with learning rate per feature).
            - tf.keras.optimizers.Adadelta: Adadelta optimizer (Adaptive optimizer with adaptive learning rates).
            - tf.keras.optimizers.Adamax: Adamax optimizer (Variant of Adam with better convergence properties).
            - tf.keras.optimizers.Nadam: Nadam optimizer (Adam optimizer with Nesterov momentum).
            - tf.keras.optimizers.Ftrl: FTRL optimizer (Follow the Regularized Leader optimizer).
            """
            return [
                tf.keras.optimizers.Adam,
                tf.keras.optimizers.SGD,
                tf.keras.optimizers.RMSprop,
                tf.keras.optimizers.Adagrad,
                tf.keras.optimizers.Adadelta,
                tf.keras.optimizers.Adamax,
                tf.keras.optimizers.Nadam,
                tf.keras.optimizers.Ftrl,
            ]

        def get_output_activation_functions():
            """
            Returns a list of common activation functions for output layers in neural networks.

            - tf.keras.activations.linear: Linear activation (default for regression tasks).
            - tf.keras.activations.sigmoid: Sigmoid activation (for binary classification).
            - tf.keras.activations.softmax: Softmax activation (for multiclass classification).
            - tf.keras.activations.tanh: Hyperbolic tangent activation.
            - tf.keras.activations.relu: Rectified Linear Unit (ReLU) activation.
            - tf.keras.activations.elu: Exponential Linear Unit (ELU) activation.
            - tf.keras.activations.selu: Scaled Exponential Linear Unit (SELU) activation.
            """
            return [
                tf.keras.activations.linear,
                tf.keras.activations.sigmoid,
                tf.keras.activations.softmax,
                tf.keras.activations.tanh,
                tf.keras.activations.relu,
                tf.keras.activations.elu,
                tf.keras.activations.selu,
            ]

        class dominate_params_file:
            def __init__(self, params_file_path, train_folder):
                if not os.path.exists(params_file_path) or not os.path.isfile(
                    params_file_path
                ):
                    raise FileNotFoundError(
                        "Cant find path 🤷\nparams_file_path: " + params_file_path
                    )
                try:
                    with open(params_file_path, "r", encoding="utf-8") as p:
                        (
                            all_params,
                            skiped_models,
                            epochs,
                            best_acc,
                            save_to_path,
                            old_best_model_path,
                            base_models_accs,
                            patience,
                            def_patience,
                            model_name,
                            device_memory,
                            data_path,
                            use_multiprocessing,
                            def_params,
                        ) = eval(p.read())

                        del (
                            all_params,
                            skiped_models,
                            epochs,
                            best_acc,
                            save_to_path,
                            old_best_model_path,
                            base_models_accs,
                            patience,
                            def_patience,
                            model_name,
                            device_memory,
                            data_path,
                            use_multiprocessing,
                            def_params,
                        )
                except Exception as e:
                    raise Exception("Params file is corrupted 💥")
                if not os.path.exists(train_folder):
                    raise FileNotFoundError(
                        "Cant find path 🤷\nparams_file_path: " + train_folder
                    )
                self.params_file_path = params_file_path
                self.train_folder = train_folder

            def update_data_path(self, new_data_path):
                BurobotOutput.clear_and_memory_to()
                if type(new_data_path) in [list, tuple]:
                    for p in new_data_path:
                        if not os.path.exists(p):
                            raise FileNotFoundError(
                                "Cant find path 🤷\nnew_data_path: " + str(p)
                            )
                    if len(new_data_path) != 3:
                        raise ValueError(
                            "Cant find paths. if new_data_paths value is list or tuple value must like this ['train/datas/path', 'test/datas/path', 'val/datas/path]\nnew_data_paths"
                            + str(new_data_path)
                        )
                elif type(new_data_path) == str:
                    if not os.path.exists(new_data_path):
                        raise FileNotFoundError(
                            "Cant find path 🤷\nnew_data_path: " + str(new_data_path)
                        )
                else:
                    raise ValueError(
                        "new_data_paths value is not valid please check your data.\nnew_data_paths:"
                        + str(new_data_path)
                    )
                with open(self.params_file_path, "r", encoding="utf-8") as p:
                    print("Updating data_path 🔃")
                    (
                        all_params,
                        skiped_models,
                        epochs,
                        best_acc,
                        save_to_path,
                        _,
                        base_models_accs,
                        patience,
                        def_patience,
                        model_name,
                        device_memory,
                        data_path,
                        use_multiprocessing,
                        def_params,
                    ) = eval(p.read())
                    save_to_path = self.train_folder
                    if type(new_data_path) in [list, tuple]:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + save_to_path + '"').replace("'", '"'),
                                str(
                                    '"'
                                    + os.path.join(
                                        save_to_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                device_memory,
                                [
                                    '"' + str(new_data_path[0]) + '"',
                                    '"' + str(new_data_path[1]) + '"',
                                    '"' + str(new_data_path[2]) + '"',
                                ],
                                use_multiprocessing,
                                str(def_params).replace("'", '"'),
                            ],
                            save_to_path,
                            model_name,
                        )
                    elif type(new_data_path) == str:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + save_to_path + '"').replace("'", '"'),
                                str(
                                    '"'
                                    + os.path.join(
                                        save_to_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                device_memory,
                                '"' + str(new_data_path) + '"',
                                use_multiprocessing,
                                str(def_params).replace("'", '"'),
                            ],
                            save_to_path,
                            model_name,
                        )

            def update_train_folder_path(self, new_train_folder_path: str):
                if not os.path.exists(new_train_folder_path):
                    raise FileNotFoundError(
                        "Cant find path 🤷\nnew_train_folder_path: "
                        + str(new_train_folder_path)
                    )
                with open(self.params_file_path, "r", encoding="utf-8") as p:
                    print("Updating folder_path 🔃")
                    (
                        all_params,
                        skiped_models,
                        epochs,
                        best_acc,
                        save_to_path,
                        _,
                        base_models_accs,
                        patience,
                        def_patience,
                        model_name,
                        device_memory,
                        data_path,
                        use_multiprocessing,
                        def_params,
                    ) = eval(p.read())
                    if type(data_path) == list:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + new_train_folder_path + '"').replace(
                                    "'", '"'
                                ),
                                str(
                                    '"'
                                    + os.path.join(
                                        new_train_folder_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                device_memory,
                                [
                                    '"' + str(data_path[0]) + '"',
                                    '"' + str(data_path[1]) + '"',
                                    '"' + str(data_path[2]) + '"',
                                ],
                                use_multiprocessing,
                                str(def_params).replace("'", '"'),
                            ],
                            new_train_folder_path,
                            model_name,
                        )
                    else:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + new_train_folder_path + '"').replace(
                                    "'", '"'
                                ),
                                str(
                                    '"'
                                    + os.path.join(
                                        new_train_folder_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                device_memory,
                                '"' + str(data_path) + '"',
                                use_multiprocessing,
                                str(def_params).replace("'", '"'),
                            ],
                            new_train_folder_path,
                            model_name,
                        )

            def update_device_memory(self, new_device_memory: int):
                if new_device_memory <= 0:
                    raise ValueError(
                        "invalid device memory! device memory must be bigger than 0 🔢"
                    )
                with open(self.params_file_path, "r", encoding="utf-8") as p:
                    print("Updating device_memory 🔃")
                    (
                        all_params,
                        skiped_models,
                        epochs,
                        best_acc,
                        save_to_path,
                        _,
                        base_models_accs,
                        patience,
                        def_patience,
                        model_name,
                        device_memory,
                        data_path,
                        use_multiprocessing,
                        def_params,
                    ) = eval(p.read())
                    save_to_path = self.train_folder
                    if type(data_path) in [list, tuple]:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + save_to_path + '"').replace("'", '"'),
                                str(
                                    '"'
                                    + os.path.join(
                                        save_to_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                new_device_memory,
                                [
                                    '"' + str(data_path[0]) + '"',
                                    '"' + str(data_path[1]) + '"',
                                    '"' + str(data_path[2]) + '"',
                                ],
                                str(def_params).replace("'", '"'),
                            ],
                            save_to_path,
                            model_name,
                        )
                    else:
                        TransferLearning._save_unused_params(  # type: ignore
                            [
                                all_params,
                                str(skiped_models).replace("'", '"'),
                                epochs,
                                best_acc,
                                str('"' + save_to_path + '"').replace("'", '"'),
                                str(
                                    '"'
                                    + os.path.join(
                                        save_to_path, model_name + "_best.h5"
                                    )
                                    + '"'
                                ),
                                str(base_models_accs)
                                .replace("'", '"')
                                .replace('"[', "")
                                .replace(']"', ""),
                                patience,
                                def_patience,
                                '"' + str(model_name) + '"',
                                new_device_memory,
                                '"' + str(data_path) + '"',
                                use_multiprocessing,
                                str(def_params).replace("'", '"'),
                            ],
                            save_to_path,
                            model_name,
                        )

            def split(self, divide: int = 2):
                if divide <= 1:
                    raise ValueError("Divide value must be bigger than 1 🔢")
                with open(self.params_file_path, encoding="utf-8") as p:
                    (
                        all_params,
                        skiped_models,
                        epochs,
                        best_acc,
                        save_to_path,
                        old_best_model_path,
                        base_models_accs,
                        patience,
                        def_patience,
                        model_name,
                        device_memory,
                        data_path,
                        use_multiprocessing,
                        def_params,
                    ) = eval(p.read())

                divided_all_params = []
                divided_all_params_len_start = 0
                divided_all_params_len_end = len(all_params) // divide

                for _ in range(divide):
                    divided_all_params.append(
                        all_params[
                            divided_all_params_len_start:divided_all_params_len_end
                        ]
                    )
                    divided_all_params_len_start = divided_all_params_len_end
                    divided_all_params_len_end += len(all_params) // divide
                split_path = os.path.join(
                    save_to_path, model_name + "_unused_params(split)"
                )
                os.mkdir(split_path)
                for params in divided_all_params:
                    TransferLearning._save_unused_params(  # type: ignore
                        [
                            params,
                            str(skiped_models).replace("'", '"'),
                            epochs,
                            best_acc,
                            str('"' + save_to_path + '"').replace("'", '"'),
                            str(
                                '"'
                                + os.path.join(save_to_path, model_name + "_best.h5")
                                + '"'
                            ),
                            str(base_models_accs)
                            .replace("'", '"')
                            .replace('"[', "")
                            .replace(']"', ""),
                            patience,
                            def_patience,
                            '"' + str(model_name) + '"',
                            device_memory,
                            '"' + str(data_path) + '"',
                            use_multiprocessing,
                            str(def_params).replace("'", '"'),
                        ],
                        split_path,
                        model_name,
                        split=True,
                    )

    def create_model(
        train_data: str,  # type: ignore
        val_data: str,
        img_shape,
        batch_size,
        save_to_path: str,
        model_name: str,
        dense_units: int,
        dense_count: int,
        conv_filters: int,
        conv_count: int,
        conv_layer_repeat_limit: int,
        drop_out: float,
        activation_function: tf.keras.activations,  # type: ignore
        loss_function: tf.keras.losses,  # type: ignore
        optimizer: tf.keras.optimizers,  # type: ignore
        output_activation_function: tf.keras.activations,  # type: ignore
        frozen_layer: float,
        epochs: int,
        base_model: tf.keras.applications,  # type: ignore
        use_multiprocessing: bool = False,
    ):
        input_tensor = tf.keras.layers.Input(shape=img_shape)
        cur_model = base_model(
            weights="imagenet",
            include_top=False,
            input_shape=img_shape,
            input_tensor=input_tensor,
        )  # type: ignore

        model = tf.keras.models.Sequential()
        for layer in cur_model.layers[: int(len(cur_model.layers) * frozen_layer)]:
            layer.trainable = False
            try:
                model.add(layer)
            except:
                pass
        global conv0
        x = cur_model.output
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        if (conv_filters > 0 and conv_count == 0) and (TransferLearning.conv0):
            raise Exception("Model is unnecessary 🚮")
        for i in range(1, conv_count + 1):
            b = False
            if (conv_filters > 0 and conv_count == 0) and not (TransferLearning.conv0):
                TransferLearning.conv0 = True
            filters = conv_filters if i % 2 != 0 else int(conv_filters / 2)
            for _ in range(conv_layer_repeat_limit):
                try:
                    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3))(x)
                except:
                    b = True
                    break
            if b:
                break
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        x = tf.keras.layers.Flatten()(x)

        for i in range(1, dense_count + 1):
            dense_units = dense_units if i % 2 != 0 else int(dense_units / 2)
            x = tf.keras.layers.Dropout(drop_out)(x)
            x = tf.keras.layers.Dense(
                units=dense_units, activation=activation_function
            )(x)

        predictions = tf.keras.layers.Dense(
            units=len(train_data.class_names), activation=output_activation_function  # type: ignore
        )(x)
        model = tf.keras.models.Model(inputs=cur_model.input, outputs=predictions)

        model.compile(optimizer=optimizer(), loss=loss_function, metrics=["accuracy"])  # type: ignore

        num_cores = os.cpu_count()

        system_memory_gb = psutil.virtual_memory().total / (1024**3)

        workers = min(num_cores // 2, int(system_memory_gb // 2))  # type: ignore
        workers = max(1, int(workers * 0.8))

        prime_divisors = []
        for p in range(2, epochs // 2 + 1):
            if epochs % p == 0:
                prime_divisors.append(p)
        if len(prime_divisors) == 0:
            prime_divisors = [0]
        patience = max(1, prime_divisors[int(len(prime_divisors) * 0.3)])
        reduce = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.005, patience=patience, verbose=1, mode="min"
        )
        early_stoping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience + 1, mode="min"
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_to_path, model_name + "_checkpoint.h5"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )
        print("🍕 batch_size: " + str(batch_size))
        print("🧘 model patience: " + str(patience))
        if use_multiprocessing:
            print("👷 workers: " + str(workers))
        history = model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            batch_size=batch_size,
            callbacks=[early_stoping, reduce, checkpoint],
            verbose=1,  # type: ignore
            use_multiprocessing=use_multiprocessing,
            workers=workers,
        )

        gc.collect()
        tf.keras.backend.clear_session()

        return model, os.path.join(save_to_path, model_name + "_checkpoint.h5"), history

    def _check_errs(
        gpu: bool,  # type: ignore
        device_memory: int,
        split_to_data_path,
        patience,
        params,
        data_path: str,
        data_split_ratio: tuple,
        save_to_path: str,
        stop_massages: bool,
    ):
        if device_memory <= 0:
            raise Exception("device_memory must be bigger than 0 🔢")
        if patience is not None and type(patience) != int:
            raise Exception("patience must be int 🔢")
        if type(patience) == int:
            if patience <= 0:
                raise Exception("patience must be larger than 0 🔢")
        if not gpu or len(tf.config.experimental.get_visible_devices("GPU")) == 0:
            if stop_massages:
                BurobotOutput.print_burobot()
                q = input("🙅 Training with CPU. Are you sure? y/N")
                q = q.lower()
                if q != "y":
                    sys.exit()
                gpu = False
                del q
            else:
                BurobotOutput.print_burobot()
                print("Training with CPU 💾😢")
                time.sleep(5)
        BurobotOutput.clear_and_memory_to()

        if type(params) != TransferLearning.Params and type(params) != str:
            raise Exception(
                "Your params value is not true.\nUse TransferLearning.Params(...) 🎛️"
            )
        if type(data_path) == str:
            if not os.path.exists(data_path):
                raise FileNotFoundError(
                    f"Can't find path 🤷‍♂️\nsave_to_path: {save_to_path}"
                )
        elif type(data_path) == list or type(data_path) == tuple:
            if len(data_path) != 3:
                raise ValueError(
                    "If you want to use splited data. You must give data_path value like this: [train_path, test_path, val_path]"
                )
            train_path, test_path, val_path = data_path  # type: ignore
            if (
                not os.path.exists(train_path)
                or not os.path.exists(test_path)
                or not os.path.exists(val_path)
            ):
                raise FileNotFoundError(
                    f"Can't find path 🤷‍♂️\ntrain_path: {train_path}\ntest_path: {test_path}\nval_path: {val_path}"
                )
            if stop_massages:
                BurobotOutput.print_burobot()
                print(
                    f"Your data paths is:\ntrain_path: {train_path}\ntest_path: {test_path}\nval_path: {val_path}"
                )
                q = input("Are you sure? Y/n")
                q.lower()
                if q == "n":
                    sys.exit()
                del q
            else:
                BurobotOutput.print_burobot()
                print(
                    f"Your data paths is:\ntrain_path: {train_path}\ntest_path: {test_path}\nval_path: {val_path}"
                )
                time.sleep(5)
            BurobotOutput.clear_and_memory_to()
        else:
            raise ValueError("data_path value must str, list or tuple 🔢")
        if not os.path.exists(save_to_path):
            raise FileNotFoundError(
                f"Can't find path 🤷‍♂️\nsave_to_path: {save_to_path}"
            )
        if split_to_data_path is not None:
            if not os.path.exists(split_to_data_path):
                raise FileNotFoundError(
                    f"Can't find path 🤷‍♂️\nsplit_to_data_path: {split_to_data_path}"
                )

        if type(params) == str:
            if not os.path.exists(params):
                raise FileNotFoundError(f"Can't find path 🤷‍♂️\nparams: {params}")
        if (
            data_split_ratio[0]
            + data_split_ratio[1]
            + (1 - (data_split_ratio[0] + data_split_ratio[1]))
            != 1
        ):
            raise ValueError(
                "Your data_split_ratio value not valid. data_split_ratio:tuple=(train, test) note: val= 1-(train+test). sum of data_split_ratio must be 1. example: (0.8, 0.1) => %80 train, %10 test, %10 validation"
            )

    def _print_info(
        best_values: dict,  # type: ignore
        model_name: str,
        params,
        dense_units,
        dense_count,
        conv_filters,
        conv_count,
        conv_layer_repeat,
        drop_out,
        activation_function,
        loss_function,
        optimizer,
        output_activation_function,
        frozen_layer,
        base_model,
        last_acc: str,
        best_acc: str,
        all_: int,
        c: int,
        patience,
        def_patience,
    ):
        BurobotOutput.print_burobot()
        print("Sit back and relax. This process will take a LONG time 😎\n")
        print("MODEL:" + model_name + "\n")
        poutput = ""
        emoji_dict = {
            "dense_units": "🧠",
            "dense_count": "🔢",
            "conv_filters": "💄",
            "conv_count": "🎨",
            "conv_layer_repeat": "🖌️",
            "drop_out": "👋",
            "activation_function": "⚡️",
            "loss_function": "💔",
            "optimizer": "🚀",
            "output_activation_function": "🧽",
            "frozen_layer": "🥶",
            "base_model": "📦",
        }
        for key in best_values.keys():
            if (
                key == "base_model"
                or key == "activation_function"
                or key == "loss_function"
                or key == "activation_function"
                or key == "output_activation_function"
            ):
                list_val = list(getattr(params, key + "s"))
                for i, val in enumerate(list_val):
                    list_val[i] = str(val).split(" ")[1]

                poutput += (
                    str(
                        emoji_dict[key]
                        + " "
                        + key
                        + ": "
                        + str(list_val)
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(
                            str(eval(key)).split(" ")[1],
                            "[" + str(eval(key)).split(" ")[1] + "]",
                            1,
                        )
                    )
                    + "\n"
                )
            elif key == "optimizer":
                list_val = list(getattr(params, key + "s"))
                for i, val in enumerate(list_val):
                    list_val[i] = (
                        str(val).split(" ")[1].split(".")[-1].replace("'>", "")
                    )

                poutput += (
                    str(
                        emoji_dict[key]
                        + " "
                        + key
                        + ": "
                        + str(list_val)
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(
                            str(val).split(" ")[1].split(".")[-1].replace("'>", ""),  # type: ignore
                            "["
                            + str(val).split(" ")[1].split(".")[-1].replace("'>", "")  # type: ignore
                            + "]",
                        )
                    )
                    + "\n"
                )

            elif key == "conv_layer_repeat":
                item = str(list(range(1, getattr(params, str(key) + "_limit") + 1)))
                poutput += (
                    str(
                        emoji_dict[key]
                        + " "
                        + key
                        + ": "
                        + item.replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(str(eval(key)), "[" + str(eval(key)) + "]")
                    )
                    + "\n"
                )
            else:
                try:
                    list_val = getattr(params, str(key) + "s")
                except:
                    list_val = getattr(params, str(key))
                poutput += (
                    str(
                        emoji_dict[key]
                        + " "
                        + key
                        + ": "
                        + str(list_val)
                        .replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace(str(eval(key)), "[" + str(eval(key)) + "]")
                    )
                    + "\n"
                )

        print("🎓 The model is training with these values:\n" + poutput)

        print("📊 parameters tried: " + str(c) + "/" + str(all_))
        print("💪 Train %" + str((c - 0) / (all_ - 0) * 100))

        print(
            "🏆 Best Accuracy: "
            + (("%" + str(best_acc)) if best_acc is not None else "None")
        )
        if last_acc is not None:
            print(
                ("🕦 Last Accuracy: ")
                + (
                    ("%" + str(last_acc))
                    if last_acc >= 0  # type: ignore
                    else "Overfitting"
                    if last_acc == -1
                    else "Error"
                )
            )
        else:
            print("🕦 Last Accuracy: None")
        if patience is None:
            print("😛🧁 patience: deactive\n")
        elif not patience <= 0:
            if def_patience > 4:
                if patience > def_patience * 0.8:
                    print("🥱 patience: " + str(patience) + "\n")
                elif patience > def_patience * 0.6:
                    print("😐 patience: " + str(patience) + "\n")
                elif patience > def_patience * 0.4:
                    print("🤨 patience: " + str(patience) + "\n")
                else:
                    print("😫 patience: " + str(patience) + "\n")
            elif def_patience > 2:
                if patience > def_patience * 0.5:
                    print("🤨 patience: " + str(patience) + "\n")
                else:
                    print("😫 patience: " + str(patience) + "\n")
            else:
                print("😫 patience: " + str(patience) + "\n")

    def _save_unused_params(
        params: list, save_to_path: str, model_name: str, split: bool = False  # type: ignore
    ):
        n_params = []
        for p in params[0]:
            n_params.append(list(p))
        params[0] = n_params
        del n_params
        for i, p in enumerate(params[0]):
            # base_model, activation_function, loss_function, optimizer, output_activation_function, dense_count, conv_filters, conv_layer_repeat, conv_count, dense_units, frozen_layer, drop_out = all_params[0]
            params[0][i][0] = "tf.keras.applications." + str(p[0]).split(" ")[1]
            params[0][i][1] = "tf.keras.activations." + str(p[1]).split(" ")[1]
            params[0][i][4] = "tf.keras.activations." + str(p[4]).split(" ")[1]
            params[0][i][2] = "tf.keras.losses." + str(p[2]).split(" ")[1]
            params[0][i][3] = "tf.keras.optimizers." + str(p[3]).split(" ")[1].split(
                "."
            )[-1].replace("'>", "")
        if not split:
            with open(
                os.path.join(save_to_path, model_name + "_unused_params.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(params).replace("'", ""))
        else:
            params_file_end = "_splited_unused_params"
            while os.path.exists(
                os.path.join(save_to_path, model_name + params_file_end + ".txt")
            ):
                try:
                    file_id = int(params_file_end[-1])
                    file_id += 1
                    params_file_end = params_file_end[:-1] + str(file_id)
                except ValueError:
                    params_file_end += "_1"

            with open(
                os.path.join(save_to_path, model_name + params_file_end + ".txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(str(params).replace("'", ""))

    def _write_to_log_file(text, save_to_path):  # type: ignore
        if not os.path.exists(save_to_path):
            raise FileNotFoundError(
                f"Can't find path 🤷‍♂️\nsave_to_path: {save_to_path}"
            )
        mode = "w"
        if os.path.exists(os.path.join(save_to_path, "log.txt")):
            text = time.strftime("[%d-%m-%Y %H:%M:%S]") + text  # type: ignore
            mode = "a"
            text = "=" * 83 + "\n" + text
        else:
            text = (
                BurobotOutput.print_burobot(True)  # type: ignore
                + "\nStarted:"
                + time.strftime("[%d-%m-%Y %H:%M:%S]")
                + "\n"
                + text
            )
        text = text + "\n" + "=" * 83
        with open(
            os.path.join(save_to_path, "log.txt"), mode, encoding="utf-8"
        ) as log_file:
            if mode == "a":
                log_file.write("\n" + text + "\n")
            else:
                log_file.write(text + "\n")

    def FindModel(
        params,  # type: ignore
        data_path=None,
        save_to_path: str = None,  # type: ignore
        model_name: str = None,  # type: ignore
        device_memory: int = None,  # type: ignore
        split_to_data_path: str = None,  # type: ignore
        data_split_ratio: tuple = (0.7, 0.1),
        gpu: bool = True,
        patience=3,
        stop_massages: bool = True,
        use_multiprocessing: bool = False,
    ):
        def_patience = patience
        def_params = {}
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        BurobotOutput.clear_and_memory_to()

        def _save_unused_params_(save_data_paths):
            if type(save_data_paths) in [tuple, list]:
                TransferLearning._save_unused_params(  # type: ignore
                    [
                        all_params,
                        str(skiped_models).replace("'", '"'),
                        epochs,
                        best_acc,
                        str('"' + save_to_path + '"').replace("'", '"'),
                        str(
                            '"'
                            + os.path.join(save_to_path, model_name + "_best.h5")
                            + '"'
                        ),
                        str(base_models_accs)
                        .replace("'", '"')
                        .replace('"[', "")
                        .replace(']"', ""),
                        patience,
                        def_patience,
                        '"' + str(model_name) + '"',
                        device_memory,
                        [
                            '"' + save_data_paths[0] + '"',
                            '"' + save_data_paths[1] + '"',
                            '"' + save_data_paths[2] + '"',
                        ],
                        use_multiprocessing,
                        str(def_params).replace("'", '"'),
                    ],
                    save_to_path,
                    model_name,
                )
            else:
                TransferLearning._save_unused_params(  # type: ignore
                    [
                        all_params,
                        str(skiped_models).replace("'", '"'),
                        epochs,
                        best_acc,
                        str('"' + save_to_path + '"').replace("'", '"'),
                        str(
                            '"'
                            + os.path.join(save_to_path, model_name + "_best.h5")
                            + '"'
                        ),
                        str(base_models_accs)
                        .replace("'", '"')
                        .replace('"[', "")
                        .replace(']"', ""),
                        patience,
                        def_patience,
                        '"' + str(model_name) + '"',
                        device_memory,
                        '"' + save_data_paths + '"',
                        use_multiprocessing,
                        str(def_params).replace("'", '"'),
                    ],
                    save_to_path,
                    model_name,
                )

        if type(params) != str:
            TransferLearning._check_errs(  # type: ignore
                gpu,
                device_memory,
                split_to_data_path,
                patience,
                params,
                data_path,  # type: ignore
                data_split_ratio,
                save_to_path,
                stop_massages,
            )
        if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpu:
                tf.config.experimental.set_visible_devices(gpus[0], "GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                tf.config.set_visible_devices([], "GPU")
        best_model = None
        best_acc = None
        best_history = None
        best_values = {
            "dense_units": None,
            "dense_count": None,
            "conv_filters": None,
            "conv_count": None,
            "conv_layer_repeat": None,
            "drop_out": None,
            "activation_function": None,
            "loss_function": None,
            "optimizer": None,
            "output_activation_function": None,
            "frozen_layer": None,
            "base_model": None,
        }
        skiped_models = []
        base_models_accs = [[], []]
        all_params = []
        epochs = 1
        len_base_models = 0

        if type(params) != str:
            for key, value in params.__dict__.items():
                if type(value) in [tuple, list]:
                    new_list = []
                    for item in value:
                        if callable(item) and hasattr(item, "__module__"):
                            module = ".".join(str(item.__module__).split(".")[:2])
                            qual_name = item.__qualname__
                            new_list.append(f"tf.{module}.{qual_name}")
                        else:
                            new_list.append(item)
                    def_params[key] = new_list
                else:
                    if callable(value) and hasattr(value, "__module__"):
                        def_params[key] = f"tf.{value.__module__}.{value.__qualname__}"
                    else:
                        def_params[key] = value

            if stop_massages:
                if len(params.base_models) == 1:  # type: ignore
                    patience = None
                    q = input(
                        "❓ Patiance deactivated because you only have 1 base model. Than means I can't chance model order. Are sure about using just 1 base model? N/y"
                    )
                    q.lower()
                    if q != "y":
                        sys.exit()
                if use_multiprocessing:
                    q = input(
                        "❓ Multiprocessing is activated. Multiprocessing might be can speed up training but its requires high level hardware. If you doesnt, training will be unsuccessfull. You can use colab if you want. Are you sure abount openning Multiprocessing? N/y"
                    )
                    q.lower()
                    if q != "y":
                        sys.exit()
                    del q
                else:
                    q = input(
                        "❓ Multiprocessing is deactivated. Multiprocessing might be can speed up training but its requires high level hardware. If you doesnt, training will be unsuccessfull. You can use colab if you want. Are you sure abount deactivate Multiprocessing? N/y"
                    )
                    q.lower()
                    if q != "y":
                        sys.exit()
                    del q

            else:
                if len(params.base_models) == 1:  # type: ignore
                    patience = None
                    print(
                        "⚠️ Patiance deactivated because you only have 1 base model. Than means I can't chance model order."
                    ),  # type: ignore
                    time.sleep(5)
            BurobotOutput.clear_and_memory_to()

            skiped_models = []
            base_models_accs[1] = np.zeros(len(params.base_models)).tolist()  # type: ignore
            for b in params.base_models:  # type: ignore
                base_models_accs[0].append(str(b).split(" ")[1])
            epochs = params.epochs  # type: ignore

            all_params = list(
                itertools.product(
                    params.base_models,  # type: ignore
                    params.activation_functions,  # type: ignore
                    params.loss_functions,  # type: ignore
                    params.optimizers,  # type: ignore
                    params.output_activation_functions,  # type: ignore
                    params.dense_count,  # type: ignore
                    params.conv_filters,  # type: ignore
                    range(1, params.conv_layer_repeat_limit + 1),  # type: ignore
                    params.conv_count,  # type: ignore
                    params.dense_units,  # type: ignore
                    params.frozen_layers,  # type: ignore
                    params.drop_outs,  # type: ignore
                )
            )
            i = 0
            while True:
                try:
                    if i == 0:
                        os.mkdir(os.path.join(save_to_path, model_name + "_train"))
                        save_to_path = os.path.join(save_to_path, model_name + "_train")
                    else:
                        os.mkdir(
                            os.path.join(save_to_path, model_name + "_train_" + str(i))
                        )
                        save_to_path = os.path.join(
                            save_to_path, model_name + "_train_" + str(i)
                        )
                    TransferLearning._write_to_log_file("", save_to_path)  # type: ignore
                    del i
                    break
                except FileExistsError:
                    i += 1
                    pass
        else:
            with open(params, "r", encoding="utf-8") as p:
                (
                    all_params,
                    skiped_models,
                    epochs,
                    best_acc,
                    save_to_path,
                    old_best_model_path,
                    base_models_accs,
                    _patience,
                    def_patience,
                    model_name,
                    _device_memory,
                    _data_path,
                    _use_multiprocessing,
                    def_params,
                ) = eval(p.read())
                params = TransferLearning.Params(epochs=epochs, kwargs=def_params)
                if device_memory is None or device_memory <= 0:
                    device_memory = _device_memory
                if data_path is None:
                    data_path = _data_path
                if patience == 3:
                    patience = _patience
                    def_patience = patience
                if use_multiprocessing == False:
                    use_multiprocessing = _use_multiprocessing
                del _use_multiprocessing, _patience, _data_path, _device_memory

                if data_path is not None:
                    if type(data_path) == str:
                        if not os.path.exists(data_path):
                            raise FileNotFoundError(
                                "Cant find path 🤷\ndata_path: " + data_path
                            )
                    elif type(data_path) == list or type(data_path) == tuple:
                        for pa in data_path:
                            if not os.path.exists(pa):
                                raise FileNotFoundError(
                                    "Cant find path(s) 🤷\ndata_path: " + data_path  # type: ignore
                                )
                    else:
                        raise ValueError("data_path Value must be str, list or tuple 🔢")

                TransferLearning._write_to_log_file(
                    "Loaded un_used_params file\nContinuing training", save_to_path  # type: ignore
                )

            if os.path.exists(old_best_model_path):
                try:
                    best_model = tf.keras.models.load_model(old_best_model_path)
                except:
                    if old_best_model_path.split(".")[-1] == "keras":
                        raise Exception(
                            "Error on loading Old Best Model Path. Please check your tensorflow version must be >=2.12 or change file type to .h5 with convert_keras_file_to_h5 metod"
                        )
            else:
                print(
                    "⚠️ Can't find old best model .h5"
                    + " file! best model and best accuracy is setting to empty 🫙"
                )
                time.sleep(2)
                BurobotOutput.clear_and_memory_to()
                best_model = None
                best_acc = None

        len_base_models = len(params.base_models)  # type: ignore
        train_path, test_path, val_path = None, None, None
        BurobotOutput.clear_and_memory_to()
        if type(data_path) == str:
            BurobotOutput.print_burobot()
            try:
                if split_to_data_path is None:
                    os.makedirs(os.path.join(save_to_path, "splited_data"))
                    train_path, test_path, val_path = DominateImage.split_data(
                        data_path,
                        os.path.join(save_to_path, "splited_data"),
                        data_split_ratio,
                    )
                else:
                    os.makedirs(os.path.join(split_to_data_path, "splited_data"))
                    train_path, test_path, val_path = DominateImage.split_data(
                        data_path,
                        os.path.join(split_to_data_path, "splited_data"),
                        data_split_ratio,
                    )

            except:
                file_tree = """
                data/
                |   A/
                |   |   file1.data-type(jpg, png, jpeg)
                |   |   ...
                |   B/
                |   |   file1.data-type(jpg, png, jpeg)
                |   |   ...
                |   C/
                |   |   file1.data-type(jpg, png, jpeg)
                |   |   ...
                |   ...
                """
                TransferLearning._write_to_log_file(
                    "Error in the splitting data! Please check your folder tree. folder tree must be like this:\n"
                    + file_tree,  # type: ignore
                    save_to_path,
                )
                raise Exception(
                    "Error in the splitting data! Please check your folder tree. folder tree must be like this:\n"
                    + file_tree
                )
        else:
            train_path, test_path, val_path = data_path  # type: ignore
        BurobotOutput.clear_and_memory_to()
        all_ = len(all_params)
        last_acc = None
        BurobotOutput.print_burobot()
        if stop_massages:
            if type(patience) is not None:
                q = input(
                    "❓ Up to "
                    + str(all_)
                    + " possibilities will be tried.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models.\nDo you want to continue? Y/n "
                )
                q = q.lower()
                if q == "n":
                    sys.exit()
                del q
            else:
                q = input(
                    "❓ Up to "
                    + str(all_)
                    + " possibilities will be tried. But patience is active! This means training might be take shorter time than predicted time.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models.\nDo you want to continue? Y/n "
                )
                q = q.lower()
                if q == "n":
                    sys.exit()
                del q
        else:
            if patience is None:
                print(
                    "⚠️ Up to "
                    + str(all_)
                    + " possibilities will be tried.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models."
                )
                time.sleep(5)
            else:
                print(
                    "⚠️ Up to "
                    + str(all_)
                    + " possibilities will be tried. But patience is active! This means training might be take shorter time than predicted time.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models."
                )
                time.sleep(5)
        BurobotOutput.clear_and_memory_to()
        c = 0
        print("Loading data ⚙️")
        print("📚 Train data:")
        my_data_loader = BurobotImageData.ImageLoader()
        train_data, batch_size, img_shape = my_data_loader.load_data(
            path=train_path, return_data_only=False, device_memory=device_memory
        )
        print()
        print("📝 Validation data:")
        val_data = my_data_loader.load_data(path=val_path, device_memory=device_memory)
        print()
        print("🧪 Test data:")
        test_data = my_data_loader.load_data(
            path=test_path, device_memory=device_memory
        )
        save_data_paths = [train_path, test_path, val_path]

        if img_shape[-1] != 3:  # type: ignore
            raise ValueError("Images must be RGB please check your data 🔢")
        last_base_model = None
        while len(all_params) != 0:
            (
                base_model,
                activation_function,
                loss_function,
                optimizer,
                output_activation_function,
                dense_count,
                conv_filters,
                conv_layer_repeat,
                conv_count,
                dense_units,
                frozen_layer,
                drop_out,
            ) = all_params[0]
            if last_base_model != base_model:
                last_base_model = base_model
                TransferLearning.conv0 = False
            BurobotOutput.clear_and_memory_to()
            try:
                if patience is not None:
                    if patience == 0:
                        patience = def_patience
                        print("😡 Patience is 0; changing model order 🔃")
                        time.sleep(2)
                        skiped_model_list = []
                        c_all_params = list(all_params)
                        if len(skiped_models) + 1 != len_base_models:
                            for p in c_all_params:
                                if (
                                    p[0] is base_model
                                    and str(base_model).split(" ")[1]
                                    not in skiped_models
                                ):
                                    all_params.remove(p)
                                    skiped_model_list.append(p)
                            if str(base_model).split(" ")[1] not in skiped_models:
                                skiped_models.append(str(base_model).split(" ")[1])

                                all_params.extend(skiped_model_list)

                        else:
                            all_params.sort(
                                key=lambda x: base_models_accs[1][
                                    base_models_accs[0].index(str(x[0]).split(" ")[1])
                                ]
                            )
                            all_params.reverse()

                            patience = None
                            print(
                                "⚠️ Patience is disabled because there are no models to be tested. Running all params with ordered base models."
                            )
                            time.sleep(2)
                        _save_unused_params_(save_data_paths)  # type: ignore
                        continue
                TransferLearning._print_info(  # type: ignore
                    best_values,
                    model_name,
                    params,
                    dense_units,
                    dense_count,
                    conv_filters,
                    conv_count,
                    conv_layer_repeat,
                    drop_out,
                    activation_function,
                    loss_function,
                    optimizer,
                    output_activation_function,
                    frozen_layer,
                    base_model,
                    last_acc,  # type: ignore
                    best_acc,  # type: ignore
                    all_,
                    c,
                    patience,
                    def_patience,
                )
                model_exception = None
                try:
                    _save_unused_params_(save_data_paths)  # type: ignore
                    (
                        model,
                        checkpoint_model_path,
                        history,
                    ) = TransferLearning.create_model(  # type: ignore
                        train_data,  # type: ignore
                        val_data,  # type: ignore
                        img_shape,
                        batch_size,
                        save_to_path,
                        model_name,
                        dense_units,
                        dense_count,
                        conv_filters,
                        conv_count,
                        conv_layer_repeat,
                        drop_out,
                        activation_function,
                        loss_function,
                        optimizer,
                        output_activation_function,
                        frozen_layer,
                        epochs,
                        base_model,
                        use_multiprocessing,
                    )
                except Exception as e:
                    model_exception = str(e)
                    TransferLearning._write_to_log_file(
                        "Model train error:\n"
                        + str(e)
                        + "\nParams:"
                        + str(all_params[0]),  # type: ignore
                        save_to_path,
                    )
                    model, history = None, None
                if model is None and history is None:
                    if (
                        base_models_accs[1][
                            base_models_accs[0].index(str(base_model).split(" ")[1])
                        ]
                        < -2
                    ):
                        base_models_accs[1][
                            base_models_accs[0].index(str(base_model).split(" ")[1])
                        ] = -2
                    c += 1
                    last_acc = -2
                    if (
                        patience is not None
                        and model_exception != "Model is unnecessary 🚮"
                    ):
                        patience -= 1

                    del all_params[0]
                    _save_unused_params_(save_data_paths)  # type: ignore
                    continue
                print("Testing Model 🥼")
                test_acc, predictions = test_model(
                    model, test_data, device_memory, return_predictions=True
                )  # type: ignore
                TransferLearning._write_to_log_file(
                    "Tested Model:\nAccuracy:"
                    + (("%" + str(test_acc)) if test_acc != -1 else "Overfitting"),  # type: ignore
                    save_to_path,
                )
                checkpoint_model = tf.keras.models.load_model(checkpoint_model_path)  # type: ignore
                print("Testing Checkpoint Model 🥼")
                checkpoint_test_acc, checkpoint_predictions = test_model(
                    checkpoint_model, test_data, device_memory, return_predictions=True
                )  # type: ignore
                TransferLearning._write_to_log_file(
                    "Tested Checkpoint Model:\nAccuracy:"
                    + (
                        ("%" + str(checkpoint_test_acc))
                        if checkpoint_test_acc != -1
                        else "Overfitting"
                    ),  # type: ignore
                    save_to_path,
                )
                model.save(os.path.join(save_to_path, model_name + "_last.h5"))  # type: ignore
                checkpoint_model.save(  # type: ignore
                    os.path.join(save_to_path, model_name + "_last(checkpoint).h5")
                )
                my_data_loader = BurobotImageData.ImageLoader()
                _draw_model(
                    history,
                    predictions,
                    model_name,
                    model_name + "_last",
                    test_acc,
                    {
                        "dense_units": dense_units,
                        "dense_count": dense_count,
                        "drop_out": drop_out,
                        "activation_function": str(activation_function).split(" ")[1],
                        "loss_function": str(loss_function).split(" ")[1],
                        "optimizer": str(optimizer)
                        .split(" ")[1]
                        .split(".")[-1]
                        .replace("'>", ""),
                        "frozen_layer": frozen_layer,
                        "base_model": str(base_model).split(" ")[1],
                    },
                    save_to_path,
                    my_data_loader.count_images(test_path),
                )

                if checkpoint_test_acc > test_acc:
                    predictions = checkpoint_predictions
                    test_acc = checkpoint_test_acc
                    model = checkpoint_model
                    del (
                        checkpoint_test_acc,
                        checkpoint_model_path,  # type: ignore
                        checkpoint_model,
                        checkpoint_predictions,
                    )

                last_acc = test_acc

                if (
                    base_models_accs[1][
                        base_models_accs[0].index(str(base_model).split(" ")[1])
                    ]
                    < test_acc
                ):
                    base_models_accs[1][
                        base_models_accs[0].index(str(base_model).split(" ")[1])
                    ] = test_acc
                if test_acc > (best_acc if best_acc is not None else 0):
                    if patience is not None and patience != def_patience:
                        patience += 1
                    best_model = model
                    del model
                    best_history = history
                    best_acc = test_acc
                    for key in best_values.keys():
                        if (
                            key == "base_model"
                            or key == "activation_function"
                            or key == "loss_function"
                            or key == "output_activation_function"
                        ):
                            best_values[key] = str(eval(key)).split(" ")[1]  # type: ignore
                        elif key == "optimizer":
                            best_values[key] = (  # type: ignore
                                str(eval(key))
                                .split(" ")[1]
                                .split(".")[-1]
                                .replace("'>", "")
                            )
                        else:
                            best_values[key] = eval(key)
                    _draw_model(
                        history,
                        predictions,
                        model_name,
                        model_name + "_best",
                        test_acc,
                        best_values,
                        save_to_path,
                        my_data_loader.count_images(test_path),
                    )
                    best_model.save(os.path.join(save_to_path, model_name + "_best.h5"))  # type: ignore
                else:
                    if patience is not None:
                        patience -= 1
                c += 1

                del all_params[0]

                _save_unused_params_(save_data_paths)  # type: ignore
            except KeyboardInterrupt:
                TransferLearning._write_to_log_file("User stoped", save_to_path)  # type: ignore
                print("\nI-i s-stopped 🙌")
                sys.exit()
                pass
            except Exception as e:
                TransferLearning._write_to_log_file("Error: " + str(e), save_to_path)  # type: ignore
                print("Something went wrong. Skiping to next model 😵‍💫")
                continue

        if best_model is None:
            BurobotOutput.clear_and_memory_to()
            print("I can't find best model 😥 Please check your data and parameters.")
            return None, None, None, None
        return best_model, best_acc, best_values, best_history


# BUROBOT
