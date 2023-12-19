import numpy as np
import pandas as pd
import json, json, sklearn, random, string, os, sys, itertools, threading, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except:
    pass
import tensorflow as tf

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput
from Burobot.tools import BurobotTextData


class Learning:
    class Params:
        def __init__(
            self,
            embedding_outputs=None,
            lstm_units=None,
            lstm_activation_functions=None,
            lstm_count_limit=None,
            dense_units=None,
            dense_count_limit=None,
            dense_activation_functions=None,
            output_activation_functions=None,
            loss_functions=None,
            optimizers=None,
            epochs=None,
            **kwargs,
        ):
            if kwargs:
                embedding_outputs = kwargs["kwargs"]["embedding_outputs"]
                lstm_units = kwargs["kwargs"]["lstm_units"]
                lstm_activation_functions = kwargs["kwargs"][
                    "lstm_activation_functions"
                ]
                lstm_count_limit = kwargs["kwargs"]["lstm_count_limit"]
                dense_units = kwargs["kwargs"]["dense_units"]
                dense_count_limit = kwargs["kwargs"]["dense_count_limit"]
                dense_activation_functions = kwargs["kwargs"][
                    "dense_activation_functions"
                ]
                output_activation_functions = kwargs["kwargs"][
                    "output_activation_functions"
                ]
                loss_functions = kwargs["kwargs"]["loss_functions"]
                optimizers = kwargs["kwargs"]["optimizers"]
                epochs = kwargs["kwargs"]["epochs"]
            if None in [
                embedding_outputs,
                lstm_units,
                lstm_activation_functions,
                lstm_count_limit,
                dense_units,
                dense_count_limit,
                dense_activation_functions,
                output_activation_functions,
                loss_functions,
                optimizers,
                epochs,
            ]:
                raise ValueError(
                    "embedding_outputs, lstm_units, lstm_activation_functions, lstm_count_limit, dense_units, dense_count_limit, dense_activation_functions, output_activation_functions, loss_functions, optimizers orepochs is None 🔢"
                )
            for param, i in enumerate(
                [
                    embedding_outputs,
                    lstm_units,
                    lstm_activation_functions,
                    lstm_count_limit,
                    dense_units,
                    dense_count_limit,
                    dense_activation_functions,
                    output_activation_functions,
                    loss_functions,
                    optimizers,
                    epochs,
                ]
            ):
                if i in [
                    3,
                    5,
                    10,
                ]:  # lstm_count_limit, dense_count_limit, epochs
                    if type(param) != int:
                        raise ValueError(
                            "lstm_count_limit, dense_count_limit, epochs must be intager 🔢"
                        )
                    if param <= 0:
                        raise ValueError(
                            "lstm_count_limit, dense_count_limit, epochs must be bigger than 0 🔢"
                        )

                elif i in [
                    0,
                    1,
                    4,
                    6,
                    7,
                    8,
                    9,
                ]:  # embedding_outputs, lstm_units, lstm_activation_functions, dense_units, dense_activation_functions, output_activation_functions, loss_functions, optimizers
                    if type(param) not in [list, tuple]:
                        raise ValueError(
                            "lstm_units, lstm_activation_functions, dense_units, dense_activation_functions, output_activation_functions, loss_functions, optimizers must be list or tuple 🔢"
                        )
                    for p in param:
                        if i in [0, 1, 6]:
                            if p <= 0:
                                raise ValueError(
                                    "embedding_outputs, lstm_units, dense_units must be bigger than 0 🔢"
                                )
                        else:
                            if (
                                not param in tf.keras.optimizers.__dict__.values()
                                and not param in tf.keras.activations.__dict__.values()
                                and not param in tf.keras.losses.__dict__.values()
                            ):
                                raise ValueError(
                                    "Invalid lstm_activation_functions, dense_activation_functions, output_activation_functions, loss_functions or optimizers please use tensorflow.keras.(optimizers, activations, losses).[your function] 🔢"
                                )
            self.embedding_outputs = embedding_outputs
            self.lstm_units = lstm_units
            self.lstm_activation_functions = lstm_activation_functions
            self.lstm_count_limit = lstm_count_limit
            self.dense_units = dense_units
            self.dense_count_limit = dense_count_limit
            self.dense_activation_functions = dense_activation_functions
            self.output_activation_functions = output_activation_functions
            self.loss_functions = loss_functions
            self.optimizers = optimizers
            self.epochs = epochs

        def get_activation_functions():
            return [
                tf.keras.activations.relu,
                tf.keras.activations.elu,
                tf.keras.activations.tanh,
                tf.keras.activations.sigmoid,
            ]

        def get_loss_functions():
            return [
                tf.keras.losses.sparse_categorical_crossentropy,
                tf.keras.losses.hinge,
                tf.keras.losses.mean_squared_error,
                tf.keras.losses.mean_absolute_error,
                tf.keras.losses.huber,
            ]

        def get_optimizers():
            return [
                tf.keras.optimizers.Adam,
                tf.keras.optimizers.RMSprop,
                tf.keras.optimizers.Adagrad,
                tf.keras.optimizers.SGD,
                tf.keras.optimizers.Adadelta,
                tf.keras.optimizers.Adamax,
            ]

        def get_output_activation_functions():
            return [
                tf.keras.activations.softmax,
                tf.keras.activations.sigmoid,
                tf.keras.activations.linear,
                tf.keras.activations.tanh,
                tf.keras.activations.relu,
            ]

    def create_model(
        x_train,
        y_train,
        save_to_path,
        model_name,
        uniqueWordCount,
        outputLength,
        embedding_output: int,
        lstm_unit: int,
        lstm_activation_function,
        lstm_count,
        dense_unit,
        dense_count,
        dense_activation_function,
        output_activation_function,
        loss_function,
        optimizer,
        epochs: int,
    ):
        try:
            i = tf.keras.layers.Input(shape=(x_train.shape[1]))

            x = tf.keras.layers.Embedding(uniqueWordCount + 1, embedding_output)(i)
            x = tf.keras.layers.MaxPooling1D()(x)
            for _ in range(lstm_count):
                x = tf.keras.layers.LSTM(
                    lstm_unit,
                    activation=lstm_activation_function,
                    return_sequences=True,
                )(x)
            x = tf.keras.layers.Flatten()(x)
            for _ in range(dense_count):
                x = tf.keras.layers.Dense(
                    dense_unit, activation=dense_activation_function
                )(x)
            x = tf.keras.layers.Dense(
                outputLength, activation=output_activation_function
            )(x)
            model = tf.keras.models.Model(i, x)
            model.compile(
                loss=loss_function,
                optimizer=optimizer(),
                metrics=["accuracy"],
            )

            prime_divisors = []
            for p in range(2, epochs // 2 + 1):
                if epochs % p == 0:
                    prime_divisors.append(p)
            if len(prime_divisors) == 0:
                prime_divisors = [0]
            patience = max(1, prime_divisors[int(len(prime_divisors) * 0.3)])
            reduce = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.005, patience=patience, verbose=1, mode="min"
            )
            early_stoping = tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=patience + 1, mode="min"
            )
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(save_to_path, model_name + "_checkpoint.h5"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            )
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                callbacks=[reduce, early_stoping, checkpoint],
            )
            return (
                model,
                history,
                os.path.join(save_to_path, model_name + "_checkpoint.h5"),
            )
        except:
            return None, None, None

    def test_model(
        model,
        x_test,
        y_test,
        train_data_shape,
        tokenizer,
        return_predictions: bool = False,
    ):
        last_predected = None
        predicted_same_c = 0
        predictions = []
        true_v = 0
        test_acc = 0
        for i, x in enumerate(x_test):
            print(f"test progress {(i / len(x_test)) * 100}%\r", end="")
            y = y_test[i]
            text_p = []
            # to lower
            value = [
                letters.lower() for letters in x if letters not in string.punctuation
            ]
            value = "".join(value)
            text_p.append(value)

            value = tokenizer.texts_to_sequences(text_p)
            value = np.array(value).reshape(-1)
            value = tf.keras.preprocessing.sequence.pad_sequences(
                [value], train_data_shape[0]
            )
            output = model.predict(value, verbose=0)
            output = output.argmax()
            le = LabelEncoder()
            le.fit(y_test)
            predicted_tag = le.inverse_transform([output])[0]
            predictions.append(predicted_tag)
            if predicted_tag == y:
                if predicted_tag == last_predected:
                    predicted_same_c += 1
                    if predicted_same_c >= len(y_test) * 0.5:
                        test_acc = -1
                        break
                else:
                    predicted_same_c = 0
                predicted_tag = last_predected
                true_v += 1
        if test_acc != -1:
            test_acc = true_v / len(x_test) * 100
        if not return_predictions:
            return test_acc
        return test_acc, predictions

    def _print_info(current_params, params, best_acc, last_acc, all_, c):
        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        print("Sit back and relax. This process will take a LONG time 😎")
        print("MODEL:Alzheimer")
        print("🎓 The model is training with these values:")
        output = ""
        emojis = {
            "embedding_outputs": "✒️",
            "lstm_units": "🔢",
            "lstm_activation_functions": "⚡️",
            "lstm_count_limit": "",
            "dense_units": "🧠",
            "dense_count_limit": "🔢",
            "dense_activation_functions": "⚡️",
            "output_activation_functions": "🧽",
            "loss_functions": "💔",
            "optimizers": "🚀",
        }
        for i, p in enumerate(params.__dict__.items()):
            t, p = str(p[0]), str(p[1])
            if t == "epochs":
                continue
            c_p = current_params[t]
            if i in [2, 6, 7, 8]:
                p = (
                    str(p)
                    .replace("<", "")
                    .replace(">", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split("at")[0]
                    .split("function")[-1]
                    .strip()
                )
                c_p = (
                    str(c_p)
                    .replace("<", "")
                    .replace(">", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split("at")[0]
                    .split("function")[-1]
                    .strip()
                )
            elif i in [9]:
                p = (
                    str(p)
                    .replace("<", "")
                    .replace(">", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split("object")[0]
                    .split(".")[-1]
                    .strip()
                )
                c_p = (
                    str(c_p)
                    .replace("<", "")
                    .replace(">", "")
                    .replace("[", "")
                    .replace("]", "")
                    .split("object")[0]
                    .split(".")[-1]
                    .strip()
                )
            output += f"{emojis[t]}{t} : {str(p).replace('[','').replace(']','').replace(str(c_p), '['+str(c_p)+']')}\n"
        print(output + "\n")
        print(f"📊 parameters tried: {c}/{all_}")
        print(f"💪 Train {str((c - 0) / (all_ - 0) * 100)}%")
        print(f"🏆 Best Accuracy: {'Overfitting' if best_acc == -1 else str(best_acc)}%")
        print(f"🕦 Last Accuracy: {'Overfitting' if last_acc == -1 else str(last_acc)}%")

    def FindModel(
        params: Params,
        data_path: str,
        test_data: str,
        save_to_path: str,
        model_name: str,
        tokenizer_num_words: int,
        gpu: bool = True,
        stop_massages: bool = True,
    ):
        try:
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            if not os.path.exists(save_to_path):
                raise FileNotFoundError(
                    "Cant find path🤷\ndata_path: " + str(save_to_path)
                )
            if tokenizer_num_words <= 0:
                raise ValueError("tokenizer_num_words must be bigger than 0 🔢")
            stpi = 0
            while True:
                if os.path.exists(
                    os.path.join(
                        save_to_path,
                        model_name
                        + "_Train"
                        + (("_" + str(stpi)) if stpi != 0 else ""),
                    )
                ):
                    stpi += 1
                    continue
                save_to_path = os.path.join(
                    save_to_path,
                    model_name + "_Train" + (("_" + str(stpi)) if stpi != 0 else ""),
                )
                os.mkdir(os.path.join(save_to_path))
                break
            best_model = None
            best_acc = None
            last_acc = None
            best_history = None
            best_values = {
                "embedding_output": None,
                "lstm_unit": None,
                "lstm_activation_function": None,
                "lstm_count": None,
                "dense_unit": None,
                "dense_count": None,
                "dense_activation_function": None,
                "output_activation_function": None,
                "loss_function": None,
                "optimizer": None,
                "epochs": None,
            }
            all_params = []
            if gpu:
                if stop_massages:
                    q = input("Using GPU. Are you sure about using GPU Y/n? 🙏")
                    q = q.lower()
                    if q == "n":
                        print("Using CPU 🥹")
                        gpu = False
                    print("Using GPU 😁")
                    time.sleep(3)
                else:
                    print("Using GPU 😁")
                    time.sleep(3)
            else:
                if stop_massages:
                    q = input("Using CPU. Are you sure about using CPU y/N? 🙅")
                    q = q.lower()
                    if q == "n":
                        print("Using GPU 😁")
                        gpu = True
                else:
                    print("Using CPU 🥹")
                    time.sleep(3)
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            print("Loading data 📚")
            
            (
                x_train,
                y_train,
                tokenizer,
                unique_word_count,
                output_length,
            ) = BurobotTextData.load_data_from_json(data_path, tokenizer_num_words)

            print(f"Loaded data 🗄️\nTrain Data length:{len(x_train)}")
            x_test, y_test = BurobotTextData.load_data_from_json_for_test(test_data)
            # test_x_train, _ = BurobotTextData.load_data_from_json_for_test(data_path)
            print(f"Loaded data 🗄️\nTest Data length:{len(x_test)}")
            time.sleep(3)
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            all_params = list(
                itertools.product(
                    params.output_activation_functions,
                    params.optimizers,
                    params.loss_functions,
                    params.dense_activation_functions,
                    params.lstm_activation_functions,
                    range(1, params.dense_count_limit + 1),
                    range(1, params.lstm_count_limit + 1),
                    params.dense_units,
                    params.lstm_units,
                    params.embedding_outputs,
                )
            )
            all_ = len(all_params)
            c = 0
            if stop_massages:
                q = input(
                    str(all_)
                    + " possibilities will be tried.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models.\nDo you want to continue? Y/n"
                )
                q = q.lower()
                if q == "n":
                    sys.exit()
            else:
                print(
                    str(all_)
                    + " possibilities will be tried.\nThis value depends on the parameters you enter.\nIf the number of possibilities is too high for you, you can change your parameters or you can use patience, patience value skips unsuccessful models."
                )
                time.sleep(5)
            epochs = params.epochs
            x_train_shape_1 = (x_train.shape[1],)
            while len(all_params) != 0:
                param = all_params[0]
                output_activation_function = param[0]
                optimizer = param[1]
                loss_function = param[2]
                dense_activation_function = param[3]
                lstm_activation_function = param[4]
                dense_count = param[5]
                lstm_count = param[6]
                dense_unit = param[7]
                lstm_unit = param[8]
                embedding_output = param[9]
                Learning._print_info(
                    {
                        "embedding_outputs": embedding_output,
                        "lstm_units": lstm_unit,
                        "lstm_activation_functions": lstm_activation_function,
                        "lstm_count_limit": lstm_count,
                        "dense_units": dense_unit,
                        "dense_count_limit": dense_count,
                        "dense_activation_functions": dense_activation_function,
                        "output_activation_functions": output_activation_function,
                        "loss_functions": loss_function,
                        "optimizers": optimizer,
                    },
                    params,
                    best_acc,
                    last_acc,
                    all_,
                    c,
                )
                c += 1
                model, history, checkpoint_path = Learning.create_model(
                    x_train,
                    y_train,
                    save_to_path,
                    model_name,
                    unique_word_count,
                    output_length,
                    embedding_output,
                    lstm_unit,
                    lstm_activation_function,
                    lstm_count,
                    dense_unit,
                    dense_count,
                    dense_activation_function,
                    output_activation_function,
                    loss_function,
                    optimizer,
                    epochs,
                )
                del all_params[0]
                if model is None or history is None:
                    last_acc = -1
                    continue
                model.save(os.path.join(save_to_path, model_name+"_last.h5"))
                print("Testing model 🥼")
                test_acc = Learning.test_model(
                    model, x_test, y_test, x_train_shape_1, tokenizer
                )
                if os.path.exists(checkpoint_path):
                    print("Testing checkpoint model 🥼")
                    checkpoint_model = tf.keras.models.load_model(checkpoint_path)
                    checkpoint_model.save(os.path.join(save_to_path, model_name+"_last(checkpoint).h5"))
                    checkpoint_test_acc = Learning.test_model(
                        checkpoint_model, x_test, y_test, x_train_shape_1, tokenizer
                    )
                    if checkpoint_test_acc > test_path:
                        test_path = checkpoint_test_acc
                        model = checkpoint_model
                        del checkpoint_model
                        checkpoint_path = None
                last_acc = test_acc
                if test_acc > (best_acc if best_acc is not None else 0):
                    best_acc = test_acc
                    best_model = model
                    best_model.save(os.path.join(save_to_path, model_name+"_best.h5"))
                    best_history = history
                    best_values = {
                        "embedding_output": embedding_output,
                        "lstm_unit": lstm_unit,
                        "lstm_activation_function": lstm_activation_function,
                        "lstm_count": lstm_count,
                        "dense_unit": dense_unit,
                        "dense_count": dense_count,
                        "dense_activation_function": dense_activation_function,
                        "output_activation_function": output_activation_function,
                        "loss_function": loss_function,
                        "optimizer": optimizer,
                        "epochs": epochs,
                    }

            return best_model, best_acc, best_values, best_history, tokenizer
        except KeyboardInterrupt:
            print("I-i stopped 🙌")
            sys.exit()
