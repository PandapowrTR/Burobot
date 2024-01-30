# BUROBOT
import os, sys, itertools, json, copy, typing, importlib, datetime

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput
from Burobot.tools import BurobotOther


class ModelSchemes:
    class ImageClassification:
        class Scheme1:
            def __init__(self, classCount: int, imageShape: tuple):
                """
                :classCount (int): This variable determines how many classes are in your data set.
                :imageShape (tuple): This variable determines the size of the images in the data set. (width, height, channel(1:gray, 3:rgb))
                """
                import tensorflow as tf

                self.params = {
                    "epochs": [50],
                    "batchSizes": [32],
                    "convRepeat": [1, 2, 3],
                    "convCount": [1, 2, 3],
                    "convActivationFunction": ["relu"],
                    "convFilters": [64, 128, 256, 512],
                    "convKernelSizes": [(3, 3)],
                    "convRegularizers": [
                        tf.keras.regularizers.l2,
                        tf.keras.regularizers.l1_l2,
                        None,
                    ],
                    "convRegularizerCoefficients": [(0.01, 0.01), (0.1, 0.1)],
                    "pool2DKernelSizes": [(2, 2)],
                    "denseCounts": [1, 2, 3],
                    "denseUnits": [256, 512],
                    "denseActivationFunctions": ["relu"],
                    "dropOuts": [0, 0.3, 0.5, 0.7],
                    "denseRegularizers": [
                        tf.keras.regularizers.l2,
                        tf.keras.regularizers.l1_l2,
                        None,
                    ],
                    "denseRegularizerCoefficients": [
                        (0.01, 0.01),
                        (0.1, 0.1),
                    ],  # if l1_l2 ind 0: l1, ind 1: l2 | if l2: ind 0:l2
                    "useBatchNormalization": [True, False],
                    "outputClassificationFunction": ["softmax"],
                    "compileOptimizerFunctions": ["adam"],
                    "compileLossFunctions": ["categorical_crossentropy"],
                    "modelValueTypes": ["uint8", "bfloat16", "float32", "float16"],
                }
                self.staticValues = {
                    "classCount": classCount,
                    "imageShape": imageShape,
                    "useMultiprocessing": True,
                    "libraryImportCodes": ["import tensorflow as tf"],
                }

            @staticmethod
            def trainModel(
                param: dict,
                _values: dict,
            ):
                """
                :param (dict): This variable represents the data to be used in the training of the model. It should be a dictionary obtained from the ModelSchemes.ImageClassification.Scheme1.params dictionary.
                :_values (dict): The following variables must be included in this variable: staticValues(static values to be used in model training (e.g. ModelSchemes.ImageClassification.Scheme1().staticValues)), trainData(training data to be used in model slope (such as Model.Schemes.ImageClassification.Scheme().loadData) must be loaded)), valData(validation data to be used in the model slope (must be loaded like Model.Schemes.ImageClassification.Scheme().loadData))
                """
                import tensorflow as tf
                import os, psutil, gc

                staticValues = None
                trainData = None
                valData = None
                saveToPath = None
                modelName = None
                if _values is not None:
                    staticValues = _values["staticValues"]
                    trainData = _values["trainData"]
                    valData = _values["valData"]
                    saveToPath = _values["saveToPath"]
                    modelName = _values["modelName"]

                model = tf.keras.models.Sequential()
                for c in range(param["convRepeat"]):
                    for _ in range(param["convCount"]):
                        convRegularizers = param["convRegularizers"]
                        if convRegularizers is not None:
                            if convRegularizers == tf.keras.regularizers.l2:
                                convRegularizers = convRegularizers(
                                    param["convRegularizerCoefficients"][0]
                                )
                            elif convRegularizers == tf.keras.regularizers.l1_l2:
                                convRegularizers = convRegularizers(
                                    param["convRegularizerCoefficients"][0],
                                    param["convRegularizerCoefficients"][1],
                                )
                            else:
                                # if no compatible regularizer. Throw error.
                                raise ValueError(
                                    "Incompatible Conv regularizer. Please use l2 or l1_l2."
                                )
                            if c == 0:
                                model.add(
                                    tf.keras.layers.Conv2D(
                                        param["convFilters"],
                                        param["convKernelSizes"],
                                        activation=param["convActivationFunction"],
                                        input_shape=staticValues["imageShape"],
                                        kernel_regularizer=convRegularizers,
                                        dtype=param["modelValueTypes"],
                                    )
                                )
                            else:
                                model.add(
                                    tf.keras.layers.Conv2D(
                                        param["convFilters"],
                                        param["convKernelSizes"],
                                        activation=param["convActivationFunction"],
                                        kernel_regularizer=convRegularizers,
                                        dtype=param["modelValueTypes"],
                                    )
                                )
                        else:
                            if c == 0:
                                model.add(
                                    tf.keras.layers.Conv2D(
                                        param["convFilters"],
                                        param["convKernelSizes"],
                                        activation=param["convActivationFunction"],
                                        input_shape=staticValues["imageShape"],
                                        dtype=param["modelValueTypes"],
                                    )
                                )
                            else:
                                model.add(
                                    tf.keras.layers.Conv2D(
                                        param["convFilters"],
                                        param["convKernelSizes"],
                                        activation=param["convActivationFunction"],
                                        dtype=param["modelValueTypes"],
                                    )
                                )

                    if c != len(range(param["convRepeat"])):
                        model.add(
                            tf.keras.layers.MaxPooling2D(
                                param["pool2DKernelSizes"], padding="same"
                            )
                        )
                model.add(tf.keras.layers.Flatten())
                for _ in range(param["denseCounts"]):
                    if param["denseRegularizers"] is not None:
                        # find what regularizer using
                        denseRegularizers = param["denseRegularizers"]
                        if denseRegularizers == tf.keras.regularizers.l2:
                            denseRegularizers = denseRegularizers(
                                param["denseRegularizerCoefficients"][0]
                            )
                        elif denseRegularizers == tf.keras.regularizers.l1_l2:
                            denseRegularizers = denseRegularizers(
                                param["denseRegularizerCoefficients"][0],
                                param["denseRegularizerCoefficients"][1],
                            )
                        else:
                            # if no compatible regularizer. Throw error.
                            raise ValueError(
                                "Incompatible Dense regularizer. Please use l2 or l1_l2."
                            )
                        model.add(
                            tf.keras.layers.Dense(
                                param["denseUnits"],
                                activation=param["denseActivationFunctions"],
                                kernel_regularizer=denseRegularizers,
                                dtype=param["modelValueTypes"],
                            )
                        )
                    else:
                        model.add(
                            tf.keras.layers.Dense(
                                param["denseUnits"],
                                activation=param["denseActivationFunctions"],
                                dtype=param["modelValueTypes"],
                            )
                        )

                    model.add(tf.keras.layers.Dropout(param["dropOuts"]))
                    if param["useBatchNormalization"]:
                        model.add(tf.keras.layers.BatchNormalization())
                model.add(
                    tf.keras.layers.Dense(
                        staticValues["classCount"],
                        activation=param["outputClassificationFunction"],
                        dtype=param["modelValueTypes"],
                    )
                )

                model.compile(
                    optimizer=param["compileOptimizerFunctions"],
                    loss=param["compileLossFunctions"],
                    metrics=["accuracy"],
                )
                prime_divisors = []
                for p in range(2, param["epochs"] // 2 + 1):
                    if param["epochs"] % p == 0:
                        prime_divisors.append(p)
                if len(prime_divisors) == 0:
                    prime_divisors = [0]
                patience = max(1, prime_divisors[int(len(prime_divisors) * 0.3)])
                reduce = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.005,
                    patience=patience,
                    verbose=1,
                    mode="min",
                    min_delta=0.0100,
                )
                early_stoping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=patience + 1, mode="min"
                )
                checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        saveToPath,
                        modelName + "-Checkpoint.h5",
                    ),
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                )
                num_cores = os.cpu_count()
                system_memory_gb = psutil.virtual_memory().total / (1024**3)

                workers = min(num_cores // 2, int(system_memory_gb // 2))
                workers = max(1, int(workers * 0.8))
                history = model.fit(
                    trainData,
                    epochs=param["epochs"],
                    validation_data=valData,
                    batch_size=param["batchSizes"],
                    callbacks=[early_stoping, reduce, checkpoint],
                    verbose=1,
                    use_multiprocessing=staticValues["useMultiprocessing"],
                    workers=workers,
                )

                gc.collect()
                tf.keras.backend.clear_session()
                return (
                    model,
                    history,
                    os.path.join(
                        saveToPath,
                        modelName + "-Checkpoint.h5",
                    ),
                )

            @staticmethod
            def saveModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: model, saveToPath
                """
                _values["model"].save(_values["saveToPath"] + ".h5")

            @staticmethod
            def loadModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: modelPath
                """
                import tensorflow as tf

                return tf.keras.models.load_model(_values["modelPath"])

            @staticmethod
            def hardwareSetup(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method.
                """
                import warnings
                import tensorflow as tf

                warnings.warn("ignore")
                if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
                    gpus = tf.config.experimental.list_physical_devices("GPU")
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.set_visible_devices([], "GPU")

            @staticmethod
            def loadData(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: data, batchSize, returnDataOnly
                """
                from Burobot.tools import BurobotImageData

                return BurobotImageData.ImageLoader().loadData(None, 1, True, _values)

            @staticmethod
            def splitData(_values: dict):
                """
                :_values (dict): If you want to split your data in a ratio you want. Add the following variable to this variable: splitRatio(split rates of your data (train, test) note: val = 1- (train + test)). The following values will be automatically populated during Grid Search: sourcePath, saveToPath, splitRatio
                """
                from Burobot.Data.Dominate import DominateImage

                return DominateImage.splitData("", "", (0.8, 0.1), _values)

            @staticmethod
            def testModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: model, testData, returnPredictions
                """
                import tensorflow as tf
                import numpy as np
                import gc

                model = None
                testData = None
                returnPredictions = False
                if _values is not None:
                    model = _values["model"]
                    testData = _values["testData"]
                    if "returnPredictions" in list(_values.keys()):
                        returnPredictions = _values["returnPredictions"]
                test_class_counts = testData.reduce(
                    tf.zeros(len(testData.class_names), dtype=tf.int32),
                    lambda x, y: x + tf.reduce_sum(tf.cast(y[1], tf.int32), axis=0),
                ).numpy()
                test_class_counts = dict(zip(testData.class_names, test_class_counts))
                test_size = 0

                test_acc = 0
                same_class_count = 0
                previous_prediction = 0
                predictions = []
                progress = 0
                for images, labels in testData:
                    progress += 1
                    percentage = (progress / len(testData)) * 100

                    print("\r" + " " * 80, end="")
                    print(f"\rTest Progress: %{percentage:.2f}", end="")
                    for i in range(len(images)):
                        test_size += 1
                        img = np.expand_dims(images[i].numpy() / 255, 0)
                        pred = np.argmax(model.predict(img, verbose=0))
                        pred = testData.class_names[pred]
                        real = testData.class_names[np.argmax(labels[i])]
                        # if prediction is true
                        if real == pred:
                            test_acc += 1
                        predictions.append(
                            {
                                "Real": real,
                                "Predicted": pred,
                                "result": real == pred,
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
                        if same_class_count >= int(
                            test_class_counts[previous_prediction] * 1.3
                        ):
                            tf.keras.backend.clear_session()
                            gc.collect()
                            print("\r" + " " * 80, end="")
                            print("\rTest Progress: %100", end="")
                            print()
                            del model
                            if returnPredictions:
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
                        class_predictions[key]
                        >= class_counts[key] * overfitting_threshold_big
                        or class_predictions[key]
                        <= class_counts[key] * overfitting_threshold_small
                    ):
                        if returnPredictions:
                            return -1, predictions
                        return -1

                print()
                test_acc = test_acc / test_size * 100
                if returnPredictions:
                    return test_acc, predictions
                return test_acc

    class NamedEntityRecognition:
        class Scheme1:
            def __init__(self, language: str, pipesAndLabels: dict):
                """
                :language (str): Your model's language.
                :pipesAndLabels (dict): A dictionary containing the pipes and label values to be used in the model. Each key of the dictionary is referred to as a pipe, and the value opposite each key is referred to as the labels of that pipe.
                """
                self.staticValues = {
                    "language": language,
                    "pipesAndLabels": pipesAndLabels,
                }
                self.params = {
                    "epochs": [10, 30, 50, 100],
                    "drops": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                }

            @staticmethod
            def trainModel(param, _values: dict):
                """
                :param (dict): This variable represents the data to be used in the training of the model. It should be a dictionary obtained from the ModelSchemes.NamedEntityRecognition.Scheme1.params dictionary.
                :_values (dict): The following variables must be included in this variable: staticValues(static values to be used in model training (e.g. ModelSchemes.NamedEntityRecognition.Scheme1().staticValues)), trainData(training data to be used in model slope (such as Model.Schemes.NamedEntityRecognition.Scheme().loadData) must be loaded))
                """
                import spacy, random
                from spacy.training.example import Example

                staticValues = _values["staticValues"]
                trainData = _values["trainData"]

                nlp = spacy.blank(staticValues["language"])
                for key, item in staticValues["pipesAndLabels"].items():
                    ner = nlp.add_pipe(key)
                    for l in item:
                        ner.add_label(l)

                data = []
                for text, annotations in trainData:
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    data.append(example)

                trainData = data
                nlp.begin_training()
                for epoch in range(param["epochs"]):
                    random.shuffle(trainData)
                    print(
                        f"Epoch {str(epoch)}/{str(len(range(param['epochs'])))}", end=""
                    )
                    for i, example in enumerate(trainData):
                        progress = ((i + 1) / len(trainData)) * 100
                        print(f" {progress:.1f}%", end="\r")
                        nlp.update([example], drop=param["drops"])
                return nlp, None, None

            @staticmethod
            def saveModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: model, saveToPath
                """
                import spacy

                _values["model"].to_disk(_values["saveToPath"])

            @staticmethod
            def loadModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: modelPath
                """
                import spacy

                return spacy.load(_values["modelPath"])

            @staticmethod
            def hardwareSetup(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method.
                """
                pass

            @staticmethod
            def loadData(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: data, batchSize, returnDataOnly
                """
                try:
                    import json

                    data = _values["data"]
                    with open(data, "r", encoding="utf-8") as j:
                        data = json.load(j)
                    data = data["data"]
                    for i, d in enumerate(data.copy()):
                        data[i] = tuple(d)
                    return data
                except:
                    pass

            @staticmethod
            def splitData(_values: dict):
                """
                :_values (dict): If you want to split your data in a ratio you want. Add the following variable to this variable: splitRatio(split rates of your data (train, test) note: val = 1- (train + test)). The following values will be automatically populated during Grid Search: sourcePath, saveToPath, splitRatio
                """
                from Burobot.Data.Dominate import DominateLabel

                splitRatio = (0.7, 0.3)
                if "splitRatio" in list(_values.keys()):
                    splitRatio = _values["splitRatio"]
                return DominateLabel.NamedEntityRecognition.splitData(
                    _values["sourcePath"], _values["saveToPath"], splitRatio
                )

            @staticmethod
            def testModel(_values: dict):
                """
                :_values (dict): There are no mandatory variables in this variable. However, this variable must be in the method. The following values will be automatically populated during Grid Search: model, testData, returnPredictions
                """
                import spacy

                correctPredictions = 0
                totalPredictions = 0

                testData = _values["testData"]
                model = _values["model"]
                for text, annotations in testData:
                    doc = model(text)
                    predictedEntities = [(ent.text, ent.label_) for ent in doc.ents]
                    actualEntities = [
                        (text[start:end], label)
                        for start, end, label in annotations.get("entities", [])
                    ]

                    for predictedEntity in predictedEntities:
                        if predictedEntity in actualEntities:
                            correctPredictions += 1

                    totalPredictions += len(predictedEntities)

                accuracy = (
                    correctPredictions / totalPredictions if totalPredictions > 0 else 0
                )
                return accuracy * 100

    def generateIterFromParamDict(paramDict: dict):
        """
        This function generates a list from a params dict for grid search.
        :paramsDict (dict): dict for grid search example: modelSchemes.scheme1
        """
        keys = list(paramDict.keys())
        values = list(paramDict.values())
        combinations = list(itertools.product(*values))
        result = [dict(zip(keys, c)) for c in combinations]
        return result


class GridSearchTrain:
    @staticmethod
    def __cleanDictForJson(d):
        cleanedDict = {}
        for key, value in d.items():
            try:
                json.dumps({key: value})
                cleanedDict[key] = value
            except:
                pass

        return cleanedDict

    @staticmethod
    def __saveParams(
        modelTrainmethod,
        modelTrainValues,
        modelSaveMethod,
        modelSaveMethodValues,
        modelLoadMethod,
        modelLoadMethodValues,
        hardwareSetupmethod,
        hardwareSetupmethodValues,
        loadDatamethod,
        loadDatamethodValues,
        splitDatamethod,
        splitDatamethodValues,
        modelTestmethod,
        modelTestmethodValues,
        paramsIter,
        usedParams,
        data,
        bestAccuracy,
        bestModelPath,
        bestHistory,
        saveToPath,
        modelName: str,
        loopIndex,
        saveName: str,
    ):
        paramsSaveToPath = os.path.join(saveToPath, "params")
        try:
            os.mkdir(paramsSaveToPath)
        except:
            pass
        paramsFile = {
            "modelName": modelName,
            "data": data,
            "saveToPath": saveToPath,
            "bestAccuracy": bestAccuracy,
            "bestModelPath": bestModelPath,
            "bestHistory": bestHistory,
            "loopIndex": loopIndex,
        }

        values = {
            "modelTrainValues": GridSearchTrain.__cleanDictForJson(modelTrainValues),
            "modelSaveMethodValues": GridSearchTrain.__cleanDictForJson(
                modelSaveMethodValues
            ),
            "modelLoadMethodValues": GridSearchTrain.__cleanDictForJson(
                modelLoadMethodValues
            ),
            "hardwareSetupmethodValues": GridSearchTrain.__cleanDictForJson(
                hardwareSetupmethodValues
            ),
            "loadDatamethodValues": GridSearchTrain.__cleanDictForJson(
                loadDatamethodValues
            ),
            "splitDatamethodValues": GridSearchTrain.__cleanDictForJson(
                splitDatamethodValues
            ),
            "modelTestmethodValues": GridSearchTrain.__cleanDictForJson(
                modelTestmethodValues
            ),
        }
        for key, value in values.copy().items():
            try:
                if type(value) == dict:
                    for key, p in value.copy().items():
                        newValue = []
                        changeValue = False
                        for _p in p:
                            fncStr = BurobotOther.convertModelFunctionToString(_p)
                            if fncStr is not None:
                                newValue.append(fncStr)
                                changeValue = True
                        if changeValue:
                            value[key] = newValue
                    values[key] = value
            except:
                pass
        paramsFile["methodValues"] = values

        functionNames = BurobotOther.saveFunctionsAsPythonFile(
            paramsSaveToPath,
            "functions",
            modelTrainmethod,
            modelSaveMethod,
            modelLoadMethod,
            hardwareSetupmethod,
            loadDatamethod,
            splitDatamethod,
            modelTestmethod,
        )
        paramsFile["functionsPath"] = os.path.join(paramsSaveToPath, "functions.py")
        paramsFile["modelTrainmethodName"] = functionNames[0].split("(")[0]
        paramsFile["modelSaveMethodName"] = functionNames[1].split("(")[0]
        paramsFile["modelLoadMethodName"] = functionNames[2].split("(")[0]
        paramsFile["hardwareSetupmethodName"] = functionNames[3].split("(")[0]
        paramsFile["loadDatamethodName"] = functionNames[4].split("(")[0]
        paramsFile["splitDatamethodName"] = functionNames[5].split("(")[0]
        paramsFile["modelTestmethodName"] = functionNames[6].split("(")[0]

        for i, p in enumerate(paramsIter):
            for key, _p in p.copy().items():
                fncStr = BurobotOther.convertModelFunctionToString(_p)
                if fncStr is not None:
                    paramsIter[i][key] = fncStr
        paramsFile["paramsIter"] = paramsIter

        for i, p in enumerate(usedParams):
            for key, _p in p.copy().items():
                fncStr = BurobotOther.convertModelFunctionToString(_p)
                if fncStr is not None:
                    usedParams[i][key] = fncStr
        paramsFile["usedParams"] = usedParams

        with open(
            os.path.join(paramsSaveToPath, saveName + ".json"),
            "w",
            encoding="utf-8",
        ) as j:
            json.dump(paramsFile, j, ensure_ascii=False)

    @staticmethod
    def __getFunctionFromFile(filePath, functionName):
        spec = importlib.util.spec_from_file_location("module", filePath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, functionName)

    @staticmethod
    def __readSavedParams(paramsPath: str):
        p = ""
        with open(paramsPath, "r", encoding="utf-8") as j:
            p = json.load(j)
        libs = [None]
        for lib in p["methodValues"]["modelTrainValues"]["staticValues"][
            "libraryImportCodes"
        ]:
            moduleName, alias = lib.split(" as ")
            moduleName = "".join(moduleName.split("import")[1:]).strip()
            libs.append(moduleName)
        for itemInd, item in enumerate(p.copy()["paramsIter"]):
            for key, value in item.copy().items():
                if type(value) in [str]:
                    for lib in libs:
                        try:
                            module, nvalue = value.rsplit(".", 1)
                            if lib is not None:
                                module = lib + "." + ".".join(module.split(".")[1:])
                            nvalue = getattr(importlib.import_module(module), nvalue)
                            p["paramsIter"][itemInd][key] = nvalue
                            break
                        except:
                            pass
        loopIndex = p["loopIndex"]
        paramsIter = p["paramsIter"]
        data = p["data"]
        saveToPath = p["saveToPath"]
        modelName = p["modelName"]
        usedParams = p["usedParams"]
        bestAccuracy = p["bestAccuracy"]
        bestHistory = p["bestHistory"]
        trainPath = p["data"][0]
        testPath = p["data"][1]
        valPath = p["data"][2]
        loadDatamethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["loadDatamethodName"]
        )
        loadDatamethodValues = p["methodValues"]["loadDatamethodValues"]
        modelTestmethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["modelTestmethodName"]
        )
        modelTestmethodValues = p["methodValues"]["modelTestmethodValues"]
        modelTrainmethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["modelTrainmethodName"]
        )
        modelTrainValues = p["methodValues"]["modelTrainValues"]
        modelSaveMethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["modelSaveMethodName"]
        )
        modelSaveMethodValues = p["methodValues"]["modelSaveMethodValues"]
        modelLoadMethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["modelLoadMethodName"]
        )
        modelLoadMethodValues = p["methodValues"]["modelLoadMethodValues"]
        hardwareSetupmethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["hardwareSetupmethodName"]
        )
        hardwareSetupmethodValues = p["methodValues"]["hardwareSetupmethodValues"]
        splitDatamethod = GridSearchTrain.__getFunctionFromFile(
            p["functionsPath"], p["splitDatamethodName"]
        )
        splitDatamethodValues = p["methodValues"]["splitDatamethodValues"]
        bestModel = None
        try:
            bestModel = modelLoadMethod({"modelPath": p["bestModelPath"]})
        except:
            pass
        bestModelPath = p["bestModelPath"]
        return (
            loopIndex,
            paramsIter,
            data,
            saveToPath,
            modelName,
            usedParams,
            bestAccuracy,
            bestModel,
            bestModelPath,
            bestHistory,
            trainPath,
            testPath,
            valPath,
            loadDatamethod,
            loadDatamethodValues,
            modelTestmethod,
            modelTestmethodValues,
            modelTrainmethod,
            modelTrainValues,
            modelSaveMethod,
            modelSaveMethodValues,
            modelLoadMethod,
            modelLoadMethodValues,
            hardwareSetupmethod,
            hardwareSetupmethodValues,
            splitDatamethod,
            splitDatamethodValues,
        )

    @staticmethod
    def __writeLog(
        message: str,
        saveToPath: str,
        fileNameAndFormat: str,
        writeTime: bool = True,
        splitterText: str = "\n" + ("_" * 100) + "\n",
    ):
        with open(os.path.join(saveToPath, fileNameAndFormat), "a") as l:
            l.write(
                (str(datetime.datetime.now()) + "\n" if writeTime else "") + message
            )
            l.write(splitterText)

    @staticmethod
    def __gridLoop(
        loopIndex,
        param,
        paramsIter,
        data,
        saveToPath,
        modelName,
        usedParams,
        bestAccuracy,
        bestModel,
        bestModelPath,
        bestHistory,
        trainPath,
        testPath,
        valPath,
        loadDatamethod,
        loadDatamethodValues,
        modelTestmethod,
        modelTestmethodValues,
        modelTrainmethod,
        modelTrainValues,
        modelSaveMethod,
        modelSaveMethodValues,
        modelLoadMethod,
        modelLoadMethodValues,
        hardwareSetupmethod,
        hardwareSetupmethodValues,
        splitDatamethod,
        splitDatamethodValues,
    ):
        try:
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            if "batchSize" in list(param.keys()):
                loadDatamethodValues.update({"batchSize": param["batchSize"]})
            # Getting train and test data
            loadDatamethodValues.update({"data": trainPath})
            trainData = loadDatamethod(loadDatamethodValues)
            loadDatamethodValues.update({"data": testPath})
            testData = loadDatamethod(loadDatamethodValues)
            loadDatamethodValues.update({"data": valPath})
            valData = loadDatamethod(loadDatamethodValues)

            modelTestmethodValues.update({"testData": testData})
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            # save last params
            GridSearchTrain.__saveParams(
                modelTrainmethod,
                modelTrainValues.copy(),
                modelSaveMethod,
                modelSaveMethodValues.copy(),
                modelLoadMethod,
                modelLoadMethodValues.copy(),
                hardwareSetupmethod,
                hardwareSetupmethodValues.copy(),
                loadDatamethod,
                loadDatamethodValues.copy(),
                splitDatamethod,
                splitDatamethodValues.copy(),
                modelTestmethod,
                modelTestmethodValues.copy(),
                copy.deepcopy(paramsIter),
                copy.deepcopy(usedParams),
                copy.deepcopy(data),
                copy.deepcopy(bestAccuracy),
                copy.deepcopy(bestModelPath),
                copy.deepcopy(bestHistory),
                copy.deepcopy(saveToPath),
                modelName,
                loopIndex,
                "unusedParams",
            )
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            currentSaveFolder = os.path.join(saveToPath, str(loopIndex) + "Train")
            while True:
                currentSaveFolder = os.path.join(saveToPath, str(loopIndex) + "Train")
                if not os.path.exists(currentSaveFolder):
                    os.mkdir(currentSaveFolder)
                    break
                loopIndex += 1
            print(
                "Model training with these values\n"
                + str("\n".join([(str(k) + ": " + str(p)) for (k, p) in param.items()]))
            )
            GridSearchTrain.__writeLog(
                "Model training with these values\n"
                + str(
                    "\n".join([(str(k) + ": " + str(p)) for (k, p) in param.items()])
                ),
                saveToPath,
                "log.log",
            )
            modelTrainValues.update(
                {
                    "modelName": modelName,
                    "trainData": trainData,
                    "valData": valData,
                    "saveToPath": currentSaveFolder,
                    "modelName": modelName,
                }
            )
            try:
                (
                    trainedModel,
                    trainedHistory,
                    checkpointModelPath,
                ) = modelTrainmethod(
                    param,
                    modelTrainValues,
                )
                modelSaveMethodValues.update(
                    {
                        "saveToPath": os.path.join(currentSaveFolder, "lastModel"),
                        "model": trainedModel,
                    }
                )
                modelSaveMethod(modelSaveMethodValues)
                modelTestmethodValues.update({"model": trainedModel})
            except Exception as e:
                usedParams.append(param)
                # save usedParams
                GridSearchTrain.__saveParams(
                    modelTrainmethod,
                    modelTrainValues.copy(),
                    modelSaveMethod,
                    modelSaveMethodValues.copy(),
                    modelLoadMethod,
                    modelLoadMethodValues.copy(),
                    hardwareSetupmethod,
                    hardwareSetupmethodValues.copy(),
                    loadDatamethod,
                    loadDatamethodValues.copy(),
                    splitDatamethod,
                    splitDatamethodValues.copy(),
                    modelTestmethod,
                    modelTestmethodValues.copy(),
                    copy.deepcopy(paramsIter),
                    copy.deepcopy(usedParams),
                    copy.deepcopy(data),
                    copy.deepcopy(bestAccuracy),
                    copy.deepcopy(bestModelPath),
                    copy.deepcopy(bestHistory),
                    copy.deepcopy(saveToPath),
                    modelName,
                    loopIndex,
                    "unusedParams",
                )
                GridSearchTrain.__writeLog(
                    "An error occurred during model training. Error: " + str(e),
                    saveToPath,
                    "log.log",
                )
                try:
                    BurobotOther.zipFolder(
                        currentSaveFolder,
                        os.path.join(saveToPath, str(loopIndex) + "Train.zip"),
                    )
                    BurobotOther.deleteFilesInFolder(currentSaveFolder)
                    os.rmdir(currentSaveFolder)
                except:
                    pass
                return None, None, None, usedParams

            accuracy = modelTestmethod(modelTestmethodValues)
            modelTestmethodValues.update({"model": None})
            GridSearchTrain.__writeLog(
                "Tested model. Accuracy: " + str(accuracy), saveToPath, "log.log"
            )
            if checkpointModelPath is not None:
                try:
                    modelLoadMethodValues.update({"modelPath": checkpointModelPath})
                    checkpointModel = modelLoadMethod(modelLoadMethodValues)
                    modelTestmethodValues.update({"model": checkpointModel})
                    checkpointAccuracy = modelTestmethod(modelTestmethodValues)
                    GridSearchTrain.__writeLog(
                        "Tested checkpoint model. Accuracy: " + str(checkpointAccuracy),
                        saveToPath,
                        "log.log",
                    )
                    if checkpointAccuracy > accuracy:
                        trainedModel = checkpointModel
                        accuracy = checkpointAccuracy
                        del checkpointModel, checkpointAccuracy
                except Exception as e:
                    GridSearchTrain.__writeLog(
                        "An error occurred in Checkpoint model testing. Checkpoint model testing is skipped. Error: "
                        + str(e),
                        saveToPath,
                        "log.log",
                    )

            if bestAccuracy < accuracy:
                bestModel = trainedModel
                bestAccuracy = accuracy
                bestHistory = trainedHistory
                GridSearchTrain.__writeLog(
                    "New best model! New best accuracy: " + str(bestAccuracy),
                    saveToPath,
                    "log.log",
                )
                modelSaveMethodValues.update(
                    {
                        "saveToPath": os.path.join(saveToPath, "bestModel"),
                        "model": bestModel,
                    }
                )
                modelSaveMethod(modelSaveMethodValues)
            BurobotOther.zipFolder(
                currentSaveFolder,
                os.path.join(saveToPath, str(loopIndex) + "Train.zip"),
            )
            BurobotOther.deleteFilesInFolder(currentSaveFolder)
            try:
                os.rmdir(currentSaveFolder)
            except:
                pass
            usedParams.append(param)
            # save usedParams
            GridSearchTrain.__saveParams(
                modelTrainmethod,
                modelTrainValues.copy(),
                modelSaveMethod,
                modelSaveMethodValues.copy(),
                modelLoadMethod,
                modelLoadMethodValues.copy(),
                hardwareSetupmethod,
                hardwareSetupmethodValues.copy(),
                loadDatamethod,
                loadDatamethodValues.copy(),
                splitDatamethod,
                splitDatamethodValues.copy(),
                modelTestmethod,
                modelTestmethodValues.copy(),
                copy.deepcopy(paramsIter),
                copy.deepcopy(usedParams),
                copy.deepcopy(data),
                copy.deepcopy(bestAccuracy),
                copy.deepcopy(bestModelPath),
                copy.deepcopy(bestHistory),
                copy.deepcopy(saveToPath),
                modelName,
                loopIndex,
                "unusedParams",
            )
        except KeyboardInterrupt:
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            GridSearchTrain.__writeLog(
                "Training stopped by user", saveToPath, "log.log"
            )
            print("Stoped!!!")
            sys.exit()
        except Exception as e:
            GridSearchTrain.__writeLog(
                "An unknown error occurred. Error: " + str(e), saveToPath, "log.log"
            )
        finally:
            try:
                BurobotOther.zipFolder(
                    currentSaveFolder,
                    os.path.join(saveToPath, str(loopIndex) + "Train.zip"),
                )
                BurobotOther.deleteFilesInFolder(currentSaveFolder)
                os.rmdir(currentSaveFolder)
            except:
                pass
        return bestModel, bestAccuracy, bestHistory, usedParams

    @staticmethod
    def newModel(
        modelName: str,
        data: typing.Union[str, list],
        saveToPath: str,
        modelTrainmethod: callable,
        modelTrainValues: dict,
        modelSaveMethod: callable,
        modelSaveMethodValues: dict,
        modelLoadMethod: callable,
        modelLoadMethodValues: dict,
        hardwareSetupmethod: callable,
        hardwareSetupmethodValues: dict,
        loadDatamethod: callable,
        loadDatamethodValues: dict,
        splitDatamethod: callable,
        splitDatamethodValues: dict,
        modelTestmethod: callable,
        modelTestmethodValues: dict,
    ):
        """A method that finds the best model using grid search.

        :modelName (str): Name of the model to be trained.

        :data (str | list): This is the path(s) to your data, split or unsplit.

        :saveToPath (str): The path to create the folder where all files related to the model will be saved.

        :modelTrainmethod (callable): The model slope method to be used in training your model. The values this method should take are: param (Dictionary containing the variables that your model will use in training), _values (Dictionary containing the other variables you use in the model).

        :modelTrainValues (dict): This variable is a dictionary containing the variables to be used in model training. This dictionary must necessarily contain the following values: params (Dictionary containing the variables to be used in model training), staticValues (static values to be used in model training). If you are using a ready-made method, you must enter the values in the method description.

        :modelSaveMethod (callable): This is the method used to save your model.

        :modelSaveMethodValues (dict): It is the dictionary that contains the variables to be used when saving your model. If you are using a ready-made method, you must enter the values in the method description.

        :modelLoadMethod (callable): This method is the method that allows loading the model.

        :modelLoadMethodValues (dict): A dictionary containing the values that your model loading method will use. If you are using a ready-made method, you must enter the values in the method description.

        :hardwareSetupmethod (callable): This is a method for setting up your hardware. (E.g. ModelSchemes.ImageClassification.Scheme1.hardwareSetup).

        :hardwareSetupmethodValues (dict): A dictionary containing the values used by your hardware Setup method. If you are using a ready-made method, you must enter the values in the method description.

        :loadDatamethod (callable): A method to upload your data. This method receives a dictionary data that contains the data that your method will use. The method must return the following values: loaded data.

        :loadDatamethodValues (dict): A dictionary containing the variables that your data loading method will use.

        :splitDatamethod (callable): This method is the method that will separate your data into training, testing and validation data. The values that this method must return are: training data path, test data path, validation data path

        :splitDatamethodValues (dict): This variable is a dictionary that contains the variables used by your data splitting method. If you are using a ready-made data splitting method, make sure that your method contains the variables in its description.

        :modelTestmethod (callable): This method is the testing method of your model. This method must return: accuracy(The accuracy of your model (you can return -1 for overfitting, -2 for error))

        :modelTestmethodValues (dict): This model is a dictionary that contains the variables that your testing method will use. If you are using a ready-made method, you must enter the values in the method description.

        Returns:
            bestModel, bestAccuracy, bestHistory
        """
        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()

        hardwareSetupmethod(hardwareSetupmethodValues)

        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()

        def catchError(
            dataPath: typing.Union[str, list],
            saveToPath: str,
        ):
            if not any([os.path.exists(dataPath), os.path.exists(saveToPath)]):
                raise FileNotFoundError(
                    f"One or more paths are not found. Please check your values.\ndataPath:{type(str(dataPath))}\n{str(dataPath)}\nsaveToPath:{str(type(saveToPath))}\n{str(saveToPath)}"
                )

        catchError(data, saveToPath)
        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()

        # Creating a folder that includes log file, unused params file, used params file, models etc.
        i = 0
        while True:
            trainFolderName = os.path.join(
                saveToPath, modelName + "Train" + str(i if i != 0 else "")
            )
            if os.path.exists(trainFolderName):
                i += 1
                continue
            os.mkdir(trainFolderName)
            saveToPath = trainFolderName
            del trainFolderName, i
            break
        # Write a log about training has starded.
        GridSearchTrain.__writeLog("Model training has started", saveToPath, "log.log")

        trainPath = None
        testPath = None
        valPath = None
        # if data is string that means we must split the dataset
        if type(data) == str:
            splitDatamethodValues.update({"sourcePath": data, "saveToPath": saveToPath})
            trainPath, testPath, valPath = splitDatamethod(splitDatamethodValues)
            data = [trainPath, testPath, valPath]
        # if data is list that means data is already splitted
        elif type(data) == list:
            trainPath, testPath, valPath = data

        bestModel = None
        bestModelPath = None
        bestAccuracy = 0
        bestHistory = None

        params = modelTrainValues["params"]
        staticValues = modelTrainValues["staticValues"]
        # Creating all params value. all params value contains all possibilities.
        paramsIter = ModelSchemes.generateIterFromParamDict(params)
        usedParams = []
        for loopIndex, param in enumerate(paramsIter):
            try:
                (
                    bestModel,
                    bestAccuracy,
                    bestHistory,
                    usedParams,
                ) = GridSearchTrain.__gridLoop(
                    loopIndex,
                    param,
                    paramsIter,
                    data,
                    saveToPath,
                    modelName,
                    usedParams,
                    bestAccuracy,
                    bestModel,
                    bestModelPath,
                    bestHistory,
                    trainPath,
                    testPath,
                    valPath,
                    loadDatamethod,
                    loadDatamethodValues,
                    modelTestmethod,
                    modelTestmethodValues,
                    modelTrainmethod,
                    modelTrainValues,
                    modelSaveMethod,
                    modelSaveMethodValues,
                    modelLoadMethod,
                    modelLoadMethodValues,
                    hardwareSetupmethod,
                    hardwareSetupmethodValues,
                    splitDatamethod,
                    splitDatamethodValues,
                )
            except:
                pass
            paramsIter.remove(param)
        return bestModel, bestAccuracy, bestHistory

    @staticmethod
    def oldModel(paramsFilePath: str):
        """
        A method that continues finding the best model using grid search.
        :paramsFilePath (str): The path to the .json file saved when training your model.
        """
        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()

        def _catchError(paramsFilePath: str):
            if not os.path.exists(paramsFilePath):
                raise FileNotFoundError(
                    "params file not found. Please check the path. paramsFilePath:\n"
                    + str(paramsFilePath)
                )

        _catchError(paramsFilePath)
        (
            loopIndex,
            paramsIter,
            data,
            saveToPath,
            modelName,
            usedParams,
            bestAccuracy,
            bestModel,
            bestModelPath,
            bestHistory,
            trainPath,
            testPath,
            valPath,
            loadDatamethod,
            loadDatamethodValues,
            modelTestmethod,
            modelTestmethodValues,
            modelTrainmethod,
            modelTrainValues,
            modelSaveMethod,
            modelSaveMethodValues,
            modelLoadMethod,
            modelLoadMethodValues,
            hardwareSetupmethod,
            hardwareSetupmethodValues,
            splitDatamethod,
            splitDatamethodValues,
        ) = GridSearchTrain.__readSavedParams(paramsFilePath)

        for param in paramsIter:
            (
                bestModel,
                bestAccuracy,
                bestHistory,
                usedParams,
            ) = GridSearchTrain.__gridLoop(
                loopIndex,
                param,
                paramsIter,
                data,
                saveToPath,
                modelName,
                usedParams,
                bestAccuracy,
                bestModel,
                bestModelPath,
                bestHistory,
                trainPath,
                testPath,
                valPath,
                loadDatamethod,
                loadDatamethodValues,
                modelTestmethod,
                modelTestmethodValues,
                modelTrainmethod,
                modelTrainValues,
                modelSaveMethod,
                modelSaveMethodValues,
                modelLoadMethod,
                modelLoadMethodValues,
                hardwareSetupmethod,
                hardwareSetupmethodValues,
                splitDatamethod,
                splitDatamethodValues,
            )
            loopIndex += 1
            paramsIter.remove(param)
        return bestModel, bestAccuracy, bestHistory, usedParams
