import sys, os, json

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))


class ObjectDetection:
    def convertAlbumentationsLabelsToYolo(labelsPath: str):
        if not os.path.exists(labelsPath):
            raise FileNotFoundError("Can't find path 🤷\nlabelsPath:" + str(labelsPath))

        classLabels = []

        for root, _, files in os.walk(labelsPath):
            for file in files:
                if str(file).endswith(".json"):
                    imgWidth, imgHeight = 0, 0
                    with open(os.path.join(root, file), "r") as labelFile:
                        labelData = json.load(labelFile)
                        imgWidth, imgHeight = (
                            labelData["imageWidth"],
                            labelData["imageHeight"],
                        )
                        corrs = labelData["shapes"]
                        for corr in corrs:
                            labelName = corr["label"]
                            if labelName not in classLabels:
                                classLabels.append(labelName)

        classLabels.sort()

        for root, _, files in os.walk(labelsPath):
            for file in files:
                if str(file).endswith(".json"):
                    with open(os.path.join(root, file), "r") as labelFile:
                        labelData = json.load(labelFile)
                        corrs = labelData["shapes"]
                        yoloLabelList = []

                        for corr in corrs:
                            labelName = corr["label"]
                            classIndex = classLabels.index(labelName)
                            corr = corr["points"][0]
                            xMin, yMin, xMax, yMax = corr[0], corr[1], corr[2], corr[3]
                            xCenter = (xMin + xMax) / 2.0
                            yCenter = (yMin + yMax) / 2.0
                            width = xMax - xMin
                            height = yMax - yMin

                            yoloLabel = [classIndex, xCenter, yCenter, width, height]
                            yoloLabelList.append(yoloLabel)

                        yoloTxtFilePath = os.path.join(
                            root, ".".join(file.split(".")[:-1]) + ".txt"
                        )
                        with open(yoloTxtFilePath, "w") as yoloTxt:
                            for i, yoloLabel in enumerate(yoloLabelList):
                                end = "\n"
                                if i == len(yoloLabelList) - 1:
                                    end = ""
                                yoloTxt.write(
                                    f"{yoloLabel[0]} {yoloLabel[1]} {yoloLabel[2]} {yoloLabel[3]} {yoloLabel[4]}"
                                    + end
                                )

                    # JSON dosyasını silin
                    os.remove(os.path.join(root, file))

    def splitLabelsToTxt(
        labelsPath: str,
        saveToPath: str,
        splitRatio: tuple = (0.8, 0.1),
        addToStart: str = "",
        labelsFileTypes: tuple = (".json", ".txt"),
    ):
        """NOTE splitRatio = (train, test), val = 1 - (train+test)"""
        if not os.path.exists(labelsPath):
            raise FileNotFoundError("Can't find path 🤷\nlabelsPath:" + str(labelsPath))
        if not os.path.exists(saveToPath):
            raise FileNotFoundError("Can't find path 🤷\nsaveToPath:" + str(saveToPath))

        if len(splitRatio) != 2 or any(
            splitRatio[i] < 0 for i in range(len(splitRatio))
        ):
            raise ValueError("split_raito value is invalid. Please check the data 🔢")

        print("Splitting labels to txt files 🔪📰")
        labels = []

        for root, _, files in os.walk(labelsPath):
            for file in files:
                if str(file).endswith(labelsFileTypes):
                    labels.append(str(addToStart) + file)

        trainLabels = []
        testLabels = []
        valLabels = []

        trainLabels = labels[: int(len(labels) * splitRatio[0])]
        testLabels = labels[
            int(len(labels) * splitRatio[0]) : int(
                len(labels) * (splitRatio[0] + splitRatio[1])
            )
        ]
        valLabels = labels[int(len(labels) * (splitRatio[0] + splitRatio[1])) :]
        train = ""
        for t in trainLabels:
            train += t + "\n"
        with open(os.path.join(saveToPath, "train.txt"), "w") as trainFile:
            trainFile.write(train.replace("\\", "/"))

        test = ""
        for t in testLabels:
            test += t + "\n"
        with open(os.path.join(saveToPath, "test.txt"), "w") as testFile:
            testFile.write(test.replace("\\", "/"))

        val = ""
        for v in valLabels:
            val += v + "\n"
        with open(os.path.join(saveToPath, "val.txt"), "w") as valFile:
            valFile.write(val.replace("\\", "/"))


class NamedEntityRecognition:
    def splitData(dataPath: str, saveToPath: str, splitRatio: tuple = (0.7, 0.2)):
        """
        Data splitter for Named Entity Recognition Datas.
        :dataPath (str): Path to data json file.
        :splitRatio (tuple): The value that determines the ratio in which your data will be divided. Note: (train, test), val: 1-(train+test).
        """

        data = None
        fileName = os.path.split(dataPath)[-1]
        with open(dataPath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data is None:
            raise ValueError(
                "An error occurred while reading the json file. dataPath:"
                + str(dataPath)
            )

        data = data["data"]

        trainItemsCount = int(len(data) * splitRatio[0])
        testItemsCount = int((len(data) - trainItemsCount) * splitRatio[1])
        valItemsCount = int(
            (len(data) - (trainItemsCount + testItemsCount))
            * (1 - (splitRatio[0] + splitRatio[1]))
        )

        trainData = []
        testData = []
        valData = []

        for train in data[:trainItemsCount]:
            trainData.append(train)

        for test in data[trainItemsCount : trainItemsCount + testItemsCount]:
            testData.append(test)

        for val in data[trainItemsCount + testItemsCount :]:
            valData.append(val)

        paths = []
        for item in [
            {"data": trainData, "name": "Train"},
            {"data": testData, "name": "Test"},
            {"data": valData, "name": "Val"},
        ]:
            paths.append(os.path.join(saveToPath, item["name"] + "-" + fileName))
            if len(item["data"]) != 0:
                with open(
                    os.path.join(saveToPath, item["name"] + "-" + fileName),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump({"data": item["data"]}, f, ensure_ascii=False)
        return paths[0], paths[1], paths[2]
