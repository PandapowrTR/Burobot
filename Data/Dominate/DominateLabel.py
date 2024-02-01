import sys, os, json
import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))


class ObjectDetection:
    @staticmethod
    def convertAllLabelPoints(labelsPath: str, currentType, targetType):
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

                            yoloLabel = [
                                classIndex
                            ] + ObjectDetection.convertLabelPoints(
                                corr, currentType, targetType, imgWidth, imgHeight
                            )
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

                    os.remove(os.path.join(root, file))

    @staticmethod
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

    @staticmethod
    def convertLabelPoints(
        points, currentType, targetType, imgWidth=None, imgHeight=None
    ):
        """Converts annotation labels between different formats.

        :points (list): List of annotation points or information.
        :currentType (str): Current annotation format (e.g., 'pascal_voc', 'yolo', 'albumentations', 'coco').
        :targetType (str): Target annotation format.
        :imgWidth (int): The image width to use if required in the conversion process.
        :imgHeight (int): The image height to use if required in the conversion process.

        Returns:
            list: Converted annotation points in the target format.
        """
        if len(points) == 1:
            points = [points[0][0], points[0][1], points[0][2], points[0][3]]
        elif len(points) == 2:
            points = [points[0][0], points[0][1], points[1][0], points[1][1]]
        if currentType == targetType:
            return points

        if currentType == "pascal_voc":
            # convert to albumentations format
            if targetType == "albumentations":
                return np.divide(
                    [
                        points[0],
                        points[1],
                        points[2],
                        points[3],
                    ],
                    [imgWidth, imgHeight, imgWidth, imgHeight],
                )
            # Convert to yolo format
            elif targetType == "yolo":
                xCenter = (points[0] + points[2]) / (2 * imgWidth)
                yCenter = (points[1] + points[3]) / (2 * imgHeight)
                width = abs(points[2] - points[0]) / imgWidth
                height = abs(points[3] - points[1]) / imgHeight

                return [xCenter, yCenter, width, height]
            # Convert to coco format
            elif targetType == "coco":
                xmin, ymin, xmax, ymax = points
                x = xmin
                y = ymin
                width = xmax - xmin
                height = ymax - ymin

                return [x, y, width, height]

            else:
                raise ValueError(
                    f"Conversion from {currentType} to {targetType} is not supported."
                )

        elif currentType == "yolo":
            if targetType == "pascal_voc":
                xCenter, yCenter, width, height = points
                xMin = int((xCenter - width / 2) * imgWidth)
                yMin = int((yCenter - height / 2) * imgHeight)
                xMax = int((xCenter + width / 2) * imgWidth)
                yMax = int((yCenter + height / 2) * imgHeight)
                return [xMin, yMin, xMax, yMax]

            elif targetType == "albumentations":
                points = ObjectDetection.convertLabelPoints(
                    points, currentType, "pascal_voc", imgWidth, imgHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetType, imgWidth, imgHeight
                )

            elif targetType == "coco":
                xCenter, yCenter, width, height = points

                xMin = int((xCenter - width / 2) * imgWidth)
                yMin = int((yCenter - height / 2) * imgHeight)
                cocoWidth = int(width * imgWidth)
                cocoHeight = int(height * imgHeight)

                return [xMin, yMin, cocoWidth, cocoHeight]

            else:
                raise ValueError(
                    f"Conversion from {currentType} to {targetType} is not supported."
                )

        elif currentType == "albumentations":
            if targetType == "pascal_voc":
                xMin, yMin, xMax, yMax = points
                return [
                    xMin * imgWidth,
                    yMin * imgHeight,
                    xMax * imgWidth,
                    yMax * imgHeight,
                ]

            elif targetType == "yolo":
                points = ObjectDetection.convertLabelPoints(
                    points, currentType, "pascal_voc", imgWidth, imgHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetType, imgWidth, imgHeight
                )

            elif targetType == "coco":
                points = ObjectDetection.convertLabelPoints(
                    points, currentType, "pascal_voc", imgWidth, imgHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetType, imgWidth, imgHeight
                )

            else:
                raise ValueError(
                    f"Conversion from {currentType} to {targetType} is not supported."
                )

        elif currentType == "coco":
            if targetType == "pascal_voc":
                xmin, ymin, width, height = points

                xmax = xmin + width
                ymax = ymin + height

                return [xmin, ymin, xmax, ymax]

            elif targetType == "yolo":
                points = ObjectDetection.convertLabelPoints(
                    points, currentType, "pascal_voc", imgWidth, imgHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetType, imgWidth, imgHeight
                )

            elif targetType == "albumentations":

                points = ObjectDetection.convertLabelPoints(
                    points, currentType, "pascal_voc", imgWidth, imgHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetType, imgWidth, imgHeight
                )

            else:
                raise ValueError(
                    f"Conversion from {currentType} to {targetType} is not supported."
                )

        else:
            raise ValueError(f"Conversion from {currentType} is not supported.")

        return points


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
