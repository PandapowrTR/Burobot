import sys, os, json
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))


class ObjectDetection:

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
        points, currentFormat, targetFormat, imageWidth, imageHeight
    ):
        """Converts annotation labels between different formats.

        :points (list): List of annotation points or information.
        :currentFormat (str): Current annotation format (e.g., 'pascal_voc', 'yolo', 'albumentations', 'coco').
        :targetFormat (str): Target annotation format.
        :imageWidth (int): The image width to use if required in the conversion process.
        :imageHeight (int): The image height to use if required in the conversion process.

        Returns:
            list: Converted annotation points in the target format.
        """
        if len(points) == 1:
            points = [points[0][0], points[0][1], points[0][2], points[0][3]]
        elif len(points) == 2:
            points = [points[0][0], points[0][1], points[1][0], points[1][1]]
        if currentFormat == targetFormat:
            return points

        if currentFormat == "pascal_voc":
            # convert to albumentations format
            if targetFormat == "albumentations":
                return np.divide(
                    [
                        points[0],
                        points[1],
                        points[2],
                        points[3],
                    ],
                    [imageWidth, imageHeight, imageWidth, imageHeight],
                )
            # Convert to yolo format
            elif targetFormat == "yolo":
                xCenter = (points[0] + points[2]) / (2 * imageWidth)
                yCenter = (points[1] + points[3]) / (2 * imageHeight)
                width = abs(points[2] - points[0]) / imageWidth
                height = abs(points[3] - points[1]) / imageHeight

                return [xCenter, yCenter, width, height]
            # Convert to coco format
            elif targetFormat == "coco":
                xmin, ymin, xmax, ymax = points
                x = xmin
                y = ymin
                width = xmax - xmin
                height = ymax - ymin

                return [x, y, width, height]

            else:
                raise ValueError(
                    f"Conversion from {currentFormat} to {targetFormat} is not supported."
                )

        elif currentFormat == "yolo":
            if targetFormat == "pascal_voc":
                xCenter, yCenter, width, height = points
                xMin = int((xCenter - width / 2) * imageWidth)
                yMin = int((yCenter - height / 2) * imageHeight)
                xMax = int((xCenter + width / 2) * imageWidth)
                yMax = int((yCenter + height / 2) * imageHeight)
                return [xMin, yMin, xMax, yMax]

            elif targetFormat == "albumentations":
                points = ObjectDetection.convertLabelPoints(
                    points, currentFormat, "pascal_voc", imageWidth, imageHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetFormat, imageWidth, imageHeight
                )

            elif targetFormat == "coco":
                xCenter, yCenter, width, height = points

                xMin = int((xCenter - width / 2) * imageWidth)
                yMin = int((yCenter - height / 2) * imageHeight)
                cocoWidth = int(width * imageWidth)
                cocoHeight = int(height * imageHeight)

                return [xMin, yMin, cocoWidth, cocoHeight]

            else:
                raise ValueError(
                    f"Conversion from {currentFormat} to {targetFormat} is not supported."
                )

        elif currentFormat == "albumentations":
            if targetFormat == "pascal_voc":
                xMin, yMin, xMax, yMax = points
                return [
                    xMin * imageWidth,
                    yMin * imageHeight,
                    xMax * imageWidth,
                    yMax * imageHeight,
                ]

            elif targetFormat == "yolo":
                points = ObjectDetection.convertLabelPoints(
                    points, currentFormat, "pascal_voc", imageWidth, imageHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetFormat, imageWidth, imageHeight
                )

            elif targetFormat == "coco":
                points = ObjectDetection.convertLabelPoints(
                    points, currentFormat, "pascal_voc", imageWidth, imageHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetFormat, imageWidth, imageHeight
                )

            else:
                raise ValueError(
                    f"Conversion from {currentFormat} to {targetFormat} is not supported."
                )

        elif currentFormat == "coco":
            if targetFormat == "pascal_voc":
                xmin, ymin, width, height = points

                xmax = xmin + width
                ymax = ymin + height

                return [xmin, ymin, xmax, ymax]

            elif targetFormat == "yolo":
                points = ObjectDetection.convertLabelPoints(
                    points, currentFormat, "pascal_voc", imageWidth, imageHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetFormat, imageWidth, imageHeight
                )

            elif targetFormat == "albumentations":

                points = ObjectDetection.convertLabelPoints(
                    points, currentFormat, "pascal_voc", imageWidth, imageHeight
                )
                return ObjectDetection.convertLabelPoints(
                    points, "pascal_voc", targetFormat, imageWidth, imageHeight
                )

            else:
                raise ValueError(
                    f"Conversion from {currentFormat} to {targetFormat} is not supported."
                )

        else:
            raise ValueError(f"Conversion from {currentFormat} is not supported.")

        return points

    @staticmethod
    def convertAllLabelPoints(
        points,
        currentLabelFormat,
        targetFormat,
        imageWidth,
        imgHeight,
    ):
        for key, value in points.copy().items():
            for i, v in enumerate(value.copy()):
                value[i]["bbox"] = ObjectDetection.convertLabelPoints(
                    v["bbox"], currentLabelFormat, targetFormat, imageWidth, imgHeight
                )
                value[i]["labelFormat"] = targetFormat
            points[key] = value
        return points

    @staticmethod
    def convertAllLabelFiles(
        labelsPath,
        dataPath,
        saveToPath,
        currentLabelFormat,
        targetFormat,
        imageWidth,
        imageHeight,
    ):
        points, classes = ObjectDetection.loadAllLabels(
            labelsPath, dataPath, currentLabelFormat, True
        )
        for key, value in points.copy().items():
            for i, v in enumerate(value.copy()):
                value[i]["bbox"] = ObjectDetection.convertLabelPoints(
                    v["bbox"], currentLabelFormat, targetFormat, imageWidth, imageHeight
                )
                value[i]["labelFormat"] = targetFormat
                if targetFormat in ["yolo", "coco"]:
                    classNumbers = {}
                    for clsi, clsl in enumerate(classes):
                        classNumbers[clsl] = clsi
                    value[i]["classNumbers"] = classNumbers
            points[key] = value
            ObjectDetection.saveLabel(saveToPath+"/"+str(key), points[key])

    @staticmethod
    def loadLabel(labelPath, labelFormat, imageWidth, imageHeight):
        """
        Load labels from different formats.

        :labelPath (str): Path to the label file.
        :labelFormat (str): Format of the labels ("pascal_voc", "yolo", "coco").
        :imageWidth (int): Width of the image.
        :imageHeight (int): Height of the image.

        Returns:
            list: List of dictionaries containing label information.
        """
        if labelFormat == "pascal_voc":
            tree = ET.parse(labelPath)
            root = tree.getroot()
            labels = []
            for obj in root.findall(".//object"):
                label = obj.find("name").text
                bbox = obj.find("bndbox")
                bbox = [
                    float(x)
                    for x in [
                        bbox.find("xmin").text,
                        bbox.find("ymin").text,
                        bbox.find("xmax").text,
                        bbox.find("ymax").text,
                    ]
                ]
                labels.append(
                    {
                        "labelFormat": labelFormat,
                        "label": label,
                        "bbox": bbox,
                        "imageWidth": imageWidth,
                        "imageHeight": imageHeight,
                    }
                )

            return labels
        elif labelFormat == "yolo":
            with open(labelPath, "r") as file:
                lines = file.readlines()

                labels = []
                for line in lines:
                    label, x, y, width, height = map(float, line.split())
                    labels.append(
                        {
                            "labelFormat": labelFormat,
                            "label": int(label),
                            "bbox": [x, y, width, height],
                            "imageWidth": imageWidth,
                            "imageHeight": imageHeight,
                        }
                    )

                return labels

        elif labelFormat == "coco":
            with open(labelPath, "r") as file:
                cocoData = json.load(file)

            labels = []
            for annotation in cocoData["annotations"]:
                labelId = annotation["category_id"]
                label = [
                    category
                    for category in cocoData["categories"]
                    if category["id"] == labelId
                ][0]
                label = label["name"]

                bbox = annotation["bbox"]

                labels.append(
                    {
                        "labelFormat": labelFormat,
                        "label": label,
                        "bbox": bbox,
                        "categories": cocoData["categories"],
                        "imageWidth": imageWidth,
                        "imageHeight": imageHeight,
                    }
                )

            return labels
        else:
            raise ValueError("Unsupported label format")

    def saveLabel(saveToPath, label):
        """
        label (list): must be loaded like ObjectDetection.loadLabel
        """
        if label[0]["labelFormat"] == "pascal_voc":
            xmlPath = saveToPath + ".xml"
            root = ET.Element("annotation")

            folder = ET.SubElement(root, "folder")
            folder.text = os.path.split(xmlPath)[0]
            filename = ET.SubElement(root, "filename")
            filename.text = ".".join(os.path.split(xmlPath)[1].split(".")[:-1])
            size = ET.SubElement(root, "size")
            width = ET.SubElement(size, "width")
            height = ET.SubElement(size, "height")

            width.text = str(label[0]["imageWidth"])
            height.text = str(label[0]["imageHeight"])
            for l in label:
                objectElem = ET.SubElement(root, "object")
                name = ET.SubElement(objectElem, "name")
                name.text = l["label"]
                bbox = ET.SubElement(objectElem, "bndbox")
                xmin = ET.SubElement(bbox, "xmin")
                xmin.text = str(l["bbox"][0])
                ymin = ET.SubElement(bbox, "ymin")
                ymin.text = str(l["bbox"][1])
                xmax = ET.SubElement(bbox, "xmax")
                xmax.text = str(l["bbox"][2])
                ymax = ET.SubElement(bbox, "ymax")
                ymax.text = str(l["bbox"][3])

            tree = ET.ElementTree(root)
            with open(xmlPath, "wb") as file:
                tree.write(file)

        elif label[0]["labelFormat"] == "yolo":
            txtPath = saveToPath + ".txt"
            yoloData = ""
            classes = set()
            for l in label:
                if "classNumbers" in list(l.keys()):
                    classes.add(l["label"])
                    l["label"] = l["classNumbers"][l["label"]]
                yoloData += f"{l['label']} {l['bbox'][0]} {l['bbox'][1]} {l['bbox'][2]} {l['bbox'][3]}\n"
            with open(txtPath, "w") as file:
                file.write(yoloData)
            classFile = os.path.join(os.path.split(txtPath)[0], "classes.txt")

            with open(classFile, "a+") as cl:
                cl.seek(0)
                existing_classes = cl.read().splitlines()

                classes.update(filter(None, existing_classes))

                cl.seek(0)
                cl.truncate()
                cl.write("\n".join(sorted(classes)))
        elif label[0]["labelFormat"] == "coco":
            jsonPath = saveToPath + ".json"
            cocoData = {
                "images": [],
                "annotations": [],
                "categories": [],
            }
            classes = set()
            if "classNumbers" in list(label[0].keys()):
                for clsk in label[0]["classNumbers"].keys():
                    classes.add(clsk)
            classes = sorted(list(classes))
            fileName = os.path.split(saveToPath)[1]
            for l in label:
                cocoData["images"].append(
                    {
                        "id": 1,
                        "width": l["imageWidth"],
                        "height": l["imageHeight"],
                        "file_name": fileName,
                    }
                )
                cocoData["annotations"].append(
                    {
                        "id": 1,
                        "image_id": 1,
                        "bbox": [
                            l["bbox"][0],
                            l["bbox"][1],
                            l["bbox"][2],
                            l["bbox"][3],
                        ],
                        "category_id": classes.index(l["label"]),
                        "area": l["bbox"][2] * l["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                cocoData["categories"].append(
                    {
                        "id": classes.index(l["label"]),
                        "name": l["label"],
                        "supercategory": l["label"],
                    }
                )
            newCategories = []

            for c in cocoData["categories"].copy():
                if c not in newCategories:
                    newCategories.append(c)
            cocoData["categories"] = newCategories
            del newCategories

            with open(jsonPath, "w") as file:
                json.dump(cocoData, file)
        else:
            raise ValueError("Unsupported label format")

    @staticmethod
    def loadAllLabels(labelsPath, dataPath, labelsFormat, returnClasses: bool = False):
        """
        Load object detection labels from the specified directory using the given label format.

        :labelsPath (str): The path to the directory containing label files.
        :labelsFormat (str): The format of the label files. Supported formats: "pascal_voc", "coco", "yolo".
        :imageWidth (int): The width of the images associated with the labels.
        :imageHeight (int): The height of the images associated with the labels.
        :returnClasses (bool): Option to return all classes

        Returns:
        - labels (dict): A dictionary containing all loaded labels. The key values of the dictionary are the extentions of the file names removed.
        - classes (list): A list containing all classes.
        """

        data = {}

        for root, _, files in os.walk(dataPath):
            for file in files:
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(os.path.join(root, file))
                    data[".".join(file.split(".")[:-1])] = {
                        "imageWidth": img.width,
                        "imageHeight": img.height,
                    }

        labels = {}
        classes = []
        for root, _, files in os.walk(labelsPath):
            labelExtention = ""
            if labelsFormat == "pascal_voc":
                labelExtention == ".xml"
            elif labelsFormat == "coco":
                labelExtention = ".json"
            elif labelsFormat == "yolo":
                labelExtention == ".txt"
            else:
                raise ValueError(
                    "Invalid labels format. labelsFormat: " + str(labelsFormat)
                )
            for file in files:
                if file.lower().endswith(labelExtention):
                    label = ObjectDetection.loadLabel(
                        os.path.join(root, file),
                        labelsFormat,
                        data[".".join(file.split(".")[:-1])]["imageWidth"],
                        data[".".join(file.split(".")[:-1])]["imageHeight"],
                    )
                    labels[".".join(file.split(".")[:-1])] = label
                    if returnClasses:
                        for l in label:
                            classes.append(l["label"])
        if returnClasses:
            classes = np.unique(classes).tolist()
            return labels, sorted(classes)
        return labels


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
