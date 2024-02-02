import os, shutil, os, gc, json, time, sys, uuid, copy

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from PIL import Image
import numpy as np
import albumentations as alb
from concurrent.futures import ThreadPoolExecutor
from Burobot.tools import BurobotOutput
from Burobot.Data.Dominate.DominateImage import *  # type: ignore
from Burobot.Data.Dominate.DominateLabel import *


def splitDataToTxt(
    dataPath: str,
    labelsPath: str,
    saveToPath: str,
    splitRatio: tuple = (0.8, 0.1),
    addToStart: str = "",
    labelsFileTypes: tuple = (".json", ".txt"),
):
    """NOTE splitRatio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find path 🤷\ndataPath:" + str(dataPath))
    if not os.path.exists(labelsPath):
        raise FileNotFoundError("Can't find path 🤷\nlabelsPath:" + str(labelsPath))
    if not os.path.exists(saveToPath):
        raise FileNotFoundError("Can't find path 🤷\nsaveToPath:" + str(saveToPath))

    if len(splitRatio) != 2 or any(splitRatio[i] < 0 for i in range(len(splitRatio))):
        raise ValueError("splitRatio value is invalid. Please check the data 🔢")

    print("Splitting images and labels to txt files 🔪📰")
    images = []
    labels = []

    for root, _, files in os.walk(dataPath):
        for file in files:
            if str(file).endswith((".png", ".jpg", ".jpeg")):
                images.append(str(addToStart) + file)

    for root, _, files in os.walk(labelsPath):
        for file in files:
            if str(file).endswith(labelsFileTypes):
                labels.append(str(addToStart) + file)

    trainImages = images[: int(len(images) * splitRatio[0])]
    testImages = images[
        int(len(images) * splitRatio[0]) : int(
            len(images) * (splitRatio[0] + splitRatio[1])
        )
    ]
    valImages = images[int(len(images) * (splitRatio[0] + splitRatio[1])) :]

    trainLabels = labels[: int(len(labels) * splitRatio[0])]
    testLabels = labels[
        int(len(labels) * splitRatio[0]) : int(
            len(labels) * (splitRatio[0] + splitRatio[1])
        )
    ]
    valLabels = labels[int(len(labels) * (splitRatio[0] + splitRatio[1])) :]

    trainData = ""
    for img, lbl in zip(trainImages, trainLabels):
        trainData += f"{img}\n{lbl}\n"

    testData = ""
    for img, lbl in zip(testImages, testLabels):
        testData += f"{img}\n{lbl}\n"

    valData = ""
    for img, lbl in zip(valImages, valLabels):
        valData += f"{img}\n{lbl}\n"

    with open(os.path.join(saveToPath, "train.txt"), "w") as trainFile:
        trainFile.write(trainData.replace("\\", "/"))

    with open(os.path.join(saveToPath, "test.txt"), "w") as testFile:
        testFile.write(testData.replace("\\", "/"))

    with open(os.path.join(saveToPath, "val.txt"), "w") as valFile:
        valFile.write(valData.replace("\\", "/"))


def splitDataToFolders(
    dataPath: str,
    labelsPath: str,
    saveToPath: str,
    splitRatio: tuple = (0.8, 0.1),
    labelsFileTypes: tuple = (".json", ".txt"),
):
    """NOTE splitRatio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find path 🤷\ndataPath:" + str(dataPath))
    if not os.path.exists(labelsPath):
        raise FileNotFoundError("Can't find path 🤷\nlabelsPath:" + str(labelsPath))
    if not os.path.exists(saveToPath):
        raise FileNotFoundError("Can't find path 🤷\nsaveToPath:" + str(saveToPath))

    if len(splitRatio) != 2 or any(splitRatio[i] < 0 for i in range(len(splitRatio))):
        raise ValueError("splitRatio value is invalid. Please check the data 🔢")

    print("Splitting images and labels to txt files 🔪📁")

    trainImgsPath = os.path.join(saveToPath, "train/images")
    testImgsPath = os.path.join(saveToPath, "test/images")
    valImgsPath = os.path.join(saveToPath, "val/images")

    trainLabelsPath = os.path.join(saveToPath, "train/labels")
    testLabelsPath = os.path.join(saveToPath, "test/labels")
    valLabelsPath = os.path.join(saveToPath, "val/labels")

    os.makedirs(trainImgsPath, exist_ok=True)
    os.makedirs(trainLabelsPath, exist_ok=True)
    os.makedirs(valImgsPath, exist_ok=True)
    os.makedirs(valLabelsPath, exist_ok=True)
    os.makedirs(testImgsPath, exist_ok=True)
    os.makedirs(testLabelsPath, exist_ok=True)

    datas = {}
    for root, _, files in os.walk(dataPath):
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png")):
                a = ".".join(file.split(".")[:-1])
                datas[".".join(file.split(".")[:-1])] = os.path.join(root, file)
    labels = {}
    for root, _, files in os.walk(labelsPath):
        for file in files:
            if file.endswith(labelsFileTypes):
                labels[".".join(file.split(".")[:-1])] = os.path.join(root, file)
    totalFiles = len(datas)
    numTrain = int(totalFiles * splitRatio[0])
    numTest = int(totalFiles * splitRatio[1])

    trainIndex = 0
    testIndex = 0
    for key, value in datas.items():
        if trainIndex < numTrain:
            trainIndex += 1
            shutil.copy(value, trainImgsPath)
            shutil.copy(labels[key], trainLabelsPath)
        elif testIndex < numTest:
            testIndex += 1
            shutil.copy(value, testImgsPath)
            shutil.copy(labels[key], testLabelsPath)
        else:
            shutil.copy(value, valImgsPath)
            shutil.copy(labels[key], valLabelsPath)

    print("Data splitting completed successfully 🥳")


def deleteAloneData(dataPath: str, labelsPath: str = None):  # type: ignore
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))
    if labelsPath is None:
        labelsPath = dataPath
    if labelsPath is not None:
        if not os.path.exists(labelsPath):
            raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))

    labelFiles = [[], []]  # 0: file with root, 1:file
    for root, _, files in os.walk(labelsPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith(("json", "txt", "xml")):
                labelFiles[0].append(os.path.join(root, file))
                labelFiles[1].append(".".join(file.split(".")[:-1]))

    imageFiles = [[], []]  # 0: file with root, 1:file
    for root, _, files in os.walk(dataPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith(("jpg", "png", "jpeg")):
                imageFiles[0].append(os.path.join(root, file))
                imageFiles[1].append(".".join(file.split(".")[:-1]))

    for i, image in enumerate(imageFiles[1]):
        try:
            labelFiles[1].index(image)
        except ValueError:
            os.remove(imageFiles[0][i])
    for i, label in enumerate(labelFiles[1]):
        try:
            imageFiles[1].index(label)
        except:
            os.remove(labelFiles[0][i])


def deleteSimilarDetections(
    dataPath, labelsPath, labelFormat="albumentations", p: float = 0.9
):
    """
    Delete detections that are similar to each other within a given directory.

    Args:
        dataPath (str): The path to the directory containing the images.
        labelsPath (str): The path to the directiory containing the labels.
        labelFormat (str): The format of label values(albumentations(default), yolo, pascal_voc, coco)
        p (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        int: The number of deleted similar detection files.
    """

    def processFile(file1):
        global deletedCount
        for file2 in files:
            if (
                file1 != file2
                and str(file1).endswith((".jpg", ".png", ".jpeg"))
                and str(file2).endswith((".jpg", ".png", ".jpeg"))
            ):
                try:
                    if imgAreSimilar(
                        os.path.join(root, file1), os.path.join(root, file2), p
                    ):
                        os.remove(os.path.join(root, file2))
                        deletedCount += 1
                        print(f"Deleted {file2} 🗑️")
                except:
                    pass
        os.remove(file1)

    deletedCount = 0
    prog = 0
    labels = {}
    for root, _, files in os.walk(labelsPath):
        for file in files:
            if file.lower().endswith((".json")):
                with open(os.path.join(root, file)) as j:
                    labels.update({file: json.load(j)})
    for root, _, files in os.walk(dataPath):
        threads = []
        maxThreads = 10
        with ThreadPoolExecutor(maxThreads) as executor:
            for file in files:
                if file.lower().endswith((".png", ".jpeg", ".jpg")):
                    label = labels[".".join(file.split(".")[:-1]) + ".json"]
                    for shape in label["shapes"]:
                        points = shape["points"]
                        if len(points) == 1:
                            points = [
                                points[0][0],
                                points[0][1],
                                points[0][2],
                                points[0][3],
                            ]
                        else:
                            points = [
                                points[0][0],
                                points[0][1],
                                points[1][0],
                                points[1][1],
                            ]
                        imgHeight = label["imageHeight"]
                        imgWidth = label["imageWidth"]
                        img = cv2.imread(os.path.join(root, file))
                        points = ObjectDetection.convertLabelPoints(
                            points, labelFormat, "pascal_voc", imgWidth, imgHeight
                        )
                        x1, y1, x2, y2 = points
                        img = img[y1:y2, x1:x2]
                        file = os.path.join(root, "temp-" + file)
                        cv2.imshow("a", img)
                        cv2.waitKey(0)
                        cv2.imwrite(file, img)
                        future = executor.submit(processFile, file)
                        threads.append(future)
                        prog += 1
                        print(
                            "progress: " + str(((prog / len(files)) * 100)) + "%\r",
                            end="",
                        )
    for t in threads:
        t.result()
    return deletedCount


def countClasses(dataPath: str, labelsPath: str = None):
    """return classes, images, labels"""
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))
    if labelsPath is None:
        labelsPath = dataPath
    if not os.path.exists(labelsPath):
        raise FileNotFoundError("Can't find folder 🤷\n labelsPath: " + str(labelsPath))
    labels = {}
    for root, _, files in os.walk(labelsPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith(("json")):
                labels[file] = {}
                labels[file]["root"] = root
                labels[file]["file"] = file
    images = {}
    for root, _, files in os.walk(dataPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith(("jpg", "png", "jpeg")):
                images[file] = {}
                images[file]["root"] = root
                images[file]["file"] = file
    classes = {}
    for _, value in labels.items():
        with open(os.path.join(value["root"], value["file"]), "r") as f:
            value = json.load(f)
        for sh in value["shapes"]:
            if sh["label"] not in list(classes.keys()):
                classes[sh["label"]] = 0
            classes[sh["label"]] += 1
    return classes, images, labels


def equlizeClassCount(dataPath: str, labelsPath: str = None):
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))
    if labelsPath is None:
        labelsPath = dataPath
    if not os.path.exists(labelsPath):
        raise FileNotFoundError("Can't find folder 🤷\n labelsPath: " + str(labelsPath))

    classes, images, labels = countClasses(dataPath, labelsPath)
    minClass = min([x for x in list(classes.values())])
    for targetClass, clcount in classes.items():
        print("Equlizing classes 📊\r", end="")
        if clcount > minClass:
            target = int(clcount - minClass)
            for labelFile in labels.values():
                label = None
                try:
                    with open(
                        os.path.join(labelFile["root"], labelFile["file"]), "r"
                    ) as f:
                        label = json.load(f)
                except:
                    continue
                for sh in label["shapes"]:
                    if sh["label"] == targetClass:
                        label["shapes"].remove(sh)
                        target -= 1
                        if target == 0:
                            break
                if len(label["shapes"]) == 0:
                    # remove label and image
                    os.remove(os.path.join(labelFile["root"], labelFile["file"]))
                    for im in images.values():
                        if ".".join(im["file"].split(".")[:-1]) == ".".join(
                            labelFile["file"].split(".")[:-1]
                        ):
                            os.remove(os.path.join(im["root"], im["file"]))
                            break
                else:
                    with open(
                        os.path.join(labelFile["root"], labelFile["file"]), "w"
                    ) as f:
                        f.truncate()
                        json.dump(label, f)
                if target == 0:
                    break

    print("Classes are equal ⚖️")


class Augmentation:
    class AugmentationRate:
        """

        high:
        alb.HorizontalFlip(),
        alb.RandomBrightnessContrast(),
        alb.Rotate(limit=90),
        alb.OneOf([
            alb.Blur(blur_limit=3),
            alb.GaussianBlur(blur_limit=3),
            alb.MotionBlur(blur_limit=3)
        ]),
        alb.OneOf([
            alb.GaussNoise(),
            alb.ISONoise(),
            alb.ImageCompression(quality_lower=90, quality_upper=100),
        ]),
        alb.OneOf([
            alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            alb.GridDistortion(),
            alb.OpticalDistortion(distort_limit=1, shift_limit=0.5),
        ]),
        alb.OneOf([
            alb.HueSaturationValue(),
            alb.RGBShift(),
            alb.ChannelShuffle(),
        ]),
        alb.OneOf([
            alb.RandomSnow(),
            alb.RandomRain(),
            alb.RandomFog(),
            alb.RandomSunFlare(),
        ])
        X50

        medium:
        alb.HorizontalFlip(p=0.5),
        alb.Rotate(limit=65),
        alb.RandomBrightnessContrast(),
        alb.Blur(blur_limit=3),
        X15

        low:
        alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        alb.Blur(blur_limit=3),
        X5
        """

        high = [
            [
                alb.HorizontalFlip(),
                alb.RandomBrightnessContrast(),
                alb.Rotate(limit=90),
                alb.OneOf(
                    [
                        alb.Blur(blur_limit=3),
                        alb.GaussianBlur(blur_limit=3),
                        alb.MotionBlur(blur_limit=3),
                    ]
                ),
                alb.OneOf(
                    [
                        alb.GaussNoise(),
                        alb.ISONoise(),
                        alb.ImageCompression(quality_lower=90, quality_upper=100),
                    ]
                ),
                alb.OneOf(
                    [
                        alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                        alb.GridDistortion(),
                        alb.OpticalDistortion(distort_limit=1, shift_limit=0.5),
                    ]
                ),
                alb.OneOf(
                    [
                        alb.HueSaturationValue(),
                        alb.RGBShift(),
                        alb.ChannelShuffle(),
                    ]
                ),
                alb.OneOf(
                    [
                        alb.RandomSnow(),
                        alb.RandomRain(),
                        alb.RandomFog(),
                        alb.RandomSunFlare(),
                    ]
                ),
            ],
            50,
        ]
        medium = [
            [
                alb.HorizontalFlip(p=0.5),
                alb.Rotate(limit=65),
                alb.RandomBrightnessContrast(),
                alb.Blur(blur_limit=3),
            ],
            15,
        ]
        low = [
            [
                alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                alb.Blur(blur_limit=3),
            ],
            5,
        ]

    @staticmethod
    def _augDataErr(dataPath: str, labelsPath: str, saveToPath: str, augRate, labelSaveFormat: str, currentLabelFormat: str):  # type: ignore
        import albumentations as alb

        if not os.path.exists(dataPath):
            raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))

        if not (os.path.exists(saveToPath)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n saveToPath: " + str(saveToPath)
            )
        if not (os.path.exists(labelsPath)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n saveToPath: " + str(labelsPath)
            )

        if not type(augRate[0]) == list:
            raise ValueError(
                "Your augRate value is not valid. Please use augRate.(augMax, augMid, augLow) or for custom use [[albumentations.[YourTransform]], 15(augRateCount)]"
            )

        if type(augRate[1]) != int:
            raise ValueError(
                "Your augRate value is not valid. Please use augRate.(augMax, augMid, augLow) or for custom use (albumentations.Compose([params...]), augRateCount:int)"
            )
        for p in [dataPath, labelsPath]:
            if not os.path.exists(p):
                raise FileNotFoundError("Can't find files 🤷\ndataPath: " + str(p))
        lTypes = ["yolo", "albumentations", "pascal_voc", "coco"]
        if labelSaveFormat not in lTypes or currentLabelFormat not in lTypes:
            raise ValueError(
                "Your labelSaveFormat/currentLabelFormat value is not valid. Please use one of this formats: albumentations, yolo, pascal_voc, coco"
            )
        gc.collect()

    @staticmethod
    def augData(
        dataPath: str,  # type: ignore
        labelsPath: str,  # type: ignore
        augRate: list,
        saveToPath: str,
        currentLabelFormat: str = "pascal_voc",
        labelSaveFormat: str = "pascal_voc",
        equlizeClasses: bool = True,
        similarity: float = 0.9,
    ):
        """
        Augment images in a directory using specified augmentation rates.

        :dataPath (str): The path to the directory containing the original images.
        :labelsPath (str): The path to the directory containing the image labes (for object detection datas).
        :augRate (list): A list containing an augmentation pipeline (albumentations.Compose) and the number of augmentations per image.
        :saveToPath (str): The path where augmented images will be saved.
        :currentLabelFormat (str): The current label format. Available formats: yolo, pascal_voc(default), coco
        :labelSaveFormat (str): The option for label save format. Available formats: yolo, pascal_voc(default), coco
        :equlizeClasses (bool): The option for equlize class count. Default is True.
        :similarity (float): Similarity threshold for deleting similar images. To disable enter under 0 or None. Default is 0.9.
        """
        BurobotOutput.clearAndMemoryTo()
        Augmentation._augDataErr(dataPath, labelsPath, saveToPath, augRate, labelSaveFormat, currentLabelFormat)  # type: ignore

        augRate[0] = alb.Compose(
            augRate[0],
            bbox_params=alb.BboxParams(
                format=labelSaveFormat, label_fields=["class_labels"]
            ),
        )
        BurobotOutput.printBurobot()

        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleting alone data 🥺💔")
        time.sleep(1)
        deleteAloneData(dataPath, labelsPath)
        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()
        labelsFiles, classes = ObjectDetection.loadAllLabels(
            labelsPath, dataPath, currentLabelFormat, True
        )
        imgSavePath = os.path.join(saveToPath, "images")
        if not os.path.exists(imgSavePath):
            os.makedirs(imgSavePath)
        labelSavePath = os.path.join(saveToPath, "labels")
        if not os.path.exists(labelSavePath):
            os.makedirs(labelSavePath)
        for root, _, files in os.walk(dataPath):
            for fi, file in enumerate(files):
                print(f"\r🔄 Data augmentating {((fi+1)/len(files))*100}% 😎", end="")
                fileCount = str(file)
                if fileCount.lower().endswith((".png", ".jpg", ".jpeg")):
                    imagePath = os.path.join(root, file)
                    image = Image.open(imagePath)
                    labels = {}
                    classLabels = []
                    copyLabels = {}
                    labels = labelsFiles[".".join(file.split(".")[:-1])]

                    imgWidth, imgHeight = image.size
                    newLabels = []
                    classLabels = []
                    copyLabels = copy.deepcopy(labels)
                    for l in copyLabels.copy():
                        newLabels.append(
                            ObjectDetection.convertLabelPoints(
                                l["bbox"],
                                currentLabelFormat,
                                labelSaveFormat,
                                imgWidth,
                                imgHeight,
                            )
                        )
                        classLabels.append(l["label"])
                    labels = newLabels
                    del newLabels
                    for i in range(augRate[1]):
                        try:
                            augmentedData = augRate[0](
                                image=np.array(image),
                                bboxes=labels,
                                class_labels=classLabels,
                            )

                            augmentedImage = Image.fromarray(augmentedData["image"])
                            for cli in range(len(copyLabels)):
                                copyLabels[cli]["imageHeight"] = augmentedImage.height
                                copyLabels[cli]["imageWidth"] = augmentedImage.width
                                copyLabels[cli]["labelFormat"] = labelSaveFormat
                                if labelSaveFormat in ["yolo", "coco"]:
                                    classNumbers = {}
                                    for clsi, clsl in enumerate(classes):
                                        classNumbers[clsl] = clsi
                                    copyLabels[cli]["classNumbers"] = classNumbers
                            uu = str(uuid.uuid1())
                            augmentedImage.save(
                                os.path.join(imgSavePath, "aug-" + str(i) + "-" + file)
                            )

                            augmentedLabels = augmentedData["bboxes"]
                            for li, l in enumerate(augmentedLabels):
                                if labelSaveFormat == "pascal_voc":
                                    l = [int(x) for x in list(l)]
                                copyLabels[li]["bbox"] = list(l)
                            ObjectDetection.saveLabel(
                                os.path.join(
                                    labelSavePath,
                                    "aug-"
                                    + str(i)
                                    + "-"
                                    + file.replace(".jpg", "")
                                    .replace(".jpg", "")
                                    .replace(".jpeg", "")
                                    .replace(".JPG", ""),
                                ),
                                copyLabels,
                            )
                        except:
                            continue

        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleting duplicate images 🎴🖼️📜 🖼️🪚")
        deletedCount = deleteDuplicateImages(saveToPath)
        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleted " + str(deletedCount) + " duplicate image(s) 🪚😋")
        time.sleep(3)

        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleting alone data 🥺💔")
        time.sleep(1)
        deleteAloneData(imgSavePath, labelSavePath)
        BurobotOutput.clearAndMemoryTo()

        if similarity is not None and similarity > 0:
            try:
                BurobotOutput.clearAndMemoryTo()
                BurobotOutput.printBurobot()
                print(f"🔄 Deleting {similarity*100}% similar or more images 🔍🧐")
                deleteSimilarDetections(
                    imgSavePath, labelSavePath, labelSaveFormat, similarity
                )
            except:
                pass
        if equlizeClasses:
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            equlizeClassCount(imgSavePath, labelSavePath)

        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()
        print("✅ Data augmentation completed successfully! 😋🎉")
