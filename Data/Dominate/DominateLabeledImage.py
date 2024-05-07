import os, shutil, os, gc, time, sys, uuid, copy, cv2, warnings
import concurrent.futures
cv2.setLogLevel(0)

warnings.warn("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from PIL import Image
import numpy as np
import albumentations as alb
from Burobot.tools import BurobotOutput
from Burobot.Data.Dominate.DominateImage import imgAreSimilar, deleteDuplicateImages
from Burobot.Data.Dominate.DominateLabel import ObjectDetection


def splitDataToTxt(
    dataPath: str,
    labelsPath: str,
    saveToPath: str,
    splitRatio: tuple = (0.8, 0.1),
    addToStart: str = "",
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
            if str(file).endswith((".xml", ".json", ".txt")):
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
            if file.endswith((".xml", ".json", ".txt")):
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
            if fileCount.lower().endswith((".json", ".txt", ".xml")):
                labelFiles[0].append(os.path.join(root, file))
                labelFiles[1].append(".".join(file.split(".")[:-1]))

    imageFiles = [[], []]  # 0: file with root, 1:file
    for root, _, files in os.walk(dataPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith((".jpg", ".png", ".jpeg")):
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
    dataPath, labelsPath, labelFormat="pascal_voc", maxSimilarity: float = 0.9
):
    """
    Delete detections that are similar to each other within a given directory.

    Args:
        dataPath (str): The path to the directory containing the images.
        labelsPath (str): The path to the directiory containing the labels.
        labelFormat (str): The format of label values(yolo, pascal_voc(default), coco)
        maxSimilarity (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        int: The number of deleted similar detection files.
    """

    def processFile(args):
        (
            file,
            root,
            labels,
            tempFolder,
            maxSimilarity,
            deletedFiles,
            files,
        ) = args

        if (
            file.lower().endswith((".png", ".jpeg", ".jpg"))
            and file not in deletedFiles
        ):
            label = labels[".".join(file.split(".")[:-1])]
            
            if not os.path.exists(os.path.join(root, file)):
                return
            img = cv2.imread(os.path.join(root, file))
            if img is None:
                return
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for l in label:
                    future = executor.submit(
                        processLabel,
                        l,
                        img,
                        args,
                    )
                    futures.append(future)

                concurrent.futures.wait(futures)

    def processLabel(l, img, args):
        (
            file,
            root,
            labels,
            tempFolder,
            maxSimilarity,
            deletedFiles,
            files,
        ) = args
        points = l["bbox"]
        points = ObjectDetection.convertLabelPoints(
            points, labelFormat, "pascal_voc", l["imageWidth"], l["imageHeight"]
        )
        xmin, ymin, xmax, ymax = [int(p) for p in points]
        img = img[ymin:ymax, xmin:xmax]
        img = cv2.resize(img, (100, 100))
        uu = str(uuid.uuid1())
        cutDetection = os.path.join(tempFolder, "temp-" + uu + file)
        cv2.imwrite(cutDetection, img)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            cFileFutures = []
            for checkFile in files.copy():
                future = executor.submit(
                    processCheckFile, checkFile, cutDetection, args
                )
                cFileFutures.append(future)
            concurrent.futures.wait(cFileFutures)

        os.remove(cutDetection)

    def processCheckFile(checkFile, cutDetection, args):
        (
            file,
            root,
            labels,
            tempFolder,
            maxSimilarity,
            deletedFiles,
            files,
        ) = args
        if (
            checkFile.lower().endswith((".png", ".jpeg", ".jpg"))
            and checkFile != file
            and checkFile not in deletedFiles
        ):
            checkLabels = labels[".".join(checkFile.split(".")[:-1])]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                cFutures = []
                for cl in checkLabels:
                    future = executor.submit(
                        processCheckLabel, cl, checkFile, cutDetection, args
                    )
                    cFutures.append(future)
            concurrent.futures.wait(cFutures)

    def processCheckLabel(cl, checkFile, cutDetection, args):
        (
            file,
            root,
            labels,
            tempFolder,
            maxSimilarity,
            deletedFiles,
            files,
        ) = args
        checkImgPoints = cl["bbox"]
        checkImgPoints = ObjectDetection.convertLabelPoints(
            checkImgPoints,
            labelFormat,
            "pascal_voc",
            cl["imageWidth"],
            cl["imageHeight"],
        )
        cxmin, cymin, cxmax, cymax = [int(p) for p in checkImgPoints]
        if not os.path.exists(os.path.join(root, checkFile)):
            return
        checkImg = cv2.imread(os.path.join(root, checkFile))
        checkImg = checkImg[cymin:cymax, cxmin:cxmax]
        checkImg = cv2.resize(checkImg, (100, 100))
        checkCutDetection = os.path.join(tempFolder, "temp-" + checkFile)
        cv2.imwrite(checkCutDetection, checkImg)
        if imgAreSimilar(cutDetection, checkCutDetection, maxSimilarity):
            os.remove(os.path.join(root, checkFile))
            deletedFiles.add(checkFile)
            os.remove(checkCutDetection)
            return
        os.remove(checkCutDetection)

    sizes = {}
    for root, _, files in os.walk(dataPath):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img = Image.open(os.path.join(root, file))
                sizes[".".join(file.split(".")[:-1])] = {
                    "imageHeight": img.height,
                    "imageWidth": img.width,
                }
    labels = {}
    for root, _, files in os.walk(labelsPath):
        for file in files:
            if file.lower().endswith((".json", ".xml", ".txt")):
                rawFileName = ".".join(file.split(".")[:-1])
                labels.update(
                    {
                        rawFileName: ObjectDetection.loadLabel(
                            os.path.join(root, file),
                            labelFormat,
                            sizes[rawFileName]["imageWidth"],
                            sizes[rawFileName]["imageHeight"],
                        )
                    }
                )
    tempFolder = os.path.join(dataPath, "tempFiles")
    try:
        os.mkdir(tempFolder)
    except FileExistsError:
        pass
    print("Checking similar detections")
    deletedFiles = set()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            processFile,
            [
                (
                    file,
                    root,
                    labels,
                    tempFolder,
                    maxSimilarity,
                    deletedFiles,
                    files.copy(),
                )
                for root, _, files in os.walk(dataPath)
                for file in files
            ],
        )

    for f in os.listdir(tempFolder):
        os.remove(os.path.join(tempFolder, f))
    os.rmdir(tempFolder)

    deleteAloneData(dataPath, labelsPath)
    return len(deletedFiles)


def countClasses(dataPath: str, labelFormat, labelsPath: str = None):
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
            if fileCount.lower().endswith((".json", ".txt", ".xml")):
                labels[file] = {}
                labels[file]["root"] = root
                labels[file]["file"] = file
    images = {}
    imageWidth = 0
    imageHeight = 0
    for root, _, files in os.walk(dataPath):
        for file in files:
            fileCount = str(file)
            if fileCount.lower().endswith((".jpg", ".png", ".jpeg")):
                if imageWidth == 0:
                    img = Image.open(os.path.join(root, file))
                    imageWidth = img.width
                    imageHeight = img.height
                images[file] = {}
                images[file]["root"] = root
                images[file]["file"] = file
    classes = {}
    for _, value in labels.items():
        label = ObjectDetection.loadLabel(
            os.path.join(value["root"], value["file"]),
            labelFormat,
            imageWidth,
            imageHeight,
        )
        for l in label:
            if l["label"] not in list(classes.keys()):
                classes[l["label"]] = 0
            classes[l["label"]] += 1
    return classes, images, labels


def equlizeClassCount(dataPath: str, labelFormat, labelsPath: str = None):
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))
    if labelsPath is None:
        labelsPath = dataPath
    if not os.path.exists(labelsPath):
        raise FileNotFoundError("Can't find folder 🤷\n labelsPath: " + str(labelsPath))

    classes, images, labels = countClasses(dataPath, labelFormat, labelsPath)
    img = images[list(images.keys())[0]]
    img = Image.open(os.path.join(img["root"], img["file"]))
    imageWidth = img.width
    imageHeight = img.height
    minClass = min([x for x in list(classes.values())])
    for targetClass, clcount in classes.items():
        print("Equlizing classes 📊\r", end="")
        if clcount > minClass:
            target = int(clcount - minClass)
            for labelKey, labelFile in labels.copy().items():
                label = ObjectDetection.loadLabel(
                    os.path.join(labelFile["root"], labelFile["file"]),
                    labelFormat,
                    imageWidth,
                    imageHeight,
                )
                for l in label.copy():
                    if l["label"] == targetClass:
                        label.remove(l)
                    if target == 0:
                        break
                if len(label) == 0:
                    os.remove(os.path.join(labelFile["root"], labelFile["file"]))
                    for im in images.values():
                        if ".".join(im["file"].split(".")[:-1]) == ".".join(
                            labelFile["file"].split(".")[:-1]
                        ):
                            os.remove(os.path.join(im["root"], im["file"]))
                            break
                    del labels[labelKey]
                else:
                    ObjectDetection.saveLabel(
                        os.path.join(
                            labelFile["root"],
                            ".".join(labelFile["file"].split(".")[:-1]),
                        ),
                        label,
                    )
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
        maxSimilarity: float = 0.9,
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
        :maxSimilarity (float): Similarity threshold for deleting similar images. To disable enter under 0 or None. Default is 0.9.
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

        if equlizeClasses:
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            equlizeClassCount(imgSavePath, labelSavePath)

        if maxSimilarity is not None and maxSimilarity > 0:
            try:
                BurobotOutput.clearAndMemoryTo()
                BurobotOutput.printBurobot()
                print(
                    f"🔄 Deleting {maxSimilarity*100}% similar or more detections 🔍🧐"
                )
                deleteSimilarDetections(
                    imgSavePath, labelSavePath, labelSaveFormat, maxSimilarity
                )
            except:
                pass

        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()
        print("✅ Data augmentation completed successfully! 😋🎉")
