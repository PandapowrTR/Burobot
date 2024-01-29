# BUROBOT
import os, gc, sys, threading, time, cv2, shutil, random, json
import albumentations as alb
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def splitData(
    sourcePath: str,
    saveToPath: str,
    splitRatio: tuple = (0.8, 0.1),
    extraValues: dict = {},
):
    """
    This function splits the image data at the specified path. The file structure in the specified path should be:\n\nMYDATAS(folder)/\n\t\tCLASS_A(folder)/\n\t\t\n\t\t\tSOMEIMAGE.(JPG/PNG/JPEG)(image folder) ...
    :sourcePath (str): Folder containing data
    :saveToPath (str): The path to save the split data.
    :splitRatio (tuple): The percentages by which the data will be divided. (train, test) val: 1- (train+test)
    \nReturns:
        trainDir, testDir, valDir
    """
    try:
        sourcePath = extraValues["sourcePath"]
        saveToPath = extraValues["saveToPath"]
        splitRatio = extraValues["splitRatio"]
    except:
        pass
    if not os.path.exists(saveToPath) or not os.path.exists(sourcePath):
        raise Exception(
            "Can't find path 🤷\ndataPath:"
            + str(saveToPath)
            + "\nsource_path:"
            + str(sourcePath)
        )
    for s in splitRatio:
        if s <= 0 or s > 1:
            raise ValueError(
                "Split ratios must be between 0 (exclusive) and 1 (inclusive) 🔢"
            )
    trainDir = os.path.join(saveToPath, "train")
    testDir = os.path.join(saveToPath, "test")
    valDir = os.path.join(saveToPath, "val")

    def copy_class(class_name, progress_callback):
        class_path = os.path.join(sourcePath, class_name)
        files = os.listdir(class_path)
        random.shuffle(files)
        train_ratio, test_ratio = splitRatio
        train_idx = int(len(files) * train_ratio)
        test_idx = int(len(files) * (train_ratio + test_ratio))
        for i, file in enumerate(files):
            src_path = os.path.join(class_path, file)
            if i < train_idx:
                dest_path = os.path.join(trainDir, class_name)
            elif i < test_idx:
                dest_path = os.path.join(testDir, class_name)
            else:
                dest_path = os.path.join(valDir, class_name)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            dest_file_path = os.path.join(dest_path, file)
            shutil.copy(src_path, dest_file_path)
            progress_callback(class_name, ((i / len(files)) * 100))

    lock = threading.Lock()
    progress_data = {}

    def progress_callback(class_name, progress):
        with lock:
            progress_data[class_name] = progress
            all_progress = sum(progress_data.values()) / len(progress_data)
            print(f"\rSplitting data %{all_progress:2f} 🔪📚", end="")

    os.makedirs(trainDir, exist_ok=True)
    os.makedirs(testDir, exist_ok=True)
    os.makedirs(valDir, exist_ok=True)

    threads = []
    for class_name in os.listdir(sourcePath):
        thread = threading.Thread(
            target=copy_class,
            args=(
                class_name,
                progress_callback,
            ),
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    return trainDir, testDir, valDir


# Function to find duplicate images
def findDuplicateImages(path):
    """
    Find duplicate images within a given directory.

    Args:
        path (str): The path to the directory containing the images.

    Returns:
        list: A list of paths to duplicate image files.
    """
    import hashlib

    allFiles = []
    for root, dirs, files in os.walk(path):
        for file in files:
            allFiles.append(os.path.join(root, file))

    hashDict = {}
    duplicateFiles = []
    for filePath in allFiles:
        with open(filePath, "rb") as file:
            content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            if file_hash in hashDict:
                duplicateFiles.append(filePath)
            else:
                hashDict[file_hash] = filePath

    return duplicateFiles


# Function to delete duplicate images
def deleteDuplicateImages(path):
    """
    Delete duplicate images within a given directory.

    Args:
        path (str): The path to the directory containing the images.

    Returns:
        int: The number of deleted duplicate image files.
    """
    BurobotOutput.printBurobot()
    duplicateFiles = findDuplicateImages(path)
    deletedCount = 0
    for filePath in duplicateFiles:
        os.remove(filePath)
        deletedCount += 1
    return deletedCount


# Function to check if two images are similar
def imgAreSimilar(img1, img2, similarity: float = 0.9, returnSimilarity: bool = False):
    """
    Check if two images are similar based on their content.

    Args:
        img1 (str): Path to the first image file.
        img2 (str): Path to the second image file.
        similarity (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        bool: True if the images are similar, False otherwise.
    """
    from difflib import SequenceMatcher

    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        content1 = f1.read()
        content2 = f2.read()
        similarityRatio = SequenceMatcher(None, content1, content2).ratio()
        if not returnSimilarity:
            return similarityRatio >= similarity
        return similarityRatio >= similarity, similarityRatio


# Function to delete similar images
def deleteSimilarImgs(path, similarity: float = 0.9):
    """
    Delete images that are similar to each other within a given directory.

    Args:
        path (str): The path to the directory containing the images.
        similarity (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        int: The number of deleted similar image files.
    """
    deletedCount = 0

    BurobotOutput.printBurobot()

    def deleteSimilarImagesInFolder(folder, progressCallback):
        nonlocal deletedCount
        images = os.listdir(os.path.join(path, folder))
        i = 0
        for mainImgIndex, mainImg in enumerate(images):
            for checkImg in images:
                try:
                    if os.path.join(path, folder, mainImg) != os.path.join(path, folder, checkImg):
                        if imgAreSimilar(
                            os.path.join(path, folder, mainImg),
                            os.path.join(path, folder, checkImg),
                            similarity,
                        ):
                            os.remove(os.path.join(path, folder, checkImg))
                            deletedCount += 1
                except:
                    pass
            progress = (mainImgIndex / len(images)) * 100
            progressCallback(folder, progress)

    progressLock = threading.Lock()
    progressData = {}

    def updateProgress(folder, progress):
        with progressLock:
            progressData[folder] = progress
            totalProgress = sum(progressData.values()) / len(progressData)
            print(
                f"🔄 Deleteing %{similarity*100} similar or more images %{totalProgress} 🔍🧐\r",
                end="",
            )

    threads = []
    for folder in os.listdir(path):
        thread = threading.Thread(
            target=deleteSimilarImagesInFolder,
            args=(
                folder,
                updateProgress,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return deletedCount


# Function to equalize image counts in folders
def equalizeImageCount(path):
    """
    Equalize the image count in folders within a given directory.

    Args:
        path (str): The path to the directory containing the folders of images.
    """
    BurobotOutput.printBurobot()

    # Function to equalize image count in a folder
    def equalizeImage(folderPath, imageCount, progressCallback):
        if imageCount > minImageCount:
            excessCount = imageCount - minImageCount
            filesToDelete = [
                file
                for file in os.listdir(folderPath)
                if os.path.splitext(file)[1].lower() in imageExtensions
            ][:excessCount]
            i = 0
            totalImages = len(filesToDelete)
            for file in filesToDelete:
                file_path = os.path.join(folderPath, file)
                os.remove(file_path)
                i += 1
                progress = int((i / totalImages) * 100)
                progressCallback(folderPath, progress)

    folderImageCounts = {}

    for folderPath, _, files in os.walk(path):
        if folderPath == path:
            continue
        image_count = 0
        imageExtensions = [".jpg", ".jpeg", ".png"]

        for file in files:
            fileExtension = os.path.splitext(file)[1].lower()
            if fileExtension in imageExtensions:
                image_count += 1

        folderImageCounts[folderPath] = image_count
    progressLock = threading.Lock()
    progressData = {}

    def update_progress(folder, progress):
        with progressLock:
            progressData[folder] = progress
            totalProgress = sum(progressData.values()) / len(progressData)
            print(f"\r🔄 Equlizing image counts %{int(totalProgress)} 🎴🔢🧐", end="")

    minImageCount = min(folderImageCounts.values())
    threads = []
    for folder_path, image_count in folderImageCounts.items():
        thread = threading.Thread(
            target=equalizeImage,
            args=(
                folder_path,
                image_count,
                update_progress,
            ),
        )
        threads.append(thread)
        thread.start()
    for t in threads:
        t.join()


# Function to convert images to RGB format
def covertImageRgb(path: str):
    """
    Convert images in a directory to the RGB color space.

    Args:
        path (str): The path to the directory containing the images.
    """
    if not os.path.exists(path):
        FileNotFoundError("Can't find path 🤷\npath:" + str(path))
    for root, _, files in os.walk(path):
        for f in files:
            try:
                img = cv2.imread(os.path.join(root, f))
                os.remove(os.path.join(root, f))
                cv2.imwrite(os.path.join(root, f), img)
            except:
                pass


def convertImageTo(path: str, color_space):
    """
    Convert images in a directory to the Any color space.

    Args:
        path (str): The path to the directory containing the images.
        color_space (cv2.COLOR_[YOURTYPE]2[WANTEDTYPE]): Color space type.
    """
    if not os.path.exists(path):
        FileNotFoundError("Can't find path 🤷\npath:" + str(path))
    for root, _, files in os.walk(path):
        for f in files:
            try:
                img = cv2.imread(os.path.join(root, f))
                img = cv2.cvtColor(img, color_space)
                os.remove(os.path.join(root, f))
                cv2.imwrite(os.path.join(root, f), img)
            except:
                pass


# Function to resize all images
def resizeAllImages(path: str, size: tuple):
    """
    Resize all images in a directory to a specified size.

    Args:
        path (str): The path to the directory containing the images.
        size (tuple): The target size in the format (width, height).
    """
    print("Resizing all images 🖼️")
    if not os.path.exists(path):
        FileNotFoundError("Can't find path 🤷\npath:" + str(path))
    if len(size) != 2:
        raise ValueError("Size must be a tuple of width and height")
    for root, _, files in os.walk(path):
        for f in files:
            try:
                img = cv2.imread(os.path.join(root, f))
                img = cv2.resize(img, size)
                os.remove(os.path.join(root, f))
                cv2.imwrite(os.path.join(root, f), img)
            except:
                pass


def splitImagesToTxt(
    dataPath: str,
    saveToPath: str,
    splitRatio: tuple = (0.8, 0.1),
    addToStart: str = "",
):
    """NOTE splitRatio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(dataPath):
        raise FileNotFoundError("Can't find path 🤷\ndataPath: " + str(dataPath))
    if not os.path.exists(saveToPath):
        raise FileNotFoundError("Can't find path 🤷\nsaveToPath: " + str(saveToPath))

    if len(splitRatio) != 2 or any(splitRatio[i] < 0 for i in range(len(splitRatio))):
        raise ValueError("splitRatio value is invalid. Please check the data 🔢")
    print("Splitting images to txt files 🔪📰")
    images = []

    for root, _, files in os.walk(dataPath):
        for file in files:
            if str(file).endswith((".png", ".jpg", ".jpeg")):  # type: ignore
                images.append(str(addToStart) + file)

    trainImages = []
    testImages = []
    valImages = []

    trainImages = images[: int(len(images) * splitRatio[0])]
    testImages = images[
        int(len(images) * splitRatio[0]) : int(
            len(images) * (splitRatio[0] + splitRatio[1])
        )
    ]
    valImages = images[int(len(images) * (splitRatio[0] + splitRatio[1])) :]
    train = ""
    for t in trainImages:
        train += t + "\n"
    with open(os.path.join(saveToPath, "train.txt"), "w") as trainFile:
        trainFile.write(train.replace("\\", "/"))

    test = ""
    for t in testImages:
        test += t + "\n"
    with open(os.path.join(saveToPath, "test.txt"), "w") as test_file:
        test_file.write(test.replace("\\", "/"))

    val = ""
    for v in valImages:
        val += v + "\n"
    with open(os.path.join(saveToPath, "val.txt"), "w") as val_file:
        val_file.write(val.replace("\\", "/"))


class ImageAugmentation:
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
    def _augDataErr(dataPath: str, saveToPath: str, augRate):  # type: ignore
        import albumentations as alb

        if not os.path.exists(dataPath):
            raise FileNotFoundError("Can't find folder 🤷\n dataPath: " + str(dataPath))

        if not (os.path.exists(saveToPath)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n saveToPath: " + str(saveToPath)
            )
        if not isinstance(augRate[0], alb.Compose):
            raise ValueError(
                "Your augRate value is not valid. Please use AugmentationRate.(augMax, augMid, augLow) or for custom use (albumentations.Compose([params...]), 15(augRateCount))"
            )

        if type(augRate[1]) != int:
            raise ValueError(
                "Your augRate value is not valid. Please use [AugmentationRate.(augMax, augMid, augLow)[0], AugmentationRate.(augMax, augMid, augLow)[1]] or for custom use (albumentations.Compose([params...]), augRateCount:int)"
            )
        folderTrue = os.path.isdir(os.path.join(dataPath, os.listdir(dataPath)[0]))
        if folderTrue:
            for folder in os.listdir(dataPath):
                for f in os.listdir(os.path.join(dataPath, folder)):
                    if not os.path.isfile(os.path.join(dataPath, folder, f)):
                        folderTrue = False
                        break

        if not folderTrue:  # type: ignore
            file_tree = """
            data/
            |   A/
            |   |   file1.dataType
            |   |   ...
            |   B/
            |   |   file1.dataType
            |   |   ...
            |   C/
            |   |   file1.dataType 
            |   |   ...
            |   ...
            """
            raise FileNotFoundError(
                "Can't find class folders 🤷\ndataPath: "
                + str(dataPath)
                + "\nfile tree must be like this:\n"
                + file_tree
            )

        gc.collect()
    @staticmethod
    def augmentateData(
        dataPath: str,  # type: ignore
        augRate: list,
        saveToPath:str,
        equlizeImgCount: bool = True,
        similarity: float = 0.9,
    ):
        """
        Augment images in a directory using specified augmentation rates.

        Args:
            dataPath (str): The path to the directory containing the original images.
            augRate (list): A tuple containing an augmentation pipeline (albumentations.Compose) and the number of augmentations per image.
            saveToPath (str): The path where augmented images will be saved.
            equlizeImgCount (bool, optional): Whether to equalize image counts after augmentation. Default is True.
            similarity (float, optional): Similarity threshold for deleting similar images. Default is 0.9.
        """

        def augDataFolder(classFolder, progressCallback):
            i = 0
            totalImages = len(os.listdir(os.path.join(dataPath, classFolder)))
            print(os.path.join(dataPath, classFolder))
            for f in os.listdir(os.path.join(dataPath, classFolder)):
                imagePath = os.path.join(dataPath, classFolder, f)
                image = Image.open(imagePath)

                for _ in range(augRate[1] + 1):
                    augmentedData = augRate[0](image=np.array(image))

                    augmentedImage = Image.fromarray(augmentedData["image"])

                    savePath = os.path.join(saveToPath, classFolder)
                    os.makedirs(savePath, exist_ok=True)
                    augmentedImage.save(
                        os.path.join(savePath, "aug-" + str(i) + "-" + f)
                    )

                    i += 1
                progressPercent = (i / (totalImages * (augRate[1] + 1))) * 100
                progressCallback(classFolder, int(progressPercent))

        BurobotOutput.clearAndMemoryTo()
        ImageAugmentation._augDataErr(dataPath, saveToPath, augRate)  # type: ignore

        BurobotOutput.printBurobot()

        progressLock = threading.Lock()
        progressData = {}

        def updateProgress(folder, progress):
            with progressLock:
                progressData[folder] = progress
                totalProgress = sum(progressData.values()) / len(progressData)
                print(f"\r🔄 Data augmenting %{int(totalProgress)} 😎", end="")

        threads = []
        for folder in os.listdir(dataPath):
            thread = threading.Thread(
                target=augDataFolder,
                args=(
                    folder,
                    updateProgress,
                ),
            )
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleting duplicate images 🎴🖼️📜 🖼️🪚")
        deletedCount = deleteDuplicateImages(saveToPath)
        BurobotOutput.clearAndMemoryTo()

        BurobotOutput.printBurobot()
        print("🔄 Deleted " + str(deletedCount) + " duplicate image(s) 🪚😋")
        time.sleep(3)

        if similarity is not None and similarity > 0:
            if equlizeImgCount:
                BurobotOutput.clearAndMemoryTo()
                BurobotOutput.printBurobot()
                equalizeImageCount(saveToPath)
            BurobotOutput.printBurobot()
            BurobotOutput.clearAndMemoryTo()
            deleteSimilarImgs(saveToPath, similarity)

        if equlizeImgCount:
            BurobotOutput.clearAndMemoryTo()
            BurobotOutput.printBurobot()
            equalizeImageCount(saveToPath)

        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()
        print("✅ Data augmentation completed successfully! 😋🎉")


# BUROBOT
