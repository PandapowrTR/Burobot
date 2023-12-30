# BUROBOT
import os, gc, sys, threading, time, cv2, shutil, random, json
import albumentations as alb
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def split_data(source_path: str, dest_path: str, split_ratio: tuple = (0.8, 0.1)):
    if not os.path.exists(dest_path) or not os.path.exists(source_path):
        raise Exception(
            "Can't find path 🤷\ndata_path:"
            + str(dest_path)
            + "\nsource_path:"
            + str(source_path)
        )
    for s in split_ratio:
        if s <= 0 or s > 1:
            raise ValueError(
                "Split ratios must be between 0 (exclusive) and 1 (inclusive) 🔢"
            )
    train_dir = os.path.join(dest_path, "train")
    test_dir = os.path.join(dest_path, "test")
    val_dir = os.path.join(dest_path, "val")

    def copy_class(class_name, progress_callback):
        class_path = os.path.join(source_path, class_name)
        files = os.listdir(class_path)
        random.shuffle(files)
        train_ratio, test_ratio = split_ratio
        train_idx = int(len(files) * train_ratio)
        test_idx = int(len(files) * (train_ratio + test_ratio))
        for i, file in enumerate(files):
            src_path = os.path.join(class_path, file)
            if i < train_idx:
                dest_path = os.path.join(train_dir, class_name)
            elif i < test_idx:
                dest_path = os.path.join(test_dir, class_name)
            else:
                dest_path = os.path.join(val_dir, class_name)
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

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    threads = []
    for class_name in os.listdir(source_path):
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

    return train_dir, test_dir, val_dir


# Function to find duplicate images
def find_duplicate_images(path):
    """
    Find duplicate images within a given directory.

    Args:
        path (str): The path to the directory containing the images.

    Returns:
        list: A list of paths to duplicate image files.
    """
    import hashlib

    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))

    hash_dict = {}
    duplicate_files = []
    for file_path in all_files:
        with open(file_path, "rb") as file:
            content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            if file_hash in hash_dict:
                duplicate_files.append(file_path)
            else:
                hash_dict[file_hash] = file_path

    return duplicate_files


# Function to delete duplicate images
def delete_duplicate_images(path):
    """
    Delete duplicate images within a given directory.

    Args:
        path (str): The path to the directory containing the images.

    Returns:
        int: The number of deleted duplicate image files.
    """
    BurobotOutput.print_burobot()
    duplicate_files = find_duplicate_images(path)
    deleted_c = 0
    for file_path in duplicate_files:
        os.remove(file_path)
        deleted_c += 1
    return deleted_c


# Function to check if two images are similar
def img_are_similar(img1, img2, p: float = 0.9, return_p:bool=False):
    """
    Check if two images are similar based on their content.

    Args:
        img1 (str): Path to the first image file.
        img2 (str): Path to the second image file.
        p (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        bool: True if the images are similar, False otherwise.
    """
    from difflib import SequenceMatcher

    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        content1 = f1.read()
        content2 = f2.read()
        similarity_ratio = SequenceMatcher(None, content1, content2).ratio()
        if not return_p:
            return similarity_ratio >= p
        return similarity_ratio >= p, similarity_ratio


# Function to delete similar images
def delete_similar_imgs(path, p: float = 0.9):
    """
    Delete images that are similar to each other within a given directory.

    Args:
        path (str): The path to the directory containing the images.
        p (float, optional): The similarity threshold to consider images as similar. Default is 0.9.

    Returns:
        int: The number of deleted similar image files.
    """
    deleted_c = 0

    BurobotOutput.print_burobot()

    def delete_similar_images_in_folder(folder, progress_callback):
        nonlocal deleted_c
        images = os.listdir(os.path.join(path, folder))
        i = 0
        for main_img_i, main_img in enumerate(images):
            for check_img in images[1:main_img_i]:
                try:
                    if img_are_similar(
                        os.path.join(path, folder, main_img),
                        os.path.join(path, folder, check_img),
                        p,
                    ):
                        os.remove(os.path.join(path, folder, check_img))
                        deleted_c += 1
                except:
                    pass
            progress = (main_img_i / len(images)) * 100
            progress_callback(folder, progress)

    progress_lock = threading.Lock()
    progress_data = {}

    def update_progress(folder, progress):
        with progress_lock:
            progress_data[folder] = progress
            total_progress = sum(progress_data.values()) / len(progress_data)
            print(
                f"🔄 Deleteing %{p*100} similar or more images %{total_progress} 🔍🧐\r",
                end="",
            )

    threads = []
    for folder in os.listdir(path):
        thread = threading.Thread(
            target=delete_similar_images_in_folder,
            args=(
                folder,
                update_progress,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return deleted_c


# Function to equalize image counts in folders
def equalize_image_count(path):
    """
    Equalize the image count in folders within a given directory.

    Args:
        path (str): The path to the directory containing the folders of images.
    """
    BurobotOutput.print_burobot()

    # Function to equalize image count in a folder
    def equalize_image(folder_path, image_count, progress_callback):
        if image_count > min_image_count:
            excess_count = image_count - min_image_count
            files_to_delete = [
                file
                for file in os.listdir(folder_path)
                if os.path.splitext(file)[1].lower() in image_extensions
            ][:excess_count]
            i = 0
            total_images = len(files_to_delete)
            for file in files_to_delete:
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                i += 1
                progress = int((i / total_images) * 100)
                progress_callback(folder_path, progress)

    folder_image_counts = {}

    for folder_path, _, files in os.walk(path):
        if folder_path == path:
            continue
        image_count = 0
        image_extensions = [".jpg", ".jpeg", ".png"]

        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in image_extensions:
                image_count += 1

        folder_image_counts[folder_path] = image_count
    progress_lock = threading.Lock()
    progress_data = {}

    def update_progress(folder, progress):
        with progress_lock:
            progress_data[folder] = progress
            total_progress = sum(progress_data.values()) / len(progress_data)
            print(f"\r🔄 Equlizing image counts %{int(total_progress)} 🎴🔢🧐", end="")

    min_image_count = min(folder_image_counts.values())
    threads = []
    for folder_path, image_count in folder_image_counts.items():
        thread = threading.Thread(
            target=equalize_image,
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
def covert_image_rgb(path: str):
    """
    Convert images in a directory to the RGB color space.

    Args:
        path (str): The path to the directory containing the images.
    """
    if not os.path.exists(path):
        FileNotFoundError("Can't find path 🤷\npath:" + str(path))
    old_path = os.getcwd()
    os.chdir(path)
    for root, _, files in os.walk(path):
        os.chdir(root)
        for f in files:
            try:
                img = cv2.imread(f)
                os.remove(f)
                cv2.imwrite(f, img)
            except:
                pass
    os.chdir(old_path)

def convert_image_to(path:str, color_space):
    """
    Convert images in a directory to the Any color space.

    Args:
        path (str): The path to the directory containing the images.
        color_space (cv2.COLOR_[YOURTYPE]2[WANTEDTYPE]): Color space type.
    """
    if not os.path.exists(path):
        FileNotFoundError("Can't find path 🤷\npath:" + str(path))
    old_path = os.getcwd()
    os.chdir(path)
    for root, _, files in os.walk(path):
        os.chdir(root)
        for f in files:
            try:
                img = cv2.imread(f)
                img = cv2.cvtColor(img, color_space)
                os.remove(f)
                cv2.imwrite(f, img)
            except:
                pass
    os.chdir(old_path)
# Function to resize all images
def resize_all_images(path: str, size: tuple):
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
    old_path = os.getcwd()
    os.chdir(path)
    for root, _, files in os.walk(path):
        os.chdir(root)
        for f in files:
            try:
                img = cv2.imread(f)
                img = cv2.resize(img, size)
                os.remove(f)
                cv2.imwrite(f, img)
            except:
                pass
    os.chdir(old_path)



def split_images_to_txt(
    data_path: str, save_to_path: str, split_ratio: tuple = (0.8, 0.1), add_to_start:str=""
):
    """NOTE split_ratio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(data_path):
        raise FileNotFoundError("Can't find path 🤷\ndata_path:" + str(data_path))
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path:" + str(save_to_path))
    
    if len(split_ratio) != 2 or any(
        split_ratio[i] < 0 for i in range(len(split_ratio))
    ):
        raise ValueError("split_raito value is invalid. Please check the data 🔢")
    print("Splitting images to txt files 🔪📰")
    images = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if str(file).endswith((".png", ".jpg", ".jpeg")):  # type: ignore
                images.append(str(add_to_start)+file)

    train_images = []
    test_images = []
    val_images = []

    train_images = images[: int(len(images) * split_ratio[0])]
    test_images = images[
        int(len(images) * split_ratio[0]) : int(
            len(images) * (split_ratio[0] + split_ratio[1])
        )
    ]
    val_images = images[int(len(images) * (split_ratio[0] + split_ratio[1])) :]
    train = ""
    for t in train_images:
        train += t + "\n"
    with open(os.path.join(save_to_path, "train.txt"), "w") as train_file:
        train_file.write(train.replace("\\", "/"))

    test = ""
    for t in test_images:
        test += t + "\n"
    with open(os.path.join(save_to_path, "test.txt"), "w") as test_file:
        test_file.write(test.replace("\\", "/"))

    val = ""
    for v in val_images:
        val += v + "\n"
    with open(os.path.join(save_to_path, "val.txt"), "w") as val_file:
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

    def _aug_data_err(data_path: str, save_to_path: str, aug_rate):  # type: ignore
        import albumentations as alb

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Can't find folder 🤷\n data_path: " + str(data_path)
            )

        if not (os.path.exists(save_to_path)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n save_to_path: " + str(save_to_path)
            )
        if not isinstance(aug_rate[0], alb.Compose):
            raise ValueError(
                "Your aug_rate value is not valid. Please use aug_rate.(aug_max, aug_mid, aug_low) or for custom use (albumentations.Compose([params...]), 15(aug_rate_count))"
            )

        if type(aug_rate[1]) != int:
            raise ValueError(
                "Your aug_rate value is not valid. Please use [aug_rate.(aug_max, aug_mid, aug_low)[0], aug_rate.(aug_max, aug_mid, aug_low)[1]] or for custom use (albumentations.Compose([params...]), aug_rate_count:int)"
            )
        folder_true = os.path.isdir(os.path.join(data_path, os.listdir(data_path)[0]))
        if folder_true:
            for folder in os.listdir(data_path):
                for f in os.listdir(os.path.join(data_path, folder)):
                    if not os.path.isfile(os.path.join(data_path, folder, f)):
                        folder_true = False
                        break

        if not folder_true:  # type: ignore
            file_tree = """
            data/
            |   A/
            |   |   file1.data-type
            |   |   ...
            |   B/
            |   |   file1.data-type
            |   |   ...
            |   C/
            |   |   file1.data-type 
            |   |   ...
            |   ...
            """
            raise FileNotFoundError(
                "Can't find class folders 🤷\ndata_path: "
                + str(data_path)
                + "\nfile tree must be like this:\n"
                + file_tree
            )

        gc.collect()

    def aug_data(
        data_path: str,  # type: ignore
        aug_rate: list,
        save_to_path,
        equlize_img_count: bool = True,
        similar_p: float = 0.9,
    ):
        """
        Augment images in a directory using specified augmentation rates.

        Args:
            data_path (str): The path to the directory containing the original images.
            aug_rate (list): A tuple containing an augmentation pipeline (albumentations.Compose) and the number of augmentations per image.
            save_to_path (str): The path where augmented images will be saved.
            equlize_img_count (bool, optional): Whether to equalize image counts after augmentation. Default is True.
            similar_p (float, optional): Similarity threshold for deleting similar images. Default is 0.9.
        """

        def aug_data_folder(class_folder, progress_callback):
            i = 0
            total_images = len(os.listdir(os.path.join(data_path, class_folder)))
            print(os.path.join(data_path, class_folder))
            for f in os.listdir(os.path.join(data_path, class_folder)):
                image_path = os.path.join(data_path, class_folder, f)
                image = Image.open(image_path)

                for _ in range(aug_rate[1] + 1):
                    augmented_data = aug_rate[0](image=np.array(image))

                    augmented_image = Image.fromarray(augmented_data["image"])

                    save_path = os.path.join(save_to_path, class_folder)
                    os.makedirs(save_path, exist_ok=True)
                    augmented_image.save(
                        os.path.join(save_path, "aug_" + str(i) + "_" + f)
                    )

                    i += 1
                progress_percent = (i / (total_images * (aug_rate[1] + 1))) * 100
                progress_callback(class_folder, int(progress_percent))

        BurobotOutput.clear_and_memory_to()
        ImageAugmentation._aug_data_err(data_path, save_to_path, aug_rate)  # type: ignore

        BurobotOutput.print_burobot()

        progress_lock = threading.Lock()
        progress_data = {}

        def update_progress(folder, progress):
            with progress_lock:
                progress_data[folder] = progress
                total_progress = sum(progress_data.values()) / len(progress_data)
                print(f"\r🔄 Data augmenting %{int(total_progress)} 😎", end="")

        threads = []
        for folder in os.listdir(data_path):
            thread = threading.Thread(
                target=aug_data_folder,
                args=(
                    folder,
                    update_progress,
                ),
            )
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleting duplicate images 🎴🖼️📜 🖼️🪚")
        deleted_c = delete_duplicate_images(save_to_path)
        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleted " + str(deleted_c) + " duplicate image(s) 🪚😋")
        time.sleep(3)

        if similar_p is not None and similar_p > 0:
            if equlize_img_count:
                BurobotOutput.clear_and_memory_to()
                BurobotOutput.print_burobot()
                equalize_image_count(save_to_path)
            BurobotOutput.print_burobot()
            BurobotOutput.clear_and_memory_to()
            delete_similar_imgs(save_to_path, similar_p)

        if equlize_img_count:
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            equalize_image_count(save_to_path)

        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        print("✅ Data augmentation completed successfully! 😋🎉")


# BUROBOT
