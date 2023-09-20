import os, shutil, os, gc, json, time, sys, uuid

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from PIL import Image
import numpy as np
import albumentations as alb
from Burobot.tools import BurobotOutput
from Burobot.Data.Dominate.DominateImage import *  # type: ignore


def split_data_to_txt(
    data_path: str,
    labels_path: str,
    save_to_path: str,
    split_ratio: tuple = (0.8, 0.1),
    add_to_start: str = "",
    labels_file_types: tuple = (".json", ".txt"),
):
    """NOTE split_ratio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(data_path):
        raise FileNotFoundError("Can't find path 🤷\ndata_path:" + str(data_path))
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Can't find path 🤷\nlabels_path:" + str(labels_path))
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path:" + str(save_to_path))

    if len(split_ratio) != 2 or any(
        split_ratio[i] < 0 for i in range(len(split_ratio))
    ):
        raise ValueError("split_ratio value is invalid. Please check the data 🔢")

    print("Splitting images and labels to txt files 🔪📰")
    images = []
    labels = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if str(file).endswith((".png", ".jpg", ".jpeg")):
                images.append(str(add_to_start) + file)

    for root, _, files in os.walk(labels_path):
        for file in files:
            if str(file).endswith(labels_file_types):
                labels.append(str(add_to_start) + file)

    train_images = images[: int(len(images) * split_ratio[0])]
    test_images = images[
        int(len(images) * split_ratio[0]) : int(
            len(images) * (split_ratio[0] + split_ratio[1])
        )
    ]
    val_images = images[int(len(images) * (split_ratio[0] + split_ratio[1])) :]

    train_labels = labels[: int(len(labels) * split_ratio[0])]
    test_labels = labels[
        int(len(labels) * split_ratio[0]) : int(
            len(labels) * (split_ratio[0] + split_ratio[1])
        )
    ]
    val_labels = labels[int(len(labels) * (split_ratio[0] + split_ratio[1])) :]

    train_data = ""
    for img, lbl in zip(train_images, train_labels):
        train_data += f"{img}\n{lbl}\n"

    test_data = ""
    for img, lbl in zip(test_images, test_labels):
        test_data += f"{img}\n{lbl}\n"

    val_data = ""
    for img, lbl in zip(val_images, val_labels):
        val_data += f"{img}\n{lbl}\n"

    with open(os.path.join(save_to_path, "train.txt"), "w") as train_file:
        train_file.write(train_data.replace("\\", "/"))

    with open(os.path.join(save_to_path, "test.txt"), "w") as test_file:
        test_file.write(test_data.replace("\\", "/"))

    with open(os.path.join(save_to_path, "val.txt"), "w") as val_file:
        val_file.write(val_data.replace("\\", "/"))


def split_data_to_folders(
    data_path: str,
    labels_path: str,
    save_to_path: str,
    split_ratio: tuple = (0.8, 0.1),
    labels_file_types: tuple = (".json", ".txt"),
):
    """NOTE split_ratio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(data_path):
        raise FileNotFoundError("Can't find path 🤷\ndata_path:" + str(data_path))
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Can't find path 🤷\nlabels_path:" + str(labels_path))
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path:" + str(save_to_path))

    if len(split_ratio) != 2 or any(
        split_ratio[i] < 0 for i in range(len(split_ratio))
    ):
        raise ValueError("split_ratio value is invalid. Please check the data 🔢")

    print("Splitting images and labels to txt files 🔪📁")

    train_imgs_path = os.path.join(save_to_path, "train/images")
    test_imgs_path = os.path.join(save_to_path, "test/images")
    val_imgs_path = os.path.join(save_to_path, "val/images")

    train_labels_path = os.path.join(save_to_path, "train/labels")
    test_labels_path = os.path.join(save_to_path, "test/labels")
    val_labels_path = os.path.join(save_to_path, "val/labels")

    os.makedirs(train_imgs_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(val_imgs_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(test_imgs_path, exist_ok=True)
    os.makedirs(test_labels_path, exist_ok=True)

    datas = {}
    label_type = ""
    for _, _, files in os.walk(data_path):
        for file in files:
            if not file.endswith((".png", ".jpg", ".jpeg")):
                label_type = "." + file.split(".")[-1]
                break
    for _, _, files in os.walk(data_path):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                datas[file] = ".".join(file.split(".")[:-1]) + label_type
    total_files = len(datas)
    num_train = int(total_files * split_ratio[0])
    num_test = int(total_files * split_ratio[1])

    train_i = 0
    test_i = 0
    for key, value in datas.items():
        if train_i < num_train:
            train_i += 1
            shutil.copy(os.path.join(data_path, key), train_imgs_path)
            shutil.copy(os.path.join(data_path, value), train_labels_path)
        elif test_i < num_test:
            test_i += 1
            shutil.copy(os.path.join(data_path, key), test_imgs_path)
            shutil.copy(os.path.join(data_path, value), test_labels_path)
        else:
            shutil.copy(os.path.join(data_path, key), val_imgs_path)
            shutil.copy(os.path.join(data_path, value), val_labels_path)

    print("Data splitting completed successfully 🥳")


def delete_alone_data(data_path: str, labels_path: str = None):  # type: ignore
    if not os.path.exists(data_path):
        raise FileNotFoundError("Can't find folder 🤷\n data_path: " + str(data_path))
    if labels_path is None:
        labels_path = data_path
    if labels_path is not None:
        if not os.path.exists(labels_path):
            raise FileNotFoundError("Can't find folder 🤷\n data_path: " + str(data_path))
    
    label_files = [[], []]  # 0: file with root, 1:file
    for root, _, files in os.walk(labels_path):
        for file in files:
            file_c = str(file)
            if file_c.lower().endswith(("json", "txt")):
                label_files[0].append(os.path.join(root, file))
                label_files[1].append(".".join(file.split(".")[:-1]))
                
    image_files = [[], []]  # 0: file with root, 1:file
    for root, _, files in os.walk(data_path):
        for file in files:
            file_c = str(file)
            if file_c.lower().endswith(("jpg", "png", "jpeg")):
                image_files[0].append(os.path.join(root, file))
                image_files[1].append(".".join(file.split(".")[:-1]))

    for i, image in enumerate(image_files[1]):
        try:
            label_files[1].index(image)
        except ValueError:
            os.remove(image_files[0][i])
    for i, label in enumerate(label_files[1]):
        try:
            image_files[1].index(label)
        except:
            os.remove(label_files[0][i])
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
    for root, dirs, files in os.walk(path):
        for file in files:
            if str(file).endswith((".jpg", ".png", ".jpeg")):
                for root2, _, check_files in os.walk(path):
                    for check_file in check_files:
                        try:
                            if img_are_similar(
                                os.path.join(root, file),
                                os.path.join(root2, check_file),
                                p,
                            ):
                                os.remove(os.path.join(root2, check_file))
                                deleted_c += 1
                                print(f"Deleted {str(check_file)} 🗑️\r", end="")
                        except:
                            pass
    return deleted_c


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

    def _aug_data_err(data_path: str, labels_path: str, save_to_path: str, aug_rate):  # type: ignore
        import albumentations as alb

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "Can't find folder 🤷\n data_path: " + str(data_path)
            )

        if not (os.path.exists(save_to_path)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n save_to_path: " + str(save_to_path)
            )
        if not (os.path.exists(labels_path)):
            raise FileNotFoundError(
                "Can't find folder 🤷\n save_to_path: " + str(labels_path)
            )

        if not type(aug_rate[0]) == list:
            raise ValueError(
                "Your aug_rate value is not valid. Please use aug_rate.(aug_max, aug_mid, aug_low) or for custom use [[albumentations.[YourTransform]], 15(aug_rate_count)]"
            )

        if type(aug_rate[1]) != int:
            raise ValueError(
                "Your aug_rate value is not valid. Please use aug_rate.(aug_max, aug_mid, aug_low) or for custom use (albumentations.Compose([params...]), aug_rate_count:int)"
            )
        error_path = ""
        for p in [data_path, labels_path]:
            if not os.path.exists(p):
                raise FileNotFoundError("Can't find files 🤷\ndata_path: " + str(p))

        gc.collect()

    def aug_data(
        data_path: str,  # type: ignore
        labels_path: str,  # type: ignore
        aug_rate: list,
        save_to_path,
        similar_p: float = 0.9,
    ):
        """
        Augment images in a directory using specified augmentation rates.

        Args:
            data_path (str): The path to the directory containing the original images.
            aug_rate (list): A list containing an augmentation pipeline (albumentations.Compose) and the number of augmentations per image.
            save_to_path (str): The path where augmented images will be saved.
            labels_path (str): The path to the directory containing the image labes (for object detection datas).
            equlize_img_count (bool, optional): Whether to equalize image counts after augmentation. Default is True.
            similar_p (float, optional): Similarity threshold for deleting similar images. To disable enter under 0 or None. Default is 0.9.
        """
        BurobotOutput.clear_and_memory_to()
        Augmentation._aug_data_err(data_path, labels_path, save_to_path, aug_rate)  # type: ignore

        aug_rate[0] = alb.Compose(
            aug_rate[0],
            bbox_params=alb.BboxParams(
                format="albumentations", label_fields=["class_labels"]
            ),
        )
        BurobotOutput.print_burobot()

        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleting alone data 🥺💔")
        time.sleep(1)
        delete_alone_data(data_path, labels_path)
        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        print("🔄 Data augmentating 😎")
        labels_files = [[], []]
        for root, dirs, files in os.walk(labels_path):
            for file in files:
                file_c = str(file)
                if file_c.endswith((".json")):
                    labels_files[0].append(
                        str(os.path.join(root, file))
                    )  # all path and file
                    labels_files[1].append(file)  # only file
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_c = str(file)
                if file_c.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(root, file)
                    image = Image.open(image_path)
                    labels = {}
                    class_labels = []
                    copy_labels = {}
                    label_file = labels_files[0][
                        labels_files[1].index(file.replace(".jpg", ".json"))
                    ]

                    img_width, img_height = image.size
                    with open(label_file, "r") as label_json:
                        labels = json.load(label_json)
                    new_labels = []
                    class_labels = []
                    copy_labels = dict(labels)
                    for l in copy_labels["shapes"]:
                        # convert corrs to albumentations type to convert other types
                        new_labels.append(
                            np.divide(
                                [
                                    l["points"][0][0],
                                    l["points"][0][1],
                                    l["points"][1][0],
                                    l["points"][1][1],
                                ],
                                [img_width, img_height, img_width, img_height],
                            )
                        )

                        class_labels.append(l["label"])
                    labels = new_labels
                    del new_labels

                    for i in range(aug_rate[1] + 1):
                        augmented_data = aug_rate[0](
                            image=np.array(image),
                            bboxes=labels,
                            class_labels=class_labels,
                        )

                        augmented_image = Image.fromarray(augmented_data["image"])
                        uu = str(uuid.uuid1())
                        save_path = os.path.join(save_to_path)
                        augmented_image.save(
                            os.path.join(
                                save_path, "aug_" + uu + "_" + file
                            )
                        )

                        augmented_labels = augmented_data["bboxes"]
                        with open(
                            os.path.join(
                                save_to_path,
                                "aug_"
                                + uu
                                + "_"
                                + file.replace(".jpg", ".json")
                                .replace(".jpg", ".png")
                                .replace(".jpeg", ".json"),
                            ),
                            "w",
                        ) as label_json:
                            for lab in range(len(augmented_labels)):
                                copy_labels["shapes"][lab]["points"] = augmented_labels
                            json.dump(copy_labels, label_json, indent=4)

        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleting duplicate images 🎴🖼️📜 🖼️🪚")
        deleted_c = delete_duplicate_images(save_to_path)
        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleted " + str(deleted_c) + " duplicate image(s) 🪚😋")
        time.sleep(3)

        BurobotOutput.clear_and_memory_to()

        BurobotOutput.print_burobot()
        print("🔄 Deleting alone data 🥺💔")
        time.sleep(1)
        delete_alone_data(save_to_path)
        BurobotOutput.clear_and_memory_to()

        if similar_p is not None and similar_p > 0:
            try:
                BurobotOutput.clear_and_memory_to()
                BurobotOutput.print_burobot()
                print(f"🔄 Deleting {similar_p*100}% similar or more images 🔍🧐")
                delete_similar_imgs(save_to_path, similar_p)
            except:
                pass

        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        print("✅ Data augmentation completed successfully! 😋🎉")
