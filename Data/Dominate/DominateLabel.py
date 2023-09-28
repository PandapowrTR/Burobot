import sys, os, json

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def convert_albumentations_labels_to_yolo(labels_path: str):
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Can't find path 🤷\nlabels_path:" + str(labels_path))

    yolo_labels = []
    class_labels = []

    for root, dirs, files in os.walk(labels_path):
        for file in files:
            if str(file).endswith(".json"):
                img_width, img_height = 0, 0
                with open(os.path.join(root, file), "r") as label_file:
                    label_data = json.load(label_file)
                    img_width, img_height = (
                        label_data["imageWidth"],
                        label_data["imageHeight"],
                    )
                    corrs = label_data["shapes"]
                    for corr in corrs:
                        label_name = corr["label"]
                        if label_name not in class_labels:
                            class_labels.append(label_name)

    for root, dirs, files in os.walk(labels_path):
        for file in files:
            if str(file).endswith(".json"):
                with open(os.path.join(root, file), "r") as label_file:
                    label_data = json.load(label_file)
                    corrs = label_data["shapes"]
                    yolo_label_list = []

                    for corr in corrs:
                        label_name = corr["label"]
                        class_index = class_labels.index(label_name)
                        corr = corr["points"][0]
                        x_min, y_min, x_max, y_max = corr[0], corr[1], corr[2], corr[3]
                        x_center = (x_min + x_max) / 2.0
                        y_center = (y_min + y_max) / 2.0
                        width = x_max - x_min
                        height = y_max - y_min

                        yolo_label = [class_index, x_center, y_center, width, height]
                        yolo_label_list.append(yolo_label)

                    yolo_txt_file_path = os.path.join(
                        root, ".".join(file.split(".")[:-1]) + ".txt"
                    )
                    with open(yolo_txt_file_path, "w") as yolo_txt:
                        for i, yolo_label in enumerate(yolo_label_list):
                            end = "\n"
                            if i == len(yolo_label_list) - 1:
                                end = ""
                            yolo_txt.write(
                                f"{yolo_label[0]} {yolo_label[1]} {yolo_label[2]} {yolo_label[3]} {yolo_label[4]}"
                                + end
                            )

                # JSON dosyasını silin
                os.remove(os.path.join(root, file))


def split_labels_to_txt(
    labels_path: str, save_to_path: str, split_ratio: tuple = (0.8, 0.1), add_to_start:str="", labels_file_types:tuple = (".json", ".txt")
):
    """NOTE split_ratio = (train, test), val = 1 - (train+test)"""
    if not os.path.exists(labels_path):
        raise FileNotFoundError("Can't find path 🤷\nlabels_path:" + str(labels_path))
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path:" + str(save_to_path))

    if len(split_ratio) != 2 or any(
        split_ratio[i] < 0 for i in range(len(split_ratio))
    ):
        raise ValueError("split_raito value is invalid. Please check the data 🔢")

    print("Splitting labels to txt files 🔪📰")
    labels = []

    for root, _, files in os.walk(labels_path):
        for file in files:
            if str(file).endswith(labels_file_types):
                labels.append(str(add_to_start)+ file)

    train_labels = []
    test_labels = []
    val_labels = []

    train_labels = labels[: int(len(labels) * split_ratio[0])]
    test_labels = labels[
        int(len(labels) * split_ratio[0]) : int(
            len(labels) * (split_ratio[0] + split_ratio[1])
        )
    ]
    val_labels = labels[int(len(labels) * (split_ratio[0] + split_ratio[1])) :]
    train = ""
    for t in train_labels:
        train += t + "\n"
    with open(os.path.join(save_to_path, "train.txt"), "w") as train_file:
        train_file.write(train.replace("\\", "/"))

    test = ""
    for t in test_labels:
        test += t + "\n"
    with open(os.path.join(save_to_path, "test.txt"), "w") as test_file:
        test_file.write(test.replace("\\", "/"))

    val = ""
    for v in val_labels:
        val += v + "\n"
    with open(os.path.join(save_to_path, "val.txt"), "w") as val_file:
        val_file.write(val.replace("\\", "/"))
        
