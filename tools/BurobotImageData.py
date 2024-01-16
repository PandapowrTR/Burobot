# BUROBOT
import os, gc
import tensorflow as tf


class ImageLoader:
    @staticmethod
    def loadData(
        path, batchSize: int, returnDataOnly: bool = True, extraValues: dict = {}
    ):
        """
        Load data from a specified path.

        Args:
            path (str): The path to the data folder.
            batchSize (int): batch size value to be used in data loading.
            returnDataOnly (bool, optional): Whether to return only the loaded data. Default is True.

        Returns:
            tuple or DataLoader: A tuple containing loaded data and labels, or a DataLoader object.
        """
        try:
            path = extraValues["data"]
            batchSize = extraValues["batchSize"]
            returnDataOnly = extraValues["returnDataOnly"]
        except:
            pass
        if not os.path.exists(path):
            raise FileNotFoundError("Can't find path 🤷\npath: " + str(path))
        import math
        from PIL import Image

        imgShape = None
        s = False
        for root, _, files in os.walk(path):
            if s:
                break
            for file in files:
                filePath = os.path.join(root, file)

                if filePath.endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(filePath)

                    imgShape = (img.size[0], img.size[1], len(img.getbands()))
                    del img
                    s = True
                    break
        print("🖼️ Image shape:" + str(imgShape))
        batchSize = max(batchSize, 1)
        data = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            batch_size=batchSize,
            seed=42,
            shuffle=True,
            image_size=(imgShape[0], imgShape[1]),  # type: ignore
        )
        gc.collect()
        tf.keras.backend.clear_session()
        if not returnDataOnly:
            return data, imgShape
        else:
            return data


def countImages(folderPath):
    """
    Count the number of image files in a given folder.

    Args:
        folderPath (str): The path to the folder containing the images.

    Returns:
        int: The number of image files in the folder.
    """
    count = 0
    for _, _, files in os.walk(folderPath):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                count += 1
    return count


# BUROBOT
