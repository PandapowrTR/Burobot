# BUROBOT
import os, gc


try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except:
    pass
import tensorflow as tf


class ImageLoader:
    def count_images(self, folder_path):
        """
        Count the number of image files in a given folder.

        Args:
            folder_path (str): The path to the folder containing the images.

        Returns:
            int: The number of image files in the folder.
        """
        count = 0
        for _, _, files in os.walk(folder_path):
            for file in files:
                if (
                    file.endswith(".jpg")
                    or file.endswith(".png")
                    or file.endswith(".jpeg")
                ):
                    count += 1
        return count

    def load_data(self, path, device_memory: int, return_data_only: bool = True):
        """
        Load data from a specified path.

        Args:
            path (str): The path to the data folder.
            device_memory (int): The available device memory for data loading.
            return_data_only (bool, optional): Whether to return only the loaded data. Default is True.

        Returns:
            tuple or DataLoader: A tuple containing loaded data and labels, or a DataLoader object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError("Can't find path 🤷\npath: " + str(path))
        import math
        from PIL import Image

        device_memory = device_memory * 0.65  # type: ignore
        data_size = self.count_images(path)
        max_batch_size = int(math.ceil(data_size / 32))
        del data_size
        img_shape = None
        s = False
        for root, _, files in os.walk(path):
            if s:
                break
            for file in files:
                file_path = os.path.join(root, file)

                if file_path.endswith((".jpg", ".png", ".jpeg")):
                    img = Image.open(file_path)

                    img_shape = (img.size[0], img.size[1], len(img.getbands()))
                    del img
                    s = True
                    break
        print("🖼️ Image shape:" + str(img_shape))
        batch_size = int(
            math.floor(
                (device_memory * 1024 * 1024)
                / (img_shape[0] * img_shape[1] * img_shape[2] * 8)  # type: ignore
            )
        )
        batch_size = max(batch_size, 1)
        if batch_size > max_batch_size:
            batch_size = max_batch_size
        del s, max_batch_size
        data = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            seed=42,
            shuffle=True,
            image_size=(img_shape[0], img_shape[1]),  # type: ignore
        )
        gc.collect()
        tf.keras.backend.clear_session()
        if not return_data_only:
            return data, batch_size, img_shape
        else:
            return data


# BUROBOT
