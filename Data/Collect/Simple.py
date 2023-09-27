# BUROBOT
import pandas as pd
import os, sys, time

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def _collect_err(path: str, save_type):
    if not os.path.exists(path):
        raise FileNotFoundError("Can't find path 🤷\n path: " + path)
    if type(save_type) == list or type(save_type) == str:

        def check():
            if type(save_type) == list:
                for t in save_type:
                    if t in ["csv", "excel", "json"]:
                        return
                raise ValueError(
                    "The 'save_type' value doesn't include these values: ['csv', 'excel', 'json']🔢"
                )

        check()
        if type(save_type) == str:
            if save_type not in ["csv", "excel", "json"]:
                raise ValueError(
                    "The 'save_type' value doesn't include these values: ['csv', 'excel', 'json']🔢"
                )


def collect(path, save_type):
    _collect_err(path, save_type)
    BurobotOutput.clear_and_memory_to()
    BurobotOutput.print_burobot()
    data = None
    if os.path.isfile(path):
        file_extension = os.path.splitext(path)[-1].lower()
        if file_extension == ".xlsx":
            existing_data_frame = pd.read_excel(path)
            data = existing_data_frame
        elif file_extension == ".csv":
            existing_data_frame = pd.read_csv(path)
            data = existing_data_frame
        elif file_extension == ".json":
            existing_data_frame = pd.read_json(path)
            data = existing_data_frame
        else:
            raise ValueError(
                "The file type doesn't include these values: ['csv', 'excel', 'json']🔢"
            )
        existing_samples = existing_data_frame.shape[0]
        print(f"Continuing from existing data at {path}.")
        time.sleep(3)
        path = os.path.split(path)[0]
    else:
        data = pd.DataFrame(columns=feature_columns)#type: ignore
        existing_data_frame = None
        existing_samples = 0
        print(f"Starting fresh at {path}.")
        time.sleep(3)

    BurobotOutput.clear_and_memory_to()
    BurobotOutput.print_burobot()
    feature_columns = []
    time.sleep(3)
    if existing_data_frame is not None:
        feature_columns = existing_data_frame.columns.tolist()
    else:
        num_features = int(input("How many columns will there be? 🧾 "))
        feature_columns = []
        for i in range(num_features):
            column_name = input(f"Name of column {i+1}: 📝 ")
            feature_columns.append(column_name)

    num_samples = existing_samples

    while True:
        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        num_samples += 1
        print(f"\nSample {num_samples} Input:")
        new_row = {}
        for col in feature_columns:
            value = input(f"Value for {col}: ")
            new_row[col] = value

        data = pd.concat([data, pd.DataFrame(new_row, index=[0])], ignore_index=True)

        if type(save_type) == str:
            if save_type == "excel":
                data.to_excel(os.path.join(path, f"data.xlsx"), index=False)
            elif save_type == "csv":
                data.to_csv(os.path.join(path, f"data.csv"), index=False)
            elif save_type == "json":
                data.to_json(os.path.join(path, f"data.json"), index=False)
        else:  # list
            for t in save_type:
                if t == "excel":
                    data.to_excel(os.path.join(path, f"data.xlsx"), index=False)
                elif t == "csv":
                    data.to_csv(os.path.join(path, f"data.csv"), index=False)
                elif t == "json":
                    data.to_json(os.path.join(path, f"data.json"), index=False)
        print(f"Data successfully saved to {path} 💾")
        time.sleep(0.5)


# BUROBOT
