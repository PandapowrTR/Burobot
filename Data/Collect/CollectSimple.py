# BUROBOT
import pandas as pd
import os, sys, time

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def _catchErr(path: str, saveType):
    if not os.path.exists(path):
        raise FileNotFoundError("Can't find path 🤷\n path: " + path)
    if type(saveType) == list or type(saveType) == str:

        def check():
            if type(saveType) == list:
                for t in saveType:
                    if t in ["csv", "excel", "json"]:
                        return
                raise ValueError(
                    "The 'saveType' value doesn't include these values: ['csv', 'excel', 'json']🔢"
                )

        check()
        if type(saveType) == str:
            if saveType not in ["csv", "excel", "json"]:
                raise ValueError(
                    "The 'saveType' value doesn't include these values: ['csv', 'excel', 'json']🔢"
                )


def collect(path, saveType):
    _catchErr(path, saveType)
    BurobotOutput.clearAndMemoryTo()
    BurobotOutput.printBurobot()
    data = None
    if os.path.isfile(path):
        fileExtension = os.path.splitext(path)[-1].lower()
        if fileExtension == ".xlsx":
            existingDataFrame = pd.read_excel(path)
            data = existingDataFrame
        elif fileExtension == ".csv":
            existingDataFrame = pd.read_csv(path)
            data = existingDataFrame
        elif fileExtension == ".json":
            existingDataFrame = pd.read_json(path)
            data = existingDataFrame
        else:
            raise ValueError(
                "The file type doesn't include these values: ['csv', 'excel', 'json']🔢"
            )
        existingSamples = existingDataFrame.shape[0]
        print(f"Continuing from existing data at {path}.")
        time.sleep(3)
        path = os.path.split(path)[0]
    else:
        data = pd.DataFrame(columns=featureColumns)  # type: ignore
        existingDataFrame = None
        existingSamples = 0
        print(f"Starting fresh at {path}.")
        time.sleep(3)

    BurobotOutput.clearAndMemoryTo()
    BurobotOutput.printBurobot()
    featureColumns = []
    time.sleep(3)
    if existingDataFrame is not None:
        featureColumns = existingDataFrame.columns.tolist()
    else:
        numFeatures = int(input("How many columns will there be? 🧾 "))
        featureColumns = []
        for i in range(numFeatures):
            columnName = input(f"Name of column {i+1}: 📝 ")
            featureColumns.append(columnName)

    numSamples = existingSamples

    while True:
        BurobotOutput.clearAndMemoryTo()
        BurobotOutput.printBurobot()
        numSamples += 1
        print(f"\nSample {numSamples} Input:")
        newRow = {}
        for col in featureColumns:
            value = input(f"Value for {col}: ")
            newRow[col] = value

        data = pd.concat([data, pd.DataFrame(newRow, index=[0])], ignore_index=True)

        if type(saveType) == str:
            if saveType == "excel":
                data.to_excel(os.path.join(path, f"data.xlsx"), index=False)
            elif saveType == "csv":
                data.to_csv(os.path.join(path, f"data.csv"), index=False)
            elif saveType == "json":
                data.to_json(os.path.join(path, f"data.json"), index=False)
        else:  # list
            for t in saveType:
                if t == "excel":
                    data.to_excel(os.path.join(path, f"data.xlsx"), index=False)
                elif t == "csv":
                    data.to_csv(os.path.join(path, f"data.csv"), index=False)
                elif t == "json":
                    data.to_json(os.path.join(path, f"data.json"), index=False)
        print(f"Data successfully saved to {path} 💾")
        time.sleep(0.5)


# BUROBOT
