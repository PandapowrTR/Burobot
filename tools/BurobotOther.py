import os, shutil, inspect, zipfile, importlib


def copyFolder(sourceFolder, destinationFolder):
    os.makedirs(destinationFolder, exist_ok=True)

    files = os.listdir(sourceFolder)

    for file in files:
        sourcePath = os.path.join(sourceFolder, file)
        destinationPath = os.path.join(destinationFolder, file)

        if os.path.isfile(sourcePath):
            shutil.copy2(sourcePath, destinationPath)
        elif os.path.isdir(sourcePath):
            copyFolder(sourcePath, destinationPath)


def deleteFilesInFolder(folderPath):
    files = os.listdir(folderPath)

    for file in files:
        filePath = os.path.join(folderPath, file)
        try:
            if os.path.isfile(filePath):
                os.remove(filePath)
            elif os.path.isdir(filePath):
                deleteFilesInFolder(filePath)
                os.rmdir(filePath)
        except:
            pass


def saveFunctionsAsPythonFile(
    saveToPath: str, saveFileName: str, *functions, message: str = None
):
    """
    This method saves the specified methods in the specified path as a python file.
    :saveToPath (str): path that python file save.
    :saveFileName (str): Name of the python file to save.
    :functions: Methods to be saved in python file.
    :message (str): A messsage on top of the file.(optional)
    """
    codes = ""
    if message is not None:
        codes += "#" + str(message) + "\n"
    functionNames = []
    for i, func in enumerate(functions):
        if i != 0:
            codes += "\n\n"
        source = inspect.getsource(func)
        spaces = 0
        for s in source:
            if s == " ":
                spaces += 1
            else:
                break
        newSource = ""
        deltededSpaces = 0
        for s in source:
            if (s == " " and deltededSpaces == spaces) or (s != " "):
                newSource += s
            if s == " " and deltededSpaces != spaces:
                deltededSpaces += 1
            if s == "\n":
                deltededSpaces = 0

        codes += newSource
        functionNames.append(
            newSource.split("def")[-1].split(":")[0].replace("()", "").replace(" ", "")
        )

    with open(os.path.join(saveToPath, saveFileName + ".py"), "w") as f:
        f.write(codes)
    return functionNames


def convertModelFunctionToString(mFunction):
    try:
        moduleName = mFunction.__module__
        if moduleName.startswith("torch.optim") and isinstance(mFunction, type):
            return "torch.optim." + mFunction.__name__
        elif moduleName.startswith("torch.nn") and isinstance(mFunction, type):
            return "torch.nn." + mFunction.__name__
        elif moduleName.startswith("tensorflow.keras.optimizers") and isinstance(
            mFunction, type
        ):
            return "tf.keras.optimizers" + mFunction.__name__
        elif moduleName.startswith("tensorflow.keras.losses") and isinstance(
            mFunction, type
        ):
            return "tf.keras.losses" + mFunction.__name__
        return None
    except:
        return None


def convertStringToModelFunction(st):
    moduleName, objectName = st.rsplit(".", 1)
    module = importlib.import_module(moduleName)
    obj = getattr(module, objectName)
    return obj


def zipFolder(folderPath: str, zip: str):
    with zipfile.ZipFile(zip, "w") as zip:
        for root, dirs, files in os.walk(folderPath):
            for file in files:
                filePath = os.path.join(root, file)
                zip.write(filePath, os.path.relpath(filePath, folderPath))
