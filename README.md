README File is currently under development
There are a few sample codes for now.
Examples:
# NEW MODEL
```py
from Burobot import Training

modelName = "Name"
data = "data/to/path" or ["train/path", "test/path", "val/path"]
saveToPath = "save/to/path"
scheme = Training.ModelSchemes.ImageClassification.Scheme1(4, (200, 200, 3))
params = scheme.params
Training.GridSearchTrain.newModel(
    modelName,
    data,
    saveToPath,
    scheme.trainModel,
    {"params": params, "staticValues": scheme.staticValues},
    scheme.saveModel,
    {},
    scheme.loadModel,
    {},
    scheme.hardwareSetup,
    {},
    scheme.loadData,
    {},
    scheme.splitData,
    {},
    scheme.testModel,
    {},
)
```
# OLD MODEL

```py
from Burobot import Training

paramsFilePath = "params/file/path.json"
Training.GridSearchTrain.oldModel(paramsFilePath)
```
