import os, json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except:
    pass
import tensorflow as tf


def load_data_from_json(data_path: str, tokenizer_num_words:int, return_datafr: bool = False):
    """
    returns(if return_dafafr true): x_train, y_train, tokenizer, unique_word_count, output_length, dataframe
    returns(if return_dafafr false): x_train, y_train, tokenizer, unique_word_count, output_length
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("Cant find path🤷\ndata_path: " + str(data_path))
    data_type = data_path.split("/")[-1].split("\\")[-1].split(".")[-1]
    if data_type != "json":
        raise FileNotFoundError("File type is not json 📜\ndata type: " + str(data_type))
    data = ""
    with open(data_path, "r") as f:
        data = json.load(f)
    if tokenizer_num_words <= 0:
        raise ValueError("tokenizer_num_words must be bigger than 0 🔢")
    tags = []
    patterns = []
    answers = {}
    if len(data["intents"]) == 0:
        raise ValueError(
            "Data is incorrect 💥 example data:\n"
            + """
{
    "intents": [
      {
        "tag": "greeting",
        "patterns": [
          "Hello!"
          ],
        "answers": ["You got a error my friend 😥"]
      }
    ]
}
"""
        )
    for intent in data["intents"]:
        answers[intent["tag"]] = intent["answers"]
        for lines in intent["patterns"]:
            patterns.append(lines)
            tags.append(intent["tag"])
    data = pd.DataFrame({"patterns": patterns, "tags": tags})
    dataframe = data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=tokenizer_num_words)
    tokenizer.fit_on_texts(data["patterns"])
    _tokenizer = tokenizer
    train = tokenizer.texts_to_sequences(data["patterns"])

    x_train = tf.keras.preprocessing.sequence.pad_sequences(train)
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(data["tags"])
    unique_word_count = len(tokenizer.word_index)
    output_length = le.classes_.shape[0]
    if return_datafr:
        return x_train, y_train, _tokenizer, unique_word_count, output_length, dataframe
    return x_train, y_train, _tokenizer, unique_word_count, output_length


def load_data_from_json_for_test(data_path: str):
    """returns: x_test, y_test"""
    if not os.path.exists(data_path):
        raise FileNotFoundError("Cant find path🤷\ndata_path: " + str(data_path))
    data_type = data_path.split("/")[-1].split("\\")[-1].split(".")[-1]
    if data_type != "json":
        raise FileNotFoundError("File type is not json 📜\ndata type: " + str(data_type))

    data = ""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    y_test = []
    x_test = []
    if len(data["intents"]) == 0:
        raise ValueError(
            "Data is incorrect 💥 example data:\n"
            + """
{
    "intents": [
      {
        "tag": "greeting",
        "patterns": [
          "Hello!"
          ],
        "answers": ["You got a error my friend 😥"]
      }
    ]
}
"""
        )
    for intent in data["intents"]:
        for lines in intent["patterns"]:
            x_test.append(lines)
            y_test.append(intent["tag"])

    return x_test, y_test


def split_data(data_path: str, train_radio: float = 0.8):
    if not os.path.exists(data_path):
        raise FileNotFoundError("Cant find path🤷\ndata_path: " + str(data_path))
    data_type = data_path.split("/")[-1].split("\\")[-1].split(".")[-1]
    if data_type != "json":
        raise FileNotFoundError("File type is not json 📜\ndata type: " + str(data_type))
    if train_radio < 0 or train_radio > 1:
        raise ValueError("train_radio must be bigger than 0 and lower than 1")

    data = None
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data["intents"]
    data_name = data_path.split("/")[-1].split("\\")[-1].split(".")[:-1]
    train_count = int(len(data) * train_radio)
    train = {"intents": data[:train_count]}
    test = {"intents": data[train_count:]}
    with open(
        os.path.join(
            ".".join(data_path.split(".")[:-1]).replace(".json", ""),
            data_path + "_train.json",
        ),
        "w",
        encoding="utf-8"
    ) as tr:
        json.dump(train, tr, ensure_ascii=False)
    with open(
        os.path.join(
            ".".join(data_path.split(".")[:-1]).replace(".json", ""),
            data_path + "_test.json",
        ),
        "w",
        encoding="utf-8"
    ) as te:
        json.dump(test, te, ensure_ascii=False)
