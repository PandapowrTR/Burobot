import json, os, random, sys

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


# Returns the synonyms of the word
def get_synonyms(word: str, syn):
    if word in syn:
        return syn[word.lower()]
    return None


class TextAugmentation:
    def _check_errs(data_path: str, save_to_path: str):
        if not (os.path.isfile(data_path) and os.path.exists(save_to_path)):
            raise FileNotFoundError(
                "Cant find path(s) 🤷\ndata_path:"
                + str(data_path)
                + "\nsave_to_path:"
                + str(save_to_path)
            )
        if len(data_path.split(".json")) < 2:
            raise FileNotFoundError("Please enter a json file 📜")

    def aug_data(data_path: str, save_to_path: str):
        TextAugmentation._check_errs(data_path, save_to_path)
        try:
            data = {}
            with open(data_path, "r", encoding="UTF-8") as f:
                data = json.load(f)
            syn = {}
            for syn_file in ["Turkish_synonyms.json", "English_synonyms.json"]:
                with open(
                    os.path.join(
                        os.path.abspath(__file__).replace("DominateTextData.py", ""),
                        "TextDatatools",
                        syn_file,
                    ),
                    "r",
                    encoding="UTF-8",
                ) as f:
                    syn = json.load(f)

            augmented_data = []
            all_ = len(data["intents"])
            i = 0
            for intent in data["intents"]:
                i +=1
                BurobotOutput.clear_and_memory_to()
                BurobotOutput.print_burobot()
                print(f"Data augmentation {str((all_/i)*100)}% 😎\r")
                for pattern in intent["patterns"]:
                    for word in pattern.split(" "):
                        syns = get_synonyms(word, syn)
                        if syns is None:
                            continue
                        new_word = random.choice(syns)
                        new_pattern = pattern.replace(word, new_word)

                        augmented_data.append(
                            {
                                "tag": intent["tag"],
                                "patterns": [new_pattern],
                                "answers": intent["answers"],
                            }
                        )
                for answer in intent["answers"]:
                    for word in answer.split(" "):
                        syns = get_synonyms(word, syn)
                        if syns is None:
                            continue
                        new_word = random.choice(syns)
                        new_answer = pattern.replace(word, new_word)

                        augmented_data.append(
                            {
                                "tag": intent["tag"],
                                "patterns": intent["patterns"],
                                "answers": [new_answer],
                            }
                        )

            data_name = data_path.split("\\")[-1].split("/")[-1].replace(".json", "")

            with open(
                os.path.join(save_to_path, str(data_name + "_aug.json")),
                "w",
                encoding="UTF-8",
            ) as f:
                json.dump({"intents": data["intents"] + augmented_data}, f, ensure_ascii=False)
        except Exception as e:
            raise Exception("😵‍💫 Something went wrong. Error:" + str(e))
