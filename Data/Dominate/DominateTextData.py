import json, os, nltk, random
nltk.download('wordnet')
# Returns the synset of the word
def get_synset(word):
    return nltk.corpus.wordnet.synsets(word)[0]


# Returns the synonyms of the word
def get_synonyms(word):
    try:
        return get_synset(word).lemmas()
    except:
        return None

# Returns the hypernyms of the word
def get_hypernyms(word):
    try:
        return get_synset(word).hypernyms()
    except:
        return None


# Returns the antonyms of the word
def get_antonyms(word):
    try:
        return get_synset(word).antonyms()
    except:
        return None


class TextAugmentation:
    class AugmentationRate:
        high = 1
        medium = 0.5
        low = 0

    def _check_errs(data_path: str, aug_rate, save_to_path: str):
        if not (os.path.isfile(data_path) and os.path.exists(save_to_path)):
            raise FileNotFoundError(
                "Cant find path(s) 🤷\ndata_path:"
                + str(data_path)
                + "\nsave_to_path:"
                + str(save_to_path)
            )
        if len(data_path.split(".json")) < 2:
            raise FileNotFoundError("Please enter a json file 📜")
        if not aug_rate in [1, 0.5, 0]:
            raise ValueError(
                "aug_rate must be 1, 0.5 or 0. Use TextAugmentation.AugmentationRate.[rate] 🔢"
            )

    def aug_data(data_path: str, aug_rate, save_to_path: str, encoding: str = "utf-8"):
        TextAugmentation._check_errs(data_path, aug_rate, save_to_path)
        # try:
        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        augmented_data = []
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                new_word = random.choice(get_synonyms(pattern))
                if new_word is None:
                    continue
                new_pattern = pattern.replace(pattern, new_word)

                if random.random() < aug_rate:
                    augmented_data.append(
                        {
                            "tag": intent["tag"],
                            "patterns": [new_pattern],
                            "answers": intent["answers"],
                        }
                    )

        data_name = os.path.splitext(data_path.split(".json")[0])[-1]

        with open(
            os.path.join(save_to_path, data_name + "_aug.json"),
            "w",
            encoding=encoding,
        ) as f:
            json.dump(augmented_data, f, indent=4)
        # except Exception as e:
        #     raise Exception("😵‍💫 Something went wrong. Error:" + str(e))
