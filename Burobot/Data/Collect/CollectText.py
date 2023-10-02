# BUROBOT
import sys, os, json, uuid

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def _check_errs(save_to_path: str):
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path: " + str(save_to_path))


def create_data(data_name: str, save_to_path: str):
    BurobotOutput.clear_and_memory_to()
    _check_errs(save_to_path)
    data = {"intents": []}

    while True:
        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        data_part = {}
        try:
            tag = input("🏷️ Please anter the tag value(for exit enter 'Ctrl + C'): ")
        except KeyboardInterrupt:
            break
        data_part["tag"] = tag
        patterns = []
        cancel = False
        while True:
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            print("Data name: "+data_name +"\n")
            print("🏷️ TAG: "+tag)
            if patterns != []:
                print("📚 Patterns: " + str(patterns) + "\n")
            try:
                pattern = input("📄 Please enter a pattern(for exit enter 'Ctrl + C', for cancel tag ' '): ")
            except KeyboardInterrupt:
                if not patterns == []:
                    break
                elif patterns == []:
                    print("⚠️ No pattern added. Please add a pattern or use ' ' for cancel tag")
            if pattern == " ":
                cancel = True
                break
            patterns.append(pattern)
        data_part["patterns"] = patterns
        answers = []
        while not cancel:
            BurobotOutput.clear_and_memory_to()
            BurobotOutput.print_burobot()
            print("Data name: "+data_name +"\n")
            print("🏷️ TAG: "+tag + "\n")
            print("📚 Patterns: " + str(patterns) + "\n")
            if answers != []:
                print("🗂️ Answers: "+str(answers))
            try:
                answer = input("💬 Please enter a answer(for exit enter 'Ctrl + C', for cancel tag ' '): ")
            except KeyboardInterrupt:

                if not answers == []:
                    break
                elif answers == []:
                    print("⚠️ No answer added. Please add a answer or use ' ' for cancel tag")
            if answer == " ":
                cancel = True
                break
            answers.append(answer)
        data_part["answers"] = answers
        if cancel:
            print("❌ Canceled tag")
            continue

        data["intents"].append(data_part)
    if os.path.exists(os.path.join(save_to_path, data_name)):
        data_name += str(uuid.uuid1().hex)
    with open(data_name+".json", "w") as f:
        f.write(json.dump(data))


# BUROBOT
