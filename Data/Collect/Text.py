# BUROBOT
import pandas as pd
import sys, os, time

sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


def _check_errs(save_to_path: str):
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path: " + str(save_to_path))


def create_data(data_name: str, save_to_path: str):
    BurobotOutput.clear_and_memory_to()
    _check_errs(save_to_path)
    data = pd.DataFrame(columns=["Text", "Responses"])
    while True:
        BurobotOutput.clear_and_memory_to()
        BurobotOutput.print_burobot()
        print("Text bot data collecting system 😎")
        text = input("📜 Enter the user text: ")
        responses = []
        i = 1
        while True:
            BurobotOutput.clear_and_memory_to()
            response = input(f"Enter the response {i}(to exit write nothing): ")
            i += 1
            if response.replace(" ", "") == "":
                break
            else:
                responses.append(response)
        q = input(
            f"text: {text}\nresponses: {str(responses).replace('[', '').replace(']', '')}\nAre you sure? n/Y/e(exit))"
        )
        if q.lower() == "e":
            print("data skiped!")
            time.sleep(1)
            return data
        elif q.lower() != "n":
            if len(responses) == 0:
                print("skiping data...\n responses is empty")
                time.sleep(3)
            new_data = pd.DataFrame(
                {"Text": [text] * len(responses), "Responses": responses}
            )
            data = pd.concat([data, new_data], ignore_index=True)
            print("data added!")
            time.sleep(1)


# BUROBOT
