# BUROBOT
import time, os, cv2, keyboard, sys

# Import the necessary modules and packages
sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


# Helper function to check input errors
def _check_errs(save_to_path: str, resolution: tuple, delay: float = 0.1, i: int = 0):
    # Check if the save path exists
    if not os.path.exists(save_to_path):
        raise FileNotFoundError("Can't find path 🤷\nsave_to_path:" + str(save_to_path))
    # Check if the resolution has correct dimensions
    if len(resolution) != 2:
        raise ValueError("resolution must be like (w>0, h>0) 🖼️")
    # Check if the delay is greater than 0
    if not delay > 0:
        raise ValueError("delay must be bigger than 0 🔢")
    # Check if the index is greater or equal to 0
    if not i >= 0:
        raise ValueError("i must be bigger or equal to 0 🔢")


# Function to collect images from the webcam
def collect_from_webcam(
    save_to_path: str,
    output_resolution: tuple,
    webcam: int = 0,
    delay: float = 0.1,
    i: int = 0,
):
    # Clear the outputs
    BurobotOutput.clear_and_memory_to()
    # Check input errors
    _check_errs(save_to_path, output_resolution, delay, i)
    # Create a 'cv2.VideoCapture' object to capture video from the webcam
    cap = cv2.VideoCapture(webcam)
    collect = False
    while True:
        # Read a frame from the webcam
        _, frame_orj = cap.read()
        # Resize the frame to the desired output resolution
        frame_orj = cv2.resize(frame_orj, output_resolution)
        frame = frame_orj.copy()
        # Add instructions to the frame
        frame = cv2.putText(
            frame,
            "Press 'C' for collect img exit with 'q'",
            (int(frame.shape[0] * 0.2), int(frame.shape[1] * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        if keyboard.is_pressed("c"):
            collect = not collect
            time.sleep(1)
        elif keyboard.is_pressed("q"):
            cv2.destroyAllWindows()
            break
        if collect:
            img_name = os.path.join(save_to_path, f"{str(i)}.jpg")
            try:
                cv2.imwrite(img_name, frame_orj)
            except:
                os.remove(img_name)
                cv2.imwrite(img_name, frame_orj)
            i += 1
            time.sleep(delay)
            frame = cv2.putText(
                frame,
                "Collected!",
                (int(frame.shape[0] * 0.3), int(frame.shape[1] * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("webcam", frame)
        cv2.waitKey(1)

    return i


# Function to collect images from the screen
def collect_from_screen(
    save_to_path: str, output_resolution: tuple, delay: float = 0.1, i: int = 0
):
    BurobotOutput.clear_and_memory_to()
    _check_errs(save_to_path, output_resolution, delay, i)
    from PIL import ImageDraw, ImageGrab
    import pyautogui

    # Take a screenshot of the screen
    screen = ImageGrab.grab()

    print("Press 'c' for get X coord")
    while not keyboard.is_pressed("c"):
        x = pyautogui.position()

    time.sleep(1)
    print("Press 'c' for get Y coord")
    while not keyboard.is_pressed("c"):
        y = pyautogui.position()

    draw = ImageDraw.Draw(screen)

    draw.rectangle([x, y], outline="red") #type: ignore

    # Show the rectangle
    screen.show()
    del screen, draw
    collect = False
    BurobotOutput.clear_and_memory_to()
    while True:
        if keyboard.is_pressed("c"):
            collect = not collect
            time.sleep(1)
        elif keyboard.is_pressed("q"):
            break

        if collect:
            img = pyautogui.screenshot()

            if not x[0] <= y[0] or not x[1] <= y[1]:#type: ignore
                temp_x = x#type: ignore
                x = y#type: ignore
                y = temp_x
                del temp_x
            img = img.crop((x[0], x[1], y[0], y[1]))#type: ignore
            img = img.resize(output_resolution)#type: ignore
            BurobotOutput.clear_and_memory_to()
            try:
                img.save(os.path.join(save_to_path, str(i) + ".jpg"))
                print(str(i) + ".jpg saved!")
            except:
                os.remove(os.path.join(save_to_path, str(i) + ".jpg"))
                print(str(i) + ".jpg deleted!")
                img.save(os.path.join(save_to_path, str(i) + ".jpg"))
                print(str(i) + ".jpg saved!")
            i += 1
            time.sleep(delay)

    return i


# BUROBOT
