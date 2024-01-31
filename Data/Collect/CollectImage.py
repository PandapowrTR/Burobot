# BUROBOT
import time, os, cv2, keyboard, sys, pyautogui
import numpy as np

# Import the necessary modules and packages
sys.path.append(os.path.join(os.path.abspath(__file__).split("Burobot")[0], "Burobot"))
from Burobot.tools import BurobotOutput


# Helper function to check input errors
def _check_errs(saveToPath: str, resolution: tuple, delay: float = 0.1, i: int = 0):
    # Check if the save path exists
    if not os.path.exists(saveToPath):
        raise FileNotFoundError("Can't find path 🤷\nsaveToPath:" + str(saveToPath))
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
def collectFromWebcam(
    saveToPath: str,
    outputResolution: tuple,
    webcam: int = 0,
    delay: float = 0.1,
    i: int = 0,
):
    # Clear the outputs
    BurobotOutput.clearAndMemoryTo()
    # Check input errors
    _check_errs(saveToPath, outputResolution, delay, i)
    # Create a 'cv2.VideoCapture' object to capture video from the webcam
    cap = cv2.VideoCapture(webcam)
    collect = False
    while True:
        # Read a frame from the webcam
        _, frameOriginal = cap.read()
        # Resize the frame to the desired output resolution
        frameOriginal = cv2.resize(frameOriginal, outputResolution)
        frame = frameOriginal.copy()
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
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        if key == ord("c"):
            collect = not collect
            time.sleep(1)
        if collect:
            imgName = os.path.join(saveToPath, f"{str(i)}.jpg")
            try:
                cv2.imwrite(imgName, frameOriginal)
            except:
                os.remove(imgName)
                cv2.imwrite(imgName, frameOriginal)
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
def collectFromScreen(
    saveToPath: str, outputResolution: tuple, delay: float = 0.1, i: int = 0
):
    BurobotOutput.clearAndMemoryTo()
    _check_errs(saveToPath, outputResolution, delay, i)
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

    draw.rectangle([x, y], outline="red")  # type: ignore

    # Show the rectangle
    screen.show()
    del screen, draw
    collect = False
    BurobotOutput.clearAndMemoryTo()
    while True:
        if keyboard.is_pressed("c"):
            collect = not collect
            time.sleep(1)
        elif keyboard.is_pressed("q"):
            break

        if collect:
            img = pyautogui.screenshot()

            if not x[0] <= y[0] or not x[1] <= y[1]:  # type: ignore
                temp_x = x  # type: ignore
                x = y  # type: ignore
                y = temp_x
                del temp_x
            img = img.crop((x[0], x[1], y[0], y[1]))  # type: ignore
            img = img.resize(outputResolution)  # type: ignore
            BurobotOutput.clearAndMemoryTo()
            try:
                img.save(os.path.join(saveToPath, str(i) + ".jpg"))
                print(str(i) + ".jpg saved!")
            except:
                os.remove(os.path.join(saveToPath, str(i) + ".jpg"))
                print(str(i) + ".jpg deleted!")
                img.save(os.path.join(saveToPath, str(i) + ".jpg"))
                print(str(i) + ".jpg saved!")
            i += 1
            time.sleep(delay)

    return i


def collectFromVideo(videoPath: str, saveToPath: str, i: int = 0, printInfo:bool = True):
    """
    Collects frames from a video by pressing 'c' key and saves them to the specified path.

    :videoPath (str): The path of the input video file.
    :saveToPath (str): The directory where the frames will be saved.
    :i (int): An optional parameter to set the starting index of the saved frames.
    """
    # Open the video file
    cap = cv2.VideoCapture(videoPath)
    screenWidth, screenHeight = pyautogui.size()

    if not cap.isOpened():
        print("Could not open the video file.")
        return


    blackImage = np.zeros((int(screenWidth*0.3), int(screenHeight*0.3), 3), dtype=np.uint8)
    save = False
    while True:
        try:
            # Read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Video ended or could not read.")
                break
            orjFrame = frame.copy()
            frame = cv2.resize(frame, (int(screenWidth*0.3), int(screenHeight*0.3)))

            # Save the frame when 'c' is pressed
            key = cv2.waitKey(30) & 0xFF
            if key == ord('v'):
                save = not save
            if key == ord("c") or save:
                framePath = f"{saveToPath}/frame_{i}.png"
                cv2.imwrite(framePath, orjFrame)
                print(f"Frame saved: {framePath}")
                i += 1
            # Exit when 'q' is pressed
            elif key == ord("q"):
                break

            # Display information messages on the screen
            if printInfo:
                infoMessage = "Press 'c' to save. Press 'v' to toggle continuous saving. Press 'q' to exit."
                frame = cv2.putText(
                    frame, infoMessage, (int(screenWidth*0.3*0.05), int(screenHeight*0.3*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                frame = cv2.putText(
                    frame, "Continuous saving: "+str("YES" if save else "NO"), (int(screenWidth*0.3*0.05), int(screenHeight*0.3*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            cv2.imshow("Video", frame)
        except:
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


# BUROBOT
