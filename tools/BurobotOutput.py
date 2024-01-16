# BUROBOT
import gc, os
import tensorflow as tf


def clearAndMemoryTo():
    """
    Clear Keras backend session and release resources, and clear console output.

    Clears backend session and releases resources in use. Also, clears the console output for better display.

    Note:
        This function also attempts to clear the output display in Google Colab if applicable.
    """
    # Clear Keras backend session and release resources
    gc.collect()
    try:
        # If running on Google Colab, clear output display
        from google.colab import output  # type: ignore

        output.clear()
        return
    except:
        pass
    # Clear console output (for non-Colab environments)
    os.system("cls" if os.name == "nt" else "clear")


def clear():
    """
    Clear console output.

    Clears the console output for better display.

    Note:
        This function also attempts to clear the output display in Google Colab if applicable.
    """
    tf.keras.backend.clear_session()
    gc.collect()
    try:
        # If running on Google Colab, clear output display
        from google.colab import output  # type: ignore

        output.clear()
        return
    except:
        pass
    # Clear console output (for non-Colab environments)
    os.system("cls" if os.name == "nt" else "clear")


def printBurobot(justReturn: bool = False):
    """
    Print a message or return the message string.

    Args:
        justReturn (bool, optional): Whether to just return the message string. Default is False.

    Returns:
        None or str: If justReturn is True, returns the message string; otherwise, prints the message and returns None.
    """
    if not justReturn:
        print("-" * 50 + "BUROBOT" + "-" * 50)
        return
    return "-" * 50 + "BUROBOT" + "-" * 50


# BUROBOT
