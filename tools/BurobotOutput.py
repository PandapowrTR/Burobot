# BUROBOT
import gc, os
try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except:
    pass
import tensorflow as tf


def clear_and_memory_to():
    """
    Clear Keras backend session and release resources, and clear console output.

    Clears backend session and releases resources in use. Also, clears the console output for better display.

    Note:
        This function also attempts to clear the output display in Google Colab if applicable.
    """
    # Clear Keras backend session and release resources
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


def clear():
    """
    Clear console output.

    Clears the console output for better display.

    Note:
        This function also attempts to clear the output display in Google Colab if applicable.
    """
    try:
        # If running on Google Colab, clear output display
        from google.colab import output  # type: ignore

        output.clear()
        return
    except:
        pass
    # Clear console output (for non-Colab environments)
    os.system("cls" if os.name == "nt" else "clear")


def print_burobot(just_return: bool = False):
    """
    Print a message or return the message string.

    Args:
        just_return (bool, optional): Whether to just return the message string. Default is False.

    Returns:
        None or str: If just_return is True, returns the message string; otherwise, prints the message and returns None.
    """
    if not just_return:
        print(
            "======================================BUROBOT======================================"
        )
        return
    return "======================================BUROBOT======================================"


# BUROBOT
