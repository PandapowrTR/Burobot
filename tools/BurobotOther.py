import os, shutil


def copy_folder(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    files = os.listdir(source_folder)

    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)

        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        elif os.path.isdir(source_path):
            copy_folder(source_path, destination_path)

def delete_files_in_folder(folder_path):
    files = os.listdir(folder_path)

    for file in files:
        file_path = os.path.join(folder_path, file)

        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delete_files_in_folder(file_path)
            os.rmdir(file_path)