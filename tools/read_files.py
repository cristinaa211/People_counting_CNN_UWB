import os

def read_directories_names_in_path(path):
    """Returns a list of directories in the given path."""
    folders_list = []
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_dir():
                folders_list.append(entry.path)
    return folders_list

def read_all_files_names(path):
    """Returns a list of files in the given path."""
    files_list = []
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                files_list.append(entry.path)
    return files_list

def read_files_multiple_directories(path):
    """Returns a list of file paths from multiple directories within a path."""
    folders_list = read_directories_names_in_path(path)
    files_list = []
    if len(folders_list) != 0:
        for dir in folders_list:
            files = read_all_files_names(dir)
            files_list.extend(files)
        return files_list
    else:
        raise ValueError("The path has no directories.")