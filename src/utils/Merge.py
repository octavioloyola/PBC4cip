import os

def CheckSuffix(file, suffix):
    if not suffix or len(suffix) == 0:
        return True
    if not suffix in file or len(suffix) >= len(file):
        return False
    filename, file_extension = os.path.splitext(file)
    return filename[(len(suffix)*-1):] == suffix and file_extension == '.dat'


def GetFilesFromDirectory(directory):
    files = []
    if os.path.isdir(directory):
        # r=root, d=directories, f = files
        for r, d, f in os.walk(directory):
            for file in f:
                files.append(os.path.join(r, file))
        return files
    else:
        raise Exception(f"Directory '{directory}' is not valid.")

