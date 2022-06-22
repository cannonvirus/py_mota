import os
import shutil
import re


def extract_folder(path, ext="dir", full_path=False):
    """
    path : search path folder
    mode : "dir", ".jpg", ".png" ...
    full_path : os.path.join(*) or not
    """

    if ext == "dir":
        result = [i for i in sorted(os.listdir(path)) if re.search("[.]", i) is None]
    elif ext == "all":
        result = sorted(os.listdir(path))
    else:
        result = [i for i in sorted(os.listdir(path)) if re.search(ext, i) is not None]

    if full_path:
        result = [os.path.join(path, i) for i in result]

    return result


def zfill_filename(path, zfill_num=3, ext=".jpg"):
    """
    path : rename file in path
    zfill_num : if 3, 1 --> 001, 11 --> 011
    ext : change file extension
    """
    for file in extract_folder(path, ext=ext, full_path=False):
        file_name = re.sub(ext, "", file).lstrip("0")
        file_name = file_name.zfill(zfill_num)
        src = os.path.join(path, file)
        dst = os.path.join(path, file_name) + ext
        os.rename(src, dst)
