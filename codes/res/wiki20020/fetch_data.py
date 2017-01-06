import os
import tarfile

try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen


URL = ("http://120.25.254.217:8000/static/"
       "input/wiki20020.tar.gz")


ARCHIVE_NAME = "wiki20020.tar.gz"
TARGET_FOLDER = "wiki20020"


if not os.path.exists(TARGET_FOLDER):
    print("Downloading dataset from %s (20 MB)" % URL)
    opener = urlopen(URL)
    with open(ARCHIVE_NAME, "wb") as f:
        f.write(opener.read())

    print("Decompressing %s" % ARCHIVE_NAME)
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall()
    os.remove(ARCHIVE_NAME)
