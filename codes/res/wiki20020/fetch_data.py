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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
    os.remove(ARCHIVE_NAME)
