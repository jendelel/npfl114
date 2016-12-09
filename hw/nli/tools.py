import os

def exp_name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
