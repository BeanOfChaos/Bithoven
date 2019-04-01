from pypianoroll import Multitrack
from os import listdir


def dataset(dirname):
    files = listdir(dirname)
    for file in files:
        if '.py' in file or '.npz' in file:
            yield Multitrack(file)
