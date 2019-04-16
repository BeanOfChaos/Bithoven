from pypianoroll import Multitrack
from os import listdir
from os.path import isdir, join
import numpy as np


def dataset(name):
    files = listdir(name)
    for file in files:
        fileName = join(name, file)
        if isdir(fileName):
            for song in dataset(fileName):
                yield song
        elif '.mid' in file or '.npz' in file:
            yield Multitrack(fileName)

def toTensor(multiTrack):
    multiTrack.pad_to_same()
    res = np.zeros(tuple(list(multiTrack.tracks[0].pianoroll.shape) + [len(multiTrack.tracks)]))
    for i, track in enumerate(multiTrack.tracks):
        res[:,:,i] = track.pianoroll
    return res

if __name__ == '__main__':
    for song in dataset('./lpd_cleansed/'):
        tensor = toTensor(song)
        print(tensor.shape)

