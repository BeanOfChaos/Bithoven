import os
from random import shuffle
from math import floor
import pickle


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

DATAPATHS = ["./dataset/Dog/", "./dataset/Cat/"]
BATCH_SIZE = 100


def loadData(data_paths, test_prop=0.1):
    data = []
    for label, path in enumerate(data_paths):
        for file in os.listdir(path):
            if file.endswith(".jpg"):
                data.append((file, label))
    shuffle(data)
    return tuple(zip(*data))


if __name__ == "__main__":
    model = Sequential()

    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(256, 256, 1), activation='relu'))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1, clipnorm=1.)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.1)

    train_generator = datagen.flow_from_directory(
            './dataset/',
            target_size=(256, 256),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            color_mode='grayscale')

    histoly = model.fit_generator(train_generator,
                                    steps_per_epoch=249,
                                    epochs=2,
                                    workers=2,
                                    use_multiprocessing=False)
    with open("histoly.pickle", "wb") as f:
        pickle.dump(histoly, f)
    model.save('mypretty.model')
