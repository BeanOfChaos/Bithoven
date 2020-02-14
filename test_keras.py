import os
from random import shuffle
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATAPATHS = ["./PetImages/Dog/", "./PetImages/Cat/"]
BATCH_SIZE = 100


if __name__ == "__main__":
    model = Sequential()

    model.add(Conv2D(16, 3, input_shape=(256, 256, 1)))
    model.add(Conv2D(16, 3))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3))
    model.add(Conv2D(32, 3))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3))
    model.add(Conv2D(64, 3))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    adam = Adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
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
            './PetImages/',
            target_size=(256, 256),
            # batch_size=BATCH_SIZE,
            class_mode='categorical',
            color_mode='grayscale')

    histoly = model.fit_generator(train_generator,
                                  # steps_per_epoch=249,
                                  epochs=2,
                                  workers=2,
                                  use_multiprocessing=False)

    model.save('mypretty.model')
