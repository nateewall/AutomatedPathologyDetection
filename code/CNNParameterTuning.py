import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#-----------------------------INITIATE_SOME VARS-------------------------------------#

TRAIN = 'CheXpert-v1.0-small/train.csv'
VALID = 'CheXpert-v1.0-small/valid.csv'

TARGET_SIZE = (320,320)
BATCH_SIZE = 16
CONV_BASE = 'DenseNet121'
EPOCHS = 3
OPT = optimizers.Adam(lr=1e-4)
WEIGHTS = 'imagenet'
TRAINABLE = True

if CONV_BASE == 'VGG16':
    from keras.applications.vgg16 import VGG16 as BASE
elif CONV_BASE == 'ResNet152':
    from keras.applications.resnet import ResNet152 as BASE
elif CONV_BASE == 'DenseNet121':
    from keras.applications.densenet import DenseNet121 as BASE
elif CONV_BASE == 'NASNetLarge':
    from keras.applications.nasnet import NASNetLarge as BASE
else:
    raise ValueError('Unknown model: {}'.format(CONV_BASE))


#-----------------------------SETUP CHECKPOINTS AND MODEL STORAGE-------------------------------------#
save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = "{m}_{b}_{e}_model.h5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

weight_path="{m}_{b}_{e}_weights.hdf5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# -----------------------------LOAD IN OUR DATA---------------------------------- #

# read in the training dataset
train = pd.read_csv(TRAIN, dtype=str)
# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train = train[train['Pleural Effusion'] != '-1.0']
num_samples = len(train)

#same for validation set even though there should be no issue here with missing data
valid['Pleural Effusion'].loc[valid['Pleural Effusion'].isna()] = '0.0'
num_valid = len(train)

# -----------------------------DATA PREPROCESSING---------------------------------- #
#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.05,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.05,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

# set up the test data set
valid_datagen = ImageDataGenerator(rescale=1. / 255)

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)


valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="rgb",
                                                    target_size=TARGET_SIZE,
                                                    batch_size=BATCH_SIZE)

# -----------------------------COMPILE THE MODEL---------------------------------- #

conv_base = BASE(weights=WEIGHTS,
                        include_top=False,
                        input_shape=train_generator.image_shape,
                        pooling=max)

conv_base.trainable = TRAINABLE

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= OPT, metrics=['accuracy'])
print(model.summary())


# -----------------------------ADD SOME CHECKPOINTS---------------------------------- #

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch= (num_samples/BATCH_SIZE),
    validation_data=valid_generator,
    validation_steps= (num_valid/BATCH_SIZE),
    callbacks=[checkpoint, reduce_lr])


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)