import pandas as pd
import os
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras.applications import densenet, VGG16


#setting some hyper parameters at the top to play with
batch_size = 16
epochs = 3
opt = optimizers.Adam(lr=1e-4)
steps_per_epoch=1000


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'vgg16_032219.h5'


#read in the training dataset
train=pd.read_csv("CheXpert-v1.0-small/train.csv", dtype = str)
#generate validation set
valid=pd.read_csv("CheXpert-v1.0-small/valid.csv", dtype = str)


#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train = train[train['Pleural Effusion'] != '-1.0']
print(train['Pleural Effusion'].value_counts())
num_samples = len(train)

#same for validation set even though there should be no issue here with missing data
valid['Pleural Effusion'].loc[valid['Pleural Effusion'].isna()] = '0.0'
num_valid = len(train)

print(valid['Pleural Effusion'].value_counts())

#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255)

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    batch_size=batch_size)


# print(dir(train_generator))
print(train_generator.image_shape)

#set up the test data set
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    batch_size=batch_size)

# print(dir(train_generator))
print(valid_generator.image_shape)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=train_generator.image_shape)

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block2_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print(conv_base.summary())

# ####BUILDING SMALL NETWORK FROM SCRATCH####
# # model = models.Sequential()
# # model.add(layers.Conv2D(32, (3, 3), activation='relu',
# #         padding = 'same', input_shape=train_generator.image_shape))
# # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# # model.add(layers.MaxPooling2D((2, 2)))
# # model.add(layers.Dropout(0.25))
# # model.add(layers.Flatten())
# # model.add(layers.Dense(512, activation='relu'))
# # model.add(layers.Dense(1, activation='sigmoid'))

####USING PRE_TRAINED MODELS RETRAINING THE TOP LAYERS
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
print(model.summary())

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
    train_generator,
    steps_per_epoch= steps_per_epoch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=20)



if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)