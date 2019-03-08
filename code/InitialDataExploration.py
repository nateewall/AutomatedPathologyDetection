import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers

#read in the training dataset
train=pd.read_csv("CheXpert-v1.0-small/train.csv", dtype = str)

#generate validation set
valid=pd.read_csv("CheXpert-v1.0-small/valid.csv", dtype = str)


#convert the unknowns to zero
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train['Pleural Effusion'].loc[train['Pleural Effusion'] == '-1.0'] = '0.0'

#pull a subset out for testing
train_subset = train.sample(10000)

#show what my distribution looks like
print(train_subset['Pleural Effusion'].value_counts())

#declare the datagen options
train_datagen = ImageDataGenerator(rescale=1./255,
                                   zca_epsilon=1e-06,
                                   # randomly shift images horizontally (fraction of total width)
                                   width_shift_range=0.1,
                                   # randomly shift images vertically (fraction of total height)
                                   height_shift_range=0.1,
                                   horizontal_flip=True,  # randomly flip images
                                   vertical_flip=False  # randomly flip images
                                   )

#generate training dataset
train_generator = train_datagen.flow_from_dataframe(dataframe=train_subset,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="grayscale",
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    validate_filenames= True)

#convert the unknown to zero
valid['Pleural Effusion'].loc[valid['Pleural Effusion'].isna()] = '0.0'
valid['Pleural Effusion'].loc[valid['Pleural Effusion'] == '-1.0'] = '0.0'

print(valid['Pleural Effusion'].value_counts())

test_datagen = ImageDataGenerator(rescale=1./255,
                                   zca_epsilon=1e-06,
                                   # randomly shift images horizontally (fraction of total width)
                                   width_shift_range=0.1,
                                   # randomly shift images vertically (fraction of total height)
                                   height_shift_range=0.1,
                                   horizontal_flip=True,  # randomly flip images
                                   vertical_flip=False  # randomly flip images
                                   )


validation_generator = test_datagen.flow_from_dataframe(dataframe=valid,
                                                    directory=None,
                                                    x_col="Path",
                                                    y_col="Pleural Effusion",
                                                    class_mode="binary",
                                                    color_mode="grayscale",
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    validate_filenames= True)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)