import pandas as pd
import os
import tensorflow as tf

from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from keras import backend as K

from sklearn.metrics import roc_auc_score, average_precision_score
from LearningFunctions import train_flow, test_flow, compile_model, roc_callback

# -----------------------------LOAD IN OUR DATA---------------------------------- #
TRAIN = 'CheXpert-v1.0-small/train.csv'
VALID = 'CheXpert-v1.0-small/valid.csv'

# read in the training dataset
train = pd.read_csv(TRAIN, dtype=str)
# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#mapping to different labels
label = {'0.0': '0', '1.0' : '1', '-1.0' : '1'}

#convert the missing data to zero as nothing means no mention of the effect
train['Pleural Effusion'].loc[train['Pleural Effusion'].isna()] = '0.0'
train['label'] = train['Pleural Effusion'].map(label)
num_samples = len(train)

print(train['label'].value_counts())

#same for validation set even though there should be no issue here with missing data
valid['label'] = valid['Pleural Effusion'].map(label)
num_valid = len(valid)

print(valid['label'].value_counts())

# -------------------------------------------------------------------------------- #

# -------------------------Process for training Data---------------------------- #
BATCH_SIZE = 16
CONV_BASE = 'DenseNet121'
EPOCHS = 3
# WEIGHTS = 'DenseNet121_24_6_weights_lr_reduce_from32_16.hdf5'


train_generator = train_flow(train, (320,320), BATCH_SIZE)
valid_generator = test_flow(valid, (320,320))

# STEPS_PER_EPOCH = int(len(train_generator.labels)/BATCH_SIZE)
STEPS_PER_EPOCH = 50
VALID_STEPS = 1


print('-----------------------------------------')
print('Batched training shapes')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

print('Batched valid shapes')
for data_batch, labels_batch in valid_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
print('-----------------------------------------')

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

model = compile_model(loss = "binary_crossentropy",
                      opt = optimizers.Adam(),
                      metrics = ["accuracy", auroc],
                      conv_base = 'DenseNet121',
                      shape = train_generator.image_shape)

print('-----------------------------------------')
print('Model Summary for Training')
print(model.summary())
print('-----------------------------------------')


#-----------------------------SETUP CHECKPOINTS, TRAIN MODEL, & STORAGE-------------------------------------#

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = "{m}_{b}_{e}_model.h5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

weight_path="{m}_{b}_{e}_weights.hdf5".format(m=CONV_BASE, b=BATCH_SIZE, e=EPOCHS)

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=1, min_lr=0.00001, cooldown = 1)

roc = roc_callback(validation_data=valid_generator)

checkitout = [checkpoint, reduce_lr, roc]


model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch= STEPS_PER_EPOCH,
    validation_data=valid_generator,
    validation_steps= VALID_STEPS,
    callbacks=checkitout)


scoreSeg = model.evaluate_generator(valid_generator, steps=1)
print('--------------------------')
print('')
print("Accuracy = ",scoreSeg[1])
print('')

pred = model.predict_generator(valid_generator, steps=1)
pr_val = average_precision_score(valid_generator.labels, pred)
roc_val = roc_auc_score(valid_generator.labels, pred)

print('--------------------------')
print('')
print('Average Precision: %s' % str(round(pr_val, 4)))
print('')
print('--------------------------')
print('')
print('AUC: %s' % str(round(roc_val, 4)))
print('')
print('--------------------------')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
print(model_path)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
