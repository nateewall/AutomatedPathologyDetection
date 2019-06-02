from keras import model

from LearningFunctions import train_flow, test_flow, compile_model, roc_callback, train_history


# -----------------------------LOAD IN OUR DATA---------------------------------- #
VALID = 'CheXpert-v1.0-small/valid.csv'

# generate validation set
valid = pd.read_csv(VALID, dtype=str)

#mapping to different labels
label = {'0.0': '0', '1.0' : '1', '-1.0' : '1'}

#same for validation set even though there should be no issue here with missing data
valid['label'] = valid['Pleural Effusion'].map(label)
num_valid = len(valid)

print(valid['label'].value_counts())

# -----------------------------LOAD MODEL---------------------------------- #
filename = 'DenseNet121_16_6_weights_lr_reduce_from32.hdf5'

model = compile_model(loss = "binary_crossentropy",
                      opt = optimizers.Adam(lr=0.0001, amsgrad = True),
                      weights = filename,
                      metrics = ["accuracy"],
                      conv_base = 'DenseNet121',
                      shape = train_generator.image_shape)

model = model.load_weights(filename)

print('-----------------------------------------')
print('Model Summary for Training')
print(model.summary())
print('-----------------------------------------')

