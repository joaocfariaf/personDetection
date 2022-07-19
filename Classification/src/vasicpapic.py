from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

model = Sequential([
    # Camada com 32 filtros,actv RELU e MAXPOOL 3X3
    Conv2D(32, 3, input_shape=(81, 81, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=3),
    # Camada com 32 filtros, actv RELU e MAXPOOL 3X3
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3), strides=3),
    # Camada com 64 filtros, actv RELU
    Conv2D(64, (3, 3), activation='relu'),
    Activation('relu'),
    # Camada com 64 filtros, actv RELU
    Conv2D(64, (3, 3), activation='relu'),

    # Fully Connected com 64 filtros, actv RELU
    Flatten(),
    Dense(64, activation='relu'),
    # Output layer com Softmax
    # Dense(1, activation='softmax')
    Dense(1, activation='sigmoid')
])

# SGD, learning rate = 0.001; accuracy as metrics
model.compile(loss='binary_crossentropy',
              optimizer=SGD(learning_rate=1e-2),
              metrics=['accuracy'])
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # plots the model and saver to a .png archive
# plot_model(model, to_file='vasicpapic.png', show_shapes=True, show_layer_names=True, show_layer_activations=True,
#            show_dtype=True)

batch_size = 64
n_epochs = 10

# Data Augmentation
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
# Nesse caso, não há uma geração de dados e sim uma rescala dos valores contidos em cada pixel
test_datagen = ImageDataGenerator(rescale=1. / 255)#rescale=1. / 255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    '../data/train',  # this is the target directory
    target_size=(81, 81),  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    '../data/validation',
    target_size=(81, 81),
    batch_size=batch_size,
    class_mode='binary')

H = model.fit(
    train_generator,
    steps_per_epoch=2800 // batch_size,
    epochs=n_epochs,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)

# plot the training loss and accuracy
N = n_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plotvp.png")


# Nesse caso, não há uma geração de dados e sim uma rescala dos valores contidos em cada pixel
test_datagen = ImageDataGenerator(rescale=1./255)
NUM_TEST_IMAGES = 6875

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
test_generator = test_datagen.flow_from_directory(
        '../data/test',  # this is the target directory
        target_size=(81, 81),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
predIdxs = model.evaluate(test_generator, steps=(NUM_TEST_IMAGES // batch_size) + 1)
# predIdxs = np.argmax(predIdxs, axis=1)
print(predIdxs)

# model.save_weights(
#     '11k_validation_dropout_0.5_25K_training.h5')  # always save your weights after training or during training

# from sklearn.metrics import roc_curve
# y_pred_keras = model.predict(X_test).ravel()
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
#
#
# from sklearn.metrics import auc
# auc_keras = auc(fpr_keras, tpr_keras)