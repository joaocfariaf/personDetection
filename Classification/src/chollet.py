from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np

model = Sequential([
        # Camada com actv RELU e MAXPOOL 2x2
        Conv2D(32, 3, input_shape=(81, 81, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Camada com actv RELU e MAXPOOL 2x2
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        # Camada com actv RELU e MAXPOOL 2x2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # the model so far outputs 3D feature maps (height, width, features)
        Flatten(),  # this converts our 3D feature maps to 1D feature vectors
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# plots the model and saves to a .png archive
# plot_model(model, to_file='kerastutorial.png', show_shapes=True, show_layer_names=True, show_layer_activations=True,
#            show_dtype=True)

batch_size = 64
n_epochs = 10

# Data Augmentation
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling 
# Nesse caso, não há uma geração de dados e sim uma rescala dos valores contidos em cada pixel
validation_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../data/train',  # this is the target directory
        target_size=(81, 81),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        '../data/validation',
        target_size=(81, 81),
        batch_size=batch_size,
        class_mode='binary')

H = model.fit_generator(
        train_generator,
        steps_per_epoch=2800 // batch_size,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# model.save_weights('11k_validation_dropout_0.5_25K_training.h5')  # always save your weights after training or during training

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
plt.savefig("plot.png")

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
# show a nicely formatted classification report

# from sklearn.metrics import classification_report
# print("[INFO] evaluating network...")
# print(classification_report(testLabels.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))