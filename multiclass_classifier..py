from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
from keras import optimizers

# Initializing the CNN
classifier = Sequential()
#
# # Convolution
# classifier.add(Conv2D(140, (3, 3), input_shape=(64, 64, 3), activation="relu"))
# classifier.add(Conv2D(140, (3, 3), activation="softmax"))
# classifier.add(Conv2D(140, (3, 3), activation="relu"))
# classifier.add(Conv2D(140, (3, 3), activation="relu"))
# classifier.add(Conv2D(140, (3, 3), activation="relu"))
# classifier.add(Conv2D(140, (3, 3), activation="softmax"))
# classifier.add(Conv2D(140, (3, 3), activation="softmax"))
# classifier.add(Conv2D(140, (3, 3), activation="relu"))
# classifier.add(Conv2D(140, (3, 3), activation="softmax"))
# classifier.add(Conv2D(140, (3, 3), activation="softmax"))
#
#
#
# # Pooling
# # classifier.add(MaxPooling2D(pool_size=(6, 6)))
#
# # Flatting
# classifier.add(Flatten())
#
# # Full connection
# classifier.add(Dense(units=4, activation="softmax"))
#
# # compiling the CNN
# classifier.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])


classifier.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 3)))
# The input of this layer is the output of the first layer, which is a 28 * 28 * 6 node matrix.
# The size of the filter used in this layer is 2 * 2, and the step length and width are both 2, so the output matrix size of this layer is 14 * 14 * 6.
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# The input matrix size of this layer is 14 * 14 * 6, the filter size used is 5 * 5, and the depth is 16. This layer does not use all 0 padding, and the step size is 1.
# The output matrix size of this layer is 10 * 10 * 16. This layer has 5 * 5 * 6 * 16 + 16 = 2416 parameters
classifier.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
# The input matrix size of this layer is 10 * 10 * 16. The size of the filter used in this layer is 2 * 2, and the length and width steps are both 2, so the output matrix size of this layer is 5 * 5 * 16.
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# The input matrix size of this layer is 5 * 5 * 16. This layer is called a convolution layer in the LeNet-5 paper, but because the size of the filter is 5 * 5, #
# So it is not different from the fully connected layer. If the nodes in the 5 * 5 * 16 matrix are pulled into a vector, then this layer is the same as the fully connected layer.
# The number of output nodes in this layer is 120, with a total of 5 * 5 * 16 * 120 + 120 = 48120 parameters.
classifier.add(Flatten())
classifier.add(Dense(120, activation='relu'))
# The number of input nodes in this layer is 120 and the number of output nodes is 84. The total parameter is 120 * 84 + 84 = 10164 (w + b)
classifier.add(Dense(84, activation='relu'))
# The number of input nodes in this layer is 84 and the number of output nodes is 10. The total parameter is 84 * 10 + 10 = 850
classifier.add(Dense(4, activation='softmax'))
classifier.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    r'C:\Users\stak\Documents\ineuron\CNN network\dogcat\dogcat_new\parent',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\stak\Documents\ineuron\CNN network\dogcat\dogcat_new\parent',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')
#
model = classifier.fit_generator(training_set,
                                 steps_per_epoch=200,
                                 epochs=2,
                                 validation_data=test_set,
                                 validation_steps=10)

classifier.save("multiclass.h5")
print("Saved model to disk")
