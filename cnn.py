
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# 32 kernals of shape 3X3 with input image resolution as 64X64 and each pixel for each image having 3 different RGB reading
# 32 kernals because assuming we have 32 important feature to focus on out of millions of shapes avaliable in an image.

# Input_shape is required only in the first layer, not required in all layers in classfier

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer




# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

# We have dense layer even inside the dense hidden layer if we visulaize in netron
# those kernals are nothing but the weights of NN.


# is a hidden layer in Forword Connection network
#128 units of neural network we have with activation fucntion as relu
classifier.add(Dense(units = 128, activation = 'relu'))
# Sigmoid because it is a binary classifier.
# It is a out put layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))#softmax for multiclass classfication

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# loss function for regrssion is 'mse'

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# 1/255 is one of the normalization technique
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Batch size means in one single batch it going to take 32 pixel in the entire network.
# binary because it is binary classifier.
# after 32 pixels it will calculate the loss and process backward propagation.
training_set = train_datagen.flow_from_directory(r'C:\Users\stak\Documents\ineuron\CNN network\dogcat\dogcat_new\parent1',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\stak\Documents\ineuron\CNN network\dogcat\dogcat_new\parent1',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# validation steps 2 means after every 2 steps it will validate the accuracy.
model = classifier.fit_generator(training_set,
                         steps_per_epoch = 400,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2)


# Here there is total of one epoch and in each epoch we are going to pass the batch size value(i.re 32 pixel)
# for 400 times randomly selecting pixels from different images
# Here the 32 pixel which is sent 400 times per epoch this pixels are received from the convolved images not the real images
# The convolved images means the image after applying kernals and pooling.





classifier.save("model.h5")
print("Saved model to disk")

# Part 3 - Making new predictions



from keras.models import load_model
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\stak\Documents\ineuron\CNN network\dogcat\dogcat_new\parent1\saif\IMG_20200806_113936940_HDR.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
# we are flattening the image rows to send to model
test_image = np.expand_dims(test_image, axis = 0)
model = load_model('model.h5')
result = model.predict(test_image)
# print which class has which indices ex sana is 1 saif is 0
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'sana'
    print(prediction)
else:
    prediction = 'saif'
    print(prediction)