from dataPreProcessing import *

from tensorflow.keras.optimizers import Adam

from vggModelUnit import vgg_model



vgg_model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
vgghist = vgg_model.fit(train_data, epochs = 100, validation_data = validation_data)
vgg_model.save("vgg_model.h5")
