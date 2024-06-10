from tensorflow.keras.preprocessing.image import ImageDataGenerator
def prepare_dataset(data_dir):
	datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 40, width_shift_range = .2, height_shift_range = .2, shear_range = .1, horizontal_flip = True, fill_mode = 'nearest', zoom_range = .2)
	generator = datagen.flow_from_directory(data_dir, target_size =(244,244), class_mode = 'binary', batch_size = 128, classes = ['non_autistic' , 'autistic'])
	return generator
