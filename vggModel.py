def create_vgg_model():
	from keras.applications.vgg16 import VGG16
	from keras.models import Model
	from keras.layers import Dense
	from keras.layers import Flatten

	model = VGG16(include_top = False, input_shape = (244,244,3))
	for layer_idx in range(len(model.layers)):
		if layer_idx not in [1,2,3,15,16,17,18]:
			model.layers[layer_idx].trainable = False

	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(256, activation = 'relu')(flat1)
	output = Dense(95, activation = 'softmax')(class1)

	model = Model(inputs=model.inputs, outputs = output)


	return model
