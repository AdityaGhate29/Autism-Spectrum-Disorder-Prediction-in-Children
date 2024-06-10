def image(bhya):

	import numpy as np
	from PIL import Image
	from skimage import transform
	img = Image.open(bhya)
	img = np.array(img).astype('float32')/255
	img = transform.resize(img, (244,244,3))
	img = np.expand_dims(img, axis = 0)
	return img
