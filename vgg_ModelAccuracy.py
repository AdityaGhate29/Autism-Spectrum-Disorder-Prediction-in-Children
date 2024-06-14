from vgg_model import vgghist
import matplotlib.pyplot as plt
import numpy as np


x = []

for i in range(1,101):
	x.append(i)

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.plot(x, vgghist.history['accuracy'], label = "VGG-16 Model Training", linestyle = '-')
plt.plot(x, vgghist.history['val_accuracy'], label = "VGG-16 Model Validation", linestyle = '-')
plt.legend()
plt.show()
