!pip install tensorflow pillow > /dev/null 2>\&1



import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.mobilenet\_v2 import decode\_predictions, preprocess\_input

from tensorflow.keras.preprocessing import image

import numpy as np

import matplotlib.pyplot as plt



model = MobileNetV2(weights='imagenet')



from google.colab import files

uploaded = files.upload()



for fn in uploaded.keys():

&nbsp;   print(f"\\nProcessing image: {fn}")

&nbsp;   img = image.load\_img(fn, target\_size=(224, 224))

&nbsp;   img\_array = image.img\_to\_array(img)

&nbsp;   img\_array = np.expand\_dims(img\_array, axis=0)

&nbsp;   img\_array = preprocess\_input(img\_array)

&nbsp;   preds = model.predict(img\_array, verbose=0)

&nbsp;   decoded = decode\_predictions(preds, top=3)\[0]

&nbsp;   print("\\nPredicted Tags:")

&nbsp;   for i, (imagenet\_id, label, prob) in enumerate(decoded):

&nbsp;       print(f"{i+1}. {label} ({prob\*100:.2f}%)")

&nbsp;   plt.imshow(img)

&nbsp;   plt.axis('off')

&nbsp;   plt.show()

