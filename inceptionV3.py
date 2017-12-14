import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

img = image.load_img("xxx.jpg", target_size=(299, 299))
input_image = image.img_to_array(img)
input_image /= 255.
input_image -= 0.5
input_image *= 2.
# Add a 4th dimension for batch size (Keras)
input_image = np.expand_dims(input_image, axis=0)

# Run the image through the NN
predictions = model.predict(input_image)

# Convert the predictions into text
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:-4}% confidence!".format(name, confidence * 100))