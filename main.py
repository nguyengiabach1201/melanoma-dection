import sys
import numpy as np
import tensorflow as tf

def predict(model, img_path, class_names):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

model = tf.keras.models.load_model('C:/Users/Acer/Downloads/model1')

class_names = ['benign', 'malignant']

for i in range(1, n):
    predicted_class, confidence = predict(model,sys.argv[i],class_names)
    print(sys.argv[i], predicted_class, confidence)
