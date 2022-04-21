import tensorflow as tf
import numpy as np
import onnxruntime as ort

class FEPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.saved_path)

    def preprocess(self, image):
        image = tf.image.resize(image, (self.image_size, self.image_size))
        image = np.expand_dims(np.expand_dims(image, -1), 0)
        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image=None):
        tensor_image = tf.convert_to_tensor(image, dtype=tf.float32)
        tensor_image = self.preprocess(tensor_image)
        shape= tensor_image.shape
        tensor_image = tf.reshape(tensor_image,[1, shape[0],shape[1], shape[2]])
        print(tensor_image.shape)
        m = ort.InferenceSession(self.model_path)
        pred = m.run(None, {'input_1': tensor_image})
        pred = int(np.argmax(pred))
        return {'output':pred}
        


if __name__ == "__main__":
    image = "image path"
    predictor = FEPredictor("models/1/emotionModel.hdf5")
    predictor.preprocess(image)
    print(predictor.infer(image))