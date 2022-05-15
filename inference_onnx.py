import tensorflow as tf
import numpy as np
import onnxruntime as rt
from PIL import Image

class FEPredictor:
    def __init__(self, model_path):
        self.ort_session = rt.InferenceSession(model_path)


    def infer(self, image=None):
        input_name = self.ort_session.get_inputs()[0].name
        # label_name = self.ort_session.get_outputs()[0].name
        tensor_image = tf.convert_to_tensor(np.array(Image.open(image)), dtype=tf.float32)[np.newaxis, :]

        return self.ort_session.run(None, {input_name: tensor_image})[0]
        


if __name__ == "__main__":
    image = "image path"
    predictor = FEPredictor("models/1/emotionModel.onnx")
    print(predictor.infer(image))