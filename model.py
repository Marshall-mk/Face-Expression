import tensorflow as tf
from tensorflow.keras import layers, Model


class FEModel:
    def __init__(self, model_name="FaceExpressionModel"):
        super(FEModel, self).__init__()
        self.base_model = tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.1),layers.RandomZoom(0.2),])
        self.model = None
        self.build()

    def build(self):
        """ Builds the Keras model based """
        inputs = tf.keras.Input(shape=[48, 48, 1]) 
        x = self.base_model(inputs)
        x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(7, activation="softmax")(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        #print(self.model.summary())
        
    def compile(self, optimizer, loss, metrics):
        """ Compiles the model """
        return self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics,
                           weighted_metrics=None,
                           run_eagerly=None,
                           steps_per_execution=None,
                           jit_compile=None)
    
    def train(self, train_dataset, batch_size=None, epochs=1,  validation_data=None, callbacks=None):
        """ Trains the model """
        return self.model.fit(train_dataset, batch_size=batch_size, epochs=epochs,  validation_data=validation_data, callbacks=callbacks)
    
    def evaluate(self, test_dataset):
        """Evaluates the model"""
        return self.model.evaluate(test_dataset)
    def save(self, file_path):
        """Saves the model"""
        self.model.save(file_path)
if __name__ == '__main__':
    model = FEModel()