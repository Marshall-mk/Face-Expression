from data import DataLoader # the data
from model import FEModel # the model
import os
import hydra  # for configurations
import tf2onnx # model conversion
import datetime # for date
import tensorflow as tf
from omegaconf.omegaconf import OmegaConf # configs
import matplotlib.pyplot as plt # plots
import mlflow.keras # for tracking
# import wandb
# from wandb.keras import WandbCallback



MLFLOW_TRACKING_URI="https://dagshub.com/Marshall-mk/FaceExpression.mlfow"
# # MLFLOW_TRACKING_USERNAME= "Marshall-mk"
# # MLFLOW_TRACKING_PASSWORD= "293464782615cda17c4724e11052f9db2a1bfb07"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "Marshall-mk"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "293464782615cda17c4724e11052f9db2a1bfb07"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.tensorflow.autolog()


@hydra.main(config_path="./configs", config_name="configs")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    """defines the data and the model"""
    
    fe_data = DataLoader()
    fe_model = FEModel()

    """Compiles and trains the model"""
    fe_model.compile(optimizer= cfg.train.optimizer, loss = cfg.train.loss,  metrics= cfg.train.metrics)

    """Model callbacks"""
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.model.ckpt_path, save_weights_only=True, save_best_only=True)
    
    """Trains the model"""
    # wandb.init(project="my-test-project", entity="marshallhamzah")
    # wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}
    with mlflow.start_run():
    # ... Define a model
        model_info = fe_model.train(
            fe_data.load_train_data(cfg.model.Train_path),
            batch_size=cfg.train.batch_size,
            epochs=cfg.train.epochs,
            validation_data=fe_data.load_val_data(cfg.model.Train_path),
            callbacks= [earlystopping, checkpointer]) # , WandbCallback()

        """Evaluates the model on the test set"""
        print(f'Model evaluation metrics: {fe_model.evaluate(fe_data.load_test_data(cfg.model.Test_path))}')
        mlflow.end_run()
    
    """Saving the model"""

    fe_model.save(cfg.model.save_path)

    # """converting the model to onnx"""
    # spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    # output_path = cfg.model.onnx_path
    # model_proto, _ = tf2onnx.convert.from_keras(fe_model, input_signature=spec, opset=13, output_path=output_path)
    
    """Model training history """
    _model_history(model_info=model_info, cfg=cfg)






def _model_history(model_info, cfg):
    accuracy = model_info.history["accuracy"]
    val_accuracy = model_info.history["val_accuracy"]
    loss = model_info.history["loss"]
    val_loss = model_info.history["val_loss"]
    epochs = range(1, len(accuracy) + 1)
    plt.figure(figsize=(20,10))
    plt.plot(epochs, accuracy, "g-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.grid()
    plt.savefig(f'{cfg.model.history_path}accuracy.png', dpi=300, bbox_inches='tight')

    plt.legend()

    plt.figure(figsize=(20,10))
    plt.plot(epochs, loss, "g-", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid()
    plt.savefig(f'{cfg.model.history_path}loss.png', bbox_inches='tight', dpi=300)
if __name__ == "__main__":
    main()
