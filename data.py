# -*- coding: utf-8 -*-
"""Data Loader"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator



class DataLoader():
    """Data Loader class"""
    def __init__(self):
        super().__init__()
    
    def load_train_data(self,path):
        """Loads dataset from path"""
        self.train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

        return self.train_datagen.flow_from_directory(
        path,
        subset='training',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
        
    
    def load_val_data(self,path):
        """Loads dataset from path"""
        self.train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
        return self.train_datagen.flow_from_directory(
        path,
        subset='validation',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
    
    
    def load_test_data(self,path):
        """Loads dataset from path"""
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        return self.test_datagen.flow_from_directory(
        path,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


if __name__ == "__main__":
    data_model = DataLoader()
    
