import numpy as np 
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import io
import imageio

import models
import utils 


 
# generator that randomly samples data  
class DataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size=24, min_rainfall = 0.0, time =None, wind = None, addon = None) :
        self.data = data
        self.addon = addon
        self.time = time
        self.wind = wind
        self.sequence = 11
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.min_rainfall = min_rainfall # Percent of minimum rainfall per image
        
        # --- 僅修改此處：預先篩選索引 ---
        self.valid_indices = []
        for i in range(self.num_samples - self.sequence):
            items_tmp = self.data[i:i+self.sequence]
            if ((np.sum(items_tmp[8] != 0) / (96*96)) >= self.min_rainfall):
                self.valid_indices.append(i)
        self.valid_indices = np.array(self.valid_indices)
        np.concatenate([self.valid_indices])
        self.num_samples_filtered = len(self.valid_indices)
        self.num_batches = int(np.floor(self.num_samples_filtered / self.batch_size))
        np.random.shuffle(self.valid_indices)
        # ----------------------------
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
                        
        result = np.zeros((self.batch_size,96,96,self.sequence + 5))
        result[:,:,:,:2] = self.addon
        
        # --- 僅修改此處：按篩選後的索引順序抓取 ---
        batch_indices = self.valid_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        for i in range(self.batch_size):
            current_idx = batch_indices[i]
            items = self.data[current_idx:current_idx+self.sequence]

            items = np.expand_dims(items, axis=-1)
            items = np.swapaxes(items, 0, 3)

            result[i,:,:,2] = (utils.date_to_sinusoidal_embedding(self.time[current_idx]) + 1) / 2
            result[i,:,:,3:5] = np.transpose(self.wind[current_idx+6:current_idx+8],(1, 2, 0))
            result[i,:,:,5:] = items[:,:96,:96,:]
        # ----------------------------
        
        return result
    
    def on_epoch_end(self):
        np.random.shuffle(self.valid_indices)
    
#generator that returns all the sequences of data, from start to finish 
class FullDataGenerator(keras.utils.Sequence):
    def __init__(self, data, batch_size=24, min_rainfall = 0.0,time = None, wind = None, addon = None):
        self.data = data
        self.addon = addon
        self.wind = wind
        self.time = time
        self.counter = 0 
        self.sequence = 11
        self.batch_size = batch_size
        self.num_samples = data.shape[0]
        self.num_batches = int(np.floor((self.num_samples - self.sequence) / self.batch_size))
        self.min_rainfall = min_rainfall # Percent of minimum rainfall per image
        
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
                        
        result = np.zeros((self.batch_size,96,96,self.sequence + 5))
        result[:,:,:,:2] = self.addon
        
        for i in range(self.batch_size):
            
            while True:
                
                items = self.data[self.counter:self.counter+self.sequence]

                items = np.expand_dims(items, axis=-1)
                items = np.swapaxes(items, 0, 3)
                
                if ((np.sum(items[:,:,:,-3] != 0) / (96*96)) < self.min_rainfall):
                    self.counter = self.counter + 1
                else:
                    result[i,:,:,2] = (utils.date_to_sinusoidal_embedding(self.time[self.counter]) + 1)/2
                    result[i,:,:,3:5] = np.transpose(self.wind[self.counter+6:self.counter+8],(1, 2, 0))
                    result[i,:,:,5:] = items[:,:96,:96,:]
                    self.counter = self.counter + 1
                    break
        
        return result