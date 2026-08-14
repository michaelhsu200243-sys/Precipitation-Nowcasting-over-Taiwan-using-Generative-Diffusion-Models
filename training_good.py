import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers
import io
import imageio

import models 
import generators 
import utils 
from setup import *

# 1. 讀取地形與海陸遮罩 (Addon Data)
addon = np.load("addons/landSeaMask.npy")
tmp = np.zeros((batch_size, 2 , 96 ,96))
for i in range(batch_size):
    tmp[i] = np.copy(addon)
addon = tmp 
addon = addon.transpose((0,2,3,1))

# 2. 讀取降雨與風場數據
train_dataset = np.load('data//test_set_gridded_20150101-20211230_96x96.npz')['arr_0']
test_dataset = np.load('data/test_set_gridded_20220101-20221231_96x96.npz')['arr_0']

train_wind_dataset = np.load('data/era5_wind_tensor_96x96.npz')['wind']
train_wind_dataset = train_wind_dataset[:,:96,:96]
test_wind_dataset = np.load('data/era5_wind_tensor_2022_96x96.npz')['wind']
test_wind_dataset = test_wind_dataset[:,:96,:96]

train_timestamps_dataset = np.load('data/timestamps_combined_2015-2021.npy',allow_pickle = True)
test_timestamps_dataset = np.load('data/timestamps_20220101-00-20221231-14.npy',allow_pickle = True)

# 3. 數據正規化 (Normalization)
maxRtrain = train_dataset.max()
maxRtesr = test_dataset.max()
train_dataset = train_dataset / maxRtrain
test_dataset = test_dataset / maxRtesr

maxWtrain = train_wind_dataset.max()
maxWtest = test_wind_dataset.max()
train_wind_dataset = train_wind_dataset / maxWtrain
test_wind_dataset = test_wind_dataset / maxWtest

# 4. 生成器定義 (Generator Definition)
train_generator50 = generators.DataGenerator(train_dataset, batch_size, 0.2, train_timestamps_dataset, train_wind_dataset, addon)
test_generator50 = generators.DataGenerator(test_dataset, batch_size, 0.2, test_timestamps_dataset, test_wind_dataset, addon)
test_generator20 = generators.DataGenerator(test_dataset, batch_size, 0.2, test_timestamps_dataset, test_wind_dataset, addon)
full_test_generator50 = generators.FullDataGenerator(test_dataset, batch_size, 0.2, test_timestamps_dataset, test_wind_dataset, addon)

# 5. 模型編譯 (Diffusion Model)
model = models.DiffusionModel(image_size, 13, 3, widths, block_depth)

optimizer = keras.optimizers.experimental.AdamW
model.compile(
    optimizer=optimizer(
        learning_rate=1e-5, weight_decay=1e-6
    ),
    loss=keras.losses.mean_absolute_error,
)

# 6. 核心修正：載入預算的 Normalizer 參數
# 先用一個 batch 進行 adapt 以初始化結構
model.normalizer.adapt(train_generator50.__getitem__(1))

# 載入權重並加入防爆保護
mean = np.load("addons/mean_normalizer_updated.npz")['means'].astype('float32')
variance = np.load("addons/variance_normalizer_updated.npz")['variances'].astype('float32')

# [關鍵修正]：確保方差不為零，防止還原時數值爆炸導致彩色雜點
variance = np.maximum(variance, 1e-4)

# [關鍵修正]：使用 set_weights 確保數值真正寫入 TensorFlow 變數
# [修正後的寫法]
# 1. 建立一個 count 變數（通常是一個純量 0）
count = np.array(10000, dtype='int64') # 假設已經看過 10000 個樣本

# 2. 傳入 3 個權重：Mean, Variance, Count
model.normalizer.set_weights([mean, variance, count])

model.test_generator50 = test_generator50
model.maxRtrain = maxRtrain

# 7. 回調函數 (Callbacks)
def saver(epoch, logs):
    model.network.save_weights("weights/"+str(epoch)+"diffusion_addons")
    model.ema_network.save_weights("weights/"+str(epoch)+"diffusion_addons_ema")

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='i_loss', factor=0.5, patience=2, min_lr=0
)

# 8. 開始訓練 (Fit)
history = model.fit(
    train_generator50,
    epochs=100,
    steps_per_epoch=len(train_generator50),
    batch_size=batch_size,
    callbacks=[
        reduce_lr,
        keras.callbacks.LambdaCallback(on_epoch_end=model.plotter),
        keras.callbacks.LambdaCallback(on_epoch_end=saver)
    ],
)
