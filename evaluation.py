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
import generators 
import utils 
from setup import *
import os
import pandas as pd

# 1. 修正 Addon 處理邏輯，確保符合 Generator 維度 (96, 96, 2)
addon_raw = np.load("addons/landSeaMask.npy")
# 轉置為 (96, 96, 2)
addon_fixed = addon_raw.transpose((1, 2, 0))
# 為了讓 batch 處理順利，我們保持與 batch_size 一致
addon = np.zeros((batch_size, 96, 96, 2))
for i in range(batch_size):
    addon[i] = addon_fixed

# load rain dataset 
train_dataset = np.load('data//test_set_gridded_20150101-20211230_96x96.npz')['arr_0']
test_dataset = np.load('data/test_set_gridded_20220101-20221231_96x96.npz')['arr_0']

# load wind dataset
train_wind_dataset = np.load('data/era5_wind_tensor_96x96.npz')['wind']
train_wind_dataset = train_wind_dataset[:,:96,:96]
test_wind_dataset = np.load('data/era5_wind_tensor_2022_96x96.npz')['wind']
test_wind_dataset = test_wind_dataset[:,:96,:96]

#load timestamp dataset 
train_timestamps_dataset = np.load('data/timestamps_combined_2015-2021.npy',allow_pickle = True)
test_timestamps_dataset = np.load('data/timestamps_20220101-00-20221231-14.npy',allow_pickle = True)

# === 日期控制切片 (僅在此處新增) ===
start_date = "2022-09-11T10" # 請根據你的 npy 格式修改字串
end_date   = "2022-09-13T20"
indices = np.where((test_timestamps_dataset >= start_date) & (test_timestamps_dataset <= end_date))[0]
if len(indices) > 0:
    test_dataset = test_dataset[indices[0]:indices[-1]+1]
    test_wind_dataset = test_wind_dataset[indices[0]:indices[-1]+1]
    test_timestamps_dataset = test_timestamps_dataset[indices[0]:indices[-1]+1]
    print(f"✅ 已切片日期範圍: {test_timestamps_dataset[0]} 至 {test_timestamps_dataset[-1]}")
# =================================

# normalization 
maxRtrain = train_dataset.max()
train_dataset = train_dataset / maxRtrain
test_dataset = test_dataset / maxRtrain

maxWtrain = train_wind_dataset.max()
maxWtest = test_wind_dataset.max()
train_wind_dataset = train_wind_dataset / maxWtrain
test_wind_dataset = test_wind_dataset / maxWtest

# 2. Generators：將 min_rainfall 設為 0.0 以獲取全年度連續數據
train_generator50 = generators.DataGenerator(train_dataset,batch_size,0.2,train_timestamps_dataset,train_wind_dataset, addon)
test_generator50 = generators.DataGenerator(test_dataset,batch_size,0.0,test_timestamps_dataset,test_wind_dataset, addon)
test_generator20 = generators.DataGenerator(test_dataset,batch_size,0.2,test_timestamps_dataset,test_wind_dataset, addon)
full_test_generator50 = generators.FullDataGenerator(test_dataset,batch_size,0.0,test_timestamps_dataset,test_wind_dataset, addon)

# diffusion model 
model = models.DiffusionModel(image_size, 13, 3, widths, block_depth)
optimizer=keras.optimizers.experimental.AdamW
model.compile(optimizer=optimizer(learning_rate=1e-5, weight_decay=1e-6), loss=keras.losses.mean_absolute_error)

# pre-calculated normalizer
model.normalizer.adapt(train_generator50.__getitem__(1))
mean = np.load("addons/mean_normalizer_updated.npz")['means'].astype('float32')
variance = np.load("addons/variance_normalizer_updated.npz")['variances'].astype('float32')
model.normalizer.mean = mean 
model.normalizer.variance = variance

# Load weights
checkpoint_prefix = "weights/99diffusion_addons" 
model.network.load_weights(checkpoint_prefix)
model.ema_network.load_weights(f"{checkpoint_prefix}_ema")

# --- 實驗函數保持原始邏輯不變 ---
def experiment(generator = test_generator50, n_iter=34):
    history = np.zeros((n_iter,3))
    raw_data = np.zeros((n_iter,batch_size,96,96,6))
    mses = np.zeros((3))
    for i in range(n_iter):
        sample = generator.__getitem__(i)
        hist = np.copy(sample)
        sample = model.normalizer(sample)
        tmp = model.generate2(np.copy(sample),50)
        hist = hist * maxRtrain
        tmp = tmp * maxRtrain
        raw_data[i,:,:,:,:3] = hist[:,:,:,-3:]
        raw_data[i,:,:,:,3:] = tmp[:,:,:,-3:]
        mse = np.mean(np.sum((hist[:,:,:,-3:]-tmp[:,:,:,-3:])**2,axis=(1,2)),axis=0)
        history[i] = mse 
        print(f"Iter {i}: {mse}")
        mses += mse[-3:]
    return mses / n_iter, history, raw_data

def experiment2(generator = test_generator50, n_iter=10, ensamble_iter = 15):
    mses = np.zeros((n_iter,3))
    raw_data = np.zeros((n_iter,batch_size,96,96,6))
    for i in range(n_iter):
        test = generator.__getitem__(i)
        print(f"Ensemble Iter {i}")
        res = np.zeros([batch_size,ensamble_iter, 96,96, 16])
        for j in range(ensamble_iter):
            sample = np.copy(test)
            sample = model.normalizer(sample)
            tmp = model.generate2(np.copy(sample),50)
            tmp = tmp * maxRtrain
            res[:,j] = tmp 
        average = np.mean(res,axis=1)
        hist = test * maxRtrain
        raw_data[i,:,:,:,:3] = np.copy(hist[:,:,:,-3:])
        raw_data[i,:,:,:,3:] = np.copy(average[:,:,:,-3:])
        mse = np.mean(np.sum((hist[:,:,:,-3:]-average[:,:,:,-3:])**2,axis=(1,2)),axis=0)
        mses[i] = mse 
        print(f"Ensemble Current MSE: {mse}")
    return mses, raw_data

# --- 3. 執行全年度實驗 ---
TOTAL_YEAR_ITERS = len(full_test_generator50)
thresh = 0.0001

print(f"🚀 開始全年度評估，總迭代次數: {TOTAL_YEAR_ITERS}")

# 重置 generator 狀態
full_test_generator50.counter = 0
exp_single, hist_single, raw_single = experiment(full_test_generator50, TOTAL_YEAR_ITERS)
single_metrics_val = utils.metrics_aggregator(raw_single, thresh).mean(axis=0)

# 重置 generator 狀態跑 Ensemble
full_test_generator50.counter = 0
res_ens, raw_ensemble = experiment2(full_test_generator50, TOTAL_YEAR_ITERS, 15)
ensemble_metrics_val = utils.metrics_aggregator(raw_ensemble, thresh).mean(axis=0)

# =========================================================
# 💾 保存數據 (保持你的目錄結構)
# =========================================================
base_dir = 'evaluation_final_archive'
for folder in [f'{base_dir}/single_diffusion', f'{base_dir}/ensemble_diffusion', f'{base_dir}/metadata']:
    os.makedirs(folder, exist_ok=True)

# 建議修正 evaluation.py
metric_columns = ['Accuracy', 'Precision', 'Recall', 'MSE'] # 依照你 tmp 的順序排列
time_indices = ['T+1', 'T+2', 'T+3']

# 合併 MSE 到指標矩陣中輸出
single_output = np.column_stack([single_metrics_val, exp_single])
ensemble_output = np.column_stack([ensemble_metrics_val, np.mean(res_ens, axis=0)])

np.save(f'{base_dir}/single_diffusion/raw_comparison_data.npy', raw_single)
pd.DataFrame(single_output, columns=metric_columns, index=time_indices).to_csv(f'{base_dir}/single_diffusion/meteorological_metrics.csv')

np.save(f'{base_dir}/ensemble_diffusion/raw_comparison_data.npy', raw_ensemble)
pd.DataFrame(ensemble_output, columns=metric_columns, index=time_indices).to_csv(f'{base_dir}/ensemble_diffusion/meteorological_metrics.csv')

with open(f'{base_dir}/metadata/experiment_config.txt', 'w') as f:
    f.write(f"Total Yearly Iterations: {TOTAL_YEAR_ITERS}\nStatus: Success\n")

print(f"\n✨ 任務完成！請將 {base_dir} 資料夾載回 Windows 繪圖。")