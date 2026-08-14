# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 2025

@author: user
"""

import numpy as np
import pandas as pd
import os
import glob
import sys

# ==============================================================================
# ----------------- 🎯 參數設定 (請檢查路徑) -----------------
# ==============================================================================

# 舊 npz 檔案資料夾 (此時應包含通道 0 海陸遮罩 & 通道 1 海平面氣壓 MSL 的統計量)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\1海陸\96x96\ERA5" 
# 原始 CSV 資料夾 (包含處理好的 850 hPa Geopotential 網格化 CSVs)
data_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\msl" 
# 新 npz 檔案和新 CSV 檔案的保存路徑 (通道 1 的輸出路徑，覆蓋 MSL)
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\2地形\96x96\ERA5"

# 舊的 npz 檔案路徑
old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
# 新的 npz 檔案路徑
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')

# 🎯 實際欄位名稱 (Geopotential 850 hPa)
GPH_COL_NAME = 'Geopotential_850hPa_m^2/s^2'

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)


# ==============================================================================
# ----------------- 步驟 1：讀取並整合所有 Geopotential 數據 -----------------
# ==============================================================================
all_csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
if not all_csv_files:
    print("錯誤：在指定的資料夾中找不到任何 CSV 檔案。")
    sys.exit()

print(f"找到 {len(all_csv_files)} 個 CSV 檔案，開始讀取...")
all_gph850_data = [] # 🎯 變數名更新
for file_path in all_csv_files:
    try:
        df = pd.read_csv(file_path)
        # 🎯 關鍵修正：使用完整的欄位名稱來提取數據
        gph850_values = df[GPH_COL_NAME].values 
        all_gph850_data.append(gph850_values) # 🎯 變數名更新
    except Exception as e:
        print(f"讀取檔案 {file_path} 時出錯：{e}")
        continue
        
# 🎯 整合所有 Geopotential 數據
all_gph850_data = np.concatenate(all_gph850_data) 
print(f"所有 850 hPa Geopotential ({GPH_COL_NAME}) 數據已成功整合。")


# ==============================================================================
# ----------------- 步驟 2：計算原始數據的統計量並正規化 -----------------
# ==============================================================================
# 為了將數據縮放到 0~1 範圍，我們需要找到最小值和最大值
min_gph850 = np.min(all_gph850_data) # 🎯 變數名更新
max_gph850 = np.max(all_gph850_data) # 🎯 變數名更新

# 正規化到 0~1 範圍
normalized_gph850_data = (all_gph850_data - min_gph850) / (max_gph850 - min_gph850) # 🎯 變數名更新

# 將正規化後的數據保存為 CSV 檔案
normalized_df = pd.DataFrame({'GPH850_normalized': normalized_gph850_data}) 
normalized_csv_path = os.path.join(output_dir, 'normalized_gph850_data.csv') 
normalized_df.to_csv(normalized_csv_path, index=False)
print(f"\n正規化後的數據已保存至：{normalized_csv_path}")


# 計算正規化後數據的均值與方差
mean_gph850_norm = np.mean(normalized_gph850_data) # 🎯 變數名更新
variance_gph850_norm = np.var(normalized_gph850_data) # 🎯 變數名更新

print(f"\n正規化後 850hPa Geopotential 總體均值: {mean_gph850_norm}")
print(f"正規化後 850hPa Geopotential 總體方差: {variance_gph850_norm}")


# ==============================================================================
# ----------------- 步驟 3：載入舊檔案並更新 -----------------
# ==============================================================================
# 檢查舊檔案是否存在，如果不存在則會報錯
if not os.path.exists(old_mean_npz_path) or not os.path.exists(old_variance_npz_path):
    print("\n錯誤：未找到之前保存的 npz 檔案。")
    print(f"請確認它們已經存在於路徑：{old_npz_dir}")
    sys.exit()

with np.load(old_mean_npz_path) as data:
    all_means = data['means']
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances']

# 將計算出的正規化均值與方差添加到正確的通道位置
# 🎯 維持您設定的通道 1 (覆蓋 MSL 的統計量)
channel_index = 1
all_means[0, 0, 0, channel_index] = mean_gph850_norm
all_variances[0, 0, 0, channel_index] = variance_gph850_norm

print(f"⚠️ **注意：** Geopotential 統計量已寫入通道 {channel_index}，**取代了**該通道原有的統計量。")

# 儲存更新後的 npz 檔案到新路徑
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)

print(f"\n均值與方差已成功更新至通道 {channel_index} (Geopotential)，並已儲存到新資料夾。")
print("--------------------------------------")


# ==============================================================================
# ----------------- 步驟 4：驗證儲存的檔案內容 -----------------
# ==============================================================================
# 讀取並印出新的 mean_normalizer.npz 的內容
print("\n=== 驗證 mean_normalizer.npz 檔案內容 ===")
with np.load(new_mean_npz_path) as data:
    loaded_means = data['means']
    print(f"陣列形狀: {loaded_means.shape}")
    print("各通道的均值:")
    # 這裡現在應該會印出通道 0 (海陸), 通道 1 (Geopotential), 通道 2 (若有) 的值
    print(loaded_means[0, 0, 0, :]) 
    print("--------------------------------------")

# 讀取並印出新的 variance_normalizer.npz 的內容
print("\n=== 驗證 variance_normalizer.npz 檔案內容 ===")
with np.load(new_variance_npz_path) as data:
    loaded_variances = data['variances']
    print(f"陣列形狀: {loaded_variances.shape}")
    print("各通道的方差:")
    print(loaded_variances[0, 0, 0, :])
    print("--------------------------------------")
    
print(f"\n**請確認通道 0 (海陸) 的數值不變，且通道 {channel_index} (Geopotential) 的數值已被正確寫入 (0~1 範圍內)。**")