# -*- coding: utf-8 -*-
"""
更新版本：自動最大值正規化 (Max-Scaling)
功能：自動抓取降雨數據的全域最大值進行縮放，確保 Mean/Variance 回到有利於訓練的微小量級。
"""

import numpy as np
import os
import sys

# ----------------- 路徑設定 (維持原樣) -----------------
# 舊 npz 檔案來源 (包含風速處理後的基礎檔案)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\4uv風\96x96\ERA5\2015_2022(no2021)"
# 原始降雨量數據路徑
rainfall_npz_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\Processed_2015_2022_GRAP(no2021)\npy_npz\test_set_gridded_20150101-20221231_96x96.npz"
# 輸出目標資料夾
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\5雨量\96x96\ERA5\2015_2022(no2021)"

old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
new_mean_npz_path = os.path.join(output_dir, '0mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, '0variance_normalizer.npz')

os.makedirs(output_dir, exist_ok=True)

# ----------------- 步驟 1：載入並自動計算最大值縮放 -----------------
if not os.path.exists(rainfall_npz_path):
    print(f"錯誤：找不到降雨量檔案：{rainfall_npz_path}")
    sys.exit(1)

print(f"載入降雨量數據中...")
with np.load(rainfall_npz_path) as data:
    # 這裡假設你的數據存在 'arr_0'，如果不是請修改 key
    raw_data = data['arr_0'].astype(np.float32)

# 自動偵測最大值作為正規化基數 (maxRtrain)
maxRtrain = np.max(raw_data)
print(f"✅ 偵測到數據全域最大值: {maxRtrain:.4f}")

# 執行預縮放：將原始降雨量壓縮到 [0, 1]
rainfall_data = raw_data / maxRtrain
print(f"數據已完成 Max-Scaling，目前範圍: {np.min(rainfall_data)} ~ {np.max(rainfall_data)}")

# ----------------- 步驟 2：計算每個時間步的統計量 -----------------
sequence_length = 11
num_samples = rainfall_data.shape[0]
mean_by_timestep = []
variance_by_timestep = []

print(f"\n正在計算 {sequence_length} 個時間步的縮放後統計量...")

for t in range(sequence_length):
    # 使用切片選取所有序列的第 t 個影格
    # 例如：t=0 是輸入的第一幀，t=10 是預測的目標幀
    timestep_data = rainfall_data[t : num_samples - sequence_length + 1 + t]
    
    mean_by_timestep.append(np.mean(timestep_data))
    variance_by_timestep.append(np.var(timestep_data))

mean_by_timestep = np.array(mean_by_timestep)
variance_by_timestep = np.array(variance_by_timestep)

print("\n--- 統計結果確認 ---")
print(f"均值 (Mean): \n{mean_by_timestep}")
print(f"方差 (Variance): \n{variance_by_timestep}")

# ----------------- 步驟 3：覆蓋更新 Normalizer 通道 -----------------
if not os.path.exists(old_mean_npz_path):
    print(f"錯誤：找不到來源 Normalizer 檔案於 {old_npz_dir}")
    sys.exit(1)

with np.load(old_mean_npz_path) as data:
    all_means = data['means'].copy() 
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances'].copy()

# 更新通道 5 到 15 (這 11 個通道對應 11 幀降雨序列)
start_idx = 5
end_idx = start_idx + sequence_length

all_means[0, 0, 0, start_idx:end_idx] = mean_by_timestep
all_variances[0, 0, 0, start_idx:end_idx] = variance_by_timestep

# ----------------- 步驟 4：儲存檔案 -----------------
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)

print(f"\n--- 任務完成 ---")
print(f"1. 縮放基數 (maxRtrain): {maxRtrain}")
print(f"2. 新 Mean 檔案: {new_mean_npz_path}")
print(f"3. 新 Variance 檔案: {new_variance_npz_path}")
print(f"✅ 降雨通道均值已降至 {mean_by_timestep.mean():.6f} 左右，這與好訓練的版本特徵一致。")