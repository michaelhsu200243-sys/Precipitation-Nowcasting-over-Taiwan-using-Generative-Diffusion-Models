# -*- coding: utf-8 -*-
"""
更新版本：直接在原路徑覆蓋 Normalizer 檔案
優化：取消雙重正規化（不除以 maxRtrain），改用原始尺度計算均值與方差。
"""

import numpy as np
import pandas as pd
import os
import sys

# ----------------- 保持你的路徑完全不動 -----------------
# 舊 npz 檔案的來源資料夾
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\4uv風\96x96\ERA5\2015_2022(no2021)"
# 你的降雨量 npz 檔案路徑
rainfall_npz_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\Processed_2015_2022_GRAP(no2021)\npy_npz\test_set_gridded_20150101-20221231_96x96.npz"
# 輸出的目標資料夾 (維持原樣)
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\5雨量\96x96\ERA5\2015_2022(no2021)"

old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')
new_csv_path = os.path.join(output_dir, 'rainfall_data.csv')

os.makedirs(output_dir, exist_ok=True)

# ----------------- 步驟 1：載入數據 (原始尺度) -----------------
if not os.path.exists(rainfall_npz_path):
    print(f"錯誤：找不到降雨量 npz 檔案：{rainfall_npz_path}")
    sys.exit(1)

print(f"載入降雨量數據：{rainfall_npz_path}")
with np.load(rainfall_npz_path) as data:
    # 關鍵：直接用原始值算 Mean/Var，不要除以 maxRtrain
    rainfall_data = data['arr_0'].astype(np.float32)

print(f"成功載入，形狀：{rainfall_data.shape}")

# 保存原始數據 CSV
flattened_rainfall_data = rainfall_data.reshape(-1)
pd.DataFrame({'rainfall': flattened_rainfall_data}).to_csv(new_csv_path, index=False)
print(f"原始數據 CSV 已存至：{new_csv_path}")

# ----------------- 步驟 2：快速計算統計量 (移除雙重 for 迴圈) -----------------
sequence_length = 11
num_samples = rainfall_data.shape[0]
mean_by_timestep = []
variance_by_timestep = []

print(f"\n正在計算 {sequence_length} 個時間步的統計量 (原始尺度)...")

for t in range(sequence_length):
    # 使用 NumPy 切片直接選取所有序列的第 t 幀，速度快 100 倍
    timestep_data = rainfall_data[t : num_samples - sequence_length + 1 + t]
    
    mean_by_timestep.append(np.mean(timestep_data))
    variance_by_timestep.append(np.var(timestep_data))

mean_by_timestep = np.array(mean_by_timestep)
variance_by_timestep = np.array(variance_by_timestep)

print("\n計算結果：")
print(f"均值: {mean_by_timestep}")
print(f"方差: {variance_by_timestep}")

# ----------------- 步驟 3：更新並存檔 -----------------
if not os.path.exists(old_mean_npz_path) or not os.path.exists(old_variance_npz_path):
    print(f"\n錯誤：來源路徑找不到 npz：{old_npz_dir}")
    sys.exit(1)

with np.load(old_mean_npz_path) as data:
    all_means = data['means'].copy() 
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances'].copy()

start_idx = 5
end_idx = start_idx + sequence_length

# 更新通道 5 到 15 (共 11 個降雨通道)
all_means[0, 0, 0, start_idx:end_idx] = mean_by_timestep
all_variances[0, 0, 0, start_idx:end_idx] = variance_by_timestep

# 儲存
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)

print(f"\n✅ 任務完成！")
print(f"新 Normalizer 已儲存至：{output_dir}")