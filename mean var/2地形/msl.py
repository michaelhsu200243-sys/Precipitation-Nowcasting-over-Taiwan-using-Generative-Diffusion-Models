import numpy as np
import pandas as pd
import os
import glob
import sys

# ----------------- 設定你的路徑 -----------------
# 舊 npz 檔案資料夾 (海陸遮罩的 npz)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\1海陸\96x96\ERA5"
# 原始 CSV 資料夾
data_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\interpolated_csvs"
# 新 npz 檔案和新 CSV 檔案的保存路徑
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\2地形\96x96\ERA5"

# 舊的 npz 檔案路徑
old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
# 新的 npz 檔案路徑
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)


# ----------------- 步驟 1：讀取並整合所有 msl 數據 -----------------
all_csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
if not all_csv_files:
    print("錯誤：在指定的資料夾中找不到任何 CSV 檔案。")
    sys.exit()

print(f"找到 {len(all_csv_files)} 個 CSV 檔案，開始讀取...")
all_msl_data = []
for file_path in all_csv_files:
    try:
        df = pd.read_csv(file_path)
        msl_values = df['msl'].values
        all_msl_data.append(msl_values)
    except Exception as e:
        print(f"讀取檔案 {file_path} 時出錯：{e}")
        continue
all_msl_data = np.concatenate(all_msl_data)
print("所有 msl 數據已成功整合。")


# ----------------- 步驟 2：計算原始數據的統計量並正規化 -----------------
# 為了將數據縮放到 0~1 範圍，我們需要找到最小值和最大值
min_msl = np.min(all_msl_data)
max_msl = np.max(all_msl_data)

# 正規化到 0~1 範圍
normalized_msl_data = (all_msl_data - min_msl) / (max_msl - min_msl)

# 將正規化後的數據保存為 CSV 檔案
normalized_df = pd.DataFrame({'msl_normalized': normalized_msl_data})
normalized_csv_path = os.path.join(output_dir, 'normalized_msl_data.csv')
normalized_df.to_csv(normalized_csv_path, index=False)
print(f"\n正規化後的數據已保存至：{normalized_csv_path}")


# 計算正規化後數據的均值與方差
mean_msl_norm = np.mean(normalized_msl_data)
variance_msl_norm = np.var(normalized_msl_data)

print(f"\n正規化後 msl 總體均值: {mean_msl_norm}")
print(f"正規化後 msl 總體方差: {variance_msl_norm}")


# ----------------- 步驟 3：載入舊檔案並更新 -----------------
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
# 根據你的模型定義，通道 1 是地形/msl
channel_index = 1
all_means[0, 0, 0, channel_index] = mean_msl_norm
all_variances[0, 0, 0, channel_index] = variance_msl_norm

# 儲存更新後的 npz 檔案到新路徑
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)

print(f"\n均值與方差已成功更新至通道 {channel_index}，並已儲存到新資料夾。")
print("--------------------------------------")


# ----------------- 步驟 4：驗證儲存的檔案內容 -----------------
# 讀取並印出新的 mean_normalizer.npz 的內容
print("\n=== 驗證 mean_normalizer.npz 檔案內容 ===")
with np.load(new_mean_npz_path) as data:
    loaded_means = data['means']
    print(f"陣列形狀: {loaded_means.shape}")
    print("各通道的均值:")
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
    
print("\n**請確認通道 0 和通道 1 的數值已被正確填入，且都在 0~1 的範圍內。**")