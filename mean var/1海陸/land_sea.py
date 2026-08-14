import numpy as np
import pandas as pd
import os

# 定義你的輸入和輸出路徑
input_file_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\taiwan_land_sea_mask_96x96_no_rasterio.npz"
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\1海陸\96x96\ERA5"

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 1. 載入原始數據檔案
# 使用你提供的正確資料鍵 'mask'
try:
    with np.load(input_file_path) as data:
        land_sea_channel = data['mask']
except FileNotFoundError:
    print(f"錯誤：檔案路徑未找到，請檢查路徑是否正確：{input_file_path}")
    exit()
except KeyError:
    print("錯誤：npz 檔案中找不到名為 'mask' 的資料鍵。")
    print(f"檔案中包含的資料鍵有：{list(data.keys())}")
    exit()

# 2. 計算海陸遮罩通道的均值和方差
# 通道 0 是海陸遮罩，形狀為 (16, 16)
mean_val = np.mean(land_sea_channel)
variance_val = np.var(land_sea_channel)

print(f"海陸遮罩通道的均值: {mean_val}")
print(f"海陸遮罩通道的方差: {variance_val}")

# 3. 創建並儲存到 CSV 檔案
csv_path = os.path.join(output_dir, 'normalization_stats_land_sea_mask.csv')
df = pd.DataFrame({
    'Channel': [0],
    'Mean': [mean_val],
    'Variance': [variance_val]
})
df.to_csv(csv_path, index=False)
print(f"\n計算結果已儲存到 {csv_path} 檔案。")

# 4. 創建兩個完整的陣列來儲存所有通道的值
# 這一步確保你可以逐步添加其他通道的數據
num_channels = 16
all_means = np.full((1, 1, 1, num_channels), np.nan)
all_variances = np.full((1, 1, 1, num_channels), np.nan)

# 5. 將計算好的海陸遮罩值放到第 0 通道
all_means[0, 0, 0, 0] = mean_val
all_variances[0, 0, 0, 0] = variance_val

# 6. 儲存到新路徑的 npz 檔案
mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')

np.savez(mean_npz_path, means=all_means)
np.savez(variance_npz_path, variances=all_variances)

print(f"\nmean_normalizer.npz 和 variance_normalizer.npz 已為海陸遮罩創建並儲存到 {output_dir}。")