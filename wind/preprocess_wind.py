import numpy as np
import os

print("--- 開始讀取原始風速資料 ---")
# 1. 載入你原本 (17484, 2, 96, 96) 的檔案
old_wind_path = 'data/era5_wind_tensor_u10v10_2022_96x96.npz'
data = np.load(old_wind_path)
wind_uv = data['wind'] 

print(f"原始維度: {wind_uv.shape}")

# 2. 計算風速大小 (sqrt(u^2 + v^2))
print("正在計算風速大小...")
u = wind_uv[:, 0, :, :]
v = wind_uv[:, 1, :, :]
wind_speed = np.sqrt(u**2 + v**2)

print(f"處理後維度: {wind_speed.shape}")

# 3. 儲存成新的檔案
new_wind_path = 'data/era5_wind_tensor_96x96_.npz'
np.savez(new_wind_path, wind=wind_speed)

print(f"✅ 成功！新檔案已儲存至: {new_wind_path}")
