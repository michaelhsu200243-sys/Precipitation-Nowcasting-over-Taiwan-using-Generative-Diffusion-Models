# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 18:29:48 2025

@author: user
Description: 整合風速數據讀取、2015-2022年份篩選(排除2021)、96x96插值、統計量計算與NPZ更新。
"""

import numpy as np
import pandas as pd
import os
import glob
import sys
from pathlib import Path
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ==============================================================================
# ----------------- 🎯 參數設定 (與您提供的一致) -----------------
# ==============================================================================

# 舊 npz 檔案的來源資料夾 (從「3時間」資料夾讀取)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\3時間\96x96_1h\ERA5\2015_2022(no2021)"

# 🎯 你的 U/V 風速合併 CSV 檔案所在的資料夾
data_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\wind\10uv(17-22)"

# 新 npz 和 CSV 檔案的保存路徑 (全部都放在「4uv風」資料夾)
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\4uv風\96x96\ERA5\2015_2022(no2021)"

# 舊的 npz 檔案路徑
old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
# 新的 npz 檔案路徑
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 🎯 統一的風速數據欄位名稱
WIND_COL_NAME = 'U10_Wind_m/s'
# 🎯 篩選的年份範圍 (已修改為 2022)
START_YEAR = 2015
END_YEAR = 2022

# 🎯 網格化設定
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
TIME_COL = 'Time'

lat_min, lat_max = 21.5, 25.5
lon_min, lon_max = 119.0, 123.0
target_shape = (96, 96) # H, W

# ==============================================================================
# ----------------- 輔助函數：插值到網格 (無改動) -----------------
# ==============================================================================

def interpolate_to_grid(df_time_slice, lon_range, lat_range, target_shape):
    """將 DataFrame 中的點資料插值到指定尺寸的網格上。"""
    new_h, new_w = target_shape
    points_orig = df_time_slice[['longitude', 'latitude']].values
    data_values = df_time_slice['value'].values
    
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    new_lons = np.linspace(lon_min, lon_max, new_w)
    new_lats = np.linspace(lat_min, lat_max, new_h)
    grid_lon, grid_lat = np.meshgrid(new_lons, new_lats)

    interpolated_data = griddata(
        points_orig,
        data_values.ravel(),
        (grid_lon, grid_lat),
        method='linear'
    )
    
    interpolated_data = np.nan_to_num(interpolated_data, nan=0.0)
    # sigma = 1.0 
    # smoothed_data = gaussian_filter(interpolated_data, sigma=sigma)
    
    return interpolated_data.flatten()


# ==============================================================================
# ----------------- 🎯 修正後的數據載入與處理函數 -----------------
# ==============================================================================

def load_filter_and_process_wind_data(data_dir, file_pattern, start_year, end_year):
    """
    讀取所有符合模式的 CSV 檔案並合併，篩選年份，對每個時間步執行 96x96 插值，
    並將所有插值後的數據展平為一個長陣列。
    """
    file_path_list = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not file_path_list:
        print(f"錯誤：在指定的資料夾中找不到符合 '{file_pattern}' 模式的 CSV 檔案。")
        sys.exit()
    
    # --- 步驟 1：讀取並合併所有 CSV 檔案 ---
    all_dfs = []
    print(f"✅ 正在讀取並合併 {len(file_path_list)} 個檔案 ({file_pattern})...")
    
    for file_path in file_path_list:
        try:
            temp_df = pd.read_csv(file_path, parse_dates=[TIME_COL])
            # 重命名欄位以配合 interpolate_to_grid 函數
            temp_df = temp_df.rename(columns={LAT_COL: 'latitude', LON_COL: 'longitude', WIND_COL_NAME: 'value'})
            temp_df = temp_df[['Time', 'latitude', 'longitude', 'value']]
            all_dfs.append(temp_df)
            print(f"  > 成功載入 {Path(file_path).name}")
        except Exception as e:
            print(f"  ❌ 讀取或重命名檔案 {Path(file_path).name} 時出錯：{e}，跳過。")
            continue
            
    if not all_dfs:
        print("錯誤：所有檔案載入失敗。")
        sys.exit()
        
    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined.sort_values(by='Time').reset_index(drop=True)
    
    # --- 步驟 2：篩選年份 (核心修改：2015-2022 且排除 2021) ---
    print(f"\n⏳ 開始根據 'Time' 欄位篩選 {start_year} 至 {end_year} 年的數據 (排除 2021)...")
    start_date = pd.to_datetime(f'{start_year}-01-01 00:00:00')
    end_date = pd.to_datetime(f'{end_year+1}-01-01 00:00:00')
    
    df_filtered = df_combined[
        (df_combined['Time'] >= start_date) & 
        (df_combined['Time'] < end_date) & 
        (df_combined['Time'].dt.year != 2021)
    ].copy()
    
    if df_filtered.empty:
        print(f"錯誤：篩選後沒有找到 {start_year}-{end_year} 年的數據。")
        sys.exit()
        
    all_timestamps = df_filtered['Time'].unique()
    print(f"✅ 篩選完成。找到 {len(all_timestamps)} 個時間步用於網格化。")
    print(f"數據範圍：{all_timestamps.min()} 至 {all_timestamps.max()}")
    
    # --- 步驟 3：網格化與展平 ---
    all_processed_data = []
    
    for i, dt in enumerate(all_timestamps):
        if i % 500 == 0:
            print(f"  處理進度: {i}/{len(all_timestamps)} (時間: {dt})")

        df_slice = df_filtered[df_filtered['Time'] == dt]
        
        # 執行插值、NaN 處理和高斯平滑
        processed_data_flat = interpolate_to_grid(
            df_slice, 
            (lon_min, lon_max), 
            (lat_min, lat_max), 
            target_shape
        )
        all_processed_data.append(processed_data_flat)

    return np.concatenate(all_processed_data).astype(np.float32)


# ==============================================================================
# ----------------- 🎯 主執行區塊 (無改動) -----------------
# ==============================================================================

# 讀取 u 和 v 數據 (現在包含了所有檔案的合併、篩選和 96x96 網格化)
u_data = load_filter_and_process_wind_data(data_dir, 'u10_extracted_*.csv', START_YEAR, END_YEAR)
v_data = load_filter_and_process_wind_data(data_dir, 'v10_extracted_*.csv', START_YEAR, END_YEAR)

print(f"\n✅ 所有 U 風速數據已成功整合 ({u_data.size} 個數值，已網格化)。")
print(f"✅ 所有 V 風速數據已成功整合 ({v_data.size} 個數值，已網格化)。")


# ----------------- 步驟 2：將風速數據保存為 CSV 檔案 (可選) -----------------
u_csv_path = os.path.join(output_dir, 'u_wind_processed_96x96_2015-2022_no2021.csv')
v_csv_path = os.path.join(output_dir, 'v_wind_processed_96x96_2015-2022_no2021.csv')

u_df = pd.DataFrame({'u10_processed_flat': u_data})
u_df.to_csv(u_csv_path, index=False)
print(f"\nU 風速展平數據已保存至：{u_csv_path}")

v_df = pd.DataFrame({'v10_processed_flat': v_data})
v_df.to_csv(v_csv_path, index=False)
print(f"\nV 風速展平數據已保存至：{v_csv_path}")


# ----------------- 步驟 3：正規化 (Normalization) 並計算統計量 -----------------
max_abs_u = np.max(np.abs(u_data))
max_abs_v = np.max(np.abs(v_data))
maxWtrain_global = max(max_abs_u, max_abs_v) 

u_data_norm = u_data / maxWtrain_global
v_data_norm = v_data / maxWtrain_global

mean_u = np.mean(u_data_norm)
variance_u = np.var(u_data_norm)
mean_v = np.mean(v_data_norm)
variance_v = np.var(v_data_norm)

print(f"\n計算出的風速正規化因子 (maxWtrain_global): {maxWtrain_global:.4f}")
print(f"U 數據正規化後範圍: [{np.min(u_data_norm):.4f}, {np.max(u_data_norm):.4f}]")
print(f"V 數據正規化後範圍: [{np.min(v_data_norm):.4f}, {np.max(v_data_norm):.4f}]")
print(f"\n計算出的 U 風速 (正規化後) 總體均值: {mean_u:.8f}")
print(f"計算出的 U 風速 (正規化後) 總體方差: {variance_u:.8f}")
print(f"\n計算出的 V 風速 (正規化後) 總體均值: {mean_v:.8f}")
print(f"計算出的 V 風速 (正規化後) 總體方差: {variance_v:.8f}")


# ----------------- 步驟 4：載入舊 npz 檔案並更新 -----------------
if not os.path.exists(old_mean_npz_path) or not os.path.exists(old_variance_npz_path):
    print("\n錯誤：未找到 npz 檔案。")
    sys.exit()

with np.load(old_mean_npz_path) as data:
    all_means = data['means']
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances']

# U 和 V 的通道索引 (U: 3, V: 4)
u_channel_index = 3
v_channel_index = 4
try:
    all_means[0, 0, 0, u_channel_index] = mean_u
    all_variances[0, 0, 0, u_channel_index] = variance_u
    all_means[0, 0, 0, v_channel_index] = mean_v
    all_variances[0, 0, 0, v_channel_index] = variance_v
except IndexError:
    print("\n錯誤：npz 檔案形狀不匹配，請確認風速通道索引是否正確。")
    print("目前陣列形狀為:", all_means.shape)
    sys.exit(1)


# ----------------- 步驟 5：儲存與驗證 -----------------
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)
print("\n均值與方差已成功更新至 npz 檔案。")

print("\n=== 驗證更新後的 npz 檔案內容 ===")
with np.load(new_mean_npz_path) as data:
    print("各通道的均值 (部分):")
    print(data['means'][0, 0, 0, :v_channel_index + 1])
with np.load(new_variance_npz_path) as data:
    print("\n各通道的方差 (部分):")
    print(data['variances'][0, 0, 0, :v_channel_index + 1])
print("--------------------------------------")
print(f"**U 風速統計量寫入通道 {u_channel_index}；V 風速統計量寫入通道 {v_channel_index}。**")