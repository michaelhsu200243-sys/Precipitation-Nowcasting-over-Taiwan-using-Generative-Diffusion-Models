# -*- coding: utf-8 -*-
"""
Created on Tue Dec 9 15:55:19 2025
Modified: 移除高斯平滑步驟
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ==============================================================================
# === 參數設定 ===
# ==============================================================================

ERA5_base_folder = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\wind\10uv(17-22)")

U10_FILENAMES = ["u10_extracted_141516.csv", "u10_extracted_171819.csv", "u10_extracted_202122.csv"]
V10_FILENAMES = ["v10_extracted_141516.csv", "v10_extracted_171819.csv", "v10_extracted_202122.csv"]

output_base_dir = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\wind\96x96\NEW_ERA5")
output_npy = output_base_dir / "era5_wind_tensor_u10v10_96x96.npz"
output_png = output_base_dir / "era5_wind_tensor_u10v10_96x96_visualization.png"

output_u_csv_folder = output_base_dir / "u_csv_processed"
output_v_csv_folder = output_base_dir / "v_csv_processed"

output_base_dir.mkdir(parents=True, exist_ok=True)
output_u_csv_folder.mkdir(parents=True, exist_ok=True)
output_v_csv_folder.mkdir(parents=True, exist_ok=True)

lat_min, lat_max = 21.5, 25.5
lon_min, lon_max = 119.0, 123.0
target_shape = (96, 96)
target_time_steps = 60853

TIME_COL = 'Time'
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
U10_COL = 'U10_Wind_m/s'
V10_COL = 'U10_Wind_m/s' 

# ==============================================================================
# === 輔助函數 ===
# ==============================================================================

def load_and_combine_data(base_folder, filenames, component_col):
    all_data = []
    print(f"正在讀取 {component_col} 檔案...")
    for filename in filenames:
        file_path = base_folder / filename
        if not file_path.exists():
            print(f"警告：檔案 {filename} 不存在，跳過。")
            continue
        try:
            df = pd.read_csv(file_path, parse_dates=[TIME_COL])
            df = df.rename(columns={LAT_COL: 'latitude', LON_COL: 'longitude', component_col: 'value'})
            df = df[['Time', 'latitude', 'longitude', 'value']]
            all_data.append(df)
            print(f"  ✅ 成功載入 {filename}")
        except Exception as e:
            print(f"  ❌ 載入或處理 {filename} 失敗: {e}")
            continue

    if not all_data:
        raise ValueError(f"未能載入任何 {component_col} 資料。")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values(by='Time').reset_index(drop=True)
    
    start_date_2015 = pd.to_datetime('2015-01-01 00:00:00')
    end_date_2022 = pd.to_datetime('2022-01-01 00:00:00')
    
    initial_count = len(combined_df)
    combined_df = combined_df[
        (combined_df['Time'] >= start_date_2015) & 
        (combined_df['Time'] < end_date_2022)
    ]
    
    removed_count = initial_count - len(combined_df)
    if removed_count > 0:
        print(f"  ⚠️ 已移除 {removed_count} 筆 2014 年及 2022 年的數據。")
    
    return combined_df

def interpolate_to_grid(df_time_slice, lon_range, lat_range, target_shape):
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
    return interpolated_data, grid_lon, grid_lat

# ==============================================================================
# === 主要處理流程 (已移除平滑步驟) ===
# ==============================================================================

print("🚀 開始處理 ERA5 風場數據 (僅網格化，不進行平滑) ...")

try:
    u_combined_df = load_and_combine_data(ERA5_base_folder, U10_FILENAMES, U10_COL)
    v_combined_df = load_and_combine_data(ERA5_base_folder, V10_FILENAMES, V10_COL) 

    u_timestamps = u_combined_df['Time'].unique()
    v_timestamps = v_combined_df['Time'].unique()
    common_timestamps = sorted(list(set(u_timestamps).intersection(set(v_timestamps))))

    if not common_timestamps:
        raise ValueError("無共同時間戳記")

    print(f"\n找到 {len(common_timestamps)} 個共同的時間步長")

    wind_list = []
    timestamps_list = []
    
    for i, dt in enumerate(common_timestamps):
        if i % 500 == 0:
            print(f"  處理進度: {i}/{len(common_timestamps)}")

        u_slice = u_combined_df[u_combined_df['Time'] == dt]
        v_slice = v_combined_df[v_combined_df['Time'] == dt]

        if u_slice.empty or v_slice.empty:
            continue 

        # 執行網格化
        u_grid, grid_lon, grid_lat = interpolate_to_grid(u_slice, (lon_min, lon_max), (lat_min, lat_max), target_shape)
        v_grid, _, _ = interpolate_to_grid(v_slice, (lon_min, lon_max), (lat_min, lat_max), target_shape)

        # NaN 處理
        u_grid = np.nan_to_num(u_grid, nan=0.0)
        v_grid = np.nan_to_num(v_grid, nan=0.0)
        
        # --- 原本的高斯平滑步驟已移除 ---

        # 儲存 CSV 範例 (t0)
        if i == 0:
            grid_df = pd.DataFrame({'longitude': grid_lon.flatten(), 'latitude': grid_lat.flatten()})
            
            u_df_save = grid_df.copy()
            u_df_save['u10'] = u_grid.flatten() # 使用原始網格數據
            u_df_save.to_csv(output_u_csv_folder / f"{dt.strftime('%Y%m%d%H%M')}_u10_raw_96x96.csv", index=False)

            v_df_save = grid_df.copy()
            v_df_save['v10'] = v_grid.flatten() # 使用原始網格數據
            v_df_save.to_csv(output_v_csv_folder / f"{dt.strftime('%Y%m%d%H%M')}_v10_raw_96x96.csv", index=False)

        # 堆疊原始數據 (2, 96, 96)
        wind_data = np.stack([u_grid, v_grid], axis=0).astype(np.float32)
        wind_list.append(wind_data)
        timestamps_list.append(dt)

    # 數據堆疊與長度補齊
    if wind_list:
        wind_array = np.stack(wind_list, axis=0)
        timestamps_array = np.array(timestamps_list, dtype=object)

        current_steps = wind_array.shape[0]
        if current_steps < target_time_steps:
            fill_count = target_time_steps - current_steps
            wind_fill = np.tile(wind_array[-1][np.newaxis, ...], (fill_count, 1, 1, 1))
            last_ts = timestamps_array[-1]
            ts_fill = np.array([last_ts + timedelta(hours=j) for j in range(1, fill_count + 1)], dtype=object)
            wind_array = np.concatenate((wind_array, wind_fill), axis=0)
            timestamps_array = np.concatenate((timestamps_array, ts_fill), axis=0)
        elif current_steps > target_time_steps:
            wind_array = wind_array[:target_time_steps]
            timestamps_array = timestamps_array[:target_time_steps]

        # 儲存結果
        np.savez(output_npy, wind=wind_array, timestamps=timestamps_array)
        print(f"📎 已儲存原始網格數據至：{output_npy}")
        
        # 可視化
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(grid_lon, grid_lat, wind_array[0, 0], cmap="coolwarm")
        plt.title("u10 Raw Grid")
        plt.subplot(1, 2, 2)
        plt.pcolormesh(grid_lon, grid_lat, wind_array[0, 1], cmap="coolwarm")
        plt.title("v10 Raw Grid")
        plt.savefig(output_png)
        plt.show()

except Exception as e:
    print(f"\n❌ 錯誤: {e}")