# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 03:07:07 2025

@author: user
Description: 處理多層次資料夾 (年資料夾/deal/96x96) 中的 CSV 降雨數據，
             僅篩選 2015-2021 年的數據，並將其合併為單個 NPZ 檔案。
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import re 
from pathlib import Path 

# ----------------------------------------------------------------------
# 函數 1: 遞迴尋找目標 CSV 檔案並進行時間篩選
# ----------------------------------------------------------------------
def find_all_target_csv_files(base_data_path):
    """
    遞迴尋找並返回所有在 *2015*-*2021*/deal.*/96x96/ 結構下的 CSV 檔案列表，
    並過濾只保留時間戳記在 2015-01-01 至 2021-12-31 之間的檔案。
    """
    base_data_path = Path(base_data_path)
    print(f"🔄 正在主資料夾中搜尋目標 CSV 檔案: {base_data_path}")
    
    # 設定時間篩選範圍 (用於CSV檔名過濾)
    START_TIME = datetime(2015, 1, 1, 0, 0)
    END_TIME = datetime(2022, 1, 1, 0, 0) # 不包含 2022 年
    
    all_target_csv_files = []
    
    # 1. 尋找所有潛在的目標資料夾: */deal.*/96x96/
    # glob 模式: [年資料夾]/deal.[日期]/96x96
    search_dirs = glob.glob(str(base_data_path / '*' / 'deal.*' / '96x96'))
    
    # 2. 針對找到的目標資料夾進行處理 (雙重過濾的第一層：年資料夾名稱)
    for target_dir_path in search_dirs:
        # 提取該資料夾的**父目錄的父目錄**，即年資料夾的名稱 (e.g., '20150101_1231test')
        year_folder_name = Path(target_dir_path).parent.parent.name
        
        # 檢查年資料夾名稱是否在 2015-2021 範圍內
        try:
            folder_year = int(year_folder_name[:4])
            if not (2015 <= folder_year <= 2021):
                continue
        except ValueError:
            # 如果資料夾名稱不是以數字開頭 (例如: 'temp' 或其他雜項資料夾)，則跳過
            continue
            
        # 3. 如果年資料夾通過篩選，則查找其中的 CSV 檔案
        current_csv_files = glob.glob(os.path.join(target_dir_path, '*.csv'))
        
        # 4. 對 CSV 檔案進行時間戳過濾 (雙重過濾的第二層：CSV 檔名時間戳)
        for filepath in current_csv_files:
            filename = os.path.basename(filepath)
            
            # 使用正規表達式匹配 'YYYYMMDD_HHMM' 格式
            match = re.search(r'(\d{8})_(\d{4})', filename)
            if match:
                time_str = match.group(1) + match.group(2)
                try:
                    current_time = datetime.strptime(time_str, '%Y%m%d%H%M')
                    
                    # 執行 CSV 檔名時間篩選
                    if START_TIME <= current_time < END_TIME:
                        all_target_csv_files.append(filepath)
                        
                except ValueError:
                    # 如果檔名匹配了格式但時間本身無效 (理論上不太可能發生在有效數據中)
                    continue
    
    all_target_csv_files.sort() # 排序以確保時間順序
    
    if not all_target_csv_files:
        print("❌ 警告: 未找到任何符合條件 (結構正確且時間在 2015-2021 之間) 的 CSV 檔案。")
    else:
        print(f"✅ 成功找到 {len(all_target_csv_files)} 個 CSV 檔案 (已過濾 2015-2021)。")
        print(f"   首檔: {Path(all_target_csv_files[0]).relative_to(base_data_path)}")
        print(f"   尾檔: {Path(all_target_csv_files[-1]).relative_to(base_data_path)}")
        
    return all_target_csv_files

# ----------------------------------------------------------------------
# 函數 2: 處理 CSV 列表並儲存為 NPZ (加入更強健的錯誤處理)
# ----------------------------------------------------------------------
def process_and_save_gridded_data(
    all_csv_files_to_process, 
    output_base_dir,
    image_size=96,
    target_lon_min=119.0,
    target_lon_max=123.0,
    target_lat_min=21.5,
    target_lat_max=25.5
):
    """
    處理已經網格化的降雨 CSV 檔案列表，將其轉換為 npz 格式並保存。
    """
    
    if not all_csv_files_to_process:
        print("沒有 CSV 檔案列表可供處理。")
        return None, None, None, None, None, None, None, None, None, None, None, None

    output_dir = os.path.join(output_base_dir, 'npy_npz')
    addons_dir = os.path.join(output_base_dir, 'npy_npz', 'addons')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(addons_dir, exist_ok=True)

    gridded_data_with_timestamps = []
    total_files = len(all_csv_files_to_process)
    files_skipped_count = 0 

    for i, filepath in enumerate(all_csv_files_to_process):
        filename = os.path.basename(filepath)

        # 1. 解析時間 (這裡的時間解析應該會成功，因為檔案已被 find_all_target_csv_files 篩選)
        current_time = None
        match = re.search(r'(\d{8})_(\d{4})', filename)
        
        if match:
            time_str = match.group(1) + match.group(2)
            try:
                current_time = datetime.strptime(time_str, '%Y%m%d%H%M')
            except ValueError:
                # 備用錯誤處理，以防萬一
                files_skipped_count += 1
                if files_skipped_count <= 5: 
                     print(f"⚠️ 警告 (第 {files_skipped_count} 次): 無法解析檔名時間格式 '{time_str}'，跳過 {filename}")
                continue
        else:
            # 檔名不含時間戳 (例如 'downscaled_20.csv')
            files_skipped_count += 1
            if files_skipped_count <= 5: 
                print(f"⚠️ 警告 (第 {files_skipped_count} 次): 檔名不含時間戳，跳過 {filename}")
            continue

        # 2. 讀取 CSV
        try:
            df = pd.read_csv(filepath)
            # 確保有 'apcp' 欄位，且數據大小正確 (96x96=9216行)
            if 'apcp' in df.columns and len(df) == image_size * image_size:
                gridded_image = df['apcp'].values.reshape((image_size, image_size))
                gridded_data_with_timestamps.append((current_time, gridded_image))
            else:
                files_skipped_count += 1
                if files_skipped_count <= 5:
                    print(f"⚠️ 警告 (第 {files_skipped_count} 次): {filename} 格式錯誤 (缺少 'apcp' 或行數不符)。跳過。")
                continue
        except Exception as e:
            files_skipped_count += 1
            if files_skipped_count <= 5:
                print(f"⚠️ 警告 (第 {files_skipped_count} 次): 讀取或處理檔案 {filename} 時發生錯誤: {e}，跳過。")
            continue

        if (i + 1) % 1000 == 0: 
            print(f"已處理 {i + 1}/{total_files} 個檔案 ({filename})。")

    if files_skipped_count > 0:
        print(f"\n📢 總共跳過 {files_skipped_count} 個不合規或錯誤的檔案。")
        
    if not gridded_data_with_timestamps:
        print("最終沒有數據可供保存。")
        return None, None, None, None, None, None, None, None, None, None, None, None

    # 3. 排序與堆疊
    gridded_data_with_timestamps.sort(key=lambda x: x[0])
    all_timestamps = np.array([item[0] for item in gridded_data_with_timestamps], dtype=object)
    all_gridded_data = np.stack([item[1] for item in gridded_data_with_timestamps], axis=0)

    print(f"\n所有時間步的數據已堆疊。總形狀: {all_gridded_data.shape}")

    # 4. 儲存 NPZ 和時間戳
    if all_timestamps.size == 0:
         print("❌ 錯誤: 時間戳陣列為空，無法確定檔案名範圍。")
         return None, None, None, None, None, None, None, None, None, None, None, None

    first_date_str = all_timestamps[0].strftime('%Y%m%d')
    last_date_str = all_timestamps[-1].strftime('%Y%m%d')
    output_data_filename = f"test_set_gridded_{first_date_str}-{last_date_str}_{image_size}x{image_size}.npz"
    output_timestamps_filename = f"timestamps_gridded_{first_date_str}-{last_date_str}.npy"

    np.savez(os.path.join(output_dir, output_data_filename), arr_0=all_gridded_data)
    np.save(os.path.join(output_dir, output_timestamps_filename), all_timestamps, allow_pickle=True)

    print(f"網格化降雨數據已保存到: {os.path.join(output_dir, output_data_filename)}")
    print(f"時間戳數據已保存到: {os.path.join(output_dir, output_timestamps_filename)}")

    # 5. 計算並儲存歸一化參數
    mean_normalizer = np.mean(all_gridded_data) if all_gridded_data.size > 0 else 0.0
    variance_normalizer = np.var(all_gridded_data) 
    if variance_normalizer == 0.0:
        variance_normalizer = 1.0 # 避免方差為零
    
    np.savez(os.path.join(addons_dir, 'mean_normalizer.npz'), arr_0=mean_normalizer)
    np.savez(os.path.join(addons_dir, 'variance_normalizer.npz'), arr_0=variance_normalizer)
    print(f"均值 normalizer ({mean_normalizer:.4f}) 已保存到: {os.path.join(addons_dir, 'mean_normalizer.npz')}")
    print(f"方差 normalizer ({variance_normalizer:.4f}) 已保存到: {os.path.join(addons_dir, 'variance_normalizer.npz')}")

    return (all_gridded_data, all_timestamps, mean_normalizer, variance_normalizer,
            output_dir, output_data_filename, output_timestamps_filename,
            target_lon_min, target_lon_max, target_lat_min, target_lat_max, image_size)


# ----------------------------------------------------------------------
# 函數 3: 可視化 NPZ 數據
# ----------------------------------------------------------------------
def visualize_precipitation_npz(
    npz_filepath,
    timestamps_filepath,
    output_save_dir,
    target_dates=None,
    lon_min=None, lon_max=None, lat_min=None, lat_max=None,
    image_size=None
):
    """
    載入 npz 數據並可視化為圖片。
    """
    if not os.path.exists(npz_filepath) or not os.path.exists(timestamps_filepath):
        print("錯誤: NPZ 或時間戳檔案不存在。")
        return

    save_images_dir = os.path.join(output_save_dir, 'visualizations')
    os.makedirs(save_images_dir, exist_ok=True)

    try:
        loaded_npz = np.load(npz_filepath)
        precipitation_data = loaded_npz['arr_0']
        loaded_timestamps = np.load(timestamps_filepath, allow_pickle=True)

        # 決定要畫哪些圖
        if target_dates is None:
            total_len = len(loaded_timestamps)
            indices_to_show = [0, total_len // 3, 2 * total_len // 3, total_len - 1] if total_len > 3 else list(range(total_len))
        else:
            if isinstance(target_dates, (tuple, list)) and len(target_dates) == 2 and all(isinstance(d, datetime) for d in target_dates):
                start_dt, end_dt = target_dates
                indices_to_show = [i for i, ts in enumerate(loaded_timestamps) if start_dt <= ts <= end_dt]
            else:
                indices_to_show = [i for i, ts in enumerate(loaded_timestamps) if ts in target_dates]

        print(f"將繪製以下時間索引影像：{indices_to_show}")
        
        image_extent = [lon_min, lon_max, lat_min, lat_max]
        max_precipitation = 350

        for idx in indices_to_show:
            img = precipitation_data[idx]
            timestamp = loaded_timestamps[idx]

            plt.figure(figsize=(6, 6))
            # 使用 pcolormesh 配合 extent 更適合地圖數據
            plt.pcolormesh(
                np.linspace(lon_min, lon_max, image_size + 1),
                np.linspace(lat_min, lat_max, image_size + 1),
                img,
                cmap='jet',
                vmin=0, vmax=max_precipitation,
                shading='auto' 
            )
            
            plt.colorbar(label='Precipitation (mm)')
            plt.title(f'Time: {timestamp.strftime("%Y-%m-%d %H:%M")}\nShape: {img.shape}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, color='white', linestyle='-', linewidth=0.5)
            plt.tight_layout()

            image_save_name = f"gridded_qpesums_{timestamp.strftime('%Y%m%d%H%M')}.png"
            plt.savefig(os.path.join(save_images_dir, image_save_name), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存: {os.path.join(save_images_dir, image_save_name)}")

        print("所有指定時間點的可視化圖片已完成。")

    except Exception as e:
        print(f"可視化過程中發生錯誤: {e}")


# ----------------------------------------------------------------------
# 主執行區塊
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- ❗ 設定輸入輸出資料路徑 (請務必修改此處) ---
    # base_data_path 應該指向包含所有 '20150101_1231test' 這樣年資料夾的那個父資料夾。
    # 這裡仍保留虛擬路徑，請替換成您的實際路徑：
    base_data_path = Path(r'D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre')
    
    # 輸出路徑：將所有結果統一存在這裡。
    output_base_directory = base_data_path / "Processed_2015_2021_GRAP" 
    
    target_image_size = 96
    lon_min_val, lon_max_val = 119.0, 123.0
    lat_min_val, lat_max_val = 21.5, 25.5

    print("======================================================")
    print("      🌧️ 降雨數據批次處理與 NPZ 轉換 (2015-2021) 🌧️     ")
    print("======================================================")
    print(f"讀取主路徑: {base_data_path}")
    print(f"輸出路徑: {output_base_directory}")
    print(f"目標尺寸: {target_image_size}x{target_image_size}")
    print("------------------------------------------------------")


    # --- 步驟 1: 尋找所有目標 CSV 檔案 (已包含 2015-2021 時間過濾) ---
    all_files_list = find_all_target_csv_files(str(base_data_path))

    # --- 步驟 2: 呼叫處理函數，一次性合併所有找到的檔案 ---
    print("\n--- 開始處理和合併所有找到的 CSV 檔案 ---")
    (processed_data, processed_timestamps, mean_val, var_val,
     output_dir, output_data_filename, output_timestamps_filename,
     returned_lon_min, returned_lon_max, returned_lat_min, returned_lat_max, returned_image_size
     ) = process_and_save_gridded_data(
         all_files_list, 
         output_base_dir=str(output_base_directory), 
         image_size=target_image_size,
         target_lon_min=lon_min_val,
         target_lon_max=lon_max_val,
         target_lat_min=lat_min_val,
         target_lat_max=lat_max_val
    )

    # --- 步驟 3: 驗證與可視化 ---
    if processed_data is not None:
        try:
            print("\n--- 驗證數據載入與可視化 ---")
            
            npz_file_path = os.path.join(output_dir, output_data_filename)
            timestamps_file_path = os.path.join(output_dir, output_timestamps_filename)

            # 載入儲存的檔案進行驗證
            loaded_npz = np.load(npz_file_path)
            loaded_data_array = loaded_npz['arr_0']
            loaded_timestamps = np.load(timestamps_file_path, allow_pickle=True)
            
            # 載入歸一化參數
            loaded_mean = np.load(os.path.join(output_base_directory, 'npy_npz', 'addons', 'mean_normalizer.npz'))['arr_0']
            loaded_variance = np.load(os.path.join(output_base_directory, 'npy_npz', 'addons', 'variance_normalizer.npz'))['arr_0']

            print(f"\n✅ 載入 NPZ 形狀: {loaded_data_array.shape}, Dtype: {loaded_data_array.dtype}")
            print(f"✅ 載入時間戳數量: {len(loaded_timestamps)}, 範圍從 {loaded_timestamps[0].strftime('%Y-%m-%d')} 到 {loaded_timestamps[-1].strftime('%Y-%m-%d')}")
            print(f"✅ 均值: {loaded_mean:.4f}, 方差: {loaded_variance:.4f}")

            # --- 範例: 可視化指定時間區間 ---
            # 選擇一個在 2015-2021 範圍內的日期進行繪圖
            target_start = datetime(2018, 7, 1, 10, 0) 
            target_end = datetime(2018, 7, 1, 15, 0)
            
            visualize_precipitation_npz(
                 npz_file_path,
                 timestamps_file_path,
                 output_save_dir=output_dir, 
                 target_dates=(target_start, target_end),
                 lon_min=returned_lon_min,
                 lon_max=returned_lon_max,
                 lat_min=returned_lat_min,
                 lat_max=returned_lat_max,
                 image_size=returned_image_size
            )

        except Exception as e:
            print(f"❌ 驗證或可視化過程發生錯誤: {e}")
    else:
        print("未成功生成任何數據，無法進行驗證和可視化。")