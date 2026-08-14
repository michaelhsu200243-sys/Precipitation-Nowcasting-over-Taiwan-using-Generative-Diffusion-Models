# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 2025
Modified: 2026-02-25
Description: 自動遍歷 2015-2022 年降雨資料夾，提取時間戳並合併，排除 2021 年。
修正重點：強化路徑匹配邏輯，解決 2022 年資料夾抓取錯誤的問題。
@author: user
"""

import pandas as pd
import numpy as np
import os
import glob
import sys
from pathlib import Path
from datetime import datetime
import re

# ==============================================================================
# 🎯 核心函數：從檔名中提取時間戳
# ==============================================================================

def extract_timestamps_from_folder(input_csv_path, year):
    """
    從指定資料夾下的 CSV 檔名中提取時間戳字串。
    """
    all_csv_files = glob.glob(os.path.join(input_csv_path, '*.csv'))
    all_csv_files.sort()  # 按檔案名排序，確保時間順序

    if not all_csv_files:
        return []

    timestamps_str = []
    time_pattern = r'(\d{8})_(\d{4})'
    
    total_files = len(all_csv_files)
    
    for i, filepath in enumerate(all_csv_files):
        filename = os.path.basename(filepath)
        match = re.search(time_pattern, filename)
        
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            datetime_str = date_str + time_str
            
            try:
                dt_object = datetime.strptime(datetime_str, '%Y%m%d%H%M')
                timestamps_str.append(dt_object.strftime('%Y-%m-%dT%H'))
                
                # 進度提示：每 2000 筆印一次，避免畫面太亂
                if i % 2000 == 0 and i > 0:
                    print(f"    ...已讀取 {i}/{total_files} 筆")
            
            except ValueError:
                pass

    return timestamps_str

# ==============================================================================
# 🎯 主程式：遍歷資料夾結構 (2015-2022, Skip 2021)
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. 路徑設定 ---
    ROOT_PRE_DIR = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre")
    OUTPUT_TIME_DIR = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\time\2015-2022_no_2021")
    
    START_YEAR = 2015
    END_YEAR = 2022
    SKIP_YEAR = 2021
    
    OUTPUT_TIME_DIR.mkdir(parents=True, exist_ok=True)
    
    all_combined_timestamps = []
    yearly_report = {} # 用於最後產出統計表
    
    print("="*60)
    print(f"🚀 啟動強化版數據提取 | 範圍: {START_YEAR} - {END_YEAR}")
    print(f"🚫 排除年份: {SKIP_YEAR}")
    print("="*60)

    # --- 2. 迴圈處理各年份 ---
    for year in range(START_YEAR, END_YEAR + 1):
        
        if year == SKIP_YEAR:
            print(f"\n⏩ [{year} 年] >>> 跳過 (設定為測試集)")
            continue
        
        print(f"\n📅 [{year} 年] >>> 正在定位正確資料夾...")
        
        # A. 找出所有年份開頭的候選目錄 (例如 2022*)
        candidate_folders = [f for f in os.listdir(ROOT_PRE_DIR) if f.startswith(str(year))]
        
        target_csv_dir = None
        
        # B. 雙重驗證：在候選目錄中尋找真正含有 deal.YYYY 的結構
        for folder_name in candidate_folders:
            current_path = ROOT_PRE_DIR / folder_name
            # 檢查是否有 deal.2022... 這種子目錄
            check_deal = glob.glob(str(current_path / f"deal.{year}*"))
            
            if check_deal:
                potential_csv_dir = Path(check_deal[0]) / "96x96"
                if potential_csv_dir.is_dir():
                    target_csv_dir = potential_csv_dir
                    print(f"  🎯 成功鎖定: {folder_name} -> {Path(check_deal[0]).name}")
                    break
        
        if target_csv_dir is None:
            print(f"  ❌ 錯誤：在 {year} 年相關目錄中找不到符合規格的 96x96 數據。")
            continue
        
        # C. 執行提取
        print(f"  📂 掃描路徑: {target_csv_dir}")
        year_ts = extract_timestamps_from_folder(target_csv_dir, year)
        
        if year_ts:
            print(f"  ✅ 提取成功: {len(year_ts)} 筆")
            all_combined_timestamps.extend(year_ts)
            yearly_report[year] = len(year_ts)
        else:
            print(f"  ⚠️ 警告: 該目錄下無有效 CSV 檔案。")

    # --- 3. 儲存與總結 ---
    if not all_combined_timestamps:
        print("\n❌ 失敗：未提取到任何數據，請檢查 D 槽路徑。")
        sys.exit()

    # 轉為 NumPy 陣列
    time_array = np.array(all_combined_timestamps, dtype='<U16')
    output_filename = f"timestamps_combined_{START_YEAR}-{END_YEAR}_no2021.npy"
    save_path = OUTPUT_TIME_DIR / output_filename
    np.save(save_path, time_array)

    print("\n" + "="*60)
    print(f"🎉 任務達成！數據已對齊")
    print("-" * 60)
    print(f"📊 年度統計明細：")
    for y, count in yearly_report.items():
        print(f"   {y} 年: {count:5d} 筆")
    print("-" * 60)
    print(f"💎 最終合併總數: {len(all_combined_timestamps)} 筆")
    print(f"📁 儲存位置: {save_path}")
    print(f"📅 涵蓋時段: {all_combined_timestamps[0]}  至  {all_combined_timestamps[-1]}")
    print("="*60)