# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 01:27:09 2025

@author: user
"""
import pandas as pd
import os
from datetime import datetime

# --- 設定路徑和篩選條件 ---
# ***重要：請將此路徑替換為您實際存放 .csv 檔案的資料夾路徑***
# 例如：input_data_folder = r"D:\QPESUMS\my_csv_files_2023"
input_folder = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\20150101_1231test" # <--- 請修改這裡，這應該是一個資料夾路徑！

# 輸出資料夾路徑（改為您指定的絕對路徑）
output_folder = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\20150101_1231test\deal.20150101-20151231"

# 設定要處理的日期範圍 (可選)
# 格式為 'YYYYMMDD'。如果不想設定起始/結束日期，請將其設為 None。
start_date_filter = "20150101" # 例如：從 2023 年 1 月 1 日開始處理
end_date_filter = "20151231"   # 例如：到 2023 年 1 月 31 日結束處理

# --- 主處理邏輯 ---
os.makedirs(output_folder, exist_ok=True)  # 確保輸出資料夾存在

print(f"--- 開始處理 CSV 檔案 ---")
print(f"輸入資料夾: {input_folder}")
print(f"輸出資料夾: {output_folder}")
print(f"日期篩選範圍: 從 {start_date_filter if start_date_filter else '無限制'} 到 {end_date_filter if end_date_filter else '無限制'}")

# 將日期篩選字串轉換為 datetime 對象，便於比較
start_date_obj = datetime.strptime(start_date_filter, "%Y%m%d") if start_date_filter else None
end_date_obj = datetime.strptime(end_date_filter, "%Y%m%d") if end_date_filter else None

processed_count = 0
skipped_count_date = 0
error_count = 0

# 遍歷輸入資料夾中的所有檔案
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        # 從文件名中提取日期部分，用於日期篩選
        try:
            # 假設文件名格式為 YYYYMMDD_HHMM.csv
            file_date_str = file_name.split("_")[0] # 提取 "YYYYMMDD" 部分
            current_file_date = datetime.strptime(file_date_str, "%Y%m%d")
        except (ValueError, IndexError):
            print(f"警告: 文件名 '{file_name}' 的日期格式不正確或無法解析 (預期 YYYYMMDD_HHMM.csv)，跳過此文件。")
            error_count += 1
            continue

        # 根據指定日期範圍進行篩選
        if (start_date_obj and current_file_date < start_date_obj) or \
           (end_date_obj and current_file_date > end_date_obj):
            print(f"資訊: 跳過文件 '{file_name}'，因為它不在指定的日期範圍內。")
            skipped_count_date += 1
            continue

        # 輸入文件的完整路徑
        input_file = os.path.join(input_folder, file_name)

        # 讀取 CSV 文件
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"錯誤: 讀取文件 {file_name} 時出錯：{e}")
            error_count += 1
            continue

        # 刪除前兩列（假定為 A 和 B）
        if df.shape[1] >= 2: # 確保至少有兩列可以刪除
            df = df.drop(df.columns[:2], axis=1)
        else:
            print(f"警告: 文件 '{file_name}' 只有 {df.shape[1]} 列，不足以刪除前兩列。跳過刪除操作。")


        # 從文件名中提取日期和時間部分，並轉換為 datetime 對象
        # 假設文件名格式為 YYYYMMDD_HHMM.csv
        date_time_str_from_filename = file_name.split(".")[0] # 去掉 .csv，得到 "YYYYMMDD_HHMM"
        try:
            current_datetime_obj = datetime.strptime(date_time_str_from_filename, "%Y%m%d_%H%M")
        except ValueError as e:
            print(f"錯誤: 文件名 {file_name} 的時間格式不正確 (預期 YYYYMMDD_HHMM.csv)，無法解析日期和時間：{e}")
            error_count += 1
            continue

        # 將時間列格式化為字符串（包含時:分:秒），並添加到 DataFrame 的第一列
        time_col_str = current_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        df.insert(0, "time", time_col_str)

        # 修改 C、D、E 列的頂端名稱
        # 假定刪除前兩列後，原始的 C, D, E 列現在是 DataFrame 中的索引 1, 2, 3
        # 即 'time' 列之後的三列。
        expected_new_cols = ["time", "longitude", "latitude", "apcp"]
        if len(df.columns) >= len(expected_new_cols):
            # 只替換前四個欄位名，其餘保持不變
            df.columns = expected_new_cols + df.columns[len(expected_new_cols):].tolist()
        else:
            print(f"警告: 文件 '{file_name}' 的欄位數量不足，無法完整重命名為 'time', 'longitude', 'latitude', 'apcp'。")
            # 僅重命名存在的欄位，如果它連3個都不到，就只重命名現有的
            df.columns = expected_new_cols[:len(df.columns)]


        # 輸出文件的完整路徑
        output_file = os.path.join(output_folder, file_name)

        # 保存處理後的數據
        df.to_csv(output_file, index=False)
        print(f"處理完成！文件已保存為 {output_file}")
        processed_count += 1

print(f"\n--- 檔案處理總結 ---")
print(f"總共處理了 {processed_count} 個檔案。")
print(f"跳過了 {skipped_count_date} 個檔案 (不在指定日期範圍內)。")
print(f"有 {error_count} 個檔案在讀取或解析時出錯。")
print(f"所有符合條件的檔案已處理完畢。")