# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:01:49 2025

@author: user
"""
import pandas as pd
import os
from datetime import datetime, timedelta

# 🔹 設定檔案路徑
input_file = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\20150101-20151231.txt"
output_folder = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\pre\20150101_1231test"
# 🔹 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

# 🔹 定義起始時間
start_time = datetime(2015, 1, 1, 0, 0, 0)  # 從 2022-05-10 00:00:00 開始

# 🔹 每小時的行數
rows_per_hour = 103041#原本77361

# 🔹 讀取 TXT 檔案，逐小時拆分
chunk_iter = pd.read_csv(input_file, delim_whitespace=True, header=None, chunksize=rows_per_hour)

for hour, chunk in enumerate(chunk_iter):
    # 計算當前時間
    current_time = start_time + timedelta(hours=hour)
    
    # 生成檔案名稱，例如：20220510_0000.csv
    filename = current_time.strftime("%Y%m%d_%H%M") + ".csv"
    output_path = os.path.join(output_folder, filename)
    
    # 🔹 存成 CSV
    chunk.to_csv(output_path, index=False, header=False)

    print(f"✅ 已存檔：{output_path}")

print("🎉 所有小時的 CSV 檔案處理完成！")
