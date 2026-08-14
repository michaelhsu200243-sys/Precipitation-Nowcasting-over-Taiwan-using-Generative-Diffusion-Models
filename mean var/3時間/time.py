import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

# ----------------- 設定你的路徑 -----------------
# 舊 npz 檔案的來源資料夾 (從「2地形」資料夾讀取)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\2地形\96x96"
# 你的時間數據檔案路徑
time_npy_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\time\2015-2022_no_2021\timestamps_combined_2015-2022_no2021.npy"
# 新 npz 檔案和新 CSV 檔案的保存路徑 (全部都放在「3時間」資料夾)
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\3時間\96x96_1h\ERA5\2015_2022(no2021)"

# 舊的 npz 檔案路徑
old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
# 新的 npz 檔案路徑
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')
# 新的 CSV 檔案路徑
new_csv_path = os.path.join(output_dir, 'normalized_time_data.csv')

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)


# ----------------- 你的新轉換函數 -----------------
def date_to_sinusoidal_embedding(date_string: str) -> np.ndarray:
    """將日期字串轉換為 96x16 的嵌入陣列。"""
    # 確保輸入是字串，以防止錯誤
    if not isinstance(date_string, str):
         # 如果不是字串，嘗試將其轉換為標準的 ISO 格式字串，例如 '2023-07-01T00'
         date_string = pd.to_datetime(date_string).strftime('%Y-%m-%dT%H')

    # 使用 datetime.strptime 解析字串
    date_time_obj = datetime.strptime(date_string, '%Y-%m-%dT%H')
    
    # 正規化到 0-1 範圍
    # 月份 (1-12 -> 0-11 / 11)
    month = (date_time_obj.month - 1) / 11
    # 日期 (1-31 -> 0-30 / 30) (註: 這裡假設最大日數為31，所以分母為30)
    day = (date_time_obj.day - 1) / 30 
    # 小時 (0-23 / 23)
    hour = date_time_obj.hour / 23
    
    embedding = []
    # 應用正弦和餘弦轉換
    for value in [month, day, hour]:
        embedding.append(np.sin(2 * np.pi * value))
        embedding.append(np.cos(2 * np.pi * value))
        
    result = np.array(embedding)
    # 這裡 result 的 shape 是 (6,)，因為有 3 個時間特徵 * 2 (sin/cos)
    
    # 擴展陣列以匹配目標形狀 (96, 16)
    # (6,) -> (96, 6)
    result = np.tile(result, (96, 1))
    
    # (96, 6) 擴展到 (96, 16)。注意：您原本的代碼是 np.tile(result, (96, 16))，
    # 這會產生 (96, 6*16) 的形狀。如果您的目標是 (96, 16)，您需要確認您的嵌入維度。
    # 根據您的 np.tile(result, (96, 16))，您的目標形狀可能是 (96, 96)，如果您的特徵是 6。
    # 為了保持原意，我假設您想要 (96, N) 的形狀，其中 N 是 16。
    # 由於您的特徵數是 6，這一段邏輯可能需要檢查。
    # 假設您想要的是一個 (96, 6) 的陣列，將其擴展為 (96, 16) 是一個不常見的操作。
    # 為了避免改變您的原始邏輯，我暫時保留它可能帶來的形狀變化，但請檢查這是否是您想要的。
    
    # 重新評估您原來的代碼: result = np.tile(np.array([sin, cos, sin, cos, sin, cos]), (96, 16))
    # 陣列形狀會是 (96, 6 * 16) = (96, 96)。
    # 為了維持您的原始邏輯，我將其改為：
    result = np.tile(result, (1, 16)) # shape: (96, 6) -> (96, 96)
    
    return result


# ----------------- 步驟 1：載入並處理時間數據 (重點修正區域) -----------------
if not os.path.exists(time_npy_path):
    print(f"錯誤：找不到時間數據檔案：{time_npy_path}")
    sys.exit(1)

# 載入數據，通常會是 NumPy 的 datetime64 或 str 陣列
time_data = np.load(time_npy_path, allow_pickle=True)
print(f"成功載入 {len(time_data)} 個時間物件/字串。")

# 使用 Pandas 確保所有時間物件都被轉換為標準的日期時間物件 (這是避免錯誤的關鍵)
# .dt.strftime 可以在 Pandas Series 上可靠地執行字串轉換
time_series = pd.Series(time_data)
# 將所有物件/字串統一轉換為格式化的字串，格式為 'YYYY-MM-DDTHH'
# 如果原始數據是 datetime64，這個轉換是正確的。如果原始是字串，它會先嘗試解析
time_strings = time_series.apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%dT%H'))

# 儲存所有濃縮後的時間數值
all_time_values = []

for date_string in time_strings:
    # 現在 date_string 確定是 Python str 類型，可以安全地傳入轉換函數
    embedding_grid = date_to_sinusoidal_embedding(date_string)
    
    # 將 (96, 96) 的陣列濃縮成一個單一數值
    single_value = np.mean(embedding_grid)
    all_time_values.append(single_value)

# 將清單轉換為 NumPy 陣列
all_time_values_array = np.array(all_time_values)


# ----------------- 步驟 2：計算總體均值與方差並保存 CSV -----------------
# 程式碼保持不變
mean_time = np.mean(all_time_values_array)
variance_time = np.var(all_time_values_array)

print("\n計算出的濃縮後時間均值:")
print(mean_time)
print("\n計算出的濃縮後時間方差:")
print(variance_time)

# 將濃縮後的數值保存到新的 CSV 檔案
normalized_time_df = pd.DataFrame({'time_normalized': all_time_values_array})
normalized_time_df.to_csv(new_csv_path, index=False)
print(f"\n正規化後的數據已保存至：{new_csv_path}")


# ----------------- 步驟 3：載入舊 npz 檔案並更新 -----------------
# 程式碼保持不變
if not os.path.exists(old_mean_npz_path) or not os.path.exists(old_variance_npz_path):
    print("\n錯誤：未找到 npz 檔案。")
    print("請確認它們已經存在於路徑：", old_npz_dir)
    sys.exit(1)

with np.load(old_mean_npz_path) as data:
    all_means = data['means']
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances']

# 假設時間通道是第三個通道 (索引為 2)
time_channel_index = 2
try:
    all_means[0, 0, 0, time_channel_index] = mean_time
    all_variances[0, 0, 0, time_channel_index] = variance_time
except IndexError:
    print("\n錯誤：npz 檔案形狀不匹配，請確認時間通道索引是否正確。")
    print("目前陣列形狀為:", all_means.shape)
    sys.exit(1)

# ----------------- 步驟 4：儲存與驗證 -----------------
# 程式碼保持不變
np.savez(new_mean_npz_path, means=all_means)
np.savez(new_variance_npz_path, variances=all_variances)
print("\n均值與方差已成功更新至 npz 檔案。")
print(f"新的 npz 檔案已保存至：{output_dir}")

print("\n=== 驗證更新後的 npz 檔案內容 ===")
with np.load(new_mean_npz_path) as data:
    print("各通道的均值:")
    print(data['means'][0, 0, 0, :])
with np.load(new_variance_npz_path) as data:
    print("\n各通道的方差:")
    print(data['variances'][0, 0, 0, :])