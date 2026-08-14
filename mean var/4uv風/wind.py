import numpy as np
import pandas as pd
import os
import glob
import sys

# ----------------- 設定你的路徑 -----------------
# 舊 npz 檔案的來源資料夾 (從「3時間」資料夾讀取)
old_npz_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\3時間\96x96_1h\ERA5"
# 你的 u 風速 CSV 資料夾
u_wind_data_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\GRAP_data\Michael_GRAP\96x96\2023測試用\u_csv"
# 你的 v 風速 CSV 資料夾
v_wind_data_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\GRAP_data\Michael_GRAP\96x96\2023測試用\v_csv"
# 新 npz 和 CSV 檔案的保存路徑 (全部都放在「4uv風」資料夾)
output_dir = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\new_mean_variances\4uv風\96x96\ERA5"

# 舊的 npz 檔案路徑
old_mean_npz_path = os.path.join(old_npz_dir, 'mean_normalizer.npz')
old_variance_npz_path = os.path.join(old_npz_dir, 'variance_normalizer.npz')
# 新的 npz 檔案路徑
new_mean_npz_path = os.path.join(output_dir, 'mean_normalizer.npz')
new_variance_npz_path = os.path.join(output_dir, 'variance_normalizer.npz')
# 新的 CSV 檔案路徑
u_csv_path = os.path.join(output_dir, 'u_wind_data.csv')
v_csv_path = os.path.join(output_dir, 'v_wind_data.csv')

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)


# ----------------- 步驟 1：讀取並整合 u 和 v 風速數據 -----------------
def get_all_wind_data(directory, column_name):
    all_csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not all_csv_files:
        print(f"錯誤：在指定的資料夾中找不到任何 {column_name} 的 CSV 檔案。")
        sys.exit()
        
    print(f"找到 {len(all_csv_files)} 個 {column_name} 的 CSV 檔案，開始讀取...")
    all_data = []
    for file_path in all_csv_files:
        try:
            df = pd.read_csv(file_path)
            # 假設 CSV 檔案格式正確，u10/v10 是唯一一列或關鍵列
            data_values = df[column_name].values.flatten()
            all_data.append(data_values)
        except Exception as e:
            print(f"讀取檔案 {file_path} 時出錯：{e}")
            continue
    # 這裡的 np.concatenate 確保所有數據被展平為一個長的一維陣列
    return np.concatenate(all_data).astype(np.float32)

# 讀取 u 和 v 數據
u_data = get_all_wind_data(u_wind_data_dir, 'u10')
v_data = get_all_wind_data(v_wind_data_dir, 'v10')

print("\n所有 U 風速數據已成功整合。")
print("所有 V 風速數據已成功整合。")


# ----------------- 步驟 2：將風速數據保存為 CSV 檔案 -----------------
# 保存 u 風速數據
u_df = pd.DataFrame({'u10': u_data})
u_df.to_csv(u_csv_path, index=False)
print(f"\nU 風速數據已保存至：{u_csv_path}")

# 保存 v 風速數據
v_df = pd.DataFrame({'v10': v_data})
v_df.to_csv(v_csv_path, index=False)
print(f"\nV 風速數據已保存至：{v_csv_path}")


# --- 關鍵修正從這裡開始 ---
# ----------------- 步驟 3：正規化 (Normalization) 並計算統計量 -----------------

# 找出 U 和 V 數據的全局絕對值最大值作為正規化因子 maxWtrain。
# 這樣可以確保 U 和 V 使用相同的縮放比例，並保留負值。
max_abs_u = np.max(np.abs(u_data))
max_abs_v = np.max(np.abs(v_data))

# 選擇 U/V 兩者中的最大絕對值，作為主訓練腳本中 maxWtrain 的等效值。
maxWtrain_global = max(max_abs_u, max_abs_v) 
print(f"\n計算出的風速全局正規化因子 (maxWtrain_global): {maxWtrain_global:.4f}")

# 步驟 3a：進行 [0, 1] 範圍的正規化 (Normalization)
# 由於風速有負值，這個縮放實際上會將數據映射到 [-X, 1] 範圍，X<1。
u_data_norm = u_data / maxWtrain_global
v_data_norm = v_data / maxWtrain_global

print(f"U 數據正規化後範圍: [{np.min(u_data_norm):.4f}, {np.max(u_data_norm):.4f}]")
print(f"V 數據正規化後範圍: [{np.min(v_data_norm):.4f}, {np.max(v_data_norm):.4f}]")


# 步驟 3b：計算正規化後數據的均值與方差 (模型需要的 Z-score 統計量)
mean_u = np.mean(u_data_norm)
variance_u = np.var(u_data_norm)

mean_v = np.mean(v_data_norm)
variance_v = np.var(v_data_norm)

print(f"\n計算出的 U 風速 (正規化後) 總體均值: {mean_u:.8f}")
print(f"計算出的 U 風速 (正規化後) 總體方差: {variance_u:.8f}")
print(f"\n計算出的 V 風速 (正規化後) 總體均值: {mean_v:.8f}")
print(f"計算出的 V 風速 (正規化後) 總體方差: {variance_v:.8f}")


# ----------------- 步驟 4：載入舊 npz 檔案並更新 -----------------
if not os.path.exists(old_mean_npz_path) or not os.path.exists(old_variance_npz_path):
    print("\n錯誤：未找到 npz 檔案。")
    print("請確認它們已經存在於路徑：", old_npz_dir)
    sys.exit()

with np.load(old_mean_npz_path) as data:
    all_means = data['means']
with np.load(old_variance_npz_path) as data:
    all_variances = data['variances']

# 將 U 和 V 的均值與方差更新到 npz 檔案的正確通道
# 假設 u 風速是第 3 個通道 (索引 3)，v 風速是第 4 個通道 (索引 4)
u_channel_index = 3
v_channel_index = 4
try:
    # 確保寫入的是正規化後的統計量 (小數)
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
print(f"新的 npz 檔案已保存至：{output_dir}")

print("\n=== 驗證更新後的 npz 檔案內容 ===")
with np.load(new_mean_npz_path) as data:
    print("各通道的均值 (部分):")
    # 顯示前後幾個關鍵通道的均值
    print(data['means'][0, 0, 0, :u_channel_index + 2]) 
with np.load(new_variance_npz_path) as data:
    print("\n各通道的方差 (部分):")
    print(data['variances'][0, 0, 0, :u_channel_index + 2])