# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:07:39 2026

@author: user
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from scipy.ndimage import uniform_filter

def calculate_fss(y_true_bin, y_pred_bin, window_size=5):
    """
    計算 Fraction Skill Score (FSS)
    用於評估空間結構，解決高解析度模型常見的位移懲罰問題。
    """
    # 使用滑動平均計算鄰域內的降雨佔比
    f_true = uniform_filter(y_true_bin.astype(float), size=window_size)
    f_pred = uniform_filter(y_pred_bin.astype(float), size=window_size)
    
    mse_neigh = np.mean((f_true - f_pred)**2)
    ref_mse = np.mean(f_true**2) + np.mean(f_pred**2)
    
    return 1 - (mse_neigh / ref_mse) if ref_mse > 0 else 1.0

def calculate_all_metrics(y_true, y_pred, threshold):
    """
    計算單一時間步的所有指標。
    y_true/y_pred: 96x96 的矩陣
    """
    mse = np.mean((y_true - y_pred)**2)
    
    # 二值化（修正：改為 >= 包含邊界值）
    y_true_bin = (y_true >= threshold)
    y_pred_bin = (y_pred >= threshold)

    tp = np.sum(y_true_bin & y_pred_bin)
    tn = np.sum(~y_true_bin & ~y_pred_bin)
    fp = np.sum(~y_true_bin & y_pred_bin)
    fn = np.sum(y_true_bin & ~y_pred_bin)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    # HSS 計算 (Heidke Skill Score)
    num_hss = 2 * (tp * tn - fp * fn)
    den_hss = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = num_hss / den_hss if den_hss > 0 else 0

    # FSS 計算 (Fraction Skill Score, 5x5)
    fss = calculate_fss(y_true_bin, y_pred_bin, window_size=5)

    return [tp, tn, fp, fn, acc, pre, rec, csi, hss, fss, mse]

def run_evaluation_pipeline(base_path, thresholds=[0.1, 0.5, 1.0, 2.0]):
    """
    針對滾動式預測資料進行自動化分層評估。
    """
    subfolders = ['single_diffusion', 'ensemble_diffusion']
    
    for folder in subfolders:
        folder_path = os.path.join(base_path, folder)
        npy_path = os.path.join(folder_path, "raw_comparison_data.npy")
        
        if not os.path.exists(npy_path):
            print(f"⚠️ 找不到資料: {npy_path}")
            continue
            
        print(f"🚀 正在分析滾動預測序列: {folder}")
        raw = np.load(npy_path)
        # raw shape: (n_iter, batch, h, w, c) -> 轉換為 (Sample, H, W, Channels)
        data = raw.reshape(-1, 96, 96, 6)
        
        for th in thresholds:
            detailed_data = []
            for i in range(data.shape[0]):
                # 遍歷 T+1, T+2, T+3
                for t_idx, t_name in enumerate(['T+1', 'T+2', 'T+3']):
                    y_true = data[i, :, :, t_idx]
                    y_pred = data[i, :, :, t_idx + 3]
                    
                    metrics = calculate_all_metrics(y_true, y_pred, th)
                    # 儲存 Sample_ID 以利後續分析滑動物理時間
                    detailed_data.append([i, t_name] + metrics)

            cols = ['Sample_ID', 'Time_Step', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'CSI', 'HSS', 'FSS', 'MSE']
            df_detailed = pd.DataFrame(detailed_data, columns=cols)
            
            # --- 【關鍵修改處】修正隱憂二：改為正確的氣象統計彙整邏輯 ---
            summary_rows = []
            for t_name in ['T+1', 'T+2', 'T+3']:
                # 篩選特定時間步的數據
                df_step = df_detailed[df_detailed['Time_Step'] == t_name]
                
                # 1. 先將所有樣本的四象限計數加總
                sum_tp = df_step['TP'].sum()
                sum_tn = df_step['TN'].sum()
                sum_fp = df_step['FP'].sum()
                sum_fn = df_step['FN'].sum()
                
                # 2. 用總數重新計算統計指標
                total = sum_tp + sum_tn + sum_fp + sum_fn
                acc = (sum_tp + sum_tn) / total if total > 0 else 0
                pre = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
                rec = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
                csi = sum_tp / (sum_tp + sum_fp + sum_fn) if (sum_tp + sum_fp + sum_fn) > 0 else 0
                
                num_hss = 2 * (sum_tp * sum_tn - sum_fp * sum_fn)
                den_hss = (sum_tp + sum_fn) * (sum_fn + sum_tn) + (sum_tp + sum_fp) * (sum_fp + sum_tn)
                hss = num_hss / den_hss if den_hss > 0 else 0
                
                # 3. 空間指標 FSS 與連續變數指標 MSE 採用平均值
                avg_fss = df_step['FSS'].mean()
                avg_mse = df_step['MSE'].mean()
                
                summary_rows.append([t_name, sum_tp, sum_tn, sum_fp, sum_fn, acc, pre, rec, csi, hss, avg_fss, avg_mse])
            
            # 建立正確的摘要 DataFrame
            summary_cols = ['Time_Step', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'CSI', 'HSS', 'FSS', 'MSE']
            df_summary = pd.DataFrame(summary_rows, columns=summary_cols).set_index('Time_Step')
            # --------------------------------------------------------
            
            # 檔案輸出
            suffix = str(th).replace('.', '_')
            df_detailed.to_csv(os.path.join(folder_path, f"{folder}_detailed_th{suffix}.csv"), index=False)
            df_summary.to_csv(os.path.join(folder_path, f"{folder}_summary_th{suffix}_2.csv")) # 加上 _2 區隔舊檔案
            print(f"    ✅ 門檻 {th} 處理完畢")

# --- 主執行區 ---
if __name__ == "__main__":
    # 設定你的根路徑
    PATH = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main測試\0.2\evaluation_final_archive(052410)新loss"
    
    # 執行多門檻分析
    run_evaluation_pipeline(PATH, thresholds=[0.1, 0.5, 5.0, 10.0])
    print("\n✨ CSV 檔案已根據滑動預測樣本全部生成完畢。")