# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:53:28 2026
@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 04:00:00 2026
@author: Gemini AI (Journal Style Visualization - Final Multi-Date Version)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib import rcParams

# ==========================================
# 視覺風格設定 (字體加大且加粗)
# ==========================================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = "Times New Roman"
# 將權重設為 bold (粗體)，並加大整體基礎字級
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
sns.set_theme(style="whitegrid", font_scale=1.4) # font_scale 從 1.2 提高到 1.4

def extract_date_from_path(path_str):
    """ 
    自動轉換日期格式: 
    輸入 (21060410) -> 輸出 20210604
    """
    match = re.search(r'\((\d{8})\)', path_str)
    if match:
        raw_date = match.group(1)
        return f"20{raw_date[:6]}"
    return "Unknown date"

def get_clean_df(file_path, is_rm=False):
    try:
        df = pd.read_csv(file_path)
        if is_rm and (df.columns[0] == 'Unnamed: 0' or 'Time' not in df.columns[0]):
            df = df.reset_index()
        df.columns = [c.strip().upper() for c in df.columns]
        time_col = df.columns[0]
        for col in df.columns:
            if any(k in col for k in ['TIME', 'STEP', 'LEAD']):
                time_col = col; break
        df['TIME_KEY'] = df[time_col].astype(str).str.extract('(\d+)').astype(float)
        
        # --- 關鍵修正：確保指標不超過 1 ---
        metric_cols = ['HSS', 'CSI', 'FSS', 'ACCURACY', 'PRECISION', 'RECALL']
        for m in metric_cols:
            if m in df.columns:
                df[m] = df[m].clip(0, 1)
                
        return df.dropna(subset=['TIME_KEY'])
    except: return None

# ==========================================
# 繪圖函式 A：HSS, CSI, FSS (Bar Plot)
# ==========================================

def plot_bar_metrics(combined_det, rm_data, date_str, threshold_th, display_th, m, output_path):
    t3_min = combined_det[combined_det['TIME_KEY'] == 3][m].min()
    rm_t3 = rm_data.sort_values('TIME_KEY')[m].values[2]
    y_min = max(0, min(t3_min, rm_t3) - 0.05)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined_det, x='TIME_KEY', y=m, hue='MODEL', 
                palette={"Single": "#85C1E9", "Ensemble": "#F8C471"},
                capsize=.1, errorbar='sd', alpha=0.9)
    
    rm_vals = rm_data.sort_values('TIME_KEY')[m].values[:3]
    plt.plot([0, 1, 2], rm_vals, marker='^', color='#27ae60', 
             markersize=12, label='Rainymotion (base)', linestyle='--', linewidth=2.5)

    # 加大標題、座標軸標籤、刻度字體
    plt.title(f'{m} temporal comparison ({date_str}, {display_th})', fontsize=18, pad=15, fontweight='bold')
    plt.xlabel('Lead time', fontsize=16, fontweight='bold')
    plt.xticks(ticks=[0, 1, 2], labels=['T+1h', 'T+2h', 'T+3h'], fontsize=14, fontweight='bold')
    plt.ylabel(f'{m}', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    plt.ylim(y_min, 1.0) 
    # 圖例字體加大
    plt.legend(title='Forecast model', loc='upper right', frameon=True, fontsize=12, title_fontsize=13)
    plt.savefig(os.path.join(output_path, f"{m}_Bar_th{threshold_th}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 繪圖函式 B：Accuracy, Precision, Recall (Line Plot)
# ==========================================

def plot_line_metrics(combined_det, rm_data, date_str, threshold_th, display_th, m, output_path):
    m_display = m.capitalize() if m not in ['CSI', 'HSS', 'FSS'] else m
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=combined_det, x='TIME_KEY', y=m, hue='MODEL', style='MODEL',
                 markers={"Single": "o", "Ensemble": "s"}, dashes=False,
                 palette={"Single": "#3498db", "Ensemble": "#e67e22"},
                 errorbar='sd', linewidth=4, markersize=12) # 線條與標記加粗加大
    
    rm_vals = rm_data.sort_values('TIME_KEY')[m].values[:3]
    plt.plot([1, 2, 3], rm_vals, marker='^', color='#2ecc71', 
             markersize=12, label='Rainymotion', linestyle='-', linewidth=3, alpha=0.7)

    # 加大標題、座標軸標籤、刻度字體
    plt.title(f'{m_display} trend ({date_str}, {display_th})', fontsize=18, pad=15, fontweight='bold')
    plt.xlabel('Lead time', fontsize=16, fontweight='bold')
    plt.xticks(ticks=[1, 2, 3], labels=['T+1h', 'T+2h', 'T+3h'], fontsize=14, fontweight='bold')
    plt.ylabel(f'{m_display}', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    
    t3_min = min(combined_det[combined_det['TIME_KEY'] == 3][m].min(), rm_vals[2])
    y_min = max(0, t3_min - 0.1)
    plt.ylim(y_min, 1.0) 
    
    plt.legend(title='Forecast model', loc='lower left', frameon=True, fontsize=12, title_fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_path, f"{m}_Line_th{threshold_th}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 執行與路徑設定 (多日期批次處理)
# ==========================================

if __name__ == "__main__":
    ROOT_DIR = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main測試\0.2\對比"
    
    target_dates = ["21060410", "21073122", "21080510", "22052420", "22091120"]

    for d_suffix in target_dates:
        PATH = os.path.join(ROOT_DIR, f"evaluation_final_archive({d_suffix})")
        RM_PATH = os.path.join(ROOT_DIR, f"new20{d_suffix}")
        
        date_log = extract_date_from_path(PATH)
        OUTPUT_PATH = os.path.join(ROOT_DIR, "圖", date_log)
        
        if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
        
        print(f"🚀 正在處理事件: {date_log} (資料夾: {d_suffix})")

        for v in [0.5, 5.0, 10.0]:
            th_str = str(v).replace('.', '_')
            display_th = f"{v} mm/h"
            
            s_det = get_clean_df(os.path.join(PATH, 'single_diffusion', f'single_diffusion_detailed_th{th_str}.csv'))
            e_det = get_clean_df(os.path.join(PATH, 'ensemble_diffusion', f'ensemble_diffusion_detailed_th{th_str}.csv'))
            rm_data = get_clean_df(os.path.join(RM_PATH, f'Rainymotion_Average_th_{th_str}.csv'), is_rm=True)

            if s_det is None or e_det is None or rm_data is None:
                print(f"  ⚠️ 缺失資料: {date_log} Threshold {v}, 跳過...")
                continue

            s_det['MODEL'] = 'Single'; e_det['MODEL'] = 'Ensemble'
            combined_det = pd.concat([s_det, e_det], ignore_index=True)
            combined_det['TIME_KEY'] = combined_det['TIME_KEY'].astype(int)

            for m in ['HSS', 'CSI', 'FSS']:
                if m in combined_det.columns:
                    plot_bar_metrics(combined_det, rm_data, date_log, th_str, display_th, m, OUTPUT_PATH)

            for m in ['ACCURACY', 'PRECISION', 'RECALL']:
                if m in combined_det.columns:
                    plot_line_metrics(combined_det, rm_data, date_log, th_str, display_th, m, OUTPUT_PATH)

    print(f"\n✨ 所有事件指標圖表處理完成！")