# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas
import re
import gc  # 導入垃圾回收模組，用於徹底釋放空間
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator

# --- 1. 顏色與色階設定 ---
nws_precip_colors = [
    '#ffffff', '#98ffff', '#00ceff', '#009aff', '#006af7',
    '#2e9c00', '#2bff00', '#fefe08', '#ffcb00', '#ff9c00',
    '#fe0005', '#c90200', '#9d0000', '#9a009d', '#cf00d7',
    '#ff00f7', '#fdcafe'
]
precip_colormap = mcolors.ListedColormap(nws_precip_colors)
clevel = [0, 1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 600]
norm = mcolors.BoundaryNorm(clevel, len(nws_precip_colors))

def plot_precipitation(ax, data, title, map_data):
    data_lon_full = [119.0, 123.0] 
    data_lat_full = [21.5, 25.5]
    orig_lons = np.linspace(data_lon_full[0], data_lon_full[1], data.shape[1])
    orig_lats = np.linspace(data_lat_full[0], data_lat_full[1], data.shape[0])
    interp_func = RegularGridInterpolator((orig_lats, orig_lons), data, method='cubic', bounds_error=False, fill_value=0)
    
    # 維持 600x600 高解析度
    grid_res = 600
    plot_lats = np.linspace(21.5, 25.5, grid_res)
    plot_lons = np.linspace(119.0, 123.0, grid_res)
    new_lon_grid, new_lat_grid = np.meshgrid(plot_lons, plot_lats)
    points = np.array([new_lat_grid.ravel(), new_lon_grid.ravel()]).T
    smooth_data = interp_func(points).reshape(grid_res, grid_res)
    smooth_data = np.maximum(smooth_data, 0)
    
    im = ax.pcolormesh(new_lon_grid, new_lat_grid, smooth_data, cmap=precip_colormap, norm=norm, shading='auto', zorder=1)
    
    if map_data is not None:
        map_data.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.8, zorder=2)
    
    view_lon = [119.5, 122.5]
    view_lat = [21.5, 25.5]
    ax.set_xlim(view_lon)
    ax.set_ylim(view_lat)
    
    lons_ticks = [120.0, 121.0, 122.0] 
    lats_ticks = np.arange(22.0, 25.5, 0.5) 
    
    ax.set_xticks(lons_ticks)
    ax.set_xticklabels([f"{x:.1f}°E" for x in lons_ticks], fontsize=14)
    ax.set_yticks(lats_ticks)
    ax.set_yticklabels([f"{y:.1f}°N" for y in lats_ticks], fontsize=14)
    
    ax.set_title(title, fontsize=20, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', color='gray', alpha=0.3)
    
    # 畫完後刪除暫存的內插數據，節省內存
    del new_lon_grid, new_lat_grid, points, smooth_data, interp_func
    return im

def visualize_comparison(iter_idx, batch_idx, timestamps, raw_data, taiwan_shp, base_output_dir, mode_name):
    base_idx = iter_idx * raw_data.shape[1] + batch_idx
    try:
        t_str = str(timestamps[base_idx])
        base_time = datetime.strptime(t_str, "%Y-%m-%dT%H")
    except:
        base_time = datetime.now()
        
    obs_frames = raw_data[iter_idx, batch_idx, :, :, :3].transpose(2, 0, 1)
    pred_frames = raw_data[iter_idx, batch_idx, :, :, 3:].transpose(2, 0, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    display_mode = "single diffusion" if mode_name == "Single" else "ensemble diffusion"
    
    fig.suptitle(f"{display_mode} {base_time.strftime('%Y-%m-%d %H:00')} ", fontsize=36, fontweight='bold', y=0.97)

    for i in range(3):
        h_time = base_time - timedelta(hours=(2-i))
        title = f"Historical (T-{2-i})\n{h_time.strftime('%Y-%m-%d %H:00')}"
        plot_precipitation(axes[0, i], obs_frames[i], title, taiwan_shp)
    for i in range(3):
        f_time = base_time + timedelta(hours=(i+1))
        title = f"Forecast (T+{i+1})\n{f_time.strftime('%Y-%m-%d %H:00')}"
        im = plot_precipitation(axes[1, i], pred_frames[i], title, taiwan_shp)

    fig.subplots_adjust(left=0.04, right=0.87, top=0.87, bottom=0.05, wspace=0.02, hspace=0.18)
    
    cbar_ax = fig.add_axes([0.90, 0.1, 0.02, 0.8]) 
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=clevel)
    cbar.set_label('Rainfall Intensity (mm/hr)', fontsize=22, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=16) 

    save_dir = os.path.join(base_output_dir, mode_name)
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{mode_name}_Forecast_Iter{iter_idx}_B{batch_idx}.png"
    plt.savefig(os.path.join(save_dir, save_name), dpi=150, bbox_inches='tight')

    # --- 關鍵釋放邏輯：跑完就清掉 ---
    plt.clf()        # 清除畫布內容
    plt.close(fig)   # 關閉視窗物件
    plt.close('all') # 確保背景所有 Figure 都關閉
    del fig, axes, obs_frames, pred_frames, im  # 刪除變數引用
    gc.collect()     # 強制執行垃圾回收回收實體記憶體

if __name__ == "__main__":
    base_search_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main測試\0.2\對比"
    SHP_PATH = r"D:\台灣地圖\onlytw_shp\onlytw_shp\taiwan.shp"
    
    tw_map = geopandas.read_file(SHP_PATH) if os.path.exists(SHP_PATH) else None
    all_cases = [d for d in os.listdir(base_search_path) if os.path.isdir(os.path.join(base_search_path, d))]
    
    for case_name in all_cases:
        match = re.search(r"\((\d{8})\)", case_name)
        if not match: continue
        
        date_tag = match.group(1)
        year_prefix = date_tag[:2]
        
        time_npy_path = (r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\time\2021\timestamps_20210101-00-20211230-04.npy" 
                         if year_prefix == "21" else 
                         r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\time\2022\timestamps_20220101-00-20221231-14.npy")
        
        if not os.path.exists(time_npy_path): continue
            
        full_timestamps = np.load(time_npy_path, allow_pickle=True)
        start_iso = f"20{year_prefix}-{date_tag[2:4]}-{date_tag[4:6]}T{date_tag[6:8]}"
        idx_match = np.where(full_timestamps == start_iso)[0]
        
        if len(idx_match) == 0: continue
        target_ts = full_timestamps[idx_match[0]:]
        
        modes = [("Single", "single_diffusion"), ("Ensemble", "ensemble_diffusion")]
        output_root = os.path.join(base_search_path, case_name, "Final_Results_Plots")

        for mode_label, sub_folder in modes:
            npy_path = os.path.join(base_search_path, case_name, sub_folder, "raw_comparison_data.npy")
            if os.path.exists(npy_path):
                print(f"🚀 正在處理: {case_name} -> {mode_label}")
                data = np.load(npy_path)
                for i in range(data.shape[0]):
                    for b in range(data.shape[1]):
                        visualize_comparison(i, b, target_ts, data, tw_map, output_root, mode_label)
                
                # 處理完一個模式的大數據後，也進行一次手動釋放
                del data
                gc.collect()
            else:
                print(f"❌ 找不到檔案: {npy_path}")

    print("\n🎉 [全部任務完成，記憶體已妥善釋放]")