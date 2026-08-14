import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# ⚠️ 新增導入：用於處理地理圖形檔案 (.shp)
try:
    import geopandas as gpd
except ImportError:
    print("⚠️ 警告：找不到 geopandas 庫。請安裝：pip install geopandas")
    # 如果找不到 geopandas，可以考慮退出或只執行不繪圖的邏輯
    # exit() 
    gpd = None


# --- 物理常數 ---
STANDARD_GRAVITY = 9.80665 

# --- 路徑設定 ---
input_folder = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\msl")
GEO_FILENAMES = ["geo850_extracted_14-15.csv", "geo850_extracted_16-22.csv"]

output_folder = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\msl\96x96_2014-2022_GEO850")
output_csv_folder = output_folder / "interpolated_geo_csvs"

output_folder.mkdir(parents=True, exist_ok=True)
output_csv_folder.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# ⚠️ 新增：台灣地圖 Shapefile 路徑
# -------------------------------------------------------------
taiwan_shapefile_path = r"D:\台灣地圖\onlytw_shp\onlytw_shp\taiwan.shp"


# --- 欄位名稱設定 ---
GEO_COL_ORIG = 'Geopotential_850hPa_m^2/s^2'
GEO_COL_NEW = 'geopotential_height'

# --- 經緯度範圍與目標網格尺寸 ---
lon_range = (119.0, 123.0)
lat_range = (21.5, 25.5)
target_size = (96, 96)

# --- 初始化列表與全域旗標 ---
all_interpolated_data = []
IS_FIRST_TIME = True

# -------------------------------------------------------------
# --- 核心優化：預先計算目標網格座標 ---
# -------------------------------------------------------------
new_h, new_w = target_size
lon_min, lon_max = lon_range
lat_min, lat_max = lat_range

new_lons = np.linspace(lon_min, lon_max, new_w)
new_lats = np.linspace(lat_max, lat_min, new_h) 
grid_lon, grid_lat = np.meshgrid(new_lons, new_lats)
target_coordinates = (grid_lon, grid_lat)


# --- 輔助函數：執行插值 ---
def interpolate_to_grid(df, target_coords):
    """
    將 DataFrame 中的點資料插值到預先定義的網格上。
    """
    points_orig = df[['longitude', 'latitude']].values
    data_values = df[GEO_COL_NEW].values

    # 執行線性插值
    interpolated_data = griddata(
        points_orig,
        data_values.ravel(),
        target_coords, 
        method='linear'
    ).astype(np.float32)
    
    # 將負值設為 Nan
    interpolated_data[interpolated_data < 0] = np.nan
    
    return interpolated_data

# -------------------------------------------------------------
# --- 主要執行區塊：讀取、轉換、插值 (無變動) ---
# -------------------------------------------------------------
print("🚀 開始處理 Geo850 CSV 檔案 (優化版本)...")

# 1. 讀取並合併檔案
all_df_parts = []
for filename in GEO_FILENAMES:
    file_path = input_folder / filename
    if not file_path.exists():
        print(f"警告：找不到檔案 {file_path}，已跳過。")
        continue

    print(f"  正在讀取並合併檔案: {filename}...")
    try:
        df_part = pd.read_csv(file_path)
        all_df_parts.append(df_part)
    except Exception as e:
        print(f"讀取檔案 {filename} 時發生錯誤: {e}")
        continue

if not all_df_parts:
    print("❌ 找不到任何 Geo850 檔案，腳本結束。")
    exit()

df_combined = pd.concat(all_df_parts, ignore_index=True)
print(f"✅ 檔案合併完成，總行數：{len(df_combined)}")

# 2. 欄位處理與位勢轉換
df_combined = df_combined.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})

if GEO_COL_ORIG not in df_combined.columns:
    print(f"❌ 錯誤：合併後的數據缺少必要的欄位 '{GEO_COL_ORIG}'。腳本結束。")
    exit()

df_combined[GEO_COL_NEW] = df_combined[GEO_COL_ORIG] / STANDARD_GRAVITY
print(f"✅ 位勢轉換完成 (除以 {STANDARD_GRAVITY} m/s^2)")

# --- 3. 逐時間步插值 (優化: 使用 GroupBy) ---
print("\n開始逐時間步進行 96x96 空間插值 (速度優化)...")
processed_files_count = 0
grouped_data = df_combined.groupby('Time') 

for time_str, df_slice in grouped_data: 
    
    try:
        if len(df_slice) < 4:
             continue
        
        # 執行插值
        interpolated_data_2d = interpolate_to_grid(df_slice, target_coordinates)
        
        # --- 驗證：只儲存第一個時間步的 NPY 檔案 ---
        if IS_FIRST_TIME:
            npy_check_path = output_folder / f"CHECK_geo850_{time_str.replace('/', '').replace(' ', '_').replace(':', '')}_96x96.npy"
            np.save(npy_check_path, interpolated_data_2d)
            print(f"  💡 已儲存第一個時間步的 NPY 驗證檔：{npy_check_path.name}")
            IS_FIRST_TIME = False 
            
        all_interpolated_data.append(interpolated_data_2d)
        processed_files_count += 1
        
        if processed_files_count % 1000 == 0:
             print(f"  處理進度：已完成 {processed_files_count} 個時間步...")

    except Exception as e:
        print(f"處理時間 {time_str} 時發生錯誤: {e}")

print(f"\n總共處理了 {processed_files_count} 個時間步長。")
print("-" * 30)

# -------------------------------------------------------------
# --- 數據堆疊、平均與正規化 (無變動) ---
# -------------------------------------------------------------
if not all_interpolated_data:
    print("沒有找到可處理的 Geo850 數據。腳本結束。")
else:
    all_geo = np.stack(all_interpolated_data, axis=0)
    
    mean_geo = np.nanmean(all_geo, axis=0)
    mean_geo = np.nan_to_num(mean_geo, nan=np.nanmean(mean_geo) if not np.all(np.isnan(mean_geo)) else 0.0)
    mean_geo_2d = mean_geo.reshape(target_size)

    geo_min = np.min(mean_geo_2d)
    geo_max = np.max(mean_geo_2d)
    geo_norm = (mean_geo_2d - geo_min) / (geo_max - geo_min)
    
    npy_path = output_folder / "mean_geo850_norm.npy"
    np.save(npy_path, geo_norm)
    print(f"✅ 已儲存正規化後 NPY 檔案：{npy_path}")

    # -------------------------------------------------------------
    # --- 繪圖 (新增 Shapefile 疊加功能) ---
    # -------------------------------------------------------------
    print("開始繪製圖片 (疊加台灣地圖)...")
    
    lon_pcolor_edges = np.linspace(lon_range[0], lon_range[1], target_size[1] + 1)
    lat_pcolor_edges = np.linspace(lat_range[0], lat_range[1], target_size[0] + 1)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # 1. 繪製熱圖
    mesh = plt.pcolormesh(lon_pcolor_edges, lat_pcolor_edges, geo_norm, cmap="plasma", shading="auto", zorder=0)

    # -------------------------------------------------------------
    # 2. 疊加台灣地圖輪廓 (Shapefile)
    # -------------------------------------------------------------
    if gpd is not None:
        try:
            # 讀取 Shapefile
            taiwan_map = gpd.read_file(taiwan_shapefile_path)
            
            # 確保 Shapefile 的 CRS (座標參考系統) 是 WGS84 (經緯度)
            # 如果 shapefile 默認不是 WGS84，則需要轉換：
            if taiwan_map.crs is not None and taiwan_map.crs.to_epsg() != 4326:
                 taiwan_map = taiwan_map.to_crs(epsg=4326) # 4326 是 WGS84
            
            # 繪製台灣地圖輪廓
            taiwan_map.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1.5, zorder=20)
            print(f"✅ 已成功疊加台灣地圖輪廓：{taiwan_shapefile_path}")

        except Exception as e:
            print(f"❌ 疊加台灣地圖時發生錯誤: {e}")
            print("請確認 shapefile 路徑正確且 geopandas 已正確安裝。")
    # -------------------------------------------------------------

    # 繪製網格線 (保持不變)
    fine_grid_color = 'dimgray'
    fine_grid_linewidth = 0.2
    fine_grid_zorder = 5
    for lon_val in lon_pcolor_edges:
        ax.axvline(x=lon_val, color=fine_grid_color, linestyle='-', linewidth=fine_grid_linewidth, zorder=fine_grid_zorder)
    for lat_val in lat_pcolor_edges:
        ax.axhline(y=lat_val, color=fine_grid_color, linestyle='-', linewidth=fine_grid_linewidth, zorder=fine_grid_zorder)

    coarse_grid_interval = 0.25
    coarse_grid_color = 'lightgray'
    coarse_grid_linewidth = 0.5
    coarse_grid_linestyle = '--'
    coarse_grid_alpha = 0.6
    coarse_grid_zorder = 10
    coarse_lon_lines = np.arange(lon_range[0], lon_range[1] + coarse_grid_interval, coarse_grid_interval)
    coarse_lat_lines = np.arange(lat_range[0], lat_range[1] + coarse_grid_interval, coarse_grid_interval)
    for lon_val in coarse_lon_lines:
        ax.axvline(x=lon_val, color=coarse_grid_color, linestyle=coarse_grid_linestyle, linewidth=coarse_grid_linewidth, alpha=coarse_grid_alpha, zorder=coarse_grid_zorder)
    for lat_val in coarse_lat_lines:
        ax.axhline(y=lat_val, color=coarse_grid_color, linestyle=coarse_grid_linestyle, linewidth=coarse_grid_linewidth, alpha=coarse_grid_alpha, zorder=coarse_grid_zorder)

    # 設定標籤與限制
    plt.colorbar(mesh, label="Normalized Geopotential Height (0~1)")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.title("Mean Normalized Geo. Height 850hPa (96x96) w/ Taiwan Overlay", fontsize=16)

    major_lon_ticks = coarse_lon_lines
    major_lat_ticks = coarse_lat_lines
    plt.xticks(major_lon_ticks, rotation=45, fontsize=10)
    plt.yticks(major_lat_ticks, fontsize=10)
    ax.set_xlim(lon_range[0], lon_range[1])
    ax.set_ylim(lat_range[0], lat_range[1])

    plt.tight_layout()
    png_path = output_folder / "mean_geo850_norm_visualization_dual_grid_with_map.png"
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    plt.show()
    print(f"✅ 已儲存可視化圖片：{png_path}")