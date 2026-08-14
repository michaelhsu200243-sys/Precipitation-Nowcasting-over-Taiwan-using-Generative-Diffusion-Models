# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:12:45 2025

@author: user
"""
import shapefile  # pyshp
from shapely.geometry import shape, Point
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 參數設定
target_lon_min = 119
target_lon_max = 123
target_lat_min = 21.5
target_lat_max = 25.5
grid_width = 96
grid_height = 96

shapefile_path = r"D:\台灣地圖\onlytw_shp\onlytw_shp\taiwan.shp"
output_npz_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\taiwan_land_sea_mask_96x96_no_rasterio.npz"
# 修改輸出文件名以區分，表示網格線已手動繪製
output_fig_path = r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\taiwan_land_sea_mask_96x96_no_rasterio_96x96grid_manual_lines.png"

# 讀取 shapefile 多邊形
sf = shapefile.Reader(shapefile_path)
polygons = [shape(record.shape.__geo_interface__) for record in sf.shapeRecords()]

# 建立經緯度網格點的中心點 (用於判斷是否在多邊形內)
lon_centers = np.linspace(target_lon_min + (target_lon_max - target_lon_min) / (2 * grid_width),
                          target_lon_max - (target_lon_max - target_lon_min) / (2 * grid_width),
                          grid_width)
lat_centers = np.linspace(target_lat_max - (target_lat_max - target_lat_min) / (2 * grid_height),
                          target_lat_min + (target_lat_max - target_lat_min) / (2 * grid_height),
                          grid_height)

# 建立空白 mask 陣列
mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

# 判斷每個格點是否在任何一個多邊形內
for i, y in enumerate(lat_centers):
    for j, x in enumerate(lon_centers):
        point = Point(x, y)
        if any(poly.contains(point) for poly in polygons):
            mask[i, j] = 1

# 儲存結果
np.savez(output_npz_path, mask=mask, lon=lon_centers, lat=lat_centers)
print(f"✔ 掩膜已儲存為 {output_npz_path}")

# 繪圖檢視
plt.figure(figsize=(8, 7))

# 定義顏色：0 (海) 為深藍色，1 (陸地) 為橙色
colors = ["#000080", "#FFA500"] # 深藍色 for sea, 橙色 for land
custom_cmap = ListedColormap(colors)

# imshow的extent參數是邊界，而不是中心。所以我們需要計算邊界點。
lon_bounds = np.linspace(target_lon_min, target_lon_max, grid_width + 1)
lat_bounds = np.linspace(target_lat_max, target_lat_min, grid_height + 1)

ax = plt.gca() # 獲取當前的軸

plt.imshow(mask, origin="upper",
           extent=[lon_bounds.min(), lon_bounds.max(), lat_bounds.min(), lat_bounds.max()],
           cmap=custom_cmap,
           interpolation='nearest',
           zorder=0) # 將 imshow 放在最底層 (zorder=0)

plt.title("Taiwan Land-Sea Mask (96x96 Grid)", fontsize=16)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)

# --- 調整刻度以顯示 16x16 網格線 ---
# 不再使用 ax.set_axisbelow(False) 和 plt.grid()

# 主要刻度用於顯示較少的、清晰的標籤
major_lon_ticks = np.arange(target_lon_min, target_lon_max + 0.1, 0.5)
major_lat_ticks = np.arange(target_lat_min, target_lat_max + 0.1, 0.5)
ax.set_xticks(major_lon_ticks, minor=False)
ax.set_yticks(major_lat_ticks, minor=False)

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

plt.tick_params(axis='both', which='major', labelsize=10)

# --- 關鍵修改部分：手動繪製所有網格線 ---
grid_line_color = 'white' # 網格線顏色
grid_line_width = 0.5     # 網格線粗細
grid_line_zorder = 10     # 確保網格線在最上層

# 繪製垂直網格線 (經線)
for lon_val in lon_bounds:
    ax.axvline(x=lon_val, color=grid_line_color, linestyle='-', linewidth=grid_line_width, zorder=grid_line_zorder)

# 繪製水平網格線 (緯線)
for lat_val in lat_bounds:
    ax.axhline(y=lat_val, color=grid_line_color, linestyle='-', linewidth=grid_line_width, zorder=grid_line_zorder)

# 為了確保軸的邊框線也在最上方，如果需要的話：
ax.spines['top'].set_linewidth(grid_line_width)
ax.spines['bottom'].set_linewidth(grid_line_width)
ax.spines['left'].set_linewidth(grid_line_width)
ax.spines['right'].set_linewidth(grid_line_width)
ax.spines['top'].set_edgecolor(grid_line_color)
ax.spines['bottom'].set_edgecolor(grid_line_color)
ax.spines['left'].set_edgecolor(grid_line_color)
ax.spines['right'].set_edgecolor(grid_line_color)


# --- 結束關鍵修改部分 ---

# 色條調整：
cbar = plt.colorbar(ticks=[0.25, 0.75])
cbar.set_ticklabels(['Sea', 'Land'])
cbar.ax.tick_params(labelsize=10)
cbar.set_label("0 = Sea, 1 = Land", fontsize=12)


plt.tight_layout()
plt.savefig(output_fig_path, dpi=400) # 保持高DPI
plt.close()
print(f"✔ 圖片已儲存為 {output_fig_path}")