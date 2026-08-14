import numpy as np
from pathlib import Path

# 讀取位勢高度 npy (16,16)
msl_path = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\ERA5\msl\96x96_2014-2022_GEO850\mean_geo850_norm.npy")
msl = np.load(msl_path)  # shape (16,16)

# 讀取海陸遮罩 npz (16,16)
mask_npz_path = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\taiwan_land_sea_mask_96x96_no_rasterio.npz")
mask_npz = np.load(mask_npz_path)
mask_key = list(mask_npz.keys())[0]
mask = mask_npz[mask_key]  # shape (16,16)

# 確認形狀一致
assert msl.shape == mask.shape == (96,96), "兩個陣列大小需相同"

# 合併成 (16,16,2)
combined = np.stack([msl, mask], axis=-1)  # shape (16,16,2)

# 使用 transpose 函式將形狀轉換為 (2, 16, 16)
transposed_combined = combined.transpose((2, 0, 1))

# 顯示轉換後的形狀以確認
print(f"原始陣列形狀: {combined.shape}")
print(f"轉換後陣列形狀: {transposed_combined.shape}")

# 儲存成 npy，檔名 landSeaMask_msl.npy
output_path = Path(r"D:\Precipitation-nowcasting-with-generative-diffusion-models-main\MYdata\map_msl\96x96_2022-2023\2022-2023landSeaMask_msl.npy")
np.save(output_path, transposed_combined)

print(f"✅ 已儲存轉換後的 npy 檔案：{output_path}")