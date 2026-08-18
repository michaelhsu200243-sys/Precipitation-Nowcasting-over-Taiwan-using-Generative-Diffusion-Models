# Precipitation-Nowcasting-over-Taiwan-using-Generative-Diffusion-Models

本專案之核心程式碼架構參考論文《Precipitation nowcasting with generative diffusion models》。
Github連結為：https://github.com/fmerizzi/Precipitation-nowcasting-with-generative-diffusion-models
本研究將此生成式擴散模型架構進一步延伸應用於臺灣地區。


檔案包含 training_good.py、evaluation.py、models.py、generators.py、utils.py 以及 setup.py。在執行模型前，需先完成各項特徵資料的預處理。

一、特徵資料預處理流程

風速資料處理：

使用 ERA5 資料，確認檔案內容並篩選目標時間範圍。

執行 ERA5_all_wind_deal_npy.py，產出 era5_wind_tensor_u10v10_96x96.npz。

執行 preprocess_wind.py，產出 era5_wind_tensor_96x96_.npz。



海平面氣壓與海陸遮罩處理：

使用 ERA5 資料，確認檔案內容並篩選時間範圍。

執行 ERA5_msl_deal.py，產出 mean_geo850_norm.npy。

執行 map.py，產出 taiwan_land_sea_mask_96x96_no_rasterio.npz。

執行 Combination.py 將氣壓與海陸遮罩合成為雙通道特徵矩陣，產出 2022-2023landSeaMask_msl.npy。



雨量資料處理：

使用QPESUMS資料

依序執行 1QPSUM_CSV.py 與 2QPSUM.csv處理.py 整理資料。

執行 ERA5_deal_npz.py，產出 test_set_gridded_20150101-20211230_96x96.npz。



時間資訊處理：

執行 time_ERA5.py。

產出 timestamps_combined_2015-2021.npy。



均值與變異數處理：
依照 mean var 資料夾內的指定順序進行腳本執行與資料處理。



二、模式執行與評估流程

執行 training_good.py 進行模型訓練並產出權重檔案。

執行 evaluation.py，將訓練好的權重與設定的時間帶入進行評估。

依序執行 plot_results2.py、全指標更新版.py 與 指標畫圖.py，進行結果繪圖與各項評估指標計算。