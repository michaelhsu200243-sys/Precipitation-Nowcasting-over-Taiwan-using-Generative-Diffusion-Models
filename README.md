# Precipitation-Nowcasting-over-Taiwan-using-Generative-Diffusion-Models

The core code architecture of this project is based on the paper Precipitation nowcasting with generative diffusion models.
GitHub Repository: https://github.com/fmerizzi/Precipitation-nowcasting-with-generative-diffusion-models
This study further extends and applies this generative diffusion model architecture to the Taiwan region.

The repository contains training_good.py, evaluation.py, models.py, generators.py, utils.py, and setup.py. Prior to running the model, preprocessing of all feature datasets must be completed.

Feature Data Preprocessing Workflow

Wind Speed Data Preprocessing:

Check file content and filter the target time range using ERA5 data.

Run ERA5_all_wind_deal_npy.py to generate era5_wind_tensor_u10v10_96x96.npz.

Run preprocess_wind.py to generate era5_wind_tensor_96x96_.npz.



Mean Sea Level Pressure & Land-Sea Mask Preprocessing:

Check file content and filter the target time range using ERA5 data.

Run ERA5_msl_deal.py to generate mean_geo850_norm.npy.

Run map.py to generate taiwan_land_sea_mask_96x96_no_rasterio.npz.

Run Combination.py to combine sea level pressure and land-sea mask into a 2-channel feature matrix, generating 2022-2023landSeaMask_msl.npy.



Precipitation Data Preprocessing:

Using QPESUMS dataset.

Run 1QPSUM_CSV.py and 2QPSUM.csv處理.py sequentially to organize the data.

Run ERA5_deal_npz.py to generate test_set_gridded_20150101-20211230_96x96.npz.



Timestamp Information Preprocessing:

Run time_ERA5.py.

Generate timestamps_combined_2015-2021.npy.



Mean and Variance Preprocessing:
Execute scripts and process data according to the specified sequence in the mean_var folder.



Model Execution and Evaluation Workflow

Run training_good.py to train the model and save weight files.

Run evaluation.py to evaluate the model using the trained weights and designated time period.

Run plot_results2.py, 全指標更新版.py, and 指標畫圖.py sequentially to plot result figures and calculate all evaluation metrics.