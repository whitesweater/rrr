@echo off
set MODEL_DIR=runwayml/stable-diffusion-v1-5
set OUTPUT_DIR=./output/
set TRAINDATA_DIR=TerrainDataset.py

accelerate launch train_controlnet.py ^
 --pretrained_model_name_or_path=%MODEL_DIR%
 --output_dir=%OUTPUT_DIR% ^
 --train_data_dir=%TRAINDATA_DIR% ^
 --resolution=128 ^
 --learning_rate=0.0001 ^
 --train_batch_size=1 ^
 --gradient_accumulation_steps=1 ^
 --conditioning_image_column="sketch" ^