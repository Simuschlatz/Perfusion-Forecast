# Perfusion-Forecast: Reducing Radiation Dose and Scan Duration in CT Perfusion with Video Prediction Models

## Overview
This repo contains part of an ongoing research project aimed at reducing radiation dose and scan time in CT Perfusion imaging for stroke diagnosis by predicting missing scans with 4D spatiotemporal models. Our methods achieve realistic perfusion maps and promising clinical results, paving the way for faster, safer, and more reliable stroke diagnostics.

## Disclaimer
Beware of the License.

## Data Availability
The Preprocessed data used for training, validation and test are not included in this repository but will be provided by the author upon justified request.
## Installation
### Clone this repository:
```bash
git clone https://github.com/Simuschlatz/Perfusion-Forecast
cd Perfusion-Forecast
```
## Installation

### Using Docker
#### Build the Image
```bash
docker build -t perfusion-forecast:latest .
```
#### Run the Docker Container
```bash
docker run --rm -it perfusion-forecast:latest
```
#### if you want to run it with jupyter-lab add the following to your docker run command:
```bash
 -c "jupyter-lab --ip 0.0.0.0 --port 9020 --allow-root --NotebookApp.token='' --NotebookApp.password='' "
 ```

### Using pip:
```bash
pip install -r requirements.txt
```
### Setting up the Environment for Preprocessing with Conda
Mac, Windows & Linux
```bash
conda env create -f environment.yml
```
Apple Silicon
```bash
conda env create -f metal.yml
```
Activate the environment
```bash
conda activate perfcast
```

## Usage

### Training
```bash
usage: train_2d.py [-h] [--data_dir DATA_DIR] [--model_name {SimVP,SimVP2,UNetPlus_temp,UNet,PredFormer}] [--device DEVICE] [--gpu GPU] [--input_frames INPUT_FRAMES] [--pred_frames PRED_FRAMES]
                   [--pred_n_frames_per_step PRED_N_FRAMES_PER_STEP] [--seed SEED] [--num_workers NUM_WORKERS] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
                   [--precision PRECISION]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory where the data is stored
  --model_name {SimVP,SimVP2,UNetPlus_temp,UNet,PredFormer}
                        Name of the model to use
  --device DEVICE       Device the model runs on.
  --gpu GPU             Number of gpu the model runs on
  --input_frames INPUT_FRAMES
                        Number of time-frames the model gets as input
  --pred_frames PRED_FRAMES
                        Number of time-frames the model needs to predict
  --pred_n_frames_per_step PRED_N_FRAMES_PER_STEP
                        Number of time-frames that are predicted per step
  --seed SEED           Seed
  --num_workers NUM_WORKERS
                        Number of workers for the dataloaders
  --epochs EPOCHS       Number of Epochs
  --learning_rate LEARNING_RATE
                        Learning-Rate
  --batch_size BATCH_SIZE
                        Batch Size
  --precision PRECISION
                        Precision
```

Example Commands
```bash
python train_2d.py --model_name PredFormer --batch_size 5 --data_dir ./NormalizedQualityFiltered/ && python **train_2d**.py --model_name SimVP --batch_size 5 --data_dir ./NormalizedQualityFiltered/ && python train_2d.py --model_name SimVP2 --batch_size 5 --data_dir ./NormalizedQualityFiltered/ && python train_2d.py --model_name UNet --batch_size 5 --data_dir ./NormalizedQualityFiltered/ && python train_2d.py --model_name UNetPlus_temp --batch_size 5 --data_dir ./NormalizedQualityFiltered/

python train_3d.py --model_name PredFormer --batch_size 1 --data_dir ./NormalizedQualityFiltered/ && python train_3d.py --model_name SimVP --batch_size 1 --data_dir ./NormalizedQualityFiltered/ && python train_3d.py --model_name SimVP2 --batch_size 1 --data_dir ./NormalizedQualityFiltered/ && python train_3d.py --model_name UNet --batch_size 1 --data_dir ./NormalizedQualityFiltered/ && python train_3d.py --model_name UNetPlus_temp --batch_size 1 --data_dir ./NormalizedQualityFiltered/
```
# Contact
If you experience any issues, feel free to reach out at simon.ma@iserv-schillerschule.de
