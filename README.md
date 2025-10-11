# Perfusion-Forecast: Reducing Radiation Dose and Scan Duration in CT Perfusion with Video Prediction Models

## Overview
ChadCTP is a deep learning project that reduces radiation dose and scan time in CT Perfusion imaging for stroke diagnosis by predicting missing scans with 4D spatiotemporal models. Our methods achieve realistic perfusion maps and promising clinical results, paving the way for faster, safer, and more reliable stroke diagnostics.

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
## Usage with Docker

### Build the Image
```bash
docker build -t perfusion-forecast:latest .
```
### Run the Docker Container
```bash
docker run --rm -it perfusion-forecast:latest
```
### if you want to run it with jupyter-lab add the following to your docker run command:
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

# Contact
If you experience any issues, feel free to reach out at simon.ma@iserv-schillerschule.de
