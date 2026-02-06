### Setup
pip install -r requirements.txt

### 1) Train AE
python -m src.train_ae

### 2) Train TCN
python -m src.train_tcn

### 3) Inference + submission
python -m src.infer_submit
