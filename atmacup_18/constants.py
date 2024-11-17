from pathlib import Path

DATA_DIR = Path(".").joinpath("data")
DATASET_DIR = DATA_DIR.joinpath("atmaCup#18_dataset")
TR_FEATURES_CSV = DATASET_DIR.joinpath("train_features.csv")
TS_FEATURES_CSV = DATASET_DIR.joinpath("test_features.csv")
IMAGES_DIR = DATASET_DIR.joinpath("images")
TRAFFIC_LIGHTS_DIR = DATASET_DIR.joinpath("traffic_lights")

TRAFFIC_LIGHTS_CSV = DATASET_DIR.joinpath("traffic_lights.csv")
