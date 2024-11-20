from pathlib import Path

# data
DATA_DIR = Path(".").joinpath("data")
DATASET_DIR = DATA_DIR.joinpath("atmaCup#18_dataset")
TR_FEATURES_CSV = DATASET_DIR.joinpath("train_features.csv")
TS_FEATURES_CSV = DATASET_DIR.joinpath("test_features.csv")
IMAGES_DIR = DATASET_DIR.joinpath("images")
TRAFFIC_LIGHTS_DIR = DATASET_DIR.joinpath("traffic_lights")

TRAFFIC_LIGHTS_CSV = DATASET_DIR.joinpath("traffic_lights.csv")

# image
IMAGE_SIZE = (64, 128)


# traffic light
TRAFFIC_LIGHT_CLASS_INDEXES = {
    "green": 0,
    "red": 1,
    "yellow": 2,
    "empty": 3,
    "right": 4,
    "straight": 5,
    "left": 6,
    "other": 7,
}

TRAFFIC_LIGHT_BBOX_IMAGE_NAME = "traffic_light_bbox_image.npy"

# optical flow
OPTICAL_FLOW_IMAGE_NAME = "optical_flow_image.npy"
