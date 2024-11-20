from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from atmacup_18 import constants


def load_images(image_dir: Path, ids: list) -> np.ndarray:
    """
    画像を読み込む

    Args:
        image_dir (Path): 画像が格納されているディレクトリ
        ids (list): 画像のIDのリスト

    Returns:
        np.ndarray: 画像の配列(shape: (id, image_type, height, width, channel))
    """
    image_names = [
        "image_t.png",
        "image_t-0.5.png",
        "image_t-1.0.png",
    ]

    def read_images_for_id(id):
        images_for_id = []
        for image_name in image_names:
            image_path = image_dir.joinpath(f"{id}/{image_name}")
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_for_id.append(image)
        return images_for_id

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(read_images_for_id, ids))

    images = np.array(images)
    return images


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(520, 960)),
        ]
    )
    batch = transforms(batch)
    return batch


def get_optical_flow_model():
    model = torchvision.models.optical_flow.raft_large(pretrained=True, progress=True)
    return model


def calc_optical_flows(model, images: np.ndarray, dev: str = "cuda"):
    """

    Args:
        images (np.ndarray): shape: (id, image_type(3), height, width, channel)。image_type軸は時系列順であること
    """
    # (id, image_type, channel, height, width)に変換
    images = torch.tensor(images).permute(0, 1, 4, 2, 3)

    flows = []
    for i_id in range(images.shape[0]):
        # (3, channel, height, width)
        img_set = images[i_id]
        img_hw = img_set.shape[-2:]
        img_set = preprocess(img_set).to(dev)

        # 3枚の画像が1セットなので、2枚ずつのペアを作成する
        # (img1_set[0], img2_set[0]), (img1_set[1], img2_set[1]) というペアとなる。
        img1_set = img_set[0:2]
        img2_set = img_set[1:3]

        with torch.no_grad():
            flow_batch = model(img1_set, img2_set)
        # num_flow_update個の要素を持つtensorが出力される。
        # 最後の要素が最終的なflowのみを取得する。
        # shape: (pair(2), flow(2), height, width)
        flow_batch = flow_batch[-1]
        # サイズをもとに戻す
        flow_batch = F.interpolate(
            flow_batch, size=img_hw, mode="bilinear", align_corners=True
        )

        # shape: (pair(2), flow(2), height, width)
        flow_batch = flow_batch.cpu().detach().numpy()

        flows.append(flow_batch)

    # shape: (id, pair(2), flow(2), height, width)
    # pair(2): 各idの1枚目及び2枚目の画像に対するflow
    flows = np.array(flows)

    # (id, height, width, pair(2), flow(2))
    flows = flows.transpose(0, 3, 4, 1, 2)
    # (id, height, width, pair(2) * flow(2))
    flows = flows.reshape(flows.shape[0], flows.shape[1], flows.shape[2], -1)

    return flows


def create_optical_flow_image(
    features_csv: Path,
    images_dir: Path,
    save_image_name: str,
    save_dir: Path,
):
    df = pl.read_csv(features_csv)
    ids = df.get_column("ID").to_list()
    # (id, 3, height, width, channel)
    images = load_images(images_dir, ids)

    dev = "cuda"
    model = get_optical_flow_model()
    model.to(dev)
    model.eval()

    for i_id, id in tqdm(enumerate(ids), total=len(ids)):
        # (1, 3, height, width, channel)
        images_ = images[i_id : i_id + 1]

        # (1, height, width, pair(2) * flow(2))
        flows = calc_optical_flows(model, images_)

        flow = flows[0]
        save_path = Path(save_dir, id, save_image_name)
        np.save(save_path, flow)

    return


if __name__ == "__main__":
    features_csv = constants.TR_FEATURES_CSV
    images_dir = constants.IMAGES_DIR
    save_image_name = constants.OPTICAL_FLOW_IMAGE_NAME
    save_dir = constants.IMAGES_DIR

    create_optical_flow_image(
        features_csv=features_csv,
        images_dir=images_dir,
        save_image_name=save_image_name,
        save_dir=save_dir,
    )

    features_csv = constants.TS_FEATURES_CSV

    create_optical_flow_image(
        features_csv=features_csv,
        images_dir=images_dir,
        save_image_name=save_image_name,
        save_dir=save_dir,
    )
