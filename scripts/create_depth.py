from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from atmacup_18 import constants


def read_feature_csv(feature_csv: Path) -> pl.DataFrame:
    """
    feature.csvを読み込む

    Args:
        feature_csv (Path): feature.csvのパス

    Returns:
        pl.DataFrame: feature.csvのデータ
    """
    df = pl.read_csv(str(feature_csv))
    df = df.with_columns(
        pl.col("ID").str.split("_").list.get(0).alias("scene_id"),
        pl.col("ID").str.split("_").list.get(1).cast(pl.Int32).alias("scene_dsec"),
        pl.arange(len(df)).alias("origin_idx"),
    )
    return df


def load_images(image_dir: Path, ids: list, image_names: list[str]) -> np.ndarray:
    """
    画像を読み込む

    Args:
        image_dir (Path): 画像が格納されているディレクトリ
        ids (list): 画像のIDのリスト
        image_names (list[str]): 画像のファイル名のリスト。"image_dir/{id}/{image_name}" が画像のパスになる

    Returns:
        np.ndarray: 画像の配列(shape: (id, image_name, height, width, channel))
    """

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, df: pl.DataFrame, image_names: list[str]):
        """
        Args:
            images (np.ndarray): 画像の配列(shape: (id, image_name, height, width, channel))
            df (pl.DataFrame): feature.csvのデータ
        """
        self.images = images
        self.df = df
        self.ids = df.get_column("ID").to_list()
        self.image_names = image_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.ids[idx]
        images = {}
        for i, image_name in enumerate(self.image_names):
            images[image_name] = self.images[idx, i]

        return id, images


def _generate_depth_for_batch(
    batch, processor, model, dev
) -> tuple[tuple[str], dict[str, np.ndarray]]:
    """
    バッチごとに深度を生成する

    Returns:
        tuple[str]: IDのtuple
        dict[str, np.ndarray]: 画像ごとの深度の辞書。keyは画像の名前。valueは深度の配列(shape: (batch, height, width, 1))
    """
    # tuple, dict[str, tensor]
    ids, images = batch

    outputs_dict = {}
    for image_name, image in images.items():
        inputs = processor(images=image, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
            # (batch, h, w)
            outputs = outputs.predicted_depth
            # (batch, 1, h, w)
            outputs = outputs.unsqueeze(1)

        image_size = image.size()[1:3]
        # 元のサイズにリサイズ
        outputs = torch.nn.functional.interpolate(
            outputs, size=image_size, mode="bilinear", align_corners=True
        )
        # (batch, h, w, c)に変換
        outputs = outputs.permute(0, 2, 3, 1)

        outputs_dict[image_name] = outputs.cpu().numpy()

    return ids, outputs_dict


def _save_images_npy(
    save_dir: Path, save_file_prefix: str, ids: list, images_dict: dict
):
    """
    画像をnpy形式で保存する

    Args:
        save_dir (Path): 保存先のディレクトリ
        save_file_prefix (str): 保存するファイル名のプレフィックス
        ids (list): IDのリスト
        images_dict (dict): 画像の辞書。keyは画像の名前。valueは画像の配列(shape: (id, height, width, channel))
    """
    save_dir.mkdir(exist_ok=True, parents=True)

    for i_id, id in enumerate(ids):
        for image_name, images in images_dict.items():
            image = images[i_id]
            save_path = save_dir.joinpath(id, f"{save_file_prefix}{image_name}.npy")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(save_path), image)

    return


def create_depth(
    feature_csv: Path, images_dir: Path, save_dir: Path, save_file_prefix: str
):
    df = read_feature_csv(feature_csv)

    image_names = ["image_t.png", "image_t-0.5.png", "image_t-1.0.png"]
    # (sample, image_name, height, width, channel)
    images = load_images(
        images_dir, ids=df.get_column("ID").to_list(), image_names=image_names
    )

    dev = "cuda"
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(dev)

    ds = Dataset(images, df, image_names)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, drop_last=False, num_workers=4
    )

    for batch in tqdm(dl):
        ids, depth_dict = _generate_depth_for_batch(batch, processor, model, dev)
        _save_images_npy(save_dir, save_file_prefix, ids, depth_dict)


if __name__ == "__main__":
    tr_features_csv = constants.TR_FEATURES_CSV
    ts_features_csv = constants.TS_FEATURES_CSV
    images_dir = constants.IMAGES_DIR
    save_dir = constants.IMAGES_DIR
    save_file_prefix = constants.DEPTH_IMAGE_FILE_PREFIX

    create_depth(tr_features_csv, images_dir, save_dir, save_file_prefix)
    create_depth(ts_features_csv, images_dir, save_dir, save_file_prefix)
