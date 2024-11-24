from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

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


def _generate_feature_for_batch(
    batch, processor, model, dev
) -> tuple[tuple[str], dict[str, np.ndarray]]:
    """
    バッチごとに画像特徴を生成する

    Returns:
        tuple[str]: IDのtuple
        dict[str, np.ndarray]: 画像ごとの特徴の辞書。keyは画像の名前。valueは特徴の配列(shape: (batch, feature))
    """
    # tuple, dict[str, tensor]
    ids, images = batch

    outputs_dict = {}
    for image_name, image in images.items():
        inputs = processor(images=image, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
            # (batch, token, feature)
            outputs = outputs.last_hidden_state
            # (batch, feature)
            outputs = outputs.mean(dim=1)

        outputs_dict[image_name] = outputs.cpu().numpy()

    return ids, outputs_dict


def _create_image_feature_df(
    ids: list, features_dict: dict[str, np.ndarray]
) -> pl.DataFrame:
    """
    画像特徴をDataFrameに変換する

    Args:
        save_file_path (Path): 保存するファイルのパス
        ids (list): IDのリスト
        features_dict (dict): 画像ごとの特徴の辞書。keyは画像の名前。valueは特徴の配列(shape: (batch, feature))
    """
    feature_cols = [f"ft_{i}" for i in range(list(features_dict.values())[0].shape[1])]

    df = []
    for image_name, features in features_dict.items():
        df_ = pl.from_numpy(features, schema=feature_cols)
        df_ = df_.with_columns(
            pl.Series("ID", ids),
            pl.lit(image_name).alias("image_name"),
        )
        df.append(df_)

    df = pl.concat(df, how="vertical")
    df = df.select(["ID", "image_name"] + feature_cols)
    df = df.sort(["ID", "image_name"])

    return df


def create_image_feature(
    feature_csv: Path,
    images_dir: Path,
    save_csv: Path,
):
    df = read_feature_csv(feature_csv)

    image_names = ["image_t.png", "image_t-0.5.png", "image_t-1.0.png"]
    # (sample, image_name, height, width, channel)
    images = load_images(
        images_dir, ids=df.get_column("ID").to_list(), image_names=image_names
    )

    dev = "cuda"
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev)

    ds = Dataset(images, df, image_names)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=128, shuffle=False, drop_last=False, num_workers=4
    )

    df = []
    for batch in tqdm(dl):
        ids_, img_feature_dict_ = _generate_feature_for_batch(
            batch, processor, model, dev
        )
        df_ = _create_image_feature_df(ids_, img_feature_dict_)
        df.append(df_)
    df = pl.concat(df, how="vertical")
    df.write_csv(save_csv)
    print(f"Save to {save_csv}")


if __name__ == "__main__":
    tr_features_csv = constants.TR_FEATURES_CSV
    ts_features_csv = constants.TS_FEATURES_CSV
    images_dir = constants.IMAGES_DIR
    save_dir = constants.IMAGES_DIR
    save_tr_csv = constants.TR_IMAGE_FEATURE_CSV
    save_ts_csv = constants.TS_IMAGE_FEATURE_CSV

    create_image_feature(tr_features_csv, images_dir, save_tr_csv)
    create_image_feature(ts_features_csv, images_dir, save_ts_csv)
