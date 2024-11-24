from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import cv2
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# ########
# utils
# ########


# ########
# data
# ########
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


def load_npy_images(image_dir: Path, ids: list, image_names: list[str]) -> np.ndarray:
    """
    npy形式の画像を読み込む

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
            image = np.load(str(image_path))
            images_for_id.append(image)
        return images_for_id

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(read_images_for_id, ids))

    images = np.array(images)
    return images


def preprocess_images(images_list: list[np.ndarray]) -> np.ndarray:
    """
    画像の前処理

    Args:
        images_list (list[np.ndarray]): list of array (sample, image_type, height, width, channel)
    """
    concat_images = []
    for images in images_list:
        # (sample, image_type, channel, height, width)
        images = images.transpose(0, 1, 4, 2, 3)
        # (sample, image_type * channel, height, width)
        images = images.reshape(images.shape[0], -1, images.shape[3], images.shape[4])
        concat_images.append(images)

    # (sample, image_type * channel, height, width)
    concat_images = np.concatenate(concat_images, axis=1)

    return concat_images


def read_image_feature_csv(
    tr_image_feature_csv: Path, ts_image_feature_csv: Path, n_components: int = 32
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    画像特徴量のCSVを読み込む
    メモリ削減のためPCAを適用

    Args:
        tr_image_feature_csv (Path): trainの画像特徴量のCSV
        ts_image_feature_csv (Path): testの画像特徴量のCSV
    """
    # columns: "ID", "image_name", "ft_0", "ft_1", ...
    tr_df = pl.read_csv(tr_image_feature_csv)
    ts_df = pl.read_csv(ts_image_feature_csv)

    feat_cols = [col for col in tr_df.columns if col.startswith("ft_")]
    tr_ft = tr_df.select(feat_cols).to_numpy()
    ts_ft = ts_df.select(feat_cols).to_numpy()

    pca = Pipeline(
        (
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        )
    )
    pca.fit(tr_ft)
    tr_ft = pca.transform(tr_ft)
    ts_ft = pca.transform(ts_ft)

    pca_feat_cols = [f"image_feat_pca_{i}" for i in range(tr_ft.shape[1])]
    tr_df = tr_df.select(["ID", "image_name"]).with_columns(
        pl.from_numpy(data=tr_ft, schema=pca_feat_cols)
    )
    ts_df = ts_df.select(["ID", "image_name"]).with_columns(
        pl.from_numpy(data=ts_ft, schema=pca_feat_cols)
    )

    # image_nameをfeatureにpivot
    tr_df = tr_df.pivot(on="image_name", index="ID", values=pca_feat_cols)
    ts_df = ts_df.pivot(on="image_name", index="ID", values=pca_feat_cols)

    return tr_df, ts_df


def read_image_feature_type2_csv(
    tr_image_feature_csv: Path,
    ts_image_feature_csv: Path,
    n_components: int = 16,
    n_patch: int = 16,
    prefix: str = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    画像特徴量のCSVを読み込む
    メモリ削減のためPCAを適用

    Args:
        tr_image_feature_csv (Path): trainの画像特徴量のCSV
        ts_image_feature_csv (Path): testの画像特徴量のCSV
    """
    # columns: "ID", "image_name", "ft_0", "ft_1", ...
    tr_df = pl.read_csv(tr_image_feature_csv)
    ts_df = pl.read_csv(ts_image_feature_csv)

    feat_cols = [col for col in tr_df.columns if col.startswith("ft_")]
    tr_ft = tr_df.select(feat_cols).to_numpy().astype(np.float32)
    ts_ft = ts_df.select(feat_cols).to_numpy().astype(np.float32)

    tr_df = tr_df.select(["ID", "image_name"])
    ts_df = ts_df.select(["ID", "image_name"])

    # feat_colsは(n_patch, n_channels)をflattenしたもの
    n_channels = len(feat_cols) // n_patch
    # (n_sample * n_patch, n_channels)
    tr_ft = tr_ft.reshape(tr_ft.shape[0] * n_patch, n_channels)
    ts_ft = ts_ft.reshape(ts_ft.shape[0] * n_patch, n_channels)

    # pca
    pca = Pipeline(
        (
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        )
    )
    pca.fit(tr_ft)
    # (n_sample * n_patch, n_components)
    tr_ft = pca.transform(tr_ft)
    ts_ft = pca.transform(ts_ft)
    # (n_sample, n_patch * n_components)
    tr_ft = tr_ft.reshape(tr_ft.shape[0] // n_patch, -1)
    ts_ft = ts_ft.reshape(ts_ft.shape[0] // n_patch, -1)

    pf = prefix if prefix is not None else ""
    pca_feat_cols = [
        f"{pf}image_feat_patch_{i // n_components}_pca_{i % n_components}"
        for i in range(tr_ft.shape[1])
    ]
    tr_df = tr_df.select(["ID", "image_name"]).with_columns(
        pl.from_numpy(data=tr_ft.astype(np.float32), schema=pca_feat_cols)
    )
    ts_df = ts_df.select(["ID", "image_name"]).with_columns(
        pl.from_numpy(data=ts_ft.astype(np.float32), schema=pca_feat_cols)
    )

    # image_nameをfeatureにpivot
    tr_df = tr_df.pivot(on="image_name", index="ID", values=pca_feat_cols)
    ts_df = ts_df.pivot(on="image_name", index="ID", values=pca_feat_cols)

    return tr_df, ts_df


# ########
# target
# ########
class CoordinateTarget:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def fit(self, df: pl.DataFrame):
        return self

    def transform(self, df: pl.DataFrame):
        pf = self.prefix
        df = df.with_columns(
            pl.col("x_0").alias(f"{pf}cood_x_0"),
            pl.col("y_0").alias(f"{pf}cood_y_0"),
            pl.col("z_0").alias(f"{pf}cood_z_0"),
            pl.col("x_1").alias(f"{pf}cood_x_1"),
            pl.col("y_1").alias(f"{pf}cood_y_1"),
            pl.col("z_1").alias(f"{pf}cood_z_1"),
            pl.col("x_2").alias(f"{pf}cood_x_2"),
            pl.col("y_2").alias(f"{pf}cood_y_2"),
            pl.col("z_2").alias(f"{pf}cood_z_2"),
            pl.col("x_3").alias(f"{pf}cood_x_3"),
            pl.col("y_3").alias(f"{pf}cood_y_3"),
            pl.col("z_3").alias(f"{pf}cood_z_3"),
            pl.col("x_4").alias(f"{pf}cood_x_4"),
            pl.col("y_4").alias(f"{pf}cood_y_4"),
            pl.col("z_4").alias(f"{pf}cood_z_4"),
            pl.col("x_5").alias(f"{pf}cood_x_5"),
            pl.col("y_5").alias(f"{pf}cood_y_5"),
            pl.col("z_5").alias(f"{pf}cood_z_5"),
        )

        self.columns = [col for col in df.columns if col.startswith(self.prefix)]
        df = df.select(self.columns)
        return df


# ########
# feature
# ########
class FeatureGBDT:
    def __init__(
        self, prefix: str, n_components_depth_pca: int = 32, random_state: int = 42
    ):
        self.prefix = prefix
        self.n_components_depth_pca = n_components_depth_pca
        self.random_state = random_state

    def fit(self, df: pl.DataFrame, depth_images: np.ndarray):
        """
        Args:
            df (pl.DataFrame): 特徴量データ (n_sample, n_features)
            depth_images (np.ndarray): 深度画像データ (n_sample, image_name, height, width, 1)
        """
        # depthのpca
        # (n_sample * image_name, height * width)
        depth_images = depth_images.reshape(
            depth_images.shape[0] * depth_images.shape[1], -1
        )
        self.depth_pca = Pipeline(
            (
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=self.n_components_depth_pca,
                        random_state=self.random_state,
                    ),
                ),
            )
        )
        self.depth_pca.fit(depth_images)

        return self

    def transform(self, df: pl.DataFrame, depth_images: np.ndarray):
        """
        Args:
            df (pl.DataFrame): 特徴量データ (n_sample, n_features)
            depth_images (np.ndarray): 深度画像データ (n_sample, image_name, height, width, 1)
        """
        pf = self.prefix

        # ##############
        # df
        # ##############
        df = df.with_columns(
            pl.col("vEgo").alias(f"{pf}vEgo"),
            pl.col("aEgo").alias(f"{pf}aEgo"),
            pl.col("steeringAngleDeg").alias(f"{pf}steeringAngleDeg"),
            pl.col("steeringTorque").alias(f"{pf}steeringTorque"),
            pl.col("brake").alias(f"{pf}brake"),
            (pl.col("brakePressed") * 1.0).alias(f"{pf}brakePressed"),
            pl.col("gas").alias(f"{pf}gas"),
            (pl.col("gasPressed") * 1.0).alias(f"{pf}gasPressed"),
            ((pl.col("gearShifter") == "drive") * 1.0).alias(
                f"{pf}is_gearShifter_drive"
            ),
            ((pl.col("gearShifter") == "neutral") * 1.0).alias(
                f"{pf}is_gearShifter_neutral"
            ),
            ((pl.col("gearShifter") == "park") * 1.0).alias(f"{pf}is_gearShifter_park"),
            ((pl.col("gearShifter") == "reverse") * 1.0).alias(
                f"{pf}is_gearShifter_reverse"
            ),
            (pl.col("leftBlinker") * 1.0).alias(f"{pf}leftBlinker"),
            (pl.col("rightBlinker") * 1.0).alias(f"{pf}rightBlinker"),
        )
        tmp_feat_cols = [col for col in df.columns if col.startswith(self.prefix)]

        df = df.with_columns(
            #
            pl.col(tmp_feat_cols).shift(1).over("scene_id").name.prefix(f"{pf}prev_"),
            #
            pl.col(tmp_feat_cols).shift(-1).over("scene_id").name.prefix(f"{pf}next_"),
            #
            pl.col(tmp_feat_cols)
            .diff(1)
            .over("scene_id")
            .name.prefix(f"{pf}prev_diff_"),
            #
            pl.col(tmp_feat_cols)
            .diff(-1)
            .over("scene_id")
            .name.prefix(f"{pf}next_diff_"),
        )

        # ##############
        # depth
        # ##############
        df = df.with_columns(
            self._calc_depth_patch_features(depth_images),
            # self._calc_depth_pca_features(depth_images),
        )

        # ##############
        # image feature
        # ##############
        # image_feat_cols = [
        #    col for col in df.columns if col.startswith("type2_image_feat_")
        # ]
        # df = df.with_columns(
        #    pl.col(image_feat_cols).name.prefix(f"{pf}image_feat_"),
        # )
        df = df.with_columns(
            self._calc_image_features(df),
        )

        # ##############
        # base_pred
        # ##############
        df = df.with_columns(
            # base_pred
            pl.col("base_pred_x_0").alias(f"{pf}base_pred_x0"),
            pl.col("base_pred_y_0").alias(f"{pf}base_pred_y0"),
            pl.col("base_pred_z_0").alias(f"{pf}base_pred_z0"),
            pl.col("base_pred_x_1").alias(f"{pf}base_pred_x1"),
            pl.col("base_pred_y_1").alias(f"{pf}base_pred_y1"),
            pl.col("base_pred_z_1").alias(f"{pf}base_pred_z1"),
            pl.col("base_pred_x_2").alias(f"{pf}base_pred_x2"),
            pl.col("base_pred_y_2").alias(f"{pf}base_pred_y2"),
            pl.col("base_pred_z_2").alias(f"{pf}base_pred_z2"),
            pl.col("base_pred_x_3").alias(f"{pf}base_pred_x3"),
            pl.col("base_pred_y_3").alias(f"{pf}base_pred_y3"),
            pl.col("base_pred_z_3").alias(f"{pf}base_pred_z3"),
            pl.col("base_pred_x_4").alias(f"{pf}base_pred_x4"),
            pl.col("base_pred_y_4").alias(f"{pf}base_pred_y4"),
            pl.col("base_pred_z_4").alias(f"{pf}base_pred_z4"),
            pl.col("base_pred_x_5").alias(f"{pf}base_pred_x5"),
            pl.col("base_pred_y_5").alias(f"{pf}base_pred_y5"),
            pl.col("base_pred_z_5").alias(f"{pf}base_pred_z5"),
        )

        # ##############
        # col
        # ##############
        self.columns = [col for col in df.columns if col.startswith(self.prefix)]

        df = df.select(self.columns)

        return df

    def _image_to_patch(
        self, images: np.array, n_patch_h: int, n_path_w: int
    ) -> np.array:
        """
        画像データをパッチごとに分割する

        Args:
            images (np.array): 画像データ (n_sample, height, width, channel)
            n_patch_h (int): 高さ方向のパッチ数
            n_path_w (int): 幅方向のパッチ

        Returns:
            np.array: パッチごとの画像データ (n_sample, n_patch, patch_h, patch_w, channel)
        """
        n_sample, height, width, channel = images.shape
        patch_h = height // n_patch_h
        patch_w = width // n_path_w

        images = images.reshape(
            n_sample, n_patch_h, patch_h, n_path_w, patch_w, channel
        )
        images = images.transpose(0, 1, 3, 2, 4, 5)
        images = images.reshape(n_sample, -1, patch_h, patch_w, channel)
        return images

    def _calc_depth_pca_features(self, depth_images: np.array) -> pl.DataFrame:
        """
        深度画像データからPCA特徴量を計算する

        Args:
            depth_images (np.array): 深度画像データ (n_sample, image_name, height, width, 1)
        """
        pf = self.prefix

        pca_fts = []
        for i_image in range(depth_images.shape[1]):
            depth = depth_images[:, i_image, :, :, 0]
            depth = depth.reshape(depth.shape[0], -1)
            # (n_sample, n_components)
            pca_ft = self.depth_pca.transform(depth)
            pca_fts.append(pca_ft)
        # (n_sample, n_image * n_components)
        pca_fts = np.concatenate(pca_fts, axis=1)

        cols = [f"{pf}depth_pca_{i}" for i in range(pca_fts.shape[1])]

        features = pl.from_numpy(data=pca_fts, schema=cols)
        return features

    def _calc_depth_patch_features(self, depth_images: np.array) -> pl.DataFrame:
        """
        深度画像データから特徴量を計算する

        Args:
            depth_images (np.array): 深度画像データ (n_sample, image_name, height, width, 1)
        """
        pf = self.prefix

        # (n_sample, height, width, image_name * 1)
        depth_images = depth_images.transpose(0, 2, 3, 1, 4).reshape(
            depth_images.shape[0], depth_images.shape[2], depth_images.shape[3], -1
        )
        n_path_h = 4
        n_path_w = 4
        # (n_sample, n_patch, patch_h, patch_w, image_name * 1)
        depth_images = self._image_to_patch(depth_images, n_path_h, n_path_w)

        # (n_sample, n_patch, patch_h * patch_w, channel)
        depth_images = depth_images.reshape(
            depth_images.shape[0], depth_images.shape[1], -1, depth_images.shape[4]
        )

        features_ = {}
        features_[f"{pf}depth_patch_mean"] = depth_images.mean(axis=2)
        features_[f"{pf}depth_patch_std"] = depth_images.std(axis=2)
        features_[f"{pf}depth_patch_max"] = depth_images.max(axis=2)
        features_[f"{pf}depth_patch_min"] = depth_images.min(axis=2)

        # 特徴を追加していく
        features = {}

        # 1つ目のチャンネルの特徴量を使う
        for ft_name, ft in features_.items():
            # ft: (n_sample, n_patch, channel)
            for i_patch in range(ft.shape[1]):
                i_channel = 0
                # (n_sample,)
                features[f"{ft_name}_p{i_patch}-c{i_channel}"] = ft[
                    :, i_patch, i_channel
                ]

        # 次のチャンネルとの差分を特徴量に追加
        for ft_name, ft in features_.items():
            # ft: (n_sample, n_patch, channel)
            for i_patch in range(ft.shape[1]):
                for i_channel in range(ft.shape[2] - 1):
                    # (n_sample,)
                    features[f"{ft_name}_p{i_patch}-c{i_channel}_ch-diff"] = (
                        ft[:, i_patch, i_channel] - ft[:, i_patch, i_channel + 1]
                    )

        features = pl.DataFrame(features)
        return features

    def _calc_image_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        画像特徴量を計算する

        Args:
            df (pl.DataFrame): 特徴量データ (n_sample, n_features)
        """
        pf = self.prefix
        col_pf = "type2_image_feat_"

        # 画像特徴の列名
        # "type2_image_feat_patch_{patch_number}_pca_{pca_number}_{image_name}"という列名になっている
        image_feat_cols = [
            col for col in df.columns if col.startswith("type2_image_feat_")
        ]
        df = df.select(["scene_id"] + image_feat_cols)

        # 時系列順の画像名
        image_name0, image_name1, image_name2 = [
            "image_t.png",
            "image_t-0.5.png",
            "image_t-1.0.png",
        ]

        image_name0_feat_cols = [
            col for col in image_feat_cols if col.endswith(image_name0)
        ]
        # パッチ数
        n_patch = (
            max(
                [
                    int(col[: -len(image_name0)].split("_")[-4])
                    for col in image_name0_feat_cols
                ]
            )
            + 1
        )
        # PCA数
        n_pca = (
            max(
                [
                    int(col[: -len(image_name0)].split("_")[-2])
                    for col in image_name0_feat_cols
                ]
            )
            + 1
        )

        # image_name0の値の特徴量
        df = df.with_columns(
            [
                pl.col(f"{col_pf}patch_{i}_pca_{j}_{image_name0}").alias(
                    f"{pf}{col_pf}patch_{i}_pca_{j}_{image_name0}"
                )
                for i in range(n_patch)
                for j in range(n_pca)
            ]
        )

        # image_name0 - image_name1の差分の特徴量
        df = df.with_columns(
            [
                (
                    pl.col(f"{col_pf}patch_{i}_pca_{j}_{image_name0}")
                    - pl.col(f"{col_pf}patch_{i}_pca_{j}_{image_name1}")
                ).alias(
                    f"{pf}{col_pf}patch_{i}_pca_{j}_diff({image_name0},{image_name1})"
                )
                for i in range(n_patch)
                for j in range(n_pca)
            ]
        )

        # image_name1 - image_name2の差分の特徴量
        df = df.with_columns(
            [
                (
                    pl.col(f"{col_pf}patch_{i}_pca_{j}_{image_name1}")
                    - pl.col(f"{col_pf}patch_{i}_pca_{j}_{image_name2}")
                ).alias(
                    f"{pf}{col_pf}patch_{i}_pca_{j}_diff({image_name1},{image_name2})"
                )
                for i in range(n_patch)
                for j in range(n_pca)
            ]
        )

        ft_cols = [col for col in df.columns if col.startswith(self.prefix)]
        df = df.select(ft_cols)

        return df


class FeatureRidge:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def fit(self, df: pl.DataFrame):
        return self

    def transform(self, df: pl.DataFrame):
        pf = self.prefix
        df = df.with_columns(
            pl.col("vEgo").alias(f"{pf}vEgo"),
            # pl.col("aEgo").alias(f"{pf}aEgo"),
            # pl.col("steeringAngleDeg").alias(f"{pf}steeringAngleDeg"),
            # pl.col("steeringTorque").alias(f"{pf}steeringTorque"),
            # pl.col("brake").alias(f"{pf}brake"),
            # (pl.col("brakePressed") * 1.0).alias(f"{pf}brakePressed"),
            # pl.col("gas").alias(f"{pf}gas"),
            # (pl.col("gasPressed") * 1.0).alias(f"{pf}gasPressed"),
            # ((pl.col("gearShifter") == "drive") * 1.0).alias(
            #    f"{pf}is_gearShifter_drive"
            # ),
            # ((pl.col("gearShifter") == "neutral") * 1.0).alias(
            #    f"{pf}is_gearShifter_neutral"
            # ),
            # ((pl.col("gearShifter") == "park") * 1.0).alias(f"{pf}is_gearShifter_park"),
            # ((pl.col("gearShifter") == "reverse") * 1.0).alias(
            #    f"{pf}is_gearShifter_reverse"
            # ),
            # (pl.col("leftBlinker") * 1.0).alias(f"{pf}leftBlinker"),
            # (pl.col("rightBlinker") * 1.0).alias(f"{pf}rightBlinker"),
        )

        self.columns = [col for col in df.columns if col.startswith(self.prefix)]

        df = df.select(self.columns)

        return df


# ########
# model
# ########
class ImageNormalizer:
    """
    画像データの正規化
    """

    def __init__(self):
        pass

    def fit(self, images: np.ndarray):
        """
        画像データの正規化

        Args:
            images (np.ndarray): 画像データ (n_sample, channel, height, width)
        """
        n_channels = images.shape[1]

        # channelごとに正規化
        self.m_ = []
        self.s_ = []
        for i_channel in range(n_channels):
            channel_imgs = images[:, i_channel, :, :]

            # 1枚の画像に対して画素値のユニークな値の数をカウントする。
            # 3つ以上の値があれば平均と標準偏差で正規化
            # 後述の全画像に対する処理は遅いので、まずは1枚の画像に対して処理を行う
            n_unique = len(np.unique(channel_imgs[0]))
            if n_unique > 2:
                m = np.mean(channel_imgs)
                s = np.std(channel_imgs)
                self.m_.append([[m]])
                self.s_.append([[s]])
                continue

            # 全画像に対して画素値のユニークな値の数をカウントし、その値によって正規化方法を変える
            n_unique = len(np.unique(channel_imgs))

            if n_unique > 2:
                # 画素値が3種類以上は平均と標準偏差で正規化
                m = np.mean(channel_imgs)
                s = np.std(channel_imgs)
            elif n_unique == 2:
                # 画素値が2種類の場合は最小値と最大値で正規化
                m = np.min(channel_imgs)
                s = np.max(channel_imgs) - m
            else:
                # 画素値が1種類の場合は正規化しない
                m = np.mean(channel_imgs)
                s = 1.0

            self.m_.append([[m]])
            self.s_.append([[s]])

        self.m_ = np.array(self.m_)
        self.s_ = np.array(self.s_)
        return self

    def transform(self, images: np.ndarray):
        """
        画像データの正規化

        Args:
            images (np.ndarray): 画像データ (n_sample, channel, height, width)
        """
        images = (images - self.m_) / self.s_
        return images

    def transform_single(self, image: np.ndarray):
        """
        画像データの正規化

        Args:
            image (np.ndarray): 画像データ (channel, height, width)
        """
        image = (image - self.m_) / self.s_
        return image


class Scaler(RobustScaler):
    def fit(self, df: pl.DataFrame, y=None):
        super().fit(df)

        # ユニークな値の数によって、スケーリングの方法を変える
        n_uniques = df.select(pl.all().n_unique()).to_dicts()[0]
        for i_col, (col, n_unique) in enumerate(n_uniques.items()):
            if n_unique > 2:
                continue
            elif n_unique == 1:
                # すべて同じ値の場合は、中央値を0、スケールを1にする
                self.center_[i_col] = float(df.get_column(col).mean())
                self.scale_[i_col] = 1.0
                continue
            elif n_unique == 2:
                # 2値の場合は、最小値を0、最大値を1にする
                self.center_[i_col] = float(df.get_column(col).min())
                self.scale_[i_col] = float(
                    df.get_column(col).max() - df.get_column(col).min()
                )
                continue
        return self


class LgbModel:
    def __init__(self, model_params: dict, fit_params: dict):
        self.model_params = model_params
        self.fit_params = fit_params

    def fit(
        self,
        tr_features: pl.DataFrame,
        tr_targets: pl.DataFrame,
        vl_features: pl.DataFrame,
        vl_targets: pl.DataFrame,
    ):
        """
        モデルの学習

        Args:
            features (pl.DataFrame): 特徴量データ (n_sample, n_features)
            targets (pl.DataFrame): ターゲットデータ (n_sample, n_targets)
        """

        self.tg_cols = deepcopy(tr_targets.columns)
        self.feature_names = tr_features.columns

        # target毎に学習
        self.models = []

        for tg_col in self.tg_cols:
            print("-----------------")
            print(f"*** Training target {tg_col}... ***")
            tr_tg = tr_targets.get_column(tg_col).to_numpy()
            vl_tg = vl_targets.get_column(tg_col).to_numpy()

            # dataset
            tr_ds = lgb.Dataset(tr_features, tr_tg)
            vl_ds = lgb.Dataset(vl_features, vl_tg, reference=tr_ds)

            callbacks = [
                lgb.log_evaluation(
                    period=self.fit_params["gbdt"]["num_boost_round"] // 20
                ),
                lgb.early_stopping(
                    stopping_rounds=self.fit_params["gbdt"]["early_stopping_rounds"],
                ),
            ]

            model = lgb.train(
                self.model_params["gbdt"],
                tr_ds,
                num_boost_round=self.fit_params["gbdt"]["num_boost_round"],
                valid_sets=[tr_ds, vl_ds],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )

            self.models.append(model)

        return self

    def predict(
        self,
        features: pl.DataFrame,
    ) -> np.ndarray:
        """
        予測
        """
        preds = []

        for model in self.models:
            # (n_sample,)
            pred = model.predict(features)
            preds.append(pred)

        # (n_sample, n_targets)
        preds = np.stack(preds, axis=1)
        return preds


class RidgeModel:
    def __init__(self, model_params: dict, fit_params: dict):
        self.model_params = model_params
        self.fit_params = fit_params

    def fit(
        self,
        tr_features: pl.DataFrame,
        tr_targets: pl.DataFrame,
        vl_features: pl.DataFrame,
        vl_targets: pl.DataFrame,
    ):
        """
        モデルの学習

        Args:
            features (pl.DataFrame): 特徴量データ (n_sample, n_features)
            targets (pl.DataFrame): ターゲットデータ (n_sample, n_targets)
        """

        self.tg_cols = deepcopy(tr_targets.columns)
        self.feature_names = tr_features.columns

        self.model = Pipeline(
            (
                ("scaler", Scaler()),
                ("ridge", Ridge(**self.model_params["ridge"])),
            )
        )
        self.model.fit(tr_features, tr_targets)

        return self

    def predict(
        self,
        features: pl.DataFrame,
    ) -> np.ndarray:
        """
        予測
        """
        preds = self.model.predict(features)
        return preds


def train(
    model_class: object,
    model_params: dict,
    fit_params: dict,
    df: pl.DataFrame,
    target_cols: list[str],
    feature_cols: list[str],
    group_col: str,
    n_splits: int,
):
    df = df.select(list(set(feature_cols + target_cols + [group_col])))

    models = []
    oof_preds = np.full((len(df), len(target_cols)), np.nan)

    group_kfold = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, vl_idx) in enumerate(
        group_kfold.split(df, groups=df.get_column(group_col))
    ):
        print("-----------------")
        print("-----------------")
        print(f"Training fold {fold}...")
        print(f"train samples: {len(tr_idx)}, valid samples: {len(vl_idx)}")

        tr_df = df[tr_idx]
        vl_df = df[vl_idx]

        model = model_class(model_params, fit_params)

        model.fit(
            tr_features=tr_df.select(feature_cols),
            tr_targets=tr_df.select(target_cols),
            vl_features=vl_df.select(feature_cols),
            vl_targets=vl_df.select(target_cols),
        )
        models.append(model)

        pred = model.predict(
            vl_df.select(feature_cols),
        )
        oof_preds[vl_idx] = pred

    oof_preds = pl.DataFrame(oof_preds, schema=target_cols)

    return models, oof_preds


def predict(
    models: list,
    df: pl.DataFrame,
    feature_cols: list[str],
    pred_cols: list[str],
):
    preds = []
    for model in models:
        pred = model.predict(df.select(feature_cols))
        preds.append(pred)
    preds = np.mean(preds, axis=0)

    preds = pl.from_numpy(preds, schema=pred_cols)
    return preds


# ########
# evaluation
# ########


def plot_lgb_importance(models: list[object], feature_names: list[str]) -> None:
    """LightGBMモデルの平均特徴量重要度をプロットする関数。

    各モデルの特徴量重要度の平均と標準偏差を計算し、横棒グラフ（barh）で表示します。

    Args:
        models (List[lgb.Booster]): 同じ特徴量を持つLightGBMモデルのリスト。

    Returns:
        None
    """
    if not models:
        raise ValueError("モデルリストが空です。")

    # 各モデルの特徴量重要度を収集
    importances = []
    for model in models:
        importance = model.feature_importance(
            importance_type="gain"
        )  # 'gain'で重要度を取得
        importances.append(importance)

    # データフレームに変換
    df_importances = pd.DataFrame(importances, columns=feature_names)

    # 平均と標準偏差を計算
    mean_importance = df_importances.mean(axis=0)
    std_importance = df_importances.std(axis=0)

    # プロット用のデータフレームを作成
    importance_df = pd.DataFrame(
        {"mean": mean_importance, "std": std_importance}, index=feature_names
    )

    # 平均重要度でソート
    importance_df = importance_df.sort_values(by="mean")

    # 横棒グラフをプロット
    plt.figure(figsize=(7, len(importance_df) / 4))
    plt.barh(
        y=importance_df.index,
        width=importance_df["mean"],
        xerr=importance_df["std"],
        align="center",
        alpha=0.8,
    )
    plt.xlabel("importance")
    plt.title("Importance of features")
    plt.tight_layout()
    plt.show()


def calc_score(df: pl.DataFrame, pred_cols: list[str]):
    tg_cols = sum([[f"x_{i}", f"y_{i}", f"z_{i}"] for i in range(6)], [])

    tg = df.select(tg_cols).to_numpy()
    pred = df.select(pred_cols).to_numpy()

    scores = np.abs(tg - pred).mean(axis=0)
    scores = {f"score_{col}": float(score) for col, score in zip(pred_cols, scores)}
    scores["avg"] = float(np.abs(tg - pred).mean())
    return scores


def _create_ax_calibration_curve(
    ax: plt.Axes,
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    n_bins: int = 30,
) -> plt.Axes:
    """
    キャリブレーションカーブのAxesを作成する

    Args:
        ax (plt.Axes): Axes
        df (pl.DataFrame): データフレーム
        x_col (str): x軸のカラム名
        y_col (str): y軸のカラム名
        n_bins (int): ビンの数

    Returns:
        plt.Axes: Axes
    """
    x_expr = pl.col(x_col)

    df = df.select([x_expr, y_col])
    df = df.with_columns(
        ((pl.col(x_col).rank("min") - 1) // (len(df) / n_bins))
        .clip(0, n_bins - 1)
        .alias("bin")
    )

    bin_df = (
        df.group_by("bin", maintain_order=True)
        .agg(
            pl.col(x_col).mean().alias(x_col),
            pl.col(y_col).mean().alias(y_col),
        )
        .sort("bin")
    )

    # calibration curve
    ax.plot(bin_df.get_column(x_col), bin_df.get_column(y_col), color="red", marker="o")
    ax.plot(bin_df.get_column(x_col), bin_df.get_column(x_col), color="black")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"mean({y_col})")

    # 第二軸にヒストグラム
    ax2 = ax.twinx()
    ax2.hist(df.get_column(x_col), bins=n_bins, alpha=0.5)
    ax2.set_ylabel("Frequency")

    return ax


def plot_calibration_curve(df: pl.DataFrame, pred_cols: list[str], n_bins: int = 10):
    tg_cols = sum([[f"x_{i}", f"y_{i}", f"z_{i}"] for i in range(6)], [])

    fig, axs = plt.subplots(6, 3, figsize=(3 * 4, 6 * 2))

    for i, (tg_col, pred_col) in enumerate(zip(tg_cols, pred_cols)):
        ax = axs[i // 3, i % 3]
        _create_ax_calibration_curve(ax, df, pred_col, tg_col, n_bins=n_bins)

    plt.tight_layout()
    plt.show()
