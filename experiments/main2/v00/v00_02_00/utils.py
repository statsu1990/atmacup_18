from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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
    def __init__(self, prefix: str):
        self.prefix = prefix

    def fit(self, df: pl.DataFrame):
        return self

    def transform(self, df: pl.DataFrame):
        pf = self.prefix
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

        self.columns = [col for col in df.columns if col.startswith(self.prefix)]

        df = df.select(self.columns)

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
    plt.figure(figsize=(7, len(importance_df) / 2))
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
