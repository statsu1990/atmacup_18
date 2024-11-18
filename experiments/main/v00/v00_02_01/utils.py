import inspect
import os
import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import segmentation_models_pytorch as smp
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


# ########
# utils
# ########
def _get_default_args(cls) -> dict:
    """
    __init__メソッドのシグネチャを取得
    """
    signature = inspect.signature(cls.__init__)
    # デフォルト値が設定されている引数を取得
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)


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
        pl.col("ID").str.split("_").list.get(1).alias("scene_dsec"),
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


def preprocess_images(camera_images: np.ndarray) -> np.ndarray:
    """
    画像の前処理

    Args:
        camera_images (np.ndarray): (sample, image_type, height, width, channel)
    """
    # (sample, image_type, channel, height, width)
    camera_images = camera_images.transpose(0, 1, 4, 2, 3)
    # (sample, image_type * channel, height, width)
    camera_images = camera_images.reshape(
        camera_images.shape[0], -1, camera_images.shape[3], camera_images.shape[4]
    )

    return camera_images


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
class Feature:
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


class Scaler(RobustScaler):
    def fit(self, df: pl.DataFrame):
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


class TargetScaler:
    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def fit(self, df: pl.DataFrame):
        n_targets = len(df.columns)

        # [n_target,]
        self.min_ = df.min().to_numpy()[0]
        self.max_ = df.max().to_numpy()[0]

        margin = (self.max_ - self.min_) * self.margin
        self.min_ = self.min_ - margin
        self.max_ = self.max_ + margin

        self.weights = np.array([1.0] * n_targets)
        # self.weights = self.max_ - self.min_

        return self

    def transform(self, df: pl.DataFrame):
        df = df.to_numpy()
        df = (df - self.min_) / (self.max_ - self.min_)
        return df

    def inverse_transform(self, y: np.ndarray):
        y = y * (self.max_ - self.min_) + self.min_
        return y


class CnnModel:
    def __init__(self, model_params: dict, fit_params: dict):
        self.model_params = model_params
        self.fit_params = fit_params
        self.dev = model_params["dev"]

    def fit(
        self,
        tr_imgs: np.ndarray,
        tr_features: pl.DataFrame,
        tr_targets: pl.DataFrame,
        vl_imgs: np.ndarray,
        vl_features: pl.DataFrame,
        vl_targets: pl.DataFrame,
    ):
        """
        モデルの学習

        Args:
            images (np.ndarray): 画像データ (n_sample, channel, height, width)
            features (pl.DataFrame): 特徴量データ (n_sample, n_features)
            targets (pl.DataFrame): ターゲットデータ (n_sample, n_targets)
        """
        # feature scaling
        self.feat_scaler = Scaler()
        self.feat_scaler.fit(tr_features)
        tr_features = self.feat_scaler.transform(tr_features)
        vl_features = self.feat_scaler.transform(vl_features)

        # image scaling
        self.img_scaler = ImageNormalizer()
        self.img_scaler.fit(tr_imgs)
        tr_imgs = self.img_scaler.transform(tr_imgs)
        vl_imgs = self.img_scaler.transform(vl_imgs)

        # target scaling
        self.target_scaler = TargetScaler(
            margin=self.fit_params["target_scaler"]["margin"]
        )
        self.target_scaler.fit(tr_targets)
        tr_targets = self.target_scaler.transform(tr_targets)
        vl_targets = self.target_scaler.transform(vl_targets)
        self.fit_params["dnn"]["trainer_params"]["criterion_params"]["weights"] = (
            self.target_scaler.weights.copy()
        )

        # dataset
        tr_ds = AtmaCup18Dataset(tr_imgs, tr_features, tr_targets)
        vl_ds = AtmaCup18Dataset(vl_imgs, vl_features, vl_targets)

        # dataloader
        tr_dl = _get_dataloader(
            tr_ds,
            batch_size=self.fit_params["dnn"]["tr_batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        vl_dl = _get_dataloader(
            vl_ds,
            batch_size=self.fit_params["dnn"]["vl_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # model
        self.dnn_model = DNN(
            **self.model_params["dnn"],
        )

        # trainer
        self.trainer = Trainer(**self.fit_params["dnn"]["trainer_params"])
        self.trainer.run(self.dnn_model, tr_dl, vl_dl)

        return self

    def predict(self, images: np.ndarray, features: pl.DataFrame) -> np.ndarray:
        """
        予測
        """
        # scaling
        features = self.feat_scaler.transform(features)
        images = self.img_scaler.transform(images)

        # dataset
        ds = AtmaCup18Dataset(images, features, targets=None)
        dl = _get_dataloader(
            ds,
            batch_size=self.fit_params["dnn"]["vl_batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # predict
        predictor = Predictor(dev="cuda", do_print=False)
        preds = predictor.run(self.dnn_model, dl)

        # inverse scaling
        preds = self.target_scaler.inverse_transform(preds)
        return preds


def train(
    model_params: dict,
    fit_params: dict,
    df: pl.DataFrame,
    images: np.ndarray,
    target_cols: list[str],
    feature_cols: list[str],
    group_col: str,
    n_splits: int,
):
    df = df.select(feature_cols + target_cols + [group_col])

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
        tr_imgs = images[tr_idx]
        vl_imgs = images[vl_idx]

        model = CnnModel(model_params, fit_params)

        model.fit(
            tr_imgs=tr_imgs,
            tr_features=tr_df.select(feature_cols),
            tr_targets=tr_df.select(target_cols),
            vl_imgs=vl_imgs,
            vl_features=vl_df.select(feature_cols),
            vl_targets=vl_df.select(target_cols),
        )
        models.append(model)

        pred = model.predict(vl_imgs, vl_df.select(feature_cols))
        oof_preds[vl_idx] = pred

    oof_preds = pl.DataFrame(oof_preds, schema=target_cols)

    return models, oof_preds


def predict(
    models: list,
    images: np.ndarray,
    df: pl.DataFrame,
    feature_cols: list[str],
    pred_cols: list[str],
):
    preds = []
    for model in models:
        pred = model.predict(images, df.select(feature_cols))
        preds.append(pred)
    preds = np.mean(preds, axis=0)

    preds = pl.from_numpy(preds, schema=pred_cols)
    return preds


# dnn architecture
class DNN(nn.Module):
    def __init__(
        self,
        n_img_channels: int,
        n_features: int,
        n_targets: int,
        segmentation_model_params: dict,
        n_bins: int,
    ):
        super().__init__()
        # backbone
        self.backbone = BackBone(n_img_channels, n_features, segmentation_model_params)
        self.last_decoder_channels = self.backbone.last_decoder_channels

        # head
        # self.head = Head(self.last_decoder_channels, n_targets)
        self.head = BinRegHead(self.last_decoder_channels, n_targets, n_bins)

    def forward(
        self, x: torch.Tensor, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (batch, channels, h, w)
            features (torch.Tensor): (batch, feat)

        Returns:
            dict[str, torch.Tensor]: {output_name: output_tensor}
        """
        # backbone
        h = self.backbone(x, features)

        # head
        h = self.head(h)
        return h


class BackBone(nn.Module):
    def __init__(
        self, n_img_channels: int, n_features: int, segmentation_model_params: dict
    ):
        super().__init__()
        # segmentation model
        segmentation_model_params = deepcopy(segmentation_model_params)
        segmentation_model_params["in_channels"] = n_img_channels

        self.sg_model = smp.Unet(**segmentation_model_params)
        self.last_encoder_channels = self.sg_model.encoder.out_channels[-1]
        self.last_decoder_channels = segmentation_model_params["decoder_channels"][-1]

        # feature
        self.ft_fc = nn.Linear(n_features, self.last_encoder_channels)

        # last_channels
        if "decoder_channels" in segmentation_model_params:
            self.last_channels = segmentation_model_params["decoder_channels"][-1]
        else:
            self.last_channels = _get_default_args(smp.Unet)["decoder_channels"][-1]

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch, channels, h, w)
            features (torch.Tensor): (batch, feat)

        Returns:
            torch.Tensor: (batch, channels, h, w)
        """
        # encoder
        # [h_layer1, h_layer2, ..., h_layerN]
        hs = self.sg_model.encoder(x)

        # feature
        # (batch, feat, 1, 1)
        features = self.ft_fc(features).unsqueeze(-1).unsqueeze(-1)
        hs[-1] = hs[-1] + features

        # decoder
        # (batch, channels, h, w)
        h = self.sg_model.decoder(*hs)

        return h


class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (batch, channels, h, w)

        Returns:
            torch.Tensor: (batch, target)
        """
        # (batch, channels)
        h = x.mean(dim=(2, 3))
        h = self.linear(h)

        return h


class BinRegHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_bins: int):
        super().__init__()
        self.out_channels = out_channels
        self.n_bins = n_bins
        self.linear = nn.Linear(in_channels, n_bins * out_channels)

        # (bin, 1)
        bin_values = torch.linspace(0, 1, n_bins).unsqueeze(-1)
        self.register_buffer("bin_values", bin_values)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): (batch, channels, h, w)

        Returns:
            torch.Tensor: (batch, target)
        """
        # (batch, channels)
        h = x.mean(dim=(2, 3))
        # (batch, bin, target)
        bin_logit = self.linear(h).view(-1, self.n_bins, self.out_channels)

        # (batch, bin, target)
        bin_proba = F.softmax(bin_logit, dim=1)
        # (batch, target)
        avg_val = torch.sum(bin_proba * self.bin_values, dim=1)

        outputs = {}
        outputs["bin_logit"] = bin_logit
        outputs["avg_val"] = avg_val

        return outputs


# dnn data
class AtmaCup18Loss(nn.Module):
    def __init__(
        self,
        weights: list[float] = None,
        ref_bin_proba_sigma: float = 0.05,
        dev: str = "cuda",
    ):
        super().__init__()
        weights = (
            torch.tensor(weights, dtype=torch.float32, device=dev)
            if weights is not None
            else None
        )
        self.register_buffer("weights", weights)

        self.ref_bin_proba_sigma = ref_bin_proba_sigma

    def forward(
        self, outputs: dict[str, torch.Tensor], target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            output (dict[str, torch.Tensor]): {output_name: output_tensor}
            target (torch.Tensor): (batch, target)

        Returns:
            torch.Tensor: loss
            dict[str, float]: loss dict
        """
        loss = 0
        loss_dict = {}

        # avg val loss
        ploss = self._avg_val_loss(outputs["avg_val"], target, self.weights)
        loss = loss + ploss
        loss_dict["loss_avg_val"] = ploss.item()

        # bin logit loss
        ploss = self._bin_logit_loss(outputs["bin_logit"], target, self.weights)
        loss = loss + ploss
        loss_dict["loss_bin_logit"] = ploss.item()
        return loss, loss_dict

    def _bin_logit_loss(
        self,
        bin_logit: torch.Tensor,
        target: torch.Tensor,
        target_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            bin_logit (torch.Tensor): (batch, bin, target)
            target (torch.Tensor): (batch, target)
            target_weights (torch.Tensor): (target,)

        Returns:
            torch.Tensor: loss
        """
        n_bins = bin_logit.size(1)

        # (bin,)
        ref_bin_proba = torch.linspace(0, 1, n_bins, device=bin_logit.device)
        # (batch, bin, target)
        ref_bin_proba = (
            -0.5
            * (
                (target.unsqueeze(1) - ref_bin_proba.unsqueeze(-1))
                / self.ref_bin_proba_sigma
            )
            ** 2
        )
        ref_bin_proba = F.softmax(ref_bin_proba, dim=1)

        # nll
        # -ref_proba * log(pred_proba)
        # (batch, bin, target)
        loss = -ref_bin_proba * F.log_softmax(bin_logit, dim=1)
        # (batch, target)
        loss = loss.sum(dim=1)

        if target_weights is not None:
            # (batch, target)
            loss = loss * target_weights

        loss = loss.mean()

        return loss

    def _avg_val_loss(
        self,
        avg_val: torch.Tensor,
        target: torch.Tensor,
        target_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            bin_logit (torch.Tensor): (batch, target)
            target (torch.Tensor): (batch, target)
            target_weights (torch.Tensor): (target,)

        Returns:
            torch.Tensor: loss
        """
        # (batch, target)
        loss = F.mse_loss(avg_val, target, reduction="none")

        if target_weights is not None:
            # (batch, target)
            loss = loss * target_weights

        loss = loss.mean()

        return loss


class AtmaCup18Dataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, features: np.ndarray, targets: np.ndarray):
        """
        Args:
            images (np.ndarray): 画像データ (n_sample, channel, height, width)
            features (np.ndarray): 特徴量データ (n_sample, n_features)
            targets (np.ndarray): ターゲットデータ (n_sample, n_targets)
        """
        self.images = images.astype(np.float32)
        self.features = features.astype(np.float32)
        if targets is not None:
            self.targets = targets.astype(np.float32)
        else:
            self.targets = None

        # validation
        if len(self.images) != len(self.features):
            raise ValueError(
                f"images ({len(self.images)}) != features ({len(self.features)})"
            )
        if self.targets is not None:
            if len(self.images) != len(self.targets):
                raise ValueError(
                    f"images ({len(self.images)}) != targets ({len(self.targets)})"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self._getitem(idx)

    def _getitem(self, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            idx (int): datasetのindex

        Returns:
            images (np.ndarray): 画像データ (channel, height, width)
            features (np.ndarray): 特徴量データ (n_features)
            targets (np.ndarray): ターゲットデータ (n_targets)。ターゲットがない場合は返されない
        """
        img = self.images[idx]
        feat = self.features[idx]

        if self.targets is None:
            return img, feat

        target = self.targets[idx]

        return img, feat, target


# dnn training
def _get_dataloader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
    )


class Trainer:
    def __init__(
        self,
        criterion_params: dict,
        opt: str,
        opt_params: dict,
        backbone_opt_params: dict,
        sch_params: dict,
        epochs: int,
        dev: str,
        prefix: str = "",
        save_best: bool = True,
        save_epochs: list[int] = None,
        maximize_score: bool = True,
        grad_max_norm: float = None,
    ):
        """
        Args:
            criterion_params (dict): criterionのパラメーター
            opt (str): "Adam", "AdamW", ...
            opt_params (dict): optimizerのパラメーター
            backbone_opt_params (dict): backboneのoptimizerのパラメーター
            sch_params (dict): schedulerのパラメーター
            epochs (int): エポック数
            dev (str): "cuda", "cpu"
            prefix (str): 保存するモデルのprefix
            save_best (bool): 最良のモデルを保存するかどうか
            save_epochs (list of int): 保存するエポック数
            maximize_score (bool): スコアを最大化するかどうか
            grad_max_norm (float): 勾配の最大ノルム
        """
        self.criterion = AtmaCup18Loss(**criterion_params)
        self.opt_params = opt_params
        self.backbone_opt_params = backbone_opt_params
        self.opt = _get_optimizer(opt)

        self.sch_params = sch_params
        self.scheduler = (
            torch.optim.lr_scheduler.OneCycleLR if sch_params is not None else None
        )

        self.epochs = epochs
        self.grad_max_norm = grad_max_norm

        self.save_best = save_best
        self.save_epochs = save_epochs if save_epochs is not None else []
        self.maximize_score = maximize_score

        self.dev = dev
        self.prefix = prefix

    def _get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        optimizer設定

        Args:
            model (nn.Module): model
        """
        if self.backbone_opt_params is None:
            return self.opt(model.parameters(), **self.opt_params)

        backbone_w_weightdecay_params = []
        backbone_wo_weightdecay_params = []
        another_w_weightdecay_params = []
        another_wo_weightdecay_params = []

        # backboneとそれ以外で最適化パラメータを分ける
        # biasはweight decayをかけない
        no_decay_name = ["bias"]
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                if any(nd in name for nd in no_decay_name):
                    backbone_wo_weightdecay_params.append(param)
                else:
                    backbone_w_weightdecay_params.append(param)
            else:
                if any(nd in name for nd in no_decay_name):
                    another_wo_weightdecay_params.append(param)
                else:
                    another_w_weightdecay_params.append(param)

        opt_params_wo_weightdecay = self.opt_params.copy()
        opt_params_wo_weightdecay["weight_decay"] = 0.0
        backbone_opt_params_wo_weightdecay = self.backbone_opt_params.copy()
        backbone_opt_params_wo_weightdecay["weight_decay"] = 0.0

        if self.opt is SAM:
            opt = self.opt(
                [
                    dict(**{"params": another_w_weightdecay_params}, **self.opt_params),
                    dict(
                        **{"params": another_wo_weightdecay_params},
                        **opt_params_wo_weightdecay,
                    ),
                    dict(
                        **{"params": backbone_w_weightdecay_params},
                        **self.backbone_opt_params,
                    ),
                    dict(
                        **{"params": backbone_wo_weightdecay_params},
                        **backbone_opt_params_wo_weightdecay,
                    ),
                ],
                base_optimizer=self.opt_params["base_optimizer"],
            )
        else:
            opt = self.opt(
                [
                    dict(**{"params": another_w_weightdecay_params}, **self.opt_params),
                    dict(
                        **{"params": another_wo_weightdecay_params},
                        **opt_params_wo_weightdecay,
                    ),
                    dict(
                        **{"params": backbone_w_weightdecay_params},
                        **self.backbone_opt_params,
                    ),
                    dict(
                        **{"params": backbone_wo_weightdecay_params},
                        **backbone_opt_params_wo_weightdecay,
                    ),
                ]
            )
        return opt

    def run(
        self,
        model: nn.Module,
        tr_loader: torch.utils.data.DataLoader,
        vl_loader: torch.utils.data.DataLoader,
    ) -> tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        学習実行

        Args:
            model (nn.Module): model
            tr_loader (DataLoader): train loader
            vl_loader (DataLoader): valid loader

        Returns:
            tuple(float, pd.DataFrame, pd.DataFrame): (best_score, tr_score_df, vl_score_df)
        """
        self.train_steps_per_epoch = int(len(tr_loader))

        optimizer = self._get_optimizer(model)

        if self.scheduler is None:
            scheduler = None
        else:
            scheduler = self.scheduler(
                optimizer,
                steps_per_epoch=self.train_steps_per_epoch,
                epochs=self.epochs,
                **self.sch_params,
            )

        grad_scaler = torch.amp.GradScaler(
            init_scale=65536.0,
        )

        model_path = self.prefix + "model.pth"
        log_path = self.prefix + "tr_log.csv"

        loglist = []
        tr_scores, vl_scores = [], []
        best_score = None
        self.save_model(model, model_path)
        for ep in range(self.epochs):
            print("\nepoch ", ep)

            for param_group in optimizer.param_groups:
                print("lr ", param_group["lr"])
                now_lr = param_group["lr"]

            tr_log, tr_scores_ = self.run_epoch(
                model, optimizer, scheduler, tr_loader, grad_scaler, train=True
            )
            vl_log, vl_scores_ = self.run_epoch(
                model, None, None, vl_loader, grad_scaler, train=False
            )
            tr_scores.extend(tr_scores_)
            vl_scores.extend(vl_scores_)

            print()
            self.print_result(tr_log, "Train")
            self.print_result(vl_log, "Valid")

            # best score
            score = vl_log["score"]
            if not np.isnan(score):
                if best_score is None:
                    best_score = score
                    print("Update best score :", best_score)

                    if self.save_best:
                        self.save_model(model, model_path)
                else:
                    if (self.maximize_score and best_score < score) or (
                        not self.maximize_score and best_score > score
                    ):
                        best_score = score
                        print("Update best score :", best_score)

                        if self.save_best:
                            self.save_model(model, model_path)

            if not self.save_best:
                self.save_model(model, model_path)

            if ep in self.save_epochs:
                self.save_model(model, self.prefix + f"epoch{ep}_" + "model.pth")

            # save log
            columns = (
                ["ep", "lr"]
                + ["tr_" + k for k in tr_log.keys()]
                + ["vl_" + k for k in tr_log.keys()]
            )
            loglist.append([ep, now_lr] + list(tr_log.values()) + list(vl_log.values()))
            rslt_df = pd.DataFrame(loglist, columns=columns)
            rslt_df.to_csv(log_path)

        # load best
        model.load_state_dict(torch.load(model_path))
        model.to(self.dev)

        # scores
        if len(tr_scores) > 0:
            tr_scores = {
                i: {f"tr_{k}": v for k, v in scores.items()}
                for i, scores in enumerate(tr_scores)
            }
            vl_scores = {
                i: {f"vl_{k}": v for k, v in scores.items()}
                for i, scores in enumerate(vl_scores)
            }
            tr_score_df = pd.DataFrame.from_dict(tr_scores, orient="index")
            vl_score_df = pd.DataFrame.from_dict(vl_scores, orient="index")
        else:
            tr_score_df = None
            vl_score_df = None

        return best_score, tr_score_df, vl_score_df

    def run_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loader: torch.utils.data.DataLoader,
        grad_scaler: torch.amp.GradScaler,
        train: bool = True,
    ) -> tuple[dict, dict]:
        """
        epoch実行

        Args:
            model (nn.Module): model
            optimizer (torch.optim.Optimizer): optimizer
            scheduler (torch.optim.lr_scheduler._LRScheduler): scheduler
            loader (DataLoader): loader
            train (bool): train or valid

        Returns:
            dict: result
            dict: score
        """
        if train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        total_loss = 0
        total_loss_dict = {}

        for data in tqdm(loader):
            imgs = data[0].to(self.dev)
            features = data[1].to(self.dev)
            targets = data[2].to(self.dev)

            with torch.set_grad_enabled(train):
                if train and isinstance(optimizer, SAM):
                    # sam first step
                    optimizer.step = optimizer.first_step

                with torch.amp.autocast(
                    device_type=self.dev, dtype=torch.bfloat16, enabled=True
                ):
                    if train:
                        outputs = model(imgs, features)
                        loss, loss_dict = self.criterion(outputs, targets)
                    else:
                        with torch.no_grad():
                            outputs = model(imgs, features)
                            loss, loss_dict = self.criterion(outputs, targets)

                if train:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)

                    if self.grad_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=model.parameters(), max_norm=self.grad_max_norm
                        )

                    # ampは学習初期にgradがnanになることがある。gradにnanがあるとoptimizer.step()は実行されない。
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()

                    if isinstance(optimizer, SAM):
                        # first stepが実行された場合のみsecond stepを実行する
                        if optimizer.first_step_is_done:
                            # sam second step
                            optimizer.step = optimizer.second_step
                            with torch.amp.autocast(
                                device_type=self.dev, dtype=torch.bfloat16, enabled=True
                            ):
                                outputs = model(imgs, features)
                                loss, loss_dict = self.criterion(outputs, targets)

                            grad_scaler.scale(loss).backward()
                            grad_scaler.unscale_(optimizer)

                            if self.grad_max_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    parameters=model.parameters(),
                                    max_norm=self.grad_max_norm,
                                )

                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            optimizer.zero_grad()
                        # first stepが実行されたことをリセット
                        optimizer.reset_first_step_is_done()

                    # scheduler
                    if scheduler is not None:
                        scheduler.step()

                total_loss += loss.item()
                if len(total_loss_dict) == 0:
                    total_loss_dict = loss_dict
                else:
                    total_loss_dict = {
                        k: total_loss_dict[k] + v for k, v in loss_dict.items()
                    }

        # loss
        total_loss = total_loss / len(loader)
        total_loss_dict = {k: v / len(loader) for k, v in total_loss_dict.items()}

        result = dict(loss=total_loss, score=total_loss, **total_loss_dict)
        scores = []
        return result, scores

    def print_result(self, result, title):
        print(title + " Loss: %.4f" % (result["loss"],))
        print({k: v for k, v in result.items() if k.startswith("loss")})

    def save_model(self, model, model_path):
        torch.save(model.to("cpu").state_dict(), model_path)
        model = model.to(self.dev)
        print("Save model :", model_path)


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.reset_first_step_is_done()

    def reset_first_step_is_done(self):
        self.first_step_is_done = False

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.first_step_is_done = True

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def _get_optimizer(opt):
    if opt == "sgd":
        return torch.optim.SGD
    elif opt == "adam":
        return torch.optim.Adam
    elif opt == "adamw":
        return torch.optim.AdamW
    elif opt == "lbfgs":
        return torch.optim.LBFGS
    else:
        return None


# dnn predict
class Predictor:
    def __init__(self, dev: str, do_print: bool):
        """
        Args:
            dev (str): "cuda", "cpu"
            do_print (bool): tqdmを表示するかどうか
        """
        self.dev = dev
        self.do_print = do_print

    def run(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        推論実行

        Args:
            model (nn.Module): model
            loader (DataLoader): loader

        Returns:
            np.ndarray: 推論結果
        """
        model.eval()
        preds = []

        if self.do_print:
            ite = enumerate(tqdm(loader))
        else:
            ite = enumerate(loader)

        for batch_idx, data in ite:
            imgs = data[0].to(self.dev)
            features = data[1].to(self.dev)

            with torch.set_grad_enabled(False):
                if self.dev != "cpu":
                    # with torch.amp.autocast(device_type=self.dev, dtype=torch.bfloat16, enabled=True):
                    #    with torch.no_grad():
                    #        output, sub_output, pitwise_sub_output = model(x)
                    with torch.no_grad():
                        outputs = model(imgs, features)
                        output = outputs["avg_val"]
                else:
                    with torch.no_grad():
                        outputs = model(imgs, features)
                        output = outputs["avg_val"]

            preds.append(output.cpu().numpy())

        preds = np.concatenate(preds)
        return preds


# ########
# evaluation
# ########
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
