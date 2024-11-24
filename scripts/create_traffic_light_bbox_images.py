import json
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from atmacup_18 import constants


def fill_rectangles(
    rectangles: list[list[int]], image_size: tuple[int, int]
) -> np.ndarray:
    """
    指定された矩形部分を塗りつぶす関数。

    Args:
        rectangles (list[list[int]]): 矩形の座標のリスト。各要素は[左上x座標, 左上y座標, 右下x座標, 右下y座標]。
        image_size (tuple[int, int]): 画像のサイズ（高さ, 幅）のタプル。

    Returns:
        np.ndarray: 塗りつぶし後の画像（numpy配列）。
    """
    # 背景が0の画像を作成
    image = np.zeros(image_size, dtype=np.int16)

    # 各矩形を塗りつぶす
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        cv2.rectangle(image, (x1, y1), (x2, y2), color=1, thickness=-1)

    return image


def load_traffic_lights_json_files(directory: Path) -> list[dict]:
    """traffic_lightsディレクトリ内のJSONファイルを読み込む。

    Args:
        directory (Path): JSONファイルを含むディレクトリ。

    Returns:
        List[Dict]: JSONファイルの内容を持つ辞書のリスト。


    Example:
        [{'ID': '00066be8e20318869c38c66be466631a_320', 'traffic_lights': []},
         {'ID': '00066be8e20318869c38c66be466631a_420', 'traffic_lights': []},
         {'ID': '00066be8e20318869c38c66be466631a_520', 'traffic_lights': []},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_120', 'traffic_lights': []},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_20',
         'traffic_lights': [{'index': 1,
         'class': 'green',
         'bbox': [63.53342819213867,
             10.685697555541992,
             65.62159729003906,
             11.599557876586914]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_220',
         'traffic_lights': [{'index': 1,
         'class': 'green',
         'bbox': [68.00301361083984,
             6.320143222808838,
             69.9793701171875,
             7.251696586608887]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_320',
         'traffic_lights': [{'index': 1,
         'class': 'green',
         'bbox': [61.16100311279297,
             3.9232115745544434,
             63.26563262939453,
             5.019723415374756]},
         {'index': 2,
         'class': 'green',
         'bbox': [71.0138168334961,
             3.411626100540161,
             73.28620910644531,
             4.54356050491333]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_420',
         'traffic_lights': [{'index': 1,
         'class': 'green',
         'bbox': [72.72760009765625,
             0.5037141442298889,
             76.14653015136719,
             1.831321358680725]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_520',
         'traffic_lights': [{'index': 1,
         'class': 'green',
         'bbox': [72.32199096679688,
             5.2213826179504395,
             74.47071838378906,
             6.218740463256836]}]},
         {'ID': '0010357962a2e77440a0ff237af69b27_120', 'traffic_lights': []}]
    """
    json_files = sorted(directory.glob("*.json"))

    traffic_lights_infos = []
    for json_file in json_files:
        traffic_lights_info = {}

        id = json_file.stem
        traffic_lights_info["ID"] = id

        with open(json_file, encoding="utf-8") as file:
            traffic_lights_info["traffic_lights"] = json.load(file)

        traffic_lights_infos.append(traffic_lights_info)

    return traffic_lights_infos


def create_traffic_light_bbox_image(
    traffic_lights_info: dict, image_size: tuple[int, int]
) -> np.ndarray:
    """交差点画像に信号機のバウンディングボックスを描画する。

    Args:
        traffic_lights_info (dict): 信号機情報を含む辞書。
        image_size (tuple[int, int]): 画像のサイズ（高さ, 幅）のタプル。

    Returns:
        np.ndarray: 信号機のバウンディングボックスが描画された画像（numpy配列）
    """
    traffic_lights = traffic_lights_info["traffic_lights"]

    # class毎のbboxを取得
    bbox_dict = {}
    for traffic_light in traffic_lights:
        class_ = traffic_light["class"]
        # (左上x座標, 左上y座標, 右下x座標, 右下y座標)
        bbox = traffic_light["bbox"]
        bbox[0] = math.floor(bbox[0])
        bbox[1] = math.floor(bbox[1])
        bbox[2] = math.ceil(bbox[2])
        bbox[3] = math.ceil(bbox[3])

        if class_ not in constants.TRAFFIC_LIGHT_CLASS_INDEXES:
            raise ValueError(f"Invalid class: {class_}")

        if class_ not in bbox_dict:
            bbox_dict[class_] = []

        bbox_dict[class_].append(bbox)

    n_channels = len(constants.TRAFFIC_LIGHT_CLASS_INDEXES)
    image = np.zeros((*image_size, n_channels), dtype=np.int16)

    for class_, bboxes in bbox_dict.items():
        class_index = constants.TRAFFIC_LIGHT_CLASS_INDEXES[class_]
        class_image = fill_rectangles(bboxes, image_size)
        image[:, :, class_index] = class_image

    return image


def create_traffic_light_bbox_images(
    traffic_lights_dir: Path,
    image_size: tuple[int, int],
    save_image_name: str,
    save_dir: Path,
):
    """
    信号機bboxを描画した画像を作成し保存する。

    Args:
        traffic_lights_dir (Path): 信号機情報が含まれるディレクトリ。
        image_size (tuple[int, int]): 画像のサイズ（高さ, 幅）のタプル。
        save_image_name (str): 保存する画像の名前。
        save_dir (Path): 画像を保存するディレクトリ。

    Note:
        - 画像の保存パスは save_dir/ID/save_image_name となる。
    """
    traffic_lights_infos = load_traffic_lights_json_files(traffic_lights_dir)

    for traffic_lights_info in tqdm(traffic_lights_infos):
        id = traffic_lights_info["ID"]
        image = create_traffic_light_bbox_image(traffic_lights_info, image_size)

        save_path = Path(save_dir, id, save_image_name)
        # npyで保存する
        np.save(save_path, image)

    return


if __name__ == "__main__":
    traffic_lights_dir = constants.TRAFFIC_LIGHTS_DIR
    image_size = constants.IMAGE_SIZE
    save_image_name = constants.TRAFFIC_LIGHT_BBOX_IMAGE_NAME
    save_dir = constants.IMAGES_DIR

    create_traffic_light_bbox_images(
        traffic_lights_dir, image_size, save_image_name, save_dir
    )
