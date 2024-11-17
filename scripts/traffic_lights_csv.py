import json
from pathlib import Path

import polars as pl

from atmacup_18 import constants


def load_traffic_lights_json_files(directory: Path) -> list[dict]:
    """traffic_lightsディレクトリ内のJSONファイルを読み込む。

    Args:
        directory (Path): JSONファイルを含むディレクトリ。

    Returns:
        List[Dict]: JSONファイルの内容を持つ辞書のリスト。


    Example:
        [{'ID': '00066be8e20318869c38c66be466631a_320', 'data': []},
         {'ID': '00066be8e20318869c38c66be466631a_420', 'data': []},
         {'ID': '00066be8e20318869c38c66be466631a_520', 'data': []},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_120', 'data': []},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_20',
         'data': [{'index': 1,
         'class': 'green',
         'bbox': [63.53342819213867,
             10.685697555541992,
             65.62159729003906,
             11.599557876586914]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_220',
         'data': [{'index': 1,
         'class': 'green',
         'bbox': [68.00301361083984,
             6.320143222808838,
             69.9793701171875,
             7.251696586608887]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_320',
         'data': [{'index': 1,
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
         'data': [{'index': 1,
         'class': 'green',
         'bbox': [72.72760009765625,
             0.5037141442298889,
             76.14653015136719,
             1.831321358680725]}]},
         {'ID': '000fb056f97572d384bae4f5fc1e0f28_520',
         'data': [{'index': 1,
         'class': 'green',
         'bbox': [72.32199096679688,
             5.2213826179504395,
             74.47071838378906,
             6.218740463256836]}]},
         {'ID': '0010357962a2e77440a0ff237af69b27_120', 'data': []}]
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


def traffic_lights_infos_to_df(traffic_lights_infos: list[dict]) -> pl.DataFrame:
    """traffic_lights_infosをpolarsのDataFrameに変換する。

    Args:
        traffic_lights_infos (List[Dict]): 信号機情報。

    Returns:
        pl.DataFrame: 信号機情報を持つDataFrame。列は以下の通り。
            - ID
            - index
            - class
            - bbox_upper_left_x
            - bbox_upper_left_y
            - bbox_lower_right_x
            - bbox_lower_right_y
    """
    records = []
    for info in traffic_lights_infos:
        id = info["ID"]
        traffic_lights = info["traffic_lights"]

        if len(traffic_lights) == 0:
            record = {"ID": id}
            records.append(record)

        for traffic_light in traffic_lights:
            record = {}
            record["ID"] = id
            record["index"] = traffic_light["index"]
            record["class"] = traffic_light["class"]
            record["bbox_upper_left_x"] = traffic_light["bbox"][0]
            record["bbox_upper_left_y"] = traffic_light["bbox"][1]
            record["bbox_lower_right_x"] = traffic_light["bbox"][2]
            record["bbox_lower_right_y"] = traffic_light["bbox"][3]
            records.append(record)

    return pl.DataFrame(records)


def create_traffic_lights_csv(
    traffic_lights_dir: Path, traffic_lights_csv: Path
) -> None:
    """信号機情報をCSVファイルに保存する。

    Args:
        traffic_lights_dir (Path): 信号機情報が含まれるディレクトリ。
        traffic_lights_csv (Path): 信号機情報を保存するCSVファイル。
    """
    traffic_lights_infos = load_traffic_lights_json_files(traffic_lights_dir)
    df = traffic_lights_infos_to_df(traffic_lights_infos)
    # sort
    df = df.sort(["ID", "index"])

    df.write_csv(traffic_lights_csv)


if __name__ == "__main__":
    traffic_lights_dir = constants.TRAFFIC_LIGHTS_DIR
    traffic_lights_csv = constants.TRAFFIC_LIGHTS_CSV
    create_traffic_lights_csv(
        traffic_lights_dir=traffic_lights_dir, traffic_lights_csv=traffic_lights_csv
    )
