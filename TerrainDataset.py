import csv
import json
import os
import datasets
import pandas as pd

_DESCRIPTION = """\
Dataset for sketch2terrain generation task.
"""

from unittest.mock import patch, mock_open


class TerrainDataset(datasets.GeneratorBasedBuilder):
    """dataset file"""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="usgs", version=VERSION),
    ]

    DEFAULT_CONFIG_NAME = "usgs"

    def _info(self):
        if self.config.name == "usgs":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "image": datasets.Image(),
                    "sketch": datasets.Image(),
                    "text": datasets.Value("string")
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
        )

    def _split_generators(self, dl_manager):
        images_dir = 'res'
        metadata_path = 'train_dataset.csv'
        conditioning_images_dir = 'res'

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)
        print('training data length:', len(metadata))

        for _, row in metadata.iterrows():
            text = row["text"]

            image_path = os.path.join(images_dir, row["image"])
            image = open(image_path, "rb").read()

            conditioning_image_path = os.path.join(conditioning_images_dir, row["sketch"])
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["image"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "sketch": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }


if __name__ == '__main__':
    # 创建数据集实例
    dataset = TerrainDataset()

    # 确保数据已下载和准备好
    dataset.download_and_prepare()
    #
    # # 使用数据
    # data = dataset.as_dataset(split='train')  # 确保指定数据集分割
    # print(data)
