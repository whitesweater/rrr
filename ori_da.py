import csv
import json
import os
import datasets
import pandas as pd
from datasets import load_dataset

if __name__ == '__main__':
    # 创建数据集实例
    # dataset = load_dataset()
    dataset = load_dataset(
        'TerrainDataset.py',
        trust_remote_code=True
    )
    # 确保数据已下载和准备好
    # dataset.download_and_prepare()
    #
    # # 使用数据
    column_names = dataset["train"].column_names
    data = dataset.as_dataset(split='train')  # 确保指定数据集分割
    print(data)
