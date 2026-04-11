import os
import numpy as np
import torch
from tqdm import tqdm


class UWBDataProcessor:
    """
    UWB数据处理类，支持多种数据转换方法
    """

    def __init__(self, folder_path: str = "cir_50", length: int = 100, train: bool = True):
        """
        初始化处理器

        参数:
        folder_path: 数据文件夹路径
        length: CIR数据长度
        """
        self.folder_path = folder_path
        self.length = length
        self.all_uwb_data = []
        self.all_labels = []
        self.train = train

    def _validate_folder(self):
        """验证文件夹是否存在"""
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"错误：文件夹 '{self.folder_path}' 不存在")

    def _read_data_files(self):
        """
        读取数据文件

        参数:
        file_range: 文件范围，默认为1到5

        返回:
        file_data: 文件数据字典 {文件路径: 数据}
        """
        file_data = {}
        if self.train:
            file_range = (1, 6)
        else:
            file_range = (6, 8)
        start, end = file_range

        for file_num in range(start, end):
            file_path = os.path.join(self.folder_path, f"{file_num}.txt")

            if not os.path.exists(file_path):
                print(f"警告：文件 {file_path} 不存在，跳过")
                continue

            print(f"正在处理文件: {file_path}")

            try:
                data = np.loadtxt(file_path)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                file_data[file_path] = data
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue

        return file_data

    def _extract_cir_data(self, data_row):
        """
        从数据行中提取CIR数据和标签

        参数:
        data_row: 单行数据

        返回:
        cir_data: CIR数据
        label: 标签
        """
        cir_data = data_row[11:11 + self.length]
        # cir_data = data_row[: -1]
        label = data_row[-1]
        return cir_data, label

    def _create_dataset(self, uwb_tensor, labels_tensor):
        """
        创建UWB数据集

        参数:
        uwb_tensor: 数据张量
        labels_tensor: 标签张量

        返回:
        UWB_dataset: UWB数据集
        """
        return uwb_tensor, labels_tensor


        uwb_tensor = torch.tensor(np.array(self.all_uwb_data), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(self.all_labels), dtype=torch.float32)

        print(f"格拉米角场处理完成！总共处理了 {len(self.all_uwb_data)} 个样本")
        print(f"数据张量形状: {uwb_tensor.shape}")
        print(f"标签张量形状: {labels_tensor.shape}")

        return self._create_dataset(uwb_tensor, labels_tensor)

    def process_time_domain(self):

        self._validate_folder()
        self.all_uwb_data = []
        self.all_rf_data = []
        self.all_labels = []

        file_data = self._read_data_files()

        for file_path, data in file_data.items():
            try:
                for i in tqdm(range(len(data)), desc=f"处理文件 {os.path.basename(file_path)}"):
                    rf_data = np.concatenate([data[i][0:1], data[i][2:11]])
                    rf_data = np.abs(rf_data)

                    cir_data = data[i][11:11 + self.length]
                    label = data[i][-1]

                    self.all_uwb_data.append(cir_data)
                    self.all_rf_data.append(rf_data)
                    self.all_labels.append(label)

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue

        # 转换为张量并创建数据集
        uwb_tensor = torch.tensor(np.array(self.all_uwb_data), dtype=torch.float32)
        rf_tensor = torch.tensor(np.array(self.all_rf_data), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(self.all_labels), dtype=torch.float32)

        print(f"总共处理了 {len(self.all_uwb_data)} 个样本")
        print(f"CIR张量形状: {uwb_tensor.shape}")
        print(f"RF张量形状: {rf_tensor.shape}")
        print(f"标签张量形状: {labels_tensor.shape}")

        return uwb_tensor, rf_tensor, labels_tensor



    def get_statistics(self):
        """
        获取数据统计信息

        返回:
        stats: 统计信息字典
        """
        if not self.all_uwb_data:
            return {"message": "没有可用的数据，请先处理数据"}

        stats = {
            "total_samples": len(self.all_uwb_data),
            "data_shape": np.array(self.all_uwb_data).shape,
            "unique_labels": len(np.unique(self.all_labels)),
            "label_distribution": np.bincount(np.array(self.all_labels).astype(int))
        }
        return stats
