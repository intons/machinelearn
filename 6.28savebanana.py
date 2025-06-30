import d2l
import zipfile
import os
from pathlib import Path

# 1. 定义数据集保存的本地路径
# 这里使用当前工作目录下的'data'文件夹，你可以根据需要修改
data_dir = './data'
Path(data_dir).mkdir(exist_ok=True)  # 创建目录（如果不存在）

# 2. 从D2L数据中心下载数据集
# D2L的DATA_HUB已经注册了banana-detection数据集
# download函数会自动处理下载和校验
zip_file_path = d2l.download('banana-detection', data_dir=data_dir)
print(f"数据集压缩包已下载到: {zip_file_path}")

# 3. 解压数据集（可选步骤，根据实际需求决定是否解压）
def unzip_file(zip_path, extract_dir):
    """解压zip文件到指定目录"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"数据集已解压到: {extract_dir}")

# 解压到与压缩包同级的目录
extract_dir = os.path.join(data_dir, 'banana-detection')
unzip_file(zip_file_path, extract_dir)

# 4. 验证数据集是否正确保存
# 检查文件是否存在
if os.path.exists(zip_file_path):
    print(f"压缩包文件存在，大小: {os.path.getsize(zip_file_path) / 1024:.2f} KB")
if os.path.exists(extract_dir):
    print(f"解压后的数据集目录存在，包含以下文件:")
    # 打印目录下的文件列表（最多10个）
    files = os.listdir(extract_dir)
    for file in files[:10]:
        print(f"  - {file}")
    if len(files) > 10:
        print(f"  ... 还有{len(files) - 10}个文件")