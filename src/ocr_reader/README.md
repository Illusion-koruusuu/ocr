# ocr_reader

ROS 2 C++ 节点：从固定路径读入图片，使用 Tesseract（C++ API）做 OCR，并发布识别结果。

参数：
- `image_path`：图片绝对路径（默认：`/home/lu/code/ocr/input.png`）
- `lang`：语言（默认：`eng`）
- `period_ms`：轮询周期，便于题目更新（默认：500ms）
- `topic`：发布话题名（默认：`ocr_text`）

## Texify 节点（LaTeX OCR）

项目中还提供了 Python 节点 `texify_reader_node`，用于识别公式并发布 LaTeX 文本。

### 1) 建议环境

- Python 虚拟环境：`/home/lu/code/ocr/.venv`
- 建议版本组合：
	- `texify==0.2.1`
	- `transformers==4.45.2`
	- `tokenizers==0.20.3`

### 2) 首次安装依赖

```bash
cd /home/lu/code/ocr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install texify "transformers==4.45.2" "tokenizers==0.20.3" pillow torch
```

### 3) 构建包

```bash
cd /home/lu/code/ocr
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select ocr_reader
```

### 4) 推荐：离线启动（避免 Hugging Face 网络抖动）

已提供 launch 文件 `launch/texify_reader.launch.py`，默认会设置：

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

启动命令：

```bash
cd /home/lu/code/ocr
source .venv/bin/activate
source /opt/ros/$ROS_DISTRO/setup.bash
source install/setup.bash
ros2 launch ocr_reader texify_reader.launch.py image_path:=/home/lu/code/ocr/png/0.png
```

### 5) 在线模式（可选）

```bash
ros2 launch ocr_reader texify_reader.launch.py \
	hf_hub_offline:=0 \
	transformers_offline:=0 \
	image_path:=/home/lu/code/ocr/png/0.png
```
