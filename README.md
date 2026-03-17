# ocr (ROS 2)

一个最小 ROS 2 工作区：从固定路径读入图片，OCR 识别文字并发布到话题。

默认提供 C++ 版本（`ocr_reader`），直接链接系统的 Tesseract（不需要 pip/venv）。Python 版本（`ocr_reader`）保留作备用。

## 依赖（C++ 版本）

- ROS 2（已 source 对应的 `setup.bash`）
- Tesseract OCR 引擎 + 开发库

在 Ubuntu/Debian 上：

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev
```

## 构建

在工作区根目录：

```bash
colcon build
source install/setup.bash
```

## 运行

默认会读取参数 `image_path` 指定的图片路径（默认：`/home/lu/code/ocr/input.png`）。

```bash
ros2 run ocr_reader ocr_reader_node
```

指定图片路径：

```bash
ros2 run ocr_reader ocr_reader_node --ros-args -p image_path:=/abs/path/to/problem.png
```

使用 launch：

```bash
ros2 launch ocr_reader ocr_reader.launch.py image_path:=/abs/path/to/problem.png
```

输出：
- 识别结果发布到话题 `ocr_text`（类型：`std_msgs/msg/String`）
