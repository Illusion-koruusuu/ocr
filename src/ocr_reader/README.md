# ocr_reader

ROS 2 C++ 节点：从固定路径读入图片，使用 Tesseract（C++ API）做 OCR，并发布识别结果。

参数：
- `image_path`：图片绝对路径（默认：`/home/lu/code/ocr/input.png`）
- `lang`：语言（默认：`eng`）
- `period_ms`：轮询周期，便于题目更新（默认：500ms）
- `topic`：发布话题名（默认：`ocr_text`）
