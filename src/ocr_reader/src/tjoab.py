from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps
import sys
import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

MODEL_ID = 'tjoab/latex_finetuned'

# Helper funtion (path to either JPEG or PNG)
def open_PIL_image(image_path: str) -> Image.Image:
  image = Image.open(image_path)
  if image.mode in ('RGBA', 'LA'):
      background = Image.new('RGB', image.size, 'white')
      background.paste(image.convert('RGB'), mask=image.getchannel('A'))
      return ImageOps.invert(background)
  return ImageOps.invert(image.convert('RGB'))


# Load model and processor from local cache only (offline mode)
try:
  processor = TrOCRProcessor.from_pretrained(MODEL_ID, local_files_only=True)
  model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, local_files_only=True)
except OSError as exc:
  raise SystemExit(
      f'Offline load failed: local cache for {MODEL_ID} was not found. '
      'Run once online to cache the model, then retry offline.'
  ) from exc


# Load all images as a batch
paths = sys.argv[1:]
if not paths:
  raise SystemExit('Usage: python tjoab.py <image1> [image2 ...]')

images = [open_PIL_image(path) for path in paths]

# Preprocess the images 
preproc_image = processor.image_processor(images=images, return_tensors="pt").pixel_values

# Generate and decode the tokens
# NOTE: max_length default value is very small, which often results in truncated inference if not set 
pred_ids = model.generate(preproc_image, max_length=128)
latex_preds = processor.batch_decode(pred_ids, skip_special_tokens=True)

for path, pred in zip(paths, latex_preds):
  print(f'{path}: {pred}')
