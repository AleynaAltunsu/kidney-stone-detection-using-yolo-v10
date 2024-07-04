# kidney-stone-detection-using-yolo-v10


---

# YOLOv10 Object Detection with Kidney Stone Images

This repository contains code for setting up and training YOLOv10 for object detection using kidney stone images.

## Setup

### Installation

Install YOLOv10 library:

```bash
pip install -q git+https://github.com/THU-MIG/yolov10.git
```

### Download YOLOv10 Weights

Download the YOLOv10 model weights:

```bash
mkdir -p /kaggle/working/yolov10/weights
wget -P /kaggle/working/yolov10/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt
```

## Training Configuration

The training configuration is defined in the script:

```python
class CFG:
    EPOCHS = 5  # Number of epochs (reduced for demonstration)
    BATCH_SIZE = 32
    SEED = 6
    LEARNING_RATE = 0.001
    NUM_SAMPLES = 16
    OPTIMIZER = 'Adam'
    
    DATA_PATH = '/kaggle/input/kidney-stone-images/data.yaml'
    SAMPLE_PATH = '/kaggle/input/kidney-stone-images/test/images/*'
```

## Random Image Selection and Visualization

Randomly select and visualize sample images from the dataset:

```python
import matplotlib.pyplot as plt
import cv2
from glob import glob
import random

images_data = glob(CFG.SAMPLE_PATH)
random_image = random.sample(images_data, CFG.NUM_SAMPLES)

plt.figure(figsize=(12,10))
for i in range(CFG.NUM_SAMPLES):
    plt.subplot(4,4,i+1)
    plt.imshow(cv2.imread(random_image[i]))
    plt.axis('off')
plt.show()
```

## Training YOLOv10

Train the YOLOv10 model:

```python
from ultralytics import YOLOv10

yolo_v10 = YOLOv10('/kaggle/working/yolov10/weights/yolov10m.pt')
v10_model = yolo_v10.train(data=CFG.DATA_PATH, seed=CFG.SEED, epochs=CFG.EPOCHS, lr0=CFG.LEARNING_RATE, optimizer=CFG.OPTIMIZER, verbose=True,
                           project='ft_models', name='yolo_v10')
```

## Object Detection and Visualization

Define a function for stone detection and visualize the results:

```python
def stone_detection(img_path, model):
    import cv2
    
    img = cv2.imread(img_path)
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img

custom_image_dir = '/kaggle/input/kidney-stone-images/test/images'
image_files = os.listdir(custom_image_dir)
selected_images = random.sample(image_files, 16)

v10_trained = YOLOv10('/kaggle/working/ft_models/yolo_v10/weights/best.pt')

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
for i, img_file in enumerate(selected_images):
    row_idx = i // 4
    col_idx = i % 4
    img_path = os.path.join(custom_image_dir, img_file)
    detect_img = stone_detection(img_path, v10_trained)
    axes[row_idx, col_idx].imshow(detect_img)
    axes[row_idx, col_idx].axis('off')
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
```

## Results

Training results and metrics can be visualized using the following code:

```python
import pandas as pd

v10_result = pd.read_csv('/kaggle/working/ft_models/yolo_v10/results.csv')

def show_v10_graphs(result):    
    result.columns = result.columns.str.strip()
    epoch_column = result['epoch']
    plt.figure(figsize=(20,10))
    plt.style.use('ggplot')  
    plt.subplot(1, 3, 1)
    plt.plot(epoch_column, result['train/box_om'], label='train_loss_om')
    plt.plot(epoch_column, result['train/box_oo'], label='train_loss_oo')
    plt.plot(epoch_column, result['val/box_om'], label='val_loss_om')
    plt.plot(epoch_column, result['val/box_oo'], label='val_loss_oo')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('Train and Validation Box Losses (OM/OO)')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epoch_column, result['train/cls_om'], label='train_loss_om')
    plt.plot(epoch_column, result['train/cls_oo'], label='train_loss_oo')
    plt.plot(epoch_column, result['val/cls_om'], label='val_loss_om')
    plt.plot(epoch_column, result['val/cls_oo'], label='val_loss_oo')
    plt.xlabel('Epoch')
    plt.ylabel('Class Loss')
    plt.title('Train and Validation Class Losses (OM/OO)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epoch_column, result['train/dfl_om'], label='train_loss_om')
    plt.plot(epoch_column, result['train/dfl_oo'], label='train_loss_oo')
    plt.plot(epoch_column, result['val/dfl_om'], label='val_loss_om')
    plt.plot(epoch_column, result['val/dfl_oo'], label='val_loss_oo')
    plt.xlabel('Epoch')
    plt.ylabel('Distribution Focal Loss')
    plt.title('Train and Validation Distribution Focal Losses (OM/OO)')
    plt.legend()
    plt.show()

show_v10_graphs(v10_result)

# Eğitim Sonuçlarını ve PR Eğrisini Görselleştirme
plt.figure(figsize=(12,8))
plt.imshow(cv2.imread('/kaggle/working/ft_models/yolo_v10/results.png'))
plt.axis('off')

plt.figure(figsize=(12,8))
plt.imshow(cv2.imread('/kaggle/working/ft_models/yolo_v10/PR_curve.png'))
plt.axis('off')
```
