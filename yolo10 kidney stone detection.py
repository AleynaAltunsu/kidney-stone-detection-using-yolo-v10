# YOLOv10 Kütüphanesini Yükleme ve Eğitim Ayarlarını Tanımlama
# YOLOv10'u yükleyin
!pip install -q git+https://github.com/THU-MIG/yolov10.git
import os
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from glob import glob
from ultralytics import YOLOv10

# YOLOv10'u yükleyin
!mkdir -p /kaggle/working/yolov10/weights
!wget -P /kaggle/working/yolov10/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt

# Veri yolunu ve örnek görüntü yolunu tanımlayın
class CFG:
    EPOCHS = 5  # Yarıya düşürülen epoch sayısı
    BATCH_SIZE = 32
    SEED = 6
    LEARNING_RATE = 0.001
    NUM_SAMPLES = 16
    OPTIMIZER = 'Adam'
    
    DATA_PATH = '/kaggle/input/kidney-stone-images/data.yaml'
    SAMPLE_PATH = '/kaggle/input/kidney-stone-images/test/images/*'

# Rastgele Örnek Görüntü Seçimi ve Görselleştirme
# Örnek veri setini yükle ve rastgele örnek görüntüleri seç
images_data = glob(CFG.SAMPLE_PATH)
random_image = random.sample(images_data, CFG.NUM_SAMPLES)

# Rastgele seçilen görüntüleri görselleştir
plt.figure(figsize=(12,10))
for i in range(CFG.NUM_SAMPLES):
    plt.subplot(4,4,i+1)
    plt.imshow(cv2.imread(random_image[i]))
    plt.axis('off')
plt.show()

# YOLOv10 Modelini Eğitme
# YOLOv10 modelini eğitin
yolo_v10 = YOLOv10('/kaggle/working/yolov10/weights/yolov10m.pt')
v10_model = yolo_v10.train(data=CFG.DATA_PATH, seed=CFG.SEED, epochs=CFG.EPOCHS, lr0=CFG.LEARNING_RATE, optimizer=CFG.OPTIMIZER, verbose=True,
                           project='ft_models', name='yolo_v10')

# Taş Tespiti ve Görselleştirme
# Fonksiyonu taş tespiti yapmak için güncelleyin
def stone_detection(img_path, model):
    img = cv2.imread(img_path)
    detect_result = model(img)
    detect_img = detect_result[0].plot()
    detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    return detect_img

# Özel görüntüleri içeren dizini tanımlayın
custom_image_dir = '/kaggle/input/kidney-stone-images/test/images'
image_files = os.listdir(custom_image_dir)
selected_images = random.sample(image_files, 16)

# Eğitilmiş modeli yükleyin
v10_trained = YOLOv10('/kaggle/working/ft_models/yolo_v10/weights/best.pt')

# Görüntüleri görselleştirin ve taş tespiti yapın
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

# Eğitim Sonuçlarını Görselleştirme
plt.figure(figsize=(12,8))