import os

train_images_dir = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\dataset\train\images"
train_labels_dir = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\dataset\train\labels"

# Verificar si las carpetas existen
print(f"Train images directory exists: {os.path.exists(train_images_dir)}")
print(f"Train labels directory exists: {os.path.exists(train_labels_dir)}")

# Verificar el contenido de las carpetas
if os.path.exists(train_images_dir):
    print(f"Number of images in train/images: {len(os.listdir(train_images_dir))}")
else:
    print("No images found in train/images.")

if os.path.exists(train_labels_dir):
    print(f"Number of labels in train/labels: {len(os.listdir(train_labels_dir))}")
else:
    print("No labels found in train/labels.")
