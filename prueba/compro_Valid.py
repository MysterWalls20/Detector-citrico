import os

valid_images_dir = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\dataset\valid\images"
valid_labels_dir = r"C:\Users\OPEN SERVICE EIRL\Documents\UPN\Ciclo 7\Machine learning\Proyecto_Final_Machine Learning\Proyecto Final_Machine learning\dataset\valid\labels"

# Verificar si las carpetas existen
print(f"Valid images directory exists: {os.path.exists(valid_images_dir)}")
print(f"Valid labels directory exists: {os.path.exists(valid_labels_dir)}")

# Verificar el contenido de las carpetas
if os.path.exists(valid_images_dir):
    print(f"Number of images in valid/images: {len(os.listdir(valid_images_dir))}")
else:
    print("No images found in valid/images.")

if os.path.exists(valid_labels_dir):
    print(f"Number of labels in valid/labels: {len(os.listdir(valid_labels_dir))}")
else:
    print("No labels found in valid/labels.")
