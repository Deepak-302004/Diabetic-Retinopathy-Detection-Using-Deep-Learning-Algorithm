import cv2
import os

# ✅ Dataset Paths
input_dir = r"D:\archive\gaussian_filtered_images\gaussian_filtered_images"
output_dir = r"D:\UG-Project\Dataset-Kaggle"

# ✅ Create output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# ✅ Process each category
for category in os.listdir(input_dir):
    category_path = os.path.join(input_dir, category)

    # ✅ Ignore non-folder files
    if not os.path.isdir(category_path):
        continue

    output_category_path = os.path.join(output_dir, category)
    os.makedirs(output_category_path, exist_ok=True)

    images = [img for img in os.listdir(category_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in images:
        img_path = os.path.join(category_path, img_name)
        output_path = os.path.join(output_category_path, img_name)

        # ✅ Read & Resize
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, img)  # ✅ Save resized image

    print(f"✅ {category}: {len(images)} images processed!")

print("🎯 All images preprocessed successfully!")
