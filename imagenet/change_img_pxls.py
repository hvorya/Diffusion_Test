from PIL import Image
import os

def resize_images_in_subdirs(input_dir, output_dir, size=(128, 128)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for root, subdirs, files in os.walk(input_dir):
        # Create a corresponding output subdirectory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file_name)
                output_path = os.path.join(output_subdir, file_name)
                try:
                    # Open and resize the image
                    with Image.open(input_path) as img:
                        resized_img = img.resize(size, Image.Resampling.LANCZOS)
                        resized_img.save(output_path)
                        print(f"Resized: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# Example usage
input_directory = "mnist/train"  # Root folder with 10 subdirectories
output_directory = "mnist/train2"  # Root folder for resized images
resize_images_in_subdirs(input_directory, output_directory)
