import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def optimize_image(image_path, output_folder):
    try:
        with Image.open(image_path) as img:
            # Set output path
            output_path = os.path.join(output_folder, os.path.basename(image_path))

            if img.format == "JPEG":
                img.save(output_path, "JPEG", quality=85, optimize=True, progressive=True)
            elif img.format == "PNG":
                img.save(output_path, "PNG", optimize=True)
            else:
                img.save(output_path)

            print(f"Optimized and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def optimize_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    # Use ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor() as executor:
        for image_path in image_files:
            executor.submit(optimize_image, image_path, output_folder)


def main():
    input_folder = './rotated_images'
    output_folder = './optimized_images'

    optimize_images_in_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
