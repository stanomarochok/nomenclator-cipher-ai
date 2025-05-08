import os
from PIL import Image


def rotate_images_in_folder(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.isdir(input_folder):
        print(f"The folder '{input_folder}' does not exist.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all files in the input folder
    files = os.listdir(input_folder)

    # Process each file
    for file_name in files:
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # Check if it's an image file
        try:
            with Image.open(input_file_path) as img:
                # Preserve the original format and settings
                format = img.format
                # Rotate the image 90 degrees clockwise
                rotated_img = img.rotate(90, expand=True)

                # Handle JPEG specifically to preserve quality settings
                if format == "JPEG":
                    # Get original quality (if available)
                    quality = img.info.get("quality", 85)
                    rotated_img.save(output_file_path, format=format, quality=quality)
                else:
                    # Save with original format for other image types
                    rotated_img.save(output_file_path, format=format)
                print(f"Rotated and saved to: {output_file_path}")
        except Exception as e:
            print(f"Skipping file '{file_name}': {e}")


def main():
    input_folder = './images'
    output_folder = './rotated_images'
    output_folder = output_folder if output_folder else None

    rotate_images_in_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
