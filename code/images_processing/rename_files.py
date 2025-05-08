import os


# Specify the directory containing the images
directory = "../../materials/rotated_images"

# Get a sorted list of files in the directory
files = sorted(os.listdir(directory), key=lambda x: int(x[3:x.find(".png")]))

# Loop through the files and rename them
for file in files:
    if file.endswith(".png"):
        # Extract the numeric part from the filename (e.g., img5.png -> 5)
        file_number = int(file[3:file.find(".png")])

        # Calculate the new number by subtracting 4
        new_file_number = file_number - 4

        # Create the new file name in the format 'imgX.png'
        new_file_name = f"img{new_file_number}.png"

        # Build the old and new file paths
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file if the new name doesn't already exist
        if not os.path.exists(new_file_path):
            os.rename(old_file_path, new_file_path)
        else:
            print(f"Skipping {new_file_name} as it already exists.")

print("Files renamed successfully!")