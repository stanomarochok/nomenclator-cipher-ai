import os
import json

# Set the directory co  ntaining the JSON files
directory = '../../materials/dataset/word_symbol_annotations_yolo'

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        try:
            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Remove the 'imageData' field if it exists
            if 'imageData' in data:
                del data['imageData']

            # Save the modified JSON back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            print(f'Updated file: {filename}')
        except json.JSONDecodeError:
            print(f'Failed to decode JSON from {filename}')
        except Exception as e:
            print(f'An error occurred with {filename}: {e}')

print("All files have been processed.")
