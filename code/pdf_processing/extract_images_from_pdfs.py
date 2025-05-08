import os
import fitz  # PyMuPDF


def extract_images_from_pdf(pdf_path, output_folder, img_counter):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            img_data = base_image["image"]

            # Define image file path with sequential numbering
            image_filename = f"img{img_counter}.png"
            image_path = os.path.join(output_folder, image_filename)

            # Write image data to file
            with open(image_path, "wb") as img_file:
                img_file.write(img_data)

            print(f"Saved image: {image_path}")
            img_counter += 1

    pdf_document.close()
    return img_counter


def extract_images_from_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    img_counter = 1  # Start image numbering from 1
    # Process each PDF in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            img_counter = extract_images_from_pdf(pdf_path, output_folder, img_counter)


def main():
    input_folder = './reshaped'
    output_folder = './images'

    extract_images_from_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
