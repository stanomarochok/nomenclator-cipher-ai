import os
from PyPDF2 import PdfReader, PdfWriter


def merge_pdfs_in_numeric_order(folder_path, output_path="merged_ordered.pdf"):
    pdf_writer = PdfWriter()

    # List all PDFs and filter out non-numeric filenames
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf") and f.split(".")[0].isdigit()]
    pdf_files.sort(key=lambda f: int(f.split(".")[0]))  # Sort by the numeric part of the filename

    # Merge the PDFs in sorted order
    for filename in pdf_files:
        pdf_path = os.path.join(folder_path, filename)
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    # Save the merged PDF
    with open(output_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

    print(f"Merged PDF saved as {output_path}")


def main():
    folder_path = './pdfs'
    output_pdf_path = "merged_ordered.pdf"  # or specify a custom name
    merge_pdfs_in_numeric_order(folder_path, output_pdf_path)


if __name__ == "__main__":
    main()
