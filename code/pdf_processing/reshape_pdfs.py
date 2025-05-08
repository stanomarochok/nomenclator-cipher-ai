import os
from PyPDF2 import PdfReader, PdfWriter


def merge_pdfs(folder_path, output_path="merged.pdf"):
    pdf_writer = PdfWriter()

    # Merge all PDFs in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)

    # Save the merged PDF
    with open(output_path, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

    print(f"Merged PDF saved as {output_path}")
    return output_path


def split_pdf(input_pdf, output_folder, pages_per_split=5):
    pdf_reader = PdfReader(input_pdf)
    total_pages = len(pdf_reader.pages)

    for start_page in range(0, total_pages, pages_per_split):
        pdf_writer = PdfWriter()
        end_page = min(start_page + pages_per_split - 1, total_pages - 1)

        # Add pages for the current split
        for page in range(start_page, end_page + 1):
            pdf_writer.add_page(pdf_reader.pages[page])

        # Naming the split PDF in the format "startPage-endPage.pdf"
        output_pdf_path = os.path.join(output_folder, f"{start_page + 1}-{end_page + 1}.pdf")
        with open(output_pdf_path, "wb") as output_pdf:
            pdf_writer.write(output_pdf)

        print(f"Saved split PDF: {output_pdf_path}")


def main():
    folder_path = '.'
    output_folder = './reshaped'

    # Merge PDFs
    merged_pdf_path = merge_pdfs(folder_path)

    # Split merged PDF
    split_pdf(merged_pdf_path, output_folder)


if __name__ == "__main__":
    main()
