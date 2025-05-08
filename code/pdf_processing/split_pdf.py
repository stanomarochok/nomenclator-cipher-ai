from PyPDF2 import PdfReader, PdfWriter
import os


def split_pdf(input_pdf, output_folder, pages_per_split=5):
    # Load the input PDF
    pdf_reader = PdfReader(input_pdf)
    total_pages = len(pdf_reader.pages)

    # Split PDF into parts of 5 pages each
    for start_page in range(0, total_pages, pages_per_split):
        pdf_writer = PdfWriter()
        end_page = min(start_page + pages_per_split - 1, total_pages - 1)

        # Add pages for the current split
        for page in range(start_page, end_page + 1):
            pdf_writer.add_page(pdf_reader.pages[page])

        # Naming the split PDF in the format "startPage-endPage.pdf"
        output_pdf_name = f"{start_page + 1}-{end_page + 1}.pdf"
        output_pdf_path = os.path.join(output_folder, output_pdf_name)
        with open(output_pdf_path, "wb") as output_pdf:
            pdf_writer.write(output_pdf)

        print(f"Saved split PDF: {output_pdf_path}")


def main():
    input_pdf = './merged_ordered.pdf'
    output_folder = './reshaped2'

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Split the PDF
    split_pdf(input_pdf, output_folder)


if __name__ == "__main__":
    main()
