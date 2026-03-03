import fitz
import pytesseract
from PIL import Image
import io 

def extract_text_from_pdf(pdf_path: str):
    pages = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start = 1):
            text = page.get_text('text')

            # if text is empty, use OCR to extract text
            if not text.strip():
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(image)

            pages.append({
                "page":page_number,
                "text": text.strip()
            }) 

    return pages