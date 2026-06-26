import pymupdf4llm as py
import fitz
import uuid

class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    def extract_markdown(self):
        # PyMuPDF4LLM is excellent at turning tables and text into clean markdown
        markdown = py.to_markdown(self.doc)
        return markdown
    
    def extract_images(self, min_bytes=20000):
        images = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            for img in page.get_images(full=True):
                xref = img[0]

                try:
                    base_image = self.doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Filter out tiny artifacts, icons, and lines early
                    if len(image_bytes) < min_bytes:
                        continue

                    images.append({
                        "image_id": str(uuid.uuid4()),
                        "page": page_num + 1,
                        "xref": xref,
                        "extension": image_ext,
                        "bytes": image_bytes
                    })
                except Exception as e:
                    print(f"Error extracting image on page {page_num + 1}, xref {xref}: {e}")

        self.doc.close()
        return images