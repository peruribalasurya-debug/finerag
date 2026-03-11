import fitz
from pathlib import Path

def load_pdf(file_path: str) -> list:
    doc = fitz.open(file_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": page_num + 1
                }
            })
    doc.close()
    print(f"Loaded {len(pages)} pages from {Path(file_path).name}")
    return pages

if __name__ == "__main__":
    pages = load_pdf("data/apple_10k.pdf")
    print(f"\nFirst page preview:")
    print("-" * 50)
    print(pages[0]["page_content"][:500])
    print("-" * 50)
    print(f"\nTotal pages loaded: {len(pages)}")