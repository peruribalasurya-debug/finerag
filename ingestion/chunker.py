from langchain.text_splitter import RecursiveCharacterTextSplitter

def fixed_size_chunks(pages: list, chunk_size: int = 512, overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["page_content"])
        for split in splits:
            chunks.append({
                "page_content": split,
                "metadata": page["metadata"]
            })

    return chunks


if __name__ == "__main__":
    from loader import load_pdf

    pages = load_pdf("data/apple_10k.pdf")
    chunks = fixed_size_chunks(pages)

    print(f"\n✅ Total pages:  {len(pages)}")
    print(f"✅ Total chunks: {len(chunks)}")
    print(f"✅ Avg chunk size: {sum(len(c['page_content']) for c in chunks) // len(chunks)} characters")
    print(f"\nSample chunk:")
    print("-" * 50)
    print(chunks[5]["page_content"])
    print("-" * 50)
    print(f"From page: {chunks[5]['metadata']['page']}")