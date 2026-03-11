import pickle

with open('faiss_chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

print('=== R&D chunks ===')
for c in chunks:
    if 'research and development' in c['page_content'].lower() and any(x in c['page_content'] for x in ['29,915', '26,251', 'R&D']):
        print(f"Page {c['metadata']['page']}: {c['page_content'][:200]}")
        print()

print('=== Risk Factor chunks ===')
for c in chunks:
    if 'risk factor' in c['page_content'].lower() and len(c['page_content']) > 200:
        print(f"Page {c['metadata']['page']}: {c['page_content'][:200]}")
        print()