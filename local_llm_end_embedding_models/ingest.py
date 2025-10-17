import os
import argparse
import faiss
import numpy as np
from tqdm import tqdm
from utils import read_pdf_texts, chunk_text, embed_texts, save_json
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
print("HF token loaded:", bool(hf_token))


def ingest_folder(pdf_dir: str,
                  index_path: str = "faiss.index",
                  meta_path: str = "metadata.json",
                  texts_path: str = "chunks.jsonl",
                  embedding_model_name: str = "google/embeddinggemma-300m",
                  device: str = "cpu",
                  chunk_size: int = 1500,
                  chunk_overlap: int = 200,
                  batch_size: int = 8):
    # collect chunks and metadata
    all_chunks = []
    metadatas = []  # list of dicts
    for root, _, files in os.walk(pdf_dir):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue
            p = os.path.join(root, fn)
            pages = read_pdf_texts(p)
            for page_num, page_text in tqdm(pages):
                chunks = chunk_text(page_text, max_chars=chunk_size, overlap=chunk_overlap)
                for i, ch in enumerate(chunks):
                    meta = {
                        "source_file": os.path.relpath(p, pdf_dir),
                        "page": page_num,
                        "chunk_index": i
                    }
                    all_chunks.append(ch)
                    metadatas.append(meta)

    if len(all_chunks) == 0:
        print("No chunks found. Make sure pdfs exist and contain text.")
        return

    print(f"Embedding {len(all_chunks)} chunks using {embedding_model_name} on device {device} ...")
    embeddings = embed_texts(all_chunks, model_name=embedding_model_name, device=device, batch_size=batch_size)
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    print("Embedding dim:", dim)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    save_json(metadatas, meta_path)
    print(f"Saved metadata to {meta_path}")

    # save chunk texts as jsonl for context display
    import json
    with open(texts_path, "w", encoding="utf-8") as out:
        for meta, txt in zip(metadatas, all_chunks):
            row = {"meta": meta, "text": txt}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved chunks to {texts_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_dir", type=str, required=True, help="Directory with PDFs")
    p.add_argument("--index_path", type=str, default="faiss.index")
    p.add_argument("--meta_path", type=str, default="metadata.json")
    p.add_argument("--texts_path", type=str, default="chunks.jsonl")
    p.add_argument("--embedding_model", type=str, default="google/embeddinggemma-300m")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chunk_size", type=int, default=1500)
    p.add_argument("--chunk_overlap", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()
    ingest_folder(args.pdf_dir, args.index_path, args.meta_path, args.texts_path,
                  args.embedding_model, args.device, args.chunk_size, args.chunk_overlap, args.batch_size)
