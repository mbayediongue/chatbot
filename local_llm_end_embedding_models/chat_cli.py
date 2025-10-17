import argparse
import os
import json
import time
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from utils import mean_pooling
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
print("HF token loaded:", bool(hf_token))

def detect_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_faiss(index_path: str, chunks_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    index = faiss.read_index(index_path)
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return index, chunks

def load_embedding_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tok, model

def embed_query(query: str, tokenizer, model, device="cpu"):
    enc = tokenizer([query], truncation=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        embs = mean_pooling(out, attention_mask)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy().astype("float32")

def retrieve(query: str, index, chunks, emb_tokenizer, emb_model, device, top_k=5):
    q_emb = embed_query(query, emb_tokenizer, emb_model, device=device)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = chunks[idx]["meta"] if idx < len(chunks) else {}
        text = chunks[idx]["text"] if idx < len(chunks) else ""
        results.append({"score": float(score), "meta": meta, "text": text, "idx": int(idx)})
    return results

def build_prompt(query, retrieved, chat_history, memory_k=3, max_context_chars=3000):
    sys_inst = (
        "Tu es un assistant repondant a des questions sur les projets de l'Etat senegalais suivi par le BOCS. "
        "Le BOCS est le Bureau Operationnel et Suivi de la primature senegalise chargee de suivie des projets et programmes prioritaires."
        "Chaque extrait est labelise par un token de citation du type [DOC_1], [DOC_2], etc."
        "Quand tu utilises une information dans un extrait, inclus un inline citation du type ( see [DOC_2]) ou bien [DOC_2] "
        "Quand la reponse n'est pas dans le contexte dis que tu ne sais pas."
        "Donne a la fin une courte liste des sources que tu as utilisees pour repondre and une ligne  "
        "Provide a short 'SOURCES' list at the end with the tokens you used and a one-line summary for each.\n\n"
    )
    

    convo = ""
    if chat_history:
        last_ex = chat_history[-memory_k:]
        if last_ex:
            convo += "Conversation history (most recent first):\n"
            for ex in reversed(last_ex):
                if ex.get("user"):
                    convo += f"User: {ex['user']}\n"
                if ex.get("assistant"):
                    convo += f"Assistant: {ex['assistant']}\n"
            convo += "\n"

    ctx = ""
    total_chars = 0
    for i, r in enumerate(retrieved, start=1):
        header = f"[DOC_{i}] Source: {r['meta'].get('source_file','-')} | page: {r['meta'].get('page','-')} | score: {r['score']:.4f}\n"
        block = header + r["text"] + "\n\n"
        if total_chars + len(block) > max_context_chars:
            break
        ctx += block
        total_chars += len(block)

    if ctx.strip() == "":
        ctx = "(No document context found.)\n\n"

    prompt = sys_inst + convo + "DOCUMENTS:\n" + ctx + "\nQUESTION:\n" + query + "\n\nAnswer with inline citations and a 'SOURCES' list at the end.\n\nAnswer:"
    return prompt

def generate_answer(prompt, llm_tokenizer, llm_model, device="cpu", max_new_tokens=256, temperature=0.2):
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = llm_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    decoded = llm_tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()

def print_bibliography(retrieved):
    print("\nBibliographie:")
    for i, r in enumerate(retrieved, start=1):
        meta = r["meta"]
        print(f"[DOC_{i}] {meta.get('source_file','-')} page {meta.get('page','-')} (score {r['score']:.4f})")

def repl_loop(index, chunks, emb_tok, emb_model, llm_tok, llm_model, device, top_k=5, memory_k=3):
    chat_history = []  # list of {"user":..., "assistant":...}
    print("chatBOCS (tapez 'exit' ou 'quit' pour arreter la convo).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        retrieved = retrieve(q, index, chunks, emb_tok, emb_model, device=device, top_k=top_k)
        prompt = build_prompt(q, retrieved, chat_history, memory_k=memory_k)
        t0 = time.time()
        ans = generate_answer(prompt, llm_tok, llm_model, device=device)
        dt = time.time() - t0
        
        chat_history.append({"user": q, "assistant": ans})
        print(f"\nAssistant (in {dt:.2f}s):\n{ans}\n")
        print_bibliography(retrieved)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="faiss.index")
    parser.add_argument("--chunks", default="chunks.jsonl")
    parser.add_argument("--embedding_model", default="google/embeddinggemma-300m")
    parser.add_argument("--llm_model", default="google/gemma-3-270m")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--memory_k", type=int, default=3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"Device: {device}")
    print("Loading index and chunks...")
    index, chunks = load_faiss(args.index, args.chunks)

    print("Loading embedding model...")
    emb_tok, emb_model = load_embedding_model(args.embedding_model, device=device)

    print("Loading LLM...")
    llm_tok = AutoTokenizer.from_pretrained(args.llm_model, use_fast=True)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model)
    llm_model.to(device)
    llm_model.eval()

    repl_loop(index, chunks, emb_tok, emb_model, llm_tok, llm_model, device, top_k=args.top_k, memory_k=args.memory_k)

if __name__ == "__main__":
    main()

