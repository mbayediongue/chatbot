import streamlit as st
import faiss
import numpy as np
import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import embed_texts, load_json, mean_pooling
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
print("HF token loaded:", bool(hf_token)) 

EMBEDDING_MODEL = "google/embeddinggemma-300m"
LLM_MODEL = "google/gemma-3-270m"
FAISS_INDEX_PATH = "faiss.index"
META_PATH = "metadata.json"
CHUNKS_PATH = "chunks.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 3000
DEFAULT_MEMORY_K = 3  # number of previous exchanges (user+assistant) to include

@st.cache_resource
def load_faiss_and_meta(index_path=FAISS_INDEX_PATH, meta_path=META_PATH):
    if not os.path.exists(index_path):
        st.error(f"FAISS index not found at {index_path}. Run ingest.py first.")
        return None, None, None
    index = faiss.read_index(index_path)
    metas = load_json(meta_path)
    chunks = []
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    else:
        chunks = [{"meta": m, "text": ""} for m in metas]
    return index, metas, chunks

@st.cache_resource
def load_models(embedding_model=EMBEDDING_MODEL, llm_model=LLM_MODEL, device=DEVICE):
    from transformers import AutoTokenizer, AutoModel
    emb_tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=True)
    emb_model = AutoModel.from_pretrained(embedding_model)
    emb_model.to(device)
    emb_model.eval()

    tok = AutoTokenizer.from_pretrained(llm_model, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(llm_model)
    llm.to(device)
    llm.eval()

    return (emb_tokenizer, emb_model), (tok, llm)

def embed_query_text(query_text: str, emb_model_pair, device=DEVICE):
    emb_tokenizer, emb_model = emb_model_pair
    enc = emb_tokenizer([query_text], truncation=True, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = emb_model(input_ids=input_ids, attention_mask=attention_mask)
        embs = mean_pooling(out, attention_mask)
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    return embs.cpu().numpy().astype("float32")

def retrieve(query, index, chunks, emb_model_pair, top_k=DEFAULT_TOP_K):
    q_emb = embed_query_text(query, emb_model_pair)
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        meta = chunks[idx]["meta"] if idx < len(chunks) else {}
        text = chunks[idx]["text"] if idx < len(chunks) else ""
        results.append({"score": float(dist), "meta": meta, "text": text, "index": int(idx)})
    return results

def build_prompt_with_memory(query, retrieved, chat_history, memory_k=DEFAULT_MEMORY_K, max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
    sys_inst = (
        "You are an assistant that answers user questions using only the provided context snippets. "
        "Each snippet is labeled with citation tokens like [DOC_1], [DOC_2], etc. "
        "When you use information from a snippet, include an inline citation such as (see [DOC_2]) or simply [DOC_2]. "
        "If the answer is not contained in the provided snippets or cannot be determined, reply honestly and say you don't know. "
        "At the end of your answer include a short 'SOURCES' list of the tokens you used and a one-line summary for each cited token.\n\n"
    )

    convo_section = ""
    if chat_history:
        last_exchanges = chat_history[-memory_k:]
        if last_exchanges:
            convo_section += "Conversation history (most recent first):\n"
            for ex in reversed(last_exchanges):  # show most recent first for the model
                u = ex.get("user","").strip()
                a = ex.get("assistant","").strip()
                if u:
                    convo_section += f"User: {u}\n"
                if a:
                    convo_section += f"Assistant: {a}\n"
            convo_section += "\n"

    ctx = ""
    used = 0
    total_chars = 0
    for i, r in enumerate(retrieved, start=1):
        header = f"[DOC_{i}] Source: {r['meta'].get('source_file','-')} | page: {r['meta'].get('page','-')} | chunk: {r['meta'].get('chunk_index','-')} | score: {r['score']:.4f}\n"
        block = header + r["text"] + "\n\n"
        if total_chars + len(block) > max_context_chars:
            break
        ctx += block
        total_chars += len(block)
        used += 1

    if ctx.strip() == "":
        ctx = "(No document context was retrieved.)\n\n"

    prompt = sys_inst + convo_section + "DOCUMENTS:\n" + ctx + "\nUSER QUESTION:\n" + query + "\n\nAnswer with inline citations and a short 'SOURCES' list at the end.\n\nAnswer:"
    return prompt

def generate_answer(prompt, llm_pair, max_new_tokens=256, temperature=0.2, device=DEVICE):
    tok, llm = llm_pair
    inputs = tok(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=False)
    decoded = tok.decode(out[0], skip_special_tokens=True)

    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    return decoded.strip()


st.set_page_config(page_title="chatBOCS", layout="wide")
st.title("chatBOCS (version test)")

index, metas, chunks = load_faiss_and_meta()
if index is None:
    st.stop()

st.sidebar.markdown("## Parametres")
#st.sidebar.write(f"Device: {DEVICE}")
top_k = st.sidebar.number_input("Top k retrieval", value=DEFAULT_TOP_K, min_value=1, max_value=20)
memory_k = st.sidebar.number_input("Num. de questions en memoire", value=DEFAULT_MEMORY_K, min_value=0, max_value=20)
max_context_chars = st.sidebar.number_input("Nombre maximal de characteres", value=DEFAULT_MAX_CONTEXT_CHARS, min_value=500, max_value=20000)
max_tokens = st.sidebar.number_input("Nombre Maximal de Toekns", value=256, min_value=32, max_value=1024)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

st.info("Chargement des modeles ...")
(emb_tokenizer, emb_model), (tok, llm) = load_models()

emb_model_pair = (emb_tokenizer, emb_model)
llm_pair = (tok, llm)

if "conversations" not in st.session_state:
    st.session_state.conversations = [] 


col1, col2 = st.columns([3,1])
with col1:
    user_query = st.text_area("Votre question", height=120)
    if st.button("Envoyer"):
        if not user_query.strip():
            st.warning("Veuillez ecrire votre question.")
        else:
            with st.spinner("Collection des informations..."):
                retrieved = retrieve(user_query, index, chunks, emb_model_pair, top_k=top_k)
            prompt = build_prompt_with_memory(user_query, retrieved, st.session_state.conversations, memory_k=memory_k, max_context_chars=max_context_chars)
            st.session_state.conversations.append({"user": user_query, "assistant": None, "retrieved": retrieved})
            with st.spinner("Generating answer..."):
                start = time.time()
                answer = generate_answer(prompt, llm_pair, max_new_tokens=int(max_tokens), temperature=float(temperature))
                duration = time.time() - start
            # Save assistant answer back to last conversation entry
            st.session_state.conversations[-1]["assistant"] = answer
            st.session_state.conversations[-1]["time"] = duration
            st.experimental_rerun()

with col2:
    st.markdown("### Documnets utilises (top k)")
    if st.session_state.conversations:
        last = st.session_state.conversations[-1]
        retrieved = last.get("retrieved", [])
        for i, r in enumerate(retrieved, start=1):
            meta = r["meta"]
            st.markdown(f"**[DOC_{i}]** {meta.get('source_file','-')} — page {meta.get('page','-')} — chunk {meta.get('chunk_index','-')}")
            txt = r["text"]
            st.write(txt[:400] + ("..." if len(txt) > 400 else ""))
            st.write(f"score: {r['score']:.4f}")
            st.markdown("---")
    else:
        st.write("Pas encore de question posee.")

st.markdown("## Conversation (le plus recent en premier)")
for entry in reversed(st.session_state.conversations):
    st.markdown("**User:**")
    st.write(entry["user"])
    st.markdown("**ChatBOCS:**")
    st.write(entry["assistant"] or "(en cours...)")
    #st.write(f"*Temps de reponse: {entry.get('time', 0):.2f}s*")
    if st.checkbox(f"Voir les documents utilises pour la reponse", key=f"ret_{hash(entry['user'])}"):
        for i, r in enumerate(entry.get("retrieved", []), start=1):
            meta = r["meta"]
            st.markdown(f"- [DOC_{i}] {meta.get('source_file','-')} page {meta.get('page','-')} (score {r['score']:.4f})")
            st.write(r["text"][:800] + ("..." if len(r["text"])>800 else ""))
    st.markdown("---")
