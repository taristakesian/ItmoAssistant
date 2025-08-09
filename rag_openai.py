#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG CLI (LangChain + OpenAI API)

  python rag_openai.py index
  python rag_openai.py ask "кто преподаёт на ai_product?" --prog ai_product --llm
"""

from __future__ import annotations
import os, re, json, pickle, argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma               
from langchain_openai import OpenAIEmbeddings     #
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from openai import OpenAI, APIConnectionError

# ----- ENV -----
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")      # обязателен
OPENAI_API_BASE  = os.getenv("OPENAI_API_BASE")     # оставь пустым, если обычный OpenAI
CHAT_MODEL       = os.getenv("OPENAI_MODEL", "gpt-5-nano")
EMB_MODEL        = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMB_BATCH_SIZE = 64

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан в .env")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE or None)

SYSTEM_PROMPT = Path("prompts/system.txt").read_text(encoding="utf-8")

# ----- PATHS -----
BASE = Path("data")
PROCESSED = BASE / "processed"
CURR_CSV  = BASE / "normalized" / "curriculum.csv"
META_DIR  = BASE / "program_meta"

IDX_DIR   = BASE / "index"
CHROMA_DIR = IDX_DIR / "chroma"
BM25_PATH  = IDX_DIR / "bm25.pkl"
IDX_DIR.mkdir(parents=True, exist_ok=True)

COLL_NAME = "itmo_kb"   # имя коллекции ≥3 символов

CHUNK_SZ = 900
CHUNK_OV = 120

# ----- LOAD DOCS -----
def load_docs() -> List[Document]:
    docs: List[Document] = []

    # PDF → txt
    for p in PROCESSED.glob("*.txt"):
        prog = p.name.split("__", 1)[0]
        raw  = p.read_text(encoding="utf-8", errors="ignore")
        parts = re.split(r"---\s*PAGE\s*(\d+)\s*---", raw)
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                page = int(parts[i]) if parts[i].isdigit() else None
                docs.append(Document(page_content=parts[i+1].strip(),
                                     metadata={"program": prog,"source":str(p),"page":page,"kind":"pdf"}))
        else:
            docs.append(Document(page_content=raw.strip(),
                                 metadata={"program": prog,"source":str(p),"kind":"pdf"}))

    # curriculum.csv
    if CURR_CSV.exists():
        df = pd.read_csv(CURR_CSV, dtype=str).fillna("")
        for _, r in df.iterrows():
            text = f'Курс: {r["course"]}. Семестр: {r["semester"]}. Тип: {r["type"]}. ЗЕТ: {r["credits"]}.'
            docs.append(Document(page_content=text,
                                 metadata={"program": r["program"],"source":"curriculum.csv","kind":"curr"}))

    # meta json
    for jp in META_DIR.glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8", errors="ignore"))
        prog = data.get("slug") or jp.stem
        about = (data.get("program") or {}).get("description","")
        if about:
            docs.append(Document(page_content=about,
                                 metadata={"program": prog,"source":str(jp),"kind":"about"}))
        for t in data.get("teachers", []):
            # безопасные поля
            name = (t.get("name") or "").strip()
            if not name:
                continue  # пропускаем мусорные записи без имени

            role = (t.get("role") or "").strip()

            # source всегда строка: profile_url → source → путь к json
            src = (t.get("profile_url") or t.get("source") or str(jp) or "").strip()

            docs.append(Document(
                page_content=f"Преподаватель: {name}. {role}" if role else f"Преподаватель: {name}.",
                metadata={
                    "program": prog,
                    "source": src,             # ← гарантированно строка
                    "kind": "teacher",
                    "teacher_name": name,      # доп. мета — удобно для фильтров/отладки
                    "teacher_role": role
                }
            ))
        for c in data.get("costs", []):
            txt = f'Стоимость: {c.get("amount")} {c.get("currency","RUB")}' if c.get("amount") else "Стоимость: см. источник."
            src = (c.get("source") or str(jp) or "").strip()
            docs.append(Document(
                page_content=txt,
                metadata={"program": prog, "source": src, "kind": "cost"}
            ))

    return docs

# ----- split -----
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SZ, chunk_overlap=CHUNK_OV,
                                          separators=["\n\n","\n"," ",""])

def chunk_docs(docs: List[Document]) -> List[Document]:
    out=[]
    for d in docs:
        if d.metadata.get("kind") in {"curr","teacher","cost"}:
            out.append(d)
        else:
            out.extend(splitter.split_documents([d]))
    return out

# ----- embeddings -----
def embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMB_MODEL,
        chunk_size=EMB_BATCH_SIZE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE or None,
    )

# ----- index -----
def build_index():
    docs = chunk_docs(load_docs())
    print("Docs:", len(docs))

    vect = Chroma(
        collection_name=COLL_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder()
    )
    try:
        vect.delete_collection()
    except Exception:
        pass
    vect = Chroma(
        collection_name=COLL_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder()
    )

    vect.add_documents(docs)      # persist() больше не нужен
    # vect._client.persist()      # ← если хочешь явно

    bm25 = BM25Retriever.from_documents(docs); bm25.k = 20
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("Индекс готов:", CHROMA_DIR)


# ----- retriever -----
def get_retriever(prog: Optional[str]):
    """dense - всегда;  dense+bm25 - если без фильтра программы"""
    dense = Chroma(
        collection_name=COLL_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedder()
    ).as_retriever(search_kwargs={
        "k": 24,
        "filter": ({"program": prog} if prog else None)
    })

    if prog:
        return dense                          # фильтр есть → только dense

    # без фильтра программы → гибрид dense + bm25
    bm25: BM25Retriever = pickle.load(open(BM25_PATH, "rb"))
    bm25.k = 16
    return EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])

# ----- ask -----
def ask(q:str, prog:str|None, k:int, with_llm:bool):
    retr = get_retriever(prog)
    docs = retr.invoke(q)[:k]

    for i,d in enumerate(docs,1):
        src = Path(d.metadata.get("source","")).name
        pg  = d.metadata.get("page")
        print(f"[{i}] {d.page_content[:200].replace('\\n',' ')}")
        print(f"    src: {src}{' p'+str(pg) if pg else ''}")

    if not with_llm:
        return "\n".join([d.page_content for d in docs])

    ctx = "\n\n".join(f"[{i}] {d.page_content}" for i,d in enumerate(docs,1))
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":f"Вопрос: {q}\n\nКонтекст:\n{ctx}"}]
        )
        print("\n--- ANSWER ---\n"+resp.choices[0].message.content.strip())
        return resp.choices[0].message.content.strip()
    except APIConnectionError as e:
        print("OpenAI APIConnectionError:", e)

# ----- CLI -----
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("index")
    a=sub.add_parser("ask")
    a.add_argument("q",nargs="+")
    a.add_argument("--prog"); a.add_argument("--k",type=int,default=8)
    a.add_argument("--llm",action="store_true")
    args = ap.parse_args()

    if args.cmd=="index": build_index()
    elif args.cmd=="ask": ask(" ".join(args.q), args.prog, args.k, args.llm)
    else: ap.print_help()
