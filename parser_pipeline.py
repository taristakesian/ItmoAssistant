#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parser_pipeline.py
Парсинг программ ИТМО (AI, AI Product) без изменений bot.py и rag_openai.py.

Делает:
1) тянет страницы, ищет ссылки на учебные планы (включая Google Drive)
2) качает PDF, сохраняет текст (с маркерами страниц) и извлекает таблицы (pdfplumber)
3) нормализует таблицы в data/normalized/curriculum.csv (+ report.json)
4) собирает метаданные программ в data/program_meta/<slug>.json
   — включает manager + team; ДУБЛИРУЕТ их в "teachers", чтобы rag_openai.py их поднял

Выходные файлы:
- data/processed/*.txt                             (текст PDF для индекса)
- data/processed/*__tables/table_XXX.csv           (сырьё парсинга таблиц)
- data/normalized/curriculum.csv                   (итог для RAG)
- data/normalized/report.json                      (какие таблицы использованы)
- data/program_meta/<slug>.json                    (описание/стоимость/контакты/команда/teachers)
"""

from __future__ import annotations
import os, re, json, time, hashlib, contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
from slugify import slugify
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
# Пути и константы
# ────────────────────────────────────────────────────────────────────────────
BASE = Path("data")
RAW_DIR      = BASE / "raw";          RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR     = BASE / "processed";    PROC_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR     = BASE / "normalized";   NORM_DIR.mkdir(parents=True, exist_ok=True)
META_DIR     = BASE / "program_meta"; META_DIR.mkdir(parents=True, exist_ok=True)

PROGRAMS = {
    "ai": "https://abit.itmo.ru/program/master/ai",
    "ai_product": "https://abit.itmo.ru/program/master/ai_product",
}
# внешние сайты — на них обычно лежит "Команда/Эксперты"
EXTERNAL_SITES = {
    "ai": "https://ai.itmo.ru/",
    "ai_product": "https://aiproduct.itmo.ru/",
}

UA = "Mozilla/5.0 (compatible; itmo-rag-pipeline/4.0)"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
TIMEOUT = 30

PDF_ABIT_RX  = re.compile(r"/file_storage/.+\.pdf", re.I)
PDF_DRIVE_RX = re.compile(r"https?://drive\.google\.com/[^\s'\"<>]+", re.I)
ANCHOR_PLAN_RX = re.compile(r"(изучить\s+учебн|учебн[ыи]й\s*план)", re.I)

EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RX = re.compile(r"(?:\+?\d[\d\-\s\(\)]{7,}\d)")
TG_RX    = re.compile(r"(?:t\.me/|@)[\w\d_]{3,}")

COST_RX  = re.compile(r"(\d[\d\s]{3,})\s*(?:₽|руб|руб\.|RUB)", re.I)
TEAM_NAME_RX = re.compile(r"^[•\-\*]?\s*([A-ZА-ЯЁ][A-Za-zА-Яа-яёЁ\.\-’' ]{3,110})$")

# ────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ────────────────────────────────────────────────────────────────────────────
def http_get(url: str, stream: bool=False) -> requests.Response:
    r = SESSION.get(url, timeout=TIMEOUT, allow_redirects=True, stream=stream)
    r.raise_for_status()
    return r

def gdrive_direct(url: str) -> Optional[str]:
    """file/d/<id> → uc?export=download&id=<id>"""
    m = re.search(r"drive\.google\.com/(?:file/d/([^/]+)|open\?id=([^&]+))", url, re.I)
    if not m: return None
    fid = m.group(1) or m.group(2)
    return f"https://drive.google.com/uc?export=download&id={fid}"

def sanitize_filename_from_url(url: str) -> str:
    p = urlparse(url)
    base = Path(p.path).name or "file.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{p.netloc}_{base}_{digest}"

# ────────────────────────────────────────────────────────────────────────────
# Поиск ссылок на план (abit + внешние сайты)
# ────────────────────────────────────────────────────────────────────────────
def _is_curriculum_anchor(tag) -> bool:
    texts = [tag.get_text(" ", strip=True) or "", tag.get("title") or "", tag.get("aria-label") or ""]
    with contextlib.suppress(Exception):
        texts.append((tag.parent.get_text(" ", strip=True) or "")[:200])
    return bool(ANCHOR_PLAN_RX.search(" ".join(texts)))

def _extract_drive_anywhere(soup: BeautifulSoup, base: str) -> List[str]:
    out=[]
    # в <a>
    for a in soup.select("a[href]"):
        href = urljoin(base, a["href"].strip())
        if "drive.google.com" in href.lower():
            out.append(href)
    # в <script> и data-атрибутах
    for s in soup.find_all("script"):
        txt = (s.string or s.text or "")
        out += PDF_DRIVE_RX.findall(txt)
    for tag in soup.find_all(True):
        for attr in ("onclick","data-href","data-url"):
            val = tag.get(attr)
            if val:
                out += PDF_DRIVE_RX.findall(val)
    # uniq
    uniq=[]; seen=set()
    for u in out:
        u=urljoin(base,u.strip().strip(')"\''))
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def find_curriculum_links(page_url: str, slug: str) -> List[str]:
    html = http_get(page_url).text
    soup = BeautifulSoup(html, "lxml")
    found: List[str] = []

    # 1) Явные PDF на abit
    for a in soup.select("a[href]"):
        href = urljoin(page_url, a["href"].strip())
        if PDF_ABIT_RX.search(href):
            found.append(href)
        # 2) Drive в якоре "Учебный план"
        if "drive.google.com" in href.lower() and (_is_curriculum_anchor(a) or "учеб" in (a.get_text(" ", strip=True) or "").lower()):
            found.append(href)

    # 3) Drive где угодно
    found += _extract_drive_anywhere(soup, page_url)

    # 4) Внешний сайт программы (если указан): ищем drive и/или ссылки с текстом «учеб»
    ext = EXTERNAL_SITES.get(slug)
    if ext:
        with contextlib.suppress(Exception):
            shtml = http_get(ext).text
            ssoup = BeautifulSoup(shtml, "lxml")
            for a in ssoup.select("a[href]"):
                href = urljoin(ext, a["href"].strip())
                txt = (a.get_text(" ", strip=True) or "").lower()
                if "drive.google.com" in href.lower() and ("учеб" in txt or True):
                    found.append(href)

    # uniq
    return list(dict.fromkeys(found))

def download_pdf(url: str) -> Path:
    direct = gdrive_direct(url) or url
    fname = sanitize_filename_from_url(direct)
    out = RAW_DIR / fname
    if out.exists():
        return out
    with http_get(direct, stream=True) as r:
        with open(out,"wb") as f:
            for chunk in r.iter_content(1024*64):
                if chunk:
                    f.write(chunk)
    return out

# ────────────────────────────────────────────────────────────────────────────
# PDF → текст и таблицы
# ────────────────────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    parts=[]
    with pdfplumber.open(pdf_path) as pdf:
        for i, pg in enumerate(pdf.pages, start=1):
            txt = pg.extract_text() or ""
            parts.append(f"--- PAGE {i} ---\n{txt}")
    return "\n\n".join(parts)

def extract_pdf_tables(pdf_path: Path) -> List[pd.DataFrame]:
    frames=[]
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            # пробуем по линиям
            try:
                tables = pg.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_x_tolerance": 5,
                    "intersection_y_tolerance": 5,
                }) or []
            except Exception:
                tables = []
            # если линий нет — fallback на потоковую эвристику: делим строки по 2+ пробелам
            if not tables:
                text = pg.extract_text() or ""
                rows = []
                for line in text.splitlines():
                    cols = [c.strip() for c in re.split(r"\s{2,}|\t+", line.strip()) if c.strip()]
                    if len(cols) >= 3:
                        rows.append(cols[:4])  # курс, кредиты, часы — чаще всего 3–4 колонки
                if rows:
                    # выравниваем кол-во колонок
                    width = max(len(r) for r in rows)
                    norm = [r + [""]*(width-len(r)) for r in rows]
                    tables = [norm]
            for t in tables:
                df = pd.DataFrame(t)
                # чистка пустоты
                df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
                df.dropna(how="all", axis=0, inplace=True)
                df.dropna(how="all", axis=1, inplace=True)
                if not df.empty:
                    frames.append(df)
    return frames

def save_processed(slug: str, pdf_path: Path, text: str, tables: List[pd.DataFrame]) -> None:
    ts = int(time.time())
    base = slugify(slug)
    # текст
    txt_path = PROC_DIR / f"{base}__{pdf_path.stem}__{ts}.txt"
    txt_path.write_text(text, encoding="utf-8")
    # таблицы
    tdir = PROC_DIR / f"{base}__{pdf_path.stem}__{ts}__tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(tables, 1):
        (tdir / f"table_{i:03d}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    # мета (для normalize)
    meta = {"program": slug, "source_pdf": str(pdf_path), "text_file": str(txt_path), "tables_dir": str(tdir)}
    (PROC_DIR / f"{base}__{pdf_path.stem}__{ts}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ────────────────────────────────────────────────────────────────────────────
# Нормализация таблиц → curriculum.csv
# ────────────────────────────────────────────────────────────────────────────
RX_SEM_INLINE = re.compile(r"\b(\d)\s*семестр\b", re.I)
RX_MULTI_SEM  = re.compile(r"^\s*(\d(?:\s*,\s*\d)+)\s*$")
RX_DIGIT      = re.compile(r"^\s*[1-8]\s*$")
AGG_RX        = re.compile(r"^(обязательные дисциплины|пул выборных дисциплин|микромодули|soft\s*skills)", re.I)

TYPE_HINTS = (
    (re.compile(r"обязательн", re.I), "обязательная"),
    (re.compile(r"пул\s*выборн", re.I), "электив"),
    (re.compile(r"soft\s*skills", re.I), "soft"),
    (re.compile(r"\bпрактика\b", re.I), "практика"),
    (re.compile(r"итогов\w*\s+аттестац", re.I), "гия"),
    (re.compile(r"факультатив", re.I), "факультатив"),
    (re.compile(r"язык", re.I), "язык"),
)

def _to_num(x) -> Optional[float]:
    if pd.isna(x): return None
    s = str(x).strip().replace(",", ".")
    s = re.sub(r"[^\d\.]", "", s)
    if not s: return None
    try: return float(s)
    except: return None

def _type_from_text(text: str, current: Optional[str]) -> Optional[str]:
    l = (text or "").lower()
    for rx, val in TYPE_HINTS:
        if rx.search(l): return val
    return current

def _extract_rows_from_df(df: pd.DataFrame, program: str, source_pdf: str) -> List[Dict[str, Any]]:
    rows=[]
    df = df.copy()
    df.columns = list(range(df.shape[1]))
    for need in range(4):
        if need not in df.columns: df[need] = ""
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
    df.fillna("", inplace=True)

    ctx = {"semester": None, "type": None}
    for _, r in df.iterrows():
        c0 = str(r[0]).strip()
        c1 = str(r[1]).strip()
        c2 = r[2]; c3 = r[3]

        # обновим контекст по "шапкам"
        if AGG_RX.match(c1):
            tp = _type_from_text(c1, None)
            if tp: ctx["type"] = tp
            m = RX_SEM_INLINE.search(c1)
            if m: ctx["semester"] = int(m.group(1))
            continue

        # 1) "SEM , COURSE , CRED , HOURS"
        if RX_DIGIT.match(c0) and c1 and _to_num(c2) is not None and _to_num(c3) is not None:
            rows.append({
                "program": program,
                "semester": int(c0),
                "course": c1.rstrip("."),
                "type": ctx.get("type"),
                "credits": _to_num(c2),
                "hours": int(_to_num(c3)) if _to_num(c3) is not None else None,
                "source_pdf": source_pdf
            }); continue

        # 1a) "1, 2, 3 , COURSE , ..."
        if RX_MULTI_SEM.match(c0) and c1 and _to_num(c2) is not None and _to_num(c3) is not None:
            sems = [int(x.strip()) for x in c0.split(",") if x.strip().isdigit()]
            for sem in sems:
                rows.append({
                    "program": program,
                    "semester": sem,
                    "course": c1.rstrip("."),
                    "type": ctx.get("type"),
                    "credits": _to_num(c2),
                    "hours": int(_to_num(c3)) if _to_num(c3) is not None else None,
                    "source_pdf": source_pdf
                }); 
            continue

        # 2) "COURSE , '... N семестр' , CRED , HOURS"
        m_in = RX_SEM_INLINE.search(c1)
        if m_in and _to_num(c2) is not None and _to_num(c3) is not None:
            sem = int(m_in.group(1))
            course = c0 if c0 else re.sub(RX_SEM_INLINE, "", c1).strip().rstrip(".")
            if not course: course = c1.rstrip(".")
            rows.append({
                "program": program,
                "semester": sem,
                "course": course,
                "type": ctx.get("type"),
                "credits": _to_num(c2),
                "hours": int(_to_num(c3)) if _to_num(c3) is not None else None,
                "source_pdf": source_pdf
            }); continue

        # 3) "COURSE , CRED , HOURS" при заданном контексте семестра
        if c1 and _to_num(c2) is not None and _to_num(c3) is not None and ctx.get("semester") is not None:
            rows.append({
                "program": program,
                "semester": int(ctx["semester"]),
                "course": c1.rstrip("."),
                "type": ctx.get("type"),
                "credits": _to_num(c2),
                "hours": int(_to_num(c3)) if _to_num(c3) is not None else None,
                "source_pdf": source_pdf
            }); continue

    return rows

def _looks_like_curriculum_tables_dir(tdir: Path) -> bool:
    files = sorted(tdir.glob("table_*.csv"))
    if not files: return False
    try:
        df = pd.read_csv(files[0], header=None, nrows=2, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(files[0], header=None, nrows=2, dtype=str, encoding="utf-8-sig")
    except Exception:
        return False
    if df.shape[0] < 2: return False
    row2 = " ".join(str(x) for x in df.iloc[1].tolist() if pd.notna(x))
    return "учебный план" in row2.lower() or len(df.columns) >= 3

def normalize_curriculum() -> Tuple[Path, Path]:
    """Собираем из всех *__tables CSV → data/normalized/curriculum.csv + report.json"""
    tdirs = [td for td in PROC_DIR.glob("*__*__*__tables") if _looks_like_curriculum_tables_dir(td)]
    if not tdirs:
        tdirs = list(PROC_DIR.glob("*__*__*__tables"))

    rows_all=[]; report=[]
    for tdir in tdirs:
        program = tdir.name.split("__")[0] if tdir.name != "__tables" else tdir.parent.name.split("__")[0]
        source_pdf = ""
        with contextlib.suppress(Exception):
            meta_json = sorted(tdir.parent.glob(f"{program}__*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if meta_json:
                source_pdf = (json.loads(meta_json[0].read_text(encoding="utf-8"))).get("source_pdf","")

        used=0
        for csvp in sorted(tdir.glob("table_*.csv")):
            try:
                df = pd.read_csv(csvp, dtype=str, encoding="utf-8").fillna("")
            except UnicodeDecodeError:
                df = pd.read_csv(csvp, dtype=str, encoding="utf-8-sig").fillna("")
            except Exception:
                continue
            rows = _extract_rows_from_df(df, program, source_pdf)
            if rows:
                rows_all.extend(rows); used += 1

        report.append({
            "program": program, "tables_dir": str(tdir),
            "source_pdf": source_pdf, "tables_total": len(list(tdir.glob("table_*.csv"))),
            "tables_used": used
        })

    df = pd.DataFrame(rows_all, columns=["program","semester","course","type","credits","hours","source_pdf"])
    if not df.empty:
        df.drop_duplicates(subset=["program","semester","course"], inplace=True)
        df["semester"] = pd.to_numeric(df["semester"], errors="coerce").astype("Int64")
        df["credits"]  = pd.to_numeric(df["credits"], errors="coerce")
        df["hours"]    = pd.to_numeric(df["hours"], errors="coerce").astype("Int64")
        df.sort_values(["program","semester","course"], na_position="last", inplace=True)

    out_csv = NORM_DIR / "curriculum.csv"
    out_rep = NORM_DIR / "report.json"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv, out_rep

# ────────────────────────────────────────────────────────────────────────────
# Метаданные программы (описание/контакты/стоимость/команда/teachers)
# ────────────────────────────────────────────────────────────────────────────
def _uniq(seq):
    seen=set(); out=[]
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def scrape_manager_from_abit(soup: BeautifulSoup) -> dict:
    manager={}
    # ищем маркер
    marker = soup.find(string=re.compile(r"Менеджер программы", re.I))
    if marker:
        blk = marker.parent
        # ближайшие 3–6 текстовых узлов пробуем как имя
        for sib in blk.find_all_next(string=True, limit=6):
            s = (sib or "").strip()
            if s and not re.search(r"Менеджер программы", s, re.I):
                manager["name"] = s; break
    text = soup.get_text("\n", strip=True)
    em = EMAIL_RX.findall(text); ph = PHONE_RX.findall(text)
    if em: manager["email"]=em[0]
    if ph: manager["phone"]=ph[0]
    if manager:
        manager["role"]="Менеджер программы"
    return manager

def scrape_team_from_external(url: str) -> list[dict]:
    try:
        html = http_get(url).text
    except Exception:
        return []
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)
    lower = text.lower()
    start = lower.find("эксперт")
    if start == -1:
        start = lower.find("команда")
    if start == -1:
        start = 0
    end = len(text)
    for m in ["профессиональные роли","вопросы","как поступить","контакты"]:
        i = lower.find(m, start+10)
        if i != -1:
            end = min(end, i)
    block = text[start:end].splitlines()

    team=[]; i=0
    while i < len(block):
        line = block[i].strip()
        if not line:
            i+=1; continue
        m = TEAM_NAME_RX.match(line)
        if m:
            name = " ".join(m.group(1).split())
            role = None
            if i+1 < len(block):
                nxt = block[i+1].strip()
                if not TEAM_NAME_RX.match(nxt):
                    role = nxt
            team.append({"name": name, "role": role, "source": url})
            i += 2
            continue
        i += 1

    # дедуп
    uniq=[]; seen=set()
    for t in team:
        nm = t.get("name","")
        if len(nm) < 3: continue
        key=(nm, t.get("role") or "")
        if key in seen: continue
        seen.add(key); uniq.append(t)
    return uniq

def scrape_program_meta():
    for slug, url in PROGRAMS.items():
        html = http_get(url).text
        soup = BeautifulSoup(html, "lxml")
        title = (soup.select_one("h1") or soup.select_one(".program__title") or soup.select_one(".hero__title"))
        desc  = (soup.select_one(".program__description") or soup.select_one(".lead") or soup.select_one("p"))
        meta = {
            "slug": slug,
            "sources": {"abit_page": url},
            "program": {
                "title": title.get_text(" ", strip=True) if title else "",
                "description": desc.get_text(" ", strip=True) if desc else "",
            },
            "links": {"external_sites": [], "social": [], "curriculum_links": [], "price_links": []},
            "contacts": {"emails": [], "phones": [], "telegram": []},
            "manager": None,
            "team": [],
            "teachers": [],  # ← для совместимости с rag_openai.py
            "costs": []
        }
        # ссылки, соцсети, цены, drive
        for a in soup.select("a[href]"):
            href = urljoin(url, a["href"].strip())
            txt  = (a.get_text(" ", strip=True) or "").lower()
            host = urlparse(href).netloc
            if "drive.google.com" in href:
                meta["links"]["curriculum_links"].append(href)
            if "price" in href or "stoimost" in href:
                meta["links"]["price_links"].append(href)
            if any(d in host for d in ("t.me","vk.com","youtube.com","x.com","twitter.com","instagram.com","linkedin.com","facebook.com")):
                meta["links"]["social"].append(href)
            if any(w in txt for w in ("сайт","о программе","подробнее")) and host not in ("abit.itmo.ru",):
                meta["links"]["external_sites"].append(href)

        text_all = soup.get_text("\n", strip=True)
        meta["contacts"]["emails"]   = _uniq(EMAIL_RX.findall(text_all))
        meta["contacts"]["phones"]   = _uniq(PHONE_RX.findall(text_all))
        meta["contacts"]["telegram"] = _uniq(TG_RX.findall(text_all))

        for m in COST_RX.finditer(text_all):
            amount = int(re.sub(r"\s+","", m.group(1)))
            meta["costs"].append({"amount": amount, "currency":"RUB", "source": url})

        # менеджер
        mgr = scrape_manager_from_abit(soup)
        if mgr:
            meta["manager"] = mgr
            # дублируем в teachers
            t = {"name": mgr.get("name"), "role": "Менеджер программы",
                 "profile_url": None, "source": url}
            if t["name"]:
                meta["teachers"].append(t)

        # команда с внешних сайтов
        ext_sites = list(dict.fromkeys(meta["links"]["external_sites"]))
        if slug in EXTERNAL_SITES and EXTERNAL_SITES[slug] not in ext_sites:
            ext_sites.append(EXTERNAL_SITES[slug])

        team_all=[]
        for site in ext_sites:
            team_all.extend(scrape_team_from_external(site))

        # дедуп + дублируем в teachers
        uniq=[]; seen=set()
        for t in team_all:
            key=(t.get("name"), t.get("role"))
            if key in seen: continue
            seen.add(key); uniq.append(t)
            # в teachers
            if t.get("name"):
                meta["teachers"].append({"name": t["name"], "role": (t.get("role") or "Эксперт"),
                                         "profile_url": None, "source": t.get("source")})

        # финал
        meta["team"] = uniq
        for k in ("external_sites","social","curriculum_links","price_links"):
            meta["links"][k] = _uniq(meta["links"][k])

        out = META_DIR / f"{slug}.json"
        out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("meta →", out)

# ────────────────────────────────────────────────────────────────────────────
# Пайплайн
# ────────────────────────────────────────────────────────────────────────────
def run_all():
    # 1) ссылки планов → качаем PDFs → текст + таблицы
    for slug, url in PROGRAMS.items():
        print(f"\n=== Программа: {slug} | {url}")
        links = find_curriculum_links(url, slug)
        if not links:
            print("  ! Не нашли ссылок на учебный план.")
            continue
        print(f"  Нашли ссылок: {len(links)}")
        for l in links:
            try:
                pdf = download_pdf(l)
                print("  ↓", pdf.name)
                text = extract_pdf_text(pdf)
                tables = extract_pdf_tables(pdf)
                save_processed(slug, pdf, text, tables)
                print(f"    ✓ текст={len(text)} символов, таблиц={len(tables)}")
            except Exception as e:
                print("    ✗ Ошибка:", e)

    # 2) нормализация → curriculum.csv
    out_csv, out_rep = normalize_curriculum()
    print("\n✓ curriculum.csv →", out_csv)
    print("  report.json     →", out_rep)

    # 3) метаданные программ (включая manager + team + teachers)
    scrape_program_meta()
    print("✓ program_meta/*.json обновлены")
