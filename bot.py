#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот ITMO-RAG
Команды:
  /ask <вопрос> [--prog ai|ai_product] [--k 8]
  /prog – показать доступные программы
"""
import os, re, asyncio
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from rag_openai import get_retriever, embedder, COLL_NAME, CHROMA_DIR, ask as rag_ask

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан в .env")

# ——— быстрый список программ (по коллекции Chroma) ———
def list_programs() -> list[str]:
    from langchain_chroma import Chroma
    vect = Chroma(collection_name=COLL_NAME,
                  persist_directory=str(CHROMA_DIR),
                  embedding_function=embedder())
    metas = vect._collection.get(include=["metadatas"])
    progs = {m.get("program") for m in metas["metadatas"] if m and m.get("program")}
    return sorted(filter(None, progs))

HELP = (
    "Я отвечаю на вопросы по учебным программам ИТМО.\n"
    "`/ask <вопрос>` – получить ответ\n"
    "Опции:\n"
    "  `--prog ai` | `ai_product` – фильтр по программе\n"
    "  `--k 6`     – сколько контекстов использовать (по умолчанию 8)\n"
    "`/prog` – список программ\n"
    "`/help` – эта справка"
)

def parse_params(text: str):
    """выделяем --prog и --k"""
    prog = None; k = 8
    if m := re.search(r"--prog\s+(\w+)", text):
        prog = m.group(1).strip()
    if m := re.search(r"--k\s+(\d+)", text):
        k = int(m.group(1))
    # сам вопрос без опций
    q = re.sub(r"--(prog|k)\s+\S+", "", text).strip()
    return q, prog, k

# ——— aiogram handlers ———
bot = Bot(BOT_TOKEN, parse_mode=None)
dp  = Dispatcher()

@dp.message(Command("start", "help"))
async def cmd_help(msg: Message):
    await msg.answer(HELP, disable_web_page_preview=True)

@dp.message(Command("prog"))
async def cmd_prog(msg: Message):
    progs = list_programs()
    await msg.answer("Доступные программы:\n" + "\n".join(f"• `{p}`" for p in progs), parse_mode="Markdown")

@dp.message(Command("ask"))
async def cmd_ask(msg: Message):
    q_full = msg.text.removeprefix("/ask").strip()
    if not q_full:
        await msg.answer("Формат: `/ask <вопрос> [--prog ai] [--k 6]`", parse_mode="Markdown")
        return
    q, prog, k = parse_params(q_full)
    await msg.answer("_Ищу ответ…_", parse_mode="Markdown")
    # RAG-ответ (reuse функции)
    try:
        answer_lines = rag_ask(q, prog, k, with_llm=True).splitlines()
        # rag_ask уже печатает; мы хотим текст → изменим rag_openai.ask:
        # return конечный ответ строкой, а не print
    except Exception as e:
        await msg.answer(f"Ошибка: {e}")
        return
    await msg.answer("\n".join(answer_lines[:4096]), disable_web_page_preview=True)

async def main():
    print("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
