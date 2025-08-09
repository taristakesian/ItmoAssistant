#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — единая точка старта (не меняет bot.py и rag_openai.py)

Команды:
  python main.py parse     # парсер: страницы → PDF → таблицы → curriculum.csv + meta
  python main.py index     # индексация (использует rag_openai.build_index)
  python main.py bot       # запуск Telegram-бота
  python main.py up        # полный цикл: parse -> index -> bot
"""
import asyncio
import subprocess
import sys

from parser_pipeline import run_all as parse_run_all

# rag_openai.py должен лежать рядом и содержать функцию build_index()
try:
    from rag_openai import build_index as rag_build_index
except Exception:
    rag_build_index = None

def cmd_parse():
    parse_run_all()

def cmd_index():
    if rag_build_index is None:
        print("rag_openai.build_index() не найден. Запускаю как скрипт: python rag_openai.py index")
        subprocess.check_call([sys.executable, "rag_openai.py", "index"])
    else:
        rag_build_index()

def cmd_bot():
    # Пытаемся импортировать bot.main(), иначе запускаем как отдельный процесс
    try:
        import bot
        if hasattr(bot, "main"):
            asyncio.run(bot.main())
        else:
            print("bot.main() не найден. Запускаю: python bot.py")
            subprocess.check_call([sys.executable, "bot.py"])
    except Exception:
        print("Не удалось импортировать bot.py. Запускаю: python bot.py")
        subprocess.check_call([sys.executable, "bot.py"])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parse","index","bot","up"])
    args = ap.parse_args()

    if args.cmd == "parse":
        cmd_parse()
    elif args.cmd == "index":
        cmd_index()
    elif args.cmd == "bot":
        cmd_bot()
    elif args.cmd == "up":
        cmd_parse()
        cmd_index()
        cmd_bot()
