"""CLI: mars run "вопрос" | mars rebuild <run_dir>."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import load_settings
from .costs import estimate
from .llm import LLMError
from .orchestrator import Orchestrator, rebuild_report


def _user_command(args, settings) -> int:
    import getpass

    from .web.store import Store

    store = Store(settings.database)
    if args.action == "list":
        print("\n".join(store.list_users()) or "Пользователей нет: сервер работает без входа.")
        return 0
    if not args.name:
        print("Укажите имя пользователя", file=sys.stderr)
        return 1
    if args.action == "remove":
        print("Удалён" if store.remove_user(args.name) else "Такого пользователя нет")
        return 0
    password = getpass.getpass("Пароль (не короче 8 символов): ")
    if password != getpass.getpass("Ещё раз: "):
        print("Пароли не совпадают", file=sys.stderr)
        return 1
    try:
        store.set_user(args.name, password)
    except ValueError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print(f"Пользователь {args.name} сохранён. Теперь веб-сервер требует вход.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mars", description="Multi-Agent Research System")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Провести исследование")
    r.add_argument("question")
    r.add_argument("--context", default="", help="Доп. контекст или путь к .txt/.md файлу")
    r.add_argument("--config", default=None)
    r.add_argument("--mock", action="store_true", help="Офлайн-режим без API (демо/тесты)")
    r.add_argument("--rounds", type=int, default=None, help="Макс. число раундов поиска")
    r.add_argument("--doc", action="append", default=[], metavar="PATH",
                   help="Свой файл-источник (PDF, TXT, MD, CSV, JSON); можно указать несколько раз")
    r.add_argument("--max-cost", type=float, default=None, metavar="USD",
                   help="Не запускать, если верхняя оценка стоимости выше этой суммы")
    e = sub.add_parser("estimate", help="Оценить стоимость запуска, не вызывая модель")
    e.add_argument("--config", default=None)
    e.add_argument("--sub-questions", type=int, default=5)
    e.add_argument("--docs", type=int, default=0)
    e.add_argument("--rounds", type=int, default=None)
    s = sub.add_parser("serve", help="Веб-интерфейс")
    s.add_argument("--config", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-auth", action="store_true",
                   help="Разрешить доступ из сети без входа (небезопасно: любой сможет тратить ваш API-ключ)")
    u = sub.add_parser("user", help="Пользователи веб-сервера")
    u.add_argument("action", choices=["add", "remove", "list"])
    u.add_argument("name", nargs="?")
    u.add_argument("--config", default=None)
    x = sub.add_parser("export", help="Экспорт отчёта в PDF и Word")
    x.add_argument("run_dir")
    x.add_argument("--format", choices=["pdf", "docx", "all"], default="all")
    x.add_argument("--font", default=None, help="Путь к .ttf с кириллицей для PDF")
    j = sub.add_parser("judge", help="Оценка отчёта моделью-судьёй")
    j.add_argument("run_dir")
    j.add_argument("--config", default=None)
    j.add_argument("--mock", action="store_true")
    j.add_argument("--verify-claims", type=int, default=0, metavar="N",
                   help="Сверить N утверждений с первоисточниками в вебе")
    v = sub.add_parser("eval", help="Прогнать набор эталонных вопросов")
    v.add_argument("questions", help="YAML-файл, см. examples/eval_questions.yaml")
    v.add_argument("--config", default=None)
    v.add_argument("--mock", action="store_true")
    v.add_argument("--out", default="evals")
    v.add_argument("--verify-claims", type=int, default=0, metavar="N")
    b = sub.add_parser("rebuild", help="Пересобрать отчёт из артефактов запуска")
    b.add_argument("run_dir")
    c = sub.add_parser("check", help="Проверить качество запуска без вызова модели")
    c.add_argument("run_dir")
    c.add_argument("--min-score", type=int, default=0, help="Код возврата 3, если оценка ниже")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    if args.cmd == "check":
        from .quality import evaluate_run

        q = evaluate_run(args.run_dir)
        print(q.table())
        return 3 if q.score < args.min_score else 0

    if args.cmd == "rebuild":
        print(rebuild_report(args.run_dir))
        return 0

    if args.cmd == "export":
        from .export import export

        try:
            for path in export(args.run_dir, args.format, args.font):
                print(path)
        except LLMError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
        return 0

    settings = load_settings(args.config)
    if args.cmd == "user":
        return _user_command(args, settings)
    if args.cmd in ("judge", "eval"):
        if args.mock:
            settings.backend = "mock"
        try:
            if args.cmd == "judge":
                from .evals import judge_run

                print(asyncio.run(judge_run(args.run_dir, settings, verify_claims=args.verify_claims)).table())
            else:
                from .evals import run_eval

                summary = asyncio.run(run_eval(args.questions, settings, args.out,
                                               verify_claims=args.verify_claims))
                print(f"\nСводка: {args.out}/summary.md · судья {summary['mean_judge']}/100 · "
                      f"структура {summary['mean_structure']}/100 · ошибок {summary['failed']}")
        except LLMError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
        return 0
    if args.cmd == "serve":
        try:
            import uvicorn

            from .web.app import create_app
        except ImportError:
            print('Для веб-интерфейса установите зависимости: pip install -e ".[web]"', file=sys.stderr)
            return 1
        app = create_app(settings)
        local = args.host in ("127.0.0.1", "localhost", "::1")
        if not local and not app.state.store.has_users() and not args.no_auth:
            print("Сервер открыт в сеть, но пользователей нет. Сначала создайте пользователя:\n"
                  "  mars user add ИМЯ\nили запустите с --no-auth, если понимаете риск.", file=sys.stderr)
            return 1
        mode = "вход по паролю" if app.state.store.has_users() else "без входа"
        print(f"MARS: http://{args.host}:{args.port} ({mode})")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.cmd == "estimate":
        if args.rounds:
            settings.max_rounds = args.rounds
        print(estimate(settings, args.sub_questions, args.docs).table())
        return 0
    if args.mock:
        settings.backend = "mock"
    if args.rounds:
        settings.max_rounds = args.rounds
    context = args.context
    if context.endswith((".txt", ".md")):
        from pathlib import Path
        context = Path(context).read_text(encoding="utf-8")

    if settings.backend == "ollama":
        print(f"Локальная модель {settings.ollama_model} через Ollama: бесплатно, но медленнее Claude.")
    elif settings.backend != "mock":
        est = estimate(settings, documents=len(args.doc))
        print(f"Оценка стоимости: ${est.usd_low:.2f} – ${est.usd_high:.2f} (~{est.calls} вызовов)")
        if args.max_cost is not None and est.usd_high > args.max_cost:
            print(f"Остановлено: верхняя оценка выше лимита ${args.max_cost:.2f}", file=sys.stderr)
            return 2

    try:
        result = asyncio.run(Orchestrator(settings).run(args.question, context, args.doc))
    except LLMError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(f"\nОтчёт: {result.report_path}\nHTML с графом: {result.run_dir / 'report.html'}"
          f"\nПапка запуска: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
