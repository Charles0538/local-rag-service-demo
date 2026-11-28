#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 · 稳定性实验：同一个问题连跑 N 次，把「概率性失败」量出来

背景：Q003「师傅爽约了怎么投诉」在基线温度 0.2 下 11 跑 8 过（3 次漏「50 元」）。
本脚本回答三个问题：
    实验 B①  temperature 0.2 连跑 N 次 → 复测概率性失败
    实验 B②  temperature 0   连跑 N 次 → 采样随机性是否被压掉
    实验 C   --strict（结构化抽取+生成后校验）连跑 N 次 → 能否稳定到 N/N

判定：复用 05_eval 的锚点定义（judge），与基线同一把尺子。

用法：
    python3 06_experiment.py Q003                # temp 0.2，跑 11 次
    python3 06_experiment.py Q003 --temp 0       # temp 0，跑 11 次
    python3 06_experiment.py Q003 --strict       # 结构化抽取模式，跑 11 次
    python3 06_experiment.py Q003 --n 5          # 自定义次数
"""

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

_spec = importlib.util.spec_from_file_location("rag04", os.path.join(HERE, "04_rag.py"))
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)

_spec2 = importlib.util.spec_from_file_location("eval05", os.path.join(HERE, "05_eval.py"))
ev = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(ev)


def load_question(qid: str) -> str:
    with open(os.path.join(HERE, "evals_template.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["id"] == qid:
                return row["输入问题"]
    sys.exit(f"[错误] 评测集里没有 {qid}")


def run_once(question: str, index: list, strict: bool, temperature: float) -> dict:
    """跑一条完整链路，返回 {layer, answer, seconds, retry}。"""
    t0 = time.time()
    guard = rag.check_guard(question)
    if guard:
        return {"layer": "护栏", "answer": guard[1], "seconds": time.time() - t0, "retry": 0}

    with contextlib.redirect_stdout(io.StringIO()):
        hits = rag.retrieve(question, index)
        hits = rag.filter_by_entity(question, hits)

    if not hits or hits[0][0] < rag.MIN_SCORE:
        return {"layer": "拒答", "answer": "[拒答]", "seconds": time.time() - t0, "retry": 0}

    facts = rag.extract_facts(hits[:1]) if strict else None
    r = rag.ask_with_context(question, hits, facts=facts, temperature=temperature)
    answer = r["message"]["content"].strip()
    retried = 0

    if strict:
        # 三级解法第 3 级：生成后校验，缺了重试一次
        missing = rag.verify_facts(answer, facts)
        if missing:
            retried = 1
            r2 = rag.ask_with_context(
                question + f"\n\n（你上一稿遗漏了以下数值，必须原样写出：{'、'.join(missing)}）",
                hits, facts=facts, temperature=temperature)
            a2 = r2["message"]["content"].strip()
            if len(rag.verify_facts(a2, facts)) < len(missing):
                answer = a2

    return {"layer": "生成", "answer": answer,
            "seconds": time.time() - t0, "retry": retried}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("qid")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=11)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    spec = ev.EXPECT.get(args.qid)
    if not spec:
        sys.exit(f"[错误] 05_eval 里没有 {args.qid} 的锚点定义")
    question = load_question(args.qid)
    index = rag.get_index()

    mode = "结构化抽取+校验" if args.strict else f"默认 prompt"
    print(f"[实验] {args.qid}「{question}」｜ {mode} ｜ temperature={args.temp} ｜ 连跑 {args.n} 次")
    print("=" * 64)

    passed, secs, retries = 0, [], 0
    for i in range(1, args.n + 1):
        r = run_once(question, index, args.strict, args.temp)
        ok, missed = ev.judge(r["answer"], spec)
        passed += ok
        secs.append(r["seconds"])
        retries += r["retry"]
        mark = "✅" if ok else "❌"
        extra = f"（重试 {r['retry']} 次）" if r["retry"] else ""
        print(f"{mark} 第{i:>2}轮  {r['seconds']:5.1f}s{extra}  "
              f"{'缺锚点 ' + str(missed) if not ok else '通过'}")
        if not ok:
            print(f"        答案：{r['answer'][:120]}…")

    print("=" * 64)
    print(f"[结果] 通过 {passed}/{args.n}"
          f"  ｜ 均耗 {sum(secs)/len(secs):.1f}s  ｜ 触发重试 {retries} 次")
    if args.strict:
        print("       对照组：默认 prompt 曾 11 跑 3 漏（2026-11-24 实测）")
    print(f"[存档] 实验时间 {time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
