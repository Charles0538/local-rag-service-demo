#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 · 流式调用 + 首字延迟（TTFT）测量

面试价值：TTFT（Time To First Token，首字延迟）是用户体感的核心指标，
比"总耗时"更能说明产品体验——这也是你做外呼/客服类产品时最该关注的数。

用法：
    python3 02_stream.py "写一段100字的产品介绍"
"""

import json
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"


def stream_ask(prompt: str, model: str = MODEL) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    ttft = None
    chunks = 0
    text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            piece = (obj.get("message") or {}).get("content", "")
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                chunks += 1
                text.append(piece)
                # 实时打印，体会"一个字一个字蹦出来"的流式效果
                sys.stdout.write(piece)
                sys.stdout.flush()
            if obj.get("done"):
                total = time.time() - t0
                eval_count = obj.get("eval_count", 0)
                tps = eval_count / ((obj.get("eval_duration") or 1) / 1e9)
                print("\n" + "-" * 60)
                print(f"首字延迟 TTFT : {ttft:.2f} s" if ttft else "首字延迟：—")
                print(f"总耗时        : {total:.2f} s")
                print(f"chunk 数      : {chunks}")
                print(f"出字速度      : {tps:.1f} token/s")


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "用100字介绍什么是RAG，面向不懂技术的业务方。"
    print(f"[问题] {q}\n{'-' * 60}")
    stream_ask(q)
