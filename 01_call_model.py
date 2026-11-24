#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 · 调用本地 Ollama 模型（零第三方依赖，只用标准库）

用法：
    python3 01_call_model.py                      # 用默认问题
    python3 01_call_model.py "解释一下什么是 RAG"   # 问自己的问题

它会打印：模型回答 + 本次调用的真实耗时与出字速度（token/s）
这两个数字都是你自己机器测出来的，面试可以直接报。
"""

import json
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"


def ask(prompt: str, model: str = MODEL) -> dict:
    """调用 /api/chat，非流式，返回完整响应体。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # 可选参数（Day4 实验用）：
        # "options": {"temperature": 0.7, "num_predict": 256}
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_wall_seconds"] = time.time() - t0
    return body


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "用三句话解释什么是 RAG，说清楚它解决什么问题。"
    print(f"[模型] {MODEL}\n[问题] {prompt}\n{'-' * 60}")

    r = ask(prompt)
    print(r["message"]["content"][:1500])
    print("-" * 60)

    # Ollama 返回的计时字段单位是纳秒
    eval_count = r.get("eval_count", 0)
    eval_ns = r.get("eval_duration", 1) or 1
    load_ns = r.get("load_duration", 0)
    tps = eval_count / (eval_ns / 1e9) if eval_count else 0

    print(f"整轮墙钟耗时   : {r['_wall_seconds']:.2f} s")
    print(f"模型加载耗时   : {load_ns / 1e9:.3f} s（常驻后接近 0）")
    print(f"生成 token 数  : {eval_count}")
    print(f"出字速度       : {tps:.1f} token/s")
    print(f"prompt 长度    : {r.get('prompt_eval_count', 0)} token")


if __name__ == "__main__":
    main()
