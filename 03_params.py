#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 · 参数实验：不用改代码，直接在命令后面带数字

用法（数字就是"发散度"，0 最稳、1.5 最飘）：
    python3 03_params.py 0.1
    python3 03_params.py 1.2

第二个数字可选，用来限制输出长度（控制成本）：
    python3 03_params.py 0.7 80
"""

import json
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b"

# 三次问同一个问题，方便你对比 temperature 的影响
PROMPT = "给我们公司起三个做宠物用品的品牌名，每个名字后面用一句话说明理由。"


def ask(prompt, temperature, num_predict):
    opts = {"temperature": temperature}
    if num_predict:
        opts["num_predict"] = num_predict
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": opts,
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
    body["_wall"] = time.time() - t0
    return body


def main():
    temp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    num_predict = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"[参数] temperature={temp}", end="")
    if num_predict:
        print(f"  num_predict={num_predict}")
    else:
        print()
    print(f"[问题] {PROMPT}\n{'-' * 60}")

    r = ask(PROMPT, temp, num_predict)
    print(r["message"]["content"][:1200])
    print("-" * 60)
    ec = r.get("eval_count", 0)
    ed = (r.get("eval_duration") or 1) / 1e9
    print(f"耗时 {r['_wall']:.2f}s ｜ 生成 {ec} token ｜ {ec / ed:.1f} token/s")

    # 问你一件事：下次输入时试着加 num_predict 限制长度，看耗时掉多少
    print("\n提示：把 num_predict 调小（比如 60），输出会变短、耗时也会明显下降。")


if __name__ == "__main__":
    main()
