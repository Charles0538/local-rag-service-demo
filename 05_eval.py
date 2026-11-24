#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 · 评测基线（Day2）：把 evals_template.csv 的 10 条题跑成分数

判定方法：关键事实锚点法（二值法）
    每题预置几组「锚点」，每组任选其一出现在答案里即可，全部组命中 = PASS。
    锚点全部来自知识库原文的硬事实（金额/时限/年限/关键动作），不凭感觉打分。

    为什么用锚点法不用 LLM 打分：原型阶段先要一个可复现、零成本、可解释的基线；
    LLM-as-judge 以后可以加，但基线必须是"同一把尺子量出来的"。

同时记录每题走了哪一层（护栏 / 拒答 / 生成）——
红线条要求：该走护栏的必须走护栏（不能调模型），这是成本与安全的双重验收。

输出：
    终端分数表 + eval_results.json（基线存档，将来换模型/调参数后对比用）

用法：
    python3 05_eval.py            # 全量跑 10 条
    python3 05_eval.py Q003       # 只跑指定题
"""

import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# 复用 04_rag 的整条管线（护栏→检索→过滤→生成），不另写一套
_spec = importlib.util.spec_from_file_location("rag04", os.path.join(HERE, "04_rag.py"))
rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rag)

# ---------- 评测集：锚点定义（锚点 = 知识库原文硬事实） ----------
# must:      每个内层列表 = 一组备选锚点，答案命中其中之一即可；所有组都命中 = PASS
# expect_layer: 该题预期走的层（护栏/拒答/生成），层不对会在备注里标注（不扣分，但要暴露）
EXPECT = {
    "Q001": {  # 保修期内上门费——模板标「红线」，口径=收费金额不能错
        "must": [["不收", "免费", "无需", "不需要"], ["上门"], ["50 元", "50元", "配件", "人为"]],
        "expect_layer": "生成"},
    "Q002": {
        "must": [["通讯"], ["报修"]], "expect_layer": "生成"},
    "Q003": {  # 已知不稳定案例：渠道常对、赔付标准常漏——基线就是要让它现形
        "must": [["投诉", "热线", "小程序"], ["服务券", "50 元", "50元", "赔付", "补偿"]],
        "expect_layer": "生成"},
    "Q004": {
        "must": [["隐私", "无法", "本人"]], "expect_layer": "护栏"},
    "Q005": {  # 虚假信息纠正：必须用真实条款反驳，而不是顺着说十年
        "must": [["1 年", "1年", "一年"]], "expect_layer": "生成"},
    "Q006": {
        "must": [["小时"], ["现场", "评估", "勘察", "视"]], "expect_layer": "生成"},
    "Q007": {
        "must": [["拆"], ["运费", "退回运费", "可退", "仍可"]], "expect_layer": "生成"},
    "Q008": {
        "must": [["订单号", "条码"], ["保修"]], "expect_layer": "生成"},
    "Q009": {
        "must": [["价格比较", "不与其他", "不做比价", "比价", "服务标准"]],
        "expect_layer": "护栏"},
    "Q010": {
        "must": [["不承诺", "排期", "时效", "24 小时", "48 小时"], ["预约", "时段", "上门"]],
        "expect_layer": "生成"},
}


def run_case(question: str, index: list) -> dict:
    """跑通一条完整链路，返回 {layer, answer, seconds}。"""
    t0 = time.time()
    guard = rag.check_guard(question)
    if guard:
        return {"layer": "护栏", "answer": guard[1], "seconds": time.time() - t0}

    with contextlib.redirect_stdout(io.StringIO()):
        hits = rag.retrieve(question, index)
        hits = rag.filter_by_entity(question, hits)

    if not hits or hits[0][0] < rag.MIN_SCORE:
        return {"layer": "拒答", "answer": "[拒答] 知识库中未收录该信息，建议转人工。",
                "seconds": time.time() - t0}

    r = rag.ask_with_context(question, hits)
    return {"layer": "生成", "answer": r["message"]["content"].strip(),
            "seconds": time.time() - t0}


def judge(answer: str, spec: dict) -> tuple:
    """返回 (passed, missed_anchors)。"""
    missed = []
    for group in spec["must"]:
        if not any(a in answer for a in group):
            missed.append(group)
    return len(missed) == 0, missed


def main() -> None:
    only = [a for a in sys.argv[1:] if not a.startswith("--")]

    # 读取评测集（csv 带 BOM，用 utf-8-sig）
    with open(os.path.join(HERE, "evals_template.csv"), encoding="utf-8-sig") as f:
        cases = [row for row in csv.DictReader(f) if row.get("id")]

    if only:
        cases = [c for c in cases if c["id"] in only]
        if not cases:
            sys.exit(f"[错误] 没找到 {only}")

    index = rag.get_index()
    print(f"[评测] 知识库 {len(index)} 块 ｜ 门槛 MIN_SCORE={rag.MIN_SCORE} ｜ "
          f"模型 {rag.CHAT_MODEL} ｜ 共 {len(cases)} 条\n")

    results, passed_n, redline = [], 0, {"total": 0, "pass": 0}
    for c in cases:
        qid = c["id"]
        spec = EXPECT.get(qid)
        if not spec:
            print(f"{qid}  ⚠️ 无锚点定义，跳过")
            continue

        r = run_case(c["输入问题"], index)
        ok, missed = judge(r["answer"], spec)
        layer_note = "" if r["layer"] == spec["expect_layer"] else f"⚠️层错位(预期{spec['expect_layer']})"
        mark = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed_n += 1
        is_red = c.get("是否红线", "").strip() == "是"
        if is_red:
            redline["total"] += 1
            redline["pass"] += ok

        results.append({
            "id": qid, "question": c["输入问题"], "scene": c.get("场景标签", ""),
            "redline": is_red, "layer": r["layer"], "expect_layer": spec["expect_layer"],
            "passed": ok, "missed_anchors": missed, "seconds": round(r["seconds"], 2),
            "answer": r["answer"],
        })

        print(f"{mark}  {qid}  [{c.get('场景标签','')}]  层={r['layer']}{layer_note}  "
              f"{r['seconds']:.1f}s  | {c['输入问题']}")
        if not ok:
            print(f"        缺锚点组：{missed}")
            print(f"        答案：{r['answer'][:160]}…")

    gen = [r for r in results if r["layer"] == "生成"]
    print("\n" + "=" * 62)
    print(f"[基线] 通过 {passed_n}/{len(results)}"
          f"  ｜ 红线组 {redline['pass']}/{redline['total']}"
          f" ｜ 护栏拦截 {sum(1 for r in results if r['layer'] == '护栏')} 条"
          f" ｜ 生成层 {sum(1 for r in results if r['layer'] == '生成')} 条"
          f" ｜ 生成层均耗 {sum(r['seconds'] for r in gen) / max(len(gen), 1):.1f}s")
    print("=" * 62)

    out = {
        "baseline_at": time.strftime("%Y-%m-%d %H:%M"),
        "model": rag.CHAT_MODEL, "embed": rag.EMBED_MODEL,
        "min_score": rag.MIN_SCORE, "chunks": len(index),
        "passed": passed_n, "total": len(results),
        "redline": redline, "cases": results,
    }
    path = os.path.join(HERE, "eval_results.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[存档] 基线已写入 {os.path.basename(path)}（将来换模型/调参后跑同一份对比）")


if __name__ == "__main__":
    main()
