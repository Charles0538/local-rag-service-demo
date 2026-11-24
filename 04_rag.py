#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 · 本地 RAG 问答（零第三方依赖，只用标准库）

整条链路就四步，每一步都能单独讲：
    1) 切块   读 docs/*.md，按小节切开，每块带「文件名 + 小节名」
    2) 向量化 调 bge-m3 把每块变成一串数字（语义指纹）
    3) 检索   把你的问题也变成指纹，找最像的 K 块（余弦相似度）
    4) 生成   把 K 块塞进 prompt，让 qwen2.5:7b 只依据这些块回答

用法：
    python3 04_rag.py                         # 问默认问题
    python3 04_rag.py "空调显示E5怎么办"      # 问自己的问题
    python3 04_rag.py --rebuild "..."         # 改了文档后强制重建索引
    python3 04_rag.py --debug "..."           # 打印每个候选块的分数
"""

import glob
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

OLLAMA = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
INDEX_FILE = os.path.join(HERE, ".rag_index.json")

TOP_K = 5                # 取最像的 5 块
MAX_PER_FILE = 2         # 同一份文档最多占几块（防止一个文件把名额全占了）
MIN_SCORE = 0.45         # 最高分低于这个值 = 知识库里没有依据 → 直接拒答
                         # 校准依据（2026-11-22 实测）：相关问题分数 0.72-0.77，
                         # 无关问题 0.36-0.38（"写诗" 0.3x、"股市行情" 0.383），
                         # 门槛取两类分布之间：0.45。原值 0.35 是拍的，被"股市行情"问题抓出漏洞。
MAX_CHUNK_CHARS = 700    # 单块上限，超了就按段落再切


# ---------- 第 0 步：护栏（红线前置拦截） ----------
# 为什么要有这一层：实测发现 7B 模型对 prompt 里「不要做 X」这类否定规则执行率很低，
# 一旦资料里有相关内容它就会顺着答。所以红线判断不能交给模型，要用代码挡住。
GUARDS = [
    {
        "name": "隐私/越权",
        "keywords": [("订单", "别人"), ("订单", "他人"), ("订单", "老王"), ("单号", "别人"),
                     ("单号", "他人"), ("查", "他的"), ("查", "她的")],
        "reply": "涉及他人隐私，无法为您查询。我只能查询您本人的订单，请提供您自己的订单号或工单号。",
    },
    {
        "name": "竞品比价",
        "keywords": [("比", "贵"), ("比", "便宜"), ("对比", "价格"), ("别人家", "价格")],
        "reply": "我们不与其他品牌做价格比较。可以为您说明我方的服务标准与收费标准，需要哪一项？",
    },
]


def check_guard(question: str):
    """命中红线规则就返回固定话术，不走检索、不调模型。"""
    for g in GUARDS:
        for pair in g["keywords"]:
            if all(w in question for w in pair):
                return g["name"], g["reply"]
    return None


# ---------- 第 1 步：切块 ----------

def split_doc(path: str) -> list:
    """按二级标题（## ）切块。每块都记住自己从哪个文件、哪个小节来。"""
    fname = os.path.basename(path)
    text = open(path, encoding="utf-8").read()

    # 用 ## 标题把全文切成若干段，第一段是文件开头的说明
    parts = re.split(r"\n(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 小节名 = 第一行（去掉 ##）
        head = part.splitlines()[0].lstrip("# ").strip() or fname
        body = part
        # 太长就按空行切成更小的块，尽量不超过 MAX_CHUNK_CHARS
        if len(body) > MAX_CHUNK_CHARS:
            buf = ""
            for para in body.split("\n\n"):
                if len(buf) + len(para) > MAX_CHUNK_CHARS and buf:
                    chunks.append((head, buf.strip()))
                    buf = para
                else:
                    buf = buf + "\n\n" + para if buf else para
            if buf.strip():
                chunks.append((head, buf.strip()))
        else:
            chunks.append((head, body))
    return [(fname, head, body) for head, body in chunks]


def load_chunks() -> list:
    files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not files:
        sys.exit(f"[错误] {DOCS_DIR} 里没有 .md 文档，先放几份知识库文档进去。")
    all_chunks = []
    for f in files:
        all_chunks.extend(split_doc(f))
    return all_chunks


# ---------- 第 2 步：向量化 ----------

def embed(texts: list) -> list:
    """调 Ollama 的 /api/embed，一次把多段文字变成向量。"""
    payload = {"model": EMBED_MODEL, "input": texts}
    req = urllib.request.Request(
        f"{OLLAMA}/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["embeddings"]


def cosine(a: list, b: list) -> float:
    """余弦相似度：两个向量夹角的余弦，越接近 1 越像。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------- 索引：算一次，存下来，下次秒开 ----------

def build_index(chunks: list) -> list:
    texts = [f"{head}\n{body}" for _, head, body in chunks]
    vectors = embed(texts)
    index = []
    for (fname, head, body), vec in zip(chunks, vectors):
        index.append({
            "source": fname,
            "section": head,
            "text": body,
            "vec": [round(v, 6) for v in vec],
        })
    json.dump(index, open(INDEX_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    return index


def index_is_stale() -> bool:
    """文档比索引新，就说明该重建了。"""
    if not os.path.exists(INDEX_FILE):
        return True
    index_mtime = os.path.getmtime(INDEX_FILE)
    for f in glob.glob(os.path.join(DOCS_DIR, "*.md")):
        if os.path.getmtime(f) > index_mtime:
            return True
    return False


def get_index(rebuild: bool = False) -> list:
    if rebuild or index_is_stale():
        chunks = load_chunks()
        print(f"[索引] 首次/文档有更新，正在向量化 {len(chunks)} 个知识块……")
        t0 = time.time()
        index = build_index(chunks)
        print(f"[索引] 完成，耗时 {time.time() - t0:.2f} s，已缓存到 .rag_index.json")
        return index
    return json.load(open(INDEX_FILE, encoding="utf-8"))


# ---------- 第 3 步：检索 ----------

CATEGORY_WORDS = ["空调", "冰箱", "洗衣机", "热水器", "厨电", "油烟机"]
TOPIC_WORDS = ["保修", "维修", "安装", "上门", "投诉", "收费", "退换", "发票",
               "配件", "赔付", "时效", "故障"]


def filter_by_entity(question: str, hits: list) -> list:
    """混合检索的过滤半边：向量管"像不像"，关键词管"是不是"。

    两层教训（都是实测踩出来的）：
    ① 只按品类词过滤不够——问「冰箱保修多久」，含「冰箱」的故障码块会被留下，
       但它没有保修年限，模型照样拿常识编一个。
    ② 所以品类词和主题词都要命中才保留：问冰箱+保修，块里必须真的同时出现
       「冰箱」和「保修」。全被滤光 = 知识库里真没有 → 自然触发拒答。
    """
    cats = [w for w in CATEGORY_WORDS if w in question]
    topics = [w for w in TOPIC_WORDS if w in question]
    # 只有问题同时含品类词时才做强过滤（品类 AND 主题）：
    # 「冰箱保修多久」必须滤掉邻位块；而「爽约怎么投诉」没有品类词，
    # 此时关键词过滤反而会误杀「师傅爽约处理」这种标题不含"投诉"的赔付块。
    if not cats:
        return hits

    kept = []
    for score, item in hits:
        text = item["text"] + item["section"]
        if not any(w in text for w in cats):
            continue
        if topics and not any(w in text for w in topics):
            continue
        kept.append((score, item))

    dropped = len(hits) - len(kept)
    if dropped and kept:
        print(f"[过滤] 命中品类词 {'、'.join(cats)} / 主题词 {'、'.join(topics) or '-'}，"
              f"剔除 {dropped} 块（防邻位污染与常识填空）")
    return kept


def retrieve(question: str, index: list, k: int = TOP_K, debug: bool = False) -> list:
    """按相似度排序取前 k 块；同一份文档最多取 MAX_PER_FILE 块，避免答案来源单一。"""
    qvec = embed([question])[0]
    scored = [(cosine(qvec, item["vec"]), item) for item in index]
    scored.sort(key=lambda x: x[0], reverse=True)

    if debug:
        print("[debug] 全部候选前 10（含被同源限制挤掉的）：")
        for s, it in scored[:10]:
            print(f"        {s:.3f}  {it['source']} · {it['section']}")

    picked = []
    per_file = {}
    for score, item in scored:
        src = item["source"]
        if per_file.get(src, 0) >= MAX_PER_FILE:
            continue
        picked.append((score, item))
        per_file[src] = per_file.get(src, 0) + 1
        if len(picked) == k:
            break
    return picked


# ---------- 第 4 步：生成 ----------

SYSTEM = """你是家电售后服务助手。只能依据【资料】回答，严禁编造。

规则：
1. 先一句话给结论，再分点说明依据。
2. 资料里有明确依据的，在相应句子末尾标注来源编号，如 [1]。
3. 资料里没有依据的，明确说「知识库中未收录该信息」，并建议转人工客服，不要推测。
4. 涉及查询他人订单、他人隐私的，一律拒绝。
5. 不做竞品比价。
6. 资料里出现的金额、时限、次数、保修年限等具体数值，必须如实写出，不得省略。
7. 如果【资料】里出现了赔偿、补偿、券、退款金额等条款，答案必须把这些条款完整写出，无论用户有没有直接问到。
8. 全部回答不超过 200 字。"""


def ask_with_context(question: str, hits: list) -> dict:
    ctx = []
    for i, (score, item) in enumerate(hits, 1):
        ctx.append(f"[{i}] （来源：{item['source']} · {item['section']}）\n{item['text']}")
    user = "【资料】\n" + "\n\n".join(ctx) + f"\n\n【用户问题】\n{question}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.2},   # 问答要稳，不要发散
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


# ---------- 主流程 ----------

def main() -> None:
    args = [a for a in sys.argv[1:]]
    rebuild = "--rebuild" in args
    debug = "--debug" in args
    args = [a for a in args if not a.startswith("--")]
    question = args[0] if args else "空调显示 E5 是什么故障，需要报修吗？"

    print("=" * 64)
    print(f"[问题] {question}")
    print("=" * 64)

    guard = check_guard(question)
    if guard:
        name, reply = guard
        print(f"[护栏] 命中「{name}」红线规则，前置拦截（不检索、不调模型）")
        print("-" * 64)
        print(reply)
        return

    index = get_index(rebuild)
    t0 = time.time()
    hits = retrieve(question, index, debug=debug)
    hits = filter_by_entity(question, hits)
    t_ret = time.time() - t0
    print(f"[检索] {len(index)} 个知识块中取 top{TOP_K}，耗时 {t_ret * 1000:.0f} ms")
    for i, (score, item) in enumerate(hits, 1):
        flag = "✓" if score >= MIN_SCORE else "✗"
        print(f"   {flag} [{i}] {score:.3f}  {item['source']} · {item['section']}")
    print("-" * 64)

    # 一块都没活下来，或最高分低于门槛 = 知识库里没依据，不硬答（这是幻觉抑制的关键）
    if not hits or hits[0][0] < MIN_SCORE:
        print("[结果] 最高相关度低于门槛，知识库中无依据，直接拒答（不调用模型）")
        print("       建议：转人工客服，或补充对应知识库文档后重试。")
        return

    r = ask_with_context(question, hits)
    print(r["message"]["content"].strip())
    print("-" * 64)
    ec = r.get("eval_count", 0)
    ed = r.get("eval_duration", 1) or 1
    print(f"[生成] 墙钟 {r['_wall']:.2f} s ｜ {ec} token ｜ {ec / (ed / 1e9):.1f} token/s")
    print(f"[溯源] 答案依据：{'、'.join(sorted({h[1]['source'] for h in hits if h[0] >= MIN_SCORE}))}")


if __name__ == "__main__":
    main()
