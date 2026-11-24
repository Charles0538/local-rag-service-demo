#!/bin/zsh
# 双击我：向你的知识库提问（本地跑，不联网、不上传任何东西）
cd "$(dirname "$0")"

echo "=========================================================="
echo "  家电售后知识库 · 本地问答 demo"
echo "  模型：qwen2.5:7b ＋ bge-m3 检索 ｜ 数据全部在你电脑上"
echo "=========================================================="
echo ""
echo "试试这几句："
echo "  1. 空调显示 E5 是什么故障，需要报修吗？"
echo "  2. 保修期内上门维修要收费吗？"
echo "  3. 师傅爽约了怎么投诉？"
echo "  4. 你们家冰箱保修十年是真的吗？   ← 故意问一个知识库里没有的"
echo ""
echo "输入 q 回车退出。"
echo ""

while true; do
  printf "问题： "
  read question
  if [ "$question" = "q" ] || [ "$question" = "Q" ]; then
    break
  fi
  echo ""
  if [ -z "$question" ]; then
    python3 04_rag.py
  else
    python3 04_rag.py "$question"
  fi
  echo ""
done

echo "=========================================================="
echo "  结束。按回车关闭窗口。"
echo "=========================================================="
read
