#!/bin/zsh
# 双击我就能跑：不用打开终端、不用敲命令
cd "$(dirname "$0")"

echo "================================================"
echo "  本地 AI 提问台（模型：qwen2.5:7b）"
echo "================================================"
echo ""
echo "请输入你的问题，然后按回车。"
echo "（什么都不输入直接回车 = 用默认问题）"
echo ""
printf "问题： "
read question

echo ""
echo "正在思考……（第一次可能要等几秒加载模型）"
echo ""

if [ -z "$question" ]; then
  python3 01_call_model.py
else
  python3 01_call_model.py "$question"
fi

echo ""
echo "================================================"
echo "  跑完了。按回车关闭这个窗口。"
echo "================================================"
read
