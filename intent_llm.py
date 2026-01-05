# intent_llm.py
from typing import Optional, Dict, Any
import json
import os

from openai import OpenAI

# 使用你已经规划好的环境变量名
OPENAI_KEY_ENV = "AutoCircuitChatbot"


def parse_intent_llm(query: str) -> Optional[Dict[str, Any]]:
    """
    使用 GPT-4o-mini 解析用户搜索意图
    只做信息抽取，不做判断
    失败时返回 None
    """
    api_key = os.getenv(OPENAI_KEY_ENV)
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
你是一个【车辆电路图搜索意图解析器】。
任务：从用户搜索中提取结构化信息，不要解释。

只返回 JSON，不要多余文字。
字段不存在就返回空数组。

JSON 格式：
{{
  "brand": [],
  "series": [],
  "model": [],
  "part": [],
  "confidence": 0.0
}}

用户输入：
{query}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = resp.choices[0].message.content.strip()
        data = json.loads(text)

        # 最低限度校验
        if not isinstance(data, dict):
            return None

        return data

    except Exception:
        # ⚠️ 一切异常都直接兜底
        return None
