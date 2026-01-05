# utils.py
from __future__ import annotations

import os
import re
import math
import json
from typing import List, Optional, Dict, Any

import pandas as pd
from fuzzywuzzy import fuzz
from openai import OpenAI

DATA_CSV_DEFAULT = "data/资料清单.csv"
KEYWORDS_TXT_DEFAULT = "data/keywords.txt"

COL_ID = "ID"
COL_PATH = "层级路径"
COL_TITLE = "关联文件名称"

OPENAI_KEY_ENV = "AutoCircuitChatbot"

# =========================
# 同义词扩展
# =========================
SYNONYMS = {
    "保险丝": ["熔断器", "fuse"],
    "接线盒": ["接线箱", "junction", "j/b", "jb"],
    "针脚": ["pin", "针脚定义", "端子", "插头", "接头"],
    "ecu": ["ECU", "电脑板", "控制器", "电控", "发动机电脑"],
    "仪表": ["组合仪表", "仪表盘", "meter"],
    "住友": ["sumitomo", "SUMITOMO", "住友挖机", "住友挖掘机"],
}


# =========================
# CSV / keywords 加载
# =========================
def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


def load_data(csv_path: str = DATA_CSV_DEFAULT) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到CSV文件：{csv_path}")

    df = _read_csv_robust(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    required = {COL_ID, COL_PATH, COL_TITLE}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CSV缺少必要列：{missing}")

    df[COL_PATH] = df[COL_PATH].astype(str).fillna("")
    df[COL_TITLE] = df[COL_TITLE].astype(str).fillna("")
    df["_search_blob"] = (df[COL_TITLE] + " " + df[COL_PATH]).astype(str)
    return df


def load_keywords(path: str = KEYWORDS_TXT_DEFAULT) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到keywords文件：{path}")

    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                lines = [line.strip() for line in f.readlines()]
            return [x for x in lines if x]
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


# =========================
# Query 规范化
# =========================
def _expand_synonyms(q: str) -> str:
    q_lower = q.lower()
    extra = []
    for k, vals in SYNONYMS.items():
        hit = (k.lower() in q_lower) or any(v.lower() in q_lower for v in vals)
        if hit:
            extra.append(k)
            extra.extend(vals)
    return (q + " " + " ".join(extra)).strip()


def normalize_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\becu\b", "ECU", q, flags=re.IGNORECASE)

    q = re.sub(r"([A-Za-z0-9]+)([\u4e00-\u9fff])", r"\1 \2", q)
    q = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9]+)", r"\1 \2", q)

    q = re.sub(r"[@#￥$%^&*()（）\[\]{}<>【】|\\/]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    q = _expand_synonyms(q)
    return q


# =========================
# LLM 意图解析（轻量、可回退）
# =========================
def parse_intent_llm(query: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv(OPENAI_KEY_ENV)
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""
你是一个【车辆电路图搜索意图解析器】。
只做信息抽取，不做判断。

只返回 JSON，不要多余文字。
字段不存在请返回空数组。

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
        print("✅ LLM called: gpt-4o-mini")

        text = resp.choices[0].message.content.strip()
        data = json.loads(text)

        if not isinstance(data, dict):
            return None

        return data

    except Exception:
        return None


# =========================
# 检索与排序
# =========================
def score_row(query: str, title: str, path: str) -> float:
    s_title = max(
        fuzz.token_set_ratio(query, title),
        fuzz.partial_ratio(query, title),
        fuzz.WRatio(query, title),
    )
    s_path = max(
        fuzz.token_set_ratio(query, path),
        fuzz.partial_ratio(query, path),
        fuzz.WRatio(query, path),
    )
    return 0.7 * s_title + 0.3 * s_path


def search_topk(
    df: pd.DataFrame,
    query: str,
    k: int = 5,
    candidate_pool: int = 300,
    min_score: float = 0.0,
    use_llm_intent: bool = False,
) -> pd.DataFrame:
    query_n = normalize_query(query)
    if not query_n:
        return pd.DataFrame(columns=[COL_ID, COL_TITLE, COL_PATH, "score"])

    tokens = [t for t in query_n.split() if len(t) > 1]
    tokens_lower = [t.lower() for t in tokens]

    blob_lower = df["_search_blob"].astype(str).str.lower()
    N = len(df)

    token_idf = {}
    for t in tokens_lower:
        df_count = int(blob_lower.str.contains(re.escape(t), na=False).sum())
        token_idf[t] = math.log((N + 1) / (df_count + 1)) + 1.0

    mask = None
    for t in token_idf.keys():
        m = blob_lower.str.contains(re.escape(t), na=False)
        mask = m if mask is None else (mask | m)

    candidates = df[mask].copy() if mask is not None and mask.any() else df.copy()

    def weighted_hit_score(s: str) -> float:
        s = str(s).lower()
        score = 0.0
        for t, w in token_idf.items():
            if t in s:
                score += w
        return score

    candidates["_whit"] = candidates["_search_blob"].apply(weighted_hit_score)
    candidates = candidates.sort_values("_whit", ascending=False).head(candidate_pool)

    scores = []
    for _, r in candidates.iterrows():
        base = score_row(query_n, r[COL_TITLE], r[COL_PATH])

        blob_l = str(r["_search_blob"]).lower()
        hit_cnt = sum(1 for t in token_idf if t in blob_l)

        bonus = r["_whit"] * 22.0
        bonus += hit_cnt * 12.0

        if ("住友" in blob_l or "sumitomo" in blob_l) and ("4hk1" in blob_l):
            bonus += 80.0

        scores.append(base + bonus)

    ranked = candidates.assign(score=scores).sort_values("score", ascending=False)

    # ===== LLM 意图辅助加权（可回退）=====
    if use_llm_intent:
        intent = parse_intent_llm(query)
        if intent:
            def intent_bonus(row) -> float:
                blob = f"{row[COL_TITLE]} {row[COL_PATH]}".lower()
                b = 0.0
                for brand in intent.get("brand", []):
                    if brand.lower() in blob:
                        b += 30.0
                for model in intent.get("model", []):
                    if model.lower() in blob:
                        b += 40.0
                for part in intent.get("part", []):
                    if part.lower() in blob:
                        b += 20.0
                return b

            ranked["score"] = ranked["score"] + ranked.apply(intent_bonus, axis=1)
            ranked = ranked.sort_values("score", ascending=False)

    if min_score > 0:
        filtered = ranked[ranked["score"] >= min_score]
        if not filtered.empty:
            ranked = filtered

    return ranked[[COL_ID, COL_TITLE, COL_PATH, "score"]].head(k).reset_index(drop=True)
