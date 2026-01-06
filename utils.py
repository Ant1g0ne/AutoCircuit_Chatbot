# utils.py
from __future__ import annotations
import os
import json
import re
import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from openai import OpenAI

# =========================
# 1. 知识库配置
# =========================
DATA_CSV_DEFAULT = "data/资料清单.csv"

# 同义词库 (核心：让搜索懂行话)
SYNONYMS_MAP = {
    # 车型/通用
    "2OOO": ["2000"],
    "2000": ["2OOO"],
    "小忪": ["小松"],
    "杰师": ["杰狮"],
    # 电器类
    "保险丝": ["熔断器", "fuse", "保险", "配电盒", "接线盒", "电器盒"],
    "保险盒": ["保险丝", "熔断器", "配电盒"],
    "ECU": ["电脑板", "控制器", "电控", "ECM", "VECU", "CBCU"],
    "电脑板": ["ECU", "控制器", "模块"],
    "仪表": ["组合仪表", "显示屏", "盘"],
    "针脚": ["管脚", "端子", "定义", "接头", "插头"],
    "线路图": ["电路图", "原理图", "接线图", "示意图"],
    "整车": ["全车", "系统图"],
    "供电": ["电源", "充电", "起动"],
    "玻璃": ["门窗", "升降器"],
    "差速器": ["差速锁", "桥"]
}

# 品牌库
KNOWN_BRANDS = [
    "三一", "徐工", "东风", "解放", "住友", "小松", "日立", "雷沃", 
    "卡特", "五十铃", "豪沃", "陕汽", "福田", "江淮", "红岩", 
    "大通", "宇通", "金龙", "比亚迪", "吉利", "长城", "柳汽", "乘龙", "欧曼"
]

# 系列关键词
ALL_SERIES_KEYWORDS = [
    "天龙", "KL", "KC", "VL", "旗舰", "大力神", "津威", 
    "J6", "J6P", "J6L", "J7", "JH6", "虎V", "龙V",                   
    "豪沃", "T7", "TX", "汕德卡", "斯太尔", "豪瀚",                  
    "X3000", "X5000", "X6000", "M3000", "F3000", "德龙",                      
    "乘龙", "H7", "H5", "M3", "霸龙",                                       
    "SY75", "SY135", "SY215", "SY245", "SY365",              
    "ZX200", "ZX240", "4HK1", "6HK1", "2000", "3000",
    "杰狮", "杰卡", "金刚", "欧曼", "GTL", "EST"
]

# 类型关键词
ALL_TYPE_KEYWORDS = [
    "整车", "ECU", "仪表", "底盘", "发动机", "ABS", "车身", 
    "门窗", "灯光", "空调", "后处理", "供电", "起动", "充电",
    "针脚", "线路图", "原理图", "接线图", "保险丝", "电脑板", 
    "接线盒", "继电器", "差速器", "玻璃升降"
]

# =========================
# 2. LLM 模块
# =========================
def get_api_key():
    try: return st.secrets["OPENAI_API_KEY"]
    except: return os.getenv("OPENAI_API_KEY")

def llm_parse_query(query: str) -> dict:
    """GPT-4o-mini 解析"""
    api_key = get_api_key()
    if not api_key: return {}

    client = OpenAI(api_key=api_key)
    try:
        prompt = f"""
        Extract vehicle info.
        Role: Auto Expert.
        Rules:
        1. Correct typos ("小忪"->"小松", "2ooo"->"2000").
        2. "红岩杰狮" -> Brand:红岩, Series:杰狮.
        Return JSON: {{"brand": "...", "series": "...", "part": "..."}}
        Query: {query}
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        txt = resp.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', txt, re.DOTALL)
        return json.loads(match.group(0)) if match else {}
    except:
        return {}

# =========================
# 3. 数据与搜索 (增强版)
# =========================
def _read_csv_robust(path):
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    raise ValueError("无法读取CSV")

def load_data(path=DATA_CSV_DEFAULT):
    if not os.path.exists(path): return pd.DataFrame()
    df = _read_csv_robust(path)
    df.columns = df.columns.astype(str).str.strip()
    df["层级路径"] = df["层级路径"].astype(str).fillna("")
    df["关联文件名称"] = df["关联文件名称"].astype(str).fillna("")
    df["_search_blob"] = (df["关联文件名称"] + " " + df["层级路径"]).astype(str)
    return df

def get_expanded_keywords(keyword: str) -> list[str]:
    """同义词扩展"""
    keywords = [keyword]
    for k, v in SYNONYMS_MAP.items():
        if keyword.lower() == k.lower():
            keywords.extend(v)
        elif keyword.lower() in [x.lower() for x in v]:
            keywords.append(k)
            keywords.extend([x for x in v if x.lower() != keyword.lower()])
    return list(set(keywords))

def check_text_matches_any(text: str, keywords: list[str]) -> bool:
    text_u = text.upper()
    for k in keywords:
        if k.upper() in text_u:
            return True
    return False

def search_topk(df, query, k=200):
    if not query: return pd.DataFrame()
    df_copy = df.copy()
    
    def calculate_score(text):
        return fuzz.partial_token_set_ratio(query, str(text))

    df_copy["_score"] = df_copy["_search_blob"].apply(calculate_score)
    # 阈值40，保证召回
    return df_copy[df_copy["_score"] >= 40].sort_values("_score", ascending=False).head(k)

# =========================
# 4. 选项检测 (修正函数名)
# =========================
def detect_options(current_df: pd.DataFrame, keyword_list: list) -> list[str]:
    """🔥 修复：函数名统一为 detect_options"""
    if current_df.empty: return []
    text_blob = " ".join(current_df.head(100)["_search_blob"].astype(str).tolist()).upper()
    found = []
    for k in keyword_list:
        if k.upper() in text_blob:
            found.append(k)
    return found