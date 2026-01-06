# app.py
import streamlit as st
import pandas as pd
from utils import (
    load_data, search_topk, llm_parse_query, detect_options, 
    get_expanded_keywords, check_text_matches_any, get_api_key,
    DATA_CSV_DEFAULT, ALL_SERIES_KEYWORDS, ALL_TYPE_KEYWORDS
)

# --- 1. 配置 ---
st.set_page_config(page_title="AutoCircuit Pro", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .status-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
    .green { background-color: #28a745; }
    .red { background-color: #dc3545; }
    
    /* 按钮样式优化：支持长文本自动换行 */
    .stButton button {
        border-radius: 8px; text-align: left; padding: 12px 15px;
        border: 1px solid #ddd; background: #fff; color: #333; 
        width: 100%; height: auto; white-space: normal; line-height: 1.5;
        transition: 0.2s;
    }
    .stButton button:hover { border-color: #ff4b4b; color: #ff4b4b; background: #fff5f5; }
    
    .result-card {
        padding: 12px; margin-top: 8px; border-radius: 8px;
        background: #fff; border: 1px solid #eee; border-left: 4px solid #ff4b4b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .hero-box { text-align: center; padding: 40px; background: #f8f9fa; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. 状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "state" not in st.session_state:
    st.session_state.state = {
        "active_query": "",
        "filters": [],           
        "results": None,         
        "step": "INIT",  
        "options": [],
        "debug": {}
    }

# --- 3. 数据 ---
@st.cache_data
def get_data():
    return load_data(DATA_CSV_DEFAULT)

try:
    df = get_data()
    db_ready = True
    db_count = len(df)
except:
    df = pd.DataFrame()
    db_ready = False
    db_count = 0

# --- 4. 侧边栏 ---
with st.sidebar:
    st.title("🎛️ AutoCircuit Pro")
    st.markdown(f"📚 资料库: <span class='status-dot {'green' if db_ready else 'red'}'></span>{db_count}", unsafe_allow_html=True)
    
    use_llm = st.toggle("启用 AI 意图辅助", value=True)
    if use_llm: st.caption("🚀 模型: **GPT-4o-mini**")
    
    with st.expander("🛠️ 调试 (Debug)"):
        st.json(st.session_state.state.get("debug", {}))

    if st.button("🗑️ 清空历史"):
        st.session_state.messages = []
        st.session_state.state = {"active_query": "", "filters": [], "results": None, "step": "INIT", "options": [], "debug": {}}
        st.rerun()

# --- 5. 核心逻辑 ---

def apply_filters_smart(base_df, filters):
    res = base_df.copy()
    for f in filters:
        synonyms = get_expanded_keywords(f)
        res = res[res["_search_blob"].apply(lambda x: check_text_matches_any(str(x), synonyms))]
    return res

def append_msg(role, content, msg_type="text", data=None):
    st.session_state.messages.append({
        "role": role, "content": content, "type": msg_type, "data": data
    })

def start_search(query):
    raw_res = search_topk(df, query, k=150)
    intent = {}
    filters = []
    
    if use_llm:
        intent = llm_parse_query(query)
        for k in ["brand", "series", "part"]:
            val = intent.get(k)
            if val and val.lower() != "null":
                filters.append(val)

    # 贪婪匹配策略
    strategies = [
        (filters, "精准匹配"),
        (filters[:-1], "部分匹配") if len(filters) > 1 else ([], ""),
        ([filters[0]], "品牌匹配") if len(filters) > 0 else ([], "") 
    ]
    
    final_df = pd.DataFrame()
    used_filters = []
    
    for flt, mtype in strategies:
        if not flt: continue
        temp_df = apply_filters_smart(raw_res, flt)
        if not temp_df.empty:
            final_df = temp_df
            used_filters = flt
            break
            
    if final_df.empty:
        final_df = raw_res
        used_filters = []

    st.session_state.state.update({
        "active_query": query,
        "filters": used_filters,
        "results": final_df,
        "debug": {"intent": intent, "applied": used_filters}
    })
    
    check_next_step()

    count = len(final_df)
    if count == 0:
        return f"😔 抱歉，没有找到“{query}”相关的电路图。"
    
    base_msg = f"我找到了 {count} 份资料。"
    if used_filters:
        base_msg = f"已识别 **{' '.join(used_filters)}**。{base_msg}"
            
    return base_msg

def check_next_step():
    current_df = st.session_state.state["results"]
    count = len(current_df)
    filters = st.session_state.state["filters"]
    
    has_series = any(check_text_matches_any(f, ALL_SERIES_KEYWORDS) for f in filters)
    has_type = any(check_text_matches_any(f, ALL_TYPE_KEYWORDS) for f in filters)
    
    if count <= 5 or (has_series and has_type):
        finalize_results()
        return

    curr_filters_str = "".join(filters).upper()
    
    # Check Series
    series_opts = detect_options(current_df, ALL_SERIES_KEYWORDS)
    valid_series = [o for o in series_opts if o.upper() not in curr_filters_str]
    if len(valid_series) > 1:
        st.session_state.state["step"] = "SERIES_SELECT"
        st.session_state.state["options"] = valid_series
        return

    # Check Type
    type_opts = detect_options(current_df, ALL_TYPE_KEYWORDS)
    valid_types = [o for o in type_opts if o.upper() not in curr_filters_str]
    if len(valid_types) > 1:
        st.session_state.state["step"] = "TYPE_SELECT"
        st.session_state.state["options"] = valid_types
        return

    finalize_results()

def on_option_click(option, display_label):
    # 记录用户点击的是"完整长句"，体验更好
    append_msg("user", display_label) 
    st.session_state.state["filters"].append(option)
    curr_df = st.session_state.state["results"]
    new_df = apply_filters_smart(curr_df, [option])
    st.session_state.state["results"] = new_df
    check_next_step()

def finalize_results():
    final_df = st.session_state.state["results"]
    count = len(final_df)
    top_docs = []
    for _, row in final_df.head(5).iterrows():
        top_docs.append({
            "title": row['关联文件名称'],
            "path": row['层级路径'],
            "id": row['ID']
        })
    append_msg("assistant", f"已为您找到以下 {min(5, count)} 份电路图：", msg_type="result_card", data=top_docs)
    st.session_state.state["step"] = "IDLE"

# --- 6. 界面渲染 ---

st.title("AutoCircuit Pro")

# A. 历史
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("type") == "result_card" and msg.get("data"):
            for doc in msg["data"]:
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-weight:bold; font-size:16px;">📄 {doc['title']}</div>
                        <div style="color:#666; font-size:12px;">📂 {doc['path']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button(f"⬇️ 下载", data=f"ID:{doc['id']}", file_name="doc.txt", key=f"dl_{doc['id']}_{msg_idx}")

# B. 开屏
if not st.session_state.messages:
    with st.container():
        st.markdown("<div class='hero-box'><h3>👋 欢迎使用智能电路图助手</h3><p>请直接输入车型，或点击下方示例</p></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("🚛 东风天龙"): append_msg("user", "东风天龙"); start_search("东风天龙"); st.rerun()
        if c2.button("🚜 红岩杰狮保险丝"): append_msg("user", "红岩杰狮保险丝"); start_search("红岩杰狮保险丝"); st.rerun()
        if c3.button("🚌 豪沃"): append_msg("user", "豪沃"); start_search("豪沃"); st.rerun()
        if c4.button("🔧 4HK1 ECU"): append_msg("user", "4HK1 ECU"); start_search("4HK1 ECU"); st.rerun()

# C. 输入
if prompt := st.chat_input("请输入需求..."):
    append_msg("user", prompt)
    start_search(prompt)
    st.rerun()

# D. 动态操作区 (核心修改：智能拼接按钮文案)
curr_step = st.session_state.state["step"]
opts = st.session_state.state["options"]
curr_filters = st.session_state.state["filters"]
# 获取当前上下文 (例如: "东风 天龙")
context_str = " ".join(curr_filters)

if curr_step in ["SERIES_SELECT", "TYPE_SELECT"]:
    
    if curr_step == "SERIES_SELECT":
        prompt_text = "请问您需要的是："
        # 按钮后缀
        suffix = "系列"
    else:
        prompt_text = "请问您需要哪种类型的电路图："
        suffix = "图纸"
    
    with st.chat_message("assistant"):
        st.write(prompt_text)
        cols = st.columns(3)
        for i, opt in enumerate(opts):
            # 🔥 动态生成文案：{上下文} {选项} {后缀}
            # 例如: "东风天龙 KL 系列"
            # 如果上下文已经包含选项(极少情况)，则不重复
            if opt in context_str:
                display_label = f"👉 {context_str} {suffix}"
            else:
                display_label = f"👉 {context_str} {opt} {suffix}"
            
            with cols[i % 3]:
                # key 保持唯一性
                if st.button(display_label, key=f"btn_{opt}_{i}", use_container_width=True):
                    # 传入 原始选项用于搜索，传入 显示文案用于记录历史
                    on_option_click(opt, display_label)
                    st.rerun()