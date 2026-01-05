import random
import streamlit as st
from collections import Counter
from utils import (
    load_data,
    load_keywords,
    search_topk,
    DATA_CSV_DEFAULT,
    KEYWORDS_TXT_DEFAULT,
)

st.set_page_config(page_title="AutoCircuit Chatbot", layout="wide")

st.title("AutoCircuit Chatbot")
st.caption("多轮选择题引导 · 稳定定位车辆电路图文档")

# =========================
# Session State 初始化
# =========================
for key, default in {
    "res_full": None,
    "selected_brand": None,
    "selected_model": None,
    "search_history": [],
    "current_search_entry": None,  # 用来存储当前正在进行的搜索条目
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================
# 侧边栏
# =========================
with st.sidebar:
    st.header("数据配置")
    csv_path = st.text_input("资料清单.csv 路径", value=DATA_CSV_DEFAULT)

    st.divider()
    st.header("检索参数")
    st.caption("根据您的需求调整检索参数。")
    k = st.slider("最终返回条数 k", 1, 10, 5, help="设置返回的文档数量")
    candidate_pool = st.slider("候选池大小", 30, 200, 100, step=10, help="候选文档池大小")
    min_score = st.slider("最低相关性分数", 0, 100, 55, step=1, help="设置最低相关性分数")

    st.divider()
    use_llm = st.checkbox("启用 LLM 意图辅助", value=False, help="启用 GPT-4o-mini 解析用户输入，提升检索精度")

    st.divider()
    if st.button("🔄 重新开始"):
        st.session_state.clear()
        st.session_state.search_history = []  # 清空搜索历史
        st.session_state.current_search_entry = None  # 清空当前搜索条目
        st.rerun()  # 改为 st.rerun()

@st.cache_data(show_spinner=True)
def cached_load(csv_path: str):
    return load_data(csv_path)

try:
    df = cached_load(csv_path)
    st.success(f"数据加载成功：{len(df)} 条")
except Exception as e:
    st.error(f"数据加载失败：{e}")
    st.stop()

tab1, tab2 = st.tabs(["🔎 查询", "🧪 keywords.txt 抽测"])

# =========================
# 工具函数
# =========================
def extract_brand(path: str):
    parts = str(path).split("->")
    for i, p in enumerate(parts):
        if p in ("工程机械", "商用车"):
            if i + 1 < len(parts):
                return parts[i + 1]
    return None

def extract_model(path: str):
    parts = str(path).split("->")
    if len(parts) >= 2:
        return parts[-1]
    return None

# =========================
# 查询 + 双轮引导
# =========================
def query_search():
    query = st.session_state.query_input  # 使用 text_input 获取实时输入内容
    if query.strip():
        with st.spinner("检索中..."):
            st.session_state.res_full = search_topk(
                df,
                query=query,
                k=candidate_pool,
                candidate_pool=candidate_pool,
                min_score=min_score,
                use_llm_intent=use_llm,
            )
        # 创建新的搜索条目
        st.session_state.current_search_entry = {
            "query": query,
            "brand": None,
            "model": None,
        }

# 监听回车事件，触发查询
st.text_input(
    "请输入你的查询（自然语言 / 关键词）",
    placeholder="例如：4HK1住友ecu电路图 / 东风天龙仪表图",
    key="query_input",  # 使用 `key` 让它能实时同步到 `st.session_state.query_input`
    on_change=query_search  # 回车时触发查询
)

# 添加一个“搜索”按钮
if st.button("搜索"):
    query_search()  # 点击搜索按钮后进行查询

res = st.session_state.res_full
if res is None or res.empty:
    st.stop()

# =========================
# 已选条件展示（UX 核心） 
# =========================
st.markdown("### ✅ 已选择条件")
if st.session_state.current_search_entry["brand"]:
    st.write(f"- 品牌：{st.session_state.current_search_entry['brand']}")
if st.session_state.current_search_entry["model"]:
    st.write(f"- 型号：{st.session_state.current_search_entry['model']}")
if not st.session_state.current_search_entry["brand"] and not st.session_state.current_search_entry["model"]:
    st.caption("（尚未选择任何筛选条件）")

# =========================
# 第一轮：品牌
# =========================
if st.session_state.current_search_entry["brand"] is None:
    st.subheader("请选择品牌")
    brands = [extract_brand(p) for p in res["层级路径"] if extract_brand(p)]
    counter = Counter(brands).most_common(5)

    num_cols = max(1, len(counter))  # 如果 counter 为空，至少为 1 列
    cols = st.columns(num_cols)

    for i, (b, c) in enumerate(counter):
        with cols[i]:
            if st.button(f"{b}（{c}）"):
                st.session_state.current_search_entry["brand"] = b
                if st.session_state.current_search_entry["brand"] and st.session_state.current_search_entry["model"]:
                    st.session_state.search_history.append(st.session_state.current_search_entry.copy())
                st.rerun()  # 改为 st.rerun()

# 应用品牌过滤
if st.session_state.current_search_entry["brand"]:
    res = res[res["层级路径"].str.contains(st.session_state.current_search_entry["brand"])]

# =========================
# 第二轮：型号
# =========================
if st.session_state.current_search_entry["brand"] and st.session_state.current_search_entry["model"] is None:
    if len(res) > k:
        st.subheader("请选择型号 / 系列")
        models = [extract_model(p) for p in res["层级路径"] if extract_model(p)]
        counter = Counter(models).most_common(5)

        cols = st.columns(len(counter))
        for col, (m, c) in zip(cols, counter):
            with col:
                if st.button(f"{m}（{c}）"):
                    st.session_state.current_search_entry["model"] = m
                    if st.session_state.current_search_entry["brand"] and st.session_state.current_search_entry["model"]:
                        st.session_state.search_history.append(st.session_state.current_search_entry.copy())
                    st.rerun()  # 改为 st.rerun()

# 应用型号过滤
if st.session_state.current_search_entry["model"]:
    res = res[res["层级路径"].str.contains(st.session_state.current_search_entry["model"])]

# =========================
# 最终结果
# =========================
st.subheader("📄 匹配结果")
if res.empty:
    st.warning("筛选后无结果，可尝试重新开始。")
else:
    st.dataframe(res.head(k), use_container_width=True)

# =========================
# 显示搜索历史
# =========================
with st.expander("🕒 搜索历史", expanded=True):
    if st.session_state.search_history:
        for entry in st.session_state.search_history:
            st.write(f"**查询**: {entry['query']}")
            if entry["brand"]:
                st.write(f"- **品牌**: {entry['brand']}")
            if entry["model"]:
                st.write(f"- **型号**: {entry['model']}")
            st.markdown("---")
    else:
        st.caption("没有搜索历史。")
