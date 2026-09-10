import json
import pandas as pd
import streamlit as st
from groq import Groq
from tavily import TavilyClient

st.set_page_config(page_title="Storyboard Suggester",  layout="wide")

GROQ_API_KEY = ""
TAVILY_API_KEY = ""

groq_key = GROQ_API_KEY
tavily_key = TAVILY_API_KEY

if "graphs" not in st.session_state:
    st.session_state.graphs = [] 
if "result" not in st.session_state:
    st.session_state.result = None


with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-120b"],
        index=0,
    )
    top_n = st.slider("How many graphs should the final storyboard have?", 2, 10, 5)
    use_web_context = st.checkbox(
        "Use Tavily to pull in data-storytelling best practices", value=True
    )
    domain_hint = st.text_input(
        "Domain / audience (optional)",
        placeholder="e.g. quarterly sales review for a retail exec team",
        help="Helps Tavily search for relevant storytelling conventions, and helps "
             "the LLM judge what 'important' means for this audience.",
    )

st.title("Storyboard Suggester")
st.caption(
    "Upload your data, list the graphs you've built, and get an LLM-backed "
    "recommendation for which ones belong in the final storyboard."
)

st.subheader("1. Upload your dataset")
uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])

df = None
dataset_summary = None
if uploaded is not None:
    try:
        filename = uploaded.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            excel_file = pd.ExcelFile(uploaded)
            sheet_names = excel_file.sheet_names
            if len(sheet_names) > 1:
                chosen_sheet = st.selectbox("Sheet", sheet_names)
            else:
                chosen_sheet = sheet_names[0]
            df = excel_file.parse(chosen_sheet)

        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("Preview data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        desc = df.describe(include="all").transpose()
        schema_rows = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique(dropna=True)
            row = {"column": col, "dtype": dtype, "n_unique": n_unique}
            if pd.api.types.is_numeric_dtype(df[col]):
                row["min"] = round(float(df[col].min()), 3) if df[col].notna().any() else None
                row["max"] = round(float(df[col].max()), 3) if df[col].notna().any() else None
                row["mean"] = round(float(df[col].mean()), 3) if df[col].notna().any() else None
            schema_rows.append(row)
        dataset_summary = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "columns": schema_rows,
        }
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")

st.divider()

st.subheader("2. List the graphs / plots you've created")

with st.form("add_graph_form", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 2])
    g_name = c1.text_input("Graph name", placeholder="Revenue trend by region")
    g_x = c2.text_input("X-axis (optional)", placeholder="month")
    g_y = c3.text_input("Y-axis (optional)", placeholder="revenue")
    g_type = c4.selectbox("Chart type (optional)",
                           ["", "line", "bar", "scatter", "pie", "heatmap",
                            "histogram", "box", "area", "other"])
    g_note = st.text_input("Note (optional)", placeholder="e.g. shows Q4 spike")
    submitted = st.form_submit_button("Add graph")
    if submitted:
        if g_name:
            st.session_state.graphs.append(
                {"name": g_name, "x": g_x, "y": g_y, "chart_type": g_type, "note": g_note}
            )
        else:
            st.warning("Graph name is required.")

if st.session_state.graphs:
    st.write(f"**{len(st.session_state.graphs)} graph(s) added:**")
    for i, g in enumerate(st.session_state.graphs):
        cols = st.columns([5, 1])
        axis_bits = []
        if g["x"]:
            axis_bits.append(f"x: `{g['x']}`")
        if g["y"]:
            axis_bits.append(f"y: `{g['y']}`")
        axis_str = " · ".join(axis_bits) if axis_bits else "no axes specified"
        label = f"**{g['name']}** — {axis_str}"
        if g["chart_type"]:
            label += f" · type: {g['chart_type']}"
        if g["note"]:
            label += f"  \n_{g['note']}_"
        cols[0].markdown(label)
        if cols[1].button("Remove", key=f"rm_{i}"):
            st.session_state.graphs.pop(i)
            st.rerun()
else:
    st.info("Add at least 2 graphs to get a storyboard recommendation.")

st.divider()

def get_web_context(client: TavilyClient, domain_hint: str) -> str:
    """Pull a short block of best-practice context from Tavily to ground the LLM."""
    query = "data storytelling dashboard best practices which charts to include"
    if domain_hint:
        query = f"{domain_hint} — {query}"
    try:
        res = client.search(query=query, search_depth="basic", max_results=4)
        snippets = []
        for r in res.get("results", []):
            snippets.append(f"- {r.get('title', '')}: {r.get('content', '')[:300]}")
        return "\n".join(snippets)
    except Exception as e:
        return f"(Tavily search failed: {e})"


SYSTEM_PROMPT = """You are a senior data storytelling consultant. You are given:
1. A summary of a dataset's schema (columns, types, basic stats).
2. A list of graphs/plots the user has already built (name, x-axis, y-axis, chart type, notes).
3. Optionally, some web context on data-storytelling best practices.

Your job: decide which of the listed graphs deserve a spot in a final "storyboard"
(a short, curated sequence of visuals that tells a clear, non-redundant story to
the stated audience), and which should be cut.

Judge importance using signals like:
- Relevance of the x/y fields to the dataset's key numeric/categorical columns
- Whether the graph reveals a trend, comparison, distribution, or relationship
  that materially matters (not just any visual on the data)
- Redundancy with other graphs in the list (near-duplicate stories should not
  both make the cut)
- Fit for the stated audience/domain, if given
- Narrative flow: does an ordered subset of the kept graphs tell a coherent story?

Return STRICT JSON only, no markdown fences, no prose outside the JSON, in this shape:
{
  "recommended_storyboard": [
    {"name": "...", "rank": 1, "reason": "1-2 sentence reason this belongs, tied to the data/audience"}
  ],
  "cut": [
    {"name": "...", "reason": "1 sentence reason this was cut (e.g. redundant with X, low signal)"}
  ],
  "narrative_summary": "2-3 sentences describing the story the recommended storyboard tells in order"
}
Only use graph names exactly as given. Keep the recommended_storyboard to at most the requested count.
"""


def call_groq(client: Groq, model: str, dataset_summary: dict, graphs: list,
              web_context: str, top_n: int, domain_hint: str) -> dict:
    user_payload = {
        "dataset_summary": dataset_summary,
        "graphs": graphs,
        "max_storyboard_size": top_n,
        "audience_or_domain": domain_hint or "general / not specified",
        "web_context_on_best_practices": web_context or "(none)",
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


st.subheader("3. Get your storyboard recommendation")
run = st.button("Suggest storyboard", type="primary",
                 disabled=(df is None or len(st.session_state.graphs) < 2))

if df is None:
    st.caption("Upload a dataset above to enable this.")
elif len(st.session_state.graphs) < 2:
    st.caption("Add at least 2 graphs above to enable this.")

if run:
    if not groq_key or groq_key == "your-groq-api-key-here":
        st.error("Set GROQ_API_KEY near the top of app.py.")
    elif use_web_context and (not tavily_key or tavily_key == "your-tavily-api-key-here"):
        st.error("Set TAVILY_API_KEY near the top of app.py, or uncheck the Tavily option.")
    else:
        with st.spinner("Gathering context and reasoning about your graphs..."):
            web_context = ""
            if use_web_context:
                tavily_client = TavilyClient(api_key=tavily_key)
                web_context = get_web_context(tavily_client, domain_hint)

            groq_client = Groq(api_key=groq_key)
            try:
                result = call_groq(
                    groq_client, model, dataset_summary, st.session_state.graphs,
                    web_context, top_n, domain_hint,
                )
                st.session_state.result = result
            except Exception as e:
                st.error(f"Groq call failed: {e}")
                st.session_state.result = None


if st.session_state.result:
    result = st.session_state.result
    st.success("Storyboard ready")

    st.markdown("### Narrative")
    st.write(result.get("narrative_summary", ""))

    st.markdown("### Recommended for the storyboard")
    for item in sorted(result.get("recommended_storyboard", []), key=lambda r: r.get("rank", 999)):
        st.markdown(f"**{item.get('rank', '?')}. {item.get('name', '')}**")
        st.write(item.get("reason", ""))

    with st.expander(f"Cut ({len(result.get('cut', []))})"):
        for item in result.get("cut", []):
            st.markdown(f"- **{item.get('name', '')}** — {item.get('reason', '')}")

    st.download_button(
        "Download result as JSON",
        data=json.dumps(result, indent=2),
        file_name="storyboard_recommendation.json",
        mime="application/json",
    )