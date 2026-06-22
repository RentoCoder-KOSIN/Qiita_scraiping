import os
import re
import json
import requests
import pandas as pd
import streamlit as st

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

QIITA_TOKEN = os.getenv("QIITA_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def get_gemini_model():
    if not GENAI_AVAILABLE or not GEMINI_KEY:
        return None
    try:
        genai.configure(api_key=GEMINI_KEY)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
        return None


def ai_score(model, keyword: str, df: pd.DataFrame) -> pd.DataFrame:
    context = "\n".join(
        f"ID:{r['id']} | Title:{r['title']} | Snippet:{r['content']}"
        for _, r in df.iterrows()
    )
    prompt = f"""
あなたは技術情報の査読者です。検索クエリ「{keyword}」に対する以下の記事の適合性を判定してください。

【採点基準】
- 100-80点: クエリへの直接的な回答、または深い技術解説がある
- 79-50点: 関連はあるが内容が一般的すぎる、または断片的
- 49-0点: タイトル詐欺、クエリと無関係、または内容が薄い

【出力ルール】
JSONのみ出力すること。他の文章は一切禁止。
{{"results": [{{"id": 0, "score": 90}}, {{"id": 1, "score": 20}}]}}

記事リスト:
{context}
"""
    try:
        resp = model.generate_content(prompt)
        text = resp.text
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("JSONが見つかりませんでした")
        payload = json.loads(m.group(0))
        scores_df = pd.DataFrame(payload.get("results", []))
        if "id" in scores_df.columns:
            df = df.merge(scores_df, on="id", how="left")
        df["score"] = df.get("score", pd.Series([0] * len(df))).fillna(0)
    except Exception as e:
        st.warning(f"AIスコアリングをスキップしました: {e}")
        df["score"] = 0
    return df


def fetch_qiita(keyword: str, count: int) -> list:
    if not QIITA_TOKEN:
        st.error("QIITA_API_KEY が環境変数に設定されていません。")
        st.stop()
    headers = {"Authorization": f"Bearer {QIITA_TOKEN}"}
    params = {"page": 1, "per_page": count, "query": keyword}
    try:
        res = requests.get(
            "https://qiita.com/api/v2/items",
            headers=headers, params=params, timeout=10
        )
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        st.error(f"Qiita API エラー: {e}")
        st.stop()


# --- UI ---
st.set_page_config(page_title="Qiita Precision Ranker", layout="wide")
st.title("🎯 Qiita Web Scraping")
st.markdown("---")

col_search, col_count = st.columns([4, 1])
with col_search:
    keyword = st.text_input("検索キーワード", placeholder="例: Rust メモリ管理 仕組み")
with col_count:
    count = st.number_input("取得数", min_value=5, max_value=50, value=15)

if st.button("検索", type="primary", use_container_width=True) and keyword:
    with st.spinner("Qiitaから記事を収集し、AIが精査中..."):
        items = fetch_qiita(keyword, int(count))

        if not items:
            st.warning("記事が見つかりませんでした。")
            st.stop()

        df = pd.DataFrame([
            {
                "id": i,
                "title": item.get("title", ""),
                "user": item.get("user", {}).get("id", ""),
                "likes": item.get("likes_count", 0),
                "url": item.get("url", ""),
                "content": (item.get("body", "")[:500] or "").replace("\n", " "),
                "score": 0,
            }
            for i, item in enumerate(items)
        ])

        model = get_gemini_model()
        if model:
            df = ai_score(model, keyword, df)

        df = df.sort_values("score", ascending=False)

    st.subheader("📊 AI解析済み・推奨記事リスト")
    st.dataframe(
        df[["score", "likes", "title", "user", "url"]],
        column_config={
            "score": st.column_config.ProgressColumn("AI適合度", min_value=0, max_value=100, format="%d pts"),
            "likes": "👍 いいね",
            "title": "記事タイトル",
            "user": "投稿者",
            "url": st.column_config.LinkColumn("リンク"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### 🔍 各記事の概要")
    for _, row in df.iterrows():
        with st.expander(f"[{int(row['score'])}点] {row['title']}"):
            st.write(f"**投稿者:** {row['user']} | **URL:** {row['url']}")
            st.write(f"**AIが見た中身:** {row['content']}...")
