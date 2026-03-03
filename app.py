import streamlit as st
import requests
import pandas as pd
import os
import json
import google.generativeai as genai

# --- 1. 環境設定 ---
QIITA_TOKEN = os.getenv('QIITA_API_KEY')
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')

st.set_page_config(page_title="Qiita Precision Ranker", layout="wide")

# --- 2. ユーザーインターフェース ---
st.title("🎯 Qiita Web Scraping")
st.markdown("---")

col_search, col_count = st.columns([4, 1])
with col_search:
    keyword = st.text_input("検索キーワード", placeholder="例: Rust メモリ管理 仕組み")
with col_count:
    count = st.number_input("取得数", min_value=5, max_value=50, value=15)

search_button = st.button("検索", type="primary", use_container_width=True)

# --- 3. メインロジック ---
if search_button and keyword:
    if not QIITA_TOKEN:
        st.error("QIITA_API_KEY が環境変数にありません。")
        st.stop()

    headers = {'Authorization': f'Bearer {QIITA_TOKEN}'}
    params = {'page': 1, 'per_page': count, 'query': keyword}
    
    with st.spinner('Qiitaから記事を収集し、AIが精査中...'):
        res = requests.get("https://qiita.com/api/v2/items", headers=headers, params=params)
        
        if res.status_code == 200:
            items = res.json()
            if not items:
                st.warning("記事が見つかりませんでした。")
                st.stop()

            # データフレーム作成（AIに渡す情報量をアップ）
            df = pd.DataFrame([
                {
                    'id': i,
                    'title': item['title'],
                    'user': item['user']['id'],
                    'likes': item['likes_count'],
                    'url': item['url'],
                    'content': item['body'][:500].replace('\n', ' ') # 500文字に指定
                } for i, item in enumerate(items)
            ])

            # --- 4. 高精度AIスコアリング ---
            if GEMINI_KEY:
                # AIへの命令をより厳格に
                context = "\n".join([f"ID:{r['id']} | Title:{r['title']} | Snippet:{r['content']}" for _, r in df.iterrows()])
                prompt = f"""
                あなたは技術情報の査読者です。検索クエリ「{keyword}」に対する以下の記事の適合性を判定してください。
                
                【採点基準】
                - 100-80点: クエリに対する直接的な回答、または深い技術解説がある。
                - 79-50点: 関連はあるが、内容が一般的すぎる、または断片的。
                - 49-0点: タイトル詐欺、クエリと無関係、または中身が薄い。

                【出力ルール】
                JSONのみ出力すること。他の文章は一切禁止。
                検索キーワードがあればその文字を色を変えたりして強調させる。
                {{"results": [ {{"id": 0, "score": 90}}, {{"id": 1, "score": 20}} ]}}

                記事リスト:
                {context}
                """
                
                try:
                    response = model.generate_content(prompt)
                    raw_json = response.text.replace('```json', '').replace('```', '').strip()
                    ai_results = json.loads(raw_json)['results']
                    
                    scores_df = pd.DataFrame(ai_results)
                    df = df.merge(scores_df, on='id', how='left').fillna(0)
                    df = df.sort_values(by='score', ascending=False)
                except:
                    st.warning("AIスコアリングがスキップされました（形式エラー）。")

            # --- 5. 結果表示 (視認性重視) ---
            # 表形式で一気に表示（文字化けや色の問題を回避）
            st.subheader("📊 AI解析済み・推奨記事リスト")
            
            # 見やすいテーブル形式
            st.dataframe(
                df[['score', 'likes', 'title', 'user', 'url']],
                column_config={
                    "score": st.column_config.ProgressColumn("AI適合度", min_value=0, max_value=100, format="%d pts"),
                    "likes": "👍 いいね",
                    "title": "記事タイトル",
                    "user": "投稿者",
                    "url": st.column_config.LinkColumn("リンク")
                },
                hide_index=True,
                use_container_width=True
            )

            # 詳細を確認したい場合のエキスパンダー
            st.markdown("### 🔍 各記事の概要")
            for _, row in df.iterrows():
                with st.expander(f"[{int(row.get('score', 0))}点] {row['title']}"):
                    st.write(f"**投稿者:** {row['user']} | **URL:** {row['url']}")
                    st.write(f"**AIが見た中身:** {row['content']}...")

        else:
            st.error(f"APIエラー: {res.status_code}")
