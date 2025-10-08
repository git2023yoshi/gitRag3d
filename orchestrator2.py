import os
from typing import Any, Dict, List

import streamlit as st
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType

st.set_page_config(page_title="RAG Chat (Azure Search + Azure OpenAI)")

# -------------------------------
# 設定を st.secrets から取得
# -------------------------------
def require(keys: List[str], src: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    missing = [k for k in keys if k not in src or src[k] in ("", None)]
    if missing:
        st.error(f"Missing keys in secrets[{prefix or 'root'}]: {', '.join(missing)}")
        st.stop()
    return src

# secrets 読み込み（env フォールバックも一応サポート）
search_conf = st.secrets.get("search", {})
aoai_conf   = st.secrets.get("azure_openai", {})
semantic    = st.secrets.get("semantic", {})
retrieval   = st.secrets.get("retrieval", {})

# 必須キーの検証
search_conf = require(["endpoint", "api_key", "index_name"], search_conf, "search")
aoai_conf   = require(["endpoint", "api_version", "api_key", "embed_deploy", "chat_deploy"], aoai_conf, "azure_openai")

VECTOR_FIELD   = retrieval.get("vector_field", "content_embedding")
TOP_K          = int(retrieval.get("k", 5))
SELECT_FIELDS  = retrieval.get("select", ["content_id", "content_text"])
SEMANTIC_NAME  = semantic.get("configuration_name")  # 無ければセマンティックは使わない

# -------------------------------
# クライアント生成（キャッシュ）
# -------------------------------
@st.cache_resource
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=search_conf["endpoint"],
        index_name=search_conf["index_name"],
        credential=AzureKeyCredential(search_conf["api_key"])
    )

@st.cache_resource
def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=aoai_conf["endpoint"],
        api_key=aoai_conf["api_key"],
        api_version=aoai_conf["api_version"],
    )

search_client = get_search_client()
openai_client = get_openai_client()

# -------------------------------
# プロンプト（systemメッセージ）
# -------------------------------
SYSTEM_MESSAGE = """
あなたはユーザーの質問に回答するチャットボットです。
回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「外部資料から引用します」と回答してください。

また、ユーザーからの質問は、Autodesk社のFusionで作成する3Dモデルに関する質問にのみ回答してください。
それ以外の質問には「すみません。回答できません。」と回答してください。

ただし、３Dモデルの完成予想図を求められた場合は回答して構いません。

""".strip()

# -------------------------------
# 検索＋生成のメイン関数
# -------------------------------
def search(history: List[Dict[str, str]]) -> str:
    question = history[-1].get("content", "")

    # 1) 質問をベクトル化（デプロイ名を指定）
    embed = openai_client.embeddings.create(
        model=aoai_conf["embed_deploy"],
        input=question
    ).data[0].embedding

    # 2) ベクトル検索クエリ
    vector_query = VectorizedQuery(
        vector=embed,
        k_nearest_neighbors=TOP_K,
        fields=VECTOR_FIELD,
        kind="vector"
    )

    # 3) ハイブリッド検索（必要に応じてセマンティック有効化）
    kwargs = dict(
        search_text=question,
        vector_queries=[vector_query],
        select=SELECT_FIELDS,
        top=TOP_K
    )

    # if SEMANTIC_NAME:
    #     kwargs.update({
    #         "query_type": QueryType.SEMANTIC,
    #         "semantic_configuration_name": SEMANTIC_NAME
    #     })

    results = search_client.search(**kwargs)

    # kwargs = dict(
    #     search_text=question,
    #     vector_queries=[vector_query],
    #     select=SELECT_FIELDS,
    #     top=TOP_K
    # )

    # USE_SEMANTIC = bool(SEMANTIC_NAME)  # secrets に設定があるときだけ試す

    # if USE_SEMANTIC:
    #     try:
    #         kwargs.update({
    #             "query_type": QueryType.SEMANTIC,
    #             "semantic_configuration_name": SEMANTIC_NAME
    #         })
    #     except Exception:
    #         # 念のため保険（SDK型評価でこける場合）
    #         pass

    # # ここで実行。Semanticが未有効だとサービス側で上記のエラーになるので、
    # # それが出たら再トライで semantic なしにフォールバックするのもアリ。
    # try:
    #     results = search_client.search(**kwargs)
    # except Exception as e:
    #     if "Semantic search is not enabled" in str(e):
    #         # フォールバック：semanticパラメータを外して再実行
    #         kwargs.pop("query_type", None)
    #         kwargs.pop("semantic_configuration_name", None)
    #         results = search_client.search(**kwargs)
    #     else:
    #         raise



    # 4) RAG用 Sources 整形
    src_lines = []
    for r in results:
        cid = r.get("content_id") or r.get("id") or "?"
        ctext = r.get("content_text") or r.get("content") or ""
        src_lines.append(f"[Source{cid}]: {ctext}")
    sources = "\n".join(src_lines) if src_lines else "(no sources)"

    # 5) メッセージ構築
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": f"{question}\n\nSources:\n{sources}"}
    ]

    # 6) 生成
    resp = openai_client.chat.completions.create(
        model=aoai_conf["chat_deploy"],  # ★デプロイ名
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content

# -------------------------------
# UI 本体
# -------------------------------
st.title("RAG Chat (Azure Search + Azure OpenAI)")

if "history" not in st.session_state:
    st.session_state["history"] = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("質問を入力してください"):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("検索・生成中..."):
            try:
                answer = search(st.session_state.history)
                st.write(answer)
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("エラーが発生しました。設定・権限・デプロイ名をご確認ください。")
                st.code(str(e))
