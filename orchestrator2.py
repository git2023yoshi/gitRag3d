import os
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import streamlit as st
from dotenv import load_dotenv

from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import QueryType

# .envファイルから環境変数を読み込む。
load_dotenv(verbose=True)

# 環境変数から各種Azureリソースへの接続情報を取得する。
SEARCH_SERVICE_ENDPOINT = os.environ.get("SEARCH_SERVICE_ENDPOINT") # Azure AI Searchのエンドポイント
SEARCH_SERVICE_API_KEY = os.environ.get("SEARCH_SERVICE_API_KEY") # Azure AI SearchのAPIキー
SEARCH_SERVICE_INDEX_NAME = os.environ.get("SEARCH_SERVICE_INDEX_NAME") # Azure AI Searchのインデックス名
AOAI_ENDPOINT = os.environ.get("AOAI_ENDPOINT") # Azure OpenAI Serviceのエンドポイント
AOAI_API_VERSION = os.environ.get("AOAI_API_VERSION") # Azure OpenAI ServiceのAPIバージョン
AOAI_API_KEY = os.environ.get("AOAI_API_KEY") # Azure OpenAI ServiceのAPIキー
AOAI_EMBEDDING_MODEL_NAME = os.environ.get("AOAI_EMBEDDING_MODEL_NAME") # Azure OpenAI Serviceの埋め込みモデル名
AOAI_CHAT_MODEL_NAME = os.environ.get("AOAI_CHAT_MODEL_NAME") # Azure OpenAI Serviceのチャットモデル名

# AIのキャラクターを決めるためのシステムメッセージを定義する。
# system_message_chat_conversation = """
# あなたはユーザーの質問に回答するチャットボットです。
# 回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
# 「Sources:」に記載されている情報以外の回答はしないでください。
# 情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
# また、ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「すみません。わかりません。」と回答してください。
# 回答の中に情報源の提示は含めないでください。例えば、回答の中に「[Source1]」や「Sources:」という形で情報源を示すことはしないでください。
# """


system_message_chat_conversation = """
あなたはユーザーの質問に回答するチャットボットです。
回答については、「Sources:」以下に記載されている内容に基づいて回答してください。回答は簡潔にしてください。
情報が複数ある場合は「Sources:」のあとに[Source1]、[Source2]、[Source3]のように記載されますので、それに基づいて回答してください。
ユーザーの質問に対して、Sources:以下に記載されている内容に基づいて適切な回答ができない場合は、「外部資料から引用します」と回答してください。
"""

# ユーザーの質問に対して回答を生成するための関数を定義する。
# 引数はチャット履歴を表すJSON配列とする。
def search(history):
    # [{'role': 'user', 'content': '有給は何日取れますか？'},{'role': 'assistant', 'content': '10日です'},
    # {'role': 'user', 'content': '一日の労働上限時間は？'}...]というJSON配列から
    # 最も末尾に格納されているJSONオブジェクトのcontent(=ユーザーの質問)を取得する。
    question = history[-1].get('content')

    # Azure AI SearchのAPIに接続するためのクライアントを生成する
    search_client = SearchClient(
        endpoint=SEARCH_SERVICE_ENDPOINT,
        index_name=SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_SERVICE_API_KEY)
    )

    # Azure OpenAI ServiceのAPIに接続するためのクライアントを生成する
    openai_client = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version=AOAI_API_VERSION
    )

    # # Azure OpenAI Serviceの埋め込み用APIを用いて、ユーザーからの質問をベクトル化する。
    # response = openai_client.embeddings.create(
    #     input = question,
    #     model = AOAI_EMBEDDING_MODEL_NAME
    # )

    # # ベクトル化された質問をAzure AI Searchに対して検索するためのクエリを生成する。
    # vector_query = VectorizedQuery(
    #     vector=response.data[0].embedding,
    #     k_nearest_neighbors=3,
    #     fields="contentVector"
    # )

    # # ベクトル化された質問を用いて、Azure AI Searchに対してベクトル検索を行う。
    # results = search_client.search(
    #     vector_queries=[vector_query],
    #     select=['id', 'content'])


    # 1) 質問を埋め込み
    embed = openai_client.embeddings.create(
        model=AOAI_EMBEDDING_MODEL_NAME,
        input=question
    ).data[0].embedding

    # 2) ベクトルクエリ（kind 明示、必要なら exhaustive）
    vector_query = VectorizedQuery(
        vector=embed,
        k_nearest_neighbors=5,     # 少し広めに
        fields="content_embedding",
        kind="vector",             # 追加
        # exhaustive=True          # 厳密探索が必要なら
    )

    # 3) ハイブリッド検索 + セマンティック（設定していれば）
    results = search_client.search(
        search_text=question,              # 追加：ハイブリッド
        vector_queries=[vector_query],
        # query_type=QueryType.SEMANTIC,     # セマンティック有効時
        semantic_configuration_name="default",  # 既定名は環境に合わせて
        # select=["id", "content"],
        select=['content_id', 'content_text'],
        top=5                               # 返却件数を明示
    )




    # チャット履歴の中からユーザーの質問に対する回答を生成するためのメッセージを生成する。
    messages = []

    # 先頭にAIのキャラ付けを行うシステムメッセージを追加する。
    messages.insert(0, {"role": "system", "content": system_message_chat_conversation})

    # 回答を生成するためにAzure AI Searchから取得した情報を整形する。
    sources = ["[Source" + result["content_id"] + "]: " + result["content_text"] for result in results]
    source = "\n".join(sources)

    # ユーザーの質問と情報源を含むメッセージを生成する。
    user_message = """
    {query}

    Sources:
    {source}
    """.format(query=question, source=source)

    # メッセージを追加する。
    messages.append({"role": "user", "content": user_message})

    # Azure OpenAI Serviceに回答生成を依頼する。
    response = openai_client.chat.completions.create(
        model=AOAI_CHAT_MODEL_NAME,
        messages=messages
    )
    answer = response.choices[0].message.content

    # 回答を返す。
    return answer

# ここからは画面を構築するためのコード
# チャット履歴を初期化する。
if "history" not in st.session_state:
    st.session_state["history"] = []

# チャット履歴を表示する。
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ユーザーが質問を入力したときの処理を記述する。
if prompt := st.chat_input("質問を入力してください"):

    # ユーザーが入力した質問を表示する。
    with st.chat_message("user"):
        st.write(prompt)

    # ユーザの質問をチャット履歴に追加する
    st.session_state.history.append({"role": "user", "content": prompt})

    # ユーザーの質問に対して回答を生成するためにsearch関数を呼び出す。
    response = search(st.session_state.history)

    # 回答を表示する。
    with st.chat_message("assistant"):
        st.write(response)

    # 回答をチャット履歴に追加する。
    st.session_state.history.append({"role": "assistant", "content": response})