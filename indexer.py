import os
import sys
from azure.search.documents import SearchClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from pypdf import PdfReader
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

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

# ドキュメント内のテキストをチャンクに分割する際の区切り文字を指定する。
separator = ["\n\n", "\n", "。", "、", " ", ""]

# チャンクをインデクシングする関数を定義する。
# 引数はチャンクのリストとする。
def index_docs(chunks: list):
    # Azure AI SearchのAPIに接続するためのクライアントを生成する。
    searchClient = SearchClient(
        endpoint=SEARCH_SERVICE_ENDPOINT,
        index_name=SEARCH_SERVICE_INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_SERVICE_API_KEY)
    )

    # Azure OpenAIのAPIに接続するためのクライアントを生成する。
    openAIClient = AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        api_version = AOAI_API_VERSION
    )


    # 引数によって渡されたチャンクのリストをベクトル化して、Azure AI Searchに登録する。
    for i, chunk in enumerate(chunks):
        print(f"{i+1}個目のチャンクを処理中...")
        response = openAIClient.embeddings.create(
            input = chunk,
            model = AOAI_EMBEDDING_MODEL_NAME
        )

        # チャンクのテキストと、そのチャンクをベクトル化したものをAzure AI Searchに登録する。
        document = {"id": str(i), "content": chunk, "contentVector": response.data[0].embedding}
        searchClient.upload_documents([document])

# テキストを指定したサイズで分割する関数を定義する。
def create_chunk(content: str, separator: str, chunk_size: int = 1000, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_overlap=overlap,
        chunk_size=chunk_size,
        separators=separator
    )

    chunks = splitter.split_text(content)
    return chunks

# ドキュメントからテキストを抽出する関数を定義する。
def extract_text_from_docs(filepath):
    print(f"{filepath}内のテキストを抽出中...")
    text = ""
    reader = PdfReader(filepath)
    for page in reader.pages:
        text += page.extract_text()

    print("テキストの抽出が完了しました")
    return text

if __name__ == "__main__":
    # インデクサーのコマンドライン引数からドキュメントのファイルパスを取得する。
    if len(sys.argv) < 2:
        print("ドキュメントのファイルパスを指定してください")
        sys.exit(1)

    filename = sys.argv[1]

    # ドキュメントからテキストを抽出する。
    content = extract_text_from_docs(filename)

    # ドキュメントから抽出したテキストをチャンクに分割する。
    chunks = create_chunk(content, separator)

    # チャンクをAzure AI Searchにインデックスする
    index_docs(chunks)

    print("インデックスの作成が完了しました")


