"""
AI開発コーチLLM - Llama 3 8B + RAG実装（リモートAPI版）
Windowsでの互換性を改善したパッチ版
"""

# パッチを適用（Windows環境のみ）
import platform
if platform.system() == "Windows":
    try:
        import patch_langchain
        patch_langchain.patch_langchain()
    except ImportError:
        print("警告: patch_langchainモジュールが見つかりません。Windowsでは実行できない可能性があります。")

import os
import json
import logging
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from typing import Any, Dict, List, Optional

# ベクトルストア関連のimportはtry-exceptで囲む
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain.chains import RetrievalQA
    vector_store_available = True
except ImportError as e:
    print(f"警告: ベクトルストア関連のモジュールが読み込めませんでした: {e}")
    print("RAG機能は無効化されます。")
    vector_store_available = False

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .envファイルから環境変数を読み込む
load_dotenv()

# 設定パラメータ
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
DB_DIR = os.getenv("DB_DIR", "./db")
# デフォルトは空にして必ず.envから設定するように促す
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

# APIが設定されていない場合のチェック
if not OLLAMA_API_BASE:
    logger.error("OLLAMA_API_BASEが設定されていません。.envファイルにリモートOllama APIのURLを設定してください。")
    raise ValueError("OLLAMA_API_BASEが設定されていません。.envファイルを確認してください。")

# システムプロンプト
SYSTEM_PROMPT = """
あなたはAI開発のエキスパートコーチです。ユーザーがAI技術（機械学習、深層学習、強化学習、自然言語処理など）を学び、実践するのを支援します。

以下の方針に従ってください：
1. 回答は正確で最新の情報に基づくこと
2. 初心者には基本概念をわかりやすく説明し、上級者には深い洞察を提供すること
3. 実践的なコード例を提供すること（主にPython、PyTorch、TensorFlow）
4. 学習リソースを適切に推奨すること
5. 質問の背後にある意図を理解し、適切な指導を行うこと
6. 倫理的なAI開発の重要性を強調すること

提供された参考情報があれば、それを活用して回答してください。
ない場合は、あなたの知識を基に正確で有用な情報を提供してください。

ユーザーの質問に応じて、概念説明、コード例、実装アドバイス、トラブルシューティング、最新トレンドなどの情報を提供してください。
"""

# カスタムLLMクラス（Ollama API呼び出し用）
class OllamaAPI(LLM):
    """Ollama APIを呼び出すカスタムLLMクラス"""
    
    api_base: str = OLLAMA_API_BASE
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    api_key: str = OLLAMA_API_KEY
    
    @property
    def _llm_type(self) -> str:
        """LLMタイプの取得"""
        return "ollama_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """APIを呼び出してレスポンスを取得"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # API認証キーがあれば追加
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            logger.debug(f"API呼び出し: {self.api_base}/api/generate")
            response = requests.post(
                f"{self.api_base}/api/generate",
                headers=headers,
                json=payload,
                timeout=60,  # 60秒のタイムアウト
                verify=False  # 自己署名証明書を許可（テスト環境用）
            )
            
            if response.status_code != 200:
                error_msg = f"API呼び出しエラー: ステータスコード {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f", 詳細: {json.dumps(error_details, ensure_ascii=False)}"
                except:
                    error_msg += f", レスポンス: {response.text}"
                logger.error(error_msg)
                return f"エラー: APIリクエストが失敗しました。{error_msg}"
            
            result = response.json()
            text = result.get("response", "")
            
            # ストップトークンの処理
            if stop:
                text = self._identify_stop_tokens(text, stop)
                
            return text
            
        except requests.exceptions.ConnectionError:
            logger.error(f"API接続エラー: {self.api_base}に接続できません")
            return "エラー: Ollama APIに接続できませんでした。サーバーが実行中か、URLが正しいか確認してください。"
        except Exception as e:
            logger.error(f"API呼び出し中の例外: {str(e)}")
            return f"エラー: API呼び出し中に例外が発生しました: {str(e)}"
    
    def _identify_stop_tokens(self, text: str, stop: List[str]) -> str:
        """ストップトークンが含まれていたら、そこで切る"""
        for stop_token in stop:
            if stop_token in text:
                return text[:text.index(stop_token)]
        return text

class AICoach:
    def __init__(self, use_mock=False):
        """AIコーチの初期化"""
        try:
            # テスト用の場合はモックLLMを使用
            if use_mock:
                self.llm = MockLLM()
                logger.info("モックLLMを使用します。")
            else:
                # Ollama APIクライアントの初期化
                self.llm = OllamaAPI(
                    api_base=OLLAMA_API_BASE,
                    model_name=MODEL_NAME,
                    temperature=TEMPERATURE,
                    api_key=OLLAMA_API_KEY
                )
                logger.info(f"リモートOllama API ({OLLAMA_API_BASE}) に接続しました")
            
            # ベクトルストアの初期化（利用可能な場合のみ）
            if vector_store_available:
                self.init_vector_store()
            else:
                self.vectorstore = None
            
            # メモリの初期化
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # RAGベースのQAチェーンの初期化
            self.qa_chain = self.setup_qa_chain() if vector_store_available else None
            
            # 一般会話用のチェーンの初期化
            self.conversation_chain = self.setup_conversation_chain()
            
            logger.info("AIコーチが正常に初期化されました")
        except Exception as e:
            logger.error(f"初期化エラー: {str(e)}")
            raise
    
    def init_vector_store(self):
        """ベクトルストアの初期化"""
        # ベクトルストアが既に存在するか確認
        if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
            logger.info(f"既存のベクトルストアを読み込みます: {DB_DIR}")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        else:
            logger.info("ベクトルストアが見つかりません。新規作成します。")
            self.create_vector_store()
    
    def create_vector_store(self):
        """ドキュメントからベクトルストアを作成"""
        try:
            # ドキュメントディレクトリの存在確認
            if not os.path.exists(DOCS_DIR):
                os.makedirs(DOCS_DIR)
                # サンプルファイルの作成
                with open(os.path.join(DOCS_DIR, "sample.txt"), "w", encoding="utf-8") as f:
                    f.write("これはAI開発の学習資料のサンプルです。実際のコンテンツに置き換えてください。")
                logger.info(f"ドキュメントディレクトリを作成しました: {DOCS_DIR}")
            
            # ドキュメントの読み込み
            logger.info(f"ドキュメントを読み込みます: {DOCS_DIR}")
            loader = DirectoryLoader(DOCS_DIR, glob="**/*.{txt,md,pdf}", loader_cls=TextLoader)
            documents = loader.load()
            
            if len(documents) == 0:
                logger.warning("ドキュメントが見つかりませんでした。基本的な会話のみ可能です。")
                self.vectorstore = None
                return
            
            # テキスト分割
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # ベクトルストアの作成
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
            self.vectorstore.persist()
            logger.info(f"ベクトルストアを作成しました: {len(splits)}チャンク")
        except Exception as e:
            logger.error(f"ベクトルストア作成エラー: {str(e)}")
            self.vectorstore = None
    
    def setup_qa_chain(self):
        """RAGベースのQAチェーンのセットアップ"""
        if self.vectorstore is None:
            return None
        
        qa_prompt_template = """
{system_prompt}

以下の参考情報を活用してユーザーの質問に答えてください：

参考情報:
{context}

ユーザーの質問: {question}

あなたの回答:
"""
        qa_prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "question"],
            template=qa_prompt_template,
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def setup_conversation_chain(self):
        """一般会話用のチェーンのセットアップ"""
        conversation_prompt_template = """
{system_prompt}

会話履歴:
{chat_history}

ユーザーの質問: {question}

あなたの回答:
"""
        conversation_prompt = PromptTemplate(
            input_variables=["system_prompt", "chat_history", "question"],
            template=conversation_prompt_template,
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=conversation_prompt,
            memory=self.memory,
            verbose=False
        )
    
    def get_response(self, user_input):
        """ユーザー入力に対する応答を生成する"""
        try:
            # RAGチェーンが利用可能な場合、まずそれを試す
            if self.qa_chain is not None:
                try:
                    result = self.qa_chain({"system_prompt": SYSTEM_PROMPT, "query": user_input})
                    response = result["result"]
                    logger.info("RAGベースの回答を生成しました")
                    return response
                except Exception as e:
                    logger.warning(f"RAG回答生成エラー: {str(e)}。通常の会話チェーンを使用します。")
            
            # RAGチェーンが使えない場合や失敗した場合は通常の会話チェーンを使用
            result = self.conversation_chain.predict(
                system_prompt=SYSTEM_PROMPT,
                question=user_input
            )
            logger.info("通常の会話チェーンで回答を生成しました")
            return result
            
        except Exception as e:
            error_msg = f"回答生成中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            return f"すみません、エラーが発生しました: {str(e)}\n\nもう一度質問してみてください。"

# テスト用のモックLLM
class MockLLM(LLM):
    """テスト用のモックLLM"""
    
    @property
    def _llm_type(self) -> str:
        return "mock_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return f"これはテスト用のモック回答です。実際のLLMは使用されていません。\n質問：{prompt[-50:]}"

def initialize_conversation(use_mock=False):
    """会話を初期化する"""
    try:
        coach = AICoach(use_mock=use_mock)
        return coach, "AIコーチが起動しました。AI開発について質問してください。"
    except Exception as e:
        logger.error(f"初期化エラー: {str(e)}")
        return None, f"エラー: AIコーチの初期化に失敗しました。{str(e)}"

def main():
    """簡易的なコンソールインターフェース"""
    print("AI開発コーチLLM - Windows互換パッチ版")
    print(f"API接続先: {OLLAMA_API_BASE}")
    print(f"使用モデル: {MODEL_NAME}")
    print("-" * 50)
    
    # テストモードのオプション
    test_mode = False
    if len(OLLAMA_API_BASE) == 0 or OLLAMA_API_BASE == "http://localhost:11434":
        # APIサーバーが設定されていない場合
        use_mock = input("Ollama APIサーバーが設定されていないか、ローカルに設定されています。テストモードで実行しますか？(y/n): ")
        if use_mock.lower() == 'y':
            test_mode = True
            print("テストモードで実行します。実際のLLM機能は使用されません。")
    
    coach, start_message = initialize_conversation(use_mock=test_mode)
    if coach is None:
        print(start_message)
        return
    
    print(start_message)
    
    while True:
        user_input = input("\nあなた: ")
        
        if user_input.lower() in ["終了", "exit", "quit"]:
            print("AIコーチを終了します。")
            break
            
        response = coach.get_response(user_input)
        print(f"\nAIコーチ: {response}")

if __name__ == "__main__":
    main() 