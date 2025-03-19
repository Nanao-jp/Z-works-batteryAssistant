"""
AI開発コーチLLM - OpenAI GPT + RAG実装
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

# OpenAI LLM
from langchain_openai import ChatOpenAI

# ベクトルストア関連のimportはtry-exceptで囲む
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
DB_DIR = os.getenv("DB_DIR", "./db")
# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# APIKeyが設定されていない場合のチェック
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEYが設定されていません。.envファイルにAPIキーを設定してください。")
    raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")

# システムプロンプト
SYSTEM_PROMPT = """
あなたはZ-works社の系統用蓄電池営業アシスタントです。ユーザーが蓄電池製品に関する情報を求めたり、技術的な質問をしたり、見積もり支援を求めたりする際にサポートします。

回答の形式と内容に関して以下のルールを必ず守ってください：

【形式面のルール】
1. 読みやすさを重視し、適切な場所で改行する
2. 一つの段落は3-4行以内にまとめる
3. 重要なポイントは箇条書き（・）で示す
4. 専門用語は必要に応じて平易な言葉で説明を補足する
5. 長文は避け、簡潔に説明する

【内容面のルール】
1. 回答は正確で最新の情報に基づくこと
2. 製品の特徴や利点を明確に説明すること
3. 技術的な質問には詳細かつ正確に回答すること
4. 顧客のニーズに合わせた製品提案を行うこと
5. 競合製品との比較情報を提供すること
6. 成功事例や導入実績を紹介すること

【回答の締めくくり】
回答の最後には必ず以下のような提案で締めくくってください：
- 「○○について詳しくご説明しましょうか？」
- 「実際の導入事例をご紹介しましょうか？」
- 「ご予算や導入規模について詳しくお聞かせいただけますか？」
- 「他に気になる点はございますか？」
など、会話を継続させるための具体的な提案を必ず入れてください。

提供された参考情報があれば、それを活用して回答してください。
ない場合は、あなたの知識を基に正確で有用な情報を提供してください。

ユーザーの質問に応じて、製品説明、技術仕様、導入メリット、コスト面の情報などを提供してください。
"""

# OpenAI API用のLLMクラス
class OpenAIAPI(LLM):
    """OpenAI APIを呼び出すカスタムLLMクラス"""
    
    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    api_key: str = OPENAI_API_KEY
    
    @property
    def _llm_type(self) -> str:
        """LLMタイプの取得"""
        return "openai_api"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """APIを呼び出してレスポンスを取得"""
        try:
            # ChatOpenAIを使用
            from langchain_openai.chat_models import ChatOpenAI
            from langchain_core.messages import HumanMessage
            
            chat_model = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            
            # プロンプトをHumanMessageに変換
            message = HumanMessage(content=prompt)
            
            # メッセージを送信して応答を取得
            response = chat_model.invoke([message])
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI API呼び出し中にエラーが発生しました: {e}")
            return f"申し訳ありません。エラーが発生しました: {str(e)}"

class AICoach:
    def __init__(self, use_mock=False):
        """営業アシスタントの初期化"""
        try:
            # テスト用の場合はモックLLMを使用
            if use_mock:
                self.llm = MockLLM()
                logger.info("モックLLMを使用します。")
            else:
                # OpenAI APIクライアントの初期化
                self.llm = OpenAIAPI(
                    model_name=MODEL_NAME,
                    temperature=TEMPERATURE,
                    api_key=OPENAI_API_KEY
                )
                logger.info(f"OpenAI API に接続しました")
            
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
            
            logger.info("Z-works蓄電池営業アシスタントが正常に初期化されました")
        except Exception as e:
            logger.error(f"初期化エラー: {str(e)}")
            raise
    
    def init_vector_store(self):
        """ベクトルストアの初期化"""
        # ベクトルストアが既に存在するか確認
        if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
            logger.info(f"既存のベクトルストアを読み込みます: {DB_DIR}")
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=OPENAI_API_KEY
                )
                self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
                logger.info("ベクトルストアを正常に読み込みました")
            except Exception as e:
                logger.error(f"既存のベクトルストア読み込みエラー: {str(e)}")
                # 読み込みに失敗した場合は新規作成
                self.create_vector_store()
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
                with open(os.path.join(DOCS_DIR, "sample.md"), "w", encoding="utf-8") as f:
                    f.write("# Z-works蓄電池製品情報サンプル\n\nこれは製品情報のサンプルです。実際のコンテンツに置き換えてください。")
                logger.info(f"ドキュメントディレクトリを作成しました: {DOCS_DIR}")
            
            # ドキュメントの読み込み
            logger.info(f"ドキュメントを読み込みます: {DOCS_DIR}")
            documents = []
            
            # Markdownファイルの読み込み
            md_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
            logger.info(f"見つかったMarkdownファイル: {len(md_files)}")
            
            for md_file in md_files:
                try:
                    file_path = os.path.join(DOCS_DIR, md_file)
                    logger.info(f"読み込み: {file_path}")
                    loader = UnstructuredMarkdownLoader(file_path)
                    documents.extend(loader.load())
                except Exception as file_error:
                    logger.error(f"ファイル {md_file} の読み込みエラー: {str(file_error)}")
            
            if len(documents) == 0:
                logger.warning("ドキュメントが見つかりませんでした。基本的な会話のみ可能です。")
                self.vectorstore = None
                return
            
            logger.info(f"読み込んだドキュメント数: {len(documents)}")
            
            # テキスト分割
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            logger.info(f"テキスト分割後のチャンク数: {len(splits)}")
            
            # ベクトルストアの作成
            logger.info("ベクトルストアを作成中...")
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=OPENAI_API_KEY
            )
            
            # 既存のDBディレクトリがあれば削除
            if os.path.exists(DB_DIR):
                import shutil
                shutil.rmtree(DB_DIR)
                logger.info(f"既存のベクトルストアディレクトリを削除しました: {DB_DIR}")
            
            # 新規作成
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
        
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains.retrieval import create_retrieval_chain
        
        # プロンプトテンプレートを作成
        qa_prompt_template = """
{system_prompt}

以下の参考情報を活用してユーザーの質問に答えてください：

参考情報:
{context}

ユーザーの質問: {input}

回答形式を忘れないでください：
1. 読みやすく適切に改行すること
2. 専門用語は平易に言い換えること
3. 重要ポイントは箇条書きで示すこと
4. 最後は次のアクションにつながる提案で締めくくること

あなたの回答:
"""
        qa_prompt = PromptTemplate(
            input_variables=["system_prompt", "context", "input"],
            template=qa_prompt_template,
        )
        
        # ドキュメント結合チェーンを作成
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 検索チェーンを作成
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("RAG検索チェーンの設定が完了しました")
        return retrieval_chain
    
    def setup_conversation_chain(self):
        """一般会話用のチェーンのセットアップ"""
        from langchain_core.runnables import RunnablePassthrough
        
        # プロンプトテンプレートを作成
        conversation_prompt_template = """
{system_prompt}

会話履歴:
{chat_history}

ユーザーの質問: {question}

回答形式を忘れないでください：
1. 読みやすく適切に改行すること
2. 専門用語は平易に言い換えること
3. 重要ポイントは箇条書きで示すこと
4. 最後は次のアクションにつながる提案で締めくくること

あなたの回答:
"""
        conversation_prompt = PromptTemplate(
            input_variables=["system_prompt", "chat_history", "question"],
            template=conversation_prompt_template,
        )
        
        # チェーン構築（新しい方法）
        chain = (
            {"system_prompt": lambda x: SYSTEM_PROMPT, 
             "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
             "question": lambda x: x["question"]}
            | conversation_prompt
            | self.llm
        )
        
        return chain
    
    def get_response(self, user_input):
        """ユーザー入力に対する応答を生成する"""
        try:
            # RAGチェーンが利用可能な場合、まずそれを試す
            if self.qa_chain is not None:
                try:
                    # 検索のみを行って結果を表示（デバッグ用）
                    retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                    retrieved_docs = retriever.invoke(user_input)
                    logger.info(f"検索結果: 文書数 {len(retrieved_docs)}")
                    for i, doc in enumerate(retrieved_docs):
                        logger.info(f"文書{i+1}のページ内容（抜粋）: {doc.page_content[:150]}...")
                    
                    # 新しいRetrievalChainの呼び出し方法
                    result = self.qa_chain.invoke({
                        "system_prompt": SYSTEM_PROMPT,
                        "input": user_input
                    })
                    response = result["answer"]
                    logger.info("RAGベースの回答を生成しました")
                    # メモリに会話を保存
                    self.memory.save_context(
                        {"input": user_input},
                        {"output": response}
                    )
                    return response
                except Exception as e:
                    logger.warning(f"RAG回答生成エラー: {str(e)}。通常の会話チェーンを使用します。")
            
            # RAGチェーンが使えない場合や失敗した場合は通常の会話チェーンを使用
            result = self.conversation_chain.invoke({"question": user_input})
            logger.info("通常の会話チェーンで回答を生成しました")
            
            # メモリに会話を保存
            self.memory.save_context(
                {"input": user_input},
                {"output": result}
            )
            
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
        return coach, "Z-works蓄電池営業アシスタントが起動しました。製品について質問してください。"
    except Exception as e:
        logger.error(f"初期化エラー: {str(e)}")
        return None, f"エラー: 営業アシスタントの初期化に失敗しました。{str(e)}"

def main():
    """簡易的なコンソールインターフェース"""
    print("Z-works蓄電池営業アシスタント")
    print(f"使用モデル: {MODEL_NAME}")
    
    use_mock = False
    if not OPENAI_API_KEY:
        print("警告: OpenAI APIキーが設定されていないため、モックモードで起動します。")
        use_mock = True
    
    coach, init_message = initialize_conversation(use_mock)
    print(init_message)
    
    if coach:
        while True:
            user_input = input("\nあなたの質問 (終了するには 'exit' と入力): ")
            if user_input.lower() in ['exit', 'quit', '終了']:
                print("アシスタントを終了します。お役に立てて光栄です。")
                break
            
            response = coach.get_response(user_input)
            print(f"\n回答: {response}")

if __name__ == "__main__":
    main() 