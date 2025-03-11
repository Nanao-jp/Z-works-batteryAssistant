"""
Ollama APIと連携するFastAPIバックエンド
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel
import os
import sys
import platform

# Windowsでpwdモジュールのパッチを適用
if platform.system() == "Windows":
    try:
        import patch_langchain
        patch_langchain.patch_langchain()
    except ImportError:
        print("警告: patch_langchainモジュールが見つかりません。")

from implementation_patched import AICoach, OLLAMA_API_BASE, MODEL_NAME, MockLLM

app = FastAPI(title="AI開発コーチLLM API")

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のドメインのみ許可するように変更
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルなAICoachインスタンス
coach = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.on_event("startup")
async def startup():
    global coach
    try:
        # APIコーチを初期化
        use_mock = False
        if not OLLAMA_API_BASE or OLLAMA_API_BASE == "http://localhost:11434":
            print("警告: Ollama APIが設定されていないか、ローカルアドレスが指定されています。")
            print("モックモードで起動するか、.envファイルで正しいAPIアドレスを設定してください。")
            use_mock = True
        
        coach = AICoach(use_mock=use_mock)
        print(f"AIコーチを初期化しました。使用モデル: {MODEL_NAME}, API: {OLLAMA_API_BASE}")
        print(f"モックモード: {use_mock}")
    except Exception as e:
        print(f"エラー: AIコーチの初期化に失敗しました: {e}")
        sys.exit(1)

@app.get("/api/info")
async def info():
    return {
        "status": "online",
        "model": MODEL_NAME,
        "api_base": OLLAMA_API_BASE,
        "mock_mode": isinstance(coach.llm, MockLLM) if coach else True
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global coach
    if not coach:
        return ChatResponse(response="エラー: AIコーチが初期化されていません。")
    
    response = coach.get_response(request.message)
    return ChatResponse(response=response)

# 静的ファイル配信の設定
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    # staticディレクトリが存在しない場合は作成
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # サーバー起動
    print("サーバーを起動しています... http://localhost:8000/ でアクセスしてください")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 