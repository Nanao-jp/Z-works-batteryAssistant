"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [messages, setMessages] = useState<{ text: string; isUser: boolean }[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // ユーザーメッセージを追加
    const userMessage = { text: input, isUser: true };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // APIリクエスト - バックエンドのURLに合わせて調整してください
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error("APIレスポンスエラー");
      }

      const data = await response.json();
      
      // AIの応答を追加
      setMessages((prev) => [...prev, { text: data.response, isUser: false }]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { text: "エラーが発生しました。後でもう一度お試しください。", isUser: false },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 bg-gray-50">
      <div className="w-full max-w-3xl mx-auto flex flex-col h-screen">
        <header className="py-4 bg-blue-600 text-white rounded-t-lg mb-2 shadow-md">
          <h1 className="text-2xl font-bold text-center">Z-works 蓄電池営業アシスタント</h1>
        </header>

        <div className="flex-1 overflow-auto p-4 bg-white rounded-lg shadow-md mb-4">
          {messages.length === 0 ? (
            <div className="text-center text-gray-500 my-8">
              <p>営業アシスタントAIへようこそ</p>
              <p className="mt-2">Z-works社の蓄電池製品についてお気軽にお尋ねください</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`mb-4 ${
                  msg.isUser ? "text-right" : "text-left"
                }`}
              >
                <div
                  className={`inline-block p-3 rounded-lg ${
                    msg.isUser
                      ? "bg-blue-600 text-white rounded-br-none"
                      : "bg-gray-200 text-gray-800 rounded-bl-none"
                  } whitespace-pre-wrap`}
                >
                  {msg.text}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="text-left mb-4">
              <div className="inline-block p-3 bg-gray-200 text-gray-800 rounded-lg rounded-bl-none">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-gray-500 rounded-full mr-1 animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full mr-1 animate-bounce" style={{ animationDelay: "0.2s" }}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="メッセージを入力..."
            className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            className="bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            disabled={isLoading || !input.trim()}
          >
            送信
          </button>
        </form>
      </div>
    </main>
  );
}
