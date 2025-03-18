import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://z-works-batteryassistant-production.up.railway.app';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_URL}/chat`, { 
        message: input,
        history: messages.map(m => ({ role: m.role, content: m.content }))
      });
      
      if (response.data && response.data.response) {
        setMessages(prev => [...prev, { role: 'assistant', content: response.data.response }]);
      }
    } catch (error) {
      console.error('Error communicating with API:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'すみません、エラーが発生しました。もう一度お試しください。' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>蓄電池営業アシスタント</h1>
      </header>
      
      <main>
        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <p>こんにちは！蓄電池に関するご質問がありましたら、お気軽にお尋ねください。</p>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-content">{message.content}</div>
                </div>
              ))}
              {isLoading && (
                <div className="message assistant">
                  <div className="message-content">
                    <div className="loading">考え中...</div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="メッセージを入力してください..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            送信
          </button>
        </form>
      </main>
      
      <style jsx>{`
        .container {
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
        }
        
        header {
          margin-bottom: 20px;
          padding-bottom: 10px;
          border-bottom: 1px solid #eaeaea;
        }
        
        h1 {
          color: #333;
          font-size: 1.8rem;
        }
        
        .chat-container {
          border: 1px solid #eaeaea;
          border-radius: 8px;
          height: 500px;
          overflow-y: auto;
          padding: 15px;
          background-color: #f9f9f9;
          margin-bottom: 20px;
        }
        
        .welcome-message {
          text-align: center;
          color: #666;
          margin-top: 200px;
        }
        
        .messages {
          display: flex;
          flex-direction: column;
        }
        
        .message {
          max-width: 80%;
          margin-bottom: 15px;
          padding: 10px 15px;
          border-radius: 8px;
          line-height: 1.5;
        }
        
        .user {
          align-self: flex-end;
          background-color: #007bff;
          color: white;
        }
        
        .assistant {
          align-self: flex-start;
          background-color: #e9ecef;
          color: #333;
        }
        
        .loading {
          display: inline-block;
          font-style: italic;
        }
        
        .input-form {
          display: flex;
          gap: 10px;
        }
        
        input {
          flex: 1;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 16px;
        }
        
        button {
          padding: 10px 20px;
          background-color: #007bff;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 16px;
        }
        
        button:disabled {
          background-color: #cccccc;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
} 