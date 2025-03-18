import { NextResponse } from 'next/server';

// バックエンドのURLを設定（実際のデプロイ環境のURLに変更してください）
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const message = body.message;

    // バックエンドAPIにリクエストを転送
    const response = await fetch(`${API_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`APIエラー: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json(
      { response: 'エラーが発生しました。後でもう一度お試しください。' },
      { status: 500 }
    );
  }
} 