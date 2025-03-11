"""
LangChainのWindowsでの互換性問題を解決するためのパッチスクリプト
"""

import sys
import os
import importlib.util

def patch_langchain():
    """
    Windowsでpwdモジュールが見つからない問題を解決するためのパッチを適用します。
    """
    print("LangChainのWindowsパッチを適用しています...")
    
    # 現在のスクリプトのディレクトリを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # windows_compat.pyが存在するか確認
    compat_path = os.path.join(current_dir, "windows_compat.py")
    if not os.path.exists(compat_path):
        print(f"エラー: {compat_path} が見つかりません。")
        return False
    
    # pwdモジュールをインポートしようとしているモジュールのパスを特定
    try:
        # langchain_communityのパスを特定
        langchain_community_spec = importlib.util.find_spec("langchain_community")
        if langchain_community_spec is None:
            print("エラー: langchain_communityモジュールが見つかりません。")
            return False
        
        # ダミーのpwdモジュールをsys.modulesに追加
        sys.path.insert(0, current_dir)
        import windows_compat
        sys.modules["pwd"] = windows_compat.pwd
        print("pwdモジュールのパッチを適用しました。")
        
        return True
    except Exception as e:
        print(f"パッチの適用中にエラーが発生しました: {str(e)}")
        return False

if __name__ == "__main__":
    # スタンドアロンで実行された場合は、パッチを適用して終了
    success = patch_langchain()
    if success:
        print("パッチは正常に適用されました。")
    else:
        print("パッチの適用に失敗しました。") 