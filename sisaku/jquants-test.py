import os
import json
import requests
from dotenv import load_dotenv, find_dotenv

# requestsのセッションをグローバルに作成
session = requests.Session()

def get_jquants_token():
    """J-QuantsのAPIトークンを取得する"""
    # 環境変数ファイルから認証情報を読み込み
    # 親ディレクトリまで遡って 'J-Quants.env' を探して読み込む
    load_dotenv(find_dotenv('J-Quants.env'))
    JQUANTS_MAIL = os.getenv("JQUANTS_EMAIL")
    JQUANTS_PASS = os.getenv("JQUANTS_PASSWORD")
    # 認証情報の確認
    if not JQUANTS_MAIL or not JQUANTS_PASS:
        print("❌ 環境変数ファイル 'J-Quants.env' に認証情報が設定されていません")
        print("   以下の内容でファイルを作成してください：")
        print("   JQUANTS_EMAIL=your_email@example.com")
        print("   JQUANTS_PASSWORD=your_password")
        return None, None

    try:
        # 1. ユーザー認証（リフレッシュトークン取得）
        print("   📤 ユーザー認証中...")
        auth_data = {"mailaddress": JQUANTS_MAIL, "password": JQUANTS_PASS}
        auth_response = session.post(
            "https://api.jquants.com/v1/token/auth_user",
            json=auth_data,  # data=json.dumps(auth_data) から変更
            timeout=30
        )
        auth_response.raise_for_status()
        refresh_token = auth_response.json()["refreshToken"]
        print("   ✅ リフレッシュトークン取得完了")

        # 2. IDトークン取得
        print("   🔄 IDトークン取得中...")
        token_response = session.post(
            "https://api.jquants.com/v1/token/auth_refresh",
            params={"refreshtoken": refresh_token},
            timeout=30
        )
        token_response.raise_for_status()
        id_token = token_response.json()["idToken"]
        print("   ✅ IDトークン取得完了")

        # 3. ヘッダー設定
        headers = {"Authorization": f"Bearer {id_token}"}
        print("🎉 J-QuantsのIDトークン取得完了！")
        return headers, session
    except requests.exceptions.RequestException as e:
        print(f"❌ 認証中にエラーが発生しました: {e}")
        if 'auth_response' in locals() and auth_response is not None:
            print(f"   レスポンス: {auth_response.text}")
        return None, None

def test_api_request(headers, session):
    """APIリクエストのテスト"""
    if not headers or not session:
        print("❌ 認証が完了していません")
        return False

    print("\n🧪 APIリクエストのテスト中...")
    try:
        # 簡単なテスト：上場企業一覧の取得
        print("   📊 上場企業一覧を取得中...")
        url = "https://api.jquants.com/v1/listed/info"
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 200番台以外のステータスコードで例外を発生

        data = response.json()
        companies = data.get('info', [])
        print(f"   ✅ 取得成功: {len(companies)}社の情報")

        # 最初の3社の情報を表示
        print("\n📋 取得した企業情報（最初の3社）:")
        for i, company in enumerate(companies[:3]):
            print(f"   {i+1}. {company.get('CompanyName', 'N/A')} ({company.get('Code', 'N/A')})")

        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ APIリクエスト中にエラーが発生しました: {e}")
        if 'response' in locals() and response is not None:
            print(f"   ステータスコード: {response.status_code}")
            print(f"   レスポンス: {response.text}")
        return False

def main():
    """メイン実行"""
    print("🚀 J-Quants API 認証テスト開始")
    print("=" * 50)

    # 1. 認証
    headers, session = get_jquants_token()
    if not headers:
        print("\n❌ 認証に失敗しました")
        print("   以下を確認してください：")
        print("   - J-Quants.env ファイルが存在するか")
        print("   - メールアドレスとパスワードが正しいか")
        print("   - J-Quantsアカウントが有効か")
        return

    # 2. APIテスト
    success = test_api_request(headers, session)

    # 3. 結果表示
    print("\n" + "=" * 50)
    if success:
        print("🎉 認証テスト完了！J-Quants APIが正常に動作しています")
        print("\n📝 次のステップ:")
        print("   1. 財務データの取得")
        print("   2. 株価データの取得")
        print("   3. その他の金融データの取得")
    else:
        print("⚠️  認証は成功しましたが、APIリクエストでエラーが発生しました")
        print("   ネットワーク環境やAPI制限を確認してください")

if __name__ == "__main__":
    main()
