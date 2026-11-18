# らくがきするぞ

紙のサイズを選んで描けるホワイトボードアプリ。AIに頼むと、ネットで調べた見た目をもとに、アプリのペンで絵を描く。

- 画面：`index.html`（1ファイル）
- サーバー：`app.py`（Flask。画面の配信と、AIへの依頼 `/api/research` `/api/draw`）
- 設定：`.env`（`.env.example` を参照。APIキーはリポジトリに入れない）

## 手元で動かす

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY を入れる
venv/bin/python app.py # http://127.0.0.1:8010/
```
