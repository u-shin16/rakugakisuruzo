"""らくがきするぞ：画面の配信と「AIに描いてもらう」API。"""
from __future__ import annotations

import base64
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types as genai_types

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
RESEARCH_MODEL = os.getenv("GEMINI_RESEARCH_MODEL", "gemini-2.5-flash")
PROMPT_MAX = 200

# 使いすぎ防止（1人あたり1分・1日の回数と、全体の1日の回数）
PER_MINUTE = int(os.getenv("AI_PER_MINUTE", "5"))
PER_DAY = int(os.getenv("AI_PER_DAY", "40"))
GLOBAL_PER_DAY = int(os.getenv("AI_GLOBAL_PER_DAY", "300"))
_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_global: deque = deque()

# 画像を作るAIへの指示。頼みの中身と画風をそのまま守ってもらう
IMAGE_GUIDE = """次の頼みのとおりに、1枚の絵を描いてください。

頼み：{prompt}
{research}
守ること：
- 頼みに書かれたもの（数・色・形・ポーズ・位置・組み合わせ）を一つも落とさず、そのとおりに描く
- 画風の指定があれば必ず従う（例：リアル・写真のよう→本物の写真のような写実、油絵・水彩・鉛筆・アニメ風など→その画材や画風）
- 有名な絵画・作品の名前が出たら、完成した絵画の複製ではなく、白い画用紙に描いた模写スケッチにする。色の指定がなければ鉛筆の線と陰影だけで描く。「フルカラー」「カラー」「色付き」「色あり」など色の指定があれば、原画の色をしっかり付けて（色鉛筆や絵の具で）フルカラーで描く。画用紙の縁や机は写さない
- 名画以外でも、「白黒」「モノクロ」「フルカラー」など色の指定があれば必ず従う
- 画風の指定がなければ、はっきりした輪郭線と少ない色のベタ塗りで描いた、上手な手描きイラストにする（ペンでなぞって描き直すため）
- 描くもの以外の背景の指定がなければ、背景は白い無地にする
- 文字・枠・署名・透かしは入れない（頼みに文字があるときだけ入れる）
- 頼みに作品名・キャラクター名が書かれていなければ、既存の作品やキャラクターに似せず、オリジナルの絵にする
"""
# 開いている紙に、すでに描いてあるときの指示
EDIT_GUIDE = """渡した画像は、利用者がいま描いている紙です。この紙に対して、次の指示のとおりにしてください。

指示：{prompt}
{research}
守ること：
- 指示されたこと以外は変えない。すでに描かれている線・色・形・位置・大きさはそのまま残す
- 紙の向き・縦横の比率・構図を変えない。切り抜き・拡大・縮小・回転をしない
- 描き足すものは、すでにある絵の画風・線の太さ・色合いに合わせ、すでにある絵との位置関係が自然になるように置く
- 「消して」と言われたものは、その部分を白い紙に戻す
- 何も描かれていない所は白いまま。背景を足すように言われたときだけ背景を描く
- 文字・枠・署名・透かしは入れない（指示に文字があるときだけ入れる）
"""
_DATA_URL_RE = re.compile(r"^data:(image/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=]+)$")


# 描く前に、頼まれたものの見た目をGoogle検索で調べてもらう
RESEARCH_GUIDE = """次の「描いてほしいもの」を、Google検索で調べてください。
そのうえで、絵を描く人がそのまま描けるように、本物の見た目を日本語で短くまとめてください。

描いてほしいもの：{prompt}

まとめ方：
- 1行目：それが何か（正式な名前・種類）を一文で
- 2行目以降：形・色・模様・特徴的な部分・よくある構図を、箇条書きで4〜6個
- 有名な作品なら、構図・人物の向きやポーズ・背景・主な色を具体的に
- 頼みが一般的なもの（例：帽子をかぶったネコ）なら、特定の作品やキャラクターに決めつけず、実物（ネコ・帽子など）の見た目を調べる。作品名やキャラクター名がはっきり書かれているときだけ、その作品を調べる
- 調べても分からないことは書かない。前置きや感想は書かない
- 全体で300文字以内
"""
ASPECTS = {"1:1": 1.0, "2:3": 2 / 3, "3:2": 1.5, "3:4": 0.75, "4:3": 4 / 3, "9:16": 9 / 16, "16:9": 16 / 9}


def nearest_aspect(value) -> str:
    try:
        r = float(value)
    except (TypeError, ValueError):
        return "1:1"
    if not (0.1 < r < 10):
        return "1:1"
    return min(ASPECTS, key=lambda k: abs(math.log(ASPECTS[k] / r)))


def research_block(text) -> str:
    text = str(text or "").strip()[:600]
    if not text:
        return ""
    return f"\n描くものについて調べた本物の見た目（これに合わせて正確に描く）：\n{text}\n"


def client_id() -> str:
    # nginx の後ろでは X-Real-IP / X-Forwarded-For に利用者のIPが入る
    fwd = request.headers.get("X-Forwarded-For", "")
    return (request.headers.get("X-Real-IP") or (fwd.split(",")[0].strip() if fwd else "") or request.remote_addr or "?")


def rate_limited(key: str) -> str | None:
    now = time.time()
    with _lock:
        while _global and now - _global[0] > 86400:
            _global.popleft()
        if len(_global) >= GLOBAL_PER_DAY:
            return "今日はAIの利用が上限に達しました。明日また試してください。"
        q = _hits[key]
        while q and now - q[0] > 86400:
            q.popleft()
        if len(q) >= PER_DAY:
            return "今日のAIの利用回数を使い切りました。明日また試してください。"
        if sum(1 for t in q if now - t < 60) >= PER_MINUTE:
            return "少し待ってから、もう一度試してください。"
        q.append(now)
        _global.append(now)
    return None


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/research")
def api_research():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt or len(prompt) > PROMPT_MAX:
        return jsonify(error="入力を確認してください。"), 400
    msg = rate_limited("r:" + client_id())
    if msg:
        return jsonify(error=msg), 429
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify(error="AIの設定がまだできていません。"), 503
    try:
        client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(api_version="v1beta"))
        response = client.models.generate_content(
            model=RESEARCH_MODEL,
            contents=RESEARCH_GUIDE.format(prompt=prompt),
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                temperature=0.3,
            ),
        )
        summary = (response.text or "").strip()[:600]
        sources, seen = [], set()
        for cand in response.candidates or []:
            gm = getattr(cand, "grounding_metadata", None)
            for ch in (getattr(gm, "grounding_chunks", None) or []):
                web = getattr(ch, "web", None)
                if web and web.uri and web.uri not in seen:
                    seen.add(web.uri)
                    sources.append({"title": (web.title or "")[:80], "uri": web.uri})
    except Exception:
        app.logger.exception("調べものに失敗")
        return jsonify(summary="", sources=[])
    return jsonify(summary=summary, sources=sources[:5])


@app.post("/api/draw")
def api_draw():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify(error="何を描くか入力してください。"), 400
    if len(prompt) > PROMPT_MAX:
        return jsonify(error=f"{PROMPT_MAX}文字以内で入力してください。"), 400

    msg = rate_limited(client_id())
    if msg:
        return jsonify(error=msg), 429

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify(error="AIの設定がまだできていません。"), 503

    try:
        client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(api_version="v1beta"))
        research = research_block(data.get("research"))
        contents = IMAGE_GUIDE.format(prompt=prompt, research=research)
        m = _DATA_URL_RE.match(str(data.get("image") or ""))
        if m:
            contents = [
                genai_types.Part.from_bytes(data=base64.b64decode(m.group(2)), mime_type=m.group(1)),
                EDIT_GUIDE.format(prompt=prompt, research=research),
            ]
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=genai_types.ImageConfig(aspect_ratio=nearest_aspect(data.get("aspect"))),
            ),
        )
        image = None
        for cand in response.candidates or []:
            for part in (cand.content.parts if cand.content else []) or []:
                if part.inline_data and (part.inline_data.mime_type or "").startswith("image/"):
                    image = part.inline_data
                    break
            if image:
                break
    except Exception:
        app.logger.exception("Gemini の呼び出しに失敗")
        return jsonify(error="AIが描けませんでした。少し待ってから試してください。"), 502

    if not image:
        return jsonify(error="AIが描けませんでした。頼み方を変えて試してください。"), 502
    b64 = base64.b64encode(image.data).decode("ascii")
    return jsonify(image=f"data:{image.mime_type};base64,{b64}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8010")), debug=False)
