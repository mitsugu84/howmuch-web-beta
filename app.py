import base64
import json
import os
import re
from io import BytesIO
from urllib.parse import quote_plus

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_IMAGES = int(os.getenv("MAX_IMAGES", "10"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def image_to_data_url(file_storage):
    """Convert uploaded image to compressed JPEG data URL."""
    try:
        img = Image.open(file_storage.stream)
        img = img.convert("RGB")
        img.thumbnail((1400, 1400))
    except UnidentifiedImageError:
        raise ValueError("画像ファイルとして読み込めませんでした。JPG/PNG/HEIC変換済み画像を使ってください。")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def extract_json(text):
    """Extract JSON object from model output."""
    if not text:
        raise ValueError("AIの返答が空でした。")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def safe_int(value, default=0):
    try:
        if isinstance(value, str):
            value = re.sub(r"[^0-9]", "", value)
        return int(value)
    except Exception:
        return default


def build_links(keyword):
    q = quote_plus(keyword or "")
    return {
        "mercari": f"https://jp.mercari.com/search?keyword={q}",
        "yahoo_auction": f"https://auctions.yahoo.co.jp/search/search?p={q}",
        "jimoty": f"https://jmty.jp/all/sale-kw-{q}",
    }


def analyze_one_image(data_url):
    prompt = """
あなたは中古品相場を調査するAIエージェントです。

画像の商品を特定し、メルカリ・ヤフオク・ジモティの現在の公開情報をWeb検索で確認し、
価格帯・出品傾向・売れそう度を日本語で要約してください。

重要ルール:
- 巡回・定期収集・DB保存ではなく、この1回限りの調査として扱う
- 価格は必ず「参考価格」として出す
- 不確実な場合は不確実と書く
- 画像だけで分からない付属品・動作状態は断定しない
- メルカリ手数料は10%で概算する
- 送料は商品の大きさから概算する
- ジャンク/箱なし/本体のみ/付属品ありの可能性を分かる範囲で推定する
- 返答は必ずJSONだけ。説明文やMarkdownは禁止。

JSON形式:
{
  "item_name": "商品名",
  "condition_guess": "状態推定",
  "confidence": "高/中/低",
  "mercari": {
    "price_range": "〇〇〜〇〇円",
    "listing_hint": "出品数や傾向。分からない場合は不明",
    "comment": "コメント"
  },
  "yahoo_auction": {
    "price_range": "〇〇〜〇〇円",
    "listing_hint": "出品数や傾向。分からない場合は不明",
    "comment": "コメント"
  },
  "jimoty": {
    "price_range": "〇〇〜〇〇円",
    "listing_hint": "出品数や傾向。分からない場合は不明",
    "comment": "コメント"
  },
  "shop_buyback": {
    "price_range": "〇〇〜〇〇円",
    "comment": "店頭買取の参考予想"
  },
  "shipping_estimate": 850,
  "fee_estimate": 2200,
  "expected_profit": 18950,
  "recommendation": "おすすめの売り方",
  "search_keywords": "検索キーワード",
  "caution": "注意点"
}
"""

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": "web_search"}],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    )

    raw_text = response.output_text
    data = extract_json(raw_text)
    keyword = data.get("search_keywords") or data.get("item_name") or ""
    data["links"] = build_links(keyword)
    data["shipping_estimate"] = safe_int(data.get("shipping_estimate"), 0)
    data["fee_estimate"] = safe_int(data.get("fee_estimate"), 0)
    data["expected_profit"] = safe_int(data.get("expected_profit"), 0)
    return data


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, errors=None, max_images=MAX_IMAGES)


@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]

    errors = []
    results = []

    if not files:
        errors.append("画像を1枚以上選択してください。")
        return render_template("index.html", results=None, errors=errors, max_images=MAX_IMAGES)

    if len(files) > MAX_IMAGES:
        errors.append(f"一度に解析できる画像は最大{MAX_IMAGES}枚です。先頭{MAX_IMAGES}枚だけ解析します。")
        files = files[:MAX_IMAGES]

    for idx, file in enumerate(files, start=1):
        try:
            data_url = image_to_data_url(file)
            result = analyze_one_image(data_url)
            result["index"] = idx
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            errors.append(f"{file.filename or idx}：解析に失敗しました。{e}")

    return render_template("index.html", results=results, errors=errors, max_images=MAX_IMAGES)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
