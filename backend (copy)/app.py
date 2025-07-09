import streamlit as st
import easyocr
import tempfile
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
# Constants
SYSTEM_PROMPT = """
You will receive raw nutrition-label text (e.g. "Total Sugars: 15.1 g; Saturated Fat: 0.0 g; Sodium: 3.3 mg; Fiber: —; Protein: 0.0 g; Energy: 60.8 kcal").

Extract and return the following fields in **strict JSON format** (no explanations, no extra text):

- sugar    : float (grams of sugar)
- sat_fat  : float (grams of saturated fat)
- sodium   : int   (milligrams of sodium)
- fiber    : float (grams of dietary fiber; use 0.0 if missing or marked as —)
- protein  : float (grams of protein)
- calories : float (kcal)

Return the result exactly like this:

{
  "sugar": 12.0,
  "sat_fat": 4.5,
  "sodium": 250,
  "fiber": 3.0,
  "protein": 7.0,
  "calories": 180.0
}
"""

# Set up OpenRouter / OpenAI client
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY
)

reader = easyocr.Reader(['en'], gpu=False)

# Nutrition scoring
def score_food(label_dict):
    max_sugar, max_sat_fat, max_sodium = 30.0, 15.0, 1000.0
    max_cal, max_fiber, max_prot = 500.0, 10.0, 20.0
    
    sugar = min(label_dict.get('sugar', 0), max_sugar)
    sat_fat = min(label_dict.get('sat_fat', 0), max_sat_fat)
    sodium = min(label_dict.get('sodium', 0), max_sodium)
    cal = min(label_dict.get('calories', 0), max_cal)
    fiber = min(label_dict.get('fiber', 0), max_fiber)
    prot = min(label_dict.get('protein', 0), max_prot)
    
    n_sugar = sugar / max_sugar
    n_satfat = sat_fat / max_sat_fat
    n_sodium = sodium / max_sodium
    n_cal = cal / max_cal
    n_fiber = fiber / max_fiber
    n_prot = prot / max_prot
    
    weights = {'sugar': -0.25, 'satfat': -0.20, 'sodium': -0.20, 'cal': -0.15, 'fiber': +0.10, 'protein': +0.10}
    
    score_raw = (
        weights['sugar'] * n_sugar +
        weights['satfat'] * n_satfat +
        weights['sodium'] * n_sodium +
        weights['cal'] * n_cal +
        weights['fiber'] * n_fiber +
        weights['protein'] * n_prot
    )
    return round(max(0, min(100, (score_raw + 1) * 50)), 2)

def get_class(score):
    if score < 20:
        return "Very Harmful"
    elif score < 40:
        return "Harmful"
    else:
        return "Safe"

def call_llm(system_prompt, user_content):
    resp = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content

def parse_nutrition_json(json_str: str) -> dict:
    clean = json_str.strip("```json").strip("```").strip()
    return json.loads(clean)

def extract_text_from_image(image_path):
    results = reader.readtext(image_path)
    return " ".join([text for _, text, prob in results if prob > 0.5])

# Streamlit UI
st.set_page_config(page_title="Nutrition Analyzer", layout="centered")
st.title("🥗 Nutrition Label Analyzer")
st.write("Upload a nutrition label image or paste the text manually to get health insights.")

tab1, tab2 = st.tabs(["📷 Upload Image", "✍️ Paste Text"])

with tab1:
    file = st.file_uploader("Upload a nutrition label image", type=["jpg", "jpeg", "png"])
    if file:
        st.image(file, use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        with st.spinner("Extracting text and analyzing..."):
            try:
                text = extract_text_from_image(tmp_path)
                st.code(text, language="text")

                formatted_json = call_llm(SYSTEM_PROMPT, text)
                nutrition = parse_nutrition_json(formatted_json)
                score = score_food(nutrition)
                label = get_class(score)

                if label == "Safe":
                    message = "This product appears to be safe."
                    better = "This is already a better product."
                else:
                    advice_prompt = f"""
                    You are a nutrition coach. Given this nutrition data: {json.dumps(nutrition)}
                    and classification: {label}

                    Respond with a JSON object containing:
                    - "message": A brief 1-2 sentence explanation of why this product is {label.lower()}
                    - "better_product": Name of a specific healthier alternative product

                    Only return the JSON, no extra text.
                    """
                    advice_json = call_llm(advice_prompt, "")
                    try:
                        advice = json.loads(advice_json.strip("```json").strip("```").strip())
                        message = advice.get("message", "")
                        better = advice.get("better_product", "")
                    except:
                        message = "This product may contain high sugar, fat or sodium."
                        better = "Consider lower-sugar alternatives."

                st.success(message)
                st.metric("Score", f"{score} / 100")
                st.write("**Classification:**", label)
                st.write("**Suggested Alternative:**", better)
                st.json(nutrition)

            except Exception as e:
                st.error(f"Failed to process image: {e}")

with tab2:
    text_input = st.text_area("Paste nutrition label text here:")
    if st.button("Analyze Text"):
        with st.spinner("Analyzing..."):
            try:
                formatted_json = call_llm(SYSTEM_PROMPT, text_input)
                nutrition = parse_nutrition_json(formatted_json)
                score = score_food(nutrition)
                label = get_class(score)

                if label == "Safe":
                    message = "This product appears to be safe."
                    better = "This is already a better product."
                else:
                    advice_prompt = f"""
                    You are a nutrition coach. Given this nutrition data: {json.dumps(nutrition)}
                    and classification: {label}

                    Respond with a JSON object containing:
                    - "message": A brief 1-2 sentence explanation of why this product is {label.lower()}
                    - "better_product": Name of a specific healthier alternative product

                    Only return the JSON, no extra text.
                    """
                    advice_json = call_llm(advice_prompt, "")
                    try:
                        advice = json.loads(advice_json.strip("```json").strip("```").strip())
                        message = advice.get("message", "")
                        better = advice.get("better_product", "")
                    except:
                        message = "This product may contain high sugar, fat or sodium."
                        better = "Consider lower-sugar alternatives."

                st.success(message)
                st.metric("Score", f"{score} / 100")
                st.write("**Classification:**", label)
                st.write("**Suggested Alternative:**", better)
                st.json(nutrition)

            except Exception as e:
                st.error(f"Failed to analyze text: {e}")
