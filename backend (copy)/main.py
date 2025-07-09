import os
import json
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import easyocr
import shutil

load_dotenv()

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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)



app = FastAPI(title="Nutrition Scoring API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader globally for better performance
reader = easyocr.Reader(['en'])

class NutritionInfo(BaseModel):
    sugar: float     # g
    sat_fat: float   # g
    sodium: int      # mg
    fiber: float     # g
    protein: float   # g
    calories: float  # kcal

def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using EasyOCR"""
    try:
        results = reader.readtext(image_path)
        
        # Combine all detected text into a single string
        extracted_text = ""
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Only include text with confidence > 50%
                extracted_text += text + " "
        
        return extracted_text.strip()
    except Exception as e:
        raise ValueError(f"Error extracting text from image: {str(e)}")

def parse_nutrition_json(json_str: str) -> dict:
    clean = json_str.strip("```json").strip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def score_food(label_dict):
    max_sugar = 30.0     # g
    max_sat_fat = 15.0   # g
    max_sodium = 1000.0  # mg
    max_cal = 500.0      # kcal
    max_fiber = 10.0     # g
    max_prot = 20.0      # g
    
    # Get values and cap them
    sugar = min(label_dict.get('sugar', 0), max_sugar)
    sat_fat = min(label_dict.get('sat_fat', 0), max_sat_fat)
    sodium = min(label_dict.get('sodium', 0), max_sodium)
    cal = min(label_dict.get('calories', 0), max_cal)
    fiber = min(label_dict.get('fiber', 0), max_fiber)
    prot = min(label_dict.get('protein', 0), max_prot)
    
    # Normalize each to 0-1
    n_sugar = sugar / max_sugar
    n_satfat = sat_fat / max_sat_fat
    n_sodium = sodium / max_sodium
    n_cal = cal / max_cal
    n_fiber = fiber / max_fiber
    n_prot = prot / max_prot
    
    # Weights: negative for "bad" nutrients, positive for "good" ones
    weights = {
        'sugar': -0.25,
        'satfat': -0.20,
        'sodium': -0.20,
        'cal': -0.15,
        'fiber': +0.10,
        'protein': +0.10,
    }
    
    # Compute weighted sum
    score_raw = (
        weights['sugar'] * n_sugar +
        weights['satfat'] * n_satfat +
        weights['sodium'] * n_sodium +
        weights['cal'] * n_cal +
        weights['fiber'] * n_fiber +
        weights['protein'] * n_prot
    )
    
    # Convert to 0-100 scale
    score = (score_raw + 1) * 50
    return max(0, min(100, score))

def get_class(score: float) -> str:
    if score < 20:
        return "Very Harmful"
    elif score < 40:
        return "Harmful"
    else:
        return "Safe"

def call_llm(system_prompt: str, user_content: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return resp.choices[0].message.content

@app.post("/analyze", summary="Analyze nutrition from uploaded image and return classification")
async def analyze_nutrition(file: UploadFile = File(...)):
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        with open(temp_file.name, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        extracted_text = extract_text_from_image(temp_file.name)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the image")
        
        # Process the extracted text through LLM to get structured nutrition data
        formatted_json = call_llm(SYSTEM_PROMPT, extracted_text)
        nutrition_data = parse_nutrition_json(formatted_json)
        
        # Calculate score and classification
        score = score_food(nutrition_data)
        classification = get_class(score)
        
        # Generate appropriate response based on classification
        if classification == "Safe":
            message = "Safe product"
            better_product = "This is a better product"
        else:
            advice_prompt = f"""
            You are a nutrition coach. Given this nutrition data: {json.dumps(nutrition_data)}
            and classification: {classification}

            Respond with a JSON object containing:
            - "message": A brief 1-2 sentence explanation of why this product is {classification.lower()}
            - "better_product": Name of a specific healthier alternative product

            Only return the JSON, no extra text.
            """
            
            llm_response = call_llm(advice_prompt, "")
            try:
                advice_data = json.loads(llm_response.strip("```json").strip("```").strip())
                message = advice_data.get("message", f"This product is {classification.lower()} due to nutritional content")
                better_product = advice_data.get("better_product", "Consider a healthier alternative")
            except:
                message = f"This product is {classification.lower()} due to high levels of harmful nutrients"
                better_product = "Consider a product with lower sugar, sodium, and saturated fat"

        return {
            "class": classification,
            "message": message,
            "better_product": better_product,
            "extracted_text": extracted_text,  # Include extracted text for debugging
            "nutrition_data": nutrition_data,   # Include parsed nutrition data
            "score": round(score, 2)           # Include numerical score
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
class NutritionRequest(BaseModel):
    nutrition_text: str