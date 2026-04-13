from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd  # <-- Pandas import korte hobe

# API Initialize
app = FastAPI(title="Supershop Recommendation API")

print("Loading saved Attention model...")
model = tf.keras.models.load_model("saved_model/attention_wide_deep")


# ==========================================
# 1. Pydantic Models (Apnar Request Format)
# ==========================================
class ItemRequest(BaseModel):
    itemid: int
    quantity: float


class RecRequest(BaseModel):
    customerid: int
    date: str
    items: List[ItemRequest]


class RecResponse(BaseModel):
    itemid: int
    item_name: str
    score: float


# ==========================================
# 2. Dynamic Database/Candidate Items (From CSV)
# ==========================================
print("Loading Item Catalog from Dataset...")
try:
    # Apnar dataset.csv theke shudhu itemId ebong itemName read korchi
    df_items = pd.read_csv("data/dataset.csv", usecols=["itemId", "itemName"])

    # Duplicate item gulo remove kore ekta fresh catalog toiri kora
    df_unique_items = df_items.drop_duplicates(subset=["itemId"])

    # Pandas dataframe theke ekta python dictionary toiri kora {itemId: "itemName"}
    ITEM_CATALOG = dict(zip(df_unique_items.itemId, df_unique_items.itemName))

    # Ebar shob unique item ID guloke amra candidate hishebe nilam
    CANDIDATE_IDS = list(ITEM_CATALOG.keys())
    print(f"✅ Successfully loaded {len(CANDIDATE_IDS)} unique items for recommendation.")

except Exception as e:
    print(f"Error loading dataset: {e}")
    # Jodi file na thake tahole empty rakhbe
    ITEM_CATALOG = {}
    CANDIDATE_IDS = []


# ==========================================
# 3. Context Helper Function
# ==========================================
def extract_context(date_str: str):
    try:
        # Example: '2023-01-01 09:00:00'
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Fallback if string is not in exact format
        dt = datetime.now()

    # Extract timeSlot
    if 5 <= dt.hour < 12:
        slot = "Morning"
    elif 12 <= dt.hour < 17:
        slot = "Afternoon"
    elif 17 <= dt.hour < 20:
        slot = "Evening"
    else:
        slot = "Night"

    # Extract season
    if dt.month in [11, 12, 1, 2]:
        season = "Winter"
    elif dt.month in [3, 4, 5]:
        season = "Summer"
    else:
        season = "Monsoon"

    # In real application, you might check a calendar API/DB for these
    is_holiday = 0.0
    is_festival = 0.0

    return season, slot, is_holiday, is_festival


# ==========================================
# 4. Recommendation Endpoint
# ==========================================
@app.post("/recommend", response_model=List[RecResponse])
def get_recommendations(req: RecRequest):
    # Context ber kora
    season, slot, is_holiday, is_festival = extract_context(req.date)

    batch_size = len(CANDIDATE_IDS)

    # Model er jonno Data Input toiri kora (Must be 2D arrays / reshape(-1, 1) equivalent)
    inputs = {
        "customerId": np.array([req.customerid] * batch_size).reshape(-1, 1),
        "itemId": np.array(CANDIDATE_IDS).reshape(-1, 1),
        "season": np.array([season] * batch_size).reshape(-1, 1),
        "timeSlot": np.array([slot] * batch_size).reshape(-1, 1),
        "isHoliday": np.array([is_holiday] * batch_size).reshape(-1, 1),
        "isFestival": np.array([is_festival] * batch_size).reshape(-1, 1),
        "quantity": np.array([1.0] * batch_size).reshape(-1, 1)  # Default assumption
    }

    # Model theke Prediction/Score neya
    preds = model.predict(inputs, verbose=0)

    # Score format kora
    output = []
    for i, cid in enumerate(CANDIDATE_IDS):
        score = float(preds[i][0])
        output.append({
            "itemid": int(cid),
            "item_name": ITEM_CATALOG.get(cid, "Unknown Item"),
            "score": round(score, 4)
        })

    # ✅ Ebar ei line gulo for loop er baire! (Loop shesh hobar por sort hobe)
    sorted_output = sorted(output, key=lambda x: x['score'], reverse=True)

    return sorted_output[:5]
