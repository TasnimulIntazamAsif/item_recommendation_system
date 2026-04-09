from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from datetime import datetime
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("saved_model/attention_wide_deep")


# Request/Response Models
class ItemRequest(BaseModel):
    itemid: int
    quantity: int


class RecRequest(BaseModel):
    customerid: int
    date: str
    items: List[ItemRequest]


class RecResponse(BaseModel):
    itemid: int
    item_name: str
    score: float


# Dummy Item Lookup (Real life e database theke ashbe)
ITEM_NAMES = {
    2165: "Pran mango juice", 2166: "Pran orange juice",
    2167: "Pran lemon juice", 2168: "Pran pineapple juice",
    2169: "Pran strawberry juice", 4969: "Sample Biscuit"
}


def get_context(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    slot = "Morning" if 5 <= dt.hour < 12 else "Afternoon" if 12 <= dt.hour < 17 else "Evening" if 17 <= dt.hour < 21 else "Night"
    season = "Winter" if dt.month in [11, 12, 1, 2] else "Summer" if dt.month in [3, 4, 5] else "Monsoon"
    return season, slot


@app.post("/recommend", response_model=List[RecResponse])
def recommend(req: RecRequest):
    season, slot = get_context(req.date)

    # Candidates for recommendation (In real life, fetch from DB)
    candidate_ids = [2165, 2166, 2167, 2168, 2169]
    n = len(candidate_ids)

    # Prepare model inputs
    inputs = [
        np.array([req.customerid] * n),
        np.array(candidate_ids),
        np.array([season] * n),
        np.array([slot] * n),
        np.array([0.0] * n),  # is_holiday
        np.array([0.0] * n),  # is_festival
        np.array([1.0] * n)  # default quantity
    ]

    preds = model.predict(inputs, verbose=0)

    output = []
    for i, cid in enumerate(candidate_ids):
        output.append({
            "itemid": cid,
            "item_name": ITEM_NAMES.get(cid, "Unknown Item"),
            "score": round(float(preds[i][0]), 4)
        })

    return sorted(output, key=lambda x: x['score'], reverse=True)