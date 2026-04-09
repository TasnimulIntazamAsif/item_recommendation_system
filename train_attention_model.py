import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Concatenate, Dense,
    StringLookup, IntegerLookup, CategoryEncoding,
    HashedCrossing, Normalization, Multiply
)
from tensorflow.keras.models import Model

# ==========================================
# 1. Load Real Data & Generate Negative Samples
# ==========================================
print("Loading data from data/dataset.csv...")
# Apnar file e header thik eivabe thakte hobe: customerId, itemId, itemName, quantity, orderDate, isHoliday, isFestival, season, timeSlot
df = pd.read_csv("data/dataset.csv")
df = df.dropna()

# Positive Data (Bortomane apnar CSV te ja ache shob e positive)
df['label'] = 1

# 🟢 Automated Negative Sampling 🟢
print("Generating Negative Samples for better training...")
df_neg = df.copy()
df_neg['itemId'] = np.random.permutation(df_neg['itemId'].values)
df_neg['label'] = 0

# ✅ FIX: Ekhane 0.0 er bodole 1.0 din jate model quantity diye cheat na korte pare!
df_neg['quantity'] = 1.0

final_df = pd.concat([df, df_neg]).sample(frac=1).reset_index(drop=True)

# Positive o Negative data eksathe mix kora
print(f"Total rows after negative sampling: {len(final_df)}")

# ==========================================
# 2. Train-Test Split (80% Train, 20% Test)
# ==========================================
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

# ==========================================
# 3. Context-Aware Attention Model Architecture
# ==========================================
# Inputs (Names strictly matched with your column names)
cust_id_in = Input(shape=(1,), name="customerId", dtype=tf.int64)
item_id_in = Input(shape=(1,), name="itemId", dtype=tf.int64)
season_in = Input(shape=(1,), name="season", dtype=tf.string)
timeslot_in = Input(shape=(1,), name="timeSlot", dtype=tf.string)
holiday_in = Input(shape=(1,), name="isHoliday", dtype=tf.float32)
festival_in = Input(shape=(1,), name="isFestival", dtype=tf.float32)
qty_in = Input(shape=(1,), name="quantity", dtype=tf.float32)

# Normalization & Encoding
qty_norm = Normalization()
qty_norm.adapt(train_df['quantity'].values.reshape(-1, 1))

season_lookup = StringLookup(vocabulary=list(train_df['season'].unique()))
season_enc = CategoryEncoding(num_tokens=season_lookup.vocabulary_size(), output_mode="one_hot")(season_lookup(season_in))

ts_lookup = StringLookup(vocabulary=list(train_df['timeSlot'].unique()))
ts_enc = CategoryEncoding(num_tokens=ts_lookup.vocabulary_size(), output_mode="one_hot")(ts_lookup(timeslot_in))

context = Concatenate()([season_enc, ts_enc, holiday_in, festival_in])

# Embeddings
cust_lookup = IntegerLookup(vocabulary=list(train_df['customerId'].unique()))
cust_emb = Flatten()(Embedding(cust_lookup.vocabulary_size(), 32)(cust_lookup(cust_id_in)))

item_lookup = IntegerLookup(vocabulary=list(train_df['itemId'].unique()))
item_emb = Flatten()(Embedding(item_lookup.vocabulary_size(), 32)(item_lookup(item_id_in)))

# Attention Magic
combined_emb = Concatenate()([cust_emb, item_emb])
att_gate = Dense(64, activation="sigmoid", name="attention_gate")(context)
weighted_emb = Multiply()([combined_emb, att_gate])

# Deep Path
deep = Dense(64, activation="relu")(Concatenate()([weighted_emb, context, qty_norm(qty_in)]))
deep = Dense(32, activation="relu")(deep)

# Wide Path (Crossing your specific columns)
wide = Dense(1)(Concatenate()([
    CategoryEncoding(num_tokens=10000, output_mode="one_hot")(HashedCrossing(10000)([cust_id_in, item_id_in])),
    CategoryEncoding(num_tokens=5000, output_mode="one_hot")(HashedCrossing(5000)([timeslot_in, item_id_in]))
]))

# Final Output
out = Dense(1, activation="sigmoid")(Concatenate()([deep, wide]))
model = Model(inputs=[cust_id_in, item_id_in, season_in, timeslot_in, holiday_in, festival_in, qty_in], outputs=out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. Training Model
# ==========================================
from tensorflow.keras.callbacks import EarlyStopping

def get_input_dict(dataset):
    return [
        dataset["customerId"].values.reshape(-1, 1),
        dataset["itemId"].values.reshape(-1, 1),
        dataset["season"].values.reshape(-1, 1),
        dataset["timeSlot"].values.reshape(-1, 1),
        dataset["isHoliday"].values.reshape(-1, 1),
        dataset["isFestival"].values.reshape(-1, 1),
        dataset["quantity"].values.reshape(-1, 1)
    ]

print("\n🚀 Model Training Started (Advanced Setup)...")

# ✅ FIX: Early Stopping add kora holo jate overfit na hoy
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5, # Jodi 5 epoch dhore accuracy na bare, tahole training theme jabe
    restore_best_weights=True
)

# ✅ FIX: Epoch 50 ebong batch_size 128 kora holo
history = model.fit(
    get_input_dict(train_df),
    train_df['label'].values,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop]
)

# ==========================================
# 5. Model Evaluation (Metrics)
# ==========================================
print("\n📊 Calculating Real Metrics on Test Data...")
y_pred_prob = model.predict(get_input_dict(test_df))
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_df['label'].values

print("\n--- Context-Aware Model Performance ---")
print(classification_report(y_true, y_pred, target_names=["Not Recommended (0)", "Recommended (1)"]))
print(f"Overall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# ==========================================
# 6. Save Model
# ==========================================
model.save("saved_model/attention_wide_deep")
print("\n✅ Model successfully saved to 'saved_model/attention_wide_deep'")