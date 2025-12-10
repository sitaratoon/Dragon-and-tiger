"""
dragon_tiger_ml_bot.py
Single-file Telegram bot + ML predictor for Dragon & Tiger.
Provides data collection, ML training, prediction, and backtesting.

CONFIG: edit the top section with your BOT_TOKEN / API_ID / API_HASH.
"""

import os
import csv
import math
import joblib
from datetime import datetime
from collections import Counter, defaultdict, deque
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyrogram import Client, filters
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ------------------ CONFIG (EDIT) ------------------
API_ID = 1234567                     # <- your API ID
API_HASH = "your_api_hash"           # <- your API HASH
BOT_TOKEN = "123456:ABC-DEF..."      # <- your Bot token

USE_MONGO = False                    # optional: set True to use MongoDB (see MONGO_URI)
MONGO_URI = "mongodb+srv://user:pass@cluster/db"
ADMIN_ID = 0                         # set your Telegram user id to restrict /clear and /train

LOCAL_HISTORY = "history_ml.csv"     # where outcomes saved (ts, outcome)

MODEL_FILE = "model.joblib"          # trained model file
SCALER_FILE = "scaler.joblib"        # optional scaler (inside pipeline not needed)
LAGS = 10                            # number of lag features to use (tunable)
# --------------------------------------------------

# ------------------ STORAGE ------------------
class Storage:
    def __init__(self):
        self.local_file = LOCAL_HISTORY
        try:
            open(self.local_file, "x").close()
        except FileExistsError:
            pass

    def add(self, outcome: str):
        assert outcome in ("D", "T")
        ts = datetime.utcnow().isoformat()
        with open(self.local_file, "a", newline="") as f:
            csv.writer(f).writerow([ts, outcome])

    def bulk_add(self, seq: List[str]):
        for s in seq:
            if s in ("D","T"):
                self.add(s)

    def get_all(self) -> List[Tuple[str,str]]:
        rows = []
        with open(self.local_file, newline="") as f:
            reader = csv.reader(f)
            for r in reader:
                if len(r) >= 2:
                    rows.append((r[0], r[1]))
        return rows

    def clear(self):
        open(self.local_file, "w").close()

storage = Storage()

# ------------------ FEATURE ENGINEERING ------------------
def sequence_to_dataframe(history: List[Tuple[str,str]], lags: int = LAGS) -> pd.DataFrame:
    """
    Build supervised dataset:
    Each row predicts outcome at time t using previous `lags` outcomes and a few engineered stats.
    Outcome encoded as 1 for Dragon (D), 0 for Tiger (T).
    """
    outcomes = [o for (_, o) in history]
    n = len(outcomes)
    if n <= lags:
        return pd.DataFrame()  # not enough data

    # numeric encoding
    y = []
    X_rows = []

    # helper functions
    def streak_at(seq, idx):
        # streak length ending at idx (including idx)
        v = seq[idx]
        s = 1
        j = idx - 1
        while j >= 0 and seq[j] == v:
            s += 1; j -= 1
        return s

    for t in range(lags, n):
        # target is outcomes[t]
        target = 1 if outcomes[t] == "D" else 0
        y.append(target)

        row = {}
        # lag features: t-1 .. t-lags
        for L in range(1, lags+1):
            val = outcomes[t-L]
            row[f"lag_{L}"] = 1 if val == "D" else 0

        # recent counts
        window = 20
        start = max(0, t-window)
        recent = outcomes[start:t]
        c = Counter(recent)
        row["recent_D_count"] = c["D"]
        row["recent_T_count"] = c["T"]
        row["recent_D_frac"] = c["D"] / max(1, len(recent))

        # streak features
        row["last_streak_len"] = streak_at(outcomes, t-1)
        row["last_was_D"] = 1 if outcomes[t-1] == "D" else 0

        # transitions: how many times last->D / last->T occurred historically
        trans_D = trans_T = 0
        last = outcomes[t-1]
        for a,b in zip(outcomes[:t-1], outcomes[1:t]):
            if a == last and b == "D": trans_D += 1
            if a == last and b == "T": trans_T += 1
        tot = trans_D + trans_T
        row["trans_last_to_D_frac"] = trans_D / tot if tot>0 else 0.5

        # positional: round index normalized
        row["pos_index"] = t / n

        X_rows.append(row)

    dfX = pd.DataFrame(X_rows)
    dfy = pd.Series(y, name="y")
    df = pd.concat([dfX, dfy], axis=1)
    return df

# ------------------ MODEL TRAIN / BACKTEST ------------------
def train_model(history: List[Tuple[str,str]]):
    df = sequence_to_dataframe(history, lags=LAGS)
    if df.empty or df.shape[0] < 50:
        return None, "Not enough data to train (need >=50 supervised rows)."

    X = df.drop(columns=["y"])
    y = df["y"].values

    # simple pipeline with scaler + gradient boosting
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    pipeline.fit(X, y)

    # save pipeline
    joblib.dump(pipeline, MODEL_FILE)
    return pipeline, {"cv_mean": float(scores.mean()), "cv_std": float(scores.std()), "n_samples": len(y)}

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

def backtest(history: List[Tuple[str,str]], model, stake=1.0):
    """
    Simulate backtest using model predictions on historical data (walk-forward style).
    We will do a simple walk-forward with expanding window:
    - Start training on first `init_train` supervised rows, then predict next, then add that outcome to train, repeat.
    For speed we can reuse the same pipeline parameters but re-fit each step on available data.
    """
    df_all = sequence_to_dataframe(history, lags=LAGS)
    if df_all.empty or df_all.shape[0] < 50:
        return {"error": "Not enough data for backtest"}

    X_all = df_all.drop(columns=["y"])
    y_all = df_all["y"].values
    n = len(y_all)
    init_train = int(n * 0.3)
    correct = 0
    total = 0
    profit = 0.0
    bankroll = 0.0

    for i in range(init_train, n):
        # train on [0:i)
        X_train = X_all.iloc[:i]
        y_train = y_all[:i]
        X_test = X_all.iloc[i:i+1]
        y_test = y_all[i]

        # train small model
        mdl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        # scaling
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        mdl.fit(Xtr, y_train)
        p = mdl.predict_proba(scaler.transform(X_test))[0]
        # p[1] is probability of Dragon (encoded 1)
        pred = 1 if p[1] > 0.5 else 0

        # betting simulation: if predicted, bet on predicted side with stake
        # payout 1:1 for simplicity (real games may differ or have multipliers)
        total += 1
        if pred == y_test:
            profit += stake  # win +stake profit (assuming return = stake)
            correct += 1
        else:
            profit -= stake
        bankroll += profit

    acc = correct / total if total>0 else 0.0
    return {"n_trades": total, "accuracy": acc, "profit": profit, "bankroll":bankroll}

# ------------------ FALLBACK ENSEMBLE (if no model) ------------------
def ensemble_predict(history: List[Tuple[str,str]]):
    # simple heuristics as earlier: frequency + recent + markov
    seq = [o for (_, o) in history]
    if not seq:
        return {"D":0.5,"T":0.5,"confidence":0.0}
    # freq
    last200 = seq[-200:]
    c = Counter(last200)
    p_freq_D = c["D"] / (c["D"] + c["T"])
    # recent weighted
    recent = seq[-80:][::-1]
    w_d=w_t=0.0; w=1.0; decay=0.92
    for s in recent:
        if s=="D": w_d += w
        else: w_t += w
        w *= decay
    p_recent = w_d / (w_d+w_t) if (w_d+w_t)>0 else 0.5
    # markov
    trans = defaultdict(lambda:Counter())
    for a,b in zip(seq, seq[1:]):
        trans[a][b]+=1
    last = seq[-1]
    c2 = trans.get(last, {})
    tot = c2["D"]+c2["T"]
    p_mark = c2["D"]/tot if tot>0 else 0.5
    # weighted avg
    pD = 0.3*p_freq_D + 0.4*p_recent + 0.3*p_mark
    pT = 1-pD
    return {"D":pD, "T":pT, "confidence": abs(pD-pT)}

# ------------------ TELEGRAM BOT ------------------
app = Client("dragon_tiger_ml_bot", bot_token=BOT_TOKEN, api_id=API_ID, api_hash=API_HASH)

@app.on_message(filters.command("start") & filters.private)
async def start(_, m):
    await m.reply(
        "**Dragon & Tiger ML Predictor**\n\n"
        "Commands:\n"
        "/add D — add Dragon\n"
        "/add T — add Tiger\n"
        "/bulk D T D — add multiple\n"
        "/train — train ML model (admin only)\n"
        "/predict — predict next using ML model (or fallback)\n"
        "/backtest — run quick backtest\n"
        "/stats — show dataset stats\n"
        "/export — export CSV\n"
        "/clear — clear history (admin only)\n"
    )

@app.on_message(filters.command("add") & filters.private)
async def add(_, m):
    if len(m.command) < 2:
        return await m.reply("Use: /add D or /add T")
    v = m.command[1].upper()
    if v not in ("D","T"):
        return await m.reply("Use D or T only.")
    storage.add(v)
    await m.reply(f"Added: {v}\nTotal rounds: {len(storage.get_all())}")

@app.on_message(filters.command("bulk") & filters.private)
async def bulk(_, m):
    seq = [x.upper() for x in m.command[1:] if x.upper() in ("D","T")]
    storage.bulk_add(seq)
    await m.reply(f"Added {len(seq)} items. Total: {len(storage.get_all())}")

@app.on_message(filters.command("stats") & filters.private)
async def stats(_, m):
    hist = storage.get_all()
    seq = [o for (_,o) in hist]
    c = Counter(seq)
    await m.reply(f"Total rounds: {len(seq)}\nDragon: {c['D']} | Tiger: {c['T']}\nPreview: {''.join(seq[-80:])}")

@app.on_message(filters.command("export") & filters.private)
async def export(_, m):
    await m.reply_document(LOCAL_HISTORY)

@app.on_message(filters.command("train") & filters.private)
async def train_cmd(_, m):
    if ADMIN_ID != 0 and m.from_user.id != ADMIN_ID:
        return await m.reply("You're not allowed to run training.")
    hist = storage.get_all()
    model, info = train_model(hist)
    if model is None:
        return await m.reply(str(info))
    await m.reply(f"Training complete.\nCV mean acc: {info['cv_mean']:.4f} ± {info['cv_std']:.4f}\nSamples: {info['n_samples']}\nModel saved: {MODEL_FILE}")

@app.on_message(filters.command("predict") & filters.private)
async def predict_cmd(_, m):
    hist = storage.get_all()
    mdl = load_model()
    if mdl is not None:
        # build a single-row features from last LAGS entries (we need to create an artificial dataset with last index as t)
        df = sequence_to_dataframe(hist, lags=LAGS)
        if df.empty:
            # not enough data -> fallback
            ens = ensemble_predict(hist)
            return await m.reply(f"Not enough data for ML prediction. Fallback: P(D)={ens['D']:.2f}, P(T)={ens['T']:.2f}")
        # take last row as the one representing predicting next
        Xlast = df.drop(columns=["y"]).iloc[[-1]]
        proba = mdl.predict_proba(Xlast)[0]
        pD = proba[1]; pT = proba[0]
        conf = abs(pD - pT)
        sug = "Dragon (D)" if pD>pT else "Tiger (T)"
        await m.reply(f"ML Suggestion: {sug}\nP(Dragon)={pD:.3f}\nP(Tiger)={pT:.3f}\nConfidence={conf:.3f}")
    else:
        ens = ensemble_predict(hist)
        await m.reply(f"No ML model found. Ensemble fallback:\nP(Dragon)={ens['D']:.3f}\nP(Tiger)={ens['T']:.3f}\nConfidence={ens['confidence']:.3f}")

@app.on_message(filters.command("backtest") & filters.private)
async def backtest_cmd(_, m):
    hist = storage.get_all()
    if len(hist) < 200:
        return await m.reply("Need at least 200 rounds for meaningful backtest.")
    res = backtest(hist, model=None)
    if "error" in res:
        return await m.reply(res["error"])
    await m.reply(f"Backtest results:\nTrades: {res['n_trades']}\nAccuracy: {res['accuracy']:.3f}\nProfit (simulated): {res['profit']:.2f}")

@app.on_message(filters.command("clear") & filters.private)
async def clear_cmd(_, m):
    if ADMIN_ID != 0 and m.from_user.id != ADMIN_ID:
        return await m.reply("You're not allowed to clear history.")
    storage.clear()
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    await m.reply("History and model cleared.")

# ------------------ RUN ------------------
if __name__ == "__main__":
    print("Starting Dragon & Tiger ML Bot...")
    app.run()
