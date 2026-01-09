import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from fastapi import FastAPI
from pydantic import BaseModel

# Define input schema
class WaterSample(BaseModel):
    pH: float
    turbidity: float
    dissolved_oxygen: float
    nitrate: float
    temperature: float
    lang: str = "en"  # default language

# Localized messages
MESSAGES = {
    "en": {
        "suitable": "Water is safe for use.",
        "unsuitable": "Water is unsafe. Do not drink."
    },
    "sw": {
        "suitable": "Maji ni salama kwa matumizi.",
        "unsuitable": "Maji si salama. Usiyatumie."
    },
    "ki": {
        "suitable": "Maai ni meega ma kuhuthiruo.",
        "unsuitable": "Maai ti meega ma kuhuthiruo."
    }
}

# Build a simple model (or load pre-trained one)
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train quickly on synthetic data for demo
np.random.seed(42)
n_samples = 500
pH = np.random.normal(7.0, 0.5, n_samples)
turbidity = np.random.normal(3, 1.5, n_samples)
dissolved_oxygen = np.random.normal(7.5, 1.5, n_samples)
nitrate = np.random.normal(5, 2, n_samples)
temperature = np.random.normal(25, 3, n_samples)

suitable = ((pH >= 6.5) & (pH <= 8.5) &
            (turbidity < 5) &
            (dissolved_oxygen > 5) &
            (nitrate < 10)).astype(int)

X = np.stack([pH, turbidity, dissolved_oxygen, nitrate, temperature], axis=1)
y = suitable
model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(sample: WaterSample):
    data = np.array([[sample.pH, sample.turbidity, sample.dissolved_oxygen, sample.nitrate, sample.temperature]])
    prob = float(model.predict(data)[0][0])
    suitability = prob > 0.5

    # Pick language dictionary
    lang = sample.lang if sample.lang in MESSAGES else "en"
    message = MESSAGES[lang]["suitable"] if suitability else MESSAGES[lang]["unsuitable"]

    return {
        "probability": round(prob, 4),
        "suitability": "Suitable" if suitability else "Unsuitable",
        "message": message
    }
