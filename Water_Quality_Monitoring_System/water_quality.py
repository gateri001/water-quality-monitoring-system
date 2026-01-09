import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Simulate water parameters
np.random.seed(42)
n_samples = 1000
pH = np.random.normal(7.0, 0.5, n_samples)
turbidity = np.random.normal(3, 1.5, n_samples)
dissolved_oxygen = np.random.normal(7.5, 1.5, n_samples)
nitrate = np.random.normal(5, 2, n_samples)
temperature = np.random.normal(25, 3, n_samples)

# Label safe vs unsafe
suitable = ((pH >= 6.5) & (pH <= 8.5) &
            (turbidity < 5) &
            (dissolved_oxygen > 5) &
            (nitrate < 10)).astype(int)

X = np.stack([pH, turbidity, dissolved_oxygen, nitrate, temperature], axis=1)
y = suitable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Water Quality Model Accuracy: {acc:.4f}")
# Predict for a few test samples
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Sample {i+1}: {'✅ Suitable' if preds[i] else '❌ Unsuitable'} (Actual: {'✅' if y_test[i] else '❌'})")

# Custom scenarios
custom_samples = np.array([
    [7.2, 2.0, 8.0, 4.0, 25],   # Safe water
    [6.0, 6.0, 4.0, 12.0, 28],  # Unsafe water
    [8.0, 4.5, 6.0, 9.0, 22],   # Borderline safe
])

custom_preds = (model.predict(custom_samples) > 0.5).astype(int).flatten()
for i, p in enumerate(custom_preds):
    print(f"Custom Sample {i+1}: {'✅ Suitable' if p else '❌ Unsuitable'}")

