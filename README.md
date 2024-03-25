# MICROPROJECT
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df = pd.read_csv('/content/Medical diagnosis .csv')
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Diabetes'] = label_encoder.fit_transform(df['Diabetes'])
df['Hypertension'] = label_encoder.fit_transform(df['Hypertension'])
df['Label'] = label_encoder.fit_transform(df['Label'])
data = df[['Age', 'Gender', 'Diabetes', 'Hypertension']].values.astype(np.float32)
labels = df['Label'].values.astype(np.float32)
model = Sequential([
    Dense(64, activation='relu', input_shape=(data.shape[1],)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10, batch_size=1)
new_patient_data = np.array([[32, 0, 1, 0]], dtype=np.float32)  # Gender: Female, Diabetes: Yes, Hypertension: No
prediction = model.predict(new_patient_data)

