import joblib

model = joblib.load("model.pkl")

while True:
    text = input("Enter review (or type 'exit'): ")
    if text.lower() == "exit":
        break

    prediction = model.predict([text])[0]
    print("Prediction:", prediction)