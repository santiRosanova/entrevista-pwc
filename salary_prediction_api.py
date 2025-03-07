from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from salary_prediction import preprocess_data, train_model, load_dataset
import uvicorn

app = FastAPI()

file_path = "employee_salary_dataset.csv"
df = load_dataset(file_path)
X, y, encoder, scaler = preprocess_data(df)
model = train_model(X, y)

joblib.dump(model, "salary_model.pkl")

@app.post("/predict_salary/")
def predict_salary(age: int, gender: str, education: str, job_title: str, experience: int):
    try:
        model = joblib.load("salary_model.pkl")
        
        input_data = pd.DataFrame([[age, gender, education, job_title, experience]], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])
        
        X_input, _, _, _ = preprocess_data(pd.concat([df, input_data], ignore_index=True))
        X_input = X_input.iloc[-1:].values
        
        predicted_salary = model.predict(X_input)[0]

        return {"predicted_salary": round(predicted_salary, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
