from fastapi import FastAPI
from pydantic import BaseModel
from HybridModel import hybrid_model
from fastapi.responses import FileResponse
import pandas as pd

class UserData(BaseModel):
    listed_price: int
    myear: int
    Length: int
    body: str
    transmission: str
    fuel: str
    Color: str

app = FastAPI()

@app.post("/recommendations/")
def get_recommendations(user_data: UserData):
    user_data_dict = user_data.model_dump()
    print(user_data)
    print(user_data_dict)
    recommendations = hybrid_model(user_data_dict)
    return recommendations.to_dict(orient='records')

@app.get("/dataset/")
def get_data_set():
    return FileResponse(path="Dataset\cars_data_clean.csv", filename="cars_data_clean.csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
