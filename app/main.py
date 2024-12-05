import uvicorn
from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}



app.state.model =



@app.get("/predict")
app.state.model.predict(...)
