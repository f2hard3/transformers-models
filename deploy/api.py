from typing import Dict
from pydantic import BaseModel
from fastapi import FastAPI
import os
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
print('loading tokenizer + model')
CLF = pipeline(
    'text-classification', 
    'f2hard3/distilbert-toxic-classifier',
    use_fast=True, 
    return_all_scores=True,
    use_auth_token=os.environ.get('HUGGING_API_KEY')
)
print('loaded tokenizer + model')

class Request(BaseModel):
    text:str

class Response(BaseModel):
    probabilities: Dict[str, float]
    label: str
    confidence: float

@app.post('/predict', response_model=Response)
def predict(request: Request):
    output = sorted(CLF(request.text)[0], key=lambda x: x['score'], reverse=True)
    return Response(
        label=output[0]['label'], confidence=output[0]['score'],
        probabilities={item['label']: item['score'] for item in output}
    )