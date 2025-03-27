from fastapi import FastAPI, HTTPException
import cohere
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change later for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TutorRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "AI Tutor API is running!"}

@app.post("/tutor/")
async def tutor(request: TutorRequest):
    try:
        user_question = request.question
        response = co.generate(
            model="command",
            prompt=user_question,
            max_tokens=100
        )
        return {"answer": response.generations[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




