from fastapi import FastAPI, HTTPException
from PyPDF2 import PdfReader
from contextlib import asynccontextmanager
from typing import List, Optional, Any
from pydantic_mongo import PydanticObjectId
import requests
from io import BytesIO
from motor import MotorClient
from bson import ObjectId
from pydantic import BaseModel,Field
import certifi
from gradio_client import Client


extractor = Client('mrfirdauss/CV-Extractor')
class CV(BaseModel):
    id: Optional[PydanticObjectId] = Field(alias='_id')
    name: str = Field(...)
    skills: List[str] = Field(...)
    experiences: List[dict[str, Any]] = Field(...)
    educations: List[dict[str, Any]] = Field(...)
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
                ObjectId: str
            }
class InsertedText(BaseModel):
    text: str

uri = "mongodb+srv://dutee_1:KqBeHX6ybIHmV7N7@cluster0.fngwqyd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
db = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the DB
    client = MotorClient(uri, tlsCAFile=certifi.where())
    try:
        db['database'] = client.get_database("tugas-akhir")
        yield
    finally:
        client.close()

app =  FastAPI(lifespan=lifespan)
@app.get("/", response_model=dict[str, str])
def getall():
    return {"hello":"world"}

@app.get("/cvs", response_model=List[CV])
async def get_all():
    res = []
    async for i in db['database']["application-cvs"].find(): 
        res.append(CV(**i))
    return res


@app.post("/cv/", response_model=CV)
async def extract(text: InsertedText):
    response = requests.get(text.text)
    if response.status_code == 200:
        # Open the PDF from bytes in memory
        pdf_reader = PdfReader(BytesIO(response.content))
        number_of_pages = len(pdf_reader.pages)
        # Optionally, read text from the first page
        page = pdf_reader.pages[0]
        text = page.extract_text()
        for i in range(1, number_of_pages):
            text+= '\n' + pdf_reader.pages[i].extract_text()
    else:
        #return error, make 500 because file server error
        raise HTTPException(status_code=response.status_code, detail="File server error")
    dictresult = extractor.predict(text)
    #dict to CV model
    cv = CV(**dictresult)
    return cv


