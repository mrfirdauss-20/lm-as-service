from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import List, Optional, Any
from pydantic_mongo import PydanticObjectId
from motor import MotorClient
import re
import numpy as np
from bson import ObjectId
from pydantic import BaseModel,Field
import certifi
from transformers import RobertaTokenizerFast, AutoModelForTokenClassification
import torch

tokenizer = RobertaTokenizerFast.from_pretrained("philschmid/distilroberta-base-ner-conll2003")
model = AutoModelForTokenClassification.from_pretrained("mrfirdauss/robert-base-finetuned-cv")

class CV(BaseModel):
    job_vacancy_id: int = Field(...)
    user_id: int = Field(...)
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
id2label = {0: 'O',
 1: 'B-NAME',
 3: 'B-NATION',
 5: 'B-EMAIL',
 7: 'B-URL',
 9: 'B-CAMPUS',
 11: 'B-MAJOR',
 13: 'B-COMPANY',
 15: 'B-DESIGNATION',
 17: 'B-GPA',
 19: 'B-PHONE NUMBER',
 21: 'B-ACHIEVEMENT',
 23: 'B-EXPERIENCES DESC',
 25: 'B-SKILLS',
 27: 'B-PROJECTS',
 2: 'I-NAME',
 4: 'I-NATION',
 6: 'I-EMAIL',
 8: 'I-URL',
 10: 'I-CAMPUS',
 12: 'I-MAJOR',
 14: 'I-COMPANY',
 16: 'I-DESIGNATION',
 18: 'I-GPA',
 20: 'I-PHONE NUMBER',
 22: 'I-ACHIEVEMENT',
 24: 'I-EXPERIENCES DESC',
 26: 'I-SKILLS',
 28: 'I-PROJECTS'}
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

# @app.post("/cv", response_model=CV)
# async def post_one(cv: CV):
#     res = await db['database']["application-cvs"].insert_one(cv.dict())
#     cv.id = res.inserted_id
#     return cv
def process_tokens(tokens, tag_prefix):
    # Extract and join tokens that belong to the same entity
    entity_parts = []
    current_entity = []
    for token, tag in tokens:
        if tag == 'O' and current_entity:
            entity_parts.append(' '.join(current_entity))
            current_entity = []
        elif tag.startswith(tag_prefix):
            if tag.startswith('B-'):
                if current_entity:
                    entity_parts.append(' '.join(current_entity))
                    current_entity = [token]
                else:
                    current_entity.append(token)
            elif tag.startswith('I-') and current_entity:
                current_entity.append(token)
    if current_entity:
        entity_parts.append(' '.join(current_entity))
    return entity_parts

@app.post("/cv/", response_model=str)
async def extract(text: InsertedText):
    print(text)
    tokens = re.findall(r'\w+|[^\w\s]', text.dict()['text'], re.UNICODE)
    tok = tokenizer(tokens, 
                    return_offsets_mapping=True,
                    padding='max_length',
                    truncation=True,
                    is_split_into_words=True,   
                    max_length=512)
    input_ids = torch.as_tensor(tok['input_ids']).unsqueeze(0)
    attention_mask = torch.as_tensor(tok['attention_mask']).unsqueeze(0)
    res = model(input_ids, attention_mask=attention_mask)
    print('flat', res[0].shape)
    active_logits = res[0].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    eval_per = np.array([id2label[id.item()] for id in flattened_predictions])
    data = list(zip(tokens, eval_per))
    profile = {
        "job_vacancy_id": 123,  # Example ID
        "user_id": 456,  # Example ID
        "name": "",
        "skills": [],
        "experiences": [],
        "educations": []
    }
    profile['name'] = ' '.join(process_tokens(data, 'NAME'))
    profile['experiences'].append({
        "start": "2022-01-01",
        "end": "Present",
        "designation": ' '.join(process_tokens(data, 'DESIGNATION')),
        "company": ' '.join(process_tokens(data, 'COMPANY')),
        "experience_description": ' '.join(process_tokens(data, 'EXPERIENCES DESC'))
    })

    profile['educations'].append({
        "start": "2018-01-01",
        "end": "2020-12-31",
        "major": ' '.join(process_tokens(data, 'MAJOR')),
        "campus": ' '.join(process_tokens(data, 'CAMPUS')),
        "GPA": ''.join(process_tokens(data, 'GPA'))
    })
    return profile


