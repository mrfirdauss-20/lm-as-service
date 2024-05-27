from fastapi import FastAPI, HTTPException
from PyPDF2 import PdfReader
from typing import List, Any
import requests
from io import BytesIO
from pydantic import BaseModel,Field
from gradio_client import Client
from datetime import datetime

extractor = Client('mrfirdauss/CV-Extractor')
classificator = Client('mrfirdauss/JobClassification')
class CVExtracted(BaseModel):
    name: str = Field(...)
    skills: List[str] = Field(...)
    links: List[str] = Field(...)
    experiences: List[dict[str, Any]] = Field(...)
    educations: List[dict[str, Any]] = Field(...)

class InsertedText(BaseModel):
    text: str

class CVToClassify(BaseModel):
    educations: List[dict[str, Any]]
    skills: List[str]
    experiences: List[dict[str, Any]]

class JobToClassify(BaseModel):
    minYoE: int
    jobDesc: str
    skills: List[str]
    role: str
    majors: List[str]


class JobAndCV(BaseModel):
    cv: CVToClassify
    job: JobToClassify

class ClassificationResult(BaseModel):
    score: float
    is_accepted: bool
class InsertedLink(BaseModel):
    link: str

uri = "mongodb+srv://dutee_1:KqBeHX6ybIHmV7N7@cluster0.fngwqyd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
db = {}



app =  FastAPI()
@app.get("/", response_model=dict[str, str])
def getall():
    return {"hello":"world"}


@app.post("/cv/ext", response_model=CVExtracted)
async def extract(text: InsertedText):
    dictresult = extractor.predict(text.text)
    # res = await db['database']["application-cvs"].insert_one(dictresult)
    # cv = await db['database']["application-cvs"].find_one({"_id": res.inserted_id})
    return CVExtracted(**dictresult)


@app.post("/cv/classify", response_model=ClassificationResult)
async def classify(body:JobAndCV ):
    # measure yoe from list of dict in body['cv']['experiences']
    # reduce the gratest end_date in list of dict exp by the smallest start date
    mininmal_start = 0
    maximal_end = 0
    positions = []
    userMajors = []
    if len(body.cv.experiences) > 0:
        mininmal_start = datetime.strptime(body.cv.experiences[0]['start'], "%Y-%m-%d").date() if body.cv.experiences[0].get('start') != None else datetime.today().date()
        maximal_end = datetime.strptime(body.cv.experiences[0]['end'], "%Y-%m-%d").date()
        for exp in body.cv.experiences:
            positions.append(exp['position'])
            if exp.get('end') == None:
                exp['end'] = datetime.today().strftime("%Y-%m-%d")

            if datetime.strptime(exp['start'], "%Y-%m-%d").date() < mininmal_start:
                mininmal_start = datetime.strptime(exp['start'], "%Y-%m-%d").date()
            if datetime.strptime(exp['end'], "%Y-%m-%d").date() > maximal_end:
                maximal_end = datetime.strptime(exp['end'], "%Y-%m-%d").date()
    else:
        mininmal_start = 0
        maximal_end = 0
    
    for edu in body.cv.educations:
        userMajors.append(edu['major'])
    
    yoe = (maximal_end - mininmal_start).days//365  

    #classiffy by calling api
    #params [exp, listOfPosition, major_applicant, skills_applicant, yoe, jobdesc, rolename, major_vacancy, skills_vacancy, minimumYoe] all in str
    results = classificator.predict(str(body.cv.experiences), str(positions), str(userMajors), str(body.cv.skills), yoe, body.job.jobDesc, body.job.role, str(body.job.majors), str(body.job.skills), body.job.minYoE)
    return ClassificationResult(**results)

@app.post("/cv/", response_model=CVExtracted)
async def extract(link: InsertedLink):
    response = requests.get(link.link)
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
    return CVExtracted(**dictresult)


