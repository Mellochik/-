from typing import Dict, List

from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

import pandas as pd

import uvicorn

from pydantic import BaseModel

import sqlite3

import io

from utils.utils import classificate, classificate_file


class RequestModel(BaseModel):
    text: str
    num: int = 10


class ResponseModel(BaseModel):
    text: str
    predicted: list


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/api/v1/ksr", response_model=ResponseModel)
def ksr(request: RequestModel):
    connect: sqlite3.Connection = sqlite3.connect("database/database.db")
    ksr = pd.read_sql_query("SELECT * FROM ksr", connect)
    predicted = classificate(request.text, ksr, request.num)
    response: ResponseModel = ResponseModel(
        text=request.text, predicted=predicted)
    connect.close()
    return response


@app.post("/api/v1/ksr/upload", response_class=StreamingResponse)
async def ksr_upload(file: UploadFile = File(..., accept=".csv")):
    connect: sqlite3.Connection = sqlite3.connect("database/database.db")
    ksr = pd.read_sql_query("SELECT * FROM ksr", connect)
    df = pd.read_csv(io.BytesIO(await file.read()), header=None)
    csv_data = classificate_file(df, ksr).encode()
    return StreamingResponse(io.BytesIO(csv_data), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=processed_data.csv"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request}, status_code=status.HTTP_200_OK)


if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
