from typing import Dict, List

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import uvicorn

from pydantic import BaseModel

import sqlite3

from utils.utils import classificate


class RequestModel(BaseModel):
    text: str
    num: int = 10


class ResponseModel(BaseModel):
    name: str
    predicted: Dict[str, float]


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/api/v1/ksr", response_model=ResponseModel)
def ksr(request: RequestModel):
    connect: sqlite3.Connection = sqlite3.connect("database/database.db")
    cursor: sqlite3.Cursor = connect.cursor()
    rows = cursor.execute("SELECT name FROM ksr")
    classes = [row[0] for row in rows]
    predicted = classificate(request.text, classes, request.num)
    response: ResponseModel = ResponseModel(
        name=request.text, predicted=predicted)
    connect.close()
    return response


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request}, status_code=status.HTTP_200_OK)


if __name__ == '__main__':
    uvicorn.run("main:app", reload=True)
