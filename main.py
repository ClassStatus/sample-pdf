# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import tempfile
import shutil
import os
import uuid
from app.processor import process_pdf

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form_page():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/extract/")
async def extract(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    uid = str(uuid.uuid4())
    temp_pdf_path = os.path.join(tmp_dir, f"{uid}.pdf")

    with open(temp_pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    csv_path = process_pdf(temp_pdf_path)
    return FileResponse(csv_path, filename="transactions.csv")
