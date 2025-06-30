# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import tempfile
import shutil
import os
import uuid
from processor import main

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>Welcome to the PDF Processor API</h2>"

@app.post("/extract/")
async def extract(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp()
    uid = str(uuid.uuid4())
    temp_pdf_path = os.path.join(tmp_dir, f"{uid}.pdf")

    with open(temp_pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    csv_path = process_pdf(temp_pdf_path)
    return FileResponse(csv_path, filename="transactions.csv")
