from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import tempfile
import shutil
import os
import uuid

# Import the `main` function from processor.py and alias it as process_pdf
from processor import main as process_pdf

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>Welcome to the PDF Processor API</h2>"

@app.post("/extract/")
async def extract(file: UploadFile = File(...)):
    # Create a temporary directory to save the uploaded PDF
    tmp_dir = tempfile.mkdtemp()
    uid = str(uuid.uuid4())
    temp_pdf_path = os.path.join(tmp_dir, f"{uid}.pdf")

    # Save the uploaded PDF to the temporary file
    with open(temp_pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process the PDF and return the path to the CSV file
    csv_path = process_pdf(temp_pdf_path)

    # Return the CSV file as a downloadable response
    return FileResponse(csv_path, filename="transactions.csv")
