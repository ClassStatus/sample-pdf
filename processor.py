import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random

# === PDF Text Extraction Functions ===

def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += text + "\n"
    return full_text.strip()

def extract_text_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img, config='--psm 6')
        full_text += text + "\n"
    return full_text.strip()

# === Expanded TRAIN_DATA (India + USA formats) ===

TRAIN_DATA = [
    ("05-06-2025 ATM Withdrawal Rs. 1,000.00 DR", {"entities": [(0, 10, "DATE"), (11, 28, "DESCRIPTION"), (29, 41, "AMOUNT"), (42, 44, "DR_CR")]}),
    ("10/06/2025 Salary Credit ₹50,000.00 CR", {"entities": [(0, 10, "DATE"), (11, 25, "DESCRIPTION"), (26, 36, "AMOUNT"), (37, 39, "DR_CR")]}),
    ("2025-06-15 POS Purchase Rs. 2,500.00 DB", {"entities": [(0, 10, "DATE"), (11, 25, "DESCRIPTION"), (26, 38, "AMOUNT"), (39, 41, "DR_CR")]}),
    ("06/05/2025 Cheque Deposit $1,200.50 CR", {"entities": [(0, 10, "DATE"), (11, 27, "DESCRIPTION"), (28, 37, "AMOUNT"), (38, 40, "DR_CR")]}),
    ("15-06-2025 Mobile Recharge Rs. 350.00 DR", {"entities": [(0, 10, "DATE"), (11, 28, "DESCRIPTION"), (29, 37, "AMOUNT"), (38, 40, "DR_CR")]}),
    ("2025/06/10 Utility Bill Payment ₹2,345.75 DB", {"entities": [(0, 10, "DATE"), (11, 32, "DESCRIPTION"), (33, 43, "AMOUNT"), (44, 46, "DR_CR")]}),
    ("07-06-2025 Online Transfer USD 500.00 CR", {"entities": [(0, 10, "DATE"), (11, 28, "DESCRIPTION"), (29, 39, "AMOUNT"), (40, 42, "DR_CR")]}),
    ("2025-06-05 Grocery Store Rs 1,999.99 DR", {"entities": [(0, 10, "DATE"), (11, 24, "DESCRIPTION"), (25, 36, "AMOUNT"), (37, 39, "DR_CR")]}),
    ("10/06/2025 Cashback ₹150.00 CR", {"entities": [(0, 10, "DATE"), (11, 19, "DESCRIPTION"), (20, 28, "AMOUNT"), (29, 31, "DR_CR")]}),
    ("2025-06-08 Interest Credit Rs 75.50 CR", {"entities": [(0, 10, "DATE"), (11, 26, "DESCRIPTION"), (27, 34, "AMOUNT"), (35, 37, "DR_CR")]}),
    ("08-06-2025 Fund Transfer $1,000.00 DR", {"entities": [(0, 10, "DATE"), (11, 24, "DESCRIPTION"), (25, 36, "AMOUNT"), (37, 39, "DR_CR")]}),
    ("2025/06/12 ATM Withdrawal ₹2,200.00 DR", {"entities": [(0, 10, "DATE"), (11, 28, "DESCRIPTION"), (29, 41, "AMOUNT"), (42, 44, "DR_CR")]}),
    ("06-10-2025 Deposit Rs. 5,000.00 CR", {"entities": [(0, 10, "DATE"), (11,18, "DESCRIPTION"), (19,31, "AMOUNT"), (32,34, "DR_CR")]}),
    ("2025-06-09 POS Purchase USD 250.00 DB", {"entities": [(0,10,"DATE"), (11,25,"DESCRIPTION"), (26,36,"AMOUNT"), (37,39,"DR_CR")]}),
    ("10/06/2025 Salary Credit ₹60,000.00 CR", {"entities": [(0,10,"DATE"), (11,25,"DESCRIPTION"), (26,37,"AMOUNT"), (38,40,"DR_CR")]}),
    ("2025-06-07 Mobile Recharge Rs. 450.00 DR", {"entities": [(0,10,"DATE"), (11,28,"DESCRIPTION"), (29,37,"AMOUNT"), (38,40,"DR_CR")]}),
    ("07/06/2025 Loan Repayment ₹15,000.00 DR", {"entities": [(0,10,"DATE"), (11,28,"DESCRIPTION"), (29,42,"AMOUNT"), (43,45,"DR_CR")]}),
    ("2025-06-11 Credit Interest Rs. 100.00 CR", {"entities": [(0,10,"DATE"), (11,27,"DESCRIPTION"), (28,36,"AMOUNT"), (37,39,"DR_CR")]}),
    ("06-10-2025 Online Purchase USD 75.99 DR", {"entities": [(0,10,"DATE"), (11,27,"DESCRIPTION"), (28,37,"AMOUNT"), (38,40,"DR_CR")]}),
    ("2025/06/13 Utility Bill ₹1,150.00 DB", {"entities": [(0,10,"DATE"), (11,22,"DESCRIPTION"), (23,33,"AMOUNT"), (34,36,"DR_CR")]}),

    # USA style
    ("06/05/2025 ATM Withdrawal $1,200.00 DEBIT", {"entities": [(0, 10, "DATE"), (11, 27, "DESCRIPTION"), (28, 39, "AMOUNT"), (40, 45, "DR_CR")]}),
    ("06-10-2025 Payroll Deposit $3,500.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,27, "DESCRIPTION"), (28,39, "AMOUNT"), (40,46, "DR_CR")]}),
    ("06/15/2025 POS Purchase $120.45 DEBIT", {"entities": [(0, 10, "DATE"), (11,26, "DESCRIPTION"), (27,35, "AMOUNT"), (36,41, "DR_CR")]}),
    ("06/18/2025 Check #1234 $450.00 DEBIT", {"entities": [(0, 10, "DATE"), (11,23, "DESCRIPTION"), (24,32, "AMOUNT"), (33,38, "DR_CR")]}),
    ("06/20/2025 Online Transfer $2,000.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,29, "DESCRIPTION"), (30,41, "AMOUNT"), (42,48, "DR_CR")]}),
    ("06-22-2025 ACH Credit $1,250.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,21, "DESCRIPTION"), (22,33, "AMOUNT"), (34,40, "DR_CR")]}),
    ("06/25/2025 Grocery Store $89.75 DEBIT", {"entities": [(0, 10, "DATE"), (11,24, "DESCRIPTION"), (25,32, "AMOUNT"), (33,38, "DR_CR")]}),
    ("06/27/2025 Refund $50.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,17, "DESCRIPTION"), (18,25, "AMOUNT"), (26,32, "DR_CR")]}),
    ("06-30-2025 Interest Payment $5.35 CREDIT", {"entities": [(0, 10, "DATE"), (11,29, "DESCRIPTION"), (30,36, "AMOUNT"), (37,43, "DR_CR")]}),
    ("06/30/2025 Service Fee $15.00 DEBIT", {"entities": [(0, 10, "DATE"), (11,22, "DESCRIPTION"), (23,29, "AMOUNT"), (30,35, "DR_CR")]}),
    ("07/01/2025 Mobile Deposit $500.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,27, "DESCRIPTION"), (28,35, "AMOUNT"), (36,42, "DR_CR")]}),
    ("07-02-2025 Transfer to Savings $1,000.00 DEBIT", {"entities": [(0, 10, "DATE"), (11,33, "DESCRIPTION"), (34,45, "AMOUNT"), (46,51, "DR_CR")]}),
    ("07/03/2025 ATM Deposit $300.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,24, "DESCRIPTION"), (25,32, "AMOUNT"), (33,39, "DR_CR")]}),
    ("07/04/2025 Wire Transfer USD 5,000.00 CREDIT", {"entities": [(0, 10, "DATE"), (11,25, "DESCRIPTION"), (26,38, "AMOUNT"), (39,45, "DR_CR")]}),
    ("07-05-2025 Bill Payment $250.00 DEBIT", {"entities": [(0, 10, "DATE"), (11,25, "DESCRIPTION"), (26,33, "AMOUNT"), (34,39, "DR_CR")]}),
]

# === SpaCy Training Function ===

def train_ner(train_data, iterations=20):
    nlp = spacy.blank("en")  # create blank English model
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(train_data)
            losses = {}
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.5))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.2, losses=losses, sgd=optimizer)
            print(f"Iteration {itn + 1} Losses: {losses}")
    return nlp

# === Function to extract transactions using the trained NER model ===

def extract_transactions_with_ner(text, nlp):
    doc = nlp(text)
    transactions = []
    current_transaction = {}
    
    # This is a simple heuristic grouping per line; you can improve this logic
    for ent in doc.ents:
        label = ent.label_
        if label == "DATE":
            if current_transaction:
                transactions.append(current_transaction)
                current_transaction = {}
            current_transaction["Date"] = ent.text
        elif label == "DESCRIPTION":
            current_transaction["Description"] = ent.text
        elif label == "AMOUNT":
            # Remove commas and currency symbols for numeric conversion
            amount_str = re.sub(r"[^\d.]", "", ent.text)
            try:
                current_transaction["Amount"] = float(amount_str)
            except:
                current_transaction["Amount"] = None
        elif label == "DR_CR":
            current_transaction["DR_CR"] = ent.text

    if current_transaction:
        transactions.append(current_transaction)

    return transactions

# === Main function to run everything ===

def main(pdf_path):
    print("Trying direct text extraction with PyMuPDF...")
    text = extract_text_pymupdf(pdf_path)

    if len(text) < 50:
        print("Direct text extraction failed or too little text, running OCR...")
        text = extract_text_ocr(pdf_path)
    else:
        print("Direct text extraction successful.")

    print("Raw extracted text preview:")
    print(text[:500])

    # Train NER model on example data
    print("Training NER model on example training data...")
    nlp = train_ner(TRAIN_DATA, iterations=30)

    # Extract transactions using NER
    print("Extracting transactions using trained NER model...")
    transactions = extract_transactions_with_ner(text, nlp)

    if not transactions:
        print("No transactions found by NER model.")
    else:
        df = pd.DataFrame(transactions)
        print("Extracted transactions:")
        print(df.head())

        csv_path = pdf_path.replace(".pdf", "_transactions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Transactions saved to {csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bank_statement_converter.py your_statement.pdf")
    else:
        main(sys.argv[1])
