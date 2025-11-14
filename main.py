import os
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from pydantic import BaseModel
from bson import ObjectId
from gridfs import GridFS

from database import db, create_document, get_documents
from schemas import Document as DocumentSchema

import requests
from PyPDF2 import PdfReader

app = FastAPI(title="PDF Uploader with Text Extraction/OCR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "PDF Uploader API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
        return "\n".join(texts).strip()
    except Exception:
        return ""


def ocr_with_ocrspace(file_bytes: bytes, api_key: str) -> str:
    try:
        url = "https://api.ocr.space/parse/image"
        payload = {
            "language": "eng",
            "isOverlayRequired": False,
            "OCREngine": 2,
        }
        files = {"file": ("file.pdf", file_bytes, "application/pdf")}
        headers = {"apikey": api_key}
        r = requests.post(url, data=payload, files=files, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("IsErroredOnProcessing"):
            return ""
        parsed_results = data.get("ParsedResults") or []
        all_text = "\n".join([pr.get("ParsedText", "") for pr in parsed_results])
        return all_text.strip()
    except Exception:
        return ""


class UploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    ocr_used: bool
    extracted_text_preview: str


@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    use_ocr: bool = Query(False, description="Force OCR using OCR.space"),
    ocr_api_key: Optional[str] = Query(None, description="OCR.space API key if OCR is desired"),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    size = len(file_bytes)

    # 1) Try native text extraction
    text = extract_text_from_pdf(file_bytes)
    text_ok = len(text.strip()) > 20  # minimal threshold

    # 2) Use OCR if forced or native extraction seems empty
    ocr_used = False
    if use_ocr or not text_ok:
        if not ocr_api_key:
            # If OCR requested/needed but no key, keep text as is
            ocr_used = False
        else:
            ocr_text = ocr_with_ocrspace(file_bytes, ocr_api_key)
            if ocr_text:
                text = ocr_text
                ocr_used = True

    if not text:
        text = ""  # store empty, still useful to keep file

    # Store file in GridFS
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    fs = GridFS(db)
    file_id = fs.put(file_bytes, filename=file.filename, contentType=file.content_type)

    # Store document record
    doc = DocumentSchema(
        filename=file.filename,
        content_type=file.content_type,
        size=size,
        extracted_text=text,
        ocr_used=ocr_used,
        file_id=str(file_id),
    )
    doc_id = create_document("document", doc)

    preview = (text[:300] + ("..." if len(text) > 300 else "")) if text else ""
    return UploadResponse(
        id=str(doc_id),
        filename=file.filename,
        size=size,
        ocr_used=ocr_used,
        extracted_text_preview=preview,
    )


@app.get("/api/documents")
def list_documents(limit: int = 50):
    docs = get_documents("document", {}, limit)
    def serialize(d):
        d["id"] = str(d.get("_id"))
        d["file_id"] = str(d.get("file_id")) if d.get("file_id") else None
        # Reduce extracted_text payload for listing
        if "extracted_text" in d and isinstance(d["extracted_text"], str):
            txt = d["extracted_text"]
            d["extracted_text_preview"] = (txt[:200] + ("..." if len(txt) > 200 else ""))
            del d["extracted_text"]
        return d
    return {"items": [serialize(x) for x in docs]}


@app.get("/api/documents/{doc_id}/download")
def download_document(doc_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    fs = GridFS(db)
    try:
        # find document record to get filename
        from pymongo import ReturnDocument
        doc = db["document"].find_one({"_id": ObjectId(doc_id)})
        if not doc or not doc.get("file_id"):
            raise HTTPException(status_code=404, detail="Document not found")
        gridout = fs.get(ObjectId(doc["file_id"]))
        return StreamingResponse(gridout, media_type=doc.get("content_type", "application/pdf"), headers={
            "Content-Disposition": f"attachment; filename={doc.get('filename', 'document.pdf')}"
        })
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found")
