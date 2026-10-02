"""
ingest.py
---------
Loads all FAQ source documents from the data/ folder — supports Word
(.docx), PDF (.pdf), plain text (.txt), and JSON (.json) files — splits
them into chunks, embeds them with Google's Gemini embedding API, and
uploads the vectors to a Pinecone index.

For .docx files, this also extracts embedded images and runs OCR on
them so text inside pictures/screenshots is indexed too.

Run this once whenever your source documents change:
    python ingest.py

IMPORTANT: this must use the SAME embedding model (and output dimension)
as rag_engine.py, and your Pinecone index must be created with that same
dimension. gemini-embedding-001 outputs 3072-dim vectors by default.
If you change EMBEDDING_MODEL or EMBEDDING_DIM here, update rag_engine.py
and your Pinecone index to match, or retrieval will fail.
"""

import os
import glob
import zipfile
import io
import json
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from PIL import Image
import pytesseract

# Load .env from the same folder as this file.
load_dotenv()

# --- Windows users: point this to your actual tesseract.exe location ---
# After installing Tesseract (https://github.com/UB-Mannheim/tesseract/wiki),
# it's usually installed at one of these paths. Update this to match your
# actual install path. NOTE: every backslash-separated folder needs its
# own backslash — a missing one silently merges two folder names together
# and breaks this path.
#
# Example:  r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Example:  r"D:\Project_packages\tesseract.exe"
TESSERACT_CMD = r"D:\Project_packages\tesseract.exe"

# Mac/Linux users: comment out the two lines below — tesseract is normally
# already on PATH after `brew install tesseract` or `sudo apt install tesseract-ocr`.
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
if not os.path.isfile(TESSERACT_CMD):
    print(
        f"WARNING: Tesseract not found at '{TESSERACT_CMD}'. "
        "OCR on images inside Word files will fail until this path is corrected. "
        "Double-check every backslash in the path above."
    )

DATA_DIR = "data"

# gemini-embedding-001 replaced the deprecated text-embedding-004 /
# embedding-001 models (Google retired those - see error:
# "models/text-embedding-004 is not found for API version v1beta").
# Must match rag_engine.py exactly.
EMBEDDING_MODEL = "models/gemini-embedding-001"

# gemini-embedding-001 defaults to 3072 dimensions. Your Pinecone index
# MUST be created with this same dimension value.
EMBEDDING_DIM = 3072

# Google's free tier for embed_content is capped at 100 requests/minute.
# We upload in small batches with a pause between them, and retry with
# backoff if we still hit the quota (e.g. from other traffic on the key).
BATCH_SIZE = 20          # chunks per batch (well under the 100/min limit)
PAUSE_BETWEEN_BATCHES = 15  # seconds to wait between batches
MAX_RETRIES_PER_BATCH = 5
RETRY_BACKOFF_SECONDS = 40  # base wait on a RESOURCE_EXHAUSTED error

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not set in .env. This is required to build embeddings "
        "with the same model rag_engine.py uses to query them."
    )

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "erp-faq-index").strip()

if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not set in .env.")


# ---------------------------------------------------------------------------
# Word (.docx) — text + OCR on embedded images
# ---------------------------------------------------------------------------

def extract_images_from_docx(path: str) -> list[bytes]:
    """
    A .docx is just a zip file. Embedded images live under word/media/.
    This pulls out the raw bytes of each image without needing to
    parse the document structure.
    """
    image_bytes_list = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.startswith("word/media/"):
                image_bytes_list.append(z.read(name))
    return image_bytes_list


def ocr_images(image_bytes_list: list[bytes], source_name: str) -> list[Document]:
    """
    Runs OCR on each image and returns one Document per image that
    contains any recognizable text. Images with no text (logos, icons,
    decorative lines) are skipped automatically since OCR returns empty.
    """
    docs = []
    for idx, img_bytes in enumerate(image_bytes_list):
        try:
            image = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(image).strip()
        except Exception as e:
            print(f"  Skipping image {idx} in {source_name} (couldn't process: {e})")
            continue

        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source_name, "type": "image_ocr", "image_index": idx},
                )
            )
    return docs


def load_docx(path: str) -> list[Document]:
    source_name = os.path.basename(path)
    docs = []

    loader = Docx2txtLoader(path)
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = source_name
    docs.extend(pages)

    image_bytes_list = extract_images_from_docx(path)
    if image_bytes_list:
        image_docs = ocr_images(image_bytes_list, source_name)
        print(f"  Found {len(image_bytes_list)} image(s), extracted text from {len(image_docs)} of them.")
        docs.extend(image_docs)

    return docs


# ---------------------------------------------------------------------------
# PDF (.pdf)
# ---------------------------------------------------------------------------

def load_pdf(path: str) -> list[Document]:
    source_name = os.path.basename(path)
    loader = PyPDFLoader(path)
    pages = loader.load()
    for p in pages:
        p.metadata["source"] = source_name
    return pages


# ---------------------------------------------------------------------------
# Plain text (.txt)
# ---------------------------------------------------------------------------

def load_txt(path: str) -> list[Document]:
    source_name = os.path.basename(path)
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = source_name
    return docs


# ---------------------------------------------------------------------------
# JSON (.json)
# ---------------------------------------------------------------------------

def load_json(path: str) -> list[Document]:
    """
    Handles three shapes:
      1. Module-grouped FAQs (this project's format), e.g.:
         {
           "Sales": {
             "module_title": "Sales Module FAQs",
             "questions": [{"q": "...", "a": "..."}, ...]
           },
           "Inventory": {...}
         }
         For each Q&A pair, the module name and title are woven directly
         into the embedded text (not just metadata). This is the key fix:
         without it, a bare question like "how do I do e-invoice" has to
         compete against near-identical questions in other modules with
         no signal to disambiguate, so retrieval only worked when the
         user manually typed the module name into their question.
      2. A flat list of FAQ objects, e.g. [{"question": "...", "answer": "..."}, ...]
      3. Any other JSON structure — falls back to indexing the whole
         file as one readable text block.
    """
    source_name = os.path.basename(path)
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def is_module_grouped(d) -> bool:
        if not isinstance(d, dict) or not d:
            return False
        return all(
            isinstance(v, dict) and isinstance(v.get("questions"), list)
            for v in d.values()
        )

    if is_module_grouped(data):
        for module_key, module_data in data.items():
            module_title = module_data.get("module_title", module_key)
            for i, item in enumerate(module_data.get("questions", [])):
                q = item.get("q") or item.get("question") or item.get("Question")
                a = item.get("a") or item.get("answer") or item.get("Answer")
                if not (q and a):
                    continue

                # Module name/title is embedded in the actual page_content
                # (not just metadata) so the embedding model "sees" it and
                # matches even when the user's question omits it.
                content = (
                    f"Module: {module_title} ({module_key})\n"
                    f"Q: {q}\n"
                    f"A: {a}"
                )
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": source_name,
                            "module": module_key,
                            "module_title": module_title,
                            "row": i,
                            "faq_atomic": True,  # tells the splitter to leave this chunk whole
                        },
                    )
                )
        return docs

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        for i, item in enumerate(data):
            q = item.get("question") or item.get("Question")
            a = item.get("answer") or item.get("Answer")
            if q and a:
                content = f"Q: {q}\nA: {a}"
            else:
                content = "\n".join(f"{k}: {v}" for k, v in item.items())
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": source_name, "row": i, "faq_atomic": True},
                )
            )
        return docs

    content = json.dumps(data, indent=2, ensure_ascii=False)
    docs.append(Document(page_content=content, metadata={"source": source_name}))
    return docs


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    ".docx": load_docx,
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".json": load_json,
}


def load_all_documents(data_dir: str) -> list[Document]:
    """Loads every supported file type found in data_dir."""
    docs = []

    for ext, loader_fn in LOADERS.items():
        paths = glob.glob(os.path.join(data_dir, f"*{ext}"))
        for path in paths:
            print(f"Loading {ext} file: {path}")
            try:
                docs.extend(loader_fn(path))
            except Exception as e:
                print(f"  Skipping '{path}' — failed to read it ({e})")

    return docs


def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    """
    Creates the Pinecone index if it doesn't already exist. If it exists
    with a DIFFERENT dimension than expected, raises an error rather than
    silently corrupting retrieval later.
    """
    existing = [idx["name"] for idx in pc.list_indexes()]

    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dimension={dimension})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        desc = pc.describe_index(index_name)
        existing_dim = desc.dimension
        if existing_dim != dimension:
            raise ValueError(
                f"Pinecone index '{index_name}' already exists with dimension "
                f"{existing_dim}, but EMBEDDING_DIM is set to {dimension}. "
                "Either delete the index in the Pinecone console and re-run "
                "this script, or fix EMBEDDING_DIM to match the existing index."
            )
        print(f"Using existing Pinecone index '{index_name}' (dimension={existing_dim}).")


def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading documents...")
    raw_docs = load_all_documents(DATA_DIR)

    if not raw_docs:
        print(
            f"No supported files found in '{DATA_DIR}/'. "
            "Supported types: .docx, .pdf, .txt, .json. "
            "Add your files there and re-run this script."
        )
        return

    print(f"Loaded {len(raw_docs)} raw documents/pages.")

    # FAQ Q&A pairs (from JSON) are already the right size/shape as one
    # unit — each one already carries its module context. Splitting them
    # further would risk severing the module tag or question from parts
    # of the answer, which is exactly what broke retrieval before.
    atomic_docs = [d for d in raw_docs if d.metadata.get("faq_atomic")]
    splittable_docs = [d for d in raw_docs if not d.metadata.get("faq_atomic")]

    print("Splitting non-FAQ documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_chunks = splitter.split_documents(splittable_docs)
    chunks = atomic_docs + split_chunks
    print(
        f"Kept {len(atomic_docs)} FAQ Q&A chunks whole, "
        f"split the rest into {len(split_chunks)} chunks. "
        f"Total: {len(chunks)} chunks."
    )

    print(f"Loading embedding model ({EMBEDDING_MODEL}) via Gemini API...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
        output_dimensionality=EMBEDDING_DIM,
    )

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, PINECONE_INDEX_NAME, EMBEDDING_DIM)

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    total = len(chunks)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Uploading {total} chunks to Pinecone index '{PINECONE_INDEX_NAME}' "
        f"in {num_batches} batches of {BATCH_SIZE} "
        f"(pausing {PAUSE_BETWEEN_BATCHES}s between batches to respect Gemini's free-tier rate limit)..."
    )

    for batch_num, start in enumerate(range(0, total, BATCH_SIZE), start=1):
        batch = chunks[start:start + BATCH_SIZE]

        attempt = 0
        while True:
            attempt += 1
            try:
                vectorstore.add_documents(batch)
                print(f"  Batch {batch_num}/{num_batches} uploaded ({len(batch)} chunks).")
                break
            except Exception as e:
                is_quota_error = "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower()
                if is_quota_error and attempt <= MAX_RETRIES_PER_BATCH:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    print(
                        f"  Batch {batch_num}/{num_batches}: hit rate limit "
                        f"(attempt {attempt}/{MAX_RETRIES_PER_BATCH}). Waiting {wait}s before retrying..."
                    )
                    time.sleep(wait)
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to upload batch {batch_num}/{num_batches} after {attempt} attempt(s): {e}"
                    ) from e

        if batch_num < num_batches:
            time.sleep(PAUSE_BETWEEN_BATCHES)

    print(f"Done. {total} chunks uploaded to Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    build_index()