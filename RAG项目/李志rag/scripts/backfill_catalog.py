import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.database import SessionLocal, initialize_database
from app.rag_main import extract_pdf_text, update_document_catalog
from app.models import Document


def document_text(document: Document) -> str:
    stored = Path(document.stored_path)
    sample = Path("output/pdf") / document.filename
    for path in (stored, sample):
        if path.exists():
            try:
                return extract_pdf_text(path)
            except ValueError:
                continue
    return document.summary or document.filename


def main() -> None:
    initialize_database()
    with SessionLocal() as database:
        documents = database.scalars(
            select(Document).where(Document.status == "ready")
        ).all()
        for document in documents:
            update_document_catalog(document, document_text(document), database)
            database.commit()
            print(f"Catalog updated: {document.filename}")


if __name__ == "__main__":
    main()
