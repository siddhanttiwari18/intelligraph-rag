import os
import datetime
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.utils.security import validate_filename, validate_file_content
from rag.utils.exceptions import SecurityError, OCRError, UploadError, GraphError
from rag.graph.entity_extractor import EntityExtractor
from rag.graph.relationship_extractor import RelationshipExtractor

logger = logging.getLogger("rag_platform")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise UploadError(f"Failed to read text file: {e}")


def table_to_markdown(table: List[List[str]]) -> str:
    # Filter empty rows
    cleaned_rows = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)
    if not cleaned_rows:
        return ""

    headers = cleaned_rows[0]
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in cleaned_rows[1:]:
        # Align lengths
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        else:
            row = row[:len(headers)]
        markdown += "| " + " | ".join(row) + " |\n"
    return markdown


def run_ocr_on_image(img_path: Path) -> str:
    # 1. Try PaddleOCR
    try:
        import paddle
        from paddleocr import PaddleOCR
        
        # Disable verbose paddle logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr.ocr(str(img_path), cls=True)

        text_lines = []
        if result and result[0]:
            for line in result[0]:
                text_lines.append(line[1][0])

        text = "\n".join(text_lines)
        if text.strip():
            logger.info("Successfully extracted text using PaddleOCR.")
            return text
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug(f"PaddlePaddle/PaddleOCR not fully installed: {e}. Falling back to Tesseract...")
    except Exception as e:
        logger.warning(f"PaddleOCR runtime error: {e}. Falling back to Tesseract...")

    # 2. Try Tesseract fallback
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(img_path))
        if text.strip():
            logger.info("Successfully extracted text using Tesseract OCR.")
            return text
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}. Returning fallback placeholder text.")
        raise OCRError(f"OCR engines failed to process image: {e}") from e

    return "Warning: Scanned PDF page could not be OCR-processed. No text extracted."


def load_pdf_pages(path: Path) -> List[Document]:
    import pdfplumber
    import fitz  # PyMuPDF

    # 1. Open with pdfplumber to check if scanned and to extract tables
    total_text_len = 0
    pages_data = []

    try:
        with pdfplumber.open(path) as pdf:
            pages_count = len(pdf.pages)
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                total_text_len += len(text.strip())

                # Extract tables
                tables = page.extract_tables()
                md_tables = []
                for t in tables:
                    md = table_to_markdown(t)
                    if md.strip():
                        md_tables.append(md)

                pages_data.append({
                    "page_number": idx + 1,
                    "text": text,
                    "tables": md_tables,
                    "table_raw": tables,
                })
    except Exception as e:
        raise UploadError(f"Failed to parse PDF file structures: {e}") from e

    is_scanned = pages_count > 0 and (total_text_len / pages_count) < 50
    pdf_type = "Scanned PDF (OCR)" if is_scanned else "Digital PDF"
    logger.info(f"PDF identified as '{pdf_type}' based on text density.")

    documents = []

    # If scanned, perform OCR
    if is_scanned:
        try:
            doc = fitz.open(path)
            for idx in range(len(doc)):
                page_num = idx + 1
                page = doc[idx]
                pix = page.get_pixmap()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    tmp_img_path = Path(tmp_img.name)
                try:
                    pix.save(str(tmp_img_path))
                    ocr_text = run_ocr_on_image(tmp_img_path)
                finally:
                    tmp_img_path.unlink(missing_ok=True)

                # Create text document
                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "source": path.name,
                            "page_number": page_num,
                            "document_type": ".pdf",
                            "pdf_type": pdf_type,
                            "chunk_type": "text",
                        }
                    )
                )
        except Exception as e:
            raise OCRError(f"Failed to run scanned PDF OCR processing: {e}") from e
    else:
        # Digital PDF
        for p_data in pages_data:
            documents.append(
                Document(
                    page_content=p_data["text"],
                    metadata={
                        "source": path.name,
                        "page_number": p_data["page_number"],
                        "document_type": ".pdf",
                        "pdf_type": pdf_type,
                        "chunk_type": "text",
                    }
                )
            )

    # Add extracted tables as documents
    for p_data in pages_data:
        page_num = p_data["page_number"]
        for t_idx, md_table in enumerate(p_data["tables"]):
            # Get headers from raw tables if available to store in metadata
            headers = []
            if p_data["table_raw"] and len(p_data["table_raw"]) > t_idx:
                raw_table = p_data["table_raw"][t_idx]
                if raw_table and len(raw_table) > 0:
                    headers = [str(h).strip() for h in raw_table[0] if h is not None]

            documents.append(
                Document(
                    page_content=md_table,
                    metadata={
                        "source": path.name,
                        "page_number": page_num,
                        "document_type": ".pdf",
                        "pdf_type": pdf_type,
                        "chunk_type": "table",
                        "table_columns": headers,
                    }
                )
            )

    return documents


def load_document_pages(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        return load_pdf_pages(path)
    else:
        text = load_text_file(path)
        if not text.strip():
            raise ValueError(f"Document is empty: {path.name}")
        return [
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page_number": 1,
                    "document_type": suffix,
                    "pdf_type": "Text Document",
                    "chunk_type": "text",
                },
            )
        ]


def parent_child_chunk_documents(
    documents: List[Document],
    parent_size: int = 1500,
    parent_overlap: int = 200,
    child_size: int = 400,
    child_overlap: int = 50,
) -> List[Dict[str, Any]]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    upload_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for doc in documents:
        filename = doc.metadata.get("source", "unknown")
        page_number = doc.metadata.get("page_number", 1)
        doc_type = doc.metadata.get("document_type", Path(filename).suffix.lower())
        pdf_type = doc.metadata.get("pdf_type", "Digital Document")
        chunk_type = doc.metadata.get("chunk_type", "text")
        table_columns = doc.metadata.get("table_columns", [])

        if chunk_type == "table":
            t_text = doc.page_content
            t_id = f"{filename}_p{page_number}_table_{hash(t_text) % 10000}"
            chunks.append({
                "chunk_id": t_id,
                "text": t_text,
                "parent_text": f"Table on Page {page_number} of {filename}:\n{t_text}",
                "parent_id": t_id,
                "filename": filename,
                "page_number": page_number,
                "document_type": doc_type,
                "pdf_type": pdf_type,
                "chunk_type": "table",
                "table_columns": table_columns,
                "upload_date": upload_date,
            })
        else:
            parent_docs = parent_splitter.split_documents([doc])
            for p_idx, p_doc in enumerate(parent_docs):
                p_text = p_doc.page_content
                p_id = f"{filename}_p{page_number}_parent{p_idx}"

                child_docs = child_splitter.split_documents([p_doc])
                for c_idx, c_doc in enumerate(child_docs):
                    c_text = c_doc.page_content
                    c_id = f"{filename}_p{page_number}_parent{p_idx}_child{c_idx}"

                    chunks.append({
                        "chunk_id": c_id,
                        "text": c_text,
                        "parent_text": p_text,
                        "parent_id": p_id,
                        "filename": filename,
                        "page_number": page_number,
                        "document_type": doc_type,
                        "pdf_type": pdf_type,
                        "chunk_type": "text",
                        "upload_date": upload_date,
                    })
    return chunks


class IngestPipeline:
    """Orchestrates document uploading, validation, OCR/parsing, chunking, 
    vector store additions, and Knowledge Graph extraction.
    """
    def __init__(self, vector_store, graph_store, graph_retriever, entity_extractor, relationship_extractor, telemetry, llm=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.graph_retriever = graph_retriever
        self.entity_extractor = entity_extractor
        self.relationship_extractor = relationship_extractor
        self.telemetry = telemetry
        self.llm = llm

    def ingest_file(
        self, 
        file_name: str, 
        file_bytes: bytes, 
        parent_size: int = 1500, 
        parent_overlap: int = 200, 
        child_size: int = 400, 
        child_overlap: int = 50
    ) -> dict:
        logger.info(f"Starting ingestion for file: {file_name}")
        
        # 1. Security validation (filename checks + header magic bytes checks)
        sanitized_name = validate_filename(file_name)
        validate_file_content(file_bytes, sanitized_name)
        
        suffix = Path(sanitized_name).suffix.lower()
        file_size_bytes = len(file_bytes)
        
        # Write to temporary file for parsing libraries
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            # 2. Parsing pages
            pages = load_document_pages(tmp_path)
            
            ocr_processed = any(p.metadata.get("pdf_type") == "Scanned PDF (OCR)" for p in pages)
            ocr_success = False
            if ocr_processed:
                ocr_success = any("Warning: Scanned PDF page could not be OCR-processed" not in p.page_content for p in pages)
            
            tables_extracted = sum(1 for p in pages if p.metadata.get("chunk_type") == "table")

            # 3. Parent-child chunking
            chunks = parent_child_chunk_documents(
                pages,
                parent_size=parent_size,
                parent_overlap=parent_overlap,
                child_size=child_size,
                child_overlap=child_overlap,
            )

            # Associate original filename instead of temp filename
            for chunk in chunks:
                chunk["filename"] = sanitized_name
                if "_parent" in chunk["chunk_id"]:
                    parts = chunk["chunk_id"].split("_parent")
                    chunk["chunk_id"] = f"{sanitized_name}_parent{parts[1]}"
                elif "_table_" in chunk["chunk_id"]:
                    parts = chunk["chunk_id"].split("_table_")
                    chunk["chunk_id"] = f"{sanitized_name}_table_{parts[1]}"

            # 4. Insert chunks into Vector Store
            count = self.vector_store.add_documents(chunks)
            total_chars = sum(len(p.page_content) for p in pages)

            # 5. Extract and Index Graph Relationships
            unique_parents = {}
            for chunk in chunks:
                p_id = chunk.get("parent_id") or chunk["chunk_id"]
                if p_id not in unique_parents:
                    unique_parents[p_id] = {
                        "text": chunk.get("parent_text") or chunk["text"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_ref": p_id
                    }

            for p_data in unique_parents.values():
                text = p_data["text"]
                page_num = p_data["page_number"]
                c_ref = p_data["chunk_ref"]
                
                source_meta = {
                    "source_document": sanitized_name,
                    "page_number": page_num,
                    "chunk_ref": c_ref
                }

                # Rule-based entities lookup for type mapping
                entities = self.entity_extractor.extract_entities(text)
                entity_type_map = {ent["name"].lower(): ent["type"] for ent in entities}

                # Extract relationships
                relations = self.relationship_extractor.extract_relationships(text, self)
                for rel in relations:
                    src_name = rel["source"]
                    tgt_name = rel["target"]
                    
                    src_type = entity_type_map.get(src_name.lower(), rel.get("source_type", "Unknown"))
                    tgt_type = entity_type_map.get(tgt_name.lower(), rel.get("target_type", "Unknown"))
                    conf = rel.get("confidence", 1.0)

                    self.graph_store.add_entity(src_name, src_type, source_meta, confidence=conf)
                    self.graph_store.add_entity(tgt_name, tgt_type, source_meta, confidence=conf)
                    self.graph_store.add_relationship(src_name, tgt_name, rel["relation_type"], source_meta, confidence=conf)

            # 6. Save graph and clear retrievers search caches
            self.graph_retriever.clear_cache()
            self.graph_store.save()
            
            # Invalidate query caches
            from rag.services.cache import platform_cache
            platform_cache.invalidate()

            # Record success ingestion
            self.telemetry.record_ingestion(
                file_name=sanitized_name,
                file_type=suffix,
                file_size_bytes=file_size_bytes,
                chunks_count=count,
                tables_extracted=tables_extracted,
                ocr_processed=ocr_processed,
                ocr_success=ocr_success,
                success=True
            )

            return {
                "file_name": sanitized_name,
                "chunks_added": count,
                "characters": total_chars,
            }
        except Exception as e:
            # Record failure ingestion
            self.telemetry.record_ingestion(
                file_name=sanitized_name,
                file_type=suffix,
                file_size_bytes=file_size_bytes,
                chunks_count=0,
                tables_extracted=0,
                ocr_processed=False,
                ocr_success=False,
                success=False,
                error_message=str(e)
            )
            self.telemetry.record_error("ocr" if suffix == ".pdf" else "ingestion", str(e), {"file_name": sanitized_name})
            raise e
        finally:
            tmp_path.unlink(missing_ok=True)

    def delete_document(self, filename: str) -> dict:
        logger.info(f"Deleting document: {filename}")
        deleted_count = self.vector_store.delete_document(filename)
        # Clean references inside graph store
        self.graph_store.delete_document_references(filename)
        self.graph_retriever.clear_cache()
        
        # Invalidate query caches
        from rag.services.cache import platform_cache
        platform_cache.invalidate()
        return {
            "file_name": filename,
            "chunks_deleted": deleted_count,
            "success": deleted_count > 0,
        }

    def rebuild_graph(self) -> dict:
        logger.info("Rebuilding knowledge graph store from scratch using all stored vector chunks...")
        try:
            self.graph_store.clear()
            
            # Group chunks in vector store by parent_id/chunk_id to avoid redundant processing
            unique_parents = {}
            for chunk in self.vector_store.chunks:
                p_id = chunk.get("parent_id") or chunk["chunk_id"]
                if p_id not in unique_parents:
                    unique_parents[p_id] = {
                        "text": chunk.get("parent_text") or chunk["text"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_ref": p_id
                    }
                    
            # Extract and add relationships
            for p_data in unique_parents.values():
                text = p_data["text"]
                file_name = p_data["filename"]
                page_num = p_data["page_number"]
                c_ref = p_data["chunk_ref"]
                
                source_meta = {
                    "source_document": file_name,
                    "page_number": page_num,
                    "chunk_ref": c_ref
                }
                
                # Map types
                entities = self.entity_extractor.extract_entities(text)
                entity_type_map = {ent["name"].lower(): ent["type"] for ent in entities}
                
                # Extract relationships
                relations = self.relationship_extractor.extract_relationships(text, self)
                for rel in relations:
                    src_name = rel["source"]
                    tgt_name = rel["target"]
                    
                    src_type = entity_type_map.get(src_name.lower(), rel.get("source_type", "Unknown"))
                    tgt_type = entity_type_map.get(tgt_name.lower(), rel.get("target_type", "Unknown"))
                    
                    conf = rel.get("confidence", 1.0)
                    
                    self.graph_store.add_entity(src_name, src_type, source_meta, confidence=conf)
                    self.graph_store.add_entity(tgt_name, tgt_type, source_meta, confidence=conf)
                    self.graph_store.add_relationship(src_name, tgt_name, rel["relation_type"], source_meta, confidence=conf)
                    
            self.graph_store.save()
            self.graph_retriever.clear_cache()
            
            # Invalidate query caches
            from rag.services.cache import platform_cache
            platform_cache.invalidate()
            
            return {
                "total_nodes": len(self.graph_store.graph.nodes()),
                "total_edges": len(self.graph_store.graph.edges())
            }
        except Exception as e:
            self.telemetry.record_error("graph_build", str(e))
            raise GraphError(f"Failed to rebuild graph: {e}") from e
