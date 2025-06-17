import PyPDF2
import re
from typing import Optional
import io
from services.prompts import CHUNKING_SETTINGS

def parse_input(file_obj: Optional[object] = None, text_input: Optional[str] = None) -> str:
    try:
        content = ""
        if file_obj:
            file_format = _detect_format(file_obj)
            if file_format == 'pdf':
                content = _extract_pdf_text(file_obj)
            elif file_format == 'txt':
                content = _read_text_file(file_obj)
            else:
                return ""
        elif text_input:
            content = text_input
        if content:
            content = _clean_text(content)
            content = _preprocess_text(content)
            if len(content.strip()) < CHUNKING_SETTINGS['min_chunk_length']:
                return ""
            return content
        return ""
    except Exception:
        return ""

def _clean_text(text: str) -> str:
    try:
        text = text.encode('ascii', 'ignore').decode()
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        text = re.sub(r'([.,;:!?])\s*', r'\1 ', text)
        return text.strip()
    except Exception:
        return text

def _extract_pdf_text(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = []
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text.append(_clean_text(content))
        return '\n\n'.join(text)
    except Exception:
        return ""

def _read_text_file(text_file) -> str:
    try:
        if isinstance(text_file, (str, bytes, io.TextIOWrapper)):
            return text_file.read()
        elif hasattr(text_file, 'read'):
            content = text_file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return content
        return ""
    except Exception:
        return ""

def _detect_format(file_obj) -> str:
    if hasattr(file_obj, 'name'):
        filename = file_obj.name.lower()
        if filename.endswith('.pdf'):
            return 'pdf'
        elif filename.endswith('.txt'):
            return 'txt'
    return 'unknown'

def _preprocess_text(text: str) -> str:
    try:
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1\n\2', text)
        text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)
        text = re.sub(r'^\s*[•∙○●]\s*', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*(\d+)\)\s*', r'\1. ', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    except Exception:
        return text

