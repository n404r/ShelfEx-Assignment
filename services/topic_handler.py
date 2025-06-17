import re
import torch
from typing import List, Dict, Union
from services.prompts import (
    get_topic_prompt,
    CHUNKING_SETTINGS,
)

class TopicHandler:
    def __init__(self, model=None, tokenizer=None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect_structure(self, text: str) -> List[Dict[str, Union[str, List[str]]]]:
        if not text.strip():
            return [{
                'title': 'General',
                'content': text,
                'level': 1,
                'topics': ['General']
            }]
        topics = self._extract_topics(text)
        sections = self._detect_sections(text)
        if not sections:
            sections = [{
                'title': 'Main Content',
                'content': text,
                'level': 1,
                'topics': topics
            }]
        for section in sections:
            section['topics'] = topics
        return sections

    def _extract_topics(self, text: str) -> List[str]:
        if not self.model or not self.tokenizer:
            return ['General']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            prompt = get_topic_prompt(text[:1000])
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs, max_length=128, num_beams=4, early_stopping=True
                )
                topics_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return [t.strip() for t in topics_text.split(',') if t.strip()] or ['General']
        except Exception:
            return ['General']

    def _detect_sections(self, text: str) -> List[Dict[str, Union[str, int]]]:
        sections = []
        patterns = [
            (r'^Chapter \d+[:.]\s*([^\n]+)', 1),
            (r'^\d+\.\s+([^\n]+)', 2),
            (r'^[A-Z][^.!?\n]+:', 2)
        ]
        for pattern, level in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                title = match.group(1) if '(' in pattern else match.group(0)
                content = self._extract_section_content(text, title)
                if content:
                    sections.append({
                        'title': title.strip(':'),
                        'content': content.strip(),
                        'level': level
                    })
        sections.sort(key=lambda x: text.find(x['content']))
        return sections

    def _extract_section_content(self, text: str, title: str) -> str:
        try:
            title_pattern = re.escape(title)
            matches = list(re.finditer(title_pattern, text))
            if not matches:
                return ""
            start = matches[0].end()
            next_section = float('inf')
            patterns = [
                r'^Chapter \d+[:.]\s*\w+',
                r'^\d+\.\s+\w+',
                r'^[A-Z][^.!?]+:'
            ]
            for pattern in patterns:
                next_match = re.search(pattern, text[start:], re.MULTILINE)
                if next_match:
                    next_section = min(next_section, start + next_match.start())
            return text[start:next_section if next_section < float('inf') else None].strip()
        except Exception:
            return ""

    def chunk_content(self, text: str) -> List[str]:
        if not text.strip():
            return []
        if len(text) <= CHUNKING_SETTINGS['max_chunk_length']:
            return [text]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()] or [text]
        chunks, current_chunk, current_length = [], [], 0
        for paragraph in paragraphs:
            if len(paragraph) > CHUNKING_SETTINGS['max_chunk_length']:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk, current_length = [], 0
                sentences = [s.strip() + '.' for s in re.split(r'(?<=[.!?])\s+', paragraph) if s.strip()]
                temp_chunk, temp_length = [], 0
                for sentence in sentences:
                    if temp_length + len(sentence) > CHUNKING_SETTINGS['max_chunk_length']:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk, temp_length = [sentence], len(sentence)
                    else:
                        temp_chunk.append(sentence)
                        temp_length += len(sentence)
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            elif current_length + len(paragraph) > CHUNKING_SETTINGS['max_chunk_length']:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_length = [paragraph], len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks or [text]
