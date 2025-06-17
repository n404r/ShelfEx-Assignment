import json
import csv
import os
from typing import List, Dict
from datetime import datetime

def export_flashcards(flashcards: List[Dict], export_format: str, subject: str = 'General') -> str:
    if not flashcards:
        raise ValueError("No flashcards to export")
    
    export_dir = os.path.join(os.getcwd(), 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    format_name = export_format.lower().replace(' ', '_')
    
    try:
        if 'json' in format_name:
            return _export_json(flashcards, export_dir, timestamp, subject)
        elif 'csv' in format_name:
            return _export_csv(flashcards, export_dir, timestamp, subject)
        elif 'anki' in format_name:
            return _export_anki(flashcards, export_dir, timestamp, subject)
        elif 'quizlet' in format_name:
            return _export_quizlet(flashcards, export_dir, timestamp, subject)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    except Exception as e:
        raise

def _export_json(cards: List[Dict], export_dir: str, timestamp: str, subject: str) -> str:
    try:
        organized_cards = {}
        for card in cards:
            for topic in card.get('topics', ['General']):
                if topic not in organized_cards:
                    organized_cards[topic] = []
                organized_cards[topic].append({
                    'question': card['question'],
                    'answer': card['answer'],
                    'difficulty': card.get('difficulty', 'medium')
                })
        
        export_data = {
            'subject': subject,
            'total_cards': len(cards),
            'topics': organized_cards,
            'timestamp': timestamp
        }
        
        filename = f"flashcards_{subject.lower()}_{timestamp}.json"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    except Exception as e:
        raise

def _export_csv(cards: List[Dict], export_dir: str, timestamp: str, subject: str) -> str:
    try:
        filename = f"flashcards_{subject.lower()}_{timestamp}.csv"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Topic', 'Question', 'Answer', 'Difficulty'])
            for card in cards:
                topics = ', '.join(card.get('topics', ['General']))
                writer.writerow([
                    topics,
                    card['question'],
                    card['answer'],
                    card.get('difficulty', 'medium')
                ])
        
        return filepath
    except Exception as e:
        raise

def _export_anki(cards: List[Dict], export_dir: str, timestamp: str, subject: str) -> str:
    try:
        filename = f"flashcards_{subject.lower()}_{timestamp}.txt"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for card in cards:
                tags = [subject] + card.get('topics', ['General']) + [card.get('difficulty', 'medium')]
                tags = ' '.join([f"#{tag.replace(' ', '_')}" for tag in tags])
                question = _clean_text(card['question'])
                answer = _clean_text(card['answer'])
                f.write(f"{question};{answer};{tags}\n")
        
        return filepath
    except Exception as e:
        raise

def _export_quizlet(cards: List[Dict], export_dir: str, timestamp: str, subject: str) -> str:
    try:
        filename = f"flashcards_{subject.lower()}_{timestamp}.txt"
        filepath = os.path.join(export_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for card in cards:
                question = _clean_text(card['question'])
                answer = _clean_text(card['answer'])
                f.write(f"{question}\t{answer}\n")
        
        return filepath
    except Exception as e:
        raise

def _clean_text(text: str) -> str:
    text = text.replace('"', "'")
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\', '/')
    text = text.replace(';', ',')
    return ' '.join(text.split())
