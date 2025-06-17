from typing import Dict, List
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DIFFICULTY_LEVELS, GENERATION_PARAMS, MAX_CHUNK_LENGTH


QUESTION_GENERATION_PROMPTS = {
    'Easy': [
        "Create a basic question about a key concept from this text ending with a question mark.",
        "Generate a simple factual question based on the main ideas in this text.",
        "What factual question would test knowledge of this material?"
    ],
    'Medium': [
        "Create a question about relationships between concepts in this text ending with a question mark.",
        "Write a question about how or why something happens based on this text.",
        "Generate a question about the significance or function of a concept in this text."
    ],
    'Hard': [
        "Create a challenging question requiring critical thinking about concepts in this text.",
        "Generate a question connecting multiple concepts that requires deep understanding.",
        "Write a question asking for analysis or application of the concepts in this text."
    ]
}

SUBJECT_TEMPLATES = {
    'Biology': {
        'cell': "Describe {component} in cells.",
        'process': "How does {biological_process} work?",
        'function': "What is the role of {component}?"
    },
    'Physics': {
        'law': "Explain {law_name}.",
        'application': "How is {concept} applied?"
    },
    'Computer Science': {
        'algorithm': "How does the {algorithm} algorithm work?",
        'data_structure': "Describe {data_structure}.",
        'system': "Explain {system}."
    },
    'Mathematics': {
        'theorem': "Explain {theorem_name}.",
        'concept': "What is {concept}?",
        'application': "How is {concept} used?"
    }
}

ANSWER_GENERATION_PROMPTS = {
    'Easy': {
        'template': (
            "Answer this flashcard question based on the context.\n"
            "Keep it simple and brief (1-2 sentences).\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer: "
        ),
        'style': "simple, factual"
    },
    'Medium': {
        'template': (
            "Answer this flashcard question based on the context.\n"
            "Include key points in 2-3 sentences.\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer: "
        ),
        'style': "explanatory"
    },
    'Hard': {
        'template': (
            "Answer this flashcard question based on the context.\n"
            "Provide a detailed explanation with deeper implications (3-5 sentences).\n\n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Answer: "
        ),
        'style': "analytical"
    }
}


TEXT_PROCESSING_PROMPTS = {
    'topic_extraction': (
        "List main topics:\n"
        "Text: {text}\n"
        "Topics:"
    ),
    'section_detection': (
        "List sections:\n"
        "Text: {text}\n"
        "Sections:"
    )
}

# Validation rules
VALIDATION_RULES = {
    'question': {
        'Easy': {'min_length': 5, 'max_length': 15, 'must_end_with': '?'},
        'Medium': {'min_length': 8, 'max_length': 20, 'must_end_with': '?'},
        'Hard': {'min_length': 10, 'max_length': 25, 'must_end_with': '?'}
    },
    'answer': {
        'Easy': {'min_words': 10, 'max_words': 50, 'min_sentences': 1},
        'Medium': {'min_words': 20, 'max_words': 100, 'min_sentences': 2},
        'Hard': {'min_words': 40, 'max_words': 150, 'min_sentences': 3}
    }
}

CHUNKING_SETTINGS = {
    'min_chunk_length': 100, 
    'max_chunk_length': MAX_CHUNK_LENGTH,
    'min_sentences': 1,          # Reduced for more reliable chunking
    'max_overlap': 50,           # Allow some overlap for context continuity
    'preserve_lists': True       # Preserve list structure in chunks
}

def get_question_prompt(difficulty: str, subject: str = None) -> List[str]:
    """Get question generation prompts."""
    try:
        if difficulty not in DIFFICULTY_LEVELS:
            raise KeyError(f"Invalid difficulty level. Must be one of: {DIFFICULTY_LEVELS}")
        
        if difficulty not in QUESTION_GENERATION_PROMPTS:
            raise KeyError(f"Missing prompts for {difficulty}. Available: {list(QUESTION_GENERATION_PROMPTS.keys())}")
            
        prompts = QUESTION_GENERATION_PROMPTS[difficulty].copy()
        
        if subject and subject != 'General':
            if subject not in SUBJECT_TEMPLATES:
                raise KeyError(f"Invalid subject. Must be one of: General, {list(SUBJECT_TEMPLATES.keys())}")
            
            subject_templates = list(SUBJECT_TEMPLATES[subject].values())
            prompts.extend(subject_templates[:3])
            
        return prompts
    except Exception as e:
        return QUESTION_GENERATION_PROMPTS['Medium']

def get_answer_prompt(difficulty: str) -> str:
    """Get answer generation prompt."""
    try:
        if difficulty not in DIFFICULTY_LEVELS:
            raise KeyError(f"Invalid difficulty level. Must be one of: {DIFFICULTY_LEVELS}")
            
        if difficulty not in ANSWER_GENERATION_PROMPTS:
            raise KeyError(f"Missing answer prompts for {difficulty}")
            
        return ANSWER_GENERATION_PROMPTS[difficulty]['template']
    except Exception as e:
        return ANSWER_GENERATION_PROMPTS['Medium']['template']

def get_topic_prompt(text: str) -> str:
    return TEXT_PROCESSING_PROMPTS['topic_extraction'].format(text=text)

def get_processing_prompt(prompt_type: str) -> str:
    return TEXT_PROCESSING_PROMPTS.get(prompt_type, "")

def get_validation_rules(difficulty: str) -> Dict:
    try:
        # Validate that difficulty is one of the standard levels
        if difficulty not in DIFFICULTY_LEVELS:
            raise KeyError(f"Invalid difficulty level. Must be one of: {DIFFICULTY_LEVELS}")
            
        # Ensure difficulty is in the validation rules
        if difficulty not in VALIDATION_RULES['question'] or difficulty not in VALIDATION_RULES['answer']:
            raise KeyError(f"Missing validation rules for {difficulty}. Available: {list(VALIDATION_RULES['question'].keys())}")
            
        return {
            'question': VALIDATION_RULES['question'][difficulty],
            'answer': VALIDATION_RULES['answer'][difficulty]
        }
    except Exception as e:
        return {  # Fallback to medium difficulty
            'question': VALIDATION_RULES['question']['Medium'],
            'answer': VALIDATION_RULES['answer']['Medium']
        }

def get_generation_settings(difficulty: str) -> Dict:
    if difficulty not in DIFFICULTY_LEVELS:
        raise KeyError(f"Invalid difficulty level. Must be one of: {DIFFICULTY_LEVELS}")
        
    # Ensure difficulty is in the generation settings
    if difficulty not in GENERATION_PARAMS:
        difficulty = 'Medium'
        
    settings = GENERATION_PARAMS[difficulty].copy()
    
    generation_params = {
        'max_length': settings['max_length'],
        'temperature': settings['temperature'],
        'num_beams': settings['num_beams'],
        'do_sample': True,  #varied output
        'num_return_sequences': 1,
        'early_stopping': True
    }
    
    generation_params['additional_settings'] = settings.get('additional_settings', {})
    
    return generation_params
