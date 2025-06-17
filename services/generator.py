import logging
import nltk
import os
import torch
import time
import sys
import re
import random
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from services.prompts import (
    get_question_prompt,
    get_answer_prompt,
    get_topic_prompt,
    get_validation_rules,
    get_generation_settings,
    CHUNKING_SETTINGS
)
from services.topic_handler import TopicHandler

# basic logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console)

class FlashcardGenerator:
    def __init__(self):
        """initialize generator"""
        try:
            # setup nltk
            nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
            os.makedirs(nltk_data_path, exist_ok=True)
            nltk.data.path.append(nltk_data_path)
            nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
            
            # setup device
            use_gpu_available = USE_GPU and torch.cuda.is_available()
            self.device = torch.device('cuda' if use_gpu_available else 'cpu')
            logger.info(f"using device: {self.device}")
            
            # load model
            self.model_name = MODEL_NAME
            if USE_GPU and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto',
                use_cache=True,
            )
            
            self.model.eval()
            
            # initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=MODEL_MAX_LENGTH,
                padding_side='right',
                truncation_side='right'
            )
              # setup topic handler
            self.topic_handler = TopicHandler(self.model, self.tokenizer, self.device)
            
        except Exception as e:
            logger.error(f"init error: {str(e)}")
            raise
            
    def generate_flashcards(
        self,
        text: str,
        difficulty: str = 'Medium',
        subject: str = 'General',
        stop_callback: callable = None
    ) -> List[Dict[str, str]]:
        """generate flashcards from text"""
        try:
            if not text.strip():
                logger.warning("empty text")
                return []
            
            self.difficulty = difficulty
            self.subject = subject
            self.stop_callback = stop_callback
            
            # get settings
            settings = get_generation_settings(difficulty)
            validation = get_validation_rules(difficulty)
            
            # manage memory
            if torch.cuda.is_available():
                if CLEAR_CUDA_CACHE:
                    torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
            
            # check stop flag
            if self._should_stop():
                return []
            
            # process sections
            sections = self.topic_handler.detect_structure(text)
            flashcards = []
            
            # set target count
            min_required = settings.get('additional_settings', {}).get('min_cards', 10)     
            for section in sections:
                if self._should_stop():
                    break
                
                section_cards = self._generate_section_cards(
                    section['content'],
                    section['topics'],
                    settings,
                    validation
                )
                flashcards.extend(section_cards)            # check count
                current_count = len(flashcards)
                if current_count >= min_required:
                    break
            
            # generate more if needed
            if len(flashcards) < min_required and not self._should_stop():
                for section in sections:
                    additional_cards = self._generate_additional_cards(
                        section['content'],
                        section['topics'],
                        settings,
                        validation,
                        min_required - len(flashcards)
                    )
                    flashcards.extend(additional_cards)
                    if len(flashcards) >= min_required:
                        break
            return flashcards
        except Exception as e:
            logger.error(f"generation error: {str(e)}")
            return []
            
    def _should_stop(self) -> bool:
        """check stop condition"""
        if self.stop_callback and callable(self.stop_callback):
            try:                # check stop flag
                stop_requested = self.stop_callback()
                if stop_requested:
                    return True
            except Exception as e:
                logger.error(f"stop callback error: {str(e)}")
        return False
    def _generate_section_cards(
        self,
        text: str,
        topics: List[str],
        settings: Dict,
        validation: Dict
    ) -> List[Dict]:
        """generate section cards"""
        try:
            cards = []
            # create chunks
            chunks = self.topic_handler.chunk_content(text)
            
            # use all text if no chunks
            if not chunks:
                chunks = [text]
            
            if self._should_stop():
                return cards            
            total_chunks = len(chunks)
            
            # get min cards
            min_cards = settings.get('additional_settings', {}).get('min_cards', MIN_CARDS_TOTAL)
            
            # set cards per chunk
            cards_per_chunk = 2
            
            # determine chunks to process
            max_chunks_to_process = min(10, (min_cards // cards_per_chunk))
            # use initial chunks
            prioritized_indices = list(range(min(max_chunks_to_process, total_chunks)))
            
            used_chunks = 0
            questions_set = set() # unique questions
            
            for idx in prioritized_indices:
                if self._should_stop():
                    break
                
                chunk = chunks[idx]
                questions = self._generate_questions(chunk)
                
                if questions:
                    used_chunks += 1
                    unique_questions = []
                    for q in questions:
                        normalized_q = q.strip().lower()
                        is_duplicate = False
                        for existing_q in questions_set:
                            # similarity check
                            words1 = set(normalized_q.split())
                            words2 = set(existing_q.split())
                            if len(words1.intersection(words2)) / max(1, len(words1.union(words2))) > 0.7:
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
                            questions_set.add(normalized_q)
                            unique_questions.append(q)
                      # check stop
                    if self._should_stop():
                        break
                    
                    # generate answers
                    for question in unique_questions[:cards_per_chunk]:
                        # check stop
                        if self._should_stop():
                            break
                            
                        # generate answer
                        answer = self._generate_answer(question, chunk)
                        
                        # create card
                        card = {
                            'question': question.strip(),
                            'answer': answer.strip(),
                            'topics': topics,
                            'difficulty': self.difficulty,
                            'subject': self.subject,
                            'chunk_index': idx
                        }
                        cards.append(card)                # check if enough cards
                if len(cards) >= min_cards:
                    break
            
            return cards
            
        except Exception as e:
            logger.error(f"section error: {str(e)}")
            return []
            
    def _generate_questions(self, context: str) -> List[str]:
        """Generate questions from context."""
        
        try:
            questions = []
            prompts = get_question_prompt(self.difficulty, self.subject)
            
            generation_settings = get_generation_settings(self.difficulty)
            
            # prepare generation params
            generation_params = {k: v for k, v in generation_settings.items() 
            if k != 'additional_settings'}
                
            if self._should_stop():
                return []
                  # prepare prompts
            formatted_prompts = []
            
            # use first 3 prompts
            for prompt in prompts[:3]:
                # format prompt
                full_prompt = f"""PROMPT: {prompt}

TEXT:
{context}"""
                formatted_prompts.append(full_prompt)
                
            # generate questions
            if formatted_prompts:# prepare params
                batch_generation_params = generation_params.copy()
                
                # handle hard difficulty
                if self.difficulty == "Hard":
                    hard_settings = GENERATION_PARAMS.get("Hard", {})
                    batch_generation_params["temperature"] = batch_generation_params.get("temperature", 0.7)
                    hard_settings.get("temperature", 0.9)
                    batch_generation_params["max_length"] = max(batch_generation_params.get("max_length", 256), 
                                                            hard_settings.get("max_length", 480))
                    batch_generation_params["num_beams"] = max(batch_generation_params.get("num_beams", 4), 
                                                           hard_settings.get("num_beams", 5))
                
                batch_results = self._batch_generate(formatted_prompts, batch_generation_params)
                
                # process results
                for i, question in enumerate(batch_results):
                    # validate question
                    if question and len(question.strip()) > 10:
                        questions.append(question)
                    else:
                        pass
              # try direct approach if no questions
            if not questions and not self._should_stop():
                logger.warning("trying direct prompt")
                  # direct prompt
                direct_prompt = f"""Create one question about this text:
{context}"""
                # adjust parameters
                final_params = generation_params.copy()
                final_params["max_length"] = min(final_params.get("max_length", 128), 64)
                final_params["no_repeat_ngram_size"] = 2
                
                try:
                    # use batch generate
                    final_results = self._batch_generate([direct_prompt], final_params)
                    
                    if final_results and len(final_results) > 0:
                        final_question = final_results[0]
                        
                        # add if valid
                        if final_question and len(final_question.strip()) > 5:
                            questions.append(final_question)
                            logger.info(f"generated fallback question")
                except Exception as e:
                    logger.error(f"question generation error: {str(e)}")
                
                # create fallback question
                if not questions and not self._should_stop():
                    context_start = context.strip()[:30].replace('\n', ' ').strip()
                    contextual_question = f"What does the text tell us about: {context_start}...?"
                    questions = [contextual_question]
            
            return questions
        except Exception as e:            
            logger.error(f"question error: {str(e)}")
            return []
            
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question."""
        try:            # check for stop
            if self._should_stop():
                return ""
                
            # clean context
            def clean_text_for_prompt(text):
                """clean text for prompt"""
                import re
                text = text.encode("ascii", errors="ignore").decode()
                text = re.sub(r'[^A-Za-z0-9\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()
            
            # prepare context
            clean_context = clean_text_for_prompt(context)
              # get template and format it 
            prompt_template = get_answer_prompt(self.difficulty)
            
            # simple formatting
            prompt = prompt_template.format(
                context=clean_context,
                question=question
            )            # get generation settings
            generation_settings = get_generation_settings(self.difficulty)
            
            # prepare params
            generation_params = {k: v for k, v in generation_settings.items() 
                            if k != 'additional_settings'}
            
            # generate answer
            results = self._batch_generate([prompt], generation_params)
            
            if not results:
                logger.warning("no answer generated")
                return ""
                
            answer = results[0]
            answer_text = answer.strip()
              # use answer as is
            answer_text = answer.strip()
            
            if not answer_text:
                logger.warning("empty answer")
            return answer_text
                
        except Exception as e:
            logger.error(f"answer error: {str(e)}")
            return ""
            
    def _generate_additional_cards(
        self,
        text: str,
        topics: List[str],
        settings: Dict,
        validation: Dict,
        cards_needed: int
    ) -> List[Dict]:
        try:
            if self._should_stop():
                return []
                
            chunk_settings = CHUNKING_SETTINGS.copy()
            chunk_settings['max_chunk_length'] = min(
                chunk_settings['max_chunk_length'],
                400
            )
            
            chunks = []
            current_pos = 0
            chunk_length = chunk_settings['max_chunk_length']
            
            while current_pos < len(text):
                chunk = text[current_pos:current_pos + chunk_length]
                if len(chunk) >= chunk_settings['min_chunk_length']:
                    chunks.append(chunk)
                current_pos += chunk_length // 2
            
            additional_cards = []
            cards_per_chunk = max(2, cards_needed // (len(chunks) or 1))
            for i, chunk in enumerate(chunks):
                if self._should_stop() or len(additional_cards) >= cards_needed:
                    break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                questions = self._generate_questions(chunk)
                questions = list(set(questions))
                
                if self._should_stop():
                    break
                
                for question in questions[:cards_per_chunk]:
                    if self._should_stop():
                        break
                        
                    answer = self._generate_answer(question, chunk)
                    
                    card = {
                        'question': question.strip(),
                        'answer': answer.strip(),
                        'topics': topics,
                        'difficulty': self.difficulty,
                        'subject': self.subject
                    }
                    additional_cards.append(card)
                    
                    if len(additional_cards) >= cards_needed:
                        break            
                    return additional_cards
            
        except Exception as e:
            logger.error(f"additional cards error: {str(e)}")
            return []
        
    def _batch_generate(self, prompts, generation_params=None):
        """generate batch outputs"""
        try:
            if not prompts:
                return []
                  # check for stop
            if self._should_stop():
                return []
                
            # use default params if needed
            if generation_params is None:
                generation_params = {}
                
            # create batch inputs
            batch_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MODEL_MAX_LENGTH
            ).to(self.device)
            
            batch_size = len(prompts)
            
            optimal_batch_size = BATCH_SIZE if USE_GPU and torch.cuda.is_available() else 4
            sub_batch_size = min(optimal_batch_size, batch_size)
            
            results = []
            
            # try processing in one batch
            try:
                # check for stop
                if self._should_stop():
                    return []
                  
                # process batch
                with torch.no_grad():
                    # generate outputs
                    start_time = time.time()
                    outputs = self.model.generate(
                        **batch_inputs,
                        **generation_params
                    )                    
                    end_time = time.time()
                    # batch completed
                    
                    # decode outputs
                    for i, output in enumerate(outputs):
                        if self._should_stop():
                            break
                        
                        # decode output
                        decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                        results.append(decoded)
            
            except RuntimeError as e:
                logger.warning(f"batch failed: {e}, using smaller batches")
                
                for i in range(0, batch_size, sub_batch_size):                    # check for stop
                    if self._should_stop():
                        break
                        
                    end_idx = min(i + sub_batch_size, batch_size)
                    sub_prompts = prompts[i:end_idx]
                    
                    # tokenize sub-batch
                    sub_inputs = self.tokenizer(
                        sub_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=MODEL_MAX_LENGTH
                    ).to(self.device)
                  
                    # generate for sub-batch
                    with torch.no_grad():
                        sub_outputs = self.model.generate(
                            **sub_inputs,
                            **generation_params
                        )
                        
                        # decode outputs
                        for j, output in enumerate(sub_outputs):
                            if self._should_stop():
                                break
                            decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                            results.append(decoded)
            
            # cleanup gpu memory
            if torch.cuda.is_available() and CLEAR_CUDA_CACHE:
                torch.cuda.empty_cache()
                    
            return results
        except Exception as e:
            logger.error(f"batch error: {str(e)}")
            return [""] * len(prompts)
