# ⚡ LLM-powered Flashcard Generator

LLM-powered Flashcard Generator tool that transforms educational content into effective question-answer flashcards. It processes text or documents to create categorized, subject-specific flashcards with configurable difficulty levels. Built with Streamlit and optimized for GPU acceleration.

---

## 🎯Objective

This project demonstrates:

- **LLM Integration**: Utilizes Hugging Face models to power flashcard generation
- **Educational Content Processing**: Ingests textbook excerpts, lecture notes, and educational materials
- **Automatic Flashcard Generation**: Extracts relevant Q&A pairs from content
- **User Interface**: Clean, simple Streamlit UI for interaction and card management
- **Export Functionality**: Multiple export formats for compatibility with study systems

The system meets all core requirements and implements several bonus functionalities from the assignment specifications.

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch the App

```bash
streamlit run app.py
```

---

## 🧠 Core Features

### ✅ Input Processing
* Accepts raw **text input** or **PDF/TXT/DOCX file uploads** as per assignment requirements
* Performs **text cleaning** and **chunking** for optimal processing of educational content
* Supports **subject detection** and **topic-based organization** for better flashcard categorization

### 🃏 Flashcard Generation
* Three difficulty levels: `Easy`, `Medium`, `Hard`
* Subject-specific templates for:
  * General
  * Computer Science
  * Biology
  * Physics
* Generates **minimum of 10-15 flashcards** per input submission (meets core requirement)
* Each flashcard contains a clear question and factually correct answer
* **GPU-optimized** processing for faster generation
* **Editable cards** - make manual adjustments before export (bonus functionality)
* Clean **stop/start functionality** during generation

### 📤 Export Options (Bonus Functionality)
* **Edit cards** directly in the UI before export
* Export formats:
  * `JSON`: Organized by topic
  * `CSV`: Spreadsheet-compatible
  * `Anki`: Ready for import

---

## ⚙️ Configuration

The application is easily configurable through the `config.py` file:

```python

MODEL_NAME = "google/flan-t5-large"

# front end settings
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
DEFAULT_DIFFICULTY = "Medium"
SUBJECTS = ["General", "Biology", "Physics", "Computer Science"]
DEFAULT_SUBJECT = "General"

# generator settings
MAX_CARDS_PER_TOPIC = 20  
MIN_CARDS_TOTAL = 10 
MAX_CHUNK_LENGTH = 1400    
BATCH_SIZE = 20           
USE_GPU = True       

# difficulty parameters
GENERATION_PARAMS = {
    "Easy": {
        "max_length": 128,
        "temperature": 0.7,
        "num_beams": 2,
        "additional_settings": {
            "min_cards": 10,
            "max_cards": 14
        }
    },
    "Medium": {
        "max_length": 186,
        "temperature": 0.8,
        "num_beams": 3,
        "additional_settings": {
            "min_cards": 10,
            "max_cards": 13
        }
    },
    "Hard": {
        "max_length": 248, 
        "temperature": 0.85, 
        "num_beams": 4,
        "additional_settings": {
            "min_cards": 10,
            "max_cards": 12
        }
    }
}

# export parameters
EXPORT_FORMATS = ["JSON", "CSV", "Anki"]
DEFAULT_EXPORT_FORMAT = "JSON"
EXPORTS_DIR = "exports"

MODEL_MAX_LENGTH = 384  # output lenght
DO_SAMPLE = True        # varied output
EARLY_STOPPING = True

CLEAR_CUDA_CACHE = True  # always for testing
MEMORY_FRACTION = 0.9 
```


## ⚠️ Troubleshooting

* **Memory issues**: Reduce `BATCH_SIZE` in config.py
* **Slow generation**: Ensure GPU is being utilized properly
* **Quality issues**: Try a different difficulty level or subject

---
### Input Expectations

* **Text Content**: Educational material from textbooks, lecture notes, articles, or other learning resources
* **File Formats**: Supports .txt, .pdf and .docx uploads.
* **Subject Selection**: Optional subject specification helps tailor flashcards to specific domains

### Output Expectations

#### Flashcards:

* Minimum of **10–15 cards** per input submission
* Format: **Question** – **Answer**
* Questions are clear and concise
* Answers are factually correct and self-contained

```json
{
  "subject": "General",
  "total_cards": 10,
  "topics": {
    "Algorithm": [
      {
        "question": "What is the name of the book that discusses algorithms and problem solving in computer science?",
        "answer": "Foundations of Algorithms and Problem Solving in Computer Science.",
        "difficulty": "Medium"
      },
      {
        "question": "What is the purpose of this article?",
        "answer": "The purpose of this article is to teach the basics of algorithms and problem solving in computer science.",
        "difficulty": "Medium"
      },
      {
        "question": "What are the characteristics of an algorithm?",
        "answer": "An algorithm must have several important characteristics First it must be finite it should complete in a limited number of steps Second it must be welldefined meaning each instruction must be clear and unambiguous Third it must produce one or more outputs after receiving zero or more inputs",
        "difficulty": "Medium"
      },
      {
        "question": "What is the purpose of an algorithm?",
        "answer": "An algorithm is a precise stepbystep set of instructions used to carry out a task or solve a problem.",
        "difficulty": "Medium"
      },
      {
        "question": "What is the upper bound of an algorithm's growth rate?",
        "answer": "BigO notation describes the upper bound of an algorithm's growth rate.",
        "difficulty": "Medium"
      },
      {
        "question": "How is time complexity measured?",
        "answer": "Time complexity refers to the amount of time an algorithm takes to run as a function of the input size.",
        "difficulty": "Medium"
      },
      {
        "question": "What is the difference between a quadratic algorithm and a log n algorithm?",
        "answer": "A quadratic algorithm On2 takes significantly longer as input size increases.",
        "difficulty": "Medium"
      },
      {
        "question": "Which algorithm is used to find data within a structure?",
        "answer": "Search algorithms are used to find data.",
        "difficulty": "Medium"
      },
      {
        "question": "Which algorithm finds the shortest path in weighted graphs?",
        "answer": "Dijkstras algorithm.",
        "difficulty": "Medium"
      },
      {
        "question": "What are some examples of algorithms used in machine learning?",
        "answer": "Dijkstras algorithm finds the shortest path in weighted graphs. Kruskals and Prims algorithms are used to find minimum spanning trees. Regular expressions also play a significant role in string manipulation and data validation.",
        "difficulty": "Medium"
      }
    ]
  },
  "timestamp": "20250617_195302"
}
```
#### Organization:

* Grouped by **topics** 
* Hierarchical and logically related
* Structure preservation from original content

---

## 🔍 Validation Criteria

### ✅ Questions

* Clear, grammatically correct
* End with a **question mark**
* Match selected **difficulty level**

### ✅ Answers

* Factually accurate
* Self-contained and contextual

### ✅ Topics

* Logically grouped
* Relevant to subject
* Intelligently extracted

---

## ⚙️ CUDA + GPU Support

### Enable CUDA Acceleration

* GPU will be automatically detected if available
* Uses PyTorch's `cuda` for model inference

### Checklist

* [x] Ensure `torch.cuda.is_available()` returns `True`
* [x] Log GPU usage to console
* [x] Monitor VRAM during inference

---

## 🛠 Troubleshooting & Export Tips

### Model Loading

* Check for:

  * CUDA availability
  * Sufficient VRAM
  * Model cache integrity

### Export Errors

* Validate:

### Maintenance

* Regularly update:

  * Python dependencies
  * HuggingFace models
  * Prompt.py templates and logic

---


## 👨‍💻 Author

Created by Nischay Raj as part of the LLM-powered Flashcard Generator assignment.

---
