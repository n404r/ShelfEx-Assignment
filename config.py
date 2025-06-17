
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
