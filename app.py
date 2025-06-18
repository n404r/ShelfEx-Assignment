import streamlit as st
import logging
import os
import json
from services.parser import parse_input
from services.export_utils import export_flashcards
import time
from config import SUBJECTS, DIFFICULTY_LEVELS, DEFAULT_DIFFICULTY, DEFAULT_SUBJECT, EXPORT_FORMATS, DEFAULT_EXPORT_FORMAT, MODEL_NAME

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initialize_generator():
    if 'generator' not in st.session_state:
        try:
            from services.generator import FlashcardGenerator
            with st.spinner('Loading model...'):
                st.session_state.generator = FlashcardGenerator()
        except Exception as e:
            logger.error(f"Error initializing generator: {str(e)}")
            st.error(f"Error loading model: {str(e)}")
            st.stop()

def initialize_session_state():
    if 'generate_requested' not in st.session_state:
        st.session_state.generate_requested = False
    if 'generation_in_progress' not in st.session_state:
        st.session_state.generation_in_progress = False
    if 'stop_generation' not in st.session_state:
        st.session_state.stop_generation = False
    if 'export_format' not in st.session_state:
        st.session_state.export_format = DEFAULT_EXPORT_FORMAT
    if 'edited_cards' not in st.session_state:
        st.session_state.edited_cards = {}
    if 'show_stop_button' not in st.session_state:
        st.session_state.show_stop_button = False

def trigger_generation():
    st.session_state.stop_generation = False
    st.session_state.generation_in_progress = False
    st.session_state.generate_requested = True
    st.session_state.show_stop_button = False

def stop_generation():
    if st.session_state.generation_in_progress:
        st.session_state.stop_generation = True
        st.session_state.generate_requested = False

def update_flashcard(card_id, field, new_value):
    try:
        if 'flashcards' in st.session_state and st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards):
                if f"{i}" == card_id:
                    key = f"{field}_{card_id}"
                    if key in st.session_state:
                        current_value = st.session_state[key]
                        st.session_state.flashcards[i][field] = current_value
                        
                        if 'edited_cards' not in st.session_state:
                            st.session_state.edited_cards = {}
                        st.session_state.edited_cards[card_id] = True
    except Exception as e:
        logger.error(f"Error updating flashcard: {str(e)}")

def main():
    st.set_page_config(
        page_title="AI FlashCard Generator",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI Flashcard Generator")
    st.markdown("Made with ❤️ by Nischay Raj")
    st.markdown(f"`` Current Model: {MODEL_NAME} ``")

    
    try:
        initialize_generator()
        initialize_session_state()
        
        with st.sidebar:
            st.header("Input")
            uploaded_file = st.file_uploader("Upload PDF/TXT/DOCX file only", type=["pdf", "txt", "docx"])
            text_input = st.text_area(
                "Or paste your text here",
                height=300
            )

            st.header("Settings")
            subject = st.selectbox(
                "Subject",
                SUBJECTS,
                index=SUBJECTS.index(DEFAULT_SUBJECT) if DEFAULT_SUBJECT in SUBJECTS else 0,
            )

            difficulty = st.selectbox(
                "Difficulty",
                DIFFICULTY_LEVELS,
                index=DIFFICULTY_LEVELS.index(DEFAULT_DIFFICULTY) if DEFAULT_DIFFICULTY in DIFFICULTY_LEVELS else 1
            )

            generate_btn = st.button("Generate Flashcards", type="primary", on_click=trigger_generation)

        should_generate = st.session_state.generate_requested and not st.session_state.generation_in_progress

        if not should_generate:
            if 'flashcards' not in st.session_state:
                st.info("Upload a file or paste text, then click 'Generate Flashcards'")
                st.stop()
        else:
            if not uploaded_file and not text_input.strip():
                st.error("Please provide input text or upload a file")
                st.session_state.generate_requested = False
                st.stop()

            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0, "Starting...")

            message_placeholder = st.empty()
            message_placeholder.error("Processing - Click 'Stop Generation' below to cancel if needed")
            stop_col1, stop_col2 = st.columns([1, 4])
            with stop_col1:
                st.session_state.show_stop_button = True
                stop_btn_placeholder = st.empty()
                stop_btn_placeholder.button("Stop Generation", type="primary", on_click=stop_generation)

            try:
                st.session_state.generation_in_progress = True
                st.session_state.stop_generation = False

                progress_bar.progress(20, "Processing input...")
                content = parse_input(uploaded_file if uploaded_file else None, text_input)
                if not content:
                    message_placeholder.error("Could not process input text")
                    progress_placeholder.empty()
                    stop_btn_placeholder.empty()
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False
                    return

                progress_bar.progress(40, "Generating flashcards...")
                
                if st.session_state.stop_generation:
                    progress_placeholder.empty()
                    message_placeholder.warning("Generation stopped by user")
                    stop_btn_placeholder.empty()
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False
                    return

                def get_stop_flag():
                    return st.session_state.stop_generation

                if st.session_state.stop_generation:
                    progress_placeholder.empty()
                    message_placeholder.warning("Generation stopped by user")
                    stop_btn_placeholder.empty()
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False
                    st.stop()
                    
                flashcards = st.session_state.generator.generate_flashcards(
                    content,
                    difficulty=difficulty,
                    subject=subject,
                    stop_callback=get_stop_flag
                )

                if st.session_state.stop_generation:
                    progress_placeholder.empty()
                    message_placeholder.warning("Generation stopped by user. You can generate again.")
                    stop_btn_placeholder.empty()
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False

                elif not flashcards:
                    message_placeholder.warning("No flashcards could be generated. Please try with different input.")
                    progress_placeholder.empty()
                    stop_btn_placeholder.empty()
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False
                
                else:
                    progress_bar.progress(80, "Organizing results...")
                    st.session_state.generate_requested = False
                    st.session_state.generation_in_progress = False
                    st.session_state.stop_generation = False

                    st.session_state.flashcards = flashcards

                    topics = {}
                    for card in flashcards:
                        for topic in card.get('topics', ['General']):
                            if topic not in topics:
                                topics[topic] = []
                            topics[topic].append(card)

                    message_placeholder.success(f"Generated {len(flashcards)} flashcards!")
                    progress_bar.progress(100, "Complete!")
                    time.sleep(1)
                    progress_placeholder.empty()
                    message_placeholder.empty()
                    stop_btn_placeholder.empty()
                    st.session_state.show_stop_button = False
                    
            except Exception as e:
                progress_placeholder.empty()
                message_placeholder.error(f"An error occurred during processing: {str(e)}")
                stop_btn_placeholder.empty()                
                logger.error(f"Processing error: {str(e)}")
                st.session_state.generate_requested = False
                st.session_state.generation_in_progress = False
                st.session_state.stop_generation = False
                
        if 'flashcards' in st.session_state and st.session_state.flashcards:
            topics = {}
            for card in st.session_state.flashcards:
                for topic in card.get('topics', ['General']):
                    if topic not in topics:
                        topics[topic] = []
                    topics[topic].append(card)
                    
            if len(topics) > 1:
                selected_topic = st.selectbox("Select Topic", list(topics.keys()))
                topic_cards = topics[selected_topic]
                st.subheader(f"Topic: {selected_topic}")
            else:
                selected_topic = list(topics.keys())[0]
                topic_cards = st.session_state.flashcards
                st.subheader("Generated Flashcards")
                
            for idx, card in enumerate(topic_cards):
                card_id = f"{idx}"
                
                with st.container():
                    st.markdown(f"Question: {idx+1} ")
                    edited_question = st.text_area(
                        "Edit question",
                        card['question'],
                        height=100,
                        key=f"q_{card_id}",
                        on_change=update_flashcard,
                        args=(card_id, "question", None),
                        label_visibility="collapsed"
                    )
                    
                    st.markdown("**Answer:**")
                    edited_answer = st.text_area(
                        "Edit answer",
                        card['answer'],                        height=150,
                        key=f"a_{card_id}",
                        on_change=update_flashcard,
                        args=(card_id, "answer", None),
                        label_visibility="collapsed"
                    )
                    st.markdown("---")
            
            with st.sidebar:
                st.markdown("---")
                st.header("Export")
                export_format = st.selectbox(
                    "Format",                    EXPORT_FORMATS,
                    index=EXPORT_FORMATS.index(DEFAULT_EXPORT_FORMAT) if DEFAULT_EXPORT_FORMAT in EXPORT_FORMATS else 0
                )
                st.session_state.export_format = export_format
                
                if hasattr(st.session_state, 'edited_cards') and st.session_state.edited_cards:
                    st.success(f"✅ {len(st.session_state.edited_cards)} cards have been edited")
                
                if st.button("Export Flashcards"):
                    try:
                        output_path = export_flashcards(
                            st.session_state.flashcards,
                            st.session_state.export_format,
                            subject=subject
                        )
                        with open(output_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                "Download Flashcards",
                                f.read(),
                                file_name=os.path.basename(output_path),
                                mime="text/plain"
                            )
                        st.success("Ready for download!")
                    except Exception as e:                        
                        st.error(f"Export failed: {str(e)}")
                        logger.error(f"Export error: {str(e)}")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        st.error("Application failed to start.")
