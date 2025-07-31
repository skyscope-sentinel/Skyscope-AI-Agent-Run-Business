#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Voice & Multimodal Interface
================================================================

This module provides comprehensive voice control and multimodal interface capabilities
for the Skyscope Sentinel Intelligence AI system, enabling natural interaction through
speech, vision, gestures, and other modalities.

Features:
- Speech-to-text with multiple engines (Whisper, Google Speech, Azure)
- Natural language understanding and intent recognition
- Voice commands for all system functions
- Text-to-speech with natural voices and emotion
- Computer vision integration for visual context
- Gesture recognition for touchless control
- Emotion detection from voice, text and facial expressions
- Real-time translation between multiple languages
- Accessibility features for diverse user needs
- VR/AR support for immersive interfaces
- Brain-computer interface compatibility
- Multi-language support with dialect detection
- Voice biometrics for secure authentication

Dependencies:
- speech_recognition, pyttsx3, openai-whisper
- transformers, pytorch, tensorflow
- opencv-python, mediapipe
- azure-cognitiveservices-speech, google-cloud-speech
- nltk, spacy, gensim
- pydub, sounddevice, pyaudio
- langchain, langdetect
"""

import os
import sys
import json
import time
import uuid
import queue
import base64
import logging
import asyncio
import datetime
import threading
import traceback
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque

# Audio processing
try:
    import pyaudio
    import wave
    import sounddevice as sd
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio libraries not available. Audio features will be limited.")

# Speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning("Speech recognition library not available. Speech-to-text will be limited.")

# Whisper (OpenAI)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Whisper not available. OpenAI Whisper speech recognition will be disabled.")

# Text-to-speech
try:
    import pyttsx3
    TTS_BASIC_AVAILABLE = True
except ImportError:
    TTS_BASIC_AVAILABLE = False
    logging.warning("pyttsx3 not available. Basic text-to-speech will be limited.")

# Azure Speech Services
try:
    import azure.cognitiveservices.speech as azure_speech
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    logging.warning("Azure Speech SDK not available. Azure speech services will be disabled.")

# Google Cloud Speech
try:
    from google.cloud import speech as google_speech
    from google.cloud import texttospeech as google_tts
    GOOGLE_SPEECH_AVAILABLE = True
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    logging.warning("Google Cloud Speech not available. Google speech services will be disabled.")

# Natural Language Processing
try:
    import nltk
    import spacy
    NLP_BASIC_AVAILABLE = True
    # Download necessary NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except ImportError:
    NLP_BASIC_AVAILABLE = False
    logging.warning("NLP libraries not available. Natural language understanding will be limited.")

# Advanced NLP with transformers
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    NLP_ADVANCED_AVAILABLE = True
except ImportError:
    NLP_ADVANCED_AVAILABLE = False
    logging.warning("Transformers library not available. Advanced NLP features will be limited.")

# Computer Vision
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Computer vision features will be limited.")

# Gesture Recognition
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Gesture recognition will be limited.")

# Language Detection and Translation
try:
    from langdetect import detect as lang_detect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # For deterministic results
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False
    logging.warning("LangDetect not available. Language detection will be limited.")

# Try to import security module for authentication
try:
    from security_enhanced_module import MFAManager
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logging.warning("Security module not available. Voice biometrics authentication will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/multimodal_interface.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice_multimodal")

# --- Constants and Enums ---

class InputModality(Enum):
    """Input modalities supported by the system."""
    VOICE = "voice"
    TEXT = "text"
    GESTURE = "gesture"
    VISION = "vision"
    TOUCH = "touch"
    BCI = "brain_computer_interface"
    AR_VR = "ar_vr"

class OutputModality(Enum):
    """Output modalities supported by the system."""
    VOICE = "voice"
    TEXT = "text"
    VISUAL = "visual"
    HAPTIC = "haptic"
    AR_VR = "ar_vr"

class SpeechRecognitionEngine(Enum):
    """Speech recognition engines supported by the system."""
    SYSTEM = "system"  # speech_recognition default
    WHISPER = "whisper"
    GOOGLE = "google"
    AZURE = "azure"
    VOSK = "vosk"
    SPHINX = "sphinx"

class TTSEngine(Enum):
    """Text-to-speech engines supported by the system."""
    SYSTEM = "system"  # pyttsx3
    GOOGLE = "google"
    AZURE = "azure"
    PICO = "pico"
    FESTIVAL = "festival"

class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    PORTUGUESE = "pt"
    AUTO = "auto"  # Auto-detect

class EmotionType(Enum):
    """Types of emotions that can be detected."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CONFUSED = "confused"

class CommandCategory(Enum):
    """Categories of voice commands."""
    SYSTEM = "system"
    NAVIGATION = "navigation"
    QUERY = "query"
    ACTION = "action"
    CONTROL = "control"
    COMMUNICATION = "communication"
    CUSTOM = "custom"

class GestureType(Enum):
    """Types of gestures that can be recognized."""
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"
    SPREAD = "spread"
    POINT = "point"
    GRAB = "grab"
    RELEASE = "release"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    VICTORY = "victory"
    WAVE = "wave"
    FIST = "fist"
    CUSTOM = "custom"

# --- Data Models ---

@dataclass
class VoiceProfile:
    """Data model for voice profiles used in voice biometrics."""
    user_id: str
    profile_id: str
    created_at: datetime.datetime
    last_updated: datetime.datetime
    features: Dict[str, Any]
    verification_threshold: float = 0.85
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SpeechRecognitionResult:
    """Data model for speech recognition results."""
    text: str
    confidence: float
    engine: SpeechRecognitionEngine
    language: Language
    timestamp: datetime.datetime
    audio_duration: float
    alternatives: List[Tuple[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class IntentRecognitionResult:
    """Data model for intent recognition results."""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    raw_text: str
    language: Language
    timestamp: datetime.datetime
    alternatives: List[Tuple[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EmotionAnalysisResult:
    """Data model for emotion analysis results."""
    primary_emotion: EmotionType
    emotion_scores: Dict[EmotionType, float]
    source: str  # 'voice', 'text', 'facial'
    timestamp: datetime.datetime
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class GestureRecognitionResult:
    """Data model for gesture recognition results."""
    gesture: GestureType
    confidence: float
    timestamp: datetime.datetime
    duration: float
    position: Tuple[float, float, float]  # x, y, z coordinates
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class VoiceCommand:
    """Data model for voice commands."""
    command_id: str
    phrases: List[str]
    category: CommandCategory
    action: Callable
    description: str
    is_active: bool = True
    requires_confirmation: bool = False
    context_dependent: bool = False
    contexts: List[str] = None
    parameters: List[str] = None

@dataclass
class TranslationResult:
    """Data model for translation results."""
    original_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float
    timestamp: datetime.datetime
    metadata: Optional[Dict[str, Any]] = None

# --- Speech Recognition ---

class SpeechRecognizer:
    """Handles speech recognition with multiple engines."""
    
    def __init__(self, default_engine: SpeechRecognitionEngine = SpeechRecognitionEngine.SYSTEM,
                 default_language: Language = Language.ENGLISH):
        """Initialize the speech recognizer.
        
        Args:
            default_engine: Default speech recognition engine
            default_language: Default language
        """
        self.default_engine = default_engine
        self.default_language = default_language
        
        # Initialize speech recognition engines
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.whisper_model = None
        self.google_client = None
        self.azure_speech_config = None
        
        # Initialize engines if available
        self._initialize_engines()
        
        # Audio recording settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.record_seconds = 5
        
        # For continuous listening
        self.is_listening = False
        self.listen_thread = None
        self.audio_queue = queue.Queue()
    
    def _initialize_engines(self):
        """Initialize available speech recognition engines."""
        # Initialize Whisper
        if WHISPER_AVAILABLE and self.default_engine == SpeechRecognitionEngine.WHISPER:
            try:
                # Load a small model by default for efficiency
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
        
        # Initialize Google Speech
        if GOOGLE_SPEECH_AVAILABLE and self.default_engine == SpeechRecognitionEngine.GOOGLE:
            try:
                self.google_client = google_speech.SpeechClient()
                logger.info("Google Speech client initialized")
            except Exception as e:
                logger.error(f"Error initializing Google Speech client: {str(e)}")
        
        # Initialize Azure Speech
        if AZURE_SPEECH_AVAILABLE and self.default_engine == SpeechRecognitionEngine.AZURE:
            try:
                subscription_key = os.environ.get("AZURE_SPEECH_KEY")
                region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
                
                if subscription_key:
                    self.azure_speech_config = azure_speech.SpeechConfig(
                        subscription=subscription_key,
                        region=region
                    )
                    logger.info("Azure Speech config initialized")
                else:
                    logger.warning("Azure Speech key not found in environment variables")
            except Exception as e:
                logger.error(f"Error initializing Azure Speech: {str(e)}")
    
    def recognize_from_microphone(self, engine: Optional[SpeechRecognitionEngine] = None,
                                 language: Optional[Language] = None,
                                 timeout: int = 5) -> SpeechRecognitionResult:
        """Recognize speech from the microphone.
        
        Args:
            engine: Speech recognition engine to use
            language: Language to recognize
            timeout: Recording timeout in seconds
            
        Returns:
            Speech recognition result
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("Speech recognition library not available")
        
        engine = engine or self.default_engine
        language = language or self.default_language
        
        try:
            # Record audio from microphone
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                logger.info(f"Listening for {timeout} seconds...")
                audio_data = self.recognizer.listen(source, timeout=timeout)
                
                start_time = time.time()
                
                # Process audio with selected engine
                return self._process_audio(audio_data, engine, language, time.time() - start_time)
                
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out")
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                engine=engine,
                language=language,
                timestamp=datetime.datetime.now(),
                audio_duration=timeout
            )
        except Exception as e:
            logger.error(f"Error recognizing speech: {str(e)}")
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                engine=engine,
                language=language,
                timestamp=datetime.datetime.now(),
                audio_duration=0.0
            )
    
    def recognize_from_file(self, file_path: str, engine: Optional[SpeechRecognitionEngine] = None,
                           language: Optional[Language] = None) -> SpeechRecognitionResult:
        """Recognize speech from an audio file.
        
        Args:
            file_path: Path to the audio file
            engine: Speech recognition engine to use
            language: Language to recognize
            
        Returns:
            Speech recognition result
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("Speech recognition library not available")
        
        engine = engine or self.default_engine
        language = language or self.default_language
        
        try:
            # Load audio file
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
                
                # Get audio duration
                with wave.open(file_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate)
                
                # Process audio with selected engine
                return self._process_audio(audio_data, engine, language, duration)
                
        except Exception as e:
            logger.error(f"Error recognizing speech from file: {str(e)}")
            return SpeechRecognitionResult(
                text="",
                confidence=0.0,
                engine=engine,
                language=language,
                timestamp=datetime.datetime.now(),
                audio_duration=0.0
            )
    
    def _process_audio(self, audio_data: sr.AudioData, engine: SpeechRecognitionEngine,
                      language: Language, duration: float) -> SpeechRecognitionResult:
        """Process audio data with the specified engine.
        
        Args:
            audio_data: Audio data to process
            engine: Speech recognition engine to use
            language: Language to recognize
            duration: Audio duration in seconds
            
        Returns:
            Speech recognition result
        """
        text = ""
        confidence = 0.0
        alternatives = []
        
        lang_code = language.value
        
        try:
            if engine == SpeechRecognitionEngine.SYSTEM:
                # Use the default system recognizer
                text = self.recognizer.recognize_sphinx(audio_data, language=lang_code)
                confidence = 0.8  # Sphinx doesn't provide confidence scores
            
            elif engine == SpeechRecognitionEngine.WHISPER and WHISPER_AVAILABLE:
                # Convert audio data to numpy array for Whisper
                audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                
                # Recognize with Whisper
                result = self.whisper_model.transcribe(audio_np, language=lang_code if lang_code != "auto" else None)
                text = result["text"]
                confidence = 0.9  # Whisper doesn't provide confidence scores
                
                # Get alternatives if available
                if "segments" in result and result["segments"]:
                    for segment in result["segments"]:
                        if "alternatives" in segment:
                            for alt in segment["alternatives"]:
                                alternatives.append((alt.get("text", ""), alt.get("probability", 0.0)))
            
            elif engine == SpeechRecognitionEngine.GOOGLE and GOOGLE_SPEECH_AVAILABLE:
                # Configure Google Speech recognition
                config = google_speech.RecognitionConfig(
                    encoding=google_speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                    language_code=lang_code if lang_code != "auto" else "en-US",
                    enable_automatic_punctuation=True,
                    model="default",
                    use_enhanced=True
                )
                
                # Convert audio data to bytes
                audio_bytes = audio_data.get_raw_data()
                
                # Recognize with Google Speech
                google_audio = google_speech.RecognitionAudio(content=audio_bytes)
                response = self.google_client.recognize(config=config, audio=google_audio)
                
                # Process results
                if response.results:
                    result = response.results[0]
                    if result.alternatives:
                        text = result.alternatives[0].transcript
                        confidence = result.alternatives[0].confidence
                        
                        # Get alternatives
                        for alt in result.alternatives[1:]:
                            alternatives.append((alt.transcript, alt.confidence))
            
            elif engine == SpeechRecognitionEngine.AZURE and AZURE_SPEECH_AVAILABLE:
                # Configure Azure Speech recognition
                self.azure_speech_config.speech_recognition_language = lang_code if lang_code != "auto" else "en-US"
                
                # Convert audio data to bytes
                audio_bytes = audio_data.get_raw_data()
                
                # Create audio configuration
                audio_config = azure_speech.audio.AudioConfig(stream=azure_speech.audio.PushAudioInputStream())
                
                # Create recognizer
                recognizer = azure_speech.SpeechRecognizer(
                    speech_config=self.azure_speech_config,
                    audio_config=audio_config
                )
                
                # Push audio data to stream
                stream = audio_config.get_audio_input_stream()
                stream.write(audio_bytes)
                stream.close()
                
                # Recognize with Azure Speech
                result = recognizer.recognize_once_async().get()
                
                # Process result
                if result.reason == azure_speech.ResultReason.RecognizedSpeech:
                    text = result.text
                    confidence = 0.9  # Azure doesn't provide confidence scores in basic API
                elif result.reason == azure_speech.ResultReason.NoMatch:
                    logger.warning("Azure Speech could not recognize the audio")
                elif result.reason == azure_speech.ResultReason.Canceled:
                    cancellation = azure_speech.CancellationDetails(result)
                    logger.error(f"Azure Speech recognition canceled: {cancellation.reason}")
                    if cancellation.reason == azure_speech.CancellationReason.Error:
                        logger.error(f"Error details: {cancellation.error_details}")
            
            else:
                # Fallback to system recognizer
                text = self.recognizer.recognize_sphinx(audio_data, language=lang_code)
                confidence = 0.7
        
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Could not request results from service: {str(e)}")
        except Exception as e:
            logger.error(f"Error in speech recognition: {str(e)}")
        
        return SpeechRecognitionResult(
            text=text,
            confidence=confidence,
            engine=engine,
            language=language,
            timestamp=datetime.datetime.now(),
            audio_duration=duration,
            alternatives=alternatives
        )
    
    def start_continuous_listening(self, callback: Callable[[SpeechRecognitionResult], None],
                                  engine: Optional[SpeechRecognitionEngine] = None,
                                  language: Optional[Language] = None):
        """Start continuous listening for speech.
        
        Args:
            callback: Function to call with recognition results
            engine: Speech recognition engine to use
            language: Language to recognize
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("Speech recognition library not available")
        
        if self.is_listening:
            logger.warning("Already listening")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(
            target=self._continuous_listening_thread,
            args=(callback, engine or self.default_engine, language or self.default_language)
        )
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        logger.info("Started continuous listening")
    
    def stop_continuous_listening(self):
        """Stop continuous listening."""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
            self.listen_thread = None
        
        logger.info("Stopped continuous listening")
    
    def _continuous_listening_thread(self, callback: Callable[[SpeechRecognitionResult], None],
                                    engine: SpeechRecognitionEngine, language: Language):
        """Thread function for continuous listening.
        
        Args:
            callback: Function to call with recognition results
            engine: Speech recognition engine to use
            language: Language to recognize
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                while self.is_listening:
                    try:
                        logger.debug("Listening for speech...")
                        audio_data = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=10.0)
                        
                        # Process in a separate thread to avoid blocking
                        threading.Thread(
                            target=self._process_and_callback,
                            args=(audio_data, engine, language, callback)
                        ).start()
                    
                    except sr.WaitTimeoutError:
                        # Timeout, continue listening
                        pass
                    except Exception as e:
                        logger.error(f"Error in continuous listening: {str(e)}")
                        time.sleep(1.0)  # Prevent tight loop on error
        
        except Exception as e:
            logger.error(f"Error setting up microphone: {str(e)}")
            self.is_listening = False
    
    def _process_and_callback(self, audio_data: sr.AudioData, engine: SpeechRecognitionEngine,
                             language: Language, callback: Callable[[SpeechRecognitionResult], None]):
        """Process audio and call the callback with the result.
        
        Args:
            audio_data: Audio data to process
            engine: Speech recognition engine to use
            language: Language to recognize
            callback: Function to call with recognition results
        """
        try:
            # Estimate duration
            duration = len(audio_data.get_raw_data()) / (self.sample_rate * self.channels * 2)  # 16-bit audio
            
            # Process audio
            result = self._process_audio(audio_data, engine, language, duration)
            
            # Call callback if text was recognized
            if result.text:
                callback(result)
        
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

# --- Natural Language Understanding ---

class IntentRecognizer:
    """Recognizes intents from natural language input."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the intent recognizer.
        
        Args:
            model_path: Path to the intent recognition model
        """
        self.model_path = model_path
        self.nlp = None
        self.intent_classifier = None
        self.entity_recognizer = None
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Load custom intents
        self.custom_intents = {}
        self.intent_patterns = {}
        self._load_custom_intents()
    
    def _initialize_nlp(self):
        """Initialize NLP components."""
        # Initialize spaCy if available
        if NLP_BASIC_AVAILABLE:
            try:
                # Try to load a spaCy model
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # If not installed, use a basic model
                    self.nlp = spacy.blank("en")
                
                logger.info("Initialized spaCy NLP")
            except Exception as e:
                logger.error(f"Error initializing spaCy: {str(e)}")
        
        # Initialize transformers if available
        if NLP_ADVANCED_AVAILABLE:
            try:
                # Load intent classification pipeline
                self.intent_classifier = pipeline(
                    "text-classification",
                    model="facebook/bart-large-mnli",
                    return_all_scores=True
                )
                
                # Load named entity recognition pipeline
                self.entity_recognizer = pipeline(
                    "ner",
                    aggregation_strategy="simple"
                )
                
                logger.info("Initialized transformers NLP pipelines")
            except Exception as e:
                logger.error(f"Error initializing transformers: {str(e)}")
    
    def _load_custom_intents(self):
        """Load custom intents from configuration."""
        intents_file = Path("config/intents.json")
        
        # Create default intents if file doesn't exist
        if not intents_file.exists():
            default_intents = {
                "greeting": {
                    "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "greetings"],
                    "responses": ["Hello!", "Hi there!", "Greetings!"]
                },
                "farewell": {
                    "patterns": ["goodbye", "bye", "see you", "later", "farewell"],
                    "responses": ["Goodbye!", "See you later!", "Farewell!"]
                },
                "help": {
                    "patterns": ["help", "assist", "support", "what can you do", "how does this work"],
                    "responses": ["I can help you with various tasks. What do you need?"]
                }
            }
            
            # Create directory if it doesn't exist
            intents_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Save default intents
            with open(intents_file, "w") as f:
                json.dump(default_intents, f, indent=2)
            
            self.custom_intents = default_intents
        else:
            # Load intents from file
            try:
                with open(intents_file, "r") as f:
                    self.custom_intents = json.load(f)
            except Exception as e:
                logger.error(f"Error loading intents: {str(e)}")
                self.custom_intents = {}
        
        # Compile patterns for efficient matching
        for intent, data in self.custom_intents.items():
            self.intent_patterns[intent] = [p.lower() for p in data.get("patterns", [])]
    
    def recognize_intent(self, text: str, language: Language = Language.ENGLISH) -> IntentRecognitionResult:
        """Recognize the intent from text.
        
        Args:
            text: Input text
            language: Language of the text
            
        Returns:
            Intent recognition result
        """
        intent = ""
        confidence = 0.0
        entities = {}
        alternatives = []
        
        # Check if text is empty
        if not text:
            return IntentRecognitionResult(
                intent="none",
                confidence=1.0,
                entities={},
                raw_text=text,
                language=language,
                timestamp=datetime.datetime.now(),
                alternatives=[]
            )
        
        # First, check for custom intent patterns
        text_lower = text.lower()
        custom_intent_scores = {}
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    score = len(pattern) / len(text_lower)  # Simple scoring based on pattern length
                    if intent_name not in custom_intent_scores or score > custom_intent_scores[intent_name]:
                        custom_intent_scores[intent_name] = score
        
        # If we found custom intents, use the highest scoring one
        if custom_intent_scores:
            best_intent = max(custom_intent_scores.items(), key=lambda x: x[1])
            intent = best_intent[0]
            confidence = best_intent[1]
            
            # Add alternatives
            for intent_name, score in custom_intent_scores.items():
                if intent_name != intent:
                    alternatives.append((intent_name, score))
        
        # If no custom intent or low confidence, try advanced NLP
        if not intent or confidence < 0.5:
            if NLP_ADVANCED_AVAILABLE and self.intent_classifier:
                try:
                    # Define candidate intents based on our custom intents
                    candidate_labels = list(self.custom_intents.keys())
                    
                    # Add some common intents if we have few custom ones
                    if len(candidate_labels) < 5:
                        candidate_labels.extend([
                            "greeting", "farewell", "help", "information", "query",
                            "command", "confirmation", "rejection", "gratitude"
                        ])
                    
                    # Classify with transformers
                    results = self.intent_classifier(text, candidate_labels)
                    
                    # Get the top result
                    top_result = max(results[0], key=lambda x: x['score'])
                    intent = top_result['label']
                    confidence = top_result['score']
                    
                    # Get alternatives
                    alternatives = [(r['label'], r['score']) for r in results[0] if r['label'] != intent]
                    
                except Exception as e:
                    logger.error(f"Error in transformer intent classification: {str(e)}")
            
            # If still no intent or transformers not available, use spaCy
            if (not intent or confidence < 0.5) and NLP_BASIC_AVAILABLE and self.nlp:
                try:
                    # Process with spaCy
                    doc = self.nlp(text)
                    
                    # Simple rule-based intent recognition
                    if any(token.text.lower() in ["hello", "hi", "hey", "greetings"] for token in doc):
                        intent = "greeting"
                        confidence = 0.8
                    elif any(token.text.lower() in ["goodbye", "bye", "farewell"] for token in doc):
                        intent = "farewell"
                        confidence = 0.8
                    elif any(token.text.lower() in ["help", "assist", "support"] for token in doc):
                        intent = "help"
                        confidence = 0.8
                    elif any(token.text.lower() in ["what", "who", "when", "where", "why", "how"] for token in doc):
                        intent = "query"
                        confidence = 0.7
                    elif any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc):
                        intent = "command"
                        confidence = 0.6
                    else:
                        intent = "unknown"
                        confidence = 0.5
                
                except Exception as e:
                    logger.error(f"Error in spaCy intent recognition: {str(e)}")
        
        # Extract entities
        if NLP_ADVANCED_AVAILABLE and self.entity_recognizer:
            try:
                # Extract entities with transformers
                ner_results = self.entity_recognizer(text)
                
                # Group entities by type
                for entity in ner_results:
                    entity_type = entity['entity_group']
                    entity_text = entity['word']
                    entity_score = entity['score']
                    
                    if entity_type not in entities:
                        entities[entity_type] = []
                    
                    entities[entity_type].append({
                        'text': entity_text,
                        'confidence': entity_score
                    })
            
            except Exception as e:
                logger.error(f"Error in transformer entity recognition: {str(e)}")
        
        # If transformers not available or failed, use spaCy
        if not entities and NLP_BASIC_AVAILABLE and self.nlp:
            try:
                # Process with spaCy
                doc = self.nlp(text)
                
                # Extract entities
                for ent in doc.ents:
                    if ent.label_ not in entities:
                        entities[ent.label_] = []
                    
                    entities[ent.label_].append({
                        'text': ent.text,
                        'confidence': 0.7  # spaCy doesn't provide confidence scores
                    })
            
            except Exception as e:
                logger.error(f"Error in spaCy entity recognition: {str(e)}")
        
        # If still no intent, use "unknown"
        if not intent:
            intent = "unknown"
            confidence = 0.5
        
        return IntentRecognitionResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            raw_text=text,
            language=language,
            timestamp=datetime.datetime.now(),
            alternatives=alternatives
        )
    
    def add_custom_intent(self, intent_name: str, patterns: List[str], responses: List[str] = None) -> bool:
        """Add a custom intent.
        
        Args:
            intent_name: Name of the intent
            patterns: List of patterns that trigger the intent
            responses: Optional list of responses for the intent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add to custom intents
            self.custom_intents[intent_name] = {
                "patterns": patterns,
                "responses": responses or []
            }
            
            # Update patterns
            self.intent_patterns[intent_name] = [p.lower() for p in patterns]
            
            # Save to file
            intents_file = Path("config/intents.json")
            intents_file.parent.mkdir(exist_ok=True, parents=True)
            
            with open(intents_file, "w") as f:
                json.dump(self.custom_intents, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding custom intent: {str(e)}")
            return False
    
    def get_response_for_intent(self, intent: str) -> Optional[str]:
        """Get a response for an intent.
        
        Args:
            intent: Intent name
            
        Returns:
            Response text or None if not found
        """
        if intent in self.custom_intents and self.custom_intents[intent].get("responses"):
            import random
            return random.choice(self.custom_intents[intent]["responses"])
        
        return None

# --- Text to Speech ---

class TextToSpeech:
    """Handles text-to-speech synthesis with multiple engines."""
    
    def __init__(self, default_engine: TTSEngine = TTSEngine.SYSTEM,
                 default_language: Language = Language.ENGLISH,
                 default_voice: Optional[str] = None):
        """Initialize the text-to-speech system.
        
        Args:
            default_engine: Default TTS engine
            default_language: Default language
            default_voice: Default voice name
        """
        self.default_engine = default_engine
        self.default_language = default_language
        self.default_voice = default_voice
        
        # Initialize TTS engines
        self.pyttsx3_engine = None
        self.google_tts_client = None
        self.azure_speech_config = None
        
        # Initialize engines if available
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize available TTS engines."""
        # Initialize pyttsx3
        if TTS_BASIC_AVAILABLE and self.default_engine == TTSEngine.SYSTEM:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                
                # Set properties
                self.pyttsx3_engine.setProperty('rate', 150)  # Speed
                self.pyttsx3_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
                
                # Set voice if specified
                if self.default_voice:
                    voices = self.pyttsx3_engine.getProperty('voices')
                    for voice in voices:
                        if self.default_voice in voice.id:
                            self.pyttsx3_engine.setProperty('voice', voice.id)
                            break
                
                logger.info("pyttsx3 TTS engine initialized")
            except Exception as e:
                logger.error(f"Error initializing pyttsx3: {str(e)}")
        
        # Initialize Google TTS
        if GOOGLE_SPEECH_AVAILABLE and self.default_engine == TTSEngine.GOOGLE:
            try:
                self.google_tts_client = google_tts.TextToSpeechClient()
                logger.info("Google TTS client initialized")
            except Exception as e:
                logger.error(f"Error initializing Google TTS: {str(e)}")
        
        # Initialize Azure Speech
        if AZURE_SPEECH_AVAILABLE and self.default_engine == TTSEngine.AZURE:
            try:
                subscription_key = os.environ.get("AZURE_SPEECH_KEY")
                region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
                
                if subscription_key:
                    self.azure_speech_config = azure_speech.SpeechConfig(
                        subscription=subscription_key,
                        region=region
                    )
                    logger.info("Azure Speech config initialized for TTS")
                else:
                    logger.warning("Azure Speech key not found in environment variables")
            except Exception as e:
                logger.error(f"Error initializing Azure Speech for TTS: {str(e)}")
    
    def speak(self, text: str, engine: Optional[TTSEngine] = None,
             language: Optional[Language] = None, voice: Optional[str] = None,
             rate: Optional[int] = None, volume: Optional[float] = None,
             pitch: Optional[float] = None, emotion: Optional[EmotionType] = None,
             wait: bool = True) -> bool:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            engine: TTS engine to use
            language: Language to use
            voice: Voice to use
            rate: Speech rate (words per minute)
            volume: Volume (0.0 to 1.0)
            pitch: Pitch adjustment
            emotion: Emotion to convey
            wait: Whether to wait for speech to complete
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            return False
        
        engine = engine or self.default_engine
        language = language or self.default_language
        voice = voice or self.default_voice
        
        try:
            if engine == TTSEngine.SYSTEM and TTS_BASIC_AVAILABLE:
                return self._speak_pyttsx3(text, language, voice, rate, volume, pitch, wait)
            elif engine == TTSEngine.GOOGLE and GOOGLE_SPEECH_AVAILABLE:
                return self._speak_google(text, language, voice, rate, pitch, emotion)
            elif engine == TTSEngine.AZURE and AZURE_SPEECH_AVAILABLE:
                return self._speak_azure(text, language, voice, rate, pitch, emotion, wait)
            else:
                # Fallback to pyttsx3 if available
                if TTS_BASIC_AVAILABLE:
                    return self._speak_pyttsx3(text, language, voice, rate, volume, pitch, wait)
                else:
                    logger.error("No TTS engine available")
                    return False
        
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return False
    
    def _speak_pyttsx3(self, text: str, language: Language, voice: Optional[str],
                      rate: Optional[int], volume: Optional[float],
                      pitch: Optional[float], wait: bool) -> bool:
        """Synthesize speech using pyttsx3.
        
        Args:
            text: Text to synthesize
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            volume: Volume
            pitch: Pitch adjustment
            wait: Whether to wait for speech to complete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pyttsx3_engine:
            try:
                self.pyttsx3_engine = pyttsx3.init()
            except Exception as e:
                logger.error(f"Error initializing pyttsx3: {str(e)}")
                return False
        
        try:
            # Set voice if specified
            if voice:
                voices = self.pyttsx3_engine.getProperty('voices')
                for v in voices:
                    if voice in v.id or voice in v.name:
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        break
            
            # Set language-specific voice if no specific voice is set
            elif language != Language.ENGLISH:
                voices = self.pyttsx3_engine.getProperty('voices')
                lang_code = language.value
                
                for v in voices:
                    if lang_code in v.id or lang_code in v.languages:
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        break
            
            # Set rate if specified
            if rate is not None:
                self.pyttsx3_engine.setProperty('rate', rate)
            
            # Set volume if specified
            if volume is not None:
                self.pyttsx3_engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            # Speak
            self.pyttsx3_engine.say(text)
            
            if wait:
                self.pyttsx3_engine.runAndWait()
            else:
                # Run in a separate thread
                threading.Thread(target=self.pyttsx3_engine.runAndWait).start()
            
            return True
        
        except Exception as e:
            logger.error(f"Error in pyttsx3 speech synthesis: {str(e)}")
            return False
    
    def _speak_google(self, text: str, language: Language, voice: Optional[str],
                     rate: Optional[int], pitch: Optional[float],
                     emotion: Optional[EmotionType]) -> bool:
        """Synthesize speech using Google TTS.
        
        Args:
            text: Text to synthesize
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            pitch: Pitch adjustment
            emotion: Emotion to convey
            
        Returns:
            True if successful, False otherwise
        """
        if not self.google_tts_client:
            try:
                self.google_tts_client = google_tts.TextToSpeechClient()
            except Exception as e:
                logger.error(f"Error initializing Google TTS: {str(e)}")
                return False
        
        try:
            # Set up the voice
            lang_code = language.value
            
            # Default voice name based on language
            if not voice:
                if lang_code == "en":
                    voice = "en-US-Wavenet-D"
                elif lang_code == "es":
                    voice = "es-ES-Wavenet-B"
                elif lang_code == "fr":
                    voice = "fr-FR-Wavenet-C"
                elif lang_code == "de":
                    voice = "de-DE-Wavenet-B"
                elif lang_code == "ja":
                    voice = "ja-JP-Wavenet-B"
                else:
                    # Default to English
                    voice = "en-US-Wavenet-D"
            
            # Set up speaking rate and pitch
            speaking_rate = 1.0
            if rate is not None:
                # Convert words per minute to relative rate (1.0 is normal)
                speaking_rate = rate / 150.0
            
            pitch_value = 0.0
            if pitch is not None:
                pitch_value = pitch
            
            # Adjust for emotion if specified
            if emotion:
                if emotion == EmotionType.HAPPY:
                    pitch_value += 2.0
                    speaking_rate *= 1.1
                elif emotion == EmotionType.SAD:
                    pitch_value -= 2.0
                    speaking_rate *= 0.9
                elif emotion == EmotionType.ANGRY:
                    pitch_value += 1.0
                    speaking_rate *= 1.2
                elif emotion == EmotionType.FEARFUL:
                    pitch_value -= 1.0
                    speaking_rate *= 1.1
            
            # Ensure values are within bounds
            speaking_rate = max(0.25, min(4.0, speaking_rate))
            pitch_value = max(-20.0, min(20.0, pitch_value))
            
            # Set up the synthesis input
            synthesis_input = google_tts.SynthesisInput(text=text)
            
            # Build the voice request
            voice_params = google_tts.VoiceSelectionParams(
                language_code=lang_code,
                name=voice
            )
            
            # Set up audio config
            audio_config = google_tts.AudioConfig(
                audio_encoding=google_tts.AudioEncoding.LINEAR16,
                speaking_rate=speaking_rate,
                pitch=pitch_value
            )
            
            # Perform the text-to-speech request
            response = self.google_tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            # Save the audio to a temporary file and play it
            temp_file = f"temp_tts_{uuid.uuid4()}.wav"
            with open(temp_file, "wb") as out:
                out.write(response.audio_content)
            
            # Play the audio
            self._play_audio_file(temp_file)
            
            # Clean up
            try:
                os.remove(temp_file)
            except Exception:
                pass
            
            return True
        
        except Exception as e:
            logger.error(f"Error in Google TTS: {str(e)}")
            return False
    
    def _speak_azure(self, text: str, language: Language, voice: Optional[str],
                    rate: Optional[int], pitch: Optional[float],
                    emotion: Optional[EmotionType], wait: bool) -> bool:
        """Synthesize speech using Azure TTS.
        
        Args:
            text: Text to synthesize
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            pitch: Pitch adjustment
            emotion: Emotion to convey
            wait: Whether to wait for speech to complete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.azure_speech_config:
            try:
                subscription_key = os.environ.get("AZURE_SPEECH_KEY")
                region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
                
                if subscription_key:
                    self.azure_speech_config = azure_speech.SpeechConfig(
                        subscription=subscription_key,
                        region=region
                    )
                else:
                    logger.error("Azure Speech key not found in environment variables")
                    return False
            except Exception as e:
                logger.error(f"Error initializing Azure Speech for TTS: {str(e)}")
                return False
        
        try:
            # Set up the voice
            lang_code = language.value
            
            # Default voice name based on language
            if not voice:
                if lang_code == "en":
                    voice = "en-US-AriaNeural"
                elif lang_code == "es":
                    voice = "es-ES-ElviraNeural"
                elif lang_code == "fr":
                    voice = "fr-FR-DeniseNeural"
                elif lang_code == "de":
                    voice = "de-DE-KatjaNeural"
                elif lang_code == "ja":
                    voice = "ja-JP-NanamiNeural"
                else:
                    # Default to English
                    voice = "en-US-AriaNeural"
            
            # Set the voice
            self.azure_speech_config.speech_synthesis_voice_name = voice
            
            # Create a speech synthesizer
            synthesizer = azure_speech.SpeechSynthesizer(
                speech_config=self.azure_speech_config,
                audio_config=None
            )
            
            # Prepare SSML for advanced control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang_code}">
                <voice name="{voice}">
            """
            
            # Add prosody adjustments if specified
            if rate is not None or pitch is not None or emotion is not None:
                prosody_attrs = []
                
                if rate is not None:
                    # Convert words per minute to relative rate
                    relative_rate = rate / 150.0
                    if relative_rate < 0.5:
                        rate_str = "x-slow"
                    elif relative_rate < 0.8:
                        rate_str = "slow"
                    elif relative_rate < 1.2:
                        rate_str = "medium"
                    elif relative_rate < 1.5:
                        rate_str = "fast"
                    else:
                        rate_str = "x-fast"
                    prosody_attrs.append(f'rate="{rate_str}"')
                
                if pitch is not None:
                    # Convert to semitones
                    pitch_str = f"{pitch:+.1f}st"
                    prosody_attrs.append(f'pitch="{pitch_str}"')
                
                # Adjust for emotion if specified
                if emotion:
                    if emotion == EmotionType.HAPPY:
                        prosody_attrs.append('pitch="+2st"')
                        prosody_attrs.append('rate="1.1"')
                    elif emotion == EmotionType.SAD:
                        prosody_attrs.append('pitch="-2st"')
                        prosody_attrs.append('rate="0.9"')
                    elif emotion == EmotionType.ANGRY:
                        prosody_attrs.append('pitch="+1st"')
                        prosody_attrs.append('rate="1.2"')
                        prosody_attrs.append('volume="loud"')
                    elif emotion == EmotionType.FEARFUL:
                        prosody_attrs.append('pitch="-1st"')
                        prosody_attrs.append('rate="1.1"')
                
                if prosody_attrs:
                    ssml += f"<prosody {' '.join(prosody_attrs)}>{text}</prosody>"
                else:
                    ssml += text
            else:
                ssml += text
            
            ssml += """
                </voice>
            </speak>
            """
            
            # Synthesize speech
            if wait:
                result = synthesizer.speak_ssml(ssml)
                
                # Check result
                if result.reason == azure_speech.ResultReason.SynthesizingAudioCompleted:
                    return True
                elif result.reason == azure_speech.ResultReason.Canceled:
                    cancellation = azure_speech.SpeechSynthesisCancellationDetails(result)
                    logger.error(f"Azure TTS canceled: {cancellation.reason}")
                    if cancellation.reason == azure_speech.CancellationReason.Error:
                        logger.error(f"Error details: {cancellation.error_details}")
                    return False
            else:
                # Run in a separate thread
                def synthesize_async():
                    result = synthesizer.speak_ssml(ssml)
                    if result.reason != azure_speech.ResultReason.SynthesizingAudioCompleted:
                        if result.reason == azure_speech.ResultReason.Canceled:
                            cancellation = azure_speech.SpeechSynthesisCancellationDetails(result)
                            logger.error(f"Azure TTS canceled: {cancellation.reason}")
                
                threading.Thread(target=synthesize_async).start()
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error in Azure TTS: {str(e)}")
            return False
    
    def _play_audio_file(self, file_path: str):
        """Play an audio file.
        
        Args:
            file_path: Path to the audio file
        """
        try:
            if sys.platform == "win32":
                # Windows
                os.system(f'start /min "" powershell -c "(New-Object Media.SoundPlayer \'{file_path}\').PlaySync()"')
            elif sys.platform == "darwin":
                # macOS
                os.system(f"afplay {file_path}")
            else:
                # Linux
                os.system(f"aplay {file_path}")
        except Exception as e:
            logger.error(f"Error playing audio file: {str(e)}")
    
    def save_to_file(self, text: str, file_path: str, engine: Optional[TTSEngine] = None,
                    language: Optional[Language] = None, voice: Optional[str] = None,
                    rate: Optional[int] = None, pitch: Optional[float] = None,
                    emotion: Optional[EmotionType] = None) -> bool:
        """Save synthesized speech to a file.
        
        Args:
            text: Text to synthesize
            file_path: Path to save the audio file
            engine: TTS engine to use
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            pitch: Pitch adjustment
            emotion: Emotion to convey
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            return False
        
        engine = engine or self.default_engine
        language = language or self.default_language
        voice = voice or self.default_voice
        
        try:
            if engine == TTSEngine.SYSTEM and TTS_BASIC_AVAILABLE:
                return self._save_pyttsx3(text, file_path, language, voice, rate)
            elif engine == TTSEngine.GOOGLE and GOOGLE_SPEECH_AVAILABLE:
                return self._save_google(text, file_path, language, voice, rate, pitch, emotion)
            elif engine == TTSEngine.AZURE and AZURE_SPEECH_AVAILABLE:
                return self._save_azure(text, file_path, language, voice, rate, pitch, emotion)
            else:
                # Fallback to pyttsx3 if available
                if TTS_BASIC_AVAILABLE:
                    return self._save_pyttsx3(text, file_path, language, voice, rate)
                else:
                    logger.error("No TTS engine available")
                    return False
        
        except Exception as e:
            logger.error(f"Error saving speech to file: {str(e)}")
            return False
    
    def _save_pyttsx3(self, text: str, file_path: str, language: Language,
                     voice: Optional[str], rate: Optional[int]) -> bool:
        """Save synthesized speech to a file using pyttsx3.
        
        Args:
            text: Text to synthesize
            file_path: Path to save the audio file
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pyttsx3_engine:
            try:
                self.pyttsx3_engine = pyttsx3.init()
            except Exception as e:
                logger.error(f"Error initializing pyttsx3: {str(e)}")
                return False
        
        try:
            # Set voice if specified
            if voice:
                voices = self.pyttsx3_engine.getProperty('voices')
                for v in voices:
                    if voice in v.id or voice in v.name:
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        break
            
            # Set language-specific voice if no specific voice is set
            elif language != Language.ENGLISH:
                voices = self.pyttsx3_engine.getProperty('voices')
                lang_code = language.value
                
                for v in voices:
                    if lang_code in v.id or lang_code in v.languages:
                        self.pyttsx3_engine.setProperty('voice', v.id)
                        break
            
            # Set rate if specified
            if rate is not None:
                self.pyttsx3_engine.setProperty('rate', rate)
            
            # Save to file
            self.pyttsx3_engine.save_to_file(text, file_path)
            self.pyttsx3_engine.runAndWait()
            
            return os.path.exists(file_path)
        
        except Exception as e:
            logger.error(f"Error saving pyttsx3 speech to file: {str(e)}")
            return False
    
    def _save_google(self, text: str, file_path: str, language: Language,
                    voice: Optional[str], rate: Optional[int],
                    pitch: Optional[float], emotion: Optional[EmotionType]) -> bool:
        """Save synthesized speech to a file using Google TTS.
        
        Args:
            text: Text to synthesize
            file_path: Path to save the audio file
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            pitch: Pitch adjustment
            emotion: Emotion to convey
            
        Returns:
            True if successful, False otherwise
        """
        if not self.google_tts_client:
            try:
                self.google_tts_client = google_tts.TextToSpeechClient()
            except Exception as e:
                logger.error(f"Error initializing Google TTS: {str(e)}")
                return False
        
        try:
            # Set up the voice
            lang_code = language.value
            
            # Default voice name based on language
            if not voice:
                if lang_code == "en":
                    voice = "en-US-Wavenet-D"
                elif lang_code == "es":
                    voice = "es-ES-Wavenet-B"
                elif lang_code == "fr":
                    voice = "fr-FR-Wavenet-C"
                elif lang_code == "de":
                    voice = "de-DE-Wavenet-B"
                elif lang_code == "ja":
                    voice = "ja-JP-Wavenet-B"
                else:
                    # Default to English
                    voice = "en-US-Wavenet-D"
            
            # Set up speaking rate and pitch
            speaking_rate = 1.0
            if rate is not None:
                # Convert words per minute to relative rate (1.0 is normal)
                speaking_rate = rate / 150.0
            
            pitch_value = 0.0
            if pitch is not None:
                pitch_value = pitch
            
            # Adjust for emotion if specified
            if emotion:
                if emotion == EmotionType.HAPPY:
                    pitch_value += 2.0
                    speaking_rate *= 1.1
                elif emotion == EmotionType.SAD:
                    pitch_value -= 2.0
                    speaking_rate *= 0.9
                elif emotion == EmotionType.ANGRY:
                    pitch_value += 1.0
                    speaking_rate *= 1.2
                elif emotion == EmotionType.FEARFUL:
                    pitch_value -= 1.0
                    speaking_rate *= 1.1
            
            # Ensure values are within bounds
            speaking_rate = max(0.25, min(4.0, speaking_rate))
            pitch_value = max(-20.0, min(20.0, pitch_value))
            
            # Set up the synthesis input
            synthesis_input = google_tts.SynthesisInput(text=text)
            
            # Build the voice request
            voice_params = google_tts.VoiceSelectionParams(
                language_code=lang_code,
                name=voice
            )
            
            # Set up audio config
            audio_config = google_tts.AudioConfig(
                audio_encoding=google_tts.AudioEncoding.LINEAR16,
                speaking_rate=speaking_rate,
                pitch=pitch_value
            )
            
            # Perform the text-to-speech request
            response = self.google_tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice_params,
                audio_config=audio_config
            )
            
            # Save the audio to the specified file
            with open(file_path, "wb") as out:
                out.write(response.audio_content)
            
            return os.path.exists(file_path)
        
        except Exception as e:
            logger.error(f"Error saving Google TTS to file: {str(e)}")
            return False
    
    def _save_azure(self, text: str, file_path: str, language: Language,
                   voice: Optional[str], rate: Optional[int],
                   pitch: Optional[float], emotion: Optional[EmotionType]) -> bool:
        """Save synthesized speech to a file using Azure TTS.
        
        Args:
            text: Text to synthesize
            file_path: Path to save the audio file
            language: Language to use
            voice: Voice to use
            rate: Speech rate
            pitch: Pitch adjustment
            emotion: Emotion to convey
            
        Returns:
            True if successful, False otherwise
        """
        if not self.azure_speech_config:
            try:
                subscription_key = os.environ.get("AZURE_SPEECH_KEY")
                region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
                
                if subscription_key:
                    self.azure_speech_config = azure_speech.SpeechConfig(
                        subscription=subscription_key,
                        region=region
                    )
                else:
                    logger.error("Azure Speech key not found in environment variables")
                    return False
            except Exception as e:
                logger.error(f"Error initializing Azure Speech for TTS: {str(e)}")
                return False
        
        try:
            # Set up the voice
            lang_code = language.value
            
            # Default voice name based on language
            if not voice:
                if lang_code == "en":
                    voice = "en-US-AriaNeural"
                elif lang_code == "es":
                    voice = "es-ES-ElviraNeural"
                elif lang_code == "fr":
                    voice = "fr-FR-DeniseNeural"
                elif lang_code == "de":
                    voice = "de-DE-KatjaNeural"
                elif lang_code == "ja":
                    voice = "ja-JP-NanamiNeural"
                else:
                    # Default to English
                    voice = "en-US-AriaNeural"
            
            # Set the voice
            self.azure_speech_config.speech_synthesis_voice_name = voice
            
            # Create audio configuration for file output
            audio_config = azure_speech.AudioConfig(filename=file_path)
            
            # Create a speech synthesizer
            synthesizer = azure_speech.SpeechSynthesizer(
                speech_config=self.azure_speech_config,
                audio_config=audio_config
            )
            
            # Prepare SSML for advanced control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang_code}">
                <voice name="{voice}">
            """
            
            # Add prosody adjustments if specified
            if rate is not None or pitch is not None or emotion is not None:
                prosody_attrs = []
                
                if rate is not None:
                    # Convert words per minute to relative rate
                    relative_rate = rate / 150.0
                    if relative_rate < 0.5:
                        rate_str = "x-slow"
                    elif relative_rate < 0.8:
                        rate_str = "slow"
                    elif relative_rate < 1.2:
                        rate_str = "medium"
                    elif relative_rate < 1.5:
                        rate_str = "fast"
                    else:
                        rate_str = "x-fast"
                    prosody_attrs.append(f'rate="{rate_str}"')
                
                if pitch is not None:
                    # Convert to semitones
                    pitch_str = f"{pitch:+.1f}st"
                    prosody_attrs.append(f'pitch="{pitch_str}"')
                
                # Adjust for emotion if specified
                if emotion:
                    if emotion == EmotionType.HAPPY:
                        prosody_attrs.append('pitch="+2st"')
                        prosody_attrs.append('rate="1.1"')
                    elif emotion == EmotionType.SAD:
                        prosody_attrs.append('pitch="-2st"')
                        prosody_attrs.append('rate="0.9"')
                    elif emotion == EmotionType.ANGRY:
                        prosody_attrs.append('pitch="+1st"')
                        prosody_attrs.append('rate="1.2"')
                        prosody_attrs.append('volume="loud"')
                    elif emotion == EmotionType.FEARFUL:
                        prosody_attrs.append('pitch="-1st"')
                        prosody_attrs.append('rate="1.1"')
                
                if prosody_attrs:
                    ssml += f"<prosody {' '.join(prosody_attrs)}>{text}</prosody>"
                else:
                    ssml += text
            else:
                ssml += text
            
            ssml += """
                </voice>
            </speak>
            """
            
            # Synthesize speech
            result = synthesizer.speak_ssml(ssml)
            
            # Check result
            if result.reason == azure_speech.ResultReason.SynthesizingAudioCompleted:
                return os.path.exists(file_path)
            elif result.reason == azure_speech.ResultReason.Canceled:
                cancellation = azure_speech.SpeechSynthesisCancellationDetails(result)
                logger.error(f"Azure TTS canceled: {cancellation.reason}")
                if cancellation.reason == azure_speech.CancellationReason.Error:
                    logger.error(f"Error details: {cancellation.error_details}")
                return False
            
            return False
        
        except Exception as e:
            logger.error(f"Error saving Azure TTS to file: {str(e)}")
            return False

# --- Voice Command Handler ---

class VoiceCommandHandler:
    """Handles voice commands and their execution."""
    
    def __init__(self):
        """Initialize the voice command handler."""
        self.commands = {}  # command_id -> VoiceCommand
        self.command_history = deque(maxlen=100)
        
        # Load commands from configuration
        self._load_commands()
    
    def _load_commands(self):
        """Load commands from configuration."""
        commands_file = Path("config/voice_commands.json")
        
        # Create default commands if file doesn't exist
        if not commands_file.exists():
            default_commands = {
                "system_help": {
                    "phrases": ["help", "what can you do", "show commands"],
                    "category": "system",
                    "description": "Show available commands",
                    "requires_confirmation": False,
                    "context_dependent": False
                },
                "system_exit": {
                    "phrases": ["exit", "quit", "close"],
                    "category": "system",
                    "description": "Exit the application",
                    "requires_confirmation": True,
                    "context_dependent": False
                }
            }
            
            # Create directory if it doesn't exist
            commands_file.parent.mkdir(exist_ok=True, parents=True)
            
            # Save default commands
            with open(commands_file, "w") as f:
                json.dump(default_commands, f, indent=2)
            
            # Register default commands
            for command_id, command_data in default_commands.items():
                self.register_command(
                    command_id=command_id,
                    phrases=command_data["phrases"],
                    category=CommandCategory(command_data["category"]),
                    action=lambda: logger.info(f"Command {command_id} executed"),
                    description=command_data["description"],
                    requires_confirmation=command_data.get("requires_confirmation", False),
                    context_dependent=command_data.get("context_dependent", False),
                    contexts=command_data.get("contexts")
                )
        else:
            # Load commands from file
            try:
                with open(commands_file, "r") as f:
                    commands_data = json.load(f)
                
                # Register commands
                for command_id, command_data in commands_data.items():
                    self.register_command(
                        command_id=command_id,
                        phrases=command_data["phrases"],
                        category=CommandCategory(command_data["category"]),
                        action=lambda: logger.info(f"Command {command_id} executed"),
                        description=command_data["description"],
                        requires_confirmation=command_data.get("requires_confirmation", False),
                        context_dependent=command_data.get("context_dependent", False),
                        contexts=command_data.get("contexts")
                    )
            except Exception as e:
                logger.error(f"Error loading voice commands: {str(e)}")
    
    def register_command(self, command_id: str, phrases: List[str], category: CommandCategory,
                        action: Callable, description: str, requires_confirmation: bool = False,
                        context_dependent: bool = False, contexts: List[str] = None,
                        parameters: List[str] = None) -> bool:
        """Register a voice command.
        
        Args:
            command_id: Unique identifier for the command
            phrases: List of phrases that trigger the command
            category: Category of the command
            action: Function to execute when the command is triggered
            description: Description of the command
            requires_confirmation: Whether the command requires confirmation
            context_dependent: Whether the command is context-dependent
            contexts: List of contexts in which the command is valid
            parameters: List of parameters for the command
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create command object
            command = VoiceCommand(
                command_id=command_id,
                phrases=phrases,
                category=category,
                action=action,
                description=description,
                is_active=True,
                requires_confirmation=requires_confirmation,
                context_dependent=context_dependent,
                contexts=contexts,
                parameters=parameters
            )
            
            # Register command
            self.commands[command_id] = command
            
            return True
        except Exception as e:
            logger.error(f"Error registering command: {str(e)}")
            return False
    
    def process_command(self, text: str, context: Optional[str] = None) -> Optional[VoiceCommand]:
        """Process a voice command.
        
        Args:
            text: Command text
            context: Current context
            
        Returns:
            Matched command or None if no command matched
        """
        if not text:
            return None
        
        text_lower = text.lower()
        matched_command = None
        max_match_length = 0
        
        # Find the best matching command
        for command in self.commands.values():
            if not command.is_active:
                continue
            
            # Skip context-dependent commands if context doesn't match
            if command.context_dependent and context and command.contexts:
                if context not in command.contexts:
                    continue
            
            # Check if any phrase matches
            for phrase in command.phrases:
                phrase_lower = phrase.lower()
                if phrase_lower in text_lower:
                    # Use the longest matching phrase
                    if len(phrase) > max_match_length:
                        max_match_length = len(phrase)
                        matched_command = command
        
        if matched_command:
            # Add to command history
            self.command_history.append({
                "command": matched_command.command_id,
                "text": text,
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context
            })
            
            return matched_command
        
        return None
    
    def execute_command(self, command: VoiceCommand, parameters: Dict[str, Any] = None) -> bool:
        """Execute a voice command.
        
        Args:
            command: Command to execute
            parameters: Command parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if command.requires_confirmation:
                # In a real implementation, would prompt for confirmation
                logger.info(f"Command {command.command_id} requires confirmation")
            
            # Execute the command
            if parameters:
                command.action(**parameters)
            else:
                command.action()
            
            return True
        except Exception as e:
            logger.error(f"Error executing command {command.command_id}: {str(e)}")
            return False
    
    def get_commands_by_category(self, category: CommandCategory) -> List[VoiceCommand]:
        """Get commands by category.
        
        Args:
            category: Command category
            
        Returns:
            List of commands in the category
        """
        return [cmd for cmd in self.commands.values() if cmd.category == category and cmd.is_active]
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get command history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of command history items
        """
        history =