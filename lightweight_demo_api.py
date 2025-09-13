"""
Lightweight Demo API for SHEMS
Immediate deployment with core functionality only
No heavy dependencies, runs on basic hardware
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml
from pathlib import Path
import time
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import json
from scipy import signal
from scipy.signal import butter, filtfilt
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Minimal configuration for demo"""
    # EEG Settings
    SAMPLING_RATE = 256  # Hz (standard for most EEG devices)
    EEG_CHANNELS = 8     # Typical consumer EEG channel count
    
    # Frequency bands (Hz)
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    # Performance
    CACHE_SIZE = 100
    MAX_TEXT_LENGTH = 1000
    
    # API
    VERSION = "1.0-demo"
    PORT = 8000

config = Config()

# ============================================================================
# EEG ANALYZER with Goertzel/Kalman/SNR
# ============================================================================

class EEGAnalyzer:
    """Optimized EEG analysis with Goertzel algorithm and Kalman filtering"""
    
    def __init__(self, sampling_rate: int = 256):
        self.sampling_rate = sampling_rate
        self.kalman_state = None
        self.kalman_covariance = np.eye(5) * 0.1  # For 5 frequency bands
        
    def goertzel_algorithm(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm for efficient single frequency detection
        Much faster than FFT for specific frequencies
        """
        N = len(samples)
        k = int(0.5 + N * target_freq / self.sampling_rate)
        omega = 2 * np.pi * k / N
        
        # Goertzel coefficients
        coeff = 2 * np.cos(omega)
        
        # Filter implementation
        s_prev = 0
        s_prev2 = 0
        
        for sample in samples:
            s = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        
        # Calculate power
        power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        return np.sqrt(power) / N
    
    def extract_band_powers(self, eeg_signal: np.ndarray) -> Dict[str, float]:
        """Extract power in each frequency band using Goertzel"""
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in config.BANDS.items():
            # Use Goertzel for center frequency of each band
            center_freq = (low_freq + high_freq) / 2
            power = self.goertzel_algorithm(eeg_signal, center_freq)
            
            # Normalize to z-score
            band_powers[band_name] = (power - 0.5) * 2  # Simple normalization
        
        return band_powers
    
    def kalman_filter(self, measurement: np.ndarray) -> np.ndarray:
        """Kalman filter for noise reduction"""
        # Initialize state if needed
        if self.kalman_state is None:
            self.kalman_state = measurement
            return measurement
        
        # Prediction step
        predicted_state = self.kalman_state
        predicted_covariance = self.kalman_covariance + np.eye(5) * 0.01
        
        # Update step
        innovation = measurement - predicted_state
        innovation_covariance = predicted_covariance + np.eye(5) * 0.1
        kalman_gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
        
        # Update state and covariance
        self.kalman_state = predicted_state + kalman_gain @ innovation
        self.kalman_covariance = (np.eye(5) - kalman_gain) @ predicted_covariance
        
        return self.kalman_state
    
    def calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        # Simple SNR estimation using high-frequency content as noise
        noise_band = butter(4, [40, 50], btype='band', fs=self.sampling_rate)
        noise = filtfilt(noise_band[0], noise_band[1], signal)
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 40.0  # Good SNR
        
        return snr
    
    def process_eeg(self, raw_eeg: np.ndarray) -> Dict:
        """Complete EEG processing pipeline"""
        # Check SNR
        snr = self.calculate_snr(raw_eeg)
        
        if snr < 10:
            logger.warning(f"Low SNR detected: {snr:.2f} dB")
        
        # Extract band powers
        band_powers = self.extract_band_powers(raw_eeg)
        
        # Apply Kalman filtering
        band_array = np.array(list(band_powers.values()))
        filtered_bands = self.kalman_filter(band_array)
        
        # Update band powers with filtered values
        for i, band_name in enumerate(config.BANDS.keys()):
            band_powers[band_name] = float(filtered_bands[i])
        
        return {
            "band_powers": band_powers,
            "snr": snr,
            "quality": "good" if snr > 20 else "fair" if snr > 10 else "poor"
        }

# ============================================================================
# OPTIMIZED EMOTION ENGINE
# ============================================================================

class OptimizedEmotionEngine:
    """Lightweight emotion processing engine"""
    
    def __init__(self):
        self.yaml_mappings = self._load_yaml_mappings()
        self.cache = {}
        
    def _load_yaml_mappings(self) -> Dict:
        """Load YAML mappings (simplified for demo)"""
        # Embedded minimal mappings for demo
        return {
            'eeg_emotion_profiles': {
                'joy': {'delta': -0.10, 'theta': -0.05, 'alpha': 0.35, 'beta': 0.25, 'gamma': 0.15},
                'sadness': {'delta': 0.05, 'theta': 0.20, 'alpha': -0.25, 'beta': 0.10, 'gamma': 0.05},
                'anger': {'delta': 0.10, 'theta': 0.15, 'alpha': -0.30, 'beta': 0.40, 'gamma': 0.25},
                'fear': {'delta': 0.05, 'theta': 0.25, 'alpha': -0.35, 'beta': 0.30, 'gamma': 0.40},
                'disgust': {'delta': 0.10, 'theta': 0.20, 'alpha': -0.20, 'beta': 0.25, 'gamma': 0.10},
                'surprise': {'delta': -0.05, 'theta': 0.05, 'alpha': 0.10, 'beta': 0.30, 'gamma': 0.35},
                'neutral': {'delta': 0.00, 'theta': 0.00, 'alpha': 0.00, 'beta': 0.00, 'gamma': 0.00}
            },
            'korean_endings': {
                '-네요': [0.25, 0.05, 0.00, 0.30, 0.00, 0.40, -0.05],
                '-군요': [0.10, 0.05, 0.00, 0.15, 0.00, 0.30, -0.05],
                '-거든요': [0.00, 0.00, 0.35, 0.05, 0.00, 0.05, -0.10]
            }
        }
    
    def map_eeg_to_emotion(self, band_powers: Dict[str, float]) -> np.ndarray:
        """Map EEG band powers to emotion vector"""
        emotion_scores = {}
        
        # Compare with each emotion profile
        for emotion, profile in self.yaml_mappings['eeg_emotion_profiles'].items():
            score = 0
            for band, expected_value in profile.items():
                if band in band_powers:
                    # Calculate similarity (inverse of distance)
                    diff = abs(band_powers[band] - expected_value)
                    score += np.exp(-diff * 2)  # Exponential decay
            
            emotion_scores[emotion] = score / len(profile)
        
        # Convert to 7D emotion vector
        emotion_vector = np.array([
            emotion_scores.get('joy', 0),
            emotion_scores.get('sadness', 0),
            emotion_scores.get('anger', 0),
            emotion_scores.get('fear', 0),
            emotion_scores.get('disgust', 0),
            emotion_scores.get('surprise', 0),
            emotion_scores.get('neutral', 0)
        ])
        
        # Normalize
        if emotion_vector.sum() > 0:
            emotion_vector = emotion_vector / emotion_vector.sum()
        
        return emotion_vector
    
    def process_text(self, text: str) -> np.ndarray:
        """Simple text emotion analysis"""
        # Cache check
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        emotion_vector = np.zeros(7)
        
        # Check for Korean endings
        for ending, emotion_values in self.yaml_mappings['korean_endings'].items():
            if ending in text:
                emotion_vector += np.array(emotion_values)
        
        # Simple keyword matching (for demo)
        emotion_keywords = {
            0: ['기쁨', '행복', '좋아', '사랑'],  # joy
            1: ['슬픔', '우울', '눈물', '아프'],   # sadness
            2: ['화나', '분노', '짜증', '열받'],   # anger
            3: ['무서', '두려', '공포', '겁나'],   # fear
            4: ['역겨', '더러', '혐오', '싫어'],   # disgust
            5: ['놀라', '깜짝', '헉', '어머'],    # surprise
            6: ['그냥', '보통', '평범', '중립']    # neutral
        }
        
        for idx, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_vector[idx] += 0.3
        
        # Normalize
        if emotion_vector.sum() > 0:
            emotion_vector = emotion_vector / emotion_vector.sum()
        else:
            emotion_vector[6] = 1.0  # Default to neutral
        
        # Cache result
        self.cache[text_hash] = emotion_vector
        
        return emotion_vector
    
    def emotion_to_music(self, emotion_vector: np.ndarray) -> Dict:
        """Convert emotion to music parameters"""
        # Valence and arousal from emotion vector
        valence = emotion_vector[0] - emotion_vector[1]  # joy - sadness
        arousal = (emotion_vector[2] + emotion_vector[3] + emotion_vector[5]) / 3
        
        # Generate music parameters
        music = {
            "tempo_bpm": int(60 + arousal * 60),  # 60-120 BPM
            "key": "C" if valence > 0 else "A",
            "mode": "major" if valence > 0 else "minor",
            "dynamics": self._get_dynamics(arousal),
            "chord_progression": self._get_progression(valence, arousal),
            "complexity": float(np.std(emotion_vector))
        }
        
        return music
    
    def _get_dynamics(self, arousal: float) -> str:
        """Map arousal to musical dynamics"""
        if arousal < 0.2:
            return "pp"
        elif arousal < 0.4:
            return "p"
        elif arousal < 0.6:
            return "mf"
        elif arousal < 0.8:
            return "f"
        else:
            return "ff"
    
    def _get_progression(self, valence: float, arousal: float) -> List[str]:
        """Select chord progression based on emotion"""
        if valence > 0 and arousal > 0.5:
            return ["I", "V", "vi", "IV"]  # Uplifting
        elif valence > 0:
            return ["I", "IV", "V", "I"]   # Classical happy
        elif valence < 0 and arousal > 0.5:
            return ["i", "iv", "VII", "III"]  # Dramatic minor
        else:
            return ["i", "iv", "v", "i"]   # Sad/contemplative

# ============================================================================
# API MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for /analyze endpoint"""
    text: Optional[str] = Field(None, max_length=config.MAX_TEXT_LENGTH)
    eeg_data: Optional[List[float]] = Field(None, description="Raw EEG samples")
    eeg_bands: Optional[Dict[str, float]] = Field(None, description="Pre-computed band powers")
    sampling_rate: Optional[int] = Field(config.SAMPLING_RATE, description="EEG sampling rate in Hz")

class AnalyzeResponse(BaseModel):
    """Response model for /analyze endpoint"""
    emotion_vector: List[float]
    confidence: float
    music_parameters: Dict
    processing_time_ms: float
    eeg_quality: Optional[str] = None
    snr_db: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float

# ============================================================================
# FAST API APPLICATION
# ============================================================================

app = FastAPI(
    title="SHEMS Lightweight Demo API",
    description="Emotion-Music Analysis with EEG Support",
    version=config.VERSION
)

# Initialize components
eeg_analyzer = EEGAnalyzer(sampling_rate=config.SAMPLING_RATE)
emotion_engine = OptimizedEmotionEngine()
app_start_time = time.time()

@app.get("/", response_model=Dict)
async def root():
    """API information"""
    return {
        "name": "SHEMS Demo API",
        "version": config.VERSION,
        "endpoints": ["/analyze", "/health", "/docs"],
        "features": ["EEG Analysis", "Text Emotion", "Music Mapping"]
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Main analysis endpoint"""
    start_time = time.time()
    
    emotion_vectors = []
    eeg_quality = None
    snr_db = None
    
    try:
        # Process EEG if provided
        if request.eeg_data:
            # Convert to numpy array
            eeg_signal = np.array(request.eeg_data)
            
            # Process EEG
            eeg_result = eeg_analyzer.process_eeg(eeg_signal)
            
            # Map to emotion
            eeg_emotion = emotion_engine.map_eeg_to_emotion(eeg_result["band_powers"])
            emotion_vectors.append(eeg_emotion)
            
            eeg_quality = eeg_result["quality"]
            snr_db = eeg_result["snr"]
            
        elif request.eeg_bands:
            # Use pre-computed band powers
            eeg_emotion = emotion_engine.map_eeg_to_emotion(request.eeg_bands)
            emotion_vectors.append(eeg_emotion)
        
        # Process text if provided
        if request.text:
            text_emotion = emotion_engine.process_text(request.text)
            emotion_vectors.append(text_emotion)
        
        # Combine emotions if multiple sources
        if len(emotion_vectors) > 1:
            final_emotion = np.mean(emotion_vectors, axis=0)
        elif len(emotion_vectors) == 1:
            final_emotion = emotion_vectors[0]
        else:
            # No input provided
            raise HTTPException(status_code=400, detail="No input data provided")
        
        # Generate music parameters
        music_params = emotion_engine.emotion_to_music(final_emotion)
        
        # Calculate confidence
        confidence = 1.0 - np.std(final_emotion)  # Higher std = lower confidence
        
        processing_time = (time.time() - start_time) * 1000
        
        return AnalyzeResponse(
            emotion_vector=final_emotion.tolist(),
            confidence=float(confidence),
            music_parameters=music_params,
            processing_time_ms=processing_time,
            eeg_quality=eeg_quality,
            snr_db=snr_db
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy",
        version=config.VERSION,
        uptime_seconds=uptime
    )

# ============================================================================
# UNIT TESTS
# ============================================================================

import unittest

class TestEEGAnalyzer(unittest.TestCase):
    """Test EEG analysis functions"""
    
    def setUp(self):
        self.analyzer = EEGAnalyzer()
    
    def test_goertzel_algorithm(self):
        """Test Goertzel frequency detection"""
        # Generate test signal with known frequency
        t = np.linspace(0, 1, 256)  # 1 second at 256 Hz
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        
        # Test Goertzel for 10 Hz
        power = self.analyzer.goertzel_algorithm(signal, 10)
        self.assertGreater(power, 0.1) # Should detect significant power
        
        # Test Goertzel for a frequency not in signal
        power_noise = self.analyzer.goertzel_algorithm(signal, 5)
        self.assertLess(power_noise, 0.01) # Should detect very low power

    def test_extract_band_powers(self):
        """Test band power extraction"""
        # Generate a signal with dominant alpha and beta
        t = np.linspace(0, 2, 512) # 2 seconds at 256 Hz
        eeg_signal = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
        
        band_powers = self.analyzer.extract_band_powers(eeg_signal)
        
        self.assertIn('alpha', band_powers)
        self.assertIn('beta', band_powers)
        self.assertGreater(band_powers['alpha'], band_powers['delta'])
        self.assertGreater(band_powers['beta'], band_powers['theta'])

    def test_kalman_filter(self):
        """Test Kalman filter smoothing"""
        # Simulate noisy measurements
        measurements = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.15, 0.25, 0.35, 0.45, 0.55],
            [0.08, 0.18, 0.28, 0.38, 0.48]
        ])
        
        filtered_output = []
        for m in measurements:
            filtered_output.append(self.analyzer.kalman_filter(m))
        
        # Check if smoothing occurred (variance should decrease)
        original_variance = np.var(measurements[:, 0])
        filtered_variance = np.var(np.array(filtered_output)[:, 0])
        
        self.assertLess(filtered_variance, original_variance * 0.5) # Expect significant reduction

    def test_calculate_snr(self):
        """Test SNR calculation"""
        # High SNR signal (pure sine wave)
        t = np.linspace(0, 1, 256)
        signal_high_snr = np.sin(2 * np.pi * 10 * t)
        snr_high = self.analyzer.calculate_snr(signal_high_snr)
        self.assertGreater(snr_high, 30)
        
        # Low SNR signal (mostly noise)
        signal_low_snr = np.random.randn(256) * 0.5 + 0.01 * np.sin(2 * np.pi * 10 * t)
        snr_low = self.analyzer.calculate_snr(signal_low_snr)
        self.assertLess(snr_low, 10)

    def test_process_eeg(self):
        """Test full EEG processing pipeline"""
        t = np.linspace(0, 2, 512)
        raw_eeg = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t) + np.random.randn(512) * 0.1
        
        result = self.analyzer.process_eeg(raw_eeg)
        
        self.assertIn('band_powers', result)
        self.assertIn('snr', result)
        self.assertIn('quality', result)
        self.assertIsInstance(result['band_powers'], dict)
        self.assertIsInstance(result['snr'], float)
        self.assertIsInstance(result['quality'], str)

class TestOptimizedEmotionEngine(unittest.TestCase):
    """Test emotion engine functions"""
    
    def setUp(self):
        self.engine = OptimizedEmotionEngine()
    
    def test_map_eeg_to_emotion(self):
        """Test mapping EEG bands to emotion vector"""
        # Simulate joy
        band_powers_joy = {'delta': -0.1, 'theta': -0.05, 'alpha': 0.35, 'beta': 0.25, 'gamma': 0.15}
        emotion_joy = self.engine.map_eeg_to_emotion(band_powers_joy)
        self.assertGreater(emotion_joy[0], 0.5) # Joy should be dominant
        
        # Simulate sadness
        band_powers_sadness = {'delta': 0.05, 'theta': 0.20, 'alpha': -0.25, 'beta': 0.10, 'gamma': 0.05}
        emotion_sadness = self.engine.map_eeg_to_emotion(band_powers_sadness)
        self.assertGreater(emotion_sadness[1], 0.5) # Sadness should be dominant

    def test_process_text(self):
        """Test text emotion analysis"""
        # Test with Korean ending
        emotion_text1 = self.engine.process_text("오늘 정말 기쁘네요!")
        self.assertGreater(emotion_text1[0], 0.5) # Joy should be dominant
        
        # Test with keyword
        emotion_text2 = self.engine.process_text("너무 슬퍼요.")
        self.assertGreater(emotion_text2[1], 0.5) # Sadness should be dominant
        
        # Test neutral
        emotion_text3 = self.engine.process_text("그냥 그래요.")
        self.assertGreater(emotion_text3[6], 0.5) # Neutral should be dominant

    def test_emotion_to_music(self):
        """Test emotion to music mapping"""
        # Joyful emotion
        music_joy = self.engine.emotion_to_music(np.array([1.0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(music_joy['key'], 'C')
        self.assertEqual(music_joy['mode'], 'major')
        self.assertGreater(music_joy['tempo_bpm'], 90)
        
        # Sad emotion
        music_sad = self.engine.emotion_to_music(np.array([0, 1.0, 0, 0, 0, 0, 0]))
        self.assertEqual(music_sad['key'], 'A')
        self.assertEqual(music_sad['mode'], 'minor')
        self.assertLess(music_sad['tempo_bpm'], 90)

if __name__ == '__main__':
    # This allows running unit tests directly
    # For API server, use `uvicorn lightweight_demo_api:app --reload`
    unittest.main(argv=['first-arg-is-ignored'], exit=False)