import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import sqlite3
import pickle
from datetime import datetime
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class EmotionLexiconEntry:
    """Single entry in the emotion lexicon with complete metadata"""
    word: str
    emotion_vector: List[float]  # 7D vector [joy, sadness, anger, fear, disgust, surprise, neutral]
    confidence: float
    frequency: float  # Word frequency in corpus
    context_dependency: float  # How much context affects emotion
    cultural_specificity: float  # Korean-specific emotional nuance
    phonetic_features: Dict[str, float] = field(default_factory=dict)
    morphological_features: Dict[str, Any] = field(default_factory=dict)
    semantic_relations: List[str] = field(default_factory=list)  # Related words
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for computation"""
        return np.array(self.emotion_vector)
    
    def similarity(self, other: 'EmotionLexiconEntry') -> float:
        """Compute similarity with another entry"""
        v1 = self.to_vector()
        v2 = other.to_vector()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


@dataclass 
class MusicMapping:
    """Complete music mapping for an emotional state"""
    emotion_state: List[float]  # 7D emotion vector
    tempo_range: Tuple[float, float]  # BPM range
    key_preferences: Dict[str, float]  # Key -> probability
    mode_distribution: Dict[str, float]  # Mode -> probability  
    chord_progressions: List[List[str]]  # Possible progressions
    rhythm_patterns: List[List[float]]  # Rhythmic patterns
    dynamics_curve: List[float]  # Dynamic progression
    texture_density: float  # Polyphonic complexity
    harmonic_rhythm: float  # Chord change frequency
    melodic_contour: str  # ascending, descending, arch, wave
    
    def select_optimal_progression(self, context: Dict = None) -> List[str]:
        """Select best chord progression given context"""
        if not self.chord_progressions:
            return ["I", "IV", "V", "I"]  # Default
        
        # Context-aware selection
        if context and 'energy' in context:
            energy = context['energy']
            # Higher energy -> more movement
            if energy > 0.7:
                # Prefer progressions with more chord changes
                return max(self.chord_progressions, key=len)
        
        return self.chord_progressions[0]


@dataclass
class KoreanMorpheme:
    """Korean morpheme with complete linguistic analysis"""
    surface: str  # Surface form
    lemma: str  # Base form
    pos_tag: str  # Part of speech
    semantic_role: str  # Semantic function
    emotion_modifier: List[float]  # How it modifies base emotion
    politeness_level: int  # 1-7 scale
    formality: float  # 0-1 scale
    grammatical_mood: str  # declarative, interrogative, imperative, etc.
    
    def apply_to_emotion(self, base_emotion: np.ndarray) -> np.ndarray:
        """Apply morpheme's emotional modification"""
        modifier = np.array(self.emotion_modifier)
        # Weighted combination
        return 0.7 * base_emotion + 0.3 * modifier


# ============================================================================
# DATA STORAGE & RETRIEVAL
# ============================================================================

class QuantumDataStore:
    """High-performance data storage with multiple backends"""
    
    def __init__(self, base_path: Path = Path("data")):
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize storage backends
        self.memory_cache = {}  # In-memory cache
        self.db_path = self.base_path / "cosmos.db"
        self.initialize_database()
        
        # Load core datasets
        self.emotion_lexicon: Dict[str, EmotionLexiconEntry] = {}
        self.music_mappings: List[MusicMapping] = []
        self.korean_morphemes: Dict[str, KoreanMorpheme] = {}
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def initialize_database(self):
        """Initialize SQLite database with optimized schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Emotion lexicon table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emotion_lexicon (
                word TEXT PRIMARY KEY,
                emotion_vector BLOB,
                confidence REAL,
                frequency REAL,
                context_dependency REAL,
                cultural_specificity REAL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Music mappings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS music_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emotion_hash TEXT UNIQUE,
                emotion_vector BLOB,
                mapping_data JSON,
                confidence REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Korean morphemes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS korean_morphemes (
                surface TEXT PRIMARY KEY,
                lemma TEXT,
                pos_tag TEXT,
                emotion_modifier BLOB,
                linguistic_features JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion_confidence ON emotion_lexicon(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_music_usage ON music_mappings(usage_count)")
        
        conn.commit()
        conn.close()
    
    async def load_emotion_lexicon(self, path: Optional[Path] = None) -> Dict[str, EmotionLexiconEntry]:
        """Load emotion lexicon from file or database"""
        
        # Try cache first
        cache_key = "emotion_lexicon"
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        self.cache_misses += 1
        
        # Load from file if provided
        if path and path.exists():
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            if path.suffix == '.json':
                data = json.loads(content)
            elif path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(content)
            else:
                # Assume CSV
                df = pd.read_csv(path)
                data = df.to_dict('records')
            
            # Convert to EmotionLexiconEntry objects
            lexicon = {}
            for item in data:
                if isinstance(item, dict):
                    entry = EmotionLexiconEntry(
                        word=item['word'],
                        emotion_vector=item.get('emotion_vector', [0]*7),
                        confidence=item.get('confidence', 0.5),
                        frequency=item.get('frequency', 0.0),
                        context_dependency=item.get('context_dependency', 0.3),
                        cultural_specificity=item.get('cultural_specificity', 0.5)
                    )
                    lexicon[entry.word] = entry
                    
            # Store in database
            await self.store_emotion_lexicon(lexicon)
            
        else:
            # Load from database
            lexicon = await self._load_lexicon_from_db()
        
        # Cache the result
        self.memory_cache[cache_key] = lexicon
        self.emotion_lexicon = lexicon
        
        return lexicon
    
    async def _load_lexicon_from_db(self) -> Dict[str, EmotionLexiconEntry]:
        """Load lexicon from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT word, emotion_vector, confidence, frequency, 
                   context_dependency, cultural_specificity, metadata
            FROM emotion_lexicon
        """)
        
        lexicon = {}
        for row in cursor.fetchall():
            word, emotion_blob, conf, freq, context, cultural, meta_json = row
            
            entry = EmotionLexiconEntry(
                word=word,
                emotion_vector=pickle.loads(emotion_blob) if emotion_blob else [0]*7,
                confidence=conf,
                frequency=freq,
                context_dependency=context,
                cultural_specificity=cultural
            )
            
            if meta_json:
                metadata = json.loads(meta_json)
                entry.phonetic_features = metadata.get('phonetic_features', {})
                entry.morphological_features = metadata.get('morphological_features', {})
                entry.semantic_relations = metadata.get('semantic_relations', [])
            
            lexicon[word] = entry
        
        conn.close()
        return lexicon
    
    async def store_emotion_lexicon(self, lexicon: Dict[str, EmotionLexiconEntry]):
        """Store emotion lexicon to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for word, entry in lexicon.items():
            metadata = {
                'phonetic_features': entry.phonetic_features,
                'morphological_features': entry.morphological_features,
                'semantic_relations': entry.semantic_relations
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO emotion_lexicon 
                (word, emotion_vector, confidence, frequency, 
                 context_dependency, cultural_specificity, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                word,
                pickle.dumps(entry.emotion_vector),
                entry.confidence,
                entry.frequency,
                entry.context_dependency,
                entry.cultural_specificity,
                json.dumps(metadata)
            ))
        
        conn.commit()
        conn.close()
        
        # Invalidate cache
        if "emotion_lexicon" in self.memory_cache:
            del self.memory_cache["emotion_lexicon"]
    
    def get_music_mapping(self, emotion_vector: np.ndarray) -> Optional[MusicMapping]:
        """Retrieve best music mapping for emotion vector"""
        
        # Generate hash for lookup
        emotion_hash = hashlib.md5(emotion_vector.tobytes()).hexdigest()
        
        # Check cache
        cache_key = f"music_{emotion_hash}"
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        self.cache_misses += 1
        
        # Find closest mapping
        best_mapping = None
        min_distance = float('inf')
        
        for mapping in self.music_mappings:
            distance = np.linalg.norm(emotion_vector - np.array(mapping.emotion_state))
            if distance < min_distance:
                min_distance = distance
                best_mapping = mapping
        
        # If no close mapping, generate new one
        if min_distance > 0.5 or best_mapping is None:
            best_mapping = self._generate_music_mapping(emotion_vector)
            self.music_mappings.append(best_mapping)
        
        # Cache result
        self.memory_cache[cache_key] = best_mapping
        
        # Update usage count in database
        self._update_mapping_usage(emotion_hash)
        
        return best_mapping
    
    def _generate_music_mapping(self, emotion_vector: np.ndarray) -> MusicMapping:
        """Generate new music mapping for emotion"""
        
        # Extract key features
        valence = emotion_vector[0] - emotion_vector[1]  # joy - sadness
        arousal = (emotion_vector[2] + emotion_vector[3] + emotion_vector[5]) / 3  # anger + fear + surprise
        
        # Tempo based on arousal
        base_tempo = 60 + arousal * 60
        tempo_range = (base_tempo - 10, base_tempo + 10)
        
        # Key preferences based on emotion
        if valence > 0:
            key_prefs = {"C": 0.3, "G": 0.3, "D": 0.2, "A": 0.2}
            mode_dist = {"major": 0.7, "minor": 0.3}
        else:
            key_prefs = {"A": 0.3, "E": 0.3, "D": 0.2, "G": 0.2}
            mode_dist = {"minor": 0.7, "major": 0.3}
        
        # Chord progressions
        if valence > 0 and arousal > 0:
            progressions = [
                ["I", "V", "vi", "IV"],  # Pop
                ["I", "IV", "V", "I"]     # Classic
            ]
        elif valence < 0 and arousal > 0:
            progressions = [
                ["i", "iv", "VII", "III"],  # Dramatic minor
                ["i", "VII", "iv", "i"]     # Alternative minor
            ]
        else:
            progressions = [
                ["I", "vi", "IV", "V"],     # Contemplative
                ["i", "iv", "v", "i"]       # Traditional minor
            ]
        
        return MusicMapping(
            emotion_state=emotion_vector.tolist(),
            tempo_range=tempo_range,
            key_preferences=key_prefs,
            mode_distribution=mode_dist,
            chord_progressions=progressions,
            rhythm_patterns=[[0.25, 0.25, 0.5] * 4],
            dynamics_curve=[0.5, 0.6, 0.7, 0.6],
            texture_density=0.5 + arousal * 0.3,
            harmonic_rhythm=0.5 + valence * 0.2,
            melodic_contour="arch" if valence > 0 else "descending"
        )
    
    def _update_mapping_usage(self, emotion_hash: str):
        """Update usage count for mapping"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE music_mappings 
            SET usage_count = usage_count + 1 
            WHERE emotion_hash = ?
        """, (emotion_hash,))
        
        conn.commit()
        conn.close()
    
    async def augment_korean_data(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Augment Korean emotion data with variations"""
        
        augmented_rows = []
        
        # Korean-specific augmentation strategies
        endings_variations = {
            '-네요': ['-네', '-군요', '-구나'],
            '-습니다': ['-어요', '-아요', '-ㅂ니다'],
            '-잖아요': ['-잖아', '-지 않아요'],
            '-거든요': ['-거든', '-는데요']
        }
        
        emotion_modifiers = {
            '정말': 1.3,
            '너무': 1.4,
            '조금': 0.7,
            '약간': 0.6,
            '엄청': 1.5
        }
        
        for _, row in base_data.iterrows():
            # Original row
            augmented_rows.append(row.to_dict())
            
            text = row['text']
            base_emotion = np.array(row['emotion_vector'])
            
            # Augment with ending variations
            for original, variations in endings_variations.items():
                if original in text:
                    for variant in variations:
                        new_text = text.replace(original, variant)
                        # Slightly modify emotion based on formality change
                        formality_shift = np.random.randn(7) * 0.05
                        new_emotion = base_emotion + formality_shift
                        
                        augmented_rows.append({
                            'text': new_text,
                            'emotion_vector': new_emotion.tolist(),
                            'source': 'augmented_ending',
                            'original_text': text
                        })
            
            # Augment with intensity modifiers
            for modifier_word, intensity_factor in emotion_modifiers.items():
                if modifier_word in text:
                    new_text = f"{modifier_word} {text}"
                    new_emotion = base_emotion * intensity_factor
                    augmented_rows.append({
                        'text': new_text,
                        'emotion_vector': new_emotion.tolist(),
                        'source': 'augmented_modifier',
                        'original_text': text
                    })
        
        return pd.DataFrame(augmented_rows)

    async def get_korean_morpheme(self, surface: str) -> Optional[KoreanMorpheme]:
        """Retrieve Korean morpheme by surface form"""
        # Check cache
        cache_key = f"morpheme_{surface}"
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        self.cache_misses += 1
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT surface, lemma, pos_tag, emotion_modifier, linguistic_features
            FROM korean_morphemes
            WHERE surface = ?
        """, (surface,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            surface, lemma, pos_tag, emotion_blob, features_json = row
            morpheme = KoreanMorpheme(
                surface=surface,
                lemma=lemma,
                pos_tag=pos_tag,
                emotion_modifier=pickle.loads(emotion_blob) if emotion_blob else [0]*7,
                semantic_role=features_json.get('semantic_role', 'N/A'),
                politeness_level=features_json.get('politeness_level', 1),
                formality=features_json.get('formality', 0.5),
                grammatical_mood=features_json.get('grammatical_mood', 'declarative')
            )
            self.memory_cache[cache_key] = morpheme
            return morpheme
        
        return None
    
    async def store_korean_morphemes(self, morphemes: Dict[str, KoreanMorpheme]):
        """Store Korean morphemes to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for surface, entry in morphemes.items():
            linguistic_features = {
                'semantic_role': entry.semantic_role,
                'politeness_level': entry.politeness_level,
                'formality': entry.formality,
                'grammatical_mood': entry.grammatical_mood
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO korean_morphemes 
                (surface, lemma, pos_tag, emotion_modifier, linguistic_features)
                VALUES (?, ?, ?, ?, ?)
            """, (
                surface,
                entry.lemma,
                entry.pos_tag,
                pickle.dumps(entry.emotion_modifier),
                json.dumps(linguistic_features)
            ))
        
        conn.commit()
        conn.close()
        
        # Invalidate cache
        for surface in morphemes.keys():
            cache_key = f"morpheme_{surface}"
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics"""
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {
            'emotion_lexicon_size': len(self.emotion_lexicon),
            'music_mappings_count': len(self.music_mappings),
            'korean_morphemes_count': len(self.korean_morphemes),
            'cache_stats': self.get_cache_stats()
        }
        
        # Additional stats from emotion lexicon
        if self.emotion_lexicon:
            all_emotions = np.array([entry.emotion_vector for entry in self.emotion_lexicon.values()])
            stats['emotion_distribution'] = {
                'mean': all_emotions.mean(axis=0).tolist(),
                'std': all_emotions.std(axis=0).tolist()
            }
        
        # Additional stats from music mappings
        if self.music_mappings:
            all_tempos = [m.tempo_range[0] for m in self.music_mappings]
            all_keys = {}
            for m in self.music_mappings:
                for k, v in m.key_preferences.items():
                    all_keys[k] = all_keys.get(k, 0) + v
            
            stats['music_preferences'] = {
                'avg_tempo': np.mean(all_tempos),
                'tempo_std': np.std(all_tempos),
                'key_distribution': all_keys
            }
        
        return stats


# ============================================================================
# DATA QUALITY & VALIDATION
# ============================================================================

class DataQualityValidator:
    """Ensure data quality and consistency across all layers"""
    
    def __init__(self):
        self.validation_rules = {
            'emotion_vector': self.validate_emotion_vector,
            'music_mapping': self.validate_music_mapping,
            'korean_text': self.validate_korean_text
        }
        self.quality_metrics = {}
    
    def validate_emotion_vector(self, vector: Union[List, np.ndarray]) -> Tuple[bool, str]:
        """Validate emotion vector format and values"""
        
        # Convert to numpy if needed
        if isinstance(vector, list):
            vector = np.array(vector)
        
        # Check dimensions
        if vector.shape != (7,):
            return False, f"Invalid shape: {vector.shape}, expected (7,)"
        
        # Check range (z-scale typically -3 to 3)
        if np.any(np.abs(vector) > 3):
            return False, f"Values outside expected range: {vector}"
        
        # Check for NaN or Inf
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            return False, "Contains NaN or Inf values"
        
        return True, "Valid"
    
    def validate_music_mapping(self, mapping: MusicMapping) -> Tuple[bool, str]:
        """Validate music mapping consistency"""
        
        # Check tempo range
        if mapping.tempo_range[0] > mapping.tempo_range[1]:
            return False, "Invalid tempo range"
        
        if not (20 <= mapping.tempo_range[0] <= 300):
            return False, "Tempo outside reasonable range"
        
        # Check probability distributions
        key_sum = sum(mapping.key_preferences.values())
        if abs(key_sum - 1.0) > 0.01:
            return False, f"Key probabilities don't sum to 1: {key_sum}"
        
        mode_sum = sum(mapping.mode_distribution.values())
        if abs(mode_sum - 1.0) > 0.01:
            return False, f"Mode probabilities don't sum to 1: {mode_sum}"
        
        # Check chord progressions
        valid_chords = {"I", "i", "II", "ii", "III", "iii", "IV", "iv", 
                       "V", "v", "VI", "vi", "VII", "vii", "bVII", "bIII", "bVI"}
        
        for progression in mapping.chord_progressions:
            for chord in progression:
                base_chord = chord.rstrip('7965sus').rstrip('dim').rstrip('aug')
                if base_chord not in valid_chords:
                    return False, f"Invalid chord: {chord}"
        
        return True, "Valid"
    
    def validate_korean_text(self, text: str) -> Tuple[bool, str]:
        """Validate Korean text format and encoding"""
        
        # Check for proper encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            return False, "Encoding error"
        
        # Check for Korean characters
        korean_chars = sum(1 for c in text if '\uAC00' <= c <= '\uD7A3')
        if korean_chars == 0 and len(text) > 10:
            return False, "No Korean characters found"
        
        # Check for mixed scripts (potential data corruption)
        has_korean = any('\uAC00' <= c <= '\uD7A3' for c in text)
        has_chinese = any('\u4E00' <= c <= '\u9FFF' for c in text)
        has_japanese = any('\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' for c in text)
        
        mixed_count = sum([has_korean, has_chinese, has_japanese])
        if mixed_count > 1:
            logger.warning(f"Mixed scripts detected in: {text[:50]}")
        
        return True, "Valid"
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate entire dataset and return quality metrics"""
        
        results = {
            'total_rows': len(df),
            'valid_rows': 0,
            'invalid_rows': 0,
            'issues': []
        }
        
        for idx, row in df.iterrows():
            row_valid = True
            
            # Validate emotion vector if present
            if 'emotion_vector' in row:
                valid, msg = self.validate_emotion_vector(row['emotion_vector'])
                if not valid:
                    row_valid = False
                    results['issues'].append(f"Row {idx}: {msg}")
            
            # Validate text if present
            if 'text' in row:
                valid, msg = self.validate_korean_text(row['text'])
                if not valid:
                    row_valid = False
                    results['issues'].append(f"Row {idx}: {msg}")
            
            if row_valid:
                results['valid_rows'] += 1
            else:
                results['invalid_rows'] += 1
        
        results['quality_score'] = results['valid_rows'] / results['total_rows']
        
        # Additional quality metrics
        if 'emotion_vector' in df.columns:
            emotions = np.array(df['emotion_vector'].tolist())
            results['emotion_metrics'] = {
                'mean': emotions.mean(axis=0).tolist(),
                'std': emotions.std(axis=0).tolist(),
                'correlation': np.corrcoef(emotions.T).tolist()
            }
        
        self.quality_metrics = results
        return results
    
    def generate_quality_report(self) -> str:
        """Generate human-readable quality report"""
        
        if not self.quality_metrics:
            return "No quality metrics available. Run validate_dataset first."
        
        report = []
        report.append("="*60)
        report.append("DATA QUALITY REPORT")
        report.append("="*60)
        report.append(f"Total Rows: {self.quality_metrics['total_rows']}")
        report.append(f"Valid Rows: {self.quality_metrics['valid_rows']}")
        report.append(f"Invalid Rows: {self.quality_metrics['invalid_rows']}")
        report.append(f"Quality Score: {self.quality_metrics['quality_score']:.2%}")
        
        if self.quality_metrics['issues']:
            report.append("\nTop Issues:")
            for issue in self.quality_metrics['issues'][:10]:
                report.append(f"  - {issue}")
        
        if 'emotion_metrics' in self.quality_metrics:
            report.append("\nEmotion Distribution:")
            emotions = ['Joy', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise', 'Neutral']
            means = self.quality_metrics['emotion_metrics']['mean']
            for emotion, mean in zip(emotions, means):
                report.append(f"  {emotion}: {mean:.3f}")
        
        report.append("="*60)
        
        return "\n".join(report)


# ============================================================================
# INTELLIGENT DATA LOADER
# ============================================================================

class IntelligentDataLoader:
    """Smart data loading with automatic format detection and optimization"""
    
    def __init__(self, data_store: QuantumDataStore):
        self.data_store = data_store
        self.validator = DataQualityValidator()
        self.supported_formats = ['.json', '.yaml', '.yml', '.csv', '.parquet', '.pkl', '.txt']
        
    async def load_auto(self, path: Path) -> pd.DataFrame:
        """Automatically detect format and load data"""
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Load based on format
        if suffix == '.json':
            df = await self._load_json(path)
        elif suffix in ['.yaml', '.yml']:
            df = await self._load_yaml(path)
        elif suffix == '.csv':
            df = await self._load_csv(path)
        elif suffix == '.parquet':
            df = await self._load_parquet(path)
        elif suffix == '.pkl':
            df = await self._load_pickle(path)
        elif suffix == '.txt':
            df = await self._load_text(path)
        else:
            raise ValueError(f"Handler not implemented for: {suffix}")
        
        # Validate loaded data
        validation_results = self.validator.validate_dataset(df)
        logger.info(f"Loaded {len(df)} rows, quality score: {validation_results['quality_score']:.2%}")
        
        # Auto-correct common issues
        df = self._auto_correct(df, validation_results)
        
        return df
    
    async def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON data"""
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        data = json.loads(content)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try to infer structure
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
    
    async def _load_yaml(self, path: Path) -> pd.DataFrame:
        """Load YAML data"""
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        data = yaml.safe_load(content)
        
        # Similar handling as JSON
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
    
    async def _load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV data with intelligent parsing"""
        # Try to detect encoding
        encodings = ['utf-8', 'cp949', 'euc-kr']  # Common Korean encodings
        
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                
                # Try to parse emotion vectors if stored as strings
                if 'emotion_vector' in df.columns:
                    df['emotion_vector'] = df['emotion_vector'].apply(
                        lambda x: json.loads(x) if isinstance(x, str) else x
                    )
                
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        raise ValueError(f"Could not parse CSV with any encoding: {encodings}")
    
    async def _load_parquet(self, path: Path) -> pd.DataFrame:
        """Load Parquet data"""
        return pd.read_parquet(path)
    
    async def _load_pickle(self, path: Path) -> pd.DataFrame:
        """Load pickled data"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            return data
        else:
            return pd.DataFrame([data])
    
    async def _load_text(self, path: Path) -> pd.DataFrame:
        """Load plain text data (Korean emotion words)"""
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            lines = await f.readlines()
        
        data = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for section headers
            if line.startswith('단어,'):
                current_section = 'words'
                continue
            
            if current_section == 'words' and ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    word = parts[0].strip()
                    category = parts[1].strip()
                    
                    # Map category to emotion vector
                    emotion_vector = self._category_to_emotion(category)
                    
                    data.append({
                        'word': word,
                        'category': category,
                        'emotion_vector': emotion_vector
                    })
        
        return pd.DataFrame(data)
    
    def _category_to_emotion(self, category: str) -> List[float]:
        """Map emotion category to vector"""
        category_map = {
            '기쁨': [0.8, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            '슬픔': [0.0, 0.8, 0.0, 0.1, 0.0, 0.0, 0.1],
            '분노': [0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 0.0],
            '공포': [0.0, 0.1, 0.0, 0.8, 0.0, 0.1, 0.0],
            '혐오': [0.0, 0.0, 0.1, 0.0, 0.8, 0.0, 0.1],
            '놀람': [0.1, 0.0, 0.0, 0.1, 0.0, 0.8, 0.0],
            '중성': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            '흥미': [0.3, 0.0, 0.0, 0.0, 0.0, 0.3, 0.4],
            '지루함': [-0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.6],
            '통증': [0.0, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0],
            '기타': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        return category_map.get(category, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    
    def _auto_correct(self, df: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """Auto-correct common data issues"""
        
        # Fill missing emotion vectors with neutral
        if 'emotion_vector' in df.columns:
            df['emotion_vector'] = df['emotion_vector'].apply(
                lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) == 7 
                else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            )
        
        # Remove duplicate entries
        if 'text' in df.columns:
            df = df.drop_duplicates(subset=['text'])
        elif 'word' in df.columns:
            df = df.drop_duplicates(subset=['word'])
        
        # Normalize emotion vectors
        if 'emotion_vector' in df.columns:
            df['emotion_vector'] = df['emotion_vector'].apply(
                lambda x: np.array(x) / (np.linalg.norm(x) + 1e-10) if np.linalg.norm(x) > 1.5 else x
            )
        
        logger.info(f"Auto-corrected data: {len(df)} rows remaining")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def demonstrate_data_layer():
    """Demonstrate the unified data layer capabilities"""
    
    # Initialize components
    data_store = QuantumDataStore()
    validator = DataQualityValidator()
    loader = IntelligentDataLoader(data_store)
    
    print("="*60)
    print("UNIFIED DATA LAYER DEMONSTRATION")
    print("="*60)
    
    # Load emotion words from text file
    emotion_words_path = Path("new_emotion_words.txt")
    if emotion_words_path.exists():
        print("\n1. Loading Korean emotion words...")
        df = await loader.load_auto(emotion_words_path)
        print(f"   Loaded {len(df)} emotion words")
        
        # Validate
        validation = validator.validate_dataset(df)
        print(f"   Quality Score: {validation['quality_score']:.2%}")
        
        # Augment data
        print("\n2. Augmenting Korean data...")
        augmented = await data_store.augment_korean_data(df)
        print(f"   Generated {len(augmented)} augmented samples")
        
        # Store in data store
        print("\n3. Building emotion lexicon...")
        lexicon = {}
        for _, row in augmented.iterrows():
            if 'word' in row:
                entry = EmotionLexiconEntry(
                    word=row['word'],
                    emotion_vector=row.get('emotion_vector', [0]*7),
                    confidence=0.7,
                    frequency=0.1,
                    context_dependency=0.3,
                    cultural_specificity=0.8
                )
                lexicon[row['word']] = entry
        
        await data_store.store_emotion_lexicon(lexicon)
        print(f"   Stored {len(lexicon)} lexicon entries")
    
    # Generate music mappings
    print("\n4. Generating music mappings...")
    test_emotions = [
        np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1]), # Joy
        np.array([0.0, 0.7, 0.0, 0.1, 0.0, 0.0, 0.2]), # Sadness
        np.array([0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0])  # Anger
    ]
    
    for emotion_vec in test_emotions:
        mapping = data_store.get_music_mapping(emotion_vec)
        if mapping:
            print(f"   Emotion: {emotion_vec[:3]}..., Tempo: {mapping.tempo_range[0]:.0f} BPM, Key: {list(mapping.key_preferences.keys())[0]}")
    
    print(f"\nCache Stats: {data_store.get_cache_stats()}")
    print(f"System Stats: {data_store.get_system_stats()}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(demonstrate_data_layer())


