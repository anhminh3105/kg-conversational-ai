"""
Natural Language to Cypher Query Converter
A lightweight ML model that converts natural language questions into Cypher queries
"""

import re
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import fasttext
import fasttext.util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
import joblib
import warnings

from neo4j import GraphDatabase


class QuestionEncoder:
    """Encodes questions into embeddings using FastText"""
    
    def __init__(self, model_name='cc.en.300.bin'):
        print(f"Loading FastText model: {model_name}...")
        print("(This may take a moment on first run - downloading if needed)")
        
        try:
            # Try to load local model first
            self.model = fasttext.load_model(model_name)
            print("✓ Model loaded successfully from local file")
        except:
            # Download pretrained model if not found
            print(f"Downloading FastText model (this is a one-time download)...")
            try:
                fasttext.util.download_model('en', if_exists='ignore')
                self.model = fasttext.load_model('cc.en.300.bin')
                print("✓ Model downloaded and loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Attempting to use smaller FastText model...")
                # Fallback to compressed model
                try:
                    fasttext.util.download_model('en', if_exists='ignore')
                    self.model = fasttext.load_model('cc.en.300.bin')
                    fasttext.util.reduce_model(self.model, 100)  # Reduce to 100 dimensions
                    print("✓ Compressed model loaded (100-dim)")
                except:
                    raise ImportError("Could not load FastText model. Please download manually.")
        
        self.dim = self.model.get_dimension()
        print(f"✓ Model ready (embedding dimension: {self.dim})")
    
    def encode(self, texts):
        """
        Encode text(s) into embeddings using FastText
        
        Args:
            texts: String or list of strings
            
        Returns:
            numpy array of embeddings (300-dim by default for FastText)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Tokenize and get word vectors
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                try:
                    vec = self.model.get_word_vector(word)
                    word_vectors.append(vec)
                except:
                    pass  # Skip words that can't be embedded
            
            # Average word vectors to get sentence embedding
            if word_vectors:
                sentence_embedding = np.mean(word_vectors, axis=0)
            else:
                # Fallback: zero vector if no words can be embedded
                sentence_embedding = np.zeros(self.dim)
            
            embeddings.append(sentence_embedding)
        
        return np.array(embeddings)


class IntentClassifier:
    """Classifies question intent using ML"""
    
    def __init__(self):
        # if not DEPENDENCIES_AVAILABLE:
        #     raise ImportError("Please install: pip install scikit-learn")
        
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.encoder = None
        self.is_trained = False
        self.label_names = None
    
    def train(self, questions: List[str], labels: List[str]):
        """
        Train the intent classifier
        
        Args:
            questions: List of training questions
            labels: List of corresponding intent labels
        """
        if self.encoder is None:
            self.encoder = QuestionEncoder()
        
        print(f"\nTraining intent classifier on {len(questions)} examples...")
        
        # Get unique labels
        self.label_names = sorted(set(labels))
        print(f"Intent categories: {self.label_names}")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.encoder.encode(questions)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(embeddings, labels)
        self.is_trained = True
        
        print("✓ Training complete!")
    
    def predict(self, question: str) -> Tuple[str, float]:
        """
        Predict intent for a question
        
        Args:
            question: Input question
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        embedding = self.encoder.encode([question])
        intent = self.classifier.predict(embedding)[0]
        confidence = self.classifier.predict_proba(embedding).max()
        
        return intent, confidence
    
    def save(self, path: str):
        """Save the trained classifier"""
        joblib.dump({
            'classifier': self.classifier,
            'label_names': self.label_names,
            'is_trained': self.is_trained
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained classifier"""
        data = joblib.load(path)
        self.classifier = data['classifier']
        self.label_names = data['label_names']
        self.is_trained = data['is_trained']
        if self.encoder is None:
            self.encoder = QuestionEncoder()
        print(f"✓ Model loaded from {path}")


class EntityExtractor:
    """Extract entities (actors, movies) from questions"""
    
    def __init__(self, neo4j_driver=None, use_spacy=True):
        self.driver = neo4j_driver
        self.known_entities = {'actors': [], 'movies': []}
        self.entity_embeddings = None
        self.encoder = None
        self.nlp = None
        
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("✓ SpaCy model loaded")
                except OSError:
                    print("Warning: SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            except ImportError:
                print("Warning: SpaCy not installed. Run: pip install spacy")
        
        # Load known entities from database
        if neo4j_driver:
            self._load_entities_from_db()
    
    def _load_entities_from_db(self):
        """Load known actors and movies from Neo4j"""
        print("\nLoading entities from database...")
        
        with self.driver.session() as session:
            # Get all actors
            result = session.run("MATCH (p:Person) RETURN p.name as name")
            self.known_entities['actors'] = [record['name'] for record in result]
            
            # Get all movies
            result = session.run("MATCH (m:Movie) RETURN m.title as title")
            self.known_entities['movies'] = [record['title'] for record in result]
        
        print(f"✓ Loaded {len(self.known_entities['actors'])} actors")
        print(f"✓ Loaded {len(self.known_entities['movies'])} movies")
        
        # Pre-compute embeddings for similarity matching
        self.encoder = QuestionEncoder()
        all_entities = self.known_entities['actors'] + self.known_entities['movies']
        if all_entities:
            self.entity_embeddings = self.encoder.encode(all_entities)
    
    def extract(self, question: str) -> Dict[str, List[str]]:
        """
        Extract entities from question
        
        Args:
            question: Input question
            
        Returns:
            Dict with 'actors', 'movies', 'genres', and 'years' lists
        """
        entities = {'actors': [], 'movies': [], 'genres': [], 'years': []}
        
        # Extract year (4-digit numbers starting with 19 or 20)
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
        if year_matches:
            entities['years'] = [int(year) for year in year_matches]
        
        # Extract genre (common movie genres)
        genres = ['action', 'drama', 'comedy', 'thriller', 'horror', 'sci-fi', 
                  'science fiction', 'romance', 'documentary', 'fantasy', 'adventure']
        question_lower = question.lower()
        for genre in genres:
            if genre in question_lower:
                # Capitalize properly
                if genre == 'sci-fi':
                    entities['genres'].append('Sci-Fi')
                elif genre == 'science fiction':
                    entities['genres'].append('Sci-Fi')
                else:
                    entities['genres'].append(genre.capitalize())
        
        # Method 1: Use SpaCy NER
        if self.nlp:
            doc = self.nlp(question)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    # Match to known actors
                    matched = self._find_similar_entity(ent.text, 'actors')
                    if matched:
                        entities['actors'].append(matched)
                elif ent.label_ == 'WORK_OF_ART':
                    # Match to known movies
                    matched = self._find_similar_entity(ent.text, 'movies')
                    if matched:
                        entities['movies'].append(matched)
        
        # Method 2: Pattern matching for quoted strings (movies often in quotes)
        quoted = re.findall(r'"([^"]+)"', question)
        for q in quoted:
            matched = self._find_similar_entity(q, 'movies')
            if matched and matched not in entities['movies']:
                entities['movies'].append(matched)
        
        # Method 3: Match capitalized phrases to known entities (case-insensitive now)
        words = question.split()
        for i in range(len(words)):
            for length in [3, 2, 1]:  # Try 3-word, 2-word, 1-word phrases
                if i + length <= len(words):
                    phrase = ' '.join(words[i:i+length])
                    # Remove common words and punctuation
                    phrase = re.sub(r'[?!.,]', '', phrase)
                    
                    if phrase:  # Removed capitalization requirement
                        # Try matching to actors
                        matched = self._find_similar_entity(phrase, 'actors')
                        if matched and matched not in entities['actors']:
                            entities['actors'].append(matched)
                        
                        # Try matching to movies
                        matched = self._find_similar_entity(phrase, 'movies')
                        if matched and matched not in entities['movies']:
                            entities['movies'].append(matched)
        
        return entities
    
    def _find_similar_entity(self, text: str, entity_type: str, threshold: float = 0.60) -> Optional[str]:
        """
        Find most similar entity using embeddings
        
        Args:
            text: Text to match
            entity_type: 'actors' or 'movies'
            threshold: Similarity threshold (0-1), lowered to 0.60 for better matching
            
        Returns:
            Matched entity name or None
        """
        known = self.known_entities.get(entity_type, [])
        
        if not known:
            return None
        
        # Exact match first
        if text in known:
            return text
        
        # Case-insensitive exact match
        for entity in known:
            if text.lower() == entity.lower():
                return entity
        
        # Partial match (for "Hanks" matching "Tom Hanks")
        text_lower = text.lower()
        for entity in known:
            entity_lower = entity.lower()
            if text_lower in entity_lower or entity_lower in text_lower:
                # If one is substring of other (minimum 4 chars to avoid false matches)
                if len(text_lower) >= 4 or len(entity_lower) >= 4:
                    return entity
        
        # Embedding-based similarity matching
        if self.encoder and self.entity_embeddings is not None and known:
            text_emb = self.encoder.encode([text])
            
            # Get embeddings for this entity type
            all_entities = self.known_entities['actors'] + self.known_entities['movies']
            start_idx = 0 if entity_type == 'actors' else len(self.known_entities['actors'])
            end_idx = len(self.known_entities['actors']) if entity_type == 'actors' else len(all_entities)
            
            type_embeddings = self.entity_embeddings[start_idx:end_idx]
            similarities = cosine_similarity(text_emb, type_embeddings)[0]
            
            best_idx = similarities.argmax()
            if similarities[best_idx] > threshold:
                return known[best_idx]
        
        return None


class QueryGenerator:
    """Generate Cypher queries from intent and entities"""
    
    def __init__(self):
        self.templates = {
            'find_actors': [
                "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {title: '{movie}'}) RETURN p.name as actor, r.role as role",
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title CONTAINS '{movie}' RETURN p.name as actor",
            ],
            'find_movies_by_actor': [
                "MATCH (p:Person {name: '{actor}'})-[:ACTED_IN]->(m:Movie) RETURN m.title as movie, m.year as year ORDER BY m.year",
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE p.name CONTAINS '{actor}' RETURN m.title as movie, m.year as year",
            ],
            'find_director': [
                "MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: '{movie}'}) RETURN p.name as director",
                "MATCH (p:Person)-[:DIRECTED]->(m:Movie) WHERE m.title CONTAINS '{movie}' RETURN p.name as director",
            ],
            'find_movies_by_director': [
                "MATCH (p:Person {name: '{director}'})-[:DIRECTED]->(m:Movie) RETURN m.title as movie, m.year as year ORDER BY m.year",
                "MATCH (p:Person)-[:DIRECTED]->(m:Movie) WHERE p.name CONTAINS '{director}' RETURN m.title as movie, m.year as year",
            ],
            'find_movies_by_year': [
                "MATCH (m:Movie) WHERE m.year = {year} RETURN m.title as movie, m.year as year ORDER BY m.title",
                "MATCH (m:Movie) WHERE m.year >= {year} RETURN m.title as movie, m.year as year ORDER BY m.year",
            ],
            'find_movies_by_genre': [
                "MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: '{genre}'}) RETURN m.title as movie, m.year as year ORDER BY m.year DESC",
                "MATCH (m:Movie) WHERE toLower(m.genre) CONTAINS toLower('{genre}') RETURN m.title as movie, m.year as year ORDER BY m.year DESC",
                "MATCH (m:Movie) RETURN m.title as movie, m.genre as genre, m.year as year ORDER BY m.year DESC LIMIT 20",
            ],
            'find_release_year': [
                "MATCH (m:Movie {title: '{movie}'}) RETURN m.year as year",
                "MATCH (m:Movie) WHERE m.title CONTAINS '{movie}' RETURN m.title as movie, m.year as year",
            ],
            'count_movies': [
                "MATCH (p:Person {name: '{actor}'})-[:ACTED_IN]->(m:Movie) RETURN count(m) as count",
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE p.name CONTAINS '{actor}' RETURN count(m) as count",
                "MATCH (m:Movie) RETURN count(m) as count",  # Fallback: count all movies
            ],
            # Legacy intent names for backwards compatibility
            'FIND_ACTORS_IN_MOVIE': [
                "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {title: '{movie}'}) RETURN p.name as actor, r.role as role",
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE m.title CONTAINS '{movie}' RETURN p.name as actor",
            ],
            'FIND_MOVIES_BY_ACTOR': [
                "MATCH (p:Person {name: '{actor}'})-[:ACTED_IN]->(m:Movie) RETURN m.title as movie, m.year as year ORDER BY m.year",
                "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) WHERE p.name CONTAINS '{actor}' RETURN m.title as movie, m.year as year",
            ],
            'FIND_DIRECTOR': [
                "MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: '{movie}'}) RETURN p.name as director",
                "MATCH (p:Person)-[:DIRECTED]->(m:Movie) WHERE m.title CONTAINS '{movie}' RETURN p.name as director",
            ],
            'FIND_MOVIES_BY_DIRECTOR': [
                "MATCH (p:Person {name: '{director}'})-[:DIRECTED]->(m:Movie) RETURN m.title as movie, m.year as year ORDER BY m.year",
            ],
            'FIND_PATH': [
                "MATCH path = shortestPath((p1:Person {name: '{person1}'})-[*]-(p2:Person {name: '{person2}'})) RETURN path",
            ],
            'COUNT_MOVIES': [
                "MATCH (p:Person {name: '{actor}'})-[:ACTED_IN]->(m:Movie) RETURN count(m) as movie_count",
            ],
            'FIND_COACTORS': [
                "MATCH (p1:Person {name: '{actor}'})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) RETURN DISTINCT p2.name as coactor, m.title as movie",
            ],
            'FIND_MOVIES_BY_YEAR': [
                "MATCH (m:Movie) WHERE m.year = {year} RETURN m.title as movie, m.year as year",
                "MATCH (m:Movie) WHERE m.year >= {year} RETURN m.title as movie, m.year as year ORDER BY m.year",
            ],
        }
    
    def generate(self, intent: str, entities: Dict) -> Optional[str]:
        """
        Generate Cypher query from intent and entities
        
        Args:
            intent: Classified intent
            entities: Extracted entities dict
            
        Returns:
            Cypher query string or None
        """
        templates = self.templates.get(intent, [])
        if not templates:
            return None
        
        # Select best template based on available entities
        template = self._select_template(templates, entities)
        if not template:
            return None
        
        # Fill template with entities
        query = self._fill_template(template, entities)
        return query
    
    def _select_template(self, templates: List[str], entities: Dict) -> Optional[str]:
        """Select best template based on available entities"""
        for template in templates:
            # Check if all required placeholders can be filled
            required_fields = re.findall(r'\{(\w+)\}', template)
            
            can_fill = True
            for field in required_fields:
                if field == 'movie' and not entities.get('movies'):
                    can_fill = False
                elif field == 'actor' and not entities.get('actors'):
                    can_fill = False
                elif field == 'director' and not entities.get('actors'):
                    can_fill = False
                elif field == 'genre' and not entities.get('genres'):
                    can_fill = False
                elif field == 'year' and not entities.get('years'):
                    can_fill = False
                elif field.startswith('person') and not entities.get('actors'):
                    can_fill = False
            
            if can_fill:
                return template
        
        # Return last template as fallback (usually a generic query)
        return templates[-1] if templates else None
    
    def _fill_template(self, template: str, entities: Dict) -> str:
        """Fill template with entity values"""
        query = template
        
        # Fill movie
        if '{movie}' in query and entities.get('movies'):
            query = query.replace('{movie}', entities['movies'][0])
        
        # Fill actor
        if '{actor}' in query and entities.get('actors'):
            query = query.replace('{actor}', entities['actors'][0])
        
        # Fill director (same as actor)
        if '{director}' in query and entities.get('actors'):
            query = query.replace('{director}', entities['actors'][0])
        
        # Fill genre
        if '{genre}' in query and entities.get('genres'):
            query = query.replace('{genre}', entities['genres'][0])
        
        # Fill year
        if '{year}' in query:
            if entities.get('years'):
                query = query.replace('{year}', str(entities['years'][0]))
            else:
                # Try to extract year from template itself
                year_match = re.search(r'\b(19|20)\d{2}\b', template)
                if year_match:
                    query = query.replace('{year}', year_match.group())
        
        # Fill person1 and person2 for path queries
        if '{person1}' in query and entities.get('actors'):
            if len(entities['actors']) >= 2:
                query = query.replace('{person1}', entities['actors'][0])
                query = query.replace('{person2}', entities['actors'][1])
            elif len(entities['actors']) == 1:
                query = query.replace('{person1}', entities['actors'][0])
        
        return query


class QueryValidator:
    """Validate generated Cypher queries"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def validate(self, query: str) -> Tuple[Optional[str], str]:
        """
        Validate a Cypher query
        
        Args:
            query: Cypher query string
            
        Returns:
            Tuple of (valid_query or None, message)
        """
        if not query:
            return None, "No query generated"
        
        # Basic syntax checks
        if not query.strip().upper().startswith('MATCH'):
            return None, "Invalid query: must start with MATCH"
        
        if 'RETURN' not in query.upper():
            return None, "Invalid query: must contain RETURN"
        
        # Try to explain the query (doesn't execute, just validates)
        try:
            with self.driver.session() as session:
                session.run(f"EXPLAIN {query}")
            return query, "Valid"
        except Exception as e:
            return None, f"Query validation error: {str(e)}"


class MLNLPToCypher:
    """Complete ML-based NLP to Cypher pipeline"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        print("\n" + "="*60)
        print("Initializing ML NLP to Cypher Converter")
        print("="*60)
        
        # Connect to Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("✓ Connected to Neo4j")
        
        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor(self.driver, use_spacy=True)
        self.query_generator = QueryGenerator()
        self.validator = QueryValidator(self.driver)
        
        self.feedback_data = []
    
    def train(self, training_data: List[Dict]):
        """
        Train the model on training data
        
        Args:
            training_data: List of dicts with 'question' and 'intent' keys
        """
        questions = [item['question'] for item in training_data]
        intents = [item['intent'] for item in training_data]
        
        self.intent_classifier.train(questions, intents)
    
    def convert(self, question: str, confidence_threshold: float = 0.6) -> Dict:
        """
        Convert natural language question to Cypher query
        
        Args:
            question: Natural language question
            confidence_threshold: Minimum confidence to accept prediction
            
        Returns:
            Dict with query, intent, confidence, entities, and message
        """
        result = {
            'question': question,
            'query': None,
            'intent': None,
            'confidence': 0.0,
            'entities': {},
            'message': '',
            'success': False
        }
        
        try:
            # Step 1: Classify intent
            intent, confidence = self.intent_classifier.predict(question)
            result['intent'] = intent
            result['confidence'] = confidence
            
            if confidence < confidence_threshold:
                result['message'] = f"Low confidence ({confidence:.2f}). Please rephrase your question."
                return result
            
            # Step 2: Extract entities
            entities = self.entity_extractor.extract(question)
            result['entities'] = entities
            
            if not entities['actors'] and not entities['movies']:
                result['message'] = "Could not identify any actors or movies in your question."
                return result
            
            # Step 3: Generate query
            query = self.query_generator.generate(intent, entities)
            
            if not query:
                result['message'] = f"Could not generate query for intent: {intent}"
                return result
            
            # Step 4: Validate query
            valid_query, message = self.validator.validate(query)
            
            if valid_query:
                result['query'] = valid_query
                result['message'] = "Query generated successfully"
                result['success'] = True
            else:
                result['message'] = message
            
        except Exception as e:
            result['message'] = f"Error: {str(e)}"
        
        return result
    
    def execute_query(self, query: str) -> List[Dict]:
        """
        Execute a Cypher query and return results
        
        Args:
            query: Cypher query string
            
        Returns:
            List of result records as dicts
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def ask(self, question: str) -> None:
        """
        Ask a question, generate query, execute it, and display results
        
        Args:
            question: Natural language question
        """
        print("\n" + "="*60)
        print(f"Question: {question}")
        print("="*60)
        
        # Convert to Cypher
        result = self.convert(question)
        
        print(f"\nIntent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Entities: {result['entities']}")
        print(f"Message: {result['message']}")
        
        if result['success'] and result['query']:
            print(f"\nGenerated Cypher:\n{result['query']}")
            
            # Execute query
            try:
                print("\nExecuting query...")
                records = self.execute_query(result['query'])
                
                print(f"\nResults ({len(records)} records):")
                print("-" * 60)
                for i, record in enumerate(records, 1):
                    print(f"{i}. {record}")
                
            except Exception as e:
                print(f"\nExecution error: {e}")
        else:
            print("\n❌ Could not generate valid query")
    
    def save_model(self, path: str = "intent_classifier.pkl"):
        """Save the trained model"""
        self.intent_classifier.save(path)
    
    def load_model(self, path: str = "intent_classifier.pkl"):
        """Load a trained model"""
        self.intent_classifier.load(path)
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
        print("\n✓ Connection closed")


# Training data
TRAINING_DATA = [
    # FIND_ACTORS_IN_MOVIE
    {"question": "Who acted in Forrest Gump?", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Who starred in Forrest Gump?", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Who was in Forrest Gump?", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Show me the cast of Forrest Gump", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Forrest Gump actors", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "List actors in Titanic", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Who played in Cast Away?", "intent": "FIND_ACTORS_IN_MOVIE"},
    {"question": "Cast of Titanic", "intent": "FIND_ACTORS_IN_MOVIE"},
    
    # FIND_MOVIES_BY_ACTOR
    {"question": "What movies did Tom Hanks act in?", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "Tom Hanks movies", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "Show me films with Tom Hanks", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "List Leonardo DiCaprio movies", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "Which movies has Kate Winslet been in?", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "Robin Wright filmography", "intent": "FIND_MOVIES_BY_ACTOR"},
    {"question": "What has Tom Hanks starred in?", "intent": "FIND_MOVIES_BY_ACTOR"},
    
    # FIND_DIRECTOR
    {"question": "Who directed Forrest Gump?", "intent": "FIND_DIRECTOR"},
    {"question": "Who is the director of Titanic?", "intent": "FIND_DIRECTOR"},
    {"question": "Forrest Gump director", "intent": "FIND_DIRECTOR"},
    {"question": "Director of Cast Away", "intent": "FIND_DIRECTOR"},
    {"question": "Who made Titanic?", "intent": "FIND_DIRECTOR"},
    
    # FIND_MOVIES_BY_DIRECTOR
    {"question": "What movies did Robert Zemeckis direct?", "intent": "FIND_MOVIES_BY_DIRECTOR"},
    {"question": "Robert Zemeckis films", "intent": "FIND_MOVIES_BY_DIRECTOR"},
    {"question": "Movies directed by Robert Zemeckis", "intent": "FIND_MOVIES_BY_DIRECTOR"},
    {"question": "Show me Robert Zemeckis movies", "intent": "FIND_MOVIES_BY_DIRECTOR"},
    
    # FIND_PATH
    {"question": "How are Tom Hanks and Robin Wright connected?", "intent": "FIND_PATH"},
    {"question": "Connection between Tom Hanks and Kate Winslet", "intent": "FIND_PATH"},
    {"question": "Path from Tom Hanks to Leonardo DiCaprio", "intent": "FIND_PATH"},
    {"question": "How is Tom Hanks related to Robin Wright?", "intent": "FIND_PATH"},
    
    # COUNT_MOVIES
    {"question": "How many movies did Tom Hanks make?", "intent": "COUNT_MOVIES"},
    {"question": "Count Tom Hanks movies", "intent": "COUNT_MOVIES"},
    {"question": "Number of Leonardo DiCaprio films", "intent": "COUNT_MOVIES"},
    {"question": "How many films has Kate Winslet acted in?", "intent": "COUNT_MOVIES"},
    
    # FIND_COACTORS
    {"question": "Who acted with Tom Hanks?", "intent": "FIND_COACTORS"},
    {"question": "Tom Hanks co-stars", "intent": "FIND_COACTORS"},
    {"question": "Who has worked with Tom Hanks?", "intent": "FIND_COACTORS"},
    {"question": "Leonardo DiCaprio co-actors", "intent": "FIND_COACTORS"},
]


def load_training_data(json_path: str) -> List[Dict]:
    """
    Load training data from JSON file and append to existing TRAINING_DATA
    
    Args:
        json_path: Path to the training data JSON file
        
    Returns:
        Combined list of training examples (TRAINING_DATA + loaded data)
    """
    print(f"\nLoading training data from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
                
        # Combine hardcoded data with loaded data
        combined_data = TRAINING_DATA + data
        
        print(f"✓ Combine hardcoded data with loaded data. Total training examples: {len(combined_data)}")
        
        # Display intent distribution for combined data
        intent_counts = {}
        for item in combined_data:
            intent = item.get('intent', 'UNKNOWN')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("\nCombined intent distribution:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  • {intent}: {count} examples")
        
        return combined_data
        
    except FileNotFoundError:
        print(f"✗ Error: Training data file not found: {json_path}")
        print(f"  Using only hardcoded training data ({len(TRAINING_DATA)} examples)")
        return TRAINING_DATA
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in training data file: {e}")
        print(f"  Using only hardcoded training data ({len(TRAINING_DATA)} examples)")
        return TRAINING_DATA
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        print(f"  Using only hardcoded training data ({len(TRAINING_DATA)} examples)")
        return TRAINING_DATA


def main():
    # Neo4j connection details
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password123"
    
    # Path to training data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(script_dir, '..', 'data', 'training_data_complete.json')
    
    try:
        # Initialize the converter
        converter = MLNLPToCypher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Load training data from JSON file
        training_data = load_training_data(training_data_path)
        
        # Train the model
        print("\n" + "="*60)
        print("Training Phase")
        print("="*60)
        converter.train(training_data)
        
        # Save the model
        model_dir = os.path.join(script_dir, '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'intent_classifier.pkl')
        converter.save_model(model_path)
        
        # Test questions
        print("\n" + "="*60)
        print("Testing Phase")
        print("="*60)
        
        test_questions = [
            "Who acted in Top Gun: Maverick?",
            "What movies did Tom Cruise act in?",
            "Who acted in The Matrix?",
            "Show me the cast of Barbie",
            "Tom Hanks movies",
            "How are Keanu Reeves and Margot Robbie connected?",
            "Who has worked with Tom Cruise?",
            "Count Tom Hanks movies",
        ]
        
        for question in test_questions:
            converter.ask(question)
            print()
        
        # Interactive mode
        print("\n" + "="*60)
        print("Interactive Mode (type 'quit' to exit)")
        print("="*60)
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if question:
                converter.ask(question)
        
        # Close connection
        converter.close()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
