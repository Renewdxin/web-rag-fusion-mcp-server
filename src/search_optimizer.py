"""
Enhanced Search Optimizer for RAG MCP Server.

This module provides comprehensive search optimization capabilities including:
- Query expansion with WordNet and custom synonyms
- Hybrid ranking combining BM25 and vector similarity
- Search result summarization and highlighting
- Search personalization and preference learning
- Spell correction with context awareness
- Semantic query understanding and intent recognition
- Analytics dashboard data generation
- A/B testing framework for search strategies

Architecture:
- QueryExpander: Handles query expansion and synonym detection
- HybridRanker: Combines multiple ranking algorithms
- ResultSummarizer: Generates summaries and highlights
- PersonalizationEngine: Learns user preferences
- SpellCorrector: Provides spelling correction
- SemanticAnalyzer: Understands query intent and entities
- SearchAnalytics: Tracks metrics and generates insights
- ABTestFramework: Manages A/B testing of search strategies
- SearchOptimizer: Main orchestrator class
"""

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import uuid

# NLP libraries with graceful degradation
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None
    wordnet = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Local imports
from .vector_store import Document, SearchResult


# Type definitions and enums
class QueryType(Enum):
    FACTUAL = "factual"
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    UNKNOWN = "unknown"


class RankingStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID_BALANCED = "hybrid_balanced"
    HYBRID_VECTOR_WEIGHTED = "hybrid_vector_weighted"
    HYBRID_BM25_WEIGHTED = "hybrid_bm25_weighted"
    PERSONALIZED = "personalized"


class SearchIntent(Enum):
    SEARCH = "search"
    NAVIGATION = "navigation"
    TRANSACTION = "transaction"
    INFORMATION = "information"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


@dataclass
class ExpandedQuery:
    """Expanded query with synonyms and related terms."""
    original_query: str
    expanded_terms: List[str]
    synonyms: Dict[str, List[str]]
    related_concepts: List[str]
    boost_terms: List[str]
    expansion_confidence: float
    processing_time: float


@dataclass
class RankingScore:
    """Combined ranking score with components."""
    final_score: float
    vector_score: float
    bm25_score: float
    personalization_score: float
    boost_score: float
    strategy_used: RankingStrategy
    explanation: str


@dataclass
class SearchSummary:
    """Search result summary with highlights."""
    summary_text: str
    key_sentences: List[str]
    highlights: List[Dict[str, Any]]
    relevance_score: float
    summary_length: int
    extraction_method: str


@dataclass
class UserPreference:
    """User preference data."""
    user_id: str
    preferred_content_types: List[str]
    frequent_queries: List[str]
    click_through_rates: Dict[str, float]
    dwell_times: Dict[str, float]
    feedback_scores: Dict[str, int]
    last_updated: datetime
    preference_strength: float


@dataclass
class SpellingSuggestion:
    """Spelling correction suggestion."""
    original_word: str
    corrected_word: str
    confidence: float
    edit_distance: int
    context_score: float


@dataclass
class QueryAnalysis:
    """Semantic query analysis results."""
    query_type: QueryType
    intent: SearchIntent
    entities: List[Dict[str, Any]]
    keywords: List[str]
    sentiment: str
    complexity_score: float
    confidence: float


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    query_count: int
    avg_response_time: float
    click_through_rate: float
    user_satisfaction: float
    result_relevance: float
    conversion_rate: float
    bounce_rate: float
    popular_queries: List[Tuple[str, int]]


@dataclass
class ABTestResult:
    """A/B test experiment result."""
    experiment_id: str
    variant_a: str
    variant_b: str
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    statistical_significance: float
    winner: Optional[str]
    confidence_interval: float
    sample_size: int


class QueryExpander:
    """Handles query expansion with synonyms and related terms."""
    
    def __init__(self, 
                 custom_synonyms: Optional[Dict[str, List[str]]] = None,
                 enable_wordnet: bool = True,
                 max_expansions: int = 5):
        self.custom_synonyms = custom_synonyms or {}
        self.enable_wordnet = enable_wordnet and NLTK_AVAILABLE
        self.max_expansions = max_expansions
        self.logger = logging.getLogger(f"{__name__}.QueryExpander")
        
        # Initialize NLTK components
        if self.enable_wordnet:
            try:
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
                self._download_nltk_data()
            except Exception as e:
                self.logger.warning(f"Failed to initialize WordNet: {e}")
                self.enable_wordnet = False
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {e}")
    
    async def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original search query
            
        Returns:
            ExpandedQuery with expansion results
        """
        start_time = time.time()
        
        try:
            # Tokenize and clean query
            tokens = self._tokenize_query(query)
            
            # Get synonyms from multiple sources
            synonyms = {}
            expanded_terms = []
            
            for token in tokens:
                if token.lower() not in self.stop_words:
                    # Custom synonyms
                    custom_syns = self.custom_synonyms.get(token.lower(), [])
                    
                    # WordNet synonyms
                    wordnet_syns = []
                    if self.enable_wordnet:
                        wordnet_syns = self._get_wordnet_synonyms(token)
                    
                    # Combine and deduplicate
                    all_synonyms = list(set(custom_syns + wordnet_syns))
                    if all_synonyms:
                        synonyms[token] = all_synonyms[:self.max_expansions]
                        expanded_terms.extend(all_synonyms[:self.max_expansions])
            
            # Get related concepts
            related_concepts = self._get_related_concepts(tokens)
            
            # Identify boost terms (important terms that should be weighted higher)
            boost_terms = self._identify_boost_terms(tokens)
            
            # Calculate expansion confidence
            expansion_confidence = self._calculate_expansion_confidence(
                len(synonyms), len(related_concepts), len(boost_terms)
            )
            
            processing_time = time.time() - start_time
            
            return ExpandedQuery(
                original_query=query,
                expanded_terms=expanded_terms,
                synonyms=synonyms,
                related_concepts=related_concepts,
                boost_terms=boost_terms,
                expansion_confidence=expansion_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Query expansion failed: {e}")
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                synonyms={},
                related_concepts=[],
                boost_terms=[],
                expansion_confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize and clean query."""
        if NLTK_AVAILABLE:
            tokens = word_tokenize(query.lower())
        else:
            tokens = re.findall(r'\b\w+\b', query.lower())
        
        return [token for token in tokens if token.isalpha() and len(token) > 1]
    
    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms from WordNet."""
        if not self.enable_wordnet:
            return []
        
        synonyms = set()
        
        try:
            # Get lemmatized form
            lemmatized = self.lemmatizer.lemmatize(word)
            
            # Get synsets for the word
            synsets = wordnet.synsets(lemmatized)
            
            for synset in synsets[:3]:  # Limit to top 3 synsets
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym != lemmatized:
                        synonyms.add(synonym)
            
            return list(synonyms)
            
        except Exception as e:
            self.logger.warning(f"WordNet lookup failed for '{word}': {e}")
            return []
    
    def _get_related_concepts(self, tokens: List[str]) -> List[str]:
        """Get conceptually related terms."""
        related = []
        
        # Domain-specific concept mappings
        concept_mappings = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks'],
            'machine': ['algorithm', 'automation', 'computer', 'technology'],
            'learning': ['training', 'education', 'knowledge', 'skill'],
            'data': ['information', 'dataset', 'statistics', 'analytics'],
            'python': ['programming', 'coding', 'development', 'script'],
            'web': ['internet', 'online', 'website', 'browser'],
            'server': ['hosting', 'backend', 'infrastructure', 'cloud'],
            'database': ['storage', 'sql', 'nosql', 'query'],
        }
        
        for token in tokens:
            if token in concept_mappings:
                related.extend(concept_mappings[token])
        
        return related[:self.max_expansions]
    
    def _identify_boost_terms(self, tokens: List[str]) -> List[str]:
        """Identify terms that should be boosted in ranking."""
        boost_terms = []
        
        # Technical terms that are usually important
        technical_terms = {
            'algorithm', 'implementation', 'optimization', 'performance',
            'architecture', 'design', 'pattern', 'framework', 'library',
            'api', 'interface', 'protocol', 'standard', 'specification'
        }
        
        # Action terms that indicate intent
        action_terms = {
            'how', 'what', 'why', 'when', 'where', 'implement', 'create',
            'build', 'develop', 'design', 'optimize', 'improve', 'fix'
        }
        
        for token in tokens:
            if token in technical_terms or token in action_terms:
                boost_terms.append(token)
        
        return boost_terms
    
    def _calculate_expansion_confidence(self, 
                                      synonym_count: int,
                                      concept_count: int,
                                      boost_count: int) -> float:
        """Calculate confidence in query expansion."""
        # Higher confidence with more expansions available
        base_confidence = min(0.8, (synonym_count + concept_count) * 0.1)
        
        # Boost confidence if important terms are identified
        boost_factor = min(0.2, boost_count * 0.05)
        
        return min(1.0, base_confidence + boost_factor)
    
    def add_custom_synonyms(self, synonyms: Dict[str, List[str]]) -> None:
        """Add custom synonym dictionary."""
        self.custom_synonyms.update(synonyms)
        self.logger.info(f"Added {len(synonyms)} custom synonym mappings")


class HybridRanker:
    """Combines BM25 and vector similarity for hybrid ranking."""
    
    def __init__(self,
                 vector_weight: float = 0.6,
                 bm25_weight: float = 0.4,
                 k1: float = 1.2,
                 b: float = 0.75):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k1 = k1  # BM25 parameter
        self.b = b    # BM25 parameter
        self.logger = logging.getLogger(f"{__name__}.HybridRanker")
        
        # Document statistics for BM25
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.doc_count = 0
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.document_frequencies = defaultdict(int)
        
    def update_document_stats(self, documents: List[Document]) -> None:
        """Update document statistics for BM25 calculation."""
        self.doc_count = len(documents)
        total_length = 0
        
        # Reset statistics
        self.doc_lengths = {}
        self.term_frequencies = defaultdict(lambda: defaultdict(int))
        self.document_frequencies = defaultdict(int)
        
        for doc in documents:
            doc_id = doc.id
            tokens = self._tokenize_document(doc.page_content)
            
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # Count term frequencies
            term_counts = Counter(tokens)
            unique_terms = set(tokens)
            
            for term, count in term_counts.items():
                self.term_frequencies[doc_id][term] = count
            
            # Count document frequencies
            for term in unique_terms:
                self.document_frequencies[term] += 1
        
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
        self.logger.info(f"Updated BM25 statistics for {self.doc_count} documents")
    
    def _tokenize_document(self, text: str) -> List[str]:
        """Tokenize document for BM25 calculation."""
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text.lower())
        else:
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        return [token for token in tokens if token.isalpha() and len(token) > 1]
    
    def calculate_bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document."""
        if doc_id not in self.doc_lengths:
            return 0.0
        
        doc_length = self.doc_lengths[doc_id]
        score = 0.0
        
        for term in query_terms:
            if term in self.term_frequencies[doc_id]:
                tf = self.term_frequencies[doc_id][term]
                df = self.document_frequencies[term]
                
                if df > 0:
                    idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
        
        return score
    
    async def rank_results(self,
                          query: str,
                          expanded_query: ExpandedQuery,
                          search_results: List[SearchResult],
                          strategy: RankingStrategy = RankingStrategy.HYBRID_BALANCED,
                          user_preferences: Optional[UserPreference] = None) -> List[Tuple[SearchResult, RankingScore]]:
        """
        Rank search results using hybrid approach.
        
        Args:
            query: Original query
            expanded_query: Expanded query with synonyms
            search_results: Initial search results
            strategy: Ranking strategy to use
            user_preferences: User preferences for personalization
            
        Returns:
            List of (result, score) tuples, sorted by score
        """
        if not search_results:
            return []
        
        try:
            # Prepare query terms
            query_terms = self._tokenize_document(query)
            expanded_terms = expanded_query.expanded_terms
            all_terms = list(set(query_terms + expanded_terms))
            
            ranked_results = []
            
            for result in search_results:
                # Calculate BM25 score
                bm25_score = self.calculate_bm25_score(all_terms, result.document.id)
                
                # Get vector similarity score (already in result)
                vector_score = result.score
                
                # Calculate personalization score
                personalization_score = 0.0
                if user_preferences:
                    personalization_score = self._calculate_personalization_score(
                        result, user_preferences
                    )
                
                # Calculate boost score from expanded query
                boost_score = self._calculate_boost_score(
                    result, expanded_query.boost_terms
                )
                
                # Combine scores based on strategy
                final_score, explanation = self._combine_scores(
                    vector_score, bm25_score, personalization_score, boost_score, strategy
                )
                
                ranking_score = RankingScore(
                    final_score=final_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    personalization_score=personalization_score,
                    boost_score=boost_score,
                    strategy_used=strategy,
                    explanation=explanation
                )
                
                ranked_results.append((result, ranking_score))
            
            # Sort by final score (descending)
            ranked_results.sort(key=lambda x: x[1].final_score, reverse=True)
            
            return ranked_results
            
        except Exception as e:
            self.logger.error(f"Ranking failed: {e}")
            # Return original results with basic scores
            return [(result, RankingScore(
                final_score=result.score,
                vector_score=result.score,
                bm25_score=0.0,
                personalization_score=0.0,
                boost_score=0.0,
                strategy_used=strategy,
                explanation="Fallback to vector score due to error"
            )) for result in search_results]
    
    def _calculate_personalization_score(self,
                                       result: SearchResult,
                                       preferences: UserPreference) -> float:
        """Calculate personalization score based on user preferences."""
        score = 0.0
        
        # Content type preference
        doc_type = result.document.metadata.get('file_type', 'unknown')
        if doc_type in preferences.preferred_content_types:
            score += 0.2
        
        # Check against frequent queries
        content_lower = result.document.page_content.lower()
        for freq_query in preferences.frequent_queries[:5]:  # Top 5 frequent queries
            if freq_query.lower() in content_lower:
                score += 0.1
        
        # Historical click-through rate
        doc_id = result.document.id
        if doc_id in preferences.click_through_rates:
            score += preferences.click_through_rates[doc_id] * 0.3
        
        # Dwell time (time spent on document)
        if doc_id in preferences.dwell_times:
            normalized_dwell = min(1.0, preferences.dwell_times[doc_id] / 300)  # 5 minutes max
            score += normalized_dwell * 0.2
        
        # Explicit feedback
        if doc_id in preferences.feedback_scores:
            # Normalize feedback score (-1 to 1) to (0 to 0.3)
            normalized_feedback = (preferences.feedback_scores[doc_id] + 1) / 2 * 0.3
            score += normalized_feedback
        
        return min(1.0, score)
    
    def _calculate_boost_score(self,
                             result: SearchResult,
                             boost_terms: List[str]) -> float:
        """Calculate boost score based on important terms."""
        if not boost_terms:
            return 0.0
        
        content_lower = result.document.page_content.lower()
        matches = 0
        
        for term in boost_terms:
            if term.lower() in content_lower:
                matches += 1
        
        return min(1.0, matches / len(boost_terms))
    
    def _combine_scores(self,
                       vector_score: float,
                       bm25_score: float,
                       personalization_score: float,
                       boost_score: float,
                       strategy: RankingStrategy) -> Tuple[float, str]:
        """Combine scores based on strategy."""
        
        if strategy == RankingStrategy.VECTOR_ONLY:
            final_score = vector_score + boost_score * 0.1
            explanation = "Vector similarity only with boost"
            
        elif strategy == RankingStrategy.BM25_ONLY:
            # Normalize BM25 score
            normalized_bm25 = min(1.0, bm25_score / 10.0)
            final_score = normalized_bm25 + boost_score * 0.1
            explanation = "BM25 only with boost"
            
        elif strategy == RankingStrategy.HYBRID_BALANCED:
            normalized_bm25 = min(1.0, bm25_score / 10.0)
            final_score = (
                self.vector_weight * vector_score +
                self.bm25_weight * normalized_bm25 +
                0.1 * personalization_score +
                0.1 * boost_score
            )
            explanation = f"Balanced hybrid: {self.vector_weight:.1f} vector + {self.bm25_weight:.1f} BM25"
            
        elif strategy == RankingStrategy.HYBRID_VECTOR_WEIGHTED:
            normalized_bm25 = min(1.0, bm25_score / 10.0)
            final_score = (
                0.7 * vector_score +
                0.2 * normalized_bm25 +
                0.05 * personalization_score +
                0.05 * boost_score
            )
            explanation = "Vector-weighted hybrid: 70% vector + 20% BM25"
            
        elif strategy == RankingStrategy.HYBRID_BM25_WEIGHTED:
            normalized_bm25 = min(1.0, bm25_score / 10.0)
            final_score = (
                0.3 * vector_score +
                0.6 * normalized_bm25 +
                0.05 * personalization_score +
                0.05 * boost_score
            )
            explanation = "BM25-weighted hybrid: 30% vector + 60% BM25"
            
        elif strategy == RankingStrategy.PERSONALIZED:
            normalized_bm25 = min(1.0, bm25_score / 10.0)
            final_score = (
                0.4 * vector_score +
                0.3 * normalized_bm25 +
                0.2 * personalization_score +
                0.1 * boost_score
            )
            explanation = "Personalized ranking: 40% vector + 30% BM25 + 20% personal"
            
        else:
            final_score = vector_score
            explanation = "Default vector similarity"
        
        return final_score, explanation
    
    def update_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """Update ranking weights."""
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.logger.info(f"Updated weights: vector={vector_weight:.2f}, BM25={bm25_weight:.2f}")


class ResultSummarizer:
    """Generates summaries and highlights for search results."""
    
    def __init__(self, max_summary_length: int = 300):
        self.max_summary_length = max_summary_length
        self.logger = logging.getLogger(f"{__name__}.ResultSummarizer")
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    async def generate_summary(self,
                             document: Document,
                             query: str,
                             expanded_query: ExpandedQuery) -> SearchSummary:
        """
        Generate extractive summary for a document.
        
        Args:
            document: Document to summarize
            query: Original query
            expanded_query: Expanded query with synonyms
            
        Returns:
            SearchSummary with summary and highlights
        """
        try:
            content = document.page_content
            
            # Split into sentences
            sentences = self._split_into_sentences(content)
            
            # Score sentences based on relevance
            sentence_scores = self._score_sentences(sentences, query, expanded_query)
            
            # Select top sentences for summary
            top_sentences = self._select_top_sentences(
                sentences, sentence_scores, self.max_summary_length
            )
            
            # Generate highlights
            highlights = self._generate_highlights(content, query, expanded_query)
            
            # Create summary text
            summary_text = ' '.join(top_sentences)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                summary_text, query, expanded_query
            )
            
            return SearchSummary(
                summary_text=summary_text,
                key_sentences=top_sentences,
                highlights=highlights,
                relevance_score=relevance_score,
                summary_length=len(summary_text),
                extraction_method="extractive_scoring"
            )
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            # Return basic summary
            preview = document.page_content[:self.max_summary_length]
            return SearchSummary(
                summary_text=preview,
                key_sentences=[preview],
                highlights=[],
                relevance_score=0.5,
                summary_length=len(preview),
                extraction_method="fallback_preview"
            )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentences(self,
                        sentences: List[str],
                        query: str,
                        expanded_query: ExpandedQuery) -> List[float]:
        """Score sentences based on relevance to query."""
        query_terms = set(self._tokenize_text(query.lower()))
        expanded_terms = set(term.lower() for term in expanded_query.expanded_terms)
        boost_terms = set(term.lower() for term in expanded_query.boost_terms)
        
        scores = []
        
        for sentence in sentences:
            sentence_terms = set(self._tokenize_text(sentence.lower()))
            
            # Base score from query term overlap
            query_overlap = len(query_terms.intersection(sentence_terms))
            base_score = query_overlap / len(query_terms) if query_terms else 0
            
            # Bonus for expanded terms
            expanded_overlap = len(expanded_terms.intersection(sentence_terms))
            expanded_score = expanded_overlap * 0.5 / len(expanded_terms) if expanded_terms else 0
            
            # Bonus for boost terms
            boost_overlap = len(boost_terms.intersection(sentence_terms))
            boost_score = boost_overlap * 0.3 / len(boost_terms) if boost_terms else 0
            
            # Position bonus (earlier sentences get slight bonus)
            position_bonus = 0.1 / (sentences.index(sentence) + 1)
            
            # Length penalty for very short or very long sentences
            length_penalty = 0
            if len(sentence) < 20:
                length_penalty = -0.2
            elif len(sentence) > 200:
                length_penalty = -0.1
            
            final_score = base_score + expanded_score + boost_score + position_bonus + length_penalty
            scores.append(max(0, final_score))
        
        return scores
    
    def _select_top_sentences(self,
                            sentences: List[str],
                            scores: List[float],
                            max_length: int) -> List[str]:
        """Select top sentences for summary within length limit."""
        # Sort by score (descending)
        scored_sentences = list(zip(sentences, scores))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        current_length = 0
        
        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= max_length:
                selected.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        # Ensure at least one sentence if possible
        if not selected and sentences:
            selected = [sentences[0][:max_length]]
        
        return selected
    
    def _generate_highlights(self,
                           content: str,
                           query: str,
                           expanded_query: ExpandedQuery) -> List[Dict[str, Any]]:
        """Generate highlights for query terms in content."""
        highlights = []
        
        # Combine all terms to highlight
        all_terms = set()
        all_terms.update(self._tokenize_text(query.lower()))
        all_terms.update(term.lower() for term in expanded_query.expanded_terms)
        all_terms.update(term.lower() for term in expanded_query.boost_terms)
        
        content_lower = content.lower()
        
        for term in all_terms:
            if term in content_lower:
                # Find all occurrences
                start = 0
                while True:
                    pos = content_lower.find(term, start)
                    if pos == -1:
                        break
                    
                    # Extract context around the term
                    context_start = max(0, pos - 50)
                    context_end = min(len(content), pos + len(term) + 50)
                    context = content[context_start:context_end]
                    
                    highlight = {
                        'term': term,
                        'position': pos,
                        'context': context,
                        'type': 'query_term' if term in query.lower() else 'expanded_term'
                    }
                    highlights.append(highlight)
                    
                    start = pos + 1
        
        return highlights[:10]  # Limit to top 10 highlights
    
    def _calculate_relevance_score(self,
                                 summary: str,
                                 query: str,
                                 expanded_query: ExpandedQuery) -> float:
        """Calculate relevance score for the summary."""
        query_terms = set(self._tokenize_text(query.lower()))
        summary_terms = set(self._tokenize_text(summary.lower()))
        
        if not query_terms:
            return 0.0
        
        # Base relevance from query term overlap
        overlap = len(query_terms.intersection(summary_terms))
        base_relevance = overlap / len(query_terms)
        
        # Bonus for expanded terms
        expanded_terms = set(term.lower() for term in expanded_query.expanded_terms)
        expanded_overlap = len(expanded_terms.intersection(summary_terms))
        expanded_bonus = expanded_overlap * 0.1 / len(expanded_terms) if expanded_terms else 0
        
        return min(1.0, base_relevance + expanded_bonus)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback tokenization
        return re.findall(r'\b\w+\b', text)


class PersonalizationEngine:
    """Handles search personalization and preference learning."""
    
    def __init__(self, preference_decay_days: int = 30):
        self.preference_decay_days = preference_decay_days
        self.user_preferences: Dict[str, UserPreference] = {}
        self.logger = logging.getLogger(f"{__name__}.PersonalizationEngine")
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """Get user preferences."""
        return self.user_preferences.get(user_id)
    
    def update_user_preferences(self,
                              user_id: str,
                              query: str,
                              clicked_results: List[str],
                              dwell_times: Dict[str, float],
                              feedback_scores: Dict[str, int]) -> None:
        """Update user preferences based on interaction."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreference(
                user_id=user_id,
                preferred_content_types=[],
                frequent_queries=[],
                click_through_rates={},
                dwell_times={},
                feedback_scores={},
                last_updated=datetime.now(),
                preference_strength=0.0
            )
        
        prefs = self.user_preferences[user_id]
        
        # Update frequent queries
        if query not in prefs.frequent_queries:
            prefs.frequent_queries.append(query)
        prefs.frequent_queries = prefs.frequent_queries[-20:]  # Keep last 20
        
        # Update click-through rates
        for doc_id in clicked_results:
            current_ctr = prefs.click_through_rates.get(doc_id, 0.0)
            prefs.click_through_rates[doc_id] = min(1.0, current_ctr + 0.1)
        
        # Update dwell times
        for doc_id, dwell_time in dwell_times.items():
            current_dwell = prefs.dwell_times.get(doc_id, 0.0)
            # Exponential moving average
            prefs.dwell_times[doc_id] = 0.7 * current_dwell + 0.3 * dwell_time
        
        # Update feedback scores
        prefs.feedback_scores.update(feedback_scores)
        
        # Update preference strength
        prefs.preference_strength = self._calculate_preference_strength(prefs)
        prefs.last_updated = datetime.now()
        
        self.logger.info(f"Updated preferences for user {user_id}")
    
    def _calculate_preference_strength(self, prefs: UserPreference) -> float:
        """Calculate overall preference strength."""
        factors = [
            min(1.0, len(prefs.frequent_queries) / 10),  # Query history
            min(1.0, len(prefs.click_through_rates) / 20),  # Click history
            min(1.0, len(prefs.dwell_times) / 15),  # Dwell time history
            min(1.0, len(prefs.feedback_scores) / 10),  # Feedback history
        ]
        
        return sum(factors) / len(factors)
    
    def apply_preference_decay(self) -> None:
        """Apply time-based decay to preferences."""
        cutoff_date = datetime.now() - timedelta(days=self.preference_decay_days)
        
        for user_id, prefs in self.user_preferences.items():
            if prefs.last_updated < cutoff_date:
                # Apply decay
                decay_factor = 0.8
                
                # Decay click-through rates
                for doc_id in prefs.click_through_rates:
                    prefs.click_through_rates[doc_id] *= decay_factor
                
                # Decay dwell times
                for doc_id in prefs.dwell_times:
                    prefs.dwell_times[doc_id] *= decay_factor
                
                # Remove very old feedback
                old_feedback = {
                    doc_id: score for doc_id, score in prefs.feedback_scores.items()
                    if score > -0.5  # Keep only positive-leaning feedback
                }
                prefs.feedback_scores = old_feedback
                
                prefs.preference_strength *= decay_factor
                
        self.logger.info("Applied preference decay")
    
    def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user's search behavior."""
        if user_id not in self.user_preferences:
            return {"error": "User not found"}
        
        prefs = self.user_preferences[user_id]
        
        # Most frequent queries
        query_counter = Counter(prefs.frequent_queries)
        top_queries = query_counter.most_common(5)
        
        # Most clicked documents
        top_documents = sorted(
            prefs.click_through_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Average dwell time
        avg_dwell = (
            sum(prefs.dwell_times.values()) / len(prefs.dwell_times)
            if prefs.dwell_times else 0
        )
        
        # Feedback distribution
        feedback_dist = Counter(prefs.feedback_scores.values())
        
        return {
            "user_id": user_id,
            "preference_strength": prefs.preference_strength,
            "top_queries": top_queries,
            "top_documents": top_documents,
            "average_dwell_time": avg_dwell,
            "feedback_distribution": dict(feedback_dist),
            "last_updated": prefs.last_updated.isoformat(),
            "total_interactions": len(prefs.click_through_rates)
        }


class SpellCorrector:
    """Provides spell correction with context awareness."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SpellCorrector")
        
        # Common misspellings dictionary
        self.common_misspellings = {
            'algoritm': 'algorithm',
            'machien': 'machine',
            'learing': 'learning',
            'artifical': 'artificial',
            'intelligance': 'intelligence',
            'programing': 'programming',
            'databse': 'database',
            'sofware': 'software',
            'computor': 'computer',
            'langauge': 'language',
            'tehnology': 'technology',
            'implemantation': 'implementation',
            'performace': 'performance',
            'optimiztion': 'optimization',
            'architechture': 'architecture',
        }
        
        # Domain-specific vocabulary
        self.domain_vocabulary = {
            'ai', 'ml', 'nlp', 'api', 'sql', 'nosql', 'json', 'xml', 'html', 'css',
            'javascript', 'python', 'java', 'cpp', 'rust', 'go', 'scala', 'kotlin',
            'react', 'vue', 'angular', 'node', 'express', 'django', 'flask',
            'tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy', 'matplotlib',
            'algorithm', 'data', 'structure', 'function', 'variable', 'class',
            'object', 'method', 'property', 'parameter', 'return', 'boolean',
            'string', 'integer', 'float', 'array', 'list', 'dictionary', 'tuple',
            'database', 'table', 'query', 'index', 'primary', 'foreign', 'key',
            'server', 'client', 'request', 'response', 'http', 'https', 'tcp',
            'udp', 'ip', 'domain', 'url', 'uri', 'endpoint', 'authentication',
            'authorization', 'token', 'session', 'cookie', 'cache', 'memory',
            'storage', 'file', 'directory', 'path', 'configuration', 'deployment'
        }
    
    async def correct_spelling(self, query: str) -> List[SpellingSuggestion]:
        """
        Correct spelling in query.
        
        Args:
            query: Input query to correct
            
        Returns:
            List of spelling suggestions
        """
        suggestions = []
        words = query.lower().split()
        
        for word in words:
            if word in self.domain_vocabulary:
                continue  # Skip known domain terms
            
            # Check common misspellings
            if word in self.common_misspellings:
                suggestions.append(SpellingSuggestion(
                    original_word=word,
                    corrected_word=self.common_misspellings[word],
                    confidence=0.9,
                    edit_distance=self._calculate_edit_distance(
                        word, self.common_misspellings[word]
                    ),
                    context_score=1.0
                ))
                continue
            
            # Find closest matches in vocabulary
            best_matches = self._find_closest_matches(word, max_suggestions=3)
            
            for match, distance in best_matches:
                if distance <= 2:  # Only suggest if edit distance is reasonable
                    context_score = self._calculate_context_score(word, match, query)
                    confidence = max(0.1, 1.0 - (distance / len(word)))
                    
                    suggestions.append(SpellingSuggestion(
                        original_word=word,
                        corrected_word=match,
                        confidence=confidence,
                        edit_distance=distance,
                        context_score=context_score
                    ))
        
        # Sort by confidence and context score
        suggestions.sort(key=lambda x: (x.confidence + x.context_score) / 2, reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _calculate_edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein distance between two words."""
        if len(word1) < len(word2):
            return self._calculate_edit_distance(word2, word1)
        
        if len(word2) == 0:
            return len(word1)
        
        previous_row = list(range(len(word2) + 1))
        for i, c1 in enumerate(word1):
            current_row = [i + 1]
            for j, c2 in enumerate(word2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _find_closest_matches(self, word: str, max_suggestions: int = 3) -> List[Tuple[str, int]]:
        """Find closest matches in vocabulary."""
        matches = []
        
        for vocab_word in self.domain_vocabulary:
            if abs(len(word) - len(vocab_word)) <= 2:  # Length filter
                distance = self._calculate_edit_distance(word, vocab_word)
                if distance <= 2:
                    matches.append((vocab_word, distance))
        
        # Sort by edit distance
        matches.sort(key=lambda x: x[1])
        
        return matches[:max_suggestions]
    
    def _calculate_context_score(self, original: str, correction: str, query: str) -> float:
        """Calculate context-aware score for correction."""
        # Higher score if correction appears in common domain combinations
        common_pairs = {
            'machine': ['learning', 'intelligence', 'algorithm'],
            'data': ['science', 'structure', 'analysis', 'mining'],
            'artificial': ['intelligence', 'neural', 'network'],
            'deep': ['learning', 'neural', 'network'],
            'natural': ['language', 'processing'],
            'software': ['engineering', 'development', 'architecture'],
            'computer': ['science', 'vision', 'graphics'],
            'web': ['development', 'application', 'service'],
            'database': ['management', 'system', 'design'],
            'algorithm': ['design', 'analysis', 'optimization'],
        }
        
        query_words = query.lower().split()
        context_score = 0.5  # Base score
        
        if correction in common_pairs:
            for pair_word in common_pairs[correction]:
                if pair_word in query_words:
                    context_score += 0.3
                    break
        
        # Check if correction makes the query more coherent
        if correction in ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
            context_score += 0.2  # Boost for common words
        
        return min(1.0, context_score)
    
    def add_domain_vocabulary(self, words: List[str]) -> None:
        """Add words to domain vocabulary."""
        self.domain_vocabulary.update(word.lower() for word in words)
        self.logger.info(f"Added {len(words)} words to domain vocabulary")
    
    def add_common_misspellings(self, misspellings: Dict[str, str]) -> None:
        """Add common misspellings dictionary."""
        self.common_misspellings.update(misspellings)
        self.logger.info(f"Added {len(misspellings)} common misspellings")


class SemanticAnalyzer:
    """Analyzes queries for semantic understanding."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SemanticAnalyzer")
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Query type patterns
        self.query_patterns = {
            QueryType.DEFINITIONAL: [
                r'\bwhat\s+is\b', r'\bdefine\b', r'\bmeaning\s+of\b',
                r'\bexplain\b', r'\bdefinition\b'
            ],
            QueryType.PROCEDURAL: [
                r'\bhow\s+to\b', r'\bsteps\b', r'\bprocess\b',
                r'\bimplement\b', r'\bcreate\b', r'\bbuild\b'
            ],
            QueryType.COMPARATIVE: [
                r'\bcompare\b', r'\bdifference\b', r'\bvs\b', r'\bversus\b',
                r'\bbetter\b', r'\bworse\b', r'\balternative\b'
            ],
            QueryType.TEMPORAL: [
                r'\bwhen\b', r'\bhistory\b', r'\btimeline\b', r'\bchronology\b',
                r'\brecent\b', r'\blatest\b', r'\bnew\b'
            ],
            QueryType.CAUSAL: [
                r'\bwhy\b', r'\bcause\b', r'\breason\b', r'\bbecause\b',
                r'\bresult\b', r'\beffect\b', r'\bimpact\b'
            ],
            QueryType.FACTUAL: [
                r'\bwho\b', r'\bwhere\b', r'\bwhich\b', r'\blist\b',
                r'\bexample\b', r'\binstance\b'
            ]
        }
        
        # Intent patterns
        self.intent_patterns = {
            SearchIntent.INFORMATION: [
                r'\blearn\b', r'\bunderstand\b', r'\bknow\b', r'\binformation\b',
                r'\bdetails\b', r'\bexplain\b', r'\btell\s+me\b'
            ],
            SearchIntent.NAVIGATION: [
                r'\bfind\b', r'\blocate\b', r'\bwhere\s+is\b', r'\blink\b',
                r'\bpage\b', r'\bsection\b', r'\bgo\s+to\b'
            ],
            SearchIntent.COMPARISON: [
                r'\bcompare\b', r'\bversus\b', r'\bvs\b', r'\bbetter\b',
                r'\bchoose\b', r'\bselect\b', r'\boption\b'
            ],
            SearchIntent.TRANSACTION: [
                r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bdownload\b',
                r'\binstall\b', r'\bget\b', r'\bacquire\b'
            ]
        }
        
        # Technical entities
        self.technical_entities = {
            'TECHNOLOGY': [
                'python', 'java', 'javascript', 'react', 'angular', 'vue',
                'tensorflow', 'pytorch', 'docker', 'kubernetes', 'aws', 'gcp',
                'azure', 'mongodb', 'postgresql', 'mysql', 'redis', 'nginx'
            ],
            'CONCEPT': [
                'algorithm', 'data structure', 'machine learning', 'ai',
                'deep learning', 'neural network', 'api', 'microservice',
                'blockchain', 'cybersecurity', 'devops', 'agile', 'scrum'
            ],
            'METHODOLOGY': [
                'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'cicd',
                'tdd', 'bdd', 'mvc', 'mvp', 'solid', 'dry', 'kiss'
            ]
        }
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query for semantic understanding.
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryAnalysis with semantic information
        """
        try:
            # Detect query type
            query_type = self._detect_query_type(query)
            
            # Detect intent
            intent = self._detect_intent(query)
            
            # Extract entities
            entities = self._extract_entities(query)
            
            # Extract keywords
            keywords = self._extract_keywords(query)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(query)
            
            # Calculate complexity
            complexity_score = self._calculate_complexity(query)
            
            # Calculate overall confidence
            confidence = self._calculate_confidence(query_type, intent, entities)
            
            return QueryAnalysis(
                query_type=query_type,
                intent=intent,
                entities=entities,
                keywords=keywords,
                sentiment=sentiment,
                complexity_score=complexity_score,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return QueryAnalysis(
                query_type=QueryType.UNKNOWN,
                intent=SearchIntent.UNKNOWN,
                entities=[],
                keywords=[],
                sentiment="neutral",
                complexity_score=0.5,
                confidence=0.0
            )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query."""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.FACTUAL  # Default
    
    def _detect_intent(self, query: str) -> SearchIntent:
        """Detect search intent."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return SearchIntent.INFORMATION  # Default
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from query."""
        entities = []
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8
                })
        
        # Extract technical entities
        query_lower = query.lower()
        for entity_type, terms in self.technical_entities.items():
            for term in terms:
                if term in query_lower:
                    entities.append({
                        'text': term,
                        'label': entity_type,
                        'start': query_lower.find(term),
                        'end': query_lower.find(term) + len(term),
                        'confidence': 0.9
                    })
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords."""
        # Remove stop words and extract meaningful terms
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(query.lower())
                stop_words = set(stopwords.words('english'))
                keywords = [token for token in tokens 
                          if token.isalpha() and token not in stop_words and len(token) > 2]
                return keywords
            except:
                pass
        
        # Fallback keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an'}
        
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of query."""
        # Simple sentiment analysis based on keywords
        positive_words = {'good', 'great', 'excellent', 'amazing', 'awesome', 'perfect', 'love', 'like', 'best', 'better', 'superior', 'outstanding', 'fantastic', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst', 'worse', 'inferior', 'poor', 'disappointing', 'frustrating', 'annoying', 'useless'}
        
        query_lower = query.lower()
        
        positive_count = sum(1 for word in positive_words if word in query_lower)
        negative_count = sum(1 for word in negative_words if word in query_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score."""
        # Factors that increase complexity
        word_count = len(query.split())
        unique_words = len(set(query.lower().split()))
        avg_word_length = sum(len(word) for word in query.split()) / word_count if word_count > 0 else 0
        
        # Complexity indicators
        has_operators = any(op in query.lower() for op in ['and', 'or', 'not', 'but'])
        has_questions = any(q in query.lower() for q in ['what', 'how', 'why', 'when', 'where', 'which', 'who'])
        has_comparisons = any(comp in query.lower() for comp in ['better', 'worse', 'compare', 'versus', 'vs'])
        
        # Calculate complexity score
        complexity = 0.0
        
        # Word count factor
        complexity += min(0.3, word_count / 20)
        
        # Vocabulary diversity
        complexity += min(0.2, unique_words / word_count if word_count > 0 else 0)
        
        # Average word length
        complexity += min(0.2, avg_word_length / 10)
        
        # Linguistic complexity
        if has_operators:
            complexity += 0.1
        if has_questions:
            complexity += 0.1
        if has_comparisons:
            complexity += 0.1
        
        return min(1.0, complexity)
    
    def _calculate_confidence(self, 
                            query_type: QueryType,
                            intent: SearchIntent,
                            entities: List[Dict[str, Any]]) -> float:
        """Calculate confidence in analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we detected specific patterns
        if query_type != QueryType.UNKNOWN:
            confidence += 0.2
        
        if intent != SearchIntent.UNKNOWN:
            confidence += 0.2
        
        # Boost confidence based on entities found
        if entities:
            confidence += min(0.3, len(entities) * 0.1)
        
        return min(1.0, confidence)


class SearchAnalytics:
    """Generates analytics and insights for search performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SearchAnalytics")
        
        # Analytics storage
        self.query_history: List[Dict[str, Any]] = []
        self.click_data: List[Dict[str, Any]] = []
        self.user_sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_metrics: List[Dict[str, Any]] = []
        
        # Cache for computed metrics
        self._metrics_cache = {}
        self._cache_expiry = None
    
    def track_query(self,
                   query: str,
                   user_id: str,
                   results_count: int,
                   response_time: float,
                   query_analysis: QueryAnalysis) -> None:
        """Track a search query."""
        query_data = {
            'query': query,
            'user_id': user_id,
            'timestamp': datetime.now(),
            'results_count': results_count,
            'response_time': response_time,
            'query_type': query_analysis.query_type.value,
            'intent': query_analysis.intent.value,
            'complexity': query_analysis.complexity_score,
            'entities': query_analysis.entities,
            'keywords': query_analysis.keywords
        }
        
        self.query_history.append(query_data)
        self.user_sessions[user_id].append(query_data)
        
        # Keep only recent data (last 10000 queries)
        if len(self.query_history) > 10000:
            self.query_history = self.query_history[-10000:]
    
    def track_click(self,
                   query: str,
                   user_id: str,
                   document_id: str,
                   position: int,
                   dwell_time: float = 0.0) -> None:
        """Track a click on search results."""
        click_data = {
            'query': query,
            'user_id': user_id,
            'document_id': document_id,
            'position': position,
            'dwell_time': dwell_time,
            'timestamp': datetime.now()
        }
        
        self.click_data.append(click_data)
        
        # Keep only recent data
        if len(self.click_data) > 5000:
            self.click_data = self.click_data[-5000:]
    
    def track_performance(self,
                         operation: str,
                         duration: float,
                         success: bool,
                         metadata: Dict[str, Any] = None) -> None:
        """Track performance metrics."""
        perf_data = {
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        self.performance_metrics.append(perf_data)
        
        # Keep only recent data
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        # Check cache
        if (self._cache_expiry and 
            datetime.now() < self._cache_expiry and 
            self._metrics_cache):
            return self._metrics_cache
        
        try:
            dashboard_data = {
                'overview': self._generate_overview_metrics(),
                'query_analytics': self._generate_query_analytics(),
                'user_behavior': self._generate_user_behavior_analytics(),
                'performance': self._generate_performance_analytics(),
                'content_insights': self._generate_content_insights(),
                'trends': self._generate_trend_analytics(),
                'recommendations': self._generate_recommendations()
            }
            
            # Cache results for 5 minutes
            self._metrics_cache = dashboard_data
            self._cache_expiry = datetime.now() + timedelta(minutes=5)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            return {"error": "Failed to generate dashboard data"}
    
    def _generate_overview_metrics(self) -> Dict[str, Any]:
        """Generate overview metrics."""
        if not self.query_history:
            return {"no_data": True}
        
        # Time ranges
        now = datetime.now()
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(weeks=1)
        month_ago = now - timedelta(days=30)
        
        # Filter queries by time
        recent_queries = [q for q in self.query_history if q['timestamp'] >= day_ago]
        weekly_queries = [q for q in self.query_history if q['timestamp'] >= week_ago]
        
        # Basic metrics
        total_queries = len(self.query_history)
        daily_queries = len(recent_queries)
        unique_users = len(set(q['user_id'] for q in self.query_history))
        
        # Response time metrics
        response_times = [q['response_time'] for q in self.query_history]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Click-through rate
        total_clicks = len(self.click_data)
        ctr = total_clicks / total_queries if total_queries > 0 else 0
        
        # User satisfaction (based on dwell time)
        avg_dwell_time = (
            sum(c['dwell_time'] for c in self.click_data) / len(self.click_data)
            if self.click_data else 0
        )
        
        return {
            'total_queries': total_queries,
            'daily_queries': daily_queries,
            'weekly_queries': len(weekly_queries),
            'unique_users': unique_users,
            'avg_response_time': round(avg_response_time, 3),
            'click_through_rate': round(ctr, 3),
            'avg_dwell_time': round(avg_dwell_time, 2),
            'queries_per_user': round(total_queries / unique_users, 2) if unique_users > 0 else 0
        }
    
    def _generate_query_analytics(self) -> Dict[str, Any]:
        """Generate query analytics."""
        if not self.query_history:
            return {"no_data": True}
        
        # Query types distribution
        query_types = Counter(q['query_type'] for q in self.query_history)
        
        # Intent distribution
        intents = Counter(q['intent'] for q in self.query_history)
        
        # Complexity distribution
        complexities = [q['complexity'] for q in self.query_history]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        
        # Most popular queries
        query_counts = Counter(q['query'] for q in self.query_history)
        popular_queries = query_counts.most_common(10)
        
        # Failed queries (no results)
        failed_queries = [q for q in self.query_history if q['results_count'] == 0]
        failure_rate = len(failed_queries) / len(self.query_history) if self.query_history else 0
        
        # Most common keywords
        all_keywords = []
        for q in self.query_history:
            all_keywords.extend(q['keywords'])
        keyword_counts = Counter(all_keywords)
        
        return {
            'query_types': dict(query_types),
            'intents': dict(intents),
            'avg_complexity': round(avg_complexity, 3),
            'popular_queries': popular_queries,
            'failure_rate': round(failure_rate, 3),
            'failed_queries_count': len(failed_queries),
            'top_keywords': keyword_counts.most_common(15),
            'avg_query_length': round(
                sum(len(q['query'].split()) for q in self.query_history) / len(self.query_history),
                2
            ) if self.query_history else 0
        }
    
    def _generate_user_behavior_analytics(self) -> Dict[str, Any]:
        """Generate user behavior analytics."""
        if not self.query_history:
            return {"no_data": True}
        
        # Session analysis
        session_lengths = []
        session_durations = []
        
        for user_id, sessions in self.user_sessions.items():
            if len(sessions) > 1:
                session_lengths.append(len(sessions))
                duration = (sessions[-1]['timestamp'] - sessions[0]['timestamp']).total_seconds()
                session_durations.append(duration)
        
        # Click position analysis
        click_positions = [c['position'] for c in self.click_data]
        
        # User engagement metrics
        engaged_users = set()
        for user_id, sessions in self.user_sessions.items():
            if len(sessions) > 1:  # More than one query
                engaged_users.add(user_id)
        
        # Bounce rate (single query sessions)
        single_query_sessions = sum(1 for sessions in self.user_sessions.values() if len(sessions) == 1)
        bounce_rate = single_query_sessions / len(self.user_sessions) if self.user_sessions else 0
        
        return {
            'avg_session_length': round(
                sum(session_lengths) / len(session_lengths), 2
            ) if session_lengths else 0,
            'avg_session_duration': round(
                sum(session_durations) / len(session_durations), 2
            ) if session_durations else 0,
            'engagement_rate': round(
                len(engaged_users) / len(self.user_sessions), 3
            ) if self.user_sessions else 0,
            'bounce_rate': round(bounce_rate, 3),
            'avg_click_position': round(
                sum(click_positions) / len(click_positions), 2
            ) if click_positions else 0,
            'total_sessions': len(self.user_sessions),
            'returning_users': len(engaged_users)
        }
    
    def _generate_performance_analytics(self) -> Dict[str, Any]:
        """Generate performance analytics."""
        if not self.performance_metrics:
            return {"no_data": True}
        
        # Success rate
        successes = sum(1 for m in self.performance_metrics if m['success'])
        success_rate = successes / len(self.performance_metrics) if self.performance_metrics else 0
        
        # Performance by operation
        operation_stats = defaultdict(list)
        for metric in self.performance_metrics:
            operation_stats[metric['operation']].append(metric['duration'])
        
        operation_performance = {}
        for op, durations in operation_stats.items():
            operation_performance[op] = {
                'avg_duration': round(sum(durations) / len(durations), 3),
                'min_duration': round(min(durations), 3),
                'max_duration': round(max(durations), 3),
                'count': len(durations)
            }
        
        # Overall performance
        all_durations = [m['duration'] for m in self.performance_metrics]
        
        return {
            'success_rate': round(success_rate, 3),
            'avg_response_time': round(sum(all_durations) / len(all_durations), 3),
            'min_response_time': round(min(all_durations), 3),
            'max_response_time': round(max(all_durations), 3),
            'total_operations': len(self.performance_metrics),
            'operation_performance': operation_performance
        }
    
    def _generate_content_insights(self) -> Dict[str, Any]:
        """Generate content insights."""
        if not self.click_data:
            return {"no_data": True}
        
        # Most clicked documents
        doc_clicks = Counter(c['document_id'] for c in self.click_data)
        popular_docs = doc_clicks.most_common(10)
        
        # Click distribution by position
        position_clicks = Counter(c['position'] for c in self.click_data)
        
        # Content performance
        doc_performance = defaultdict(list)
        for click in self.click_data:
            doc_performance[click['document_id']].append(click['dwell_time'])
        
        top_performing_docs = []
        for doc_id, dwell_times in doc_performance.items():
            avg_dwell = sum(dwell_times) / len(dwell_times)
            top_performing_docs.append((doc_id, avg_dwell, len(dwell_times)))
        
        top_performing_docs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'popular_documents': popular_docs,
            'click_distribution': dict(position_clicks),
            'top_performing_docs': top_performing_docs[:10],
            'total_unique_docs_clicked': len(doc_clicks)
        }
    
    def _generate_trend_analytics(self) -> Dict[str, Any]:
        """Generate trend analytics."""
        if not self.query_history:
            return {"no_data": True}
        
        # Query volume trends (last 7 days)
        now = datetime.now()
        daily_trends = {}
        
        for i in range(7):
            day = now - timedelta(days=i)
            day_key = day.strftime('%Y-%m-%d')
            day_queries = [
                q for q in self.query_history 
                if q['timestamp'].date() == day.date()
            ]
            daily_trends[day_key] = len(day_queries)
        
        # Query type trends
        recent_queries = [
            q for q in self.query_history 
            if q['timestamp'] >= now - timedelta(days=7)
        ]
        
        type_trends = Counter(q['query_type'] for q in recent_queries)
        
        # Performance trends
        recent_performance = [
            m for m in self.performance_metrics 
            if m['timestamp'] >= now - timedelta(days=7)
        ]
        
        if recent_performance:
            avg_recent_perf = sum(m['duration'] for m in recent_performance) / len(recent_performance)
        else:
            avg_recent_perf = 0
        
        return {
            'daily_query_volume': daily_trends,
            'query_type_trends': dict(type_trends),
            'recent_avg_performance': round(avg_recent_perf, 3),
            'trend_period': '7 days'
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Analyze query analytics
        if self.query_history:
            # High failure rate
            failed_queries = [q for q in self.query_history if q['results_count'] == 0]
            failure_rate = len(failed_queries) / len(self.query_history)
            
            if failure_rate > 0.15:  # 15% failure rate
                recommendations.append({
                    'type': 'content_gap',
                    'priority': 'high',
                    'title': 'High Query Failure Rate',
                    'description': f'Failure rate is {failure_rate:.1%}. Consider adding more content or improving query expansion.',
                    'action': 'Add more diverse content to knowledge base'
                })
            
            # Slow response time
            response_times = [q['response_time'] for q in self.query_history]
            avg_response = sum(response_times) / len(response_times)
            
            if avg_response > 2.0:  # 2 seconds
                recommendations.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'title': 'Slow Response Time',
                    'description': f'Average response time is {avg_response:.2f}s. Consider optimizing search algorithms.',
                    'action': 'Optimize search indexing and caching'
                })
        
        # Analyze user behavior
        if self.user_sessions:
            single_query_sessions = sum(1 for sessions in self.user_sessions.values() if len(sessions) == 1)
            bounce_rate = single_query_sessions / len(self.user_sessions)
            
            if bounce_rate > 0.7:  # 70% bounce rate
                recommendations.append({
                    'type': 'user_experience',
                    'priority': 'medium',
                    'title': 'High Bounce Rate',
                    'description': f'Bounce rate is {bounce_rate:.1%}. Users are not engaging with results.',
                    'action': 'Improve result relevance and presentation'
                })
        
        # Analyze click patterns
        if self.click_data:
            click_positions = [c['position'] for c in self.click_data]
            avg_click_position = sum(click_positions) / len(click_positions)
            
            if avg_click_position > 3:  # Average click beyond position 3
                recommendations.append({
                    'type': 'ranking',
                    'priority': 'high',
                    'title': 'Poor Result Ranking',
                    'description': f'Average click position is {avg_click_position:.1f}. Top results may not be relevant.',
                    'action': 'Improve ranking algorithm and result ordering'
                })
        
        return recommendations


class ABTestFramework:
    """Framework for A/B testing different search strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ABTestFramework")
        
        # Active experiments
        self.experiments: Dict[str, Dict[str, Any]] = {}
        
        # Results storage
        self.experiment_results: Dict[str, ABTestResult] = {}
        
        # User assignments
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        # Metrics collection
        self.experiment_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def create_experiment(self,
                         experiment_id: str,
                         name: str,
                         description: str,
                         variant_a: Dict[str, Any],
                         variant_b: Dict[str, Any],
                         traffic_split: float = 0.5,
                         duration_days: int = 7,
                         success_metric: str = 'click_through_rate') -> None:
        """Create a new A/B test experiment."""
        self.experiments[experiment_id] = {
            'name': name,
            'description': description,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'traffic_split': traffic_split,
            'duration_days': duration_days,
            'success_metric': success_metric,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=duration_days),
            'status': 'active'
        }
        
        self.logger.info(f"Created experiment {experiment_id}: {name}")
    
    def assign_user_to_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to experiment variant."""
        if experiment_id not in self.experiments:
            return 'control'
        
        if user_id in self.user_assignments[experiment_id]:
            return self.user_assignments[experiment_id][user_id]
        
        # Consistent assignment based on user ID hash
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        hash_int = int(user_hash, 16)
        
        experiment = self.experiments[experiment_id]
        if (hash_int % 100) < (experiment['traffic_split'] * 100):
            variant = 'variant_a'
        else:
            variant = 'variant_b'
        
        self.user_assignments[experiment_id][user_id] = variant
        return variant
    
    def track_experiment_metric(self,
                              experiment_id: str,
                              user_id: str,
                              metric_name: str,
                              value: float) -> None:
        """Track metric for experiment."""
        if experiment_id not in self.experiments:
            return
        
        variant = self.assign_user_to_variant(experiment_id, user_id)
        metric_key = f"{variant}_{metric_name}"
        
        self.experiment_metrics[experiment_id][metric_key].append(value)
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ABTestResult]:
        """Get results for an experiment."""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        metrics = self.experiment_metrics[experiment_id]
        
        # Calculate metrics for both variants
        success_metric = experiment['success_metric']
        
        variant_a_key = f"variant_a_{success_metric}"
        variant_b_key = f"variant_b_{success_metric}"
        
        metrics_a = metrics.get(variant_a_key, [])
        metrics_b = metrics.get(variant_b_key, [])
        
        if not metrics_a or not metrics_b:
            return None
        
        # Calculate summary statistics
        avg_a = sum(metrics_a) / len(metrics_a)
        avg_b = sum(metrics_b) / len(metrics_b)
        
        # Simple statistical significance test (simplified)
        sample_size = min(len(metrics_a), len(metrics_b))
        
        if sample_size < 30:
            statistical_significance = 0.0
        else:
            # Simplified significance calculation
            diff = abs(avg_a - avg_b)
            pooled_variance = (
                sum((x - avg_a) ** 2 for x in metrics_a) +
                sum((x - avg_b) ** 2 for x in metrics_b)
            ) / (len(metrics_a) + len(metrics_b) - 2)
            
            standard_error = (pooled_variance * (1/len(metrics_a) + 1/len(metrics_b))) ** 0.5
            
            if standard_error > 0:
                t_statistic = diff / standard_error
                # Rough significance approximation
                statistical_significance = min(0.95, t_statistic / 10)
            else:
                statistical_significance = 0.0
        
        # Determine winner
        winner = None
        if statistical_significance > 0.8:
            winner = 'variant_a' if avg_a > avg_b else 'variant_b'
        
        # Confidence interval (simplified)
        confidence_interval = 0.95 if statistical_significance > 0.8 else 0.5
        
        result = ABTestResult(
            experiment_id=experiment_id,
            variant_a='variant_a',
            variant_b='variant_b',
            metrics_a={success_metric: avg_a},
            metrics_b={success_metric: avg_b},
            statistical_significance=statistical_significance,
            winner=winner,
            confidence_interval=confidence_interval,
            sample_size=sample_size
        )
        
        self.experiment_results[experiment_id] = result
        return result
    
    def get_active_experiments(self) -> List[str]:
        """Get list of active experiments."""
        now = datetime.now()
        active = []
        
        for exp_id, exp_data in self.experiments.items():
            if exp_data['status'] == 'active' and now <= exp_data['end_date']:
                active.append(exp_id)
        
        return active
    
    def stop_experiment(self, experiment_id: str) -> None:
        """Stop an experiment."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['status'] = 'stopped'
            self.logger.info(f"Stopped experiment {experiment_id}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        summary = {
            'total_experiments': len(self.experiments),
            'active_experiments': len(self.get_active_experiments()),
            'completed_experiments': len(self.experiment_results),
            'experiments': {}
        }
        
        for exp_id, exp_data in self.experiments.items():
            result = self.experiment_results.get(exp_id)
            
            summary['experiments'][exp_id] = {
                'name': exp_data['name'],
                'status': exp_data['status'],
                'start_date': exp_data['start_date'].isoformat(),
                'end_date': exp_data['end_date'].isoformat(),
                'has_results': result is not None,
                'winner': result.winner if result else None
            }
        
        return summary


class SearchOptimizer:
    """
    Main SearchOptimizer class that orchestrates all search enhancement features.
    """
    
    def __init__(self,
                 enable_query_expansion: bool = True,
                 enable_hybrid_ranking: bool = True,
                 enable_summarization: bool = True,
                 enable_personalization: bool = True,
                 enable_spell_correction: bool = True,
                 enable_semantic_analysis: bool = True,
                 enable_analytics: bool = True,
                 enable_ab_testing: bool = True):
        
        self.logger = logging.getLogger(f"{__name__}.SearchOptimizer")
        
        # Initialize components
        self.query_expander = QueryExpander() if enable_query_expansion else None
        self.hybrid_ranker = HybridRanker() if enable_hybrid_ranking else None
        self.result_summarizer = ResultSummarizer() if enable_summarization else None
        self.personalization_engine = PersonalizationEngine() if enable_personalization else None
        self.spell_corrector = SpellCorrector() if enable_spell_correction else None
        self.semantic_analyzer = SemanticAnalyzer() if enable_semantic_analysis else None
        self.search_analytics = SearchAnalytics() if enable_analytics else None
        self.ab_framework = ABTestFramework() if enable_ab_testing else None
        
        # Configuration
        self.config = {
            'default_ranking_strategy': RankingStrategy.HYBRID_BALANCED,
            'max_results': 20,
            'enable_auto_spell_correction': True,
            'spell_correction_threshold': 0.7,
            'personalization_weight': 0.2,
            'cache_results': True,
            'cache_ttl': 300  # 5 minutes
        }
        
        # Results cache
        self.results_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        self.logger.info("SearchOptimizer initialized with all components")
    
    async def optimize_search(self,
                            query: str,
                            user_id: str,
                            search_results: List[SearchResult],
                            ranking_strategy: Optional[RankingStrategy] = None,
                            enable_personalization: bool = True,
                            enable_summarization: bool = True,
                            max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Main method to optimize search results.
        
        Args:
            query: Original search query
            user_id: User identifier
            search_results: Initial search results
            ranking_strategy: Ranking strategy to use
            enable_personalization: Whether to apply personalization
            enable_summarization: Whether to generate summaries
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with optimized results and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, user_id, ranking_strategy)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.debug(f"Cache hit for query: {query}")
                return cached_result
            
            # Initialize result structure
            optimization_result = {
                'query': query,
                'user_id': user_id,
                'optimized_results': [],
                'metadata': {
                    'original_results_count': len(search_results),
                    'processing_time': 0.0,
                    'optimizations_applied': []
                }
            }
            
            # Step 1: Spell correction
            spelling_suggestions = []
            corrected_query = query
            
            if self.spell_corrector:
                spelling_suggestions = await self.spell_corrector.correct_spelling(query)
                if (spelling_suggestions and 
                    self.config['enable_auto_spell_correction'] and
                    spelling_suggestions[0].confidence > self.config['spell_correction_threshold']):
                    corrected_query = spelling_suggestions[0].corrected_word
                    optimization_result['metadata']['optimizations_applied'].append('spell_correction')
            
            # Step 2: Semantic analysis
            query_analysis = None
            if self.semantic_analyzer:
                query_analysis = await self.semantic_analyzer.analyze_query(corrected_query)
                optimization_result['metadata']['query_analysis'] = {
                    'type': query_analysis.query_type.value,
                    'intent': query_analysis.intent.value,
                    'complexity': query_analysis.complexity_score,
                    'confidence': query_analysis.confidence
                }
                optimization_result['metadata']['optimizations_applied'].append('semantic_analysis')
            
            # Step 3: Query expansion
            expanded_query = None
            if self.query_expander:
                expanded_query = await self.query_expander.expand_query(corrected_query)
                optimization_result['metadata']['expanded_query'] = {
                    'synonyms': expanded_query.synonyms,
                    'related_concepts': expanded_query.related_concepts,
                    'boost_terms': expanded_query.boost_terms,
                    'confidence': expanded_query.expansion_confidence
                }
                optimization_result['metadata']['optimizations_applied'].append('query_expansion')
            
            # Step 4: Get user preferences
            user_preferences = None
            if self.personalization_engine and enable_personalization:
                user_preferences = self.personalization_engine.get_user_preferences(user_id)
                if user_preferences:
                    optimization_result['metadata']['optimizations_applied'].append('personalization')
            
            # Step 5: Hybrid ranking
            ranked_results = []
            if self.hybrid_ranker and expanded_query:
                strategy = ranking_strategy or self.config['default_ranking_strategy']
                ranked_results = await self.hybrid_ranker.rank_results(
                    corrected_query, expanded_query, search_results, strategy, user_preferences
                )
                optimization_result['metadata']['ranking_strategy'] = strategy.value
                optimization_result['metadata']['optimizations_applied'].append('hybrid_ranking')
            else:
                # Fallback to original results
                ranked_results = [(result, RankingScore(
                    final_score=result.score,
                    vector_score=result.score,
                    bm25_score=0.0,
                    personalization_score=0.0,
                    boost_score=0.0,
                    strategy_used=RankingStrategy.VECTOR_ONLY,
                    explanation="No hybrid ranking available"
                )) for result in search_results]
            
            # Step 6: Generate summaries
            summarized_results = []
            max_results = max_results or self.config['max_results']
            
            for result, ranking_score in ranked_results[:max_results]:
                result_data = {
                    'document': result.document,
                    'original_score': result.score,
                    'ranking_score': ranking_score,
                    'summary': None
                }
                
                # Generate summary if enabled
                if self.result_summarizer and enable_summarization and expanded_query:
                    summary = await self.result_summarizer.generate_summary(
                        result.document, corrected_query, expanded_query
                    )
                    result_data['summary'] = summary
                
                summarized_results.append(result_data)
            
            if enable_summarization:
                optimization_result['metadata']['optimizations_applied'].append('summarization')
            
            optimization_result['optimized_results'] = summarized_results
            
            # Step 7: Track analytics
            if self.search_analytics:
                self.search_analytics.track_query(
                    corrected_query, user_id, len(summarized_results),
                    time.time() - start_time, query_analysis or QueryAnalysis(
                        query_type=QueryType.UNKNOWN,
                        intent=SearchIntent.UNKNOWN,
                        entities=[], keywords=[], sentiment="neutral",
                        complexity_score=0.5, confidence=0.0
                    )
                )
            
            # Step 8: A/B testing
            if self.ab_framework:
                active_experiments = self.ab_framework.get_active_experiments()
                for exp_id in active_experiments:
                    variant = self.ab_framework.assign_user_to_variant(exp_id, user_id)
                    # Track metrics based on variant
                    self.ab_framework.track_experiment_metric(
                        exp_id, user_id, 'search_quality', 
                        sum(r['ranking_score'].final_score for r in summarized_results) / len(summarized_results) if summarized_results else 0
                    )
            
            # Finalize metadata
            optimization_result['metadata']['processing_time'] = time.time() - start_time
            optimization_result['metadata']['final_results_count'] = len(summarized_results)
            optimization_result['metadata']['spelling_suggestions'] = spelling_suggestions
            
            # Cache result
            self._cache_result(cache_key, optimization_result)
            
            self.logger.info(f"Search optimization completed for query '{query}' in {optimization_result['metadata']['processing_time']:.3f}s")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Search optimization failed: {e}")
            # Return basic results on error
            return {
                'query': query,
                'user_id': user_id,
                'optimized_results': [
                    {
                        'document': result.document,
                        'original_score': result.score,
                        'ranking_score': RankingScore(
                            final_score=result.score,
                            vector_score=result.score,
                            bm25_score=0.0,
                            personalization_score=0.0,
                            boost_score=0.0,
                            strategy_used=RankingStrategy.VECTOR_ONLY,
                            explanation="Error fallback"
                        ),
                        'summary': None
                    }
                    for result in search_results
                ],
                'metadata': {
                    'error': str(e),
                    'processing_time': time.time() - start_time,
                    'optimizations_applied': []
                }
            }
    
    def _generate_cache_key(self, query: str, user_id: str, strategy: Optional[RankingStrategy]) -> str:
        """Generate cache key for results."""
        key_data = f"{query}_{user_id}_{strategy.value if strategy else 'default'}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid."""
        if not self.config['cache_results']:
            return None
        
        if cache_key in self.results_cache:
            result, timestamp = self.results_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.config['cache_ttl']):
                return result
            else:
                del self.results_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache search result."""
        if self.config['cache_results']:
            self.results_cache[cache_key] = (result, datetime.now())
            
            # Clean old cache entries
            if len(self.results_cache) > 1000:
                oldest_key = min(self.results_cache.keys(), 
                               key=lambda k: self.results_cache[k][1])
                del self.results_cache[oldest_key]
    
    def track_user_feedback(self,
                          user_id: str,
                          query: str,
                          clicked_results: List[str],
                          dwell_times: Dict[str, float],
                          feedback_scores: Dict[str, int]) -> None:
        """Track user feedback for personalization."""
        if self.personalization_engine:
            self.personalization_engine.update_user_preferences(
                user_id, query, clicked_results, dwell_times, feedback_scores
            )
        
        if self.search_analytics:
            for doc_id in clicked_results:
                position = clicked_results.index(doc_id) + 1
                dwell_time = dwell_times.get(doc_id, 0.0)
                self.search_analytics.track_click(query, user_id, doc_id, position, dwell_time)
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get analytics dashboard data."""
        if not self.search_analytics:
            return {"error": "Analytics not enabled"}
        
        return self.search_analytics.generate_dashboard_data()
    
    def get_ab_test_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        if not self.ab_framework:
            return {"error": "A/B testing not enabled"}
        
        return self.ab_framework.get_experiment_summary()
    
    def create_ab_test(self,
                      experiment_id: str,
                      name: str,
                      description: str,
                      variant_a: Dict[str, Any],
                      variant_b: Dict[str, Any],
                      **kwargs) -> None:
        """Create A/B test experiment."""
        if self.ab_framework:
            self.ab_framework.create_experiment(
                experiment_id, name, description, variant_a, variant_b, **kwargs
            )
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """Update optimizer configuration."""
        self.config.update(config_updates)
        self.logger.info(f"Configuration updated: {config_updates}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health."""
        return {
            'components': {
                'query_expander': self.query_expander is not None,
                'hybrid_ranker': self.hybrid_ranker is not None,
                'result_summarizer': self.result_summarizer is not None,
                'personalization_engine': self.personalization_engine is not None,
                'spell_corrector': self.spell_corrector is not None,
                'semantic_analyzer': self.semantic_analyzer is not None,
                'search_analytics': self.search_analytics is not None,
                'ab_framework': self.ab_framework is not None,
            },
            'dependencies': {
                'nltk': NLTK_AVAILABLE,
                'spacy': SPACY_AVAILABLE,
                'numpy': NUMPY_AVAILABLE,
            },
            'configuration': self.config,
            'cache_stats': {
                'cached_results': len(self.results_cache),
                'cache_enabled': self.config['cache_results']
            }
        }


# Export main classes
__all__ = [
    'SearchOptimizer',
    'QueryExpander',
    'HybridRanker',
    'ResultSummarizer',
    'PersonalizationEngine',
    'SpellCorrector',
    'SemanticAnalyzer',
    'SearchAnalytics',
    'ABTestFramework',
    'QueryType',
    'RankingStrategy',
    'SearchIntent',
    'ExpandedQuery',
    'RankingScore',
    'SearchSummary',
    'UserPreference',
    'SpellingSuggestion',
    'QueryAnalysis',
    'SearchMetrics',
    'ABTestResult',
]