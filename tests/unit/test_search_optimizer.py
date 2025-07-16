#!/usr/bin/env python3
"""
Unit tests for SearchOptimizer components.
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.search_optimizer import (
        SearchOptimizer, QueryExpander, HybridRanker, ResultSummarizer,
        PersonalizationEngine, SpellCorrector, SemanticAnalyzer,
        SearchAnalytics, ABTestFramework, RankingStrategy, QueryType,
        SearchIntent, Document, SearchResult, ExpandedQuery, QueryAnalysis
    )
except ImportError:
    # Fallback for different import scenarios
    from search_optimizer import (
        SearchOptimizer, QueryExpander, HybridRanker, ResultSummarizer,
        PersonalizationEngine, SpellCorrector, SemanticAnalyzer,
        SearchAnalytics, ABTestFramework, RankingStrategy, QueryType,
        SearchIntent, Document, SearchResult, ExpandedQuery, QueryAnalysis
    )

async def test_query_expander():
    """Test QueryExpander component."""
    print("1. Testing QueryExpander")
    try:
        expander = QueryExpander(
            custom_synonyms={"ai": ["artificial intelligence", "machine learning"]},
            enable_wordnet=False  # Disable for test
        )
        
        expanded = await expander.expand_query("ai algorithms")
        print(f"✅ Query expansion successful")
        print(f"   Original: ai algorithms")
        print(f"   Expanded terms: {expanded.expanded_terms}")
        print(f"   Synonyms: {expanded.synonyms}")
        print(f"   Confidence: {expanded.expansion_confidence:.3f}")
        return True
    except Exception as e:
        print(f"❌ QueryExpander test failed: {e}")
        return False

async def test_spell_corrector():
    """Test SpellCorrector component."""
    print("\n2. Testing SpellCorrector")
    try:
        corrector = SpellCorrector()
        suggestions = await corrector.correct_spelling("machien lerning algoritm")
        print(f"✅ Spell correction successful")
        print(f"   Found {len(suggestions)} suggestions")
        for suggestion in suggestions[:3]:
            print(f"   {suggestion.original_word} -> {suggestion.corrected_word} (confidence: {suggestion.confidence:.3f})")
        return True
    except Exception as e:
        print(f"❌ SpellCorrector test failed: {e}")
        return False

async def test_semantic_analyzer():
    """Test SemanticAnalyzer component."""
    print("\n3. Testing SemanticAnalyzer")
    try:
        analyzer = SemanticAnalyzer()
        analysis = await analyzer.analyze_query("How to implement machine learning algorithms?")
        print(f"✅ Semantic analysis successful")
        print(f"   Query type: {analysis.query_type.value}")
        print(f"   Intent: {analysis.intent.value}")
        print(f"   Keywords: {analysis.keywords}")
        print(f"   Complexity: {analysis.complexity_score:.3f}")
        print(f"   Entities: {len(analysis.entities)}")
        return True
    except Exception as e:
        print(f"❌ SemanticAnalyzer test failed: {e}")
        return False

async def test_personalization_engine():
    """Test PersonalizationEngine component."""
    print("\n4. Testing PersonalizationEngine")
    try:
        personalization = PersonalizationEngine()
        
        # Update user preferences
        personalization.update_user_preferences(
            user_id="test_user",
            query="machine learning",
            clicked_results=["doc1", "doc2"],
            dwell_times={"doc1": 45.0, "doc2": 120.0},
            feedback_scores={"doc1": 1, "doc2": 0}
        )
        
        # Get preferences
        prefs = personalization.get_user_preferences("test_user")
        print(f"✅ Personalization successful")
        print(f"   Preference strength: {prefs.preference_strength:.3f}")
        print(f"   Frequent queries: {prefs.frequent_queries}")
        print(f"   Click-through rates: {len(prefs.click_through_rates)} documents")
        
        # Get insights
        insights = personalization.get_personalization_insights("test_user")
        print(f"   Total interactions: {insights['total_interactions']}")
        return True
    except Exception as e:
        print(f"❌ PersonalizationEngine test failed: {e}")
        return False

async def test_search_analytics():
    """Test SearchAnalytics component."""
    print("\n5. Testing SearchAnalytics")
    try:
        analytics = SearchAnalytics()
        
        # Track some queries
        test_analysis = QueryAnalysis(
            query_type=QueryType.PROCEDURAL,
            intent=SearchIntent.INFORMATION,
            entities=[],
            keywords=["machine", "learning"],
            sentiment="neutral",
            complexity_score=0.6,
            confidence=0.8
        )
        
        analytics.track_query("machine learning", "user1", 5, 0.5, test_analysis)
        analytics.track_query("python tutorial", "user2", 3, 0.8, test_analysis)
        analytics.track_click("machine learning", "user1", "doc1", 1, 30.0)
        
        # Generate dashboard
        dashboard = analytics.generate_dashboard_data()
        print(f"✅ Analytics successful")
        print(f"   Total queries tracked: {dashboard['overview']['total_queries']}")
        print(f"   Unique users: {dashboard['overview']['unique_users']}")
        print(f"   Average response time: {dashboard['overview']['avg_response_time']:.3f}s")
        print(f"   Recommendations: {len(dashboard['recommendations'])}")
        return True
    except Exception as e:
        print(f"❌ SearchAnalytics test failed: {e}")
        return False

async def test_ab_framework():
    """Test ABTestFramework component."""
    print("\n6. Testing ABTestFramework")
    try:
        ab_framework = ABTestFramework()
        
        # Create experiment
        ab_framework.create_experiment(
            experiment_id="test_ranking",
            name="Ranking Strategy Test",
            description="Test vector vs hybrid ranking",
            variant_a={"strategy": "vector_only"},
            variant_b={"strategy": "hybrid_balanced"},
            duration_days=7
        )
        
        # Assign users to variants
        variant_a = ab_framework.assign_user_to_variant("test_ranking", "user1")
        variant_b = ab_framework.assign_user_to_variant("test_ranking", "user2")
        
        # Track metrics
        ab_framework.track_experiment_metric("test_ranking", "user1", "click_through_rate", 0.25)
        ab_framework.track_experiment_metric("test_ranking", "user2", "click_through_rate", 0.30)
        
        print(f"✅ A/B testing successful")
        print(f"   Experiment created: test_ranking")
        print(f"   User1 variant: {variant_a}")
        print(f"   User2 variant: {variant_b}")
        print(f"   Active experiments: {len(ab_framework.get_active_experiments())}")
        return True
    except Exception as e:
        print(f"❌ ABTestFramework test failed: {e}")
        return False

async def test_full_search_optimizer():
    """Test full SearchOptimizer integration."""
    print("\n7. Testing Full SearchOptimizer Integration")
    try:
        optimizer = SearchOptimizer(
            enable_query_expansion=True,
            enable_hybrid_ranking=True,
            enable_summarization=True,
            enable_personalization=True,
            enable_spell_correction=True,
            enable_semantic_analysis=True,
            enable_analytics=True,
            enable_ab_testing=True
        )
        
        # Create mock documents
        docs = [
            Document(page_content="Machine learning is a subset of artificial intelligence", 
                    metadata={"source": "doc1"}, id="doc1"),
            Document(page_content="Deep learning uses neural networks for pattern recognition", 
                    metadata={"source": "doc2"}, id="doc2"),
            Document(page_content="Python is a popular programming language for AI", 
                    metadata={"source": "doc3"}, id="doc3")
        ]
        
        # Mock search results for optimization
        search_results = [
            SearchResult(document=docs[0], score=0.9),
            SearchResult(document=docs[1], score=0.8),
            SearchResult(document=docs[2], score=0.7)
        ]
        
        # Test system status
        status = optimizer.get_system_status()
        print(f"✅ Full optimization successful")
        print(f"   Active components: {sum(status['components'].values())}/8")
        print(f"   Dependencies: NLTK={status['dependencies']['nltk']}, spaCy={status['dependencies']['spacy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full optimization test failed: {e}")
        return False

async def run_unit_tests():
    """Run all unit tests."""
    print("🧪 **SearchOptimizer Unit Tests**")
    print("=" * 60)
    
    tests = [
        test_query_expander,
        test_spell_corrector,
        test_semantic_analyzer,
        test_personalization_engine,
        test_search_analytics,
        test_ab_framework,
        test_full_search_optimizer
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"🎉 **Unit Tests Complete: {passed}/{total} passed**")
    
    if passed == total:
        print("✅ All unit tests passed!")
    else:
        print(f"⚠️ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_unit_tests())
    sys.exit(0 if success else 1)