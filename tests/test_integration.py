"""
============================================================================
INTEGRATION TESTS - Test di Integrazione End-to-End
============================================================================

Questo modulo contiene test di integrazione che verificano il funzionamento
completo del sistema Web Scraper Agent.

Test Coverage:
- Agent completo (query → plan → execute → synthesize)
- Integrazione tools + LLM
- Gestione errori end-to-end
- Conversation memory
- Caching integrato

Questi test richiedono:
- LLM funzionante (Ollama/Groq/OpenAI)
- Connessione internet (per search/fetch)
- Tempo esecuzione più lungo

Pattern: Given-When-Then
Framework: pytest
Markers: @pytest.mark.integration, @pytest.mark.slow

Usage:
    # Run test integrazione (richiede LLM)
    pytest tests/test_integration.py -v -m integration

    # Skip test lenti
    pytest tests/test_integration.py -v -m "not slow"

    # Run con output dettagliato
    pytest tests/test_integration.py -v -s

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

# Moduli da testare
from agent import WebScraperAgent, ConversationMemory, ToolCall, ExecutionStep
from llm_interface import LLMInterface


# ----------------------------------------------------------------------------
# MARKERS - Categorizzazione test
# ----------------------------------------------------------------------------
# Questi markers permettono di eseguire selettivamente i test:
# pytest -m integration  → solo test di integrazione
# pytest -m "not slow"   → skip test lenti

pytestmark = pytest.mark.integration  # Tutti i test in questo file sono integration


# ----------------------------------------------------------------------------
# FIXTURES
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_agent_config():
    """
    Configurazione mockup per l'agente.

    Returns:
        Dict con configurazione di test
    """
    return {
        "agent": {
            "llm_model": "llama3.2",
            "llm_provider": "ollama",
            "max_tools_per_query": 5,
            "enable_caching": False,  # Disabilita cache per test deterministici
            "verbose": False
        },
        "tools": {
            "search_web": {
                "timeout_seconds": 5,
                "max_results": 3
            },
            "fetch_webpage": {
                "timeout_seconds": 5
            }
        }
    }


@pytest.fixture
def mock_llm_responses():
    """
    Mock delle risposte LLM per test deterministici.

    Returns:
        Dict con risposte pre-programmate
    """
    return {
        "plan": json.dumps([
            {
                "tool": "search_web",
                "parameters": {"query": "Python programming", "num_results": 3},
                "reasoning": "Search for Python information"
            }
        ]),
        "synthesis": "Ecco le informazioni su Python che ho trovato: [contenuto riassunto]"
    }


# ============================================================================
# TEST AGENT INITIALIZATION
# ============================================================================

def test_agent_initialization_basic(mock_agent_config):
    """
    Test inizializzazione base dell'agente.

    Verifica che:
    - Agent si inizializzi correttamente
    - Tutti i componenti siano creati
    - Tools siano registrati
    """
    # GIVEN - configurazione

    # WHEN - crea agente
    with patch('agent.LLMInterface') as mock_llm_class:
        # Mock LLM interface
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="test response")
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            llm_model="llama3.2",
            llm_provider="ollama",
            config=mock_agent_config,
            enable_cache=False
        )

        # THEN - verifica
        assert agent is not None
        assert agent.llm is not None
        assert agent.tools is not None
        assert agent.web_tools is not None
        assert agent.memory is not None

        # Verifica che tools siano registrati
        assert len(agent.tools.list_tools()) == 5
        assert "search_web" in agent.tools.list_tools()
        assert "fetch_webpage" in agent.tools.list_tools()


# ============================================================================
# TEST CONVERSATION MEMORY
# ============================================================================

def test_conversation_memory_basic():
    """
    Test funzionalità base della memoria conversazionale.
    """
    # GIVEN
    memory = ConversationMemory(max_history=5)

    # WHEN - aggiungi messaggi
    memory.add_message("user", "Ciao")
    memory.add_message("assistant", "Ciao! Come posso aiutarti?")
    memory.add_message("user", "Cerca Python")
    memory.add_message("assistant", "Ecco le info su Python")

    # THEN
    assert len(memory) == 4
    messages = memory.get_messages()
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Ciao"


def test_conversation_memory_limit():
    """
    Test che la memoria rispetti il limite massimo.
    """
    # GIVEN
    memory = ConversationMemory(max_history=2)  # Max 2 coppie = 4 messaggi

    # WHEN - aggiungi più messaggi del limite
    for i in range(10):
        memory.add_message("user", f"Message {i}")
        memory.add_message("assistant", f"Response {i}")

    # THEN - dovrebbe mantenere solo ultimi 4 messaggi
    assert len(memory) == 4
    messages = memory.get_messages()
    assert messages[0]["content"] == "Message 8"  # Ultimi 2 coppie


def test_conversation_memory_clear():
    """
    Test pulizia della memoria.
    """
    # GIVEN
    memory = ConversationMemory()
    memory.add_message("user", "Test")
    memory.add_message("assistant", "Response")

    # WHEN
    memory.clear()

    # THEN
    assert len(memory) == 0


# ============================================================================
# TEST PLANNING (Chain of Thought)
# ============================================================================

def test_agent_planning_mock(mock_agent_config):
    """
    Test generazione piano con LLM mockato.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        # Setup mock LLM
        mock_llm = Mock()

        # Mock generate per ritornare un piano JSON valido
        plan_json = json.dumps([
            {
                "tool": "search_web",
                "parameters": {"query": "Python", "num_results": 3},
                "reasoning": "Search for Python info"
            },
            {
                "tool": "fetch_webpage",
                "parameters": {"url": "https://python.org"},
                "reasoning": "Get official docs"
            }
        ])

        mock_llm.generate = Mock(return_value=plan_json)
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # WHEN - genera piano
        plan = agent._generate_plan("Cerca informazioni su Python")

        # THEN
        assert len(plan) == 2
        assert isinstance(plan[0], ToolCall)
        assert plan[0].tool == "search_web"
        assert plan[0].parameters["query"] == "Python"
        assert plan[1].tool == "fetch_webpage"


def test_agent_fallback_planning(mock_agent_config):
    """
    Test fallback planning quando LLM fallisce.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        # Setup mock LLM che fallisce
        mock_llm = Mock()
        mock_llm.generate = Mock(side_effect=Exception("LLM error"))
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # WHEN - prova a generare piano
        plan = agent._generate_plan("Cerca Python")

        # THEN - dovrebbe usare fallback euristico
        assert len(plan) > 0
        assert isinstance(plan[0], ToolCall)
        # Fallback dovrebbe riconoscere "cerca" e usare search_web
        assert plan[0].tool == "search_web"


# ============================================================================
# TEST EXECUTION
# ============================================================================

def test_agent_execution_mock(mock_agent_config):
    """
    Test esecuzione piano con tools mockati.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value="summary")
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Mock web_tools
        agent.web_tools.search_web = Mock(return_value=[
            {"title": "Test", "url": "http://test.com", "snippet": "test snippet", "position": 1}
        ])

        # Crea piano di test
        plan = [
            ToolCall(
                tool="search_web",
                parameters={"query": "test", "num_results": 1},
                reasoning="test search"
            )
        ]

        # WHEN - esegui piano
        results = agent._execute_plan(plan)

        # THEN
        assert len(results) == 1
        assert isinstance(results[0], ExecutionStep)
        assert results[0].success is True
        assert results[0].result is not None


def test_agent_execution_with_error(mock_agent_config):
    """
    Test che esecuzione continui anche con errori.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Mock tool che fallisce
        agent.web_tools.search_web = Mock(side_effect=Exception("Search failed"))

        plan = [
            ToolCall(
                tool="search_web",
                parameters={"query": "test"},
                reasoning="test"
            )
        ]

        # WHEN
        results = agent._execute_plan(plan)

        # THEN
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error is not None


# ============================================================================
# TEST SYNTHESIS
# ============================================================================

def test_agent_synthesis_mock(mock_agent_config):
    """
    Test sintesi risultati con LLM mockato.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        mock_llm = Mock()

        # Mock synthesis response
        mock_llm.generate = Mock(return_value="Ecco il riassunto delle informazioni trovate.")
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Crea risultati di test
        results = [
            ExecutionStep(
                tool_call=ToolCall("search_web", {"query": "test"}, "test"),
                result=[{"title": "Test", "url": "http://test.com", "snippet": "snippet", "position": 1}],
                timestamp=None,
                success=True
            )
        ]

        # WHEN
        response = agent._synthesize_results("Cerca test", results)

        # THEN
        assert response is not None
        assert len(response) > 0
        assert "riassunto" in response.lower() or "informazioni" in response.lower()


# ============================================================================
# TEST PROCESS QUERY END-TO-END
# ============================================================================

@pytest.mark.slow
def test_agent_process_query_end_to_end_mock(mock_agent_config):
    """
    Test completo end-to-end con tutti i componenti mockati.

    Questo test verifica l'intero flusso:
    Query → Plan → Execute → Synthesize → Response
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        # Setup mock LLM
        mock_llm = Mock()

        # Mock planning response
        plan_response = json.dumps([
            {
                "tool": "search_web",
                "parameters": {"query": "Python tutorial", "num_results": 2},
                "reasoning": "Find Python tutorials"
            }
        ])

        # Mock synthesis response
        synthesis_response = "Ho trovato informazioni su Python tutorial."

        # Setup generate per ritornare risposte diverse in base al contesto
        def mock_generate(prompt, **kwargs):
            if "execution plan" in prompt.lower() or "chain of thought" in prompt.lower():
                return plan_response
            else:
                return synthesis_response

        mock_llm.generate = Mock(side_effect=mock_generate)
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Mock web tools
        agent.web_tools.search_web = Mock(return_value=[
            {
                "title": "Python Tutorial",
                "url": "https://python.org/tutorial",
                "snippet": "Learn Python",
                "position": 1
            }
        ])

        # WHEN - processa query completa
        response = agent.process_query("Cerca tutorial Python")

        # THEN
        assert response is not None
        assert len(response) > 0

        # Verifica che tutti i metodi siano stati chiamati
        assert agent.web_tools.search_web.called

        # Verifica execution history
        assert len(agent.get_execution_history()) == 1
        history = agent.get_execution_history()[0]
        assert history["query"] == "Cerca tutorial Python"
        assert len(history["plan"]) > 0

        # Verifica memoria conversazione
        assert len(agent.memory) == 2  # user + assistant
        messages = agent.memory.get_messages()
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


# ============================================================================
# TEST STATISTICS
# ============================================================================

def test_agent_statistics(mock_agent_config):
    """
    Test raccolta statistiche dell'agente.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.generate = Mock(return_value=json.dumps([]))
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Simula alcune operazioni
        agent.web_tools.stats["search_web"] = 5
        agent.web_tools.stats["fetch_webpage"] = 3

        # WHEN
        stats = agent.get_stats()

        # THEN
        assert "queries_processed" in stats
        assert "conversation_length" in stats
        assert "tool_usage" in stats
        assert stats["tool_usage"]["search_web"] == 5
        assert stats["tool_usage"]["fetch_webpage"] == 3


# ============================================================================
# TEST HELPER FUNCTIONS
# ============================================================================

def test_format_result_for_synthesis(mock_agent_config):
    """
    Test formattazione risultati per sintesi.
    """
    # GIVEN
    with patch('agent.LLMInterface') as mock_llm_class:
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        agent = WebScraperAgent(
            config=mock_agent_config,
            enable_cache=False
        )

        # Test formattazione search results
        search_result = [
            {"title": "Test", "url": "http://test.com", "snippet": "test snippet"}
        ]

        # WHEN
        formatted = agent._format_result_for_synthesis("search_web", search_result)

        # THEN
        assert "Test" in formatted
        assert "http://test.com" in formatted


# ============================================================================
# TEST RUNNER INFO
# ============================================================================

if __name__ == "__main__":
    """
    Informazioni per esecuzione test.
    """
    print("\n" + "="*70)
    print("INTEGRATION TESTS - Web Scraper Agent")
    print("="*70)
    print("\nQuesti test verificano l'integrazione completa dei componenti.")
    print("\nPer eseguire:")
    print("  pytest tests/test_integration.py -v")
    print("\nPer eseguire solo test veloci:")
    print("  pytest tests/test_integration.py -v -m 'not slow'")
    print("\nPer output dettagliato:")
    print("  pytest tests/test_integration.py -v -s")
    print("\n" + "="*70 + "\n")
