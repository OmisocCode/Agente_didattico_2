"""
============================================================================
UNIT TESTS - Test Unitari per Web Tools
============================================================================

Questo modulo contiene test unitari per tutti i tools implementati
nel sistema Web Scraper Agent.

Test Coverage:
- WebTools: Tutti i 5 tools
- ToolRegistry: Registrazione e chiamata tools
- ResultCache: Caching e TTL
- LLMInterface: Inizializzazione providers (mock)

Framework: pytest
Pattern: Arrange-Act-Assert (AAA)
Mocking: responses library per HTTP, mock per LLM

Usage:
    # Run tutti i test
    pytest tests/test_tools.py -v

    # Run test specifico
    pytest tests/test_tools.py::test_search_web_basic -v

    # Con coverage
    pytest tests/test_tools.py --cov=web_tools --cov-report=html

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import pytest
import responses
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import time

# Moduli da testare
from web_tools import WebTools
from tool_registry import ToolRegistry, Tool
from cache import ResultCache
from llm_interface import LLMInterface


# ----------------------------------------------------------------------------
# FIXTURES - Setup condiviso tra test
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """
    Crea mock LLM per test senza chiamate reali.

    Returns:
        Mock object che simula LLMInterface
    """
    llm = Mock(spec=LLMInterface)

    # Mock generate method
    llm.generate = Mock(return_value="Questo è un riassunto generato dal mock LLM.")

    # Mock chat method
    llm.chat = Mock(return_value="Risposta chat dal mock LLM.")

    # Mock function_call method
    llm.function_call = Mock(return_value={
        "tool": "search_web",
        "parameters": {"query": "test query"},
        "reasoning": "Mock reasoning"
    })

    return llm


@pytest.fixture
def web_tools(mock_llm):
    """
    Crea istanza WebTools con LLM mockato.

    Args:
        mock_llm: LLM fixture mockato

    Returns:
        WebTools instance ready for testing
    """
    config = {
        "tools": {
            "search_web": {
                "timeout_seconds": 5,
                "max_results": 10
            },
            "fetch_webpage": {
                "timeout_seconds": 5,
                "user_agent": "Test Agent"
            }
        }
    }

    return WebTools(config=config, llm_interface=mock_llm)


@pytest.fixture
def temp_cache(tmp_path):
    """
    Crea cache temporaneo per testing.

    Args:
        tmp_path: pytest fixture per directory temporanea

    Returns:
        ResultCache in directory temporanea
    """
    cache_dir = tmp_path / "test_cache"
    return ResultCache(
        cache_dir=str(cache_dir),
        ttl_seconds=10,  # TTL breve per test
        max_size_mb=1
    )


# ============================================================================
# TEST SEARCH_WEB
# ============================================================================

@responses.activate
def test_search_web_basic(web_tools):
    """
    Test ricerca web base.

    Verifica che search_web:
    - Faccia richiesta HTTP corretta
    - Ritorni risultati nel formato atteso
    - Gestisca correttamente i parametri
    """
    # ARRANGE - Prepara
    # Mock risposta DuckDuckGo
    # Nota: DuckDuckGo usa un'API interna, quindi mocchiamo a livello più alto

    with patch('web_tools.DDGS') as mock_ddgs:
        # Configura mock
        mock_results = [
            {
                "title": "Python Tutorial",
                "href": "https://example.com/python",
                "body": "Learn Python programming"
            },
            {
                "title": "Python Documentation",
                "href": "https://python.org/docs",
                "body": "Official Python docs"
            }
        ]

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance

        # ACT - Esegui
        results = web_tools.search_web("Python programming", num_results=2)

        # ASSERT - Verifica
        assert len(results) == 2
        assert all("title" in r for r in results)
        assert all("url" in r for r in results)
        assert all("snippet" in r for r in results)
        assert all("position" in r for r in results)

        # Verifica contenuto
        assert results[0]["title"] == "Python Tutorial"
        assert results[0]["url"] == "https://example.com/python"
        assert results[0]["position"] == 1


def test_search_web_empty_query(web_tools):
    """
    Test che query vuota sollevi ValueError.
    """
    # ACT & ASSERT
    with pytest.raises(ValueError, match="cannot be empty"):
        web_tools.search_web("")


def test_search_web_num_results_limit(web_tools):
    """
    Test che num_results sia limitato correttamente.
    """
    with patch('web_tools.DDGS') as mock_ddgs:
        # Setup mock
        mock_results = [{"title": f"Result {i}", "href": f"http://ex{i}.com", "body": f"Body {i}"}
                       for i in range(25)]

        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__.return_value.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance

        # Richiedi 30 risultati (oltre il limite di 20)
        results = web_tools.search_web("test", num_results=30)

        # Verifica che sia limitato a 20
        assert len(results) <= 20


# ============================================================================
# TEST FETCH_WEBPAGE
# ============================================================================

@responses.activate
def test_fetch_webpage_basic():
    """
    Test download pagina web base.
    """
    # ARRANGE
    url = "https://example.com/test"

    html_content = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Main Content</h1>
        <p>This is a test paragraph.</p>
        <a href="/link1">Link 1</a>
        <a href="https://example.com/link2">Link 2</a>
    </body>
    </html>
    """

    # Mock HTTP response
    responses.add(
        responses.GET,
        url,
        body=html_content,
        status=200,
        content_type="text/html"
    )

    # ACT
    tools = WebTools()
    result = tools.fetch_webpage(url, extract_main_content=False)

    # ASSERT
    assert result["url"] == url
    assert result["title"] == "Test Page"
    assert result["status_code"] == 200
    assert "Main Content" in result["content"]
    assert "test paragraph" in result["content"]
    assert len(result["links"]) >= 2


@responses.activate
def test_fetch_webpage_invalid_url(web_tools):
    """
    Test che URL invalido sollevi ValueError.
    """
    with pytest.raises(ValueError, match="Invalid URL"):
        web_tools.fetch_webpage("not-a-valid-url")


@responses.activate
def test_fetch_webpage_timeout():
    """
    Test gestione timeout.
    """
    url = "https://slow-site.com"

    # Mock timeout
    responses.add(
        responses.GET,
        url,
        body=Exception("Timeout")
    )

    tools = WebTools()

    with pytest.raises(RuntimeError):
        tools.fetch_webpage(url, timeout=1)


# ============================================================================
# TEST EXTRACT_STRUCTURED_DATA
# ============================================================================

def test_extract_structured_data_basic(web_tools):
    """
    Test estrazione dati strutturati base.
    """
    # ARRANGE
    html = """
    <div class="product">
        <h2 class="name">Laptop</h2>
        <span class="price">$999</span>
        <a href="/laptop">Buy</a>
    </div>
    <div class="product">
        <h2 class="name">Mouse</h2>
        <span class="price">$25</span>
        <a href="/mouse">Buy</a>
    </div>
    """

    schema = {
        "selector": ".product",
        "fields": {
            "name": ".name",
            "price": ".price",
            "link": {
                "selector": "a",
                "attr": "href"
            }
        }
    }

    # ACT
    results = web_tools.extract_structured_data(html, schema)

    # ASSERT
    assert len(results) == 2

    assert results[0]["name"] == "Laptop"
    assert results[0]["price"] == "$999"
    assert results[0]["link"] == "/laptop"

    assert results[1]["name"] == "Mouse"
    assert results[1]["price"] == "$25"


def test_extract_structured_data_empty_html(web_tools):
    """
    Test con HTML senza elementi che matchano.
    """
    html = "<div>No matching elements</div>"

    schema = {
        "selector": ".product",
        "fields": {"name": ".name"}
    }

    results = web_tools.extract_structured_data(html, schema)

    assert len(results) == 0


def test_extract_structured_data_invalid_schema(web_tools):
    """
    Test con schema invalido.
    """
    with pytest.raises(ValueError):
        web_tools.extract_structured_data("<html></html>", {"invalid": "schema"})


# ============================================================================
# TEST SUMMARIZE_CONTENT
# ============================================================================

def test_summarize_content_basic(web_tools, mock_llm):
    """
    Test riassunto contenuto base.
    """
    # ARRANGE
    long_text = " ".join(["This is a test sentence."] * 200)  # ~1000 parole

    # ACT
    summary = web_tools.summarize_content(long_text, max_length=100)

    # ASSERT
    assert summary is not None
    assert len(summary) > 0

    # Verifica che LLM sia stato chiamato
    mock_llm.generate.assert_called_once()

    # Verifica parametri chiamata
    call_args = mock_llm.generate.call_args
    assert "summarize" in call_args[0][0].lower() or "riassumi" in call_args[0][0].lower()


def test_summarize_content_short_text(web_tools):
    """
    Test che testo già corto non venga riassunto.
    """
    short_text = "This is a short text."

    result = web_tools.summarize_content(short_text, max_length=100)

    # Dovrebbe ritornare il testo originale
    assert result == short_text


def test_summarize_content_no_llm():
    """
    Test che senza LLM sollevi errore.
    """
    tools = WebTools(llm_interface=None)

    with pytest.raises(RuntimeError, match="LLM interface"):
        tools.summarize_content("Some text")


# ============================================================================
# TEST COMPARE_SOURCES
# ============================================================================

@responses.activate
def test_compare_sources_basic(web_tools, mock_llm):
    """
    Test confronto sorgenti base.
    """
    # ARRANGE
    # Mock HTTP responses per URLs
    responses.add(
        responses.GET,
        "https://site1.com",
        body="<html><head><title>Site 1</title></head><body>Content from site 1</body></html>",
        status=200
    )

    responses.add(
        responses.GET,
        "https://site2.com",
        body="<html><head><title>Site 2</title></head><body>Content from site 2</body></html>",
        status=200
    )

    # Mock LLM response
    mock_llm.generate.return_value = json.dumps({
        "consensus": "Both sources agree on main points",
        "differences": "Site 1 has more details",
        "summary": "Overall comparison summary"
    })

    # ACT
    result = web_tools.compare_sources(
        ["https://site1.com", "https://site2.com"],
        topic="Test topic"
    )

    # ASSERT
    assert "consensus" in result
    assert "differences" in result
    assert "summary" in result
    assert result["sources_analyzed"] == 2


def test_compare_sources_insufficient_sources(web_tools):
    """
    Test che meno di 2 sorgenti sollevi errore.
    """
    with pytest.raises(ValueError, match="At least 2 sources"):
        web_tools.compare_sources(["https://site1.com"])


# ============================================================================
# TEST TOOL REGISTRY
# ============================================================================

def test_tool_registry_register():
    """
    Test registrazione tool nel registry.
    """
    # ARRANGE
    registry = ToolRegistry()

    def sample_function(param1: str) -> str:
        return f"Result: {param1}"

    # ACT
    tool = registry.register(
        name="test_tool",
        function=sample_function,
        description="Test tool",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": ["param1"]
        }
    )

    # ASSERT
    assert isinstance(tool, Tool)
    assert "test_tool" in registry.list_tools()
    assert len(registry) == 1


def test_tool_registry_call_tool():
    """
    Test chiamata tool dal registry.
    """
    # ARRANGE
    registry = ToolRegistry()

    def add(a: int, b: int) -> int:
        return a + b

    registry.register(
        name="add",
        function=add,
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    )

    # ACT
    result = registry.call_tool("add", a=5, b=3)

    # ASSERT
    assert result == 8


def test_tool_registry_missing_required_param():
    """
    Test che parametro mancante sollevi errore.
    """
    registry = ToolRegistry()

    def sample(required_param: str):
        return required_param

    registry.register(
        name="sample",
        function=sample,
        description="Sample",
        parameters={
            "type": "object",
            "properties": {
                "required_param": {"type": "string"}
            },
            "required": ["required_param"]
        }
    )

    # ACT & ASSERT
    with pytest.raises(ValueError, match="Missing required parameter"):
        registry.call_tool("sample")


# ============================================================================
# TEST RESULT CACHE
# ============================================================================

def test_cache_set_and_get(temp_cache):
    """
    Test salvataggio e recupero da cache.
    """
    # ARRANGE
    tool = "test_tool"
    params = {"param1": "value1"}
    result = {"data": "test result"}

    # ACT
    temp_cache.set(tool, params, result)
    cached = temp_cache.get(tool, params)

    # ASSERT
    assert cached == result


def test_cache_miss(temp_cache):
    """
    Test cache miss con parametri diversi.
    """
    # ARRANGE
    temp_cache.set("tool1", {"p": "v1"}, "result1")

    # ACT
    cached = temp_cache.get("tool1", {"p": "v2"})  # Parametri diversi

    # ASSERT
    assert cached is None


def test_cache_ttl_expiration(temp_cache):
    """
    Test scadenza TTL.
    """
    # ARRANGE
    temp_cache.set("tool", {"p": "v"}, "result")

    # ACT
    # Aspetta che TTL scada (10 secondi nel fixture)
    time.sleep(11)

    cached = temp_cache.get("tool", {"p": "v"})

    # ASSERT
    assert cached is None  # Dovrebbe essere scaduto


def test_cache_clear(temp_cache):
    """
    Test svuotamento cache.
    """
    # ARRANGE
    temp_cache.set("tool1", {"p": "v1"}, "result1")
    temp_cache.set("tool2", {"p": "v2"}, "result2")

    # ACT
    temp_cache.clear()

    # ASSERT
    assert temp_cache.get("tool1", {"p": "v1"}) is None
    assert temp_cache.get("tool2", {"p": "v2"}) is None


def test_cache_stats(temp_cache):
    """
    Test statistiche cache.
    """
    # ARRANGE & ACT
    temp_cache.set("tool", {"p": "v"}, "result")
    temp_cache.get("tool", {"p": "v"})  # Hit
    temp_cache.get("tool", {"p": "different"})  # Miss

    stats = temp_cache.get_stats()

    # ASSERT
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["sets"] == 1


# ============================================================================
# TEST RUNNER INFO
# ============================================================================

if __name__ == "__main__":
    """
    Esegui test direttamente (senza pytest).

    Per uso normale, usa: pytest tests/test_tools.py
    """
    print("\n" + "="*70)
    print("UNIT TESTS - Web Tools")
    print("="*70)
    print("\nPer eseguire i test, usa:")
    print("  pytest tests/test_tools.py -v")
    print("\nPer coverage:")
    print("  pytest tests/test_tools.py --cov=web_tools --cov-report=html")
    print("\n" + "="*70 + "\n")
