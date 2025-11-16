# 📋 Piano di Sviluppo - Agente Web Scraper Intelligente

Questo documento fornisce un piano di sviluppo dettagliato step-by-step per implementare l'agente web scraper intelligente.

---

## 🎯 Overview

Il progetto verrà sviluppato in **4 fasi principali**:

1. **Fase 1: Fondamenta** (Step 1-3) - Setup e infrastruttura base
2. **Fase 2: Tools** (Step 4-8) - Implementazione dei tools specifici
3. **Fase 3: Agente** (Step 9-13) - Logica principale dell'agente
4. **Fase 4: Testing e Docs** (Step 14-16) - Test e documentazione

**Tempo stimato totale**: 15-20 ore di sviluppo

---

## FASE 1: FONDAMENTA

### Step 1: Setup Iniziale del Progetto
**Tempo stimato**: 30 minuti
**Priorità**: CRITICA
**Dipendenze**: Nessuna

#### Obiettivi
- Creare struttura cartelle completa
- Setup file di configurazione
- Creare requirements.txt con tutte le dipendenze
- Configurare variabili ambiente

#### Task Specifici

1. **Creare struttura cartelle**:
```bash
mkdir -p web-scraper-agent/{tests,docs,examples/{queries,cached_results}}
touch web-scraper-agent/{agent.py,web_tools.py,html_parser.py,llm_interface.py,main.py}
touch web-scraper-agent/{config.yaml,.env.example,requirements.txt}
```

2. **Popolare requirements.txt**:
```txt
# LLM
ollama>=0.1.0

# Web Scraping
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
readability-lxml>=0.8.1

# Search
duckduckgo-search>=4.0.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
validators>=0.22.0
tqdm>=4.66.0
tenacity>=8.2.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
responses>=0.23.0
```

3. **Creare config.yaml**:
```yaml
# Configurazione Agente
agent:
  llm_model: "llama3.2"
  max_tools_per_query: 5
  enable_caching: true
  cache_ttl_seconds: 3600

# Configurazione Tools
tools:
  search_web:
    enabled: true
    max_results: 10
    search_engine: "duckduckgo"  # o "google", "bing"

  fetch_webpage:
    enabled: true
    timeout_seconds: 10
    max_retries: 3
    user_agent: "Mozilla/5.0 (Educational Web Scraper Agent)"

  extract_structured_data:
    enabled: true

  summarize_content:
    enabled: true
    max_length: 500

  compare_sources:
    enabled: true
    min_sources: 2

# Logging
logging:
  level: "INFO"
  file: "agent.log"
```

4. **Creare .env.example**:
```bash
# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OPENAI_API_KEY=sk-...  # Opzionale
ANTHROPIC_API_KEY=sk-...  # Opzionale

# Search APIs (Opzionali)
GOOGLE_API_KEY=...
GOOGLE_SEARCH_ENGINE_ID=...
SERPAPI_KEY=...

# General
LOG_LEVEL=INFO
CACHE_DIR=.cache
```

#### Criteri di Successo
- [ ] Struttura cartelle completa
- [ ] requirements.txt con tutte le dipendenze
- [ ] config.yaml configurato
- [ ] .env.example creato
- [ ] `pip install -r requirements.txt` funziona

---

### Step 2: Implementare LLMInterface
**Tempo stimato**: 1.5 ore
**Priorità**: CRITICA
**Dipendenze**: Step 1

#### Obiettivi
Creare un'interfaccia unificata per interagire con diversi LLM (Ollama, OpenAI, Anthropic)

#### Design

```python
# llm_interface.py

from typing import List, Dict, Optional, Any
import os
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Interfaccia base per tutti gli LLM"""

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Genera una risposta semplice"""
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Conversazione con contesto"""
        pass

    @abstractmethod
    def function_call(self, query: str, tools: List[Dict]) -> Dict:
        """Tool calling (se supportato dal modello)"""
        pass


class OllamaLLM(BaseLLM):
    """Implementazione per Ollama (locale)"""

    def __init__(self, model: str = "llama3.2", host: str = None):
        import ollama
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages
        )
        return response['message']['content']

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages
        )
        return response['message']['content']

    def function_call(self, query: str, tools: List[Dict]) -> Dict:
        # Ollama supporta function calling con alcuni modelli
        # Per ora: simulate con prompt engineering
        tools_desc = self._format_tools_for_prompt(tools)

        prompt = f"""
        User query: {query}

        Available tools:
        {tools_desc}

        Based on the query, decide which tool to call and with what parameters.
        Respond in JSON format:
        {{
            "tool": "tool_name",
            "parameters": {{...}},
            "reasoning": "why this tool"
        }}
        """

        response = self.generate(prompt)
        return self._parse_function_call(response)

    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """Formatta tools per il prompt"""
        lines = []
        for tool in tools:
            lines.append(f"- {tool['name']}: {tool['description']}")
            lines.append(f"  Parameters: {tool['parameters']}")
        return "\n".join(lines)

    def _parse_function_call(self, response: str) -> Dict:
        """Parse risposta LLM in formato function call"""
        import json
        # Estrai JSON dal testo
        try:
            # Cerca blocco JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            return json.loads(json_str)
        except:
            return {"error": "Failed to parse function call"}


class OpenAILLM(BaseLLM):
    """Implementazione per OpenAI API"""

    def __init__(self, model: str = "gpt-4", api_key: str = None):
        import openai
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    def function_call(self, query: str, tools: List[Dict]) -> Dict:
        # OpenAI ha function calling nativo
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": query}],
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "tool": tool_call.function.name,
                "parameters": json.loads(tool_call.function.arguments)
            }
        return {"error": "No tool called"}


class LLMInterface:
    """
    Interfaccia unificata che auto-seleziona il provider giusto
    """

    def __init__(self, model: str = "llama3.2", provider: str = "ollama"):
        self.model = model
        self.provider = provider

        # Inizializza il provider appropriato
        if provider == "ollama":
            self.llm = OllamaLLM(model)
        elif provider == "openai":
            self.llm = OpenAILLM(model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Genera una risposta semplice"""
        return self.llm.generate(prompt, system)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Conversazione con contesto"""
        return self.llm.chat(messages)

    def function_call(self, query: str, tools: List[Dict]) -> Dict:
        """Tool calling"""
        return self.llm.function_call(query, tools)
```

#### Task Specifici
1. Creare `llm_interface.py` con struttura base
2. Implementare `OllamaLLM` (priorità alta)
3. Implementare `OpenAILLM` (opzionale)
4. Creare tests per verificare connessione

#### Criteri di Successo
- [ ] LLMInterface funziona con Ollama
- [ ] Test di connessione passa
- [ ] Supporto per generate() e chat()
- [ ] Function calling base implementato

---

### Step 3: Creare ToolRegistry
**Tempo stimato**: 1 ora
**Priorità**: CRITICA
**Dipendenze**: Step 1

#### Obiettivi
Sistema centralizzato per registrare, gestire e invocare tools

#### Design

```python
# tool_registry.py

from typing import Callable, Dict, Any, List
import inspect
from functools import wraps

class Tool:
    """Rappresentazione di un singolo tool"""

    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        self.name = name
        self.function = function
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> Dict:
        """Serializza per LLM function calling"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def validate_params(self, params: Dict) -> bool:
        """Valida parametri prima di chiamare il tool"""
        required = self.parameters.get("required", [])
        for param in required:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        return True

    def __call__(self, **kwargs) -> Any:
        """Esegui il tool"""
        self.validate_params(kwargs)
        return self.function(**kwargs)


class ToolRegistry:
    """
    Registro centralizzato di tutti i tools disponibili.
    Gestisce registrazione, discovery e invocazione.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Registra un nuovo tool"""
        tool = Tool(name, function, description, parameters)
        self.tools[name] = tool
        return tool

    def register_decorator(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Decorator per registrare tools"""
        def decorator(func: Callable):
            self.register(name, func, description, parameters)
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def get_tool(self, name: str) -> Tool:
        """Recupera un tool per nome"""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]

    def call_tool(self, name: str, **kwargs) -> Any:
        """Esegue un tool con validazione"""
        tool = self.get_tool(name)
        return tool(**kwargs)

    def get_all_tools(self) -> List[Tool]:
        """Ottieni tutti i tools registrati"""
        return list(self.tools.values())

    def get_tool_descriptions(self) -> str:
        """Genera descrizioni testuali per prompt LLM"""
        descriptions = []
        for tool in self.tools.values():
            desc = f"**{tool.name}**\n"
            desc += f"{tool.description}\n"
            desc += f"Parameters: {tool.parameters}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def get_tools_for_function_calling(self) -> List[Dict]:
        """Formato per LLM function calling API"""
        return [tool.to_dict() for tool in self.tools.values()]

    def list_tools(self) -> List[str]:
        """Lista nomi di tutti i tools"""
        return list(self.tools.keys())
```

#### Esempio d'uso

```python
# Esempio di registrazione tools
registry = ToolRegistry()

# Metodo 1: Registrazione diretta
def search_web(query: str, num_results: int = 5):
    # ... implementazione
    pass

registry.register(
    name="search_web",
    function=search_web,
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "Number of results"}
        },
        "required": ["query"]
    }
)

# Metodo 2: Usando decorator
@registry.register_decorator(
    name="fetch_webpage",
    description="Fetch and parse a webpage",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"}
        },
        "required": ["url"]
    }
)
def fetch_webpage(url: str):
    # ... implementazione
    pass

# Uso
result = registry.call_tool("search_web", query="AI news", num_results=5)
```

#### Criteri di Successo
- [ ] ToolRegistry implementato
- [ ] Sistema di validazione parametri funzionante
- [ ] Supporto per decorator pattern
- [ ] Generazione descrizioni per LLM

---

## FASE 2: TOOLS

### Step 4: Implementare search_web
**Tempo stimato**: 1.5 ore
**Priorità**: ALTA
**Dipendenze**: Step 3

#### Obiettivi
Tool per cercare informazioni sul web usando DuckDuckGo

#### Implementazione

```python
# web_tools.py

from typing import List, Dict, Optional
from duckduckgo_search import DDGS
import validators
from tenacity import retry, stop_after_attempt, wait_exponential

class WebTools:
    """Collezione di tools per web search e scraping"""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search_web(
        self,
        query: str,
        num_results: int = 5,
        region: str = "it-it"
    ) -> List[Dict[str, str]]:
        """
        Cerca informazioni sul web usando DuckDuckGo.

        Args:
            query: Query di ricerca
            num_results: Numero di risultati (default 5, max 20)
            region: Regione per i risultati (default: it-it)

        Returns:
            Lista di dizionari con:
            - title: Titolo del risultato
            - url: URL della pagina
            - snippet: Anteprima del contenuto

        Raises:
            ValueError: Se query è vuota
            RuntimeError: Se la ricerca fallisce
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if num_results > 20:
            num_results = 20

        try:
            # Usa DuckDuckGo per la ricerca
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(
                    query,
                    region=region,
                    safesearch='moderate',
                    max_results=num_results
                ):
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", "")
                    })

                return results

        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")
```

#### Task Specifici
1. Implementare `search_web()` con DuckDuckGo
2. Aggiungere retry logic per errori di rete
3. Validare input (query non vuota, num_results ragionevole)
4. Gestire errori e timeout
5. Registrare nel ToolRegistry

#### Test

```python
# tests/test_search_tool.py

def test_search_web_basic():
    tools = WebTools()
    results = tools.search_web("Python programming", num_results=3)

    assert len(results) <= 3
    assert all("url" in r for r in results)
    assert all("title" in r for r in results)
    assert all("snippet" in r for r in results)

def test_search_web_empty_query():
    tools = WebTools()
    with pytest.raises(ValueError):
        tools.search_web("")
```

#### Criteri di Successo
- [ ] search_web funziona con DuckDuckGo
- [ ] Retry logic implementato
- [ ] Gestione errori robusta
- [ ] Test passano

---

### Step 5: Implementare fetch_webpage
**Tempo stimato**: 2 ore
**Priorità**: ALTA
**Dipendenze**: Step 3

#### Obiettivi
Tool per scaricare e parsare pagine HTML, estraendo contenuto pulito

#### Implementazione

```python
# web_tools.py (continuazione)

import requests
from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urljoin, urlparse

class WebTools:
    # ... codice precedente ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def fetch_webpage(
        self,
        url: str,
        timeout: int = 10,
        extract_main_content: bool = True
    ) -> Dict[str, Any]:
        """
        Scarica e parsifica una pagina web.

        Args:
            url: URL della pagina da scaricare
            timeout: Timeout in secondi (default 10)
            extract_main_content: Se True, estrae solo contenuto principale

        Returns:
            Dizionario con:
            - url: URL effettivo (dopo redirect)
            - title: Titolo della pagina
            - content: Testo pulito del contenuto
            - html: HTML originale
            - links: Lista di link trovati
            - meta: Metadata (author, date, description, etc.)
        """
        # Valida URL
        if not validators.url(url):
            raise ValueError(f"Invalid URL: {url}")

        # Headers per evitare blocchi
        headers = {
            'User-Agent': self.config.get(
                'user_agent',
                'Mozilla/5.0 (Educational Web Scraper Agent)'
            )
        }

        try:
            # Scarica pagina
            response = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')

            # Estrai contenuto principale usando readability
            if extract_main_content:
                doc = Document(response.text)
                title = doc.title()
                main_html = doc.summary()
                main_soup = BeautifulSoup(main_html, 'lxml')
                content = main_soup.get_text(separator='\n', strip=True)
            else:
                title = soup.title.string if soup.title else ""
                content = soup.get_text(separator='\n', strip=True)

            # Estrai metadata
            meta = self._extract_metadata(soup)

            # Estrai links
            links = self._extract_links(soup, url)

            return {
                "url": response.url,
                "title": title,
                "content": content,
                "html": response.text,
                "links": links,
                "meta": meta,
                "status_code": response.status_code
            }

        except requests.Timeout:
            raise RuntimeError(f"Timeout fetching {url}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch {url}: {str(e)}")

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Estrae metadata da HTML"""
        meta = {}

        # Meta tags
        for tag in soup.find_all('meta'):
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                meta[name] = content

        # Common metadata
        meta['author'] = (
            meta.get('author') or
            meta.get('article:author') or
            ""
        )
        meta['date'] = (
            meta.get('article:published_time') or
            meta.get('date') or
            ""
        )
        meta['description'] = (
            meta.get('description') or
            meta.get('og:description') or
            ""
        )

        return meta

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str
    ) -> List[Dict[str, str]]:
        """Estrae tutti i link dalla pagina"""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Converti link relativi in assoluti
            absolute_url = urljoin(base_url, href)

            # Valida URL
            if validators.url(absolute_url):
                links.append({
                    "url": absolute_url,
                    "text": a_tag.get_text(strip=True)
                })

        return links
```

#### Criteri di Successo
- [ ] fetch_webpage scarica e parsifica HTML
- [ ] Estrazione contenuto principale con readability
- [ ] Metadata extraction funzionante
- [ ] Links extraction implementato
- [ ] Test passano

---

### Step 6: Implementare extract_structured_data
**Tempo stimato**: 1.5 ore
**Priorità**: MEDIA
**Dipendenze**: Step 5

#### Implementazione

```python
# web_tools.py (continuazione)

class WebTools:
    # ... codice precedente ...

    def extract_structured_data(
        self,
        html: str,
        schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Estrae dati strutturati da HTML usando schema con selettori CSS.

        Args:
            html: HTML da cui estrarre dati
            schema: Schema di estrazione con selettori CSS

        Schema format:
        {
            "selector": ".item",  # Selettore per contenitore item
            "fields": {
                "name": ".item-name",  # Selettori per campi
                "price": ".item-price",
                "link": {"selector": "a", "attr": "href"}
            }
        }

        Returns:
            Lista di dizionari con dati estratti
        """
        soup = BeautifulSoup(html, 'lxml')
        results = []

        # Trova tutti i contenitori
        containers = soup.select(schema.get("selector", "body"))

        for container in containers:
            item = {}
            fields = schema.get("fields", {})

            for field_name, field_selector in fields.items():
                # Selettore può essere string o dict
                if isinstance(field_selector, str):
                    element = container.select_one(field_selector)
                    if element:
                        item[field_name] = element.get_text(strip=True)
                elif isinstance(field_selector, dict):
                    selector = field_selector.get("selector")
                    attr = field_selector.get("attr")
                    element = container.select_one(selector)
                    if element:
                        if attr:
                            item[field_name] = element.get(attr, "")
                        else:
                            item[field_name] = element.get_text(strip=True)

            if item:
                results.append(item)

        return results
```

---

### Step 7: Implementare summarize_content
**Tempo stimato**: 1 ora
**Priorità**: ALTA
**Dipendenze**: Step 2

#### Implementazione

```python
# web_tools.py (continuazione)

class WebTools:
    # ... codice precedente ...

    def __init__(self, config: Dict = None, llm_interface = None):
        self.config = config or {}
        self.llm = llm_interface

    def summarize_content(
        self,
        text: str,
        max_length: int = 500,
        style: str = "concise"
    ) -> str:
        """
        Riassume un testo lungo usando LLM.

        Args:
            text: Testo da riassumere
            max_length: Lunghezza massima del riassunto (in parole)
            style: Stile del riassunto ("concise", "detailed", "bullet_points")

        Returns:
            Testo riassunto
        """
        if not self.llm:
            raise RuntimeError("LLM interface not configured")

        if len(text.split()) < max_length:
            return text  # Già abbastanza corto

        style_instructions = {
            "concise": "Create a concise summary",
            "detailed": "Create a detailed summary maintaining key points",
            "bullet_points": "Create a bullet-point summary"
        }

        prompt = f"""
        {style_instructions.get(style, "Summarize the following text")}
        in approximately {max_length} words.

        Text:
        {text}

        Summary:
        """

        summary = self.llm.generate(prompt)
        return summary.strip()
```

---

### Step 8: Implementare compare_sources
**Tempo stimato**: 1.5 ore
**Priorità**: MEDIA
**Dipendenze**: Step 2, Step 5

#### Implementazione

```python
# web_tools.py (continuazione)

class WebTools:
    # ... codice precedente ...

    def compare_sources(
        self,
        sources: List[str],
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Confronta informazioni da fonti multiple.

        Args:
            sources: Lista di URL o testi da confrontare
            topic: Topic specifico su cui focalizzare (opzionale)

        Returns:
            Dizionario con:
            - consensus: Punti su cui le fonti concordano
            - differences: Differenze tra le fonti
            - summary: Sintesi generale
        """
        if not self.llm:
            raise RuntimeError("LLM interface not configured")

        # Fetch content se sono URLs
        contents = []
        for source in sources:
            if validators.url(source):
                page = self.fetch_webpage(source)
                contents.append({
                    "url": source,
                    "title": page["title"],
                    "content": page["content"][:2000]  # Limita lunghezza
                })
            else:
                contents.append({
                    "url": "text",
                    "content": source[:2000]
                })

        # Crea prompt per confronto
        sources_text = "\n\n---\n\n".join([
            f"Source {i+1} ({c['url']}):\n{c['content']}"
            for i, c in enumerate(contents)
        ])

        topic_instruction = f" focusing on: {topic}" if topic else ""

        prompt = f"""
        Compare the following sources{topic_instruction}.

        {sources_text}

        Provide:
        1. CONSENSUS: What all sources agree on
        2. DIFFERENCES: Where sources disagree or provide unique information
        3. SUMMARY: Overall synthesis

        Format your response as JSON:
        {{
            "consensus": "...",
            "differences": "...",
            "summary": "..."
        }}
        """

        response = self.llm.generate(prompt)

        # Parse JSON response
        import json
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            result = json.loads(response[start:end])
            return result
        except:
            return {
                "consensus": "",
                "differences": "",
                "summary": response
            }
```

---

## FASE 3: AGENTE

### Step 9: Creare WebScraperAgent
**Tempo stimato**: 2 ore
**Priorità**: CRITICA
**Dipendenze**: Step 2, Step 3, Step 4-8

#### Design Architetturale

```python
# agent.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from llm_interface import LLMInterface
from tool_registry import ToolRegistry
from web_tools import WebTools

@dataclass
class ToolCall:
    """Rappresenta una chiamata a un tool"""
    tool: str
    parameters: Dict[str, Any]
    reasoning: str = ""

@dataclass
class ExecutionStep:
    """Rappresenta un passo nell'esecuzione"""
    tool_call: ToolCall
    result: Any
    timestamp: datetime
    success: bool
    error: Optional[str] = None

class ConversationMemory:
    """Gestisce la memoria della conversazione"""

    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Aggiungi messaggio alla storia"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Mantieni solo ultimi N messaggi
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def get_messages(self) -> List[Dict]:
        """Ottieni messaggi per LLM"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]

class WebScraperAgent:
    """
    Agente principale che orchestra i tools.

    Capabilities:
    - Analizza query utente
    - Genera piano di azioni (Chain of Thought)
    - Esegue tools in sequenza
    - Sintetizza risultati
    - Mantiene memoria conversazione
    """

    def __init__(
        self,
        llm_model: str = "llama3.2",
        llm_provider: str = "ollama",
        config: Dict = None
    ):
        self.config = config or {}

        # Inizializza LLM
        self.llm = LLMInterface(llm_model, llm_provider)

        # Inizializza tool registry
        self.tools = ToolRegistry()

        # Inizializza web tools
        self.web_tools = WebTools(config, self.llm)

        # Registra tutti i tools
        self._register_tools()

        # Memoria conversazione
        self.memory = ConversationMemory()

        # Storia esecuzioni
        self.execution_history: List[Dict] = []

    def _register_tools(self):
        """Registra tutti i tools disponibili"""

        # search_web
        self.tools.register(
            name="search_web",
            function=self.web_tools.search_web,
            description="Search the web for information using a search engine",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20)"
                    }
                },
                "required": ["query"]
            }
        )

        # fetch_webpage
        self.tools.register(
            name="fetch_webpage",
            function=self.web_tools.fetch_webpage,
            description="Fetch and parse the content of a webpage",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        )

        # extract_structured_data
        self.tools.register(
            name="extract_structured_data",
            function=self.web_tools.extract_structured_data,
            description="Extract structured data from HTML using CSS selectors",
            parameters={
                "type": "object",
                "properties": {
                    "html": {"type": "string"},
                    "schema": {"type": "object"}
                },
                "required": ["html", "schema"]
            }
        )

        # summarize_content
        self.tools.register(
            name="summarize_content",
            function=self.web_tools.summarize_content,
            description="Summarize long text using LLM",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_length": {"type": "integer"}
                },
                "required": ["text"]
            }
        )

        # compare_sources
        self.tools.register(
            name="compare_sources",
            function=self.web_tools.compare_sources,
            description="Compare information from multiple sources",
            parameters={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "topic": {"type": "string"}
                },
                "required": ["sources"]
            }
        )

    def process_query(self, query: str) -> str:
        """
        Pipeline principale per processare una query.

        Steps:
        1. Analizza query
        2. Genera piano (Chain of Thought)
        3. Esegue piano
        4. Sintetizza risultati

        Args:
            query: Query dell'utente

        Returns:
            Risposta finale
        """
        print(f"🤔 Analyzing query: {query}")

        # Aggiungi query alla memoria
        self.memory.add_message("user", query)

        # 1. Genera piano
        plan = self._generate_plan(query)
        print(f"📋 Plan generated with {len(plan)} steps")

        # 2. Esegui piano
        execution_results = self._execute_plan(plan)
        print(f"✓ Executed {len(execution_results)} steps")

        # 3. Sintetizza risultati
        final_response = self._synthesize_results(query, execution_results)

        # Salva nella memoria
        self.memory.add_message("assistant", final_response)

        # Salva nell'history
        self.execution_history.append({
            "query": query,
            "plan": [
                {"tool": tc.tool, "params": tc.parameters, "reasoning": tc.reasoning}
                for tc in plan
            ],
            "results": execution_results,
            "response": final_response,
            "timestamp": datetime.now().isoformat()
        })

        return final_response

    def _generate_plan(self, query: str) -> List[ToolCall]:
        """
        Genera un piano di azioni usando Chain of Thought.

        L'LLM analizza la query e decide quali tools usare e in che ordine.
        """
        # Implementazione nel prossimo step
        pass

    def _execute_plan(self, plan: List[ToolCall]) -> List[ExecutionStep]:
        """
        Esegue il piano step-by-step.
        """
        # Implementazione nel prossimo step
        pass

    def _synthesize_results(
        self,
        query: str,
        results: List[ExecutionStep]
    ) -> str:
        """
        Combina i risultati in una risposta coerente.
        """
        # Implementazione nel prossimo step
        pass
```

---

### Step 10: Implementare Chain of Thought
**Tempo stimato**: 2 ore
**Priorità**: CRITICA
**Dipendenze**: Step 9

#### Implementazione

```python
# agent.py (continuazione di WebScraperAgent)

def _generate_plan(self, query: str) -> List[ToolCall]:
    """
    Genera piano usando Chain of Thought.

    L'LLM ragiona step-by-step su come rispondere alla query,
    decidendo quali tools usare e in che ordine.
    """
    # Crea prompt con descrizione tools
    tools_desc = self.tools.get_tool_descriptions()

    cot_prompt = f"""
You are a web research assistant. A user has asked you:

"{query}"

Available tools:
{tools_desc}

Think step-by-step about how to answer this query:

1. What is the user asking for?
2. What information do you need to gather?
3. Which tools should you use?
4. In what order should you use them?
5. What parameters do you need for each tool?

Generate a plan as a JSON array of tool calls:

[
  {{
    "tool": "tool_name",
    "parameters": {{"param1": "value1"}},
    "reasoning": "why use this tool"
  }},
  ...
]

IMPORTANT:
- Be efficient: use minimum number of tools needed
- Be specific: provide exact parameters
- Be logical: order tools correctly (e.g., search before fetch)

Plan:
"""

    # Genera piano
    response = self.llm.generate(cot_prompt)

    # Parse JSON
    try:
        # Estrai JSON dal response
        start = response.find('[')
        end = response.rfind(']') + 1
        plan_json = json.loads(response[start:end])

        # Converti in ToolCall objects
        plan = []
        for step in plan_json:
            tool_call = ToolCall(
                tool=step["tool"],
                parameters=step.get("parameters", {}),
                reasoning=step.get("reasoning", "")
            )
            plan.append(tool_call)

        # Valida che i tools esistano
        for tc in plan:
            if tc.tool not in self.tools.list_tools():
                print(f"⚠️  Warning: Unknown tool '{tc.tool}', skipping")
                plan.remove(tc)

        return plan

    except Exception as e:
        print(f"❌ Error parsing plan: {e}")
        # Fallback: piano di default basato su euristiche
        return self._generate_fallback_plan(query)

def _generate_fallback_plan(self, query: str) -> List[ToolCall]:
    """
    Piano di fallback usando euristiche semplici.
    """
    plan = []

    # Se contiene "cerca", "trova" → search_web
    if any(word in query.lower() for word in ["cerca", "trova", "search", "find"]):
        plan.append(ToolCall(
            tool="search_web",
            parameters={"query": query, "num_results": 5},
            reasoning="User wants to search for information"
        ))

    # Se contiene URL → fetch_webpage
    import re
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', query)
    for url in urls:
        plan.append(ToolCall(
            tool="fetch_webpage",
            parameters={"url": url},
            reasoning=f"User provided URL: {url}"
        ))

    return plan
```

#### Esecuzione Piano

```python
def _execute_plan(self, plan: List[ToolCall]) -> List[ExecutionStep]:
    """
    Esegue piano step-by-step con error handling.
    """
    results = []

    for i, tool_call in enumerate(plan):
        print(f"⚙️  Step {i+1}/{len(plan)}: {tool_call.tool}")
        print(f"   Reasoning: {tool_call.reasoning}")

        try:
            # Esegui tool
            result = self.tools.call_tool(
                tool_call.tool,
                **tool_call.parameters
            )

            # Crea execution step
            step = ExecutionStep(
                tool_call=tool_call,
                result=result,
                timestamp=datetime.now(),
                success=True
            )
            results.append(step)

            print(f"   ✓ Success")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

            # Salva errore
            step = ExecutionStep(
                tool_call=tool_call,
                result=None,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
            results.append(step)

            # Continua con prossimo step
            continue

    return results
```

#### Sintesi Risultati

```python
def _synthesize_results(
    self,
    query: str,
    results: List[ExecutionStep]
) -> str:
    """
    Sintetizza risultati in risposta coerente.
    """
    # Prepara sommario dei risultati
    results_summary = []

    for i, step in enumerate(results):
        if step.success:
            # Formatta risultato in modo leggibile
            result_str = self._format_result(step.tool_call.tool, step.result)
            results_summary.append(f"Step {i+1} ({step.tool_call.tool}):\n{result_str}")
        else:
            results_summary.append(f"Step {i+1} ({step.tool_call.tool}): FAILED - {step.error}")

    results_text = "\n\n".join(results_summary)

    # Crea prompt per sintesi finale
    synthesis_prompt = f"""
User query: "{query}"

I executed the following steps and got these results:

{results_text}

Based on these results, provide a comprehensive answer to the user's query.

Requirements:
- Answer in Italian
- Be clear and concise
- Cite sources when relevant (include URLs)
- If some steps failed, acknowledge limitations
- Structure the response well

Answer:
"""

    # Genera risposta finale
    final_answer = self.llm.generate(synthesis_prompt)

    return final_answer

def _format_result(self, tool: str, result: Any) -> str:
    """Formatta risultato tool per display"""

    if tool == "search_web":
        # Format search results
        formatted = []
        for r in result[:5]:  # Top 5
            formatted.append(f"- {r['title']}\n  {r['url']}\n  {r['snippet'][:100]}...")
        return "\n".join(formatted)

    elif tool == "fetch_webpage":
        return f"Title: {result['title']}\nContent: {result['content'][:500]}..."

    elif tool == "summarize_content":
        return result

    elif tool == "compare_sources":
        return f"Consensus: {result.get('consensus', 'N/A')}\nDifferences: {result.get('differences', 'N/A')}"

    else:
        return str(result)[:500]
```

---

### Step 11: Error Handling e Retry
**Tempo stimato**: 1 ora
**Priorità**: ALTA
**Dipendenze**: Step 10

Già implementato nel codice precedente con:
- `@retry` decorator per network requests
- Try-except blocks nell'esecuzione plan
- Fallback plans
- Error tracking in ExecutionStep

---

### Step 12: Caching System
**Tempo stimato**: 1.5 ore
**Priorità**: MEDIA
**Dipendenze**: Step 4-8

#### Implementazione

```python
# cache.py

import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional

class ResultCache:
    """
    Sistema di caching per risultati tools.
    Evita richieste duplicate e velocizza testing.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        ttl_seconds: int = 3600
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)

    def _generate_key(self, tool: str, params: dict) -> str:
        """Genera cache key da tool + params"""
        key_str = f"{tool}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, tool: str, params: dict) -> Optional[Any]:
        """Recupera da cache se esiste e non scaduto"""
        key = self._generate_key(tool, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        if not cache_file.exists():
            return None

        # Carica cache
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)

        # Check TTL
        cached_time = datetime.fromisoformat(cached['timestamp'])
        if datetime.now() - cached_time > self.ttl:
            return None  # Scaduto

        return cached['result']

    def set(self, tool: str, params: dict, result: Any):
        """Salva in cache"""
        key = self._generate_key(tool, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        cached = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'tool': tool,
            'params': params
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cached, f)

    def clear(self):
        """Svuota cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
```

#### Integrazione in WebTools

```python
# web_tools.py - modifica

class WebTools:
    def __init__(self, config: Dict = None, llm_interface = None, cache = None):
        self.config = config or {}
        self.llm = llm_interface
        self.cache = cache

    def search_web(self, query: str, num_results: int = 5, **kwargs):
        # Check cache
        if self.cache:
            cached = self.cache.get("search_web", {"query": query, "num_results": num_results})
            if cached:
                print("   📦 Using cached result")
                return cached

        # Esegui ricerca
        result = self._do_search_web(query, num_results, **kwargs)

        # Salva in cache
        if self.cache:
            self.cache.set("search_web", {"query": query, "num_results": num_results}, result)

        return result
```

---

### Step 13: CLI Interattiva
**Tempo stimato**: 1.5 ore
**Priorità**: ALTA
**Dipendenze**: Step 9-10

#### Implementazione

```python
# main.py

import argparse
from pathlib import Path
import yaml
from dotenv import load_dotenv

from agent import WebScraperAgent
from cache import ResultCache

def load_config(config_path: str = "config.yaml") -> dict:
    """Carica configurazione da YAML"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def print_banner():
    """Stampa banner iniziale"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🌐 AGENTE WEB SCRAPER INTELLIGENTE 🌐                ║
║                                                            ║
║  Cerca informazioni online, estrai dati e sintetizza!     ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_help():
    """Mostra comandi disponibili"""
    help_text = """
Comandi disponibili:
  query <domanda>      - Fai una domanda all'agente
  search <query>       - Cerca direttamente sul web
  fetch <url>          - Scarica una pagina specifica
  history              - Mostra cronologia query
  clear                - Pulisci schermo
  help                 - Mostra questo aiuto
  exit                 - Esci dal programma

Esempi:
  query Ultime notizie sull'intelligenza artificiale
  search Python web scraping tutorial
  fetch https://example.com
"""
    print(help_text)

def interactive_mode(agent: WebScraperAgent):
    """Modalità interattiva CLI"""

    print_banner()
    print_help()
    print()

    while True:
        try:
            # Input utente
            user_input = input("\n💬 Tu: ").strip()

            if not user_input:
                continue

            # Parse comando
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            # Gestisci comandi
            if command in ["exit", "quit", "q"]:
                print("👋 Ciao!")
                break

            elif command == "help":
                print_help()

            elif command == "clear":
                import os
                os.system('clear' if os.name != 'nt' else 'cls')
                print_banner()

            elif command == "history":
                print("\n📜 Cronologia query:")
                for i, item in enumerate(agent.execution_history[-10:]):
                    print(f"\n{i+1}. Query: {item['query']}")
                    print(f"   Timestamp: {item['timestamp']}")
                    print(f"   Tools usati: {len(item['plan'])}")

            elif command == "query":
                if not args:
                    print("❌ Specifica una query!")
                    continue

                print(f"\n🤖 Processsing query...")
                response = agent.process_query(args)
                print(f"\n🤖 Risposta:\n{response}")

            elif command == "search":
                if not args:
                    print("❌ Specifica una query di ricerca!")
                    continue

                results = agent.web_tools.search_web(args, num_results=5)
                print(f"\n🔍 Risultati ricerca ({len(results)}):")
                for i, r in enumerate(results, 1):
                    print(f"\n{i}. {r['title']}")
                    print(f"   {r['url']}")
                    print(f"   {r['snippet'][:150]}...")

            elif command == "fetch":
                if not args:
                    print("❌ Specifica un URL!")
                    continue

                page = agent.web_tools.fetch_webpage(args)
                print(f"\n📄 Pagina: {page['title']}")
                print(f"   URL: {page['url']}")
                print(f"   Content: {page['content'][:500]}...")

            else:
                # Default: tratta come query
                print(f"\n🤖 Processing query...")
                response = agent.process_query(user_input)
                print(f"\n🤖 Risposta:\n{response}")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Use 'exit' to quit.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Web Scraper Agent")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--model", default="llama3.2", help="LLM model")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("query", nargs="*", help="Single query mode")

    args = parser.parse_args()

    # Carica env
    load_dotenv()

    # Carica config
    config = load_config(args.config) if Path(args.config).exists() else {}

    # Setup cache
    cache = None if args.no_cache else ResultCache()

    # Inizializza agente
    print("🚀 Initializing agent...")
    agent = WebScraperAgent(
        llm_model=args.model,
        llm_provider=args.provider,
        config=config
    )

    # Inietta cache in web_tools
    if cache:
        agent.web_tools.cache = cache

    print("✓ Agent ready!")

    # Modalità single query o interattiva
    if args.query:
        query = " ".join(args.query)
        response = agent.process_query(query)
        print(response)
    else:
        interactive_mode(agent)

if __name__ == "__main__":
    main()
```

---

## FASE 4: TESTING & DOCS

### Step 14: Test Unitari Tools
**Tempo stimato**: 2 ore
**Priorità**: ALTA
**Dipendenze**: Step 4-8

```python
# tests/test_tools.py

import pytest
from web_tools import WebTools
from llm_interface import LLMInterface

@pytest.fixture
def web_tools():
    llm = LLMInterface("llama3.2", "ollama")
    return WebTools(llm_interface=llm)

def test_search_web_basic(web_tools):
    """Test basic web search"""
    results = web_tools.search_web("Python programming", num_results=3)

    assert len(results) <= 3
    assert all("url" in r for r in results)
    assert all("title" in r for r in results)
    assert all("snippet" in r for r in results)

def test_search_web_empty_query(web_tools):
    """Test search with empty query raises error"""
    with pytest.raises(ValueError):
        web_tools.search_web("")

def test_fetch_webpage_valid_url(web_tools):
    """Test fetching a valid webpage"""
    result = web_tools.fetch_webpage("https://example.com")

    assert "title" in result
    assert "content" in result
    assert "url" in result
    assert result["status_code"] == 200

def test_fetch_webpage_invalid_url(web_tools):
    """Test fetching invalid URL raises error"""
    with pytest.raises(ValueError):
        web_tools.fetch_webpage("not-a-url")

def test_extract_structured_data(web_tools):
    """Test structured data extraction"""
    html = """
    <div class="item">
        <span class="name">Item 1</span>
        <span class="price">$10</span>
    </div>
    <div class="item">
        <span class="name">Item 2</span>
        <span class="price">$20</span>
    </div>
    """

    schema = {
        "selector": ".item",
        "fields": {
            "name": ".name",
            "price": ".price"
        }
    }

    results = web_tools.extract_structured_data(html, schema)

    assert len(results) == 2
    assert results[0]["name"] == "Item 1"
    assert results[0]["price"] == "$10"

# ... altri test
```

---

### Step 15: Test Integrazione
**Tempo stimato**: 1.5 ore
**Priorità**: ALTA
**Dipendenze**: Step 9-13

```python
# tests/test_integration.py

import pytest
from agent import WebScraperAgent

@pytest.fixture
def agent():
    return WebScraperAgent(llm_model="llama3.2")

def test_simple_search_query(agent):
    """Test end-to-end simple search"""
    query = "Capitale della Francia"
    response = agent.process_query(query)

    assert "parigi" in response.lower() or "paris" in response.lower()
    assert len(agent.execution_history) == 1

def test_multi_step_query(agent):
    """Test query that requires multiple tools"""
    query = "Cerca ultime notizie Python e riassumile"
    response = agent.process_query(query)

    # Dovrebbe aver usato search + summarize
    last_execution = agent.execution_history[-1]
    tools_used = [step["tool"] for step in last_execution["plan"]]

    assert "search_web" in tools_used
    assert len(response) > 0

def test_error_recovery(agent):
    """Test that agent handles errors gracefully"""
    query = "Fetch page from invalid-url-12345"

    # Non dovrebbe crashare
    response = agent.process_query(query)
    assert response  # Qualche risposta, anche se parziale
```

---

### Step 16: Documentazione Aggiuntiva
**Tempo stimato**: 2 ore
**Priorità**: MEDIA
**Dipendenze**: Tutti gli step precedenti

Creare:
- `docs/TUTORIAL.md`: Tutorial passo-passo per usare l'agente
- `docs/ARCHITECTURE.md`: Architettura dettagliata del sistema
- `docs/EXAMPLES.md`: Esempi pratici e use cases

---

## ✅ Checklist Finale

Prima di considerare il progetto completo:

- [ ] Tutti i test passano
- [ ] Documentazione completa
- [ ] README aggiornato con esempi reali
- [ ] requirements.txt aggiornato
- [ ] config.yaml configurato correttamente
- [ ] CLI funzionante
- [ ] Caching implementato
- [ ] Error handling robusto
- [ ] Esempi funzionanti testati
- [ ] Code review e refactoring completati

---

## 🚀 Prossimi Passi (Estensioni)

Dopo completamento base:

1. **Web UI** con Streamlit/Gradio
2. **API REST** con FastAPI
3. **Async execution** per parallelizzare richieste
4. **Database** per persistenza (SQLite/PostgreSQL)
5. **Authentication** e multi-user support
6. **Monitoring** con Prometheus/Grafana
7. **Deploy** su cloud (Heroku/Railway/Fly.io)

---

**Tempo totale stimato**: 20-25 ore
**Difficoltà**: Intermedio
**Prerequisiti**: Python, HTTP, HTML basics, LLM concepts
