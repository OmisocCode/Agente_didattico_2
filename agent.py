"""
============================================================================
WEB SCRAPER AGENT - Agente Intelligente per Web Scraping
============================================================================

Questo modulo implementa l'agente principale che orchestra tutti i tools
per rispondere a query dell'utente.

L'agente:
1. Riceve query in linguaggio naturale
2. Analizza la query usando LLM
3. Genera un piano di azioni (Chain of Thought)
4. Esegue i tools necessari
5. Sintetizza i risultati in una risposta coerente

Architettura:
- WebScraperAgent: Classe principale dell'agente
- ConversationMemory: Gestisce storia conversazione
- ToolCall: Dataclass per rappresentare chiamata a tool
- ExecutionStep: Dataclass per rappresentare step eseguito

Chain of Thought:
L'agente usa "Chain of Thought" prompting per pianificare le azioni:
1. Analizza la query
2. Identifica obiettivo
3. Decide quali tools usare
4. Determina ordine di esecuzione
5. Genera piano strutturato

Esempio Flusso:
Query: "Cerca ultime notizie AI e riassumile"
→ Plan: [search_web, fetch_webpage, summarize_content]
→ Execute: Esegue tools in sequenza
→ Synthesize: Combina risultati in risposta finale

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

# Moduli del progetto
from llm_interface import LLMInterface
from tool_registry import ToolRegistry
from web_tools import WebTools
from cache import ResultCache

# Configurazione logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


# ----------------------------------------------------------------------------
# DATACLASSES - Strutture dati
# ----------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    Rappresenta una chiamata a un tool pianificata.

    Questo oggetto viene creato durante la fase di planning
    e contiene tutte le informazioni necessarie per eseguire il tool.

    Attributes:
        tool: Nome del tool da chiamare (es: "search_web")
        parameters: Dizionario con parametri per il tool
        reasoning: Spiegazione del perché usare questo tool (opzionale)
    """
    tool: str
    parameters: Dict[str, Any]
    reasoning: str = ""

    def __repr__(self) -> str:
        """Rappresentazione leggibile."""
        params_str = ", ".join(f"{k}={v}" for k, v in list(self.parameters.items())[:2])
        return f"ToolCall({self.tool}({params_str}...))"


@dataclass
class ExecutionStep:
    """
    Rappresenta un passo eseguito nel piano.

    Creato dopo l'esecuzione di un ToolCall, contiene il risultato
    e informazioni sull'esecuzione.

    Attributes:
        tool_call: Il ToolCall che è stato eseguito
        result: Risultato dell'esecuzione
        timestamp: Quando è stato eseguito
        success: Se l'esecuzione è riuscita
        error: Messaggio di errore se fallito (None se success)
        execution_time_ms: Tempo di esecuzione in millisecondi
    """
    tool_call: ToolCall
    result: Any
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def __repr__(self) -> str:
        """Rappresentazione leggibile."""
        status = "✓" if self.success else "✗"
        return f"ExecutionStep({status} {self.tool_call.tool}, {self.execution_time_ms:.0f}ms)"


# ----------------------------------------------------------------------------
# CONVERSATION MEMORY
# ----------------------------------------------------------------------------
class ConversationMemory:
    """
    Gestisce la memoria della conversazione.

    Mantiene uno storico dei messaggi scambiati tra utente e agente,
    permettendo di mantenere contesto conversazionale.

    Features:
    - Memorizza messaggi user/assistant
    - Limita dimensione storia (evita context overflow)
    - Formato compatibile con LLM chat API
    - Timestamp per ogni messaggio

    Esempio:
        memory = ConversationMemory(max_history=10)
        memory.add_message("user", "Ciao, come stai?")
        memory.add_message("assistant", "Bene grazie! Come posso aiutarti?")
        messages = memory.get_messages()  # Per LLM
    """

    def __init__(self, max_history: int = 10):
        """
        Inizializza memory.

        Args:
            max_history: Numero massimo di coppie user-assistant da mantenere
                        (default 10 = ultimi 10 scambi = 20 messaggi)
        """
        logger.info(f"Initializing ConversationMemory (max_history={max_history})")

        self.max_history = max_history
        self.history: List[Dict] = []

        logger.success("✓ ConversationMemory initialized")

    def add_message(self, role: str, content: str):
        """
        Aggiungi messaggio alla storia.

        Args:
            role: "user" o "assistant"
            content: Contenuto del messaggio
        """
        logger.debug(f"Adding message: {role} - {content[:50]}...")

        # Crea messaggio con timestamp
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        # Aggiungi a storia
        self.history.append(message)

        # Mantieni solo ultimi N*2 messaggi (N coppie user-assistant)
        max_messages = self.max_history * 2

        if len(self.history) > max_messages:
            removed = len(self.history) - max_messages
            self.history = self.history[-max_messages:]
            logger.debug(f"Trimmed history: removed {removed} old messages")

        logger.debug(f"History size: {len(self.history)} messages")

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Ottieni messaggi in formato per LLM.

        Returns:
            Lista di dict {"role": "user/assistant", "content": "..."}
            (senza timestamp, solo role e content)
        """
        # Ritorna solo role e content (LLM non ha bisogno di timestamp)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]

    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """
        Ottieni ultimi N messaggi.

        Args:
            n: Numero di messaggi da ritornare

        Returns:
            Ultimi N messaggi in formato LLM
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history[-n:]
        ]

    def clear(self):
        """Svuota la memoria."""
        logger.info("Clearing conversation memory")
        self.history.clear()

    def __len__(self) -> int:
        """Numero di messaggi nella storia."""
        return len(self.history)

    def __repr__(self) -> str:
        """Rappresentazione stringa."""
        return f"ConversationMemory(messages={len(self.history)})"


# ----------------------------------------------------------------------------
# WEB SCRAPER AGENT - Agente Principale
# ----------------------------------------------------------------------------
class WebScraperAgent:
    """
    Agente principale per web scraping intelligente.

    Questo è il cuore del sistema. L'agente:
    1. Riceve query in linguaggio naturale
    2. Usa LLM per capire l'intento
    3. Genera un piano di azioni (Chain of Thought)
    4. Esegue i tools necessari
    5. Sintetizza i risultati in risposta coerente

    L'agente ha accesso a tutti i tools implementati in WebTools:
    - search_web: Ricerca sul web
    - fetch_webpage: Download pagine
    - extract_structured_data: Estrazione dati
    - summarize_content: Riassunti
    - compare_sources: Confronto fonti

    Esempio:
        agent = WebScraperAgent(
            llm_model="llama3.2",
            llm_provider="ollama",
            config=config
        )

        response = agent.process_query("Cerca notizie Python e riassumile")
        print(response)
    """

    def __init__(
        self,
        llm_model: str = "llama3.2",
        llm_provider: str = "ollama",
        config: Dict = None,
        enable_cache: bool = True
    ):
        """
        Inizializza l'agente.

        Args:
            llm_model: Modello LLM da usare
            llm_provider: Provider LLM ("ollama", "openai", "groq")
            config: Configurazione generale (da config.yaml)
            enable_cache: Se True, abilita caching risultati
        """
        logger.info("="*70)
        logger.info("Initializing WebScraperAgent")
        logger.info("="*70)

        # Salva configurazione
        self.config = config or {}

        # ====================================================================
        # INIZIALIZZA LLM
        # ====================================================================
        logger.info(f"Setting up LLM: {llm_provider}/{llm_model}")

        try:
            self.llm = LLMInterface(
                model=llm_model,
                provider=llm_provider
            )
            logger.success(f"✓ LLM ready: {llm_provider}/{llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        # ====================================================================
        # INIZIALIZZA TOOL REGISTRY
        # ====================================================================
        logger.info("Setting up Tool Registry")

        self.tools = ToolRegistry()
        logger.success("✓ Tool Registry created")

        # ====================================================================
        # INIZIALIZZA WEB TOOLS
        # ====================================================================
        logger.info("Setting up Web Tools")

        # Configurazione tools
        tools_config = self.config.get("tools", {})

        self.web_tools = WebTools(
            config=tools_config,
            llm_interface=self.llm
        )
        logger.success("✓ Web Tools initialized")

        # ====================================================================
        # INIZIALIZZA CACHE (se abilitato)
        # ====================================================================
        self.cache = None

        if enable_cache:
            logger.info("Setting up Result Cache")

            cache_config = self.config.get("cache", {})

            try:
                self.cache = ResultCache(
                    cache_dir=cache_config.get("directory", ".cache"),
                    ttl_seconds=self.config.get("agent", {}).get("cache_ttl_seconds", 3600),
                    max_size_mb=cache_config.get("max_size_mb", 500),
                    auto_cleanup=cache_config.get("auto_cleanup", True)
                )

                # Inietta cache nei web tools
                self.web_tools.cache = self.cache

                logger.success(f"✓ Cache enabled: {self.cache}")
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}")
                logger.warning("Continuing without cache")
        else:
            logger.info("Cache disabled")

        # ====================================================================
        # REGISTRA TUTTI I TOOLS
        # ====================================================================
        logger.info("Registering tools in registry...")

        self._register_tools()

        logger.success(f"✓ Registered {len(self.tools)} tools")

        # ====================================================================
        # INIZIALIZZA MEMORIA CONVERSAZIONE
        # ====================================================================
        max_history = self.config.get("agent", {}).get("max_conversation_history", 10)

        self.memory = ConversationMemory(max_history=max_history)
        logger.success(f"✓ Conversation Memory initialized (max_history={max_history})")

        # ====================================================================
        # STORAGE PER ESECUZIONI
        # ====================================================================
        self.execution_history: List[Dict] = []
        logger.debug("Execution history storage initialized")

        # ====================================================================
        # LIMITI E CONFIGURAZIONE
        # ====================================================================
        self.max_tools_per_query = self.config.get("agent", {}).get("max_tools_per_query", 5)
        self.verbose = self.config.get("agent", {}).get("verbose", True)

        logger.debug(f"Max tools per query: {self.max_tools_per_query}")
        logger.debug(f"Verbose mode: {self.verbose}")

        # ====================================================================
        # AGENT READY
        # ====================================================================
        logger.info("="*70)
        logger.success("✓ WebScraperAgent initialized and ready!")
        logger.info("="*70)
        logger.info("")

    def _register_tools(self):
        """
        Registra tutti i tools nel ToolRegistry.

        Ogni tool di WebTools viene registrato con nome, descrizione
        e schema parametri per il function calling.
        """
        logger.debug("Registering tools...")

        # ====================================================================
        # TOOL 1: search_web
        # ====================================================================
        self.tools.register(
            name="search_web",
            function=self.web_tools.search_web,
            description="Search the web for information using a search engine. Use this when you need to find current information, news, articles, or general knowledge from the internet.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'Python web scraping tutorial', 'latest AI news')"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 20)"
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code for localized results (default 'it-it')"
                    }
                },
                "required": ["query"]
            }
        )
        logger.debug("✓ Registered: search_web")

        # ====================================================================
        # TOOL 2: fetch_webpage
        # ====================================================================
        self.tools.register(
            name="fetch_webpage",
            function=self.web_tools.fetch_webpage,
            description="Fetch and parse the content of a specific webpage. Use this when you have a URL and need to extract its content, including text, metadata, and links.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to fetch (must be a valid URL)"
                    },
                    "extract_main_content": {
                        "type": "boolean",
                        "description": "If true, extracts only main content (removes ads, navigation, etc.)"
                    },
                    "extract_metadata": {
                        "type": "boolean",
                        "description": "If true, extracts metadata (author, date, description, etc.)"
                    },
                    "extract_links": {
                        "type": "boolean",
                        "description": "If true, extracts all links from the page"
                    }
                },
                "required": ["url"]
            }
        )
        logger.debug("✓ Registered: fetch_webpage")

        # ====================================================================
        # TOOL 3: extract_structured_data
        # ====================================================================
        self.tools.register(
            name="extract_structured_data",
            function=self.web_tools.extract_structured_data,
            description="Extract structured data from HTML using CSS selectors. Use this when you need to extract specific data like product lists, tables, or repeated elements from a webpage.",
            parameters={
                "type": "object",
                "properties": {
                    "html": {
                        "type": "string",
                        "description": "The HTML content to extract data from"
                    },
                    "schema": {
                        "type": "object",
                        "description": "Extraction schema with CSS selectors"
                    }
                },
                "required": ["html", "schema"]
            }
        )
        logger.debug("✓ Registered: extract_structured_data")

        # ====================================================================
        # TOOL 4: summarize_content
        # ====================================================================
        self.tools.register(
            name="summarize_content",
            function=self.web_tools.summarize_content,
            description="Summarize long text content using AI. Use this when you have a long article or text that needs to be condensed into key points.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to summarize"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum length of summary in words (default 500)"
                    },
                    "style": {
                        "type": "string",
                        "description": "Summary style: 'concise', 'detailed', or 'bullet_points'"
                    }
                },
                "required": ["text"]
            }
        )
        logger.debug("✓ Registered: summarize_content")

        # ====================================================================
        # TOOL 5: compare_sources
        # ====================================================================
        self.tools.register(
            name="compare_sources",
            function=self.web_tools.compare_sources,
            description="Compare information from multiple sources. Use this when you need to cross-reference information, find consensus, or identify differences between sources.",
            parameters={
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of URLs or text content to compare"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional topic to focus the comparison on"
                    }
                },
                "required": ["sources"]
            }
        )
        logger.debug("✓ Registered: compare_sources")

    # ========================================================================
    # PUBLIC API - Metodi principali
    # ========================================================================

    def process_query(self, query: str) -> str:
        """
        Processa una query utente end-to-end.

        Questo è il metodo principale che:
        1. Analizza la query
        2. Genera un piano (Chain of Thought)
        3. Esegue il piano
        4. Sintetizza i risultati
        5. Ritorna risposta finale

        Args:
            query: Query in linguaggio naturale

        Returns:
            Risposta dell'agente come stringa

        Example:
            >>> agent = WebScraperAgent()
            >>> response = agent.process_query("Cerca notizie Python")
            >>> print(response)
        """
        logger.info("\n" + "="*70)
        logger.info(f"🔍 Processing Query")
        logger.info("="*70)
        logger.info(f"Query: {query}")
        logger.info("")

        # Timestamp inizio
        start_time = datetime.now()

        # ====================================================================
        # STEP 1: AGGIUNGI QUERY ALLA MEMORIA
        # ====================================================================
        self.memory.add_message("user", query)

        # ====================================================================
        # STEP 2: GENERA PIANO (Chain of Thought)
        # ====================================================================
        logger.info("📋 STEP 1: Generating execution plan...")
        logger.info("-" * 70)

        try:
            plan = self._generate_plan(query)

            if not plan:
                logger.warning("No plan generated, returning direct response")
                return self._generate_direct_response(query)

            logger.success(f"✓ Plan generated with {len(plan)} steps")
            logger.info("")

            # Log piano
            if self.verbose:
                logger.info("Plan:")
                for i, tool_call in enumerate(plan, 1):
                    logger.info(f"  {i}. {tool_call.tool}({', '.join(f'{k}={v}' for k,v in list(tool_call.parameters.items())[:2])}...)")
                    if tool_call.reasoning:
                        logger.info(f"     → {tool_call.reasoning}")
                logger.info("")

        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            logger.exception(e)
            return f"Mi dispiace, ho avuto un errore nel pianificare la risposta: {str(e)}"

        # ====================================================================
        # STEP 3: ESEGUI PIANO
        # ====================================================================
        logger.info("⚙️  STEP 2: Executing plan...")
        logger.info("-" * 70)

        try:
            execution_results = self._execute_plan(plan)

            successful = sum(1 for r in execution_results if r.success)
            logger.success(f"✓ Executed {len(execution_results)} steps ({successful} successful)")
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to execute plan: {e}")
            logger.exception(e)
            return f"Mi dispiace, ho avuto un errore nell'esecuzione: {str(e)}"

        # ====================================================================
        # STEP 4: SINTETIZZA RISULTATI
        # ====================================================================
        logger.info("🎯 STEP 3: Synthesizing results...")
        logger.info("-" * 70)

        try:
            final_response = self._synthesize_results(query, execution_results)

            logger.success(f"✓ Response generated ({len(final_response)} chars)")
            logger.info("")

        except Exception as e:
            logger.error(f"Failed to synthesize results: {e}")
            logger.exception(e)
            return f"Mi dispiace, ho avuto un errore nel creare la risposta: {str(e)}"

        # ====================================================================
        # STEP 5: SALVA NELLA MEMORIA E STORIA
        # ====================================================================
        # Aggiungi risposta alla memoria conversazione
        self.memory.add_message("assistant", final_response)

        # Tempo totale
        execution_time = (datetime.now() - start_time).total_seconds()

        # Salva nell'execution history
        self.execution_history.append({
            "query": query,
            "plan": [
                {
                    "tool": tc.tool,
                    "parameters": tc.parameters,
                    "reasoning": tc.reasoning
                }
                for tc in plan
            ],
            "execution_results": [
                {
                    "tool": step.tool_call.tool,
                    "success": step.success,
                    "error": step.error,
                    "execution_time_ms": step.execution_time_ms
                }
                for step in execution_results
            ],
            "response": final_response,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time
        })

        # ====================================================================
        # DONE
        # ====================================================================
        logger.info("="*70)
        logger.success(f"✓ Query processed successfully in {execution_time:.2f}s")
        logger.info("="*70)
        logger.info("")

        return final_response

    # ========================================================================
    # CHAIN OF THOUGHT - Planning
    # ========================================================================

    def _generate_plan(self, query: str) -> List[ToolCall]:
        """
        Genera piano di azioni usando Chain of Thought.

        Usa l'LLM per analizzare la query e decidere:
        1. Quale obiettivo ha l'utente
        2. Quali informazioni servono
        3. Quali tools usare
        4. In che ordine usarli
        5. Con quali parametri

        Questo è il cuore del sistema "intelligente" dell'agente.

        Args:
            query: Query dell'utente

        Returns:
            Lista di ToolCall da eseguire in ordine
        """
        logger.debug("Generating plan with Chain of Thought...")

        # ====================================================================
        # PREPARA DESCRIZIONI TOOLS
        # ====================================================================
        # Ottieni lista tools in formato function calling
        tools_for_fc = self.tools.get_tools_for_function_calling()

        logger.debug(f"Available tools: {len(tools_for_fc)}")

        # ====================================================================
        # CREA PROMPT CHAIN OF THOUGHT
        # ====================================================================
        cot_prompt = f"""You are an intelligent web research assistant. A user has asked you:

"{query}"

Available tools:
{self.tools.get_tool_descriptions()}

Think step-by-step about how to answer this query (Chain of Thought):

1. What is the user asking for?
2. What information do I need to gather?
3. Which tools should I use?
4. In what order should I use them?
5. What parameters do I need for each tool?

Generate an execution plan as a JSON array of tool calls. Be efficient: use the minimum number of tools needed.

IMPORTANT:
- Be specific with parameters
- Order tools logically (e.g., search before fetch, fetch before summarize)
- Don't use more than {self.max_tools_per_query} tools
- Respond ONLY with a JSON array, no other text

Example format:
[
  {{
    "tool": "search_web",
    "parameters": {{"query": "Python tutorials", "num_results": 5}},
    "reasoning": "Need to find Python learning resources"
  }},
  {{
    "tool": "fetch_webpage",
    "parameters": {{"url": "https://example.com"}},
    "reasoning": "Get detailed content from top result"
  }}
]

Your plan (JSON array only):"""

        logger.debug(f"Chain of Thought prompt: {len(cot_prompt)} chars")

        # ====================================================================
        # GENERA PIANO CON LLM
        # ====================================================================
        try:
            logger.info("Calling LLM for planning...")

            response = self.llm.generate(
                cot_prompt,
                temperature=0.3,  # Bassa temperatura per output deterministico
                max_tokens=1000
            )

            logger.success("✓ LLM response received")
            logger.debug(f"Response preview: {response[:200]}...")

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback: genera piano semplice con euristiche
            logger.warning("Falling back to heuristic planning")
            return self._generate_fallback_plan(query)

        # ====================================================================
        # PARSE PIANO JSON
        # ====================================================================
        try:
            # Estrai JSON dalla risposta
            start = response.find('[')
            end = response.rfind(']') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON array found in response")

            json_str = response[start:end]
            logger.debug(f"Extracted JSON: {json_str[:500]}...")

            # Parse JSON
            plan_json = json.loads(json_str)

            # Valida che sia una lista
            if not isinstance(plan_json, list):
                raise ValueError("Plan is not a JSON array")

            logger.success(f"✓ Parsed {len(plan_json)} tool calls from plan")

        except Exception as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            logger.debug(f"Response was: {response}")
            # Fallback
            logger.warning("Falling back to heuristic planning")
            return self._generate_fallback_plan(query)

        # ====================================================================
        # CONVERTI IN TOOLCALL OBJECTS
        # ====================================================================
        plan = []

        for i, step in enumerate(plan_json):
            try:
                # Valida campi
                if "tool" not in step:
                    logger.warning(f"Step {i} missing 'tool' field, skipping")
                    continue

                tool_name = step["tool"]
                parameters = step.get("parameters", {})
                reasoning = step.get("reasoning", "")

                # Valida che il tool esista
                if tool_name not in self.tools.list_tools():
                    logger.warning(f"Unknown tool '{tool_name}', skipping")
                    continue

                # Crea ToolCall
                tool_call = ToolCall(
                    tool=tool_name,
                    parameters=parameters,
                    reasoning=reasoning
                )

                plan.append(tool_call)

                logger.debug(f"Added to plan: {tool_call}")

            except Exception as e:
                logger.error(f"Failed to process step {i}: {e}")
                continue

        # ====================================================================
        # LIMITA NUMERO TOOLS
        # ====================================================================
        if len(plan) > self.max_tools_per_query:
            logger.warning(f"Plan has {len(plan)} tools, limiting to {self.max_tools_per_query}")
            plan = plan[:self.max_tools_per_query]

        return plan

    def _generate_fallback_plan(self, query: str) -> List[ToolCall]:
        """
        Genera piano di fallback usando euristiche semplici.

        Usato quando il Chain of Thought planning fallisce.
        Usa pattern matching sulla query per decidere tool.

        Args:
            query: Query utente

        Returns:
            Piano semplice basato su euristiche
        """
        logger.info("Generating fallback plan with heuristics...")

        query_lower = query.lower()
        plan = []

        # Euristica: se contiene "cerca", "trova", "search" → search_web
        if any(word in query_lower for word in ["cerca", "trova", "search", "ricerca"]):
            logger.debug("Query seems to be a search request")

            plan.append(ToolCall(
                tool="search_web",
                parameters={"query": query, "num_results": 5},
                reasoning="User wants to search for information"
            ))

        # Euristica: se contiene URL → fetch_webpage
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', query)

        for url in urls:
            logger.debug(f"Found URL in query: {url}")

            plan.append(ToolCall(
                tool="fetch_webpage",
                parameters={"url": url},
                reasoning=f"User provided URL: {url}"
            ))

        # Se nessun tool identificato, usa search come default
        if not plan:
            logger.debug("No specific pattern matched, defaulting to search")

            plan.append(ToolCall(
                tool="search_web",
                parameters={"query": query, "num_results": 3},
                reasoning="Default: search for query"
            ))

        logger.info(f"Fallback plan generated with {len(plan)} tools")

        return plan

    def _generate_direct_response(self, query: str) -> str:
        """
        Genera risposta diretta senza usare tools.

        Usato quando non serve usare tools (es: domande generiche).

        Args:
            query: Query utente

        Returns:
            Risposta diretta dall'LLM
        """
        logger.info("Generating direct response without tools...")

        try:
            response = self.llm.generate(
                query,
                system="You are a helpful assistant. Answer the user's question directly and concisely."
            )

            return response

        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            return f"Mi dispiace, non sono riuscito a elaborare una risposta."

    # ========================================================================
    # EXECUTION - Esecuzione Piano
    # ========================================================================

    def _execute_plan(self, plan: List[ToolCall]) -> List[ExecutionStep]:
        """
        Esegue piano step-by-step.

        Per ogni ToolCall nel piano:
        1. Valida parametri
        2. Esegue tool
        3. Gestisce errori
        4. Memorizza risultato
        5. Continua anche se un tool fallisce

        Args:
            plan: Lista di ToolCall da eseguire

        Returns:
            Lista di ExecutionStep con risultati
        """
        logger.debug(f"Executing plan with {len(plan)} steps...")

        results = []

        for i, tool_call in enumerate(plan, 1):
            logger.info(f"Step {i}/{len(plan)}: {tool_call.tool}")

            if self.verbose and tool_call.reasoning:
                logger.info(f"  Reasoning: {tool_call.reasoning}")

            # Timestamp inizio
            step_start = datetime.now()

            try:
                # ============================================================
                # ESEGUI TOOL
                # ============================================================
                logger.debug(f"  Calling tool with parameters: {tool_call.parameters}")

                result = self.tools.call_tool(
                    tool_call.tool,
                    **tool_call.parameters
                )

                # Calcola tempo esecuzione
                execution_time = (datetime.now() - step_start).total_seconds() * 1000  # ms

                # Crea ExecutionStep di successo
                step = ExecutionStep(
                    tool_call=tool_call,
                    result=result,
                    timestamp=datetime.now(),
                    success=True,
                    execution_time_ms=execution_time
                )

                results.append(step)

                logger.success(f"  ✓ Success ({execution_time:.0f}ms)")

                # Log preview risultato
                if self.verbose:
                    self._log_result_preview(tool_call.tool, result)

            except Exception as e:
                # ============================================================
                # ERRORE ESECUZIONE
                # ============================================================
                logger.error(f"  ✗ Error: {str(e)}")

                # Calcola tempo esecuzione
                execution_time = (datetime.now() - step_start).total_seconds() * 1000

                # Crea ExecutionStep di errore
                step = ExecutionStep(
                    tool_call=tool_call,
                    result=None,
                    timestamp=datetime.now(),
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time
                )

                results.append(step)

                # Continua con prossimo step (non bloccare tutto il piano)
                logger.info("  Continuing with next step...")
                continue

            logger.info("")  # Linea vuota tra steps

        return results

    def _log_result_preview(self, tool: str, result: Any):
        """
        Log preview del risultato di un tool.

        Args:
            tool: Nome del tool
            result: Risultato del tool
        """
        try:
            if tool == "search_web":
                if isinstance(result, list) and result:
                    logger.info(f"  Found {len(result)} results")
                    logger.debug(f"  First result: {result[0].get('title', 'N/A')}")

            elif tool == "fetch_webpage":
                if isinstance(result, dict):
                    title = result.get('title', 'N/A')
                    content_len = len(result.get('content', ''))
                    logger.info(f"  Fetched: {title}")
                    logger.debug(f"  Content: {content_len} chars")

            elif tool == "summarize_content":
                if isinstance(result, str):
                    logger.info(f"  Summary: {result[:100]}...")

            elif tool == "compare_sources":
                if isinstance(result, dict):
                    logger.info(f"  Compared {result.get('sources_analyzed', 0)} sources")

            elif tool == "extract_structured_data":
                if isinstance(result, list):
                    logger.info(f"  Extracted {len(result)} items")

        except Exception as e:
            logger.debug(f"Failed to log result preview: {e}")

    # ========================================================================
    # SYNTHESIS - Sintesi Risultati
    # ========================================================================

    def _synthesize_results(
        self,
        query: str,
        results: List[ExecutionStep]
    ) -> str:
        """
        Sintetizza risultati dei tools in risposta coerente.

        Usa LLM per combinare i risultati di tutti i tools
        in una risposta finale che risponde alla query originale.

        Args:
            query: Query originale dell'utente
            results: Risultati dell'esecuzione dei tools

        Returns:
            Risposta finale come stringa
        """
        logger.debug("Synthesizing results into final response...")

        # ====================================================================
        # PREPARA SOMMARIO RISULTATI
        # ====================================================================
        results_summary = []

        for i, step in enumerate(results, 1):
            if step.success:
                # Formatta risultato in modo leggibile
                result_str = self._format_result_for_synthesis(
                    step.tool_call.tool,
                    step.result
                )

                results_summary.append(
                    f"Step {i} - {step.tool_call.tool}:\n{result_str}"
                )
            else:
                # Tool fallito
                results_summary.append(
                    f"Step {i} - {step.tool_call.tool}: FAILED ({step.error})"
                )

        results_text = "\n\n" + "---\n\n".join(results_summary)

        logger.debug(f"Results summary: {len(results_text)} chars")

        # ====================================================================
        # CREA PROMPT PER SINTESI
        # ====================================================================
        synthesis_prompt = f"""User query: "{query}"

I executed several tools and got these results:

{results_text}

Based on these results, provide a comprehensive answer to the user's query.

Requirements:
- Answer in Italian
- Be clear, concise, and well-structured
- Cite sources when relevant (include URLs)
- If some steps failed, acknowledge limitations
- If results are empty or insufficient, explain why

Provide your answer:"""

        logger.debug(f"Synthesis prompt: {len(synthesis_prompt)} chars")

        # ====================================================================
        # GENERA RISPOSTA FINALE
        # ====================================================================
        try:
            logger.info("Calling LLM for final synthesis...")

            final_answer = self.llm.generate(
                synthesis_prompt,
                temperature=0.5  # Temperatura moderata per bilanciare creatività e precisione
            )

            logger.success(f"✓ Final answer generated")

            return final_answer.strip()

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

            # Fallback: crea risposta semplice manuale
            return self._create_fallback_response(query, results)

    def _format_result_for_synthesis(self, tool: str, result: Any) -> str:
        """
        Formatta risultato tool per sintesi LLM.

        Args:
            tool: Nome del tool
            result: Risultato del tool

        Returns:
            Stringa formattata per il prompt di sintesi
        """
        try:
            if tool == "search_web":
                # Format search results
                if isinstance(result, list):
                    formatted = []
                    for i, r in enumerate(result[:5], 1):
                        formatted.append(
                            f"{i}. {r.get('title', 'N/A')}\n"
                            f"   URL: {r.get('url', 'N/A')}\n"
                            f"   {r.get('snippet', 'N/A')[:200]}..."
                        )
                    return "\n".join(formatted)

            elif tool == "fetch_webpage":
                # Format webpage content
                if isinstance(result, dict):
                    content = result.get('content', '')[:1000]  # Primi 1000 chars
                    return (
                        f"Title: {result.get('title', 'N/A')}\n"
                        f"URL: {result.get('url', 'N/A')}\n"
                        f"Content preview:\n{content}..."
                    )

            elif tool == "summarize_content":
                # Summary is already formatted
                return result if isinstance(result, str) else str(result)

            elif tool == "compare_sources":
                # Format comparison
                if isinstance(result, dict):
                    return (
                        f"Consensus: {result.get('consensus', 'N/A')}\n"
                        f"Differences: {result.get('differences', 'N/A')}\n"
                        f"Summary: {result.get('summary', 'N/A')}"
                    )

            elif tool == "extract_structured_data":
                # Format extracted items
                if isinstance(result, list):
                    items_preview = result[:5]  # Primi 5 items
                    return json.dumps(items_preview, indent=2, ensure_ascii=False)

            # Default: converti a stringa
            return str(result)[:500]

        except Exception as e:
            logger.error(f"Failed to format result: {e}")
            return f"[Error formatting result: {str(e)}]"

    def _create_fallback_response(
        self,
        query: str,
        results: List[ExecutionStep]
    ) -> str:
        """
        Crea risposta di fallback se sintesi LLM fallisce.

        Args:
            query: Query originale
            results: Risultati esecuzione

        Returns:
            Risposta semplice senza usare LLM
        """
        logger.info("Creating fallback response...")

        # Conta successi e fallimenti
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        response_parts = [
            f"Ho elaborato la tua richiesta: '{query}'",
            f"",
            f"Risultato: {successful}/{len(results)} operazioni completate con successo."
        ]

        # Aggiungi info risultati
        for i, step in enumerate(results, 1):
            if step.success:
                tool_name = step.tool_call.tool
                response_parts.append(f"✓ Step {i}: {tool_name} completato")
            else:
                response_parts.append(f"✗ Step {i}: {step.tool_call.tool} fallito ({step.error})")

        return "\n".join(response_parts)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_execution_history(self) -> List[Dict]:
        """
        Ritorna storia delle esecuzioni.

        Returns:
            Lista di dict con info su ogni query processata
        """
        return self.execution_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Ritorna statistiche sull'agente.

        Returns:
            Dict con varie statistiche
        """
        tool_stats = self.web_tools.get_stats()

        cache_info = self.cache.get_info() if self.cache else {}

        return {
            "queries_processed": len(self.execution_history),
            "conversation_length": len(self.memory),
            "tool_usage": tool_stats,
            "cache": cache_info
        }

    def clear_history(self):
        """Pulisce storia esecuzioni e conversazione."""
        self.execution_history.clear()
        self.memory.clear()
        logger.info("History and memory cleared")


# ----------------------------------------------------------------------------
# TESTING / DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Test dell'agente.

    Esegui: python agent.py
    """
    print("\n" + "="*70)
    print("Testing WebScraperAgent")
    print("="*70 + "\n")

    try:
        # Crea agente
        print("Creating agent (using Ollama with llama3.2)...")

        agent = WebScraperAgent(
            llm_model="llama3.2",
            llm_provider="ollama",
            enable_cache=True
        )

        print(f"\nAgent created: {agent}\n")

        # Test query
        print("-" * 70)
        print("Test Query: Cerca informazioni su Python")
        print("-" * 70)

        response = agent.process_query("Cerca le ultime informazioni su Python programming")

        print("\nResponse:")
        print(response)
        print()

        # Stats
        print("-" * 70)
        print("Agent Statistics")
        print("-" * 70)
        stats = agent.get_stats()
        print(json.dumps(stats, indent=2))

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
