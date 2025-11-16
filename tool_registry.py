"""
============================================================================
TOOL REGISTRY - Sistema di Gestione Tools per Agent
============================================================================

Questo modulo implementa il sistema centralizzato per registrare, gestire
e invocare "tools" (strumenti) che l'agente può usare.

Un "tool" è una funzione che l'agente può chiamare per eseguire azioni:
- search_web: cerca informazioni online
- fetch_webpage: scarica una pagina web
- extract_data: estrae dati strutturati
- etc.

Il ToolRegistry si occupa di:
1. Registrare tools con descrizioni e schema parametri
2. Validare parametri prima di chiamare i tools
3. Fornire descrizioni in formato LLM-friendly
4. Gestire errori in modo uniforme

Pattern Architetturali:
- Registry Pattern: registro centralizzato di componenti
- Decorator Pattern: registrazione via decorator
- Factory Pattern: creazione tools standardizzata

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import inspect
import json
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from functools import wraps
from loguru import logger

# Configurazione logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


# ----------------------------------------------------------------------------
# TOOL CLASS - Rappresentazione di un singolo tool
# ----------------------------------------------------------------------------
@dataclass
class Tool:
    """
    Rappresenta un singolo tool disponibile all'agente.

    Un tool è composto da:
    - name: nome identificativo (es: "search_web")
    - function: la funzione Python da eseguire
    - description: descrizione testuale di cosa fa
    - parameters: schema JSON dei parametri (formato OpenAI function calling)

    Questa classe incapsula tutta la logica per:
    - Validare parametri prima dell'esecuzione
    - Serializzare il tool per LLM function calling
    - Eseguire il tool con error handling
    """

    # Nome del tool (es: "search_web")
    name: str

    # Funzione Python da eseguire
    function: Callable

    # Descrizione testuale del tool
    # Usata dall'LLM per capire quando usare questo tool
    description: str

    # Schema parametri in formato JSON Schema
    # Definisce quali parametri il tool accetta
    parameters: Dict[str, Any]

    # Metadata addizionale (opzionale)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Eseguito dopo __init__ per validazione e setup.

        Valida che il tool sia configurato correttamente.
        """
        logger.debug(f"Initializing tool: {self.name}")

        # Valida che function sia effettivamente una funzione
        if not callable(self.function):
            raise ValueError(f"Tool '{self.name}': function must be callable")

        # Valida schema parametri
        if not isinstance(self.parameters, dict):
            raise ValueError(f"Tool '{self.name}': parameters must be a dict")

        # Assicurati che ci sia la struttura base
        if "type" not in self.parameters:
            self.parameters["type"] = "object"

        if "properties" not in self.parameters:
            self.parameters["properties"] = {}

        logger.debug(f"✓ Tool '{self.name}' initialized successfully")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializza tool in formato per LLM function calling.

        Questo formato è quello usato da OpenAI e compatibili
        per il function calling.

        Returns:
            Dict nel formato:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "...",
                    "parameters": {...}
                }
            }
        """
        logger.debug(f"Serializing tool '{self.name}' to dict")

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def to_text_description(self) -> str:
        """
        Genera descrizione testuale del tool.

        Utile per prompt engineering quando l'LLM non supporta
        function calling nativo.

        Returns:
            Stringa con descrizione human-readable
        """
        lines = []

        # Header
        lines.append(f"**{self.name}**")
        lines.append(f"{self.description}")
        lines.append("")

        # Parametri
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        if props:
            lines.append("Parameters:")
            for param_name, param_info in props.items():
                # Tipo parametro
                param_type = param_info.get("type", "any")

                # Descrizione parametro
                param_desc = param_info.get("description", "No description")

                # Required indicator
                req_indicator = "(required)" if param_name in required else "(optional)"

                lines.append(f"  - {param_name} ({param_type}) {req_indicator}: {param_desc}")

        return "\n".join(lines)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Valida parametri prima di eseguire il tool.

        Controlla che:
        1. Tutti i parametri required siano presenti
        2. I tipi siano corretti (basic validation)

        Args:
            params: Parametri da validare

        Returns:
            True se validi

        Raises:
            ValueError: Se validazione fallisce
        """
        logger.debug(f"Validating parameters for tool '{self.name}'")
        logger.debug(f"Received params: {params}")

        # Ottieni parametri required
        required_params = self.parameters.get("required", [])
        logger.debug(f"Required params: {required_params}")

        # Verifica che tutti i required siano presenti
        for param in required_params:
            if param not in params:
                error_msg = f"Missing required parameter: '{param}' for tool '{self.name}'"
                logger.error(error_msg)
                raise ValueError(error_msg)

        # TODO: Validazione tipi più sofisticata
        # Per ora ci limitiamo a controllare presence

        logger.debug(f"✓ Parameters validated successfully for '{self.name}'")
        return True

    def __call__(self, **kwargs) -> Any:
        """
        Esegue il tool con i parametri forniti.

        Questo permette di chiamare l'oggetto Tool direttamente:
            tool = Tool(...)
            result = tool(param1="value1", param2="value2")

        Args:
            **kwargs: Parametri per il tool

        Returns:
            Risultato dell'esecuzione del tool

        Raises:
            ValueError: Se parametri non validi
            RuntimeError: Se esecuzione fallisce
        """
        logger.info(f"Executing tool: {self.name}")
        logger.debug(f"Parameters: {kwargs}")

        try:
            # Valida parametri
            self.validate_params(kwargs)

            # Esegui funzione
            logger.debug(f"Calling function '{self.function.__name__}'")
            result = self.function(**kwargs)

            logger.success(f"✓ Tool '{self.name}' executed successfully")
            logger.debug(f"Result type: {type(result).__name__}")

            return result

        except ValueError as e:
            # Errore di validazione parametri
            logger.error(f"Parameter validation failed for '{self.name}': {e}")
            raise

        except Exception as e:
            # Errore durante esecuzione
            logger.error(f"Tool '{self.name}' execution failed: {e}")
            logger.exception(e)  # Log full stack trace
            raise RuntimeError(f"Tool '{self.name}' failed: {str(e)}")


# ----------------------------------------------------------------------------
# TOOL REGISTRY - Registro centralizzato di tutti i tools
# ----------------------------------------------------------------------------
class ToolRegistry:
    """
    Registro centralizzato di tutti i tools disponibili.

    Il ToolRegistry gestisce l'intera collezione di tools che l'agente
    può usare. Fornisce metodi per:

    1. Registrare nuovi tools
    2. Recuperare tools per nome
    3. Elencare tutti i tools disponibili
    4. Generare descrizioni per LLM
    5. Eseguire tools con validazione

    Pattern: Registry Pattern + Singleton (può essere singleton se necessario)

    Esempio d'uso:
        registry = ToolRegistry()

        # Registra tool
        registry.register(
            name="search_web",
            function=search_web_func,
            description="Search the web",
            parameters={...}
        )

        # Usa tool
        result = registry.call_tool("search_web", query="Python")
    """

    def __init__(self):
        """
        Inizializza il registry vuoto.

        Il registry parte senza tools, che vanno registrati
        manualmente o via decorator.
        """
        logger.info("Initializing ToolRegistry")

        # Dizionario nome -> Tool
        self.tools: Dict[str, Tool] = {}

        # Statistiche d'uso (opzionale, per monitoring)
        self.stats: Dict[str, int] = {}

        logger.success("✓ ToolRegistry initialized (empty)")

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tool:
        """
        Registra un nuovo tool nel registry.

        Questo è il metodo principale per aggiungere tools.

        Args:
            name: Nome univoco del tool
            function: Funzione Python da eseguire
            description: Descrizione testuale
            parameters: Schema parametri (JSON Schema format)
            metadata: Metadata addizionale (opzionale)

        Returns:
            L'oggetto Tool creato

        Raises:
            ValueError: Se tool con stesso nome già esiste
        """
        logger.info(f"Registering tool: {name}")

        # Check se già esiste
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        # Crea oggetto Tool
        tool = Tool(
            name=name,
            function=function,
            description=description,
            parameters=parameters,
            metadata=metadata or {}
        )

        # Salva nel registry
        self.tools[name] = tool

        # Inizializza stats
        self.stats[name] = 0

        logger.success(f"✓ Tool '{name}' registered successfully")
        logger.debug(f"Total tools: {len(self.tools)}")

        return tool

    def register_decorator(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator per registrare tools.

        Permette di registrare tools usando la sintassi decorator:

        @registry.register_decorator(
            name="my_tool",
            description="Does something",
            parameters={...}
        )
        def my_tool(param1, param2):
            # implementation
            pass

        Args:
            name: Nome tool
            description: Descrizione
            parameters: Schema parametri
            metadata: Metadata opzionale

        Returns:
            Decorator function
        """
        logger.debug(f"Creating decorator for tool: {name}")

        def decorator(func: Callable):
            """
            Inner decorator che wrappa la funzione.

            Args:
                func: Funzione da registrare

            Returns:
                Funzione wrappata
            """
            logger.debug(f"Applying decorator to function: {func.__name__}")

            # Registra il tool
            self.register(
                name=name,
                function=func,
                description=description,
                parameters=parameters,
                metadata=metadata
            )

            # Usa functools.wraps per preservare metadata della funzione originale
            @wraps(func)
            def wrapper(*args, **kwargs):
                """
                Wrapper che aggiunge logging quando il tool viene chiamato.
                """
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_tool(self, name: str) -> Tool:
        """
        Recupera un tool per nome.

        Args:
            name: Nome del tool

        Returns:
            Oggetto Tool

        Raises:
            ValueError: Se tool non trovato
        """
        logger.debug(f"Getting tool: {name}")

        if name not in self.tools:
            logger.error(f"Tool not found: {name}")
            logger.debug(f"Available tools: {list(self.tools.keys())}")
            raise ValueError(f"Tool '{name}' not found in registry")

        return self.tools[name]

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Esegue un tool per nome con parametri.

        Questo è il metodo principale per eseguire tools.
        Include validazione automatica e error handling.

        Args:
            name: Nome del tool da eseguire
            **kwargs: Parametri per il tool

        Returns:
            Risultato dell'esecuzione

        Raises:
            ValueError: Se tool non trovato o parametri invalidi
            RuntimeError: Se esecuzione fallisce
        """
        logger.info(f"Calling tool: {name}")
        logger.debug(f"Parameters: {kwargs}")

        # Recupera tool
        tool = self.get_tool(name)

        # Esegui tool (validazione automatica dentro __call__)
        result = tool(**kwargs)

        # Aggiorna statistiche
        self.stats[name] = self.stats.get(name, 0) + 1
        logger.debug(f"Tool '{name}' called {self.stats[name]} times total")

        return result

    def get_all_tools(self) -> List[Tool]:
        """
        Ottieni lista di tutti i tools registrati.

        Returns:
            Lista di oggetti Tool
        """
        logger.debug(f"Getting all tools (count: {len(self.tools)})")
        return list(self.tools.values())

    def list_tools(self) -> List[str]:
        """
        Lista nomi di tutti i tools.

        Returns:
            Lista di nomi
        """
        names = list(self.tools.keys())
        logger.debug(f"Tool names: {names}")
        return names

    def get_tool_descriptions(self) -> str:
        """
        Genera descrizioni testuali di tutti i tools.

        Utile per prompt engineering con LLM che non supportano
        function calling nativo.

        Returns:
            Stringa con tutte le descrizioni
        """
        logger.debug("Generating text descriptions for all tools")

        descriptions = []

        for tool in self.tools.values():
            desc = tool.to_text_description()
            descriptions.append(desc)
            descriptions.append("")  # Linea vuota tra tools

        result = "\n".join(descriptions)

        logger.debug(f"Generated descriptions (length: {len(result)} chars)")

        return result

    def get_tools_for_function_calling(self) -> List[Dict[str, Any]]:
        """
        Formatta tutti i tools per LLM function calling.

        Returns formato compatibile con OpenAI function calling:
        [
            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": {...}
                }
            },
            ...
        ]

        Returns:
            Lista di dict con tools serializzati
        """
        logger.debug("Formatting tools for function calling")

        tools_list = [tool.to_dict() for tool in self.tools.values()]

        logger.debug(f"Formatted {len(tools_list)} tools for function calling")

        return tools_list

    def unregister(self, name: str) -> bool:
        """
        Rimuove un tool dal registry.

        Args:
            name: Nome del tool da rimuovere

        Returns:
            True se rimosso, False se non trovato
        """
        logger.info(f"Unregistering tool: {name}")

        if name in self.tools:
            del self.tools[name]
            if name in self.stats:
                del self.stats[name]

            logger.success(f"✓ Tool '{name}' unregistered")
            return True
        else:
            logger.warning(f"Tool '{name}' not found, cannot unregister")
            return False

    def clear(self):
        """
        Rimuove tutti i tools dal registry.

        Utile per testing o reset completo.
        """
        logger.warning("Clearing all tools from registry")

        count = len(self.tools)
        self.tools.clear()
        self.stats.clear()

        logger.success(f"✓ Registry cleared ({count} tools removed)")

    def get_stats(self) -> Dict[str, int]:
        """
        Ottieni statistiche d'uso dei tools.

        Returns:
            Dict con nome tool -> numero di chiamate
        """
        logger.debug("Getting tool usage statistics")
        return self.stats.copy()

    def __len__(self) -> int:
        """
        Numero di tools registrati.

        Returns:
            Numero di tools
        """
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        """
        Check se un tool esiste nel registry.

        Args:
            name: Nome del tool

        Returns:
            True se esiste, False altrimenti
        """
        return name in self.tools

    def __repr__(self) -> str:
        """
        Rappresentazione stringa del registry.

        Returns:
            Stringa informativa
        """
        return f"ToolRegistry(tools={len(self.tools)}, names={list(self.tools.keys())})"


# ----------------------------------------------------------------------------
# TESTING / DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Codice di test per verificare il ToolRegistry.

    Esegui: python tool_registry.py
    """
    print("\n" + "="*70)
    print("Testing Tool Registry")
    print("="*70 + "\n")

    # Crea registry
    registry = ToolRegistry()
    print(f"Created empty registry: {registry}\n")

    # --- Test 1: Registrazione diretta ---
    print("Test 1: Direct registration")
    print("-" * 70)

    def example_search(query: str, num_results: int = 5):
        """Funzione di esempio per search"""
        logger.info(f"Searching for: {query} (max {num_results} results)")
        return [f"Result {i+1} for '{query}'" for i in range(num_results)]

    registry.register(
        name="search_web",
        function=example_search,
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    )

    print(f"Registry after registration: {registry}\n")

    # --- Test 2: Registrazione con decorator ---
    print("\nTest 2: Decorator registration")
    print("-" * 70)

    @registry.register_decorator(
        name="fetch_page",
        description="Fetch a webpage",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                }
            },
            "required": ["url"]
        }
    )
    def fetch_webpage(url: str):
        """Funzione di esempio per fetch"""
        logger.info(f"Fetching: {url}")
        return f"Content of {url}"

    print(f"Registry after decorator: {registry}\n")

    # --- Test 3: Chiamata tools ---
    print("\nTest 3: Calling tools")
    print("-" * 70)

    # Chiama search_web
    result1 = registry.call_tool("search_web", query="Python programming", num_results=3)
    print(f"\nSearch result: {result1}\n")

    # Chiama fetch_page
    result2 = registry.call_tool("fetch_page", url="https://example.com")
    print(f"\nFetch result: {result2}\n")

    # --- Test 4: Descrizioni ---
    print("\nTest 4: Tool descriptions")
    print("-" * 70)

    # Descrizione testuale
    print("\nText descriptions:")
    print(registry.get_tool_descriptions())

    # Formato function calling
    print("\nFunction calling format:")
    tools_fc = registry.get_tools_for_function_calling()
    print(json.dumps(tools_fc, indent=2))

    # --- Test 5: Statistiche ---
    print("\nTest 5: Usage statistics")
    print("-" * 70)

    stats = registry.get_stats()
    print(f"\nTool usage stats: {stats}\n")

    # --- Test 6: Error handling ---
    print("\nTest 6: Error handling")
    print("-" * 70)

    try:
        # Tool inesistente
        registry.call_tool("nonexistent_tool")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    try:
        # Parametro mancante
        registry.call_tool("search_web")  # Manca 'query' required
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
