"""
============================================================================
LLM INTERFACE - Interfaccia Unificata per Large Language Models
============================================================================

Questo modulo fornisce un'interfaccia unificata per interagire con diversi
provider di LLM (Large Language Models):

- Ollama (modelli locali come llama3.2, mistral, etc.)
- OpenAI (GPT-4, GPT-3.5-turbo, etc.)
- Groq (llama3, mixtral, gemma - velocissimi)
- Anthropic (Claude)

L'interfaccia permette di:
1. Switchare facilmente tra provider
2. Usare la stessa API indipendentemente dal provider
3. Supportare function calling (tool use)
4. Gestire errori in modo uniforme

Architettura:
- BaseLLM: Classe astratta che definisce l'interfaccia
- OllamaLLM: Implementazione per Ollama
- OpenAILLM: Implementazione per OpenAI
- GroqLLM: Implementazione per Groq
- AnthropicLLM: Implementazione per Anthropic
- LLMInterface: Factory class che crea l'istanza giusta

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import json
from typing import List, Dict, Optional, Any, Literal
from abc import ABC, abstractmethod
from loguru import logger

# Configurazione logging
# Rimuovi handler di default e configura uno personalizzato
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# ----------------------------------------------------------------------------
# BASE CLASS - Interfaccia Astratta
# ----------------------------------------------------------------------------
class BaseLLM(ABC):
    """
    Classe base astratta per tutti i provider LLM.

    Definisce l'interfaccia che tutti i provider devono implementare.
    Questo permette di usare lo stesso codice indipendentemente dal provider.

    Pattern: Abstract Base Class (ABC)
    """

    def __init__(self, model: str, **kwargs):
        """
        Inizializza il provider LLM.

        Args:
            model: Nome del modello da usare
            **kwargs: Parametri addizionali specifici del provider
        """
        self.model = model
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with model: {model}")

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Genera una risposta singola dato un prompt.

        Questo è il metodo più semplice: invii un prompt, ricevi una risposta.

        Args:
            prompt: Il prompt dell'utente
            system: Prompt di sistema (opzionale) che definisce il comportamento
            **kwargs: Parametri addizionali (temperature, max_tokens, etc.)

        Returns:
            La risposta generata dal modello come stringa

        Raises:
            RuntimeError: Se la generazione fallisce
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Genera risposta in una conversazione multi-turn.

        Permette di mantenere contesto conversazionale con messaggi alternati
        tra user e assistant.

        Args:
            messages: Lista di messaggi nel formato:
                [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            **kwargs: Parametri addizionali

        Returns:
            La risposta generata dal modello

        Raises:
            RuntimeError: Se la chat fallisce
        """
        pass

    @abstractmethod
    def function_call(
        self,
        query: str,
        tools: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Richiede al modello di chiamare una funzione/tool.

        Il modello analizza la query e decide quale tool chiamare
        e con quali parametri.

        Args:
            query: Query dell'utente
            tools: Lista di tool disponibili nel formato:
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "description": "Search the web...",
                            "parameters": {...}
                        }
                    },
                    ...
                ]
            **kwargs: Parametri addizionali

        Returns:
            Dizionario con:
                {
                    "tool": "nome_tool",
                    "parameters": {...},
                    "reasoning": "perché questo tool" (opzionale)
                }

        Raises:
            RuntimeError: Se il function calling fallisce
        """
        pass


# ----------------------------------------------------------------------------
# OLLAMA IMPLEMENTATION
# ----------------------------------------------------------------------------
class OllamaLLM(BaseLLM):
    """
    Implementazione per Ollama - LLM locale.

    Ollama permette di eseguire modelli LLM localmente senza API keys.
    Modelli supportati: llama3.2, llama3.1, mistral, codellama, phi, etc.

    Requirements:
    - Ollama installato e in esecuzione (ollama serve)
    - Modello scaricato (ollama pull llama3.2)

    Vantaggi:
    - Gratuito, no API limits
    - Privacy: tutto locale
    - Veloce su hardware adeguato

    Svantaggi:
    - Richiede GPU/RAM adeguate
    - Modelli meno capaci di GPT-4
    """

    def __init__(self, model: str = "llama3.2", host: str = None, **kwargs):
        """
        Inizializza client Ollama.

        Args:
            model: Nome modello (default: llama3.2)
            host: URL Ollama server (default: http://localhost:11434)
            **kwargs: Parametri addizionali
        """
        super().__init__(model, **kwargs)

        # Determina host (env var > parametro > default)
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

        logger.info(f"Connecting to Ollama at {self.host}")

        # Importa e inizializza client Ollama
        try:
            import ollama
            self.client = ollama.Client(host=self.host)
            logger.success(f"✓ Connected to Ollama successfully")
        except ImportError:
            logger.error("Ollama package not installed. Run: pip install ollama")
            raise RuntimeError("Ollama package not found")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}")

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Genera risposta con Ollama.

        Args:
            prompt: Prompt utente
            system: System prompt (opzionale)
            **kwargs: temperature, max_tokens, etc.

        Returns:
            Risposta generata
        """
        logger.debug(f"Generating response for prompt (length: {len(prompt)} chars)")

        # Costruisci messaggi
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
            logger.debug(f"Using system prompt: {system[:100]}...")

        messages.append({"role": "user", "content": prompt})

        try:
            # Chiama Ollama
            logger.info(f"Calling Ollama model: {self.model}")

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2000),
                }
            )

            # Estrai contenuto
            content = response['message']['content']

            logger.success(f"✓ Generated response (length: {len(content)} chars)")
            logger.debug(f"Response preview: {content[:200]}...")

            return content

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat multi-turn con Ollama.

        Args:
            messages: Lista messaggi conversazione
            **kwargs: Parametri addizionali

        Returns:
            Risposta generata
        """
        logger.debug(f"Chat with {len(messages)} messages")

        try:
            # Log messaggi (solo preview)
            for i, msg in enumerate(messages):
                preview = msg['content'][:100]
                logger.debug(f"  Message {i} ({msg['role']}): {preview}...")

            logger.info(f"Calling Ollama chat with model: {self.model}")

            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": kwargs.get("temperature", 0.7),
                }
            )

            content = response['message']['content']
            logger.success(f"✓ Chat response generated (length: {len(content)} chars)")

            return content

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise RuntimeError(f"Chat failed: {e}")

    def function_call(
        self,
        query: str,
        tools: List[Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simula function calling con prompt engineering.

        Nota: Ollama non ha function calling nativo come OpenAI.
        Usiamo prompt engineering per ottenere lo stesso risultato.

        Args:
            query: Query utente
            tools: Lista tools disponibili
            **kwargs: Parametri addizionali

        Returns:
            Dict con tool e parametri da chiamare
        """
        logger.info(f"Function calling for query: {query[:100]}...")
        logger.debug(f"Available tools: {len(tools)}")

        # Formatta tools per il prompt
        tools_desc = self._format_tools_for_prompt(tools)

        # Crea prompt per function calling
        prompt = f"""You are a function calling assistant. Analyze the user query and decide which tool to call.

User query: "{query}"

Available tools:
{tools_desc}

Based on the query, decide which tool to call and with what parameters.

IMPORTANT: Respond ONLY with a JSON object in this exact format:
{{
    "tool": "tool_name",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "reasoning": "brief explanation why this tool"
}}

JSON response:"""

        logger.debug("Sending function calling prompt to Ollama")

        try:
            # Genera risposta
            response = self.generate(prompt, temperature=0.1)  # Bassa temp per output deterministico

            # Parse JSON dalla risposta
            result = self._parse_function_call(response)

            logger.success(f"✓ Function call: {result['tool']} with {len(result.get('parameters', {}))} params")
            logger.debug(f"Reasoning: {result.get('reasoning', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            raise RuntimeError(f"Function call failed: {e}")

    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """
        Formatta tools in descrizione testuale per il prompt.

        Args:
            tools: Lista tools

        Returns:
            Stringa con descrizioni tools
        """
        lines = []
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            desc = func.get('description', 'No description')
            params = func.get('parameters', {})

            lines.append(f"- {name}: {desc}")

            # Aggiungi info parametri
            props = params.get('properties', {})
            if props:
                lines.append(f"  Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}")

            lines.append("")  # Linea vuota

        return "\n".join(lines)

    def _parse_function_call(self, response: str) -> Dict[str, Any]:
        """
        Parse risposta LLM per estrarre function call.

        Il modello dovrebbe rispondere con JSON, ma potrebbe aggiungere
        testo extra. Questa funzione cerca ed estrae il JSON.

        Args:
            response: Risposta del modello

        Returns:
            Dict con tool e parametri

        Raises:
            ValueError: Se parsing fallisce
        """
        logger.debug("Parsing function call from response")

        try:
            # Cerca blocco JSON nella risposta
            # Trova primo { e ultimo }
            start = response.find('{')
            end = response.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start:end]
            logger.debug(f"Extracted JSON: {json_str[:200]}...")

            # Parse JSON
            result = json.loads(json_str)

            # Valida struttura
            if 'tool' not in result:
                raise ValueError("Missing 'tool' field in response")

            # Aggiungi campi di default se mancanti
            if 'parameters' not in result:
                result['parameters'] = {}
            if 'reasoning' not in result:
                result['reasoning'] = ""

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON in response: {e}")
        except Exception as e:
            logger.error(f"Function call parsing failed: {e}")
            raise ValueError(f"Failed to parse function call: {e}")


# ----------------------------------------------------------------------------
# OPENAI IMPLEMENTATION
# ----------------------------------------------------------------------------
class OpenAILLM(BaseLLM):
    """
    Implementazione per OpenAI (GPT-4, GPT-3.5, etc.).

    OpenAI offre i modelli più capaci ma richiede API key e ha costi.

    Modelli comuni:
    - gpt-4-turbo: Più capace, costoso
    - gpt-4: Molto capace, molto costoso
    - gpt-3.5-turbo: Veloce ed economico

    Vantaggi:
    - Modelli molto capaci
    - Function calling nativo
    - Veloce e affidabile

    Svantaggi:
    - Richiede API key
    - Ha costi per utilizzo
    - Dati inviati a OpenAI
    """

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None, **kwargs):
        """
        Inizializza client OpenAI.

        Args:
            model: Nome modello (default: gpt-3.5-turbo)
            api_key: API key OpenAI (o usa env var OPENAI_API_KEY)
            **kwargs: Parametri addizionali
        """
        super().__init__(model, **kwargs)

        # Ottieni API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            logger.error("OpenAI API key not found!")
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter")

        logger.info("Initializing OpenAI client")

        # Importa e inizializza client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.success("✓ OpenAI client initialized")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise RuntimeError("OpenAI package not found")

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Genera risposta con OpenAI."""
        logger.debug(f"Generating with OpenAI model: {self.model}")

        # Costruisci messaggi
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(f"Calling OpenAI API...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )

            content = response.choices[0].message.content
            logger.success(f"✓ OpenAI response received (length: {len(content)} chars)")

            return content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat con OpenAI."""
        logger.debug(f"Chat with {len(messages)} messages")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7)
            )

            content = response.choices[0].message.content
            logger.success(f"✓ OpenAI chat response received")

            return content

        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise RuntimeError(f"Chat failed: {e}")

    def function_call(self, query: str, tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Function calling nativo di OpenAI.

        OpenAI supporta function calling in modo nativo,
        quindi non serve prompt engineering.
        """
        logger.info(f"OpenAI function calling for: {query[:100]}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Check se ha chiamato un tool
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                result = {
                    "tool": tool_call.function.name,
                    "parameters": json.loads(tool_call.function.arguments),
                    "reasoning": "OpenAI native function call"
                }
                logger.success(f"✓ Function call: {result['tool']}")
                return result
            else:
                logger.warning("No tool called by OpenAI")
                return {"error": "No tool called"}

        except Exception as e:
            logger.error(f"OpenAI function calling failed: {e}")
            raise RuntimeError(f"Function call failed: {e}")


# ----------------------------------------------------------------------------
# GROQ IMPLEMENTATION
# ----------------------------------------------------------------------------
class GroqLLM(BaseLLM):
    """
    Implementazione per Groq - LLM ultrarapidi in cloud.

    Groq offre inference estremamente veloce per modelli open source
    grazie alla loro hardware personalizzato (LPU - Language Processing Unit).

    Modelli supportati:
    - llama-3.1-70b-versatile: Llama 3.1 70B (molto capace)
    - llama-3.1-8b-instant: Llama 3.1 8B (velocissimo)
    - llama3-70b-8192: Llama 3 70B
    - llama3-8b-8192: Llama 3 8B
    - mixtral-8x7b-32768: Mixtral 8x7B (ottimo mix velocità/qualità)
    - gemma-7b-it: Google Gemma 7B

    Vantaggi:
    - VELOCITÀ ESTREMA: inference 10-100x più veloce di altri provider
    - Modelli open source di alta qualità
    - API gratuita (con rate limits)
    - Supporta function calling
    - Ottimo per prototipazione rapida

    Svantaggi:
    - Richiede API key
    - Rate limits su tier gratuito
    - Meno modelli disponibili rispetto a OpenAI
    - Dati inviati a Groq (cloud)

    API Key:
    - Si legge da file: groq/API_groq.txt
    - Oppure da env var: GROQ_API_KEY
    - Ottieni chiave su: https://console.groq.com/
    """

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: str = None, **kwargs):
        """
        Inizializza client Groq.

        Args:
            model: Nome modello (default: llama-3.1-8b-instant)
            api_key: API key Groq (o usa file groq/API_groq.txt o env var GROQ_API_KEY)
            **kwargs: Parametri addizionali
        """
        super().__init__(model, **kwargs)

        # Ottieni API key con priorità: parametro > env var > file
        self.api_key = api_key or self._load_api_key()

        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            logger.error("Groq API key not found or invalid!")
            logger.info("To use Groq:")
            logger.info("1. Get API key from https://console.groq.com/")
            logger.info("2. Put it in groq/API_groq.txt")
            logger.info("3. Or set GROQ_API_KEY environment variable")
            raise ValueError("Groq API key required. See logs for instructions.")

        logger.info("Initializing Groq client")

        # Importa e inizializza client
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.success("✓ Groq client initialized")
            logger.info(f"Using Groq model: {self.model}")
        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
            raise RuntimeError("Groq package not found")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise RuntimeError(f"Groq initialization failed: {e}")

    def _load_api_key(self) -> Optional[str]:
        """
        Carica API key da diverse fonti.

        Priorità:
        1. Environment variable GROQ_API_KEY
        2. File groq/API_groq.txt

        Returns:
            API key o None se non trovata
        """
        logger.debug("Loading Groq API key...")

        # 1. Prova environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            logger.debug("✓ API key loaded from environment variable")
            return api_key

        # 2. Prova file groq/API_groq.txt
        api_key_file = "groq/API_groq.txt"
        try:
            with open(api_key_file, 'r') as f:
                # Leggi tutte le righe
                lines = f.readlines()

                # Trova la prima riga che non è commento o vuota
                for line in lines:
                    line = line.strip()

                    # Salta commenti e righe vuote
                    if not line or line.startswith('#'):
                        continue

                    # Questa dovrebbe essere l'API key
                    logger.debug(f"✓ API key loaded from file: {api_key_file}")
                    return line

            logger.warning(f"File {api_key_file} exists but contains no valid API key")
            return None

        except FileNotFoundError:
            logger.warning(f"API key file not found: {api_key_file}")
            return None
        except Exception as e:
            logger.error(f"Error reading API key file: {e}")
            return None

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Genera risposta con Groq.

        Args:
            prompt: Prompt utente
            system: System prompt (opzionale)
            **kwargs: temperature, max_tokens, etc.

        Returns:
            Risposta generata
        """
        logger.debug(f"Generating with Groq model: {self.model}")

        # Costruisci messaggi
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
            logger.debug(f"Using system prompt: {system[:100]}...")

        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(f"Calling Groq API (model: {self.model})...")

            # Chiama Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
            )

            # Estrai contenuto
            content = response.choices[0].message.content

            logger.success(f"✓ Groq response received (length: {len(content)} chars)")
            logger.debug(f"Response preview: {content[:200]}...")

            # Log statistiche usage se disponibili
            if hasattr(response, 'usage'):
                logger.debug(f"Tokens used - prompt: {response.usage.prompt_tokens}, "
                           f"completion: {response.usage.completion_tokens}, "
                           f"total: {response.usage.total_tokens}")

            return content

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise RuntimeError(f"Groq generation failed: {e}")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat multi-turn con Groq.

        Args:
            messages: Lista messaggi conversazione
            **kwargs: Parametri addizionali

        Returns:
            Risposta generata
        """
        logger.debug(f"Groq chat with {len(messages)} messages")

        try:
            # Log messaggi (solo preview)
            for i, msg in enumerate(messages):
                preview = msg['content'][:100]
                logger.debug(f"  Message {i} ({msg['role']}): {preview}...")

            logger.info(f"Calling Groq chat API (model: {self.model})...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000)
            )

            content = response.choices[0].message.content
            logger.success(f"✓ Groq chat response received (length: {len(content)} chars)")

            return content

        except Exception as e:
            logger.error(f"Groq chat failed: {e}")
            raise RuntimeError(f"Chat failed: {e}")

    def function_call(self, query: str, tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Function calling con Groq.

        Groq supporta function calling nativo per alcuni modelli.
        Per modelli che non lo supportano, usa prompt engineering.

        Args:
            query: Query utente
            tools: Lista tools disponibili
            **kwargs: Parametri addizionali

        Returns:
            Dict con tool da chiamare e parametri
        """
        logger.info(f"Groq function calling for: {query[:100]}...")
        logger.debug(f"Available tools: {len(tools)}")

        # Lista modelli Groq con function calling nativo
        # (al momento Groq sta estendendo il supporto)
        native_fc_models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768"
        ]

        # Prova function calling nativo se il modello lo supporta
        if any(model in self.model for model in native_fc_models):
            logger.debug("Model supports native function calling, trying native approach")

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": query}],
                    tools=tools,
                    tool_choice="auto"
                )

                message = response.choices[0].message

                # Check se ha chiamato un tool
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    result = {
                        "tool": tool_call.function.name,
                        "parameters": json.loads(tool_call.function.arguments),
                        "reasoning": "Groq native function call"
                    }
                    logger.success(f"✓ Function call: {result['tool']}")
                    return result

            except Exception as e:
                logger.warning(f"Native function calling failed, falling back to prompt engineering: {e}")

        # Fallback: usa prompt engineering (come Ollama)
        logger.debug("Using prompt engineering for function calling")

        # Formatta tools per il prompt
        tools_desc = self._format_tools_for_prompt(tools)

        # Crea prompt per function calling
        prompt = f"""You are a function calling assistant. Analyze the user query and decide which tool to call.

User query: "{query}"

Available tools:
{tools_desc}

Based on the query, decide which tool to call and with what parameters.

IMPORTANT: Respond ONLY with a JSON object in this exact format:
{{
    "tool": "tool_name",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "reasoning": "brief explanation why this tool"
}}

JSON response:"""

        logger.debug("Sending function calling prompt to Groq")

        try:
            # Genera risposta con temperatura bassa per output più deterministico
            response = self.generate(prompt, temperature=0.1)

            # Parse JSON dalla risposta
            result = self._parse_function_call(response)

            logger.success(f"✓ Function call: {result['tool']} with {len(result.get('parameters', {}))} params")
            logger.debug(f"Reasoning: {result.get('reasoning', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            raise RuntimeError(f"Function call failed: {e}")

    def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """
        Formatta tools in descrizione testuale per il prompt.

        Args:
            tools: Lista tools

        Returns:
            Stringa con descrizioni tools
        """
        lines = []
        for tool in tools:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            desc = func.get('description', 'No description')
            params = func.get('parameters', {})

            lines.append(f"- {name}: {desc}")

            # Aggiungi info parametri
            props = params.get('properties', {})
            if props:
                lines.append(f"  Parameters:")
                for param_name, param_info in props.items():
                    param_type = param_info.get('type', 'any')
                    param_desc = param_info.get('description', '')
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}")

            lines.append("")  # Linea vuota

        return "\n".join(lines)

    def _parse_function_call(self, response: str) -> Dict[str, Any]:
        """
        Parse risposta LLM per estrarre function call.

        Il modello dovrebbe rispondere con JSON, ma potrebbe aggiungere
        testo extra. Questa funzione cerca ed estrae il JSON.

        Args:
            response: Risposta del modello

        Returns:
            Dict con tool e parametri

        Raises:
            ValueError: Se parsing fallisce
        """
        logger.debug("Parsing function call from response")

        try:
            # Cerca blocco JSON nella risposta
            # Trova primo { e ultimo }
            start = response.find('{')
            end = response.rfind('}') + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[start:end]
            logger.debug(f"Extracted JSON: {json_str[:200]}...")

            # Parse JSON
            result = json.loads(json_str)

            # Valida struttura
            if 'tool' not in result:
                raise ValueError("Missing 'tool' field in response")

            # Aggiungi campi di default se mancanti
            if 'parameters' not in result:
                result['parameters'] = {}
            if 'reasoning' not in result:
                result['reasoning'] = ""

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Invalid JSON in response: {e}")
        except Exception as e:
            logger.error(f"Function call parsing failed: {e}")
            raise ValueError(f"Failed to parse function call: {e}")


# ----------------------------------------------------------------------------
# LLM INTERFACE - Factory Class
# ----------------------------------------------------------------------------
class LLMInterface:
    """
    Interfaccia unificata che auto-seleziona il provider giusto.

    Questa è la classe principale che dovresti usare.
    Si occupa di:
    1. Scegliere il provider giusto (Ollama, OpenAI, etc.)
    2. Inizializzare il client
    3. Delegare le chiamate al provider

    Esempio:
        llm = LLMInterface(model="llama3.2", provider="ollama")
        response = llm.generate("Ciao, come stai?")
        print(response)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        provider: Literal["ollama", "openai", "groq", "anthropic"] = "ollama",
        **kwargs
    ):
        """
        Inizializza l'interfaccia LLM.

        Args:
            model: Nome del modello da usare
            provider: Provider LLM ("ollama", "openai", "groq", "anthropic")
            **kwargs: Parametri addizionali per il provider
        """
        self.model = model
        self.provider = provider.lower()

        logger.info(f"Initializing LLM Interface: {provider}/{model}")

        # Seleziona e inizializza il provider
        if self.provider == "ollama":
            self.llm = OllamaLLM(model, **kwargs)

        elif self.provider == "openai":
            self.llm = OpenAILLM(model, **kwargs)

        elif self.provider == "groq":
            self.llm = GroqLLM(model, **kwargs)

        elif self.provider == "anthropic":
            # TODO: Implementare AnthropicLLM
            logger.error("Anthropic provider not yet implemented")
            raise NotImplementedError("Anthropic support coming soon")

        else:
            logger.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'openai', 'groq', or 'anthropic'")

        logger.success(f"✓ LLM Interface ready: {provider}/{model}")

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Genera una risposta.

        Args:
            prompt: Prompt utente
            system: System prompt (opzionale)
            **kwargs: Parametri addizionali

        Returns:
            Risposta generata
        """
        return self.llm.generate(prompt, system, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat multi-turn.

        Args:
            messages: Lista messaggi
            **kwargs: Parametri addizionali

        Returns:
            Risposta generata
        """
        return self.llm.chat(messages, **kwargs)

    def function_call(self, query: str, tools: List[Dict], **kwargs) -> Dict[str, Any]:
        """
        Function calling / tool use.

        Args:
            query: Query utente
            tools: Lista tools disponibili
            **kwargs: Parametri addizionali

        Returns:
            Dict con tool da chiamare e parametri
        """
        return self.llm.function_call(query, tools, **kwargs)


# ----------------------------------------------------------------------------
# TESTING / DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Codice di testing per verificare che l'interfaccia funzioni.

    Esegui: python llm_interface.py
    """
    print("\n" + "="*70)
    print("Testing LLM Interface")
    print("="*70 + "\n")

    # Test con Ollama
    try:
        print("Testing Ollama...")
        llm = LLMInterface(model="llama3.2", provider="ollama")

        # Test generate
        response = llm.generate("Di' ciao in una frase breve")
        print(f"\nRisposta: {response}\n")

        # Test function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        result = llm.function_call("Cerca informazioni su Python", tools)
        print(f"\nFunction call result: {result}\n")

    except Exception as e:
        print(f"Ollama test failed: {e}")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
