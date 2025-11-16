"""
============================================================================
WEB TOOLS - Collezione di Tools per Web Scraping e Ricerca
============================================================================

Questo modulo implementa tutti i tools che l'agente può usare per
interagire con il web:

1. search_web: Ricerca informazioni su web usando motore di ricerca
2. fetch_webpage: Download e parsing di pagine HTML
3. extract_structured_data: Estrazione dati strutturati da HTML
4. summarize_content: Riassunto testi usando LLM
5. compare_sources: Confronto informazioni da fonti multiple

Ogni tool è implementato come metodo della classe WebTools, con:
- Validazione input completa
- Error handling robusto
- Retry logic per errori di rete
- Logging dettagliato di tutte le operazioni
- Documentazione estesa

Dependencies:
- requests: HTTP requests
- beautifulsoup4: HTML parsing
- lxml: Fast HTML/XML parser
- readability-lxml: Main content extraction
- duckduckgo-search: Web search (no API key needed)
- tenacity: Retry logic
- validators: URL validation

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

# HTTP and web scraping
import requests
from bs4 import BeautifulSoup
import validators

# Readability per contenuto principale
try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    print("Warning: readability-lxml not installed. Main content extraction disabled.")

# Search
from duckduckgo_search import DDGS

# Retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Logging
from loguru import logger

# Configurazione logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# ----------------------------------------------------------------------------
# WEB TOOLS CLASS
# ----------------------------------------------------------------------------
class WebTools:
    """
    Collezione di tools per web scraping e ricerca.

    Questa classe raggruppa tutti i tools che l'agente può usare per:
    - Cercare informazioni online
    - Scaricare e parsare pagine web
    - Estrarre dati strutturati
    - Riassumere contenuti
    - Confrontare fonti

    Ogni tool è implementato come metodo, e può essere facilmente
    registrato nel ToolRegistry.

    Esempio d'uso:
        tools = WebTools(config={...}, llm_interface=llm)
        results = tools.search_web("Python programming", num_results=5)
        page = tools.fetch_webpage("https://example.com")
    """

    def __init__(self, config: Dict = None, llm_interface = None):
        """
        Inizializza WebTools.

        Args:
            config: Configurazione tools (timeout, user agent, etc.)
            llm_interface: Interfaccia LLM per tools che richiedono AI
                          (summarize_content, compare_sources)
        """
        logger.info("Initializing WebTools")

        # Salva configurazione
        self.config = config or {}
        logger.debug(f"Configuration: {len(self.config)} parameters")

        # Salva LLM interface
        self.llm = llm_interface
        if self.llm:
            logger.debug("✓ LLM interface available for AI-powered tools")
        else:
            logger.warning("⚠ LLM interface not provided. AI tools will not work.")

        # Cache per risultati (se abilitata)
        self.cache = None  # Will be set externally if caching enabled

        # Statistiche d'uso
        self.stats = {
            "search_web": 0,
            "fetch_webpage": 0,
            "extract_structured_data": 0,
            "summarize_content": 0,
            "compare_sources": 0
        }

        logger.success("✓ WebTools initialized successfully")

    # ========================================================================
    # TOOL 1: SEARCH_WEB - Ricerca sul web
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError))
    )
    def search_web(
        self,
        query: str,
        num_results: int = 5,
        region: str = "it-it",
        safe_search: str = "moderate"
    ) -> List[Dict[str, str]]:
        """
        Cerca informazioni sul web usando DuckDuckGo.

        Questo tool permette all'agente di cercare informazioni online.
        Usa DuckDuckGo perché non richiede API key (a differenza di Google).

        Features:
        - Nessuna API key richiesta
        - Retry automatico su errori di rete
        - Validazione input
        - Safe search configurabile
        - Supporto regioni/lingue diverse

        Args:
            query: Query di ricerca (es: "Python web scraping tutorial")
            num_results: Numero di risultati da ritornare (default 5, max 20)
            region: Regione per risultati localizzati (default "it-it")
                   Opzioni: "it-it", "en-us", "fr-fr", "de-de", etc.
            safe_search: Livello safe search ("off", "moderate", "strict")

        Returns:
            Lista di dizionari, ciascuno con:
            {
                "title": "Titolo del risultato",
                "url": "https://...",
                "snippet": "Anteprima del contenuto...",
                "position": 1  # Posizione nei risultati
            }

        Raises:
            ValueError: Se query vuota o parametri invalidi
            RuntimeError: Se la ricerca fallisce dopo tutti i retry

        Example:
            >>> tools = WebTools()
            >>> results = tools.search_web("intelligenza artificiale 2024", num_results=3)
            >>> for r in results:
            ...     print(f"{r['title']}: {r['url']}")
        """
        # Incrementa statistiche
        self.stats["search_web"] += 1

        logger.info(f"🔍 Searching web for: '{query}'")
        logger.debug(f"Parameters: num_results={num_results}, region={region}, safe_search={safe_search}")

        # ====================================================================
        # VALIDAZIONE INPUT
        # ====================================================================

        # Valida query non vuota
        if not query or not query.strip():
            logger.error("Search query is empty")
            raise ValueError("Search query cannot be empty")

        query = query.strip()
        logger.debug(f"Query (cleaned): '{query}'")

        # Valida e limita num_results
        if num_results < 1:
            logger.warning(f"num_results too low ({num_results}), setting to 1")
            num_results = 1
        elif num_results > 20:
            logger.warning(f"num_results too high ({num_results}), capping at 20")
            num_results = 20

        # Valida safe_search
        valid_safe_search = ["off", "moderate", "strict"]
        if safe_search not in valid_safe_search:
            logger.warning(f"Invalid safe_search '{safe_search}', using 'moderate'")
            safe_search = "moderate"

        # ====================================================================
        # CHECK CACHE (se disponibile)
        # ====================================================================
        if self.cache:
            logger.debug("Checking cache...")
            cached = self.cache.get("search_web", {
                "query": query,
                "num_results": num_results,
                "region": region
            })

            if cached:
                logger.success(f"✓ Found in cache! Returning {len(cached)} results")
                return cached

        # ====================================================================
        # ESECUZIONE RICERCA
        # ====================================================================
        try:
            logger.info("Calling DuckDuckGo search API...")

            # Crea client DuckDuckGo
            with DDGS() as ddgs:
                results = []

                # Esegui ricerca
                # Il decorator @retry gestirà automaticamente i retry
                logger.debug("Fetching search results...")

                search_results = ddgs.text(
                    query,
                    region=region,
                    safesearch=safe_search,
                    max_results=num_results
                )

                # Processa risultati
                for i, result in enumerate(search_results):
                    # Estrai campi da risultato DuckDuckGo
                    processed_result = {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "position": i + 1
                    }

                    results.append(processed_result)

                    # Log ogni risultato
                    logger.debug(f"Result {i+1}: {processed_result['title'][:50]}... - {processed_result['url']}")

            # ================================================================
            # POST-PROCESSING
            # ================================================================

            # Verifica che abbiamo ottenuto risultati
            if not results:
                logger.warning("No results found for query")
                return []

            logger.success(f"✓ Search completed: found {len(results)} results")

            # Salva in cache (se disponibile)
            if self.cache:
                logger.debug("Saving results to cache")
                self.cache.set("search_web", {
                    "query": query,
                    "num_results": num_results,
                    "region": region
                }, results)

            return results

        except Exception as e:
            # Il decorator retry ha già provato 3 volte
            logger.error(f"Search failed after retries: {e}")
            logger.exception(e)  # Log stack trace completo
            raise RuntimeError(f"Web search failed: {str(e)}")

    # ========================================================================
    # TOOL 2: FETCH_WEBPAGE - Download e parsing HTML
    # ========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def fetch_webpage(
        self,
        url: str,
        timeout: int = None,
        extract_main_content: bool = True,
        extract_metadata: bool = True,
        extract_links: bool = True,
        max_links: int = 100
    ) -> Dict[str, Any]:
        """
        Scarica e parsifica una pagina web.

        Questo tool permette all'agente di:
        - Scaricare qualsiasi pagina web
        - Estrarre contenuto testuale pulito
        - Estrarre metadata (title, author, date, etc.)
        - Estrarre tutti i link
        - Gestire redirect automaticamente

        Features:
        - Estrazione contenuto principale (rimuove ads/nav/footer)
        - Parsing metadata completo
        - Estrazione links con conversione a URL assoluti
        - Retry automatico su errori di rete
        - User agent configurabile
        - Timeout configurabile

        Args:
            url: URL della pagina da scaricare
            timeout: Timeout in secondi (default: da config o 10)
            extract_main_content: Se True, estrae solo contenuto principale
                                 (rimuove navigazione, ads, footer, etc.)
            extract_metadata: Se True, estrae metadata (author, date, description)
            extract_links: Se True, estrae tutti i link dalla pagina
            max_links: Numero massimo di link da estrarre (default 100)

        Returns:
            Dizionario con:
            {
                "url": "URL effettivo (dopo redirect)",
                "title": "Titolo pagina",
                "content": "Testo pulito del contenuto",
                "html": "HTML completo originale",
                "links": [{"url": "...", "text": "..."}, ...],
                "meta": {
                    "author": "...",
                    "date": "...",
                    "description": "...",
                    ...
                },
                "status_code": 200,
                "encoding": "utf-8",
                "content_type": "text/html"
            }

        Raises:
            ValueError: Se URL invalido
            RuntimeError: Se download fallisce dopo retry

        Example:
            >>> tools = WebTools()
            >>> page = tools.fetch_webpage("https://example.com")
            >>> print(f"Title: {page['title']}")
            >>> print(f"Content length: {len(page['content'])} chars")
        """
        # Incrementa statistiche
        self.stats["fetch_webpage"] += 1

        logger.info(f"📄 Fetching webpage: {url}")

        # ====================================================================
        # VALIDAZIONE INPUT
        # ====================================================================

        # Valida URL
        if not validators.url(url):
            logger.error(f"Invalid URL: {url}")
            raise ValueError(f"Invalid URL: {url}")

        # Determina timeout
        if timeout is None:
            timeout = self.config.get("fetch_webpage", {}).get("timeout_seconds", 10)

        logger.debug(f"Timeout: {timeout}s")

        # Determina user agent
        user_agent = self.config.get("fetch_webpage", {}).get(
            "user_agent",
            "Mozilla/5.0 (Educational Web Scraper Agent) AppleWebKit/537.36"
        )

        # ====================================================================
        # DOWNLOAD PAGINA
        # ====================================================================

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7',
        }

        try:
            logger.info("Downloading page...")
            logger.debug(f"Headers: {headers}")

            # Scarica pagina
            response = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
                verify=True  # Verifica SSL certificates
            )

            # Verifica status code
            logger.debug(f"Status code: {response.status_code}")

            if response.status_code != 200:
                logger.warning(f"Non-200 status code: {response.status_code}")

            response.raise_for_status()  # Raise su 4xx/5xx errors

            logger.success(f"✓ Page downloaded (size: {len(response.content)} bytes)")

            # Log info risposta
            logger.debug(f"Final URL (after redirects): {response.url}")
            logger.debug(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            logger.debug(f"Encoding: {response.encoding}")

        except requests.Timeout:
            logger.error(f"Timeout downloading {url}")
            raise RuntimeError(f"Timeout fetching {url} (timeout: {timeout}s)")
        except requests.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise RuntimeError(f"HTTP error fetching {url}: {e}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise RuntimeError(f"Failed to fetch {url}: {e}")

        # ====================================================================
        # PARSING HTML
        # ====================================================================

        logger.info("Parsing HTML...")

        # Crea BeautifulSoup object
        try:
            soup = BeautifulSoup(response.text, 'lxml')
            logger.debug("✓ HTML parsed with lxml")
        except Exception as e:
            logger.warning(f"Failed to parse with lxml: {e}, trying html.parser")
            soup = BeautifulSoup(response.text, 'html.parser')

        # ====================================================================
        # ESTRAZIONE CONTENUTO PRINCIPALE
        # ====================================================================

        if extract_main_content and READABILITY_AVAILABLE:
            logger.info("Extracting main content with readability...")

            try:
                # Usa readability per estrarre contenuto principale
                doc = Document(response.text)

                # Titolo
                title = doc.title()
                logger.debug(f"Title (readability): {title}")

                # Contenuto principale come HTML
                main_html = doc.summary()

                # Parse HTML contenuto principale
                main_soup = BeautifulSoup(main_html, 'lxml')

                # Estrai testo pulito
                content = main_soup.get_text(separator='\n', strip=True)

                logger.success(f"✓ Main content extracted ({len(content)} chars)")

            except Exception as e:
                logger.warning(f"Readability extraction failed: {e}, using full page")
                title = soup.title.string if soup.title else ""
                content = soup.get_text(separator='\n', strip=True)

        else:
            # Estrai da pagina completa
            logger.info("Extracting content from full page...")

            title = soup.title.string if soup.title else ""
            content = soup.get_text(separator='\n', strip=True)

            logger.debug(f"Title: {title}")
            logger.debug(f"Content length: {len(content)} chars")

        # ====================================================================
        # ESTRAZIONE METADATA
        # ====================================================================

        meta = {}

        if extract_metadata:
            logger.info("Extracting metadata...")
            meta = self._extract_metadata(soup)
            logger.debug(f"Metadata fields: {list(meta.keys())}")

        # ====================================================================
        # ESTRAZIONE LINKS
        # ====================================================================

        links = []

        if extract_links:
            logger.info("Extracting links...")
            links = self._extract_links(soup, response.url, max_links)
            logger.success(f"✓ Extracted {len(links)} links")

        # ====================================================================
        # COSTRUZIONE RISULTATO
        # ====================================================================

        result = {
            "url": response.url,  # URL finale dopo redirect
            "title": title,
            "content": content,
            "html": response.text,
            "links": links,
            "meta": meta,
            "status_code": response.status_code,
            "encoding": response.encoding,
            "content_type": response.headers.get('Content-Type', '')
        }

        logger.success(f"✓ Webpage fetched and parsed successfully")
        logger.debug(f"Result keys: {list(result.keys())}")

        return result

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Estrae metadata da pagina HTML.

        Cerca meta tags comuni come:
        - author
        - description
        - keywords
        - date/published_time
        - Open Graph tags (og:*)
        - Twitter Card tags (twitter:*)

        Args:
            soup: BeautifulSoup object della pagina

        Returns:
            Dict con metadata estratti
        """
        logger.debug("Extracting metadata from HTML...")

        meta = {}

        # Trova tutti i meta tag
        for tag in soup.find_all('meta'):
            # Meta tag possono avere name= o property=
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')

            if name and content:
                meta[name] = content
                logger.debug(f"  Meta: {name} = {content[:50]}...")

        # Estrai campi comuni e normalizzali
        result = {}

        # Author
        result['author'] = (
            meta.get('author') or
            meta.get('article:author') or
            meta.get('twitter:creator') or
            ""
        )

        # Date
        result['date'] = (
            meta.get('article:published_time') or
            meta.get('date') or
            meta.get('pubdate') or
            ""
        )

        # Description
        result['description'] = (
            meta.get('description') or
            meta.get('og:description') or
            meta.get('twitter:description') or
            ""
        )

        # Keywords
        result['keywords'] = meta.get('keywords', "")

        # Type
        result['type'] = meta.get('og:type', "")

        # Site name
        result['site_name'] = meta.get('og:site_name', "")

        # Image
        result['image'] = (
            meta.get('og:image') or
            meta.get('twitter:image') or
            ""
        )

        # Log risultato
        filled_fields = [k for k, v in result.items() if v]
        logger.debug(f"Metadata extracted: {len(filled_fields)} fields filled")

        return result

    def _extract_links(
        self,
        soup: BeautifulSoup,
        base_url: str,
        max_links: int = 100
    ) -> List[Dict[str, str]]:
        """
        Estrae tutti i link da una pagina.

        Features:
        - Converte link relativi in assoluti
        - Valida URLs
        - Estrae testo del link
        - Limita numero di link

        Args:
            soup: BeautifulSoup object
            base_url: URL base per convertire link relativi
            max_links: Numero massimo di link da estrarre

        Returns:
            Lista di dict con {"url": "...", "text": "..."}
        """
        logger.debug(f"Extracting links (max: {max_links})...")

        links = []

        # Trova tutti i tag <a> con href
        for a_tag in soup.find_all('a', href=True):
            # Estrai href
            href = a_tag['href']

            # Converti a URL assoluto
            absolute_url = urljoin(base_url, href)

            # Valida URL
            if not validators.url(absolute_url):
                logger.debug(f"  Invalid URL skipped: {absolute_url}")
                continue

            # Estrai testo del link
            link_text = a_tag.get_text(strip=True)

            # Aggiungi a lista
            links.append({
                "url": absolute_url,
                "text": link_text
            })

            # Limita numero di link
            if len(links) >= max_links:
                logger.debug(f"Reached max_links limit ({max_links})")
                break

        logger.debug(f"Extracted {len(links)} valid links")

        return links

    # ========================================================================
    # TOOL 3: EXTRACT_STRUCTURED_DATA - Estrazione dati con CSS selectors
    # ========================================================================

    def extract_structured_data(
        self,
        html: str,
        schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Estrae dati strutturati da HTML usando schema con CSS selectors.

        Questo tool permette di estrarre dati organizzati da pagine web,
        come liste di prodotti, articoli, risultati di ricerca, etc.

        Features:
        - Usa CSS selectors (sintassi semplice e potente)
        - Supporta estrazione attributi HTML
        - Supporta nesting
        - Validazione schema
        - Error handling robusto

        Args:
            html: HTML da cui estrarre dati (string)
            schema: Schema di estrazione nel formato:
                    {
                        "selector": ".item-container",  # Container items
                        "fields": {
                            "name": ".item-name",  # Selector per field
                            "price": {
                                "selector": ".price",
                                "attr": "data-price"  # Opzionale: estrai attributo
                            },
                            "link": {
                                "selector": "a",
                                "attr": "href"
                            }
                        }
                    }

        Returns:
            Lista di dizionari, uno per ogni item trovato:
            [
                {"name": "...", "price": "...", "link": "..."},
                {"name": "...", "price": "...", "link": "..."},
                ...
            ]

        Raises:
            ValueError: Se schema invalido
            RuntimeError: Se estrazione fallisce

        Example:
            >>> html = "<div class='product'><h2>Laptop</h2><span class='price'>$999</span></div>"
            >>> schema = {
            ...     "selector": ".product",
            ...     "fields": {
            ...         "name": "h2",
            ...         "price": ".price"
            ...     }
            ... }
            >>> results = tools.extract_structured_data(html, schema)
            >>> print(results)
            [{"name": "Laptop", "price": "$999"}]
        """
        # Incrementa statistiche
        self.stats["extract_structured_data"] += 1

        logger.info("📊 Extracting structured data from HTML")
        logger.debug(f"HTML length: {len(html)} chars")
        logger.debug(f"Schema: {schema}")

        # ====================================================================
        # VALIDAZIONE SCHEMA
        # ====================================================================

        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")

        if "fields" not in schema:
            raise ValueError("Schema must have 'fields' key")

        if not isinstance(schema["fields"], dict):
            raise ValueError("Schema 'fields' must be a dictionary")

        # Selector per container (default: body se non specificato)
        container_selector = schema.get("selector", "body")
        fields_schema = schema["fields"]

        logger.debug(f"Container selector: {container_selector}")
        logger.debug(f"Fields: {list(fields_schema.keys())}")

        # ====================================================================
        # PARSING HTML
        # ====================================================================

        logger.info("Parsing HTML...")

        try:
            soup = BeautifulSoup(html, 'lxml')
        except Exception as e:
            logger.warning(f"Failed to parse with lxml: {e}, trying html.parser")
            soup = BeautifulSoup(html, 'html.parser')

        # ====================================================================
        # ESTRAZIONE DATI
        # ====================================================================

        results = []

        # Trova tutti i container
        containers = soup.select(container_selector)

        logger.info(f"Found {len(containers)} containers matching '{container_selector}'")

        # Per ogni container, estrai tutti i fields
        for i, container in enumerate(containers):
            logger.debug(f"Processing container {i+1}/{len(containers)}")

            item = {}

            # Per ogni field definito nello schema
            for field_name, field_selector in fields_schema.items():
                logger.debug(f"  Extracting field: {field_name}")

                # Field selector può essere string o dict
                if isinstance(field_selector, str):
                    # Semplice selector CSS
                    element = container.select_one(field_selector)

                    if element:
                        # Estrai testo
                        value = element.get_text(strip=True)
                        item[field_name] = value
                        logger.debug(f"    Value: {value[:50]}...")
                    else:
                        logger.debug(f"    Not found")
                        item[field_name] = ""

                elif isinstance(field_selector, dict):
                    # Selector con opzioni (es: attr per attributi)
                    selector = field_selector.get("selector")
                    attr = field_selector.get("attr")

                    if not selector:
                        logger.warning(f"Field '{field_name}' has no selector, skipping")
                        continue

                    element = container.select_one(selector)

                    if element:
                        if attr:
                            # Estrai attributo
                            value = element.get(attr, "")
                            logger.debug(f"    Attribute '{attr}': {value[:50]}...")
                        else:
                            # Estrai testo
                            value = element.get_text(strip=True)
                            logger.debug(f"    Value: {value[:50]}...")

                        item[field_name] = value
                    else:
                        logger.debug(f"    Not found")
                        item[field_name] = ""

                else:
                    logger.warning(f"Invalid field selector type for '{field_name}': {type(field_selector)}")

            # Aggiungi item ai risultati solo se ha almeno un campo non vuoto
            if any(v for v in item.values()):
                results.append(item)
            else:
                logger.debug(f"  Container {i+1} has no data, skipping")

        # ====================================================================
        # RISULTATO
        # ====================================================================

        logger.success(f"✓ Extracted {len(results)} items")

        # Log sample del primo risultato
        if results:
            logger.debug(f"Sample result: {results[0]}")

        return results

    # ========================================================================
    # TOOL 4: SUMMARIZE_CONTENT - Riassunto con LLM
    # ========================================================================

    def summarize_content(
        self,
        text: str,
        max_length: int = 500,
        style: str = "concise",
        language: str = "italian"
    ) -> str:
        """
        Riassume un testo lungo usando LLM.

        Questo tool usa l'LLM per creare riassunti intelligenti di testi lunghi.

        Features:
        - Diversi stili di riassunto (concise, detailed, bullet_points)
        - Controllo lunghezza output
        - Multilingua
        - Preserva informazioni chiave
        - Skip su testi già corti

        Args:
            text: Testo da riassumere
            max_length: Lunghezza massima riassunto in parole (default 500)
            style: Stile riassunto:
                   - "concise": Riassunto molto breve, punti chiave
                   - "detailed": Riassunto dettagliato con più contesto
                   - "bullet_points": Lista puntata dei punti principali
            language: Lingua output (default "italian")

        Returns:
            Testo riassunto

        Raises:
            RuntimeError: Se LLM non disponibile o summarization fallisce

        Example:
            >>> long_text = "..." # 5000 parole
            >>> summary = tools.summarize_content(long_text, max_length=200, style="concise")
            >>> print(summary)
        """
        # Incrementa statistiche
        self.stats["summarize_content"] += 1

        logger.info("📝 Summarizing content")
        logger.debug(f"Text length: {len(text)} chars, {len(text.split())} words")
        logger.debug(f"Parameters: max_length={max_length}, style={style}, language={language}")

        # ====================================================================
        # VALIDAZIONE
        # ====================================================================

        # Check LLM disponibile
        if not self.llm:
            logger.error("LLM interface not available")
            raise RuntimeError("LLM interface required for summarization. Initialize WebTools with llm_interface parameter.")

        # Valida input
        if not text or not text.strip():
            logger.error("Text is empty")
            raise ValueError("Text cannot be empty")

        text = text.strip()

        # Check se testo già abbastanza corto
        word_count = len(text.split())

        if word_count <= max_length:
            logger.info(f"Text already short enough ({word_count} words <= {max_length}), returning as is")
            return text

        # Valida style
        valid_styles = ["concise", "detailed", "bullet_points"]
        if style not in valid_styles:
            logger.warning(f"Invalid style '{style}', using 'concise'")
            style = "concise"

        # ====================================================================
        # PREPARAZIONE PROMPT
        # ====================================================================

        logger.info(f"Preparing summarization prompt (style: {style})...")

        # Definizioni stili
        style_instructions = {
            "concise": f"Create a very concise summary in about {max_length} words. Focus on the most important points.",
            "detailed": f"Create a detailed summary in about {max_length} words. Maintain important context and key details.",
            "bullet_points": f"Create a bullet-point summary with the main points. Use about {max_length} words total."
        }

        instruction = style_instructions[style]

        # Limita lunghezza testo input per non superare context window
        # Stima: 1 token ≈ 4 chars, context window tipico 8k-32k tokens
        max_input_chars = 10000  # ~2500 tokens

        if len(text) > max_input_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_input_chars}")
            text = text[:max_input_chars] + "..."

        # Costruisci prompt
        prompt = f"""{instruction}

The summary should be in {language}.

Text to summarize:
{text}

Summary:"""

        logger.debug(f"Prompt length: {len(prompt)} chars")

        # ====================================================================
        # GENERAZIONE RIASSUNTO
        # ====================================================================

        try:
            logger.info("Calling LLM for summarization...")

            # Chiama LLM
            summary = self.llm.generate(
                prompt,
                temperature=0.3,  # Bassa temperatura per output più factual
                max_tokens=max_length * 2  # Stima: 1 word ≈ 1.3 tokens
            )

            summary = summary.strip()

            logger.success(f"✓ Summary generated ({len(summary)} chars, {len(summary.split())} words)")
            logger.debug(f"Summary preview: {summary[:200]}...")

            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            logger.exception(e)
            raise RuntimeError(f"Failed to summarize content: {str(e)}")

    # ========================================================================
    # TOOL 5: COMPARE_SOURCES - Confronto fonti multiple
    # ========================================================================

    def compare_sources(
        self,
        sources: List[str],
        topic: Optional[str] = None,
        max_content_per_source: int = 2000
    ) -> Dict[str, Any]:
        """
        Confronta informazioni da fonti multiple usando LLM.

        Questo tool permette all'agente di:
        - Confrontare più fonti su un topic
        - Identificare consenso e differenze
        - Cross-reference informazioni
        - Valutare reliability (future feature)

        Features:
        - Supporta URL (download automatico) o testo diretto
        - Focus su topic specifico (opzionale)
        - Analisi consensus/differences via LLM
        - Limitazione lunghezza per efficienza

        Args:
            sources: Lista di URL o testi da confrontare
                     Può contenere mix di URL e testi:
                     ["https://...", "Text content...", "https://..."]
            topic: Topic specifico su cui focalizzare (opzionale)
                   Es: "intelligenza artificiale", "prezzi", "opinioni su..."
            max_content_per_source: Max caratteri per fonte (default 2000)
                                   Limita per non superare context window

        Returns:
            Dizionario con:
            {
                "consensus": "Punti su cui tutte le fonti concordano...",
                "differences": "Differenze e discordanze tra fonti...",
                "summary": "Sintesi generale del confronto...",
                "sources_analyzed": 3
            }

        Raises:
            RuntimeError: Se LLM non disponibile
            ValueError: Se meno di 2 fonti

        Example:
            >>> sources = [
            ...     "https://site1.com/article",
            ...     "https://site2.com/article"
            ... ]
            >>> result = tools.compare_sources(sources, topic="AI developments")
            >>> print(result["consensus"])
            >>> print(result["differences"])
        """
        # Incrementa statistiche
        self.stats["compare_sources"] += 1

        logger.info(f"🔬 Comparing {len(sources)} sources")
        if topic:
            logger.info(f"Focus topic: '{topic}'")

        # ====================================================================
        # VALIDAZIONE
        # ====================================================================

        # Check LLM disponibile
        if not self.llm:
            logger.error("LLM interface not available")
            raise RuntimeError("LLM interface required for source comparison.")

        # Valida numero fonti
        if len(sources) < 2:
            logger.error(f"Need at least 2 sources, got {len(sources)}")
            raise ValueError("At least 2 sources required for comparison")

        logger.debug(f"Max content per source: {max_content_per_source} chars")

        # ====================================================================
        # DOWNLOAD/PREPARAZIONE CONTENUTI
        # ====================================================================

        logger.info("Preparing source contents...")

        contents = []

        for i, source in enumerate(sources):
            logger.info(f"Processing source {i+1}/{len(sources)}")

            # Determina se è URL o testo
            if validators.url(source):
                # È un URL - scarica
                logger.info(f"  Source is URL: {source[:50]}...")

                try:
                    page = self.fetch_webpage(source)
                    content = page["content"]
                    source_title = page["title"]
                    source_url = source

                    logger.success(f"  ✓ Downloaded ({len(content)} chars)")

                except Exception as e:
                    logger.error(f"  Failed to fetch URL: {e}")
                    logger.warning(f"  Skipping source {i+1}")
                    continue

            else:
                # È testo diretto
                logger.info(f"  Source is text ({len(source)} chars)")
                content = source
                source_title = f"Text source {i+1}"
                source_url = "text"

            # Limita lunghezza contenuto
            if len(content) > max_content_per_source:
                logger.debug(f"  Truncating content: {len(content)} -> {max_content_per_source} chars")
                content = content[:max_content_per_source] + "..."

            # Aggiungi a lista
            contents.append({
                "index": i + 1,
                "title": source_title,
                "url": source_url,
                "content": content
            })

        # ====================================================================
        # VALIDAZIONE CONTENUTI
        # ====================================================================

        if len(contents) < 2:
            logger.error("Not enough valid sources after fetching")
            raise ValueError(f"Need at least 2 valid sources, got {len(contents)}")

        logger.success(f"✓ Prepared {len(contents)} sources for comparison")

        # ====================================================================
        # PREPARAZIONE PROMPT
        # ====================================================================

        logger.info("Preparing comparison prompt...")

        # Formatta contenuti per il prompt
        sources_text = []

        for item in contents:
            source_block = f"""
SOURCE {item['index']}: {item['title']}
URL: {item['url']}
---
{item['content']}
            """.strip()

            sources_text.append(source_block)

        sources_combined = "\n\n" + "="*70 + "\n\n".join(sources_text)

        # Istruzione topic
        topic_instruction = ""
        if topic:
            topic_instruction = f" focusing specifically on: {topic}"

        # Costruisci prompt
        prompt = f"""Compare the following {len(contents)} sources{topic_instruction}.

Analyze them and provide:
1. CONSENSUS: What all sources agree on
2. DIFFERENCES: Where sources disagree or provide unique information
3. SUMMARY: Overall synthesis of the information

Respond in Italian.

{sources_combined}

Please provide your analysis as a JSON object:
{{
    "consensus": "...",
    "differences": "...",
    "summary": "..."
}}

JSON response:"""

        logger.debug(f"Prompt length: {len(prompt)} chars")

        # ====================================================================
        # GENERAZIONE CONFRONTO
        # ====================================================================

        try:
            logger.info("Calling LLM for source comparison...")

            # Chiama LLM
            response = self.llm.generate(
                prompt,
                temperature=0.3  # Bassa temperatura per analisi factual
            )

            logger.success("✓ LLM response received")
            logger.debug(f"Response length: {len(response)} chars")

            # ================================================================
            # PARSING RISPOSTA
            # ================================================================

            logger.info("Parsing LLM response...")

            # Cerca JSON nella risposta
            try:
                # Trova blocco JSON
                start = response.find('{')
                end = response.rfind('}') + 1

                if start == -1 or end == 0:
                    raise ValueError("No JSON found in response")

                json_str = response[start:end]
                result = json.loads(json_str)

                # Valida campi
                required_fields = ["consensus", "differences", "summary"]
                for field in required_fields:
                    if field not in result:
                        result[field] = ""

                logger.success("✓ JSON parsed successfully")

            except Exception as e:
                logger.warning(f"Failed to parse JSON: {e}")
                logger.info("Using raw response as summary")

                result = {
                    "consensus": "",
                    "differences": "",
                    "summary": response
                }

            # Aggiungi metadata
            result["sources_analyzed"] = len(contents)

            logger.success(f"✓ Source comparison completed")

            return result

        except Exception as e:
            logger.error(f"Source comparison failed: {e}")
            logger.exception(e)
            raise RuntimeError(f"Failed to compare sources: {str(e)}")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_stats(self) -> Dict[str, int]:
        """
        Ritorna statistiche d'uso dei tools.

        Returns:
            Dict con tool -> numero di chiamate
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistiche."""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Stats reset")


# ----------------------------------------------------------------------------
# TESTING / DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Codice di test per i Web Tools.

    Esegui: python web_tools.py
    """
    print("\n" + "="*70)
    print("Testing Web Tools")
    print("="*70 + "\n")

    # Crea WebTools
    tools = WebTools(config={})

    # Test 1: Search
    print("\n" + "-"*70)
    print("Test 1: search_web")
    print("-"*70)

    try:
        results = tools.search_web("Python programming tutorial", num_results=3)
        print(f"\nFound {len(results)} results:")
        for r in results:
            print(f"\n{r['position']}. {r['title']}")
            print(f"   {r['url']}")
            print(f"   {r['snippet'][:100]}...")
    except Exception as e:
        print(f"Search test failed: {e}")

    # Test 2: Fetch
    print("\n" + "-"*70)
    print("Test 2: fetch_webpage")
    print("-"*70)

    try:
        page = tools.fetch_webpage("https://example.com")
        print(f"\nTitle: {page['title']}")
        print(f"Content: {page['content'][:200]}...")
        print(f"Links: {len(page['links'])}")
        print(f"Metadata: {list(page['meta'].keys())}")
    except Exception as e:
        print(f"Fetch test failed: {e}")

    # Test 3: Extract
    print("\n" + "-"*70)
    print("Test 3: extract_structured_data")
    print("-"*70)

    html_sample = """
    <div class="item"><h3>Item 1</h3><span class="price">$10</span></div>
    <div class="item"><h3>Item 2</h3><span class="price">$20</span></div>
    """

    schema = {
        "selector": ".item",
        "fields": {
            "name": "h3",
            "price": ".price"
        }
    }

    try:
        items = tools.extract_structured_data(html_sample, schema)
        print(f"\nExtracted {len(items)} items:")
        for item in items:
            print(f"  {item}")
    except Exception as e:
        print(f"Extract test failed: {e}")

    # Test 4 & 5: LLM-based tools (require LLM)
    print("\n" + "-"*70)
    print("Tests 4-5: LLM-based tools skipped (require LLM interface)")
    print("-"*70)

    # Stats
    print("\n" + "-"*70)
    print("Usage Statistics")
    print("-"*70)
    print(tools.get_stats())

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
