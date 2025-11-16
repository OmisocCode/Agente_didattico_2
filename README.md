# 🌐 Agente Web Scraper Intelligente

Un progetto didattico di livello **intermedio** per imparare a costruire agenti AI che interagiscono con il web. Questo agente può cercare informazioni online, estrarre dati da siti web e sintetizzare risultati in modo intelligente.

---

## 🎯 Obiettivi di Apprendimento

Questo progetto ti insegna concetti avanzati degli agenti AI:

### 1. **Tool Calling (Function Calling)**
Come l'agente decide QUALE strumento usare per risolvere un compito:
- Analisi della query utente
- Selezione del tool appropriato
- Chiamata dello strumento con parametri corretti
- Gestione dei risultati

### 2. **Web Interaction**
Come l'agente interagisce con il mondo esterno:
- Ricerca su web (Google/Bing)
- Scraping di pagine HTML
- Parsing e pulizia dati
- Gestione errori di rete

### 3. **Chain of Thought (CoT)**
Come l'agente "pensa" prima di agire:
- Ragionamento esplicito sui passi da seguire
- Pianificazione multi-step
- Auto-correzione se un approccio fallisce

### 4. **Information Synthesis**
Come l'agente combina informazioni da fonti multiple:
- Aggregazione dati
- Rimozione duplicati
- Sintesi e riassunto
- Citazione delle fonti

### 5. **Gestione Asincrona**
Come gestire operazioni che richiedono tempo:
- Richieste HTTP asincrone
- Rate limiting
- Timeout e retry logic
- Caching dei risultati

---

## 🗂️ Struttura del Progetto

```
web-scraper-agent/
│
├── 📄 CODICE PRINCIPALE
│   ├── agent.py                 # Agente principale con tool calling
│   ├── web_tools.py            # Tools per web search e scraping
│   ├── html_parser.py          # Parser HTML intelligente
│   ├── llm_interface.py        # Interfaccia unificata per LLM
│   ├── main.py                 # CLI interattiva
│   └── requirements.txt        # Dipendenze
│
├── 📚 CONFIGURAZIONE
│   ├── config.yaml             # Configurazione agente e tools
│   └── .env.example            # Template variabili ambiente
│
├── 🧪 TESTING
│   ├── test_agent.py           # Test dell'agente
│   ├── test_tools.py           # Test dei singoli tools
│   └── test_integration.py     # Test end-to-end
│
├── 📖 DOCUMENTAZIONE
│   ├── README.md               # Questa guida
│   ├── TUTORIAL.md             # Tutorial passo-passo
│   ├── ARCHITECTURE.md         # Architettura dettagliata
│   └── EXAMPLES.md             # Esempi pratici
│
└── 📁 ESEMPI
    ├── queries/                # Query di esempio
    └── cached_results/         # Cache risultati per testing
```

---

## 🛠️ Tools Implementati

L'agente ha accesso a diversi "tools" (strumenti) specializzati:

### 1. **search_web(query: str, num_results: int)**
```python
"""
Cerca informazioni sul web usando un motore di ricerca.

Quando usarlo:
- Query informative ("Chi è Elon Musk?")
- Ricerca di notizie recenti
- Trovare fonti su un topic

Esempio:
>>> search_web("Ultime novità AI 2024", num_results=5)
[
  {
    "title": "Le 10 innovazioni AI del 2024",
    "url": "https://...",
    "snippet": "Quest'anno ha visto progressi..."
  },
  ...
]
"""
```

**Tecnologie**: API DuckDuckGo/SerpAPI/Google Custom Search

### 2. **fetch_webpage(url: str)**
```python
"""
Scarica e parsifica il contenuto di una pagina web.

Quando usarlo:
- Estrarre contenuto da URL specifico
- Leggere articoli completi
- Analizzare pagine web

Esempio:
>>> fetch_webpage("https://example.com/article")
{
  "title": "Titolo articolo",
  "content": "Testo completo pulito...",
  "author": "Mario Rossi",
  "date": "2024-01-15",
  "links": [...]
}
"""
```

**Tecnologie**: requests, BeautifulSoup4, readability-lxml

### 3. **extract_structured_data(html: str, schema: dict)**
```python
"""
Estrae dati strutturati da HTML usando selettori CSS.

Quando usarlo:
- Estrarre tabelle
- Scaricare liste di prodotti
- Parsing dati strutturati

Esempio:
>>> extract_structured_data(html, {
...   "products": {
...     "selector": ".product-card",
...     "fields": {
...       "name": ".product-name",
...       "price": ".product-price"
...     }
...   }
... })
[{"name": "Laptop", "price": "999.99"}, ...]
"""
```

**Tecnologie**: BeautifulSoup4, CSS selectors, XPath

### 4. **summarize_content(text: str, max_length: int)**
```python
"""
Riassume testo lungo usando l'LLM.

Quando usarlo:
- Sintetizzare articoli lunghi
- Estrarre punti chiave
- Creare abstract

Esempio:
>>> summarize_content(long_article, max_length=200)
"L'articolo discute 3 punti principali: 1) ..."
"""
```

**Tecnologie**: LLM (Ollama/OpenAI)

### 5. **compare_sources(sources: List[str])**
```python
"""
Confronta informazioni da fonti multiple.

Quando usarlo:
- Fact-checking
- Identificare consenso/disaccordo
- Cross-reference

Esempio:
>>> compare_sources([url1, url2, url3])
{
  "consensus": "Tutti concordano che...",
  "differences": "Fonte 1 afferma X, Fonte 2 afferma Y",
  "reliability": {"source1": 0.9, "source2": 0.7}
}
"""
```

**Tecnologie**: LLM reasoning, similarity metrics

---

## 🧠 Architettura dell'Agente

### Il Ciclo Tool Calling

```
┌─────────────────────────────────────────────────────────────┐
│                    UTENTE FA UNA QUERY                      │
│   "Trova le ultime notizie su SpaceX e riassumile"         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: ANALISI QUERY                                      │
│  L'agente analizza la richiesta e identifica:              │
│  - Intent: "ricerca + sintesi"                              │
│  - Entità: "SpaceX", "notizie"                             │
│  - Azioni necessarie: [search, fetch, summarize]           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: PIANIFICAZIONE (Chain of Thought)                 │
│  L'agente crea un piano:                                    │
│  1. search_web("SpaceX news 2024", num=5)                  │
│  2. fetch_webpage(top_3_results)                            │
│  3. summarize_content(combined_articles)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: ESECUZIONE TOOLS                                   │
│  Per ogni tool nel piano:                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 3a. Valida parametri                                  │  │
│  │ 3b. Esegui tool                                       │  │
│  │ 3c. Gestisci errori/retry                            │  │
│  │ 3d. Memorizza risultato                              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: SINTESI FINALE                                     │
│  L'agente combina i risultati dei tools e genera           │
│  una risposta coerente citando le fonti                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    RISPOSTA ALL'UTENTE                      │
│  "Ecco le ultime notizie su SpaceX:                        │
│   1. Lancio Starship... [fonte: ...]                       │
│   2. Contratto NASA... [fonte: ...]                        │
│   Riassunto: ..."                                           │
└─────────────────────────────────────────────────────────────┘
```

### Componenti Chiave

#### 1. **ToolRegistry**
```python
class ToolRegistry:
    """
    Registro di tutti i tools disponibili.
    Gestisce registrazione, discovery e invocazione.
    """
    
    def register_tool(self, name: str, function: Callable, 
                     description: str, parameters: dict):
        """Registra un nuovo tool"""
        
    def get_tool_descriptions(self) -> str:
        """Genera descrizioni per il prompt LLM"""
        
    def call_tool(self, name: str, **kwargs) -> Any:
        """Esegue un tool con validazione"""
```

#### 2. **WebScraperAgent**
```python
class WebScraperAgent:
    """
    Agente principale che orchestra i tools.
    """
    
    def __init__(self, llm_model: str = "llama3.2"):
        self.llm = LLMInterface(llm_model)
        self.tools = ToolRegistry()
        self.memory = ConversationMemory()
        
    def process_query(self, query: str) -> str:
        """
        Pipeline principale:
        1. Analizza query
        2. Genera piano
        3. Esegue tools
        4. Sintetizza risultato
        """
        
    def _generate_plan(self, query: str) -> List[ToolCall]:
        """Chain of Thought: pianifica azioni"""
        
    def _execute_plan(self, plan: List[ToolCall]) -> List[Any]:
        """Esegue il piano step-by-step"""
        
    def _synthesize_results(self, results: List[Any]) -> str:
        """Combina risultati in risposta finale"""
```

#### 3. **LLMInterface**
```python
class LLMInterface:
    """
    Interfaccia unificata per diversi LLM.
    Supporta: Ollama, OpenAI, Anthropic.
    """
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generazione semplice"""
        
    def chat(self, messages: List[Dict]) -> str:
        """Conversazione con contesto"""
        
    def function_call(self, query: str, 
                     tools: List[Dict]) -> Dict:
        """Tool calling nativo (se supportato)"""
```

---

## 🔄 Flusso di Esecuzione Dettagliato

### Esempio: "Confronta i prezzi dei laptop su 3 siti"

```python
# 1. ANALISI QUERY
agent.process_query("Confronta i prezzi dei laptop su Amazon, eBay e MediaWorld")

# Internamente l'agente:

# 2. PIANIFICAZIONE
plan = [
    ToolCall("search_web", {"query": "laptop prices Amazon"}),
    ToolCall("search_web", {"query": "laptop prices eBay"}),
    ToolCall("search_web", {"query": "laptop prices MediaWorld"}),
    ToolCall("fetch_webpage", {"url": result1.url}),
    ToolCall("fetch_webpage", {"url": result2.url}),
    ToolCall("fetch_webpage", {"url": result3.url}),
    ToolCall("extract_structured_data", {
        "html": page1,
        "schema": laptop_schema
    }),
    # ... stessa cosa per page2 e page3
    ToolCall("compare_sources", {"sources": [data1, data2, data3]})
]

# 3. ESECUZIONE
results = agent._execute_plan(plan)
# [search_results, ..., comparison]

# 4. SINTESI
final_response = agent._synthesize_results(results)
# "Ho confrontato i prezzi su 3 siti:
#  - Amazon: Laptop X a €899 [link]
#  - eBay: Laptop X a €850 [link]
#  - MediaWorld: Laptop X a €920 [link]
#  Migliore offerta: eBay (-€49 vs Amazon)"
```

---

## 📋 Requisiti e Installazione

### Prerequisiti

```bash
# Python 3.8+
python --version

# Ollama (per LLM locale)
ollama --version

# Variabili ambiente (opzionale)
# Per API esterne: Google Custom Search, SerpAPI, etc.
```

### Installazione

```bash
# 1. Clone/Download progetto
cd web-scraper-agent

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Configura (opzionale)
cp .env.example .env
# Edita .env con le tue API keys

# 4. Configura tools
# Edita config.yaml per abilitare/disabilitare tools

# 5. Scarica modello LLM
ollama pull llama3.2

# 6. Test setup
python test_agent.py
```

### Dipendenze Principali

```txt
# LLM
ollama                    # LLM locale
openai                    # OpenAI API (opzionale)
anthropic                 # Claude API (opzionale)

# Web Scraping
requests                  # HTTP client
beautifulsoup4            # HTML parsing
lxml                      # XML/HTML parser veloce
readability-lxml          # Estrazione contenuto principale
selenium                  # Browser automation (opzionale)

# Search
duckduckgo-search        # Search senza API key
google-api-python-client # Google Custom Search (opzionale)

# Utilities
pyyaml                    # Config files
python-dotenv             # Variabili ambiente
validators                # Validazione URL
tqdm                      # Progress bars
tenacity                  # Retry logic

# Testing
pytest                    # Framework testing
pytest-asyncio            # Test asincroni
responses                 # Mock HTTP requests
```

---

## 💻 Utilizzo

### CLI Interattiva

```bash
python main.py
```

```
🌐 AGENTE WEB SCRAPER INTELLIGENTE
═══════════════════════════════════════

Comandi disponibili:
  search <query>        - Cerca sul web
  fetch <url>          - Scarica una pagina
  extract <url>        - Estrai dati strutturati
  compare <urls>       - Confronta più fonti
  help                 - Mostra aiuto
  exit                 - Esci

💬 Tu: search ultime notizie AI Italia

🤔 Analizzo la richiesta...
📋 Piano:
  1. search_web("AI news Italy 2024")
  2. fetch_webpage(top_3_results)
  3. summarize_content(combined)

⚙️  Esecuzione...
✓ search_web completato (5 risultati)
✓ fetch_webpage completato (3 pagine)
✓ summarize_content completato

🤖 Risposta:
Ho trovato 5 articoli recenti sull'AI in Italia:

1. "Investimenti AI crescono del 40%" - Il Sole 24 Ore
   https://...
   Sintesi: Le aziende italiane hanno aumentato...

2. "Milano hub europeo AI" - Corriere della Sera
   https://...
   Sintesi: Milano si posiziona come...

[Fonti consultate: 3 articoli, ultimo aggiornamento: oggi]
```

### Uso Programmatico

```python
from agent import WebScraperAgent

# Inizializza agente
agent = WebScraperAgent(
    llm_model="llama3.2",
    max_tools_per_query=5,
    enable_caching=True
)

# Query semplice
result = agent.query("Qual è il prezzo attuale del Bitcoin?")
print(result)

# Query complessa con tool multipli
result = agent.query(
    "Trova le migliori offerte di voli Milano-New York per Dicembre, "
    "confronta 3 siti diversi e dammi i pro/contro di ciascuna opzione"
)
print(result)

# Accesso allo storico
print(agent.get_execution_history())
# [
#   {
#     "query": "...",
#     "plan": [...],
#     "tools_executed": [...],
#     "result": "...",
#     "timestamp": "..."
#   }
# ]
```

---

## 🎓 Concetti Avanzati

### 1. Chain of Thought (CoT) Prompting

```python
# L'agente usa CoT per pianificare azioni
cot_prompt = f"""
Query utente: {user_query}

Pensa step-by-step a come rispondere:

1. Qual è l'obiettivo della query?
2. Quali informazioni servono?
3. Quali tools posso usare?
4. In che ordine li uso?
5. Come combino i risultati?

Genera un piano di azione in JSON:
{{
  "reasoning": "...",
  "steps": [
    {{"tool": "search_web", "params": {{}}, "why": "..."}},
    ...
  ]
}}
"""
```

### 2. Tool Selection

```python
# L'agente decide quale tool usare
def select_tool(self, intent: str, context: dict) -> str:
    """
    Euristiche per selezione tool:
    
    - "cerca", "trova" → search_web
    - URL specifico → fetch_webpage
    - "confronta", "differenze" → compare_sources
    - "riassumi", "sintetizza" → summarize_content
    - "estrai tabella", "lista" → extract_structured_data
    """
    
    # Oppure usa LLM per decidere
    decision_prompt = f"""
    Dato l'intent '{intent}' e il contesto {context},
    quale tool è più appropriato?
    
    Tools disponibili:
    {self.tools.get_descriptions()}
    
    Rispondi con il nome del tool e perché.
    """
```

### 3. Error Handling & Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

class WebTools:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def fetch_webpage(self, url: str) -> dict:
        """
        Retry automatico in caso di:
        - Network errors
        - Timeout
        - 5xx errors
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return self._parse_html(response.text)
```

### 4. Result Caching

```python
class CachedTools:
    """
    Cache risultati per evitare richieste duplicate
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache = {}
        self.cache_dir = cache_dir
    
    def search_web(self, query: str, **kwargs) -> List[dict]:
        # Crea cache key
        cache_key = hashlib.md5(
            f"{query}{kwargs}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Esegui ricerca
        results = self._do_search(query, **kwargs)
        
        # Salva in cache
        self.cache[cache_key] = results
        return results
```

---

## 🧪 Testing

### Test Unitari

```bash
# Test singoli tools
pytest test_tools.py -v

# Test agente
pytest test_agent.py -v

# Test integration
pytest test_integration.py -v

# Coverage
pytest --cov=. --cov-report=html
```

### Test Esempio

```python
def test_search_tool():
    """Test search_web tool"""
    tools = WebTools()
    
    results = tools.search_web("Python programming", num_results=5)
    
    assert len(results) == 5
    assert all("url" in r for r in results)
    assert all("title" in r for r in results)
    
def test_agent_planning():
    """Test che l'agente generi un piano sensato"""
    agent = WebScraperAgent()
    
    query = "Trova i migliori ristoranti a Roma"
    plan = agent._generate_plan(query)
    
    # Dovrebbe usare search_web
    assert any(step.tool == "search_web" for step in plan)
    
def test_end_to_end():
    """Test completo query → risultato"""
    agent = WebScraperAgent()
    
    result = agent.query("Qual è la capitale della Francia?")
    
    assert "parigi" in result.lower()
    assert len(agent.execution_history) > 0
```

---

## 📖 Esempi Pratici

### Esempio 1: Ricerca e Sintesi

```python
# Query: "Ultimi sviluppi sulla fusione nucleare"

# Piano generato:
# 1. search_web("nuclear fusion breakthrough 2024")
# 2. fetch_webpage(top_3_results)
# 3. summarize_content(combined_text)

# Risultato:
"""
Ho trovato 3 articoli recenti sulla fusione nucleare:

1. "Record energetico al NIF" - Nature
   https://nature.com/...
   Il National Ignition Facility ha raggiunto un guadagno netto...

2. "ITER anticipa timeline" - Science
   https://science.org/...
   Il reattore ITER potrebbe essere operativo prima del...

Sintesi: Progressi significativi sia negli USA (NIF) che in Europa (ITER)...

[Consultate 3 fonti | Ultimo aggiornamento: 2 ore fa]
"""
```

### Esempio 2: Confronto Prodotti

```python
# Query: "Confronta iPhone 15 vs Samsung S24"

# Piano:
# 1. search_web("iPhone 15 specs review")
# 2. search_web("Samsung S24 specs review")
# 3. fetch_webpage(official_spec_pages)
# 4. extract_structured_data(specs_tables)
# 5. compare_sources([iphone_data, samsung_data])

# Risultato:
"""
Confronto iPhone 15 vs Samsung Galaxy S24:

DISPLAY:
- iPhone 15: 6.1" OLED 2556x1179 [fonte: apple.com]
- Samsung S24: 6.2" AMOLED 2340x1080 [fonte: samsung.com]
→ Vantaggio: iPhone (risoluzione superiore)

FOTOCAMERA:
- iPhone 15: 48MP principale + 12MP ultra-wide
- Samsung S24: 50MP principale + 12MP ultra-wide + 10MP tele
→ Vantaggio: Samsung (lente aggiuntiva)

PREZZO:
- iPhone 15: €979 [fonte: amazon.it]
- Samsung S24: €899 [fonte: unieuro.it]
→ Vantaggio: Samsung (€80 meno costoso)

RACCOMANDAZIONE: Dipende dalle priorità...
"""
```

### Esempio 3: Analisi Tendenze

```python
# Query: "Analizza le tendenze del mercato AI negli ultimi 3 mesi"

# Piano:
# 1. search_web("AI market trends Q4 2024")
# 2. search_web("AI investment news october november december 2024")
# 3. fetch_webpage(industry_reports)
# 4. extract_structured_data(investment_figures)
# 5. summarize_content(trend_analysis)

# Risultato con grafici testuali
"""
TENDENZE MERCATO AI - Q4 2024

INVESTIMENTI:
Ottobre: $12.5B ████████████░░░░
Novembre: $15.2B ███████████████░
Dicembre: $18.1B ████████████████████ (nuovo record)

TOP SETTORI:
1. Healthcare AI: $5.2B (+45% vs Q3)
2. Enterprise AI: $4.8B (+32%)
3. Autonomous Systems: $3.1B (+28%)

INSIGHTS CHIAVE:
• Crescita accelerata del 44% rispetto a Q3
• Focus su AI generativa per enterprise
• Europa supera Asia in investimenti per prima volta

[Fonti: 12 report analizzati | Periodo: Ott-Dic 2024]
"""
```

---

## 🚀 Estensioni e Miglioramenti

### Livello Intermedio

- [ ] **Multi-language support**: Ricerca in lingue diverse
- [ ] **Image search**: Integra ricerca immagini
- [ ] **PDF extraction**: Scarica e analizza PDF
- [ ] **API integration**: Weather, finance, news APIs

### Livello Avanzato

- [ ] **Async execution**: Parallelizza richieste web
- [ ] **Browser automation**: Usa Selenium per siti dinamici
- [ ] **ML-based extraction**: Usa NLP per estrarre entities
- [ ] **Knowledge graph**: Costruisci grafo delle informazioni

### Livello Produzione

- [ ] **Web UI**: Frontend React/Streamlit
- [ ] **Authentication**: Sistema di login
- [ ] **API REST**: Esponi agente via API
- [ ] **Database**: Salva queries e risultati
- [ ] **Monitoring**: Metrics, logging, alerting

---

## ⚠️ Limitazioni e Note Legali

### Limitazioni Tecniche

- Rate limiting dei siti web
- Siti con JavaScript pesante (serve Selenium)
- CAPTCHAs e anti-bot measures
- Paywall e contenuti protetti

### Note Legali

⚖️ **IMPORTANTE**: Rispetta sempre:

- `robots.txt` dei siti web
- Terms of Service
- Rate limits
- Copyright e proprietà intellettuale
- Privacy e GDPR

**Questo progetto è SOLO per scopi educativi.**

---

## 🤝 Confronto con Progetto 1

| Aspetto | Progetto 1 (File Agent) | Progetto 2 (Web Scraper) |
|---------|------------------------|--------------------------|
| **Complessità** | Base | Intermedio |
| **Tools** | 1 (file reading) | 5+ (web tools) |
| **I/O** | File system | Internet |
| **Errori** | File not found | Network, timeout, parsing |
| **Caching** | Non necessario | Essenziale |
| **Async** | No | Consigliato |
| **Tool Selection** | Fisso | Dinamico |
| **Planning** | Singolo step | Multi-step |

---

## 📚 Risorse Utili

- **Web Scraping**: https://realpython.com/beautiful-soup-web-scraper-python/
- **Tool Calling**: https://platform.openai.com/docs/guides/function-calling
- **BeautifulSoup Docs**: https://www.crummy.com/software/BeautifulSoup/
- **Requests Docs**: https://requests.readthedocs.io/
- **Async Python**: https://realpython.com/async-io-python/

---

## 🎯 Prossimi Passi

Dopo questo progetto, puoi passare al **Progetto 3: Sistema Multi-Agente** che combina:
- Più agenti specializzati
- Orchestrazione e comunicazione inter-agente
- Task delegation
- Collaborative problem solving

---

**Buon coding! 🚀**

*Ricorda: questo è un progetto didattico. Usa responsabilmente le tecniche apprese e rispetta sempre i ToS dei servizi che usi.*
