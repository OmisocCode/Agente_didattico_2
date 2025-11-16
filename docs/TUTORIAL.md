# Tutorial: Web Scraper Agent

## Indice
1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Configurazione](#configurazione)
4. [Primi Passi](#primi-passi)
5. [Modalità Interattiva](#modalità-interattiva)
6. [Esempi Pratici](#esempi-pratici)
7. [Troubleshooting](#troubleshooting)

---

## Introduzione

Il **Web Scraper Agent** è un agente AI intelligente che può:
- 🔍 Cercare informazioni sul web
- 📄 Scaricare e analizzare pagine web
- 📊 Estrarre dati strutturati
- 📝 Riassumere contenuti lunghi
- 🔄 Confrontare fonti multiple

L'agente utilizza **Chain of Thought** per pianificare autonomamente
quali strumenti usare per rispondere alle tue domande.

---

## Installazione

### Prerequisiti
- Python 3.8 o superiore
- pip (package manager Python)
- Almeno uno tra:
  - Ollama installato localmente (consigliato per iniziare)
  - Account OpenAI con API key
  - Account Groq con API key

### Passo 1: Clonare il Repository
```bash
git clone <repository-url>
cd Agente_didattico_2
```

### Passo 2: Creare Virtual Environment (Consigliato)
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Passo 3: Installare le Dipendenze
```bash
pip install -r requirements.txt
```

Questo installerà:
- `ollama`: Client per Ollama (LLM locale)
- `openai`: Client per OpenAI GPT
- `groq`: Client per Groq (LLM ultra-veloce)
- `beautifulsoup4`, `lxml`: Parsing HTML
- `duckduckgo-search`: Ricerca web senza API
- `requests`: HTTP client
- `pyyaml`: Configurazione YAML
- `loguru`: Logging avanzato
- E altre dipendenze...

### Passo 4: Verificare l'Installazione
```bash
python main.py --help
```

Dovresti vedere l'help con le opzioni disponibili.

---

## Configurazione

### Opzione 1: Ollama (Più Semplice per Iniziare)

1. **Installa Ollama**
   ```bash
   # Linux
   curl https://ollama.ai/install.sh | sh

   # Mac
   brew install ollama

   # Windows: scarica da https://ollama.ai
   ```

2. **Scarica un modello**
   ```bash
   ollama pull llama3.1:8b
   ```

3. **Configura il file .env**
   ```bash
   cp .env.example .env
   ```

   Modifica `.env`:
   ```env
   # LLM Provider
   LLM_PROVIDER=ollama
   LLM_MODEL=llama3.1:8b
   OLLAMA_BASE_URL=http://localhost:11434
   ```

4. **Pronto!** Puoi usare l'agente:
   ```bash
   python main.py
   ```

### Opzione 2: OpenAI GPT

1. **Ottieni API Key**
   - Vai su https://platform.openai.com
   - Crea un account e genera una API key

2. **Configura .env**
   ```env
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o-mini
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
   ```

3. **Usa l'agente**
   ```bash
   python main.py
   ```

### Opzione 3: Groq (Ultra-Veloce)

1. **Ottieni API Key**
   - Vai su https://console.groq.com
   - Registrati e crea una API key

2. **Salva la chiave**

   Puoi scegliere uno di questi metodi:

   **Metodo A - File (Consigliato)**
   ```bash
   mkdir -p groq
   echo "gsk_xxxxxxxxxxxxx" > groq/API_groq.txt
   ```

   **Metodo B - Variabile d'Ambiente**
   ```env
   # In .env
   LLM_PROVIDER=groq
   LLM_MODEL=llama-3.1-8b-instant
   GROQ_API_KEY=gsk_xxxxxxxxxxxxx
   ```

3. **Usa l'agente**
   ```bash
   python main.py --provider groq --model llama-3.1-8b-instant
   ```

### Personalizzare config.yaml

Il file `config.yaml` contiene tutte le impostazioni dell'agente:

```yaml
# LLM Settings
llm:
  provider: "ollama"          # ollama, openai, groq
  model: "llama3.1:8b"        # Nome del modello
  temperature: 0.7            # Creatività (0.0-1.0)
  max_tokens: 4096            # Lunghezza massima risposta

# Agent Settings
agent:
  max_iterations: 5           # Massimo numero di tool da usare
  timeout_seconds: 300        # Timeout totale query
  max_history: 10             # Conversazioni da ricordare

# Web Tools Settings
web_tools:
  search:
    default_results: 5        # Risultati ricerca default
    max_results: 20           # Massimo risultati
  fetch:
    timeout: 30               # Timeout download pagina
    max_retries: 3            # Tentativi di retry

# Cache Settings
cache:
  enabled: true               # Abilita cache
  ttl_seconds: 3600           # Validità cache (1 ora)
  max_size_mb: 500            # Dimensione massima cache

# Logging Settings
logging:
  level: "INFO"               # DEBUG, INFO, SUCCESS, WARNING, ERROR
  format: "detailed"          # simple, detailed, json
```

**Esempio di customizzazione per uso avanzato:**
```yaml
llm:
  temperature: 0.2            # Più preciso, meno creativo

agent:
  max_iterations: 10          # Più strumenti per query complesse

web_tools:
  search:
    default_results: 10       # Più risultati di ricerca

cache:
  ttl_seconds: 7200           # Cache valida 2 ore

logging:
  level: "DEBUG"              # Log molto dettagliati
```

---

## Primi Passi

### Test Rapido: Single Query Mode

Prova l'agente con una singola domanda:

```bash
python main.py "Chi ha vinto il campionato di calcio 2024?"
```

L'agente:
1. 🧠 Pianifica quali strumenti usare (Chain of Thought)
2. 🔍 Cerca informazioni sul web
3. 📄 Scarica pagine rilevanti
4. 📝 Estrae le informazioni
5. ✅ Risponde in modo chiaro e citando le fonti

**Output esempio:**
```
[INFO] Inizializzazione Web Scraper Agent...
[SUCCESS] Agent pronto! Provider: ollama, Model: llama3.1:8b

[INFO] ============================================================
[INFO] Elaborazione query: Chi ha vinto il campionato di calcio 2024?
[INFO] ============================================================

[INFO] === FASE 1: Pianificazione (Chain of Thought) ===
[SUCCESS] Piano generato: 2 step(s)
  Step 1: search_web
    Reasoning: Cerco informazioni recenti sui campionati 2024
  Step 2: summarize_content
    Reasoning: Riassumo i risultati trovati

[INFO] === FASE 2: Esecuzione Piano ===
[INFO] ---- Step 1/2: search_web ----
[SUCCESS] Tool 'search_web' completato in 1234ms
[INFO] ---- Step 2/2: summarize_content ----
[SUCCESS] Tool 'summarize_content' completato in 567ms

[INFO] === FASE 3: Sintesi Risultati ===
[SUCCESS] Risposta finale generata

╔════════════════════════════════════════════════════════════╗
║                     RISPOSTA AGENTE                        ║
╚════════════════════════════════════════════════════════════╝

Nel 2024, diversi campionati si sono conclusi:

- **Serie A (Italia)**: Vinto dall'Inter Milano
- **La Liga (Spagna)**: Vinto dal Real Madrid
- **Premier League (Inghilterra)**: Vinto dal Manchester City

Fonti:
- https://www.gazzetta.it/...
- https://www.marca.com/...
- https://www.bbc.com/sport/...

[INFO] Query completata in 1.8s
```

### Test Interattivo

Lancia la modalità interattiva:

```bash
python main.py
```

Vedrai il prompt:
```
╔════════════════════════════════════════════════════════════╗
║            WEB SCRAPER AGENT - Modalità Interattiva       ║
╚════════════════════════════════════════════════════════════╝

Comandi disponibili:
  query <domanda>     - Fai una domanda intelligente
  search <query>      - Cerca direttamente sul web
  fetch <url>         - Scarica una pagina web
  history             - Mostra storico query
  stats               - Statistiche agente
  cache info          - Info sulla cache
  cache clear         - Pulisci cache
  help                - Mostra questo help
  exit, quit          - Esci

Agent> _
```

Prova a fare una domanda:
```
Agent> query Quali sono i principali framework Python per il machine learning?
```

---

## Modalità Interattiva

### Comandi Disponibili

#### 1. `query <domanda>` - Domanda Intelligente
L'agente usa Chain of Thought per pianificare e rispondere.

**Esempi:**
```
Agent> query Come funziona il protocollo HTTPS?

Agent> query Confronta i linguaggi Python e JavaScript per il backend

Agent> query Quali sono le novità di Python 3.12?
```

L'agente decide automaticamente:
- Se cercare sul web
- Se scaricare pagine specifiche
- Se confrontare fonti multiple
- Se riassumere contenuti lunghi

#### 2. `search <query>` - Ricerca Diretta
Usa direttamente il tool `search_web`.

**Esempi:**
```
Agent> search tutorial FastAPI italiano

Agent> search migliori framework frontend 2024

Agent> search "machine learning" python beginners
```

Output:
```
[SUCCESS] Trovati 5 risultati:

1. Tutorial FastAPI - Guida Completa
   https://fastapi.tiangolo.com/it/
   Una guida completa a FastAPI in italiano...

2. FastAPI: Creare API REST in Python
   https://realpython.com/fastapi-python-web-apis/
   Impara a creare API REST moderne con FastAPI...

[... altri risultati ...]
```

#### 3. `fetch <url>` - Scarica Pagina
Scarica e analizza una pagina specifica.

**Esempi:**
```
Agent> fetch https://realpython.com/tutorials/

Agent> fetch https://it.wikipedia.org/wiki/Python
```

Output:
```
[SUCCESS] Pagina scaricata e analizzata:

Titolo: Python (programming language) - Wikipedia
Autore: Wikipedia contributors
Data: 2024-03-15

Contenuto principale:
Python è un linguaggio di programmazione ad alto livello...

Metadata:
  - Language: it
  - Word count: 4523
  - Links found: 127

Links principali:
  - https://www.python.org
  - https://docs.python.org
  [...]
```

#### 4. `history` - Storico Conversazioni
Mostra le ultime query e risposte.

```
Agent> history

╔════════════════════════════════════════════════════════════╗
║                  STORICO CONVERSAZIONI                     ║
╚════════════════════════════════════════════════════════════╝

[1] 2024-03-15 10:23:45
User: Chi ha vinto il campionato 2024?
Assistant: Nel 2024, diversi campionati...

[2] 2024-03-15 10:25:12
User: Quali sono i framework Python per ML?
Assistant: I principali framework sono TensorFlow, PyTorch...

[3] 2024-03-15 10:27:33
User: Come funziona HTTPS?
Assistant: HTTPS usa crittografia TLS/SSL...
```

#### 5. `stats` - Statistiche Agente
Mostra statistiche di utilizzo.

```
Agent> stats

╔════════════════════════════════════════════════════════════╗
║                  STATISTICHE AGENTE                        ║
╚════════════════════════════════════════════════════════════╝

Query totali: 15
Tool chiamati: 42

Tool più usati:
  search_web:              18 volte
  fetch_webpage:           12 volte
  summarize_content:        8 volte
  compare_sources:          3 volte
  extract_structured_data:  1 volta

Tempo medio per query: 2.3s
Success rate: 93.3%
Cache hit rate: 38.5%
```

#### 6. `cache info` - Informazioni Cache
Mostra stato della cache.

```
Agent> cache info

╔════════════════════════════════════════════════════════════╗
║                   CACHE INFORMAZIONI                       ║
╚════════════════════════════════════════════════════════════╝

Posizione: /home/user/Agente_didattico_2/.cache
Elementi: 42
Dimensione totale: 12.5 MB
Dimensione media: 304.8 KB
Hit rate: 38.5%

Elementi più vecchi:
  - search_web_abc123def456: 3.2 ore fa (expires in 0.8h)
  - fetch_webpage_789ghi012: 2.8 ore fa (expires in 1.2h)

Elementi più grandi:
  - fetch_webpage_345jkl678: 2.1 MB
  - compare_sources_901mno: 1.8 MB
```

#### 7. `cache clear` - Pulisci Cache
Rimuove tutti gli elementi in cache.

```
Agent> cache clear

[INFO] Pulizia cache in corso...
[SUCCESS] Cache pulita: 42 elementi rimossi, 12.5 MB liberati
```

#### 8. `help` - Aiuto
Mostra l'help dei comandi.

```
Agent> help
```

#### 9. `exit` o `quit` - Esci
Esce dalla modalità interattiva.

```
Agent> exit

[INFO] Salvataggio stato agente...
[SUCCESS] Arrivederci!
```

---

## Esempi Pratici

### Esempio 1: Ricerca Semplice

**Domanda:**
```bash
python main.py "Cos'è FastAPI?"
```

**Cosa fa l'agente:**
1. Cerca "FastAPI" su DuckDuckGo
2. Scarica la pagina ufficiale
3. Estrae le informazioni principali
4. Riassume in modo chiaro

**Risposta:**
```
FastAPI è un framework web moderno e veloce per costruire API
con Python 3.8+, basato su standard come OpenAPI e JSON Schema.

Caratteristiche principali:
- Veloce: prestazioni comparabili a NodeJS e Go
- Facile da usare: documentazione automatica
- Type hints: validazione automatica dei dati
- Asincrono: supporto nativo per async/await

Fonti:
- https://fastapi.tiangolo.com
```

### Esempio 2: Confronto

**Domanda:**
```bash
python main.py "Confronta React e Vue.js"
```

**Cosa fa l'agente:**
1. Cerca informazioni su React
2. Cerca informazioni su Vue.js
3. Scarica documentazione ufficiale di entrambi
4. Usa `compare_sources` per trovare differenze e somiglianze
5. Sintetizza il confronto

**Risposta:**
```
Confronto React vs Vue.js:

SOMIGLIANZE:
- Entrambi usano Virtual DOM
- Supporto per componenti riutilizzabili
- Ecosistema ricco di librerie

DIFFERENZE:
- React: Creato da Meta, più popolare, JSX obbligatorio
- Vue: Creato da Evan You, curva di apprendimento più dolce,
       template HTML opzionali

QUANDO USARE REACT:
- Progetti enterprise complessi
- Team grande con esperienza JavaScript
- Ecosistema React Native per mobile

QUANDO USARE VUE:
- Progetti small-to-medium
- Team più piccolo
- Preferenza per template HTML

Fonti:
- https://react.dev
- https://vuejs.org
- https://2023.stateofjs.com
```

### Esempio 3: Estrazione Dati Strutturati

**Scenario:** Vuoi estrarre le API endpoints da una pagina di documentazione.

**Modalità interattiva:**
```
Agent> query Trova tutti gli endpoint API nella documentazione di JSONPlaceholder

[L'agente scarica la pagina e usa extract_structured_data]

Risposta:
Ho trovato i seguenti endpoints API di JSONPlaceholder:

GET /posts - Ottieni tutti i post
GET /posts/1 - Ottieni post specifico
POST /posts - Crea nuovo post
PUT /posts/1 - Aggiorna post
DELETE /posts/1 - Elimina post

GET /comments - Ottieni tutti i commenti
GET /users - Ottieni tutti gli utenti
GET /albums - Ottieni tutti gli album
GET /photos - Ottieni tutte le foto
GET /todos - Ottieni tutti i todo

Fonte: https://jsonplaceholder.typicode.com/guide/
```

### Esempio 4: Ricerca Accademica

**Domanda:**
```bash
python main.py "Quali sono le ultime ricerche su Large Language Models?"
```

**Cosa fa l'agente:**
1. Cerca "latest research large language models 2024"
2. Filtra risultati da fonti accademiche
3. Scarica abstract dei paper
4. Riassume i principali temi di ricerca

**Risposta:**
```
Principali aree di ricerca sui Large Language Models (2024):

1. EFFICIENZA E OTTIMIZZAZIONE
   - Quantizzazione dei modelli (4-bit, 8-bit)
   - Pruning e distillazione
   - Mixture of Experts (MoE) architecture

2. SAFETY E ALIGNMENT
   - RLHF (Reinforcement Learning from Human Feedback)
   - Constitutional AI
   - Riduzione bias e allucinazioni

3. MULTIMODALITÀ
   - Modelli vision-language
   - Audio, video e testo integrati
   - Generazione cross-modale

4. RETRIEVAL-AUGMENTED GENERATION (RAG)
   - Integrazione con knowledge bases
   - Riduzione allucinazioni
   - Informazioni aggiornate

Fonti:
- https://arxiv.org/list/cs.CL/recent
- https://huggingface.co/blog
- https://openai.com/research
```

### Esempio 5: Tutorial Step-by-Step

**Domanda:**
```
Agent> query Come creare una API REST con FastAPI step by step
```

**Risposta:**
```
Tutorial: Creare API REST con FastAPI

PREREQUISITI:
- Python 3.8+
- pip installato

STEP 1: INSTALLAZIONE
```bash
pip install fastapi uvicorn[standard]
```

STEP 2: FILE BASE (main.py)
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

STEP 3: ESEGUIRE L'API
```bash
uvicorn main:app --reload
```

STEP 4: TESTARE
Apri http://localhost:8000
Documentazione: http://localhost:8000/docs

STEP 5: AGGIUNGERE ENDPOINTS
```python
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

STEP 6: MODELLI PYDANTIC
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
def create_item(item: Item):
    return item
```

Fonti:
- https://fastapi.tiangolo.com/tutorial/
- https://fastapi.tiangolo.com/tutorial/first-steps/
```

---

## Troubleshooting

### Problema: "Connection refused" con Ollama

**Sintomo:**
```
[ERROR] Errore connessione a Ollama: Connection refused
```

**Soluzione:**
```bash
# Verifica se Ollama è in esecuzione
ollama list

# Se non risponde, avvia Ollama
ollama serve

# In un altro terminale
python main.py
```

### Problema: "Model not found"

**Sintomo:**
```
[ERROR] Modello 'llama3.1:8b' non trovato
```

**Soluzione:**
```bash
# Scarica il modello
ollama pull llama3.1:8b

# Verifica modelli disponibili
ollama list

# Usa un modello disponibile
python main.py --model <nome-modello-disponibile>
```

### Problema: "Invalid API key" (OpenAI/Groq)

**Sintomo:**
```
[ERROR] API key non valida
```

**Soluzione:**
1. Verifica che la chiave sia corretta
2. Per OpenAI:
   ```bash
   # In .env
   OPENAI_API_KEY=sk-proj-... (inizia con sk-proj o sk-)
   ```
3. Per Groq:
   ```bash
   # Opzione 1: File
   echo "gsk_..." > groq/API_groq.txt

   # Opzione 2: Env var
   export GROQ_API_KEY=gsk_...
   ```

### Problema: Timeout durante ricerca web

**Sintomo:**
```
[ERROR] Timeout durante search_web
```

**Soluzione:**
1. Aumenta il timeout in `config.yaml`:
   ```yaml
   web_tools:
     search:
       timeout: 60  # Aumenta a 60 secondi
   ```

2. Oppure riduci il numero di risultati:
   ```yaml
   web_tools:
     search:
       default_results: 3  # Riduci da 5 a 3
   ```

### Problema: Cache troppo grande

**Sintomo:**
```
[WARNING] Cache size (520 MB) exceeds max_size_mb (500 MB)
```

**Soluzione:**
```bash
# Opzione 1: Pulisci manualmente
python main.py
Agent> cache clear

# Opzione 2: Aumenta limite in config.yaml
cache:
  max_size_mb: 1000  # 1 GB

# Opzione 3: Riduci TTL (cache scade prima)
cache:
  ttl_seconds: 1800  # 30 minuti invece di 1 ora
```

### Problema: Risposte in inglese invece che italiano

**Sintomo:**
L'agente risponde in inglese.

**Soluzione:**
1. Sii esplicito nella domanda:
   ```
   Agent> query In italiano: cos'è FastAPI?
   ```

2. Oppure modifica il prompt di sistema in `agent.py`:
   ```python
   # Cerca questa riga e assicurati ci sia "in italiano"
   system_prompt = """
   Sei un assistente che risponde SEMPRE in italiano.
   [...]
   """
   ```

### Problema: "No module named 'bs4'"

**Sintomo:**
```
ModuleNotFoundError: No module named 'bs4'
```

**Soluzione:**
```bash
# Reinstalla le dipendenze
pip install -r requirements.txt

# Se persiste, installa manualmente
pip install beautifulsoup4 lxml
```

### Problema: Log troppo verbosi

**Sintomo:**
Troppi log DEBUG che rendono difficile leggere l'output.

**Soluzione:**
1. In `config.yaml`:
   ```yaml
   logging:
     level: "INFO"  # Invece di DEBUG
   ```

2. Oppure da CLI:
   ```bash
   # Disabilita log dettagliati temporaneamente
   python main.py --log-level WARNING "domanda qui"
   ```

### Problema: Permission denied su .cache

**Sintomo:**
```
[ERROR] Permission denied: .cache/abc123def456.pkl
```

**Soluzione:**
```bash
# Cambia permessi della directory cache
chmod -R u+rw .cache/

# Oppure rimuovi e ricrea
rm -rf .cache/
mkdir .cache
```

### Problema: Query troppo lente

**Sintomi:**
- Query impiegano > 10 secondi
- Troppe chiamate a tool non necessari

**Soluzione:**
1. Usa Groq invece di Ollama (molto più veloce):
   ```bash
   python main.py --provider groq --model llama-3.1-8b-instant
   ```

2. Abilita cache:
   ```yaml
   cache:
     enabled: true
   ```

3. Riduci max_iterations:
   ```yaml
   agent:
     max_iterations: 3  # Invece di 5
   ```

### Supporto Aggiuntivo

Se i problemi persistono:

1. **Controlla i log dettagliati:**
   ```bash
   python main.py --log-level DEBUG "query qui"
   ```

2. **Verifica la configurazione:**
   ```bash
   python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
   ```

3. **Test delle dipendenze:**
   ```bash
   python -c "import ollama, openai, groq, bs4, requests; print('OK')"
   ```

4. **Esegui i test:**
   ```bash
   pytest tests/ -v
   ```

---

## Conclusioni

Ora dovresti essere in grado di:
✅ Installare e configurare l'agente
✅ Usare la modalità interattiva
✅ Fare domande intelligenti
✅ Usare i singoli tool direttamente
✅ Risolvere problemi comuni

### Prossimi Passi

- Leggi `ARCHITECTURE.md` per capire come funziona internamente
- Guarda `EXAMPLES.md` per casi d'uso avanzati
- Sperimenta con query complesse
- Personalizza `config.yaml` per le tue esigenze

### Risorse Aggiuntive

- 📖 Documentazione Ollama: https://ollama.ai
- 📖 Documentazione OpenAI: https://platform.openai.com/docs
- 📖 Documentazione Groq: https://console.groq.com/docs
- 📖 BeautifulSoup docs: https://www.crummy.com/software/BeautifulSoup/
- 🐍 Python docs: https://docs.python.org

---

**Buon coding! 🚀**
