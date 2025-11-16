# Esempi Pratici: Web Scraper Agent

## Indice
1. [Esempi Base](#esempi-base)
2. [Esempi Intermedi](#esempi-intermedi)
3. [Esempi Avanzati](#esempi-avanzati)
4. [Casi d'Uso Reali](#casi-duso-reali)
5. [Esempi di Programmazione](#esempi-di-programmazione)
6. [Best Practices](#best-practices)

---

## Esempi Base

### Esempio 1: Ricerca Semplice

#### Query
```
python main.py "Cos'è Python?"
```

#### Piano Generato dall'Agent
```
Step 1: search_web
  Parametri: {"query": "Cos'è Python", "num_results": 5}
  Reasoning: Cerco informazioni generali su Python
```

#### Output
```
╔════════════════════════════════════════════════════════════╗
║                     RISPOSTA AGENTE                        ║
╚════════════════════════════════════════════════════════════╝

Python è un linguaggio di programmazione ad alto livello,
interpretato e orientato agli oggetti. Creato da Guido van Rossum
e rilasciato nel 1991, è noto per la sua sintassi chiara e leggibile.

CARATTERISTICHE PRINCIPALI:
- Sintassi semplice e intuitiva
- Tipizzazione dinamica
- Ampia libreria standard
- Supporto multi-paradigma (OOP, funzionale, procedurale)

USI COMUNI:
- Web development (Django, Flask)
- Data Science (Pandas, NumPy)
- Machine Learning (TensorFlow, PyTorch)
- Automazione e scripting
- Game development

Fonti consultate:
- https://www.python.org
- https://it.wikipedia.org/wiki/Python
- https://realpython.com/what-is-python
```

**Tempo di esecuzione**: ~1.5s

---

### Esempio 2: Informazioni su una Pagina Specifica

#### Query
```
Agent> fetch https://fastapi.tiangolo.com
```

#### Piano Generato
```
Step 1: fetch_webpage
  Parametri: {"url": "https://fastapi.tiangolo.com", "extract_main_content": true}
  Reasoning: Query contiene URL esplicito, scarico direttamente la pagina
```

#### Output
```
[SUCCESS] Pagina scaricata e analizzata

╔════════════════════════════════════════════════════════════╗
║                    INFORMAZIONI PAGINA                     ║
╚════════════════════════════════════════════════════════════╝

URL: https://fastapi.tiangolo.com
Titolo: FastAPI - Modern Python Web Framework

METADATA:
  Autore: Sebastián Ramírez
  Lingua: en
  Parole: 4,523
  Links trovati: 87

CONTENUTO PRINCIPALE:
FastAPI is a modern, fast (high-performance) web framework for
building APIs with Python 3.7+ based on standard Python type hints.

Key features:
- Fast: Very high performance, on par with NodeJS and Go
- Fast to code: Increase development speed by 200-300%
- Fewer bugs: Reduce human errors by 40%
- Intuitive: Great editor support with autocompletion
- Easy: Designed to be easy to use and learn
- Short: Minimize code duplication
- Robust: Production-ready code with automatic documentation

PRINCIPALI LINKS:
  - https://fastapi.tiangolo.com/tutorial/
  - https://fastapi.tiangolo.com/features/
  - https://github.com/tiangolo/fastapi
  - https://fastapi.tiangolo.com/deployment/
```

**Tempo di esecuzione**: ~800ms

---

### Esempio 3: Riassunto di Contenuto

#### Query Modalità Interattiva
```
Agent> query Riassumi la pagina di Wikipedia su Albert Einstein
```

#### Piano Generato
```
Step 1: search_web
  Parametri: {"query": "Wikipedia Albert Einstein", "num_results": 3}
  Reasoning: Cerco la pagina Wikipedia specifica

Step 2: fetch_webpage
  Parametri: {"url": "https://it.wikipedia.org/wiki/Albert_Einstein"}
  Reasoning: Scarico il contenuto completo della pagina

Step 3: summarize_content
  Parametri: {"style": "detailed", "max_length": 300}
  Reasoning: Riassumo il contenuto in modo dettagliato
```

#### Output
```
╔════════════════════════════════════════════════════════════╗
║                     RISPOSTA AGENTE                        ║
╚════════════════════════════════════════════════════════════╝

ALBERT EINSTEIN - Riassunto

Albert Einstein (1879-1955) è stato un fisico teorico tedesco,
naturalizzato svizzero e americano. È considerato uno dei più
grandi scienziati di tutti i tempi.

CONTRIBUTI PRINCIPALI:
1. Teoria della Relatività Ristretta (1905)
   - E=mc² (equivalenza massa-energia)
   - Concetto di spazio-tempo

2. Teoria della Relatività Generale (1915)
   - Gravità come curvatura dello spazio-tempo
   - Prevede buchi neri e onde gravitazionali

3. Premio Nobel per la Fisica (1921)
   - Per la scoperta dell'effetto fotoelettrico
   - Fondamentale per la meccanica quantistica

VITA:
- Nato a Ulm, Germania
- Studi al Politecnico di Zurigo
- Emigrato negli USA nel 1933 (fuga dal nazismo)
- Professore a Princeton fino alla morte

EREDITÀ:
Einstein ha rivoluzionato la fisica del XX secolo e rimane
un'icona della scienza. Le sue teorie sono ancora validate
da esperimenti moderni (LIGO ha confermato le onde
gravitazionali nel 2016).

Fonte: https://it.wikipedia.org/wiki/Albert_Einstein
```

**Tempo di esecuzione**: ~3.2s

---

## Esempi Intermedi

### Esempio 4: Confronto tra Tecnologie

#### Query
```
python main.py "Confronta React e Vue.js per sviluppo frontend"
```

#### Piano Generato
```
Step 1: search_web
  Parametri: {"query": "React framework", "num_results": 3}
  Reasoning: Cerco informazioni su React

Step 2: search_web
  Parametri: {"query": "Vue.js framework", "num_results": 3}
  Reasoning: Cerco informazioni su Vue.js

Step 3: compare_sources
  Parametri: {"topic": "React vs Vue.js frontend development"}
  Reasoning: Confronto le due tecnologie
```

#### Output
```
╔════════════════════════════════════════════════════════════╗
║                     CONFRONTO REACT VS VUE.JS              ║
╚════════════════════════════════════════════════════════════╝

REACT
Sviluppatore: Meta (Facebook)
Prima release: 2013
Tipo: Libreria JavaScript per UI

PRO:
✓ Ecosistema vastissimo (React Native, Next.js, Remix)
✓ Enorme community e supporto
✓ Usato da grandi aziende (Facebook, Netflix, Airbnb)
✓ Virtual DOM performante
✓ Ottimo per applicazioni large-scale
✓ React Native per app mobile native

CONTRO:
✗ Curva di apprendimento più ripida
✗ JSX può essere confuso per principianti
✗ Molte decisioni da prendere (router, state management)
✗ Frequenti breaking changes tra major versions

---

VUE.JS
Sviluppatore: Evan You (community-driven)
Prima release: 2014
Tipo: Framework JavaScript progressivo

PRO:
✓ Curva di apprendimento dolce
✓ Template HTML-like familiari
✓ Documentazione eccellente
✓ Tutto integrato (router, state, build tools)
✓ Performance eccellenti
✓ Perfetto per small-to-medium projects

CONTRO:
✗ Ecosistema più piccolo
✗ Meno job opportunities
✗ Community più piccola
✗ Meno risorse di apprendimento

---

SOMIGLIANZE:
• Entrambi usano Virtual DOM
• Component-based architecture
• Reactive data binding
• Supporto TypeScript
• SPA (Single Page Applications)

---

QUANDO USARE REACT:
→ Progetti enterprise large-scale
→ Team grandi
→ Necessità di app mobile (React Native)
→ Ecosistema ricco importante

QUANDO USARE VUE:
→ Progetti small-medium
→ Team piccoli o sviluppatori singoli
→ Prototipazione rapida
→ Graduale integrazione in app esistenti

---

VERDICT:
Non esiste una scelta "migliore" in assoluto. React è più
adatto per progetti complessi con team grandi, mentre Vue
eccelle in progetti più piccoli dove la semplicità è prioritaria.

Per principianti: Vue.js è più accessibile.
Per carriera/job market: React ha più opportunità.

Fonti consultate:
- https://react.dev
- https://vuejs.org
- https://2023.stateofjs.com/en-US/libraries/front-end-frameworks/
- https://stackoverflow.blog/2023/01/26/comparing-frameworks-react-vue-angular/
```

**Tempo di esecuzione**: ~5.8s

---

### Esempio 5: Estrazione Dati Strutturati

#### Scenario
Estrarre informazioni su corsi di programmazione da una pagina HTML.

#### Codice Python
```python
from web_tools import WebTools
from llm_interface import LLMInterface

# Setup
llm = LLMInterface(model="llama3.1:8b", provider="ollama")
tools = WebTools(llm)

# Scarica la pagina
page = tools.fetch_webpage("https://www.coursera.org/courses?query=python")

# Schema di estrazione
schema = {
    "_container": "div.course-card",  # Container per ogni corso
    "title": "h3.course-title",
    "instructor": "span.instructor-name",
    "rating": "span.rating-value",
    "students": "span.enrollment-count",
    "price": "span.price",
    "link": "a.course-link::attr(href)"
}

# Estrai dati
courses = tools.extract_structured_data(page["html"], schema)

# Stampa risultati
for i, course in enumerate(courses[:5], 1):
    print(f"\n{i}. {course['title']}")
    print(f"   Instructor: {course['instructor']}")
    print(f"   Rating: {course['rating']}/5")
    print(f"   Students: {course['students']}")
    print(f"   Price: {course['price']}")
    print(f"   Link: {course['link']}")
```

#### Output
```
1. Python for Everybody Specialization
   Instructor: Dr. Charles Severance
   Rating: 4.8/5
   Students: 2.3M
   Price: Free
   Link: /specializations/python

2. Python Data Structures
   Instructor: Dr. Charles Severance
   Rating: 4.9/5
   Students: 845K
   Price: Free
   Link: /learn/python-data-structures

3. Complete Python Bootcamp
   Instructor: Jose Portilla
   Rating: 4.7/5
   Students: 1.2M
   Price: $84.99
   Link: /learn/complete-python-bootcamp

4. Machine Learning with Python
   Instructor: Andrew Ng
   Rating: 4.8/5
   Students: 3.5M
   Price: Free (audit)
   Link: /learn/machine-learning

5. Python for Data Science and AI
   Instructor: IBM
   Rating: 4.6/5
   Students: 567K
   Price: Free
   Link: /professional-certificates/ibm-data-science
```

---

### Esempio 6: Ricerca Accademica

#### Query
```
Agent> query Quali sono le ultime ricerche su GPT-4 e large language models?
```

#### Piano Generato
```
Step 1: search_web
  Parametri: {"query": "GPT-4 research papers 2024", "num_results": 5}
  Reasoning: Cerco paper recenti su GPT-4

Step 2: search_web
  Parametri: {"query": "large language models latest research", "num_results": 5}
  Reasoning: Cerco ricerche generali su LLM

Step 3: fetch_webpage
  Parametri: {"url": "https://arxiv.org/list/cs.CL/recent"}
  Reasoning: Scarico lista recente da arXiv

Step 4: summarize_content
  Parametri: {"style": "technical", "max_length": 400}
  Reasoning: Riassunto tecnico delle ricerche trovate
```

#### Output
```
╔════════════════════════════════════════════════════════════╗
║         ULTIME RICERCHE SU GPT-4 E LLM (2024)              ║
╚════════════════════════════════════════════════════════════╝

AREE DI RICERCA PRINCIPALI:

1. EFFICIENZA E OTTIMIZZAZIONE
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Quantizzazione (4-bit, 8-bit)
     - Paper: "GPTQ: Accurate Post-Training Quantization for GPT"
     - Riduce dimensione modello del 75% con <1% perdita accuracy

   • Mixture of Experts (MoE)
     - Paper: "Mixtral 8x7B: A Sparse Mixture of Experts Model"
     - Attiva solo subset di parametri per token
     - 5x più efficiente senza perdita performance

   • LoRA (Low-Rank Adaptation)
     - Fine-tuning efficiente con <1% parametri
     - Usato in produzione da Anthropic, OpenAI

2. SAFETY E ALIGNMENT
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • RLHF (Reinforcement Learning from Human Feedback)
     - Tecnica usata in GPT-4, Claude 3
     - Allinea modello a preferenze umane

   • Constitutional AI
     - Paper di Anthropic
     - Modello supervisiona se stesso usando "costituzione"

   • Riduzione Allucinazioni
     - Retrieval-Augmented Generation (RAG)
     - Citation mechanisms
     - Confidence scores

3. MULTIMODALITÀ
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Vision-Language Models
     - GPT-4V (vision), Gemini Pro Vision
     - Comprendono immagini + testo

   • Audio Integration
     - Whisper per speech-to-text
     - GPT-4o con audio nativo

   • Video Understanding
     - Sora (OpenAI): text-to-video
     - Gemini 1.5: analisi video lunghi

4. RETRIEVAL-AUGMENTED GENERATION (RAG)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Integrazione con knowledge bases
     - Vector databases (Pinecone, Weaviate)
     - Riduce allucinazioni del ~40%

   • Long-context models
     - Gemini 1.5: 1M token context
     - Claude 3: 200K token context
     - Elimina necessità di chunking

5. REASONING E PLANNING
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • Chain-of-Thought (CoT)
     - Modelli "pensano" step-by-step
     - Migliora accuracy su math/reasoning

   • Tree-of-Thoughts
     - Esplora multiple reasoning paths
     - Backtracking se path sbagliato

   • Agent architectures
     - ReAct (Reasoning + Acting)
     - AutoGPT, BabyAGI per task autonomi

BREAKTHROUGH RECENTI (2024):

→ GPT-4 Turbo: 128K context, più economico
→ Claude 3 Opus: Supera GPT-4 su molti benchmark
→ Gemini Ultra: Multimodal nativo, SOTA su MMLU
→ Mixtral 8x22B: Open-source MoE competitivo con GPT-4
→ LLaMA 3: 400B parametri, open-source di Meta

TREND EMERGENTI:

• Small Language Models (SLM): Phi-3, Gemma
  - 1-7B parametri ma performance sorprendenti
  - Runnable on device (smartphone, laptop)

• Specialized models: CodeLLaMA, Med-PaLM, Galactica
  - Domain-specific outperforma general models

• Democratizzazione: Ollama, LM Studio
  - LLM locali facili da usare

SFIDE APERTE:

⚠ Hallucinations: Non ancora risolte completamente
⚠ Bias: Presenti nonostante alignment
⚠ Interpretability: "Black box" difficili da spiegare
⚠ Copyright/IP: Questioni legali su training data
⚠ Compute cost: Ancora molto costoso per training

Fonti consultate:
- https://arxiv.org/list/cs.CL/recent
- https://huggingface.co/blog
- https://openai.com/research
- https://www.anthropic.com/research
- https://deepmind.google/research/
```

**Tempo di esecuzione**: ~8.5s

---

## Esempi Avanzati

### Esempio 7: Multi-Step Research con Cache

#### Query Complessa
```
Agent> query Analizza i top 3 framework Python per web development, confronta performance, community e casi d'uso
```

#### Piano Generato (7 steps)
```
Step 1: search_web
  Query: "top Python web frameworks 2024"
  Reasoning: Identifico i framework principali

Step 2: search_web
  Query: "Django framework overview features"
  Reasoning: Approfondisco Django

Step 3: search_web
  Query: "Flask framework overview features"
  Reasoning: Approfondisco Flask

Step 4: search_web
  Query: "FastAPI framework overview features"
  Reasoning: Approfondisco FastAPI

Step 5: fetch_webpage
  URL: "https://www.techempower.com/benchmarks/"
  Reasoning: Scarico benchmark performance

Step 6: compare_sources
  Topic: "Django vs Flask vs FastAPI"
  Reasoning: Confronto sistematico

Step 7: summarize_content
  Style: "detailed"
  Reasoning: Sintesi finale completa
```

#### Output (Estratto)
```
╔════════════════════════════════════════════════════════════╗
║     TOP 3 FRAMEWORK PYTHON WEB - ANALISI COMPLETA          ║
╚════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DJANGO - "The Web Framework for Perfectionists"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW:
Full-stack framework "batteries-included" creato nel 2005.
Filosofia: "Don't Repeat Yourself" (DRY)

FEATURES:
✓ ORM potente (database abstraction)
✓ Admin panel automatico
✓ Sistema di autenticazione integrato
✓ Template engine
✓ Forms con validazione
✓ Sicurezza built-in (CSRF, XSS, SQL injection)
✓ Internazionalizzazione
✓ Caching framework

PERFORMANCE:
- Requests/sec: ~3,500 (TechEmpower)
- Latency media: ~15ms
- Rating: ⭐⭐⭐ (3/5)

COMMUNITY:
- GitHub stars: 76K
- Contributors: 2,500+
- Packages (PyPI): 5,000+
- StackOverflow questions: 350K+
- Rating: ⭐⭐⭐⭐⭐ (5/5)

CASI D'USO:
→ CMS e content-heavy sites
→ E-commerce (Oscar, Saleor)
→ Social networks
→ Applicazioni enterprise
→ Progetti con deadline stretti

USATO DA:
Instagram, Mozilla, Pinterest, The Washington Post

PRO:
+ Tutto incluso, meno decisioni
+ Scalabile per progetti grandi
+ Eccellente per rapid development
+ Admin panel incredibilmente utile

CONTRO:
- Più lento di FastAPI/Flask
- Può essere "too much" per API semplici
- Monolitico, meno flessibile

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. FLASK - "The Microframework"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW:
Micro-framework minimalista creato nel 2010.
Filosofia: "Explicit is better than implicit"

FEATURES:
✓ Core minimale (~7 dependencies)
✓ Routing semplice
✓ Template engine (Jinja2)
✓ WSGI compliant
✓ Ecosystem ricco (extensions)
✗ No ORM (usa SQLAlchemy a parte)
✗ No admin panel built-in

PERFORMANCE:
- Requests/sec: ~5,200
- Latency media: ~10ms
- Rating: ⭐⭐⭐⭐ (4/5)

COMMUNITY:
- GitHub stars: 66K
- Contributors: 800+
- Extensions: 1,000+
- StackOverflow questions: 180K+
- Rating: ⭐⭐⭐⭐⭐ (5/5)

CASI D'USO:
→ API REST semplici
→ Microservices
→ Prototyping rapido
→ Applicazioni small-to-medium
→ Quando serve controllo fine

USATO DA:
LinkedIn, Netflix (alcuni servizi), Reddit (originariamente)

PRO:
+ Estremamente flessibile
+ Curva apprendimento dolce
+ Perfetto per microservices
+ Deploy semplice

CONTRO:
- Serve configurare tutto manualmente
- Meno structure per progetti grandi
- Non async-native (pre Flask 2.0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. FASTAPI - "The Modern Framework"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW:
Framework moderno (2018) per API ad alte performance.
Filosofia: "Fast to code, fast to run"

FEATURES:
✓ Async/await nativo (basato su Starlette)
✓ Type hints → validazione automatica
✓ Documentazione automatica (OpenAPI/Swagger)
✓ Performance altissime
✓ Dependency Injection
✓ WebSocket support
✗ No template engine built-in (API-first)
✗ No ORM built-in (usa SQLAlchemy/Tortoise)

PERFORMANCE:
- Requests/sec: ~18,000 (async)
- Latency media: ~3ms
- Rating: ⭐⭐⭐⭐⭐ (5/5)
- Uno dei framework Python PIÙ VELOCI

COMMUNITY:
- GitHub stars: 70K
- Contributors: 600+
- Crescita: +400% negli ultimi 2 anni
- StackOverflow questions: 25K+ (crescente)
- Rating: ⭐⭐⭐⭐ (4/5 - più giovane)

CASI D'USO:
→ API REST moderne
→ Microservices ad alte performance
→ ML model serving
→ Real-time applications (WebSocket)
→ Quando performance è critica

USATO DA:
Microsoft, Uber, Netflix (nuovi servizi)

PRO:
+ Performance incredibili
+ Dev experience eccellente
+ Auto-documentation
+ Type safety con Pydantic
+ Async nativo

CONTRO:
- Meno maturo di Django/Flask
- Community più piccola (ma crescente)
- Non ideale per frontend-heavy apps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFRONTO DIRETTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────┬─────────┬─────────┬──────────┐
│ Caratteristica  │ Django  │ Flask   │ FastAPI  │
├─────────────────┼─────────┼─────────┼──────────┤
│ Performance     │ ⭐⭐⭐    │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐⭐    │
│ Learning Curve  │ ⭐⭐      │ ⭐⭐⭐⭐⭐  │ ⭐⭐⭐⭐     │
│ Community       │ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐⭐⭐  │ ⭐⭐⭐⭐     │
│ Async Support   │ ⭐⭐⭐     │ ⭐⭐⭐    │ ⭐⭐⭐⭐⭐    │
│ API Development │ ⭐⭐⭐     │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐⭐    │
│ Full-Stack      │ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐    │ ⭐⭐       │
│ Documentation   │ ⭐⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐⭐    │
└─────────────────┴─────────┴─────────┴──────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RACCOMANDAZIONI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USA DJANGO SE:
✓ Stai costruendo un'applicazione full-stack
✓ Hai bisogno di admin panel
✓ Il progetto è content-heavy (blog, CMS, e-commerce)
✓ Preferisci convention over configuration
✓ Team junior o deadline stretto

USA FLASK SE:
✓ Vuoi massima flessibilità
✓ Stai costruendo microservices semplici
✓ Prototipazione rapida
✓ Vuoi scegliere ogni componente
✓ Progetto small-to-medium

USA FASTAPI SE:
✓ Stai costruendo API moderne
✓ Performance è critica
✓ ML model serving / real-time apps
✓ Ami type hints e validazione automatica
✓ Documentazione automatica è importante

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[INFO] Cache statistics:
  - search_web: 4 hits / 4 calls (100% - query ripetuta)
  - fetch_webpage: 1 hit / 1 call (100%)
  - Tempo risparmiato: ~3.2s

Fonti consultate:
- https://www.djangoproject.com
- https://flask.palletsprojects.com
- https://fastapi.tiangolo.com
- https://www.techempower.com/benchmarks/
- https://pypi.org/
- https://github.com/
```

**Tempo di esecuzione**: ~6.2s (con cache), ~9.4s (senza cache)

---

## Casi d'Uso Reali

### Caso d'Uso 1: Ricerca di Mercato

#### Scenario
Un'azienda vuole analizzare la concorrenza nel settore e-commerce.

#### Query
```
Agent> query Analizza i top 3 e-commerce in Italia: caratteristiche, prezzi, punti di forza
```

#### Utilizzo dell'Agent
```python
# L'agent eseguirà automaticamente:
# 1. search_web → Identifica top e-commerce italiani
# 2. fetch_webpage → Scarica homepage di Amazon.it, eBay.it, ePrice
# 3. extract_structured_data → Estrae prezzi, categorie, offerte
# 4. compare_sources → Confronta caratteristiche
# 5. summarize_content → Sintesi competitiva
```

#### Output Utile Per
- Strategia di pricing
- Feature gap analysis
- Positioning nel mercato

---

### Caso d'Uso 2: Aggregazione Notizie

#### Scenario
Un giornalista vuole un riassunto delle ultime notizie su un tema.

#### Query
```
Agent> query Dammi le ultime notizie sulla COP29 sul clima con riassunto
```

#### Risultato
L'agent:
1. Cerca "COP29 climate conference latest news"
2. Scarica articoli da fonti multiple (Reuters, BBC, Guardian)
3. Riassume ogni articolo
4. Identifica consensus e divergenze
5. Fornisce timeline degli eventi

**Beneficio**: 10 minuti di lavoro manuale → 30 secondi automatici

---

### Caso d'Uso 3: Due Diligence Tecnica

#### Scenario
Un investitore valuta una startup tech e vuole capire la loro tech stack.

#### Query
```
Agent> query Analizza il tech stack di Stripe: linguaggi, database, infrastruttura
```

#### Processo
```python
# L'agent:
# 1. Cerca "Stripe tech stack engineering blog"
# 2. Scarica blog posts tecnici
# 3. Cerca su StackShare/GitHub
# 4. Estrae menzioni di tecnologie
# 5. Sintetizza in report strutturato
```

#### Output
```
STRIPE - TECH STACK ANALYSIS

BACKEND:
- Linguaggio principale: Ruby (Rails)
- Microservices: Scala, Go
- API: REST + GraphQL

DATABASE:
- Primary: PostgreSQL
- Cache: Redis, Memcached
- Message Queue: Kafka, RabbitMQ

INFRASTRUCTURE:
- Cloud: AWS (multi-region)
- Containers: Docker + Kubernetes
- CDN: Cloudflare

FRONTEND:
- React.js
- TypeScript
- Next.js per SSR

DEVOPS:
- CI/CD: Jenkins + GitHub Actions
- Monitoring: Datadog, Prometheus
- Logging: ELK Stack

SECURITY:
- PCI DSS Level 1 compliant
- Encryption: TLS 1.2+, AES-256

Fonti:
- https://stripe.com/blog/engineering
- https://stackshare.io/stripe/stripe
- Various engineering blog posts
```

**Valore**: Risparmio ore di ricerca manuale

---

## Esempi di Programmazione

### Esempio 8: Integrazione in un'Applicazione Python

#### Scenario
Integrare l'agent in un'app Flask per fornire ricerca intelligente.

#### Codice
```python
from flask import Flask, request, jsonify
from agent import WebScraperAgent
from llm_interface import LLMInterface
import yaml

app = Flask(__name__)

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize agent
llm = LLMInterface(
    model=config["llm"]["model"],
    provider=config["llm"]["provider"]
)
agent = WebScraperAgent(llm, config)

@app.route("/api/search", methods=["POST"])
def intelligent_search():
    """
    Endpoint per ricerca intelligente.

    Body:
    {
        "query": "Come funziona blockchain?",
        "include_sources": true
    }

    Response:
    {
        "answer": "...",
        "sources": [...],
        "execution_time_ms": 2341
    }
    """
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Process query con l'agent
        import time
        start = time.time()

        answer = agent.process_query(query)

        execution_time = (time.time() - start) * 1000

        # Extract sources dalla risposta
        sources = []
        if "Fonti consultate:" in answer:
            sources_text = answer.split("Fonti consultate:")[1]
            sources = [
                s.strip().lstrip("-").strip()
                for s in sources_text.split("\n")
                if s.strip()
            ]

        return jsonify({
            "answer": answer,
            "sources": sources if data.get("include_sources") else None,
            "execution_time_ms": execution_time,
            "cached": False  # TODO: detect cache hits
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Statistiche dell'agent.
    """
    stats = agent.get_statistics()
    return jsonify(stats)

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """
    Pulisci la cache.
    """
    agent.cache.clear()
    return jsonify({"message": "Cache cleared successfully"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

#### Test
```bash
# Start server
python app.py

# Test endpoint
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Cos'\''è FastAPI?", "include_sources": true}'

# Response
{
  "answer": "FastAPI è un framework web moderno...",
  "sources": [
    "https://fastapi.tiangolo.com",
    "https://realpython.com/fastapi-python-web-apis/"
  ],
  "execution_time_ms": 2341,
  "cached": false
}
```

---

### Esempio 9: Batch Processing

#### Scenario
Processare una lista di query in batch.

#### Codice
```python
import csv
from agent import WebScraperAgent
from llm_interface import LLMInterface
import yaml
from tqdm import tqdm

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize agent
llm = LLMInterface(
    model=config["llm"]["model"],
    provider=config["llm"]["provider"]
)
agent = WebScraperAgent(llm, config)

# Load queries da CSV
queries = []
with open("queries.csv") as f:
    reader = csv.DictReader(f)
    queries = [row["query"] for row in reader]

print(f"Processing {len(queries)} queries...")

# Process con progress bar
results = []
for query in tqdm(queries, desc="Processing queries"):
    try:
        answer = agent.process_query(query)
        results.append({
            "query": query,
            "answer": answer,
            "status": "success"
        })
    except Exception as e:
        results.append({
            "query": query,
            "answer": None,
            "status": "error",
            "error": str(e)
        })

# Save results
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "answer", "status", "error"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to results.csv")

# Statistics
successful = sum(1 for r in results if r["status"] == "success")
print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
```

#### Input (queries.csv)
```csv
query
"Cos'è Python?"
"Come funziona blockchain?"
"Differenze tra SQL e NoSQL"
"Migliori pratiche di sicurezza web"
"Cosa sono i microservices?"
```

#### Output (results.csv)
```csv
query,answer,status,error
"Cos'è Python?","Python è un linguaggio...",success,
"Come funziona blockchain?","Blockchain è una tecnologia...",success,
"Differenze tra SQL e NoSQL","SQL è relazionale...",success,
"Migliori pratiche di sicurezza web","Le migliori pratiche...",success,
"Cosa sono i microservices?","I microservices sono...",success,
```

---

### Esempio 10: Custom Tool

#### Scenario
Aggiungere un tool personalizzato per traduzione.

#### Codice
```python
# In web_tools.py

def translate_text(
    self,
    text: str,
    target_language: str = "it",
    source_language: str = "auto"
) -> str:
    """
    Traduce testo usando il LLM.

    Args:
        text: Testo da tradurre
        target_language: Lingua target (ISO 639-1)
        source_language: Lingua sorgente ("auto" per auto-detect)

    Returns:
        Testo tradotto
    """
    logger.info(f"Traduzione: {source_language} → {target_language}")

    # Check cache
    if self.cache:
        cache_key = {
            "text": text,
            "target": target_language,
            "source": source_language
        }
        cached = self.cache.get("translate_text", cache_key)
        if cached:
            logger.success("Traduzione caricata da cache")
            return cached

    # Prompt per traduzione
    if source_language == "auto":
        prompt = f"""
        Traduci il seguente testo in {target_language}.
        Mantieni lo stile e il tono originale.
        Rispondi SOLO con la traduzione, senza spiegazioni.

        TESTO:
        {text}

        TRADUZIONE:
        """
    else:
        prompt = f"""
        Traduci il seguente testo da {source_language} a {target_language}.
        Mantieni lo stile e il tono originale.
        Rispondi SOLO con la traduzione, senza spiegazioni.

        TESTO:
        {text}

        TRADUZIONE:
        """

    # Genera traduzione con LLM
    translation = self.llm.generate(
        prompt=prompt,
        temperature=0.3  # Bassa per traduzioni precise
    )

    # Save to cache
    if self.cache:
        self.cache.set("translate_text", cache_key, translation)

    logger.success("Traduzione completata")
    return translation.strip()

# Registrazione in _register_all_tools()
def _register_all_tools(self):
    # ... altri tools ...

    self.registry.register(
        name="translate_text",
        function=self.translate_text,
        description="Traduce testo in un'altra lingua usando il LLM",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Testo da tradurre"
                },
                "target_language": {
                    "type": "string",
                    "description": "Codice lingua target (es: 'it', 'en', 'es', 'fr')",
                    "default": "it"
                },
                "source_language": {
                    "type": "string",
                    "description": "Codice lingua sorgente ('auto' per auto-detect)",
                    "default": "auto"
                }
            },
            "required": ["text"]
        }
    )
```

#### Utilizzo
```python
# Modalità interattiva
Agent> query Traduci in italiano: "The quick brown fox jumps over the lazy dog"

# Output
[SUCCESS] Piano generato: 1 step(s)
  Step 1: translate_text
    Reasoning: Query richiede traduzione esplicita

[INFO] Esecuzione tool: translate_text
[SUCCESS] Traduzione completata

Risposta:
"La veloce volpe marrone salta sopra il cane pigro"
```

---

## Best Practices

### 1. Ottimizzare le Query

#### ❌ Query Vaga
```
Agent> query python
```

#### ✅ Query Specifica
```
Agent> query Quali sono le best practices per gestire eccezioni in Python con esempi
```

**Perché**: Query specifiche generano piani migliori e risposte più accurate.

---

### 2. Usare la Cache Intelligentemente

#### Strategia
```yaml
# config.yaml
cache:
  enabled: true
  ttl_seconds: 3600  # 1 ora per ricerche generali

  # Per dati che cambiano raramente
  # ttl_seconds: 86400  # 24 ore
```

#### Quando Pulire la Cache
```bash
# Prima di cercare notizie recenti
Agent> cache clear
Agent> query Ultime notizie su...
```

---

### 3. Logging per Debugging

#### Configura Livello Appropriato
```yaml
# Sviluppo
logging:
  level: "DEBUG"  # Vedi tutto

# Produzione
logging:
  level: "INFO"   # Solo informazioni importanti
```

---

### 4. Gestire Rate Limits

#### Per API a Pagamento
```python
# In web_tools.py
import time

def search_web(self, query: str, ...):
    # Aggiungi rate limiting
    time.sleep(0.5)  # 500ms delay tra chiamate
    # ... resto del codice
```

---

### 5. Error Handling Robusto

#### Nel Codice
```python
try:
    answer = agent.process_query(query)
except TimeoutError:
    print("Query timeout, riprova con query più semplice")
except ConnectionError:
    print("Problema di connessione, controlla internet")
except Exception as e:
    print(f"Errore: {e}")
    # Log per debugging
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

---

### 6. Monitorare le Performance

#### Usa Stats Regolarmente
```python
Agent> stats

# Analizza:
# - Quali tool sono più usati?
# - Tempo medio per query troppo alto?
# - Cache hit rate basso? (aumenta TTL)
# - Success rate basso? (query troppo complesse?)
```

---

### 7. Validare le Fonti

#### Sempre Controlla le Fonti
```
# L'agent cita le fonti - verificale manualmente per info critiche
Fonti consultate:
- https://example.com  ← Clicca e verifica!
```

**Importante**: Gli LLM possono commettere errori. Per decisioni critiche, verifica sempre le fonti originali.

---

## Conclusioni

Questi esempi dimostrano la versatilità del Web Scraper Agent:

✅ **Ricerca semplice**: Rispondere a domande generali
✅ **Analisi comparative**: Confrontare tecnologie, prodotti, servizi
✅ **Ricerca accademica**: Aggregare paper e ricerche
✅ **Data extraction**: Estrarre dati strutturati da HTML
✅ **Integration**: Usare l'agent in applicazioni custom
✅ **Batch processing**: Processare liste di query
✅ **Extensibility**: Aggiungere tools personalizzati

### Prossimi Passi

1. Sperimenta con query diverse
2. Personalizza config.yaml per le tue esigenze
3. Aggiungi tools custom per il tuo dominio
4. Integra in progetti esistenti
5. Monitora e ottimizza le performance

**Buon coding! 🚀**
