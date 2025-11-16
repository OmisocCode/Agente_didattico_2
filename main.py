"""
============================================================================
MAIN - CLI Interattiva per Web Scraper Agent
============================================================================

Questo modulo fornisce un'interfaccia a riga di comando (CLI) per
interagire con il Web Scraper Agent.

Features:
- Modalità interattiva con REPL loop
- Modalità single-query (per automazione)
- Comandi speciali (help, history, stats, clear, etc.)
- Configurazione da file YAML
- Gestione variabili ambiente
- Output colorato e formattato
- Error handling user-friendly

Comandi disponibili:
- query <domanda>   - Fai una domanda all'agente
- search <query>    - Cerca direttamente sul web
- fetch <url>       - Scarica una pagina specifica
- history           - Mostra cronologia query
- stats             - Mostra statistiche
- cache info        - Info sulla cache
- cache clear       - Svuota cache
- clear             - Pulisci schermo
- help              - Mostra aiuto
- exit/quit         - Esci

Usage:
    # Modalità interattiva
    python main.py

    # Single query
    python main.py "Cerca ultime notizie AI"

    # Con configurazione custom
    python main.py --config my_config.yaml

    # Con LLM specifico
    python main.py --provider groq --model llama-3.1-8b-instant

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from loguru import logger

# Moduli del progetto
from agent import WebScraperAgent

# Per clear screen
import platform

# ----------------------------------------------------------------------------
# CONFIGURAZIONE LOGGING
# ----------------------------------------------------------------------------
# Rimuovi handler default di loguru
logger.remove()

# Aggiungi handler per console
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


# ----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Carica configurazione da file YAML.

    Args:
        config_path: Path al file di configurazione

    Returns:
        Dict con configurazione

    Raises:
        FileNotFoundError: Se file non trovato
    """
    logger.info(f"Loading configuration from: {config_path}")

    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.success(f"✓ Configuration loaded")
        logger.debug(f"Config keys: {list(config.keys())}")

        return config

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        logger.warning("Using default configuration")
        return {}


def print_banner():
    """
    Stampa banner iniziale dell'applicazione.
    """
    banner = """
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              🌐 AGENTE WEB SCRAPER INTELLIGENTE 🌐                     ║
║                                                                        ║
║  Cerca informazioni online, estrai dati e sintetizza risultati!       ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """
    Mostra comandi disponibili.
    """
    help_text = """
╔════════════════════════════════════════════════════════════════════════╗
║ COMANDI DISPONIBILI                                                    ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  📝 QUERY & RICERCA                                                    ║
║  ──────────────────                                                    ║
║  query <domanda>       - Fai una domanda all'agente (intelligente)    ║
║  search <query>        - Cerca direttamente sul web (DuckDuckGo)      ║
║  fetch <url>           - Scarica e mostra una pagina specifica        ║
║                                                                        ║
║  📊 INFO & STATISTICHE                                                 ║
║  ──────────────────────                                                ║
║  history               - Mostra cronologia delle query                ║
║  stats                 - Mostra statistiche agente e tools            ║
║  cache info            - Informazioni sulla cache                     ║
║                                                                        ║
║  🔧 UTILITY                                                            ║
║  ───────────                                                           ║
║  cache clear           - Svuota la cache                              ║
║  clear                 - Pulisci schermo                              ║
║  help                  - Mostra questo aiuto                          ║
║  exit / quit / q       - Esci dal programma                           ║
║                                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║ ESEMPI                                                                 ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  query Ultime notizie sull'intelligenza artificiale                   ║
║  search Python web scraping tutorial                                  ║
║  fetch https://example.com                                            ║
║  history                                                               ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    print(help_text)


def clear_screen():
    """
    Pulisce lo schermo del terminale.
    """
    # Determina comando in base al sistema operativo
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')


def format_response(response: str) -> str:
    """
    Formatta risposta per output.

    Args:
        response: Risposta da formattare

    Returns:
        Risposta formattata con box
    """
    lines = response.split('\n')

    # Trova lunghezza massima linea
    max_len = max(len(line) for line in lines) if lines else 70
    max_len = min(max_len, 70)  # Cap a 70 caratteri

    # Crea box
    top = "╔" + "═" * (max_len + 2) + "╗"
    bottom = "╚" + "═" * (max_len + 2) + "╝"

    formatted_lines = [top]

    for line in lines:
        # Padding per allineare
        padded = line.ljust(max_len)
        formatted_lines.append(f"║ {padded} ║")

    formatted_lines.append(bottom)

    return "\n".join(formatted_lines)


# ----------------------------------------------------------------------------
# INTERACTIVE MODE
# ----------------------------------------------------------------------------

def interactive_mode(agent: WebScraperAgent):
    """
    Modalità interattiva con REPL loop.

    Permette all'utente di interagire con l'agente tramite comandi.

    Args:
        agent: Istanza di WebScraperAgent
    """
    logger.info("Entering interactive mode")

    # Banner
    print_banner()
    print_help()
    print()

    # Loop principale
    while True:
        try:
            # ================================================================
            # INPUT UTENTE
            # ================================================================
            user_input = input("\n💬 Tu: ").strip()

            # Skip input vuoto
            if not user_input:
                continue

            # ================================================================
            # PARSE COMANDO
            # ================================================================
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            # ================================================================
            # GESTIONE COMANDI
            # ================================================================

            # EXIT
            if command in ["exit", "quit", "q"]:
                print("\n👋 Ciao! Alla prossima!")
                break

            # HELP
            elif command == "help":
                print_help()

            # CLEAR SCREEN
            elif command == "clear":
                clear_screen()
                print_banner()

            # HISTORY
            elif command == "history":
                print("\n📜 CRONOLOGIA QUERY")
                print("═" * 70)

                history = agent.get_execution_history()

                if not history:
                    print("Nessuna query eseguita ancora.")
                else:
                    # Mostra ultimi 10
                    for i, item in enumerate(history[-10:], 1):
                        print(f"\n{i}. Query: {item['query']}")
                        print(f"   Timestamp: {item['timestamp']}")
                        print(f"   Tools usati: {len(item['plan'])}")
                        print(f"   Tempo: {item['execution_time_seconds']:.2f}s")

                print()

            # STATS
            elif command == "stats":
                print("\n📊 STATISTICHE AGENTE")
                print("═" * 70)

                stats = agent.get_stats()

                print(f"\nQuery processate: {stats['queries_processed']}")
                print(f"Lunghezza conversazione: {stats['conversation_length']} messaggi")

                print("\nUtilizzo Tools:")
                for tool, count in stats['tool_usage'].items():
                    print(f"  - {tool}: {count} chiamate")

                if 'cache' in stats and stats['cache']:
                    cache_info = stats['cache']
                    print(f"\nCache:")
                    print(f"  - File: {cache_info['num_files']}")
                    print(f"  - Dimensione: {cache_info['total_size_mb']:.2f} MB")
                    print(f"  - Hit rate: {cache_info['hit_rate_percent']:.1f}%")

                print()

            # CACHE INFO
            elif command == "cache" and args.lower() == "info":
                if not agent.cache:
                    print("\n⚠️  Cache non abilitata")
                else:
                    print("\n💾 INFORMAZIONI CACHE")
                    print("═" * 70)

                    info = agent.cache.get_info()

                    print(f"\nDirectory: {info['cache_dir']}")
                    print(f"File: {info['num_files']}")
                    print(f"Dimensione: {info['total_size_mb']:.2f} MB / {info['max_size_mb']:.0f} MB")
                    print(f"TTL: {info['ttl_seconds']}s")

                    print(f"\nStatistiche:")
                    for key, value in info['stats'].items():
                        print(f"  - {key}: {value}")

                    print(f"\nHit rate: {info['hit_rate_percent']:.1f}%")
                    print()

            # CACHE CLEAR
            elif command == "cache" and args.lower() == "clear":
                if not agent.cache:
                    print("\n⚠️  Cache non abilitata")
                else:
                    confirm = input("\n⚠️  Confermi di voler svuotare la cache? (s/n): ")

                    if confirm.lower() in ['s', 'si', 'y', 'yes']:
                        agent.cache.clear()
                        print("✓ Cache svuotata")
                    else:
                        print("Operazione annullata")

            # QUERY (comando esplicito)
            elif command == "query":
                if not args:
                    print("❌ Specifica una query!")
                    continue

                print("\n🤖 Elaborazione query...")
                print("─" * 70)

                response = agent.process_query(args)

                print("\n🤖 RISPOSTA")
                print("═" * 70)
                print(response)
                print("═" * 70)

            # SEARCH (chiamata diretta a tool)
            elif command == "search":
                if not args:
                    print("❌ Specifica una query di ricerca!")
                    continue

                print(f"\n🔍 Cercando: {args}")
                print("─" * 70)

                try:
                    results = agent.web_tools.search_web(args, num_results=5)

                    print(f"\n📋 Risultati ({len(results)}):")
                    print()

                    for i, r in enumerate(results, 1):
                        print(f"{i}. {r['title']}")
                        print(f"   {r['url']}")
                        print(f"   {r['snippet'][:150]}...")
                        print()

                except Exception as e:
                    print(f"❌ Errore: {e}")

            # FETCH (chiamata diretta a tool)
            elif command == "fetch":
                if not args:
                    print("❌ Specifica un URL!")
                    continue

                print(f"\n📄 Scaricando: {args}")
                print("─" * 70)

                try:
                    page = agent.web_tools.fetch_webpage(args)

                    print(f"\n📄 Pagina:")
                    print(f"Titolo: {page['title']}")
                    print(f"URL: {page['url']}")
                    print(f"Dimensione: {len(page['content'])} caratteri")
                    print(f"Links: {len(page['links'])}")
                    print()
                    print("Contenuto (preview):")
                    print(page['content'][:500])
                    print("...")
                    print()

                except Exception as e:
                    print(f"❌ Errore: {e}")

            # DEFAULT: tratta come query
            else:
                # Se non riconosce il comando, tratta tutto l'input come query
                print("\n🤖 Elaborazione query...")
                print("─" * 70)

                response = agent.process_query(user_input)

                print("\n🤖 RISPOSTA")
                print("═" * 70)
                print(response)
                print("═" * 70)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrotto. Usa 'exit' per uscire.")
            continue

        except Exception as e:
            print(f"\n❌ Errore: {str(e)}")
            logger.exception(e)
            continue


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    """
    Entry point principale dell'applicazione.
    """
    # ========================================================================
    # PARSE ARGUMENTS
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Web Scraper Agent - Agente intelligente per ricerca e scraping web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python main.py                                    # Modalità interattiva
  python main.py "Cerca notizie AI"                 # Single query
  python main.py --config my_config.yaml            # Config custom
  python main.py --provider groq --model llama-3.1  # LLM specifico
  python main.py --no-cache                         # Disabilita cache
        """
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Query in modalità single-shot (opzionale)"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="File di configurazione YAML (default: config.yaml)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Modello LLM da usare (sovrascrive config)"
    )

    parser.add_argument(
        "--provider",
        default=None,
        choices=["ollama", "openai", "groq", "anthropic"],
        help="Provider LLM (sovrascrive config)"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disabilita caching"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Abilita output verboso"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Livello di logging"
    )

    args = parser.parse_args()

    # ========================================================================
    # CONFIGURA LOGGING
    # ========================================================================
    # Rimuovi handler esistenti
    logger.remove()

    # Aggiungi nuovo handler con livello specificato
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=args.log_level,
        colorize=True
    )

    # ========================================================================
    # CARICA ENVIRONMENT VARIABLES
    # ========================================================================
    logger.info("Loading environment variables...")
    load_dotenv()

    # ========================================================================
    # CARICA CONFIGURAZIONE
    # ========================================================================
    config = load_config(args.config)

    # ========================================================================
    # DETERMINA LLM MODEL E PROVIDER
    # ========================================================================
    # Priorità: args > config > default

    llm_model = (
        args.model or
        config.get("agent", {}).get("llm_model", "llama3.2")
    )

    llm_provider = (
        args.provider or
        config.get("agent", {}).get("llm_provider", "ollama")
    )

    logger.info(f"Using LLM: {llm_provider}/{llm_model}")

    # ========================================================================
    # CACHE ENABLED?
    # ========================================================================
    enable_cache = not args.no_cache and config.get("agent", {}).get("enable_caching", True)

    logger.info(f"Cache: {'enabled' if enable_cache else 'disabled'}")

    # ========================================================================
    # INIZIALIZZA AGENTE
    # ========================================================================
    print("\n🚀 Initializing Web Scraper Agent...")
    print("─" * 70)

    try:
        agent = WebScraperAgent(
            llm_model=llm_model,
            llm_provider=llm_provider,
            config=config,
            enable_cache=enable_cache
        )

        print("✓ Agent ready!")
        print()

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        logger.exception(e)
        sys.exit(1)

    # ========================================================================
    # MODALITÀ SINGLE QUERY o INTERATTIVA
    # ========================================================================
    if args.query:
        # ====================================================================
        # SINGLE QUERY MODE
        # ====================================================================
        query = " ".join(args.query)

        logger.info(f"Single query mode: {query}")

        print(f"\n💬 Query: {query}")
        print("─" * 70)
        print()

        try:
            response = agent.process_query(query)

            print("\n🤖 RISPOSTA")
            print("═" * 70)
            print(response)
            print("═" * 70)
            print()

        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception(e)
            sys.exit(1)

    else:
        # ====================================================================
        # INTERACTIVE MODE
        # ====================================================================
        try:
            interactive_mode(agent)
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            logger.exception(e)
            sys.exit(1)


# ----------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
