"""
============================================================================
CACHE SYSTEM - Sistema di Caching per Risultati Tools
============================================================================

Questo modulo implementa un sistema di caching per i risultati dei tools,
evitando richieste HTTP duplicate e velocizzando l'esecuzione dell'agente.

Features:
- Caching basato su file (persistente tra sessioni)
- TTL (Time-To-Live) configurabile
- Chiavi generate da hash MD5 di tool + parametri
- Cleanup automatico di cache scadute
- Gestione dimensione massima cache
- Serializzazione con pickle (supporta oggetti Python complessi)

Vantaggi:
- Riduce chiamate HTTP duplicate
- Velocizza testing e sviluppo
- Risparmia rate limits API
- Persistente tra riavvii

Use Cases:
- Sviluppo e testing (evita richieste ripetute)
- Rate limiting (evita di superare limiti API)
- Performance (risparmia latenza rete)
- Offline mode (usa risultati cached)

Author: Web Scraper Agent Team
License: MIT
============================================================================
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
import os
import hashlib
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from loguru import logger

# Configurazione logging
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


# ----------------------------------------------------------------------------
# RESULT CACHE CLASS
# ----------------------------------------------------------------------------
class ResultCache:
    """
    Sistema di caching per risultati tools.

    Il cache funziona in questo modo:
    1. Ogni risultato viene salvato in un file pickle
    2. Il nome file è un hash MD5 di: tool_name + parametri
    3. Ogni file contiene: risultato, timestamp, metadata
    4. Prima di ritornare un risultato, verifica TTL

    Questo permette:
    - Cache persistente tra sessioni
    - Facile debugging (file leggibili)
    - Cleanup selettivo
    - Nessun database necessario

    Esempio:
        cache = ResultCache(cache_dir=".cache", ttl_seconds=3600)

        # Salva risultato
        cache.set("search_web", {"query": "Python"}, results)

        # Recupera risultato (se non scaduto)
        cached = cache.get("search_web", {"query": "Python"})
        if cached:
            print("Cache hit!")
        else:
            print("Cache miss, executing tool...")
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        ttl_seconds: int = 3600,
        max_size_mb: int = 500,
        auto_cleanup: bool = True
    ):
        """
        Inizializza sistema di caching.

        Args:
            cache_dir: Directory dove salvare file cache
            ttl_seconds: Time-to-live in secondi (default: 3600 = 1 ora)
            max_size_mb: Dimensione massima cache in MB (default: 500)
            auto_cleanup: Se True, pulisce cache scadute automaticamente
        """
        logger.info("Initializing ResultCache")

        # Parametri configurazione
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Converti MB in bytes
        self.auto_cleanup = auto_cleanup

        logger.debug(f"Cache directory: {self.cache_dir}")
        logger.debug(f"TTL: {ttl_seconds}s ({self.ttl})")
        logger.debug(f"Max size: {max_size_mb}MB ({self.max_size_bytes} bytes)")
        logger.debug(f"Auto cleanup: {auto_cleanup}")

        # Crea directory se non esiste
        if not self.cache_dir.exists():
            logger.info(f"Creating cache directory: {self.cache_dir}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.success(f"✓ Cache directory created")
        else:
            logger.debug("Cache directory already exists")

        # Statistiche
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "expired": 0,
            "errors": 0
        }

        # Auto-cleanup all'inizializzazione se abilitato
        if self.auto_cleanup:
            logger.info("Running initial cleanup...")
            self._cleanup_expired()

        logger.success("✓ ResultCache initialized")

    def _generate_key(self, tool: str, params: Dict) -> str:
        """
        Genera chiave univoca da tool + parametri.

        La chiave è un hash MD5 di: tool_name + JSON(params)
        Questo garantisce che stessi parametri = stessa chiave.

        Args:
            tool: Nome del tool
            params: Parametri usati

        Returns:
            Chiave hash MD5 (32 caratteri esadecimali)
        """
        # Crea stringa univoca da tool + params
        # Uso JSON con sort_keys per garantire ordine consistente
        params_json = json.dumps(params, sort_keys=True)
        key_string = f"{tool}:{params_json}"

        # Genera hash MD5
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        logger.debug(f"Generated cache key: {key_hash}")
        logger.debug(f"  Tool: {tool}")
        logger.debug(f"  Params: {params_json[:100]}...")

        return key_hash

    def get(self, tool: str, params: Dict) -> Optional[Any]:
        """
        Recupera risultato dalla cache.

        Controlla se esiste un risultato cached per il tool + parametri.
        Se esiste e non è scaduto, lo ritorna.
        Se è scaduto o non esiste, ritorna None.

        Args:
            tool: Nome del tool
            params: Parametri del tool

        Returns:
            Risultato cached o None se non trovato/scaduto
        """
        logger.debug(f"Cache lookup: {tool} with params {params}")

        # Genera chiave
        key = self._generate_key(tool, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        logger.debug(f"Cache file: {cache_file}")

        # Check se file esiste
        if not cache_file.exists():
            logger.debug("Cache miss: file not found")
            self.stats["misses"] += 1
            return None

        # Leggi file cache
        try:
            logger.debug("Reading cache file...")

            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # Estrai informazioni
            result = cached_data['result']
            timestamp = datetime.fromisoformat(cached_data['timestamp'])
            cached_tool = cached_data.get('tool', 'unknown')
            cached_params = cached_data.get('params', {})

            logger.debug(f"Cache entry found:")
            logger.debug(f"  Tool: {cached_tool}")
            logger.debug(f"  Timestamp: {timestamp}")
            logger.debug(f"  Age: {datetime.now() - timestamp}")

            # Check TTL (Time-To-Live)
            age = datetime.now() - timestamp

            if age > self.ttl:
                logger.debug(f"Cache expired (age: {age} > TTL: {self.ttl})")
                logger.info("🕐 Cache expired, will fetch fresh data")
                self.stats["expired"] += 1

                # Rimuovi file scaduto
                cache_file.unlink()
                logger.debug("Removed expired cache file")

                return None

            # Cache valida!
            logger.success(f"✓ Cache HIT! (age: {age})")
            self.stats["hits"] += 1

            return result

        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            logger.exception(e)
            self.stats["errors"] += 1

            # Rimuovi file corrotto
            try:
                cache_file.unlink()
                logger.debug("Removed corrupted cache file")
            except:
                pass

            return None

    def set(self, tool: str, params: Dict, result: Any):
        """
        Salva risultato nella cache.

        Serializza il risultato in un file pickle con metadata.

        Args:
            tool: Nome del tool
            params: Parametri usati
            result: Risultato da cachare
        """
        logger.debug(f"Caching result for: {tool}")

        # Genera chiave
        key = self._generate_key(tool, params)
        cache_file = self.cache_dir / f"{key}.pkl"

        logger.debug(f"Cache file: {cache_file}")

        try:
            # Prepara dati da salvare
            cache_data = {
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'tool': tool,
                'params': params,
                'ttl_seconds': self.ttl.total_seconds()
            }

            # Salva in file pickle
            logger.debug("Writing cache file...")

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            # Statistiche
            file_size = cache_file.stat().st_size
            logger.debug(f"Cache file size: {file_size} bytes")

            self.stats["sets"] += 1

            logger.success(f"✓ Result cached successfully")

            # Check dimensione totale cache
            if self.auto_cleanup:
                self._check_cache_size()

        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            logger.exception(e)
            self.stats["errors"] += 1

    def clear(self):
        """
        Svuota completamente la cache.

        Rimuove tutti i file nella directory cache.
        Usa con cautela!
        """
        logger.warning("Clearing entire cache...")

        removed_count = 0

        # Rimuovi tutti i file .pkl
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {cache_file}: {e}")

        logger.success(f"✓ Cache cleared ({removed_count} files removed)")

    def _cleanup_expired(self):
        """
        Rimuove file cache scaduti.

        Scansiona tutti i file cache e rimuove quelli più vecchi del TTL.
        """
        logger.debug("Cleaning up expired cache entries...")

        removed_count = 0
        now = datetime.now()

        # Scansiona tutti i file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                # Leggi file
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Check TTL
                timestamp = datetime.fromisoformat(cached_data['timestamp'])
                age = now - timestamp

                if age > self.ttl:
                    # Scaduto - rimuovi
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed expired: {cache_file.name} (age: {age})")

            except Exception as e:
                # File corrotto - rimuovi
                logger.warning(f"Corrupted cache file {cache_file.name}, removing")
                try:
                    cache_file.unlink()
                    removed_count += 1
                except:
                    pass

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache entries")
        else:
            logger.debug("No expired entries found")

    def _check_cache_size(self):
        """
        Controlla dimensione totale cache.

        Se supera il limite, rimuove i file più vecchi fino a rientrare nel limite.
        """
        logger.debug("Checking cache size...")

        # Calcola dimensione totale
        total_size = 0
        cache_files = []

        for cache_file in self.cache_dir.glob("*.pkl"):
            size = cache_file.stat().st_size
            mtime = cache_file.stat().st_mtime  # Modification time
            total_size += size
            cache_files.append((cache_file, size, mtime))

        logger.debug(f"Total cache size: {total_size / (1024*1024):.2f} MB")
        logger.debug(f"Max cache size: {self.max_size_bytes / (1024*1024):.2f} MB")

        # Se non supera il limite, tutto ok
        if total_size <= self.max_size_bytes:
            logger.debug("Cache size within limits")
            return

        # Supera il limite - rimuovi file più vecchi
        logger.warning(f"Cache size exceeds limit, cleaning up...")

        # Ordina per modification time (più vecchi prima)
        cache_files.sort(key=lambda x: x[2])

        removed_count = 0
        removed_size = 0

        # Rimuovi file più vecchi fino a rientrare nel limite
        for cache_file, size, mtime in cache_files:
            if total_size - removed_size <= self.max_size_bytes:
                break

            try:
                cache_file.unlink()
                removed_count += 1
                removed_size += size
                logger.debug(f"Removed old file: {cache_file.name} ({size} bytes)")
            except Exception as e:
                logger.error(f"Failed to remove {cache_file}: {e}")

        logger.success(f"✓ Removed {removed_count} old files ({removed_size / (1024*1024):.2f} MB)")
        logger.info(f"New cache size: {(total_size - removed_size) / (1024*1024):.2f} MB")

    def get_stats(self) -> Dict[str, int]:
        """
        Ritorna statistiche cache.

        Returns:
            Dict con hits, misses, sets, etc.
        """
        return self.stats.copy()

    def get_info(self) -> Dict[str, Any]:
        """
        Ritorna informazioni dettagliate sulla cache.

        Returns:
            Dict con info su dimensione, numero file, hit rate, etc.
        """
        logger.debug("Gathering cache info...")

        # Conta file e dimensione totale
        num_files = 0
        total_size = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            num_files += 1
            total_size += cache_file.stat().st_size

        # Calcola hit rate
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        info = {
            "cache_dir": str(self.cache_dir),
            "num_files": num_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "ttl_seconds": self.ttl.total_seconds(),
            "stats": self.stats,
            "hit_rate_percent": hit_rate
        }

        logger.debug(f"Cache info: {num_files} files, {info['total_size_mb']:.2f} MB, {hit_rate:.1f}% hit rate")

        return info

    def __repr__(self) -> str:
        """Rappresentazione stringa."""
        info = self.get_info()
        return f"ResultCache(files={info['num_files']}, size={info['total_size_mb']:.1f}MB, hit_rate={info['hit_rate_percent']:.1f}%)"


# ----------------------------------------------------------------------------
# TESTING / DEMO
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Test del sistema di caching.

    Esegui: python cache.py
    """
    print("\n" + "="*70)
    print("Testing Cache System")
    print("="*70 + "\n")

    # Crea cache
    cache = ResultCache(
        cache_dir=".test_cache",
        ttl_seconds=10,  # 10 secondi per test rapidi
        max_size_mb=1
    )

    print(f"\nCache initialized: {cache}\n")

    # Test 1: Set e Get
    print("-" * 70)
    print("Test 1: Set and Get")
    print("-" * 70)

    # Salva risultato
    test_result = ["result1", "result2", "result3"]
    cache.set("test_tool", {"param": "value"}, test_result)

    # Recupera risultato
    cached = cache.get("test_tool", {"param": "value"})

    if cached:
        print(f"✓ Cache hit: {cached}")
    else:
        print("✗ Cache miss")

    # Test 2: Cache miss
    print("\n" + "-" * 70)
    print("Test 2: Cache miss")
    print("-" * 70)

    cached = cache.get("test_tool", {"param": "different_value"})

    if cached:
        print(f"✗ Unexpected cache hit: {cached}")
    else:
        print("✓ Cache miss as expected")

    # Test 3: TTL expiration
    print("\n" + "-" * 70)
    print("Test 3: TTL expiration (wait 11 seconds...)")
    print("-" * 70)

    import time
    print("Waiting for cache to expire...")
    time.sleep(11)  # Wait longer than TTL

    cached = cache.get("test_tool", {"param": "value"})

    if cached:
        print(f"✗ Cache should have expired: {cached}")
    else:
        print("✓ Cache expired as expected")

    # Test 4: Multiple entries
    print("\n" + "-" * 70)
    print("Test 4: Multiple cache entries")
    print("-" * 70)

    for i in range(5):
        cache.set(f"tool_{i}", {"id": i}, f"result_{i}")

    print(f"\nCache info: {cache.get_info()}")
    print(f"Stats: {cache.get_stats()}")

    # Test 5: Clear cache
    print("\n" + "-" * 70)
    print("Test 5: Clear cache")
    print("-" * 70)

    cache.clear()
    print(f"\nCache after clear: {cache}\n")

    # Cleanup test directory
    import shutil
    shutil.rmtree(".test_cache", ignore_errors=True)
    print("Test cache directory removed")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
