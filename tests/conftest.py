"""Run every test offline: hashing embedder, mock generator, no cache unless a
test opts in. Set before `src.config` is first imported, because the config
reads the environment at import time."""

import os

os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("CACHE_ENABLED", "false")
