# Mandala Bridge

A lightweight retrieval layer between ChromaDB and AI agents, built around one idea: a knowledge base needs a lifecycle, not just a lookup.

Each chunk is tagged with lightweight metadata ("mandalas"). The bridge uses those tags to link related chunks, retire stale ones, and combine chunks into new derived entries — instead of treating the store as a static pile that only grows.

## Why

Standard RAG stores chunks and returns the top-k by similarity. Nothing gets pruned, nothing ages out, and there's no signal for what's actually still useful. Mandala Bridge adds a second layer on top: tag-based cross-linking between chunks, and a heuristic "breeding" process that combines related chunks and lets weakly-tagged ones fall out of rotation.

**A note on the framing:** the genetics language (crossover, Hardy-Weinberg, breeding) is a design metaphor, not literal population genetics. It's a memorable way to describe a heuristic scoring system — not a claim that formal genetics math applies to text chunks. The scoring curves are chosen design constraints that produce useful behavior, not derived laws. Full explanation in `REASONING.md`.

## Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| POST | `/ingest` | Chunk a document, generate tags, store |
| POST | `/query` | Substring search over chunks, with tag-based cross-linking to related chunks |
| POST | `/crossover` | Combine two chunks into a derived entry, weighted by shared tags |
| POST | `/population` | Tag-frequency analysis across a collection — which tags are common, rare, or dominant |
| POST | `/equilibrium` | Boost under-represented tags by pairing them with dominant ones |
| POST | `/steady` | Score how "balanced" a pair of chunks is before combining them |
| POST | `/invariant` | Secondary balance check on a pair, using a different scoring curve |
| POST | `/breed` | Run N rounds of combining + pruning across a collection |
| GET | `/health` | Status |
| GET | `/collections` | List collections |
| GET | `/neurons/<id>` | Inspect a chunk and its tags |

## Quickstart

```bash
export MANDALA_CHROMA=/path/to/chroma.sqlite3
python3 server.py
```

```bash
# ingest a document
curl -X POST http://127.0.0.1:8855/ingest \
  -H "Content-Type: application/json" \
  -d '{"name":"doc.md","text":"...","collection":"notes"}'

# search
curl -X POST http://127.0.0.1:8855/query \
  -H "Content-Type: application/json" \
  -d '{"collection":"notes","query":"database","limit":5}'

# run 3 rounds of combine + prune
curl -X POST http://127.0.0.1:8855/breed \
  -H "Content-Type: application/json" \
  -d '{"collection":"notes","generations":3}'
```

## Design

- SQLite-backed, zero external dependencies (Python stdlib only)
- Talks to ChromaDB's SQLite layer directly — no client library
- Deterministic tagging: regex + keyword extraction, not black-box embeddings
- Search is substring matching (`LIKE`) over stored chunks, not a full-text index — good enough for small-to-medium local collections; swapping in SQLite FTS5 is a natural next step for larger ones
- The scoring heuristics are explained in `REASONING.md`

## License

MIT
