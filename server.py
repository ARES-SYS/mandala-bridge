#!/usr/bin/env python3
"""
MANDALA BRIDGE — A tag-linked retrieval layer between ChromaDB and agents.

Each chunk is tagged ("mandalas"). Tags link related chunks together,
and a heuristic scoring system decides which chunks are worth combining
or letting go. The genetics-flavored naming below is a design metaphor,
not a claim that formal genetics applies — see REASONING.md.

POST /ingest     — Document → chunks + tags
POST /query      — Search with tag-based cross-linking
POST /crossover  — Combine two chunks into a derived entry
POST /population — Tag-frequency analysis across a collection
POST /breed      — N-round combine-and-prune run
GET  /health     — Status
GET  /neurons/<id> — Inspect a chunk and its tags

MIT License
"""

import json, math, os, re, sys, hashlib, uuid, sqlite3, time, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

CHROMA = os.path.expanduser(os.environ.get("MANDALA_CHROMA", "~/.mandala-bridge/chroma.sqlite3"))
CHUNK_LINES = int(os.environ.get("CHUNK_LINES", 2000))
PORT = int(os.environ.get("MANDALA_PORT", 8855))

# ─── MANDALA TAG GENERATION ──────────────────────────────

def generate_mandalas(text):
    """Extracts semantic tags from a chunk. #type #concept #entity"""
    tags = set()

    # Markdown headers → structure tags
    for h in re.findall(r'^#{1,4}\s+(.+?)$', text, re.MULTILINE):
        tags.add(f"#header:{h.strip().lower()[:60]}")

    # Sustained UPPERCASE words → entities
    for ent in re.findall(r'\b[A-Z][A-Z0-9_]{2,}(?:\s+[A-Z][A-Z0-9_]{2,})*\b', text):
        if len(ent) > 4 and ent not in ('TODO','NOTE','FIXME'):
            tags.add(f"#entity:{ent.strip().lower()}")

    # Technical keywords
    keywords = ['docker','jenkins','kubernetes','api','oauth','jwt','sql',
                'injection','xss','csrf','proxy','vpn','dns','firewall',
                'chromadb','sqlite','embedding','vector','rag','llm','gpt',
                'ollama','openai','agent','database','backend','frontend',
                'http','server','client','python','javascript','typescript',
                'react','node','redis','postgres']
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            tags.add(f"#tech:{kw}")

    # Domains / IPs
    for domain in re.findall(r'\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}\b', text):
        tags.add(f"#host:{domain.lower()}")

    # CVEs
    for cve in re.findall(r'CVE-\d{4}-\d{4,}', text):
        tags.add(f"#{cve.lower()}")

    # Paths / files
    for fpath in re.findall(r'[\w/.-]+\.(md|py|sh|yaml|json|toml|conf|db|sqlite3)', text):
        tags.add(f"#file:{os.path.basename(fpath).lower()}")

    return sorted(tags)[:30]


def chunk_text(text, max_lines=CHUNK_LINES):
    lines = text.split("\n")
    chunks, i = [], 0
    while i < len(lines):
        end = min(i + max_lines, len(lines))
        if end < len(lines):
            for j in range(end, max(i, end - 80), -1):
                if j < len(lines) and lines[j].strip() == "":
                    end = j; break
        chunks.append("\n".join(lines[i:end]))
        i = end
    return chunks


def doc_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ─── CHROMADB OPS ─────────────────────────────────────────

def chroma_conn():
    c = sqlite3.connect(CHROMA)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def ensure_collection(cur, name):
    cur.execute("SELECT id FROM collections WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0], False
    cid = str(uuid.uuid4())
    schema_tpl = '{"keys":{"#document":{"string":{"fts_index":{"enabled":true}}}},"defaults":{"float_list":{"vector_index":{"enabled":false}},"string":{"string_inverted_index":{"enabled":true}}}}'
    cur.execute(
        "INSERT INTO collections (id, name, dimension, database_id, config_json_str, schema_str) VALUES (?,?,?,?,?,?)",
        (cid, name, 384, "00000000-0000-0000-0000-000000000000", "{}", schema_tpl))
    return cid, True


def ensure_segment(cur, col_id):
    cur.execute("SELECT id FROM segments WHERE collection=? AND type LIKE '%hnsw%' LIMIT 1", (col_id,))
    row = cur.fetchone()
    if row:
        return row[0], False
    sid = str(uuid.uuid4())
    cur.execute("INSERT INTO segments (id, type, scope, collection) VALUES (?,?,?,?)",
                (sid, "urn:chroma:segment/vector/hnsw-local-persisted", "VECTOR", col_id))
    return sid, True


def next_seq_id(cur):
    cur.execute("SELECT COALESCE(MAX(seq_id),0)+1 FROM (SELECT MAX(seq_id) as seq_id FROM embeddings_queue UNION ALL SELECT MAX(seq_id) FROM embeddings)")
    return cur.fetchone()[0]


def insert_chunk(cur, seg_id, seq, embedding_id, doc_id, idx, total, text, mandalas, collection, col_id):
    # Embedding row
    cur.execute("INSERT INTO embeddings (segment_id, embedding_id, seq_id) VALUES (?,?,?)",
                (seg_id, embedding_id, seq))

    # Metadata
    metas = [
        ("chroma:document", text),
        ("parent_doc", doc_id),
        ("chunk_index", str(idx)),
        ("total_chunks", str(total)),
        ("collection", collection),
        ("mandalas", " ".join(mandalas)),
    ]
    for k, v in metas:
        cur.execute("INSERT INTO embedding_metadata (id, key, string_value) VALUES (?,?,?)",
                    (seq, k, v))

    # Queue for embedding
    topic = f"persistent://default/default/{col_id}"
    cur.execute("INSERT INTO embeddings_queue (seq_id, operation, topic, id, encoding) VALUES (?,0,?,?,'FLOAT32')",
                (seq, topic, embedding_id))


# ─── HTTP HANDLER ─────────────────────────────────────────

class MandalaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._json({"status": "mandala", "chunk_lines": CHUNK_LINES, "chroma": CHROMA})
        elif path.startswith("/neurons/"):
            nid = path.split("/neurons/")[1]
            self._get_neuron(nid)
        elif path == "/collections":
            conn = chroma_conn()
            cur = conn.cursor()
            cur.execute("SELECT name, id FROM collections")
            self._json([{"name": r[0], "id": r[1]} for r in cur.fetchall()])
            conn.close()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}

            if path == "/ingest":
                self._ingest(body)
            elif path == "/query":
                self._query(body)
            elif path == "/crossover":
                self._crossover(body)
            elif path == "/population":
                self._population(body)
            elif path == "/equilibrium":
                self._equilibrium(body)
            elif path == "/steady":
                self._steady_state(body)
            elif path == "/breed":
                self._breed(body)
            elif path == "/invariant":
                self._invariant(body)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e), "trace": traceback.format_exc()[:500]}, 500)

    # ── endpoints ─────────────────────────────────────────

    def _ingest(self, body):
        name = body.get("name", "untitled")
        text = body["text"]
        collection = body.get("collection", "default")
        chunks = chunk_text(text)
        dhash = doc_hash(text)

        conn = chroma_conn()
        cur = conn.cursor()
        col_id, _ = ensure_collection(cur, collection)
        seg_id, _ = ensure_segment(cur, col_id)
        topic = f"persistent://default/default/{col_id}"

        result_chunks = []
        for idx, chunk_text_content in enumerate(chunks):
            mandalas = generate_mandalas(chunk_text_content)
            embedding_id = f"mandala_{dhash}_{idx:04d}"
            seq = next_seq_id(cur)

            insert_chunk(cur, seg_id, seq, embedding_id, dhash, idx, len(chunks), chunk_text_content, mandalas, collection, col_id)

            result_chunks.append({
                "chunk_index": idx,
                "lines": len(chunk_text_content.split("\n")),
                "neurons": len(mandalas),
                "mandalas": mandalas[:8],
                "embedding_id": embedding_id
            })

        conn.commit()
        conn.close()

        self._json({
            "doc_id": dhash,
            "name": name,
            "collection": collection,
            "total_lines": len(text.split("\n")),
            "total_chunks": len(chunks),
            "total_neurons": sum(len(c["mandalas"]) for c in result_chunks),
            "chunks": result_chunks
        })

    def _query(self, body):
        collection = body["collection"]
        query = body["query"]
        limit = body.get("limit", 5)
        follow_links = body.get("follow_links", True)

        conn = chroma_conn()
        cur = conn.cursor()

        # Find collection
        cur.execute("SELECT id FROM collections WHERE name=?", (collection,))
        col = cur.fetchone()
        if not col:
            conn.close()
            self._json({"results": [], "error": f"collection '{collection}' not found"})
            return
        col_id = col[0]

        # FTS5 search
        try:
            cur.execute("""
                SELECT DISTINCT
                    substr(em_doc.string_value, 1, 2000) as content,
                    em_parent.string_value as parent_doc,
                    COALESCE(CAST(em_idx.string_value AS INTEGER), 0) as chunk_idx,
                    COALESCE(CAST(em_total.string_value AS INTEGER), 0) as total_chunks,
                    COALESCE(em_mandalas.string_value, '') as mandalas
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN embedding_metadata em_doc ON e.id = em_doc.id AND em_doc.key = 'chroma:document'
                JOIN embedding_metadata em_parent ON e.id = em_parent.id AND em_parent.key = 'source'
                LEFT JOIN embedding_metadata em_idx ON e.id = em_idx.id AND em_idx.key = 'chunk_index'
                LEFT JOIN embedding_metadata em_total ON e.id = em_total.id AND em_total.key = 'total_chunks'
                LEFT JOIN embedding_metadata em_mandalas ON e.id = em_mandalas.id AND em_mandalas.key = 'mandalas'
                WHERE s.collection = ?
                AND em_doc.string_value LIKE ?
                LIMIT ?
            """, (col_id, f"%{query}%", limit * 3))
        except:
            # Fallback without mandala tags
            cur.execute("""
                SELECT DISTINCT
                    substr(em_doc.string_value, 1, 2000) as content,
                    em_parent.string_value as parent_doc,
                    COALESCE(CAST(em_idx.string_value AS INTEGER), 0) as chunk_idx,
                    COALESCE(CAST(em_total.string_value AS INTEGER), 0) as total_chunks,
                    '' as mandalas
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN embedding_metadata em_doc ON e.id = em_doc.id AND em_doc.key = 'chroma:document'
                JOIN embedding_metadata em_parent ON e.id = em_parent.id AND em_parent.key = 'source'
                LEFT JOIN embedding_metadata em_idx ON e.id = em_idx.id AND em_idx.key = 'chunk_index'
                LEFT JOIN embedding_metadata em_total ON e.id = em_total.id AND em_total.key = 'total_chunks'
                WHERE s.collection = ?
                AND em_doc.string_value LIKE ?
                LIMIT ?
            """, (col_id, f"%{query}%", limit * 3))

        rows = cur.fetchall()

        # Deduplicate + link mandalas
        seen_parents, results = set(), []
        for content, parent, idx, total, mandalas_str in rows:
            if parent in seen_parents:
                continue
            seen_parents.add(parent)
            mandala_list = mandalas_str.split() if mandalas_str else []

            # Follow links: find chunks related by shared mandalas
            linked = []
            if follow_links and mandala_list:
                placeholders = ",".join("?" * len(mandala_list))
                cur.execute(f"""
                    SELECT DISTINCT em_parent.string_value, CAST(em_idx.string_value AS INTEGER)
                    FROM embedding_metadata em
                    JOIN embedding_metadata em_parent ON em.id = em_parent.id AND em_parent.key = 'source'
                    LEFT JOIN embedding_metadata em_idx ON em.id = em_idx.id AND em_idx.key = 'chunk_index'
                    WHERE em.key = 'mandalas'
                    AND em_parent.string_value != ?
                    AND ({' OR '.join(['em.string_value LIKE ?' for _ in mandala_list])})
                    LIMIT 5
                """, [parent] + [f"%{m}%" for m in mandala_list])
                linked = [{"doc": r[0], "chunk": r[1]} for r in cur.fetchall()]

            results.append({
                "parent_doc": parent,
                "chunk_index": idx,
                "total_chunks": total,
                "mandalas": mandala_list[:10],
                "linked_neurons": linked,
                "content": content[:1500]
            })
            if len(results) >= limit:
                break

        conn.close()
        self._json({"results": results, "total_docs_matched": len(seen_parents)})

    def _crossover(self, body):
        """Combine two chunks into a derived entry.

        Two source chunks (parent_a, parent_b). Tags shared between them,
        plus tags unique to either side, all carry over to the result,
        plus a small number of new synthetic tags generated by pairing
        tags across the two sources.

        Rule of thumb for how many synthetic tags to add: 2x + 1, capped
        at 30, where x = number of shared tags. Chosen because more shared
        context tends to give the pairing step more to work with — not
        derived from any formal principle.
        """
        doc_a = body["parent_a"]
        doc_b = body["parent_b"]
        collection = body.get("collection", "default")

        conn = chroma_conn()
        cur = conn.cursor()

        # Fetch mandalas from both parents
        def get_mandalas(doc_id):
            cur.execute("""
                SELECT em.string_value FROM embedding_metadata em
                WHERE em.id IN (
                    SELECT e.seq_id FROM embeddings e
                    JOIN embedding_metadata p ON e.seq_id = p.id
                    WHERE p.key='source' AND p.string_value=?
                ) AND em.key='mandalas'
                LIMIT 1
            """, (doc_id,))
            row = cur.fetchone()
            return set(row[0].split()) if row else set()

        # Fetch content from both parents
        def get_content(doc_id):
            cur.execute("""
                SELECT em.string_value FROM embedding_metadata em
                WHERE em.id IN (
                    SELECT e.seq_id FROM embeddings e
                    JOIN embedding_metadata p ON e.seq_id = p.id
                    WHERE p.key='source' AND p.string_value=?
                ) AND em.key='chroma:document'
                LIMIT 1
            """, (doc_id,))
            row = cur.fetchone()
            return row[0] if row else ""

        mandalas_a = get_mandalas(doc_a)
        mandalas_b = get_mandalas(doc_b)
        content_a = get_content(doc_a)
        content_b = get_content(doc_b)

        # ── TAG INHERITANCE ───────────────────────
        # Shared tags carry over
        inherited_tags = mandalas_a & mandalas_b
        # Tags unique to either side also carry over
        inherited_tags = inherited_tags | (mandalas_a ^ mandalas_b)  # XOR = union of the two "unique" sets
        # Tags absent from both sources are never included

        # Heuristic cap on new synthetic tags: 2x + 1, x = shared-tag count
        x = len(mandalas_a & mandalas_b)
        n_offspring = min(x * 2 + 1, 30)

        # Generate the synthetic mandala (the "+1" of the formula)
        # Emerges from the combination of the two parents
        synthetic_mandalas = set()
        for ma in list(mandalas_a)[:5]:
            for mb in list(mandalas_b)[:5]:
                if ma != mb:
                    # Combine categories: #tech:database + #entity:acme → #hybrid:database_acme
                    cat_a = ma.split(":")[0] if ":" in ma else "#hybrid"
                    name_a = ma.split(":")[1] if ":" in ma else ma.replace("#","")
                    name_b = mb.split(":")[1] if ":" in mb else mb.replace("#","")
                    synthetic_mandalas.add(f"{cat_a}:{name_a}_{name_b}"[:60])
        synthetic_mandalas = set(list(synthetic_mandalas)[:5])

        # Final offspring = inherited tags + synthetic
        offspring_mandalas = sorted(inherited_tags)[:n_offspring] + sorted(synthetic_mandalas)
        offspring_mandalas = offspring_mandalas[:n_offspring]

        # ── Generate offspring content ──────────
        # Content crossover: first N lines of A + last M of B
        lines_a = content_a.split("\n")
        lines_b = content_b.split("\n")
        crossover_point = len(lines_a) // 2
        offspring_content = (
            f"/* OFFSPRING: {doc_a[:12]} × {doc_b[:12]} */\n"
            f"/* Inherited tags: {sorted(inherited_tags)[:5]} */\n"
            f"/* Synthetic: {sorted(synthetic_mandalas)[:5]} */\n"
            + "\n".join(lines_a[:crossover_point])
            + "\n\n/* ── CROSSOVER ── */\n\n"
            + "\n".join(lines_b[crossover_point:])
        )

        # ── Store offspring ─────────────────────
        col_id, _ = ensure_collection(cur, collection)
        seg_id, _ = ensure_segment(cur, col_id)
        offspring_id = doc_hash(offspring_content)
        embedding_id = f"offspring_{offspring_id}_{x}"
        seq = next_seq_id(cur)

        insert_chunk(cur, seg_id, seq, embedding_id, offspring_id, 0, 1,
                     offspring_content, offspring_mandalas, collection, col_id)
        conn.commit()
        conn.close()

        # ── Tag overlap summary ──────────────────
        shared = len(mandalas_a & mandalas_b)
        unique_a = len(mandalas_a - mandalas_b)
        unique_b = len(mandalas_b - mandalas_a)

        overlap_class = "high" if shared > max(unique_a, unique_b) else ("some" if shared > 0 else "none")

        self._json({
            "offspring_id": offspring_id,
            "parents": {"a": doc_a, "b": doc_b},
            "tag_summary": {
                "overlap": overlap_class,
                "shared_tags": shared,
                "unique_tags": unique_a + unique_b,
                "synthetic_tags": len(synthetic_mandalas),
                "total_offspring_tags": len(offspring_mandalas),
                "synthetic_tag_formula": f"{shared}×2+1 = {n_offspring}"
            },
            "offspring_mandalas": offspring_mandalas[:10],
            "content_preview": offspring_content[:500]
        })

    def _population(self, body):
        """Tag-frequency analysis across a collection.

        For each tag in the collection:
          p = fraction of chunks that carry the tag
          q = 1 - p

        This borrows the shape of a Hardy-Weinberg comparison (expected
        vs. observed frequency, via p² / 2pq / q²) as a heuristic signal
        for whether a tag's spread looks "expected" or unusual — it is
        not a rigorous population-genetics analysis. In particular, the
        "observed" counts below are simple presence/absence counts per
        chunk, not real heterozygote/homozygote genotype observations,
        so treat the chi-squared value as a rough signal, not a
        statistically rigorous test.
        """
        collection = body["collection"]
        conn = chroma_conn()
        cur = conn.cursor()

        # Fetch all mandalas and their frequencies
        cur.execute("""
            SELECT em.string_value FROM embedding_metadata em
            JOIN embeddings e ON em.id = e.seq_id
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ? AND em.key = 'mandalas'
        """, (collection,))
        
        all_mandalas = []
        for (mstr,) in cur.fetchall():
            all_mandalas.extend(mstr.split())
        
        if not all_mandalas:
            conn.close()
            self._json({"error": "no mandalas in collection"})
            return

        # Frequency of each mandala
        total_chunks = cur.execute("""
            SELECT COUNT(DISTINCT e.seq_id) FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ?
        """, (collection,)).fetchone()[0] or 1

        from collections import Counter
        mandala_counts = Counter(all_mandalas)
        
        population_analysis = []
        for mandala, count in mandala_counts.most_common(30):
            p = count / total_chunks  # allele frequency B
            q = 1 - p                  # allele frequency b
            
            # Expected frequencies if tag presence followed a
            # Hardy-Weinberg-shaped distribution (heuristic, not literal biology)
            BB_expected = p * p * total_chunks
            Bb_expected = 2 * p * q * total_chunks
            bb_expected = q * q * total_chunks
            
            # Observed: how many chunks actually carry this tag
            cur.execute("""
                SELECT COUNT(DISTINCT e.seq_id) FROM embedding_metadata em
                JOIN embeddings e ON em.id = e.seq_id
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ? AND em.key = 'mandalas' AND em.string_value LIKE ?
            """, (collection, f"%{mandala}%"))
            BB_obs = cur.fetchone()[0]  # chunks that have the tag
            bb_obs = total_chunks - BB_obs  # chunks that don't
            Bb_obs = 0  # not tracked separately at this level of detail
            
            # Simple chi-squared comparison (rough signal, not a rigorous test)
            chi2 = 0
            if BB_expected > 0:
                chi2 += (BB_obs - BB_expected)**2 / BB_expected
            if bb_expected > 0:
                chi2 += (bb_obs - bb_expected)**2 / bb_expected
            
            # Interpretation
            if chi2 < 0.5:
                status = "EQUILIBRIUM"
            elif p > 0.7:
                status = "DOMINANT_SWEEP"  # mandala dominating the population
            elif p < 0.1:
                status = "NEAR_EXTINCTION"  # about to disappear
            elif chi2 > 3:
                status = "SELECTION_PRESSURE"
            else:
                status = "DRIFT"
            
            population_analysis.append({
                "mandala": mandala,
                "frequency_B": round(p, 3),
                "frequency_b": round(q, 3),
                "observed": {"has_mandala": BB_obs, "no_mandala": bb_obs},
                "expected_HW": {"BB": round(BB_expected, 1), "Bb": round(Bb_expected, 1), "bb": round(bb_expected, 1)},
                "chi2": round(chi2, 3),
                "status": status
            })

        conn.close()
        
        # Population summary
        equilibrium = sum(1 for a in population_analysis if a["status"] == "EQUILIBRIUM")
        sweeps = sum(1 for a in population_analysis if a["status"] == "DOMINANT_SWEEP")
        extinct = sum(1 for a in population_analysis if a["status"] == "NEAR_EXTINCTION")
        
        self._json({
            "collection": collection,
            "population_size": total_chunks,
            "unique_mandalas": len(mandala_counts),
            "summary": {
                "in_equilibrium": equilibrium,
                "dominant_sweeps": sweeps,
                "near_extinction": extinct,
                "evolving": len(population_analysis) - equilibrium - sweeps - extinct
            },
            "top_mandalas": population_analysis[:15]
        })

    def _equilibrium(self, body):
        """Boost under-represented tags by pairing them with common ones.

        Identifies tags with p < 0.1 (rare) and pairs them with tags that
        have p > 0.7 (common), creating a small derived entry that carries
        both — a heuristic way to keep rare-but-present tags from dropping
        out of the collection entirely. "Hardy-Weinberg" language has been
        removed here; this is a rarity-boosting rule, not a genetics
        computation.
        """
        collection = body["collection"]
        conn = chroma_conn()
        cur = conn.cursor()

        # Same analysis as _population but action-oriented
        cur.execute("""
            SELECT em.string_value FROM embedding_metadata em
            JOIN embeddings e ON em.id = e.seq_id
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ? AND em.key = 'mandalas'
        """, (collection,))
        
        all_mandalas = []
        for (mstr,) in cur.fetchall():
            all_mandalas.extend(mstr.split())
        
        if not all_mandalas:
            conn.close()
            self._json({"error": "no mandalas"})
            return

        total_chunks = cur.execute("""
            SELECT COUNT(DISTINCT e.seq_id) FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ?
        """, (collection,)).fetchone()[0] or 1

        from collections import Counter
        mandala_counts = Counter(all_mandalas)
        
        # Classify mandalas
        dominant = []   # p > 0.7
        stable = []     # 0.3 < p < 0.7
        endangered = [] # p < 0.1
        
        for mandala, count in mandala_counts.items():
            p = count / total_chunks
            if p > 0.7:
                dominant.append((mandala, p))
            elif p < 0.1:
                endangered.append((mandala, p))
            else:
                stable.append((mandala, p))
        
        # Pair rare tags with common ones to keep them from dropping out
        rescued = []
        for em, ep in endangered[:5]:
            for dm, dp in dominant[:3]:
                # Create a small derived entry carrying the rare tag
                offspring_mandalas = [em, dm, f"#boost:{em.split(':')[-1] if ':' in em else em}_{dm.split(':')[-1] if ':' in dm else dm}"[:60]]
                
                hybrid_id = doc_hash(em + dm + "tag_boost")
                col_id, _ = ensure_collection(cur, collection)
                seg_id, _ = ensure_segment(cur, col_id)
                seq = next_seq_id(cur)
                
                content = f"/* TAG BOOST: {em} × {dm} */\nPaired rare tag {em} with common tag {dm} to keep it from dropping out of rotation."
                
                insert_chunk(cur, seg_id, seq, f"boost_{hybrid_id}", hybrid_id, 0, 1, 
                             content, offspring_mandalas, collection, col_id)
                
                rescued.append({
                    "rare_tag": em,
                    "common_tag": dm,
                    "offspring_id": hybrid_id[:16],
                    "new_frequency_estimate": round((ep * total_chunks + 1) / (total_chunks + 1), 3)
                })
                break  # one boost per rare tag

        conn.commit()
        conn.close()
        
        self._json({
            "collection": collection,
            "population_before": {
                "common_tags": len(dominant),
                "stable_tags": len(stable),
                "rare_tags": len(endangered)
            },
            "boosts_performed": len(rescued),
            "boosted_tags": rescued
        })

    def _steady_state(self, body):
        """Score how "balanced" a pair of chunks is, using a chosen curve.

        x = fraction of tags unique to chunk A
        y = fraction of tags unique to chunk B
        z = fraction of tags shared (implicit: z = 1 - x - y)

        Reference curve: 2xy + x + y = 1 — a threshold shape chosen
        because it penalizes both near-duplicate pairs and unrelated
        pairs, not a physical law.
          - On the curve → balanced
          - Well above it → the pair is mostly non-overlapping
          - Well below it → the pair is mostly overlapping (near-duplicate)
        """
        doc_a = body["parent_a"]
        doc_b = body["parent_b"]
        collection = body.get("collection", "default")

        conn = chroma_conn()
        cur = conn.cursor()

        def get_mandalas(doc_id):
            cur.execute("""
                SELECT em.string_value FROM embedding_metadata em
                JOIN embeddings e ON em.id = e.seq_id
                JOIN embedding_metadata p ON e.seq_id = p.id
                WHERE p.key='source' AND p.string_value=? AND em.key='mandalas'
                LIMIT 1
            """, (doc_id,))
            row = cur.fetchone()
            return set(row[0].split()) if row else set()

        mandalas_a = get_mandalas(doc_a)
        mandalas_b = get_mandalas(doc_b)
        conn.close()

        # Tag overlap components
        expansion = len(mandalas_a - mandalas_b)    # tags unique to A
        contraction = len(mandalas_b - mandalas_a)  # tags unique to B
        shared = len(mandalas_a & mandalas_b)       # tags shared by both
        
        # Normalize as fractions of the combined tag pool
        total = expansion + contraction + shared
        x = expansion / max(total, 1)
        y = contraction / max(total, 1)
        
        # Distance from the reference curve 2xy + x + y = 1
        curve_value = 2 * x * y + x + y
        deviation = curve_value - 1  # 0 = sitting on the curve
        
        # Interpretation against the curve 2xy + x + y = 1
        # deviation = 0 → on the curve (balanced)
        # deviation > 0 → mostly non-overlapping
        # deviation < 0 → mostly overlapping
        
        if abs(deviation) < 0.1:
            state = "BALANCED"
            interpretation = f"On the curve: 2({x:.2f})({y:.2f}) + {x:.2f} + {y:.2f} = {curve_value:.2f} ≈ 1."
        elif deviation < 0:
            state = "OVERLAPPING"
            interpretation = f"Below curve: {curve_value:.2f} < 1. Pair is mostly overlapping (near-duplicate)."
        elif deviation < 1.0:
            state = "MILD_GAP"
            interpretation = f"Slightly above curve: {curve_value:.2f}. Some non-overlap, within a normal range."
        elif deviation < 5.0:
            state = "LOW_OVERLAP"
            interpretation = f"Above curve: {curve_value:.2f}. Pair shares little in common."
        else:
            state = "NO_OVERLAP"
            interpretation = f"Far above curve: {curve_value:.2f}. Pair shares almost nothing."

        # Action
        if state == "BALANCED":
            action = "CROSSOVER_SAFE"
        elif state in ("MILD_GAP", "OVERLAPPING"):
            action = "CROSSOVER_WITH_CAUTION"
        elif state == "LOW_OVERLAP":
            action = "PAIR_CLOSER_DOCS"
        else:
            action = "SKIP_PAIR"

        # ─── SECONDARY CHECK: x = y^z ─────────────
        # A second, independently-chosen scoring curve on the same
        # three numbers — an extra filter, not a physical law.
        z_hybrid = shared / max(total, 1)
        if y > 0 and z_hybrid > 0:
            check_value = y ** z_hybrid
            check_dev = x - check_value
        elif z_hybrid == 0:
            check_value = 1.0
            check_dev = x - 1.0
        else:
            check_value = 0.0
            check_dev = x

        check_passes = abs(check_dev) < 0.05

        self._json({
            "parents": {"a": doc_a[:16], "b": doc_b[:16]},
            "tag_overlap": {
                "x_unique_a": round(x, 3),
                "y_unique_b": round(y, 3),
                "shared_count": shared,
                "raw_unique_a": expansion,
                "raw_unique_b": contraction,
                "curve_equation": f"2xy + x + y = {round(curve_value, 3)}",
                "target": 1,
                "deviation": round(deviation, 3),
            },
            "secondary_check": {
                "equation": f"x = y^z → {round(x,3)} = {round(y,3)}^{round(z_hybrid,3)}",
                "check_value": round(check_value, 3),
                "deviation": round(check_dev, 3),
                "passes": check_passes,
            },
            "curve_value": round(curve_value, 3),
            "deviation": round(deviation, 3),
            "state": state,
            "interpretation": interpretation,
            "recommended_action": action
        })

    def _invariant(self, body):
        """Secondary balance check on a pair, using a different curve: x = y^z

        x = fraction of tags unique to chunk A
        y = fraction of tags unique to chunk B
        z = fraction of tags shared by both

        This is a second, independently-chosen scoring curve — used
        alongside /steady so a pair has to look reasonable on two
        different scales before being combined, not a biological law.
        - If x ≈ y^z → the pair scores well on this second curve too
        - If not → the pair is lopsided on this curve (may still be a
          fine pair; this is one signal among several, not a verdict)
        """
        doc_a = body["parent_a"]
        doc_b = body["parent_b"]
        collection = body.get("collection", "default")

        conn = chroma_conn()
        cur = conn.cursor()

        def get_mandalas(doc_id):
            cur.execute("""
                SELECT em.string_value FROM embedding_metadata em
                JOIN embeddings e ON em.id = e.seq_id
                JOIN embedding_metadata p ON e.seq_id = p.id
                WHERE p.key='source' AND p.string_value=? AND em.key='mandalas'
                LIMIT 1
            """, (doc_id,))
            row = cur.fetchone()
            return set(row[0].split()) if row else set()

        mandalas_a = get_mandalas(doc_a)
        mandalas_b = get_mandalas(doc_b)
        conn.close()

        # Tag overlap: x=unique to A, y=unique to B, z=shared
        expansion = len(mandalas_a - mandalas_b)    # tags unique to A
        contraction = len(mandalas_b - mandalas_a)  # tags unique to B
        shared = len(mandalas_a & mandalas_b)       # tags shared by both

        total = expansion + contraction + shared
        x = expansion / max(total, 1)
        y = contraction / max(total, 1)
        z = shared / max(total, 1)

        # ─── SECONDARY CHECK: x = y^z ─────────────
        if y > 0 and z > 0:
            check_value = y ** z
            deviation = x - check_value
        elif z == 0:
            # No shared tags: x should be 1 (everything is unique to A)
            check_value = 1.0
            deviation = x - 1.0
        else:
            # y = 0: nothing unique to B
            check_value = 0.0
            deviation = x

        check_passes = abs(deviation) < 0.05

        # Diagnosis
        if check_passes:
            verdict = "BALANCED"
            interpretation = (
                f"x = y^z → {x:.3f} = {y:.3f}^{z:.3f} = {check_value:.3f}. "
                "Pair scores well on this curve."
            )
        elif deviation > 0:
            verdict = "SKEWED_TOWARD_A"
            interpretation = (
                f"x > y^z → {x:.3f} > {check_value:.3f}. "
                "More unique to A than this curve expects for a good pairing."
            )
        else:
            verdict = "SKEWED_TOWARD_B"
            interpretation = (
                f"x < y^z → {x:.3f} < {check_value:.3f}. "
                "Less unique to A than this curve expects for a good pairing."
            )

        # Cross-check against the /steady curve
        curve_value = 2 * x * y + x + y
        steady_dev = curve_value - 1

        self._json({
            "parents": {"a": doc_a[:16], "b": doc_b[:16]},
            "tag_overlap": {
                "x_unique_a": round(x, 3),
                "y_unique_b": round(y, 3),
                "z_shared": round(z, 3),
                "check_equation": f"x = y^z → {round(x, 3)} = {round(y, 3)}^{round(z, 3)}",
                "check_value": round(check_value, 3),
                "deviation": round(deviation, 3),
                "passes": check_passes,
            },
            "steady_curve": {
                "curve_2xy_plus_x_plus_y": round(curve_value, 3),
                "deviation_from_1": round(steady_dev, 3),
            },
            "verdict": verdict,
            "interpretation": interpretation,
            "recommended_action": (
                "CROSSOVER_SAFE" if check_passes
                else "CROSSOVER_WITH_CAUTION" if abs(deviation) < 0.2
                else "SKIP_PAIR"
            ),
        })

    def _breed(self, body):
        """Run N rounds of scoring, combining, and pruning across a collection.

        OPTIMIZATION: only score and test √n · n/2 pairs per round instead
        of all n·(n-1)/2 — a practical cap to keep larger collections fast,
        not a proven statistical bound.

        FILTERS:
          1. shared=0 → immediate skip (no overlap, no combining)
          2. max_docs → initial pool cap for large collections

        TWO SCORING CURVES (both must pass, within tolerance):
          1. Balance curve: |2xy + x + y - 1| ≤ tolerance
          2. Secondary check: |x - y^z| ≤ tolerance

        Runs for N rounds. Only pairs that pass both curves survive into
        the next round's pool.
        """
        collection = body["collection"]
        generations = body.get("generations", 3)
        tolerance = body.get("tolerance", 0.3)
        max_docs = body.get("max_docs", 200)  # cap para colecciones gigantes

        conn = chroma_conn()
        cur = conn.cursor()

        # Fetch all original docs (generation 0)
        cur.execute("""
            SELECT DISTINCT em.string_value FROM embedding_metadata em
            JOIN embeddings e ON em.id = e.seq_id
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ? AND em.key = 'source'
        """, (collection,))
        gene_pool = [r[0] for r in cur.fetchall()]
        conn.close()

        # Apply max_docs: take a representative sample if too many
        original_pool_size = len(gene_pool)
        if len(gene_pool) > max_docs:
            import random
            random.seed(42)
            gene_pool = random.sample(gene_pool, max_docs)

        if len(gene_pool) < 2:
            self._json({"error": "need at least 2 docs in collection", "pool": len(gene_pool)})
            return

        history = []
        current_pool = gene_pool[:]
        total_created = 0
        total_purged = 0
        total_skipped = 0  # pares no probados (optimización)

        for gen in range(generations):
            if len(current_pool) < 2:
                break

            n = len(current_pool)
            total_pairs = n * (n - 1) // 2
            max_pairs = int(math.sqrt(n) * n / 2)
            # Cap: no superar total_pairs ni bajar de 1
            max_pairs = max(1, min(max_pairs, total_pairs))

            gen_created, gen_purged, gen_skipped = 0, 0, 0
            survivors = []
            scored_pairs = []

            # ─── PHASE 1: Pre-score without opening the DB ─────
            # Fast scan: only count mandalas for ranking
            conn = chroma_conn()
            cur = conn.cursor()

            def get_mandalas_fast(doc_id):
                cur.execute("""
                    SELECT em.string_value FROM embedding_metadata em
                    JOIN embeddings e ON em.id = e.seq_id
                    JOIN embedding_metadata p ON e.seq_id = p.id
                    WHERE p.key='source' AND p.string_value=? AND em.key='mandalas'
                    LIMIT 1
                """, (doc_id,))
                row = cur.fetchone()
                return set(row[0].split()) if row else set()

            # Mandala cache by doc_id
            mandala_cache = {}
            for doc_id in current_pool:
                mandala_cache[doc_id] = get_mandalas_fast(doc_id)
            conn.close()

            # Score all pairs by closeness to the two curves
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = current_pool[i], current_pool[j]
                    ma = mandala_cache.get(a, set())
                    mb = mandala_cache.get(b, set())
                    if not ma or not mb:
                        gen_skipped += 1
                        continue

                    expansion = len(ma - mb)
                    contraction = len(mb - ma)
                    shared = len(ma & mb)
                    total = expansion + contraction + shared
                    if total == 0 or shared == 0:
                        gen_skipped += 1
                        continue

                    x = expansion / total
                    y = contraction / total
                    z = shared / total

                    # Combined score: balance curve + secondary check
                    classic_score = abs(2 * x * y + x + y - 1)
                    
                    if y > 0 and z > 0:
                        invariant_score = abs(x - y ** z)
                    elif z == 0:
                        invariant_score = abs(x - 1.0)
                    else:
                        invariant_score = abs(x)

                    # Combined score: the worse of the two (both must be decent)
                    score = max(classic_score, invariant_score)

                    scored_pairs.append((score, a, b, x, y, z, classic_score, invariant_score))

            # ─── PHASE 2: Sort and take the best ─────
            scored_pairs.sort(key=lambda p: p[0])
            
            if len(scored_pairs) > max_pairs:
                gen_skipped += len(scored_pairs) - max_pairs
                scored_pairs = scored_pairs[:max_pairs]

            # ─── PHASE 3: Breed the best ─────
            pair_idx = 0
            for score, a, b, x, y, z, classic_score, invariant_score in scored_pairs:
                if score <= tolerance:
                    # Viable: do the crossover
                    ma = mandala_cache.get(a, set())
                    mb = mandala_cache.get(b, set())

                    conn = chroma_conn()
                    cur = conn.cursor()
                    col_id, _ = ensure_collection(cur, collection)
                    seg_id, _ = ensure_segment(cur, col_id)

                    offspring_mandalas = sorted(ma | mb)
                    offspring_content = (
                        f"/* GEN {gen}: {a[:20]} × {b[:20]} */\n"
                        f"/* classic={classic_score:.3f} invariant={invariant_score:.3f} */\n"
                        f"/* x={x:.3f} y={y:.3f} z={z:.3f} */\n" +
                        "\n".join([f"# {m}" for m in offspring_mandalas[:20]])
                    )

                    offspring_id = doc_hash(offspring_content + f"_{gen}_{pair_idx}_{time.time()}")
                    seq = next_seq_id(cur)
                    insert_chunk(cur, seg_id, seq, f"gen{gen}_{offspring_id}", offspring_id, 0, 1,
                                 offspring_content, offspring_mandalas, collection, col_id)
                    conn.commit()
                    conn.close()

                    survivors.append(offspring_id)
                    gen_created += 1
                else:
                    gen_purged += 1
                pair_idx += 1  # always advance, viable or not

            total_skipped += gen_skipped

            history.append({
                "generation": gen,
                "pool_size": n,
                "total_pairs": total_pairs,
                "max_pairs_limit": max_pairs,
                "pairs_tested": len(scored_pairs),
                "pairs_skipped": gen_skipped,
                "created": gen_created,
                "purged": gen_purged,
                "survivors": gen_created,
                "optimization": "√n·n/2",
            })

            total_created += gen_created
            total_purged += gen_purged

            if survivors:
                current_pool = survivors
            else:
                break

        self._json({
            "collection": collection,
            "original_pool": original_pool_size,
            "sampled_pool": len(gene_pool),
            "generations_run": len(history),
            "total_created": total_created,
            "total_purged": total_purged,
            "total_skipped": total_skipped,
            "survival_rate": f"{total_created}/{total_created + total_purged} ({round(100*total_created/max(total_created+total_purged,1))}%)",
            "optimization": f"√n·n/2 ({round(100*(total_created+total_purged)/max(total_created+total_purged+total_skipped,1))}% tested)",
            "validators": ["|2xy+x+y-1| ≤ tol", "|x-y^z| ≤ tol"],
            "tolerance": tolerance,
            "history": history,
            "final_pool": current_pool[:10],
        })

    def _get_neuron(self, doc_id):
        conn = chroma_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT em.key, em.string_value FROM embedding_metadata em
            WHERE em.id IN (SELECT e.seq_id FROM embeddings e
                            JOIN embedding_metadata p ON e.seq_id = p.id
                            WHERE p.key='source' AND p.string_value=?)
            ORDER BY em.id, em.key
        """, (doc_id,))
        rows = cur.fetchall()
        conn.close()

        neuron = {"doc_id": doc_id, "chunks": {}}
        current_chunk = None
        for key, val in rows:
            if key == "chunk_index":
                current_chunk = val
                neuron["chunks"][val] = {}
            elif current_chunk and key in ("chroma:document", "mandalas", "collection"):
                neuron["chunks"][current_chunk][key] = val[:500] if key == "chroma:document" else val

        self._json(neuron)

    # ── helpers ───────────────────────────────────────────

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Mandala", "v1")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"  🕸️  MANDALA BRIDGE → http://127.0.0.1:{PORT}")
    print(f"     ChromaDB: {CHROMA}")
    print(f"     Chunk: {CHUNK_LINES} lines")
    print(f"     Each chunk = neuron, tags = synapses")
    HTTPServer(("127.0.0.1", PORT), MandalaHandler).serve_forever()
