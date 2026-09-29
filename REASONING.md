# Why a "breeding" engine?

Standard RAG is a lookup: chunk documents, embed them, retrieve by similarity. Nothing has a lifecycle — stale chunks accumulate forever, and there's no signal for what's still useful.

I wanted a retrieval layer that also decides what to keep and what to let go — so I built a heuristic scoring system on top of tag overlap between chunks, and gave it a genetics-flavored vocabulary because it made the design easier to reason about while building it. Worth saying plainly: the equations below are chosen design constraints, not derived scientific laws. An earlier version of this doc implied otherwise — that was overclaiming, and it's fixed here.

## The model

Treat a collection as a population. Each chunk carries tags ("mandalas"). Two chunks can be combined: shared tags carry over, and the combination gets one new synthetic tag generated from pairing tags across the two chunks.

Three numbers drive the scoring, per pair of chunks:

- `x` = tags unique to chunk A
- `y` = tags unique to chunk B
- `z` = tags shared by both

These are normalized so x + y + z = 1 — true by construction, not a discovery.

## The scoring

Two curves decide whether a pair is worth combining:

1. **Balance curve** — `2xy + x + y = 1`. A pair sitting close to this curve gets treated as "balanced": not near-duplicates, not totally unrelated. This is a chosen threshold shape, not a derived law — I picked it because it naturally penalizes both ends (near-duplicate pairs and unrelated pairs) better than a flat similarity cutoff did in testing.
2. **Secondary check** — `x = y^z`. An independent second filter on the same three numbers. A pair has to look reasonable on two different scales before it's combined, not just one.

Both curves are heuristics I chose because they produced results I liked — not principles borrowed validly from genetics, physics, or population biology. Naming them "Hardy-Weinberg" or "biological invariant" was more evocative than accurate, and that language has been toned down here and in the README.

## The optimization

Combining every chunk against every other is O(n²) and wasteful for larger collections. Instead of testing all pairs, `/breed` pre-scores every pair using cached tag sets (cheap), sorts by closeness to the two curves, and only writes the best-looking √n · n/2 of them to the database. This is a practical compute-saving heuristic, not a proven statistical bound — it keeps things fast on bigger collections without a real accuracy cost I could measure at the scale I've tested so far.

## What I'd do differently

- The tag generator is regex + keyword extraction. Fine for structured docs, weak on prose. A small local embedding model would be a better tagger.
- `/query` uses `LIKE` substring matching, not SQLite FTS5, despite what an earlier version of the README claimed. Swapping in real FTS5 is a natural next step and wouldn't take long.
- I'd want to actually test the "~30% of pairs are viable" claim across a range of collection sizes before stating it as a number, rather than asserting it from a handful of runs.
