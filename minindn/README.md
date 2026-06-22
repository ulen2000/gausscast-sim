# GaussCast Mini-NDN harness

A self-contained **Mini-NDN / NFD emulation harness** that runs the GaussCast
shared-block retrieval mechanism over **real NFD forwarders moving real Data
packets**. It complements the trace-driven delivery simulator in
[`../gausscast_sim`](../gausscast_sim): the simulator does full-scale byte and
fairness accounting over EyeNavGS traces, while this harness demonstrates that
the same delivery mechanism operates as described on an actual Named Data
Networking dataplane — named layered blocks, PIT Interest aggregation, Content
Store reuse, digest-linked verification, and closure-aware planning.

The harness is **pure standard library + [python-ndn]**, with no `numpy`,
`pandas`, or dataset dependency, so each module also runs inside the per-node
namespaces that Mini-NDN creates.

[python-ndn]: https://github.com/named-data/python-ndn

## What it demonstrates

The paper's prototype plans a **shared base** of prerequisite blocks that is
fetched once upstream and referenced by every user that needs it, then spends the
remaining budget on per-user refinement layers, admitting each block only
together with its still-missing dependency closure. On an NDN dataplane this maps
directly onto two forwarder mechanisms:

* **PIT aggregation** — concurrent Interests for the same name collapse into a
  single upstream Interest at each forwarder;
* **Content Store reuse** — a name already fetched is served from an on-path
  cache without reaching the origin.

Because every segment name carries a content digest
(`.../seg/<s>/<digest>`), a cached copy is **self-certifying**: the consumer
recomputes SHA-256 over the payload and checks it against the name, so a segment
served from any cache is accepted without contacting the publisher, yet still
verified.

The harness measures, per policy, how many segments the **origin actually
serves** versus how many Interests the clients express. When many users share
lower-layer prerequisites, the origin serves each distinct name **once** and the
forwarders satisfy the repeats.

## Topology

The default is the paper's three-tier tree: one publisher, one core forwarder,
two aggregation forwarders, four edge forwarders, and synthetic users attached at
the edge layer. Each edge runs the planner (Algorithm 1) over the demand of the
users attached to it.

```
      publisher --- core --- agg0 --- edge0, edge1
                          \-- agg1 --- edge2, edge3
```

Links are direct host-to-host links (no software switch), and carry no `tc`
qdisc, so the harness runs on kernels without `sch_netem`. Latency and bandwidth
accounting are covered by the trace-driven simulator; here the focus is the
forwarding mechanism (aggregation, caching, verification, dependency ordering).

## Components

| File | Role |
|---|---|
| `gc_blocks.py` | Cell/layer/chunk byte model, full-prefix closures, and the versioned digest-linked NDN naming `/gc/scene/<scid>/ver/<v>/cell/<cid>/layer/<l>/chunk/<k>/seg/<s>/<digest>`. Mirrors `gausscast_sim/scene_model.py` without `numpy`. |
| `gc_planner.py` | Closure-aware two-stage planner (shared base ranked by cross-user sharing density, then per-user supplements) and a synthetic multi-user demand generator with a controllable cross-user overlap. |
| `gc_producer.py` | Origin producer: serves digest-named segment Data on demand (stateless, payload regenerated from `gc_blocks`), counting the segments it actually serves — the upstream fetch volume. |
| `gc_consumer.py` | Client: expands a per-user plan into segment Interests, fetches them over NFD, and verifies each payload's SHA-256 against the name digest; records time-to-first-segment and useful/late counts. |
| `gc_experiment.py` | Orchestrator: builds the tree, starts NFD on every node, configures upstream `/gc` routes, runs the planner per policy, dispatches per-user plans to the edges concurrently, and reports origin vs. client volumes. |

## Requirements

* [Mini-NDN](https://github.com/named-data/mini-ndn) with NFD installed
  (the orchestrator uses `minindn` and `mininet`, run as root).
* `python-ndn` (already present in the Mini-NDN node Python).

## Usage

```bash
# from a Mini-NDN environment, as root
sudo python3 gc_experiment.py \
    --scene truck --users 8 --overlap 0.6 \
    --policies PerUser-ICN,GC-noClosure,GC-Full \
    --out results.json
```

The policy flags select the planning variant:

| Policy | Shared base | Closure admission |
|---|---|---|
| `PerUser-ICN`  | – | full prefix |
| `GC-noClosure` | ✓ | target layer only |
| `GC-Full`      | ✓ | full prefix |

Individual modules can also be exercised standalone, e.g. `python3 gc_blocks.py`
and `python3 gc_planner.py` print a self-test of the byte model and the planner's
sharing behavior without requiring NFD.

## Reported metrics

For each policy the orchestrator prints a row:

| Column | Meaning |
|---|---|
| `distinct` | distinct segments in the plan — the theoretical upstream floor |
| `origin` | segments the publisher actually served (upstream fetch volume) |
| `cli_int` | Interests expressed across all clients |
| `recv` | Data packets the clients received |
| `verif` | payloads whose SHA-256 matched the name digest |
| `reuse` | `1 − origin / cli_int` — share of client Interests satisfied by aggregation and on-path caches |
| `ttff_ms` | mean time to first received segment |

`origin` converging to `distinct` while `cli_int` is larger is the mechanism made
measurable: each distinct name is fetched from the origin once and the repeated
Interests are collapsed on path. The gap `cli_int − distinct` (the collapsed
Interests) grows as cross-user overlap rises, which is the shared-base reuse the
paper describes.

## Relationship to the simulator

This harness runs the GaussCast retrieval mechanism on a **real NDN dataplane**,
validating on live NFD forwarders the behaviors the simulator accounts for: named
layered blocks, PIT Interest aggregation, Content Store reuse, digest-linked
verification, and closure-aware planning. End-to-end byte ratios, edge hit ratios,
late-miss ratios, and Jain fairness over full-scale scenes and real EyeNavGS traces
are produced by [`../gausscast_sim`](../gausscast_sim). The two share the same block
model, naming, dependency closures, and planning logic, so the dataplane results and
the trace-driven results corroborate one another.

