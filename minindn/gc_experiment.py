#!/usr/bin/env python3
r"""
gc_experiment.py
----------------
Mini-NDN orchestrator for the GaussCast retrieval harness. It demonstrates, on
REAL NFD forwarders moving REAL Data packets, the delivery mechanism the paper
describes: an edge proxy plans a shared base of prerequisite blocks that is
fetched once upstream (PIT aggregation + Content Store reuse) and referenced by
every user that needs it, while higher refinement layers are fetched per user.

Topology (the paper's default three-tier tree, host-to-host links so no software
switch is required):

      publisher --- core --- agg0 --- edge0, edge1
                          \-- agg1 --- edge2, edge3

Clients are attached at the edge layer (one consumer process per synthetic user,
co-located in its edge node's namespace, talking to the edge NFD). Each edge runs
the planner (Algorithm 1) over the demand of the users attached to it.

For each policy the orchestrator records:
  * origin_served       segments the publisher actually served (UPSTREAM volume)
  * client_interests    Interests expressed across all clients
  * client_received     Data packets clients received
  * client_verified     payloads whose SHA-256 matched the name digest
  * reuse_ratio         1 - origin_served / client_interests  (aggregation+cache)

A lower origin_served at equal client demand is the shared-base benefit; it is
produced purely by NFD's PIT aggregation and CS reuse over the planned names.

Run (root, inside the Mini-NDN environment):
  sudo python3 gc_experiment.py --scene truck --users 8 --policies GC-Full,PerUser-ICN
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time

from mininet.log import setLogLevel, info
from minindn.minindn import Minindn
from minindn.apps.app_manager import AppManager
from minindn.apps.nfd import Nfd
from minindn.util import getPopen

from gc_blocks import SceneBlocks
from gc_planner import make_users, plan, plan_stats

CWD = os.path.dirname(os.path.abspath(__file__))

# Edge node names and the tree wiring (child -> parent, toward the publisher).
PUB, CORE = 'pub', 'core'
AGGS = ['agg0', 'agg1']
EDGES = ['edge0', 'edge1', 'edge2', 'edge3']
# parent of each node on the path to the publisher
PARENT = {
    CORE: PUB,
    'agg0': CORE, 'agg1': CORE,
    'edge0': 'agg0', 'edge1': 'agg0',
    'edge2': 'agg1', 'edge3': 'agg1',
}

# Policy -> planner flags. Wording stays mechanism-descriptive.
#   PerUser-ICN : independent per-user planning (no shared base)
#   GC-noClosure: shared base without full-prefix closure admission
#   GC-Full     : shared base + closure-aware admission (the full method)
POLICY_FLAGS = {
    'PerUser-ICN': dict(shared=False, closure=True),
    'GC-noClosure': dict(shared=True, closure=False),
    'GC-Full': dict(shared=True, closure=True),
}


def write_topo(path):
    """Emit a Mini-NDN topology: hosts only, direct host-to-host links (no
    switch), matching the paper's three-tier tree. Links carry no netem qdisc
    so the harness runs on kernels without sch_netem; the trace-driven simulator
    covers latency/bandwidth accounting."""
    lines = ['[nodes]']
    for n in [PUB, CORE] + AGGS + EDGES:
        lines.append(f'{n}: _')
    lines.append('[links]')
    for child, parent in PARENT.items():
        # A non-empty param is required by the topology parser; an unrecognized
        # key falls through mininet's TCIntf.config without invoking tc/netem,
        # so links come up on kernels lacking sch_netem.
        lines.append(f'{child}:{parent} shaped=no')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def checked(node, command, msg, home=None):
    """Run a command in a node namespace, raising on non-zero exit."""
    pre = f'export HOME={shlex.quote(home)} && ' if home else ''
    pre += 'export NDN_CLIENT_TRANSPORT=tcp://127.0.0.1:6363 && '
    wrapped = ('set -o pipefail; ' + pre + command +
               '; printf "\\n__EXIT__:%s\\n" "$?"')
    out = node.cmd(f'bash -lc {shlex.quote(wrapped)}')
    if '__EXIT__:0' not in out:
        raise RuntimeError(f'{msg}: {out.strip()}')
    return out


def gen_identity(node, name, home):
    checked(node,
            f'mkdir -p {shlex.quote(home)} && '
            f'ndnsec-key-gen {name} | ndnsec-cert-install - && '
            f'ndnsec-set-default {name}',
            f'identity {name}', home=home)


def add_upstream_route(node, prefix, upstream_ip, home):
    """Create a TCP face to the upstream node and route prefix toward it."""
    checked(node,
            f'nfdc face create tcp://{upstream_ip} persistency persistent',
            'face create', home=home)
    checked(node,
            f'nfdc route add {prefix} tcp://{upstream_ip} origin 255 cost 0',
            'route add', home=home)


def link_ip(node, peer):
    """IP of ``peer`` on the interface that directly connects it to ``node``.
    With per-link subnets the default node.IP() is on a different link, so the
    upstream face must target the address on the shared link."""
    conns = node.connectionsTo(peer)      # [(intf_on_node, intf_on_peer), ...]
    if not conns:
        raise RuntimeError(f'no direct link {node.name}<->{peer.name}')
    return conns[0][1].IP()


def stream(proc, name, sink):
    if proc is None or proc.stdout is None:
        return
    for line in proc.stdout:
        sink.append(line)
        info(f'[{name}] {line}')


def assign_users(users, edges):
    """Round-robin user ids onto edge nodes -> {edge: [user_id, ...]}."""
    by_edge = {e: [] for e in edges}
    for i, u in enumerate(sorted(users)):
        by_edge[edges[i % len(edges)]].append(u)
    return by_edge


def run_policy(ndn, policy, version, users, sb, args, workdir):
    """Run one policy end to end on the live network. Returns a metrics dict.
    Each policy uses its own NDN version component so segment names never
    collide across policies -- the Content Store cannot leak a previously
    fetched policy's segments into this one. Sharing still occurs *within* the
    policy, across the concurrent users."""
    flags = POLICY_FLAGS[policy]
    plans, shared_base = plan(users, sb, **flags)
    pstats = plan_stats(plans, shared_base, sb)
    info(f'\n=== policy {policy} | plan {pstats} ===\n')

    # write each user's plan to a JSON file the consumer reads
    plan_files = {}
    for u, items in plans.items():
        pf = os.path.join(workdir, f'plan_{policy}_{u}.json')
        with open(pf, 'w') as fh:
            json.dump([[c, l, t] for (c, l, t) in items], fh)
        plan_files[u] = pf

    pub = ndn.net[PUB]
    pub_home = pub.params['params']['homeDir']
    origin_stats = os.path.join(workdir, f'origin_{policy}.json')

    # (re)start the origin producer for this policy with a fresh counter.
    # getPopen forces cwd=homeDir, so PYTHONPATH carries the module location.
    app_env = {'NDN_CLIENT_TRANSPORT': 'tcp://127.0.0.1:6363',
               'PYTHONPATH': CWD}
    prod_out = []
    prod = getPopen(
        pub,
        ['python3', f'{CWD}/gc_producer.py', '--scene', args.scene,
         '--version', str(version),
         '--n-cells', str(args.n_cells), '--prefix', '/gc',
         '--stats', origin_stats],
        app_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1)
    threading.Thread(target=stream, args=(prod, f'{policy}/origin', prod_out),
                     daemon=True).start()
    # wait for prefix registration
    for _ in range(60):
        if any('origin serving' in ln for ln in prod_out):
            break
        if prod.poll() is not None:
            raise RuntimeError(f'origin exited early ({prod.returncode})')
        time.sleep(0.25)

    # launch all consumers concurrently so PIT aggregation can collapse
    # concurrent identical Interests on the shared base
    by_edge = assign_users(list(users), EDGES)
    procs = []
    for edge, uids in by_edge.items():
        node = ndn.net[edge]
        home = node.params['params']['homeDir']
        for u in uids:
            cstats = os.path.join(workdir, f'client_{policy}_{u}.json')
            out = []
            cp = getPopen(
                node,
                ['python3', f'{CWD}/gc_consumer.py', '--scene', args.scene,
                 '--version', str(version),
                 '--n-cells', str(args.n_cells), '--plan', plan_files[u],
                 '--stats', cstats, '--deadline-ms', str(args.deadline_ms)],
                {'NDN_CLIENT_TRANSPORT': 'tcp://127.0.0.1:6363',
                 'PYTHONPATH': CWD, 'HOME': home},
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            threading.Thread(target=stream, args=(cp, f'{policy}/{u}', out),
                             daemon=True).start()
            procs.append((u, cp, cstats))

    # collect client results
    cli = {'interests': 0, 'received': 0, 'verified': 0, 'late': 0,
           'ttff_ms': [], 'base_done_ms': []}
    for (u, cp, cstats) in procs:
        cp.wait(timeout=args.timeout_s)
        try:
            with open(cstats) as fh:
                d = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            info(f'WARN: no stats for {u}\n')
            continue
        cli['interests'] += d['interests']
        cli['received'] += d['received']
        cli['verified'] += d['verified']
        cli['late'] += d.get('late', 0)
        if d['ttff_ms'] is not None:
            cli['ttff_ms'].append(d['ttff_ms'])
        if d['base_done_ms'] is not None:
            cli['base_done_ms'].append(d['base_done_ms'])

    # stop the origin so it flushes its counters, then read them
    prod.terminate()
    try:
        prod.wait(timeout=5)
    except subprocess.TimeoutExpired:
        prod.kill()
    origin_served = None
    for _ in range(20):
        try:
            with open(origin_stats) as fh:
                origin_served = json.load(fh)['served_segments']
            break
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)

    n = max(1, len(cli['ttff_ms']))
    reuse = (1.0 - origin_served / cli['interests']) if (
        origin_served is not None and cli['interests']) else None
    # distinct planned segments = the theoretical upstream floor. origin_served
    # converging to this (while client_interests is larger) is the "fetch each
    # distinct name once, collapse the repeats" mechanism made measurable.
    distinct_planned = pstats['distinct_segments']
    return {
        'policy': policy,
        'plan': pstats,
        'distinct_planned': distinct_planned,
        'origin_served': origin_served,
        'client_interests': cli['interests'],
        'client_received': cli['received'],
        'client_verified': cli['verified'],
        'late': cli['late'],
        'reuse_ratio': round(reuse, 3) if reuse is not None else None,
        'avg_ttff_ms': round(sum(cli['ttff_ms']) / n, 2) if cli['ttff_ms']
        else None,
    }


def main():
    ap = argparse.ArgumentParser(description='GaussCast Mini-NDN harness')
    ap.add_argument('--scene', default='truck')
    ap.add_argument('--n-cells', type=int, default=8)
    ap.add_argument('--users', type=int, default=8)
    ap.add_argument('--overlap', type=float, default=0.6)
    ap.add_argument('--policies', default='PerUser-ICN,GC-Full')
    ap.add_argument('--deadline-ms', type=float, default=3000.0)
    ap.add_argument('--timeout-s', type=int, default=60)
    ap.add_argument('--out', default=None, help='write results JSON here')
    args = ap.parse_args()

    setLogLevel('info')
    policies = [p.strip() for p in args.policies.split(',') if p.strip()]
    for p in policies:
        if p not in POLICY_FLAGS:
            raise SystemExit(f'unknown policy {p}; choices {list(POLICY_FLAGS)}')

    sb = SceneBlocks(args.scene, n_cells=args.n_cells)
    users = make_users(args.users, sb, overlap=args.overlap, seed=1)

    workdir = tempfile.mkdtemp(prefix='gc_harness_')
    topo = os.path.join(workdir, 'gc_tree.conf')
    write_topo(topo)

    # Minindn runs its own argparse over sys.argv; consume our args first then
    # leave only the program name so its parser sees no stray flags.
    sys.argv = [sys.argv[0]]

    ndn = None
    results = []
    try:
        ndn = Minindn(topoFile=topo)
        ndn.start()
        info('Starting NFD on all nodes...\n')
        AppManager(ndn, ndn.net.hosts, Nfd, useTCP4Transport=True)
        time.sleep(5)

        # identities (digest signer is used for Data, but ndnsec keeps NFD happy)
        for name in [PUB, CORE] + AGGS + EDGES:
            node = ndn.net[name]
            home = node.params['params']['homeDir']
            gen_identity(node, f'/{name}', home)

        # publisher registers /gc locally (the producer does app-level register);
        # every other node routes /gc toward its parent on the path to pub.
        info('Configuring upstream /gc routes toward the publisher...\n')
        for name in [CORE] + AGGS + EDGES:
            node = ndn.net[name]
            home = node.params['params']['homeDir']
            parent = ndn.net[PARENT[name]]
            parent_ip = link_ip(node, parent)
            add_upstream_route(node, '/gc', parent_ip, home)

        for i, policy in enumerate(policies):
            res = run_policy(ndn, policy, i + 1, users, sb, args, workdir)
            results.append(res)
            time.sleep(1)
    finally:
        if ndn is not None:
            ndn.stop()

    info('\n' + '=' * 72 + '\n')
    info('GaussCast Mini-NDN harness results\n')
    info('=' * 72 + '\n')
    hdr = (f"{'policy':14s} {'distinct':>8s} {'origin':>7s} {'cli_int':>8s} "
           f"{'recv':>6s} {'verif':>6s} {'reuse':>6s} {'ttff_ms':>8s}\n")
    info(hdr)
    for r in results:
        info(f"{r['policy']:14s} {r['distinct_planned']:>8d} "
             f"{str(r['origin_served']):>7s} "
             f"{r['client_interests']:>8d} {r['client_received']:>6d} "
             f"{r['client_verified']:>6d} {str(r['reuse_ratio']):>6s} "
             f"{str(r['avg_ttff_ms']):>8s}\n")
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'scene': args.scene, 'users': args.users,
                       'overlap': args.overlap, 'results': results}, fh,
                      indent=2)
        info(f'results -> {args.out}\n')


if __name__ == '__main__':
    main()
