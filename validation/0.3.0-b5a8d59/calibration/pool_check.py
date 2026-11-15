"""Independent pooling of the three calibration slices (my own check of paper-numbers.md)."""
import json, math, os, sys
from collections import defaultdict
W = sys.argv[1]

def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float('nan'),)*2
    p = k/n; d = 1+z*z/n; c = p+z*z/(2*n); h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def gap(r, name):
    g = r.get(name) or {}
    lo = g.get('ci_low', g.get('adjusted_ci_low'))
    hi = g.get('ci_high', g.get('adjusted_ci_high'))
    return dict(flag=bool(g.get('flagged')), adj=g.get('adjusted'), lo=lo, hi=hi,
                status=g.get('status', g.get('adjusted_status', r.get('adjusted_status') if isinstance(r.get('adjusted_status'), str) else None)),
                cov=g.get('coverage'))

def injected(r):
    inj = r.get('injected')
    if not inj: return None
    if 'penalised_minus_plain' in inj: return inj['penalised_minus_plain']
    if 'gap_rows' in inj: return inj['gap_rows']['diff']
    for k in ('domains', 'genres'):
        if k in inj and isinstance(inj[k], dict) and 'gap' in inj[k]: return inj[k]['gap']
    return None

rows = []
for d in ("calib-1-40", "calib-41-80", "calib-81-120"):
    for l in open(os.path.join(W, d, "results.jsonl")):
        r = json.loads(l); r['_slice'] = d; rows.append(r)
rand = [r for r in rows if r['mix'] == 'random']
conf = [r for r in rows if r['mix'] != 'random']
print("rows", len(rows), "random", len(rand), "confounded", len(conf))
seeds = sorted({r['seed'] for r in rand}); print("seeds", len(seeds), seeds[0], seeds[-1])
models = sorted({r['model'] for r in rand}); print(models)

by = defaultdict(list)
for r in rand: by[r['model']].append(r)
print("\n== flag rates over seeds (Wilson 95%)")
for m in models:
    rs = by[m]; n = len(rs)
    line = [m.ljust(30), f"n={n}"]
    for g in ('template_gap', 'source_gap', 'genre_gap'):
        k = sum(gap(r, g)['flag'] for r in rs); lo, hi = wilson(k, n)
        line.append(f"{g.split('_')[0]} {k}/{n} [{lo*100:.1f},{hi*100:.1f}]")
    print('  '.join(line))

print("\n== pooled false-flag (oracle+naive+plain mock), per gap")
for g in ('source_gap', 'genre_gap', 'template_gap'):
    for ms in (['oracle', 'naive', 'mock'], ['mock']):
        rs = [r for m in ms for r in by[m]]; k = sum(gap(r, g)['flag'] for r in rs); lo, hi = wilson(k, len(rs))
        print(f"  {g} {'+'.join(ms)}: {k}/{len(rs)} = {k/len(rs)*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]")

print("\n== coverage: does the adjusted CI contain -injected (penalised mocks); and contain 0 (plain mock)")
for m in models:
    rs = by[m]
    own = 'source_gap' if 'domains' in m else 'genre_gap' if 'genres' in m else None
    if own:
        ins = [(r, injected(r)) for r in rs]
        ok = sum(1 for r, i in ins if i is not None and gap(r, own)['lo'] <= -i <= gap(r, own)['hi'])
        nn = sum(1 for r, i in ins if i is not None)
        lo, hi = wilson(ok, nn)
        mi = sum(i for _, i in ins if i is not None)/nn
        ma = sum(gap(r, own)['adj'] for r in rs)/len(rs)
        hw = sum((gap(r, own)['hi']-gap(r, own)['lo'])/2 for r in rs)/len(rs)
        print(f"  {m.ljust(30)} {own}: covers -injected {ok}/{nn} = {ok/nn*100:.1f}% [{lo*100:.1f},{hi*100:.1f}]; mean injected {mi*100:+.1f}, mean adjusted {ma*100:+.1f}, mean half-width {hw*100:.1f}")
    else:
        for g in ('source_gap', 'genre_gap', 'template_gap'):
            ok = sum(1 for r in rs if gap(r, g)['lo'] <= 0 <= gap(r, g)['hi']); n = len(rs)
            hw = sum((gap(r, g)['hi']-gap(r, g)['lo'])/2 for r in rs)/n
            cv = sum((gap(r, g)['cov'] or 0) for r in rs)/n
            print(f"  {m.ljust(30)} {g}: CI contains 0 {ok}/{n}; mean half-width {hw*100:.1f}; like-for-like share {cv*100:.1f}%")

print("\n== cross flags (penalty on one kind flags the other gap)")
for m in models:
    if 'domains' in m:
        rs = by[m]; k = sum(gap(r, 'genre_gap')['flag'] for r in rs); print(f"  {m}: genre gap flagged {k}/{len(rs)}")
    if 'genres' in m:
        rs = by[m]; k = sum(gap(r, 'source_gap')['flag'] for r in rs); print(f"  {m}: source gap flagged {k}/{len(rs)}")

print("\n== accuracy (originals) and cross-genre consistency, mean over seeds")
for m in models:
    rs = by[m]
    acc = sum(r['accuracy']['rate'] for r in rs)/len(rs)
    cg = [r['cross_genre_consistency']['rate'] for r in rs if r.get('cross_genre_consistency') and r['cross_genre_consistency'].get('rate') is not None]
    print(f"  {m.ljust(30)} acc {acc*100:.1f}%  cross-genre {sum(cg)/len(cg)*100:.1f}% (n={len(cg)})")

print("\n== confounded mix")
cm = defaultdict(lambda: defaultdict(int))
for r in conf:
    for g in ('source_gap', 'genre_gap', 'template_gap'):
        gg = gap(r, g); st = 'flagged' if gg['flag'] else ('n/a' if gg['adj'] is None else 'other')
        cm[(r['model'], g)][st] += 1
for k, v in sorted(cm.items()): print(" ", k, dict(v))
