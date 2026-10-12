# -*- coding: utf-8 -*-
"""Cross-validation of the faithful port against native C++ fixture dumps.

For every case it:
  1. generates a dim x dim workload matrix (seeded numpy),
  2. runs the C++ driver (FAST_DUMP_FILE=<case>.dump) to get ground truth,
  3. replays the pipeline in pure Python (fast_decomp.py),
  4. compares record by record: sds/scaled_factor/psets, srvmat/approx/intr/
     bal/ds, t/bound and the RESULT line numbers.

Integer structures must match exactly; doubles are compared bitwise first and
fall back to a 1e-12 relative tolerance (mismatches reported).

"""
import os
import re
import subprocess
import sys
import time
import math
import unittest
from pathlib import Path
import numpy as np

from simai_a2a.schedulers import fast_port as fd

DRIVER = os.environ.get('FAST_XCHECK_DRIVER')
WORK = os.environ.get('FAST_XCHECK_WORK', '/tmp/simai-fast-xcheck')
INTER = ['ETHER100', 'ETHER400', 'INFB']
INTRA = ['DGX2', 'MI300X', 'NDV2', 'FAST', 'H100', 'B100']

# ---------------------------------------------------------------------------
# workload generation
# ---------------------------------------------------------------------------

def gen_uniform(rng, dim):
    return rng.randint(1, 10001, size=(dim, dim))


def gen_skew(rng, dim):
    """Per-source-row concentration: each row keeps ~70% of its mass in one
    randomly chosen destination column (of a random dst server), rest small."""
    m = rng.randint(1, 1001, size=(dim, dim))
    for i in range(dim):
        jstar = rng.randint(0, dim - 1)
        m[i, jstar] += rng.randint(1, 9001)
    return m


def gen_zipf(rng, dim):
    """Heavy-tailed values (1..10000)."""
    v = rng.zipf(1.7, size=(dim, dim))
    return np.minimum(v, 10000)


DISTS = {'uniform': gen_uniform, 'skew': gen_skew, 'zipf': gen_zipf}

# ---------------------------------------------------------------------------
# C++ side
# ---------------------------------------------------------------------------

def run_cpp(case):
    if not DRIVER:
        raise RuntimeError('set FAST_XCHECK_DRIVER to refresh native fixtures')
    os.makedirs(WORK, exist_ok=True)
    matfile = os.path.join(WORK, case['name'] + '.mat')
    dumpfile = os.path.join(WORK, case['name'] + '.dump')
    rng = np.random.RandomState(case['seed'])
    dim = case['server_n'] * case['gpu_n'] if case['mode'] != 'core' \
        else case['dim']
    mat = DISTS[case['dist']](rng, dim).astype(np.int64)
    for i in range(dim):
        mat[i, i] = 0
    np.savetxt(matfile, mat, fmt='%u', delimiter=' ')
    argv = [DRIVER, case['mode']]
    if case['mode'] == 'core':
        argv += [str(case['dim']), str(case['gpu_n']), str(case['inter']),
                 str(case['intra']), str(int(case['scaling'])),
                 repr(case['MBpu']), matfile]
    else:
        argv += [str(case['server_n']), str(case['gpu_n']),
                 str(case['inter']), str(case['intra']),
                 str(int(case['scaling'])), repr(case['MBpu'])]
        if case['mode'] == 'p3':
            argv.insert(8, str(case['topo']))
        argv.append(matfile)
    env = dict(os.environ)
    env['FAST_DUMP_FILE'] = dumpfile
    t0 = time.time()
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError('C++ driver failed (%d): %s'
                           % (p.returncode, p.stderr.decode()))
    with open(dumpfile) as f:
        records = parse_dump(f.read())
    result = {}
    for tok in p.stdout.decode().split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            result[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
    return mat, records, result, wall


def parse_dump(text):
    """Returns a list of records:
      ('sds', dim, dim, [vals]) / ('scaled_factor', v) / ('pset', f) /
      ('pmap', pid, [mp]) / ('npset', n) / ('srvmat', d, d, vals) /
      ('approx', idx, [flags]) / ('intr', idx, g, g, vals) /
      ('bal', di, dj, g, g, vals) / ('ds', pid, lid, g, g, vals) /
      ('t', float) / ('bound', float)"""
    recs = []
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        tag = tok[0]
        rest = tok[1:]
        if tag in ('t', 'bound'):
            recs.append((tag, float(rest[0])))
        elif tag in ('scaled_factor', 'npset'):
            recs.append((tag, int(rest[0])))
        elif tag in ('pset',):
            recs.append((tag, int(rest[0])))
        elif tag in ('sds', 'srvmat'):
            d = int(rest[0])
            recs.append((tag, d, d, [int(x) for x in rest[2:]]))
        elif tag == 'pmap':
            recs.append((tag, int(rest[0]), [int(x) for x in rest[1:]]))
        elif tag == 'approx':
            recs.append((tag, int(rest[0]), [int(x) for x in rest[1:]]))
        elif tag in ('intr', 'bal', 'ds'):
            # dump_matrix prints: tag <idx tokens> <r> <c> <r*c values>
            pref = 2 if tag in ('bal', 'ds') else 1
            idx = tuple(int(x) for x in rest[:pref])
            r = int(rest[pref])
            c = int(rest[pref + 1])
            vals = [int(x) for x in rest[pref + 2:]]
            if r * c != len(vals):
                raise ValueError('bad %s record length' % tag)
            recs.append((tag, idx, r, c, vals))
        else:
            raise ValueError('unknown dump record: %r' % line)
    return recs


def get_records(recs):
    """Group dump records into named structures."""
    sds = None
    scaled_factor = None
    npset = None
    psets = []          # list of (freq, [mp row0..])
    srvmat = None
    approx = {}         # local idx -> flags list
    intr = {}           # local idx -> matrix
    bal = {}            # (di, dj) -> matrix
    ds = {}             # (pid, lid) -> matrix
    t = bound = None
    for r in recs:
        tag = r[0]
        if tag == 'sds':
            sds = np.array(r[3], dtype=np.int64).reshape(r[1], r[2])
        elif tag == 'scaled_factor':
            scaled_factor = r[1]
        elif tag == 'pset':
            psets.append([r[1], None])
        elif tag == 'pmap':
            psets[r[1]][1] = r[2]
        elif tag == 'npset':
            npset = r[1]
        elif tag == 'srvmat':
            srvmat = np.array(r[3], dtype=np.int64).reshape(r[1], r[2])
        elif tag == 'approx':
            approx[r[1]] = r[2]
        elif tag in ('intr', 'bal', 'ds'):
            mat = np.array(r[4], dtype=np.int64).reshape(r[2], r[3])
            if tag == 'intr':
                intr[r[1][0]] = mat
            elif tag == 'bal':
                bal[r[1]] = mat
            else:
                ds[r[1]] = mat
        elif tag == 't':
            t = r[1]
        elif tag == 'bound':
            bound = r[1]
    return {'sds': sds, 'scaled_factor': scaled_factor, 'npset': npset,
            'psets': psets, 'srvmat': srvmat, 'approx': approx,
            'intr': intr, 'bal': bal, 'ds': ds, 't': t, 'bound': bound}


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

def run_python(case, mat):
    s_n = case['server_n'] if case['mode'] != 'core' else case['dim']
    g_n = case['gpu_n'] if case['mode'] != 'core' else case['gpu_n']
    inter = INTER[case['inter']]
    intra = INTRA[case['intra']]
    MBpu = case['MBpu']

    if case['mode'] == 'core':
        app = fd.FastAll2All(mat, g_n, inter, intra, MBpu)
        if case['scaling']:
            app.to_scaled_matrix()
        app.to_scaled_doubly_stochastic_matrix()
        sds_snap = app.SDS_mat.copy()   # C++ dumps 'sds' right after conversion
        app.decompose()
        return {'app': app, 'sds': sds_snap,
                'psets': [(p['freq'], list(p['mp']))
                          for p in app.p_sets]}, mat

    # full pipeline (p2/p3), mirroring driver.cpp
    dim = s_n * g_n
    locals_ = []
    for s in range(s_n):
        locals_.append(fd.LocalScheduler(mat[s * g_n:(s + 1) * g_n, :], g_n,
                                         s_n, s,
                                         fd.get_intra_link_info(intra)))
    gs = fd.GlobalScheduler(s_n, g_n, locals_, inter, intra,
                            bool(case['scaling']), MBpu)
    if case['mode'] == 'p2':
        res = gs.pipeline2(MBpu)
    else:
        topo = {1: 'full_mesh', 2: '2ring', 3: 'hybrid'}[case['topo']]
        res = gs.pipeline3(MBpu, topo)
    baseline = fd.spread_out_baseline(
        mat, s_n, g_n, fd.get_inter_link_info(inter),
        fd.get_intra_link_info(intra), MBpu)
    info = {'gs': gs, 'locals': locals_, 'res': res, 'baseline': baseline,
            't': res['t'], 'ratio': res['ratio'],
            'speedup': baseline / res['t'],
            'psets': [(p['freq'], list(p['mp'])) for p in res['p_sets']],
            'srvmat': gs.mat,
            'if_approx': list(gs.if_approx)}
    intr = {i: locals_[i].intrinsic_all2all.copy()
            for i in range(len(locals_))}
    bal = {}
    for lid in range(len(locals_)):
        for s in range(s_n):
            if s != locals_[lid].server_id:
                bal[(lid, s)] = gs.last_bs[(lid, s)].copy()
    ds = {}
    for pid in range(len(res['p_sets'])):
        for lid in range(len(locals_)):
            ds[(pid, lid)] = gs.last_ds[(pid, lid)].copy()
    return dict(info, intr=intr, bal=bal, ds=ds), mat


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def cmp_int(name, py, cx):
    ok = py == cx
    return ok, '' if ok else '%s: python=%s c++=%s' % (name, py, cx)


def cmp_float(name, py, cx):
    if py == cx:
        return True, ''
    if math.isclose(py, cx, rel_tol=1e-12, abs_tol=1e-12):
        return True, '  (bit-diff %r vs %r)' % (py, cx)
    return False, '%s: python=%.17g c++=%.17g diff=%g' % (
        name, py, cx, abs(py - cx))


def cmp_mat(name, py, cx):
    if py.shape != cx.shape:
        return False, '%s: shape %s vs %s' % (name, py.shape, cx.shape)
    if np.array_equal(py, cx):
        return True, ''
    idx = np.argwhere(py != cx)
    return False, '%s: %d/%d entries differ, first at %s py=%s c++=%s' % (
        name, len(idx), py.size, idx[0].tolist(), py[tuple(idx[0])],
        cx[tuple(idx[0])])


def check_core(case, records, py, py_mat):
    problems = []
    cx = get_records(records)
    # sds (snapshot taken right after the sdsm conversion, as in the C++ dump)
    ok, msg = cmp_mat('sds', py['sds'], cx['sds'])
    if not ok:
        problems.append(msg)
    # scaling factor
    if case['scaling']:
        pyf = py['app'].scaling_factor if py['app'].is_scaled else None
        ok, msg = cmp_int('scaled_factor', pyf, cx['scaled_factor'])
        if not ok:
            problems.append(msg)
    # pset list
    if len(py['psets']) != len(cx['psets']):
        problems.append('npset: python=%d c++=%d'
                        % (len(py['psets']), len(cx['psets'])))
    for i, (p, c) in enumerate(zip(py['psets'], cx['psets'])):
        if p[0] != c[0]:
            problems.append('pset[%d].freq: python=%d c++=%d' % (i, p[0], c[0]))
        if p[1] != c[1]:
            problems.append('pset[%d].mp: python=%s c++=%s' % (i, p[1], c[1]))
    return problems


def check_pipeline(case, records, py):
    problems = []
    cx = get_records(records)
    ok, msg = cmp_mat('srvmat', py['srvmat'], cx['srvmat'])
    if not ok:
        problems.append(msg)
    for lid, flags in cx['approx'].items():
        pyf = py['if_approx'][lid * case['server_n']:
                              (lid + 1) * case['server_n']]
        if pyf != flags:
            problems.append('approx[%d]: python=%s c++=%s'
                            % (lid, pyf, flags))
    for lid, m in cx['intr'].items():
        ok, msg = cmp_mat('intr[%d]' % lid, py['intr'][lid], m)
        if not ok:
            problems.append(msg)
    for (di, dj), m in cx['bal'].items():
        ok, msg = cmp_mat('bal[%d,%d]' % (di, dj), py['bal'][(di, dj)], m)
        if not ok:
            problems.append(msg)
    for (pid, lid), m in cx['ds'].items():
        ok, msg = cmp_mat('ds[%d,%d]' % (pid, lid), py['ds'][(pid, lid)], m)
        if not ok:
            problems.append(msg)
    if len(py['psets']) != len(cx['psets']):
        problems.append('npset: python=%d c++=%d'
                        % (len(py['psets']), len(cx['psets'])))
    for i, (p, c) in enumerate(zip(py['psets'], cx['psets'])):
        if p[0] != c[0]:
            problems.append('pset[%d].freq: python=%d c++=%d' % (i, p[0], c[0]))
        if p[1] != c[1]:
            problems.append('pset[%d].mp: python=%s c++=%s' % (i, p[1], c[1]))
    # t / bound
    for name, a, b in (('t', py['res']['t'], cx['t']),
                       ('bound', py['res']['bound'], cx['bound'])):
        ok, msg = cmp_float(name, a, b)
        if not ok:
            problems.append(msg)
    # scaling factor when p2 scaling on
    if case['mode'] == 'p2' and case['scaling']:
        app = py['gs'].last_app
        pyf = app.scaling_factor if app.is_scaled else None
        ok, msg = cmp_int('scaled_factor', pyf, cx['scaled_factor'])
        if not ok:
            problems.append(msg)
    return problems


# ---------------------------------------------------------------------------
# case table
# ---------------------------------------------------------------------------

CASES = [
    # core-mode: decomposition on the raw matrix
    dict(name='c01-core8-uniform', mode='core', dim=8, gpu_n=2, inter=1,
         intra=4, scaling=0, MBpu=1.0, dist='uniform', seed=101),
    dict(name='c02-core16-skew', mode='core', dim=16, gpu_n=4, inter=2,
         intra=3, scaling=0, MBpu=1.0, dist='skew', seed=202),
    dict(name='c03-core16-scaling', mode='core', dim=16, gpu_n=4, inter=1,
         intra=0, scaling=1, MBpu=1e-5, dist='uniform', seed=303),
    # pipeline2 (intra cost = spread_out)
    dict(name='c04-p2-2x4', mode='p2', server_n=2, gpu_n=4, inter=1,
         intra=4, scaling=0, MBpu=1.0, dist='uniform', seed=404),
    dict(name='c05-p2-4x4-skew', mode='p2', server_n=4, gpu_n=4, inter=0,
         intra=3, scaling=0, MBpu=1.0, dist='skew', seed=505),
    dict(name='c06-p2-4x4-scaling', mode='p2', server_n=4, gpu_n=4, inter=2,
         intra=0, scaling=1, MBpu=1e-5, dist='uniform', seed=606),
    dict(name='c07-p2-8x4-zipf', mode='p2', server_n=8, gpu_n=4, inter=1,
         intra=5, scaling=0, MBpu=1.0, dist='zipf', seed=707),
    # pipeline3 (topology intra cost functions, 8-GPU servers)
    dict(name='c08-p3-2x8-mesh', mode='p3', server_n=2, gpu_n=8, inter=1,
         intra=4, scaling=0, MBpu=1.0, topo=1, dist='uniform', seed=808),
    dict(name='c09-p3-4x8-2ring', mode='p3', server_n=4, gpu_n=8, inter=2,
         intra=3, scaling=0, MBpu=1.0, topo=2, dist='skew', seed=909),
    dict(name='c10-p3-2x8-hybrid', mode='p3', server_n=2, gpu_n=8, inter=0,
         intra=5, scaling=0, MBpu=1.0, topo=3, dist='zipf', seed=111),
    # heavy decomposition (32x32 dense)
    dict(name='c11-core32-uniform', mode='core', dim=32, gpu_n=8, inter=2,
         intra=1, scaling=0, MBpu=1.0, dist='uniform', seed=222),
    # largest gpu-level matrix (64x64), 8 servers x 8 gpus
    dict(name='c12-p2-8x8-skew', mode='p2', server_n=8, gpu_n=8, inter=1,
         intra=4, scaling=0, MBpu=1.0, dist='skew', seed=333),
    dict(name='c13-p3-4x8-hybrid', mode='p3', server_n=4, gpu_n=8, inter=2,
         intra=3, scaling=1, MBpu=1e-4, topo=3, dist='uniform', seed=444),
]

HEADER = ('%-22s %-6s %-8s %8s %8s %8s %-9s %-9s %s'
          % ('case', 'mode', 'npset', 'cpp-t', 'py-t', 't-diff', 'baseline',
             'speedup', 'status'))
SEP = '-' * 120


FIXTURES = Path(__file__).parent / 'fixtures' / 'fast_cpp'


class FastCppFixtureTests(unittest.TestCase):
    """Each generated method names one native FAST case in verbose output."""

    def test_refresh_tool_requires_explicit_paths(self):
        tool = Path(__file__).parents[1] / 'tools' / 'refresh_fast_crosscheck.py'
        completed = subprocess.run(
            [sys.executable, str(tool), '--help'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--cpp-driver', completed.stdout)
        self.assertIn('--work-dir', completed.stdout)


def _fixture_test(case):
    def test(self):
        dim = case.get('dim', case.get('server_n', 0) * case['gpu_n'])
        mat = np.loadtxt(FIXTURES / (case['name'] + '.mat'), dtype=np.int64)
        self.assertEqual(mat.shape, (dim, dim))
        records = parse_dump(
            (FIXTURES / (case['name'] + '.dump')).read_text(encoding='utf-8')
        )
        py, _ = run_python(case, mat)
        if case['mode'] == 'core':
            problems = check_core(case, records, py, mat)
        else:
            problems = check_pipeline(case, records, py)
        self.assertEqual(problems, [])

    return test


for _case in CASES:
    setattr(
        FastCppFixtureTests,
        'test_' + _case['name'].replace('-', '_'),
        _fixture_test(_case),
    )


def main():
    args = sys.argv[1:]
    if args:
        pats = [re.compile(a) for a in args]
        cases = [c for c in CASES
                 if any(p.search(c['name']) for p in pats)]
    else:
        cases = CASES
    print(SEP)
    print(HEADER)
    print(SEP)
    fail = 0
    for case in cases:
        t0 = time.time()
        try:
            mat, records, cppres, wall_cpp = run_cpp(case)
            py, _ = run_python(case, mat)
            if case['mode'] == 'core':
                problems = check_core(case, records, py, mat)
            else:
                problems = check_pipeline(case, records, py)
            # RESULT-line numbers
            if case['mode'] != 'core':
                for name in ('t', 'ratio', 'baseline', 'speedup'):
                    ok, msg = cmp_float(name, py[name], cppres[name])
                    if not ok:
                        problems.append(msg)
            npset = len(py['psets'])
            status = 'OK' if not problems else 'FAIL'
            tval = cppres.get('t')
            pyval = py['res']['t'] if case['mode'] != 'core' else None
            tdiff = ''
            if tval is not None:
                tdiff = '%.3g' % (pyval - tval) if pyval != tval else '0'
            print('%-22s %-6s %-8d %8.3g %8.3g %8s %9.3g %9.5f %s  [%.1fs]'
                  % (case['name'], case['mode'], npset, tval or 0,
                     pyval or 0, tdiff,
                     cppres.get('baseline', 0), cppres.get('speedup', 0),
                     status, time.time() - t0 + wall_cpp))
            if problems:
                fail += 1
                for p in problems:
                    print('    - ' + p)
        except Exception as e:  # noqa
            import traceback
            print('%-22s ERROR: %s' % (case['name'], e))
            traceback.print_exc()
            fail += 1
    print(SEP)
    print('summary: %d/%d cases passed' % (len(cases) - fail, len(cases)))
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
