from __future__ import annotations

import json, math, os

import sys

import pyperf


EMPTY = ({}, 2000)
SIMPLE_DATA = {'key1': 0, 'key2': True, 'key3': 'value', 'key4': 'foo',
               'key5': 'string'}
SIMPLE = (SIMPLE_DATA, 1000)
NESTED_DATA = {'key1': 0, 'key2': SIMPLE[0], 'key3': 'value', 'key4': SIMPLE[0],
               'key5': SIMPLE[0], 'key': '\u0105\u0107\u017c'}
NESTED = (NESTED_DATA, 1000)
HUGE = ([NESTED[0]] * 1000, 1)

###################################################################
import copy

def deep_nest(base, depth=6, fanout=3):
    obj = base
    for d in range(depth):
        obj = {"layer": d, "payload": obj, "arr": [obj] * fanout}
    return obj

# 10x bigger than HUGE: same nested dict repeated 10k times (great for caching)
GIANT = ([NESTED[0]] * 10_000, 1)

# Same size as GIANT but run 10 iterations → better for perf.sample density
GIANT_REPEAT = ([NESTED[0]] * 1_000, 10)

# Super wide dict: 5k keys, all pointing to the SAME SIMPLE dict (max cache hits)
WIDE_DUP = ({f"k{i}": SIMPLE[0] for i in range(5_000)}, 1)

# Wide dict with UNIQUE copies (no identity reuse → cache can’t help containers)
WIDE_UNQ = ({f"k{i}": copy.deepcopy(SIMPLE[0]) for i in range(2_000)}, 1)

# Deeply nested structure to stress recursion + string escaping
DEEP = (deep_nest(SIMPLE[0], depth=10, fanout=2), 50)


CASES = ['EMPTY', 'SIMPLE', 'NESTED', 'HUGE','GIANT', 'GIANT_REPEAT', 'WIDE_DUP', 'WIDE_UNQ', 'DEEP']

def bench_json_dumps(data):
    for obj, count_it in data:
        for _ in count_it:
            json.dumps(obj)


def add_cmdline_args(cmd, args):
    if args.cases:
        cmd.extend(("--cases", args.cases))



# parallel_caching_encoder.py
# Linux-friendly: coarse-grained parallelism with ProcessPoolExecutor + caching.

import json, math, os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from json.encoder import encode_basestring_ascii as _esc_ascii, encode_basestring as _esc_utf8

# ---------- worker (must be top-level & importable) ----------

def _encode_values_batch(values, ensure_ascii: bool) -> list[str]:
    dumps = json.dumps
    if ensure_ascii:
        return [dumps(v, ensure_ascii=True) for v in values]
    else:
        return [dumps(v, ensure_ascii=False) for v in values]

# ---------- base caching encoder (serial) ----------

class CachingEncoder(json.JSONEncoder):
    def __init__(self, *, ensure_ascii=True, separators=None, **kw):
        super().__init__(ensure_ascii=ensure_ascii, separators=separators, **kw)
        self._memo = {}
        self._in_progress = set()
        self._str_cache = {}                  # (string, ensure_ascii) -> quoted/escaped
        self._int_cache = {i: str(i) for i in range(-1024, 2048)}
        self._depth = 0

    def encode(self, o):
        self._memo.clear(); self._in_progress.clear(); self._str_cache.clear(); self._depth = 0
        return self._enc(o)

    def _esc_str(self, s: str) -> str:
        key = (s, self.ensure_ascii)
        v = self._str_cache.get(key)
        if v is not None:
            return v
        v = _esc_ascii(s) if self.ensure_ascii else _esc_utf8(s)  # C-accelerated in CPython
        self._str_cache[key] = v
        return v

    def _enc(self, o):
        if o is None:  return "null"
        if o is True:  return "true"
        if o is False: return "false"
        t = type(o)

        if t is str:
            return self._esc_str(o)

        if t is int:
            return self._int_cache[o] if -1024 <= o < 2048 else str(o)

        if t is float:
            if math.isnan(o): return "NaN"
            if math.isinf(o): return "Infinity" if o > 0 else "-Infinity"
            return repr(o)

        if t is dict:
            oid = id(o)
            if oid in self._memo: return self._memo[oid]
            if oid in self._in_progress: raise ValueError("Circular reference detected")
            self._in_progress.add(oid); self._depth += 1
            try:
                if not o:
                    s = "{}"
                else:
                    parts = []
                    ap = parts.append
                    it = iter(o.items())
                    k, v = next(it)
                    if not isinstance(k, str): raise TypeError("keys must be str")
                    ap(self._esc_str(k)); ap(self.key_separator); ap(self._enc(v))
                    for k, v in it:
                        if not isinstance(k, str): raise TypeError("keys must be str")
                        ap(self.item_separator); ap(self._esc_str(k)); ap(self.key_separator); ap(self._enc(v))
                    s = "".join(("{", *parts, "}"))
            finally:
                self._depth -= 1; self._in_progress.remove(oid)
            if len(o) >= 2: self._memo[oid] = s
            return s

        if t in (list, tuple):
            oid = id(o)
            if oid in self._memo: return self._memo[oid]
            if oid in self._in_progress: raise ValueError("Circular reference detected")
            self._in_progress.add(oid); self._depth += 1
            try:
                if not o:
                    s = "[]"
                else:
                    parts = []
                    ap = parts.append
                    it = iter(o)
                    x = next(it); ap(self._enc(x))
                    for x in it:
                        ap(self.item_separator); ap(self._enc(x))
                    s = "".join(("[", *parts, "]"))
            finally:
                self._depth -= 1; self._in_progress.remove(oid)
            if len(o) >= 2: self._memo[oid] = s
            return s

        return self._enc(self.default(o))

# ---------- parallel variant ----------

class ParallelCachingEncoder(CachingEncoder):
    """
    Parallelizes *values* of large top-level dicts/lists using a ProcessPoolExecutor.
    Only triggers when:
      - recursion depth == parallel_max_depth (default 0, i.e., top level)
      - container length >= parallel_threshold
      - executor is configured via the context manager.
    """
    executor: Optional[ProcessPoolExecutor] = None
    parallel_threshold: int = 2000
    parallel_max_depth: int = 0
    batch_size: int = 256  # values per task

    @classmethod
    def use_pool(cls, max_workers: Optional[int] = None):
        class _PoolCtx:
            def __enter__(self_inner):
                cls.executor = ProcessPoolExecutor(max_workers=max_workers or os.cpu_count())
                return cls.executor
            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    cls.executor.shutdown()
                finally:
                    cls.executor = None
        return _PoolCtx()

    def _enc(self, o):
        t = type(o)

        # ---- Parallel dict ----
        if t is dict and self.executor and self._depth == self.parallel_max_depth and len(o) >= self.parallel_threshold:
            oid = id(o)
            if oid in self._memo: return self._memo[oid]
            if oid in self._in_progress: raise ValueError("Circular reference detected")
            self._in_progress.add(oid)
            try:
                if not o:
                    s = "{}"
                else:
                    items = list(o.items())

                    # validate keys first (clear error path)
                    for k, _ in items:
                        if not isinstance(k, str):
                            raise TypeError("keys must be str")

                    # encode keys locally (keeps string cache hot)
                    key_strs = [self._esc_str(k) for k, _ in items]

                    # submit value batches
                    bs = max(1, self.batch_size)
                    futures = []
                    for i in range(0, len(items), bs):
                        chunk_vals = [v for _, v in items[i:i+bs]]
                        futures.append(self.executor.submit(_encode_values_batch, chunk_vals, self.ensure_ascii))
                    val_chunks = [f.result() for f in futures]

                    # stitch JSON preserving order
                    parts = []
                    ap = parts.append
                    # first pair
                    ap(key_strs[0]); ap(self.key_separator); ap(val_chunks[0][0])
                    idx = 1
                    for ci, chunk in enumerate(val_chunks):
                        start = 1 if ci == 0 else 0
                        for j in range(start, len(chunk)):
                            ap(self.item_separator); ap(key_strs[idx]); ap(self.key_separator); ap(chunk[j])
                            idx += 1
                    s = "".join(("{", *parts, "}"))
            finally:
                self._in_progress.remove(oid)
            if len(o) >= 2: self._memo[oid] = s
            return s

        # ---- Parallel list/tuple ----
        if t in (list, tuple) and self.executor and self._depth == self.parallel_max_depth and len(o) >= self.parallel_threshold:
            oid = id(o)
            if oid in self._memo: return self._memo[oid]
            if oid in self._in_progress: raise ValueError("Circular reference detected")
            self._in_progress.add(oid)

def bench_json_dumps_cache(data):
    for obj, count_it in data:
        for _ in count_it:
            json.dumps(obj, cls=CachingEncoder)

def bench_json_dumps_cache_par(data):
    for obj, count_it in data:
        for _ in count_it:
            with ParallelCachingEncoder.use_pool(max_workers=None):  # defaults to os.cpu_count()
                ParallelCachingEncoder.parallel_threshold = 2000  # tune
                ParallelCachingEncoder.batch_size = 256  # tune
                ParallelCachingEncoder.parallel_max_depth = 0  # top level only
                s2 = json.dumps(obj, cls=ParallelCachingEncoder)


def bench_json_dumps_cache_par(data):
    for obj, count_it in data:
        # create the pool once per case
        with ParallelCachingEncoder.use_pool(max_workers=None):  # defaults to os.cpu_count()
            ParallelCachingEncoder.parallel_threshold = 2000  # tune if needed
            ParallelCachingEncoder.batch_size = 256           # tune if needed
            ParallelCachingEncoder.parallel_max_depth = 0     # top level only
            for _ in count_it:
                json.dumps(obj, cls=ParallelCachingEncoder)


def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument(
        "--cases",
        help="Comma separated list of cases. Available cases: %s. By default, run all cases."
             % ', '.join(CASES)
    )
    runner.metadata['description'] = "Benchmark json.dumps()"

    args = runner.parse_args()
    if args.cases:
        cases = []
        for case in args.cases.split(','):
            case = case.strip()
            if case:
                cases.append(case)
        if not cases:
            print("ERROR: empty list of cases")
            sys.exit(1)
    else:
        cases = CASES

    # register three benchmarks per case, each with its own (obj, range(count))
    for case in cases:
        obj, count = globals()[case]
        data = [(obj, range(count))]
        runner.bench_func(f'json_dumps[{case}]',               bench_json_dumps,            data)
        runner.bench_func(f'json_dumps_cache[{case}]',         bench_json_dumps_cache,      data)
        runner.bench_func(f'json_dumps_cache_par[{case}]',     bench_json_dumps_cache_par,  data)

if __name__ == '__main__':
    main()
