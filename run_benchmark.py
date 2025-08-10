import json
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


CASES = ['EMPTY', 'SIMPLE', 'NESTED', 'HUGE',
         'GIANT', 'GIANT_REPEAT', 'WIDE_DUP', 'WIDE_UNQ', 'DEEP']

##################################################################

#CASES = ['EMPTY', 'SIMPLE', 'NESTED', 'HUGE']



def bench_json_dumps(data):
    for obj, count_it in data:
        for _ in count_it:
            json.dumps(obj)




def add_cmdline_args(cmd, args):
    if args.cases:
        cmd.extend(("--cases", args.cases))





##################################################
import json, math
from json.encoder import encode_basestring_ascii as esc  # C fast-path if available

class CachingEncoder(json.JSONEncoder):
    def __init__(self, *, ensure_ascii=True, separators=None, **kw):
        # Keep defaults compatible with stdlib; separators trimmed for speed
        if separators is None:
            separators = (',', ':')
        super().__init__(ensure_ascii=ensure_ascii, separators=separators, **kw)
        self._ensure_ascii = ensure_ascii
        # Per-call caches
        self._memo = {}        # id(container) -> encoded str
        self._in_progress = set()
        self._str_cache = {}   # raw str -> encoded (quoted+escaped) str
        self._int_cache = {i: str(i) for i in range(-1024, 2048)}

    # Public entry: one-shot encode with caches cleared per call
    def encode(self, o):
        self._memo.clear(); self._in_progress.clear(); self._str_cache.clear()
        return self._enc(o)

    def _enc(self, o):
        # Primitives fast-paths
        if o is None:   return "null"
        if o is True:   return "true"
        if o is False:  return "false"
        t = type(o)

        if t is str:
            hit = self._str_cache.get(o)
            if hit is not None:
                return hit
            s = esc(o) if self._ensure_ascii else '"' + o.replace('"', '\\"') + '"'
            self._str_cache[o] = s
            return s

        if t is int:
            if -1024 <= o < 2048:
                return self._int_cache[o]
            return str(o)

        if t is float:
            # match stdlib behavior for specials
            if math.isnan(o):   return "NaN"
            if math.isinf(o):   return "Infinity" if o > 0 else "-Infinity"
            return repr(o)

        if t is dict:
            oid = id(o)
            if oid in self._memo:
                return self._memo[oid]
            if oid in self._in_progress:
                raise ValueError("Circular reference detected")
            self._in_progress.add(oid)

            if not o:
                s = "{}"
            else:
                # No sort: preserve insertion order (fastest)
                parts = []
                ap = parts.append
                for k, v in o.items():
                    if not isinstance(k, str):
                        raise TypeError("keys must be str")
                    ap(self._enc(k) + ":" + self._enc(v))
                s = "{" + ",".join(parts) + "}"

            self._in_progress.remove(oid)
            # Memoize only if likely to be reused: ≥2 items or nested
            if len(o) >= 2:
                self._memo[oid] = s
            return s

        if t in (list, tuple):
            oid = id(o)
            if oid in self._memo:
                return self._memo[oid]
            if oid in self._in_progress:
                raise ValueError("Circular reference detected")
            self._in_progress.add(oid)

            if not o:
                s = "[]"
            else:
                s = "[" + ",".join(self._enc(x) for x in o) + "]"

            self._in_progress.remove(oid)
            if len(o) >= 2:
                self._memo[oid] = s
            return s

        # Fallback for unsupported types
        return self._enc(self.default(o))


def bench_json_dumps_opt(data):
    for obj, count_it in data:
        for _ in count_it:
            json.dumps(obj,cls=CachingEncoder)

##################################################

def main():
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.argparser.add_argument("--cases",
                                  help="Comma separated list of cases. Available cases: %s. By default, run all cases."
                                       % ', '.join(CASES))
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

    data = []
    for case in cases:
        obj, count = globals()[case]
        data.append((obj, range(count)))

    runner.bench_func('json_dumps', bench_json_dumps, data)
    runner.bench_func('json_dumps_cache', bench_json_dumps_opt, data)
if __name__ == '__main__':
    main()
