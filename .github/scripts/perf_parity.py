"""Correctness sweep: splintr against HuggingFace `tokenizers`, vocab by vocab.

HuggingFace is the reference — splintr's ids must match it exactly for every
`tokenizer.json`. A speed comparison is only meaningful once this passes, and a
whole class of loader bugs only shows up on vocabularies whose `tokenizer.json`
is shaped differently from the handful in the test suite.

    perf_parity.py <cache-dir> [repo ...]

With no repos, sweeps a default list spanning the major families. Gated repos
are skipped unless HF_TOKEN is set. Exits non-zero if any vocabulary mismatches.
"""

import json
import os
import sys
import urllib.error
import urllib.request

# One representative per tokenizer family, tracking gigatoken's `families.json`.
DEFAULT_REPOS = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen3-8B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "allenai/OLMo-2-0425-1B",
    "answerdotai/ModernBERT-base",
    "codellama/CodeLlama-7b-hf",
    "deepseek-ai/DeepSeek-V3",
    "microsoft/Phi-4-mini-instruct",
    "microsoft/phi-4",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "openai-community/gpt2",
    "openai/gpt-oss-20b",
    "Xenova/gpt-4",
    "Xenova/gpt-4o",
    "zai-org/GLM-4.5",
]

# Shapes that have historically broken loaders: indentation runs (the `\s+(?!\S)`
# rule), trailing and leading whitespace, mixed scripts, digit runs, and text
# that is entirely whitespace.
CASES = [
    "def f(x):\n    return x",
    "class A:\n\tdef b(self):\n\t\treturn 1",
    "  leading and\ttabs\n\nnewlines  ",
    "trailing spaces   ",
    "Unicode: café naïve 日本語 한국어 العربية 🚀 combining é",
    "那么，线性代数又是如何来解决这些问题的呢？",
    '{"event":"segment","page":12,"text":"A quick brown fox."}',
    "1234567890 42 3.14159 0x1F",
    "     ",
    "",
    "a" * 200,
    "🚀🚀🚀 emoji run 🚀🚀🚀",
]


def fetch(repo, cache_dir):
    path = os.path.join(cache_dir, repo.replace("/", "__") + ".json")
    if os.path.exists(path):
        return path
    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request) as response, open(path, "wb") as out:
            out.write(response.read())
    except urllib.error.HTTPError as e:
        if os.path.exists(path):
            os.remove(path)
        if e.code in (401, 403, 404):
            raise SystemExit("gated")
        raise
    return path


def main():
    cache_dir = sys.argv[1]
    repos = sys.argv[2:] or DEFAULT_REPOS
    os.makedirs(cache_dir, exist_ok=True)

    import splintr
    from tokenizers import Tokenizer as HF

    failures = {}
    skipped = []
    for repo in repos:
        try:
            path = fetch(repo, cache_dir)
        except SystemExit:
            skipped.append(repo)
            print(f"{repo:45} SKIP (gated)")
            continue
        except Exception as e:  # network, 404 — report rather than abort the sweep
            skipped.append(repo)
            print(f"{repo:45} SKIP ({type(e).__name__})")
            continue

        try:
            sp = splintr.from_json(path)
            hf = HF.from_file(path)
        except Exception as e:
            failures[repo] = [("<load>", str(e), "")]
            print(f"{repo:45} LOAD FAILED: {e}")
            continue

        bad = []
        for case in CASES:
            got = sp.encode_raw(case)
            want = hf.encode(case, add_special_tokens=False).ids
            if got != want:
                bad.append((case, got, want))
        if bad:
            failures[repo] = bad
        print(f"{repo:45} {'OK' if not bad else f'{len(bad)}/{len(CASES)} MISMATCH'}")

    if failures:
        print("\n--- mismatches ---")
        for repo, bad in failures.items():
            print(f"\n{repo}")
            for case, got, want in bad[:3]:
                print(f"  case    {case!r}")
                print(f"  splintr {got}")
                print(f"  HF      {want}")

    print(
        f"\n{len(repos) - len(failures) - len(skipped)} ok, "
        f"{len(failures)} mismatched, {len(skipped)} skipped"
    )
    sys.exit(1 if failures else 0)


main()
