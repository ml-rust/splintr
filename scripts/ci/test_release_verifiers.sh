#!/usr/bin/env bash
# Self-test for the release-path verifiers.
#
#   bash scripts/ci/test_release_verifiers.sh
#
# The scripts under test run exactly once per release, on the one path where a
# mistake is public and irreversible. Without this they are only ever exercised
# by the release they are supposed to protect. Run by the lint job in
# .github/workflows/test.yml, so a broken verifier fails a PR instead of a
# publish.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifacts_sh="$repo_root/scripts/ci/verify_release_artifacts.sh"
prepare_sh="$repo_root/scripts/ci/verify_prepare_run.sh"
changelog_sh="$repo_root/scripts/ci/changelog_section.sh"
smoke_sh="$repo_root/scripts/ci/smoke_test_wheel.sh"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

failures=0

# Both helpers report the case name rather than a line number: a failure here is
# read by whoever is mid-release, not by whoever wrote the test.
ok() { # ok <case> <cmd...>  — the command must succeed
  local case_name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '  ok    %s\n' "$case_name"
  else
    printf '  FAIL  %s (expected success)\n' "$case_name"
    failures=$((failures + 1))
  fi
}

rejects() { # rejects <case> <cmd...>  — the command must fail
  local case_name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '  FAIL  %s (expected rejection)\n' "$case_name"
    failures=$((failures + 1))
  else
    printf '  ok    %s\n' "$case_name"
  fi
}

# ── verify_release_artifacts.sh ─────────────────────────────────────────────
VERSION="9.9.9"
PREFIX="splintr_rs-${VERSION}"

# Tags are the ones the release actually produces, compressed manylinux pair
# included: a fixture that names a simpler tag than reality would let a pattern
# pass here and fail on the wheel it was written for.
populate() { # populate <dir> [version] — a complete, valid distribution set
  local dir="$1"
  local pfx="splintr_rs-${2:-$VERSION}"
  mkdir -p "$dir"
  : > "$dir/${pfx}.tar.gz"
  : > "$dir/${pfx}-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
  : > "$dir/${pfx}-cp310-abi3-musllinux_1_2_x86_64.whl"
  : > "$dir/${pfx}-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
  : > "$dir/${pfx}-cp310-abi3-musllinux_1_2_aarch64.whl"
  : > "$dir/${pfx}-cp310-abi3-macosx_10_12_x86_64.whl"
  : > "$dir/${pfx}-cp310-abi3-macosx_11_0_arm64.whl"
  : > "$dir/${pfx}-cp310-abi3-win_amd64.whl"
  : > "$dir/${pfx}-cp310-abi3-win_arm64.whl"
}

echo "verify_release_artifacts.sh"

complete="$work/complete"
populate "$complete"
ok "complete set" bash "$artifacts_sh" "$complete" "$VERSION"

prerelease="$work/prerelease"
populate "$prerelease" "9.9.9b1"
ok "PEP 440 prerelease version" bash "$artifacts_sh" "$prerelease" "9.9.9b1"

rejects "missing directory" bash "$artifacts_sh" "$work/absent" "$VERSION"

empty="$work/empty"
mkdir -p "$empty"
rejects "empty directory" bash "$artifacts_sh" "$empty" "$VERSION"

no_sdist="$work/no-sdist"
populate "$no_sdist"
rm "$no_sdist/${PREFIX}.tar.gz"
rejects "sdist absent" bash "$artifacts_sh" "$no_sdist" "$VERSION"

# One case per platform family, each removing only that family's wheel, so a
# pattern that stopped matching shows up as its own failure rather than being
# masked by a neighbour.
#
# The two Linux glibc/musl pairs are the point of the exercise: `manylinux` and
# `musllinux` wheels sit side by side in the same directory and differ by one
# substring, so a `*linux*` pattern accepts either as proof of both. Each of
# these four cases leaves its sibling in place — the set is complete except for
# the one family named, and it must still be rejected.
drop() { # drop <case> <suffix> — the set minus one wheel must be rejected
  local case_name="$1" suffix="$2" dir="$work/drop-${1// /-}"
  populate "$dir"
  rm "$dir/${PREFIX}-cp310-abi3-${suffix}.whl"
  rejects "$case_name" bash "$artifacts_sh" "$dir" "$VERSION"
}

drop "glibc x86_64 wheel absent, musl wheel present" \
  "manylinux_2_17_x86_64.manylinux2014_x86_64"
drop "musl x86_64 wheel absent, glibc wheel present" "musllinux_1_2_x86_64"
drop "glibc aarch64 wheel absent, musl wheel present" \
  "manylinux_2_17_aarch64.manylinux2014_aarch64"
drop "musl aarch64 wheel absent, glibc wheel present" "musllinux_1_2_aarch64"
drop "macOS x86_64 wheel absent" "macosx_10_12_x86_64"
drop "macOS arm64 wheel absent" "macosx_11_0_arm64"
drop "Windows x86_64 wheel absent" "win_amd64"
drop "Windows arm64 wheel absent" "win_arm64"

stale="$work/stale"
populate "$stale"
: > "$stale/splintr_rs-9.9.8-cp310-abi3-musllinux_1_2_aarch64.whl"
rejects "wheel from another version" bash "$artifacts_sh" "$stale" "$VERSION"

intruder="$work/intruder"
populate "$intruder"
: > "$intruder/notes.txt"
rejects "unexpected attachment" bash "$artifacts_sh" "$intruder" "$VERSION"

link="$work/link"
populate "$link"
rm "$link/${PREFIX}-cp310-abi3-win_amd64.whl"
ln -s /nonexistent "$link/${PREFIX}-cp310-abi3-win_amd64.whl"
rejects "symlinked distribution" bash "$artifacts_sh" "$link" "$VERSION"

rejects "malformed version" bash "$artifacts_sh" "$complete" "v9.9.9"

# ── smoke_test_wheel.sh ─────────────────────────────────────────────────────
# Only the rejection cases: the accepting path installs the wheel and imports
# it, which needs a real build rather than a fixture. Every check below runs
# before the script reaches pip, so an empty file named like a wheel is enough.
echo "smoke_test_wheel.sh"

smoke() { # smoke <dir> <wheel-name...>
  local dir="$work/$1"
  shift
  mkdir -p "$dir"
  local name
  for name in "$@"; do : > "$dir/$name"; done
  printf '%s' "$dir"
}

rejects "dist directory absent" bash "$smoke_sh" "$work/no-dist"
rejects "no wheel built" bash "$smoke_sh" "$(smoke smoke-empty)"
rejects "more than one wheel built" bash "$smoke_sh" \
  "$(smoke smoke-two "${PREFIX}-cp310-abi3-win_amd64.whl" \
    "${PREFIX}-cp310-abi3-win_arm64.whl")"
rejects "wheel is not abi3" bash "$smoke_sh" \
  "$(smoke smoke-versioned "${PREFIX}-cp312-cp312-win_amd64.whl")"

# ── verify_prepare_run.sh ───────────────────────────────────────────────────
echo "verify_prepare_run.sh"

COMMIT="0123456789abcdef0123456789abcdef01234567"

run_json() { # run_json <name> <jq-assignment...>
  local name="$1"
  shift
  local path="$work/$name.json"
  jq -n \
    --arg sha "$COMMIT" \
    '{head_sha: $sha,
      path: ".github/workflows/release-prepare.yml",
      event: "push",
      status: "completed",
      conclusion: "success"}' > "$path"
  # A broken jq expression must abort the run, not silently leave the fixture
  # in its unmodified (valid) state and turn a rejection case into a pass.
  if test "$#" -gt 0; then
    jq "$@" "$path" > "$path.tmp"
    mv "$path.tmp" "$path"
  fi
  printf '%s' "$path"
}

ok "successful prepare run" bash "$prepare_sh" "$(run_json good)" "$COMMIT"
rejects "different commit" bash "$prepare_sh" "$(run_json good)" \
  "89abcdef0123456789abcdef0123456789abcdef"
rejects "still running" \
  bash "$prepare_sh" "$(run_json running '.status = "in_progress" | .conclusion = null')" "$COMMIT"
rejects "failed run" \
  bash "$prepare_sh" "$(run_json failed '.conclusion = "failure"')" "$COMMIT"
rejects "different workflow" \
  bash "$prepare_sh" "$(run_json other '.path = ".github/workflows/ci.yml"')" "$COMMIT"
rejects "dispatched, not tag-pushed" \
  bash "$prepare_sh" "$(run_json dispatch '.event = "workflow_dispatch"')" "$COMMIT"
rejects "malformed commit" bash "$prepare_sh" "$(run_json good)" "not-a-sha"
rejects "missing metadata file" bash "$prepare_sh" "$work/absent.json" "$COMMIT"

not_json="$work/not-json.json"
printf 'ordinary text\n' > "$not_json"
rejects "metadata is not JSON" bash "$prepare_sh" "$not_json" "$COMMIT"

# ── changelog_section.sh ────────────────────────────────────────────────────
# Run from a scratch directory so the real CHANGELOG.md is never the thing
# under test — the fixtures below decide the outcome, not the repo's history.
echo "changelog_section.sh"

changelog_dir="$work/changelog"
mkdir -p "$changelog_dir"
cat > "$changelog_dir/CHANGELOG.md" <<'EOF'
# Changelog

## [1.0.0] - 2026-01-01

### Added

- A thing.

---

## [0.9.0] - 2025-12-01

### Fixed

- Another thing.

## [0.8.0] - 2025-11-01

EOF

section() { # section <version>
  (cd "$changelog_dir" && bash "$changelog_sh" "$1" "$work/section.md")
}

ok "section present" section "1.0.0"
rejects "section absent" section "2.0.0"
rejects "section empty" section "0.8.0"

if section "1.0.0" >/dev/null 2>&1; then
  if grep -q '^- A thing\.$' "$work/section.md" &&
    ! grep -q 'Another thing' "$work/section.md" &&
    ! tail -n 1 "$work/section.md" | grep -q '^---$'; then
    printf '  ok    section body is that version only, rule stripped\n'
  else
    printf '  FAIL  section body leaked adjacent content\n'
    failures=$((failures + 1))
  fi
fi

empty_dir="$work/no-changelog"
mkdir -p "$empty_dir"
rejects "CHANGELOG.md absent" \
  bash -c "cd '$empty_dir' && bash '$changelog_sh' 1.0.0 '$work/section.md'"

# ────────────────────────────────────────────────────────────────────────────
if test "$failures" -gt 0; then
  printf '\n%d verifier check(s) failed\n' "$failures" >&2
  exit 1
fi

printf '\nrelease verifiers: ok\n'
