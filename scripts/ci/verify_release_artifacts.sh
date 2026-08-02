#!/usr/bin/env bash
# Verify the downloaded distribution set is what Release Prepare built for this
# version, before any of it is published to PyPI or attached to a release.
#
#   scripts/ci/verify_release_artifacts.sh <artifact-dir> <pypi-version>
#
# Provenance of the producing run is established beforehand by
# verify_prepare_run.sh; this script only decides what the set may contain.
#
# Wheel filenames carry the platform tag the runner chose (`manylinux_2_34`,
# `macosx_11_0`, …), which moves with the images, so the platform families are
# matched by glob rather than by an exact name list. The version, by contrast,
# is compared exactly against every file: a wheel built before the version stamp
# is the one failure that would otherwise publish quietly under the wrong name.

set -euo pipefail

usage='usage: verify_release_artifacts.sh <artifact-dir> <pypi-version>'

die() {
  printf 'release artifacts: %s\n' "$*" >&2
  exit 1
}

artifact_dir="${1:?$usage}"
version="${2:?$usage}"

test -d "$artifact_dir" || die "missing artifact directory: $artifact_dir"

# PEP 440 as update_version.sh emits it: 0.11.0, 0.11.0a1, 0.11.0b2, 0.11.0rc1.
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]] ||
  die "invalid PyPI version: $version"

# maturin normalizes the distribution name `splintr-rs` to `splintr_rs`.
prefix="splintr_rs-${version}"

shopt -s nullglob dotglob
entries=("$artifact_dir"/*)
shopt -u nullglob dotglob

test "${#entries[@]}" -gt 0 || die "artifact directory is empty: $artifact_dir"

sdists=0
wheels=0

for entry in "${entries[@]}"; do
  name="${entry##*/}"

  # Symlinks are rejected ahead of the file test so a dangling link is reported
  # as what it is rather than as a missing file.
  if test -L "$entry" || test ! -f "$entry"; then
    die "not a regular file: $name"
  fi

  case "$name" in
    "${prefix}.tar.gz") sdists=$((sdists + 1)) ;;
    "${prefix}-"*.whl) wheels=$((wheels + 1)) ;;
    *.whl | *.tar.gz)
      die "distribution does not belong to ${prefix}: $name"
      ;;
    *)
      die "unexpected release attachment: $name"
      ;;
  esac
done

test "$sdists" -eq 1 ||
  die "expected exactly one sdist (${prefix}.tar.gz), found $sdists"

# Mirrors the wheel matrix in .github/workflows/release-prepare.yml. Adding a
# platform means changing both lists in the same commit; dropping one silently
# is how a platform stops receiving wheels without anyone noticing.
families=(
  "linux-x86_64:${prefix}-*linux*_x86_64.whl"
  "macos-x86_64:${prefix}-*macosx*_x86_64.whl"
  "macos-arm64:${prefix}-*macosx*_arm64.whl"
  "windows-x86_64:${prefix}-*win_amd64.whl"
)

for family in "${families[@]}"; do
  label="${family%%:*}"
  pattern="${family#*:}"
  shopt -s nullglob
  # Unquoted on purpose: `$pattern` is the glob, and quoting it would match a
  # file literally named `splintr_rs-…-*win_amd64.whl`.
  # shellcheck disable=SC2206
  matches=("$artifact_dir"/$pattern)
  shopt -u nullglob
  test "${#matches[@]}" -gt 0 || die "no wheel for $label (expected $pattern)"
done

printf 'release artifacts: ok (%d wheels, %d sdist)\n' "$wheels" "$sdists"
