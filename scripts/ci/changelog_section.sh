#!/usr/bin/env bash
# Extract one version's section from CHANGELOG.md.
#
#   scripts/ci/changelog_section.sh <version> [output]   # e.g. 0.11.0
#
# Writes the body of `## [<version>]` — everything up to the next `## [` — to
# `output` (default /tmp/changelog-section.md) and echoes it. Exits non-zero if
# CHANGELOG.md is missing or that section is absent or empty.
#
# Called by release-validate.yml as the release gate, and by release.yml to
# build the GitHub release body. One implementation, so the check and the notes
# can never disagree about what a section contains.
#
# Pass the BASE version: a `v0.11.1-beta.1` tag is a candidate for the
# `[0.11.1]` section, so a prerelease rehearses the notes the final release
# will carry.

set -euo pipefail

VERSION="${1:?usage: changelog_section.sh <version> [output]}"
OUTPUT="${2:-/tmp/changelog-section.md}"

if [[ ! -f CHANGELOG.md ]]; then
    echo "::error::CHANGELOG.md is required to cut a release"
    exit 1
fi

# Trailing blank lines and the `---` rule between sections are page furniture,
# not release notes.
awk -v version="$VERSION" '
    $0 ~ "^## \\[" version "\\]" { found = 1; next }
    found && /^## \[/ { exit }
    found { lines[n++] = $0 }
    END {
        while (n > 0 && (lines[n - 1] ~ /^[[:space:]]*$/ || lines[n - 1] ~ /^-{3,}[[:space:]]*$/)) {
            n--
        }
        for (i = 0; i < n; i++) print lines[i]
    }
' CHANGELOG.md > "$OUTPUT"

if [[ ! -s "$OUTPUT" ]]; then
    echo "::error::CHANGELOG.md has no non-empty '## [$VERSION]' section. Rename the Unreleased heading to [$VERSION] and describe the release."
    exit 1
fi

echo "Changelog section for [$VERSION]:"
cat "$OUTPUT"
