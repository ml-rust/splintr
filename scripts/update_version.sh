#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VERSION_FILE="$PROJECT_ROOT/.version"
CARGO_TOML="$PROJECT_ROOT/Cargo.toml"
PYPROJECT_TOML="$PROJECT_ROOT/pyproject.toml"

# Read base version from .version file
if [[ ! -f "$VERSION_FILE" ]]; then
    echo "Error: .version file not found at $VERSION_FILE"
    exit 1
fi

BASE_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')

# Validate base version format (semver: X.Y.Z)
if [[ ! "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid base version format '$BASE_VERSION'. Expected X.Y.Z"
    exit 1
fi

# Get tag from argument or environment
TAG="${1:-$GITHUB_REF_NAME}"

if [[ -z "$TAG" ]]; then
    # No tag provided, use base version
    CARGO_VERSION="$BASE_VERSION"
    PYPI_VERSION="$BASE_VERSION"
    echo "No tag provided, using base version: $BASE_VERSION"
else
    # Remove 'v' prefix if present
    TAG_VERSION="${TAG#v}"

    # Extract base version and prerelease from tag
    # Supports: v0.1.0, v0.1.0-beta.1, v0.1.0-alpha.2, v0.1.0-rc.1
    if [[ "$TAG_VERSION" =~ ^([0-9]+\.[0-9]+\.[0-9]+)(-([a-zA-Z]+)\.([0-9]+))?$ ]]; then
        TAG_BASE="${BASH_REMATCH[1]}"
        PRERELEASE_TYPE="${BASH_REMATCH[3]}"
        PRERELEASE_NUM="${BASH_REMATCH[4]}"

        # Validate tag base matches .version file
        if [[ "$TAG_BASE" != "$BASE_VERSION" ]]; then
            echo "Error: Tag version mismatch!"
            echo "  Tag base version: $TAG_BASE"
            echo "  .version file:    $BASE_VERSION"
            echo ""
            echo "The tag must match the base version in .version file."
            echo "Valid tags for version $BASE_VERSION:"
            echo "  - v$BASE_VERSION"
            echo "  - v$BASE_VERSION-alpha.N"
            echo "  - v$BASE_VERSION-beta.N"
            echo "  - v$BASE_VERSION-rc.N"
            exit 1
        fi

        if [[ -n "$PRERELEASE_TYPE" ]]; then
            # Convert to lowercase for case-insensitive matching
            PRERELEASE_TYPE_LOWER=$(echo "$PRERELEASE_TYPE" | tr '[:upper:]' '[:lower:]')

            # Convert prerelease type for Cargo (uses hyphen) and PyPI (uses different format)
            # Cargo: 0.1.0-beta.1
            # PyPI: 0.1.0b1 (alpha=a, beta=b, rc=rc)
            CARGO_VERSION="$BASE_VERSION-$PRERELEASE_TYPE_LOWER.$PRERELEASE_NUM"

            case "$PRERELEASE_TYPE_LOWER" in
                alpha)
                    PYPI_VERSION="${BASE_VERSION}a${PRERELEASE_NUM}"
                    ;;
                beta)
                    PYPI_VERSION="${BASE_VERSION}b${PRERELEASE_NUM}"
                    ;;
                rc)
                    PYPI_VERSION="${BASE_VERSION}rc${PRERELEASE_NUM}"
                    ;;
                *)
                    echo "Error: Unknown prerelease type '$PRERELEASE_TYPE'"
                    echo "Supported types: alpha, beta, rc (case-insensitive)"
                    exit 1
                    ;;
            esac

            echo "Prerelease version detected:"
            echo "  Cargo version: $CARGO_VERSION"
            echo "  PyPI version:  $PYPI_VERSION"
        else
            # Stable release
            CARGO_VERSION="$BASE_VERSION"
            PYPI_VERSION="$BASE_VERSION"
            echo "Stable release: $BASE_VERSION"
        fi
    else
        echo "Error: Invalid tag format '$TAG'"
        echo "Expected format: vX.Y.Z or vX.Y.Z-{alpha|beta|rc}.N"
        echo "Examples: v0.1.0, v0.1.0-beta.1, v0.1.0-rc.2"
        exit 1
    fi
fi

echo ""
echo "Updating version files..."

# Update Cargo.toml - only update version in [package] section
if [[ -f "$CARGO_TOML" ]]; then
    # Use awk to update version only in [package] section
    awk -v ver="$CARGO_VERSION" '
        /^\[package\]/ { in_package=1 }
        /^\[/ && !/^\[package\]/ { in_package=0 }
        in_package && /^version = "/ { print "version = \"" ver "\""; next }
        { print }
    ' "$CARGO_TOML" > "$CARGO_TOML.tmp" && mv "$CARGO_TOML.tmp" "$CARGO_TOML"

    # Verify the update worked (grep first version line and extract quoted string)
    UPDATED_VERSION=$(grep '^version = "' "$CARGO_TOML" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [[ "$UPDATED_VERSION" != "$CARGO_VERSION" ]]; then
        echo "Error: Failed to update version in $CARGO_TOML"
        echo "  Expected: $CARGO_VERSION"
        echo "  Got: $UPDATED_VERSION"
        exit 1
    fi
    echo "  Updated $CARGO_TOML -> $CARGO_VERSION"
else
    echo "  Warning: $CARGO_TOML not found"
fi

# Update pyproject.toml - only update version in [project] section
if [[ -f "$PYPROJECT_TOML" ]]; then
    # Use awk to update version only in [project] section
    awk -v ver="$PYPI_VERSION" '
        /^\[project\]/ { in_project=1 }
        /^\[/ && !/^\[project\]/ { in_project=0 }
        in_project && /^version = "/ { print "version = \"" ver "\""; next }
        { print }
    ' "$PYPROJECT_TOML" > "$PYPROJECT_TOML.tmp" && mv "$PYPROJECT_TOML.tmp" "$PYPROJECT_TOML"

    # Verify the update worked (grep first version line and extract quoted string)
    UPDATED_VERSION=$(grep '^version = "' "$PYPROJECT_TOML" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [[ "$UPDATED_VERSION" != "$PYPI_VERSION" ]]; then
        echo "Error: Failed to update version in $PYPROJECT_TOML"
        echo "  Expected: $PYPI_VERSION"
        echo "  Got: $UPDATED_VERSION"
        exit 1
    fi
    echo "  Updated $PYPROJECT_TOML -> $PYPI_VERSION"
else
    echo "  Warning: $PYPROJECT_TOML not found"
fi

# Update python/splintr/__init__.py - update __version__ variable
PYTHON_INIT="$PROJECT_ROOT/python/splintr/__init__.py"
if [[ -f "$PYTHON_INIT" ]]; then
    # Use sed to update __version__ = "X.Y.Z" line
    sed -i "s/^__version__ = \".*\"/__version__ = \"$PYPI_VERSION\"/" "$PYTHON_INIT"

    # Verify the update worked
    UPDATED_VERSION=$(grep '^__version__ = "' "$PYTHON_INIT" | sed 's/.*"\(.*\)".*/\1/')
    if [[ "$UPDATED_VERSION" != "$PYPI_VERSION" ]]; then
        echo "Error: Failed to update version in $PYTHON_INIT"
        echo "  Expected: $PYPI_VERSION"
        echo "  Got: $UPDATED_VERSION"
        exit 1
    fi
    echo "  Updated $PYTHON_INIT -> $PYPI_VERSION"
else
    echo "  Warning: $PYTHON_INIT not found"
fi

echo ""
echo "Version update complete!"
echo "  Base version:    $BASE_VERSION"
echo "  Cargo.toml:      $CARGO_VERSION"
echo "  pyproject.toml:  $PYPI_VERSION"
echo "  __init__.py:     $PYPI_VERSION"
