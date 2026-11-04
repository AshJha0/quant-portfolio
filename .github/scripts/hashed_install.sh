#!/usr/bin/env bash
# Install local projects into the active Python from uv.lock, with every
# third-party wheel checked against the sha256 recorded in the lock.
#
#   hashed_install.sh [--extra NAME]... [--group NAME]... [--editable] PATH...
#
# 1. uv export writes the locked third-party closure of the named projects,
#    extras and root dependency groups, with hashes, to a requirements file.
#    Workspace members are left out (--no-emit-workspace).
# 2. pip installs that file with --require-hashes --only-binary :all:
#    --no-deps: a wheel whose hash is not in uv.lock, a missing hash, or an
#    sdist all fail the install, and nothing outside the file is fetched.
# 3. The projects themselves go in with --no-deps --no-build-isolation, so
#    their build backend (setuptools, from the "backend" group) also comes
#    from the hashed file rather than an unpinned isolated download.
# 4. pip check confirms the hashed set satisfies every installed project.
#
# The "backend" group is always added. Needs uv on PATH and a python with
# pip (the job's venv). Run from the repository root.
set -euo pipefail

extras=() groups=(--group backend) editable=0 paths=()
while (($#)); do
  case "$1" in
    --extra) extras+=(--extra "$2"); shift 2 ;;
    --group) groups+=(--group "$2"); shift 2 ;;
    --editable) editable=1; shift ;;
    -*) echo "hashed_install.sh: unknown option $1" >&2; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done

packages=()
for p in ${paths[@]+"${paths[@]}"}; do
  name=$(tr -d '\r' < "$p/pyproject.toml" | sed -n 's/^name *= *"\(.*\)"/\1/p' | head -n1)
  test -n "$name" || { echo "no project name in $p/pyproject.toml" >&2; exit 1; }
  packages+=(--package "$name")
done

req="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/hashed-requirements.txt"
uv export --frozen --format requirements-txt --no-header --no-emit-workspace \
  --no-editable ${packages[@]+"${packages[@]}"} ${extras[@]+"${extras[@]}"} "${groups[@]}" -o "$req" >/dev/null
echo "hashed_install.sh: $(grep -c '^[A-Za-z0-9]' "$req") locked packages for ${paths[*]:-(groups only)}"

python -m pip install --require-hashes --only-binary :all: --no-deps -r "$req"

if ((${#paths[@]})); then
  flags=()
  ((editable)) && for p in "${paths[@]}"; do flags+=(-e "$p"); done
  ((editable)) || flags=("${paths[@]}")
  python -m pip install --no-deps --no-build-isolation "${flags[@]}"
fi
python -m pip check
