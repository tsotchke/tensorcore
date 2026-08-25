#!/usr/bin/env bash
# Push the current WORKING TREE to a GPU box and optionally build/test there.
#
# Why this exists
# ---------------
# tensorcore's CUDA sources cannot be compiled or tested on the Apple
# development machine -- there is no nvcc, and the manifold kernels have to be
# validated against real hardware. The loop is therefore always: edit locally,
# run on a box.
#
# Doing that through GitHub means every trial-and-error iteration becomes a
# commit, and it forces the private repo onto the box (credentials on a shared
# spot instance). This uses the gcloud SSH channel that already exists for
# operating these boxes, so it needs no deploy key, no token, and leaves no
# persistent access behind.
#
# It ships the working tree, not HEAD: `git archive` would only carry committed
# state, which is useless mid-debug. Tracked modifications and new untracked
# files both go; .gitignore is respected, so build/ and .cache/ stay put and the
# remote build stays incremental.
#
# Usage
#   PROJECT=example HOST=gpu-builder ZONE=region-zone scripts/sync-to-gpu-box.sh
#   PROJECT=example HOST=gpu-builder ZONE=region-zone \
#     scripts/sync-to-gpu-box.sh 'cmake --build build -j4'
#
# Env
#   PROJECT  cloud project id        (required)
#   HOST     instance name           (required)
#   ZONE     instance zone           (required)
#   DEST     remote checkout path    (default ~/tensorcore-cuda-build)
#   NICE     nice level for remote work (default 10 -- these boxes are shared)
#   ALLOW_UNTRACKED_EXPORT=1         include untracked, non-ignored files

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the cloud project id}"
HOST="${HOST:?set HOST to the GPU builder instance name}"
ZONE="${ZONE:?set ZONE to the instance zone}"
DEST="${DEST:-\$HOME/tensorcore-cuda-build}"
NICE="${NICE:-10}"
ALLOW_UNTRACKED_EXPORT="${ALLOW_UNTRACKED_EXPORT:-0}"
REMOTE_CMD="${1:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Tracked files plus untracked-not-ignored, NUL-separated so paths with spaces
# survive. Streamed straight into tar rather than collected into an array:
# macOS ships bash 3.2, which has no mapfile, and this is the development
# machine the script runs from.
git_files() {
    git ls-files -z --cached
    if [[ "$ALLOW_UNTRACKED_EXPORT" == "1" ]]; then
        git ls-files -z --others --exclude-standard
    fi
}

python3 scripts/check_release_privacy.py

N_FILES="$(git_files | tr -dc '\0' | wc -c | tr -d ' ')"
if [ "$N_FILES" -eq 0 ]; then
    echo "sync-to-gpu-box: nothing to send (is this a git repo?)" >&2
    exit 1
fi

echo "sync-to-gpu-box: ${N_FILES} files -> ${HOST} (${ZONE}):${DEST}"

# tar reads the file list on stdin, so the payload goes over the ssh command's
# stdin as a single stream -- one round trip, no scp fan-out.
REMOTE_SCRIPT="mkdir -p ${DEST} && tar xf - -C ${DEST} && echo 'sync: ok'"
if [ -n "$REMOTE_CMD" ]; then
    # PATH covers the toolkit and the venv cmake; both live outside the default
    # login PATH on these boxes.
    REMOTE_SCRIPT="${REMOTE_SCRIPT} && cd ${DEST} \
&& export PATH=\$HOME/tcbuild-venv/bin:/usr/local/cuda-13.0/bin:\$PATH \
&& nice -n ${NICE} ${REMOTE_CMD}"
fi

# --no-xattrs / --no-mac-metadata: macOS bsdtar otherwise attaches
# com.apple.provenance to every entry, and GNU tar on the box warns
# "Ignoring unknown extended header keyword" once per file -- 551 lines of
# noise that buries the build output the script exists to show. Both flags
# are bsdtar-only, so they are probed rather than assumed.
TAR_FLAGS=""
for f in --no-xattrs --no-mac-metadata; do
    if tar "$f" -cf /dev/null --files-from /dev/null 2>/dev/null; then
        TAR_FLAGS="$TAR_FLAGS $f"
    fi
done

# shellcheck disable=SC2086
git_files \
    | tar $TAR_FLAGS -cf - --null -T - \
    | gcloud compute ssh "$HOST" --project "$PROJECT" --zone "$ZONE" --command "$REMOTE_SCRIPT"
