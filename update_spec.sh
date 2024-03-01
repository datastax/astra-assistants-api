#!/bin/bash

set -euo pipefail
set -o noclobber
# Optionally, use `set -x` for debugging
# set -x


HASH=$(git ls-remote https://github.com/openai/openai-openapi.git refs/heads/master | awk -F" " '{print $1}')

CURRENT_HASH=$(cat OPEN_API_SPEC_HASH)

if [[ "$HASH" != "$CURRENT_HASH" ]]; then

    curl -OL https://raw.githubusercontent.com/openai/openai-openapi/"$HASH"/openapi.yaml

    mv openapi.yaml source_openapi.yaml

    git add source_openapi.yaml

    git commit -m "update yaml to $HASH"

    git apply --ignore-whitespace --3way patches/0001-modify-yaml-for-streaming.patch

    echo  Patch Applied, perform the merge and commit:
    echo git add source_openapi.yaml
    echo git commit -m "Applied streaming patch to openapi spec"

    echo "$HASH" |> OPEN_API_SPEC_HASH

else
    echo Alread up to date with the latest spec: https://raw.githubusercontent.com/openai/openai-openapi/"$HASH"/openapi.yaml
fi

