#!/usr/bin/env bash
set -euo pipefail

YEAR=2026
AUTHOR="Victor Anderssén"

HEADER="// SPDX-License-Identifier: Apache-2.0
// Copyright ${YEAR} ${AUTHOR}
"

EXCLUDE_PATH="include/hip-img-fx/autotune/embedded_cache.h"

find src include tests \
    -type f \( -name "*.cpp" -o -name "*.h" \) | while read -r file; do

    # Skip generated file(s)
    if [[ "$file" == "$EXCLUDE_PATH" ]]; then
        continue
    fi

    # Skip files that already have SPDX headers
    if grep -q "SPDX-License-Identifier" "$file"; then
        continue
    fi

    tmp="$(mktemp)"
    {
        echo "$HEADER"
        cat "$file"
    } > "$tmp"

    mv "$tmp" "$file"
    echo "Updated: $file"
done
