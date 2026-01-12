#!/bin/bash
# Generate test coverage report for hip-img-fx

set -e

echo "=== Configuring build with coverage enabled ==="
meson setup build --native-file native/hip.ini -Db_coverage=true --reconfigure

echo ""
echo "=== Building project ==="
ninja -C build

echo ""
echo "=== Running tests ==="
meson test -C build --print-errorlogs

echo ""
echo "=== Generating coverage report ==="
ninja -C build coverage-html

echo ""
echo "=== Coverage report generated ==="
echo "HTML report: build/meson-logs/coveragereport/index.html"
echo ""
echo "To view the report:"
echo "  xdg-open build/meson-logs/coveragereport/index.html"
echo ""

# Generate summary
if command -v gcovr &> /dev/null; then
    echo "=== Coverage Summary ==="
    cd build
    gcovr --root .. --print-summary
    cd ..
else
    echo "Install gcovr for detailed coverage summary: pip install gcovr"
fi
