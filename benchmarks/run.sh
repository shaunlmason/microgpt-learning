#!/bin/bash
#
# Benchmark orchestrator script for microgpt implementations.
#
# Runs benchmarks for all registered languages and saves JSON results.
# Results are stored in benchmarks/results/ directory.
#
# Usage:
#   ./benchmarks/run.sh              # Run all benchmarks
#   ./benchmarks/run.sh python       # Run only Python
#   ./benchmarks/run.sh typescript   # Run only TypeScript
#   ./benchmarks/run.sh --list       # List available benchmarks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# BENCHMARK REGISTRY
# ============================================================================
# Format: "name|command|working_dir"
#
# To add a new language:
# 1. Create a benchmark script that outputs JSON to stdout (see README.md)
# 2. Add an entry to this array
# ============================================================================

declare -a BENCHMARKS=(
    "python|python3 benchmark_json.py|py"
    "typescript|npx tsx benchmark.ts|ts-node"
    "bun|bun run benchmark.ts|ts-bun"
    "rust|cargo run --release --bin benchmark 2>/dev/null|rust"
    # Add new languages here:
    # "go|go run ./cmd/benchmark|go"
    # "zig|zig build run-benchmark|zig"
)

# ============================================================================
# Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           microgpt Benchmark Suite                         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

list_benchmarks() {
    echo "Available benchmarks:"
    echo ""
    for entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r name cmd dir <<< "$entry"
        echo -e "  ${GREEN}$name${NC}"
        echo -e "    Command: $cmd"
        echo -e "    Directory: $dir"
        echo ""
    done
}

run_benchmark() {
    local name="$1"
    local cmd="$2"
    local dir="$3"
    local workdir="$PROJECT_ROOT/$dir"
    local outfile="$RESULTS_DIR/$name.json"

    echo -e "${YELLOW}Running ${name} benchmark...${NC}"

    if [ ! -d "$workdir" ]; then
        echo -e "${RED}  ERROR: Directory not found: $workdir${NC}"
        return 1
    fi

    # Run the benchmark and capture output
    if (cd "$workdir" && eval "$cmd") > "$outfile" 2>&1; then
        # Validate JSON output
        if command -v jq &> /dev/null; then
            if jq empty "$outfile" 2>/dev/null; then
                echo -e "${GREEN}  ✓ Completed: $outfile${NC}"
            else
                echo -e "${RED}  ERROR: Invalid JSON output${NC}"
                cat "$outfile"
                return 1
            fi
        else
            echo -e "${GREEN}  ✓ Completed: $outfile${NC}"
        fi
    else
        echo -e "${RED}  ERROR: Benchmark failed${NC}"
        cat "$outfile"
        return 1
    fi
}

run_all_benchmarks() {
    local filter="$1"
    local ran_any=false

    for entry in "${BENCHMARKS[@]}"; do
        IFS='|' read -r name cmd dir <<< "$entry"

        # Skip if filter specified and doesn't match
        if [ -n "$filter" ] && [ "$filter" != "$name" ]; then
            continue
        fi

        run_benchmark "$name" "$cmd" "$dir"
        ran_any=true
    done

    if [ "$ran_any" = false ]; then
        echo -e "${RED}No matching benchmark found: $filter${NC}"
        echo ""
        list_benchmarks
        exit 1
    fi
}

# ============================================================================
# Main
# ============================================================================

print_header

# Create results directory
mkdir -p "$RESULTS_DIR"

# Parse arguments
case "${1:-}" in
    --list|-l)
        list_benchmarks
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 [language|--list|--help]"
        echo ""
        echo "Options:"
        echo "  (none)      Run all benchmarks"
        echo "  <language>  Run only the specified language benchmark"
        echo "  --list, -l  List available benchmarks"
        echo "  --help, -h  Show this help message"
        exit 0
        ;;
    *)
        run_all_benchmarks "$1"
        ;;
esac

echo ""
echo -e "${GREEN}All benchmarks complete!${NC}"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""

# Run comparison if we have results
if [ -f "$SCRIPT_DIR/compare.sh" ]; then
    echo "Generating comparison..."
    "$SCRIPT_DIR/compare.sh"
fi
