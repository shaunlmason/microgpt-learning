#!/bin/bash
#
# Benchmark comparison script for microgpt implementations.
#
# Reads JSON results from benchmarks/results/ and generates
# a markdown comparison table.
#
# Requires: jq (for JSON parsing)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
OUTPUT_FILE="$RESULTS_DIR/comparison.md"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for jq
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required for JSON parsing${NC}"
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

# Find all result files
result_files=("$RESULTS_DIR"/*.json)

if [ ${#result_files[@]} -eq 0 ] || [ ! -f "${result_files[0]}" ]; then
    echo -e "${YELLOW}No benchmark results found in $RESULTS_DIR${NC}"
    exit 0
fi

echo -e "${GREEN}Generating comparison from ${#result_files[@]} result file(s)...${NC}"

# Start building markdown
{
    echo "# Benchmark Comparison"
    echo ""
    echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo ""

    # Collect language info
    echo "## Environment"
    echo ""
    echo "| Language | Version |"
    echo "|----------|---------|"

    for f in "${result_files[@]}"; do
        if [ -f "$f" ]; then
            lang=$(jq -r '.language' "$f")
            version=$(jq -r '.version' "$f")
            echo "| $lang | $version |"
        fi
    done

    echo ""
    echo "## Results"
    echo ""

    # Build header row dynamically
    header="| Benchmark |"
    separator="|-----------|"

    for f in "${result_files[@]}"; do
        if [ -f "$f" ]; then
            lang=$(jq -r '.language' "$f")
            header="$header $lang |"
            separator="$separator------:|"
        fi
    done

    # Add ratio column if we have exactly 2 languages
    if [ ${#result_files[@]} -eq 2 ]; then
        header="$header Ratio |"
        separator="$separator------:|"
    fi

    echo "$header"
    echo "$separator"

    # Get all benchmark names from first file
    benchmark_names=$(jq -r '.benchmarks[].name' "${result_files[0]}")

    # For each benchmark, output a row
    while IFS= read -r bench_name; do
        row="| $bench_name |"
        times=()

        for f in "${result_files[@]}"; do
            if [ -f "$f" ]; then
                time_ms=$(jq -r ".benchmarks[] | select(.name == \"$bench_name\") | .time_ms" "$f")
                if [ "$time_ms" != "null" ] && [ -n "$time_ms" ]; then
                    row="$row ${time_ms}ms |"
                    times+=("$time_ms")
                else
                    row="$row N/A |"
                    times+=("0")
                fi
            fi
        done

        # Calculate ratio if we have 2 results
        if [ ${#result_files[@]} -eq 2 ] && [ ${#times[@]} -eq 2 ]; then
            t1="${times[0]}"
            t2="${times[1]}"
            if [ "$t1" != "0" ] && [ "$t2" != "0" ]; then
                # Calculate ratio (larger/smaller) and indicate which is faster
                ratio=$(echo "scale=2; if ($t1 > $t2) $t1 / $t2 else $t2 / $t1" | bc)
                if (( $(echo "$t1 < $t2" | bc -l) )); then
                    lang1=$(jq -r '.language' "${result_files[0]}")
                    row="$row ${ratio}x (${lang1}) |"
                elif (( $(echo "$t2 < $t1" | bc -l) )); then
                    lang2=$(jq -r '.language' "${result_files[1]}")
                    row="$row ${ratio}x (${lang2}) |"
                else
                    row="$row 1.00x |"
                fi
            else
                row="$row N/A |"
            fi
        fi

        echo "$row"
    done <<< "$benchmark_names"

    echo ""
    echo "## Benchmark Descriptions"
    echo ""

    # Output descriptions from first file
    jq -r '.benchmarks[] | "- **\(.name)**: \(.description)"' "${result_files[0]}"

    echo ""
    echo "---"
    echo ""
    echo "*Lower times are better. Ratio shows how many times faster the winning language is.*"

} > "$OUTPUT_FILE"

echo -e "${GREEN}Comparison saved to: $OUTPUT_FILE${NC}"
echo ""

# Also print to terminal
cat "$OUTPUT_FILE"
