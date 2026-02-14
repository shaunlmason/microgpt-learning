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

# Collect data into temp files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Get languages and versions
for f in "${result_files[@]}"; do
    if [ -f "$f" ]; then
        jq -r '[.language, .version] | @tsv' "$f" >> "$TEMP_DIR/lang_version.txt"
    fi
done

# Get benchmark names from first file
jq -r '.benchmarks[].name' "${result_files[0]}" > "$TEMP_DIR/benchmarks.txt"

# Calculate column widths
col1_width=9  # "Benchmark"
while IFS= read -r bench; do
    [ ${#bench} -gt $col1_width ] && col1_width=${#bench}
done < "$TEMP_DIR/benchmarks.txt"

# Calculate max language name width
max_lang_width=0
while IFS=$'\t' read -r lang ver; do
    [ ${#lang} -gt $max_lang_width ] && max_lang_width=${#lang}
done < "$TEMP_DIR/lang_version.txt"

# Time values typically need 6-7 chars (e.g., "21.11ms")
col_width=$((max_lang_width > 7 ? max_lang_width : 7))

# Helper function to repeat a character
repeat_char() {
    local char="$1"
    local count="$2"
    local result=""
    for ((i=0; i<count; i++)); do
        result="${result}${char}"
    done
    echo "$result"
}

# Start building markdown
{
    echo "# Benchmark Comparison"
    echo ""
    echo "Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo ""

    # Environment table
    env_col1_width=10  # "Language"
    env_col2_width=7   # "Version"
    
    # Find max widths
    while IFS=$'\t' read -r lang ver; do
        [ ${#lang} -gt $env_col1_width ] && env_col1_width=${#lang}
        [ ${#ver} -gt $env_col2_width ] && env_col2_width=${#ver}
    done < "$TEMP_DIR/lang_version.txt"
    
    echo "## Environment"
    echo ""
    printf "| %-${env_col1_width}s | %-${env_col2_width}s |\n" "Language" "Version"
    printf "|%s|%s|\n" "$(repeat_char '-' $((env_col1_width + 2)))" "$(repeat_char '-' $((env_col2_width + 2)))"
    
    while IFS=$'\t' read -r lang ver; do
        printf "| %-${env_col1_width}s | %-${env_col2_width}s |\n" "$lang" "$ver"
    done < "$TEMP_DIR/lang_version.txt"

    echo ""
    echo "## Results"
    echo ""
    
    # Header row
    printf "| %-${col1_width}s |" "Benchmark"
    while IFS=$'\t' read -r lang ver; do
        printf " %-${col_width}s |" "$lang"
    done < "$TEMP_DIR/lang_version.txt"
    echo ""
    
    # Separator
    printf "|%s|" "$(repeat_char '-' $((col1_width + 2)))"
    while IFS=$'\t' read -r lang ver; do
        printf "%s|" "$(repeat_char '-' $((col_width + 2)))"
    done < "$TEMP_DIR/lang_version.txt"
    echo ""
    
    # Data rows - build time lookup for each file
    file_index=0
    for f in "${result_files[@]}"; do
        if [ -f "$f" ]; then
            jq -r '.benchmarks[] | [.name, .time_ms] | @tsv' "$f" > "$TEMP_DIR/times_$file_index.txt"
            ((file_index++))
        fi
    done
    
    # For each benchmark, look up times from each file
    while IFS= read -r bench; do
        printf "| %-${col1_width}s |" "$bench"
        
        for ((i=0; i<file_index; i++)); do
            time_val=$(grep "^${bench}\t" "$TEMP_DIR/times_$i.txt" 2>/dev/null | cut -f2 || echo "N/A")
            if [ "$time_val" != "N/A" ]; then
                time_str="${time_val}ms"
            else
                time_str="N/A"
            fi
            printf " %-${col_width}s |" "$time_str"
        done
        echo ""
    done < "$TEMP_DIR/benchmarks.txt"

    echo ""
    echo "## Benchmark Descriptions"
    echo ""

    # Output descriptions from first file
    jq -r '.benchmarks[] | "- **\(.name)**: \(.description)"' "${result_files[0]}"

    echo ""
    echo "---"
    echo ""
    echo "*Lower times are better. Times shown in milliseconds.*"

} > "$OUTPUT_FILE"

echo -e "${GREEN}Comparison saved to: $OUTPUT_FILE${NC}"
echo ""

# Also print to terminal
cat "$OUTPUT_FILE"
