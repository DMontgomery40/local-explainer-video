#!/bin/bash
# Compress explainer videos in projects/ to ~20MB 1080p MP4 files
# Uses libx264 CPU encoding with ultrafast preset

PROJECTS_DIR="projects"
MIN_SIZE_MB=50

# Track totals
total_original=0
total_compressed=0
files_processed=0

echo "=== Video Compression Script ==="
echo "Using libx264 ultrafast preset, CRF 35"
echo ""

# Find all MP4 files larger than MIN_SIZE_MB
find "$PROJECTS_DIR" -name "*.mp4" -type f -print0 2>/dev/null | while IFS= read -r -d '' file; do
    size_bytes=$(stat -f%z "$file")
    size_mb=$((size_bytes / 1024 / 1024))

    if [ "$size_mb" -lt "$MIN_SIZE_MB" ]; then
        continue
    fi

    echo "Processing: $file (${size_mb}MB)"

    temp_file="${file%.mp4}_temp.mp4"

    # Compress - redirect stderr to stdout for progress, but suppress most output
    ffmpeg -y -i "$file" \
        -c:v libx264 \
        -preset ultrafast \
        -crf 35 \
        -vf "scale=-2:1080" \
        -c:a aac -b:a 128k \
        -movflags +faststart \
        "$temp_file" </dev/null 2>&1 | tail -1

    if [ -f "$temp_file" ] && [ "$(stat -f%z "$temp_file")" -gt 0 ]; then
        compressed_bytes=$(stat -f%z "$temp_file")
        compressed_mb=$((compressed_bytes / 1024 / 1024))
        mv "$temp_file" "$file"
        savings=$((size_mb - compressed_mb))
        echo "  ✓ ${size_mb}MB → ${compressed_mb}MB (saved ${savings}MB)"
    else
        echo "  ✗ Failed"
        rm -f "$temp_file"
    fi
    echo ""
done

echo "=== Done ==="
