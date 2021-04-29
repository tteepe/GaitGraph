#!/bin/sh

OUTPUT_DIR=./frames

for FILE in ./video/*
do
    FILENAME=$(basename "${FILE}" .avi)
    mkdir "$OUTPUT_DIR/$FILENAME"
    ffmpeg -i  "$FILE" "$OUTPUT_DIR/$FILENAME/%06d.jpg"
    echo "$FILENAME"
done
