#!/bin/bash

# Set the URL and output file name
URL="https://www.googleapis.com/drive/v3/files/1VJFM0wuljsc2Dhdxus8h0IdcE9-0iJJu?alt=media&key=AIzaSyBo55XtefB47P_CPLKosGvnpEi3pQs5lCk"
OUTPUT_FILE="data.zip"

# Download the file using curl or wget
if command -v curl > /dev/null; then
    echo "Using curl to download the file."
    curl -L "$URL" -o "$OUTPUT_FILE"
elif command -v wget > /dev/null; then
    echo "Using wget to download the file."
    wget --no-check-certificate "$URL" -O "$OUTPUT_FILE"
else
    echo "Neither curl nor wget is installed. Please install one of them."
    exit 1
fi

# Unzip the downloaded file
if command -v unzip > /dev/null; then
    echo "Unzipping the file."
    unzip "$OUTPUT_FILE"
else
    echo "Unzip command not found. Please install unzip."
    exit 1
fi

# Remove the zip file
echo "Removing the zip file."
rm "$OUTPUT_FILE"
