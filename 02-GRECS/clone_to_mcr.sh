#!/bin/bash

# Move to the 01-JRL directory
cd "$(dirname "$0")"

# Create the necessary directories if they don't exist
mkdir -p ../03-UNICORN/data/beauty
mkdir -p ../03-UNICORN/data/cds
mkdir -p ../03-UNICORN/data/clothing
mkdir -p ../03-UNICORN/data/cellphones

# Copy the directories to the new locations
cp -r data/beauty ../03-UNICORN/data/beauty
cp -r data/cds_and_vinyl ../03-UNICORN/data/cds
cp -r data/clothing_shoes_and_jewelry ../03-UNICORN/data/clothing
cp -r data/cell_phones_and_accessories ../03-UNICORN/data/cellphones

echo "Cloning complete."
