#!/bin/bash

# Move to the 01-JRL directory
cd "$(dirname "$0")"

# Create the necessary directories if they don't exist
mkdir -p ../02-TransE/data/beauty
mkdir -p ../02-TransE/data/cds
mkdir -p ../02-TransE/data/clothing
mkdir -p ../02-TransE/data/cellphones

# Copy the directories to the new locations
cp -r data/beauty ../02-TransE/data/beauty/Amazon_Beauty
cp -r data/cds_and_vinyl ../02-TransE/data/cds/Amazon_CDs
cp -r data/clothing_shoes_and_jewelry ../02-TransE/data/clothing/Amazon_Clothing
cp -r data/cell_phones_and_accessories ../02-TransE/data/cellphones/Amazon_Cellphones

echo "Cloning complete."
