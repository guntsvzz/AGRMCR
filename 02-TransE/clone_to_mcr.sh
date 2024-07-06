#!/bin/bash

# Move to the 01-JRL directory
cd "$(dirname "$0")"

# Create the necessary directories if they don't exist
mkdir -p ../03-UNICORN/tmp/beauty
mkdir -p ../03-UNICORN/tmp/cds
mkdir -p ../03-UNICORN/tmp/clothing
mkdir -p ../03-UNICORN/tmp/cellphones

# Copy the directories to the new locations
echo "--------------------Beauty---------------------------"
cp -r data/beauty/Amazon_Beauty_01_01/* ../03-UNICORN/tmp/beauty
echo "-----------------------------------------------------"

echo "--------------------CDs------------------------------"
cp -r data/cds/Amazon_CDs_01_01/* ../03-UNICORN/tmp/cds
echo "-----------------------------------------------------"

echo "--------------------Clothing-------------------------"
cp -r data/clothing/Amazon_Clothing_01_01/* ../03-UNICORN/tmp/clothing
echo "-----------------------------------------------------"

echo "--------------------Cellphones-----------------------"
cp -r data/cellphones/Amazon_Cellphones_01_01/* ../03-UNICORN/tmp/cellphones
echo "-----------------------------------------------------"

echo "Cloning complete."
