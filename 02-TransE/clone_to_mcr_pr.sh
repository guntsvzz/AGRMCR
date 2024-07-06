#!/bin/bash

# Move to the 01-JRL directory
cd "$(dirname "$0")"

# Create the necessary directories if they don't exist
mkdir -p ../03-UNICORN/tmp/beauty
mkdir -p ../03-UNICORN/tmp/cds
mkdir -p ../03-UNICORN/tmp/clothing
mkdir -p ../03-UNICORN/tmp/cellphones

mkdir -p ../04-GREC/data/beauty
mkdir -p ../04-GREC/data/cds
mkdir -p ../04-GREC/data/clothing
mkdir -p ../04-GREC/data/cellphones

# Copy the directories to the new locations
echo "--------------------Beauty---------------------------"
cp -r data/beauty/Amazon_Beauty_01_01/* ../03-UNICORN/tmp/beauty
cp -r data/beauty/Amazon_Beauty_01_01/* ../04-GREC/data/beauty/Amazon_Cellphones_01_01/
echo "-----------------------------------------------------"

echo "--------------------CDs------------------------------"
cp -r data/cds/Amazon_CDs_01_01/* ../03-UNICORN/tmp/cds
cp -r data/cds/Amazon_CDs_01_01/* ../04-GREC/data/cds/Amazon_Cellphones_01_01/

echo "-----------------------------------------------------"

echo "--------------------Clothing-------------------------"
cp -r data/clothing/Amazon_Clothing_01_01/* ../03-UNICORN/tmp/clothing
cp -r data/clothing/Amazon_Clothing_01_01/* ../04-GREC/data/clothing/Amazon_Cellphones_01_01/
echo "-----------------------------------------------------"

echo "--------------------Cellphones-----------------------"
cp -r data/cellphones/Amazon_Cellphones_01_01/* ../03-UNICORN/tmp/cellphones
cp -r data/cellphones/Amazon_Cellphones_01_01/* ../04-GREC/data/cellphones/Amazon_Cellphones_01_01/
echo "-----------------------------------------------------"

echo "Cloning complete."
