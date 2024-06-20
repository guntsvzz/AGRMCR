#!/bin/bash

# Sports_and_Outdoors
# Video_Games
# Appliances

# dataset=Sports_and_Outdoors 
dataset=Office_Products

mkdir stats 
mkdir data 
mkdir tmp
mkdir embs
mkdir processed

# echo "---------------- step 1: feature filter ----------------"
# python3 1_feature_filter.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 2: edge extraction ---------------"
# python3 2_edge_extractor.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 3: edge filter -------------------"
# python3 3_edge_filter.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 4: data formulation --------------"
# python3 4_data_formulator.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 5: category embedding generation --------------"
# python3 5_embs_generator.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 6: train-test-validation split --------------"
# python3 6_dataset_split.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 7: mapping categories-type --------------"
# python3 7_categories_mapping.py $dataset
# echo "--------------------------------------------------------"

echo "---------------- step 8: create user_dictionary --------------"
python3 8_user_dict.py $dataset
echo "--------------------------------------------------------"

# echo "---------------- step 9: create item_dictionary --------------"
# python3 9_item_dict.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 10: create feature_dictionary --------------"
# python3 10_feature_dict.py $dataset
# echo "--------------------------------------------------------"

# echo "---------------- step 11: index_filter_review --------------"
# python3 11_index_and_filter_review_file.py $dataset ./tmp 0
# echo "--------------------------------------------------------"