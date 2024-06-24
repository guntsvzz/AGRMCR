# Preprocessing dataset
data requirement 5-core data and meatadata

```bash
review_file=reviews_Beauty_5.json.gz
indexed_data_dir=./tmp/Beauty
min_count=15
python3 ./scripts/index_and_filter_review_file.py $review_file $indexed_data_dir $min_count
```

```bash
dataset_name=Beauty
python3 ./scripts/match_cate_brand_related.py $dataset_name
```

```bash
indexed_data_dir=./tmp/Beautymin_count15/
review_sample_rate=0.3
python3 ./scripts/split_train_test.py $indexed_data_dir $review_sample_rate
```