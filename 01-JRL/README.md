## JRL - Preprocessing dataset

```bash
source JRL/preprocessing_data.sh
```
<details>
<summary> Details code </summary>

</details>


<details>
<summary> Description </summary>

### STEP 1 : Index datasets (Entity) 
`index_and_filter_review_file.py `

This script processes the review data to generate various entity files.
#### Generated Files:
- `vocab.txt`: Contains a list of unique words from the reviews.
- `user.txt`: Contains a list of unique user IDs.
- `product.txt`: Contains a list of unique product IDs.
- `review_text.txt`: Contains the text of the reviews.
- `review_u_p.txt`: Maps reviews to users and products.
- `review_id.txt`: Contains unique review IDs.

### STEP 2 : Split datasets for training and test 
`split_train_test.py`

### STEP 3 : Extract gzip to txt 
`gzip -d *.txt.gz`

### STEP 4 : Matching Relations
`match_cate_brand_related.py`

This script processes the data to generate relation files, which describe various relationships between entities such as products, brands, and categories.
#### Generated Files:
- `also_bought_p_p.txt`: Contains pairs of products that are often bought together.
- `also_view_p_p.txt`: Contains pairs of products that are often viewed together.
- `bought_together_p_p.txt`: Contains pairs of products that are frequently bought together.
- `brand_p_b.txt`: Maps products to their respective brands.
- `category_p_c.txt`: Maps products to their respective categories.
- `brand.txt`: Contains a list of unique brands.
- `category.txt`: Contains a list of unique categories.
- `related_product.txt` : Contains a list of unique related_product product IDs.

</details>