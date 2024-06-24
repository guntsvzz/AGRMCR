## JRL - Preprocessing dataset

```bash
source JRL/preprocessing_data.sh
```
<details>
<summary> Details code </summary>

```bash
DATASET_NAME=Beauty
# DATASET_NAME=CDs_and_Vinyl
# DATASET_NAME=Clothing_Shoes_and_Jewelry
# DATASET_NAME=Cell_Phones_and_Accessories

echo "Dataset Name is ${DATASET_NAME}"
echo "------------- step 1: Index datasets (Entity) --------------"
REVIEW_FILE=./raw_data/reviews_${DATASET_NAME}_5.json.gz
INDEXED_DATA_DIR=./tmp/${DATASET_NAME}_
MIN_COUNT=15
python3 ./scripts/index_and_filter_review_file.py $REVIEW_FILE $INDEXED_DATA_DIR $MIN_COUNT
echo "------------------------------------------------------------"
# <REVIEW_FILE>: the file path for the Amazon review data
# <INDEXED_DATA_DIR>: output directory for indexed data
# <MIN_COUNT>: the minimum count for terms. If a term appears less then <MIN_COUNT> times in the data, it will be ignored.

echo "------------- step 2: Split datasets for training and test --------------"
SOURCE_DIR=./tmp/${DATASET_NAME}_min_count${MIN_COUNT}
SAMPLE_RATE=0.3
python3 ./scripts/split_train_test.py $SOURCE_DIR/ $SAMPLE_RATE
echo "-------------------------------------------------------------------------"

echo "------------- step 3: Extract gzip to txt ------------------"
# Convert DATASET_NAME to lowercase
DATASET_NAME_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')
DEST_DIR=./data/${DATASET_NAME_LOWER}
# Create the destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Find all .txt.gz files in the source directory, decompress them, and move the .txt files to the destination directory
for gz_file in "$SOURCE_DIR"/*.txt.gz; 
do
    echo "Processing $gz_file"
    # Decompress the file
    gzip -d "$gz_file"

    # Extract the base filename without extension
    BASE_NAME=$(basename "$gz_file" .gz)
    txt_file="${SOURCE_DIR}/${BASE_NAME}"
    echo "Move to $txt_file"
    
    # Check if the .txt file exists after decompression
    if [ -f "$txt_file" ]; then
        # Move the decompressed .txt file to the destination directory
        mv "$txt_file" "$DEST_DIR"
    else
        echo "Error: Decompressed file '$txt_file' not found."
    fi
done
echo "------------------------------------------------------------"

echo "------------- step 4: Matching Relations --------------"
python3 ./scripts/match_cate_brand_related.py $DATASET_NAME
echo "-------------------------------------------------------"
# DATASET_NAME: the domain name 
```

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