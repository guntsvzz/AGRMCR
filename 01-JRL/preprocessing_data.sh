
# echo "------------- step 0: Stem and remove stop words from the Amazon review datasets --------------"
# jsonConfigFile=./example_jsonConfigFile.json
# REVIEW_FILE=./raw_data/reviews_Beauty_5.json
# output_REVIEW_FILE=./tmp/
# java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar $jsonConfigFile> $REVIEW_FILE $output_REVIEW_FILE
# echo "-------------------------------------------------------------------------"
## <jsonConfigFile>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter "false" if don’t want to remove stop words. 
## <REVIEW_FILE>: the path for the original Amazon review data
## <output_REVIEW_FILE>: the output path for processed Amazon review data
# Define dataset names as a list
DATASET_NAMES=("Beauty" "CDs_and_Vinyl" "Clothing_Shoes_and_Jewelry" "Cell_Phones_and_Accessories")

# Iterate over each dataset name
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
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
    # <SOURCE_DIR>: the directory for indexed data.
    # <SAMPLE_RATE>: the proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).

    echo "------------- step 3: Extract gzip to txt ------------------"
    # Convert DATASET_NAME to lowercase
    DATASET_NAME_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')
    DEST_DIR=./data/${DATASET_NAME_LOWER}
    # Create the destination directory if it does not exist
    mkdir -p "$DEST_DIR"
    # Find all .txt.gz files in the source directory, decompress them, and move the .txt files to the destination directory
    for gz_file in "$SOURCE_DIR"/*.txt.gz; do
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
    # <DATASET_NAME>: the domain name of Amazon dataset

    SOURCE_DIR=./data/${DATASET_NAME_LOWER}/
    echo "------------- step 5 : Combination Feature ------------"
    python3 ./scripts/match_feature.py $SOURCE_DIR
    echo "-------------------------------------------------------"
    # <SOURCE_DIR>: the path of domain name of Amazon dataset
done
