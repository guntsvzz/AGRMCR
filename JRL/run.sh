
echo "------------- step 1: Stem and remove stop words from the Amazon review datasets --------------"
jsonConfigFile=./example_jsonConfigFile.json
review_file=./raw_data/reviews_Beauty_5.json
output_review_file=./tmp/
# java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar $jsonConfigFile> $review_file $output_review_file
echo "-------------------------------------------------------------------------"
## <jsonConfigFile>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter "false" if don’t want to remove stop words. 
## <review_file>: the path for the original Amazon review data
## <output_review_file>: the output path for processed Amazon review data

echo "------------- step 2: Index datasets --------------"
review_file=./raw_data/reviews_Beauty_5.json.gz
indexed_data_dir=./tmp/Beauty_
min_count=15
python3 ./scripts/index_and_filter_review_file.py $review_file $indexed_data_dir $min_count
echo "-------------------------------------------------------------------------"
## <review_file>: the file path for the Amazon review data
## <indexed_data_dir>: output directory for indexed data
## <min_count>: the minimum count for terms. If a term appears less then <min_count> times in the data, it will be ignored.

echo "------------- step 3: Training --------------"
dataset_name=Beauty
python3 ./scripts/match_cate_brand_related.py $dataset_name
echo "---------------------------------------------"
## dataset_name: the domain name 

echo "------------- step 4: Split datasets for training and test --------------"
indexed_data_dir=./tmp/Beautymin_count15/
review_sample_rate=0.3
python3 ./scripts/split_train_test.py $indexed_data_dir $review_sample_rate
echo "-------------------------------------------------------------------------"
##Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 
##<indexed_data_dir>: the directory for indexed data.
##<review_sample_rate>: the proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).