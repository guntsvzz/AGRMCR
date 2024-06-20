import pandas as pd
import json
import sys
from tqdm.auto import tqdm
import os

def main():
    # Load the Office Products ID dictionary
    office_products_dict = {}
    with open(item_id, 'r') as f:
        for line in tqdm(f):
            asin, id_val = line.strip().split('\t')
            office_products_dict[asin] = int(id_val)
    
    print("Number of Item :", len(office_products_dict))
    # # Load all JSON lines into a list of dictionaries
    # with open(review_amazon_data, 'r') as fp:
    #     data = [json.loads(line.strip()) for line in fp]

    # # Create the pandas DataFrame
    # df_review = pd.DataFrame(data)
    # # Group by reviewerID and aggregate asins into a list
    # grouped_df = df_review.groupby('reviewerID')['asin'].apply(list).reset_index()
    
    df_rating = pd.read_csv(
        rating_amazon_data, 
        names=['asin','reviewerID','rating','timestamp'])
    
    # Filter for reviewerID and asin that occur at least 5 times
    reviewer_counts = df_rating['reviewerID'].value_counts()
    asin_counts = df_rating['asin'].value_counts()

    # Get the reviewerIDs and asins that occur at least 5 times
    reviewers_with_5_or_more = reviewer_counts[reviewer_counts >= 5].index
    asins_with_5_or_more = asin_counts[asin_counts >= 5].index
    # Filter the dataframe
    filtered_rating_only = df_rating[df_rating['reviewerID'].isin(reviewers_with_5_or_more) &
                                    df_rating['asin'].isin(asins_with_5_or_more)]

    group_rating = filtered_rating_only.groupby('reviewerID')['asin'].apply(list).reset_index()
    # Dropping rows where the length of the asin list is less than 5
    grouped_df = group_rating[group_rating['asin'].apply(len) >= 10].reset_index()

    print("Number of User :", grouped_df.shape[0])
    # Create the desired dictionary structure
    result = {}
    review = {}
    for index, row in tqdm(grouped_df.iterrows()):
        # Map the 'interact' ASINs to their corresponding IDs, retaining the original if not found
        # interact_ids = [office_products_dict[asin] if asin in office_products_dict else asin for asin in row['asin']]
        interact_ids = [office_products_dict[asin] for asin in row['asin'] if asin in office_products_dict]
        result[index] = {
            "interact": interact_ids,
            "friend": None,
            "like": None,
            "asin": row['reviewerID']
        }
        
        review[index] = interact_ids

    # Save the dictionary as JSON to a file
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
        
    print(f"JSON data saved to {save_path}")
     
    # Save the dictionary as JSON to a file
    with open(save_path_review, 'w') as f:
        json.dump(review, f, indent=4)

    print(f"Train JSON data saved to {save_path_review}")
    
    # Split the review dictionary into validation and test sets
    review_valid = {}
    review_test = {}

    for user, items in review.items():
        split_idx = int(len(items) * 0.7)
        review_valid[user] = items[:split_idx]
        review_test[user] = items[split_idx:]

    # Save the validation dictionary to a file
    save_path_valid = save_path_review.replace('train', 'valid')
    with open(save_path_valid, 'w') as f:
        json.dump(review_valid, f, indent=4)

    # Save the test dictionary to a file
    save_path_test = save_path_review.replace('train', 'test')
    with open(save_path_test, 'w') as f:
        json.dump(review_test, f, indent=4)

    print(f"Validation JSON data saved to {save_path_valid}")
    print(f"Test JSON data saved to {save_path_test}")

if __name__ == '__main__':
    data_name = sys.argv[1]
    # Open the JSON file
    item_id = "./tmp/{}_id_dict.txt".format(data_name)
    review_amazon_data = "./raw_data/{}_5.json".format(data_name)
    rating_amazon_data = "./raw_data/{}.csv".format(data_name)
    save_path = "./tmp/user_dict_{}.json".format(data_name)
    # save_path_review = "./tmp/review_dict_train_{}.json".format(data_name)
    
    # Get the directory path
    path_ui = f"../data/amazon_{data_name}/UI_Interaction_data/"
    # Create the directory if it does not exist
    if not os.path.exists(path_ui):
        os.makedirs(path_ui)
    save_path_review = f"{path_ui}review_dict_train_{data_name}.json"
    
    main()
