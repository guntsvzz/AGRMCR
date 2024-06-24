import os
import argparse
import json
import shutil
from collections import defaultdict, Counter

from easydict import EasyDict as edict
import numpy as np


def count_lines_in_file(filename):
    """Count the number of lines in a given file."""
    with open(filename, "r") as file:
        lines = sum(1 for _ in file)
    return lines


def copy_files(original_dir, destination_dir, ignore_files=[]):
    """Copy all the files from the original directory to the destination directory except for the ignore files."""
    for file_name in os.listdir(original_dir):
        if file_name not in ignore_files:
            original = os.path.join(original_dir, file_name)
            destination = os.path.join(destination_dir, file_name)
            shutil.copy(original, destination)


def get_word_distrib(train_file, test_file, nb_words):
    """Get the word distribution."""
    word_distrib = np.zeros(nb_words)
    with open(train_file, "r") as f:
        for line in f:
            words = line.split("\t")[2].split()
            for word in set(words):
                word_distrib[int(word)] += 1
    with open(test_file, "r") as f:
        for line in f:
            words = line.split("\t")[2].split()
            for word in set(words):
                word_distrib[int(word)] += 1
    return word_distrib


def build_new_vocab(old_vocab_path, new_vocab_path, removed_words):
    """Build and save the new vocabulary by excluding the words in removed_words."""
    new_vocab = dict()
    new_index = 0
    with open(old_vocab_path, "r") as f:
        with open(new_vocab_path, "w") as fn:
            for i, line in enumerate(f):
                if i not in removed_words:
                    new_vocab[i] = new_index
                    new_index += 1
                    fn.write(line)
    return new_vocab


def get_mentions_descriptions(train_file, test_file, new_vocab, removed_words):
    """Read the train and test files and save the words as users mentions and products descriptions relations
    using the new vocabulary and ignoring the removed words."""

    users_mentions = defaultdict(set)
    products_descriptions = defaultdict(set)

    with open(train_file, "r") as f:
        for line in f:
            user, product, words = line.split("\t")
            words = words.split()
            for word in words:
                word = int(word)
                if word not in removed_words:
                    users_mentions[int(user)].add(new_vocab[word])
                    products_descriptions[int(product)].add(new_vocab[word])

    with open(test_file, "r") as f:
        for line in f:
            user, product, words = line.split("\t")
            words = words.split()
            for word in words:
                word = int(word)
                if word not in removed_words:
                    users_mentions[int(user)].add(new_vocab[word])
                    products_descriptions[int(product)].add(new_vocab[word])
    return users_mentions, products_descriptions


def save_words_interactions(config, train_file, test_file):

    # Define the paths for the products, old vocabulary and new vocabulary
    products_path = os.path.join(config.original_data_dir, "product.txt")
    old_vocab_path = os.path.join(config.original_data_dir, "vocab.txt")
    new_vocab_path = os.path.join(config.processed_data_dir, "vocab.txt")
    users_path = os.path.join(config.original_data_dir, "users.txt")

    # Count the number of products, words and users
    nb_products = count_lines_in_file(products_path)
    nb_words = count_lines_in_file(old_vocab_path)
    nb_users = count_lines_in_file(users_path)

    # Get the word distribution
    word_distrib = get_word_distrib(train_file, test_file, nb_words)

    treshold = config.word_freq_threshold * nb_products

    # get the indexes of the words that are more frequent than the treshold
    indexes = np.where(word_distrib > treshold)[0]
    removed_words = set(indexes)

    # build and save the new vocabulary by excluding the words in removed_words
    new_vocab = build_new_vocab(old_vocab_path, new_vocab_path, removed_words)

    users_mentions = defaultdict(set)
    products_descriptions = defaultdict(set)

    # Read the train and test files and save the words as users mentions and products descriptions relations
    # using the new vocabulary and ignoring the removed words
    users_mentions, products_descriptions = get_mentions_descriptions(
        train_file, test_file, new_vocab, removed_words
    )

    # Save the users mentions and products descriptions relations
    mention_file = os.path.join(config.processed_data_dir, "mentioned_by_u_w.txt")
    with open(mention_file, "w") as f:
        for i in range(nb_users):
            f.write(" ".join(map(str, users_mentions[i])) + "\n")

    described_as_file = os.path.join(config.processed_data_dir, "described_as_p_w.txt")
    with open(described_as_file, "w") as f:
        for i in range(nb_products):
            f.write(" ".join(map(str, products_descriptions[i])) + "\n")


def save_purchases(config, train_file, test_file):
    """Save and return the purchases for the users and products."""
    purchases = defaultdict(list)
    purchases_file = os.path.join(config.processed_data_dir, "purchases.txt")
    with open(purchases_file, "w") as pf:
        # Read the train file and save the user/product in purchases_file and all the reviews in the reviews_file
        with open(train_file, "r") as f:
            for line in f:
                user, product, _ = line.split("\t")
                pf.write(f"{user} {product}\n")
                purchases[int(user)].append(int(product))

        # Read the test file and save the user/product in purchases_file and all the reviews in the reviews_file
        with open(test_file, "r") as f:
            for line in f:
                user, product, _ = line.split("\t")
                pf.write(f"{user} {product}\n")
                purchases[int(user)].append(int(product))

    return purchases


def get_p_brand(config):
    """Get the brand of the products."""
    p_brand = []
    brand_file = os.path.join(config.processed_data_dir, "brand_p_b.txt")
    with open(brand_file, "r") as f:
        for line in f:
            line = line.split()
            p_brand.append(line)
    return p_brand


def get_p_category(config):
    """Get the category of the products."""
    p_category = []
    category_file = os.path.join(config.processed_data_dir, "category_p_c.txt")
    with open(category_file, "r") as f:
        for line in f:
            line = line.split()
            p_category.append(line)
    return p_category


def make_interested_in(config, purchases, p_category, nb_users):
    """Create the "user interested_in category" relation and save it in interested_in_u_c.txt."""
    interested_in_c_u = defaultdict(Counter)
    interested_in_c_u_file = os.path.join(
        config.processed_data_dir, "interested_in_u_c.txt"
    )
    for user, products in purchases.items():
        for product in products:
            for category in p_category[product]:
                if category != "-1":
                    interested_in_c_u[int(user)][category] += 1
    # rank the categories by frequency
    for user in interested_in_c_u:
        interested_in_c_u[user] = interested_in_c_u[user].most_common()

    with open(interested_in_c_u_file, "w") as f:
        for i in range(nb_users):
            # write the categories that the user is interested in, ignore the frequency
            f.write(" ".join([category for category, _ in interested_in_c_u[i]]) + "\n")


def make_like(config, purchases, p_brand, nb_users):
    """Create the "user like brand" relation and save it in like_u_b.txt."""
    like_b_u = defaultdict(Counter)
    like_b_u_file = os.path.join(config.processed_data_dir, "like_u_b.txt")
    for user, products in purchases.items():
        for product in products:
            for brand in p_brand[product]:
                if brand != "-1":
                    like_b_u[int(user)][brand] += 1
    # rank the brands by frequency
    for user in like_b_u:
        like_b_u[user] = like_b_u[user].most_common()
    with open(like_b_u_file, "w") as f:
        for i in range(nb_users):
            # write the brands that the user likes, ignore the frequency
            f.write(" ".join([brand for brand, _ in like_b_u[i]]) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="Config file.",
        default="config/clothing/graph_reasoning/preprocess.json",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = edict(json.load(f))

    # Create the processed data directory
    os.makedirs(config.processed_data_dir, exist_ok=True)

    # Copy all the files from the original data directory to the processed data directory except for train.txt and test.txt
    ignore_files = ["train.txt", "test.txt", "vocab.txt"]
    copy_files(config.original_data_dir, config.processed_data_dir, ignore_files)

    # Define the paths for the train and test files
    train_file = os.path.join(config.original_data_dir, "train.txt")
    test_file = os.path.join(config.original_data_dir, "test.txt")

    # Save the users mentions and products descriptions relations
    save_words_interactions(config, train_file, test_file)

    # Save the purchases for the users and products
    purchases = save_purchases(config, train_file, test_file)

    # Count the number of users
    users_path = os.path.join(config.original_data_dir, "users.txt")
    nb_users = count_lines_in_file(users_path)

    # Get the category of the products
    p_category = get_p_category(config)

    # Create the "user interested_in category" relation and save it in interested_in_u_c.txt
    make_interested_in(config, purchases, p_category, nb_users)

    # Get the brand of the products
    p_brand = get_p_brand(config)

    # Create the "user like brand" relation and save it in like_u_b.txt
    make_like(config, purchases, p_brand, nb_users)

    print("Preprocessing done.")


if __name__ == "__main__":
    main()
