import sys
import json
from tqdm.auto import tqdm
from collections import OrderedDict
import os
import gzip
def main():     
    # Genereate feature.txt
    # ========== BEGIN ========== #
    # Define file paths
    brand_file_path = os.path.join(source_dir, 'brand.txt')
    category_file_path = os.path.join(source_dir, 'category.txt')
    feature_file_path = os.path.join(source_dir, 'feature.txt')
    # Read the contents of brand.txt
    with open(brand_file_path, 'r') as brand_file:
        brand_data = brand_file.readlines()
    print('Number of brand :', len(brand_data))
    # Read the contents of category.txt
    with open(category_file_path, 'r') as category_file:
        category_data = category_file.readlines()
    print('Number of category :', len(category_data))
    # Combine the contents of the two files
    combined_data = brand_data + category_data
    # Write the combined contents to feature.txt
    with open(feature_file_path, 'w') as feature_file:
        feature_file.writelines(combined_data)
    print('Number of feature :', len(combined_data))
    print(f"The files have been combined into {feature_file_path}")
    # =========== END =========== #
    
    # Genereate feature_p_bc.txt
    # ========== BEGIN ========== #
    # Define file paths
    brand_p_b_file_path = os.path.join(source_dir, 'brand_p_b.txt')
    category_p_b_file_path = os.path.join(source_dir, 'category_p_c.txt')
    feature_p_b_file_path = os.path.join(source_dir, 'feature_p_bc.txt')
    # Read the contents of brand_p_b.txt
    with open(brand_p_b_file_path, 'r') as brand_file:
        brand_p_b_data = brand_file.readlines()
    # Read the contents of category_p_b.txt
    with open(category_p_b_file_path, 'r') as category_file:
        category_p_b_data = category_file.readlines()
        
    # Calculate the offset
    offset = len(brand_data)
    # # Combine each line of brand_p_b.txt with the corresponding line of category_p_b.txt
    # feature_p_bc_data = []
    # for brand_line, category_line in zip(brand_p_b_data, category_p_b_data):
    #     combined_line = f"{brand_line.strip()} {category_line.strip()}\n"
    #     feature_p_bc_data.append(combined_line)
    
    # Combine each line of brand_p_b.txt with the corresponding line of category_p_b.txt
    feature_p_bc_data = []
    for brand_line, category_line in zip(brand_p_b_data, category_p_b_data):
        brand_line = brand_line.strip()
        category_numbers = [str(int(num) + offset) for num in category_line.strip().split()]
        combined_line = f"{brand_line} {' '.join(category_numbers)}\n"
        feature_p_bc_data.append(combined_line)
        
    # Write the combined contents to feature_p_bc.txt
    with open(feature_p_b_file_path, 'w') as feature_file:
        feature_file.writelines(feature_p_bc_data)
    print('Number of brand_p_b :', len(brand_p_b_data))
    print('Number of category_p_c :', len(category_p_b_data))
    print('Number of feature_p_bc :', len(feature_p_bc_data))
    print(f"The files have been combined into {feature_p_b_file_path}")
    # =========== END =========== #


if __name__ == '__main__':    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        
    main()