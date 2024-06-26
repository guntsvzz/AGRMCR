import sys
import json
from tqdm.auto import tqdm
from collections import OrderedDict
import os
import gzip

def main():
    # Initialize your maps
    item_map = {}
    cate_map = {}
    brand_map = {}
    item_cate = OrderedDict()
    item_brand = OrderedDict()
    item_also_bought = OrderedDict()
    item_also_viewed = OrderedDict()
    item_bought_together = OrderedDict()
    related_product_set = set()
    related_product_map = {}

    # Read product mappings
    with open(f'./data/{dataset_name}/product.txt', 'r') as f:
        i = 0
        for line in tqdm(f):
            conts = line.strip()
            item_map[conts] = i
            i += 1
    print('The number of product', len(item_map))

    # Read categories, brands, and related products, then map them to products
    with gzip.open(f'./raw_data/meta_{dataset_name}.json.gz', 'rt', encoding='utf-8') as f:
    # with open(f'./raw_data/meta_{dataset_name}.json', 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            r = eval(line.strip())
            iid = r['asin']
            cates = r['categories']
            brand = r.get('brand', 'UNK')
            related = r.get('related', {})
            also_bought = related.get('also_bought', [])
            also_viewed = related.get('also_viewed', [])
            bought_together = related.get('bought_together', [])

            if iid not in item_map:
                continue

            product_idx = item_map[iid]

            # Process categories
            if product_idx not in item_cate:
                item_cate[product_idx] = []
            for cate_list in cates:
                for cate in cate_list:
                    if cate not in cate_map:
                        cate_map[cate] = len(cate_map)
                    cate_idx = cate_map[cate]
                    if cate_idx not in item_cate[product_idx]:
                        item_cate[product_idx].append(cate_idx)

            # Process brand
            if brand not in brand_map:
                brand_map[brand] = len(brand_map)
            item_brand[product_idx] = brand_map[brand]

            # Process related products
            def map_related_products(related_list):
                related_product_set.update(related_list)

            map_related_products(also_bought)
            map_related_products(also_viewed)
            map_related_products(bought_together)
    
    assert len(item_brand) == len(item_cate), "brand_p_b is not equal to category_p_c"
    
    # Sort the dictionaries by keys
    item_cate = OrderedDict(sorted(item_cate.items()))
    item_brand = OrderedDict(sorted(item_brand.items()))
    assert len(item_brand) == len(item_cate), "brand_p_b is not equal to category_p_c"
    
    print("Number of Category :",len(cate_map))
    print("Number of Brand :",len(brand_map))
    print("Number of Item :",len(item_cate))
    
    print("Number of Item-Category :",len(item_brand))
    print("Number of Item-Brand :",len(item_cate))

    brand_map = {v: k for k, v in brand_map.items()}
    cate_map = {v: k for k, v in cate_map.items()}

    write_values_to_file(item_brand, 'brand_p_b.txt')
    write_values_to_file(item_cate, 'category_p_c.txt')
    write_values_to_file(brand_map, 'brand.txt')
    write_values_to_file(cate_map, 'category.txt')
    write_values_to_file(sorted(related_product_set), 'related_product.txt', newline=True)

    # Create a mapping for related products
    for idx, original_id in enumerate(sorted(related_product_set)):
        related_product_map[original_id] = idx
    print('Number of Related products :', len(related_product_map))

    # Read categories, brands, and related products, then map them to products
    with gzip.open(f'./raw_data/meta_{dataset_name}.json.gz', 'rt', encoding='utf-8') as f:
    # with open(f'./raw_data/meta_{dataset_name}.json', 'r') as f:
        for idx, line in enumerate(tqdm(f)):
            r = eval(line.strip())
            iid = r['asin']
            cates = r['categories']
            brand = r.get('brand', None)
            related = r.get('related', {})
            also_bought = related.get('also_bought', [])
            also_viewed = related.get('also_viewed', [])
            bought_together = related.get('bought_together', [])
            
            if iid not in item_map:
                continue

            product_idx = item_map[iid]
        
            # Process related products
            def p_related_p(related_list):
                return [related_product_map[rel_iid] for rel_iid in related_list if rel_iid in related_product_map]

            item_also_bought[product_idx] = p_related_p(also_bought)
            item_also_viewed[product_idx] = p_related_p(also_viewed)
            item_bought_together[product_idx] = p_related_p(bought_together)

    item_also_bought = OrderedDict(sorted(item_also_bought.items()))
    item_also_viewed = OrderedDict(sorted(item_also_viewed.items()))
    item_bought_together = OrderedDict(sorted(item_bought_together.items()))
    
    print("Number of item_also_bought :",len(item_also_bought))
    print("Number of item_also_viewed :",len(item_also_viewed))
    print("Number of item_bought_together :",len(item_bought_together))
    
    write_values_to_file(item_also_bought, 'also_bought_p_p.txt')
    write_values_to_file(item_also_viewed, 'also_viewed_p_p.txt')
    write_values_to_file(item_bought_together, 'bought_together_p_p.txt')

# Function to write values to file
def write_values_to_file(data, filename, newline=False):
    output_dir = f'./data/{dataset_name.lower()}'  # Define the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory if it doesn't exist
    filepath = os.path.join(output_dir, filename)  # Construct the full file path
    
    with open(filepath, 'w') as f:
        if isinstance(data, dict):
            for key, values in data.items():
                if isinstance(values, list):
                    f.write(f"{' '.join(map(str, values))}\n")
                else:  # If values is a single integer
                    f.write(f"{values}\n")
        elif isinstance(data, list):
            if newline:
                for item in data:
                    f.write(f"{item}\n")
            else:
                f.write(' '.join(map(str, data)) + '\n')
                
if __name__ == '__main__':    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        
    main()

