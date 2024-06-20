
Download [Metadata Amazon dataset 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
Download [Rating Amazon dataset 2018](https://nijianmo.github.io/amazon/index.html)

Following ScomGNN, filter categories more than 4 categories & pricing is not empty
Here's the updated table with the new columns added:

| Category             | Reviews              | Metadata             | Filter Metadata      | Filter Usesr | Type | Instance | Brand |
|----------------------|----------------------|----------------------|----------------------|--------------|------|----------|-------|
| Appliances           | 602,777 reviews      | 30,459 products      | 804 products         | xx           | xx   | xx       | xx    |
| Electronics          | 20,994,353 reviews   | 786,868 products     | 46,611 products      | xx           | xx   | xx       | xx    |
| Grocery and Gourmet Food| 5,074,160 reviews | 287,209 products     | 38,548 products      | xx           | xx   | xx       | xx    |
| Home and Kitchen     | 21,928,568 reviews   | 1,301,225 products   | 75,514 products      | xx           | xx   | xx       | xx    |
| Office Products      | 5,581,313 reviews    | 315,644 products     | 42,785 products      | xx           | xx   | xx       | xx    |
| Sports and Outdoors  | 12,980,837 reviews   | 962,876 products     | 87,076 products      | xx           | xx   | xx       | xx    |

|                   | Appliances | Electronics | Grocery and Gourmet Food | Home and Kitchen | Office Products | Sports and Outdoors |
|-------------------|------------|-------------|--------------------------|------------------|-----------------|---------------------|
| **#Users**        | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Items**        | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Interactions** | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Attributes**   | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Entities**     | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Relations**    | xx         | xx          | xx                       | xx               | xx              | xx                  |
| **#Triplets**     | xx         | xx          | xx                       | xx               | xx              | xx                  |

Head    -> Relation         -> Tail
1. USER -> INTERACT         -> ITEM
2. ITEM -> ALSO_BUY         -> ITEM 
3. ITEM -> ALSO_VIEW        -> ITEM
4. ITEM -> BELONG_TO        -> FEATURE(TYPE)
5. ITEM -> PRODUCE_BY       -> FEATURE(BRAND)
6. ITEM -> DESCRIBED_BY     -> FEATURE(FUNCTION)
7. TYPE -> BELONG_TO_LARGE  -> CATEGORIES


## GRAPH FORMAT
```
user_dict.json
{
    "0" : {
        "friend" : [],
        "like" : [],
        "interact" [item(idx), item(idx),...]
    },
    "1" : {
        "friend" : [],
        "like" : [],
        "interact" []
    },
    ...
}
```

```
item_dict.json
{
    "0" : {
        "categories" : [int, int, int],
        "brand" : int,
        "feature_index" [int, int, int]
        "asin" : str
    },
    "1" : {
        "categories" : [int, int, int],
        "brand" : int,
        "feature_index" [int, int, int]
        "asin" : str
    },
    ...
}
```

```
feature_dict.json
{
    "0" : {
        "link_to_feature" : [],
        "like" : [],
        "belong_to" [item(idx), item(idx),...]
    },
    "1" : {
        "link_to_feature" : [],
        "like" : [],
        "belong_to" [item(idx), item(idx),...]
    },
    ...
}
```

```
first-layer_merged_tag_map.json
{
    "categories#1" : 0,
    "categories#2" : 0,
    ...
}
```

```
second-layer_oringinal_tag_map.json
{
    "type#1" : 0,
    "type#2" : 0,
    ...
}
```

```
2-layer taxonomy.json
{
    "Categories#1": [TypeA, TypeB, TypeC],
    "Categories#2": [TypeD, TypeE, TypeF],
}
```