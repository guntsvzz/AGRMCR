echo "------------- step 1: Preprocessing --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/preprocess/cell_phones.py \
    --config config/cell_phones/graph_reasoning/preprocess.json

# python3 src/preprocess/beauty.py \
#     --config config/beauty/graph_reasoning/preprocess.json
# python3 src/preprocess/cds.py \
#     --config config/cds/graph_reasoning/preprocess.json
# python3 src/preprocess/cellphones.py \
#     --config config/cellphones/graph_reasoning/preprocess.json
# python3 src/preprocess/clothing.py \
#     --config config/clothing/graph_reasoning/preprocess.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Make dataset --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/make_dataset.py \
    --config config/cell_phones/graph_reasoning/UPGPR.json

# python3 src/graph_reasoning/make_dataset.py \
#     --config config/beauty/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/make_dataset.py \
#     --config config/cds/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/make_dataset.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/make_dataset.py \
#     --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 3: Train KG Embedding --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/train_transe_model.py \
    --config config/cell_phones/graph_reasoning/UPGPR.json

# python3 src/graph_reasoning/train_transe_model.py \
#     --config config/beauty/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_transe_model.py \
#     --config config/cds/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_transe_model.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_transe_model.py \
#     --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 4: Train RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/train_agent.py \
    --config config/cell_phones/graph_reasoning/UPGPR.json

# python3 src/graph_reasoning/train_agent.py \
#     --config config/beauty/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_agent.py \
#     --config config/cds/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_agent.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/train_agent.py \
#     --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 5: Evaluation --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/test_agent.py \
    --config config/cell_phones/graph_reasoning/UPGPR.json

# python3 src/graph_reasoning/test_agent.py \
#     --config config/beauty/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cds/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/clothing/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/coco/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

