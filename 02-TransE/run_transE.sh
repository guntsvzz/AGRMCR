echo "------------- step 1: Preprocessing --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/preprocess/preprocess_domain.py \
    --config config/beauty/graph_reasoning/preprocess.json
python3 src/preprocess/preprocess_domain.py \
    --config config/cds/graph_reasoning/preprocess.json
python3 src/preprocess/preprocess_domain.py \
    --config config/cellphones/graph_reasoning/preprocess.json
python3 src/preprocess/preprocess_domain.py \
    --config config/clothing/graph_reasoning/preprocess.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Make dataset --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/make_dataset.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/make_dataset.py \
    --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 3: Create review_dict --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/dataset_json.py Beauty
python3 src/graph_reasoning/dataset_json.py Cellphones
python3 src/graph_reasoning/dataset_json.py Clothing
python3 src/graph_reasoning/dataset_json.py CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

EPOCHS=30
MIN_EPOCHS=$((${EPOCHS} - 1))
echo "------------- step 4: Train KG Embedding --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/train_transe_model.py \
    --config config/beauty/graph_reasoning/UPGPR.json --epochs ${EPOCHS} --min_epochs ${MIN_EPOCHS}
python3 src/graph_reasoning/train_transe_model.py \
    --config config/cds/graph_reasoning/UPGPR.json --epochs ${EPOCHS} --min_epochs ${MIN_EPOCHS}
python3 src/graph_reasoning/train_transe_model.py \
    --config config/cellphones/graph_reasoning/UPGPR.json --epochs ${EPOCHS} --min_epochs ${MIN_EPOCHS}
python3 src/graph_reasoning/train_transe_model.py \
    --config config/clothing/graph_reasoning/UPGPR.json --epochs ${EPOCHS} --min_epochs ${MIN_EPOCHS}
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"