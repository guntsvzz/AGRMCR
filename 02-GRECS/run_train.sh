EPOCHS=30
MIN_EPOCHS=$((${EPOCHS} - 1))
echo "------------- step 3: Train KG Embedding --------------"
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

echo "------------- step 4: Create review_dict --------------"
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

EPOCHS_RL=50
MIN_EPOCHS_RL=$((${EPOCHS_RL} - 1))
echo "------------- step 5: Train RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/train_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json --epochs_rl ${EPOCHS_RL} --min_epochs_rl ${MIN_EPOCHS_RL}
python3 src/graph_reasoning/train_agent.py \
    --config config/cds/graph_reasoning/UPGPR.json --epochs_rl ${EPOCHS_RL} --min_epochs_rl ${MIN_EPOCHS_RL}
python3 src/graph_reasoning/train_agent.py \
    --config config/cellphones/graph_reasoning/UPGPR.json --epochs_rl ${EPOCHS_RL} --min_epochs_rl ${MIN_EPOCHS_RL}
python3 src/graph_reasoning/train_agent.py \
    --config config/clothing/graph_reasoning/UPGPR.json --epochs_rl ${EPOCHS_RL} --min_epochs_rl ${MIN_EPOCHS_RL}
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 6: Evaluation --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/test_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/cds/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/cellphones/graph_reasoning/UPGPR.json
python3 src/graph_reasoning/test_agent.py \
    --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

