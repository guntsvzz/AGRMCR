EPOCHS_RL=1
MIN_EPOCHS_RL=$((${EPOCHS_RL} - 1))
echo "------------- step 5: Train RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/train_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json --epochs_rl ${EPOCHS_RL} --min_epochs_rl ${MIN_EPOCHS_RL}
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
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"
