echo "------------- step 6: Evaluation --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/test_agent.py \
    --config config/beauty/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cds/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json
# python3 src/graph_reasoning/test_agent.py \
#     --config config/clothing/graph_reasoning/UPGPR.json
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

