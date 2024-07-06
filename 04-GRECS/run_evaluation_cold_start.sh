# echo "------------- step 0: make_cold_start --------------"
# start=$(date +%s)
# echo "Start time: $(date)"
# python3 src/graph_reasoning/make_test_cold_start.py \
#     --config config/beauty/graph_reasoning/UPGPR.json --domain Beauty
# python3 src/graph_reasoning/make_test_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json --domain CDs
# python3 src/graph_reasoning/make_test_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json --domain Cellphones
# python3 src/graph_reasoning/make_test_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json --domain Clothing
# end=$(date +%s)
# echo "End time: $(date)"
# duration=$((end - start))
# echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
# echo "--------------------------------------------------------"

echo "------------- step 0.5: user preference from MCR --------------"

echo "------------- step 1: make_cold_start_kg --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/make_cold_start_kg.py \
    --config config/beauty/graph_reasoning/UPGPR.json --domain Beauty
python3 src/graph_reasoning/make_cold_start_kg.py \
    --config config/cellphones/graph_reasoning/UPGPR.json --domain Cellphones
python3 src/graph_reasoning/make_cold_start_kg.py \
    --config config/clothing/graph_reasoning/UPGPR.json --domain Clothing
python3 src/graph_reasoning/make_cold_start_kg.py \
    --config config/cds/graph_reasoning/UPGPR.json --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Evaluation cold-start --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 src/graph_reasoning/test_agent.py \
#     --config config/beauty/graph_reasoning/UPGPR.json --set_name test_cold_start
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json --set_name test_cold_start
python3 src/graph_reasoning/test_agent.py \
    --config config/clothing/graph_reasoning/UPGPR.json --set_name test_cold_start
# python3 src/graph_reasoning/test_agent.py \
#     --config config/cds/graph_reasoning/UPGPR.json --set_name test_cold_start
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

