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
echo "------------- translate + no-trim (test_cold_start) --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/beauty/graph_reasoning/UPGPR.json       --set_name test_cold_start --domain Beauty
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json   --set_name test_cold_start --domain Cellphones
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json     --set_name test_cold_start --domain Clothing
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json          --set_name test_cold_start --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- translate + other + no-trim (test_cold_start_mix) --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/beauty/graph_reasoning/UPGPR.json       --set_name test_cold_start_mix --embeds_type mix --domain Beauty
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json   --set_name test_cold_start_mix --embeds_type mix --domain Cellphones
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json     --set_name test_cold_start_mix --embeds_type mix --domain Clothing
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json          --set_name test_cold_start_mix --embeds_type mix --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- other (test_cold_start) --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/beauty/graph_reasoning/UPGPR.json       --set_name test_cold_start_past --embeds_type past --domain Beauty
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json   --set_name test_cold_start_past --embeds_type past --domain Cellphones
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json     --set_name test_cold_start_past --embeds_type past --domain Clothing
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json          --set_name test_cold_start_past --embeds_type past --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- translate + trim (test_cold_start_trim) --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/test_agent_cold_start.py \
    --config config/beauty/graph_reasoning/UPGPR.json       --set_name test_cold_start_trim --trim --domain Beauty
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json   --set_name test_cold_start_trim --trim --domain Cellphones
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json     --set_name test_cold_start_trim --trim --domain Clothing
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json          --set_name test_cold_start_trim --trim --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- translate + other + trim (test_cold_start_mix_trim) --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/beauty/graph_reasoning/UPGPR.json       --set_name test_cold_start_mix_trim --embeds_type mix --trim --domain Beauty
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cellphones/graph_reasoning/UPGPR.json   --set_name test_cold_start_mix_trim --embeds_type mix --trim --domain Cellphones
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/clothing/graph_reasoning/UPGPR.json     --set_name test_cold_start_mix_trim --embeds_type mix --trim --domain Clothing
# python3 src/graph_reasoning/test_agent_cold_start.py \
#     --config config/cds/graph_reasoning/UPGPR.json          --set_name test_cold_start_mix_trim --embeds_type mix --trim --domain CDs
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"