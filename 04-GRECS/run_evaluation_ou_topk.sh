CONFIG=config/beauty/graph_reasoning/UPGPR.json
DOMAIN=Beauty
SET_NAME=test_cold_start_mix_trim

echo "------------- (${SET_NAME}) --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 src/graph_reasoning/test_agent_cold_start.py \
    --config ${CONFIG} \
    --set_name ${SET_NAME} \
    --embeds_type mix \
    --trim --domain ${DOMAIN} \
    --topk_candidate 1 \
    --preference both
python3 src/graph_reasoning/test_agent_cold_start.py \
    --config ${CONFIG} \
    --set_name ${SET_NAME} \
    --embeds_type mix \
    --trim --domain ${DOMAIN} \
    --topk_candidate 3 \
    --preference both
python3 src/graph_reasoning/test_agent_cold_start.py \
    --config ${CONFIG} \
    --set_name ${SET_NAME} \
    --embeds_type mix \
    --trim --domain ${DOMAIN} \
    --topk_candidate 5 \
    --preference both
python3 src/graph_reasoning/test_agent_cold_start.py \
    --config ${CONFIG} \
    --set_name ${SET_NAME} \
    --embeds_type mix \
    --trim --domain ${DOMAIN} \
    --topk_candidate 10 \
    --preference both
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"