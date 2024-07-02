
MAX_STEPS=10
echo "------------- step 2: Evaluation RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 evaluate.py --data_name BEAUTY      --domain Beauty     --load_rl_epoch ${MAX_STEPS} --embed transe
# python3 evaluate.py --data_name CELLPHONES  --domain Cellphones --load_rl_epoch ${MAX_STEPS} --embed transe
# python3 evaluate.py --data_name CLOTH       --domain Cloth      --load_rl_epoch ${MAX_STEPS} --embed transe
# python3 evaluate.py --data_name CDS         --domain CDs        --load_rl_epoch ${MAX_STEPS} --embed transe
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "----------------------------------------------------------"
