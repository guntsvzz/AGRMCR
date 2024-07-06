
MAX_STEPS=10
echo "------------- step 1: Evaluation RL Agent on cold_start_user --------------"
start=$(date +%s)
echo "Start time: $(date)"
echo "------------- Beauty -------------------------"
python3 evaluate.py --data_name BEAUTY      --domain Beauty     --load_rl_epoch ${MAX_STEPS} --embed transe --mode test_cold_start
echo "------------- Cellphones ---------------------"
python3 evaluate.py --data_name CELLPHONES  --domain Cellphones --load_rl_epoch ${MAX_STEPS} --embed transe --mode test_cold_start
echo "------------- Clothing------------------------"
python3 evaluate.py --data_name CLOTH       --domain Clothing   --load_rl_epoch ${MAX_STEPS} --embed transe --mode test_cold_start
echo "------------- CDs ----------------------------"
python3 evaluate.py --data_name CDS         --domain CDs        --load_rl_epoch ${MAX_STEPS} --embed transe --mode test_cold_start
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "----------------------------------------------------------"

