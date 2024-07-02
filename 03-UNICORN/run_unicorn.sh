echo "------------- step 0: TransE Embedding --------------"
echo "It was trained by GRECS"
echo "--------------------------------------------------------"

MAX_STEPS=10
SAMPLE_TIMES=1000
#Episode = max_steps * sample_times
echo "------------- step 1: Training RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 RL_model.py --data_name BEAUTY      --domain Beauty     --max_steps ${MAX_STEPS} --sample_times ${SAMPLE_TIMES} --embed transe
python3 RL_model.py --data_name CELLPHONES  --domain Cellphones --max_steps ${MAX_STEPS} --sample_times ${SAMPLE_TIMES} --embed transe
python3 RL_model.py --data_name CLOTH       --domain Cloth      --max_steps ${MAX_STEPS} --sample_times ${SAMPLE_TIMES} --embed transe
python3 RL_model.py --data_name CDS         --domain CDs        --max_steps ${MAX_STEPS} --sample_times ${SAMPLE_TIMES} --embed transe
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Evaluation RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 evaluate.py --data_name BEAUTY      --domain Beauty     --load_rl_epoch ${MAX_STEPS} --embed transe
python3 evaluate.py --data_name CELLPHONES  --domain Cellphones --load_rl_epoch ${MAX_STEPS} --embed transe
python3 evaluate.py --data_name CLOTH       --domain Cloth      --load_rl_epoch ${MAX_STEPS} --embed transe
python3 evaluate.py --data_name CDS         --domain CDs        --load_rl_epoch ${MAX_STEPS} --embed transe
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "----------------------------------------------------------"
