echo "------------- step 0: TransE Embedding --------------"
echo "It was trained by GRECS"
echo "--------------------------------------------------------"

# max_steps==train_step & sample_times=episode
echo "------------- step 1: Training RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 RL_model.py --data_name AMAZON --domain Appliances --max_steps 10 --sample_times 1 
python3 RL_model.py \
    --data_name BEAUTY --domain Beauty --max_steps 10 --sample_times 1 --embed transe
python3 RL_model.py \
    --data_name CELLPHONES --domain Cellphones --max_steps 10 --sample_times 1 --embed transe
python3 RL_model.py \
    --data_name CLOTH --domain Cloth --max_steps 10 --sample_times 1 --embed transe
python3 RL_model.py \
    --data_name CDS --domain CDs --max_steps 10 --sample_times 1 --embed transe
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Evaluation RL Agent --------------"
start=$(date +%s)
echo "Start time: $(date)"
# python3 evaluate.py --data_name AMAZON --domain Appliances --load_rl_epoch 10
python3 evaluate.py \
    --data_name BEAUTY --domain Beauty --load_rl_epoch 10 --embed transe
python3 evaluate.py \
    --data_name CELLPHONES --domain Cellphones --load_rl_epoch 10 --embed transe
python3 evaluate.py \
    --data_name CLOTH --domain Cloth --load_rl_epoch 10 --embed transe
python3 evaluate.py \
    --data_name CDS --domain CDs --load_rl_epoch 10 --embed transe
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "----------------------------------------------------------"
