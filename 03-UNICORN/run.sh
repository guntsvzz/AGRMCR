# echo "------------- step 0: TransE Embedding --------------"
# start=$(date +%s)
# echo "Start time: $(date)"
# # python3 RL_model.py --data_name LAST_FM --epochs 1000
# end=$(date +%s)
# echo "End time: $(date)"
# duration=$((end - start))
# echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
# echo "--------------------------------------------------------"


## max_steps==train_step & sample_times=episode
echo "------------- step 1: Training --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 RL_model.py --data_name AMAZON --domain Appliances --max_steps 10 --sample_times 1 
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"

echo "------------- step 2: Evaluation --------------"
start=$(date +%s)
echo "Start time: $(date)"
python3 evaluate.py --data_name AMAZON --domain Appliances --load_rl_epoch 10
end=$(date +%s)
echo "End time: $(date)"
duration=$((end - start))
echo "Duration: $(($duration / 3600)) hr $((($duration % 3600) / 60)) min $(($duration % 60)) sec"
echo "--------------------------------------------------------"
