dataset=BlogCatalog
num_hidden=128
batch_size=128
fan_out=5,10
gamma=0.002

python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma}
