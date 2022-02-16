dataset=cora
num_hidden=128
batch_size=32
fan_out=5,10
gamma=0.002

python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma} --mode full --gpu 0 &
python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma} --mode no-gnn-clf --gpu 1 &
python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma} --mode no-adj-mix --gpu 2 &
python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma} --mode no-smote --gpu 3 &
python -u ../model_acc/bench_ablationsmote_lightning.py \
  --dataset ${dataset} --num-hidden ${num_hidden} --batch-size ${batch_size}  \
  --num-workers 4 --fan-out ${fan_out} --gamma ${gamma} --mode no-enc --gpu 3
