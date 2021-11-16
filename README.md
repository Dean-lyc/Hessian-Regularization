# Hessian-Regularization



RUN 4 GPU data parallel:

```bash
CUDA_VISIBLE_DEVICE="0,1,2,3" python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 5401 run.py \
--batch_size 32 \
--lr 0.01 \
--epochs 200 \
--weight_decay 5e-4 \
--lambda_JR 0.1 \
--Hiter 5 \
--prob 1 \
--add_noise 1 \
--noise_std 1 \
--hess_interval 10 \
```





RUN with 1 GPU data parallel

```bash
python run.py \
--local_rank -1 \
--batch_size 32 \
--lr 0.01 \
--epochs 200 \
--weight_decay 5e-4 \
--lambda_JR 0.1 \
--Hiter 5 \
--prob 1 \
```

