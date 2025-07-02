args="
--log
--gpu 6,7
--method debug_distribution
--expr_name pre_norm,RMSNorm,distributed

--dataset cifar10
--agent AgentClassification
--trainer TrainerCommon

--evaluate_first
--main_metric accuracy

--lr 1e-4
--train_batch_size 16
--iteration_num 100000
--lr_scheduler cosine
--warmup_ratio 0.02

--eval_batch_size 128
--train_log_every 400
--eval_every 1000
"
# add --evaluate_first and set iteration_num 0 for test only

python src/main.py $args
