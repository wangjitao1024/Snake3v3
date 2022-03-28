nohup python -u actor.py --num_replicas=20 --data_port=5002 --param_port=5003 > ./log/actor.log &
nohup python -u learner.py --pool_size=16384 --batch_size=16384 --data_port=5002 --param_port=5003 > ./log/learner.log &
