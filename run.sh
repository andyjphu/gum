# Author: Andy Phu
# imports all config key-value pairs as environment variables


curl -d "gum started on grace" ntfy.sh/andy-phu-jobs

set -a
. ../project.conf
set +a

# consider running pip install -e .; if gum dnepip install -e .

# also consider:
# sudo mkdir -p /workspaces/ws/data/gum
# sudo chown -R $(whoami) /workspaces/ws/data

gum  --reset-cache
gum  --user-name "andrew" --model "Qwen/Qwen3-VL-8B-Instruct"



curl -d "gum finished on grace" ntfy.sh/andy-phu-jobs


# may run on lab server for long inference with nohup ./run.sh &
# discard logs: nohup ./run.sh 2>&1 | tail -n 1000 > gum.log &

# show logs
# nohup sh -c './run.sh 2>&1 | tail -n 1000 > gum.log' &


# kill with pkill -f "gum --user-name"
# ps aux | grep -v grep | grep gum