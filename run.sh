# Author: Andy Phu
# imports all config key-value pairs as environment variables
set -a
. ../project.conf
set +a

gum  --reset-cache
gum  --user-name "andrew" --model "Qwen/Qwen3-VL-8B-Instruct"


# may run on lab server for long inference with nohup ./run.sh &
# discard logs: nohup ./run.sh 2>&1 | tail -n 1000 > gum.log &


# kill with pkill -f "gum --user-name"
# ps aux | grep -v grep | grep gum