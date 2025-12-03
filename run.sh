# Author: Andy Phu
# imports all config key-value pairs as environment variables
set -a
. ../project.conf
set +a

gum  --reset-cache
gum  --user-name "andrew" --model "Qwen/Qwen3-VL-8B-Instruct"