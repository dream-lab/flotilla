EXP_NAME=./config/lenet_fedavg_iid.yaml
# requiers password less ssh https://www.strongdm.com/blog/ssh-passwordless-login
# python3 flo_session.py config/$EXP_NAME

AGG=$(cat $EXP_NAME | grep "aggregator:" -m 2 | tail -1 | cut -d ":" -f "2" | tr -d " ")
CLIENT_SELECT=$(cat $EXP_NAME | grep "client_selection:" -m 1 | cut -d ":" -f "2" | tr -d " ")_
MODEL=$(cat $EXP_NAME | grep "model_id:" -m 2 | tail -1 | cut -d ":" -f "2" | tr -d " ")_
DATA=$(cat $EXP_NAME | grep "dataset:" -m 2 | tail -1 | cut -d ":" -f "2" | tr -d " ")_
OLD_LOG_FILE=$(ls -lt ./logs/ | head -2 | tail -1 | cut -d " " -f 10)
NEW_NAME="log_$MODEL$DATA$CLIENT_SELECT$AGG.log"

mv "./logs/$OLD_LOG_FILE" "./logs/$NEW_NAME"
scp -r "./logs/$NEW_NAME" roopkathab@10.24.24.14:finished_runs/.
