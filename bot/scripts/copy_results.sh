# this script rsyncs the results from ray but not the checkpoints. you need to copy those you want manually

# conig
USR_LINUX="fsoulier"
USR_MAC="florian"
SERVER_IP="139.6.237.137"

# the following does not work, fuck it.. cyka
# show help
#if  [ "$S1"=="--help" ] || [ "$S1"=="-h" ] ; then
#echo "this script rsyncs the results from ray but not the checkpoints. you need to copy those you want manually"
#echo "usage: copy_results EXPERIMENT_ID SOURCE_FOLDER TARGET_FOLDER"
#echo "u may leave SOURCE_FOLDER and TARGET_FOLDER blank to get following result:"
#echo "usage: copy_results MISSING_EXP_ID_$(date +%Y_%m_%d_%H%M%S) $SOURCE $TARGET"
#exit 0
#fi


# get input variables
EXPERIMENT_ID=$1
if [ -z "$EXPERIMENT_ID" ]; then
EXPERIMENT_ID="MISSING_EXP_ID_$(date +%Y_%m_%d_%H%M%S)"
fi

SOURCE=$2
if [ -z "$SOURCE" ]; then
SOURCE="/home/$USR_LINUX/ray_results/"
fi

TARGET=$3
if [ -z "$TARGET" ]; then
TARGET="/Users/$USR_MAC/ray_results/$EXPERIMENT_ID/"
fi

echo $SOURCE
echo $TARGET
# for debugging:
#rsync -avz --exclude "checkpoint*" "$SOURCE/" "$TARGET/$EXPERIMENT_ID/"
# for real
rsync -chavzP --stats --exclude "checkpoint*" "$USR_LINUX@$SERVER_IP:$SOURCE" $TARGET

echo "ATTENTION: Do not forget to scopy the latest checkpoints"
echo "use: ls -Rltr | grep checkpoint"
echo "to find the latest checkpoints in the subfolders"
echo "use: scp -r $USR_LINUX@$SERVER_IP:${SOURCE}experimentId/checkpoint_XXX ${TARGET}experimentId/"
echo "to copy the latest checkpoints u wish to save"