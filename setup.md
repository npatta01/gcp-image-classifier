# Run job

```
JOB_NAME=fashion-train-np
TRAINER_PACKAGE_PATH=gcp-image-classifier/image_trainer
MAIN_TRAINER_MODULE=train.py
JOB_DIR=hs://np-training-public-temp/$JOB_NAME
REGION=us-central1


gcloud ai-platform jobs submit training $JOB_NAME \
        --scale-tier basic \
        --package-path $TRAINER_PACKAGE_PATH \
        --module-name $MAIN_TRAINER_MODULE \
        --job-dir $JOB_DIR \
        --region $REGION \
        -- \
        --user_first_arg=first_arg_value \
        --user_second_arg=second_arg_value
```        