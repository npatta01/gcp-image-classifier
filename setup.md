# download
```
git clone https://github.com/npatta01/gcp-image-classifier
mkdir -p data
wget https://storage.googleapis.com/np-training-demo-public/datsets/fashion-product-images-small.zip

unzip fashion-product-images-small.zip
```


# Run job

Serverless training
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




Bigquery ML
https://cloud.google.com/bigquery-ml/docs/making-predictions-with-imported-tensorflow-models

associate tf model
```
user=nup0013

CREATE OR REPLACE MODEL $user.imported_tf_model
   OPTIONS (MODEL_TYPE='TENSORFLOW',
    MODEL_PATH='gs://cloud-training-demos/txtclass/export/exporter/1549825580/*')
    
```    

make predictions
```
SELECT *
   FROM ML.PREDICT(MODEL nup0013.imported_tf_model,
     (
      SELECT title AS input
      FROM `bigquery-public-data.hacker_news.stories`
     )
 )
```


