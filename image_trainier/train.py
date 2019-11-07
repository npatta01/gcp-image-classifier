import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import tensorflow_hub as hub
import argparse
import logging
import subprocess



def dowload_data(location):
    return location


def parse_data(flags):

    location = flags.input
    min_images= flags.min_images
    
    DATASET_PATH=flags.scratch
    subprocess.call("gsutil -m cp -r %s %s".format(location,DATASET_PATH), shell=True)

    
    DATASET_PATH_IMAGES="%s/images".format(DATASET_PATH)
    
    
    
    df = pd.read_csv(DATASET_PATH+"/styles.csv",  error_bad_lines=False)

    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)


    df_counts = df['subCategory'].value_counts()
    df_counts


    top_classes = df_counts[df_counts>min_images].index.tolist()
    top_classes


    df = df[df['subCategory'].isin(top_classes)]
    df

    df = df.groupby('subCategory').apply(pd.DataFrame.sample, n=min_images).reset_index(drop=True)
    return df



def train_model(df, flags):

    IMAGE_SHAPE = (224, 224)

    image_generator = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest" 

    )

    training_generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=DATASET_PATH + "/images",
        x_col="image",
        y_col="subCategory",
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        subset="training"
    )

    validation_generator = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=DATASET_PATH + "/images",
        x_col="image",
        y_col="subCategory",
        target_size=IMAGE_SHAPE,
        batch_size=batch_size,
        subset="validation"
    )

    classes = len(training_generator.class_indices)

 

    # Add Layer Embedding
    classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4" #@param {type:"string"}


    base_model = hub.KerasLayer(classifier_url,  input_shape = IMAGE_SHAPE+(3,), trainable=False)

    model = tf.keras.Sequential([
        base_model
        ,tf.keras.layers.Dense(classes, activation='softmax')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()




    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        epochs=5,
        verbose=1,
        workers=8,
        #use_multiprocessing=True
    )



    model.save('model.h5')




def parse_args(argv):
"""Parses command-line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--input',
      help='''location to use in gcs.
            ''',
      required=True,
    )

    parser.add_argument(
      '--scratch',
      help='''location to use locally
            ''',
      default="/tmp"  
      required=True,
    )
    
    
    parser.add_argument(
      '--save-dir',
      help='Location in gcs',
      required=True,
    )

    
    parser.add_argument(
      '--min-images',
        default=300,
      help='number of images in sample',
      required=True,
    )
    
    
    return parser.parse_args(argv)


  parser.add_argument(
      '--log_level',
      help='Logging level.',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN',
      ],
      default='INFO',
  )



def main():
  """Entry point."""

    flags = parse_args(sys.argv[1:])
    logging.basicConfig(level=flags.log_level.upper())
    
    df = parse_data(flags)
    
    
    train_model(df, flags)


if __name__ == '__main__':
    main()