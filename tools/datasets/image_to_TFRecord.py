#!/usr/bin/env python3
'''
Filename: /home/zhihua/ml/tpu/tools/datasets/to_TFRecord.py
Path: /home/zhihua/ml/tpu/tools/datasets
Created Date: Thursday, December 26th 2019, 4:34:21 pm
Author: Saibaster

Copyright (c) 2019 Muxiv
'''
r"""Script to transfer images and labels to TFRecord format.

Make sure you have around enough of disc space available on the machine where
you're running this script. You can run the script using the following command.
```
python to_TFRecord.py \
  --csv_file=data_and_labels.csv
  --local_scratch_dir='./tfRecord' \  
  --project="TEST_PROJECT" \
  --gcs_output_path="gs://TEST_BUCKET/IMAGENET_DIR" \
  --gcs_upload="True" \
  --shards=1024 \

for example:
     GOOGLE_APPLICATION_CREDENTIALS=/home/zhihua/data/gcs.json ./image_to_TFRecord.py --csv_file=/home/zhihua/test/data/test.csv --local_scratch_dir=/home/zhihua/test/data/tfRecord --project=apt-footing-256512 --shards=16
```
"""


import math
import os
import random
import tarfile
import urllib
import logging
from absl import app
from absl import flags
import tensorflow as tf
from google.cloud import storage
flags.DEFINE_string(
    'project', None, 'Google cloud project id for uploading the dataset.')
flags.DEFINE_boolean(
    'gcs_upload', False, 'Set to false to not upload to gcs.')    
flags.DEFINE_string(
    'gcs_output_path', None, 'GCS path for uploading the dataset.')
flags.DEFINE_string(
    'csv_file', 'data.csv', 'csv file that contains files list and labels.')    
flags.DEFINE_string(
    'local_scratch_dir', 'tfRecord', 'Scratch directory path for output files on local.')
flags.DEFINE_integer(
    'shards', 1024, 'number of chucks to split the training files into')
flags.DEFINE_integer(
    'channels', 1, 'The channels dimension of image, for grayscale, it is 1, for rgb, it is 3.')
flags.DEFINE_string(
    'colorspace', 'grayscale', 'used to override the color format of the encoded output. Values can be: grayscale, rgb')
flags.DEFINE_string(
    'image_format', 'png', 'image format, png, jpeg, mat, ...')

FLAGS = flags.FLAGS


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: anytype, identifier for the ground truth for the network
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    features = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(FLAGS.colorspace.encode("utf8")),
        'image/channels': _int64_feature(FLAGS.channels),
        # 'image/class/label': _int64_feature(label),
        'image/format': _bytes_feature(FLAGS.image_format.encode("utf8")),
        'image/filename': _bytes_feature(os.path.basename(filename[list(filename.keys())[0]])),
        'image/encoded': _bytes_feature(image_buffer)
        }
    print('----------')
    print(filename)
    print(label)
    print(height, width)
    for key in label.keys():
        features['label/' + key] = _float_feature(label[key])
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    # def __init__(self):
    #     # Create a single Session to run all image coding calls.
    #     # self._sess = tf.compat.v1.Session()

    #     # Initializes function that decodes RGB JPEG data.
    #     self._decode_jpeg_data = tf.Variable(tf.string, name='_decode_jpeg_data')
    #     self._decode_jpeg = tf.image.decode_jpeg(
    #         self._decode_jpeg_data, channels=FLAGS.channels)

    #     # Initializes function that decodes RGB PNG data.
    #     self._decode_png_data = tf.Variable(tf.string, name='_decode_png_data')
    #     self._decode_png = tf.image.decode_png(
    #         self._decode_png_data, channels=FLAGS.channels)

    def decode_jpeg(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=FLAGS.channels)
        assert len(image.shape) == 3
        assert image.shape[2] == FLAGS.channels
        return image


    def decode_png(self, image_data):
        image = tf.image.decode_png(image_data, channels=FLAGS.channels)
        assert len(image.shape) == 3
        assert image.shape[2] == FLAGS.channels
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG or png encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.io.gfile.GFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    if FLAGS.image_format == 'jpeg' or FLAGS.image_format == 'jpg':
        image = coder.decode_jpeg(image_data)
    elif FLAGS.image_format == 'png':
        image = coder.decode_png(image_data)
    else:
        print('Image Format must be set through --image_format')

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == FLAGS.channels

    return image_data, height, width


def _process_image_files_batch(coder, output_file, filenames, labels):
    """Processes and saves list of images as TFRecords.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      output_file: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      labels: list of labels; 
    """
    writer = tf.io.TFRecordWriter(output_file)

    for i in range(len(filenames[list(filenames.keys())[0]])):
        filename = {}
        label = {}
        for key in filenames.keys():
            filename[key] = filenames[key][i]
        for key in labels.keys():
            label[key] = labels[key][i]            
        image_buffer, height, width = _process_image(filename[list(filename.keys())[0]], coder)
        example = _convert_to_example(filename, image_buffer, label,
                                      height, width)
        writer.write(example.SerializeToString())

    writer.close()


def _process_dataset(filenames, labels):
    """Processes and saves list of images as TFRecords.

    Args:
      filenames: list of strings; each string is a path to an image file
      labels: list of strings, integer, float...; id for all data points

    Returns:
      files: list of tf-record filepaths created from processing the dataset.
    """
    _check_or_create_dir(FLAGS.local_scratch_dir)
    chunksize = int(math.ceil(len(filenames[list(filenames.keys())[0]]) / FLAGS.shards))
    coder = ImageCoder()

    files = []

    for shard in range(FLAGS.shards):
        chunk_files = {}
        chunk_labels = {}
        for key in filenames.keys():
            chunk_files[key] = filenames[key][shard * chunksize: (shard + 1) * chunksize]
        for key in labels.keys():
            chunk_labels[key] = labels[key][shard * chunksize: (shard + 1) * chunksize]            
        output_file = os.path.join(
            FLAGS.local_scratch_dir, '%.5d-of-%.5d' % (shard, FLAGS.shards))
        _process_image_files_batch(coder, output_file, chunk_files,
                                   chunk_labels)
        logging.info('Finished writing file: %s' % output_file)
        files.append(output_file)
    return files


def read_dataset():
    """read data set from csv file
    
    Returns:
        tf dataset 
        -- A dataset, where each element is a (features, labels) tuple that corresponds to a batch of batch_size CSV rows. 
        The features dictionary maps feature column names to Tensors containing the corresponding column data, 
        and labels is a Tensor containing the column data for the label column specified by label_name.
    """
    # Get files and labels list from csv file
    return tf.data.experimental.make_csv_dataset(
        FLAGS.csv_file,
        batch_size=1024,
        num_epochs=1,
        )


def convert_to_tf_records():
    """Convert the dataset into TF-Record dumps."""
    # get file list and labels from csv file
    csv_data = read_dataset()
    # get filenames and labels.
    keys = list(next(csv_data.as_numpy_iterator()).keys())
    logging.info('file name and labels are: %s' % str(keys))
    files = {}
    labels = {}
    files[keys[0]] = []
    for key in keys[1:]:
        labels[key] = []
    for item in csv_data.as_numpy_iterator():        
        files[keys[0]] += item[keys[0]].tolist()
        for key in keys[1:]:
            labels[key] += item[key].tolist()
    logging.info('we have %s files' % len(files[keys[0]]))
    # Create TFRecord data
    logging.info('Processing data.')
    tf_records = _process_dataset(files, labels)

    return tf_records


def upload_to_gcs(tf_records):
    """Upload TF-Record files to GCS, at provided path."""

    # Find the GCS bucket_name and key_prefix for dataset files
    path_parts = FLAGS.gcs_output_path[5:].split('/', 1)
    bucket_name = path_parts[0]
    if len(path_parts) == 1:
        key_prefix = ''
    elif path_parts[1].endswith('/'):
        key_prefix = path_parts[1]
    else:
        key_prefix = path_parts[1] + '/'

    client = storage.Client(project=FLAGS.project)
    bucket = client.get_bucket(bucket_name)

    def _upload_files(filenames):
        """Upload a list of files into a specifc subdirectory."""
        for i, filename in enumerate(sorted(filenames)):
            blob = bucket.blob(key_prefix + os.path.basename(filename))
            blob.upload_from_filename(filename)
            if not i % 20:
                logging.info('Finished uploading file: %s' % filename)

    # Upload dataset
    logging.info('Uploading the data.')
    _upload_files(tf_records)


def main(argv):  # pylint: disable=unused-argument
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)

    if FLAGS.gcs_upload and FLAGS.project is None:
        raise ValueError('GCS Project must be provided.')

    if FLAGS.gcs_upload and FLAGS.gcs_output_path is None:
        raise ValueError('GCS output path must be provided.')
    elif FLAGS.gcs_upload and not FLAGS.gcs_output_path.startswith('gs://'):
        raise ValueError('GCS output path must start with gs://')

    # Convert the raw data into tf-records
    tf_records = convert_to_tf_records()

    # Upload to GCS
    if FLAGS.gcs_upload:
        upload_to_gcs(tf_records)


if __name__ == '__main__':
    app.run(main)
