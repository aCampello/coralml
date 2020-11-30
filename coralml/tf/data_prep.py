import numpy as np

import io
import json
import os

# Create tf_example

import cv2

import tensorflow as tf
from object_detection.utils import dataset_util

from coralml.data.create_masks import parse_csv_data_file


flags = tf.app.flags
flags.DEFINE_string('output_path', 'data/coral_train-00000', 'Path to output TFRecord')
flags.DEFINE_string('annotations_path', 'data/annotations_test.csv',
                    'Path to coral reef annotations file')
FLAGS = flags.FLAGS


class_map = {
    'c_algae_macro_or_leaves': 1,
    'c_fire_coral_millepora': 2,
    'c_hard_coral_boulder': 3,
    'c_hard_coral_branching': 4,
    'c_hard_coral_encrusting': 5,
    'c_hard_coral_foliose': 6,
    'c_hard_coral_mushroom': 7,
    'c_hard_coral_submassive': 8,
    'c_hard_coral_table': 9,
    'c_soft_coral': 10,
    'c_soft_coral_gorgonian': 11,
    'c_sponge': 12,
    'c_sponge_barrel': 13
}


def create_label_map_file():
    with open('data/coral_map.pbtxt', 'w') as f:
        for name, idx in class_map.items():
            f.write("item {\n")
            f.write(f"  id: {idx}\n")
            f.write(f"  name: '{name}'")
            f.write("\n}\n\n")


def create_tf_example(filename, annotation_list):
    img_path = os.path.join('data', 'images', filename)

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image_array = cv2.imread(img_path)
    height, width = image_array.shape[:2]
    image_format = b'jpeg'

    # Mask!

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    masks = []

    for annotation in annotation_list:
        class_idx = class_map[annotation['class']]
        classes_text.append(annotation['class'].encode('utf-8'))
        classes.append(class_idx)

        polygon_pts = annotation['coordinates']
        x, y = zip(*polygon_pts)

        xmin = float(min(x))
        xmax = float(max(x))
        ymin = float(min(y))
        ymax = float(max(y))

        # For some weird reason tensorflow needs the BBoxes normalised
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon_pts)], color=1)
        is_success, buffer = cv2.imencode(".png", mask)
        if not is_success:
            raise Warning(f'Could not save mask for file {filename}')

        io_buf = io.BytesIO(buffer)
        masks.append(io_buf.getvalue())

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask': dataset_util.bytes_list_feature(masks)
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # Load annotations
    annotations = parse_csv_data_file(FLAGS.annotations_path)

    for filename, annotation_list in annotations.items():
        print(f"Converting {filename}")
        tf_example = create_tf_example(filename, annotation_list)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    create_label_map_file()
    tf.app.run()
