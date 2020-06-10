import torch
import os
import argparse
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from src.utils import load_model, get_dataloaders
import glob
import itertools
import tensorflow as tf
import tensorflow_hub as hub

STEP1_THRESHOLD = 0.8
INLIERS = 14

class LandmarksDataset(Dataset):
    def __init__(self, root_dir, inverse_labels):
        self.landmarks_frame = pd.read_csv('~/gpsData.csv')
        self.root_dir = root_dir
        self.landmarks = []
        for i, row in tqdm(self.landmarks_frame.iterrows()):
            if os.path.exists(os.path.join(self.root_dir, str(row[0]) + '.jpg')) and row[1] in inverse_labels:
                self.landmarks.append((row[0], row[1]))                    

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                str(self.landmarks[idx][0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.landmarks[idx][1]
        
        image = transforms['val'](image)
        sample = (image, label, self.landmarks[idx][0])
        return sample
    


def match_images(results_dict, image_1_path, image_2_path):
    distance_threshold = 0.8

    # Read features.
    locations_1, descriptors_1 = results_dict[image_1_path]
    num_features_1 = locations_1.shape[0]
    locations_2, descriptors_2 = results_dict[image_2_path]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

  # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)
    # the number of inliers as the score for retrieved images.
    return sum(inliers)

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter parser',
    )
    
    parser.add_argument('-train_path', required=True, type=str, help='Path to train dataset')
    parser.add_argument('-new_dataset_path', required=True, type=str, help='Path to the dataset that should be filtered')
    parser.add_argument('-output_step1_csv', required=True, type=str, help='Csv with files after step1')
    parser.add_argument('-output_step2_csv', required=False, type=str, help='Csv with files after step2')
    parser.add_argument('-step', required=False, type=int, choices=[1, 2])
    
    args = parser.parse_args()
    
    datasets, dataloaders = get_dataloaders(args.train_path, args.train_path, 32)
    
    inverse_labels = {}
    for i, c in enumerate(datasets['train'].classes):
        inverse_labels[int(c)] = i
        
    d = LandmarksDataset(args.new_dataset_path, inverse_labels) # '/mnt/hdd/1/gpsData/gpsData'
    loader = DataLoader(d, batch_size=32, shuffle=False)

    if args.step == 1 or args.step is None:
        with open(args.output_step1_csv, 'w') as f:
            f.write('id,landmark_id,dist\n')
            with torch.no_grad():
                for x, y, z in tqdm(loader):
                    feats, output = model(x.to(device))
                    for i in range(len(y)):
                        d = cosine_similarity(feats.cpu()[i].reshape(1, -1), 
                                          model.center_loss.centers.cpu()[inverse_labels[y[i].item()]].reshape(1, -1))[0][0]
                        if d > STEP1_THRESHOLD:
                            f.write(str(z[i].item()) + "," + str(y[i].item()) + "," + str(d))
                            f.write('\n')
    if args.step == 2 or args.step is None:
        assert args.output_step2_csv is not None
        cleared = pd.read_csv(args.output_step1_csv)

        m = hub.Module('https://tfhub.dev/google/delf/1')
        image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

        module_inputs = {
            'image': image_placeholder,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }

        module_outputs = m(module_inputs, as_dict=True)

        with open(args.output_step2_csv, 'w') as f:
            f.write('id,landmark_id\n')
            for clazz in tqdm(datasets['train'].classes):
                pathes = glob.glob(os.path.join(args.train_path, clazz) + '/*') 
                new_pathes = []
                for i, row in tqdm(cleared.iterrows()):
                    if row['landmark_id'] == clazz:
                        new_pathes.append(os.path.join(args.new_dataset_path, row['id']))


                results_dict = {}
                image_tf = image_input_fn(list(itertools.chain(pathes, new_pathes)))
                with tf.train.MonitoredSession() as sess:
                    results_dict = {}
                    for image_path in itertools.chain(pathes, new_pathes):
                        image = sess.run(image_tf)
                        results_dict[image_path] = sess.run(
                            [module_outputs['locations'], module_outputs['descriptors']],
                            feed_dict={image_placeholder: image})

                for file in new_pathes:
                    cnt = 0
                    for old_file in pathes:
                        if match_images(results_dict, file, new_file) > INLIERS:
                            cnt += 1
                        if cnt == 5:
                            splitted = file.split('/')
                            f.write(splitted[-1], splitted[-2])
