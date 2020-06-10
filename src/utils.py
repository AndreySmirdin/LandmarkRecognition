import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_model(model, name, num):
    state = torch.load(f'models/{name}{num}.data')
    prefix = 'module.'
#     prefix = '' 
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in state.items()
                    if k.startswith(prefix)}
 
    model.load_state_dict(adapted_dict)
    return model

transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(240),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(240),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def get_dataloaders(train_path, test_path, batch_size):
    dataloaders = {}
    datasets = {}
    
    datasets['train'] = torchvision.datasets.ImageFolder(train_path, transform=transforms['train'])
    datasets['test'] = torchvision.datasets.ImageFolder(test_path, transform=transforms['val'])
    
    dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    return datasets, dataloaders


def show_images(images, titles=None, correctness=None, cols=1):
    """Display a list of images in a single figure with matplotlib.
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    if correctness is None: correctness = [0] * n_images
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(','.join(map(lambda x: "%.2f" % x, title)), fontdict={'fontsize': 80})
        from matplotlib.patches import Rectangle

        rect = Rectangle((0, 0), 240, 240, linewidth=20, edgecolor=['r', 'g'][correctness[n]], facecolor='none')
        a.add_patch(rect)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
def load_images_in_folder(path):
    files = os.listdir(path)
    images = torch.zeros(len(files), 3, 240, 240).float()
    images_no_trans = np.zeros((len(files), 240, 240, 3), dtype='uint8')
    for i, f in enumerate(files):
        images_no_trans[i] = np.asarray(Image.open(os.path.join(path, f)).resize((240, 240)))
        images[i] = transforms['val'](Image.open(os.path.join(path, f)))
        
    return images_no_trans, images

def modify_keys(d, dataset):
    fixed = {}
    for k in d:
        fixed[dataset.classes[k]] = d[k]
    return fixed


def save_results(path, results):
    classes = []
    for x in results:
        classes.append({'name': x, 'correct': results[x][0], 'total': results[x][1]})
    classes = sorted(classes, key=lambda x: x['correct'] / x['total'])
    df = pd.DataFrame(classes)
    df.to_csv(path)