# Author: Sahaj Singh Maini


from datasets import load_dataset #pip install datasets; pip install datasets[vision]
import numpy as np
from PIL import Image  
import PIL
from matplotlib import pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

def plot_images(images, Data_modified_label_list):
# Get unique classes
    classes = list(set(Data_modified_label_list))

    # Create a figure and axes for an 8x10 grid
    fig, axs = plt.subplots(8, 10, figsize=(20, 16))

    # For each class
    for i, cls in enumerate(classes):
        # Get images of this class
        cls_images = [img for img, label in zip(images, Data_modified_label_list) if label == cls]
        
        # For each image of this class
        for j, img in enumerate(cls_images[:10]):
            # Plot the image on the corresponding axes
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    
    plt.show()
    return


def create_imagenet_Baby8(num_images_per_class=10, savedir=None):
    """
    this function creates a subset of the ImageNet dataset with a specified number of images per class for the selected object categories (cup, hat, chair, car, airplane, duck, donkey, dog). It saves the images to a directory and stores the true labels and modified labels in separate lists, which are then saved to a file.
    
    Note: must log in to huggingface cli using an access token to be able to load imagenet1k
    login: huggingface-cli login
    generate access token: https://huggingface.co/docs/hub/security-tokens
    
    """
    if savedir is None:
        savedir = 'images/'
    else:
        savedir = os.path.join(savedir,'images/')
    
    # Check if the directory does not already exist
    Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # Load the ImageNet dataset using the Hugging Face datasets library
    dataset = load_dataset("imagenet-1k", 'en', split='train', streaming=True)
    
    # Define lists of class labels for different object categories
    cup_list = [438, 572, 968, 647, 504, 441]
    hat_list = [433, 439, 452, 515, 518, 560, 667, 793, 808]
    chair_list = [423, 559, 765, 857]
    car_list = [407, 436, 468, 511, 565, 576, 609, 627, 656, 661, 705, 751, 817]
    airplane_list = [404, 895]
    duck_list = [97, 98, 99]
    donkey_list = [339, 340]
    dog_list = [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 474, 934]
    
    # Create dictionaries to map object categories to their class labels and new labels
    old_labels = {"cup":cup_list, "hat":hat_list, "chair":chair_list, "car":car_list, "airplane":airplane_list, "duck":duck_list, "donkey":donkey_list, "dog":dog_list}
    new_labels = {"cup":0, "hat":1, "chair":2, "car":3, "airplane":4, "duck":5, "donkey":6, "dog":7}
    
    # Initialize a dictionary to keep track of the count of images for each object category
    new_labels_count = {"cup":0, "hat":0, "chair":0, "car":0, "airplane":0, "duck":0, "donkey":0, "dog":0}
    
    # Initialize lists to store image data, true labels, and modified labels
    Data_images_list = []
    Data_true_label_list = []
    Data_modified_label_list = []
    
    # Iterate over the old_labels dictionary and print the number of ImageNet subclasses for each object category
    for i in old_labels:
        print("Num of imagenet subclasses in ", i, " - ", len(old_labels[i]))
    
    # Set the desired number of images per class
    label_count = num_images_per_class
    
    # Initialize a counter for generating unique filenames
    img_name_counter = 0
    
    # Iterate over the ImageNet dataset
    for i in iter(dataset):
        # Check if the image's label belongs to any of the object categories
        for j in old_labels:
            if i['label'] in old_labels[j] and new_labels_count[j] < label_count:
                # Save the image to the savedir directory with a unique filename
                fname = str(img_name_counter)+".jpeg"
                i['image'].save(os.path.join(savedir, fname))
                img_name_counter+=1
                
                # Append the true label and modified label to the respective lists
                Data_true_label_list.append(i['label'])
                Data_modified_label_list.append(new_labels[j])
                
                # Increment the count for the corresponding object category
                new_labels_count[j] += 1
                break
        
        # Check if the desired number of images per class has been reached for all object categories
        print(np.array(list(new_labels_count.values())))
        if all(np.array(list(new_labels_count.values()))==label_count):
            break
    
    # Save the true labels and modified labels to a file
    np.savez("dataset_labels.npz", Data_true_label_list, Data_modified_label_list)

if __name__ == "__main__":
    create_imagenet_Baby8(num_images_per_class = 500, savedir='/N/project/baby_vision_benchmark/imagenet_baby8/')
