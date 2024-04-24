import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_image_grid(savedir, num_images_per_class=10, num_cols=10, figsize=(20, 16)):
    categories = os.listdir(savedir)
    num_categories = len(categories)
    
    fig, axes = plt.subplots(num_categories, num_cols, figsize=figsize)
    fig.tight_layout(pad=1.0)
    
    for i, category in enumerate(categories):
        category_dir = os.path.join(savedir, category)
        image_files = os.listdir(category_dir)
        random_images = random.sample(image_files, num_images_per_class)
        
        for j, image_file in enumerate(random_images):
            image_path = os.path.join(category_dir, image_file)
            image = mpimg.imread(image_path)
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
        
        axes[i, 0].text(-0.1, 0.5, category, fontsize=14, rotation=90, va='center', ha='right', transform=axes[i, 0].transAxes)
    
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.show()
    
    
def create_imagenet_Baby8(imagenet_train_root, synset_lists, num_images_per_class=1000, savedir=None):
    
    if savedir is None:
        savedir = 'images/'
#     else:
#         savedir = os.path.join(savedir,'images/')
    
    # Check if the directory does not already exist
    Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each broad category in savedir
    for category in synset_lists.keys():
        os.makedirs(os.path.join(savedir, category), exist_ok=True)
    
    # Iterate over each broad category
    for category, synsets in tqdm(synset_lists.items()):
        print(f"Processing category: {category}")
        
        # Get the list of all image paths for the current category
        image_paths = []
        for synset in synsets:
            synset_dir = os.path.join(imagenet_train_root, synset)
            if os.path.exists(synset_dir):
                synset_image_paths = [os.path.join(synset_dir, f) for f in os.listdir(synset_dir) if f.endswith('.JPEG')]
                image_paths.extend(synset_image_paths)
        
        # Randomly select num_images_per_class images from the category
        selected_images = random.sample(image_paths, min(num_images_per_class, len(image_paths)))
        
        # Copy the selected images to the corresponding category directory in savedir, keeping the original filenames
        for image_path in tqdm(selected_images):
            filename = os.path.basename(image_path)
            dest_path = os.path.join(savedir, category, filename)
            shutil.copy(image_path, dest_path)
        
        print(f"Copied {len(selected_images)} images for category: {category}")
    
    print("Dataset creation completed.")
    
    
if __name__ == "__main__":
    imagenet_train_root = '/N/project/baby_vision_curriculum/benchmarks/mainstream/imagenet/ILSVRC/Data/CLS-LOC/train/'
    
    # chosen ImageNet category to represent each baby category
    cup_synsets = ['n07930864', 'n03063599']
    hat_synsets = ['n02817516', 'n04209133', 'n04259630']
    chair_synsets = ['n02791124', 'n03376595', 'n04099969', 'n04429376']
    car_synsets = ['n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03393912', 'n03447447', 
          'n03594945', 'n03670208', 'n03770679', 'n03777568', 'n03895866', 'n04037443', 
          'n04285008']
    airplane_synsets = ['n02690373', 'n04552348']
    duck_synsets = ['n01847000', 'n01855032']
    donkey_synsets = ['n02389026', 'n02391049']
    dog_synsets = ['n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 
          'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 
          'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 
          'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 
          'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 
          'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 
          'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 
          'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 
          'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 
          'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 
          'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 
          'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 
          'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 
          'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 
          'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 
          'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 
          'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 
          'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 
          'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 
          'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02963159', 'n07697537']

    synset_lists = {"cup":cup_synsets, 
              "hat":hat_synsets, 
              "chair":chair_synsets, 
              "car":car_synsets, 
              "airplane":airplane_synsets, 
              "duck":duck_synsets, 
              "donkey":donkey_synsets, 
              "dog":dog_synsets}
    
    savedir = '/N/project/baby_vision_benchmark/imagenet_baby8/'

    create_imagenet_Baby8(imagenet_train_root, synset_lists, num_images_per_class=1000, 
                          savedir=savedir)
    
    plot_image_grid(savedir, num_images_per_class=10, num_cols=10, figsize=(20, 16))