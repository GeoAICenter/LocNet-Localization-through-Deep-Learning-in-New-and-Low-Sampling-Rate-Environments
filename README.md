# LocNet-Localization-through-Deep-Learning-in-New-and-Low-Sampling-Rate-Environments
This repository contains the code for the paper ["LocNet Localization Through Deep Learning in New and Low Sampling Rate Environments"](https://drive.google.com/file/d/1l3n6mgauMbAxEtiDcCP4a1yQhEpAWYIv/view?usp=sharing) (PAKDD-2024 Oral)
## Files
1. The <b>README.md</b> file contains the link to the dataset that was used to trained our LocNet model.

    The dataset includes:
    - DPM (a folder contains only full propagation maps); each image is a gray scale image.
    - buildings_complete (a folder contains only building maps); each image is a gray scale image.
    - antennas (a folder contains images of the transmitter ground truths; Each ground truth image only has 1 pixel that has a value of 255 and the rest of the pixels have 0 value; each image is a gray scale image. 
    - Train (a folder contains only binary images masks from DPM images that represent which pixels are sampled) - training set.
    - Val (a folder contains only binary images masks from DPM images that represent which pixels are sampled) - validation set.
    - Test (a folder contains only binary images masks from DPM images that represent which pixels are sampled) - testing set.

    Link to the [dataset](https://drive.google.com/file/d/1fZEPc5YwNNqKPGGTw5I3KTaEbBFeeSUZ/view?usp=sharing)
3. <b>LocNet_with_train.py</b> contains DataLoader function, LocNet's architecture, loss function, and LocNet's training function; We try to include only necessary parts in a file for simplistic.
4. <b>LocNet_with_use_single_image_load_the_pretrained_model.py</b> This file demonstrate how to load a pretrained LocNet model and how to use it to localize a transmitter on an image.
5. <b>Pretrained_LocNet.pt</b> contains the LocNet's weights.
6. <b>requiremens.txt</b> contains the necessary libraries that are needed to train and use LocNet.
## Dataset requirement.
To train LocNet on a custom dataset, the custom dataset must meet these requirements:
1. The dataset must have at least 5 folders in order to use LocNet_with_train.py without changing anything from the code.

    These are the must have folder:
    - full propagation maps folder (equivalent to DPM)
    - building environment folder (equivalent to buildings_complete)
    - ground truths folder (equivalent to antennas)
    - training set folder (equivalent to Train)
    - validation set folder (equivalent to Val)
2. Naming convention:
    - full propgation maps:<b><building_number>_<antenna_simulation_number>.png</b> ; building_number relates to which building environment is used from the building environment folder, antenna_simulation_number is a placebo number to distinguish different antenna positions simulations; For example: <b>1_15.png</b> means this propagation map <b>1_15.png</b> uses <b>building map 1</b> and 15 is the 15th simulation.
    - building environment folder: <b><building_number>.png</b> ; For example <b>1.png</b> means <b>building map 1</b>
    - ground truths folder: naming convention is the same as full propagation maps; <b><building_number>_<antenna_simulation_number>.png</b>
    - training set folder: <b><building_number>_<antenna_simulation_number>_<sampling_number_placebo>.png</b> ; <sampling_number_placebo> is a placebo number to distinguish between sampled images from a DPM image.
    - validation set folder: naming convention is the same as training set folder.
## How to do custom training or just recreating the result
Assume you download the dataset or you just created a custom dataset and it folder structure is like this

    .
    ├── Custom_Dataset                  
        ├── full_propagation_maps # or DPM
            ├── 0_0.png
            ├── 0_1.png
            ...
            └── 10_1.png
        ├── building_environment_maps # or buildings_complete
            ├── 0.png
            ├── 1.png
            ...
            └── 10.png
        ├── ground_truths # or antenna
            ├── 0_0.png
            ├── 0_1.png
            ...
            └── 10_1.png
        ├── training_set # or Train ;Assume training_set folder only contains simulations of even number building environments.
            ├── 0_0_0.png
            ├── 0_0_1.png
            ├── 0_1_0.png
            ...
            ├── 10_1_0.png
            └── 10_1_1.png
        ├── validation_set # or Val ;Assume validation_set folder only contains simulations of odd number building environments.
            ├── 1_0_0.png
            ├── 1_0_1.png
            ├── 1_1_0.png
            ...
            ├── 9_1_0.png
            └── 9_1_1.png
    ├── LocNet_with_train.py
    ├── LocNet_with_use_single_image_load_the_pretrained_model.py
    ├── Saved_Model
    ├── Pretrained_LocNet.pt
    ├── requirements.txt
    └── README.md

The command code to type in the terminal to train LocNet once the terminal is open the folder that contain the tree structure like above: <b>python LocNet_with_train.py -ftr ~/Custom_Dataset/DPM/ -mtr ~/Custom_Dataset/training_set/ -t ~/Custom_Dataset/antennas/ -b ~/Custom_Dataset/buildings_complete/ -mval ~/Custom_Dataset/validation_set/ -o ~/Saved_Model/</b>

## How to use LocNet model:
1. Open the terminal and use cd command (in Linux) or (chdir in Windows) (cd is stand for changing directory) go to the folder that contain the file <b>LocNet_with_use_single_image_load_the_pretrained_model.py</b>
2. Use this command to use the Pretrained LocNet model. <b>python LocNet_with_use_single_image_load_the_pretrained_model.py -i directory_path_to_the_sampling_pixel_map -k directory_path_to_the_testing_building_image -o directory_path_to_the_trained_LocNet_model</b>
    - sampling pixel map is a multiplication between full propagation map with binary sampled image mask.
## Reference:
The link to the dataset we use for full propagation map, building, and groundtruth: (https://radiomapseer.github.io/)


