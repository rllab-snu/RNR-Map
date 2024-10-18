
<h1 align="center">
Renderable Neural Radiance Map <br> for Visual Navigation‬ 
</h1>

<h3 align="center"><a href="https://arxiv.org/abs/2303.00304">Paper</a> | <a href="https://youtu.be/oLo3L0oMcWQ">Video</a> | <a href="https://rllab-snu.github.io/projects/RNR-Map/">Project Page</a></h3>
<div align="center">
</div>

<p align="center">
Official Github repository for <b>"Renderable Neural Radiance Map for Visual Navigation‬"</b>.
<br>
$\color{#58A6FF}{\textsf{Highlighted Paper at CVPR 2023}}$
<br>
  

<img src="/media/overview.gif">
<br>

## Table of contents

0. [Setup](#Setup)
1. [Mapping and Reconstruction](#Mapping-and-Reconstruction)
2. [Image-based Localization](#Image-based-Localization)
3. [Camera Tracking](#Camera-Tracking)
4. Image-Goal Navigation 

## Citing RNR-Map
If you find this code useful in your research, please consider citing:

```
@InProceedings{Kwon_2023_CVPR,
    author    = {Kwon, Obin and Park, Jeongho and Oh, Songhwai},
    title     = {Renderable Neural Radiance Map for Visual Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9099-9108}
}
```

## Setup

### Requirements
- Ubuntu 20.04 or higher 
- python 3.8 or higher
- habitat-sim 0.2.1 (commit version: b75d11cd0974e47a226d0263c70938876f605bcf)
- habitat-lab 0.2.1 (commit version: 1d187db53f931070eebe87595c4484903b2e0595)
- pytorch 1.12.1+cu102 or similar version
- torchvision 0.13.1+cu102 or similar version
- Other python libraries are included in requirements.txt

Habitat (sim/lab) libraries highly depend on the version. We recommend to use the same version as we used.
Please refer to [habitat-sim](https://github.com/facebookresearch/habitat-sim.git) and [habitat-lab](https://github.com/facebookresearch/habitat-lab.git) for installation.

### Installation
To set the environment, run:
```
pip install -r requirements.txt
```

### Habitat Dataset (Gibson, MP3D) Setup
Most of the scripts in this code build the environments assuming that the **gibson/mp3d dataset** is in **habitat-lab/data/** folder.

The recommended folder structure of habitat-api (or habitat-lab):
```
habitat-lab
  └── data
      └── datasets
      │   └── pointnav
      │       └── gibson
      │           └── v1
      │               └── train
      │               └── val
      └── scene_datasets
          └── gibson_habitat
              └── *.glb, *.navmeshs  
```

## Mapping and Reconstruction

You can try mapping with pretrained models using demo notebooks.
- [Demo] 1. Mapping using sample data.ipynb
- [Demo] 1. Mapping using simulator.ipynb

### [Mapping using sample data](%5BDemo%5D%201.%20Mapping%20using%20sample%20data.ipynb) 
In this notebok, you can build an RNR-Map using sample trajectory data.
The trajectory data contains robot poses and corresponding RGBD images.

You can embed the images and also render the embedded images simultaneously.
You will see the visualization window during embedding phase.

![mapping](demo/embedding_traj.gif)

Note that the image reconstruction is just for visualization.
You can skip the rendering part (embedder.generate) for faster embedding.

The speed of embedding and rendering depends on hardware settings.
In ours (Intel i7-9700KF CPU @ 3.60GHz, single RTX 2080 Ti), a single embedding process spends 10.9ms 
and the rendering process spends 68.0ms in average.
Note that the visualization also takes some time.


After embedding, you can freely navigate around inside the RNR-Map using the keyboard.
You can observe that RNR-Map can render the images from the novel-view, where the camera has not visited.
The visualization window will show the rendered images from RNR_map given a camera pose, as well as the location in RNR-Map.
Press 'w,a,s,d' to move the camera, and 'q' to quit navigation.

![exploring](demo/explore_RNR_map.gif)


### [Mapping Using Simulator](%5BDemo%5D%201.%20Mapping%20using%20simulator.ipynb) 
You can also build RNR-map from directly from habitat simulator.
First, generate random navigation trajectory using habitat-simulator.
While collecting the trajectory, you can build RNR-Map simulataneously.
The visualization window will show the collected images and the rendered images from RNR-Map.

![mapping_simulator](demo/embedding_traj_from_simulator.gif)

Also, note that the image reconstruction is just for visualization.
You can skip the rendering part (embedder.generate) for faster embedding.

Same as the previous notebook, you can freely navigate around inside the RNR-Map using the keyboard.
As we have simulator, you can also compare the rendered images with real images. 
You can observe that RNR-Map can render the images from the novel-view, where the camera has not visited.

The visualization window will show the rendered images from RNR_map and ground-truth images, as well as the location in RNR-Map.
Press 'w,a,s,d' to move the camera, and 'q' to quit navigation.

![exploring_simulator](demo/explore_RNR_map_from_simluator.gif)

---

## Image-based Localization

Try image-based localization with the trained model in this demo notebook!
- [Demo] 2. Image-based Localization.ipynb

First, you need to build an RNR-Map for the target environment.
Then, you can query an image to find a location in the RNR-Map.

The notebook build the RNR-Map from the simulator, but you can change it to build from offline data, similar to [mapping demo](#Mapping-using-sample-data).


![visual_loc_explain.gif](media/visual_loc_explain.gif)

The query image can be sampled from seen observations, or unseen observations from environment.

### Localize Seen observations

Run the cell named "Localize seen images."

The query image will be given to RNR-Map in order, and the RNR-map-based localization framework predicts the pose of the query image.
Then, we take a picture from the predicted pose in the simulator (named as "localized image") and compare it with the query image.

The visualization window will show the query image, localized image, pose comparison on RNR-Map, and localization probabilities.
In the pose comparison image, the red is the ground-truth pose, and the blue is the predicted pose.

![image-base-localization.gif](demo/image-based-localization.gif)


### Unseen observations
> Note that RNR-Map only contains information of the observed regions. 
RNR-Map can localize the novel view from the **observed** region. 
> 
> If the observation is sampled from unobserved region, it can only roughly predict where the place would be. 
> If you want to localize completely unseen observation from the environments, try image-goal navigation demo instead.

Run the cell name "Localize images from unseen view."

In this cell, we perturb the original trajectory with noises to make unseen view poses and re-take the query picture from the simulator.
The query images are taken from the unseen view, and RNR-Map has not observed the same image during the mapping phase.
The examples of localization from an unseen view are shown in the following:

![image-base-localization.gif](demo/image-based-localization-noise.gif)

---

## Camera Tracking
- [Demo] 3. Camera Tracking.ipynb

Try pose adjustment using RNR-Map!

As the rendering process of RNR-Map is differentiable, you can optimize the camera pose using gradient descent.
In this notebook, we will show how to optimize the camera pose using RNR-Map.

First, load pretrained models and setup a random habitat environment.

Then, we will randomly sample start position and navigate to random goal position.
While navigating, we measure odometry and add sensor noises (from [Active Neural SLAM](https://github.com/devendrachaplot/Neural-SLAM)).
The estimated poses will start to be deviated from the ground-truth poses, and the objective of this demo is to find the gt pose using RNR-Map.

In the demo notebook, 'gt_loc' refers to the ground-truth robot pose without any noises, 'bias_loc' refers to the biased robot pose because of the noises,
and 'pred_loc' (or 'curr_loc') is the optimized pose using RNR-Map.

As the rendering process of RNR-Map is differentiable, we can optimize 'pred_loc' by comparing the rendered pixels from pred_loc and observed images.

The visualization window will show the three trajectories and observation images.
In the following, the black trajectory is ground-truth, the green one is biased trajectory without optimization, and the red one is the optimized one.

![camera-tracking.gif](demo/camera-tracking.gif)

At every optimization step, we only render 200 random pixels for speed. In our setting, a single time step (including pose optimization + visualization + mapping) takes approximately 0.2 secs. 

After an episode, you can check the whole trajectories by plotting them as following:

![camera-tracking.png](demo/camera_tracking_plot.png)



---
### Acknowledgement
Large part of the source code is based on [GSN](https://github.com/apple/ml-gsn).
Also, we adapted codes and noise models from [Active Neural SLAM](https://github.com/devendrachaplot/Neural-SLAM)
