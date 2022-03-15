## Label-Efficient Point Cloud Semantic Segmentation: An Active Learning Approach
Created by <a href="https://github.com/SBXGary" target="_blank">Xian Shi</a> from South China University of Technology.

### Introduction
This work is based on our paper <a href="https://arxiv.org/abs/2101.06931">Label-Efficient Point Cloud Semantic Segmentation: An Active Learning Approach</a>. To better exploit labeling budget, we adopt a super-point based active learning strategy where we make use of manifold defined on the point cloud geometry. We further propose active learning strategy to encourage shape level diversity and local spatial consistency constraint. Experiments on ShapeNet [1] and S3DIS [2] demonstrate the efficacy of our proposed active learning strategy for label-efficient semantic segmentation of point clouds. 


Here we release python code for experiments on ShapeNet and you are welcome to report any bugs you would identify. Should you have any concerns or experience any issues please raise in Issues so that all people can benefit from the discussions.

### Citation
Please cite the following work if you feel it is helpful.

@article{shi2021label,
  title={Label-efficient point cloud semantic segmentation: An active learning approach},
  author={Shi, Xian and Xu, Xun and Chen, Ke and Cai, Lile and Foo, Chuan Sheng and Jia, Kui},
  journal={arXiv preprint arXiv:2101.06931},
  year={2021}
}

### Installation
This code has been tested on Pyhon3.8, CUDA 11.0, cuDNN 9.2 and Ubuntu 20.04

### Usage
(1) Train the backbone(<a href="https://github.com/muhanzhang/pytorch_DGCNN">dgcnn</a>) with current weakly mask

	python main_partseg_al.py
	
(2) Obtain the unlabeled samples' feature and probability outputted by the trained model

	main_partseg_aloutput.py

(3) Update labeled mask 

	main_partseg_alselect.py

(4) Repeat from (1) to (3) until the annotation budget is reached.

Reference:

[1] Li Yi, Vladimir G Kim, Duygu Ceylan, I Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, Leonidas Guibas, et al. A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics, 2016.

[2] Iro Armeni, Ozan Sener, Amir R Zamir, Helen Jiang, Ioannis Brilakis, Martin Fischer, and Silvio Savarese. 3d semantic parsing of large-scale indoor spaces. In CVPR, 2016.
