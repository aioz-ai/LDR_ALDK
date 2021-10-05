# Light-weight Deformable Registration using Adversarial Learning with Distilling Knowledge
### Summary
* [Abstract](#abstract)
* [Prerequisites](#prerequisites)
* [Datasets](#datasets)
* [Training](#training)
* [Evaluation](#evaluation)
* [Citation](#citation)
* [Acknowledgement](#acknowledgement)
* [License](#license)
* [More information](#more-information)

### Abstract
Deformable registration is a crucial step in many medical procedures such as image-guided surgery and radiation therapy. Most recent learning-based methods focus on improving the accuracy by optimizing the non-linear spatial correspondence between the input images. Therefore, these methods are computationally expensive and require modern graphic cards for real-time deployment. In this paper, we introduce a new Light-weight Deformable Registration network that significantly reduces the computational cost while achieving competitive accuracy. In particular, we propose a new adversarial learning with distilling knowledge algorithm that successfully leverages meaningful information from the effective but expensive teacher network to the student network. We design the student network such as it is light-weight and well suitable for deployment on a typical CPU. The extensively experimental results on different public datasets show that our proposed method achieves state-of-the-art accuracy while significantly faster than recent methods. We further show that the use of our adversarial learning algorithm is essential for a time-efficiency deformable registration method.

### Prerequisites
**Python 3**   
Please install dependence packages by run the following command:
```
pip install -r requirements.txt
```
### Datasets
#### 1. Liver-and-brain-scanned datasets
Create the `datasets/` directory then follow this [repo](https://github.com/microsoft/Recursive-Cascaded-Networks#datasets) to download and extract the data.
#### 2. Extracted teacher deformations
Create the `Teacher_deformation/` folder and
download our extracted teacher (`Teacher_deformations.zip`) deformations from [here](https://vision.aioz.io/d/150a081d8ea84ea685da/).   
This folder is constructed as follow:   
```
|-- Teacher_deformations   
|---|---Brain   
|---|---|-- brain_teacher_deformation_c_1.pkl   
|---|---|-- brain_teacher_deformation_c_2.pkl   
|---|---|-- brain_teacher_deformation_c_3.pkl   
|---|---Liver
|---|---|-- liver_teacher_deformation_c_1.pkl   
|---|---|-- liver_teacher_deformation_c_2.pkl   
|---|---|-- liver_teacher_deformation_c_3.pkl   
|-- ...   
```

### Training

```
python train.py -b base-network -n num_cascades -g gpus_id --round steps_per_epochs --epochs num_epochs --batch batch_size --output name_of_weights_after_training -d path_to_dataset_json --aldk (if you use our ALDK)
```
**For examples:**   
Train 1-cas LDR on liver datasets
```
python train.py -b LDR -n 1 -g 0 --round 20000 --epochs 5 --batch 4 --output 1cas_LDR_Liver -d datasets/liver.json
```
Train 1-cas LDR on brain datasets
```
python train.py -b LDR -n 1 -g 0 --round 20000 --epochs 5 --batch 4 --output 1cas_LDR_Brain -d datasets/brain.json
```
Train 1-cas LDR + ALDK on liver datasets
```
python train.py -b LDR -n 1 -g 0 --round 20000 --epochs 5 --batch 4 --output 1cas_LDR_ALDK_Liver -d datasets/liver.json --aldk
```
Train 1-cas LDR + ALDK on brain datasets
```
python train.py -b LDR -n 1 -g 0 --round 20000 --epochs 5 --batch 4 --output 1cas_LDR_ALDK_Brain -d datasets/brain.json --aldk
```

### Evaluation
Our trained weights are available [here](https://vision.aioz.io/d/150a081d8ea84ea685da/), 
you can download then extract `trained_models.zip` to `weights/` directory
```
python eval.py --gpu gpus_id --checkpoint path-to-trained-weights --batch batch_size -v evaluation_test_set
```
**For examples:**   
Evaluate our 1cas_LDR_ALDK_Liver
```
python eval.py --gpu 0 --batch 1 --checkpoint weights/1cas_LDR_ALDK_Liver
```
Evaluate our 1cas_LDR_ALDK_Brain
```
python eval.py --gpu 0 --batch 1 --checkpoint weights/1cas_LDR_ALDK_Brain
```
### Citation
If you use this code as part of any published research, we'd really appreciate it if you could cite the following paper:
```
@misc{tran2021lightweight,
      title={Light-weight Deformable Registration using Adversarial Learning with Distilling Knowledge}, 
      author={Minh Q. Tran and Tuong Do and Huy Tran and Erman Tjiputra and Quang D. Tran and Anh Nguyen},
      year={2021},
      eprint={2110.01293},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4755947.svg)](https://doi.org/10.5281/zenodo.4755947)

### Acknowledgement
Many thanks to this thorough repository ([RCN](https://github.com/microsoft/Recursive-Cascaded-Networks)).

### License
MIT License

**AIOZ © 2021 All rights reserved.**

### More information
- AIOZ AI Homepage: https://ai.aioz.io   
- AIOZ Network: https://aioz.network
