

## HySparK: Hybrid Sparse Masking for Large Scale Medical Image Pre-Training

<p align="center" width="100%">
<!---->
</p> 

![HySparK](https://github.com/FengheTan9/HySparK/blob/main/imgs/HySparK.gif)


<div align="center">
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=x1pODsMAAAAJ&hl=en" target="_blank">Fenghe Tang</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=TxjqAY0AAAAJ&hl=en" target="_blank">Ronghao Xu</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=CMiRzlAAAAAJ&hl=en" target="_blank">Qingsong Yao</a><sup>3</sup>,</span>
    <br>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=az4zv18AAAAJ&hl=en" target="_blank">Xueming Fu</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=mlTXS0YAAAAJ&hl=en" target="_blank">Quan Quan</a><sup>3</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=YkfSFekAAAAJ&hl=en" target="_blank">Heqin Zhu</a><sup>1,2</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=OkrZX0AAAAAJ&hl=en" target="_blank">Zaiyi Liu</a><sup>4,5</sup>,</span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en" target="_blank">S.Kevin Zhou</a><sup>1,2,3</sup>
    </span>
</div>
<br>

<div align="center">
    <sup>1</sup>
    <a href='https://en.ustc.edu.cn/' target='_blank'>School of Biomedical Engineering, University of Science and Technology of China</a>&emsp;
    <br>
    <sup>2</sup> <a href='http://english.ict.cas.cn/' target='_blank'>Suzhou Institute for Advanced Research, University of Science and Technology of China</a>&emsp;
    <br>
    <sup>3</sup> <a href='http://english.ict.cas.cn/' target='_blank'>Institute of Computing Technology, Chinese Academy of Sciences</a>
    <br>
    <sup>4</sup>
    <a href='https://english.ucas.ac.cn/' target='_blank'>Department of Radiology, Guangdong Provincial People’s Hospital</a>&emsp;
    </br>
    <sup>5</sup>
    <a href='https://english.ucas.ac.cn/' target='_blank'>Guangdong Provincial Key Laboratory of Artificial Intelligence in Medical Image Analysis and Application</a>&emsp;
    </br>
</div>
<br>
<br>

​                                              [![arXiv](https://img.shields.io/badge/arxiv-2408.05815-b31b1b)](https://arxiv.org/pdf/2408.05815v1)   [![github](https://img.shields.io/badge/github-HySparK-black)](https://github.com/FengheTan9/HySparK)    <a href="#LICENSE--citation"><img alt="License: Apache2.0" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue.svg"/></a>



## News

- **Code and weight now are released 😎 !**
- **HySparK is accepted by <u>MICCAI 2024 (Early accept)</u> !**
- **Code will be released soon !** 😘



## TODOs

- [x] Paper released
- [x] Code released
- [x] Weight released



### Models

#### Pre-trained weights

| Name      | Pre-trained data scale | Weights                                                      |
| --------- | ---------------------- | ------------------------------------------------------------ |
| HySparK-B | 6.8k CT Scan           | [hybird_ct_pretrained_timm_style_mask75.pth](https://github.com/FengheTan9/HySparK/blob/main/ckpt/hybird_ct_pretrained_timm_style_mask75.pth) |





## Getting Started



### Prepare Environment

```
conda create -n hyspark python=3.9
conda activate hyspark
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging timm==0.5.4
pip install transformers==4.34.1 typed-argument-parser
pip install numpy==1.21.2 opencv-python==4.5.5.64 opencv-python-headless==4.5.5.64
pip install 'monai[all]'
pip install monai==1.2.0
```



### Prepare Datasets

We recommend you to convert the dataset into the nnUNet format.

```
└── HySparK
    ├── data
        ├── Dataset060_TotalSegmentator
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
        ├── Dataset006_FLARE2022
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
        └── Other_dataset
            └── imagesTr
                ├── xxx_0000.nii.gz
                ├── ...
```

Try to use the function organize in  [nnunet-style](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) or ```organize_by_names``` to prepare your custom datasets.

Then run :

```python
python generate_js.py
```

A example ```dataset.json``` will be generated in ```./data```

The content should be like below

```json
{
    "training": [
        {
            "image": "./Dataset060_TotalSegmentator/imagesTr/xxx_0000.nii.gz"
        },
        {
            "image": "./Dataset006_FLARE2022/imagesTr/xxx_0000.nii.gz"
        },
    ]
}

```



## Start Training

Run training on multi-GPU :

```sh
# An example of training on 4 GPUs with DDP
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12351 main.py --exp_name=debug --data_path=./data  --model=hyspark --bs=12  --exp_dir=debug_hyspark_ddp_4
```

Run training on single-GPU :

```sh
# An example of training on single GPU
python main.py --exp_name=debug --data_path=./data --model=hyspark --bs=4 --exp_dir=debug_hyspark
```



## Fine-tuning

Load pre-training weights :

```python
# An example of Fine-tuning on BTCV (num_classes=14)
from models.network.hyspark_model import build_hybird

model = build_hybird(in_channel=1, n_classes=14, img_size=96).cuda()

model_dict = torch.load("./[your_ckpt_path]/hybird_ct_pretrained_timm_style_mask75.pth")   

if model.load_state_dict(model_dict, strict=False):
    print("HySpark use pretrained weights successfully !")
```



The downstream pipeline can be referred to [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)



## Acknowledgements:

This code base uses helper functions from [SparK](https://github.com/keyu-tian/SparK).



## Citation

If the code, paper and weights help your research, please cite:

```
@inproceedings{tang2024hyspark,
  title={Hyspark: Hybrid sparse masking for large scale medical image pre-training},
  author={Tang, Fenghe and Xu, Ronghao and Yao, Qingsong and Fu, Xueming and Quan, Quan and Zhu, Heqin and Liu, Zaiyi and Zhou, S Kevin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={330--340},
  year={2024},
  organization={Springer}
}
```

## License

This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
