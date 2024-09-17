# DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection

## Introduction

This repo is the official implementation of https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10504297

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![zhai2-3390206-large](https://github.com/user-attachments/assets/e3190468-872b-4e86-8c9c-baf387cb0ea9)

## Get Strat
![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-levir-cd)

![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-whu-cd)

![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-sysu-cd)
### Download three dataset LEVIR-CD, BCDD, SYSU-CD, and prepare datasets into following structure

```
├─Train
    ├─A
    ├─B
    ├─label
    └─list
├─Val
    ├─A
    ├─B
    ├─label
    └─list
├─Test
    ├─A
    ├─B
    ├─label
    └─list
```

### Train

```
run ./tools/train.py
```

### Test

```
run ./tools/test.py
```

## Acknowledgement

This repository is built under the help of the projects [TFI-GR](https://github.com/guanyuezhen/TFI-GR) for academic use only.

## Citation

@ARTICLE{10504297,  

>author={Ying, Zilu and Tan, Zijun and Zhai, Yikui and Jia, Xudong and Li, Wenba and Zeng, Junying and Genovese, Angelo and Piuri, Vincenzo and Scotti, Fabio},  
>journal={IEEE Transactions on Geoscience and Remote Sensing},  
>title={DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection},  
>year={2024},  
>volume={62},  
>pages={1-16},  
>doi={10.1109/TGRS.2024.3390206}   
>}
