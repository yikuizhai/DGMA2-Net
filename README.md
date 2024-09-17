# DGMA2-Net: A Difference-Guided Multiscale Aggregation Attention Network for Remote Sensing Change Detection

## Abtract

Remote sensing change detection (RSCD) focuses on identifying regions that have undergone changes between two remote sensing images captured at different times. Recently, convolutional neural networks (CNNs) have shown promising results in the challenging task of RSCD. However, these methods do not efficiently fuse bitemporal features and extract useful information that is beneficial to subsequent RSCD tasks. In addition, they did not consider multilevel feature interactions in feature aggregation and ignore relationships between difference features and bitemporal features, which, thus, affects the RSCD results. To address the above problems, a difference-guided multiscale aggregation attention network, DGMA2-Net, is developed. Bitemporal features at different levels are extracted through a Siamese convolutional network and a multiscale difference fusion module (MDFM) is then created to fuse bitemporal features and extract, in a multiscale manner, difference features containing rich contextual information. After the MDFM treatment, two difference aggregation modules (DAMs) are used to aggregate difference features at different levels for multilevel feature interactions. The features through DAMs are sent to the difference-enhanced attention modules (DEAMs) to strengthen the connections between bitemporal features and difference features and further refine change features. Finally, refined change features are superimposed from deep to shallow and a change map is produced. In validating the effectiveness of DGMA2-Net, a series of experiments are conducted on three public RSCD benchmark datasets [LEVIR building change detection dataset (LEVIR-CD), Wuhan University building change detection dataset (BCDD), and Sun Yat-Sen University dataset (SYSU-CD)]. The experimental results demonstrate that DGMA2-Net surpasses the current eight state-of-the-art methods in RSCD.

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![zhai2-3390206-large](https://github.com/user-attachments/assets/e3190468-872b-4e86-8c9c-baf387cb0ea9)

## Get Strat

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
