# DRRNet

##### DRRNet: Macro–Micro Feature Fusion and Dual Reverse Refinement for Camouflaged Object Detection

[[ckpt](https://drive.google.com/drive/folders/1-zu75yJucsX8d6FrR9xku3-AyRa5US6n?usp=sharing)] [[Results](https://drive.google.com/drive/folders/1aVS-cN0iUUzKF3puXVLrz99ITREMdMiN?usp=drive_link)] [[Pretrained models](https://drive.google.com/drive/folders/1_U8Kw9zs0E6Bcbjw-r58kWZHraL9b2SD?usp=sharing)]

> **Abstract**: The core challenge in Camouflage Object Detection (COD) lies in the indistinguishable similarity between targets and backgrounds in terms of color, texture, and shape. This causes existing methods to either lose edge details (such as hair-like fine structures) due to over-reliance on global semantic information or be disturbed by similar backgrounds (such as vegetation patterns) when relying solely on local features. We propose DRRNet, a four-stage architecture characterized by a ``context-detail-fusion-refinement'' pipeline to address these issues. Specifically, we introduce an Omni-Context Feature Extraction Module to capture global camouflage patterns and a Local Detail Extraction Module to supplement microstructural information for the full-scene context module. We then design a module for forming dual representations of scene understanding and structural awareness, which fuses panoramic features and local features across various scales. In the decoder, we also introduce a reverse refinement module that leverages spatial edge priors and frequency-domain noise suppression to perform a two-stage inverse refinement of the output. By applying two successive rounds of inverse refinement, the model effectively suppresses background interference and enhances the continuity of object boundaries. Experimental results demonstrate that DRRNet significantly outperforms state-of-the-art methods on benchmark datasets.
>
> ![img](https://github.com/jerrySunning/DRRNet/raw/main/tb_result.png)

------

## Usage

#### 1. Training Configuration

The pretrained model is stored in [Google Drive](https://drive.google.com/drive/folders/1_U8Kw9zs0E6Bcbjw-r58kWZHraL9b2SD?usp=sharing). After downloading, please change the file path in the corresponding code.

```
python Train.py  --epoch 100  --lr 1e-4  --batchsize 16  --trainsize 384  --train_root YOUR_TRAININGSETPATH  --val_root  YOUR_VALIDATIONSETPATH  --save_path YOUR_CHECKPOINTPATH
```

#### 2. Testing Configuration

Our well-trained model is stored in [Google Drive](https://drive.google.com/drive/folders/1-zu75yJucsX8d6FrR9xku3-AyRa5US6n?usp=sharing). After downloading, please change the file path in the corresponding code.

```
python Test.py  --testsize YOUR_IMAGESIZE  --pth_path YOUR_CHECKPOINTPATH 
```

#### 3. Evaluation

Change the file path to your GT and testing path, then run it to get your evaluation results.

#### 4. Results download

The prediction results of our DRRNet are stored on [Google Drive](https://drive.google.com/drive/folders/1aVS-cN0iUUzKF3puXVLrz99ITREMdMiN?usp=drive_link) please check.



## Concat

If you have any questions, please feel free to contact me via email at jianlinsun@seu.edu.cn
