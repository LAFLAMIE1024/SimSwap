# SimSwap: An Efficient Framework For High Fidelity Face Swapping
## Proceedings of the 28th ACM International Conference on Multimedia
**The official repository with Pytorch**

**Our method can realize **arbitrary face swapping** on images and videos with **one single trained model**.**

Training and test code are now available!
[ <a href="https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/train.ipynb)

We are working with our incoming paper SimSwap++, keeping expecting!

The high resolution version of ***SimSwap-HQ*** is supported!

[![simswaplogo](/docs/img/logo1.png)](https://github.com/neuralchen/SimSwap)

Our paper can be downloaded from [[Arxiv]](https://arxiv.org/pdf/2106.06340v1.pdf) [[ACM DOI]](https://dl.acm.org/doi/10.1145/3394171.3413630) 

<!-- [[Google Drive]](https://drive.google.com/file/d/1fcfWOGt1mkBo7F0gXVKitf8GJMAXQxZD/view?usp=sharing) 
[[Baidu Drive ]](https://pan.baidu.com/s/1-TKFuycRNUKut8hn4IimvA) Password: ```ummt``` -->

## Attention
***This project is for technical and academic use only. Please do not apply it to illegal and unethical scenarios.***

***In the event of violation of the legal and ethical requirements of the user's country or region, this code repository is exempt from liability***

***Please do not ignore the content at the end of this README!***

If you find this project useful, please star it. It is the greatest appreciation of our work.


## The first open source high resolution dataset for face swapping!!!
## High Resolution Dataset [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ)

## Dependencies
- python3.6+
- pytorch1.5+
- torchvision
- opencv
- pillow
- numpy
- imageio
- moviepy
- insightface

## Training

[Preparation](./docs/guidance/preparation.md)

The training script is slightly different from the original version, e.g., we replace the patch discriminator with the projected discriminator, which saves a lot of hardware overhead and achieves slightly better results.

In order to ensure the normal training, the batch size must be greater than 1.

Friendly reminder, due to the difference in training settings, the user-trained model will have subtle differences in visual effects from the pre-trained model we provide.

- Train 224 models with VGGFace2 224*224 [[Google Driver] VGGFace2-224 (10.8G)](https://drive.google.com/file/d/19pWvdEHS-CEG6tW3PdxdtZ5QEymVjImc/view?usp=sharing) [[Baidu Driver] ](https://pan.baidu.com/s/1OiwLJHVBSYB4AY2vEcfN0A) [Password: lrod]

For faster convergence and better results, a large batch size (more than 16) is recommended!

***We recommend training more than 400K iterations (batch size is 16), 600K~800K will be better, more iterations will not be recommended.***


```
python train.py --name simswap224_test --batchSize 8  --gpu_ids 0 --dataset /path/to/VGGFace2HQ --Gdeep False
```

[Colab demo for training 224 model](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/train.ipynb) 

For faster convergence and better results, a large batch size (more than 16) is recommended!

- Train 512 models with VGGFace2-HQ 512*512 [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ).
```
python train.py --name simswap512_test  --batchSize 16  --gpu_ids 0 --dataset /path/to/VGGFace2HQ --Gdeep True
```



## Inference with a pretrained SimSwap model
[Preparation](./docs/guidance/preparation.md)

[Inference for image or video face swapping](./docs/guidance/usage.md)

[Colab demo](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/SimSwap%20colab.ipynb)

[Colab for switching specific faces in multi-face videos](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/MultiSpecific.ipynb) 

[Image face swapping demo & Docker image on Replicate](https://replicate.ai/neuralchen/simswap-image)


## License
For academic and non-commercial use only.The whole project is under the CC-BY-NC 4.0 license. See [LICENSE](https://github.com/neuralchen/SimSwap/blob/main/LICENSE) for additional details.
