# A Learning Paradigm for Selecting Few Discriminative Stimuli in Eye-Tracking Research
This repo will contains the code for our paper *"A Learning Paradigm for Selecting Few Discriminative Stimuli in Eye-Tracking Research"* by Wenqi Zhong, Chen Xia, Linzhi Yu, Kuan Li, Zhongyu Li, Dingwen Zhang and Junwei Han.
> Eye-tracking is a reliable method for quantifying visual information processing and holds significant potential for group recognition, such as identifying autism spectrum disorder (ASD). However, eye-tracking research typically faces the heterogeneity of stimuli and is time-consuming due to the large number of observed stimuli. To address these issues, we first mathematically define the stimulus selection problem and introduce the concept of stimulus discrimination ability to reduce the computational complexity of the solution. Then, we construct a scanpath-based recognition model to mine the stimulus discrimination ability. Specifically, we propose cross-subject entropy and cross-subject divergence scores for quantitatively evaluating stimulus discrimination ability, effectively capturing differences in intra-group collective trends and inter-subject consistency within a group. Furthermore, we propose an iterative learning mechanism that employs stimulus-wise attention to focus on discriminative stimuli for discrimination purification. In the experiment, we construct an ASD eye-tracking dataset with diverse stimulus types and conduct extensive tests on three representative models to validate our approach. Remarkably, our method demonstrates superior performance using only 10 selected stimuli compared to models utilizing 220 stimuli. Additionally, we perform experiments on another eye-tracking task, gender prediction, to further validate our method. We believe that our approach is both simple and flexible for integration into existing models, promoting large-scale ASD screening and extending to other eye-tracking research domains.

## &#x1F527; Usage
### Dependencies
- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Dataset
- Download [Saliency4ASD](https://saliency4asd.ls2n.fr/datasets/) dataset
- Process the Saliency4ASD as 

### Run
- **Stage1** *Proxy Model Training*
  Train the proxy model (i.e., a scanpath-based recognition model) to extract discrimination cues from stimuli.
  ```
  python train_disc.py
  ```
- **Stage2** *Discrimination Score*
  Test the proxy model to obtain the discrimination score and use the iterative learning for discrimination purification.
  ```
  python img_score.py
  ```
- **Stage3** *Iterative Learning for Discrimination Purification*
  Iterative train the proxy model to purify discrimination scores.
  ```
  python train_disc.py
  ```
- **Stage4** *Stimulus Selection Evaluation*
  Re-train the scanpath-based recognition model with the selected stimuli according to discrimination scores. Test the recognition model to verify effectiveness discrimination score.
  ```
  python train_disc.py
  ```
