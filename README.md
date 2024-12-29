# EV load forecasting using a refined CNN-LSTM-AM
[2024-09-15] Paper:[EV load forecasting using a refined CNN-LSTM-AM](https://www.sciencedirect.com/science/article/pii/S0378779624009763)
A new method of combining different interval sequences to reconstruct the time series matrix. 

![image](https://github.com/user-attachments/assets/24583c41-779d-4972-b249-fab334efd2e6)
## Training

### Fast setup: training DINOv2 ViT-L/16 on ImageNet-1k

To train CNN-LSTM-AM:

```shell
python train.py -e 1 -is 96 -bs 288 -os 12 -k 15 -kernel 3
```

To evaluate:

```shell
python evaluate.py -is 96 -bs 288 -os 12 -k 15 -kernel 3
```
