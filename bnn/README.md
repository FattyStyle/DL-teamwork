# BNN
Binarized Neural Network (BNN) for pytorch

Refer to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

Code is based on https://github.com/itayhubara/BinaryNet.pytorch



# Quick Start

## Train BNN

```bash
cd bnn
python main_binary.py --results_dir "results" --save "vgg_cifar10_binary" --model "vgg_cifar10_binary" --dataset "cifar10" --data_path "../../Datasets."
```



## Damage Test

```bash
cd bnn
python damage_test.py -e "results/vgg_cifar10_binary/model_best.pth.tar" --data_path "../../Datasets/"
```



# Result

## BNN Damage

### VGG for Cifar10

```
original model ==> loss:{0.513}, acc:{0.917}
damaged model ==>
damage probability:  0.01
damaged loss:{1.213}, damaged acc:{0.662}, loss rate:{2.365}, acc rate:{0.722}
clipped loss:{1.743}, clipped acc:{0.482}, loss rate:{3.400}, acc rate:{0.525}
damage probability:  0.0001
damaged loss:{0.515}, damaged acc:{0.911}, loss rate:{1.003}, acc rate:{0.993}
clipped loss:{1.303}, clipped acc:{0.628}, loss rate:{2.541}, acc rate:{0.685}
damage probability:  1e-06
damaged loss:{0.512}, damaged acc:{0.917}, loss rate:{0.999}, acc rate:{1.000}
clipped loss:{1.294}, clipped acc:{0.632}, loss rate:{2.524}, acc rate:{0.689}
damage probability:  1e-08
damaged loss:{0.512}, damaged acc:{0.917}, loss rate:{0.998}, acc rate:{1.001}
clipped loss:{1.294}, clipped acc:{0.637}, loss rate:{2.523}, acc rate:{0.695}
```

### ResNet for Cifar10

```
original model ==> loss:{0.514}, acc:{0.916}
damaged model ==>
damage probability:  0.01
damaged loss:{1.073}, damaged acc:{0.720}, loss rate:{2.087}, acc rate:{0.786}
clipped loss:{2.088}, clipped acc:{0.277}, loss rate:{4.062}, acc rate:{0.302}
damage probability:  0.0001
damaged loss:{0.517}, damaged acc:{0.915}, loss rate:{1.005}, acc rate:{0.998}
clipped loss:{1.581}, clipped acc:{0.493}, loss rate:{3.075}, acc rate:{0.538}
damage probability:  1e-06
damaged loss:{0.514}, damaged acc:{0.916}, loss rate:{1.000}, acc rate:{1.000}
clipped loss:{1.532}, clipped acc:{0.519}, loss rate:{2.981}, acc rate:{0.566}
damage probability:  1e-08
damaged loss:{0.514}, damaged acc:{0.916}, loss rate:{0.999}, acc rate:{1.000}
clipped loss:{1.534}, clipped acc:{0.519}, loss rate:{2.985}, acc rate:{0.567}
```



# Conclusion

## BNN Training

 - BatchNorm对BNN至关重要，特别是最后一层。
 - Weight Decay = 0至关重要。
 - BNN训练需要较大的学习率，否则梯度的更新对sign没有作用。