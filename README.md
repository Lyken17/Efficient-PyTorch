# Efficient-Pytorch
My best practice of training large dataset using PyTorch.

## Speed overview
By following the tips, we can ahieve **~730 images/second** when training ResNet-50 on ImageNet. It is even comparable to [Tensorflow](https://www.tensorflow.org/performance/benchmarks) and [MXNet](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)

```
Epoch: [0][430/5005]    Time 0.409 (0.405)      Data 626.6 (728.0)      Loss 6.8381 (6.9754)    Error@1 100.000 (99.850) Error@5 99.609 (99.259)
Epoch: [0][440/5005]    Time 0.364 (0.404)      Data 704.2 (727.9)      Loss 6.8506 (6.9725)    Error@1 100.000 (99.851) Error@5 99.609 (99.258)
Epoch: [0][450/5005]    Time 0.350 (0.403)      Data 730.7 (727.3)      Loss 6.8846 (6.9700)    Error@1 100.000 (99.847) Error@5 99.609 (99.258)
Epoch: [0][460/5005]    Time 0.357 (0.402)      Data 716.8 (727.4)      Loss 6.9129 (6.9680)    Error@1 100.000 (99.849) Error@5 99.609 (99.256)
Epoch: [0][470/5005]    Time 0.346 (0.401)      Data 740.8 (727.4)      Loss 6.8574 (6.9657)    Error@1 100.000 (99.850) Error@5 98.828 (99.249)
Epoch: [0][480/5005]    Time 0.425 (0.400)      Data 601.8 (727.3)      Loss 6.8467 (6.9632)    Error@1 100.000 (99.849) Error@5 99.609 (99.239)
Epoch: [0][490/5005]    Time 0.358 (0.399)      Data 715.2 (727.2)      Loss 6.8319 (6.9607)    Error@1 100.000 (99.848) Error@5 99.609 (99.232)
Epoch: [0][500/5005]    Time 0.347 (0.399)      Data 737.4 (726.9)      Loss 6.8426 (6.9583)    Error@1 99.609 (99.843)  Error@5 98.047 (99.220)
Epoch: [0][510/5005]    Time 0.346 (0.398)      Data 740.5 (726.7)      Loss 6.8245 (6.9561)    Error@1 100.000 (99.839) Error@5 99.609 (99.211)
Epoch: [0][520/5005]    Time 0.350 (0.452)      Data 730.7 (724.0)      Loss 6.8270 (6.9538)    Error@1 99.609 (99.834)  Error@5 97.656 (99.193)
Epoch: [0][530/5005]    Time 0.340 (0.450)      Data 752.9 (724.4)      Loss 6.8149 (6.9516)    Error@1 100.000 (99.832) Error@5 98.047 (99.183)
Epoch: [0][540/5005]    Time 0.345 (0.449)      Data 743.0 (724.4)      Loss 6.8127 (6.9491)    Error@1 100.000 (99.827) Error@5 99.219 (99.179)
Epoch: [0][550/5005]    Time 0.341 (0.447)      Data 751.6 (724.9)      Loss 6.8608 (6.9469)    Error@1 99.609 (99.824)  Error@5 98.047 (99.170)
Epoch: [0][560/5005]    Time 0.342 (0.445)      Data 748.2 (725.1)      Loss 6.7818 (6.9444)    Error@1 99.609 (99.817)  Error@5 97.266 (99.151)
Epoch: [0][570/5005]    Time 0.339 (0.443)      Data 754.6 (725.4)      Loss 6.8105 (6.9419)    Error@1 99.609 (99.812)  Error@5 99.219 (99.133)
Epoch: [0][580/5005]    Time 0.340 (0.442)      Data 752.7 (725.7)      Loss 6.8213 (6.9391)    Error@1 100.000 (99.806) Error@5 99.219 (99.118)
Epoch: [0][590/5005]    Time 0.342 (0.440)      Data 749.3 (725.9)      Loss 6.7334 (6.9363)    Error@1 98.828 (99.801)  Error@5 98.047 (99.107)
```
