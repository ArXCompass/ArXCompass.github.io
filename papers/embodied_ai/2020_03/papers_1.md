# embodied ai - 2020_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/1904.12368v2">Towards Efficient Model Compression via Learned Global Ranking</a></div>
    <div class="paper-meta">
      📅 2020-03-14
      | 💬 CVPR 2020 Oral
    </div>
    <details class="paper-abstract">
      Pruning convolutional filters has demonstrated its effectiveness in compressing ConvNets. Prior art in filter pruning requires users to specify a target model complexity (e.g., model size or FLOP count) for the resulting architecture. However, determining a target model complexity can be difficult for optimizing various embodied AI applications such as autonomous robots, drones, and user-facing applications. First, both the accuracy and the speed of ConvNets can affect the performance of the application. Second, the performance of the application can be hard to assess without evaluating ConvNets during inference. As a consequence, finding a sweet-spot between the accuracy and speed via filter pruning, which needs to be done in a trial-and-error fashion, can be time-consuming. This work takes a first step toward making this process more efficient by altering the goal of model compression to producing a set of ConvNets with various accuracy and latency trade-offs instead of producing one ConvNet targeting some pre-defined latency constraint. To this end, we propose to learn a global ranking of the filters across different layers of the ConvNet, which is used to obtain a set of ConvNet architectures that have different accuracy/latency trade-offs by pruning the bottom-ranked filters. Our proposed algorithm, LeGR, is shown to be 2x to 3x faster than prior work while having comparable or better performance when targeting seven pruned ResNet-56 with different accuracy/FLOPs profiles on the CIFAR-100 dataset. Additionally, we have evaluated LeGR on ImageNet and Bird-200 with ResNet-50 and MobileNetV2 to demonstrate its effectiveness. Code available at https://github.com/cmu-enyac/LeGR.
    </details>
</div>
