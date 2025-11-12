# embodied ai - 2022_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2204.02587v2">Learning to Anticipate Future with Dynamic Context Removal</a></div>
    <div class="paper-meta">
      📅 2022-06-27
      | 💬 CVPR 2022
    </div>
    <details class="paper-abstract">
      Anticipating future events is an essential feature for intelligent systems and embodied AI. However, compared to the traditional recognition task, the uncertainty of future and reasoning ability requirement make the anticipation task very challenging and far beyond solved. In this filed, previous methods usually care more about the model architecture design or but few attention has been put on how to train an anticipation model with a proper learning policy. To this end, in this work, we propose a novel training scheme called Dynamic Context Removal (DCR), which dynamically schedules the visibility of observed future in the learning procedure. It follows the human-like curriculum learning process, i.e., gradually removing the event context to increase the anticipation difficulty till satisfying the final anticipation target. Our learning scheme is plug-and-play and easy to integrate any reasoning model including transformer and LSTM, with advantages in both effectiveness and efficiency. In extensive experiments, the proposed method achieves state-of-the-art on four widely-used benchmarks. Our code and models are publicly released at https://github.com/AllenXuuu/DCR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2206.11887v1">VRKitchen2.0-IndoorKit: A Tutorial for Augmented Indoor Scene Building in Omniverse</a></div>
    <div class="paper-meta">
      📅 2022-06-23
    </div>
    <details class="paper-abstract">
      With the recent progress of simulations by 3D modeling software and game engines, many researchers have focused on Embodied AI tasks in the virtual environment. However, the research community lacks a platform that can easily serve both indoor scene synthesis and model benchmarking with various algorithms. Meanwhile, computer graphics-related tasks need a toolkit for implementing advanced synthesizing techniques. To facilitate the study of indoor scene building methods and their potential robotics applications, we introduce INDOORKIT: a built-in toolkit for NVIDIA OMNIVERSE that provides flexible pipelines for indoor scene building, scene randomizing, and animation controls. Besides, combining Python coding in the animation software INDOORKIT assists researchers in creating real-time training and controlling avatars and robotics. The source code for this toolkit is available at https://github.com/realvcla/VRKitchen2.0-Tutorial, and the tutorial along with the toolkit is available at https://vrkitchen20-tutorial.readthedocs.io/en/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2206.08500v1">What do navigation agents learn about their environment?</a></div>
    <div class="paper-meta">
      📅 2022-06-17
      | 💬 CVPR 2022
    </div>
    <details class="paper-abstract">
      Today's state of the art visual navigation agents typically consist of large deep learning models trained end to end. Such models offer little to no interpretability about the learned skills or the actions of the agent taken in response to its environment. While past works have explored interpreting deep learning models, little attention has been devoted to interpreting embodied AI systems, which often involve reasoning about the structure of the environment, target characteristics and the outcome of one's actions. In this paper, we introduce the Interpretability System for Embodied agEnts (iSEE) for Point Goal and Object Goal navigation agents. We use iSEE to probe the dynamic representations produced by these agents for the presence of information about the agent as well as the environment. We demonstrate interesting insights about navigation agents using iSEE, including the ability to encode reachable locations (to avoid obstacles), visibility of the target, progress from the initial spawn location as well as the dramatic effect on the behaviors of agents when we mask out critical individual neurons. The code is available at: https://github.com/allenai/iSEE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2206.06994v1">ProcTHOR: Large-Scale Embodied AI Using Procedural Generation</a></div>
    <div class="paper-meta">
      📅 2022-06-15
      | 💬 ProcTHOR website: https://procthor.allenai.org
    </div>
    <details class="paper-abstract">
      Massive datasets and high-capacity models have driven many recent advancements in computer vision and natural language understanding. This work presents a platform to enable similar success stories in Embodied AI. We propose ProcTHOR, a framework for procedural generation of Embodied AI environments. ProcTHOR enables us to sample arbitrarily large datasets of diverse, interactive, customizable, and performant virtual environments to train and evaluate embodied agents across navigation, interaction, and manipulation tasks. We demonstrate the power and potential of ProcTHOR via a sample of 10,000 generated houses and a simple neural model. Models trained using only RGB images on ProcTHOR, with no explicit mapping and no human task supervision produce state-of-the-art results across 6 embodied AI benchmarks for navigation, rearrangement, and arm manipulation, including the presently running Habitat 2022, AI2-THOR Rearrangement 2022, and RoboTHOR challenges. We also demonstrate strong 0-shot results on these benchmarks, via pre-training on ProcTHOR with no fine-tuning on the downstream benchmark, often beating previous state-of-the-art systems that access the downstream training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2206.06489v1">BEHAVIOR in Habitat 2.0: Simulator-Independent Logical Task Description for Benchmarking Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2022-06-15
    </div>
    <details class="paper-abstract">
      Robots excel in performing repetitive and precision-sensitive tasks in controlled environments such as warehouses and factories, but have not been yet extended to embodied AI agents providing assistance in household tasks. Inspired by the catalyzing effect that benchmarks have played in the AI fields such as computer vision and natural language processing, the community is looking for new benchmarks for embodied AI. Prior work in embodied AI benchmark defines tasks using a different formalism, often specific to one environment, simulator or domain, making it hard to develop general and comparable solutions. In this work, we bring a subset of BEHAVIOR activities into Habitat 2.0 to benefit from its fast simulation speed, as a first step towards demonstrating the ease of adapting activities defined in the logic space into different simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2206.06994v1">ProcTHOR: Large-Scale Embodied AI Using Procedural Generation</a></div>
    <div class="paper-meta">
      📅 2022-06-14
      | 💬 ProcTHOR website: https://procthor.allenai.org
    </div>
    <details class="paper-abstract">
      Massive datasets and high-capacity models have driven many recent advancements in computer vision and natural language understanding. This work presents a platform to enable similar success stories in Embodied AI. We propose ProcTHOR, a framework for procedural generation of Embodied AI environments. ProcTHOR enables us to sample arbitrarily large datasets of diverse, interactive, customizable, and performant virtual environments to train and evaluate embodied agents across navigation, interaction, and manipulation tasks. We demonstrate the power and potential of ProcTHOR via a sample of 10,000 generated houses and a simple neural model. Models trained using only RGB images on ProcTHOR, with no explicit mapping and no human task supervision produce state-of-the-art results across 6 embodied AI benchmarks for navigation, rearrangement, and arm manipulation, including the presently running Habitat 2022, AI2-THOR Rearrangement 2022, and RoboTHOR challenges. We also demonstrate strong 0-shot results on these benchmarks, via pre-training on ProcTHOR with no fine-tuning on the downstream benchmark, often beating previous state-of-the-art systems that access the downstream training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2206.06489v1">BEHAVIOR in Habitat 2.0: Simulator-Independent Logical Task Description for Benchmarking Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2022-06-13
    </div>
    <details class="paper-abstract">
      Robots excel in performing repetitive and precision-sensitive tasks in controlled environments such as warehouses and factories, but have not been yet extended to embodied AI agents providing assistance in household tasks. Inspired by the catalyzing effect that benchmarks have played in the AI fields such as computer vision and natural language processing, the community is looking for new benchmarks for embodied AI. Prior work in embodied AI benchmark defines tasks using a different formalism, often specific to one environment, simulator or domain, making it hard to develop general and comparable solutions. In this work, we bring a subset of BEHAVIOR activities into Habitat 2.0 to benefit from its fast simulation speed, as a first step towards demonstrating the ease of adapting activities defined in the logic space into different simulators.
    </details>
</div>
