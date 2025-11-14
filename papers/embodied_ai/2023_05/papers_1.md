# embodied ai - 2023_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17066v1">Mindstorms in Natural Language-Based Societies of Mind</a></div>
    <div class="paper-meta">
      📅 2023-05-26
      | 💬 9 pages in main text + 7 pages of references + 38 pages of appendices, 14 figures in main text + 13 in appendices, 7 tables in appendices
    </div>
    <details class="paper-abstract">
      Both Minsky's "society of mind" and Schmidhuber's "learning to think" inspire diverse societies of large multimodal neural networks (NNs) that solve problems by interviewing each other in a "mindstorm." Recent implementations of NN-based societies of minds consist of large language models (LLMs) and other NN-based experts communicating through a natural language interface. In doing so, they overcome the limitations of single LLMs, improving multimodal zero-shot reasoning. In these natural language-based societies of mind (NLSOMs), new agents -- all communicating through the same universal symbolic language -- are easily added in a modular fashion. To demonstrate the power of NLSOMs, we assemble and experiment with several of them (having up to 129 members), leveraging mindstorms in them to solve some practical AI tasks: visual question answering, image captioning, text-to-image synthesis, 3D generation, egocentric retrieval, embodied AI, and general language-based task solving. We view this as a starting point towards much larger NLSOMs with billions of agents-some of which may be humans. And with this emergence of great societies of heterogeneous minds, many new research questions have suddenly become paramount to the future of artificial intelligence. What should be the social structure of an NLSOM? What would be the (dis)advantages of having a monarchical rather than a democratic structure? How can principles of NN economies be used to maximize the total reward of a reinforcement learning NLSOM? In this work, we identify, discuss, and try to answer some of these questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16925v1">How To Not Train Your Dragon: Training-free Embodied Object Goal Navigation with Semantic Frontiers</a></div>
    <div class="paper-meta">
      📅 2023-05-26
      | 💬 Accepted by/To be published in Robotics: Science and Systems (RSS) 2023; 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Object goal navigation is an important problem in Embodied AI that involves guiding the agent to navigate to an instance of the object category in an unknown environment -- typically an indoor scene. Unfortunately, current state-of-the-art methods for this problem rely heavily on data-driven approaches, \eg, end-to-end reinforcement learning, imitation learning, and others. Moreover, such methods are typically costly to train and difficult to debug, leading to a lack of transferability and explainability. Inspired by recent successes in combining classical and learning methods, we present a modular and training-free solution, which embraces more classic approaches, to tackle the object goal navigation problem. Our method builds a structured scene representation based on the classic visual simultaneous localization and mapping (V-SLAM) framework. We then inject semantics into geometric-based frontier exploration to reason about promising areas to search for a goal object. Our structured scene representation comprises a 2D occupancy map, semantic point cloud, and spatial scene graph. Our method propagates semantics on the scene graphs based on language priors and scene statistics to introduce semantic knowledge to the geometric frontiers. With injected semantic priors, the agent can reason about the most promising frontier to explore. The proposed pipeline shows strong experimental performance for object goal navigation on the Gibson benchmark dataset, outperforming the previous state-of-the-art. We also perform comprehensive ablation studies to identify the current bottleneck in the object navigation task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2202.00199v2">RFUniverse: A Multiphysics Simulation Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-05-16
      | 💬 Project page: https://sites.google.com/view/rfuniverse
    </div>
    <details class="paper-abstract">
      Multiphysics phenomena, the coupling effects involving different aspects of physics laws, are pervasive in the real world and can often be encountered when performing everyday household tasks. Intelligent agents which seek to assist or replace human laborers will need to learn to cope with such phenomena in household task settings. To equip the agents with such kind of abilities, the research community needs a simulation environment, which will have the capability to serve as the testbed for the training process of these intelligent agents, to have the ability to support multiphysics coupling effects. Though many mature simulation software for multiphysics simulation have been adopted in industrial production, such techniques have not been applied to robot learning or embodied AI research. To bridge the gap, we propose a novel simulation environment named RFUniverse. This simulator can not only compute rigid and multi-body dynamics, but also multiphysics coupling effects commonly observed in daily life, such as air-solid interaction, fluid-solid interaction, and heat transfer. Because of the unique multiphysics capacities of this simulator, we can benchmark tasks that involve complex dynamics due to multiphysics coupling effects in a simulation environment before deploying to the real world. RFUniverse provides multiple interfaces to let the users interact with the virtual world in various ways, which is helpful and essential for learning, planning, and control. We benchmark three tasks with reinforcement learning, including food cutting, water pushing, and towel catching. We also evaluate butter pushing with a classic planning-control paradigm. This simulator offers an enhancement of physics simulation in terms of the computation of multiphysics coupling effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2202.00199v2">RFUniverse: A Multiphysics Simulation Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-05-14
      | 💬 Project page: https://sites.google.com/view/rfuniverse
    </div>
    <details class="paper-abstract">
      Multiphysics phenomena, the coupling effects involving different aspects of physics laws, are pervasive in the real world and can often be encountered when performing everyday household tasks. Intelligent agents which seek to assist or replace human laborers will need to learn to cope with such phenomena in household task settings. To equip the agents with such kind of abilities, the research community needs a simulation environment, which will have the capability to serve as the testbed for the training process of these intelligent agents, to have the ability to support multiphysics coupling effects. Though many mature simulation software for multiphysics simulation have been adopted in industrial production, such techniques have not been applied to robot learning or embodied AI research. To bridge the gap, we propose a novel simulation environment named RFUniverse. This simulator can not only compute rigid and multi-body dynamics, but also multiphysics coupling effects commonly observed in daily life, such as air-solid interaction, fluid-solid interaction, and heat transfer. Because of the unique multiphysics capacities of this simulator, we can benchmark tasks that involve complex dynamics due to multiphysics coupling effects in a simulation environment before deploying to the real world. RFUniverse provides multiple interfaces to let the users interact with the virtual world in various ways, which is helpful and essential for learning, planning, and control. We benchmark three tasks with reinforcement learning, including food cutting, water pushing, and towel catching. We also evaluate butter pushing with a classic planning-control paradigm. This simulator offers an enhancement of physics simulation in terms of the computation of multiphysics coupling effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2202.00199v2">RFUniverse: A Multiphysics Simulation Platform for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-05-14
      | 💬 Project page: https://sites.google.com/view/rfuniverse
    </div>
    <details class="paper-abstract">
      Multiphysics phenomena, the coupling effects involving different aspects of physics laws, are pervasive in the real world and can often be encountered when performing everyday household tasks. Intelligent agents which seek to assist or replace human laborers will need to learn to cope with such phenomena in household task settings. To equip the agents with such kind of abilities, the research community needs a simulation environment, which will have the capability to serve as the testbed for the training process of these intelligent agents, to have the ability to support multiphysics coupling effects. Though many mature simulation software for multiphysics simulation have been adopted in industrial production, such techniques have not been applied to robot learning or embodied AI research. To bridge the gap, we propose a novel simulation environment named RFUniverse. This simulator can not only compute rigid and multi-body dynamics, but also multiphysics coupling effects commonly observed in daily life, such as air-solid interaction, fluid-solid interaction, and heat transfer. Because of the unique multiphysics capacities of this simulator, we can benchmark tasks that involve complex dynamics due to multiphysics coupling effects in a simulation environment before deploying to the real world. RFUniverse provides multiple interfaces to let the users interact with the virtual world in various ways, which is helpful and essential for learning, planning, and control. We benchmark three tasks with reinforcement learning, including food cutting, water pushing, and towel catching. We also evaluate butter pushing with a classic planning-control paradigm. This simulator offers an enhancement of physics simulation in terms of the computation of multiphysics coupling effects.
    </details>
</div>
