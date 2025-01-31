# embodied ai - 2024_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.03696v3">MOPA: Modular Object Navigation with PointGoal Agents</a></div>
    <div class="paper-meta">
      📅 2024-01-27
    </div>
    <details class="paper-abstract">
      We propose a simple but effective modular approach MOPA (Modular ObjectNav with PointGoal agents) to systematically investigate the inherent modularity of the object navigation task in Embodied AI. MOPA consists of four modules: (a) an object detection module trained to identify objects from RGB images, (b) a map building module to build a semantic map of the observed objects, (c) an exploration module enabling the agent to explore the environment, and (d) a navigation module to move to identified target objects. We show that we can effectively reuse a pretrained PointGoal agent as the navigation model instead of learning to navigate from scratch, thus saving time and compute. We also compare various exploration strategies for MOPA and find that a simple uniform strategy significantly outperforms more advanced exploration methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14349v1">Learning to navigate efficiently and precisely in real environments</a></div>
    <div class="paper-meta">
      📅 2024-01-25
    </div>
    <details class="paper-abstract">
      In the context of autonomous navigation of terrestrial robots, the creation of realistic models for agent dynamics and sensing is a widespread habit in the robotics literature and in commercial applications, where they are used for model based control and/or for localization and mapping. The more recent Embodied AI literature, on the other hand, focuses on modular or end-to-end agents trained in simulators like Habitat or AI-Thor, where the emphasis is put on photo-realistic rendering and scene diversity, but high-fidelity robot motion is assigned a less privileged role. The resulting sim2real gap significantly impacts transfer of the trained models to real robotic platforms. In this work we explore end-to-end training of agents in simulation in settings which minimize the sim2real gap both, in sensing and in actuation. Our agent directly predicts (discretized) velocity commands, which are maintained through closed-loop control in the real robot. The behavior of the real robot (including the underlying low-level controller) is identified and simulated in a modified Habitat simulator. Noise models for odometry and localization further contribute in lowering the sim2real gap. We evaluate on real navigation scenarios, explore different localization and point goal calculation methods and report significant gains in performance and robustness compared to prior work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.03047v3">ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2024-01-22
      | 💬 Project page: https://github.com/MarSaKi/ETPNav
    </div>
    <details class="paper-abstract">
      Vision-language navigation is a task that requires an agent to follow instructions to navigate in environments. It becomes increasingly crucial in the field of embodied AI, with potential applications in autonomous navigation, search and rescue, and human-robot interaction. In this paper, we propose to address a more practical yet challenging counterpart setting - vision-language navigation in continuous environments (VLN-CE). To develop a robust VLN-CE agent, we propose a new navigation framework, ETPNav, which focuses on two critical skills: 1) the capability to abstract environments and generate long-range navigation plans, and 2) the ability of obstacle-avoiding control in continuous environments. ETPNav performs online topological mapping of environments by self-organizing predicted waypoints along a traversed path, without prior environmental experience. It privileges the agent to break down the navigation procedure into high-level planning and low-level control. Concurrently, ETPNav utilizes a transformer-based cross-modal planner to generate navigation plans based on topological maps and instructions. The plan is then performed through an obstacle-avoiding controller that leverages a trial-and-error heuristic to prevent navigation from getting stuck in obstacles. Experimental results demonstrate the effectiveness of the proposed method. ETPNav yields more than 10% and 20% improvements over prior state-of-the-art on R2R-CE and RxR-CE datasets, respectively. Our code is available at https://github.com/MarSaKi/ETPNav.
    </details>
</div>
