# embodied ai - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05863v1">Constrained Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04411v1">Self-evolving Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) is an intelligent system formed by agents and their environment through active perception, embodied cognition, and action interaction. Existing embodied AI remains confined to human-crafted setting, in which agents are trained on given memory and construct models for given tasks, enabling fixed embodiments to interact with relatively static environments. Such methods fail in in-the-wild setting characterized by variable embodiments and dynamic open environments. This paper introduces self-evolving embodied AI, a new paradigm in which agents operate based on their changing state and environment with memory self-updating, task self-switching, environment self-prediction, embodiment self-adaptation, and model self-evolution, aiming to achieve continually adaptive intelligence with autonomous evolution. Specifically, we present the definition, framework, components, and mechanisms of self-evolving embodied AI, systematically review state-of-the-art works for realized components, discuss practical applications, and point out future research directions. We believe that self-evolving embodied AI enables agents to autonomously learn and interact with environments in a human-like manner and provide a new perspective toward general artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.14093v7">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a cornerstone of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges facing VLAs and outline promising future directions in embodied AI. A curated repository associated with this survey is available at: https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00611v2">Structured Self-Consistency:A Multi-Task Evaluation of LLMs on VirtualHome</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Embodied AI requires agents to understand goals, plan actions, and execute tasks in simulated environments. We present a comprehensive evaluation of Large Language Models (LLMs) on the VirtualHome benchmark using the Embodied Agent Interface (EAI) framework. We compare two representative 7B-parameter models OPENPANGU-7B and QWEN2.5-7B across four fundamental tasks: Goal Interpretation, Action Sequencing, Subgoal Decomposition, and Transition Modeling. We propose Structured Self-Consistency (SSC), an enhanced decoding strategy that leverages multiple sampling with domain-specific voting mechanisms to improve output quality for structured generation tasks. Experimental results demonstrate that SSC significantly enhances performance, with OPENPANGU-7B excelling at hierarchical planning while QWEN2.5-7B show advantages in action-level tasks. Our analysis reveals complementary strengths across model types, providing insights for future embodied AI system development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03200v1">Hand3R: Online 4D Hand-Scene Reconstruction in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      For Embodied AI, jointly reconstructing dynamic hands and the dense scene context is crucial for understanding physical interaction. However, most existing methods recover isolated hands in local coordinates, overlooking the surrounding 3D environment. To address this, we present Hand3R, the first online framework for joint 4D hand-scene reconstruction from monocular video. Hand3R synergizes a pre-trained hand expert with a 4D scene foundation model via a scene-aware visual prompting mechanism. By injecting high-fidelity hand priors into a persistent scene memory, our approach enables simultaneous reconstruction of accurate hand meshes and dense metric-scale scene geometry in a single forward pass. Experiments demonstrate that Hand3R bypasses the reliance on offline optimization and delivers competitive performance in both local hand reconstruction and global positioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01667v3">What does really matter in image goal navigation?</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In this large experimental study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extent. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03054v1">Towards Considerate Embodied AI: Co-Designing Situated Multi-Site Healthcare Robots from Abstract Concepts to High-Fidelity Prototypes</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 To appear in Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI 2026)
    </div>
    <details class="paper-abstract">
      Co-design is essential for grounding embodied artificial intelligence (AI) systems in real-world contexts, especially high-stakes domains such as healthcare. While prior work has explored multidisciplinary collaboration, iterative prototyping, and support for non-technical participants, few have interwoven these into a sustained co-design process. Such efforts often target one context and low-fidelity stages, limiting the generalizability of findings and obscuring how participants' ideas evolve. To address these limitations, we conducted a 14-week workshop with a multidisciplinary team of 22 participants, centered around how embodied AI can reduce non-value-added task burdens in three healthcare settings: emergency departments, long-term rehabilitation facilities, and sleep disorder clinics. We found that the iterative progression from abstract brainstorming to high-fidelity prototypes, supported by educational scaffolds, enabled participants to understand real-world trade-offs and generate more deployable solutions. We propose eight guidelines for co-designing more considerate embodied AI: attuned to context, responsive to social dynamics, mindful of expectations, and grounded in deployment. Project Page: https://byc-sophie.github.io/Towards-Considerate-Embodied-AI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02787v1">Real-World Applications of AI in LTE and 5G-NR Network Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 6 pages and 3 figures
    </div>
    <details class="paper-abstract">
      Telecommunications networks generate extensive performance and environmental telemetry, yet most LTE and 5G-NR deployments still rely on static, manually engineered configurations. This limits adaptability in rural, nomadic, and bandwidth-constrained environments where traffic distributions, propagation characteristics, and user behavior fluctuate rapidly. Artificial Intelligence (AI), more specifically Machine Learning (ML) models, provide new opportunities to transition Radio Access Networks (RANs) from rigid, rule-based systems toward adaptive, self-optimizing infrastructures that can respond autonomously to these dynamics. This paper proposes a practical architecture incorporating AI-assisted planning, reinforcement-learning-based RAN optimization, real-time telemetry analytics, and digital-twin-based validation. In parallel, the paper addresses the challenge of delivering embodied-AI healthcare services, educational tools, and large language model (LLM) applications to communities with insufficient backhaul for cloud computing. We introduce an edge-hosted execution model in which applications run directly on LTE/5G-NR base stations using containers, reducing latency and bandwidth consumption while improving resilience. Together, these contributions demonstrate how AI can enhance network performance, reduce operational overhead, and expand access to advanced digital services, aligning with broader goals of sustainable and inclusive network development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20377v2">RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.
    </details>
</div>
