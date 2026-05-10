# embodied ai - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06234v1">RobotEQ: Transitioning from Passive Intelligence to Active Intelligence in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Embodied AI is a prominent research topic in both academia and industry. Current research centers on completing tasks based on explicit user instructions. However, for robots to integrate into human society, they must understand which actions are permissible and which are prohibited, even without explicit commands. We refer to the user-guided AI as passive intelligence and the unguided AI as active intelligence. This paper introduces RobotEQ, the first benchmark for active intelligence, aiming to assess whether existing models can comprehend and adhere to social norms in embodied scenarios. First, we construct RobotEQ-Data, a dataset consisting of 1,900 egocentric images, spanning 10 representative embodied categories and 56 subcategories. Through extensive manual annotation, we provide 5,353 action judgment questions and 1,286 spatial grounding questions, specifying appropriate robot actions across diverse scenarios. Furthermore, we establish RobotEQ-Bench to evaluate the performance of state-of-the-art models on this task. Experimental results show that current models still fall short in achieving reliable active intelligence, particularly in spatial grounding. Meanwhile, we observe that leveraging RAG techniques to incorporate external social norm knowledge bases can generally enhance performance. This work can facilitate the transition of robotics from user-guided passive manipulation to active social compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05756v1">MaMi-HOI: Harmonizing Global Kinematics and Local Geometry for Human-Object Interaction Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-07
    </div>
    <details class="paper-abstract">
      Generating realistic 3D Human-Object Interactions (HOI) is a fundamental task for applications ranging from embodied AI to virtual content creation, which requires harmonizing high-level semantic intent with strict low-level physical constraints. Existing methods excel at semantic alignment, however, they struggle to maintain precise object contact. We reveal a key finding termed \textit{Geometric Forgetting}: as diffusion model depth increases, semantic feature tend to overshadow object geometry feature, causing the model to lose its perception to object geometry. To address this, we propose MaMi-HOI, a hierarchical framework reconciling \textbf{Ma}cro-level kinematic fluidity with \textbf{Mi}cro-level spatial precision. First, to counteract geometric forgetting, we introduce the Geometry-Aware Proximity Adapter (GAPA), which explicitly re-injects dense object details to perform residual snapping corrections for precise contact. Nevertheless, such aggressive local enforcement can disrupt global dynamics, leading to robotic stiffness. In response, we introduce the Kinematic Harmony Adapter (KHA), which proactively aligns whole-body posture with spatial objectives, ensuring the skeleton actively accommodates constraints without compromising naturalness. Extensive experiments validate that MaMi-HOI simultaneously achieves natural motion and precise contact. Crucially, it extends generation capabilities to long-term tasks with complex trajectories, effectively bridging the gap between global navigation and high-fidelity manipulation in 3D scenes. Code is available at https://github.com/DON738110198/MaMi-HOI.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05163v1">PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted by ICML 2026. Project Page: https://hku-mmlab.github.io/PhysForge/
    </div>
    <details class="paper-abstract">
      Synthesizing physics-grounded 3D assets is a critical bottleneck for interactive virtual worlds and embodied AI. Existing methods predominantly focus on static geometry, overlooking the functional properties essential for interaction. We propose that interactive asset generation must be rooted in functional logic and hierarchical physics. To bridge this gap, we introduce PhysForge, a decoupled two-stage framework supported by PhysDB, a large-scale dataset of 150,000 assets with four-tier physical annotations. First, a VLM acts as a "physical architect" to plan a "Hierarchical Physical Blueprint" defining material, functional, and kinematic constraints. Second, a physics-grounded diffusion model realizes this blueprint by synthesizing high-fidelity geometry alongside precise kinematic parameters via a novel KineVoxel Injection (KVI) mechanism. Experiments demonstrate that PhysForge produces functionally plausible, simulation-ready assets, providing a robust data engine for interactive 3D content and embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.05017v1">Position: Embodied AI Requires a Privacy-Utility Trade-off</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 Accepted at ICML 2026. 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Embodied AI (EAI) systems are rapidly transitioning from simulations into real-world domestic and other sensitive environments. However, recent EAI solutions have largely demonstrated advancements within isolated stages such as instruction, perception, planning and interaction, without considering their coupled privacy implications in high-frequency deployments where privacy leakage is often irreversible. This position paper argues that optimizing these components independently creates a systemic privacy crisis when deployed in sensitive settings, thereby advancing the position that privacy in EAI is a life cycle-level architectural constraint rather than a stage-local feature. To address these challenges, we propose Secure Privacy Integration in Next-generation Embodied AI (SPINE), a unified privacy-aware framework that treats privacy as a dynamic control signal governing cross-stage coupling throughout the entire EAI life cycle. SPINE decomposes the EAI pipeline into various stages and establishes a multi-criterion privacy classification matrix to orchestrate contextual sensitivity across stage boundaries. We conduct preliminary simulation and real-world case studies to conceptually validate how privacy constraints propagate downstream to reshape system behavior, illustrating the insufficiency of fragmented privacy patches and motivating future research directions into secure yet functional embodied AI systems. We detail the SPINE framework and case studies at https://github.com/rminshen03/EAI_Privacy_Position.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26509v2">3D Generation for Embodied AI and Robotic Simulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-05-06
      | 💬 27 pages, 11 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey reviews 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In Data Generator, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in Simulation Environments, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in Sim2Real Bridge, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at https://3dgen4robot.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10070v2">AhaRobot: A Low-Cost Open-Source Bimanual Mobile Manipulator for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-05
      | 💬 The first two authors contributed equally. Website: https://aha-robot.github.io
    </div>
    <details class="paper-abstract">
      Scaling Vision-Language-Action models for embodied manipulation demands large volumes of diverse manipulation data, yet the high cost of commercial mobile manipulators and teleoperation interfaces that are difficult to deploy at scale remain key bottlenecks. We present AhaRobot, a low-cost, fully open-source bimanual mobile manipulator tailored for Embodied-AI. The system contributes: (1) a SCARA-like dual-arm hardware design that reduces motor torque demands while maintaining a large vertical reachable workspace, (2) an optimized control stack that improves precision via dual-motor backlash mitigation and static-friction compensation through dithering, and (3) RoboPilot, a teleoperation interface featuring a novel 26-faced marker handle for precise, long-horizon remote data collection. Experimental results show that our hardware-control co-design achieves 0.7 mm repeatability at a total hardware cost of only $1,000. The proposed 26-faced handle reduces tracking error by 80% over a 6-faced baseline and improves data-collection efficiency by 30%, while robustly handling singularities and supporting extremely long-horizon tasks in fully remote settings. Despite its low cost, AhaRobot enables imitation learning of complex household behaviors involving bimanual coordination, upper-body mobility, and contact-rich interaction, with data quality comparable to VR-based collection. All software, CAD files, and documentation are available at https://aha-robot.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02357v1">Channel-Level Relation to Attentive Aggregation with Neighborhood-Homogeneity Constraint for Point Cloud Analysis</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      In 3D point cloud understanding, the core challenge lies in accurately capturing discriminative features within complex neighborhoods, which directly affects the execution precision of downstream tasks such as embodied AI and autonomous driving. Existing methods explore feature correlation discrimination but are limited to point-level spatial distribution or channel responses, enabling only coarse-grained level evaluation. For modern multi-scale point cloud networks, such coarse-grained metrics inevitably incur significant information loss in deeper layers. To address this issue, we propose a novel network equipped with a channel-level metric-based enhancement mechanism, termed the PointCRA network. Our core idea is to introduce temporal trend variation as a new evaluation dimension to avoid the information loss caused by weight dimension collapse in existing spatial and channel attention mechanisms. On this basis, we construct a multi-level calibration framework guided by neighborhood homogeneity for weight calibration, and design a dedicated loss function to enhance channel discriminability. The module effectively leverages the intrinsic feature priors of deep networks to adaptively correct the feature aggregation process, offering strong interpretability with low parameter overhead. Furthermore, our proposed method exhibits strong transferability, interpretability, and parameter efficiency. We validate the proposed method effectiveness on diverse datasets and benchmark models, and further demonstrate its rationality through extensive analytical experiments. Our PointCRA achieves 77.5% mIoU on the S3DIS dataset, 90.4% OA on the ScanObjectNN dataset, and 87.4% instance mIoU on the ShapeNetPart dataset. The code and pretrained weights are publicly available on GitHub:
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28489v2">Video Generation Models as World Models: Efficient Paradigms, Architectures and Algorithms</a></div>
    <div class="paper-meta">
      📅 2026-05-04
    </div>
    <details class="paper-abstract">
      The rapid evolution of video generation has enabled models to simulate complex physical dynamics and long-horizon causalities, positioning them as potential world simulators. However, a critical gap still remains between the theoretical capacity for world simulation and the heavy computational costs of spatiotemporal modeling. To address this, we comprehensively and systematically review video generation frameworks and techniques that consider efficiency as a crucial requirement for practical world modeling. We introduce a novel taxonomy in three dimensions: efficient modeling paradigms, efficient network architectures, and efficient inference algorithms. We further show that bridging this efficiency gap directly empowers interactive applications such as autonomous driving, embodied AI, and game simulation. Finally, we identify emerging research frontiers in efficient video-based world modeling, arguing that efficiency is a fundamental prerequisite for evolving video generators into general-purpose, real-time, and robust world simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01799v1">Embody4D: A Generalist 4D World Model for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-03
    </div>
    <details class="paper-abstract">
      World models have made significant progress in modeling dynamic environments; however, most embodied world models are still restricted to 2D representations, lacking the comprehensive multi-view information essential for embodied spatial reasoning. Bridging this gap is non-trivial, primarily due to challenges from severe scarcity of paired multi-view data, the difficulty of maintaining spatiotemporal consistency in generated 3D geometries, and the tendency to hallucinate manipulation details. To address these challenges, we propose Embody4D, a dedicated video-to-video world model for embodied scenarios, capable of synthesizing arbitrary novel views from a monocular video. First, to tackle data scarcity, we introduce a 3D-aware compositional synthesis pipeline to curate a heterogeneous dataset compositing cross-embodiment robotic arms with diverse backgrounds, guaranteeing broad generalization. Second, to enforce geometric stability, we devise an adaptive noise injection strategy; by leveraging confidence disparities across image regions, this method selectively regularizes the diffusion process to ensure strict spatiotemporal consistency. Finally, to guarantee manipulation fidelity, we incorporate an interaction-aware attention mechanism that explicitly attends to the robotic interaction regions. Extensive experiments demonstrate that Embody4D achieves state-of-the-art performance, serving as a robust world model that synthesizes high-fidelity, view-consistent videos to empower downstream robotic planning and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01352v1">VUDA: Breaking CUDA-Vulkan Isolation for Spatial Sharing of Compute and Graphics on the Same GPU</a></div>
    <div class="paper-meta">
      📅 2026-05-02
    </div>
    <details class="paper-abstract">
      GPU-based simulation environments for embodied AI interleave physics simulation (CUDA) and photorealistic rendering (Vulkan) on a single device. We observe that two foundational scenarios -- simulation data generation and RL training -- can be naturally adapted to execute their simulation and rendering phases concurrently, presenting a significant opportunity to improve GPU utilization through spatial multiplexing. However, a fundamental obstacle we term execution isolation prevents this: CUDA and Vulkan create separate GPU contexts whose channels are bound to different scheduling groups, confining compute and graphics to mutually exclusive time slices. Existing spatial-sharing techniques are limited to the CUDA ecosystem, while temporal-sharing approaches underutilize available resources. This paper presents VUDA, a system that breaks execution isolation to enable spatial parallelism between CUDA compute and Vulkan graphics workloads. VUDA is built on two key observations: although CUDA and Vulkan expose different programming abstractions, their execution paths converge to a common channel primitive at the driver and hardware level; meanwhile, their virtual-address spaces are inherently disjoint, making safe page-table merging feasible without remapping. VUDA exposes a thin API for developers to annotate co-schedulable CUDA streams, and realizes spatial sharing through channel redirection into Vulkan's scheduling domain and page-table grafting to unify address spaces, eliminating all data copying on the critical path. Experiments on representative embodied-AI workloads show that VUDA delivers up to 85% higher throughput than temporal-sharing baselines, while improving GPU utilization and reducing end-to-end latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.00970v1">Split and Aggregation Learning for Foundation Models Over Mobile Embodied AI Network (MEAN): A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2026-05-01
    </div>
    <details class="paper-abstract">
      The rapid advancements in foundation models and sixth-generation (6G) wireless communication systems necessitate the development of efficient, scalable, and privacy-preserving machine learning approaches. For foundation models in 6G, split learning (SL) and aggregation learning (AL) have emerged as promising paradigms that address key challenges in distributed artificial intelligence (AI), such as communication efficiency, resource allocation, and data privacy. SL enables multiple entities to collaboratively train deep learning models by partitioning neural networks, while AL focuses on aggregating intermediate results or model updates from multiple participants, improving robustness, optimizing resource utilization, and mitigating data leakage risks. Specifically, SL is ideal for scenarios requiring strict data isolation (e.g., vertical collaborations), whereas AL suits homogeneous horizontal data settings; they can be combined to balance privacy and communication efficiency. This survey provides a comprehensive analysis of SL and AL in 6G communication systems, exploring their architectures, technical methodologies, and integration with AI-native 6G communication technologies. We examine different SL configurations, aggregation techniques, and their roles in optimizing distributed foundation models. Furthermore, we discuss their applications in emerging wireless networks, including semantic communication, reconfigurable intelligent surfaces (RIS), space-air-ground integrated networks (SAGINs), and quantum communication. By analyzing the impact of SL and AL, this survey provides insights into their role in shaping distributed AI-driven communication systems in the 6G era, focusing on efficiency, privacy preservation, and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.14093v8">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-01
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a cornerstone of artificial general intelligence (AGI) because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models (LLMs) and vision-language models (VLMs), a new category of multimodal models -- referred to as vision-language-action (VLA) models -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges facing VLAs and outline promising future directions in embodied AI. A curated repository associated with this survey is available at: https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
