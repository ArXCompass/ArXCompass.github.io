# embodied ai - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30308v2">The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-07-09
    </div>
    <details class="paper-abstract">
      4D hand motion reconstruction from egocentric video is bottlenecked by clear limitations of existing methods: image-based pipelines depend on a detector that fails under heavy occlusion, while video-based methods rely on temporal modules learned only from scarce hand-pose annotations, a narrow signal insufficient to model motion dynamics, occlusion reasoning, and hand-object interaction. These capabilities, however, are exactly what video generative models must implicitly acquire when trained to synthesize coherent video at internet scale. Motivated by this, we present ViDiHand, which leverages the representations of a pretrained video diffusion model to reconstruct 4D two-hand pose. We adapt it via a hand-overlay rendering objective that specializes its features for hands while preserving its world priors. A decoder then recovers metric-scale pose from the adapted features. The whole pipeline operates directly on full frames--no detector, no infiller, and no test-time optimization. On ARCTIC, HOT3D, and HOI4D, ViDiHand substantially outperforms prior methods, establishing video diffusion models as a powerful new foundation for hand motion reconstruction and a promising route to scalable in-the-wild data collection for embodied AI. Project page: https://vidihand.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07880v1">GIRAF: Towards Generalizable Human Interactions with Articulated Objects</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 12 pages, 6 figures, 3 tables. Accepted at the Third Workshop on Human Motion Generation (HuMoGen), CVPR 2026
    </div>
    <details class="paper-abstract">
      Synthesizing realistic full-body human interactions with articulated objects is a fundamental challenge for embodied AI and graphics, with applications in robotics training and virtual agents. Existing models remain limited: some focus on simple activities with static objects, while others restrict attention to hand-only manipulation. This leaves open the problem of generating coordinated full-body motion that approaches, manipulates, and moves articulated objects in a realistic and generalizable way. The key difficulty lies in reasoning jointly about locomotion, fine-grained contact, and object articulation. Models must capture subtle hand-object correspondences that transfer across object geometries, while also producing seamless transitions from navigation to manipulation. At the same time, the scarcity of large-scale paired motion-scene data makes it difficult to generalize across diverse object positions and shapes. We introduce a text-conditioned diffusion model that addresses these challenges through three core ideas: an object-centric representation that unifies hand-object contact with object surfaces, a mixed-domain training strategy that balances locomotion and interaction, and a contact-based augmentation scheme that expands training diversity. Through experiments, our method demonstrated strong generalization to unseen object configurations, surpassing current state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07459v1">EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-07-08
    </div>
    <details class="paper-abstract">
      We present EmbodiedGen V2, a generative 3D world engine for building executable sim-ready environments for embodied intelligence. Sim-ready 3D asset generation has advanced rapidly, yet assembling such assets into policy-ready task environments remains largely manual, limiting scalable closed-loop learning. EmbodiedGen V2 addresses this gap through a unified sim-ready representation that connects cross-simulator assets, interaction affordances, task-driven worlds, large-scale multi-room scenes, and stateful Vibe Coding into a generative, editable, and reusable simulation pipeline. The generated environments support manipulation, navigation, mobile manipulation, cross-simulator deployment, and embodied policy training. In evaluation, the asset pipeline achieves 96.5% human acceptance and 98.6% collision success, and 83.3% of task-driven worlds are directly usable for downstream simulation without manual modification. Online reinforcement learning with generated environments further improves simulation success from 9.7% to 79.8%, and transfers to real robots with task success increasing from 21.7% to 75.0%. These results establish EmbodiedGen V2 as scalable simulation infrastructure for training, evaluating, and deploying embodied policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07001v1">Ego-Human Motion Prediction with 3D-Aware LLM</a></div>
    <div class="paper-meta">
      📅 2026-07-08
      | 💬 Accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      Anticipating human motion from an egocentric perspective is fundamental for proactive assistance in AR/VR, human-robot collaboration, and embodied AI. While recent works incorporate language as a semantic prior to reduce the ill-posed nature of egocentric forecasting, they largely neglect the 3D spatial and semantic context that governs how motion unfolds, and treat pose and language prediction as separate inference streams. We introduce Ego3DLM, built on two core principles: accurate motion forecasting requires explicit spatial and semantic understanding of the 3D environment, and pose and language must be predicted holistically in a single pass, since motion is inherently tied to the semantic interpretation of actions being performed. Given three-point tracking, 3D scene features, and egocentric video, Ego3DLM simultaneously decodes past pose, future pose, past narration, and future narration in a single autoregressive pass, grounding predicted poses and descriptions in one another to enforce cross-modal and temporal consistency. We adopt a three-stage training scheme: (1) spatial-semantic scene awareness pretraining; (2) holistic instruction tuning over all four outputs in a single pass; and (3) GRPO-based reinforcement finetuning with intra- and inter-modal rewards that directly optimize pose-language fidelity. Experiments on the Nymeria benchmark demonstrate that Ego3DLM achieves state-of-the-art performance across future motion prediction, past motion tracking, and motion description, showing that 3D scene grounding and holistic cross-modal prediction yield physically plausible and semantically coherent motion forecasts. The project page is available at https://jaewoo97.github.io/Ego3DLM/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.06701v1">SPEAR: A Simulator for Photorealistic Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 Accepted for publication at the European Conference on Computer Vision (ECCV) 2026
    </div>
    <details class="paper-abstract">
      Interactive simulators have become powerful tools for training embodied agents and generating synthetic visual data, but existing photorealistic simulators suffer from limited generality, programmability, and rendering speed. We address these limitations by introducing SPEAR: A Simulator for Photorealistic Embodied AI Research. At its core, SPEAR is a Python library that can connect to, and programmatically control, any Unreal Engine (UE) application via a modular plugin architecture. SPEAR exposes over 14K unique UE functions to Python, representing an order-of-magnitude increase in programmable functionality over existing UE-based simulators. Additionally, a single SPEAR instance can render 1920x1080 photorealistic beauty images directly into a user's NumPy array at 73 frames per second - an order of magnitude faster than existing UE plugins - while also providing ground truth image modalities that are not available in any existing UE-based simulator (e.g., a non-diffuse intrinsic image decomposition, material IDs, and physically based shading parameters). Finally, SPEAR introduces an expressive high-level programming model that enables users to specify complex graphs of UE work with arbitrary data dependencies among work items, and to execute these graphs deterministically within a single UE frame. We demonstrate the utility of SPEAR through a diverse collection of example applications: controlling multiple embodied agents with distinct action spaces (e.g., humans, cars, and robots) across several in-the-wild UE projects; rendering photorealistic city-scale environments; manipulating UE's procedural content generation systems; rendering synchronized multi-view images of detailed human faces; coordinating an interactive co-simulation with the MuJoCo physics simulator; and editing scenes with natural language via an AI coding assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22851v2">EgoDyn-Bench: Evaluating Ego-Motion Understanding in Vision-Centric Foundation Models for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-07-07
      | 💬 36 Pages, Accepted at ECCV 2026
    </div>
    <details class="paper-abstract">
      While Vision-Language Models (VLMs) have advanced high-level reasoning in autonomous driving, their ability to ground this reasoning in the underlying physics of ego-motion remains poorly understood. We introduce EgoDyn-Bench [Project page: (https://tum-avs.github.io/EgoDyn-Bench-Website/), Code: (https://github.com/TUM-AVS/EgoDyn-Bench), Dataset: (https://huggingface.co/datasets/fnc1901/EgoDyn-Bench)], a diagnostic benchmark for evaluating the semantic ego-motion understanding of vision-centric foundation models. By mapping continuous vehicle kinematics to discrete motion concepts via a deterministic oracle, we decouple a model's internal physical logic from its visual perception. Our large-scale empirical audit spanning 20$+$ models, including closed-source MLLMs, open-source VLMs across multiple scales, and specialized VLAs, identifies a significant Perception Bottleneck: while models exhibit logical physical concepts, they consistently fail to accurately align them with visual observations, frequently underperforming classical non-learned geometric baselines. This failure persists across model scales and domain-specific training, indicating a structural deficit in how current architectures couple visual perception with physical reasoning. We demonstrate that providing explicit trajectory encodings substantially restores physical consistency across all evaluated models, revealing a functional disentanglement between vision and language: ego-motion logic is derived almost exclusively from the language modality, while visual observations contribute negligible temporal signal. This structural finding provides a standardized diagnostic framework and a practical pathway toward physically aligned embodied AI. Ego-motion - Physical Reasoning - Foundation Models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14625v2">Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-07-06
    </div>
    <details class="paper-abstract">
      Efficient digital twin (DT) synchronization relies on maintaining high-fidelity virtual representations with minimal age of information (AoI). However, the synergistic potential of cooperative sensing and autonomous mobility of the sensing agent remains underexplored in existing DT synchronization frameworks. In this paper, we propose an agentic AI-empowered mobile embodied AI network (MEAN) framework for DT synchronization. In the proposed hybrid architecture, the base station (BS) conducts global orchestration, while the agents autonomously execute a five-stage closed-loop workflow: move-to-sense, cooperative sensing, onboard semantic processing, channel-aware mobility, and uplink transmission. To optimize synchronization performance, we formulate a joint topology dispatching and multidimensional resource allocation problem aimed at minimizing the maximum twin deviation across regions, subject to heterogeneous sensing fidelity and energy budget constraints. To tackle this, we develop a hierarchical two-layer optimization algorithm, where the outer-layer refines multi-agent assignment via a dynamic matching game, and the inner-layer iteratively optimizes the continuous resources. Extensive simulation results verify the convergence of the proposed algorithm and demonstrate its substantial superiority over multiple baseline schemes in reducing synchronization deviation. Furthermore, the results reveal that semantic compression serves as a vital substitute for channel resources in latency reduction under constrained bandwidth, while autonomous velocity adaptation provides an essential degree of freedom for the system to navigate the fundamental energy-time trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14401v2">pFedNavi: Structure-Aware Personalized Federated Vision-Language Navigation for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted by the IEEE INFOCOM 2026 Workshop on Emerging Intelligent Networks (EIN)
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation VLN requires large-scale trajectory instruction data from private indoor environments, raising significant privacy concerns. Federated Learning FL mitigates this by keeping data on-device, but vanilla FL struggles under VLNs' extreme cross-client heterogeneity in environments and instruction styles, making a single global model suboptimal. This paper proposes pFedNavi, a structure-aware and dynamically adaptive personalized federated learning framework tailored for VLN. Our key idea is to personalize where it matters: pFedNavi adaptively identifies client-specific layers via layer-wise mixing coefficients, and performs fine-grained parameter fusion on the selected components (e.g., the encoder-decoder projection and environment-sensitive decoder layers) to balance global knowledge sharing with local specialization. We evaluate pFedNavi on two standard VLN benchmarks, R2R and RxR, using both ResNet and CLIP visual representations. Across all metrics, pFedNavi consistently outperforms the FedAvg-based VLN baseline, achieving up to 7.5% improvement in navigation success rate and up to 7.8% gain in trajectory fidelity, while converging 1.38x faster under non-IID conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.04426v1">ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI</a></div>
    <div class="paper-meta">
      📅 2026-07-05
    </div>
    <details class="paper-abstract">
      Embodied AI is moving from isolated perception or action modules toward physical agents that understand, plan under goals, act through robot bodies, monitor progress, and improve from experience. Existing systems address this loop only in parts: end-to-end policies generate actions but often lack spatial reasoning, planning, and execution assessment, while robot-agent systems orchestrate tools or specialists but do not learn a shared representation. This fragmentation limits general Physical Agentic AI. We present ACE-Brain-0.5, a unified embodied foundation model that organizes robot intelligence into five coupled functions: spatial perception, decision making, embodied interaction, self-monitoring, and self-improvement. Built on ACE-Brain-0, which established spatial intelligence as a shared scaffold across robot platforms, ACE-Brain-0.5 extends an understanding-centric model into a closed-loop foundation model. A single 8B backbone instantiates the first four functions: grounding objects and affordances, reasoning over 3D and egocentric spatial relations, decomposing instructions into subgoals, generating navigation and manipulation actions, and estimating progress for verification and recovery. To unify these capabilities without cross-task interference, we introduce SSR+, which extends Scaffold-Specialize-Reconcile with a Reactivate stage after task-vector merging. The fifth function, self-improvement, is realized by a companion framework that updates external execution state, including task schemas, spatial memory, and failure-recovery cases, from rollouts. Across fifteen benchmarks, ACE-Brain-0.5 improves over ACE-Brain-0 on 14 of 18 spatial perception and grounding benchmarks, achieves competitive navigation and manipulation performance, and provides strong progress estimation in ID and OOD settings. Together, these results mark an early step toward general Physical Agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.03941v1">WSA$_1$: a 3D-Centric World-Spatial-Action Model for Generalizable Robot Control</a></div>
    <div class="paper-meta">
      📅 2026-07-04
    </div>
    <details class="paper-abstract">
      Recent advances in embodied AI have established robot foundation models (RFMs) as the dominant approach for generalist robotic systems to date. By leveraging imitation learning on extensive robot demonstrations, RFMs have achieved impressive capabilities in mapping visual observations and language instructions to continuous robotic actions. However, current RFMs lack an inherent ability to reason about physical dynamics and the causal effects of robot behaviors on the 3D physical world. This creates a fundamental mismatch between 2D-centric visual perception and 3D-centric embodied interaction, severely limiting the generalization ability of RFMs in real-world tasks.To address this gap, we present WSA$_1$, a novel RFM built upon proposed 3D-Centric World-Spatial-Action modeling paradigm. It not only learns 3D world-aware visual thought for future robot behaviors, but also models mutual constraints between 3D world state transitions and robotic actions to enhance behavior generalization. Notably, WSA$_1$ achieves highly data-efficient pre-training with 6k hours of expert demonstration data (only 1k hours from real robot), while delivering competitive manipulation performance (93% success rate) on RoboTwin2.0 simulation benchmark and achieving +20% average boosted performance over state-of-the-art RFMs on real-world robot control tasks. These results reveal that generalizable RFM can be attained without large-scale real robot data when paired with 3D-centric world-action joint modeling, which offers a practical and affordable pathway to generalist robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28489v3">Video Generation Models as World Models: Efficient Paradigms, Architectures and Algorithms</a></div>
    <div class="paper-meta">
      📅 2026-07-04
    </div>
    <details class="paper-abstract">
      The rapid evolution of video generation has enabled models to simulate complex physical dynamics and long-horizon causalities, positioning them as potential world simulators. However, a critical gap still remains between the theoretical capacity for world simulation and the heavy computational costs of spatiotemporal modeling. To address this, we comprehensively and systematically review video generation frameworks and techniques that consider efficiency as a crucial requirement for practical world modeling. We introduce a novel taxonomy in three dimensions: efficient modeling paradigms, efficient network architectures, and efficient inference algorithms. We further show that bridging this efficiency gap directly empowers interactive applications such as autonomous driving, embodied AI, and game simulation. Finally, we identify emerging research frontiers in efficient video-based world modeling, arguing that efficiency is a fundamental prerequisite for evolving video generators into general-purpose, real-time, and robust world simulators. A curated GitHub repository of the reviewed literature can be found at https://github.com/Isaachhh/Efficient-VWM-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.03470v1">PhysMirror: Physics-Aware Mirror Object Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-03
      | 💬 IROS 2026
    </div>
    <details class="paper-abstract">
      Synthesizing physically accurate mirror reflections remains a fundamental challenge for modern text-to-image diffusion models, which are increasingly critical for generating synthetic training data for embodied AI and robotic perception. These models typically struggle with strict geometric constraints, leading to hallucinations that degrade the utility of the synthetic data. To address this, we introduce a novel, end-to-end physics-aware generation framework namely PhysMirror that natively enforces projective geometry through explicit 3D spatial priors. Our method automatically lifts prompted objects into 3D meshes and constructs a lightweight, mathematically exact mirror scene within a simulated environment. By rendering this explicit 3D scene, we extract precise 2D conditioning elements, such as depth maps and segmentation maps, that serve as robust guiding signals for downstream diffusion models, guiding them to generate images with physically correct mirror reflections. Moreover, we introduce Mirror Consistency Score (MCS), reference-free, fully automated metric that quantifies physical correctness using dense feature matching and vanishing point convergence. Experimental results on our newly constructed MirrOB dataset demonstrate that our approach outperforms state-of-the-art baselines in reflection accuracy and physical realism, while maintaining strong text-to-image semantic alignment, providing a reliable pipeline for embodied AI data generation. The source code is released at https://duyphuc0701.github.io/PhysMirror.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16806v4">Insect-inspired Visual Point-goal Navigation</a></div>
    <div class="paper-meta">
      📅 2026-07-03
      | 💬 This work has been submitted for possible publication
    </div>
    <details class="paper-abstract">
      Insect neuroethology provides a compelling biological template for efficient autonomous navigation. We draw an analogy between the formal embodied AI visual point-goal navigation task and the ability of insects to discover, learn, and refine visually guided paths around obstacles between a discovered food location and their nest. We develop a novel integrative model of mushroom body and central complex, two insect brain structures, that have been implicated, respectively, in associative learning and path integration. We demonstrate the mushroom body learning triggered by collisions results in adaptive obstacle avoidance and consequently optimised paths to the goal, corroborating the hypothesis of recent behavioural work that an insect can learn continuously as they travel. The embodied insect-inspired model achieves success rates comparable to recent state-of-the-art models at many orders of magnitude less computational cost in the standardised Habitat point-goal navigation benchmark. Testing in a more realistic simulated environment validates its robustness to perturbations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02501v2">Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots</a></div>
    <div class="paper-meta">
      📅 2026-07-03
      | 💬 12 pages, 2 figures, Project website: https://github.com/SEU-PAISys/Embodied.cpp
    </div>
    <details class="paper-abstract">
      Embodied AI models now span vision-language-action (VLA) models and world-action models (WAMs), but practical deployment remains fragmented across model-specific Python stacks, backend assumptions, and robot-side glue code, especially on heterogeneous edge devices. Existing inference runtimes are designed mainly for request-response serving and therefore do not satisfy the runtime contract of embodied deployment: multi-rate execution inside closed-loop control, latency-first batch-1 inference on heterogeneous hardware, and extensible embodied interfaces beyond fixed token I/O. We present Embodied$.$cpp, a portable C++ inference runtime for embodied models. Based on an architectural analysis of representative VLA models and WAMs, Embodied$.$cpp captures a shared execution path and organizes it into five layers: input adapters, sequence builders, backbone execution, head plugins, and deployment adapters. The runtime provides modular multi-rate execution, latency-first fused inference, and extensible operator and I/O support, enabling deployment across heterogeneous devices, robots, and simulators through one backend abstraction. We evaluate Embodied$.$cpp on two VLA models, HY-VLA and pi0.5, and on a preliminary WAM benchmark using a LingBot-VA Transformer block. The VLA deployments achieve successful closed-loop execution with 100.0% and 91.0% task success rates, respectively. The WAM benchmark reduces block memory from 312.2 MiB to 88.1 MiB. These results show that Embodied$.$cpp improves deployment efficiency while preserving high accuracy across diverse embodied model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07306v3">BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-07-03
      | 💬 17 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02501v1">Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 12 pages, 2 figures, Project website: https://github.com/SEU-PAISys/Embodied.cpp
    </div>
    <details class="paper-abstract">
      Embodied AI models now span vision-language-action (VLA) models and world-action models (WAMs), but practical deployment remains fragmented across model-specific Python stacks, backend assumptions, and robot-side glue code, especially on heterogeneous edge devices. Existing inference runtimes are designed mainly for request-response serving and therefore do not satisfy the runtime contract of embodied deployment: multi-rate execution inside closed-loop control, latency-first batch-1 inference on heterogeneous hardware, and extensible embodied interfaces beyond fixed token I/O. We present Embodied.cpp, a portable C++ inference runtime for embodied models. Based on an architectural analysis of representative VLA models and WAMs, Embodied.cpp captures a shared execution path and organizes it into five layers: input adapters, sequence builders, backbone execution, head plugins, and deployment adapters. The runtime provides modular multi-rate execution, latency-first fused inference, and extensible operator and I/O support, enabling deployment across heterogeneous devices, robots, and simulators through one backend abstraction. We evaluate Embodied.cpp on two VLA models, HY-VLA and pi0.5, and on a preliminary WAM benchmark using a LingBot-VA Transformer block. The VLA deployments achieve successful closed-loop execution with 100.0% and 91.0% task success rates, respectively. The WAM benchmark reduces block memory from 312.2 MiB to 88.1 MiB. These results show that Embodied.cpp improves deployment efficiency while preserving high accuracy across diverse embodied model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02497v1">Seek to Segment: Active Perception for Panoramic Referring Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ECCV 2026, Project Page: https://henghuiding.com/APRS/
    </div>
    <details class="paper-abstract">
      Existing referring segmentation models passively process static images captured from fixed perspectives, limiting their applicability in Embodied AI, where agents must perform active perception in the continuous 360$^\circ$ environments. To bridge this gap, we introduce a novel task: Active Panoramic Referring Segmentation (APRS). In this setting, an agent is required to adjust its viewing direction ($Δθ, Δφ$) to explore the 360$^\circ$ environment, seeking the object specified by a user instruction for segmentation. To tackle this challenging task, we propose PanoSeeker, a memory-augmented agent for efficient APRS. Rather than relying on heuristic scanning, PanoSeeker integrates a Vision-Language Model (VLM) with EgoSphere, an explicit spatial visual memory. By progressively integrating sequential local observations into a unified 360$^\circ$ representation, EgoSphere enables the agent to plan efficient and non-redundant search trajectories. Once the target is found, the agent performs active viewpoint alignment and outputs the segmentation mask. Furthermore, we curate an expert-annotated search trajectory dataset with memory timelines for Supervised Fine-Tuning, followed by Reinforcement Learning post-training to explicitly optimize PanoSeeker's exploration efficiency. Extensive experiments on our newly established APRS benchmark demonstrate that PanoSeeker achieves superior search efficiency and segmentation accuracy, significantly outperforming adapted state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02466v1">Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Accepted to ICML 2026, 21 pages,6 figures
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are fundamentally bottlenecked by the scarcity of expert demonstrations -- triplets of observations, instructions, and actions that are costly to collect at scale. We argue that this bottleneck stems from conflating two distinct learning objectives: acquiring physical competence (how to move) and acquiring semantic alignment (what to do). Crucially, only the latter requires language supervision. Building on this Decomposition Hypothesis, we propose Task-Agnostic Pretraining (TAP), a two-stage framework that first learns transferable motor priors from cheap, unlabeled interaction data -- including discarded off-task trajectories and autonomous robot play -- via a self-supervised Inverse Dynamics objective. A lightweight second stage then grounds these priors in language using minimal expert data. On the SIMPLER benchmark, TAP matches models trained on over 1M expert trajectories while using orders of magnitude less labeled data, yielding a 10% absolute gain over standard behavior cloning. On a real-world WidowX platform, TAP retains 25% success under camera perturbations where internet-scale baselines collapse to 0%, demonstrating that task-agnostic pretraining produces robust, transferable physical representations and offers a scalable path forward for Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02425v1">Learning to Evolve Scenes: Reasoning about Human Activities with Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 Project page at https://francescapistilli.github.io/GLEN
    </div>
    <details class="paper-abstract">
      Understanding human behavior while interacting with the surrounding world is crucial for many applications of embodied AI. First-person videos are particularly informative for this problem, as they well capture how activities reshape the scene over time. However, existing approaches often rely on implicit visual or language-aligned representations, disregarding structured reasoning over the scene dynamic. We argue that explicit, compositional and editable representations of human-environment interactions can play a crucial role for rich grounded activity understanding. To this end, we introduce SG-Ego, a large scale annotation set extending Ego4D with spatio-temporal scene graphs, where relations triplets are consolidated over time into explicit time-evolving descriptions of the scene state. To reason over this representation, we propose GLEN, a graph-based model that operates over scene graph sequences to both align them with textual actions and model their temporal evolution. In addition, we formulate the activity-driven graph-edit forecasting (A-GEF) problem, a novel task that casts scene dynamics as a sequence of structured transformations conditioned on ongoing actions, enabling explicit reasoning about how scenes change over time. We validate our approach across multiple downstream tasks, spanning retrieval benchmarks as EgoMCQ and EgoCVR, as well as long-horizon reasoning benchmarks as EXPLORE-Bench and the newly introduced A-GEF. GLEN achieves strong results compared to raw video baselines and it excels in reasoning settings, typically addressed only with MLLMs, while enabling controllable and structured predictions of scene dynamics driven by human activities. We believe our results establish spatio-temporal scene graphs, together with models that reason over them, as strong compositional and interpretable representations for video understanding and potentially beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01938v1">PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 ECCV 2026. Code and data are available at: https://github.com/vLAR-group/PhysMani
    </div>
    <details class="paper-abstract">
      Manipulating fast and dynamically moving targets in unstructured 3D environments remains challenging for embodied AI. Existing visual-language-action models and world models struggle with accurate 3D geometry and physically meaningful forecasting. We propose PhysMani, a framework that couples a physics-principled 3D Gaussian world model with a future-aware action policy model. The world model learns a divergence-free Gaussian velocity field via online optimization for fast and physically grounded future dynamics prediction. The policy model integrates the predicted 3D scene future dynamics through a learnable token based cross-attention module. We introduce PhysMani-Bench, a dynamic manipulation benchmark with 16 tasks, and demonstrate a superior success rate over strong baselines in both simulation and real-world robot experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.01766v1">SimWorlds: A Multi-Agent System for Dynamic 3D Scene Creation</a></div>
    <div class="paper-meta">
      📅 2026-07-02
      | 💬 20 pages, 3 figures. Project page: https://dynsimworlds.github.io
    </div>
    <details class="paper-abstract">
      LLM agents are increasingly used to translate natural language into 3D scenes in a procedural way, but existing systems focus on static output. Dynamic 4D scenes from text alone, in which liquids flow, particles emit, rigid bodies cascade, and articulated mechanisms move, remain largely unexplored despite their value as editable content and as physics-grounded training data for video generation and embodied AI. Two challenges set the dynamic case apart from static text-to-scene work: an agent must jointly coordinate spatial layout, multiple physics solvers, temporal sequencing, camera, and lighting in a single coherent scene, and verifying motion correctness from rendered video is fundamentally harder than judging a single image. We present SimWorlds: a multi-agent framework that produces dynamic, editable 4D scenes from text, with Blender-specific procedural knowledge, a planner-coder-reviewer workflow driving a fixed ordered sequence of construction stages, a layered scene protocol enforced by a deterministic verifier, and a runtime-state inspection tool suite that catches mechanism failures the rendered image cannot reveal. We also introduce 4DBuildBench, a benchmark for assessing both visual fidelity and physical consistency of the procedural dynamic 3D scenes generated from text prompts. Experiments show that SimWorlds outperforms prior dynamic Blender generation baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16993v2">Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification</a></div>
    <div class="paper-meta">
      📅 2026-07-01
    </div>
    <details class="paper-abstract">
      As embodied AI transitions to real-world deployment, the success of the Vision-and-Language Navigation (VLN) task tends to evolve from mere reachability to social compliance. However, current agents suffer from a "goal-driven trap", prioritizing physical geometry ("can I go?") over semantic rules ("may I go?"), frequently overlooking subtle regulatory constraints. To bridge this gap, we establish Rule-VLN, the first large-scale urban benchmark for rule-compliant navigation. Spanning a massive 29k-node environment, it injects 177 diverse regulatory categories into 8k constrained nodes across four curriculum levels, challenging agents with fine-grained visual and behavioral constraints. We further propose the Semantic Navigation Rectification Module (SNRM), a universal, zero-shot module designed to equip pre-trained agents with safety awareness. SNRM integrates a coarse-to-fine visual perception VLM framework with an epistemic mental map for dynamic detour planning. Experiments demonstrate that while Rule-VLN challenges state-of-the-art models, SNRM significantly restores navigation capabilities, reducing CVR by 19.26% and boosting TC by 5.97%.
    </details>
</div>
