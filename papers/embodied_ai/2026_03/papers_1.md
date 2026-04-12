# embodied ai - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.29798v1">SceneTeract: Agentic Functional Affordances and VLM Grounding in 3D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-31
      | 💬 Project page: https://sceneteract.github.io/
    </div>
    <details class="paper-abstract">
      Embodied AI depends on interactive 3D environments that support meaningful activities for diverse users, yet assessing their functional affordances remains a core challenge. We introduce SceneTeract, a framework that verifies 3D scene functionality under agent-specific constraints. Our core contribution is a grounded verification engine that couples high-level semantic reasoning with low-level geometric checks. SceneTeract decomposes complex activities into sequences of atomic actions and validates each step against accessibility requirements (e.g., reachability, clearance, and navigability) conditioned on an embodied agent profile, using explicit physical and geometric simulations. We deploy SceneTeract to perform an in-depth evaluation of (i) synthetic indoor environments, uncovering frequent functional failures that prevent basic interactions, and (ii) the ability of frontier Vision-Language Models (VLMs) to reason about and predict functional affordances, revealing systematic mismatches between semantic confidence and physical feasibility even for the strongest current models. Finally, we leverage SceneTeract as a reward engine for VLM post-training, enabling scalable distillation of geometric constraints into reasoning models. We release the SceneTeract verification suite and data to bridge perception and physical reality in embodied 3D scene understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28888v1">A Semantic Observer Layer for Autonomous Vehicles: Pre-Deployment Feasibility Study of VLMs for Low-Latency Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-30
    </div>
    <details class="paper-abstract">
      Semantic anomalies-context-dependent hazards that pixel-level detectors cannot reason about-pose a critical safety risk in autonomous driving. We propose a \emph{semantic observer layer}: a quantized vision-language model (VLM) running at 1--2\,Hz alongside the primary AV control loop, monitoring for semantic edge cases, and triggering fail-safe handoffs when detected. Using Nvidia Cosmos-Reason1-7B with NVFP4 quantization and FlashAttention2, we achieve ~500 ms inference a ~50x speedup over the unoptimized FP16 baseline (no quantization, standard PyTorch attention) on the same hardware--satisfying the observer timing budget. We benchmark accuracy, latency, and quantization behavior in static and video conditions, identify NF4 recall collapse (10.6%) as a hard deployment constraint, and a hazard analysis mapping performance metrics to safety goals. The results establish a pre-deployment feasibility case for the semantic observer architecture on embodied-AI AV platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21428v2">From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 10 pages, 5 figures, Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.28010v1">HeteroHub: An Applicable Data Management Framework for Heterogeneous Multi-Embodied Agent System</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Heterogeneous Multi-Embodied Agent Systems involve coordinating multiple embodied agents with diverse capabilities to accomplish tasks in dynamic environments. This process requires the collection, generation, and consumption of massive, heterogeneous data, which primarily falls into three categories: static knowledge regarding the agents, tasks, and environments; multimodal training datasets tailored for various AI models; and high-frequency sensor streams. However, existing frameworks lack a unified data management infrastructure to support the real-world deployment of such systems. To address this gap, we present \textbf{HeteroHub}, a data-centric framework that integrates static metadata, task-aligned training corpora, and real-time data streams. The framework supports task-aware model training, context-sensitive execution, and closed-loop control driven by real-world feedback. In our demonstration, HeteroHub successfully coordinates multiple embodied AI agents to execute complex tasks, illustrating how a robust data management framework can enable scalable, maintainable, and evolvable embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27967v1">Learning Multi-View Spatial Reasoning from Cross-View Relations</a></div>
    <div class="paper-meta">
      📅 2026-03-30
      | 💬 Accepted to CVPR 2026
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have achieved impressive results on single-view vision tasks, but lack the multi-view spatial reasoning capabilities essential for embodied AI systems to understand 3D environments and manipulate objects across different viewpoints. In this work, we introduce Cross-View Relations (XVR), a large-scale dataset designed to teach VLMs spatial reasoning across multiple views. XVR comprises 100K vision-question-answer samples derived from 18K diverse 3D scenes and 70K robotic manipulation trajectories, spanning three fundamental spatial reasoning tasks: Correspondence (matching objects across views), Verification (validating spatial relationships), and Localization (identifying object positions). VLMs fine-tuned on XVR achieve substantial improvements on established multi-view and robotic spatial reasoning benchmarks (MindCube and RoboSpatial). When integrated as backbones in Vision-Language-Action models, XVR-trained representations improve success rates on RoboCasa. Our results demonstrate that explicit training on cross-view spatial relations significantly enhances multi-view reasoning and transfers effectively to real-world robotic manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.27573v1">SPREAD: Spatial-Physical REasoning via geometry Aware Diffusion</a></div>
    <div class="paper-meta">
      📅 2026-03-29
    </div>
    <details class="paper-abstract">
      Automated 3D scene generation is pivotal for applications spanning virtual reality, digital content creation, and Embodied AI. While computer graphics prioritizes aesthetic layouts, vision and robotics demand scenes that mirror real-world complexity which current data-driven methods struggle to achieve due to limited unstructured training data and insufficient spatial and physical modeling. We propose SPREAD, a diffusion-based framework that jointly learns spatial and physical relationships through a graph transformer, explicitly conditioning on posed scene point clouds for geometric awareness. Moreover, our model integrates differentiable guidance for collision avoidance, relational constraint, and gravity, ensuring physically coherent scenes without sacrificing relational context. Our experiments on 3D-FRONT and ProcTHOR datasets demonstrate state-of-the-art performance in spatial-relational reasoning and physical metrics. Moreover, \ours{} outperforms baselines in scene consistency and stability during pre- and post-physics simulation, proving its capability to generate simulation-ready environments for embodied AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08021v2">AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-28
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26997v1">ROSClaw: An OpenClaw ROS 2 Framework for Agentic Robot Control and Interaction</a></div>
    <div class="paper-meta">
      📅 2026-03-27
    </div>
    <details class="paper-abstract">
      Foundation models can endow robots with open-ended reasoning, language understanding, and adaptive planning, yet connecting a model to a physical robot today requires bespoke integration that couples perception, actuation, and safety to a single model and platform. We present ROSClaw, a model-agnostic executive layer that integrates the OpenClaw agent runtime with ROS 2, enabling any foundation model to perceive, reason about, and act on any ROS-enabled robot through (i) dynamic capability discovery with standardized affordance injection, (ii) multimodal observation normalization, (iii) pre-execution action validation within a configurable safety envelope, and (iv) structured audit logging. Swapping model backends or robot platforms is a configuration change; tool schemas, safety enforcement, and provenance logging remain invariant. We deploy ROSClaw on three platforms (wheeled, quadruped, humanoid) with four foundation-model backends. Under this controlled substrate, models exhibit up to 4.8 x differences in out-of-policy action proposal rates (3.4 x among frontier models alone) and produce qualitatively distinct physical behaviors from identical commands. A cross-framework parity protocol against ROSA confirms that executive-layer design, not just prompt wording, significantly affects both task completion and safety behavior, establishing ROSClaw as both practical agentic-robot infrastructure and a reproducible measurement instrument for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20620v2">Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-27
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at https://ai4ce.github.io/wanderland/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25981v1">Policy-Guided World Model Planning for Language-Conditioned Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Navigating to a visually specified goal given natural language instructions remains a fundamental challenge in embodied AI. Existing approaches either rely on reactive policies that struggle with long-horizon planning, or employ world models that suffer from poor action initialization in high-dimensional spaces. We present PiJEPA, a two-stage framework that combines the strengths of learned navigation policies with latent world model planning for instruction-conditioned visual navigation. In the first stage, we finetune an Octo-based generalist policy, augmented with a frozen pretrained vision encoder (DINOv2 or V-JEPA-2), on the CAST navigation dataset to produce an informed action distribution conditioned on the current observation and language instruction. In the second stage, we use this policy-derived distribution to warm-start Model Predictive Path Integral (MPPI) planning over a separately trained JEPA world model, which predicts future latent states in the embedding space of the same frozen encoder. By initializing the MPPI sampling distribution from the policy prior rather than from an uninformed Gaussian, our planner converges faster to high-quality action sequences that reach the goal. We systematically study the effect of the vision encoder backbone, comparing DINOv2 and V-JEPA-2, across both the policy and world model components. Experiments on real-world navigation tasks demonstrate that PiJEPA significantly outperforms both standalone policy execution and uninformed world model planning, achieving improved goal-reaching accuracy and instruction-following fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25544v1">Towards Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Learning motor control for muscle-driven musculoskeletal models is hindered by the computational cost of biomechanically accurate simulation and the scarcity of validated, open full-body models. Here we present MuscleMimic, an open-source framework for scalable motion imitation learning with physiologically realistic, muscle-actuated humanoids. MuscleMimic provides two validated musculoskeletal embodiments - a fixed-root upper-body model (126 muscles) for bimanual manipulation and a full-body model (416 muscles) for locomotion - together with a retargeting pipeline that maps SMPL-format motion capture data onto musculoskeletal structures while preserving kinematic and dynamic consistency. Leveraging massively parallel GPU simulation, the framework achieves order-of-magnitude training speedups over prior CPU-based approaches while maintaining comprehensive collision handling, enabling a single generalist policy to be trained on hundreds of diverse motions within days. The resulting policy faithfully reproduces a broad repertoire of human movements under full muscular control and can be fine-tuned to novel motions within hours. Biomechanical validation against experimental walking and running data demonstrates strong agreement in joint kinematics (mean correlation r = 0.90), while muscle activation analysis reveals both the promise and fundamental challenges of achieving physiological fidelity through kinematic imitation alone. By lowering the computational and data barriers to musculoskeletal simulation, MuscleMimic enables systematic model validation across diverse dynamic movements and broader participation in neuromuscular control research. Code, models, checkpoints, and retargeted datasets are available at: https://github.com/amathislab/musclemimic
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25420v1">VideoWeaver: Multimodal Multi-View Video-to-Video Transfer for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      Recent progress in video-to-video (V2V) translation has enabled realistic resimulation of embodied AI demonstrations, a capability that allows pretrained robot policies to be transferable to new environments without additional data collection. However, prior works can only operate on a single view at a time, while embodied AI tasks are commonly captured from multiple synchronized cameras to support policy learning. Naively applying single-view models independently to each camera leads to inconsistent appearance across views, and standard transformer architectures do not scale to multi-view settings due to the quadratic cost of cross-view attention. We present VideoWeaver, the first multimodal multi-view V2V translation framework. VideoWeaver is initially trained as a single-view flow-based V2V model. To achieve an extension to the multi-view regime, we propose to ground all views in a shared 4D latent space derived from a feed-forward spatial foundation model, namely, Pi3. This encourages view-consistent appearance even under wide baselines and dynamic camera motion. To scale beyond a fixed number of cameras, we train views at distinct diffusion timesteps, enabling the model to learn both joint and conditional view distributions. This in turn allows autoregressive synthesis of new viewpoints conditioned on existing ones. Experiments show superior or similar performance to the state-of-the-art on the single-view translation benchmarks and, for the first time, physically and stylistically consistent multi-view translations, including challenging egocentric and heterogeneous-camera setups central to world randomization for robot learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07237v2">Unified Camera Positional Encoding for Controlled Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-26
      | 💬 Camera Ready of CVPR2026. Project Page: https://chengzhag.github.io/publication/ucpe/ Code: https://github.com/chengzhag/UCPE
    </div>
    <details class="paper-abstract">
      Transformers have emerged as a universal backbone across 3D perception, video generation, and world models for autonomous driving and embodied AI, where understanding camera geometry is essential for grounding visual observations in three-dimensional space. However, existing camera encoding methods often rely on simplified pinhole assumptions, restricting generalization across the diverse intrinsics and lens distortions in real-world cameras. We introduce Relative Ray Encoding, a geometry-consistent representation that unifies complete camera information, including 6-DoF poses, intrinsics, and lens distortions. To evaluate its capability under diverse controllability demands, we adopt camera-controlled text-to-video generation as a testbed task. Within this setting, we further identify pitch and roll as two components effective for Absolute Orientation Encoding, enabling full control over the initial camera orientation. Together, these designs form UCPE (Unified Camera Positional Encoding), which integrates into a pretrained video Diffusion Transformer through a lightweight spatial attention adapter, adding less than 1% trainable parameters while achieving state-of-the-art camera controllability and visual fidelity. To facilitate systematic training and evaluation, we construct a large video dataset covering a wide range of camera motions and lens types. Extensive experiments validate the effectiveness of UCPE in camera-controllable video generation and highlight its potential as a general camera representation for Transformers across future multi-view, video, and 3D tasks. Code will be available at https://github.com/chengzhag/UCPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03370v3">EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation</a></div>
    <div class="paper-meta">
      📅 2026-03-26
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24684v1">KitchenTwin: Semantically and Geometrically Grounded 3D Kitchen Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-03-25
    </div>
    <details class="paper-abstract">
      Embodied AI training and evaluation require object-centric digital twin environments with accurate metric geometry and semantic grounding. Recent transformer-based feedforward reconstruction methods can efficiently predict global point clouds from sparse monocular videos, yet these geometries suffer from inherent scale ambiguity and inconsistent coordinate conventions. This mismatch prevents the reliable fusion of these dimensionless point cloud predictions with locally reconstructed object meshes. We propose a novel scale-aware 3D fusion framework that registers visually grounded object meshes with transformer-predicted global point clouds to construct metrically consistent digital twins. Our method introduces a Vision-Language Model (VLM)-guided geometric anchor mechanism that resolves this fundamental coordinate mismatch by recovering an accurate real-world metric scale. To fuse these networks, we propose a geometry-aware registration pipeline that explicitly enforces physical plausibility through gravity-aligned vertical estimation, Manhattan-world structural constraints, and collision-free local refinement. Experiments on real indoor kitchen environments demonstrate improved cross-network object alignment and geometric consistency for downstream tasks, including multi-primitive fitting and metric measurement. We additionally introduce an open-source indoor digital twin dataset with metrically scaled scenes and semantically grounded and registered object-centric mesh annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24329v1">GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-25
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.23386v1">SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM</a></div>
    <div class="paper-meta">
      📅 2026-03-24
    </div>
    <details class="paper-abstract">
      High-quality articulated 3D assets are indispensable for embodied AI and physical simulation, yet 3D generation still focuses on static meshes, leaving a gap in "sim-ready" interactive objects. Most recent articulated object creation methods rely on multi-stage pipelines that accumulate errors across decoupled modules. Alternatively, unified MLLMs offer a single-stage path to joint static asset understanding and sim-ready asset generation. However dense voxel-based 3D tokenization yields long 3D token sequences and high memory overhead, limiting scalability to complex articulated objects. To address this, we propose SIMART, a unified MLLM framework that jointly performs part-level decomposition and kinematic prediction. By introducing a Sparse 3D VQ-VAE, SIMART reduces token counts by 70% vs. dense voxel tokens, enabling high-fidelity multi-part assemblies. SIMART achieves state-of-the-art performance on PartNet-Mobility and in-the-wild AIGC datasets, and enables physics-based robotic simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02982v2">U4D: Uncertainty-Aware 4D World Modeling from LiDAR Sequences</a></div>
    <div class="paper-meta">
      📅 2026-03-24
      | 💬 CVPR 2026; 20 pages, 7 figures, 11 tables; Code at https://github.com/worldbench/U4D
    </div>
    <details class="paper-abstract">
      Modeling dynamic 3D environments from LiDAR sequences is central to building reliable 4D worlds for autonomous driving and embodied AI. Existing generative frameworks, however, often treat all spatial regions uniformly, overlooking the varying uncertainty across real-world scenes. This uniform generation leads to artifacts in complex or ambiguous regions, limiting realism and temporal stability. In this work, we present U4D, an uncertainty-aware framework for 4D LiDAR world modeling. Our approach first estimates spatial uncertainty maps from a pretrained segmentation model to localize semantically challenging regions. It then performs generation in a "hard-to-easy" manner through two sequential stages: (1) uncertainty-region modeling, which reconstructs high-entropy regions with fine geometric fidelity, and (2) uncertainty-conditioned completion, which synthesizes the remaining areas under learned structural priors. To further ensure temporal coherence, U4D incorporates a mixture of spatio-temporal (MoST) block that adaptively fuses spatial and temporal representations during diffusion. Extensive experiments show that U4D produces geometrically faithful and temporally consistent LiDAR sequences, advancing the reliability of 4D world modeling for autonomous perception and simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17657v3">SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-03-23
    </div>
    <details class="paper-abstract">
      Robotic and autonomous systems need dense spatial cues, but many monocular depth models are heavy, task-specific, or hard to attach to an existing multimodal stack. CLIP offers strong semantic representations, yet most CLIP-based depth methods still depend on text prompts or backbone updates, which complicate deployment in integrated control pipelines. We present SPACE-CLIP, a decoder-only depth framework that reads geometric cues directly from a frozen CLIP vision encoder and bypasses the text encoder at inference time. The model combines FiLM-conditioned semantic features from deep layers with structural features from shallow layers to recover both global scene layout and local geometric detail. Under the TFI-FB constraint (text-free inference and frozen vision backbone), SPACE-CLIP achieves AbsRel 0.0901 on KITTI and 0.1042 on NYU Depth V2, and the same dual-pathway decoder transfers to a frozen SigLIP backbone with comparable results. These findings show that a compact decoder can turn a shared foundation-model backbone into a reusable spatial perception module for embodied AI and autonomous robotic systems. Our model is available at https://github.com/taewan2002/space-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01547v2">Vision-language models lag human performance on physical dynamics and intent reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-23
    </div>
    <details class="paper-abstract">
      Spatial intelligence is central to embodied cognition, yet contemporary AI systems still struggle to reason about physical interactions in open-world human environments. Despite strong performance on controlled benchmarks, vision-language models often fail to jointly model physical dynamics, reference frames, and the latent human intentions that drive spatial change. We introduce Teleo-Spatial Intelligence (TSI), a reasoning capability that links spatiotemporal change to goal-directed structure. To evaluate TSI, we present EscherVerse, a large-scale open-world resource built from 11,328 real-world videos, including an 8,000-example benchmark and a 35,963-example instruction-tuning set. Across 27 state-of-the-art vision-language models and an independent analysis of first-pass human responses from 11 annotators, we identify a persistent teleo-spatial reasoning gap: the strongest proprietary model achieves 57.26\% overall accuracy, far below first-pass human performance, which ranges from 84.81\% to 95.14\% with a mean of 90.62\%. Fine-tuning on real-world, intent-aware data narrows this gap for open-weight models, but does not close it. EscherVerse provides a diagnostic testbed for purpose-aware spatial reasoning and highlights a critical gap between pattern recognition and human-level understanding in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20668v1">ROI-Driven Foveated Attention for Unified Egocentric Representations in Vision-Language-Action Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-21
    </div>
    <details class="paper-abstract">
      The development of embodied AI systems is increasingly constrained by the availability and structure of physical interaction data. Despite recent advances in vision-language-action (VLA) models, current pipelines suffer from high data collection cost, limited cross-embodiment alignment, and poor transfer from internet-scale visual data to robot control. We propose a region-of-interest (ROI) driven engineering workflow that introduces an egocentric, geometry-grounded data representation. By projecting end-effector poses via forward kinematics (FK) into a single external camera, we derive movement-aligned hand-centric ROIs without requiring wrist-mounted cameras or multi-view systems. Unlike directly downsampling the full frame, ROI is cropped from the original image before resizing, preserving high local information density for contact-critical regions while retaining global context. We present a reproducible pipeline covering calibration, synchronization, ROI generation, deterministic boundary handling, and metadata governance. The resulting representation is embodiment-aligned and viewpoint-normalized, enabling data reuse across heterogeneous robots. We argue that egocentric ROI serves as a practical data abstraction for scalable collection and cross-embodiment learning, bridging internet-scale perception and robot-specific control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.22507v2">A Unified Cloud-Edge-Terminal Framework for Multimodal Integrated Sensing and Communication</a></div>
    <div class="paper-meta">
      📅 2026-03-21
    </div>
    <details class="paper-abstract">
      The transition to 6G calls for tightly integrated sensing and communication to support mission-critical services such as autonomous driving, embodied AI, and high-precision telemedicine. However, most existing ISAC designs rely on a single sensing modality (often RF), which limits environmental understanding and becomes a bottleneck in complex and dynamic scenes. This motivates a shift from single-modal to multimodal ISAC, where heterogeneous sensors (e.g., radar, LiDAR, and cameras) complement each other to improve robustness and semantic awareness. In this article, we first summarize key challenges for multimodal ISAC, including heterogeneous fusion, communication overhead, and scalable system design. We then highlight three enabling technologies: large AI models, semantic communications, and multi-agent systems, and discuss how their combination can enable task-oriented multimodal perception. Building on these insights, we propose a unified cloud-edge-terminal (CET) framework that hierarchically distributes intelligence and supports three adaptive operation modes: global fusion mode (GFM), cooperative relay mode (CRM), and peer interaction mode (PIM). A case study evaluates the framework across three modes, demonstrating that GFM achieves the highest accuracy, PIM minimizes latency, and CRM strikes an optimal balance between performance and efficiency. Finally, we conclude with open research issues and future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26730v1">Why Cognitive Robotics Matters: Lessons from OntoAgent and LLM Deployment in HARMONIC for Safety-Critical Robot Teaming</a></div>
    <div class="paper-meta">
      📅 2026-03-20
    </div>
    <details class="paper-abstract">
      Deploying embodied AI agents in the physical world demands cognitive capabilities for long-horizon planning that execute reliably, deterministically, and transparently. We present HARMONIC, a cognitive-robotic architecture that pairs OntoAgent, a content-centric cognitive architecture providing metacognitive self-monitoring, domain-grounded diagnosis, and consequence-based action selection over ontologically structured knowledge, with a modular reactive tactical layer. HARMONIC's modular design enables a functional evaluation of whether LLMs can replicate OntoAgent's cognitive capabilities, evaluated within the same robotic system under identical conditions. Six LLMs spanning frontier and efficient tiers replace OntoAgent in a collaborative maintenance scenario under native and knowledge-equalized conditions. Results reveal that LLMs do not consistently assess their own knowledge state before acting, causing downstream failures in diagnostic reasoning and action selection. These deficits persist even with equivalent procedural knowledge, indicating the issues are architectural rather than knowledge-based. These findings support the design of physically embodied systems in which cognitive architectures retain primary authority for reasoning, owing to their deterministic and transparent characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19782v1">Embodied Science: Closing the Discovery Loop with Agentic Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-20
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Artificial intelligence has demonstrated remarkable capability in predicting scientific properties, yet scientific discovery remains an inherently physical, long-horizon pursuit governed by experimental cycles. Most current computational approaches are misaligned with this reality, framing discovery as isolated, task-specific predictions rather than continuous interaction with the physical world. Here, we argue for embodied science, a paradigm that reframes scientific discovery as a closed loop tightly coupling agentic reasoning with physical execution. We propose a unified Perception-Language-Action-Discovery (PLAD) framework, wherein embodied agents perceive experimental environments, reason over scientific knowledge, execute physical interventions, and internalize outcomes to drive subsequent exploration. By grounding computational reasoning in robust physical feedback, this approach bridges the gap between digital prediction and empirical validation, offering a roadmap for autonomous discovery systems in the life and chemical sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19607v1">Physion-Eval: Evaluating Physical Realism in Generated Video via Human Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-20
    </div>
    <details class="paper-abstract">
      Video generation models are increasingly used as world simulators for storytelling, simulation, and embodied AI. As these models advance, a key question arises: do generated videos obey the physical laws of the real world? Existing evaluations largely rely on automated metrics or coarse human judgments such as preferences or rubric-based checks. While useful for assessing perceptual quality, these methods provide limited insight into when and why generated dynamics violate real-world physical constraints. We introduce Physion-Eval, a large-scale benchmark of expert human reasoning for diagnosing physical realism failures in videos generated by five state-of-the-art models across egocentric and exocentric views, containing 10,990 expert reasoning traces spanning 22 fine-grained physical categories. Each generated video is derived from a corresponding real-world reference video depicting a clear physical process, and annotated with temporally localized glitches, structured failure categories, and natural-language explanations of the violated physical behavior. Using this dataset, we reveal a striking limitation of current video generation models: in physics-critical scenarios, 83.3% of exocentric and 93.5% of egocentric generated videos exhibit at least one human-identifiable physical glitch. We hope Physion-Eval will set a new standard for physical realism evaluation and guide the development of physics-grounded video generation. The benchmark is publicly available at https://huggingface.co/datasets/PhysionLabs/Physion-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18912v1">GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      Understanding realistic hand-object interactions from monocular RGB videos is essential for AR/VR, robotics, and embodied AI. Existing methods rely on category-specific templates or heavy computation, yet still produce physically inconsistent hand-object alignment in 3D. We introduce GHOST (Gaussian Hand-Object Splatting), a fast, category-agnostic framework for reconstructing dynamic hand-object interactions using 2D Gaussian Splatting. GHOST represents both hands and objects as dense, view-consistent Gaussian discs and introduces three key innovations: (1) a geometric-prior retrieval and consistency loss that completes occluded object regions, (2) a grasp-aware alignment that refines hand translations and object scale to ensure realistic contact, and (3) a hand-aware background loss that prevents penalizing hand-occluded object regions. GHOST achieves complete, physically consistent, and animatable reconstructions from a single RGB video while running an order of magnitude faster than prior category-agnostic methods. Extensive experiments on ARCTIC, HO3D, and in-the-wild datasets demonstrate state-of-the-art accuracy in 3D reconstruction and 2D rendering quality, establishing GHOST as an efficient and robust solution for realistic hand-object interaction modeling. Code is available at https://github.com/ATAboukhadra/GHOST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18496v1">NymeriaPlus: Enriching Nymeria Dataset with Additional Annotations and Data</a></div>
    <div class="paper-meta">
      📅 2026-03-19
    </div>
    <details class="paper-abstract">
      The Nymeria Dataset, released in 2024, is a large-scale collection of in-the-wild human activities captured with multiple egocentric wearable devices that are spatially localized and temporally synchronized. It provides body-motion ground truth recorded with a motion-capture suit, device trajectories, semi-dense 3D point clouds, and in-context narrations. In this paper, we upgrade Nymeria and introduce NymeriaPlus. NymeriaPlus features: (1) improved human motion in Momentum Human Rig (MHR) and SMPL formats; (2) dense 3D and 2D bounding box annotations for indoor objects and structural elements; (3) instance-level 3D object reconstructions; and (4) additional modalities e.g., basemap recordings, audio, and wristband videos. By consolidating these complementary modalities and annotations into a single, coherent benchmark, NymeriaPlus strengthens Nymeria into a more powerful in-the-wild egocentric dataset. We expect NymeriaPlus to bridge a key gap in existing egocentric resources and to support a broader range of research, including unique explorations of multimodal learning for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20310v1">GraphiContact: Pose-aware Human-Scene Robust Contact Perception for Interactive Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-19
      | 💬 15 pages, 9 figures, Accepted at ICME 2026
    </div>
    <details class="paper-abstract">
      Monocular vertex-level human-scene contact prediction is a fundamental capability for interactive systems such as assistive monitoring, embodied AI, and rehabilitation analysis. In this work, we study this task jointly with single-image 3D human mesh reconstruction, using reconstructed body geometry as a scaffold for contact reasoning. Existing approaches either focus on contact prediction without sufficiently exploiting explicit 3D human priors, or emphasize pose/mesh reconstruction without directly optimizing robust vertex-level contact inference under occlusion and perceptual noise. To address this gap, we propose GraphiContact, a pose-aware framework that transfers complementary human priors from two pretrained Transformer encoders and predicts per-vertex human-scene contact on the reconstructed mesh. To improve robustness in real-world scenarios, we further introduce a Single-Image Multi-Infer Uncertainty (SIMU) training strategy with token-level adaptive routing, which simulates occlusion and noisy observations during training while preserving efficient single-branch inference at test time. Experiments on five benchmark datasets show that GraphiContact achieves consistent gains on both contact prediction and 3D human reconstruction. Our code, based on the GraphiContact method, provides comprehensive 3D human reconstruction and interaction analysis, and will be publicly available at https://github.com/Aveiro-Lin/GraphiContact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15888v2">AsgardBench -- Evaluating Visually Grounded Interactive Planning Under Minimal Feedback</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 19 figures, 6 tables, including appendix
    </div>
    <details class="paper-abstract">
      With AsgardBench we aim to evaluate visually grounded, high-level action sequence generation and interactive planning, focusing specifically on plan adaptation during execution based on visual observations rather than navigation or low-level manipulation. In the landscape of embodied AI benchmarks, AsgardBench targets the capability category of interactive planning, which is more sophisticated than offline high-level planning as it requires agents to revise plans in response to environmental feedback, yet remains distinct from low-level execution. Unlike prior embodied AI benchmarks that conflate reasoning with navigation or provide rich corrective feedback that substitutes for perception, AsgardBench restricts agent input to images, action history, and lightweight success/failure signals, isolating interactive planning in a controlled simulator without low-level control noise. The benchmark contains 108 task instances spanning 12 task types, each systematically varied through object state, placement, and scene configuration. These controlled variations create conditional branches in which a single instruction can require different action sequences depending on what the agent observes, emphasizing conditional branching and plan repair during execution. Our evaluations of leading vision language models show that performance drops sharply without visual input, revealing weaknesses in visual grounding and state tracking that ultimately undermine interactive planning. Our benchmark zeroes in on a narrower question: can a model actually use what it sees to adapt a plan when things do not go as expected?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02458v2">Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Computer Vision
    </div>
    <details class="paper-abstract">
      Embodied agents are expected to assist humans by actively exploring unknown environments and reasoning about spatial contexts. When deployed in real life, agents often face sequential tasks where each new task follows the completion of the previous one and may include infeasible objectives, such as searching for non-existent objects. However, most existing research focuses on isolated goals, overlooking the core challenge of sequential tasks: the ability to reuse spatial knowledge accumulated from previous explorations to guide subsequent reasoning and exploration. In this work, we investigate this underexplored yet practically significant embodied AI challenge. Specifically, we propose 3DSPMR, a 3D SPatial Memory Reasoning framework that utilizes Field-of-View (FoV) coverage as an explicit geometric prior. By integrating FoV-based constraints, 3DSPMR significantly enhances an agent's memory, reasoning, and exploration capabilities across sequential tasks. To facilitate research in this area, we further introduce SEER-Bench, a novel Sequential Embodied Exploration and Reasoning Benchmark that spans two foundational tasks: Embodied Question Answering (EQA) and Embodied Multi-modal Navigation (EMN). SEER-Bench uniquely incorporates both feasible and infeasible tasks to provide a rigorous and comprehensive evaluation of agent performance. Extensive experiments verify that 3DSPMR achieves substantial performance gains on both sequential EQA and EMN tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20285v1">AgentComm-Bench: Stress-Testing Cooperative Embodied AI Under Latency, Packet Loss, and Bandwidth Collapse</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Cooperative multi-agent methods for embodied AI are almost universally evaluated under idealized communication: zero latency, no packet loss, and unlimited bandwidth. Real-world deployment on robots with wireless links, autonomous vehicles on congested networks, or drone swarms in contested spectrum offers no such guarantees. We introduce AgentComm-Bench, a benchmark suite and evaluation protocol that systematically stress-tests cooperative embodied AI under six communication impairment dimensions: latency, packet loss, bandwidth collapse, asynchronous updates, stale memory, and conflicting sensor evidence. AgentComm-Bench spans three task families: cooperative perception, multi-agent waypoint navigation, and cooperative zone search, and evaluates five communication strategies, including a lightweight method we propose based on redundant message coding with staleness-aware fusion. Our experiments reveal that communication-dependent tasks degrade catastrophically: stale memory and bandwidth collapse cause over 96% performance drops in navigation, while content corruption (stale or conflicting data) reduces perception F1 by over 85%. Vulnerability depends on the interaction between impairment type and task design; perception fusion is robust to packet loss but amplifies corrupted data. Redundant message coding more than doubles navigation performance under 80% packet loss. We release AgentComm-Bench as a practical evaluation protocol and recommend that cooperative embodied AI work report performance under multiple impairment conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16669v1">Kinema4D: Kinematic 4D World Modeling for Spatiotemporal Embodied Simulation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Project page: https://mutianxu.github.io/Kinema4D-project-page/
    </div>
    <details class="paper-abstract">
      Simulating robot-world interactions is a cornerstone of Embodied AI. Recently, a few works have shown promise in leveraging video generations to transcend the rigid visual/physical constraints of traditional simulators. However, they primarily operate in 2D space or are guided by static environmental cues, ignoring the fundamental reality that robot-world interactions are inherently 4D spatiotemporal events that require precise interactive modeling. To restore this 4D essence while ensuring the precise robot control, we introduce Kinema4D, a new action-conditioned 4D generative robotic simulator that disentangles the robot-world interaction into: i) Precise 4D representation of robot controls: we drive a URDF-based 3D robot via kinematics, producing a precise 4D robot control trajectory. ii) Generative 4D modeling of environmental reactions: we project the 4D robot trajectory into a pointmap as a spatiotemporal visual signal, controlling the generative model to synthesize complex environments' reactive dynamics into synchronized RGB/pointmap sequences. To facilitate training, we curated a large-scale dataset called Robo4D-200k, comprising 201,426 robot interaction episodes with high-quality 4D annotations. Extensive experiments demonstrate that our method effectively simulates physically-plausible, geometry-consistent, and embodiment-agnostic interactions that faithfully mirror diverse real-world dynamics. For the first time, it shows potential zero-shot transfer capability, providing a high-fidelity foundation for advancing next-generation embodied simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16050v1">The Era of End-to-End Autonomy: Transitioning from Rule-Based Driving to Large Driving Models</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Autonomous driving is undergoing a shift from modular rule based pipelines toward end to end (E2E) learning systems. This paper examines this transition by tracing the evolution from classical sense perceive plan control architectures to large driving models (LDMs) capable of mapping raw sensor input directly to driving actions. We analyze recent developments including Tesla's Full Self Driving (FSD) V12 V14, Rivian's Unified Intelligence platform, NVIDIA Cosmos, and emerging commercial robotaxi deployments, focusing on architectural design, deployment strategies, safety considerations and industry implications. A key emerging product category is supervised E2E driving, often referred to as FSD (Supervised) or L2 plus plus, which several manufacturers plan to deploy from 2026 onwards. These systems can perform most of the Dynamic Driving Task (DDT) in complex environments while requiring human supervision, shifting the driver's role to safety oversight. Early operational evidence suggests E2E learning handles the long tail distribution of real world driving scenarios and is becoming a dominant commercial strategy. We also discuss how similar architectural advances may extend beyond autonomous vehicles (AV) to other embodied AI systems, including humanoid robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16044v1">Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Generalization remains a core challenge in embodied AI, as robots must adapt to diverse environments. While OpenVLA represents the State-of-the-Art (SOTA) in Vision-Language-Action models by leveraging large-scale pre-training, its zero-shot performance can be limited when encountering completely new environments. This paper proposes a parameter-efficient fine-tuning strategy to enhance the linguistic generalization of OpenVLA by synthesizing a general instruction set for the Bridge Dataset V2. The paper leverages a Large Language Model (LLM) to generate a rich variety of semantically equivalent but structurally diverse commands for existing trajectories. In this experiment, Low-Rank Adaptation (LoRA) is implemented to fine-tune OpenVLA on augmented pairs, allowing the model to bridge the gap between complex natural language intent and robotic actions. Results demonstrate that the LoRA-enhanced model's robustness, suggesting that enriching the linguistic space of specialized datasets is crucial for embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15885v1">Resilience Meets Autonomy: Governing Embodied AI in Critical Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Critical infrastructure increasingly incorporates embodied AI for monitoring, predictive maintenance, and decision support. However, AI systems designed to handle statistically representable uncertainty struggle with cascading failures and crisis dynamics that exceed their training assumptions. This paper argues that Embodied AIs resilience depends on bounded autonomy within a hybrid governance architecture. We outline four oversight modes and map them to critical infrastructure sectors based on task complexity, risk level, and consequence severity. Drawing on the EU AI Act, ISO safety standards, and crisis management research, we argue that effective governance requires a structured allocation of machine capability and human judgement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15612v1">HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 https://yukangcao.github.io/HSImul3R/
    </div>
    <details class="paper-abstract">
      We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos. Existing methods suffer from a perception-simulation gap: visually plausible reconstructions often violate physical constraints, leading to instability in physics engines and failure in embodied AI applications. To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry. In the forward direction, we employ Scene-targeted Reinforcement Learning to optimize human motion under dual supervision of motion fidelity and contact stability. In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry. We further present HSIBench, a new benchmark with diverse objects and interaction scenarios. Extensive experiments demonstrate that HSImul3R produces the first stable, simulation-ready HSI reconstructions and can be directly deployed to real-world humanoid robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15558v1">Panoramic Affordance Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations, often missing critical holistic environmental context. In this paper, we present the first exploration into Panoramic Affordance Prediction, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. To facilitate this novel task, we first introduce PAP-12K, a large-scale benchmark dataset containing over 1,000 ultra-high-resolution (12k, 11904 x 5952) panoramic images with over 12k carefully annotated QA pairs and affordance masks. Furthermore, we propose PAP, a training-free, coarse-to-fine pipeline inspired by the human foveal visual system to tackle the ultra-high resolution and severe distortion inherent in panoramic images. PAP employs recursive visual routing via grid prompting to progressively locate targets, applies an adaptive gaze mechanism to rectify local geometric distortions, and utilizes a cascaded grounding pipeline to extract precise instance-level masks. Experimental results on PAP-12K reveal that existing affordance prediction methods designed for standard perspective images suffer severe performance degradation and fail due to the unique challenges of panoramic vision. In contrast, PAP framework effectively overcomes these obstacles, significantly outperforming state-of-the-art baselines and highlighting the immense potential of panoramic perception for robust embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14811v1">Ego to World: Collaborative Spatial Reasoning in Embodied Systems via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Understanding the world from distributed, partial viewpoints is a fundamental challenge for embodied multi-agent systems. Each agent perceives the environment through an ego-centric view that is often limited by occlusion and ambiguity. To study this problem, we introduce the Ego-to-World (E2W) benchmark, which evaluates a vision-language model's ability to fuse heterogeneous viewpoints across three tasks: (i) global counting, (ii) relational location reasoning, and (iii) action-oriented grasping that requires predicting view-specific image coordinates. To address this setting, we propose CoRL, a two-stage framework that combines Chain-of-Thought supervised fine-tuning with reinforcement learning using Group-Relative Policy Optimization. Its core component, the Cross-View Spatial Reward (CVSR), provides dense task-aligned feedback by linking reasoning steps to visual evidence, ensuring coherent cross-view entity resolution, and guiding the model toward correct final predictions. Experiments on E2W show that CoRL consistently surpasses strong proprietary and open-source baselines on both reasoning and perception-grounding metrics, while ablations further confirm the necessity of each CVSR component. Beyond that, CoRL generalizes to external spatial reasoning benchmarks and enables effective real-world multi-robot manipulation with calibrated multi-camera rigs, demonstrating cross-view localization and successful grasp-and-place execution. Together, E2W and CoRL provide a principled foundation for learning world-centric scene understanding from distributed, ego-centric observations, advancing collaborative embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14371v1">OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-03-15
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Embodied AI agents increasingly require parallel execution of multiple tasks, such as manipulation, conversation, and memory construction, from shared observations under distinct time constraints. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) architecturally support such heterogeneous outputs, yet existing inference systems fail to achieve efficient multi-task parallelism for on-device deployment due to redundant computation and resource contention. We identify isolated KV cache management as the root cause. To address this, we propose unified KV cache management, an inference paradigm that treats KV cache as a first-class shared resource across tasks and over time. This abstraction enables two key optimizations: cross-task KV sharing eliminates redundant prefill of shared observations, while cross-frame continuous batching decouples variable-length language decoding from fixed-rate action generation across control cycles. We implement this paradigm for $π_{0.5}$, the most popular MoT VLA, and evaluate under representative robotic configurations. OxyGen achieves up to 3.7$\times$ speedup over isolated execution, delivering over 200 tokens/s language throughput and 70 Hz action frequency simultaneously without action quality degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14076v1">SGR-OCC: Evolving Monocular Priors for Embodied 3D Occupancy Prediction via Soft-Gating Lifting and Semantic-Adaptive Geometric Refinement</a></div>
    <div class="paper-meta">
      📅 2026-03-14
      | 💬 mian paper: 20 pages, 6 figures; appendix: 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D semantic occupancy prediction is a cornerstone for embodied AI, enabling agents to perceive dense scene geometry and semantics incrementally from monocular video streams. However, current online frameworks face two critical bottlenecks: the inherent depth ambiguity of monocular estimation that causes "feature bleeding" at object boundaries , and the "cold start" instability where uninitialized temporal fusion layers distort high-quality spatial priors during early training stages. In this paper, we propose SGR-OCC (Soft-Gating and Ray-refinement Occupancy), a unified framework driven by the philosophy of "Inheritance and Evolution". To perfectly inherit monocular spatial expertise, we introduce a Soft-Gating Feature Lifter that explicitly models depth uncertainty via a Gaussian gate to probabilistically suppress background noise. Furthermore, a Dynamic Ray-Constrained Anchor Refinement module simplifies complex 3D displacement searches into efficient 1D depth corrections along camera rays, ensuring sub-voxel adherence to physical surfaces. To ensure stable evolution toward temporal consistency, we employ a Two-Phase Progressive Training Strategy equipped with identity-initialized fusion, effectively resolving the cold start problem and shielding spatial priors from noisy early gradients. Extensive experiments on the EmbodiedOcc-ScanNet and Occ-ScanNet benchmarks demonstrate that SGR-OCC achieves state-of-the-art performance. In local prediction tasks, SGR-OCC achieves a completion IoU of 58.55$\%$ and a semantic mIoU of 49.89$\%$, surpassing the previous best method, EmbodiedOcc++, by 3.65$\%$ and 3.69$\%$ respectively. In challenging embodied prediction tasks, our model reaches 55.72$\%$ SC-IoU and 46.22$\%$ mIoU. Qualitative results further confirm our model's superior capability in preserving structural integrity and boundary sharpness in complex indoor environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17657v2">SPACE-CLIP: Spatial Perception via Adaptive CLIP Embeddings for Monocular Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-03-14
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pre-training (CLIP) provides strong semantic representations, but it is not designed for dense geometric prediction. Most CLIP-based monocular depth methods still rely on text prompts and image-text matching, which adds indirection and inference overhead. We propose SPACE-CLIP, a decoder-only framework that predicts depth directly from a frozen CLIP vision encoder and fully bypasses the text encoder. Its decoder fuses scene-level context from FiLM-conditioned semantic features with fine spatial cues from shallow layers. Under the TFI-FB constraint (text-free inference and frozen vision backbone), SPACE-CLIP achieves AbsRel 0.0901 on KITTI and 0.1042 on NYU Depth V2. These results, together with ablations, show that hierarchical fusion of semantic and structural cues is effective while preserving modularity for embodied AI systems such as vision-language-action (VLA) models. We also observe stable training behavior across both datasets with the same frozen-backbone setting, which supports reproducible deployment in integration-constrained pipelines. Our model is available at https://github.com/taewan2002/space-clip
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13615v1">Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      To serve as a scalable data source for embodied AI, world models should act as true simulators that infer interaction dynamics strictly from user actions, rather than mere conditional video generators relying on privileged future object states. In this context, egocentric Human-Object Interaction (HOI) world models are critical for predicting physically grounded first-person rollouts. However, building such models is profoundly challenging due to rapid head motions, severe occlusions, and high-DoF hand articulations that abruptly alter contact topologies. Consequently, existing approaches often circumvent these physics challenges by resorting to conditional video generation with access to known future object trajectories. We introduce EgoHOI, an egocentric HOI world model that breaks away from this shortcut to simulate photorealistic, contact-consistent interactions from action signals alone. To ensure physical accuracy without future-state inputs, EgoHOI distills geometric and kinematic priors from 3D estimates into physics-informed embeddings. These embeddings regularize the egocentric rollouts toward physically valid dynamics. Experiments on the HOT3D dataset demonstrate consistent gains over strong baselines, and ablations validate the effectiveness of our physics-informed design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12936v1">MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins</a></div>
    <div class="paper-meta">
      📅 2026-03-13
      | 💬 5 figures
    </div>
    <details class="paper-abstract">
      Converting static 3D meshes into interactable articulated assets is crucial for embodied AI and robotic simulation. However, existing zero-shot pipelines struggle with complex assets due to a critical lack of physical grounding. Specifically, ungrounded Vision-Language Models (VLMs) frequently suffer from kinematic hallucinations, while unconstrained joint estimation inevitably leads to catastrophic mesh inter-penetration during physical simulation. To bridge this gap, we propose MotionAnymesh, an automated zero-shot framework that seamlessly transforms unstructured static meshes into simulation-ready digital twins. Our method features a kinematic-aware part segmentation module that grounds VLM reasoning with explicit SP4D physical priors, effectively eradicating kinematic hallucinations. Furthermore, we introduce a geometry-physics joint estimation pipeline that combines robust type-aware initialization with physics-constrained trajectory optimization to rigorously guarantee collision-free articulation. Extensive experiments demonstrate that MotionAnymesh significantly outperforms state-of-the-art baselines in both geometric precision and dynamic physical executability, providing highly reliable assets for downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08096v2">TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically plausible but geometrically inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at https://cwru-aism.github.io/triangulang/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12639v1">RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-13
    </div>
    <details class="paper-abstract">
      Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11755v1">Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Motion-controllable video generation is crucial for egocentric applications in virtual reality and embodied AI. However, existing methods often struggle to achieve 3D-consistent fine-grained hand articulation. By adopting on 2D trajectories or implicit poses, they collapse 3D geometry into spatially ambiguous signals or over rely on human-centric priors. Under severe egocentric occlusions, this causes motion inconsistencies and hallucinated artifacts, as well as preventing cross-embodiment generalization to robotic hands. To address these limitations, we propose a novel framework that generates egocentric videos from a single reference frame, leveraging sparse 3D hand joints as embodiment-agnostic control signals with clear semantic and geometric structures. We introduce an efficient control module that resolves occlusion ambiguities while fully preserving 3D information. Specifically, it extracts occlusion-aware features from the source reference frame by penalizing unreliable visual signals from hidden joints, and employs a 3D-based weighting mechanism to robustly handle dynamically occluded target joints during motion propagation. Concurrently, the module directly injects 3D geometric embeddings into the latent space to strictly enforce structural consistency. To facilitate robust training and evaluation, we develop an automated annotation pipeline that yields over one million high-quality egocentric video clips paired with precise hand trajectories. Additionally, we register humanoid kinematic and camera data to construct a cross-embodiment benchmark. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art baselines, generating high-fidelity egocentric videos with realistic interactions and exhibiting exceptional cross-embodiment generalization to robotic hands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07791v3">GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 ICLR 2026, 31 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for autonomous driving, embodied AI and general AI. Existing spatial-temporal benchmarks mainly focus on egocentric (first-person) perspective reasoning using images/video contexts, or geographic reasoning with graphical context (e.g., maps), thus fail to assess VLMs' geographic spatial-temporal intelligence that requires integrating both images/video and graphical context, which is crucial for real-world scenarios such as traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench show that even the best proprietary model, Gemini-2.5-Pro (34.9\%), significantly lags behind human performance (78.61\%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three major deficiencies of current models for geo-temporal reasoning. (1) VLMs exhibit imbalanced utilization of spatial and temporal context during reasoning. (2) they show weak temporal forecasting ability, leading to poorer performance on temporally focused tasks. (3) they lack the capability to effectively align and integrate map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26685v1">Contextual Graph Representations for Task-Driven 3D Perception and Planning</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 University of Toronto Undergraduate Thesis, 2021. 85 pages, 24 figures
    </div>
    <details class="paper-abstract">
      Recent advances in computer vision facilitate fully automatic extraction of object-centric relational representations from visual-inertial data. These state representations, dubbed 3D scene graphs, are a hierarchical decomposition of real-world scenes with a dense multiplex graph structure. While 3D scene graphs claim to promote efficient task planning for robot systems, they contain numerous objects and relations when only small subsets are required for a given task. This magnifies the state space that task planners must operate over and prohibits deployment in resource constrained settings. This thesis tests the suitability of existing embodied AI environments for research at the intersection of robot task planning and 3D scene graphs and constructs a benchmark for empirical comparison of state-of-the-art classical planners. Furthermore, we explore the use of graph neural networks to harness invariances in the relational structure of planning domains and learn representations that afford faster planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11320v1">UniCompress: Token Compression for Unified Vision-Language Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Unified models aim to support both understanding and generation by encoding images into discrete tokens and processing them alongside text within a single autoregressive framework. This unified design offers architectural simplicity and cross-modal synergy, which facilitates shared parameterization, consistent training objectives, and seamless transfer between modalities. However, the large number of visual tokens required by such models introduces substantial computation and memory overhead, and this inefficiency directly hinders deployment in resource constrained scenarios such as embodied AI systems. In this work, we propose a unified token compression algorithm UniCompress that significantly reduces visual token count while preserving performance on both image understanding and generation tasks. Our method introduces a plug-in compression and decompression mechanism guided with learnable global meta tokens. The framework is lightweight and modular, enabling efficient integration into existing models without full retraining. Experimental results show that our approach reduces image tokens by up to 4 times, achieves substantial gains in inference latency and training cost, and incurs only minimal performance degradation, which demonstrates the promise of token-efficient unified modeling for real world multimodal applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08420v2">Human-Aware Robot Behaviour in Self-Driving Labs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Self-driving laboratories (SDLs) are rapidly transforming research in chemistry and materials science to accelerate new discoveries. Mobile robot chemists (MRCs) play a pivotal role by autonomously navigating the lab to transport samples, effectively connecting synthesis, analysis, and characterisation equipment. The instruments within an SDL are typically designed or retrofitted to be accessed by both human and robotic chemists, ensuring operational flexibility and integration between manual and automated workflows. In many scenarios, human and robotic chemists may need to use the same equipment simultaneously. Currently, MRCs rely on simple LiDAR-based obstruction detection, which forces the robot to passively wait if a human is present. This lack of situational awareness leads to unnecessary delays and inefficient coordination in time-critical automated workflows in human-robot shared labs. To address this, we present an initial study of an embodied, AI-driven perception method that facilitates proactive human-robot interaction in shared-access scenarios. Our method features a hierarchical human intention prediction model that allows the robot to distinguish between preparatory actions (waiting) and transient interactions (accessing the instrument). Our results demonstrate that the proposed approach enhances efficiency by enabling proactive human-robot interaction, streamlining coordination, and potentially increasing the efficiency of autonomous scientific labs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v4">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2305.17066v2">Mindstorms in Natural Language-Based Societies of Mind</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 published in Computational Visual Media Journal (CVMJ); 9 pages in main text + 7 pages of references + 38 pages of appendices, 14 figures in main text + 13 in appendices, 7 tables in appendices
    </div>
    <details class="paper-abstract">
      Both Minsky's "society of mind" and Schmidhuber's "learning to think" inspire diverse societies of large multimodal neural networks (NNs) that solve problems by interviewing each other in a "mindstorm." Recent implementations of NN-based societies of minds consist of large language models (LLMs) and other NN-based experts communicating through a natural language interface. In doing so, they overcome the limitations of single LLMs, improving multimodal zero-shot reasoning. In these natural language-based societies of mind (NLSOMs), new agents -- all communicating through the same universal symbolic language -- are easily added in a modular fashion. To demonstrate the power of NLSOMs, we assemble and experiment with several of them (having up to 129 members), leveraging mindstorms in them to solve some practical AI tasks: visual question answering, image captioning, text-to-image synthesis, 3D generation, egocentric retrieval, embodied AI, and general language-based task solving. We view this as a starting point towards much larger NLSOMs with billions of agents-some of which may be humans. And with this emergence of great societies of heterogeneous minds, many new research questions have suddenly become paramount to the future of artificial intelligence. What should be the social structure of an NLSOM? What would be the (dis)advantages of having a monarchical rather than a democratic structure? How can principles of NN economies be used to maximize the total reward of a reinforcement learning NLSOM? In this work, we identify, discuss, and try to answer some of these questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09827v2">MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      As embodied models become powerful, humans will collaborate with multiple embodied AI agents at their workplace or home in the future. To ensure better communication between human users and the multi-agent system, it is crucial to interpret incoming information from agents in parallel and refer to the appropriate context for each query. Existing challenges include effectively compressing and communicating high volumes of individual sensory inputs in the form of video and correctly aggregating multiple egocentric videos to construct system-level memory. In this work, we first formally define a novel problem of understanding multiple long-horizon egocentric videos simultaneously collected from embodied agents. To facilitate research in this direction, we introduce MultiAgent-EgoQA (MA-EgoQA), a benchmark designed to systemically evaluate existing models in our scenario. MA-EgoQA provides 1.7k questions unique to multiple egocentric streams, spanning five categories: social interaction, task coordination, theory-of-mind, temporal reasoning, and environmental interaction. We further propose a simple baseline model for MA-EgoQA named EgoMAS, which leverages shared memory across embodied agents and agent-wise dynamic retrieval. Through comprehensive evaluation across diverse baselines and EgoMAS on MA-EgoQA, we find that current approaches are unable to effectively handle multiple egocentric streams, highlighting the need for future advances in system-level understanding across the agents. The code and benchmark are available at https://ma-egoqa.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25268v2">SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08260v1">Seed2Scale: A Self-Evolving Data Engine for Embodied AI via Small to Large Model Synergy and Multimodal Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Existing data generation methods suffer from exploration limits, embodiment gaps, and low signal-to-noise ratios, leading to performance degradation during self-iteration. To address these challenges, we propose Seed2Scale, a self-evolving data engine that overcomes the data bottleneck through a heterogeneous synergy of "small-model collection, large-model evaluation, and target-model learning". Starting with as few as four seed demonstrations, the engine employs the lightweight Vision-Language-Action model, SuperTiny, as a dedicated collector, leveraging its strong inductive bias for robust exploration in parallel environments. Concurrently, a pre-trained Vision-Language Model is integrated as a Verifer to autonomously perform success/failure judgment and quality scoring for the massive generated trajectories. Seed2Scale effectively mitigates model collapse, ensuring the stability of the self-evolution process. Experimental results demonstrate that Seed2Scale exhibits signifcant scaling potential: as iterations progress, the success rate of the target model shows a robust upward trend, achieving a performance improvement of 131.2%. Furthermore, Seed2Scale signifcantly outperforms existing data augmentation methods, providing a scalable and cost-effective pathway for the large-scale development of Generalist Embodied AI. Project page: https://terminators2025.github.io/Seed2Scale.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08131v1">UniGround: Universal 3D Visual Grounding via Training-Free Scene Parsing</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 14 pages,6 figures,3 tables
    </div>
    <details class="paper-abstract">
      Understanding and localizing objects in complex 3D environments from natural language descriptions, known as 3D Visual Grounding (3DVG), is a foundational challenge in embodied AI, with broad implications for robotics, augmented reality, and human-machine interaction. Large-scale pre-trained foundation models have driven significant progress on this front, enabling open-vocabulary 3DVG that allows systems to locate arbitrary objects in a given scene. However, their reliance on pre-trained models constrains 3D perception and reasoning within the inherited knowledge boundaries, resulting in limited generalization to unseen spatial relationships and poor robustness to out-of-distribution scenes. In this paper, we replace this constrained perception with training-free visual and geometric reasoning, thereby unlocking open-world 3DVG that enables the localization of any object in any scene beyond the training data. Specifically, the proposed UniGround operates in two stages: a Global Candidate Filtering stage that constructs scene candidates through training-free 3D topology and multi-view semantic encoding, and a Local Precision Grounding stage that leverages multi-scale visual prompting and structured reasoning to precisely identify the target object. Experiments on ScanRefer and EmbodiedScan show that UniGround achieves 46.1\%/34.1\% Acc@0.25/0.5 on ScanRefer and 28.7\% Acc@0.25 on EmbodiedScan, establishing a new state-of-the-art among zero-shot methods on EmbodiedScan without any 3D supervision. We further evaluate UniGround in real-world environments under uncontrolled reconstruction conditions and substantial domain shift, showing training-free reasoning generalizes robustly beyond curated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08096v1">TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically plausible but geometrically inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at https://cwru-aism.github.io/triangulang/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08021v1">AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24099v3">Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 Under review at ICLR 2026
    </div>
    <details class="paper-abstract">
      Generating realistic, context-aware two-person motion conditioned on diverse modalities remains a fundamental challenge for graphics, animation and embodied AI systems. Real-world applications such as VR/AR companions, social robotics and game agents require models capable of producing coordinated interpersonal behaviour while flexibly switching between interactive and reactive generation. We introduce DualFlow, the first unified and efficient framework for multi-modal two-person motion generation. DualFlow conditions 3D motion generation on diverse inputs, including text, music, and prior motion sequences. Leveraging rectified flow, it achieves deterministic straight-line sampling paths between noise and data, reducing inference time and mitigating error accumulation common in diffusion-based models. To enhance semantic grounding, DualFlow employs a novel Retrieval-Augmented Generation (RAG) module for two-person motion that retrieves motion exemplars using music features and LLM-based text decompositions of spatial relations, body movements, and rhythmic patterns. We use a contrastive rectified flow objective to further sharpen alignment with conditioning signals and add synchronisation loss to improve inter-person temporal coordination. Extensive evaluations across interactive, reactive, and multi-modal benchmarks demonstrate that DualFlow consistently improves motion quality, responsiveness, and semantic fidelity. DualFlow achieves state-of-the-art performance in two-person multi-modal motion generation, producing coherent, expressive, and rhythmically synchronized motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06280v1">SuperSuit: An Isomorphic Bimodal Interface for Scalable Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      High-quality, long-horizon demonstrations are essential for embodied AI, yet acquiring such data for tightly coupled wheeled mobile manipulators remains a fundamental bottleneck. Unlike fixed-base systems, mobile manipulators require continuous coordination between $SE(2)$ locomotion and precise manipulation, exposing limitations in existing teleoperation and wearable interfaces. We present \textbf{SuperSuit}, a bimodal data acquisition framework that supports both robot-in-the-loop teleoperation and active demonstration under a shared kinematic interface. Both modalities produce structurally identical joint-space trajectories, enabling direct data mixing without modifying downstream policies. For locomotion, SuperSuit maps natural human stepping to continuous planar base velocities, eliminating discrete command switches. For manipulation, it employs a strictly isomorphic wearable arm in both modes, while policy training is formulated in a shift-invariant delta-joint representation to mitigate calibration offsets and structural compliance without inverse kinematics. Real-world experiments on long-horizon mobile manipulation tasks show 2.6$\times$ higher demonstration throughput in active mode compared to a teleoperation baseline, comparable policy performance when substituting teleoperation data with active demonstrations at fixed dataset size, and monotonic performance improvement as active data volume increases. These results indicate that consistent kinematic representations across collection modalities enable scalable data acquisition for long-horizon mobile manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17100v4">Generative Models in Decision Making: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project page:https://github.com/xyshao23/Awesome-Generative-Models-for-Decision-Making-Taxonomy
    </div>
    <details class="paper-abstract">
      Generative models have fundamentally reshaped the landscape of decision-making, reframing the problem from pure scalar reward maximization to high-fidelity trajectory generation and distribution matching. This paradigm shift addresses intrinsic limitations in classical Reinforcement Learning (RL), particularly the limited expressivity of standard unimodal policy distributions in capturing complex, multi-modal behaviors embedded in diverse datasets. However, current literature often treats these models as isolated algorithmic improvements, rarely synthesizing them into a single comprehensive framework. This survey proposes a principled taxonomy grounding generative decision-making within the probabilistic framework of Control as Inference. By performing a variational factorization of the trajectory posterior, we conceptualize four distinct functional roles: Controllers for amortized policy inference, Modelers for dynamics priors, Optimizers for iterative trajectory refinement, and Evaluators for trajectory guidance and value assessment. Unlike existing architecture-centric reviews, this function-centric framework allows us to critically analyze representative generative families across distinct dimensions. Furthermore, we examine deployment in high-stakes domains, specifically Embodied AI, Autonomous Driving, and AI for Science, highlighting systemic risks such as dynamics hallucination in world models and proxy exploitation. Finally, we chart the path toward Generalist Physical Intelligence, identifying pivotal challenges in inference efficiency, trustworthiness, and the emergence of Physical Foundation Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21161v2">MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project Page: https://xuhu0529.github.io/MarketGen
    </div>
    <details class="paper-abstract">
      The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05684v3">D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations and 1K+ hours of pseudo-labeled gameplay), our 1B-parameter model achieves 96.6% success on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models up to 7x larger, such as π_{0} (3.3B) and OpenVLA (7B). These results demonstrate that sensorimotor primitives learned from digital interactions transfer effectively to real-world physical tasks, establishing desktop pretraining as a practical paradigm for embodied AI. All resources are publicly available at https://worv-ai.github.io/d2e.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04457v1">Capability Thresholds and Manufacturing Topology: How Embodied Intelligence Triggers Phase Transitions in Economic Geography</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      The fundamental topology of manufacturing has not undergone a paradigm-level transformation since Henry Ford's moving assembly line in 1913. Every major innovation of the past century, from the Toyota Production System to Industry 4.0, has optimized within the Fordist paradigm without altering its structural logic: centralized mega-factories, located near labor pools, producing at scale. We argue that embodied intelligence is poised to break this century-long stasis, not by making existing factories more efficient, but by triggering phase transitions in manufacturing economic geography itself. When embodied AI capabilities cross critical thresholds in dexterity, generalization, reliability, and tactile-vision fusion, the consequences extend far beyond cost reduction: they restructure where factories are built, how supply chains are organized, and what constitutes viable production scale. We formalize this by defining a Capability Space C = (d, g, r, t) and showing that the site-selection objective function undergoes topological reorganization when capability vectors cross critical surfaces. Through three pathways, weight inversion, batch collapse, and human-infrastructure decoupling, we show that embodied intelligence enables demand-proximal micro-manufacturing, eliminates "manufacturing deserts," and reverses geographic concentration driven by labor arbitrage. We further introduce Machine Climate Advantage: once human workers are removed, optimal factory locations are determined by machine-optimal conditions (low humidity, high irradiance, thermal stability), factors orthogonal to traditional siting logic, creating a production geography with no historical precedent. This paper establishes Embodied Intelligence Economics, the study of how physical AI capability thresholds reshape the spatial and structural logic of production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02008v1">Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Effective exploration in reinforcement learning requires not only tracking where an agent has been, but also understanding how the agent perceives and represents the world. To learn powerful representations, an agent should actively explore states that contribute to its knowledge of the environment. Temporal representations can capture the information necessary to solve a wide range of potential tasks while avoiding the computational cost associated with full state reconstruction. In this paper, we propose an exploration method that leverages temporal contrastive representations to guide exploration, prioritizing states with unpredictable future outcomes. We demonstrate that such representations can enable the learning of complex exploratory x in locomotion, manipulation, and embodied-AI tasks, revealing capabilities and behaviors that traditionally require extrinsic rewards. Unlike approaches that rely on explicit distance learning or episodic memory mechanisms (e.g., quasimetric-based methods), our method builds directly on temporal similarities, yielding a simpler yet effective strategy for exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19400v2">Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026. Camera-ready version. Project page: https://aaronfengzy.github.io/MV-RoboBench-Webpage/
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating CoT-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15018v2">UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026. Project page: https://urbanverseproject.github.io/
    </div>
    <details class="paper-abstract">
      Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01452v1">Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Developing generalist robots capable of mastering diverse skills remains a central challenge in embodied AI. While recent progress emphasizes scaling model parameters and offline datasets, such approaches are limited in robotics, where learning requires active interaction. We argue that effective online learning should scale the \emph{number of tasks}, rather than the number of samples per task. This regime reveals a structural advantage of model-based reinforcement learning (MBRL). Because physical dynamics are invariant across tasks, a shared world model can aggregate multi-task experience to learn robust, task-agnostic representations. In contrast, model-free methods suffer from gradient interference when tasks demand conflicting actions in similar states. Task diversity therefore acts as a regularizer for MBRL, improving dynamics learning and sample efficiency. We instantiate this idea with \textbf{EfficientZero-Multitask (EZ-M)}, a sample-efficient multi-task MBRL algorithm for online learning. Evaluated on \textbf{HumanoidBench}, a challenging whole-body control benchmark, EZ-M achieves state-of-the-art performance with significantly higher sample efficiency than strong baselines, without extreme parameter scaling. These results establish task scaling as a critical axis for scalable robotic learning. The project website is available \href{https://yewr.github.io/ez_m/}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01414v1">Jailbreaking Embodied LLMs via Action-level Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 This paper has been officially accepted for ACM SenSys 2026
    </div>
    <details class="paper-abstract">
      Embodied Large Language Models (LLMs) enable AI agents to interact with the physical world through natural language instructions and actions. However, beyond the language-level risks inherent to LLMs themselves, embodied LLMs with real-world actuation introduce a new vulnerability: instructions that appear semantically benign may still lead to dangerous real-world consequences, revealing a fundamental misalignment between linguistic security and physical outcomes. In this paper, we introduce Blindfold, an automated attack framework that leverages the limited causal reasoning capabilities of embodied LLMs in real-world action contexts. Rather than iterative trial-and-error jailbreaking of black-box embodied LLMs, Blindfold adopts an Adversarial Proxy Planning strategy: it compromises a local surrogate LLM to perform action-level manipulations that appear semantically safe but could result in harmful physical effects when executed. Blindfold further conceals key malicious actions by injecting carefully crafted noise to evade detection by defense mechanisms, and it incorporates a rule-based verifier to improve the attack executability. Evaluations on both embodied AI simulators and a real-world 6DoF robotic arm show that Blindfold achieves up to 53% higher attack success rates than SOTA baselines, highlighting the urgent need to move beyond surface-level language censorship and toward consequence-aware defense mechanisms to secure embodied LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23893v2">AoE: Always-on Egocentric Human Video Collection for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Embodied foundation models require large-scale, high-quality real-world interaction data for pre-training and scaling. However, existing data collection methods suffer from high infrastructure costs, complex hardware dependencies, and limited interaction scope, making scalable expansion challenging. In fact, humans themselves are ideal physically embodied agents. Therefore, obtaining egocentric real-world interaction data from globally distributed "human agents" offers advantages of low cost and sustainability. To this end, we propose the Always-on Egocentric (AoE) data collection system, which aims to simplify hardware dependencies by leveraging humans themselves and their smartphones, enabling low-cost, highly efficient, and scene-agnostic real-world interaction data collection to address the challenge of data scarcity. Specifically, we first employ an ergonomic neck-mounted smartphone holder to enable low-barrier, large-scale egocentric data collection through a cloud-edge collaborative architecture. Second, we develop a cross-platform mobile APP that leverages on-device compute for real-time processing, while the cloud hosts automated labeling and filtering pipelines that transform raw videos into high-quality training data. Finally, the AoE system supports distributed Ego video data collection by anyone, anytime, and anywhere. We evaluate AoE on data preprocessing quality and downstream tasks, demonstrating that high-quality egocentric data significantly boosts real-world generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18041v7">Openfly: A comprehensive platform for aerial vision-language navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02271v1">Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 3 Pages 4 Figures for Workshop paper
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are an emerging class of workloads critical for robotics and embodied AI at the edge. As these models scale, they demonstrate significant capability gains, yet they must be deployed locally to meet the strict latency requirements of real-time applications. This paper characterizes VLA performance on two generations of edge hardware, viz. the Nvidia Jetson Orin and Thor platforms. Using MolmoAct-7B, a state-of-the-art VLA model, we identify a primary execution bottleneck: up to 75% of end-to-end latency is consumed by the memory-bound action-generation phase. Through analytical modeling and simulations, we project the hardware requirements for scaling to 100B parameter models. We also explore the impact of high-bandwidth memory technologies and processing-in-memory (PIM) as promising future pathways in edge systems for embodied AI.
    </details>
</div>
