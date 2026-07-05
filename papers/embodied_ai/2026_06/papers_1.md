# embodied ai - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30608v2">UnfoldArt: Zero-Shot Recovery of Full Articulated 3D Objects from Text or Image</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Project page: https://aminebdj.github.io/unfoldart
    </div>
    <details class="paper-abstract">
      Articulated 3D objects are essential for interactive environments in embodied AI, robotics, and virtual reality, but reconstructing their structure and motion from sparse observations remains challenging. Existing approaches remain largely constrained by lack of supervised data or lack the priors needed to reliably recover articulation, hidden geometry, and internal object structure. We present the first debate-driven agentic approach to articulated 3D object reconstruction from text or image inputs that both grounds articulation reasoning in concrete motion and exposes the occluded geometry revealed under articulation. High-level agents reason about object semantics and motion using knowledge from vision-language and video models, while low-level agents estimate articulation parameters and interaction points; together, they engage in a two-round structured debate that first exploits global--local disagreement and then grounds the agents in freely generated video. The same video prior, conditioned on the agreed articulation, then drives each part through its motion to expose occluded interiors and geometry that cannot be inferred from a single static view. By combining agentic reasoning with a video generative prior, our approach jointly infers articulation and reconstructs complete 3D articulated objects, producing high-fidelity geometry, internal structure, and motion-consistent states beyond directly observed surfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31772v1">Autonomous UAV Navigation for Individual Wildlife Re-Identification</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted at 2026 CV4Animals Workshop at CVPR
    </div>
    <details class="paper-abstract">
      Reliable individual re-identification (re-ID) of wildlife is essential for population monitoring, behavioral tracking, and conservation policy evaluation, yet large-scale data collection remains labor-intensive, relying on manual efforts by ecologists or citizen scientists. We propose an autonomous drone navigation system that actively optimizes image capture for downstream re-ID, moving beyond passive aerial sensing. The system combines YOLOv11 object detection with a DINOv2-based pose classifier to guide real-time flight decisions: detecting animals, orienting to expose the lateral flank (the surface of interest for pattern-based re-ID), and approaching until the subject meets a minimum bounding-box threshold. Unlike prior drone systems that optimize for group-level behavioral video, ours targets the specific image-quality requirements of individual-identification models. We demonstrate feasibility through a case study on zebra using footage collected in Kenya, and show the approach generalizes to other species with diagnostic surface patterns, including giraffes, tigers, and elephants. Our work establishes a framework for task-aware embodied AI for ecological data collection, in which downstream re-ID requirements drive real-time perception and control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31388v1">One Video, One World: Turning Monocular Video into Physical 4D Scenes</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted by ECCV 2026. Project Page: https://OneVideoOneWorld.github.io/
    </div>
    <details class="paper-abstract">
      We introduce \textbf{OVOW}, the first training-free system that reconstructs \emph{instance-level, simulation-ready} 4D mesh scenes from a single monocular video. Recent 4D reconstruction achieves impressive rendering quality, but its outputs (\eg, implicit fields, Gaussian primitives, or point clouds) lack the watertight topology, instance separation, and standardized physical interfaces required by physics simulators and embodied AI. OVOW closes this gap with a four-stage pipeline: a vision-language model discovers, labels, and motion-classifies all instances; category-aware reconstruction yields per-instance meshes for rigid objects and topology-consistent mesh sequences for deformable ones; an iterative render-match-optimize procedure recovers metric scale and 6-DoF pose trajectories; and physics-grounded assembly enforces ground contact and inter-object support. Crucially, we model all motion, rigid and non-rigid, through direct vertex deformation without category-specific priors or skeleton rigging, producing watertight mesh scenes ready for downstream physics simulation and editing. We further establish the first benchmark for \emph{structured Video-to-4D} evaluation, with metrics for geometric correctness, instance separation, and physical plausibility beyond visual fidelity; the same pipeline doubles as a scalable engine for \emph{synthesizing} paired video-to-4D simulation data for future 4D world models and embodied AI. Across two synthetic benchmarks (static and 4D), OVOW attains the best overall layout and geometry accuracy and the lowest photometric and semantic error among all baselines, and on monocular video runs one to two orders of magnitude faster than the baselines, while downstream physics simulation confirms its physical stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.31086v1">CasaMaestro: Multi-View Panoramas for House-Scale 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-06-30
      | 💬 Accepted to ECCV2026
    </div>
    <details class="paper-abstract">
      The rise of home-deployed embodied AI systems is driving a growing need for fast, metric 3D reconstruction of residential spaces to support navigation, interaction, and long-horizon task execution. However, the commonly used pinhole-camera 3D reconstruction pipelines struggle to model large indoor residences efficiently due to their limited field of view, to which achieving full coverage across multiple rooms often requires thousands of images and incurs drift from long chains of incremental alignment. In this work, we present CasaMaestro (Spanish words meaning ``house'' and ``master''), a feedforward model that can take only twenty to fifty sparse multi-view indoor panoramas as input and directly predicts metric depth along with camera poses, allowing fast point-cloud reconstruction of the entire house with full coverage. CasaMaestro is the first model that supports house-scale reconstruction with multi-view panoramas. Experiments show that CasaMaestro can robustly provide high quality results in both real-world and synthetic scenes, which can serve as a strong foundation for acquiring house-scale 3D indoor assets to be applied in close-loop simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30638v1">Open-Vocabulary and Referring Segmentation for 3D Gaussians Using 2D Detectors</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      3D Gaussian Splatting (3DGS) has emerged at the forefront of 3D scene reconstruction. Extending 3DGS with language-driven, open-vocabulary understanding has gained significant attention for real-world applications such as embodied AI. Recent methods achieve this by learning an instance feature attribute and assigning semantics by distilling high-dimensional Contrastive Language-Image Pretraining (CLIP) features directly into the scene representation. However, the instance grouping mechanisms of these methods either require a predefined number of instances or suffer from noise in their bottom-up grouping strategies. Furthermore, the reliance on CLIP restricts semantic understanding to simple noun phrases, preventing complex spatial reasoning and referential expression grounding. We present GaussDet, a method that circumvents the need for dense CLIP features by leveraging discrete, open-vocabulary 2D object detectors with referring expression capabilities. We learn instance features for individual Gaussians to decompose the scene into 3D instance groups. By rendering these groups and aggregating semantic votes from multi-view 2D detections, we generate a robust View-Aggregated Semantic Label Distribution (VASD) for each 3D instance. This view-aggregation strategy acts as a strong regularizer, attenuating spurious labels caused by low-quality instance grouping. Our approach enables a straightforward, zero-shot extension from simple language queries to complex referential grounding. Extensive evaluations across two key tasks -- open-vocabulary segmentation (LeRF-OVS, ScanNet) and referring expression grounding (Ref-LeRF) -- demonstrate that GaussDet achieves consistent improvements over existing methods. Most notably, we achieve a substantial 16.7% mIoU improvement in referential grounding within a strict zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30308v1">The Surprising Effectiveness of Video Diffusion Models for Hand Motion Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      4D hand motion reconstruction from egocentric video is bottlenecked by clear limitations of existing methods: image-based pipelines depend on a detector that fails under heavy occlusion, while video-based methods rely on temporal modules learned only from scarce hand-pose annotations, a narrow signal insufficient to model motion dynamics, occlusion reasoning, and hand-object interaction. These capabilities, however, are exactly what video generative models must implicitly acquire when trained to synthesize coherent video at internet scale. Motivated by this, we present ViDiHand, which leverages the representations of a pretrained video diffusion model to reconstruct 4D two-hand pose. We adapt it via a hand-overlay rendering objective that specializes its features for hands while preserving its world priors. A decoder then recovers metric-scale pose from the adapted features. The whole pipeline operates directly on full frames--no detector, no infiller, and no test-time optimization. On ARCTIC, HOT3D, and HOI4D, ViDiHand substantially outperforms prior methods, establishing video diffusion models as a powerful new foundation for hand motion reconstruction and a promising route to scalable in-the-wild data collection for embodied AI. Project page: https://vidihand.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02742v2">Consistent Yet Wrong: Evidence Insensitivity in Spatial Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      Spatial reasoning is fundamental to robotics, autonomy, and embodied AI, yet modern vision-language models (VLMs) remain unreliable on metric distance queries. A common assumption is that consistent predictions across viewpoints reflect geometric grounding. We test this assumption and find the opposite: leading VLMs often produce view-invariant and consistent answers even when those answers are incorrect, indicating weak coupling between predictions and viewpoint-specific visual evidence. We introduce \textbf{ViewDiag}, a controlled multi-view evaluation protocol built from Hypersim, ScanNet, and KITTI360, comprising 176 object-pair tracks across 80 scenes with 2--10 views per track. The protocol evaluates models along three axes: metric accuracy, distributional concentration, and internal collapse, the last of which is assessed using a latent feature probe. Across diverse models, we observe a consistent pattern of high prediction stability paired with substantial error, clustering in a regime characterized by strong consistency but low accuracy. \noindent These results challenge the common use of cross-view consistency as a proxy for geometric understanding. Instead, we show that stable predictions may reflect prior-driven collapse rather than evidence-sensitive reasoning. ViewDiag provides a controlled benchmark and diagnostic framework for evaluating whether spatial VLMs are not only accurate, but also meaningfully coupled to visual evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17969v3">E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 Project page: https://k0uya.github.io/e3vs-proj/
    </div>
    <details class="paper-abstract">
      Visual search in 3D environments requires embodied agents to actively explore their surroundings and acquire task-relevant evidence. However, existing visual search and embodied AI benchmarks, including EQA, typically rely on static observations or constrained egocentric motion, and thus do not explicitly evaluate fine-grained viewpoint-dependent phenomena that arise under unrestricted 5-DoF viewpoint control in real-world 3D environments, such as visibility changes caused by vertical viewpoint shifts, revealing contents inside containers, and disambiguating object attributes that are only observable from specific angles. To address this limitation, we introduce {E3VS-Bench}, a benchmark for embodied 3D visual search where agents must control their viewpoints in 5-DoF to gather viewpoint-dependent evidence for question answering. E3VS-Bench consists of 99 high-fidelity 3D scenes reconstructed using 3D Gaussian Splatting and 2,014 question-driven episodes. 3D Gaussian Splatting enables photorealistic free-viewpoint rendering that preserves fine-grained visual details (e.g., small text and subtle attributes) often degraded in mesh-based simulators, thereby allowing the construction of questions that cannot be answered from a single view and instead require active inspection across viewpoints in 5-DoF. We evaluate multiple state-of-the-art VLMs and compare their performance with humans. Despite strong 2D reasoning ability, all models exhibit a substantial gap from humans, highlighting limitations in active perception and coherent viewpoint planning specifically under full 5-DoF viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.30014v1">Shell-Supervised Gaussian Splatting for Urban Real-to-Sim Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-06-29
      | 💬 10 pages main paper, 2 pages supplementary material
    </div>
    <details class="paper-abstract">
      Real-to-sim reconstruction for embodied AI requires geometry that is useful for collision reasoning, navigation, and agent-environment interaction, not only photorealistic novel-view synthesis. However, close-range urban facades are difficult for video-to-3D reconstruction: glass, reflections, repeated windows, and weak texture can produce visually plausible renderings with unstable surface geometry. We introduce shell-supervised Gaussian Splatting, a reconstruction-stage framework that uses an external facade structural shell as lightweight geometric supervision for video-driven Gaussian reconstruction. The method aligns an exterior shell to the video reconstruction frame, renders per-view depth, camera-space normal, and valid-mask maps, and applies these cues through mask-gated losses during Gaussian optimization. This design preserves RGB-driven appearance while regularizing only visible shell-supported facade regions. Experiments on anonymized close-range urban facade scenes show improved facade orientation and visible-surface point-cloud consistency over photo-only, monocular-cue, and surface-oriented Gaussian baselines, while maintaining comparable held-out rendering quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29850v1">Efficient Visual Pointing for Embodied AI:Agent-Driven Data Synthesis, Cross-Block Attention, and Iterative Correction</a></div>
    <div class="paper-meta">
      📅 2026-06-29
    </div>
    <details class="paper-abstract">
      Visual pointing maps a language instruction to pixel co ordinates, a core skill for embodied AI. We describe our PointArena 2026 solution, which achieves 77.2% overall accuracy and ranks second on the benchmark. The ap proach targets three failure modes. First, agent-driven syn thesis builds large semantic and anchor-relative candidate pools; the server inventory contains 55,372 processed out puts, 53,772 de-duplicated sample IDs, and 37,574 train able completed or accepted rows. Second, a determinis tic steerable-data pipeline creates a verified 10,000-sample main set, plus reserve samples, using masks, templates, and path verification. Third, two model-side modules address complementary errors: AttnRes adds gated cross-block at tention for steerability, while ABC correction encodes per turbed coordinates with visual features for general coordi nate grounding. Category-aware routing combines comple mentary specialists; local validation used to select experts records 93.9% Affordance, 82.6% Spatial Relation, 78.2% Reasoning, 70.4% Counting, and 63.0% Steerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.29384v1">Event-VLA: Action-Conditioned Event Fusion for Robust Vision-Language-Action Model</a></div>
    <div class="paper-meta">
      📅 2026-06-28
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have become an important paradigm of embodied AI. However, existing VLA models typically assume well-lit and stable indoor settings, while real-world embodied manipulation may involve degraded RGB observations caused by illumination shifts, posing critical challenges for robust robotic manipulation. To address this gap, we propose \textbf{Event-VLA}, an event-enhanced VLA framework for generalizable manipulation across varying illumination conditions. We formulate VLA-based manipulation under degraded visibility as a practical robustness problem for RGB-centric policies, and introduce event streams as an illumination-robust, motion-sensitive complementary observation to improve robustness across visibility levels. Specifically, unlike conventional multimodal fusion that directly merges event features into the global semantic token space, Event-VLA injects event information through an action-query routing pathway. It uses learnable action queries to extract task-relevant semantics from the VLA reasoning process, and selectively aggregates event tokens via gated cross-attention to construct event-aware action representations. This design preserves the pretrained RGB-language semantic priors while effectively leveraging event information for robust action prediction. Experiments in simulation and real-world deployment show that Event-VLA maintains strong manipulation performance under normal lighting and improves success rates under low-light degradation and near-dark real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07306v2">BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-06-28
      | 💬 17 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19119v2">MonoSR: Open-Vocabulary Spatial Reasoning from Monocular Images</a></div>
    <div class="paper-meta">
      📅 2026-06-28
      | 💬 Accepted by ECCV 2026
    </div>
    <details class="paper-abstract">
      Spatial reasoning (SR), the ability to infer 3D spatial information from 2D inputs, is essential for real-world applications such as embodied AI and autonomous driving. However, existing research primarily focuses on indoor environments and typically relies on multi-view observations, which limits their generalizability to outdoor scenarios and constrains their applicability to monocular images, the most common real-world setting. In this work, we propose MonoSR, a large-scale monocular spatial reasoning dataset that spans diverse scenarios including indoor, outdoor, and object-centric settings, and supports multiple question types. MonoSR provides a path toward open-world monocular spatial reasoning. Beyond introducing the dataset, we evaluate advanced vision-language models to reveal their limitations on this challenging task. We further analyze whether auxiliary information is crucial for monocular spatial reasoning and offer practical guidance for designing future models. These contributions collectively establish a foundation for advancing monocular spatial reasoning in real-world, open-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28215v1">HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-06-26
      | 💬 Accepted to ECCV 2026. 15 pages of main text and 39 pages of appendices. Project page: https://lijiaxin0111.github.io/HAT4D/
    </div>
    <details class="paper-abstract">
      Extracting dynamic 4D object interactions from massive, in-the-wild monocular videos offers a highly efficient data collection pathway for scaling Embodied AI and training VLAs. However, existing monocular 4D reconstruction methods primarily focus on isolated objects, often failing under the severe occlusions and complex dynamics inherent in multi-object interactions. To bridge this gap, we propose HAT-4D, the first agentic framework designed to reconstruct the 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single video. By integrating VLMs with a multi-level human-in-the-loop feedback mechanism, HAT-4D efficiently resolves depth ambiguities and interaction-induced occlusions during 3D generation and 4D propagation, yielding physically plausible assets without relying on expensive multicamera rigs. As a scalable data engine, HAT-4D facilitates the creation of MVOIK-4D, an open-world benchmark for monocular 4D interaction reconstruction, accompanied by a novel multi-dimensional evaluation protocol focused on physical plausibility and temporal consistency. Extensive experiments demonstrate that HAT-4D achieves SOTA performance on most evaluation metrics, while maintaining competitive semantic alignment. Ablation studies show that introducing a small amount of human feedback improves interaction reconstruction. Moreover, the data produced by HAT-4D effectively improves baseline performance when used for fine-tuning. Our data and code are available at https://lijiaxin0111.github.io/HAT4D/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.27929v1">When Multi-Robot Systems Meet Agentic AI:Towards Embodied Collective Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-26
    </div>
    <details class="paper-abstract">
      Embodied AI is increasingly becoming agentic, shifting robots from perception--control pipelines towards closed-loop systems that can retrieve context, deliberate during execution, monitor feedback, and refine future behavior. In parallel, robotics research has also moved from single-robot autonomy towards multi-robot systems, driven by the need for wider sensing, distributed action, heterogeneous capabilities, and fault tolerance. As AI agents move from single-agent use towards multi-agent collaboration, robotics faces a parallel challenge: robot teams must move beyond sharing maps, task assignments, and datasets towards sharing the state produced by embodied agent loops. This article explores Embodied Collective Intelligence (ECI), a future multi-robot paradigm in which a robot team accumulates and uses world context, task progress, and skill experience as shared resources. Specifically, we first review how embodied AI is becoming agentic and how multi-robot cooperation has evolved. We then present Embodied Collective Intelligence through Co-Perception, Co-Action, and Co-Evolution. Finally, we use an illustrative navigation study to examine one concrete component of the concept: shared world-memory inheritance. The study shows that a newly added robot can benefit from merged team memory, but it is not intended as a full evaluation of the ECI framework. Taken together, the review and conceptual framework motivate Embodied Collective Intelligence as a direction for embodied multi-agent intelligence, while the case study grounds one measurable part of the concept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26964v2">Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds</a></div>
    <div class="paper-meta">
      📅 2026-06-26
      | 💬 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      As embodied AI and world models increasingly operate in dynamic 3D environments, visual perception must move beyond passively interpreting given observations toward actively deciding what to observe. We study this problem through camera planning in dynamic 3D story worlds, where the camera must not only generate smooth motion, but also decide what visual evidence should be acquired before it moves. We formulate this capability as Narrative-Grounded World Visual Attention, where the camera acts as an embodied observer that determines what to observe, how to compose the observation, and how to shift attention over time under narrative intent and physical 3D constraints. To realize this capability, we propose Look-Before-Move, a camera planning framework that separates observation specification from motion execution. It first builds a Semantic Observation Contract to convert directorial intent into executable visual constraints, then performs Monte Carlo Viewpoint Search to find narrative-compliant and geometrically feasible viewpoints, and finally applies Semantic Trajectory Grounding to connect selected viewpoints into continuous, collision-aware, and temporally coherent camera motion. We further construct a dynamic 3D Story World Benchmark based on StoryBlender, covering 50 stories, 457 scenes, and 1585 shots with animated characters, semantic scene configurations, and executable 3D environments. Experiments show that our framework improves subject perception, intent consistency, and trajectory quality over representative baselines, demonstrating the importance of organizing visual attention before generating camera motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26964v1">Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      As embodied AI and world models increasingly operate in dynamic 3D environments, visual perception must move beyond passively interpreting given observations toward actively deciding what to observe. We study this problem through camera planning in dynamic 3D story worlds, where the camera must not only generate smooth motion, but also decide what visual evidence should be acquired before it moves. We formulate this capability as Narrative-Grounded World Visual Attention, where the camera acts as an embodied observer that determines what to observe, how to compose the observation, and how to shift attention over time under narrative intent and physical 3D constraints. To realize this capability, we propose Look-Before-Move, a camera planning framework that separates observation specification from motion execution. It first builds a Semantic Observation Contract to convert directorial intent into executable visual constraints, then performs Monte Carlo Viewpoint Search to find narrative-compliant and geometrically feasible viewpoints, and finally applies Semantic Trajectory Grounding to connect selected viewpoints into continuous, collision-aware, and temporally coherent camera motion. We further construct a dynamic 3D Story World Benchmark based on StoryBlender, covering 50 stories, 457 scenes, and 1585 shots with animated characters, semantic scene configurations, and executable 3D environments. Experiments show that our framework improves subject perception, intent consistency, and trajectory quality over representative baselines, demonstrating the importance of organizing visual attention before generating camera motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16732v3">A Comprehensive Survey on World Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-25
      | 💬 https://github.com/Li-Zn-H/AwesomeWorldModels
    </div>
    <details class="paper-abstract">
      Embodied AI requires agents that perceive, act, and anticipate how actions reshape future world states. World models serve as internal simulators that capture environment dynamics, enabling forward and counterfactual rollouts to support perception, prediction, and decision making. This survey presents a unified framework for world models in embodied AI. Specifically, we formalize the problem setting and learning objectives, and propose a three-axis taxonomy encompassing: (1) Functionality, Decision-Coupled vs. General-Purpose; (2) Temporal Modeling, Sequential Simulation and Inference vs. Global Difference Prediction; (3) Spatial Representation, Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, and Decomposed Rendering Representation. We systematize data resources and metrics across robotics, autonomous driving, and general video settings, covering pixel prediction quality, state-level understanding, and task performance. Furthermore, we offer a quantitative comparison of state-of-the-art models and distill key open challenges, including the scarcity of unified datasets and the need for evaluation metrics that assess physical consistency over pixel fidelity, the trade-off between model performance and the computational efficiency required for real-time control, and the core modeling difficulty of achieving long-horizon temporal consistency while mitigating error accumulation. Finally, we maintain a curated bibliography at https://github.com/Li-Zn-H/AwesomeWorldModels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.25337v1">AI Coaching for Accelerating Human Skill Development with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-24
    </div>
    <details class="paper-abstract">
      AI copilots can substantially boost human performance through shared control, but excessive assistance can induce over-reliance and skill atrophy. This paper studies how an embodied AI agent can act as a coach that accelerates human motor-skill development. We argue that effective coaching requires strategic scaffolding and stepping back that are aligned with the learner's capability, allowing productive failures that drive learning. We formalize the interactive AI coaching process as a non-cooperative dynamic game in which the learner optimizes task performance while the coach targets the learner's independent competence. Building on this formalism, we develop a reinforcement learning framework combining adaptive shared control with probabilistic models of the coach's causal influence on skill evolution, enabling tractable training of coaching policies. A comprehensive user study (N=33) on first-person-view drone racing shows significant gains in human learning outcomes over state-of-the-art AI coaching baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24767v1">Compact Object-Level Representations with Open-Vocabulary Understanding for Indoor Visual Relocalization</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Accepted by RA-L 2026
    </div>
    <details class="paper-abstract">
      Indoor visual relocalization plays a critical role in emerging spatial and embodied AI applications. However, prior research was predominantly devoted to low-level vision schemes, struggling to perceive scene semantics and compositions, which limits both interpretability and applicability. In this paper, we explore the issue of how to organize rich object information in a scene, including semantics, layout, and geometry, into a structured map representation, thereby utilizing object units exclusively to drive the camera relocalization task. To this end, we propose OpenReLoc, a camera relocalization system designed to provide scene understanding and accurate pose estimation capabilities. Leveraging recent foundation models, we first introduce a multi-modal mechanism to integrate open-vocabulary semantic knowledge for effective 2D-3D object matching. Additionally, we design object-oriented reference frames as position priors, paired with a reference frame selection strategy based on the Distance-IoU (DIOU), enabling extension to scalable scenes. Moreover, to ensure stable and accurate pose optimization, we also propose a dual-path 2D Iterative Closest Pixel loss guided by object shape. Experimental results demonstrate that OpenReLoc achieves superior relocalization recall and accuracy across various datasets. Our source code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.24628v1">ArtiTwinSplat: Interactable Digital Twin Reconstruction via Gaussian Splatting from RGB-D videos</a></div>
    <div class="paper-meta">
      📅 2026-06-23
      | 💬 Presented at the ICRA 2026 Workshop on Advances and Challenges in AI-Driven Automation and Robotic System Integration with Digital Twins, Vienna, June 2026
    </div>
    <details class="paper-abstract">
      Deploying robots in unstructured real-world environments needs accurate, interactive models of the objects. Constructing these models at scale remains a critical bottleneck for robotic system integration. We present ArtiTwinSplat, a framework that automatically constructs articulated, photo-realistic digital twins of objects directly from RGB-D videos, requiring no CAD models, simulation assets, or manual annotations. Our method is built on 3D Gaussian Splatting that preserve geometric fidelity and photometric realism, coupled with an unsupervised articulation discovery pipeline that recovers part structure and joint kinematics from observed motion alone. With tracking and optimization stages our method provides stable, queryable digital twins that support real-time rendering, viewpoint control, and interactive manipulation. Unlike prior methods confined to simulation, ArtiTwinSplat operates directly on real-world observations and produces twins that are immediately usable by downstream robot planning and learning systems. This method offers a practical, scalable pathway toward digital twin construction, lowering the integration barrier for articulated object manipulation in embodied AI and human-robot collaboration contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23675v1">IMAGIN-4D: Image-Guided Controllable Interaction Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 15 pages, 8 figures. Project page: https://imagin4d.github.io
    </div>
    <details class="paper-abstract">
      Generating human-object interactions (HOI) is central to character animation, robotics, AR/VR, and embodied AI. Recent HOI generation methods synthesize motion from text, object geometry, and sparse waypoints, controlling action semantics and object trajectories. However, these signals underspecify interaction: the same prompt and trajectory can produce different grasps, approach directions, body poses, object poses, contacts, and body-object layouts. We address this ambiguity with a reference image as a visual specification of the desired interaction snapshot. However, a single global image representation conflates distinct cues and conditions all frames on identical visual evidence. We therefore introduce IMAGIN-4D, a diffusion-based HOI generator that decomposes image conditioning spatio-temporally. For spatial conditioning, IMAGIN-4D extracts supervised interaction-state tokens for body pose, object pose, body-object contact, and spatial relationships at the depicted frame. For temporal conditioning, it computes frame-aware tokens by querying image patches per generated frame, allowing sequence segments to attend to different visual cues from the same image. To balance image, text, and waypoint cues, IMAGIN-4D uses role-aware conditioning: text, waypoints, and interaction-state tokens use separate AdaLN streams, while frame-aware visual tokens cross-attend with motion tokens. Since HOI motion datasets lack paired images, we build a synthetic motion-to-image rendering pipeline from FullBodyManipulation (FBM) and introduce an image-adherence metric to evaluate whether generated motions match the reference snapshot. Experiments on FBM and BEHAVE show that IMAGIN-4D improves fine-grained interaction control over single-token and uniformly image-conditioned baselines while preserving waypoint-following and motion quality. Code and models will be released at https://imagin4d.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23565v1">HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      LLM agents follow a practical execution loop in digital environments: they reason over structured states, invoke tools, inspect feedback, and revise actions. Extending this loop to physical robots is difficult because physical execution is continuous, embodiment-dependent, uncertain, and constrained by safety. Existing embodied-AI systems have advanced manipulation, spatial understanding, navigation, and humanoid control, but these capabilities often remain specialized modules or loosely coupled decision loops. In this work, we introduce HoloAgent-0, a unified embodied agent framework for real-world robot deployment. Embodied AgentOS converts language instructions into executable skill graphs, schedules robot resources, monitors execution, and triggers clarification or re-planning from runtime feedback. HoloAgent-0 organizes heterogeneous robot models and controllers through three coupled layers: Embodied AgentOS for closed-loop execution, 3D spatial memory for physical world grounding, and embodied skills for robot action. We deploy HoloAgent-0 on real hardware and evaluate its spatial memory, long-horizon navigation, and closed-loop execution across motion generation, object search, cross-robot coordination, and mobile manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23293v1">Flow6D: Discrete-to-Continuous Flow Matching for Efficient and Accurate Category-Level 6D Pose Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-22
      | 💬 Accepted for publication in IEEE Robotics and Automation Letters (RA-L), 2026
    </div>
    <details class="paper-abstract">
      6D pose estimation is a key task in computer vision and embodied AI, widely used in robotic manipulation, augmented reality, etc. Existing methods directly regress in a high-dimensional continuous space, facing two key challenges in category-level pose estimation: limited accuracy due to noise and local optima, and inefficient search over an infinite space that hinders real-time performance. This paper proposes Flow6D, a hierarchical flow matching framework with a two-stage discrete latent space localization-continuous pose regression strategy. Rotation and translation parameters are first discretized into bins, with a discrete flow matching model locking the latent space around the true pose to reduce search complexity. Then, by sampling in the latent space, a continuous flow matching model predicts local pose residuals to optimize the estimate and regress to an accurate pose. The framework also naturally extends to articulated objects, outperforming state-of-the-art methods on synthetic and real datasets with real-time inference at 70 FPS. Project website: https://flow6d.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.23256v1">P-JEPA: Procedural Video Representation Learning via Joint Embedding Predictive Architecture</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      The increasing maturity of embodied AI platforms has driven a growing interest in procedural video representation learning to support intelligent assistance systems for complex, multi-step tasks. Leveraging large-scale latent predictive training, video foundation models capture video dynamics, enabling downstream tasks such as activity understanding, spatiotemporal localization, and predictive control. However, procedural videos include actions with long-range dependencies that these models do not support, due to the quadratic complexity of self-attention. Distinct actions, for example, may be visually similar despite appearing at different points in the procedure, such as turning the stove on versus off. Here, we propose a backbone-agnostic approach that learns long-duration video representations by reducing the problem to a dense, frame-aligned action space and predicting pooled masked latent vectors. This approach allows our Procedural Joint Embedding Predictive Architecture (P-JEPA) to ingest videos over 30 minutes long, enabling effective long-form understanding of procedural steps. We evaluate P-JEPA using features extracted with VJEPA2.1, TSM, and I3D over the EgoExo4D, EgoProceL, and Assembly101 datasets, finding that it consistently improves linear separability, streaming inference, and temporal action segmentation performance, achieving state-of-the-art results on EgoExo4D fine-grained action classification while using an order of magnitude fewer parameters than LLM-based methods and running in real time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.22971v1">Humanoid-OmniOcc: Stereo-Based Full-View Occupancy Dataset for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-22
    </div>
    <details class="paper-abstract">
      Occupancy prediction at voxel-level granularity is essential for safe robotic navigation and interaction in complex environments. Existing occupancy datasets, however, are predominantly designed for autonomous driving with vehicle-centric biases -- forward-facing cameras, far-field geometry, and static road priors -- limiting their applicability to embodied humanoid perception. We present Humanoid-OmniOcc, a large-scale panoramic stereo-based occupancy dataset tailored for humanoid robots. The dataset encompasses 15 diverse simulated indoor scenes and 5 real-world environments, yielding over 155K samples with broad scene and style diversity. Importantly, the dataset is designed around a Real2Sim2Real closed-loop paradigm: real sensor specifications drive physically accurate simulation, simulation produces large-scale annotated training data, and models trained in simulation are directly evaluated on real-world captures -- enabling iterative refinement of the sim-to-real pipeline. We further propose \textbf{H}umanoid \textbf{S}urround \textbf{S}tereo-guided \textbf{Occ}upancy model (Humanoid-OmniOcc) that exploits robust depth priors for accurate 2D-to-3D lifting. Extensive experiments show that Humanoid-OmniOcc consistently outperforms monocular baselines and generalizes well to both unseen simulated test scenes and real-world environments, validating the effectiveness of the Real2Sim2Real design. Code and data will be available upon acceptance at https://d-robotics-ai-lab.github.io/humanoid-omniocc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.21018v1">LK Jam: System Architecture and Implementation of a Real-Time Human-AI Interactive Music Generation System using Role-Aware GRU</a></div>
    <div class="paper-meta">
      📅 2026-06-19
      | 💬 7 pages, 10 figures, 3 tables. This is an original technical report on real-time human-AI interactive symbolic music generation VST3 plugin based on GRU and JUCE. The source code is open-source on GitHub
    </div>
    <details class="paper-abstract">
      As artificial intelligence advances into the era of Embodied AI, live musical interaction urgently needs to break free from the limitations of offline, unidirectional generation, achieving a "virtual synergy" capable of low-latency, dynamic interplay. To address this, this technical report presents LK_Jam, a real-time, bidirectional human-computer interactive music generation system based on a lightweight Gated Recurrent Unit (GRU) and a high-performance audio host architecture. In the algorithmic representation layer, this system abandons the computationally expensive fixed time-grid. Instead, it constructs a multi-dimensional sparse event stream integrating time-shifts, continuous harmonic embeddings, and role-aware encoding, enabling the model to accurately capture turn-taking logic and micro-timing in a single-step inference. In the engineering implementation layer, this paper builds a strict multithreaded lock-free communication bridge using C++ and the JUCE framework, incorporating the RTNeural inference engine designed specifically for real-time audio. By utilizing compile-time network topology solidification and a zero-allocation (allocation-free) mechanism, the end-to-end overhead of autoregressive decoding is strictly locked at \(O(1)\) complexity, structurally mitigating the risk of audio thread dropouts in DAW plugin environments. Furthermore, this study designs a three-stage progressive training strategy, achieving a leap from basic chord harmonization to expert-level interaction. Preliminary observations and architectural analysis demonstrate that while ensuring musical coherence and interactive role-play, the proposed system successfully challenges extreme real-time engineering constraints, offering a highly robust and deployable technical paradigm for next-generation AI co-performers in live music.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15908v2">High-Fidelity 4D Hand-Object Capture via Multi-View Spatiotemporal Tracking and Physics-Aware Gaussians</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 Project page: https://hostpg.github.io/
    </div>
    <details class="paper-abstract">
      The growing demand for high-fidelity 4D hand-object interaction (HOI) data in embodied AI and spatial computing is currently bottlenecked by the reliance on pre-scanned object templates and physical markers. While recent methods have demonstrated promising results in reconstructing 4D hand-object interaction from videos, they are highly sensitive to initial estimates of hand and object poses. Yet, estimating these poses from images is challenging, in particular under severe occlusion which is inherent in hand-object interaction scenarios. We propose a novel system for the robust and accurate reconstruction of hands and objects from synchronized and calibrated multi-view videos without requiring any templates or markers. Our system consists of two main components with key innovations: (1) a multi-view feed-forward transformer model that aggregates cross-view geometry and temporal cues to provide a reliable, metric-consistent initialization for both poses and dense object geometry, and (2) a hand-object physics-aware Gaussian-based optimization framework to refine the initial estimates, integrating tetrahedral constraints, collision refinement, and appearance decomposition to produce physically plausible and visually accurate reconstruction. Validated on public benchmarks and an extensive internal dataset, our pipeline achieves highly robust, artifact-free reconstruction, providing an efficient foundation for automated 4D asset generation. Our project page are available at https://zyshen021.github.io/HOSTPG/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19983v1">A Measurement Study of Cryptographic Misuse in Embodied AI Mobile Applications</a></div>
    <div class="paper-meta">
      📅 2026-06-18
    </div>
    <details class="paper-abstract">
      Embodied AI (EAI) mobile applications are evolving from auxiliary user interfaces into active control-path components, directly linking mobile-side cryptographic security to cyber-physical trust. Despite this shift, existing security research predominantly focuses on embodied AI devices and cloud infrastructures, leaving the mobile control layer largely unexplored as a critical attack surface. To bridge this gap, we present the first large-scale measurement study of cryptographic misuse within the EAI mobile ecosystem. We construct EAIAppZoo, a benchmark of 507 real-world applications across six EAI domains, and employ an automated semantic-aware analysis pipeline to measure the prevalence and characteristics of five major cryptographic failure modes. Our measurement yields 12,975 misuse findings (with an evaluated precision of 80.74\%), revealing that these cryptographic failures are driven by EAI-specific engineering constraints rather than random developer errors. We uncover structural security trade-offs: latency-sensitive control paths systematically weaken transport protection, while the heavy reliance on offline device provisioning and legacy IoT SDKs exacerbates the local hardcoding of authentication credentials. Through real-world case studies, we demonstrate how these mobile-side cryptographic flaws bypass nominal network protections, enabling adversaries to intercept command channels and hijack the physical control of EAI entities. Ultimately, our findings highlight that mobile applications have become a fragile, yet overlooked, cryptographic trust boundary in cyber-physical systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31158v3">Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models</a></div>
    <div class="paper-meta">
      📅 2026-06-18
      | 💬 13 pages, 6 figures, 3 tables. Project page: https://2843721358l-del.github.io/Light-Interaction-Project/
    </div>
    <details class="paper-abstract">
      Interactive video world models generate video chunk by chunk in response to user-controlled camera movements, enabling applications such as real-time game simulation, virtual scene navigation, and embodied AI training. However, scaling to long interactive trajectories is prohibitively expensive due to growing context memory, quadratic attention complexity, and repeated denoising steps. We present Light Interaction, a training-free inference acceleration framework for interactive video world models. Our key insight is that interaction naturally enables trajectory-dependent adaptive computation: retrieved spatial memory can be discarded during novel exploration, temporal context can be adjusted according to local latent dynamics, and early-step model outputs can be reused when the camera revisits familiar regions. Based on this insight, Light Interaction combines adaptive context management, denoising cache acceleration, and hardware-software co-designed 3D block sparse attention with fused Triton kernels. Evaluated on HY-WorldPlay and Matrix-Game-3.0, Light Interaction achieves up to 2.59x speedup without model retraining while maintaining competitive visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19253v1">OneCanvas: 3D Scene Understanding via Panoramic Reprojection</a></div>
    <div class="paper-meta">
      📅 2026-06-17
      | 💬 Project page: https://baranowskibrt.github.io/onecanvas/
    </div>
    <details class="paper-abstract">
      Existing approaches to 3D scene understanding in Vision-Language Models (VLMs) either rely on complex, model-specific geometry encoders or large training budgets in pursuit of spatial reasoning. Instead, OneCanvas aggregates patch features from all views onto a single equirectangular panoramic canvas. Namely, each patch is unprojected to a 3D world coordinate using its depth and camera pose, then placed on the canvas at the continuous longitude and latitude of that point as seen from the canvas origin, with no rasterization or aggregation across overlapping views. A 3D position embedding of the patch's metric coordinates is added to its feature, restoring the depth lost when collapsing the world position to an angular canvas coordinate. Patches from all frames thus share one spatial coordinate system with no fusion or major architectural modifications of the backbone. The pretrained VLM consumes this representation as if it were an ordinary image. Because the canvas can be centered on any pose of interest, the same representation directly supports situated reasoning from a specific viewpoint, a common requirement in robotics and embodied AI. Thanks to this representation, we can also introduce a spatial pretraining curriculum: by procedurally placing patch features of objects, drawn from real images, at chosen 3D world positions on an otherwise empty canvas, we generate on-the-fly supervision spanning a broad range of spatial reasoning tasks, with answer distributions controlled to reduce spatial reasoning shortcuts. OneCanvas achieves state-of-the-art accuracy on SQA3D and VSI-Bench, and generalizes to out-of-distribution data on SPBench, using an order of magnitude less training compute than the strongest competing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.17639v2">ERQA-Plus: A Diagnostic Benchmark for Reasoning in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-17
    </div>
    <details class="paper-abstract">
      Generalist embodied agents require more than object recognition: they must reason about spatial relations, actions, procedures, human intentions, environmental constraints, and commonsense consequences from situated visual observations. Yet existing visual and embodied question answering benchmarks often provide limited control over the reasoning dependencies being tested, making it difficult to distinguish grounded embodied reasoning from shortcut-driven visual or linguistic pattern matching. We present ERQA-Plus, a diagnostic benchmark for reasoning in embodied AI. ERQA-Plus contains 1,766 question-answer instances grounded in 711 robot-centric images and organized according to a structured taxonomy spanning perceptual, action-centric, social-interaction, navigation-environmental, and contextual commonsense reasoning. The dataset is constructed using a multi-stage generation and validation pipeline that combines taxonomy-guided question generation, automatic quality judging, iterative revision, and human assessment to improve visual grounding, answer validity, and reasoning quality. We benchmark representative general-purpose vision-language models and embodied models, including LLaVA-NeXT-8B, Prismatic-7B, MiniCPM-V-4.5-8B, Qwen3-VL, RoboRefer-8B, and RoboBrain2.5-8B. Although the strongest model, Qwen3-VL-32B, achieves 83.4% overall accuracy and 61.4 SBERT score, category-level results reveal persistent weaknesses in spatial reasoning, procedural reasoning, event prediction, and intention inference. ERQA-Plus therefore provides a fine-grained evaluation framework for measuring not only whether embodied agents answer correctly, but also which forms of embodied reasoning they can and cannot perform reliably. The dataset is available https://huggingface.co/datasets/huggingdas/erqa-plus and the project page at https://github.com/LUNAProject22/erqa-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03609v3">A 3D Isovist World Model -- Revealing a City's Unseen Geometry and Its Emergent Cross-City Signature</a></div>
    <div class="paper-meta">
      📅 2026-06-16
    </div>
    <details class="paper-abstract">
      Embodied agents that navigate cities rely on world models that predict how their surroundings will change as they move. But for navigation, what matters is not what the buildings look like; it is where the agent can go. Most world models nonetheless predict appearance, learning how a scene looks rather than the space an agent can move through. Those that do target geometry, such as bird's-eye-view occupancy grids, flatten the three-dimensional environment onto a ground plane, discarding the above-ground and multi-level structure that shapes real navigation. What is missing is a predictive target that captures the navigable geometry an agent actually traverses, without photometric entanglement and without collapsing the third dimension. Our key idea is to model the open volume between buildings, the negative space, encoded as a 3D isovist: a spherical visibility-depth map recording the distance to the nearest surface in every direction. We introduce an embodied world model that predicts the next isovist from a short history of past isovists and a movement action. The prediction is formulated as a depth residual so the decoder inherits sharp building edges, trained with self-rollout scheduled sampling to keep corrupted context on the geometry manifold, and equipped with a persistent latent bird's-eye-view spatial map for cross-path consistency. Our central finding is emergent and unexpected: a single city-blind model trained on Manhattan and Paris develops a cross-city spatial signature, with city identity linearly decodable from its temporal latents far above single-frame baselines, so the signature lives in the learned dynamics rather than in appearance. The representation is lightweight, interpretable, and reproducible, offering a geometric substrate for spatial reasoning in embodied AI, robotics, and urban analysis, released with an open dataset and pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16996v1">ActiveSAM: Image-Conditional Class Pruning for Fast and Accurate Open-Vocabulary Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-06-15
      | 💬 Preprint. Code is available at https://github.com/VILA-Lab/ActiveSAM
    </div>
    <details class="paper-abstract">
      Segment Anything Model 3 (SAM 3) provides a strong frozen backbone for concept-prompted segmentation, but applying it directly to open-vocabulary semantic segmentation (OVSS) is inefficient: full-resolution decoding is typically run over the entire dataset vocabulary, whereas each image contains only a small active subset of classes. We introduce ActiveSAM, a training-free, zero-shot inference framework that turns SAM 3 into an active-vocabulary segmenter. ActiveSAM first canonicalizes and expands class prompts, then estimates an image-conditioned active set from a low-resolution presence preview. Only the retained classes are decoded at full resolution, using bucketed prompt multiplexing with the frozen SAM 3 decoder. The preview stage uses only class-presence evidence and skips unnecessary segmentation-head computation, while the final stage applies margin-aware background calibration to suppress low-confidence pixels. ActiveSAM requires no target-dataset training, no weight updates, and no oracle class-presence labels. Across eight OVSS benchmarks, ActiveSAM improves the speed-accuracy tradeoff of training-free open-vocabulary semantic segmentation, outperforming the current state-of-the-art SegEarth-OV3 by approximately +1.4 mIoU on average while running up to 5.5x faster on large-vocabulary datasets. ActiveSAM also demonstrates the strongest robustness under image corruption that simulates real-world distribution shift, making it well-suited for deployment in noisy-input domains such as autonomous driving and embodied AI. Code is available at https://github.com/VILA-Lab/ActiveSAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16436v1">V2P-Manip: Learning Dexterous Manipulation from Monocular Human Videos</a></div>
    <div class="paper-meta">
      📅 2026-06-15
    </div>
    <details class="paper-abstract">
      Achieving autonomous robotic dexterous manipulation requires precise, human-like action sequences at scale. As a scalable supplement to costly teleoperation data, extracting trajectories with both visual fidelity and physical plausibility from monocular videos represents a promising frontier in embodied AI. To this end, we introduce V2P-Manip, an efficient framework designed to learn dexterous manipulation policies directly from human demonstration videos. We establish an efficient, integrated pipeline encompassing 3D asset acquisition, trajectory estimation, and dexterous policy learning. To bridge the gap between visual perception and physical constraints, we introduce a two-stage refinement process to enforce spatial alignment and physical consistency. Evaluations on the TACO and OakInk benchmarks demonstrate that our approach significantly outperforms previous methods in pose accuracy, adaptability to unstructured environments, and training efficiency. Ultimately, experimental results confirm an average success rate of over 75% across multiple synthetic manipulation tasks and validate the adaptability of the extracted manipulation priors across diverse dexterous hand embodiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15898v1">VL2Spike: Spike-driven Distillation from VLMs for Low-Power Visual Perception in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-14
      | 💬 9 pages, 4 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Spiking neural networks (SNNs) are brain-inspired, event-driven models that compute with sparse spikes, which enables highly efficient visual perception in resource-constrained embodied AI models. The emergence of Spiking-Transformer models with spike self-attention has substantially improved the learning capacity of pure SNNs. Although SNNs are energy efficient, their performance is still limited by the spike-based architecture and optimization challenges, as standard gradient descent rules cannot be directly applied. Recently, vision-language models (VLMs) have shown rich multi-modal knowledge representation capabilities for visual perception. Thus, it is promising to leverage VLMs for better Spikformer training. To this end, we present VL2Spike, a novel spike-based knowledge distillation (KD) framework that bridges multi-modal knowledge from VLMs with compact Spikformer models. This design enhances the learning capacity of Spikformer models while preserving their energy-efficiency merits, thereby offering a practical pathway toward low-power robotic perception. Our VL2Spike brings two key technical contributions. To align with spiking dynamics, we first propose spatial-temporal visual spike (SVS) distillation, which achieves (1) shared manifold alignment between VLM image features and spike tokens, and (2) warm-started temporal consistency on membrane potentials and spike rates. We then design a novel spike prototype-guided linguistic (SPL) distillation strategy that aligns Spikformer's class prototypes and logits with promptable VLM text embeddings. Extensive experiments show that VL2Spike achieves 6.81% gain across three static datasets with only 15.7% energy consumption. It also exhibits strong generalization capacity on robotic visual place recognition (VPR) with a gain of 6.63%, highlighting its potential for low-power perception in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.15681v1">3D Consistency Optimization for Self-Supervised Monocular Video Depth Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-14
    </div>
    <details class="paper-abstract">
      Reliable monocular video depth estimation is crucial for downstream 3D reasoning and embodied AI in endoscopic navigation. However, existing self-supervised approaches typically treat video frames independently or rely on weak temporal regularization. These methods, lacking a holistic perception of the underlying 3D scene, inevitably suffer from geometrically inconsistent predictions and severe cross-frame drift. To address these limitations, we introduce a new paradigm that recasts sequential video depth estimation as an unconstrained multi-view 3D reconstruction problem, enabling full exploitation of the powerful geometric priors embedded in recent 3D foundation models. The core of our approach is a 3D consistency optimization framework driven by three constraints: image-level photometric rendering, explicit world-coordinate geometric alignment, and multi-scale temporal gradient consistency. Such unified optimization elegantly anchors isolated frames to a globally coherent 3D structure. Our method has been validated in both the self-supervised training scenarios and challenging zero-shot clinical environments. Results show that the proposed approach achieves state-of-the-art spatial accuracy, outperforming the frame-based, video-based depth estimators and the multi-view 3D reconstruction baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18464v3">AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-06-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models is severely bottlenecked by synchronization barriers and the high cost of environment data acquisition. To overcome these challenges, we propose AcceRL, a distributed asynchronous RL framework that physically isolates environment rollouts, model inference, and gradient updates. By eliminating the cascading long-tail idle bubbles inherent in synchronous systems, AcceRL maximizes hardware utilization and ensures scalable throughput. Furthermore, AcceRL features a modular design that supports the integration of diverse, plug-and-play world models into its distributed pipeline. Extensive experiments demonstrate that the base framework achieves highly competitive performance across all four LIBERO~\cite{liu2023libero} task suites. Systematically, the asynchronous architecture delivers a $2.4\times$ throughput speedup over leading synchronous baselines. Algorithmically, by leveraging a world model pre-trained on 1,000 offline trajectories, AcceRL achieves up to a $200\times$ improvement in online sample efficiency on LIBERO-Spatial, establishing a robust framework that is both sample-efficient and time-efficient for embodied AI. Code is included in the supplementary material. Code is available at https://github.com/distanceLu/AcceRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.14168v1">MUSE: Agentic 3D Scene Authoring via Memory-Grounded Incremental Requirement Satisfaction</a></div>
    <div class="paper-meta">
      📅 2026-06-12
    </div>
    <details class="paper-abstract">
      Text-driven 3D scene generation is a promising technique for digital content creation, embodied AI simulation, and interactive design, yet practical workflows often require refining, extending, or correcting existing scenes while preserving non-target content. Existing methods can produce realistic and structurally plausible scenes, but they generally lack editability with requirement-level state tracking, so part-level failures often lead to full-scene regeneration or manual intervention. To tackle this challenge, we formulate controllable 3D scene authoring as incremental requirement satisfaction, unifying construction and editing. In this paper, we present MUSE, a memory-grounded multi-agent framework in which an Architect compiles instructions into structured requirements, a Sculptor executes local scene operations, and an Inspector verifies each step while updating Working, Scene, and Skill Memory. To evaluate requirement-level controllability and preservation-aware editing, we introduce AuthorBench, offering 145 constrained construction cases and a 1,584-case preservation-aware editing pool paired with external structured checks. On full construction cases, MUSE improves All-Goal success from 37.9 to 80.7 and surface-constraint fulfillment from 35.0 to 92.6 over the strongest baseline. On a stratified 240-case editing test split, MUSE achieves 49.6 All-Goal success, 99.9 preservation rate, and only 0.6 unintended change rate. Beyond automated metrics, human evaluations on compared local-editing baselines support stronger alignment with user intent, and downstream navigation-proxy tests indicate stronger spatial stability. Combined with ablations validating our memory designs, these results establish MUSE as an effective framework for controllable 3D scene authoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21570v2">From Digital to Physical: Digital Agents as Autonomous Coaches for Physical Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 53 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08881v2">Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 13 pages, 9 figures,
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have demonstrated strong generalization in robotic manipulation, yet existing evaluations are primarily conducted in simulation or on expensive robotic platforms, leaving their robustness on affordable real-world robots largely unexplored. We present a standardized real-world benchmark for evaluating representative VLA and imitation learning policies on the low-cost SO-101 robotic platform. The benchmark comprises four representative manipulation tasks together with unified evaluation protocols, enabling systematic comparison under embodiment uncertainty. Using real-world teleoperated demonstrations, we fine-tune and evaluate $π_{0.5}$, SmolVLA, Wall-X, and ACT directly on the physical platform. Beyond conventional task success rates, the benchmark incorporates a structured failure taxonomy, semantic- and execution-level failure decomposition, and recovery-aware evaluation metrics to characterize policy robustness. Experimental results show that stronger pretrained VLA policies generally outperform the imitation learning baseline, although performance remains highly task-dependent under low-cost robotic deployment conditions. Execution instability emerges as the dominant failure source, while recovery capability varies substantially across architectures. These results highlight the importance of failure and recovery analysis beyond binary task success and establish SO-101 as a practical benchmark for evaluating embodied AI systems under realistic low-cost robotic deployment conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12942v2">Offline Diffusion Policy for Multi-User Delay-Constrained Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Effective multi-user delay-constrained scheduling is crucial in various real-world applications, including embodied AI, instant messaging, live streaming, and data center management, where efficient resource allocation is required among users with diverse delay sensitivities. In these scenarios, schedulers must make real-time decisions to satisfy both delay and resource constraints without prior knowledge of system dynamics, which are often time-varying and challenging to estimate. {Current learning-based methods typically require online interactions with actual systems during the training stage. Therefore, these approaches are often difficult or impractical, as they can significantly degrade system performance and incur substantial service costs.} To address these challenges, we propose a novel offline reinforcement learning-based algorithm, named \underline{S}cheduling By \underline{O}ffline Learning with \underline{C}ritic Guidance and \underline{D}iffusion Model (SOCD), to learn efficient scheduling policies purely from pre-collected \emph{offline data}. SOCD innovatively employs a diffusion policy, complemented by a sampling-free critic network for policy guidance. By integrating the Lagrangian multiplier optimization into the offline reinforcement learning, SOCD efficiently trains high-quality constraint-aware policies exclusively from available datasets, eliminating the need for online interactions with the system. Experimental results demonstrate that SOCD is resilient to various system dynamics, including partially observable and large-scale environments, and delivers superior performance compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10803v1">Beyond APIs: Probing the Limits of MLLMs in Physical Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) excel at utilizing digital APIs and increasingly serve as the "brain" of embodied AI, instructing robots to interact with the physical world. In such embodied settings, a central capability is the use of physical tools, which underpins MLLMs' ability to assist humans in real-world tasks. Despite the importance, MLLMs' proficiency in physical tool use remains largely unexplored. To address this gap, we introduce PhysTool-Bench, the first physical tool-use benchmark designed to evaluate MLLMs' ability to comprehend real-world scenarios, identify physical tools, and plan their use. PhysTool-Bench comprises 2,510 queries over 2,678 real-world physical tools spanning diverse domains, including manufacturing, electrical work, agriculture, and healthcare. Concretely, models are evaluated along two primary dimensions: 1) recognizing all physical tools present in the scene, and 2) planning the tool selection and use sequence based on the instruction and visual context. Across 13 leading MLLMs, even the strongest model (Gemini-3.1-Pro) identifies only 58.7% of tools in a scene and completes merely 21.0% of queries end-to-end. Our analysis reveals a two-level deficit: MLLMs struggle to perceive tools in realistic scenes, and the much larger drop at the planning stage further indicates a lack of functional commonsense for mapping perceived tools onto task semantics, pinpointing a critical bottleneck for the development of practical embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.01458v2">A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Navigation and manipulation are core capabilities in Embodied AI, but training agents to perform them directly in the real world is costly, time-consuming, and unsafe. Therefore, sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists. This survey examines how physics simulators address this gap by analyzing properties that have received limited attention in prior surveys. We also analyze their features for navigation and manipulation tasks, as well as their hardware requirements. Additionally, we offer a resource with benchmark datasets, metrics, simulation platforms, and methods to help researchers select suitable tools while accounting for hardware constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.06234v2">RobotEQ: Transitioning from Passive Intelligence to Active Intelligence in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Embodied AI is a prominent research topic in both academia and industry. Current research centers on completing tasks based on explicit user instructions. However, for robots to integrate into human society, they must understand which actions are permissible and which are prohibited, even without explicit commands. We refer to the user-guided AI as passive intelligence and the unguided AI as active intelligence. This paper introduces RobotEQ, the first benchmark for active intelligence, aiming to assess whether existing models can comprehend and adhere to social norms in embodied scenarios. First, we construct RobotEQ-Data, a dataset consisting of 1,894 egocentric images, spanning 10 representative embodied categories and 56 subcategories. Through extensive manual annotation, we provide 4,944 action judgment questions and 1,157 spatial grounding questions, specifying appropriate robot actions across diverse scenarios. Furthermore, we establish RobotEQ-Bench to evaluate the performance of state-of-the-art models on this task. Experimental results demonstrate that current models still fall short in achieving reliable active intelligence, particularly in spatial grounding. Meanwhile, leveraging RAG techniques to incorporate external social norm knowledge bases can generally enhance performance. This work can facilitate the transition of robotics from user-guided passive manipulation to active social compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20242v5">BadRobot: Jailbreaking Embodied LLM Agents in the Physical World</a></div>
    <div class="paper-meta">
      📅 2026-06-09
      | 💬 Accepted to ICLR 2025. Please cite the conference version. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Our code is available at https://github.com/Rookie143/BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10371v1">Test-time Adversarial Takeover: A Real-time Hijacking Interface against Robotic Diffusion Policies</a></div>
    <div class="paper-meta">
      📅 2026-06-09
    </div>
    <details class="paper-abstract">
      Diffusion-based action generation has become a foundational component of embodied AI, but its reliance on visual conditioning leaves deployed visuomotor policies vulnerable to adversarial manipulation. Most prior attacks focus on disruption: they perturb the observation stream to reduce task success or induce erratic behavior. We study a stronger threat, Test-time Adversarial Takeover (TAKO), in which an attacker obtains a real-time steering interface over a frozen robot policy and turns it into a remotely piloted instrument. TAKO learns a small vocabulary of reusable universal patches through differentiable diffusion inference; at test time, the attacker switches among these patches in the camera stream to compose attacker-chosen trajectories. This works because the perturbation acts on the visual conditioning pathway, where the induced bias can persist through iterative generative inference. We further show that the natural targeted baseline, target-policy matching, fails because the victim policy cannot reliably supervise itself on out-of-distribution target shifts. Across four tasks (2D manipulation, simulated aerial delivery, simulated ground navigation, and physical-world ground navigation), two visual encoders (ResNet-18 and EfficientNet-B0 + Transformer), and three generative inference families (DDPM, DDIM, and flow matching), human operators achieve 100\% takeover success on attacker-defined objectives in every evaluated setting. The project page is available at https://tako-attack.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09967v1">ABot-Earth 0.5: Generative 3D Earth Model</a></div>
    <div class="paper-meta">
      📅 2026-06-08
      | 💬 From Amap-cvlab, Alibaba. Official page: https://abot-earth.amap.com/
    </div>
    <details class="paper-abstract">
      We present ABot-Earth 0.5, a generative 3D framework designed to synthesize vast, seamless 3D environments from ubiquitous, geospatially referenced satellite imagery. To achieve this, we propose a novel generative model formulated directly with the 3D Gaussian Splatting (3DGS) representation. The model is trained on a diverse corpus of existing real-world urban reconstructions, learning to generate realistic geometry and textures. At inference, it synthesizes novel 3D scenes conditioned solely on satellite imagery at a scalable rate of under 10 minutes per square kilometer, while demonstrating exceptional realism. The framework is designed for accessibility, with integrated hierarchical level-of-detail (LOD) structures that permit real-time, interactive visualization on web-based map engines. This high-fidelity simulation sandbox effectively mitigates the sim-to-real domain gap, enabling critical downstream Embodied AI applications like closed-loop UAV navigation. By providing an ultra-low-cost and high-efficiency solution, ABot-Earth 0.5 significantly lowers the technical and financial barriers to large-scale 3D reconstruction and empowers the future of global digital earth visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04266v2">State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space</a></div>
    <div class="paper-meta">
      📅 2026-06-08
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08881v1">Benchmarking Vision-Language-Action Models on SO-101: Failure and Recovery Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-07
      | 💬 13 pages, 9 figures,
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have demonstrated strong generalization in robotic manipulation, yet existing evaluations are primarily conducted in simulation or on expensive robotic platforms, leaving their robustness on affordable real-world robots largely unexplored. We present a standardized real-world benchmark for evaluating representative VLA and imitation learning policies on the low-cost SO-101 robotic platform. The benchmark comprises four representative manipulation tasks together with unified evaluation protocols, enabling systematic comparison under embodiment uncertainty. Using real-world teleoperated demonstrations, we fine-tune and evaluate $π_{0.5}$, SmolVLA, Wall-X, and ACT directly on the physical platform. Beyond conventional task success rates, the benchmark incorporates a structured failure taxonomy, semantic- and execution-level failure decomposition, and recovery-aware evaluation metrics to characterize policy robustness. Experimental results show that stronger pretrained VLA policies generally outperform the imitation learning baseline, although performance remains highly task-dependent under low-cost robotic deployment conditions. Execution instability emerges as the dominant failure source, while recovery capability varies substantially across architectures. These results highlight the importance of failure and recovery analysis beyond binary task success and establish SO-101 as a practical benchmark for evaluating embodied AI systems under realistic low-cost robotic deployment conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.31158v2">Light Interaction: Training-Free Inference Acceleration for Interactive Video World Models</a></div>
    <div class="paper-meta">
      📅 2026-06-06
      | 💬 13 pages, 6 figures, 3 tables. Project page: https://2843721358l-del.github.io/Light-Interaction-Project/
    </div>
    <details class="paper-abstract">
      Interactive video world models generate video chunk by chunk in response to user-controlled camera movements, enabling applications such as real-time game simulation, virtual scene navigation, and embodied AI training. However, scaling to long interactive trajectories is prohibitively expensive due to growing context memory, quadratic attention complexity, and repeated denoising steps. We present Light Interaction, a training-free inference acceleration framework for interactive video world models. Our key insight is that interaction naturally enables trajectory-dependent adaptive computation: retrieved spatial memory can be discarded during novel exploration, temporal context can be adjusted according to local latent dynamics, and early-step model outputs can be reused when the camera revisits familiar regions. Based on this insight, Light Interaction combines adaptive context management, denoising cache acceleration, and hardware-software co-designed 3D block sparse attention with fused Triton kernels. Evaluated on HY-WorldPlay and Matrix-Game-3.0, Light Interaction achieves up to 2.59x speedup without model retraining while maintaining competitive visual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10578v3">Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-06-05
    </div>
    <details class="paper-abstract">
      The growing demand for Embodied AI and VR applications has highlighted the need for synthesizing high-quality 3D indoor scenes from sparse inputs. However, existing approaches struggle to infer massive amounts of missing geometry in large unseen areas while maintaining global consistency, often producing locally plausible but globally inconsistent reconstructions. We present Rein3D, a framework that reconstructs full 360-degree indoor environments by coupling explicit 3D Gaussian Splatting (3DGS) with temporally coherent priors from video diffusion models. Our approach follows a "restore-and-refine" paradigm: we employ a radial exploration strategy to render imperfect panoramic videos along trajectories starting from the origin, effectively uncovering occluded regions from a coarse 3DGS initialization. These sequences are restored by a panoramic video-to-video diffusion model and further enhanced via video super-resolution to synthesize high-fidelity geometry and textures. Finally, these refined videos serve as pseudo-ground truths to update the global 3D Gaussian field. To support this task, we construct PanoV2V-15K, a dataset of over 15K paired clean and degraded panoramic videos for diffusion-based scene restoration. Experiments demonstrate that Rein3D produces photorealistic and globally consistent 3D scenes and significantly improves long-range camera exploration compared with existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06895v1">Blockchain Infrastructure for Intelligent Cyber--Physical--Social Systems:Post-Quantum Security, Interoperability, and Trustworthy Data Economies in the Era of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-06-05
    </div>
    <details class="paper-abstract">
      The deployment of embodied artificial intelligence via world-model-based robotics presents a transformative opportunity for blockchain infrastructure, establishing urgent demand for trustworthy data provenance, cross-organizational governance, and incentive-compatible sharing across decentralized ecosystems. Simultaneously, quantum computing advances recognized by the 2025 Nobel Prize in Physics and the Turing Award threaten the cryptographic primitives securing these data economies, creating an interdependent imperative: long-lived verification for embodied AI depends on crypto-agile architectures capable of withstanding quantum adversaries. This tutorial examines blockchain as the coordination layer bridging this dual transition, from financial substrate to foundational Cyber-Physical-Social Systems infrastructure that simultaneously secures against quantum cryptanalysis and enables scalable, trustworthy data economies. The session opens with an immersive AWS Braket demonstration engaging participants with superconducting, trapped-ion, and neutral-atom hardware to assess cryptographic threat timelines and witness ECDSA-to-post-quantum signature transitions. Five integrated modules progress from embodied AI and world-model requirements through quantum hardware reality and evidence-based security migration, to scalable cross-shard architectures via BrokerChain protocols, trustworthy data economies implementing Croissant metadata standards and robotic learning provenance, and industry ecosystem integration for multi-modal cloud deployment. By bridging quantum hardware realities with embodied AI data requirements, this tutorial charts blockchain as unified infrastructure for next-generation decentralized intelligent environments, providing open-source frameworks and roadmaps for architecting quantum-resistant, interoperable, and data-trustworthy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.06390v1">HomeWorld: A Unified Floorplan-to-Furnished Framework for Generating Controllable, Densely Interactive Whole-Home Scenes</a></div>
    <div class="paper-meta">
      📅 2026-06-04
    </div>
    <details class="paper-abstract">
      Indoor scene generation is crucial for robot simulation and modern interior design. However, complex layouts together with scarce 3D scene data make learning-based generation challenging. Existing methods often rely on hand-crafted rules or focus on isolated sub-tasks (e.g., floorplan synthesis or single-room furnishing), producing whole-home scenes that lack global coherence, realism, and simulation readiness. To mitigate these limitations, we propose a unified hierarchical framework that decomposes indoor scene synthesis into controllable stages. First, we curate a large-scale dataset of 300K real residential floorplans to train a large language model for whole-home floorplan generation. With detailed descriptions and a K-D tree-based representation, our method enables fine-grained, controllable whole-home floorplan generation. Building upon the generated whole-home floorplan, we leverage image generation models to draft furniture layouts from multi-level roaming viewpoints, and then generate the layouts of small manipulable objects on different supporting surfaces (e.g., cabinets, desks, and dining tables) for embodied AI simulation. During furniture and object layout generation, a VLM-based refiner iteratively corrects furniture and object placement, and a 3D generative model enables flexible replacement of individual assets. We further attach basic physical attributes and simple surface texture and lighting setups to complete the pipeline for embodied AI use. Experiments and user studies demonstrate that our pipeline produces indoor spaces with greater layout diversity and stronger 3D design appeal, outperforming prior methods on both quantitative and qualitative metrics. Finally, alongside our generation pipeline, we will release the floorplan dataset and 5K fully furnished scenes to the community. Project Page: https://kairos-homeworld.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.05660v1">Safe Embodied AI for Long-horizon Tasks: A Cross-layer Analysis of Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-06-04
      | 💬 63 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Embodied AI systems are increasingly expected to reason and act over extended horizons in physical environments. This growing capability brings safety to the foreground, because failures in the physical world can harm people, damage objects, and disrupt workplaces. Although safe embodied AI has attracted substantial attention, the literature remains fragmented across planning, policy design, and runtime execution. Long-horizon robotic manipulation is a particularly revealing anchor domain for this problem because semantic misgrounding, subtask-level error propagation, execution drift, and contact-rich physical risk can accumulate within the same closed-loop system. This survey therefore provides a structured review of safety in long-horizon robotic manipulation from an embodied AI perspective. We organize the literature by intervention locus, covering planning-time, policy-time, and execution-time safety, and we analyze the strength of the evidence that each line of work provides, distinguishing formal guarantees, statistical support, and empirical safety heuristics. This framework clarifies the distinct roles of backbone capability papers, direct safety mechanisms, and benchmark or evaluation studies, while exposing where current safety claims are well supported and where they remain indirect. We identify persistent gaps, including limited evidence for policy-time safety, weak formal support for contact-rich long-horizon manipulation, immature uncertainty-triggered intervention, and a shortage of manipulation-specific safety benchmarks. We conclude by outlining research directions for cross-layer assurance, evaluation design, and safer deployment of long-horizon robotic agents in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04773v1">NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-06-03
      | 💬 23 pages, 8 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Reliable evaluation of human motion understanding is fundamental to advancing embodied AI, robotics, and animation. However, existing benchmarks suffer from coarse semantic granularity, undifferentiated difficulty, limited annotation quality, and pervasive answer ambiguity, leaving them unable to diagnose where current models fail. To bridge this gap, we introduce NextMotionQA, a comprehensive benchmark that leverages vision-language models (VLMs) for semi-automated, expert-verified dataset. NextMotionQA features three complementary tasks: multiple-choice question answering, video captioning, and fine-grained error correction. Each task is systematically structured across three core semantic axes and stratified into three task complexity levels. Our extensive evaluation of twelve representative VLMs uncovers critical capability gaps and weakness that remain invisible under conventional, single-task evaluations. In a complementary direction, recent work has begun using VLMs as judges for text-to-motion evaluation; we ask whether they show the same degradation under harder tasks. We find that VLMs align strongly with expert ratings on coarse criteria (Cohen's κ=0.70) but break down on fine-grained, part-level judgment (κ=0.10), validating the paradigm in its strong regime while clarifying its limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03609v2">A 3D Isovist World Model -- Revealing a City's Unseen Geometry and Its Emergent Cross-City Signature</a></div>
    <div class="paper-meta">
      📅 2026-06-03
    </div>
    <details class="paper-abstract">
      Embodied agents that navigate cities rely on world models that predict how their surroundings will change as they move. But for navigation, what matters is not what the buildings look like; it is where the agent can go. Most world models nonetheless predict appearance, learning how a scene looks rather than the space an agent can move through. Those that do target geometry, such as bird's-eye-view occupancy grids, flatten the three-dimensional environment onto a ground plane, discarding the above-ground and multi-level structure that shapes real navigation. What is missing is a predictive target that captures the navigable geometry an agent actually traverses, without photometric entanglement and without collapsing the third dimension. Our key idea is to model the open volume between buildings, the negative space, encoded as a 3D isovist: a spherical visibility-depth map recording the distance to the nearest surface in every direction. We introduce an embodied world model that predicts the next isovist from a short history of past isovists and a movement action. The prediction is formulated as a depth residual so the decoder inherits sharp building edges, trained with self-rollout scheduled sampling to keep corrupted context on the geometry manifold, and equipped with a persistent latent bird's-eye-view spatial map for cross-path consistency. Our central finding is emergent and unexpected: a single city-blind model trained on Manhattan and Paris develops a cross-city spatial signature, with city identity linearly decodable from its temporal latents far above single-frame baselines, so the signature lives in the learned dynamics rather than in appearance. The representation is lightweight, interpretable, and reproducible, offering a geometric substrate for spatial reasoning in embodied AI, robotics, and urban analysis, released with an open dataset and pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03686v1">The DeepSpeak-Agentic Dataset</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      We present DeepSpeak-Agentic, a dataset of videos comprising over 37 hours of semi-structured conversations between a human and an embodied AI agent. We use this dataset to evaluate the automatic forensic identification (audio, video, or text) of AI agents, study the nature of human-agent interactions, and provide a benchmark for future advances in the large-language models and AI-generated voices and faces that power embodied AI agents. We also contribute a scalable data-capture system that creates agents, automatically pairs them with human crowd workers, records audiovisual conversations across specified scenarios, and identifies and separates the human and agent in the combined stream.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03593v1">Making Embodied AI Reliable: A Community Agenda from Testing to Formal Verification</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Embodied AI systems are increasingly deployed in open-world environments, yet ensuring their reliability remains a fundamental challenge. Drawing on discussions from the AAAI'26 Bridge Program on "Making Embodied AI Reliable with Testing and Formal Verification", this article argues that reliability in embodied AI is inherently a lifecycle assurance problem arising from uncertainty, human interaction, and emergent behaviors across tightly coupled system components. We identify three complementary directions toward reliable embodied AI: (1) trustworthy scenario-based testing supported by validated specifications and meaningful coverage metrics, (2) compositional verification enabled by structured symbolic representations of system behavior and environmental context, and (3) runtime assurance mechanisms capable of adapting to uncertainty and distribution shifts during deployment. Rather than treating these approaches independently, we advocate integrated assurance workflows that connect testing, verification, and runtime adaptation through shared neuro-symbolic representations and continuous feedback across the system lifecycle. Such integration provides a foundation for building trustworthy embodied AI systems that can operate safely and reliably in complex real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.03392v1">OpenEAI-Platform: An Open-source Embodied Artificial Intelligence Hardware-Software Unified Platform</a></div>
    <div class="paper-meta">
      📅 2026-06-02
    </div>
    <details class="paper-abstract">
      Embodied AI in the real world requires both accurate hardware and robust vision-language-action (VLA) policies. We present OpenEAI-Platform, a fully open-source platform that integrates a low-cost 6+1 degree-of-freedom (dof) robotic arm (OpenEAI-Arm) and a reproducible VLA model (OpenEAI-VLA). OpenEAI-Arm provides open-source mechanical designs for low manufacturing cost and compliant control methods for higher accuracy. OpenEAI-VLA builds on Qwen3-VL-4B and uses a Diffusion Transformer action head, and is trained in two stages with only open-source robot and multimodal datasets. Across four real-world manipulation tasks, OpenEAI-Arm outperforms two commercial 6+1-dof arms under the same policy, and OpenEAI-VLA achieves success rates comparable to the large-scale pretrained pi0 baseline with only limited pretraining data. We will release the full hardware designs, drivers, models, and training/data pipelines to support reproducible research and scalable data collection. Our codes, layouts, and models will be released after the paper is accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.04046v1">Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-02
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      In embodied vision-language decision making tasks such as robotic manipulation and navigation, Vision-Language and Vision-Language-Action Models (VLMs & VLAs) are powerful tools with different benefits: VLMs are better at long-term planning, while VLAs are better at reactive control. However, their performance is limited by the same perceptual bottleneck: visual hallucinations arise due to the models' inability to distinguish task-relevant objects from distractors. In principle, accurate identification and focus on critical objects while filtering out irrelevant ones is the key to break this limitation. A straightforward solution is one-step focus: directly attending to essential objects. However, this approach proves ineffective because effective focus inherently requires deep scene understanding. To this end, we propose SceneDiver, a coarse-to-fine focus plan generation method for VLMs leveraging their long-term planning abilities, that first constructs a holistic scene graph to establish initial comprehension, then progressively decomposes the task into simpler sub-problems through an iterative cycle of recognition, understanding, and analysis. To enable reactive control, we also design a lightweight adapter for distilling the deliberate focus ability into VLAs. Evaluations on standard embodied AI benchmarks confirm that our method substantially reduces visual hallucinations for both VLMs and VLAs, while preserving computational efficiency in tasks requiring fast execution. Our code and data are released at: https://future-item.github.io/SceneDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02956v1">The Road Ahead in Autonomous Driving: The KITScenes Multimodal Dataset</a></div>
    <div class="paper-meta">
      📅 2026-06-01
      | 💬 28 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Existing autonomous driving datasets have enabled major progress, but fall short in sensor fidelity, map completeness, or geographic diversity. We present KITScenes Multimodal, a European dataset built around high-fidelity sensors and maps. Our fully synchronized sensor suite combines high-resolution global-shutter cameras, long-range lidar beyond 400m, 4D imaging radar, and redundant GNSS/INS localization. Our HD maps are, to our knowledge, the most complete of any sensor dataset, validated through autonomous driving trials on open-source software. For the first time in a public dataset, all driving-relevant traffic elements, such as traffic lights, are mapped in 3D to a reprojection-accurate level with full topological connectivity. Recorded in cities with irregular street layouts and mixed traffic modes, our dataset complements existing datasets by broadening the available geographic diversity. We also introduce four benchmarks, each advancing spatial learning for embodied AI: online HD map construction, long-range depth estimation, novel view synthesis, and end-to-end driving. Project page: https://kitscenes.com/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02753v1">MetaWorld: Scaling Multi-Agent Video World Model from Single-view Video Data</a></div>
    <div class="paper-meta">
      📅 2026-06-01
    </div>
    <details class="paper-abstract">
      Video world models are a foundational generative technology for embodied AI and the Metaverse, yet existing approaches are inherently limited to a single agent observing from a single perspective. Extending these models to multi-agent settings introduces two critical challenges: data scarcity (coordinated multi-view recordings are prohibitively expensive to collect for general open-domain scenarios) and world state alignment (independently generated video streams cannot ensure that shared physical environments and events evolve consistently across views). To address these challenges, we propose MetaWorld, a novel framework that scales multi-agent video world models to open-domain environments directly from single-view videos. First, we introduce Monocular World-State Unrolling (MWSU) to explicitly decompose monocular footage into the camera operator's ego-motion and the visible subject's spatial trajectory. This camera-trajectory decomposition naturally extracts synchronized multi-agent motion data within a shared 3D space, completely bypassing the need for multi-camera setups. Second, for precise visual control, we develop the Subject-Aware World Generator to enable appearance-driven simulation conditioned on per-agent identity images. Finally, to ensure both views are grounded in the identical physical reality, we propose World-State Alignment, a per-frame inter-branch cross-attention mechanism inserted at every transformer layer of the video DiT. By jointly synchronizing the denoising process, WSA enforces both static geometric consistency and dynamic motion consistency, encouraging that the shared 3D environment and physical events remain well-aligned across both egocentric views. Extensive experiments demonstrate that MetaWorld achieves superior cross-view consistency and identity fidelity, establishing a highly scalable, physics-driven paradigm for multi-agent video world modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02742v1">Consistent Yet Wrong: Evidence Insensitivity in Spatial Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-06-01
    </div>
    <details class="paper-abstract">
      Spatial reasoning is fundamental to robotics, autonomy, and embodied AI, yet modern vision-language models (VLMs) remain unreliable on metric distance queries. A common assumption is that consistent predictions across viewpoints reflect geometric grounding. We test this assumption and find the opposite: leading VLMs often produce view-invariant and consistent answers even when those answers are incorrect, indicating weak coupling between predictions and viewpoint-specific visual evidence. We introduce \textbf{ViewDiag}, a controlled multi-view evaluation protocol built from Hypersim, ScanNet, and KITTI360, comprising 176 object-pair tracks across 80 scenes with 2--10 views per track. The protocol evaluates models along three axes: metric accuracy, distributional concentration, and a latent feature probe for internal collapse that distinguishes decision collapse from representation collapse. Across diverse models, we observe a consistent pattern of high prediction stability paired with substantial error, clustering in a regime characterized by strong consistency but low accuracy. \noindent These results challenge the common use of cross-view consistency as a proxy for geometric understanding. Instead, we show that stable predictions may reflect prior-driven collapse rather than evidence-sensitive reasoning. ViewDiag provides a controlled benchmark and diagnostic framework for evaluating spatial VLMs beyond accuracy alone. The code and data can be found \href{https://github.com/SDivakarBhat/Consistent_Yet_Wrong.git}{here}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10958v2">WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World</a></div>
    <div class="paper-meta">
      📅 2026-06-01
      | 💬 CVPR 2026 Oral Presentation; 80 pages, 37 figures, 29 tables; Project Page at https://worldbench.github.io/worldlens GitHub at https://github.com/worldbench/WorldLens
    </div>
    <details class="paper-abstract">
      Generative world models are reshaping embodied AI, enabling agents to synthesize realistic 4D driving environments that look convincing but often fail physically or behaviorally. Despite rapid progress, the field still lacks a unified way to assess whether generated worlds preserve geometry, obey physics, or support reliable control. We introduce WorldLens, a full-spectrum benchmark evaluating how well a model builds, understands, and behaves within its generated world. It spans five aspects -- Generation, Reconstruction, Action-Following, Downstream Task, and Human Preference -- jointly covering visual realism, geometric consistency, physical plausibility, and functional reliability. Across these dimensions, no existing world model excels universally: those with strong textures often violate physics, while geometry-stable ones lack behavioral fidelity. To align objective metrics with human judgment, we further construct WorldLens-26K, a large-scale dataset of human-annotated videos with numerical scores and textual rationales, and develop WorldLens-Agent, an evaluation model distilled from these annotations to enable scalable, explainable scoring. Together, the benchmark, dataset, and agent form a unified ecosystem for measuring world fidelity -- standardizing how future models are judged not only by how real they look, but by how real they behave.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.02510v1">Not All Points Are Equal: Uncertainty-Aware 4D LiDAR Scene Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-06-01
      | 💬 CVPR 2026 E2E3D Workshop; GitHub at https://github.com/worldbench/U4D
    </div>
    <details class="paper-abstract">
      Constructing faithful 4D worlds from LiDAR-acquired sequences is crucial for embodied AI, yet current generative frameworks apply uniform modeling capacity across all spatial regions. This ignores that perceptual difficulty varies dramatically within a single scan: distant surfaces, occluded boundaries, and small-scale objects carry far higher uncertainty than well-observed structures. We present U4D, a new framework that explicitly leverages spatial uncertainty to guide LiDAR scene generation in a "hard-to-easy" schedule. U4D derives per-point uncertainty maps via Shannon Entropy from a pretrained segmentor, then applies an unconditional diffusion stage to synthesize high-entropy areas with precise geometry, followed by a conditional completion stage that fills in the remaining regions using these structures as priors. A MoST (Mixture of Spatio-Temporal) block further maintains cross-frame coherence by dynamically balancing spatial detail and temporal continuity. Extensive experiments on nuScenes and SemanticKITTI demonstrate state-of-the-art scene fidelity, temporal consistency, and downstream performance.
    </details>
</div>
