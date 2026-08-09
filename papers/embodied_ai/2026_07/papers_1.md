# embodied ai - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00289v1">RF-HOI: Recognize Human-Object Interaction with Radio Frequency Signals</a></div>
    <div class="paper-meta">
      📅 2026-07-31
      | 💬 Accepted by ACM IMWUT
    </div>
    <details class="paper-abstract">
      Recognizing Human-Object Interactions (HOI) is essential for intelligent systems, underpinning applications in virtual and augmented reality, embodied AI, and assistive robotics. However, vision-based HOI methods face challenges in privacy concerns and poor light conditions. In this work, we introduce RF-HOI, the first framework that only uses radio frequency (RF) signals for HOI recognition. A key challenge of RF-HOI is that single-modality RF sensing is insufficient to recognize both actions and the objects being interacted with. RF-HOI addresses this through a novel modality fusion that combines mmWave radar and RFID, enabling simultaneous action recognition and target identification. Another challenge is limited training data across diverse setups, which impairs the generalizability of the recognition model. To overcome this, we develop a simulator that synthesizes multimodal RF data for diverse HOIs at scale, allowing us to fine-tune with only a small amount of real-world data. Experiment results show that RF-HOI outperforms all baselines, approaching vision model performance, and that our diverse synthetic training data can significantly boost our system's performance on real-world scenarios. These results highlight the potential of multimodal RF sensing for robust and privacy-preserving HOI recognition as well as the effectiveness of our RF data synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.29596v1">FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling</a></div>
    <div class="paper-meta">
      📅 2026-07-31
    </div>
    <details class="paper-abstract">
      Vision-language-action models (VLAs), which leverage the cognition of multimodal information to infer physical-world actions, provide a generalized solution for embodied AI applications. Conventional VLAs usually concentrate on current digital cognition. While some efforts are made to enhance VLAs' reasoning capabilities by capturing temporal information, encoding the long-context history causes an efficiency-decreasing issue. To reconcile the conflict between capturing temporal information and maintaining inference efficiency in VLAs, this paper introduces FibVLA, an efficient framework featuring temporal perception of long-context history. Specifically, we leverage logarithmic hindsight sampling to both proprioceptive states and visual frames to capture long-term temporal dependencies with minimal redundancy. For the action expert, we introduce the flow matching to produce action distributions, and the Fibonacci recurrent inference strategy to generate long-range planning steps based on real-time closed-loop feedback. Experiments demonstrate that FibVLA significantly improves action smoothness and success rates without retraining large-scale visual encoders. Efficiency analysis demonstrates superior real-time responsiveness compared to video-based baselines in real-world evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28625v1">ACE-Data-0: Human-Centric Ambient Capture as Embodied Data Engine</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 Project Page: https://ace-data-engine.github.io/ACE-Data-0/
    </div>
    <details class="paper-abstract">
      Embodied intelligence faces a fundamental data bottleneck. Models must capture how first-person perception, whole-body motion, dexterous manipulation, object state, sound, and touch evolve together as humans pursue goals over time. Existing datasets fragment this experience across viewpoints, modalities, or spatial scales, leaving the full perception-action loop only partially observed. We introduce the Ambient Capture Engine (ACE), a human-centric data engine that transforms real home environments into spatially calibrated, temporally synchronized recording studios. ACE operates at two complementary scales: a table-scale configuration resolves hand-object manipulation, while a room-scale configuration captures whole-body motion, locomotion, and interactions across a furnished home. ACE records egocentric and multi-view exocentric video, full-body and articulated hand motion, object geometry and 6-DoF trajectories, audio, and tactile signals as a unified multisensory stream. Using ACE, we build ACE-Data-0, comprising 150 hours and 17M video frames across 200 task categories, performed by 50 participants in 2 environments, for a total of 75,000 interaction episodes. The dataset spans atomic manipulation, long-horizon chains of household activities, and human-scene interaction, while preserving natural behavioral variation through goal-level rather than step-by-step instructions. We further introduce a hierarchical benchmark that progresses from signals to scene components and then to interactions. Evaluations of state-of-the-art methods expose substantial gaps under contact, occlusion, egomotion, and long temporal horizons. ACE-Data-0 provides synchronized human demonstrations with aligned perceptual, kinematic, and contact supervision, offering a scalable foundation for imitation learning, world models, vision-language-action systems, and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28442v1">ViewMind3D: Modular View-Aware Inference for Training-Free 3D-QA</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled new possibilities for 3D question answering (3D-QA), a key capability for embodied AI and robotic perception. However, most existing methods rely on 3D-specific training or fine-tuning with costly annotations, limiting their scalability and real-world applicability. We present \textbf{ViewMind3D}, a fully training-free and modular framework for 3D spatial reasoning over multi-view observations of a scene without requiring complete 3D reconstruction. The framework decomposes the 3D-QA task into four interpretable components: (1) question-driven multi-view selection, (2) guided visual grounding with language-conditioned object cues, (3) spatial context encoding via a bird's-eye-view (BEV) viewpoint indicator, and (4) structured answer generation through role-based reasoning. This design enables structured, robust, and interpretable reasoning without requiring model tuning. Experimental results on ScanQA and SQA3D show that ViewMind3D achieves competitive performance compared to prior training-free and fine-tuned 3D-LLMs. In particular, our method improves performance on spatially grounded question types, such as ``What'' questions in SQA3D, while maintaining strong overall accuracy (50.8\%) and achieving 73.4 CIDEr on ScanQA. These results demonstrate that effective 3D reasoning can be achieved through modular orchestration of general-purpose LLMs and VLMs for robotic perception in real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28256v1">When Robots Exchange Meaning: A Demo of Goal-Oriented Semantic Communications for Collaborative Robotics</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 3 pages, 3 figures. This paper has been accepted for presentation as a demo paper at IEEE CSCN 2026
    </div>
    <details class="paper-abstract">
      Collaborative robotics is a representative task-oriented 6G use-case, where communication quality should be reflected in mission execution, environment understanding, and closed-loop operation rather than packet delivery alone. This demo paper presents a robot-edge semantic communication (SemCom) testbed integrating robot-side visual compression, edge-side semantic mapping, and dashboard-based mission interaction. A mobile robot equipped with RGB-D sensing and LiDAR runs ROS 2, while a Jetson Orin edge node performs reconstruction, RTAB-Map mapping, semantic object handling, and browserbased visualization. As an initial proof of concept, RGB frames are encoded on the robot into VQ-VAE tokens using an ONNX Runtime encoder and reconstructed on the edge using a PyTorch decoder. A 320 X 240 image is represented by an 80 X 60 token grid with a packed payload of 5400 bytes, corresponding to a 42.67X reduction relative to model-input RGB bytes. The reconstructed visual stream is further associated with depth, pose, and 3D mapping information to generate a semantic map for downstream robotic applications. The demo exposes the full path from semantic visual transport to object-level map interaction, and provides a practical platform for future task-aware 6G networking studies at the intersection of SemCom, embodied AI, and physical AI-enabled robotics. A video of the demo is available at https://tinyurl.com/Tos09
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28243v1">EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 project page: https://egogenesis.github.io/
    </div>
    <details class="paper-abstract">
      Egocentric video offers rich manipulation experience for embodied AI, yet collecting diverse egocentric data across scenes, objects, motions, and embodiments remains costly. We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms. Online Anchored Projective Memory (OAPM) preserves a first-frame 3D scene anchor while periodically refreshing a recent state during autoregressive generation. Action-3D Rotary Position Embedding (A3D-RoPE) encodes end-effector motion with camera-aware 3D rotary coordinates, injecting action geometry into skeleton-to-video cross-attention for precise control. Together, these components improve visual fidelity, geometric stability, and action alignment in long egocentric rollouts. Moreover, augmenting 400 real trajectories with 400 \method-generated trajectories improves out-of-distribution real-robot success from 77\% to 84\% on single-arm tasks and from 53\% to 70\% on dual-arm tasks, demonstrating that the synthesized data substantially improve downstream WAM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28226v1">Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      World models give embodied AI a predictive core: they compress observations into states, simulate action-conditioned futures, and enable planning beyond reactive control. This predictive layer, however, opens a new security boundary-compromise can propagate from data, sensors, prompts, or feedback into physical action. Rather than treating world models as an isolated component, this survey traces threats across their entire lifecycle-from data construction and representation learning, through state grounding and imagination, to trajectory evaluation, execution, and long-term adaptation via memory and tools. We show that familiar attack families: poisoning, backdoors, adversarial examples, sensor spoofing, prompt injection, trajectory manipulation, and supply-chain attacks take on distinct meanings when they corrupt world states, learned dynamics, affordance estimates, or safety costs. We also highlight a duality: world models can serve as runtime safety shields, yet when compromised or over-trusted they generate predictive safety illusions. The survey offers a lifecycle taxonomy, maps existing attacks to world-model security properties, outlines evaluation protocols for safety failures, and structures defenses across provenance, robust grounding, uncertainty-aware prediction, trajectory gating, feedback auditing, and deployment assurance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.28083v1">Layered Architecture for Mobile Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is rapidly evolving from a centralized computing capability into a pervasive infrastructure that interacts directly with the physical world. While recent perspectives highlight the roles of energy, chips, infrastructure, models, and applications in enabling large-scale AI systems, these frameworks primarily assume a static, cloud-centric computing paradigm. However, emerging intelligent applications, including autonomous vehicles, drones, robots, and wearable systems, require AI to operate in highly dynamic and mobile environments. This shift introduces mobility as a fundamental constraint across the entire AI ecosystem, affecting energy supply, computation, and intelligence deployment. In this article, we introduce the concept of the Mobile AI Stack, a mobility-aware architectural framework that integrates five tightly coupled layers: mobile energy networks, energy-efficient AI chips, cloud-edge-mobile infrastructure, distributed AI models, and embodied AI applications. The proposed framework provides a systematic perspective for understanding how energy delivery, computing architectures, communication networks, and AI algorithms must co-evolve to support large-scale mobile intelligence. We further discuss key research challenges and future directions toward building scalable, reliable, and energy-efficient mobile AI systems. Mobile AI Stack offers a conceptual blueprint of the next-generation infrastructure which deeply integrates the networks of computation, energy, and communications for mobile intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27967v1">MARS-RA: Rank Aggregation for Credit Assignment via Multimodal Comparisons in Embodied Multi-Agent Cooperation</a></div>
    <div class="paper-meta">
      📅 2026-07-30
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      Credit assignment is a fundamental challenge in cooperative multi-agent reinforcement learning, particularly in embodied AI settings characterized by limited and delayed feedback as well as dynamically changing numbers of active agents. We propose MARS-RA, a framework that reformulates credit assignment as a rank aggregation problem using contribution-based pairwise comparisons among agents generated by large multimodal models. This shift from absolute to relative estimation ensures robustness against noise and dynamic agent participation, converting comparison results into contribution scores for potential-based reward shaping. We provide theoretical justification for the convergence and robustness of the proposed framework, and show that Shapley values can be used as an interpretive reference. Experimental results on challenging tasks of different types indicate that MARS-RA can guide agents toward effective cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04227v3">Continual Learning for VLMs: A Survey and Taxonomy Beyond Forgetting</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs), spanning predictive architectures to generative Multimodal Large Language Models (MLLMs), have revolutionized artificial intelligence through powerful cross-modal alignment and zero-shot generalization. However, enabling them to learn continually from non-stationary data remains a major challenge, as their cross-modal alignment and generalization capabilities are particularly vulnerable to catastrophic forgetting. Unlike traditional unimodal continual learning (CL), VLMs face unique challenges such as cross-modal feature drift, parameter interference due to shared architectures, and zero-shot capability erosion. Furthermore, generative MLLMs exhibit a unique "alignment tax," where catastrophic forgetting manifests not merely as factual amnesia, but as a systemic collapse of deep Chain-of-Thought (CoT) reasoning. This survey presents the first comprehensive diagnostic review bridging continual learning across predictive VLMs and generative MLLMs. We systematically deconstruct the aforementioned failure modes and propose a challenge-driven taxonomy comprising four core paradigms: (1) Multi-Modal Replay Strategies addressing explicit and implicit memory drift; (2) Cross-Modal Regularization enforcing topological and geometric alignment; (3) Parameter-Efficient Adaptation utilizing dynamic routing and subspace projections; and the emerging (4) Model Fusion and Decoupling paradigms. We critically analyze the evolution of evaluation protocols, highlighting the essential shift toward dual-track benchmarks (Domain vs. Ability CL). Finally, we chart a roadmap for future research, emphasizing compositional zero-shot learning, embodied AI with sensor fusion, and autonomous agentic ecosystems. All resources are available at: https://github.com/YuyangSunshine/Awesome-Continual-learning-of-Vision-Language-Models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03438v3">Multimodal Reinforcement Learning with Adaptive Verifier for AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-30
    </div>
    <details class="paper-abstract">
      Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27443v1">Leveraging Trajectory Graphs for Pre-Execution Error Diagnosis in Agentic LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      Large Language Model~(LLM)-based agents have demonstrated exceptional performance across a wide range of complex interactive tasks. However, they often struggle with long-horizon interactive tasks common in domains, such as embodied AI. The complexity and vast action spaces in these settings lead to compounding errors, where a single suboptimal action can derail an entire trajectory, causing the agent to exhaust its limited step budget on inefficient or unrecoverable paths. To overcome this without costly fine-tuning, we draw inspiration from software debugging, where execution logs are analyzed to preemptively catch errors. We propose \textit{Trajectory Graph Copilot}, a novel framework that acts as a ``copilot'' for LLM agents by diagnosing potential action errors before they are executed. At its core,\textit{Graph Debugger} models historical trajectories as a probabilistic graph and uses a Graph Neural Network to identify sequential action patterns that frequently lead to failure. Functioning as a proactive diagnostic sandbox, our method provides early warnings on potentially flawed actions, prompting the agent to self-correct. This pre-action error diagnosis prevents costly mistakes, significantly enhancing the agent's ability to complete long-horizon tasks successfully. The extensive experiments on four benchmarks with three LLM agents demonstrate a $14.69\%$ pass ratio improvement on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.27145v1">Explainable and Resource-Efficient Spatial Reasoning in Multimodal LLMs for Decision-Critical Applications</a></div>
    <div class="paper-meta">
      📅 2026-07-29
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As Multimodal Large Language Models (MLLMs) are increasingly deployed in decision-critical pipelines such as robotics, embodied AI, and safety monitoring, the opacity of their spatial judgments limits operator trust and auditability. MLLMs demonstrate strong reasoning but often struggle with fine-grained spatial understanding and object hallucination. Prior work, ByDeWay, introduced Layered-Depth-Based Prompting (LDP), a training-free framework that mitigates hallucinations by structuring prompts using monocular depth estimation. However, coarse depth layering falls short in resolving object-to-object spatial relationships within the same geometric plane, such as projective ("left of", "above") and topological ("inside", "touching") relations. We propose ByDeWay-V2, which integrates explicit spatial relational context alongside depth cues, expressed as human-readable predicates that serve as auditable evidence for downstream decision support. Using an open-vocabulary object detector (YOLO-World-L), our framework computes pairwise geometric relations between detected objects and injects them as structured spatial predicates into the MLLM prompt, bridging 3D scene depth and 2D spatial semantics without any training. We evaluate ByDeWay-V2 on the Visual Spatial Reasoning (VSR) and BLINK benchmarks across multiple MLLMs, with hallucination grounding assessed via POPE. On the BLINK spatial subset, ByDeWay-V2 achieves a 46 percent relative F1 improvement over LDP for Qwen2.5-VL, and recovers BLIP-Base's spatial reasoning on VSR from near-random performance to a competitive F1 of 0.53. Our lightest configuration operates under a strict 40-token context budget on CPU, showing the framework's suitability for resource-constrained, real-time decision-support settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26903v1">From Passive Video to Editable Experience: Physically Grounded Experience Synthesis for Embodied Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      The key bottleneck in embodied AI is not model architecture but data. Although billions of human manipulation videos exist online, robots cannot directly learn from them due to the embodiment gap between human morphology and robot hardware. We introduce Pegasus, a low-resource framework that bridges this gap by translating human demonstrations into robot-learnable data through structured knowledge transfer. Instead of relying on raw video prompts, Pegasus constructs a graph-based intermediate representation: a Task Graph extracted from human videos is transformed through Affordance and Constraint Graphs into a Robot Planning Graph for robot-conditioned video generation. A hierarchical affordance latent space models the relationship between object states, affordances, and tasks, enabling generalization beyond object identities. A closed-loop physics verifier further filters invalid generations using kinematic feasibility, collision constraints, and joint limits. We evaluate Pegasus across a range of egocentric manipulation benchmarks, including GTEA Gaze+ and EPIC-KITCHENS-100, and diverse robot embodiments, assessing Task Correctness, Executability, State Consistency, and Learnability. Results demonstrate reliable cross-embodiment translation and show that robot data generation can be reframed from a hardware collection problem into a scalable, low-resource knowledge transfer problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26646v1">Genie Sim PanoWorld: An Infinite Indoor 3D World Generation Pipeline via Panoramic Scene Modeling and Simulation</a></div>
    <div class="paper-meta">
      📅 2026-07-29
    </div>
    <details class="paper-abstract">
      We address the problem of reconstructing a high-fidelity, freely navigable 3D scene from a single $360^\circ$ panorama, without per-scene optimization or multi-view capture. Existing methods either lack metric trajectory control, which hinders reliable downstream 3D reconstruction, or struggle with large disocclusions under long-range camera motion while requiring high-end multi-GPU servers.We present Genie Sim PanoWorld, a two-stage feed-forward pipeline that bridges generation and reconstruction via an explicit, trajectory-controllable panoramic video. A NavMesh-planned $\mathrm{SE}(3)$ roaming trajectory is injected into a latent video diffusion model through dense geometry-warped conditioning; long--short trajectory mixed training and a self-consistency objective based on shortcut models together yield high-fidelity video in four CFG-free denoising steps. A feed-forward panoramic reconstructor then lifts the generated video into a high-fidelity 3D Gaussian scene that supports real-time, free-viewpoint roaming and can be directly used as a simulation-ready asset for embodied AI applications. Experiments show that Genie Sim PanoWorld outperforms geometry-conditioned baselines in both panoramic video generation and downstream 3D reconstruction, while generalizing zero-shot to unseen indoor scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.26121v1">Towards Trustworthy Embodied Intelligence: A Systems Framework and Graded Trustworthiness Levels</a></div>
    <div class="paper-meta">
      📅 2026-07-28
      | 💬 Website: https://xsparkai.com/sparklab/towards-trustworthy-eai
    </div>
    <details class="paper-abstract">
      Embodied intelligence integrates learned perception and decision making with real-time computation, control, and physical interaction. Because failures can cause immediate physical or operational harm, task completion alone does not establish trustworthiness. We define trustworthy embodied intelligence as the sustained capacity to execute specified tasks reliably under environmental and system variation while maintaining risk within acceptable bounds. We term this objective sustained safe success. Its supporting mechanisms are organized into four interdependent layers. The model layer generates task-competent action proposals with calibrated uncertainty and explicit safety preferences. The system layer realizes authorized actions dependably through integrated sensing, computation, control, hardware safeguards, fault containment, and fallback. The evidence layer substantiates bounded claims through evaluation, verification, validation, traceability, and structured assurance arguments. The deployment layer maintains claim validity through runtime monitoring, authority management, intervention, incident response, and controlled updates. Because assumptions and failures propagate across these layers, neither model capability, isolated safeguards, nor benchmark performance alone can establish end-to-end trustworthiness. Drawing on embodied AI, robotics, control, dependable computing, distributed systems, and autonomous driving, we further propose a non-normative hierarchy of trustworthiness levels. This hierarchy grades the strength of bounded deployment claims across task capability, safety, system assurance, operational governance, and supporting evidence, providing a basis for bounded deployment, comparative evaluation, research prioritization, and future standardization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.25106v1">IMPRINT: Image-Conditioned Query Enrichment for Long-Tail Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Project page: https://github.com/JelinR/IMPRINT
    </div>
    <details class="paper-abstract">
      Embodied AI increasingly relies on queryable semantic maps built from pre-trained vision-language models to enable zero-shot Object Goal Navigation (ObjectNav). However, existing approaches typically depend on text-only queries, which become less reliable as semantic specificity increases toward fine-grained object categories. We introduce IMPRINT, a zero-shot plug-and-play framework that enriches textual object queries with web-sourced images to improve grounding in queryable maps. Retrieved images are encoded using a vision-language model, matched against the semantic map to produce similarity maps, and aggregated to yield context-aware localization. Notably, this requires no training or modification of the underlying navigation policy. To explicitly evaluate long-tail behavior, we present HSSD-rare, a new ObjectNav benchmark built on Habitat Synthetic Scenes and featuring semantically specific subcategories. Across both OVON and HSSD-rare, image-conditioned queries consistently improve object grounding and yield end-to-end navigation gains. Further analysis reveals that translating localization gains to navigation performance depends critically on downstream detection quality, highlighting a key systems bottleneck in long-tail embodied navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14366v2">Towards Embodied Cognition in Robots via Spatially Grounded Synthetic Worlds</a></div>
    <div class="paper-meta">
      📅 2026-07-27
      | 💬 Accepted to: Intelligent Autonomous Systems (IAS) 2025 as Late Breaking Report
    </div>
    <details class="paper-abstract">
      We present a conceptual framework for training Vision-Language Models (VLMs) to perform Visual Perspective Taking (VPT), a core capability for embodied cognition essential for Human-Robot Interaction (HRI). As a first step toward this goal, we introduce a synthetic dataset, generated in NVIDIA Omniverse, that enables supervised learning for spatial reasoning tasks. Each instance includes an RGB image, a natural language description, and a ground-truth 4X4 transformation matrix representing object pose. We focus on inferring Z-axis distance as a foundational skill, with future extensions targeting full 6 Degrees Of Freedom (DOFs) reasoning. The dataset is publicly available to support further research. This work serves as a foundational step toward embodied AI systems capable of spatial understanding in interactive human-robot scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.24148v1">A Motion-Aware Vector Quantization Framework with Centroid Reuse for Efficient VLA Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-27
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have demonstrated strong potential for embodied AI, yet their high inference latency on GPUs limits real-time deployment. Existing accelerators, such as Dadu-Corki, improve efficiency but treat VLA models as full-precision workloads, leaving substantial redundancy in both memory and computation underexploited. In this paper, we propose VQVLA, an algorithm-hardware co-design framework that accelerates VLA inference by exploiting weight similarity and execution dynamics. We first introduce MotionVQ, a motion-aware vector quantization scheme that dynamically adjusts quantization precision based on the robot's execution state, reducing memory access while preserving task success rate. We then propose a merged-centroid vectorized GEMM paradigm that operates on the codebook-index representation, eliminating redundant multiplications through spatial aggregation and temporal reuse of centroids. To realize these optimizations, we design an accelerator that efficiently supports dynamic precision selection and centroid-reuse computation. Experimental results show that VQVLA achieves 6.5x, 2.8x, 1.9x, 3.3x, and 4.3x speedup over the A100 GPU, Dadu-Corki, LUT-DLA, CodeGEMM, and ShiftAddLLM, respectively, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22355v1">SiPhy: Single-Image Physical Property Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-07-24
      | 💬 Accepted to ECCV 2026 (main track)
    </div>
    <details class="paper-abstract">
      Inferring physical properties such as mass, stiffness, and elasticity from a single image is essential for simulation and embodied AI, yet most existing approaches rely on multi-view reconstruction or physics-based supervision. We introduce SiPhy, a unified framework for single-image physical property reasoning that aligns 3D-aware visual cues, depth with language-based material knowledge. From one RGB image, SiPhy samples pseudo-voxel points, extracts CLIP features, and grounds them to material candidates proposed by a VLM. A part-based contrastive aggregator enforces region consistency, while a heaviness-aware refinement improves thickness and volume estimation for dense objects. Across ABO-500, MVImgNet-100, and PhysXNet-100, SiPhy achieves state-of-the-art single-image performance, surpassing multi-view reconstruction methods by improving mass MnRE by up to 93% (vs. PUGS), reducing density MAE by 35.5% (vs. NeRF2Physics), and lowering Young's modulus error by 23.5%. We further validate SiPhy on real hand-object interaction datasets, demonstrating its potential as a data annotation engine for physical understanding from single-view imagery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22014v1">Zero-Shot Mission-Level Evaluation for Aerial MLLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-07-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) are emerging as core reasoning modules for embodied agents, yet it remains unclear how well general-purpose models can solve long-horizon embodied tasks from a single high-level instruction. We introduce MissionBench, a benchmark for mission-level evaluation of MLLMs in aerial 3D environments. It comprises 120 missions across five simulated 3D environments and four task families. Agents must autonomously plan, navigate, and report outcomes using only egocentric observations and its action history, without aerial-specific fine-tuning. Across 22 open- and closed-source MLLMs, the strongest model succeeds on fewer than 35% of missions compared to 84.4% human performance, highlighting the difficulty of multi-step embodied tasks. Despite large variations between model families, we observe gains from scaling, indicating that larger general-purpose models possess stronger zero-shot embodied capabilities. Our analysis shows that mission-level competence requires coordinating multiple capabilities beyond spatial perception, including multi-step planning and adaptive reasoning. This motivates closed-loop evaluation and highlights both the promise and risk of scaling-driven improvements for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.19893v1">Odin: Primitive-Level Synchronization for Distributed Point-Based Neural Rendering</a></div>
    <div class="paper-meta">
      📅 2026-07-22
      | 💬 Accepted at HPDC 26 (Poster), further submission in progress
    </div>
    <details class="paper-abstract">
      Point-based neural rendering (PBNR) represents 3D scenes as explicit, trainable primitives and underpins high-quality reconstruction and emerging embodied AI and world-model pipelines. Unlike layer-structured neural networks, PBNR has primitive-indexed dependencies: each view reads and updates only a sparse, view-dependent subset of mutable scene state. As large scenes require distributed training and optimized renderers reduce per-view computation, global task- or iteration-level barriers increasingly place synchronization, rather than rendering, on the critical path. We present Odin, a distributed PBNR training system that replaces global barriers with primitive-level synchronization. Its ahead-of-time scheduler uses stable locality and phase order to identify low-conflict overlap windows, while the runtime validates primitive publication before later work observes mutable state. Odin provides a quality-first path that preserves synchronized-training visibility and a throughput-first path that uses overlap and gradient evidence to admit only small, low-impact delayed reads; structural changes and high-impact cases remain synchronized. Across four existing PBNR pipelines and 13 non-city scenes on 8 GPUs, Odin improves throughput by 1.22 times on average and hides 82% of critical-path wait while preserving reconstruction quality. In a MatrixCity mixed-parallel case study scaling to 64 GPUs, Odin improves throughput over Grendel by up to 1.89 times without changing renderer kernels, optimizers, training budgets, or model capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16602v2">PAVXploreRL: Physical-Action-Visual World Model Reinforcement Learning with Action Exploration</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Action-conditioned world models are a key component of embodied AI, serving as scalable policy evaluators that reduce reliance on expensive real-world rollouts. To accurately capture diverse action-induced dynamics, such models should satisfy three key objectives-Physical Plausibility (P), Action Adherence (A), and Visual Fidelity (V), collectively referred to as PAV-while remaining robust to both in-distribution (ID) expert demonstrations and out-of-distribution (OOD) actions. However, existing methods primarily rely on ID action-video pairs and pixel-level reconstruction losses, which do not explicitly optimize PAV objectives and generalize poorly beyond expert data. To address this, we propose PAVXploreRL, a reinforcement learning framework built on a pretrained latent world model that explicitly optimizes PAV objectives through reward-driven training. To improve action generalization, our method jointly leverages ID trajectories and noise-driven OOD action exploration, without paired video supervision. Experiments show that PAVXploreRL consistently outperforms pretrained baselines, achieving a 5.6% average gain across benchmarks and producing higher-quality PAV properties. As a policy evaluator, it also yields more reliable performance estimates and reduces the overestimation bias of prior expert-only world models such as Ctrl-World. Code: https://github.com/Social-AI-Studio/PAVXploreRL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11063v2">AdvNav: Behavior-Guided Black-Box Adversarial Attacks on Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-07-21
      | 💬 ACM International Conference on Multimedia 2026
    </div>
    <details class="paper-abstract">
      Despite progress in Embodied AI, Vision-and-Language Navigation systems remain vulnerable to adversarial visual disturbances. Most existing methods rely on white-box access to target model gradients, which is often unrealistic for real-world deployed systems and computationally exhaustive due to recursive backpropagation for optimization, limiting their applicability. While previous black-box methods predominantly target single-step, instantaneous decision tasks, they struggle to handle the task complexities and temporal dependencies. This highlights the need for a gradient-free attack method that can effectively disrupt the multistep sequential perception-action loop using only observable inputs and outputs. Therefore, we propose AdvNav, a behavior-guided black-box adversarial attack framework that disturbs an agent's first-person views during navigation. To construct an informative surrogate objective for effective optimization guidance in gradient-free search under the black-box setting, we design a dual-granularity behavior-based feedback, aggregating a trajectory-level performance score representing overall navigation degradation, an action-level reward score considering the potential decision risk, and a deviation indicator, all of which are extracted from the agent's self-output behaviors. This feedback guides a hybrid optimization strategy that heuristically tunes perturbation strength via adaptive updates and evolves noise spatial structure genetically, to iteratively discover the most disruptive noise configuration. Evaluated against Transformer-based HAMT and LLM-based MapGPT with two types of backbones on R2R dataset, AdvNav achieves 49.70/65.96/87.30% Attack Success Rate. The result demonstrates the effectiveness and generality of AdvNav, reveals critical perception vulnerabilities and offers insights for the design of future resilient VLN models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01483v3">Spatiotemporal Knowledge Graphs as Persistent Scene Memory for Embodied Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-07-20
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) demonstrate strong image-level scene understanding, but reasoning over long egocentric video remains costly: because VLMs maintain no persistent memory or explicit spatial representation, all sampled frames must be re-processed for every new query. We present VL-KnG, a training-free framework that constructs spatiotemporal knowledge graphs from monocular egocentric video, bridging fine-grained scene graphs and global topological graphs without 3D reconstruction. VL-KnG processes video in chunks, maintains persistent object identities via large language model (LLM)-based Spatiotemporal Object Association (STOA), and answers questions through Graph-Enhanced Retrieval (GER), which combines subgraph retrieval with visual grounding. Once constructed, the knowledge graph removes the need to revisit raw video at query time, decoupling query latency from video length. Across three embodied question answering benchmarks -- OpenEQA, NaVQA, and our newly introduced WalkieKnowledge -- VL-KnG achieves accuracy competitive with frontier VLMs while answering queries at substantially lower latency and with explainable, graph-grounded reasoning; it further surpasses prior persistent-representation baselines and open-weight VLMs in several settings. Deployment on a real robot demonstrates practical applicability, with query latency remaining stable as observation history grows. As a persistent, queryable scene memory, VL-KnG provides a concrete substrate for memory maintenance and knowledge updating in embodied AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13586v2">UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets</a></div>
    <div class="paper-meta">
      📅 2026-07-20
      | 💬 39 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Physically grounded 3D assets are increasingly important for embodied AI and robotic simulation. However, most existing 3D assets lack unified physical semantics, including articulation semantics and intrinsic physical properties, required for realistic interaction. Current approaches either treat these semantics independently or rely on canonicalized object structures, limiting robustness across heterogeneous 3D assets. We present UniPhys, a scalable framework for automatically transforming raw 3D assets into simulation-ready assets with unified physical semantics. Based on UniPhys, we construct UniPhys-40K, a large-scale physically grounded dataset, together with UniPhys-Bench, a carefully verified benchmark for unified physical grounding evaluation. We further introduce UniPhysGen, a unified physical grounding model that jointly reasons over articulation semantics and intrinsic physical properties. UniPhysGen incorporates geometry-robust articulation grounding to mitigate geometric shortcut bias under heterogeneous part decompositions. Extensive experiments demonstrate state-of-the-art performance across articulation grounding and intrinsic physical property estimation tasks, while the resulting assets can be directly deployed in robotic simulation environments for realistic physical interaction. Our code and dataset will be available at https://github.com/breezexian/UniPhysGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17097v1">HarmoHOI: Harmonizing Appearance and 3D Motion for Multi-view Hand-Object Interaction Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-07-19
    </div>
    <details class="paper-abstract">
      Hand-Object Interaction (HOI) synthesis is a cornerstone for animation production and embodied AI. Despite the strong priors of video foundation models, multi-view consistent HOI synthesis remains challenging due to complex hand motions and occlusions. We present HarmoHOI, a unified diffusion framework that jointly and harmoniously generates synchronized multi-view HOI videos and globally aligned 3D point tracks. Our core insight is that robust multi-view consistency fundamentally requires globally aligned 3D geometry and motion. To this end, we propose a Mixture of Multi-view Diffusion Transformer that co-models RGB videos and 3D point tracks. By representing point tracks as pseudo-videos, we align 3D geometric signals with the 2D latent space of foundation models, thereby minimizing the domain gap and easing adaptation of priors. To further ensure geometry consistency, we introduce Global Motion Aligning Diffusion, which refines coarse point tracks into metric-scale, globally aligned 3D trajectories. HarmoHOI enables on-the-fly co-evolution of 2D appearance and 3D motion during denoising. To overcome the scarcity of multi-view HOI data, we employ a hybrid data curriculum learning strategy that successfully transfers generic priors from single-view data to synchronized multi-view generation. Experimental results show that HarmoHOI achieves state-of-the-art performance in visual quality, motion plausibility, and multi-view geometric consistency. Project page available at https://droliven.github.io/HarmoHOI_project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.22702v1">MIME: Multimodal Interactive Motion Encoder</a></div>
    <div class="paper-meta">
      📅 2026-07-18
      | 💬 Under review at WACV 2027
    </div>
    <details class="paper-abstract">
      Text-motion representation learning has advanced rapidly, with growing interest in multi person interactions for animation, AR/VR, and embodied AI. These settings require representations that align language with both individual actor dynamics and the relationships between actors. We introduce the Multimodal Interactive Motion Encoder (MIME), which, to our knowledge, represents the first dedicated multimodal encoder designed specifically for two person interactive motion. MIME captures individual and shared structure using stream based co-attention with explicit interaction features and curriculum based contrastive training. On Inter-X text-motion retrieval, MIME consistently outperforms early and late fusion baselines across gallery sizes, achieving a 12.8% relative improvement in text-to-motion R@1 at a 2,000-sample gallery. We further evaluate MIME as a frozen auxiliary prior within TIMotion and InterMask on the unseen InterHuman dataset. MIME improves semantic alignment metrics while maintaining comparable FID in TIMotion. These results show that interaction aware multimodal encoding improves multi person motion retrieval and transfers across datasets to support downstream motion generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.15868v1">EgoExoMoCap: Distributed Ego-Exo Human Motion Capture</a></div>
    <div class="paper-meta">
      📅 2026-07-17
      | 💬 Accepted by ECCV 2026, Project page and code: https://siplab.org/projects/EgoExoMoCap
    </div>
    <details class="paper-abstract">
      Human motion capture from head-mounted devices (HMDs) offers a scalable way to acquire real-world human motion and interaction data, which is crucial for applications in embodied AI and VR/AR. Existing approaches focus on either egocentric body tracking, estimating the motion of the subject wearing the device, or exocentric tracking, capturing the movements of people in the wearer's surroundings. So far, these two paradigms have largely been explored in isolation. In this paper, we propose a novel distributed framework that jointly leverages ego- and exocentric multi-modal signals for human motion estimation from HMDs. Unlike traditional motion capture systems requiring bulky multi-camera setups or obtrusive mocap suits, our approach, EgoExoMoCap, is as simple as two (or more) people, each wearing a pair of smart glasses. The method leverages head (plus potentially wrist) tracking signals for accurate estimation of global motion in the 3D world and combines context-aware image features based on DINOv3 to achieve robustness in the presence of noise and occlusions. Extensive experiments on two in-the-wild datasets show that our approach can robustly reconstruct motion even in challenging scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22098v2">WorldWander: Bridging Egocentric and Exocentric Worlds in Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-07-16
      | 💬 Accepted by ECCV 2026; Code: https://github.com/showlab/WorldWander
    </div>
    <details class="paper-abstract">
      Recent advances in video world models enable interactive environments with free navigation, making translation between first-person (egocentric) and third-person (exocentric) perspectives increasingly important. However, existing studies focus on unidirectional exocentric-to-egocentric translation, overlooking reference-guided exocentric perspective synthesis. This capability is crucial for gaming and embodied AI applications. Motivated by this, we present WorldWander, an in-context learning framework tailored for translating between egocentric and exocentric worlds in video generation. Building upon advanced video diffusion transformers, WorldWander integrates (i) In-Context Perspective Alignment and (ii) Collaborative Position Encoding to model cross-view synchronization and character consistency. To support our task, we curate EgoExo-8K, a dynamic and scene-rich dataset containing synchronized egocentric-exocentric triplets from both synthetic and real-world scenarios. Experiments demonstrate that WorldWander achieves superior perspective synchronization, character consistency, and generalization, setting a new benchmark for egocentric-exocentric video translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13586v1">UniPhysGen: Unified Physical Grounding for Simulation-Ready 3D Assets</a></div>
    <div class="paper-meta">
      📅 2026-07-15
      | 💬 39 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Physically grounded 3D assets are increasingly important for embodied AI and robotic simulation. However, most existing 3D assets lack unified physical semantics, including articulation semantics and intrinsic physical properties, required for realistic interaction. Current approaches either treat these semantics independently or rely on canonicalized object structures, limiting robustness across heterogeneous 3D assets. We present UniPhys, a scalable framework for automatically transforming raw 3D assets into simulation-ready assets with unified physical semantics. Based on UniPhys, we construct UniPhys-40K, a large-scale physically grounded dataset, together with UniPhys-Bench, a carefully verified benchmark for unified physical grounding evaluation. We further introduce UniPhysGen, a unified physical grounding model that jointly reasons over articulation semantics and intrinsic physical properties. UniPhysGen incorporates geometry-robust articulation grounding to mitigate geometric shortcut bias under heterogeneous part decompositions. Extensive experiments demonstrate state-of-the-art performance across articulation grounding and intrinsic physical property estimation tasks, while the resulting assets can be directly deployed in robotic simulation environments for realistic physical interaction. Our code and dataset will be available at https://github.com/breezexian/UniPhysGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13560v1">Grounded world models in biological organisms and future embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-07-15
    </div>
    <details class="paper-abstract">
      Recent advances in generative and embodied AI have been driven by large-scale predictive learning over multimodal data. However, the resulting systems remain largely based on passive training regimes where linguistic regularities create the scaffold onto which information from other modalities is attached. Conversely, neuroscience and cognitive science suggest that biological intelligence is organized in the opposite way, where grounded world models acquired through interaction with the environment provide the semantic scaffold to which language is attached. Here, we illustrate five examples of neural circuits supporting grounded world modelling, which underlie navigation in physical and conceptual spaces, affordance-based perception and interaction with objects, active perception and exploratory learning, allostatic control and emotion, and the distinction between self- and world-generated outcomes. These examples highlight several features largely missing from current embodied AI, including the role of intrinsic dynamics as a foundation for learning, the centrality of action in aligning these dynamics with the external world, the prominence of autonomous experience and open-ended learning over passive assimilation of externally provided data, and the fact that early predictive and control mechanisms scaffold higher cognitive abilities such as reasoning, conceptual navigation, planning, imagination, understanding others' minds, and communication. Finally, we discuss whether and how principles derived from biological systems may inform future embodied AI, including training regimes based on social interaction to construct world models that are not only grounded but also socially shared and aligned with human norms and values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.12297v1">MobileSAM2: Lightweight Segment Anything for Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-07-14
      | 💬 Accepted to ECCV 2026
    </div>
    <details class="paper-abstract">
      The recent large video foundation model, SAM2, enables segment anything in both images and videos, serving as a powerful base model for various applications. However, many of such use cases require to operate on resource-constrained devices like mobile phones and laptops. In this work, we aim to make SAM2 more mobile-friendly by distilling the heavyweight SAM2 into a lightweight model, facilitating segment anything in both images and videos on mobile devices. To this end, we propose Hypergraphical Knowledge Distill (HyperKD), which introduces the idea of hypergraph into knowledge distillation, aiming to effectively model and transfer SAM2's generalizable and comprehensive knowledge. HyperKD consists of Temporal HyperKD and Granularity HyperKD that construct hypergraphs to explicitly model and extract the generalizable temporal knowledge and the comprehensive multi-granularity knowledge from SAM2 respectively, which are then distilled into the lightweight student model by aligning it with the constructed hypergraphs. Besides, we present MobileSAM2, a new family of lightweight SAM2 that balances efficiency and effectiveness via searching the best model architectures with HyperKD during model size reduction. Extensive experiments validate MobileSAM2 across multiple benchmarks and show promising generalization performance on embodied AI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.11063v1">AdvNav: Behavior-Guided Black-Box Adversarial Attacks on Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-07-13
      | 💬 ACM International Conference on Multimedia 2026
    </div>
    <details class="paper-abstract">
      Despite progress in Embodied AI, Vision-and-Language Navigation systems remain vulnerable to adversarial visual disturbances. Most existing methods rely on white-box access to target model gradients, which is often unrealistic for real-world deployed systems and computationally exhaustive due to recursive backpropagation for optimization, limiting their applicability. While previous black-box methods predominantly target single-step, instantaneous decision tasks, they struggle to handle the task complexities and temporal dependencies. This highlights the need for a gradient-free attack method that can effectively disrupt the multistep sequential perception-action loop using only observable inputs and outputs. Therefore, we propose AdvNav, a behavior-guided black-box adversarial attack framework that disturbs an agent's first-person views during navigation. To construct an informative surrogate objective for effective optimization guidance in gradient-free search under the black-box setting, we design a dual-granularity behavior-based feedback, aggregating a trajectory-level performance score representing overall navigation degradation, an action-level reward score considering the potential decision risk, and a deviation indicator, all of which are extracted from the agent's self-output behaviors. This feedback guides a hybrid optimization strategy that heuristically tunes perturbation strength via adaptive updates and evolves noise spatial structure genetically, to iteratively discover the most disruptive noise configuration. Evaluated against Transformer-based HAMT and LLM-based MapGPT with two types of backbones on R2R dataset, AdvNav achieves 49.70/65.96/87.30% Attack Success Rate. The result demonstrates the effectiveness and generality of AdvNav, reveals critical perception vulnerabilities and offers insights for the design of future resilient VLN models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07459v2">EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-07-12
      | 💬 Minor revisions to presentation; technical content and results remain unchanged. Project page: horizonrobotics.github.io/EmbodiedGen
    </div>
    <details class="paper-abstract">
      We present EmbodiedGen V2, a generative 3D world engine for building executable policy-ready environments for embodied intelligence. Sim-ready 3D asset generation has advanced rapidly, yet assembling such assets into policy-ready task environments remains largely manual, limiting scalable closed-loop learning. EmbodiedGen V2 addresses this gap through a unified sim-ready representation that connects cross-simulator assets, interaction affordances, task-driven worlds, large-scale multi-room scenes, and stateful Vibe Coding into a generative, editable, and reusable simulation pipeline. The generated environments support manipulation, navigation, mobile manipulation, cross-simulator deployment, and embodied policy training. In evaluation, the asset pipeline achieves 96.5% human acceptance and 98.6% collision success, and 83.3% of task-driven worlds are directly usable for downstream simulation without manual modification. Online reinforcement learning with generated environments further improves simulation success from 9.7% to 79.8%, and transfers to real robots with task success increasing from 21.7% to 75.0%. These results establish EmbodiedGen V2 as scalable simulation infrastructure for training, evaluating, and deploying embodied policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.09520v1">Seeing is Free, Speaking is Not: Uncovering the True Energy Bottleneck in Edge VLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-07-10
      | 💬 Accepted to ACM MM 2026. 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are the perceptual backbone of embodied AI, but their energy footprint on edge hardware remains poorly understood. Existing efficiency efforts focus predominantly on reducing visual tokens, implicitly treating visual processing as the dominant energy cost. We overturn this implicit assumption through the first systematic energy profiling of on-device VLM inference, spanning five models across three architecture families, four input resolutions, and two hardware platforms (NVIDIA RTX 3070 and Jetson Orin NX). Our analysis yields three findings. First, average inference power is a model-intrinsic constant, invariant to input resolution, image complexity, and prompt type, with less than 5% variation across all conditions. This means that all energy variation across inputs must arise from variation in inference time, not from variation in power draw. Second, each output token costs 11 to 39x more wall-clock time than each input token due to the compute-bound and memory-bound asymmetry between prefill and decode, making output token count the dominant driver of both latency and energy. Third, image complexity, measured by the number of objects in an image, induces up to 4.1x energy differences at identical resolution. This variation arises not from increased visual processing cost, but from differences in output length. These findings expose a fundamental limitation of visual token pruning: even removing all visual tokens saves at most 10% of total energy for fixed-token models. Across models spanning 1 billion to 8 billion parameters, controlling output length saves up to 97% of total energy, with the energy dominance of decoding growing stronger at larger model scale. In short, the true energy bottleneck in edge VLM inference is not what the model sees, but how much it says.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13059v1">GPUSimBench: Towards Scalable and Reliable GPU-Accelerated Simulators in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-07-06
      | 💬 Accepted by IROS 2026
    </div>
    <details class="paper-abstract">
      Data-driven embodied AI is rapidly transitioning into a paradigm that scales training through massively parallel simulation, where GPU-accelerated simulators serve as the foundational data infrastructure. However, as computational throughput scales, the underlying trade-offs between parallel efficiency, physical fidelity, and execution determinism remain largely unexamined, hindering the development of reliable robot learning. In this paper, we expose the hidden limits of mainstream GPU-based robotic simulators (e.g., Isaac Lab, Genesis) by introducing GPUSimBench, which focuses on scalability, physical consistency, and computational determinism. First, GPUSimBench establishes a physical grounding evaluation with a controlled inclined-plane task, quantifying the distributional alignment between simulated dynamics and their real-world counterparts. Second, we benchmark parallel scalability by measuring throughput and memory footprints across scaling environment counts. Crucially, beyond standard performance metrics, we unveil and quantify the inherent non-determinism introduced by GPU-batched execution, characterized by significant run-to-run and inter-environment variability even under identical initial conditions. Finally, we identify four empirical regimes of stochasticity within current simulator stacks, highlighting that unbounded scaling can compromise reproducibility without explicit constraints.
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
