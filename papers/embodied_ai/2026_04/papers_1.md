# embodied ai - 2026_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27621v1">Robot Learning from Human Videos: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Paper list: https://github.com/IRMVLab/awesome-robot-learning-from-human-videos
    </div>
    <details class="paper-abstract">
      A critical bottleneck hindering further advancement in embodied AI and robotics is the challenge of scaling robot data. To address this, the field of learning robot manipulation skills from human video data has attracted rapidly growing attention in recent years, driven by the abundance of human activity videos and advances in computer vision. This line of research promises to enable robots to acquire skills passively from the vast and readily available resource of human demonstrations, substantially favoring scalable learning for generalist robotic systems. Therefore, we present this survey to provide a comprehensive and up-to-date review of human-video-based learning techniques in robotics, focusing on both human-robot skill transfer and data foundations. We first review the policy learning foundations in robotics, and then describe the fundamental interfaces to incorporate human videos. Subsequently, we introduce a hierarchical taxonomy of transferring human videos to robot skills, covering task-, observation-, and action-oriented pathways, along with a cross-family analysis of their couplings with different data configurations and learning paradigms. In addition, we investigate the data foundations including widely-used human video datasets and video generation schemes, and provide large-scale statistical trends in dataset development and utilization. Ultimately, we emphasize the challenges and limitations intrinsic to this field, and delineate potential avenues for future research. The paper list of our survey is available at https://github.com/IRMVLab/awesome-robot-learning-from-human-videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27578v1">World2Minecraft: Occupancy-Driven Simulated Scenes Construction</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Embodied intelligence requires high-fidelity simulation environments to support perception and decision-making, yet existing platforms often suffer from data contamination and limited flexibility. To mitigate this, we propose World2Minecraft to convert real-world scenes into structured Minecraft environments based on 3D semantic occupancy prediction. In the reconstructed scenes, we can effortlessly perform downstream tasks such as Vision-Language Navigation(VLN). However, we observe that reconstruction quality heavily depends on accurate occupancy prediction, which remains limited by data scarcity and poor generalization in existing models. We introduce a low-cost, automated, and scalable data acquisition pipeline for creating customized occupancy datasets, and demonstrate its effectiveness through MinecraftOcc, a large-scale dataset featuring 100,165 images from 156 richly detailed indoor scenes. Extensive experiments show that our dataset provides a critical complement to existing datasets and poses a significant challenge to current SOTA methods. These findings contribute to improving occupancy prediction and highlight the value of World2Minecraft in providing a customizable and editable platform for personalized embodied AI research. Project page:https://world2minecraft.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27555v1">SpatialGrammar: A Domain-Specific Language for LLM-Based 3D Indoor Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Automatically generating interactive 3D indoor scenes from natural language is crucial for virtual reality, gaming, and embodied AI. However, existing LLM-based approaches often suffer from spatial errors and collisions, in part because common scene representations-raw coordinates or verbose code-are difficult for models to reason about 3D spatial relationships and physical constraints. We propose SpatialGrammar, a domain-specific language that represents gravity-aligned indoor layouts as BEV grid placements with deterministic compilation to valid 3D geometry, enabling verifiable constraint checking. Building on this representation, we develop (1) SG-Agent, a closed-loop system that uses compiler feedback to iteratively refine scenes and enforce collision constraints, and (2) SG-Mini, a 104M-parameter model trained entirely on compiler-validated synthetic data. Across 159 test scenes spanning five scenarios of different complexity, SG-Agent improves spatial fidelity and physical plausibility over prior methods, while SG-Mini performs competitively against larger LLM-based baselines on single-shot generation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23065v2">EAG-PT: Emission-Aware Gaussians and Path Tracing for Diffuse Indoor Scene Reconstruction and Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 SIGGRAPH 2026 Conference Paper; Project Page: https://eag-pt.github.io
    </div>
    <details class="paper-abstract">
      Recent radiance-field-based reconstruction methods, such as NeRF and 3DGS, achieve high visual fidelity for indoor scenes, but often break down under scene editing due to baked illumination and the lack of explicit light transport. In contrast, inverse path tracing methods based on mesh representations enforce correct light transport but require highly accurate geometry, making them difficult to apply robustly to real indoor scenes. We present Emission-Aware Gaussians and Path Tracing (EAG-PT), a method for physically based reconstruction and rendering of indoor scenes using a unified 2D Gaussian representation, targeting editable diffuse global illumination. Our approach consists of three key ideas: (1) representing indoor scenes with 2D Gaussians as a transport-friendly geometric proxy that avoids explicit mesh reconstruction; (2) explicitly separating emissive and non-emissive components during reconstruction to support editing; and (3) decoupling reconstruction from final rendering by using efficient single-bounce optimization and high-quality multi-bounce path tracing, respectively. Experiments on synthetic and real indoor scenes show that EAG-PT produces more natural and physically consistent edited renderings than radiance-field reconstructions, while preserving finer geometric detail and avoiding mesh-induced artifacts compared with mesh-based inverse path tracing. These results highlight the potential of our approach for applications such as interior design, XR content creation, and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03992v2">Value-Guided Iterative Refinement and the DIQ-H Benchmark for Evaluating VLM Robustness</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are essential for embodied AI and safety-critical applications, such as robotics and autonomous systems. However, existing benchmarks primarily focus on static or curated visual inputs, neglecting the challenges posed by adversarial conditions, value misalignment, and error propagation in continuous deployment. Current benchmarks either overlook the impact of real-world perturbations, or fail to account for the cumulative effect of inconsistent reasoning over time. To address these gaps, we introduce the Degraded Image Quality Leading to Hallucinations (DIQ-H) benchmark, the first to evaluate VLMs under adversarial visual conditions in continuous sequences. DIQ-H simulates real-world stressors including motion blur, sensor noise, and compression artifacts, and measures how these corruptions lead to persistent errors and misaligned outputs across time. The benchmark explicitly models error propagation and its long-term value consistency. To enhance scalability and reduce costs for safety-critical evaluation, we propose the Value-Guided Iterative Refinement (VIR) framework, which automates the generation of high-quality, ethically aligned ground truth annotations. VGIR leverages lightweight VLMs to detect and refine value misalignment, improving accuracy from 72.2% to 83.3%, representing a 15.3% relative improvement. The DIQ-H benchmark and VGIR framework provide a robust platform for embodied AI safety assessment, revealing vulnerabilities in error recovery, ethical consistency, and temporal value alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26509v1">3D Generation for Embodied AI and Robotic Simulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 26 pages, 11 figures, 8 tables. Project Page: https://3dgen4robot.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey presents the first survey of 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In \emph{Data Generator}, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in \emph{Simulation Environments}, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in \emph{Sim2Real Bridge}, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at https://3dgen4robot.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22227v3">A Co-Evolutionary Theory of Human-AI Coexistence: Mutualism, Governance, and Dynamics in Complex Societies</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Classical robot ethics is often framed around obedience, including Asimov's laws. This framing is insufficient for contemporary AI systems, which are increasingly adaptive, generative, embodied, and embedded in physical, psychological, and social environments. This paper proposes conditional mutualism under governance as a framework for human-AI coexistence: a co-evolutionary relationship in which humans and AI systems develop, specialize, and coordinate under institutional conditions that preserve reciprocity, reversibility, psychological safety, and social legitimacy. We synthesize concepts from computability, machine learning, foundation models, embodied AI, alignment, human-robot interaction, ecological mutualism, coevolution, and polycentric governance. We then formalize coexistence as a multiplex dynamical system across physical, psychological, and social layers, with reciprocal supply-demand coupling, conflict penalties, developmental freedom, and governance regularization. The model gives conditions for existence, uniqueness, and global asymptotic stability of equilibria. We complement the analytical results with deterministic ODE simulations, basin sweeps, sensitivity analyses, governance-regime comparisons, shock tests, and local stability checks. The simulations indicate that governed mutualism reaches a high coexistence index with negligible domination, whereas insufficient or excessive governance can produce domination, weak-benefit lock-in, or suppressed developmental freedom. The results suggest that human-AI coexistence should be designed as a co-evolutionary governance problem rather than as a static obedience problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26261v1">Multiple Consistent 2D-3D Mappings for Robust Zero-Shot 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Zero-shot 3D Visual Grounding (3DVG) is a critical capability for open-world embodied AI. However, existing methods are fundamentally bottlenecked by the poor quality of open-vocabulary 3D proposals, suffering from inaccurate categories and imprecise geometries, as well as the spatial redundancy of exhaustive multi-view reasoning. To address these challenges, we propose MCM-VG, a novel framework that achieves robust zero-shot 3DVG by explicitly establishing Multiple Consistent 2D-3D Mappings. Instead of passively relying on noisy 3D segments, MCM-VG enforces 2D-3D consistency across three fundamental dimensions to achieve precise target localization and reliable reasoning. First, a Semantic Alignment module corrects category mismatches via LLM-driven query parsing and coarse-to-fine 2D-3D matching. Second, an Instance Rectification module leverages VLM-guided 2D segmentations to reconstruct missing targets, back-projecting these reliable visual priors to establish accurate 3D geometries. Finally, to eliminate spatial redundancy, a Viewpoint Distillation module clusters 3D camera directions to extract optimal frames. By pairing these optimal RGB frames with Bird's Eye View maps into concise visual prompt sets, we formulate the final target disambiguation as a multiple-choice reasoning task for Vision-Language Models. Extensive evaluations on ScanRefer and Nr3D benchmarks demonstrate that MCM-VG sets a new state-of-the-art for zero-shot 3D visual grounding. Remarkably, it achieves 62.0\% and 53.6\% in Acc@0.25 and Acc@0.5 on ScanRefer, outperforming previous baselines by substantial margins of 6.4\% and 4.0\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20714v2">Inferix: A Block-Diffusion based Next-Generation Inference Engine for World Simulation</a></div>
    <div class="paper-meta">
      📅 2026-04-28
    </div>
    <details class="paper-abstract">
      World models serve as core simulators for fields such as agentic AI, embodied AI, and gaming, capable of generating long, physically realistic, and interactive high-quality videos. Moreover, scaling these models could unlock emergent capabilities in visual perception, understanding, and reasoning, paving the way for a new paradigm that moves beyond current LLM-centric vision foundation models. A key breakthrough empowering them is the semi-autoregressive (block-diffusion) decoding paradigm, which merges the strengths of diffusion and autoregressive methods by generating video tokens in block-applying diffusion within each block while conditioning on previous ones, resulting in more coherent and stable video sequences. Crucially, it overcomes limitations of standard video diffusion by reintroducing LLM-style KV Cache management, enabling efficient, variable-length, and high-quality generation. Therefore, Inferix is specifically designed as a next-generation inference engine to enable immersive world synthesis through optimized semi-autoregressive decoding processes. This dedicated focus on world simulation distinctly sets it apart from systems engineered for high-concurrency scenarios (like vLLM or SGLang) and from classic video diffusion models (such as xDiTs). Inferix further enhances its offering with interactive video streaming and profiling, enabling real-time interaction and realistic simulation to accurately model world dynamics. Additionally, it supports efficient benchmarking through seamless integration of LV-Bench, a new fine-grained evaluation benchmark tailored for minute-long video generation scenarios. We hope the community will work together to advance Inferix and foster world model exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03136v2">Limited Linguistic Diversity in Embodied AI Datasets</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted to ACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions--including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16518v2">MiMo-Embodied: X-Embodied Foundation Model Technical Report</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Code: https://github.com/XiaomiMiMo/MiMo-Embodied | Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B
    </div>
    <details class="paper-abstract">
      We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25459v1">GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Embodied AI research is undergoing a shift toward vision-centric perceptual paradigms. While massively parallel simulators have catalyzed breakthroughs in proprioception-based locomotion, their potential remains largely untapped for vision-informed tasks due to the prohibitive computational overhead of large-scale photorealistic rendering. Furthermore, the creation of simulation-ready 3D assets heavily relies on labor-intensive manual modeling, while the significant sim-to-real physical gap hinders the transfer of contact-rich manipulation policies. To address these bottlenecks, we propose GS-Playground, a multi-modal simulation framework designed to accelerate end-to-end perceptual learning. We develop a novel high-performance parallel physics engine, specifically designed to integrate with a batch 3D Gaussian Splatting (3DGS) rendering pipeline to ensure high-fidelity synchronization. Our system achieves a breakthrough throughput of 10^4 FPS at 640x480 resolution, significantly lowering the barrier for large-scale visual RL. Additionally, we introduce an automated Real2Sim workflow that reconstructs photorealistic, physically consistent, and memory-efficient environments, streamlining the generation of complex simulation-ready scenes. Extensive experiments on locomotion, navigation, and manipulation demonstrate that GS-Playground effectively bridges the perceptual and physical gaps across diverse embodied tasks. Project homepage: https://gsplayground.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10813v4">InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts</a></div>
    <div class="paper-meta">
      📅 2026-04-28
      | 💬 Accepted by NeurIPS 2025; Project page: https://marjordcpz.github.io/InternScenes.github.io
    </div>
    <details class="paper-abstract">
      The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24527v1">Interoceptive machine framework: Toward interoception-inspired regulatory architectures in artificial intelligence</a></div>
    <div class="paper-meta">
      📅 2026-04-27
    </div>
    <details class="paper-abstract">
      This review proposes an integrative framework grounded on interoception and embodied AI-termed the interoceptive machine framework-that translates biologically inspired principles of internal-state regulation into computational architectures for adaptive autonomy. Interoception, conceived as the monitoring, integration, and regulation of internal signals, has proven relevant for understanding adaptive behavior in biological systems. The proposed framework organizes interoceptive contributions into three functional principles: homeostatic, allostatic, and enactive, each associated with distinct computational roles: internal viability regulation, anticipatory uncertainty-based re-evaluation, and active data generation through interaction. These principles are not intended as direct neurophysiological mappings, but as abstractions that inform the design of artificial agents with improved self-regulation and context-sensitive behavior. By embedding internal state variables and regulatory loops within these principles, AI systems can achieve more robust decision-making, calibrated uncertainty handling, and adaptive interaction strategies, particularly in uncertain and dynamic environments. This approach provides a concrete and testable pathway toward agents capable of functionally grounded self-regulation, with direct implications for human-computer interaction and assistive technologies. Ultimately, the interoceptive machine framework offers a unifying perspective on how internal-state regulation can enhance autonomy, adaptivity, and robustness in embodied AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24039v1">AgenticCache: Cache-Driven Asynchronous Planning for Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-27
      | 💬 Accepted at MLSys 2026
    </div>
    <details class="paper-abstract">
      Embodied AI agents increasingly rely on large language models (LLMs) for planning, yet per-step LLM calls impose severe latency and cost. In this paper, we show that embodied tasks exhibit strong plan locality, where the next plan is largely predictable from the current one. Building on this, we introduce AgenticCache, a planning framework that reuses cached plans to avoid per-step LLM calls. In AgenticCache, each agent queries a runtime cache of frequent plan transitions, while a background Cache Updater asynchronously calls the LLM to validate and refine cached entries. Across four multi-agent embodied benchmarks, AgenticCache improves task success rate by 22% on average across 12 configurations (4 benchmarks x 3 models), reduces simulation latency by 65%, and lowers token usage by 50%. Cache-based plan reuse thus offers a practical path to low-latency, low-cost embodied agents. Code is available at https://github.com/hojoonleokim/MLSys26_AgenticCache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23629v1">From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-26
      | 💬 Preprint. 41 pages, 5 figures. Jiafeng Wu and Zhuofan Lou contributed equally. Project page: https://christinebobby.github.io/production-ready-3d-survey/
    </div>
    <details class="paper-abstract">
      Three-dimensional content generation has progressed from producing isolated, visually plausible shapes to constructing structured assets that can be deployed in real-time interactive environments. This trajectory is driven by converging demands from game development, embodied AI, world simulation, digital twins, and spatial computing, all of which require 3D content that goes beyond surface appearance to satisfy engine-level constraints on topology, UV parameterization, physically based materials, skeletal rigging, and physics-aware scene layout. Despite rapid advances in generative modeling, a persistent gap separates the outputs of current methods from the production-ready standard expected by interactive applications. This survey addresses that gap by organizing the literature around the asset production pipeline rather than algorithmic families. Along the horizontal axis we distinguish three asset tiers, namely general objects, characters, and scenes, while the vertical axis traces each tier through the full production lifecycle from data foundations and geometry synthesis through topology optimization, UV unwrapping, PBR appearance, rigging, and scene assembly. Through this two-dimensional taxonomy we assess not only what current methods can generate but whether their outputs are directly usable in downstream engines and simulation platforms. We further consolidate evaluation metrics and protocols that span geometric fidelity, appearance quality, asset usability, and scene-level physical plausibility. The survey concludes by identifying open challenges in data quality, generation controllability, end-to-end assetization, and physically grounded generation, and by situating production-ready 3D content as foundational infrastructure for emerging interactive world models and embodied intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23580v1">PhysCodeBench: Benchmarking Physics-Aware Symbolic Simulation of 3D Scenes via Self-Corrective Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2026-04-26
    </div>
    <details class="paper-abstract">
      Physics-aware symbolic simulation of 3D scenes is critical for robotics, embodied AI, and scientific computing, requiring models to understand natural language descriptions of physical phenomena and translate them into executable simulation environments. While large language models (LLMs) excel at general code generation, they struggle with the semantic gap between physical descriptions and simulation implementation. We introduce PhysCodeBench, the first comprehensive benchmark for evaluating physics-aware symbolic simulation, comprising 700 manually-crafted diverse samples across mechanics, fluid dynamics, and soft-body physics with expert annotations. Our evaluation framework measures both code executability and physical accuracy through automated and visual assessment. Building on this, we propose a Self-Corrective Multi-Agent Refinement Framework (SMRF) with three specialized agents (simulation generator, error corrector, and simulation refiner) that collaborate iteratively with domain-specific validation to produce physically accurate simulations. SMRF achieves 67.7 points overall performance compared to 36.3 points for the best baseline among evaluated SOTA models, representing a 31.4-point improvement. Our analysis demonstrates that error correction is critical for accurate physics-aware symbolic simulation and that specialized multi-agent approaches significantly outperform single-agent methods across the tested physical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23018v1">AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-04-24
    </div>
    <details class="paper-abstract">
      Web-scale 3D asset collections are abundant, but rarely deployment-ready. Assets ship with arbitrary metric scale, incorrect pivots and forward axes, brittle geometry, and textures that do not support relighting, which limits their utility for embodied AI, robotics simulation, game development, and AR/VR. We present AmaraSpatial-10K, a dataset of over 10,000 synthetic 3D assets designed for downstream use rather than volume alone. Each asset is released as a metric-scaled, semantically anchored .glb with separated PBR material maps, a convex collision hull, a paired reference image, and rich multi-sentence text metadata. The dataset spans indoor objects, vehicles, architecture, creatures, and props under a unified spatial convention. Alongside the dataset, we introduce an evaluation suite for 3D asset banks. The suite comprises a continuous Scale Plausibility Score (SPS) with an LLM-as-Judge interval protocol, an LLM Concept Density score for metadata, an anchor-error metric, and a cross-modal CLIP coherence protocol, and we use it to audit AmaraSpatial-10K alongside matched subsets from Objaverse, HSSD, ABO, and GSO. Compared with Objaverse-sourced assets, we demonstrate that AmaraSpatial-10K substantially improves text-based retrieval precision (CLIP Recall@5 of 0.612 vs 0.181, a 3.4x improvement with median rank falling from 267 to 3), and we establish that it satisfies the spatial and semantic prerequisites for physics-aware scene composition and embodied-AI asset banks, leaving those downstream evaluations to future work. AmaraSpatial-10K is publicly available on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17706v2">OmniVLA-RL: A Vision-Language-Action Model with Spatial Understanding and Online RL</a></div>
    <div class="paper-meta">
      📅 2026-04-24
    </div>
    <details class="paper-abstract">
      Visual-Language-Action (VLA) models represent a paradigm shift in embodied AI, yet existing frameworks often struggle with imprecise spatial perception, suboptimal multimodal fusion, and instability in reinforcement learning. To bridge these gaps, we propose OmniVLA-RL, a novel architecture that leverages a Mix-of-Transformers (MoT) design to synergistically integrate reasoning, spatial, and action experts. Furthermore, we introduce Flow-GSPO, which reformulates flow matching as a Stochastic Differential Equation (SDE) process and integrates it with Group Segmented Policy Optimization (GSPO) to enhance action precision and training robustness. Extensive evaluations on the LIBERO and LIBERO-Plus benchmarks demonstrate that OmniVLA-RL achieves decent overall performance and surpasses mainstream existing methods, effectively overcoming the fundamental limitations of current VLA models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05673v2">Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 18 pages, 7 figures, 10 tables. Code available at https://github.com/WuyangLuan/RSBM
    </div>
    <details class="paper-abstract">
      Visual navigation is a core challenge in Embodied AI, requiring autonomous agents to translate high-dimensional sensory observations into continuous, long-horizon action trajectories. While generative policies based on diffusion models and Schrödinger Bridges (SB) effectively capture multimodal action distributions, they require dozens of integration steps due to high-variance stochastic transport, posing a critical barrier for real-time robotic control. We propose Rectified Schrödinger Bridge Matching (RSBM), a framework that exploits a shared velocity-field structure between standard Schrödinger Bridges ($\varepsilon=1$, maximum-entropy transport) and deterministic Optimal Transport ($\varepsilon\to 0$, as in Conditional Flow Matching), controlled by a single entropic regularization parameter $\varepsilon$. We prove two key results: (1) the conditional velocity field's functional form is invariant across the entire $\varepsilon$-spectrum (Velocity Structure Invariance), enabling a single network to serve all regularization strengths; and (2) reducing $\varepsilon$ linearly decreases the conditional velocity variance, enabling more stable coarse-step ODE integration. Anchored to a learned conditional prior that shortens transport distance, RSBM operates at an intermediate $\varepsilon$ that balances multimodal coverage and path straightness. Empirically, while standard bridges require $\geq 10$ steps to converge, RSBM achieves over 94% cosine similarity and 92% success rate in merely 3 integration steps -- without distillation or multi-stage training -- substantially narrowing the gap between high-fidelity generative policies and the low-latency demands of Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21377v1">A Replicable Robotics Awareness Method Using LLM-Enabled Robotics Interaction: Evidence from a Corporate Challenge</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 10 pages, 8 Figures, to be submitted for journal per-review
    </div>
    <details class="paper-abstract">
      Large language models are increasingly being explored as interfaces between humans and robotic systems, yet there remains limited evidence on how such technologies can be used not only for interaction, but also as a structured means of introducing robotics to non-specialist users in real organizational settings. This paper introduces and evaluates a challenge-based method for robotics awareness, implemented through an LLM-enabled humanoid robot activity conducted with employees of AD Ports Group in the United Arab Emirates. In the event, participants engaged with a humanoid robot in a logistics-inspired task environment using voice commands interpreted through an LLM-based control framework. The activity was designed as a team-based, role-driven experience intended to expose participants to embodied AI and human-robot collaboration without requiring prior robotics expertise. To evaluate the approach, a post-event survey remained open for 16 days and collected 102 responses. Results indicate strong overall reception, with high satisfaction (8.46/10), increased interest in robotics and AI (4.47/5), and improved understanding of emerging forms of human-robot collaboration (4.45/5). Participants who interacted directly with the robot also reported natural interaction (4.37/5) and a strong sense that interaction became easier as the activity progressed (4.74/5). At the same time, lower ratings for reliability and predictability point to important technical and design challenges for future iterations. The findings suggest that challenge-based, LLM-enabled humanoid interaction can serve as a promising and replicable method for robotics awareness in industrial and operational environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17969v2">E3VS-Bench: A Benchmark for Viewpoint-Dependent Active Perception in 3D Gaussian Splatting Scenes</a></div>
    <div class="paper-meta">
      📅 2026-04-23
      | 💬 Project page: https://k0uya.github.io/e3vs-proj/
    </div>
    <details class="paper-abstract">
      Visual search in 3D environments requires embodied agents to actively explore their surroundings and acquire task-relevant evidence. However, existing visual search and embodied AI benchmarks, including EQA, typically rely on static observations or constrained egocentric motion, and thus do not explicitly evaluate fine-grained viewpoint-dependent phenomena that arise under unrestricted 5-DoF viewpoint control in real-world 3D environments, such as visibility changes caused by vertical viewpoint shifts, revealing contents inside containers, and disambiguating object attributes that are only observable from specific angles. To address this limitation, we introduce {E3VS-Bench}, a benchmark for embodied 3D visual search where agents must control their viewpoints in 5-DoF to gather viewpoint-dependent evidence for question answering. E3VS-Bench consists of 99 high-fidelity 3D scenes reconstructed using 3D Gaussian Splatting and 2,014 question-driven episodes. 3D Gaussian Splatting enables photorealistic free-viewpoint rendering that preserves fine-grained visual details (e.g., small text and subtle attributes) often degraded in mesh-based simulators, thereby allowing the construction of questions that cannot be answered from a single view and instead require active inspection across viewpoints in 5-DoF. We evaluate multiple state-of-the-art VLMs and compare their performance with humans. Despite strong 2D reasoning ability, all models exhibit a substantial gap from humans, highlighting limitations in active perception and coherent viewpoint planning specifically under full 5-DoF viewpoint changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20193v1">LLM-Guided Safety Agent for Edge Robotics with an ISO-Compliant Perception-Compute-Control Architecture</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      Ensuring functional safety in human-robot interaction is challenging because AI perception is inherently probabilistic, whereas industrial standards require deterministic behavior. We present an LLM-guided safety agent for edge robotics, built on an ISO-compliant low-latency perception-compute-control architecture. Our method translates natural-language safety regulations into executable predicates and deploys them through a redundant heterogeneous edge runtime. For fault-tolerant closed-loop execution under edge constraints, we adopt a symmetric dual-modular redundancy design with parallel independent execution for low-latency perception, computation, and control. We prototype the system on a dual-RK3588 platform and evaluate it in representative human-robot interaction scenarios. The results demonstrate a practical edge implementation path toward ISO 13849 Category 3 and PL d using cost-effective hardware, supporting practical deployment of safety-critical embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.20910v1">Planetary Exploration 3.0: A Roadmap for Software-Defined, Radically Adaptive Space Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-22
    </div>
    <details class="paper-abstract">
      The surface and subsurface of worlds beyond Mars remain largely unexplored. Yet these worlds hold keys to fundamental questions in planetary science - from potentially habitable subsurface oceans on icy moons to ancient records preserved in Kuiper Belt objects. NASA's success in Mars exploration was achieved through incrementalism: 22 progressively sophisticated missions over decades. This paradigm, which we call Planetary Exploration 2.0 (PE 2.0), is untenable for the outer Solar System, where cruise times of a decade or more make iterative missions infeasible. We propose Planetary Exploration 3.0 (PE 3.0): a paradigm in which unvisited worlds are explored by a single or a few missions with radically adaptive space systems. A PE 3.0 mission conducts both initial exploratory science and follow-on hypothesis-driven science based on its own in situ data returns, evolving spacecraft capabilities to work resiliently in previously unseen environments. The key enabler of PE 3.0 is software-defined space systems (SDSSs) - systems that can adapt their functions at all levels through software updates. This paper presents findings from a Keck Institute for Space Studies (KISS) workshop on PE 3.0, covering: (1) PE 3.0 systems engineering including science definition, architecture, design methods, and verification & validation; (2) software-defined space system technologies including reconfigurable hardware, multi-functionality, and modularity; (3) onboard intelligence including autonomous science, navigation, controls, and embodied AI; and (4) three PE 3.0 mission concepts: a Neptune/Triton smart flyby, an ocean world explorer, and an Oort cloud reconnaissance mission.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22851v1">EgoDyn-Bench: Evaluating Ego-Motion Understanding in Vision-Centric Foundation Models for Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-04-22
      | 💬 36 Pages, under review
    </div>
    <details class="paper-abstract">
      While Vision-Language Models (VLMs) have advanced highlevel reasoning in autonomous driving, their ability to ground this reasoning in the underlying physics of ego-motion remains poorly understood. We introduce EgoDyn-Bench, a diagnostic benchmark for evaluating the semantic ego-motion understanding of vision-centric foundation models. By mapping continuous vehicle kinematics to discrete motion concepts via a deterministic oracle, we decouple a model's internal physical logic from its visual perception. Our large-scale empirical audit spanning 20 + models, including closed-source MLLMs, open-source VLMs across multiple scales, and specialized VLAs, identifies a significant Perception Bottleneck: while models exhibit logical physical concepts, they consistently fail to accurately align them with visual observations, frequently underperforming classical non-learned geometric baselines. This failure persists across model scales and domain-specific training, indicating a structural deficit in how current architectures couple visual perception with physical reasoning. We demonstrate that providing explicit trajectory encodings substantially restores physical consistency across all evaluated models, revealing a functional disentanglement between vision and language: egomotion logic is derived almost exclusively from the language modality, while visual observations contribute negligible additional signal. This structural finding provides a standardized diagnostic framework and a practical pathway toward physically aligned embodied AI. Keywords: Ego-motion - Physical Reasoning - Foundation Models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08096v3">TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization</a></div>
    <div class="paper-meta">
      📅 2026-04-20
      | 💬 Tables updated with current results, typographical errors fixed
    </div>
    <details class="paper-abstract">
      Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically-plausible but geometrically-inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at https://cwru-aism.github.io/triangulang/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10813v3">InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17887v1">StableIDM: Stabilizing Inverse Dynamics Model against Manipulator Truncation via Spatio-Temporal Refinement</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Inverse Dynamics Models (IDMs) map visual observations to low-level action commands, serving as central components for data labeling and policy execution in embodied AI. However, their performance degrades severely under manipulator truncation, a common failure mode that makes state recovery ill-posed and leads to unstable control. We present StableIDM, a spatio-temporal framework that refines features from visual inputs to stabilize action predictions under such partial observability. StableIDM integrates three complementary components: (1) auxiliary robot-centric masking to suppress background clutter, (2) Directional Feature Aggregation (DFA) for geometry-aware spatial reasoning, which extracts anisotropic features along directions inferred from the visible arm and (3) Temporal Dynamics Refinement (TDR) to smooth and correct predictions via motion continuity. Extensive evaluations validate our approach: StableIDM improves strict action accuracy by 12.1% under severe truncation on the AgiBot benchmark, and increases average task success by 9.7% in real-robot replay. Moreover, it boosts end-to-end grasp success by 11.5% when decoding video-generated plans, and improves downstream VLA real-robot success by 17.6% when functioning as an automatic annotator. These results demonstrate that StableIDM provides a robust and scalable backbone for both policy execution and data generation in embodied artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17706v1">OmniVLA-RL: A Vision-Language-Action Model with Spatial Understanding and Online RL</a></div>
    <div class="paper-meta">
      📅 2026-04-20
    </div>
    <details class="paper-abstract">
      Visual-Language-Action (VLA) models represent a paradigm shift in embodied AI, yet existing frameworks often struggle with imprecise spatial perception, suboptimal multimodal fusion, and instability in reinforcement learning. To bridge these gaps, we propose OmniVLA-RL, a novel architecture that leverages a Mix-of-Transformers (MoT) design to synergistically integrate reasoning, spatial, and action experts. Furthermore, we introduce Flow-GSPO, which reformulates flow matching as a Stochastic Differential Equation (SDE) process and integrates it with Group Segmented Policy Optimization (GSPO) to enhance action precision and training robustness. Extensive evaluations on the LIBERO and LIBERO-Plus benchmarks demonstrate that OmniVLA-RL significantly outperforms state-of-the-art methods, effectively overcoming the fundamental limitations of current VLA models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17241v1">GaLa: Hypergraph-Guided Visual Language Models for Procedural Planning</a></div>
    <div class="paper-meta">
      📅 2026-04-19
      | 💬 14pages, 7figures
    </div>
    <details class="paper-abstract">
      Implicit spatial relations and deep semantic structures encoded in object attributes are crucial for procedural planning in embodied AI systems. However, existing approaches often over rely on the reasoning capabilities of vision language models (VLMs) themselves, while overlooking the rich structured semantic information that can be mined from multimodal inputs. As a result, models struggle to effectively understand functional spatial relationships in complex scenes. To fully exploit implicit spatial relations and deep semantic structures in multimodal data, we propose GaLa, a vision language framework for multimodal procedural planning. GaLa introduces a hypergraph-based representation, where object instances in the image are modeled as nodes, and region-level hyperedges are constructed by aggregating objects according to their attributes and functional semantics. This design explicitly captures implicit semantic relations among objects as well as the hierarchical organization of functional regions. Furthermore, we design a TriView HyperGraph Encoder that enforces semantic consistency across the node view, area view, and node area association view via contrastive learning, enabling hypergraph semantics to be more effectively injected into downstream VLM reasoning. Extensive experiments on the ActPlan1K and ALFRED benchmarks demonstrate that GaLa significantly outperforms existing methods in terms of execution success rate, LCS, and planning correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02008v2">Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2026-04-19
    </div>
    <details class="paper-abstract">
      Effective exploration in reinforcement learning requires not only tracking where an agent has been, but also understanding how the agent perceives and represents the world. To learn powerful representations, an agent should actively explore states that contribute to its knowledge of the environment. Temporal representations can capture the information necessary to solve a wide range of potential tasks while avoiding the computational cost associated with full state reconstruction. In this paper, we propose an exploration method that leverages temporal contrastive representations to guide exploration, prioritizing states with unpredictable future outcomes. We demonstrate that such representations can enable the learning of complex exploratory x in locomotion, manipulation, and embodied-AI tasks, revealing capabilities and behaviors that traditionally require extrinsic rewards. Unlike approaches that rely on explicit distance learning or episodic memory mechanisms (e.g., quasimetric-based methods), our method builds directly on temporal similarities, yielding a simpler yet effective strategy for exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03438v2">Multimodal Reinforcement Learning with Adaptive Verifier for AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-18
    </div>
    <details class="paper-abstract">
      Agentic reasoning models trained with multimodal reinforcement learning (MMRL) have become increasingly capable, yet they are almost universally optimized using sparse, outcome-based rewards computed based on the final answers. Richer rewards computed from the reasoning tokens can improve learning significantly by providing more fine-grained guidance. However, it is challenging to compute more informative rewards in MMRL beyond those based on outcomes since different samples may require different scoring functions and teacher models may provide noisy reward signals too. In this paper, we introduce the Argos (Agentic Reward for Grounded & Objective Scoring), a principled reward agent to train multimodal reasoning models for agentic tasks. For each sample, Argos selects from a pool of teacher-model derived and rule-based scoring functions to simultaneously evaluate: (i) final response accuracy, (ii) spatiotemporal localization of referred entities and actions, and (iii) the quality of the reasoning process. We find that by leveraging our agentic verifier across both SFT data curation and RL training, our model achieves state-of-the-art results across multiple agentic tasks such as spatial reasoning, visual hallucination as well as robotics and embodied AI benchmarks. Critically, we demonstrate that just relying on SFT post-training on highly curated reasoning data is insufficient, as agents invariably collapse to ungrounded solutions during RL without our online verification. We also show that our agentic verifier can help to reduce reward-hacking in MMRL. Finally, we also provide a theoretical justification for the effectiveness of Argos through the concept of pareto-optimality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17019v1">Mini-BEHAVIOR-Gran: Revealing U-Shaped Effects of Instruction Granularity on Language-Guided Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-18
      | 💬 23 pages, Keywords: Language Grounding, Language Granularity, Instruction Following Agent, Width-based Planning Research Area: Multimodality and Language Grounding to Vision, Robotics and Beyond Research Area Keywords: vision language navigation, multimodality, neurosymbolic approaches
    </div>
    <details class="paper-abstract">
      Instruction granularity is an important yet poorly controlled variable in language-guided embodied AI. Existing benchmarks typically pair each task with a single static instruction, making it difficult to study how agent behavior changes when the same task is described at different levels of detail. We introduce Mini-BEHAVIOR-Gran, a new benchmark for controlled studies of instruction granularity that extends Mini-BEHAVIOR with multiple instruction variants per task, ranging from high-level goal descriptions to step-by-step guidance. Using this benchmark, we compare four candidate metrics for cross-task granularity quantification: token count, entity count, action-verb count, and planning-width, and find that width correlates most consistently with agent performance. Using width to organize training and evaluation further reveals a non-monotonic U-shaped relationship between instruction granularity and performance, with peaks at both fine and coarse extremes. Further analysis suggests that the coarse-granularity performance rebound is associated with shallow grounding, where agents learn vision-dominant policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16993v1">Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification</a></div>
    <div class="paper-meta">
      📅 2026-04-18
    </div>
    <details class="paper-abstract">
      As embodied AI transitions to real-world deployment, the success of the Vision-and-Language Navigation (VLN) task tends to evolve from mere reachability to social compliance. However, current agents suffer from a "goal-driven trap", prioritizing physical geometry ("can I go?") over semantic rules ("may I go?"), frequently overlooking subtle regulatory constraints. To bridge this gap, we establish Rule-VLN, the first large-scale urban benchmark for rule-compliant navigation. Spanning a massive 29k-node environment, it injects 177 diverse regulatory categories into 8k constrained nodes across four curriculum levels, challenging agents with fine-grained visual and behavioral constraints. We further propose the Semantic Navigation Rectification Module (SNRM), a universal, zero-shot module designed to equip pre-trained agents with safety awareness. SNRM integrates a coarse-to-fine visual perception VLM framework with an epistemic mental map for dynamic detour planning. Experiments demonstrate that while Rule-VLN challenges state-of-the-art models, SNRM significantly restores navigation capabilities, reducing CVR by 19.26% and boosting TC by 5.97%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07412v2">TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains an open challenge. We introduce TwinOR, a real-to-sim infrastructure for constructing photorealistic and dynamic digital twins of ORs. The system reconstructs static geometry and continuously models human and equipment motion. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and facilitates future embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter-level accuracy while preserving dynamic interaction across surgical workflows. In our experiments, TwinOR synthesizes stereo and monocular RGB streams as well as depth observations for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 evaluated on TwinOR-synthesized data achieve performance within their reported accuracy ranges on real-world indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for emulating real-world perception and localization challenge. By establishing a perception-grounded real-to-sim pipeline, TwinOR enables the automatic construction of dynamic, photorealistic digital twins of ORs. As a safe and scalable environment for experimentation, TwinOR opens new opportunities for translating embodied intelligence from simulation to real-world clinical environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.14565v1">Model-Based Reinforcement Learning Exploits Passive Body Dynamics for High-Performance Biped Robot Locomotion</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Embodiment is a significant keyword in recent machine learning fields. This study focused on the passive nature of the body of a biped robot to generate walking and running locomotion using model-based deep reinforcement learning. We constructed two models in a simulator, one with passive elements (e.g., springs) and the other, which is similar to general humanoids, without passive elements. The training of the model with passive elements was highly affected by the attractor of the system. This lead that although the trajectories quickly converged to limit cycles, it took a long time to obtain large rewards. However, thanks to the attractor-driven learning, the acquired locomotion was robust and energy-efficient. The results revealed that robots with passive elements could efficiently acquire high-performance locomotion by utilizing stable limit cycles generated through dynamic interaction between the body and ground. This study demonstrates the importance of implementing passive properties in the body for future embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13596v2">VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Instance-level object segmentation across disparate egocentric and exocentric views is a fundamental challenge in visual understanding, critical for applications in embodied AI and remote collaboration. This task is exceptionally difficult due to severe changes in scale, perspective, and occlusion, which destabilize direct pixel-level matching. While recent geometry-aware models like VGGT provide a strong foundation for feature alignment, we find they often fail at dense prediction tasks due to significant pixel-level projection drift, even when their internal object-level attention remains consistent. To bridge this gap, we introduce VGGT-Segmentor (VGGT-S), a framework that unifies robust geometric modeling with pixel-accurate semantic segmentation. VGGT-S leverages VGGT's powerful cross-view feature representation and introduces a novel Union Segmentation Head. This head operates in three stages: mask prompt fusion, point-guided prediction, and iterative mask refinement, effectively translating high-level feature alignment into a precise segmentation mask. Furthermore, we propose a single-image self-supervised training strategy that eliminates the need for paired annotations and enables strong generalization. On the Ego-Exo4D benchmark, VGGT-S sets a new state-of-the-art, achieving 67.7% and 68.0% average IoU for Ego to Exo and Exo to Ego tasks, respectively, significantly outperforming prior methods. Notably, our correspondence-free pretrained model surpasses most fully-supervised baselines, demonstrating the effectiveness and scalability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.15495v1">GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology</a></div>
    <div class="paper-meta">
      📅 2026-04-16
    </div>
    <details class="paper-abstract">
      Navigating complex, densely packed environments like retail stores, warehouses, and hospitals poses a significant spatial grounding challenge for humans and embodied AI. In these spaces, dense visual features quickly become stale given the quasi-static nature of items, and long-tail semantic distributions challenge traditional computer vision. While Vision-Language Models (VLMs) help assistive systems navigate semantically-rich spaces, they still struggle with spatial grounding in cluttered environments. We present GIST (Grounded Intelligent Semantic Topology), a multimodal knowledge extraction pipeline that transforms a consumer-grade mobile point cloud into a semantically annotated navigation topology. Our architecture distills the scene into a 2D occupancy map, extracts its topological layout, and overlays a lightweight semantic layer via intelligent keyframe and semantic selection. We demonstrate the versatility of this structured spatial knowledge through critical downstream Human-AI interaction tasks: (1) an intent-driven Semantic Search engine that actively infers categorical alternatives and zones when exact matches fail; (2) a one-shot Semantic Localizer achieving a 1.04 m top-5 mean translation error; (3) a Zone Classification module that segments the walkable floor plan into high-level semantic regions; and (4) a Visually-Grounded Instruction Generator that synthesizes optimal paths into egocentric, landmark-rich natural language routing. In multi-criteria LLM evaluations, GIST outperforms sequence-based instruction generation baselines. Finally, an in-situ formative evaluation (N=5) yields an 80% navigation success rate relying solely on verbal cues, validating the system's capacity for universal design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21025v2">CaptionQA: Is Your Caption as Useful as the Image Itself?</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Image captions serve as efficient surrogates for visual content in multimodal systems such as retrieval, recommendation, and multi-step agentic inference pipelines. Yet current evaluation practices miss a fundamental question: Can captions stand-in for images in real downstream tasks? We propose a utility-based benchmark, CaptionQA, to evaluate model-generated captions, where caption quality is measured by how well it supports downstream tasks. CaptionQA is an extensible domain-dependent benchmark covering 4 domains--Natural, Document, E-commerce, and Embodied AI--each with fine-grained taxonomies (25 top-level and 69 subcategories) that identify useful information for domain-specific tasks. CaptionQA builds 33,027 densely annotated multiple-choice questions (50.3 per image on average) that explicitly require visual information to answer, providing a comprehensive probe of caption utility. In our evaluation protocol, an LLM answers these questions using captions alone, directly measuring whether captions preserve image-level utility and are utilizable by a downstream LLM. Evaluating state-of-the-art MLLMs reveals substantial gaps between the image and its caption utility. Notably, models nearly identical on traditional image-QA benchmarks lower by up to 32% in caption utility. We release CaptionQA along with an open-source pipeline for extension to new domains. The code is available at https://github.com/bronyayang/CaptionQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05991v2">3D Instruction Ambiguity Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at https://jiayuding031020.github.io/ambi3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13959v1">[Emerging Ideas] Artificial Tripartite Intelligence: A Bio-Inspired, Sensor-First Architecture for Physical AI</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      As AI moves from data centers to robots and wearables, scaling ever-larger models becomes insufficient. Physical AI operates under tight latency, energy, privacy, and reliability constraints, and its performance depends not only on model capacity but also on how signals are acquired through controllable sensors in dynamic environments. We present Artificial Tripartite Intelligence (ATI), a bio-inspired, sensor-first architectural contract for physical AI. ATI is tripartite at the systems level: a Brainstem (L1) provides reflexive safety and signal-integrity control, a Cerebellum (L2) performs continuous sensor calibration, and a Cerebral Inference Subsystem spanning L3/L4 supports routine skill selection and execution, coordination, and deep reasoning. This modular organization allows sensor control, adaptive sensing, edge-cloud execution, and foundation model reasoning to co-evolve within one closed-loop architecture, while keeping time-critical sensing and control on device and invoking higher-level inference only when needed. We instantiate ATI in a mobile camera prototype under dynamic lighting and motion. In our routed evaluation (L3-L4 split inference), compared to the default auto-exposure setting, ATI (L1/L2 adaptive sensing) improves end-to-end accuracy from 53.8% to 88% while reducing remote L4 invocations by 43.3%. These results show the value of co-designing sensing and inference for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13800v1">EmbodiedClaw: Conversational Workflow Execution for Embodied AI Development</a></div>
    <div class="paper-meta">
      📅 2026-04-15
      | 💬 13 pages, 7 figure
    </div>
    <details class="paper-abstract">
      Embodied AI research is increasingly moving beyond single-task, single-environment policy learning toward multi-task, multi-scene, and multi-model settings. This shift substantially increases the engineering overhead and development time required for stages such as evaluation environment construction, trajectory collection, model training, and evaluation. To address this challenge, we propose a new paradigm for embodied AI development in which users express goals and constraints through conversation, and the system automatically plans and executes the development workflow. We instantiate this paradigm with EmbodiedClaw, a conversational agent that turns high-frequency, high-cost embodied research activities, including environment creation and revision, benchmark transformation, trajectory synthesis, model evaluation, and asset expansion, into executable skills. Experiments on end-to-end workflow tasks, capability-specific evaluations, human researcher studies, and ablations show that EmbodiedClaw reduces manual engineering effort while improving executability, consistency, and reproducibility. These results suggest a shift from manual toolchains to conversationally executable workflows for embodied AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13633v1">ESCAPE: Episodic Spatial Memory and Adaptive Execution Policy for Long-Horizon Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-04-15
    </div>
    <details class="paper-abstract">
      Coordinating navigation and manipulation with robust performance is essential for embodied AI in complex indoor environments. However, as tasks extend over long horizons, existing methods often struggle due to catastrophic forgetting, spatial inconsistency, and rigid execution. To address these issues, we propose ESCAPE (Episodic Spatial Memory Coupled with an Adaptive Policy for Execution), operating through a tightly coupled perception-grounding-execution workflow. For robust perception, ESCAPE features a Spatio-Temporal Fusion Mapping module to autoregressively construct a depth-free, persistent 3D spatial memory, alongside a Memory-Driven Target Grounding module for precise interaction mask generation. To achieve flexible action, our Adaptive Execution Policy dynamically orchestrates proactive global navigation and reactive local manipulation to seize opportunistic targets. ESCAPE achieves state-of-the-art performance on the ALFRED benchmark, reaching 65.09% and 60.79% success rates in test seen and unseen environments with step-by-step instructions. By reducing redundant exploration, our ESCAPE attains substantial improvements in path-length-weighted metrics and maintains robust performance (61.24% / 56.04%) even without detailed guidance for long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.13151v1">Exploration and Exploitation Errors Are Measurable for Language Model Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Language Model (LM) agents are increasingly used in complex open-ended decision-making tasks, from AI coding to physical AI. A core requirement in these settings is the ability to both explore the problem space and exploit acquired knowledge effectively. However, systematically distinguishing and quantifying exploration and exploitation from observed actions without access to the agent's internal policy remains challenging. To address this, we design controllable environments inspired by practical embodied AI scenarios. Each environment consists of a partially observable 2D grid map and an unknown task Directed Acyclic Graph (DAG). The map generation can be programmatically adjusted to emphasize exploration or exploitation difficulty. To enable policy-agnostic evaluation, we design a metric to quantify exploration and exploitation errors from agent's actions. We evaluate a variety of frontier LM agents and find that even state-of-the-art models struggle on our task, with different models exhibiting distinct failure modes. We further observe that reasoning models solve the task more effectively and show both exploration and exploitation can be significantly improved through minimal harness engineering. We release our code \href{https://github.com/jjj-madison/measurable-explore-exploit}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10992v2">ArtiCAD: Articulated CAD Assembly Design via Multi-Agent Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      Parametric Computer-Aided Design (CAD) of articulated assemblies is essential for product development, yet generating these multi-part, movable models from high-level descriptions remains unexplored. To address this, we propose ArtiCAD, the first training-free multi-agent system capable of generating editable, articulated CAD assemblies directly from text or images. Our system divides this complex task among four specialized agents: Design, Generation, Assembly, and Review. One of our key insights is to predict assembly relationships during the initial design stage rather than the assembly stage. By utilizing a Connector that explicitly defines attachment points and joint parameters, ArtiCAD determines these relationships before geometry generation, effectively bypassing the limited spatial reasoning capabilities of current LLMs and VLMs. To further ensure high-quality outputs, we introduce validation steps in the generation and assembly stages, accompanied by a cross-stage rollback mechanism that accurately isolates and corrects design- and code-level errors. Additionally, a self-evolving experience store accumulates design knowledge to continuously improve performance on future tasks. Extensive evaluations on three datasets (ArtiCAD-Bench, CADPrompt, and ACD) validate the effectiveness of our approach. We further demonstrate the applicability of ArtiCAD in requirement-driven conceptual design, physical prototyping, and the generation of embodied AI training assets through URDF export.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12626v1">Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-04-14
      | 💬 Project page: https://zju3dv.github.io/habitat-gs/
    </div>
    <details class="paper-abstract">
      Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators rely on mesh-based rasterization with limited visual realism, and their support for dynamic human avatars, where available, is constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization, with mixed-domain training being the most effective strategy. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation. Finally, performance benchmarks validate the system's scalability across varying scene complexity and avatar counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10578v2">Rein3D: Reinforced 3D Indoor Scene Generation with Panoramic Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2026-04-14
    </div>
    <details class="paper-abstract">
      The growing demand for Embodied AI and VR applications has highlighted the need for synthesizing high-quality 3D indoor scenes from sparse inputs. However, existing approaches struggle to infer massive amounts of missing geometry in large unseen areas while maintaining global consistency, often producing locally plausible but globally inconsistent reconstructions. We present Rein3D, a framework that reconstructs full 360-degree indoor environments by coupling explicit 3D Gaussian Splatting (3DGS) with temporally coherent priors from video diffusion models. Our approach follows a "restore-and-refine" paradigm: we employ a radial exploration strategy to render imperfect panoramic videos along trajectories starting from the origin, effectively uncovering occluded regions from a coarse 3DGS initialization. These sequences are restored by a panoramic video-to-video diffusion model and further enhanced via video super-resolution to synthesize high-fidelity geometry and textures. Finally, these refined videos serve as pseudo-ground truths to update the global 3D Gaussian field. To support this task, we construct PanoV2V-15K, a dataset of over 15K paired clean and degraded panoramic videos for diffusion-based scene restoration. Experiments demonstrate that Rein3D produces photorealistic and globally consistent 3D scenes and significantly improves long-range camera exploration compared with existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14129v2">Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 16 pages, 14 figures, Published in evostar2026. Code: https://github.com/JedMuff/airevolve. Videos: https://www.youtube.com/watch?list=PL5oQiyJFx4qM9Hzs2asyoGbJo9TuO4sPS&v=playlist&feature=youtu.be
    </div>
    <details class="paper-abstract">
      Evolution and learning have historically been interrelated topics, and their interplay is attracting increased interest lately. The emerging new factor in this trend is morphological evolution, the evolution of physical forms within embodied AI systems such as robots. In this study, we investigate a system of hexacopter-type drones with evolvable morphologies and learnable controllers and make contributions to two fields. For aerial robotics, we demonstrate that the combination of evolution and learning can deliver non-conventional drones that significantly outperform the traditional hexacopter on several tasks that are more complex than previously considered in the literature. For the field of Evolutionary Computing, we introduce novel metrics and perform new analyses into the interaction of morphological evolution and learning, uncovering hitherto unidentified effects. Our analysis tools are domain-agnostic, making a methodological contribution towards building solid foundations for embodied AI systems that integrate evolution and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11585v1">GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 Accepted to the CVPR 2026 URVIS Workshop. Project page: https://geomprompt.github.io
    </div>
    <details class="paper-abstract">
      Multimodal perception systems for robotics and embodied AI often assume reliable RGB-D sensing, but in practice, depth is frequently missing, noisy, or corrupted. We thus present GeomPrompt, a lightweight cross-modal adaptation module that synthesizes a task-driven geometric prompt from RGB alone for the fourth channel of a frozen RGB-D semantic segmentation model, without depth supervision. We further introduce GeomPrompt-Recovery, an adaptation module that compensates for degraded depth by predicting the fourth channel correction relevant for the frozen segmenter. Both modules are trained solely with downstream segmentation supervision, enabling recovery of the geometric prior useful for segmentation, rather than estimating depth signals. On SUN RGB-D, GeomPrompt improves over RGB-only inference by +6.1 mIoU on DFormer and +3.0 mIoU on GeminiFusion, while remaining competitive with strong monocular depth estimators. For degraded depth, GeomPrompt-Recovery consistently improves robustness, yielding gains up to +3.6 mIoU under severe depth corruptions. GeomPrompt is also substantially more efficient than monocular depth baselines, reaching 7.8 ms latency versus 38.3 ms and 71.9 ms. These results suggest that task-driven geometric prompting is an efficient mechanism for cross-modal compensation under missing and degraded depth inputs in RGB-D perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11572v1">DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Vision-Language-Action models (VLAs) have demonstrated strong potential for embodied AI, yet their deployment on resource-limited robots remains challenging due to high memory and computational demands. While Post-Training Quantization (PTQ) provides an efficient solution, directly applying PTQ to VLAs often results in severe performance degradation during sequential control. We identify temporal error accumulation as a key factor, where quantization perturbations at the vision-language-to-action interface are progressively amplified, leading to kinematic drift in executed trajectories. To address this issue, we propose Drift-Aware Post-Training Quantization (DA-PTQ), which formulates quantization as a drift-aware optimization problem over sequential decision processes. DA-PTQ consists of two components: (1) Cross-Space Representation Compensation, which mitigates structured distortions between multimodal representations and action space to improve action consistency, and (2) Motion-Driven Mixed-Precision Allocation, which assigns bit-widths by minimizing trajectory-level motion errors. Extensive experiments show that DA-PTQ significantly reduces kinematic drift and achieves comparable performance to full-precision models under low-bit settings, enabling practical deployment of VLAs on resource-limited robotic platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12639v2">RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-13
    </div>
    <details class="paper-abstract">
      Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11174v1">EmbodiedGovBench: A Benchmark for Governance, Recovery, and Upgrade Safety in Embodied Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 34 pages, 7 tables. Code: https://github.com/s20sc/embodied-gov-bench
    </div>
    <details class="paper-abstract">
      Recent progress in embodied AI has produced a growing ecosystem of robot policies, foundation models, and modular runtimes. However, current evaluation remains dominated by task success metrics such as completion rate or manipulation accuracy. These metrics leave a critical gap: they do not measure whether embodied systems are governable -- whether they respect capability boundaries, enforce policies, recover safely, maintain audit trails, and respond to human oversight. We present EmbodiedGovBench, a benchmark for governance-oriented evaluation of embodied agent systems. Rather than asking only whether a robot can complete a task, EmbodiedGovBench evaluates whether the system remains controllable, policy-bounded, recoverable, auditable, and evolution-safe under realistic perturbations. The benchmark covers seven governance dimensions: unauthorized capability invocation, runtime drift robustness, recovery success, policy portability, version upgrade safety, human override responsiveness, and audit completeness. We define a benchmark structure spanning single-robot and fleet settings, with scenario templates, perturbation operators, governance metrics, and baseline evaluation protocols. We describe how the benchmark can be instantiated over embodied capability runtimes with modular interfaces and contract-aware upgrade workflows. Our analysis suggests that embodied governance should become a first-class evaluation target. EmbodiedGovBench provides the initial measurement framework for that shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11038v1">EgoFun3D: Modeling Interactive Objects from Egocentric Videos using Function Templates</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 Project website: https://3dlg-hcvc.github.io/EgoFun3D/
    </div>
    <details class="paper-abstract">
      We present EgoFun3D, a coordinated task formulation, dataset, and benchmark for modeling interactive 3D objects from egocentric videos. Interactive objects are of high interest for embodied AI but scarce, making modeling from readily available real-world videos valuable. Our task focuses on obtaining simulation-ready interactive 3D objects from egocentric video input. While prior work largely focuses on articulations, we capture general cross-part functional mappings (e.g., rotation of stove knob controls stove burner temperature) through function templates, a structured computational representation. Function templates enable precise evaluation and direct compilation into executable code across simulation platforms. To enable comprehensive benchmarking, we introduce a dataset of 271 egocentric videos featuring challenging real-world interactions with paired 3D geometry, segmentation over 2D and 3D, articulation and function template annotations. To tackle the task, we propose a 4-stage pipeline consisting of: 2D part segmentation, reconstruction, articulation estimation, and function template inference. Comprehensive benchmarking shows that the task is challenging for off-the-shelf methods, highlighting avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10962v1">ScoRe-Flow: Complete Distributional Control via Score-Based Reinforcement Learning for Flow Matching</a></div>
    <div class="paper-meta">
      📅 2026-04-13
      | 💬 20 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Flow Matching (FM) policies have emerged as an efficient backbone for robotic control, offering fast and expressive action generation that underpins recent large-scale embodied AI systems. However, FM policies trained via imitation learning inherit the limitations of demonstration data; surpassing suboptimal behaviors requires reinforcement learning (RL) fine-tuning. Recent methods convert deterministic flows into stochastic differential equations (SDEs) with learnable noise injection, enabling exploration and tractable likelihoods, but such noise-only control can compromise training efficiency when demonstrations already provide strong priors. We observe that modulating the drift via the score function, i.e., the gradient of log-density, steers exploration toward high-probability regions, improving stability. The score admits a closed-form expression from the velocity field, requiring no auxiliary networks. Based on this, we propose ScoRe-Flow, a score-based RL fine-tuning method that combines drift modulation with learned variance prediction to achieve decoupled control over the mean and variance of stochastic transitions. Experiments demonstrate that ScoRe-Flow achieves 2.4x faster convergence than flow-based SOTA on D4RL locomotion tasks and up to 5.4% higher success rates on Robomimic and Franka Kitchen manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.24329v2">GameplayQA: A Benchmarking Framework for Decision-Dense POV-Synced Multi-Video Understanding of 3D Virtual Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 Accepted to the Annual Meeting of the Association for Computational Linguistics (ACL 2026)
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are increasingly deployed as perceptual backbones for autonomous agents in 3D environments, from robotics to virtual worlds. These applications require agents to perceive rapid state changes, attribute actions to the correct entities, and reason about concurrent multi-agent behaviors from a first-person perspective, capabilities that existing benchmarks do not adequately evaluate. We introduce GameplayQA, a framework for evaluating agentic-centric perception and reasoning through video understanding. Specifically, we densely annotate multiplayer 3D gameplay videos at 1.22 labels/second, with time-synced, concurrent captions of states, actions, and events structured around a triadic system of Self, Other Agents, and the World, a natural decomposition for multi-agent environments. From these annotations, we refined 2.4K diagnostic QA pairs organized into three levels of cognitive complexity, accompanied by a structured distractor taxonomy that enables fine-grained analysis of where models hallucinate. Evaluation of frontier MLLMs reveals a substantial gap from human performance, with common failures in temporal and cross-video grounding, agent-role attribution, and handling the decision density of the game. We hope GameplayQA stimulates future research at the intersection of embodied AI, agentic perception, and world modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10789v1">ReplicateAnyScene: Zero-Shot Video-to-3D Composition via Textual-Visual-Spatial Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 Project Page: https://xiac20.github.io/ReplicateAnyScene/
    </div>
    <details class="paper-abstract">
      Humans exhibit an innate capacity to rapidly perceive and segment objects from video observations, and even mentally assemble them into structured 3D scenes. Replicating such capability, termed compositional 3D reconstruction, is pivotal for the advancement of Spatial Intelligence and Embodied AI. However, existing methods struggle to achieve practical deployment due to the insufficient integration of cross-modal information, leaving them dependent on manual object prompting, reliant on auxiliary visual inputs, and restricted to overly simplistic scenes by training biases. To address these limitations, we propose ReplicateAnyScene, a framework capable of fully automated and zero-shot transformation of casually captured videos into compositional 3D scenes. Specifically, our pipeline incorporates a five-stage cascade to extract and structurally align generic priors from vision foundation models across textual, visual, and spatial dimensions, grounding them into structured 3D representations and ensuring semantic coherence and physical plausibility of the constructed scenes. To facilitate a more comprehensive evaluation of this task, we further introduce the C3DR benchmark to assess reconstruction quality from diverse aspects. Extensive experiments demonstrate the superiority of our method over existing baselines in generating high-quality compositional 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10772v1">HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      3D layout generation and editing play a crucial role in Embodied AI and immersive VR interaction. However, manual creation requires tedious labor, while data-driven generation often lacks diversity. The emergence of large models introduces new possibilities for 3D scene synthesis. We present HOG-Layout that enables text-driven hierarchical scene generation, optimization and real-time scene editing with large language models (LLMs) and vision-language models (VLMs). HOG-Layout improves scene semantic consistency and plausibility through retrieval-augmented generation (RAG) technology, incorporates an optimization module to enhance physical consistency, and adopts a hierarchical representation to enhance inference and optimization, achieving real-time editing. Experimental results demonstrate that HOG-Layout produces more reasonable environments compared with existing baselines, while supporting fast and intuitive scene editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10573v1">Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2026-04-12
      | 💬 CVPR 2026
    </div>
    <details class="paper-abstract">
      Robust 3D representation learning forms the perceptual foundation of spatial intelligence, enabling downstream tasks in scene understanding and embodied AI. However, learning such representations directly from unposed multi-view images remains challenging. Recent self-supervised methods attempt to unify geometry, appearance, and semantics in a feed-forward manner, but they often suffer from weak geometry induction, limited appearance detail, and inconsistencies between geometry and semantics. We introduce UniSplat, a feed-forward framework designed to address these limitations through three complementary components. First, we propose a dual-masking strategy that strengthens geometry induction in the encoder. By masking both encoder and decoder tokens, and targeting decoder masks toward geometry-rich regions, the model is forced to infer structural information from incomplete visual cues, yielding geometry-aware representations even under unposed inputs. Second, we develop a coarse-to-fine Gaussian splatting strategy that reduces appearance-semantics inconsistencies by progressively refining the radiance field. Finally, to enforce geometric-semantic consistency, we introduce a pose-conditioned recalibration mechanism that interrelates the outputs of multiple heads by re-projecting predicted 3D point and semantic maps into the image plane using estimated camera parameters, and aligning them with corresponding RGB and semantic predictions to ensure cross-task consistency, thereby resolving geometry-semantic mismatches. Together, these components yield unified 3D representations that are robust to unposed, sparse-view inputs and generalize across diverse tasks, laying a perceptual foundation for spatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.10125v1">PhyMix: Towards Physically Consistent Single-Image 3D Indoor Scene Generation with Implicit--Explicit Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-11
    </div>
    <details class="paper-abstract">
      Existing single-image 3D indoor scene generators often produce results that look visually plausible but fail to obey real-world physics, limiting their reliability in robotics, embodied AI, and design. To examine this gap, we introduce a unified Physics Evaluator that measures four main aspects: geometric priors, contact, stability, and deployability, which are further decomposed into nine sub-constraints, establishing the first benchmark to measure physical consistency. Based on this evaluator, our analysis shows that state-of-the-art methods remain largely physics-unaware. To overcome this limitation, we further propose a framework that integrates feedback from the Physics Evaluator into both training and inference, enhancing the physical plausibility of generated scenes. Specifically, we propose PhyMix, which is composed of two complementary components: (i) implicit alignment via Scene-GRPO, a critic-free group-relative policy optimization that leverages the Physics Evaluator as a preference signal and biases sampling towards physically feasible layouts, and (ii) explicit refinement via a plug-and-play Test-Time Optimizer (TTO) that uses differentiable evaluator signals to correct residual violations during generation. Overall, our method unifies evaluation, reward shaping, and inference-time correction, producing 3D indoor scenes that are visually faithful and physically plausible. Extensive synthetic evaluations confirm state-of-the-art performance in both visual fidelity and physical plausibility, and extensive qualitative examples in stylized and real-world images further showcase the robustness of the method. We will release codes and models upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17925v3">Switch-JustDance: Benchmarking Whole Body Motion Tracking Controllers Using a Commercial Console Game</a></div>
    <div class="paper-meta">
      📅 2026-04-11
    </div>
    <details class="paper-abstract">
      Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.09415v1">PhysInOne: Visual Physics Learning and Reasoning in One Suite</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 CVPR 2026. Siyuan, Hejun, Hu, Jinxi, Dongsheng, Junwei, Yixiao, Jiayue, and Shiwei are co-first authors. Project page: https://vlar-group.github.io/PhysInOne.html
    </div>
    <details class="paper-abstract">
      We present PhysInOne, a large-scale synthetic dataset addressing the critical scarcity of physically-grounded training data for AI systems. Unlike existing datasets limited to merely hundreds or thousands of examples, PhysInOne provides 2 million videos across 153,810 dynamic 3D scenes, covering 71 basic physical phenomena in mechanics, optics, fluid dynamics, and magnetism. Distinct from previous works, our scenes feature multiobject interactions against complex backgrounds, with comprehensive ground-truth annotations including 3D geometry, semantics, dynamic motion, physical properties, and text descriptions. We demonstrate PhysInOne's efficacy across four emerging applications: physics-aware video generation, long-/short-term future frame prediction, physical property estimation, and motion transfer. Experiments show that fine-tuning foundation models on PhysInOne significantly enhances physical plausibility, while also exposing critical gaps in modeling complex physical dynamics and estimating intrinsic properties. As the largest dataset of its kind, orders of magnitude beyond prior works, PhysInOne establishes a new benchmark for advancing physics-grounded world models in generation, simulation, and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08125v2">PolySLGen: Online Multimodal Speaking-Listening Reaction Generation in Polyadic Interaction</a></div>
    <div class="paper-meta">
      📅 2026-04-10
    </div>
    <details class="paper-abstract">
      Human-like multimodal reaction generation is essential for natural group interactions between humans and embodied AI. However, existing approaches are limited to single-modality or speaking-only responses in dyadic interactions, making them unsuitable for realistic social scenarios. Many also overlook nonverbal cues and complex dynamics of polyadic interactions, both critical for engagement and conversational coherence. In this work, we present PolySLGen, an online framework for Polyadic multimodal Speaking and Listening reaction Generation. Given past conversation and motion from all participants, PolySLGen generates a future speaking or listening reaction for a target participant, including speech, body motion, and speaking state score. To model group interactions effectively, we propose a pose fusion module and a social cue encoder that jointly aggregate motion and social signals from the group. Extensive experiments, along with quantitative and qualitative evaluations, show that PolySLGen produces contextually appropriate and temporally coherent multi-modal reactions, outperforming several adapted and state-of-the-art baselines in motion quality, motion-speech alignment, speaking state prediction, and human-perceived realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08987v1">PilotBench: A Benchmark for General Aviation Agents with Safety Constraints</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 Accepted at the 2026 IEEE International Joint Conference on Neural Networks (IJCNN 2026). 6 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance toward embodied AI agents operating in physical environments, a fundamental question emerges: can models trained on text corpora reliably reason about complex physics while adhering to safety constraints? We address this through PilotBench, a benchmark evaluating LLMs on safety-critical flight trajectory and attitude prediction. Built from 708 real-world general aviation trajectories spanning nine operationally distinct flight phases with synchronized 34-channel telemetry, PilotBench systematically probes the intersection of semantic understanding and physics-governed prediction through comparative analysis of LLMs and traditional forecasters. We introduce Pilot-Score, a composite metric balancing 60% regression accuracy with 40% instruction adherence and safety compliance. Comparative evaluation across 41 models uncovers a Precision-Controllability Dichotomy: traditional forecasters achieve superior MAE of 7.01 but lack semantic reasoning capabilities, while LLMs gain controllability with 86--89% instruction-following at the cost of 11--14 MAE precision. Phase-stratified analysis further exposes a Dynamic Complexity Gap-LLM performance degrades sharply in high-workload phases such as Climb and Approach, suggesting brittle implicit physics models. These empirical discoveries motivate hybrid architectures combining LLMs' symbolic reasoning with specialized forecasters' numerical precision. PilotBench provides a rigorous foundation for advancing embodied AI in safety-constrained domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08983v1">AssemLM: Spatial Reasoning Multimodal Large Language Models for Robotic Assembly</a></div>
    <div class="paper-meta">
      📅 2026-04-10
      | 💬 Project Page: https://assemlmhome.github.io/
    </div>
    <details class="paper-abstract">
      Spatial reasoning is a fundamental capability for embodied intelligence, especially for fine-grained manipulation tasks such as robotic assembly. While recent vision-language models (VLMs) exhibit preliminary spatial awareness, they largely rely on coarse 2D perception and lack the ability to perform accurate reasoning over 3D geometry, which is crucial for precise assembly operations. To address this limitation, we propose AssemLM, a spatial multimodal large language model tailored for robotic assembly. AssemLM integrates assembly manuals, point clouds, and textual instructions to reason about and predict task-critical 6D assembly poses, enabling explicit geometric understanding throughout the assembly process. To effectively bridge raw 3D perception and high-level reasoning, we adopt a specialized point cloud encoder to capture fine-grained geometric and rotational features, which are then integrated into the multimodal language model to support accurate 3D spatial reasoning for assembly tasks. In addition, we construct AssemBench, a large-scale dataset and benchmark for assembly-oriented spatial reasoning, comprising over 900K multimodal samples with precise 6D pose annotations. AssemBench extends spatial reasoning evaluation beyond 2D and grounding tasks into full 3D geometric inference, filling a critical gap in existing embodied AI benchmarks. Extensive experiments demonstrate that AssemLM achieves state-of-the-art performance in 6D pose reasoning across diverse assembly scenarios. Furthermore, real-robot evaluations show that our model can support fine-grained and multi-step assembly execution in real-world settings, demonstrating its potential for robotic assembly applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08509v1">Visually-grounded Humanoid Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-09
      | 💬 Project page: https://alvinyh.github.io/VGHuman/
    </div>
    <details class="paper-abstract">
      Digital human generation has been studied for decades and supports a wide range of real-world applications. However, most existing systems are passively animated, relying on privileged state or scripted control, which limits scalability to novel environments. We instead ask: how can digital humans actively behave using only visual observations and specified goals in novel scenes? Achieving this would enable populating any 3D environments with digital humans at scale that exhibit spontaneous, natural, goal-directed behaviors. To this end, we introduce Visually-grounded Humanoid Agents, a coupled two-layer (world-agent) paradigm that replicates humans at multiple levels: they look, perceive, reason, and behave like real people in real-world 3D scenes. The World Layer reconstructs semantically rich 3D Gaussian scenes from real-world videos via an occlusion-aware pipeline and accommodates animatable Gaussian-based human avatars. The Agent Layer transforms these avatars into autonomous humanoid agents, equipping them with first-person RGB-D perception and enabling them to perform accurate, embodied planning with spatial awareness and iterative reasoning, which is then executed at the low level as full-body actions to drive their behaviors in the scene. We further introduce a benchmark to evaluate humanoid-scene interaction in diverse reconstructed environments. Experiments show our agents achieve robust autonomous behavior, yielding higher task success rates and fewer collisions than ablations and state-of-the-art planning methods. This work enables active digital human population and advances human-centric embodied AI. Data, code, and models will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.08125v1">PolySLGen: Online Multimodal Speaking-Listening Reaction Generation in Polyadic Interaction</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Human-like multimodal reaction generation is essential for natural group interactions between humans and embodied AI. However, existing approaches are limited to single-modality or speaking-only responses in dyadic interactions, making them unsuitable for realistic social scenarios. Many also overlook nonverbal cues and complex dynamics of polyadic interactions, both critical for engagement and conversational coherence. In this work, we present PolySLGen, an online framework for Polyadic multimodal Speaking and Listening reaction Generation. Given past conversation and motion from all participants, PolySLGen generates a future speaking or listening reaction for a target participant, including speech, body motion, and speaking state score. To model group interactions effectively, we propose a pose fusion module and a social cue encoder that jointly aggregate motion and social signals from the group. Extensive experiments, along with quantitative and qualitative evaluations, show that PolySLGen produces contextually appropriate and temporally coherent multi-modal reactions, outperforming several adapted and state-of-the-art baselines in motion quality, motion-speech alignment, speaking state prediction, and human-perceived realism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07986v1">DP-DeGauss: Dynamic Probabilistic Gaussian Decomposition for Egocentric 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Egocentric video is crucial for next-generation 4D scene reconstruction, with applications in AR/VR and embodied AI. However, reconstructing dynamic first-person scenes is challenging due to complex ego-motion, occlusions, and hand-object interactions. Existing decomposition methods are ill-suited, assuming fixed viewpoints or merging dynamics into a single foreground. To address these limitations, we introduce DP-DeGauss, a dynamic probabilistic Gaussian decomposition framework for egocentric 4D reconstruction. Our method initializes a unified 3D Gaussian set from COLMAP priors, augments each with a learnable category probability, and dynamically routes them into specialized deformation branches for background, hands, or object modeling. We employ category-specific masks for better disentanglement and introduce brightness and motion-flow control to improve static rendering and dynamic reconstruction. Extensive experiments show that DP-DeGauss outperforms baselines by +1.70dB in PSNR on average with SSIM and LPIPS gains. More importantly, our framework achieves the first and state-of-the-art disentanglement of background, hand, and object components, enabling explicit, fine-grained separation, paving the way for more intuitive ego scene understanding and editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07901v1">PanoSAM2: Lightweight Distortion- and Memory-aware Adaptions of SAM2 for 360 Video Object Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      360 video object segmentation (360VOS) aims to predict temporally-consistent masks in 360 videos, offering full-scene coverage, benefiting applications, such as VR/AR and embodied AI. Learning 360VOS model is nontrivial due to the lack of high-quality labeled dataset. Recently, Segment Anything Models (SAMs), especially SAM2 -- with its design of memory module -- shows strong, promptable VOS capability. However, directly using SAM2 for 360VOS yields implausible results as 360 videos suffer from the projection distortion, semantic inconsistency of left-right sides, and sparse object mask information in SAM2's memory. To this end, we propose PanoSAM2, a novel 360VOS framework based on our lightweight distortion- and memory-aware adaptation strategies of SAM2 to achieve reliable 360VOS while retaining SAM2's user-friendly prompting design. Concretely, to tackle the projection distortion and semantic inconsistency issues, we propose a Pano-Aware Decoder with seam-consistent receptive fields and iterative distortion refinement to maintain continuity across the 0/360 degree boundary. Meanwhile, a Distortion-Guided Mask Loss is introduced to weight pixels by distortion magnitude, stressing stretched regions and boundaries. To address the object sparsity issue, we propose a Long-Short Memory Module to maintain a compact long-term object pointer to re-instantiate and align short-term memories, thereby enhancing temporal coherence. Extensive experiments show that PanoSAM2 yields substantial gains over SAM2: +5.6 on 360VOTS and +6.7 on PanoVOS, showing the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07758v1">DailyArt: Discovering Articulation from Single Static Images via Latent Dynamics</a></div>
    <div class="paper-meta">
      📅 2026-04-09
    </div>
    <details class="paper-abstract">
      Articulated objects are essential for embodied AI and world models, yet inferring their kinematics from a single closed-state image remains challenging because crucial motion cues are often occluded. Existing methods either require multi-state observations or rely on explicit part priors, retrieval, or other auxiliary inputs that partially expose the structure to be inferred. In this work, we present DailyArt, which formulates articulated joint estimation from a single static image as a synthesis-mediated reasoning problem. Instead of directly regressing joints from a heavily occluded observation, DailyArt first synthesizes a maximally articulated opened state under the same camera view to expose articulation cues, and then estimates the full set of joint parameters from the discrepancy between the observed and synthesized states. Using a set-prediction formulation, DailyArt recovers all joints simultaneously without requiring object-specific templates, multi-view inputs, or explicit part annotations at test time. Taking estimated joints as conditions, the framework further supports part-level novel state synthesis as a downstream capability. Extensive experiments show that DailyArt achieves strong performance in articulated joint estimation and supports part-level novel state synthesis conditioned on joints. Project page is available at https://rangooo123.github.io/DaliyArt.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.07592v1">Spatio-Temporal Grounding of Large Language Models from Perception Streams</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Embodied-AI agents must reason about how objects move and interact in 3-D space over time, yet existing smaller frontier Large Language Models (LLMs) still mis-handle fine-grained spatial relations, metric distances, and temporal orderings. We introduce the general framework Formally Explainable Spatio-Temporal Scenes (FESTS) that injects verifiable spatio-temporal supervision into an LLM by compiling natural-language queries into Spatial Regular Expression (SpRE) -- a language combining regular expression syntax with S4u spatial logic and extended here with universal and existential quantification. The pipeline matches each SpRE against any structured video log and exports aligned (query, frames, match, explanation) tuples, enabling unlimited training data without manual labels. Training a 3-billion-parameter model on 27k such tuples boosts frame-level F1 from 48.5% to 87.5%, matching GPT-4.1 on complex spatio-temporal reasoning while remaining two orders of magnitude smaller, and, hence, enabling spatio-temporal intelligence for Video LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05171v3">Towards provable probabilistic safety for scalable embodied AI systems</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety-verifying system safety across all possible scenarios-remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. Instead, empirical safety evaluation is employed as an alternative, but the absence of provable guarantees imposes significant limitations. To address these issues, we argue for a paradigm shift to provable probabilistic safety that integrates provable guarantees with progressive achievement toward a probabilistic safety boundary on overall system performance. The new paradigm better leverages statistical methods to enhance feasibility and scalability, and a well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale. In this Perspective, we outline a roadmap for provable probabilistic safety, along with corresponding challenges and potential solutions. By bridging the gap between theoretical safety assurance and practical deployment, this Perspective offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.06722v1">Infrastructure First: Enabling Embodied AI for Science in the Global South</a></div>
    <div class="paper-meta">
      📅 2026-04-08
    </div>
    <details class="paper-abstract">
      Embodied AI for Science (EAI4S) brings intelligence into the laboratory by uniting perception, reasoning, and robotic action to autonomously run experiments in the physical world. For the Global South, this shift is not about adopting advanced automation for its own sake, but about overcoming a fundamental capacity constraint: too few hands to run too many experiments. By enabling continuous, reliable experimentation under limits of manpower, power, and connectivity, EAI4S turns automation from a luxury into essential scientific infrastructure. The main obstacle, however, is not algorithmic capability. It is infrastructure. Open-source AI and foundation models have narrowed the knowledge gap, but EAI4S depends on dependable edge compute, energy-efficient hardware, modular robotic systems, localized data pipelines, and open standards. Without these foundations, even the most capable models remain trapped in well-resourced laboratories. This article argues for an infrastructure-first approach to EAI4S and outlines the practical requirements for deploying embodied intelligence at scale, offering a concrete pathway for Global South institutions to translate AI advances into sustained scientific capacity and competitive research output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05749v1">Hazard Management in Robot-Assisted Mammography Support</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Robotic and embodied-AI systems have the potential to improve accessibility and quality of care in clinical settings, but their deployment in close physical contact with vulnerable patients introduces significant safety risks. This paper presents a hazard management methodology for MammoBot, an assistive robotic system designed to support patients during X-ray mammography. To ensure safety from early development stages, we combine stakeholder-guided process modelling with Software Hazard Analysis and Resolution in Design (SHARD) and System-Theoretic Process Analysis (STPA). The robot-assisted workflow is defined collaboratively with clinicians, roboticists, and patient representatives to capture key human-robot interactions. SHARD is applied to identify technical and procedural deviations, while STPA is used to analyse unsafe control actions arising from user interaction. The results show that many hazards arise not from component failures, but from timing mismatches, premature actions, and misinterpretation of system state. These hazards are translated into refined and additional safety requirements that constrain system behaviour and reduce reliance on correct human timing or interpretation alone. The work demonstrates a structured and traceable approach to safety-driven design with potential applicability to assistive robotic systems in clinical environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05673v1">Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 18 pages, 7 figures, 10 tables. Code available at https://github.com/WuyangLuan/RSBM
    </div>
    <details class="paper-abstract">
      Visual navigation is a core challenge in Embodied AI, requiring autonomous agents to translate high-dimensional sensory observations into continuous, long-horizon action trajectories. While generative policies based on diffusion models and Schrödinger Bridges (SB) effectively capture multimodal action distributions, they require dozens of integration steps due to high-variance stochastic transport, posing a critical barrier for real-time robotic control. We propose Rectified Schrödinger Bridge Matching (RSBM), a framework that exploits a shared velocity-field structure between standard Schrödinger Bridges ($\varepsilon=1$, maximum-entropy transport) and deterministic Optimal Transport ($\varepsilon\to 0$, as in Conditional Flow Matching), controlled by a single entropic regularization parameter $\varepsilon$. We prove two key results: (1) the conditional velocity field's functional form is invariant across the entire $\varepsilon$-spectrum (Velocity Structure Invariance), enabling a single network to serve all regularization strengths; and (2) reducing $\varepsilon$ linearly decreases the conditional velocity variance, enabling more stable coarse-step ODE integration. Anchored to a learned conditional prior that shortens transport distance, RSBM operates at an intermediate $\varepsilon$ that balances multimodal coverage and path straightness. Empirically, while standard bridges require $\geq 10$ steps to converge, RSBM achieves over 94% cosine similarity and 92% success rate in merely 3 integration steps -- without distillation or multi-stage training -- substantially narrowing the gap between high-fidelity generative policies and the low-latency demands of Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05595v1">Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have achieved remarkable success in robotic manipulation. However, their robustness to linguistic nuances remains a critical, under-explored safety concern, posing a significant safety risk to real-world deployment. Red teaming, or identifying environmental scenarios that elicit catastrophic behaviors, is an important step in ensuring the safe deployment of embodied AI agents. Reinforcement learning (RL) has emerged as a promising approach in automated red teaming that aims to uncover these vulnerabilities. However, standard RL-based adversaries often suffer from severe mode collapse due to their reward-maximizing nature, which tends to converge to a narrow set of trivial or repetitive failure patterns, failing to reveal the comprehensive landscape of meaningful risks. To bridge this gap, we propose a novel \textbf{D}iversity-\textbf{A}ware \textbf{E}mbodied \textbf{R}ed \textbf{T}eaming (\textbf{DAERT}) framework, to expose the vulnerabilities of VLAs against linguistic variations. Our design is based on evaluating a uniform policy, which is able to generate a diverse set of challenging instructions while ensuring its attack effectiveness, measured by execution failures in a physical simulator. We conduct extensive experiments across different robotic benchmarks against two state-of-the-art VLAs, including $π_0$ and OpenVLA. Our method consistently discovers a wider range of more effective adversarial instructions that reduce the average task success rate from 93.33\% to 5.85\%, demonstrating a scalable approach to stress-testing VLA agents and exposing critical safety blind spots before real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.05484v1">CoEnv: Driving Embodied Multi-Agent Collaboration via Compositional Environment</a></div>
    <div class="paper-meta">
      📅 2026-04-07
      | 💬 31 pages, 8 figures, including supplementary material. Project page: https://faceong.github.io/CoEnv/
    </div>
    <details class="paper-abstract">
      Multi-agent embodied systems hold promise for complex collaborative manipulation, yet face critical challenges in spatial coordination, temporal reasoning, and shared workspace awareness. Inspired by human collaboration where cognitive planning occurs separately from physical execution, we introduce the concept of compositional environment -- a synergistic integration of real-world and simulation components that enables multiple robotic agents to perceive intentions and operate within a unified decision-making space. Building on this concept, we present CoEnv, a framework that leverages simulation for safe strategy exploration while ensuring reliable real-world deployment. CoEnv operates through three stages: real-to-sim scene reconstruction that digitizes physical workspaces, VLM-driven action synthesis supporting both real-time planning with high-level interfaces and iterative planning with code-based trajectory generation, and validated sim-to-real transfer with collision detection for safe deployment. Extensive experiments on challenging multi-arm manipulation benchmarks demonstrate CoEnv's effectiveness in achieving high task success rates and execution efficiency, establishing a new paradigm for multi-agent embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21938v1">The Biggest Risk of Embodied AI is Governance Lag</a></div>
    <div class="paper-meta">
      📅 2026-04-07
    </div>
    <details class="paper-abstract">
      Embodied AI is widely discussed as a job-displacement problem. The deeper risk, however, is governance lag: the inability of public institutions to keep pace with how fast the technology spreads through the physical economy. As reusable robotic platforms are combined with increasingly general AI models, embodied AI may scale across manufacturing, logistics, care, and infrastructure faster than governance systems can observe, interpret, and respond. We argue that this lag appears in three connected forms: observational, institutional, and distributive. The central policy challenge, therefore, is not automation alone, but whether governance and compliance systems can adapt before disruption becomes entrenched.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.04843v1">InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Human-object-scene interactions (HOSI) generation has broad applications in embodied AI, simulation, and animation. Unlike human-object interaction (HOI) and human-scene interaction (HSI), HOSI generation requires reasoning over dynamic object-scene changes, yet suffers from limited annotated data. To address these issues, we propose a coarse-to-fine instruction-conditioned interaction generation framework that is explicitly aligned with the iterative denoising process of a consistency model. In particular, we adopt a dynamic perception strategy that leverages trajectories from the preceding refinement to update scene context and condition subsequent refinement at each denoising step of consistency model, yielding consistent interactions. To further reduce physical artifacts, we introduce a bump-aware guidance that mitigates collisions and penetrations during sampling without requiring fine-grained scene geometry, enabling real-time generation. To overcome data scarcity, we design a hybrid training startegy that synthesizes pseudo-HOSI samples by injecting voxelized scene occupancy into HOI datasets and jointly trains with high-fidelity HSI data, allowing interaction learning while preserving realistic scene awareness. Extensive experiments demonstrate that our method achieves state-of-the-art performance in both HOSI and HOI generation, and strong generalization to unseen scenes. Project page: https://yudezou.github.io/InfBaGel-page/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13998v2">Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-04-06
      | 💬 Embodied-R1 technical report v2; Published as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03890v1">From Prompt to Physical Action: Structured Backdoor Attacks on LLM-Mediated Robotic Control Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-04
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into robotic control pipelines enables natural language interfaces that translate user prompts into executable commands. However, this digital-to-physical interface introduces a critical and underexplored vulnerability: structured backdoor attacks embedded during fine-tuning. In this work, we experimentally investigate LoRA-based supply-chain backdoors in LLM-mediated ROS2 robotic control systems and evaluate their impact on physical robot execution. We construct two poisoned fine-tuning strategies targeting different stages of the command generation pipeline and reveal a key systems-level insight: back-doors embedded at the natural-language reasoning stage do not reliably propagate to executable control outputs, whereas backdoors aligned directly with structured JSON command formats successfully survive translation and trigger physical actions. In both simulation and real-world experiments, backdoored models achieve an average Attack Success Rate of 83% while maintaining over 93% Clean Performance Accuracy (CPA) and sub-second latency, demonstrating both reliability and stealth. We further implement an agentic verification defense using a secondary LLM for semantic consistency checking. Although this reduces the Attack Success Rate (ASR) to 20%, it increases end-to-end latency to 8-9 seconds, exposing a significant security-responsiveness trade-off in real-time robotic systems. These results highlight structural vulnerabilities in LLM-mediated robotic control architectures and underscore the need for robotics-aware defenses for embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08392v2">ST-BiBench: Benchmarking Multi-Stream Multimodal Coordination in Bimanual Embodied Tasks for MLLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-04
      | 💬 42 pages, 9 figures. Project page:https://stbibench.github.io/
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have significantly advanced the landscape of embodied AI, yet transitioning to synchronized bimanual coordination introduces formidable challenges in multi-stream multimodal integration. We introduce ST-BiBench, a comprehensive multi-tier framework for evaluating spatio-temporal multimodal coordination. Our approach centers on Strategic Coordination Planning, assessing high-level cross-modal reasoning over multiple action and perception streams. To investigate the "proximity paradox"-where semantically coherent plans fail to align with spatially grounded visual inputs-we incorporate Foundational Spatial Grounding to verify workspace awareness and arm-selection logic. Furthermore, we probe model frontiers through Fine-Grained Action Control, investigating whether MLLMs can directly synthesize high-dimensional continuous action modalities (16-Dim) from complex multimodal metadata. Evaluating 30+ state-of-the-art MLLMs, we uncover a persistent and pervasive "coordination paradox"-a significant gap between high-level strategic reasoning and fine-grained physical execution. Results reveal that while frontier MLLMs excel at logic-driven strategy, they frequently suffer from perception-logic disconnection and multi-stream interference during multimodal fusion. ST-BiBench provides a platform for identifying critical bottlenecks in multi-stream multimodal fusion and cross-modal alignment for complex embodied tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.03340v1">Learning Additively Compositional Latent Actions for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      Latent action learning infers pseudo-action labels from visual transitions, providing an approach to leverage internet-scale video for embodied AI. However, most methods learn latent actions without structural priors that encode the additive, compositional structure of physical motion. As a result, latents often entangle irrelevant scene details or information about future observations with true state changes and miscalibrate motion magnitude. We introduce Additively Compositional Latent Action Model (AC-LAM), which enforces scene-wise additive composition structure over short horizons on the latent action space. These AC constraints encourage simple algebraic structure in the latent action space~(identity, inverse, cycle consistency) and suppress information that does not compose additively. Empirically, AC-LAM learns more structured, motion-specific, and displacement-calibrated latent actions and provides stronger supervision for downstream policy learning, outperforming state-of-the-art LAMs across simulated and real-world tabletop tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15279v2">Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2026-04-03
    </div>
    <details class="paper-abstract">
      In embodied AI, visual perception should be active rather than passive: the system must decide where to look and at what scale to sense to acquire maximally informative data under pixel and spatial budget constraints. Existing vision models coupled with fixed RGB-D cameras fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. We study the task of language-guided active visual perception: given a single RGB image and a natural language instruction, the agent must output pan, tilt, and zoom adjustments of a real PTZ (pan-tilt-zoom) camera to acquire the most informative view for the specified task. We propose EyeVLA, a unified framework that addresses this task by integrating visual perception, language understanding, and physical camera control within a single autoregressive vision-language-action model. EyeVLA introduces a semantically rich and efficient hierarchical action encoding that compactly tokenizes continuous camera adjustments and embeds them into the VLM vocabulary for joint multimodal reasoning. Through a data-efficient pipeline comprising pseudo-label generation, iterative IoU-controlled data refinement, and reinforcement learning with Group Relative Policy Optimization (GRPO), we transfer the open-world understanding of a pre-trained VLM to an embodied active perception policy using only 500 real-world samples. Evaluations on 50 diverse real-world scenes across five independent evaluation runs demonstrate that EyeVLA achieves an average task completion rate of 96%. Our work establishes a new paradigm for instruction-driven active visual information acquisition in multimodal embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.22193v3">PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 Accepted to CVPR 2026 Code: https://github.com/GasaiYU/PAM
    </div>
    <details class="paper-abstract">
      Hand-object interaction (HOI) reconstruction and synthesis are becoming central to embodied AI and AR/VR. Yet, despite rapid progress, existing HOI generation research remains fragmented across three disjoint tracks: (1) pose-only synthesis that predicts MANO trajectories without producing pixels; (2) single-image HOI generation that hallucinates appearance from masks or 2D cues but lacks dynamics; and (3) video generation methods that require both the entire pose sequence and the ground-truth first frame as inputs, preventing true sim-to-real deployment. Inspired by the philosophy of Joo et al. (2018), we think that HOI generation requires a unified engine that brings together pose, appearance, and motion within one coherent framework. Thus we introduce PAM: a Pose-Appearance-Motion Engine for controllable HOI video generation. The performance of our engine is validated by: (1) On DexYCB, we obtain an FVD of 29.13 (vs. 38.83 for InterDyn), and MPJPE of 19.37 mm (vs. 30.05 mm for CosHand), while generating higher-resolution 480x720 videos compared to 256x256 and 256x384 baselines. (2) On OAKINK2, our full multi-condition model improves FVD from 68.76 to 46.31. (3) An ablation over input conditions on DexYCB shows that combining depth, segmentation, and keypoints consistently yields the best results. (4) For a downstream hand pose estimation task using SimpleHand, augmenting training with 3,400 synthetic videos (207k frames) allows a model trained on only 50% of the real data plus our synthetic data to match the 100% real baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.01720v1">Hi-LOAM: Hierarchical Implicit Neural Fields for LiDAR Odometry and Mapping</a></div>
    <div class="paper-meta">
      📅 2026-04-02
      | 💬 This manuscript is the accepted version of IEEE Transactions on Multimedia
    </div>
    <details class="paper-abstract">
      LiDAR Odometry and Mapping (LOAM) is a pivotal technique for embodied-AI applications such as autonomous driving and robot navigation. Most existing LOAM frameworks are either contingent on the supervision signal, or lack of the reconstruction fidelity, which are deficient in depicting details of large-scale complex scenes. To overcome these limitations, we propose a multi-scale implicit neural localization and mapping framework using LiDAR sensor, called Hi-LOAM. Hi-LOAM receives LiDAR point cloud as the input data modality, learns and stores hierarchical latent features in multiple levels of hash tables based on an octree structure, then these multi-scale latent features are decoded into signed distance value through shallow Multilayer Perceptrons (MLPs) in the mapping procedure. For pose estimation procedure, we rely on a correspondence-free, scan-to-implicit matching paradigm to estimate optimal pose and register current scan into the submap. The entire training process is conducted in a self-supervised manner, which waives the model pre-training and manifests its generalizability when applied to diverse environments. Extensive experiments on multiple real-world and synthetic datasets demonstrate the superior performance, in terms of the effectiveness and generalization capabilities, of our Hi-LOAM compared to existing state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23205v2">EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-02
    </div>
    <details class="paper-abstract">
      Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10394v2">RoboNeuron: A Middle-Layer Infrastructure for Agent-Driven Orchestration in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) models and LLM agents have advanced rapidly, yet reliable deployment on physical robots is often hindered by an interface mismatch between agent tool APIs and robot middleware. Current implementations typically rely on ad-hoc wrappers that are difficult to reuse, and changes to the VLA backend or serving stack often necessitate extensive re-integration. We introduce RoboNeuron, a middleware layer that connects the Model Context Protocol (MCP) for LLM agents with robot middleware such as ROS2. RoboNeuron bridges these ecosystems by deriving agent-callable tools directly from ROS schemas, providing a unified execution abstraction that supports both direct commands and modular composition, and localizing backend, runtime, and acceleration-preset changes within a stable inference boundary. We evaluate RoboNeuron in simulation and on hardware through multi-platform base control, arm motion, and VLA-based grasping tasks, demonstrating that it enables modular system orchestration under a unified interface while supporting backend transitions without system rewiring. The full code implementation of this work is available at github repo: https://github.com/guanweifan/RoboNeuron
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01184v2">Object Affordance Recognition and Grounding via Multi-scale Cross-modal Representation Learning</a></div>
    <div class="paper-meta">
      📅 2026-04-01
    </div>
    <details class="paper-abstract">
      A core problem of Embodied AI is to learn object manipulation from observation, as humans do. To achieve this, it is important to localize 3D object affordance areas through observation such as images (3D affordance grounding) and understand their functionalities (affordance classification). Previous attempts usually tackle these two tasks separately, leading to inconsistent predictions due to lacking proper modeling of their dependency. In addition, these methods typically only ground the incomplete affordance areas depicted in images, failing to predict the full potential affordance areas, and operate at a fixed scale, resulting in difficulty in coping with affordances significantly varying in scale with respect to the whole object. To address these issues, we propose a novel approach that learns an affordance-aware 3D representation and employs a stage-wise inference strategy leveraging the dependency between grounding and classification tasks. Specifically, we first develop a cross-modal 3D representation through efficient fusion and multi-scale geometric feature propagation, enabling inference of full potential affordance areas at a suitable regional scale. Moreover, we adopt a simple two-stage prediction mechanism, effectively coupling grounding and classification for better affordance understanding. Experiments demonstrate the effectiveness of our method, showing improved performance in both affordance grounding and classification.
    </details>
</div>
