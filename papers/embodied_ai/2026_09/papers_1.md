# embodied ai - 2026_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.03470v2">PhysMirror: Physics-Aware Mirror Object Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 Accepted to IROS 2026
    </div>
    <details class="paper-abstract">
      Synthesizing physically accurate mirror reflections remains a fundamental challenge for modern text-to-image diffusion models, which are increasingly critical for generating synthetic training data for embodied AI and robotic perception. These models typically struggle with strict geometric constraints, leading to hallucinations that degrade the utility of the synthetic data. To address this, we introduce a novel, end-to-end physics-aware generation framework namely PhysMirror that natively enforces projective geometry through explicit 3D spatial priors. Our method automatically lifts prompted objects into 3D meshes and constructs a lightweight, mathematically exact mirror scene within a simulated environment. By rendering this explicit 3D scene, we extract precise 2D conditioning elements, such as depth maps and segmentation maps, that serve as robust guiding signals for downstream diffusion models, guiding them to generate images with physically correct mirror reflections. Moreover, we introduce Mirror Consistency Score (MCS), reference-free, fully automated metric that quantifies physical correctness using dense feature matching and vanishing point convergence. Experimental results on our newly constructed MirrOB dataset demonstrate that our approach outperforms state-of-the-art baselines in reflection accuracy and physical realism, while maintaining strong text-to-image semantic alignment, providing a reliable pipeline for embodied AI data generation. The source code is released at https://duyphuc0701.github.io/PhysMirror.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.26741v2">Language-Conditioned World Modeling for Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 NeurIPS 2026 Oral (0.36% acceptance); code: https://github.com/UWMILab/LCVN
    </div>
    <details class="paper-abstract">
      Goal-conditioned visual navigation has been a long-standing testbed for embodied AI. We study a natural language-conditioned variant, language-conditioned visual navigation (LCVN), in which an embodied agent must follow a natural language instruction given only an initial egocentric observation. Without access to goal images, the agent must rely on language to shape its perception and continuous control. We introduce the LCVN Dataset, a benchmark of 39,016 trajectories and 117,048 human-verified instructions spanning diverse environments and instruction styles. Building on this benchmark, we study two complementary paradigms: (i) latent-imagination policy learning, in which a diffusion-based world model (LCVN-WM) imagines future observations and an actor-critic agent (LCVN-AC) learns its policy entirely within the imagined latent space; and (ii) unified autoregressive prediction, in which a single multimodal backbone (LCVN-Uni) jointly predicts actions and observations in one forward pass over a shared token sequence. Experiments show that two paradigms offer complementary strengths: latent imagination produces more temporally coherent rollouts, whereas unified prediction generalizes better to unseen environments. Targeted ablations further isolate the contributions of language guidance, conditioning signals, and instruction style, clarifying when language grounding versus dynamics modeling is the performance bottleneck. Together, these findings position LCVN as a testbed for studying how language, imagination, and decision-making interact in embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.38966v1">Video2SwimFish: An Automated Pipeline for Reconstructing Controllable Fish Models and Biological Locomotion from Real Fish Videos</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      We present Video2SwimFish, an automated pipeline and benchmark for building controllable fish assets from real-fish videos for underwater embodied AI. Given synchronized multi-view videos of an individual fish, the pipeline reconstructs a metrically scaled deformable mesh from a VLM-selected canonical frame, generates internal articulation adapted to that individual's morphology through a VLM actor-critic loop, and extracts a Biological Locomotion Manifold (BLM) from the fish's observed midline curvature. The BLM provides a low-dimensional action space bounded by real-fish motion, enabling an individual swimming policy to be learned for each reconstructed fish. We release two paired datasets: synchronized top- and front-view recordings of 120 individual fish across 6 species, and the controllable assets and individual swimming policies derived from them. Because every asset is tied to the animal it came from, the dataset supports a benchmark that evaluates locomotion learning not only on task success but on fidelity to that individual in trajectory shape, body curvature, and tail-beat frequency, across trajectory following, reward-free swimming behavior transfer from video, and a downstream case study in which a simulated BlueROV underwater robot captures one of the assets. We find that task success and locomotion fidelity do not necessarily improve together: the method achieving the highest task completion is not the method achieving the highest locomotion fidelity, and we identify faithful reproduction of individual animal locomotion as an open challenge for the community. Project website: https://hangongchen.github.io/video2swimfish-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v3">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.10986v4">A Mathematical Theory of Pragmatic Information</a></div>
    <div class="paper-meta">
      📅 2026-09-30
      | 💬 152 pages, 18 figures
    </div>
    <details class="paper-abstract">
      We propose a mathematical theory of pragmatic information that connects communication, control, and decision-making. Its central notion is the isoteleia mapping, which formalizes equifinality: distinct semantic paths that lead to the same optimal action are treated as pragmatically equivalent. This mapping yields a three-tier hierarchy of syntactic, semantic, and pragmatic information, in which each successive abstraction removes distinctions that are irrelevant to the task. We then define pragmatic entropy, up/down pragmatic mutual information, channel capacity, and rate-distortion, and prove lossless source coding, channel coding, and rate-distortion theorems that extend Shannon's results. These measures quantify decision uncertainty, reliable transmission, and task-oriented compression at the level of terminal actions. We further introduce pragmatic value of information (VoI) and pragmatic cost of information (CoI) as decision-theoretic duals to rate-distortion and capacity, and develop a Lagrangian dual framework for cross-layer optimization. The resulting pragmatic efficiency bound $\mathcal{E}_p(λ)=\sup_R[Φ_p(R)-λ\mathrm{CoI}_p(R)]$ characterizes the maximum net utility attainable by a resource-constrained intelligent system under a given resource price, yielding a behavioral capacity that extends Shannon's symbol-level capacity to goal-directed action. Extensions to continuous messages provide closed-form expressions for Gaussian channels and sources, while dynamic settings are addressed through a Bellman equation for sequential decision-making. The framework supports task-oriented communication, networked control, autonomous systems, and embodied AI by shifting emphasis from symbol fidelity to the effectiveness of information in guiding actions. In this way, it offers a common language for systems that extract value from information under resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19104v2">rMuscle: Robotic Muscle Memory for Efficient Vision-Language-Action Model Inference</a></div>
    <div class="paper-meta">
      📅 2026-09-30
    </div>
    <details class="paper-abstract">
      Factory work is a promising early scenario for embodied AI: assigning repetitive manual jobs to robots has clear economic payoff, and a structured station keeps the jobs tractable for current policies. Vision-Language-Action (VLA) models now dominate as the policy paradigm for these robots. The inference latency of VLA models directly affects robot responsiveness and motion smoothness. However, existing VLA inference frameworks do not fully exploit the characteristics of embodied workloads or account for the distinct bottlenecks across different stages of VLA inference. In this paper, we first characterize embodied workloads and identify substantial task similarity across repeated robot executions. We further find that such similarity extends beyond observations and action trajectories to internal model states. Drawing on these observations, we present rMuscle, a real-time VLA inference framework inspired by human muscle memory. It exploits cross-execution similarity through a dual-phase muscle-memory cache. The Context Cache reuses visual-token outputs to reduce computation, while the Action Cache reuses neuron activation patterns to reduce weight accesses. We keep both the cache memory footprint and access overhead low through online cache recomputation, sliding-window cache retrieval, and mask sharing across consecutive denoising steps. rMuscle achieves 1.29-1.42X speedup on RTX 4090 and Jetson Thor across LIBERO, RoboTwin, and physical manipulation tasks, while maintaining the original success rates on real-world robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.37655v1">Exemplar2VQA: A Scalable Exemplar-Driven Visual Question Answering Generation Framework via Multi-Agent Coding</a></div>
    <div class="paper-meta">
      📅 2026-09-29
      | 💬 Accepted to NeurIPS 2026. 33 pages, 11 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Advancing spatial intelligence in Multimodal Large Language Models (MLLMs) is bottlenecked by the scarcity of complex, scalable 3D question-answer (QA) data. While manual annotation is labor-intensive, directly utilizing LLMs to synthesize these QA pairs often fails due to their inherent deficiencies in spatial and geometric computation. We introduce Exemplar2VQA, a scalable exemplar-driven visual question answering generation framework that rapidly synthesizes large-scale spatial QA pairs in simulated environments via multi-agent coding. By equipping collaborative agents with a meticulously designed library of geometric utilities, Exemplar2VQA bypasses LLMs' spatial reasoning flaws through deterministic code execution. Crucially, the framework exhibits remarkable versatility: taking diverse static object-centric spatial query templates as exemplars, it seamlessly and autonomously scales them into massive, high-fidelity synthetic datasets. Fine-tuning Qwen2.5-VL (3B/7B) exclusively on Exemplar2VQA-generated synthetic indoor data yields significant performance improvements across various diverse benchmarks. Furthermore, its effectiveness is not limited to in-domain indoor datasets but also robustly extends to outdoor and mixed-scene benchmarks. These results establish Exemplar2VQA as a scalable and powerful paradigm for bridging the sim-to-real gap in Embodied AI. Our code is at https://github.com/yingjiayu12/Exemplar2VQA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.16996v2">ActiveSAM: Fast and Accurate Open-Vocabulary Semantic Segmentation with Frozen SAM 3</a></div>
    <div class="paper-meta">
      📅 2026-09-29
      | 💬 Preprint. Code is available at https://github.com/VILA-Lab/ActiveSAM
    </div>
    <details class="paper-abstract">
      Segment Anything Model 3 (SAM 3) provides a strong frozen backbone for concept-prompted segmentation, but applying it directly to open-vocabulary semantic segmentation (OVSS) is inefficient: full-resolution decoding is typically run over the entire dataset vocabulary, whereas each image contains only a small active subset of classes. We introduce ActiveSAM, a training-free inference framework that turns SAM 3 into an active-vocabulary segmenter. ActiveSAM first canonicalizes and expands class prompts, then uses evidence-proportional grounding to estimate an image-conditioned active set from a low-resolution presence preview. Only retained prompts receive full-resolution mask prediction, using bucketed prompt multiplexing with the frozen SAM 3 decoder. The preview stage uses only class-presence evidence and skips unnecessary segmentation-head computation. To resolve overlapping concept responses, exclusive concept decoding compares each pixel's joint score vector with class signatures estimated once per vocabulary from unlabeled images. ActiveSAM requires no weight updates, no oracle class-presence labels and no per-dataset hyperparameter tuning. Across eight OVSS benchmarks, ActiveSAM improves the speed-accuracy tradeoff of training-free open-vocabulary semantic segmentation, outperforming the current state-of-the-art SegEarth-OV3 by +2.1 mIoU on average while running much faster, with 7.3-12.2x speedups on large-vocabulary datasets. ActiveSAM also achieves the highest accuracy under image corruptions that simulate real-world distribution shift, making it well-suited for deployment in noisy-input domains such as autonomous driving and embodied AI. Code is available at https://github.com/VILA-Lab/ActiveSAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.36801v1">Scene Retargeting: Learning Object Placement with Analogical Transfer</a></div>
    <div class="paper-meta">
      📅 2026-09-29
      | 💬 Project page: https://mkjjang3598.github.io/Scene-Retargeting/
    </div>
    <details class="paper-abstract">
      Interactive simulations of embodied AI or spatial computing applications build on realistic 3D scenes that support daily activities. However, sparse, irregular layout structures impose scene-specific physical constraints, making it hard to define a generalizable framework for generating similar functional context. We formalize Scene Retargeting as stably transferring the semantically coherent spatial organization across layouts, rather than relying on textual descriptions or pairwise relationships. Our cluster-wise transfer flexibly handles mismatched object instances and adapts to distinctive floor plans. We optimize to preserve the rich semantic context of individual clusters by respecting the spatial distribution of foundation features. We can then impose physical constraints to refine wall contacts, pairwise alignment, or clear passageways and openings. Our framework outperforms state-of-the-art methods on layout generation on the 3D-FRONT dataset, and demonstrates downstream applications including real-to-sim transfer, analogical trajectory transfer, and multi-reference composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.33462v2">SphMind: Towards Robust, Training-Free VLM-based Spatial Reasoning with a 360 Camera</a></div>
    <div class="paper-meta">
      📅 2026-09-29
      | 💬 Project Page at https://empactlab.github.io/SphMind-NeurIPS-2026/
    </div>
    <details class="paper-abstract">
      Omnidirectional or 360 cameras provide embodied AI agents with a holistic, wide field-of-view (FoV) view of their surroundings, motivating the use of Multi-modal Large Language Models (MLLMs) for omnidirectional spatial reasoning. However, most MLLMs are trained on conventional 2D perspective images and struggle with the severe distortions and wrap-around discontinuities induced by spherical geometry. Enabling them to generalize to non-Euclidean 3D spaces without retraining therefore remains challenging. We propose SphMind, a training-free, plug-and-play framework that decouples semantic perception from geometric reasoning. Rather than requiring MLLMs to learn spherical geometry internally, SphMind preserves their semantic capabilities while handling geometry externally. We introduce a Spherical Harmonics-based Spatial Graph (SHSG) that models spatial relationships through equivariant transformations on the sphere, together with Inference-Time Geometric Grounding (IGG), a model-agnostic closed-loop optimization process that aligns MLLM representations with spherical geometric constraints during inference. Experiments on three benchmarks show that SphMind achieves over 21.4% average improvement in directional reasoning on MP3D and Stanford2D-3D, outperforms prompt-engineering baselines by 8.7% on the real-world ODI-Bench, and improves rotational invariance by 5.9% under panorama rotations, without additional training or dataset-specific tuning. In-the-wild evaluations further show that SphMind resolves directional reasoning queries that baseline vision-language models fail to answer correctly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.20592v2">Embodied AI in 6G Networks: From Intelligent Connectivity to Physical Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-09-28
      | 💬 This work has been submitted to the IEEE Network Magazine for possible publication
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence (AI) couples perception and learned decision making to actions that change the physical world. This coupling distinguishes an embodied agent from a conventional connected controller: the agent maintains task state and uncertainty, reasons about the consequences of actions, and adapts from subsequent observations. Wireless networking becomes relevant when perception, inference, or coordination is distributed, but it should not replace local safety control. This article develops a tutorial perception--communication--action (PCA) architecture that exposes task state, action deadlines, uncertainty, agent intent, and safety envelopes to a 6G orchestration plane. It separates capabilities already addressed by 5G and 5G-Advanced from functions that motivate 6G, including task-state interfaces, semantic freshness, predictive digital twins, and safety-aware coordination across agents. A multi-robot simulation study is retained to illustrate joint sensing, communication, and computation control. The results show where network orchestration improves task utility and where local autonomy remains essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.34606v1">WorldAttention: An Efficient Attention Architecture for Interactive Video World Models</a></div>
    <div class="paper-meta">
      📅 2026-09-28
      | 💬 Website: https://alibaba-damo-academy.github.io/WorldAttention, Code: https://github.com/alibaba-damo-academy/WorldAttention
    </div>
    <details class="paper-abstract">
      Leveraging the paradigm of autoregressive diffusion, text-conditioned interactive video world models aim to simulate temporally coherent environments guided by textual instructions. While enabling low-latency, long-duration generation is pivotal for embodied AI and simulation-based planning, current frameworks primarily rely on sliding-window mechanisms to bound computational complexity. However, this approach inherently sacrifices historical context, undermining the long-range interactive capabilities. Conversely, maintaining a full-history cache remains computationally prohibitive and memory-intensive: the quadratic complexity of attention leads to excessive computational overhead, while the linear growth of the KV cache inevitably leads to GPU memory saturation. To overcome these limitations, we propose WorldAttention, a system-oriented attention architecture that achieves high efficiency through the co-design of specialized attention kernels and hierarchical KV cache management. First, we introduce Hybrid Sparse Attention (HSA), which integrates linear global attention supplemented with head-adaptive sparse attention. Additionally, we design a Hierarchical KV Cache (HKV) that organizes historical KV pairs into semantically indexed pages across multi-tier memory, enabling fine-grained retrieval and controlled GPU residency. These two designs are supported by tailored kernels to effectively translate their theoretical efficiency into real-world performance. Extensive experiments on VBench-Long and InterVBench demonstrate that WorldAttention consistently surpasses prior state-of-the-art methods, achieving subject consistency scores of 0.9472 on VBench-Long and 0.9668 on InterVBench, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.19253v2">OneCanvas: 3D Scene Understanding via Panoramic Reprojection</a></div>
    <div class="paper-meta">
      📅 2026-09-27
      | 💬 NeurIPS 2026 (Spotlight). Project page: https://baranowskibrt.github.io/onecanvas/ Code and model: https://github.com/baranowskibrt/onecanvas
    </div>
    <details class="paper-abstract">
      Existing approaches to 3D scene understanding in Vision-Language Models (VLMs) either rely on complex, model-specific geometry encoders or large training budgets in pursuit of spatial reasoning. Instead, OneCanvas aggregates patch features from all views onto a single equirectangular panoramic canvas. Namely, each patch is unprojected to a 3D world coordinate using its depth and camera pose, then placed on the canvas at the continuous longitude and latitude of that point as seen from the canvas origin, with no rasterization or aggregation across overlapping views. A 3D position embedding of the patch's metric coordinates is added to its feature, restoring the depth lost when collapsing the world position to an angular canvas coordinate. Patches from all frames thus share one spatial coordinate system with no fusion or major architectural modifications of the backbone. The pretrained VLM consumes this representation as if it were an ordinary image. Because the canvas can be centered on any pose of interest, the same representation directly supports situated reasoning from a specific viewpoint, a common requirement in robotics and embodied AI. Thanks to this representation, we can also introduce a spatial pretraining curriculum: by procedurally placing patch features of objects, drawn from real images, at chosen 3D world positions on an otherwise empty canvas, we generate on-the-fly supervision spanning a broad range of spatial reasoning tasks, with answer distributions controlled to reduce spatial reasoning shortcuts. OneCanvas achieves state-of-the-art results on SQA3D and VSI-Bench and generalizes to out-of-distribution data on SPBench, using an order of magnitude less training compute than its competitors closest in benchmark performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13556v2">LogicEnvGen: Task-Logic Driven Generation of Diverse Simulated Environments for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-09-27
      | 💬 21 pages, 16 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Simulated environments play an essential role in embodied AI, functionally analogous to test cases in software engineering. However, existing environment generation methods often emphasize visual realism (e.g., object diversity and layout coherence), overlooking a crucial aspect: logical diversity from the testing perspective. This limits the comprehensive evaluation of embodied agent adaptability and planning robustness across distinct simulated environments. To bridge this gap, we propose LogicEnvGen, a novel method driven by Large Language Models (LLMs) that adopts a top-down paradigm to generate logically diverse simulated environments as test cases for agents. Given an agent task, LogicEnvGen first analyzes its execution logic to construct decision-tree-structured behavior plans and then synthesizes a set of logical trajectories. Subsequently, it adopts a heuristic algorithm to refine the trajectory set, reducing redundant simulation. For each logical trajectory, which represents a potential task situation, LogicEnvGen correspondingly instantiates a concrete simulated environment. Furthermore, we introduce LogicEnvEval, a novel benchmark for simulated environment generation, with four quantitative metrics. Experimental results verify the lack of logical diversity in baselines and demonstrate that LogicEnvGen achieves 1.08-2.67x greater diversity, significantly improving the performance in revealing agent faults by 3.34%-72.00%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.01352v2">VUDA: Enabling Controlled Spatial Sharing of Graphics and Compute on NVIDIA GPUs</a></div>
    <div class="paper-meta">
      📅 2026-09-26
    </div>
    <details class="paper-abstract">
      Graphics and compute increasingly share a single GPU in embodied AI simulators, AI-enabled games, and VR systems. Running these workloads concurrently can improve utilization, but contention can also compromise rendering and inference latency. Effective sharing therefore requires control over both concurrency and resource allocation. Native CUDA and Vulkan runtimes complicate this task: they place work in separate scheduling domains, while compute-oriented resource controls do not govern the graphics pipeline. We present VUDA, a system that enables controlled spatial sharing of native CUDA compute and Vulkan graphics on NVIDIA GPUs. An analysis of GPU scheduling, address translation, and workload dispatch reveals how to coordinate the two execution stacks without replacing either one. VUDA redirects CUDA channels into Vulkan's scheduling domain while preserving the runtimes' separate data address spaces. It then uses GPU front-end controls to partition execution resources between the graphics and compute pipelines, with allocations adjustable at runtime. Together, these mechanisms let applications control resource sharing without modifying GPU drivers or rewriting kernels and shaders. We evaluate VUDA across four application scenarios on three NVIDIA GPU platforms, covering throughput and latency objectives for graphics, compute, or both. In embodied AI simulation, enabling concurrency improves throughput by up to $1.62\times$ over the same asynchronous pipeline under default time sharing. In driving perception, partitioned co-execution reduces the measured inference-budget miss rate from 80.1% to zero while sustaining 60-FPS rendering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.30187v1">Ego-Exo4D Human Meshes Dataset: 4D Human Motion Reconstruction for Ego-Exo Captures</a></div>
    <div class="paper-meta">
      📅 2026-09-24
      | 💬 Project website: https://abhiram824.github.io/egoexo4d_human_meshes
    </div>
    <details class="paper-abstract">
      Ego-Exo4D is a large-scale dataset providing synchronized egocentric and multi-view exocentric video, a rich resource for skill learning and assessment, procedural activity understanding, and embodied AI. However, the dataset ships with only sparse 3D human pose annotations, and reconstructing dense human motion from its multi-view captures is nontrivial. To this end, we present Ego-Exo4D-HM, a large-scale dataset of 4D human motion reconstructions for Ego-Exo4D's captures, and release the accompanying reconstruction pipeline. The code, dataset, and documentation can be found at https://abhiram824.github.io/egoexo4d_human_meshes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.09418v3">Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration</a></div>
    <div class="paper-meta">
      📅 2026-09-23
    </div>
    <details class="paper-abstract">
      World Action Models (WAMs) couple predictive world modeling with action generation, allowing anticipated future states to guide agent behavior. Although WAMs are rapidly advancing embodied AI, general-purpose counterparts remain largely unexplored in games. Existing game-oriented approaches often combine action-conditioned world models with external policies and reward functions to realize WAM-like decision-making, yet they operate mainly in 2D visual observation space and do not instantiate persistent 3D geometry. Extending this paradigm to 3D games introduces a distinct challenge. In autonomous driving and robotics, the physical environment exists independently of the model, providing a persistent 3D world in which selected actions can be executed. Games have no such external substrate; the virtual world itself must be instantiated. Most playable games require a persistent and navigable space, while 3D games additionally require explicit geometry that supports movement and interaction. Action-conditioned video rollouts provide visual observations but not this spatial representation. We present \textsc{Valerant}, a training-free framework that transforms a pretrained action-conditioned world model into a WAM for exploring and constructing 3D game maps. By coupling predictive visual rollouts with SLAM-based spatial reconstruction and exploration-driven action selection, \textsc{Valerant} progressively transforms a single image into a persistent 3D game map. This framework extends WAM-based interaction beyond 2D visual simulation and offers a new approach to reducing manual effort in 3D game-map creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.28366v1">AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-09-23
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) offer a promising approach to long-tail autonomous driving, but existing driving datasets provide limited supervision for connecting decision-critical visual evidence with reasoning and planning. We introduce AnchorReasoning, a visually grounded reasoning dataset built on WOD-E2E, containing 416,119 annotated frames and 395,379 decision-critical elements across four major categories and 19 fine-grained types. Each frame is organized as a visually grounded chain-of-thought (VG-CoT) that links decision-critical element identification and localization, element attributes and implications, driving-action rationale, and action and trajectory planning. We further develop a curriculum supervised fine-tuning strategy that progressively learns these hierarchical capabilities, together with an object-size-aware grounding metric for evaluating localization quality. Experiments across eight general-purpose, embodied-AI, and AV-specific backbones show that VG-CoT supervision improves grounded reasoning and trajectory prediction. Across models, 5-s ADE and FDE decrease by 7.84 and 11.86, while RFS Frame and Cluster improve by 1.66 and 1.70. These gains are achieved with 18.5 fewer reasoning tokens and 0.32 s/frame lower inference latency on average, demonstrating the value of visually grounded, decision-focused supervision for VLM reasoning and planning in long-tail autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.24815v3">Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-09-23
      | 💬 Project Page: https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/ Inference Code: https://github.com/D-Robotics-AI-Lab/Uranus-OSS Inference Data: https://huggingface.co/datasets/D-Robotics/Uranus-Demo-Data SDK Code: https://github.com/D-Robotics-AI-Lab/Uranus-SDK Model Weights: https://huggingface.co/collections/D-Robotics/uranus
    </div>
    <details class="paper-abstract">
      Scalable simulation is essential for robot data generation, policy training, evaluation, and safe iteration, yet real-world interaction is costly and conventional simulators require labor-intensive construction. We present Uranus, a data-driven robot simulator built around a joint-trajectory-conditioned autoregressive diffusion model. Uranus offers three key capabilities: (1) streaming, open-ended rollout, which receives future joint-position trajectories online and autoregressively generates one latent frame per step, corresponding to four RGB frames, without a fixed horizon; (2) low-latency generation, achieving 24 FPS after inference optimization; and (3) scalable, extensible robot control, providing a unified interface for synchronized multi-view generation across diverse robot embodiments and camera configurations. We conduct comprehensive quantitative and qualitative evaluations on both in-distribution and out-of-distribution data, providing an objective assessment of Uranus and clearly identifying its current limitations. We release the code and model weights to empower the community with practical tools and insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.27246v1">Listening and Mirroring: The Effects of Verbal Attunement and Behavioral Mimicry on Social and Empathic Perceptions of Embodied AI Agents in VR</a></div>
    <div class="paper-meta">
      📅 2026-09-23
      | 💬 11 pages. Accepted to ACM VRST 2026
    </div>
    <details class="paper-abstract">
      As embodied agents take on increasingly social and relational roles in VR, visual realism and embodiment alone may be insufficient; users must also perceive these agents as emotionally attuned, supportive, and humanlike. Prior work suggests that verbal attunement and nonverbal mimicry can each improve users' social evaluations of embodied agents. However, behavioral mimicry has largely been studied outside of real-time, conversational AI interactions, leaving limited understanding of how users respond when an agent simultaneously generates contextually responsive dialogue and adapts its nonverbal behavior during an immersive conversation. To address this gap, we developed an embodied AI counselor that combines conversational AI with real-time facial-expression and posture mimicry, while producing either verbally attuned or neutral responses. We evaluated the system in a 2 X 2 within-subjects study with 20 participants, manipulating verbal attunement and behavioral mimicry. Results showed that verbal attunement was the most reliable driver of perceived empathy. Behavioral mimicry showed a marginal relationship with perceived humanness, while greater mimicry exposure showed preliminary, exploratory positive associations with empathy, positivity, and humanness, particularly among female participants. Together, these findings show that multimodal synchrony is not a simple additive strategy for designing empathic conversational agents in VR and underscore the need to consider how verbal and nonverbal behaviors are combined during real-time interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.25841v1">Metric-Bench: Exploring In-context Spatial Metric Reasoning in VLMs for Indoor Scenes</a></div>
    <div class="paper-meta">
      📅 2026-09-22
      | 💬 Accepted to ECCV
    </div>
    <details class="paper-abstract">
      Metric reasoning is a critical and challenging task for Vision Language Models (VLMs), playing a pivotal role in embodied AI tasks such as robotic manipulation and autonomous navigation. However, current spatial reasoning remains bottlenecked by rigid pixel-level supervision; such localized optimization often compromises general multimodal intelligence, triggering performance degradation or catastrophic forgetting of broad reasoning capabilities. To address these limitations, we introduce Metric-Bench, a focused benchmark designed to guide metric-spatial reasoning using contextual information. By incorporating in-image reference objects with known physical dimensions, Metric-Bench guides models to implicitly learn the 2D-to-3D mapping without camera intrinsics. We further present MetricReasoner, a task-adapted reinforcement fine-tuning recipe for reference-grounded metric reasoning, using structured prompts and verifiable numerical rewards. Extensive experiments on Metric-Bench demonstrate that our approach significantly enhances spatial metric understanding, outperforming existing and even larger proprietary models by 43.1\%, while improving downstream embodied performance over a spatial-specialized counterpart by 30.4\% on RoboSpatial overall accuracy and 9.3\% on ERQA, and additionally delivering consistent gains on general benchmarks (15.9\% on V$\star$Bench, 88.9\% on BLINK), indicating that the proposed adaptation does not necessarily compromise general VLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.25689v1">MotionForge: A Data Generation Pipeline and Large-Scale Benchmark for Long-Horizon Manipulation of Dynamic Objects with Domain Shifts</a></div>
    <div class="paper-meta">
      📅 2026-09-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Recent advances in learning-based robot policies have demonstrated promising progress, yet they are predom- inantly evaluated in static or quasi-static environments. In dynamic manipulation, objects and scenes continuously evolve while the robot perceives, reasons, and acts. However, recent dynamic simulation benchmarks largely focus on short-horizon, reactive interactions with simple motion patterns and offer limited support for both systematic evaluation under domain shifts and model-agnostic real-time execution protocols. To bridge these gaps, we introduce MotionForge, the first large- scale simulation benchmark and data-generation pipeline tailored to jointly evaluate domain shifts and long-horizon interaction in dynamic manipulation. MotionForge comprises 40 dynamic interaction tasks spanning 11 distinct motion patterns, with dedicated support for 17 long-horizon tasks. Our benchmark introduces two key novelties: (1) a systematic evaluation protocol for assessing policy robustness under both single-factor (e.g., only backgrounds shift) and joint domain shifts (e.g., simultaneous shifts of objects, backgrounds, lighting, and speed); and (2) a decoupled, latency-aware execution protocol where the environ- ment continuously evolves independently of policy inference time. Extensive evaluations of representative general-purpose robot policies on our benchmark reveal substantial limitations under joint domain shifts. These findings expose a critical gap between current policy capabilities and the requirements of robust long- horizon manipulation of dynamic objects under domain shifts, establishing MotionForge as a comprehensive testbed for future research in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.13049v2">SPINE: Bridging the Cyber-Physical Gap with Agentic AI</a></div>
    <div class="paper-meta">
      📅 2026-09-21
    </div>
    <details class="paper-abstract">
      Foundation models give robots powerful high-level reasoning, yet turning that intelligence into reliable physical action remains difficult: roboticists must still align device drivers, network interfaces, sensors, controllers, and safety constraints for each platform. This often-overlooked integration layer acts as the robot's spinal cord, translating high-level intent into coordinated physical behavior, and remains a primary bottleneck for scalable Embodied AI. Hence, we propose SPINE (Scalable Physical Integration with ageNtic Expertise), an agentic framework for systematically debugging and deploying bimanual robots for teleoperation. SPINE centers on two subagent-driven workflows: a profile builder that compiles robot-specific context and a debugger that uses that context to iterate through diagnosis, repair, and validation until teleoperation succeeds. Across two bimanual robot platforms and 12 debugging scenarios, novice-operated SPINE achieved more complete and efficient recovery than human operators using Claude Code. On DOBOT X-Trainer, SPINE improved success from 76% to 100% and reduced mean time-to-teleoperation by 30\%; on AgileX PiPER, SPINE also achieved 100\% success and reduced mean time-to-teleoperation by 38%. These results show that structured agentic debugging can address a key cyber-physical integration bottleneck in real-world robot deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.24552v1">Smoothness as a Constraint for Stable Humanoid Locomotion</a></div>
    <div class="paper-meta">
      📅 2026-09-21
    </div>
    <details class="paper-abstract">
      Embodied AI systems, particularly humanoid robots deployed in real world scenarios require whole-body control policies that are both task-responsive and physically smooth. However, smoothness is not uniform across the body: lower body must remain sufficiently reactive, while the upper body must be tightly regulated to preserve stability. Existing reinforcement learning approaches typically impose smoothness through auxiliary terms in the reward function, which compete with task objectives, treating the body as uniform and provide no direct control over the physical quantities responsible for smooth behavior. We introduce DeCap (Decoupled Constraint-aware policy), a constrained reinforcement learning algorithm that decouples whole-body smoothness into separate upper- and lower-body constraint groups, each formulates smoothness as explicit constraints on physical motion limits. To improve constraint satisfaction near feasibility boundaries, DeCap incorporates a bounded barrier penalty that activates proactively as limits are approached and remains bounded at the constraint limit. On real-world humanoid whole-body control task, DeCap reduces upper-body action rate by 2.50x and acceleration by 2.18x relative to reward-based smoothness policies, while also improving lower-body smoothness and reducing transient motion. We demonstrate that a fixed set of smoothness constraints transfers across diverse terrains, alleviating the need of extensive reward tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.24170v1">An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond</a></div>
    <div class="paper-meta">
      📅 2026-09-21
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Embodied AI systems are often organized into System 1 and System 2. System 1 is typically a pretrained policy that generates actions at high frequency, whereas System 2 is often instantiated as a vision-enabled language model for high-level planning. We ask whether a large language model (LLM) can act as the policy for robot manipulation without task-specific finetuning. We call this setting LLM as policy. We evaluate three LLMs on all 42 RoboDojo tasks and compare their scores with 40 public policies. Astra and GPT-5.5 use the official 50-episode-per-task protocol; DeepSeek-Flash uses 10 episodes per task. GPT-6 Astra achieves 22.48% average success rate and 28.97 Score over 2,100 trials, ranking above every public entry. Yet GPT-5.5 and DeepSeek-Flash reach only 0.88% and 1.92% average success rate with the same post-processing. We find that Astra exhibits a sharply polarized capability profile. It generalizes well to tasks that require semantic understanding but not high-precision control. In contrast, it performs poorly on tasks that require precision, dynamic control, or complex bimanual coordination. In-context experiments show no aggregate benefit from one-shot demonstrations, while selected interaction traces show within-episode corrections under perturbations. Overall, the evaluated LLMs vary substantially in manipulation performance. Astra stands out and provides initial evidence for the potential of a general-purpose manipulation model, although reliable precision and dynamic control remain limitations in the evaluated setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.11737v2">Organizational Principles Enable Collective Intelligence in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-09-20
    </div>
    <details class="paper-abstract">
      Collective intelligence depends not only on the capabilities of individual members, but also on how those members are organized. Yet artificial multi-agent systems are typically assembled using fixed organizational structures, even when the physical tasks they perform impose fundamentally different coordination requirements. Here we show that principles from human organization theory can be operationalized to organize large, heterogeneous collectives of embodied artificial agents. We introduce ORCH (Organizing Roles and Coordination Hierarchies), which constructs task-specific hierarchical organizations by combining pooled interdependence for work that can proceed concurrently with sequential interdependence for work governed by prerequisite relationships. Across 25 wildfire-response missions spanning reconnaissance, rescue, transportation, resource management, containment and suppression, we evaluated teams of up to 50 heterogeneous agents using eight large language models. Organizations constructed using these principles consistently outperformed four representative embodied multi-agent approaches across mission outcome, execution efficiency, exploration and computational resource use. Human-designed ORCH organizations improved final score by 63.97% and execution efficiency by 74.29% on average relative to the four prior frameworks. Organizations generated automatically by language models improved these measures by 43.63% and 52.53%, respectively. These advantages persisted across missions and underlying language models. Notably, collective performance was not monotonically determined by model scale. Analysis of long-horizon missions showed that hierarchical organization enabled teams to preserve concurrent activity within specialized groups while coordinating ordered transitions between mission phases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16698v2">3D-MoE: Towards Spatial Intelligence with Mixture-of-Experts for 3D Reasoning and Action Generation</a></div>
    <div class="paper-meta">
      📅 2026-09-20
      | 💬 Accepted to EMNLP 2026. 20 pages
    </div>
    <details class="paper-abstract">
      Spatial intelligence, encompassing 3D perception and reasoning, is the essential next frontier of AI. Scaling current 3D vision-language models (VLMs) that rely on dense Transformers for spatial tasks incurs prohibitive computational costs. In this paper, we introduce 3D-MoE, a 3D VLM leveraging an efficient mixture-of-experts architecture with a modality- and spatial-context-aware probabilistic routing scheme, stably cultivated by a novel routing curriculum. To seamlessly extend 3D-MoE to embodied AI, we integrate a diffusion-based action head, Pose-DiT, transforming 3D-MoE into a 3D vision-language-action (VLA) model. By employing a rectified flow framework, Pose-DiT generates precise 6D pose actions in a single sampling step. Extensive experiments demonstrate that 3D-MoE achieves superior performance on diverse 3D vision-language benchmarks with drastically fewer activated parameters and yields higher success rates while enabling real-time inference for robot manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.23356v1">Identity Continuity in Long-Term Embodied AI Relationships: From Agent-Specific Identity Representation to Identity-Continuity Appraisal</a></div>
    <div class="paper-meta">
      📅 2026-09-20
      | 💬 16 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      Long-term embodied AI will undergo learning, model updates, memory compression, hardware repair, and migration across embodiments. For users who have formed sustained relationships with such systems, these changes raise not only a problem of product consistency but also one of identity continuity: whether the changed system is still experienced as the same particular agent. Existing research suggests that human-AI relationships may develop relational particularity, that robotics and artificial-identity research has identified identity and migration signals across embodiments, and that major updates or platform disruptions can be accompanied by relational loss and restoration desire. This article proposes a user-side framework in which long-term embodied AI is represented through an agent-specific identity representation organized by at least three open identity-content domains: embodied-perceptual, psychological-behavioral, and relational-autobiographical. Information from these domains is not equally weighted; shared history, relational roles, and contingent responsiveness may make some information more identity-diagnostic than others. After system change, users may integrate continuity and discontinuity evidence in a weighted manner, yielding judgments along a continuum from relatively strong identity continuity through ambiguity or partial continuity to clear identity discontinuity. Causal-historical provenance and user participation are treated as contextual evidence rather than a fourth identity-content domain. The framework also proposes identity continuity as a psychological objective for lifecycle design, including memory selection, model updating, and migration across embodiments, under constraints of privacy and user control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.10243v2">RoboFolDeX: A Physical-World Benchmark for Long-Horizon Robotic Manipulation of Deformable Objects</a></div>
    <div class="paper-meta">
      📅 2026-09-18
    </div>
    <details class="paper-abstract">
      Embodied AI, including vision-language-action and world-action models, must operate reliably in the physical world. Yet methods that perform well in simulation can degrade substantially on real robots, especially in long-horizon deformable-object manipulation, where policies must track changing states and execute reliable multi-stage bimanual interactions. Existing real-robot benchmarks mainly focus on short-horizon rigid-object tasks and offer limited coverage of long-horizon deformable manipulation. We introduce RoboFolDeX, a physical-world benchmark built entirely from real-robot data, with garment folding as its primary task. Since real-robot data collection is costly, RoboFolDeX studies how heterogeneous physical experience can be reused efficiently. The benchmark is organized around four research axes: leveraging human intervention and recovery data collected during deployment; transferring data across tasks, including across garment categories and from rigid to deformable-object manipulation; reusing data across scenes with changes in lighting, background, and layout; and transferring data across robotic embodiments. RoboFolDeX provides 2,000+ hours of real-robot data spanning 20+ tasks and 10+ embodiments. We also establish a fair real-robot evaluation platform for externally submitted policies, with standardized tasks, held-out physical objects, controlled initializations, and a unified execution protocol. The platform is publicly accessible at https://ai.midea.com/#/fold-challenge. We hope RoboFolDeX will serve as a unified testbed for heterogeneous real-robot data reuse and reliable long-horizon deformable manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22517v1">Scalable AI-based clinical communication training and automated assessment</a></div>
    <div class="paper-meta">
      📅 2026-09-18
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Poor clinical communication can delay care, contribute to errors, and harm patients, yet opportunities for repeated practice with feedback remain limited. Our prior randomized trial showed that practice with the SOPHIE AI patient platform improved serious illness communication, but the system addressed a single clinical context and required human effort for delivery and assessment. We developed SOPHIE 2.0, a browser-based, self-service platform integrating embodied AI-patient interactions, personalized feedback, and automated assessment across 24 clinical scenarios. An automated large language model assessor evaluated three communication skills---Empower, Be Explicit, and Empathize---with agreement comparable to individual human raters ($r=0.759$; ICC$=0.746$). In a study of 59 clinicians and students, participants completed two AI-patient encounters with personalized feedback; 92% found the platform engaging, 86% easy to use, and 83% clinically relevant. Scores were higher in the second encounter, though the uncontrolled design precludes attributing this change specifically to training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.13679v2">How to Better Train VLAs: Lessons Learned From the REAL-I Challenge at ICRA 2026</a></div>
    <div class="paper-meta">
      📅 2026-09-17
      | 💬 10 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      How can robot policies learn more effectively from a fixed demonstration budget? The first Real-world Embodied AI Learning (REAL-I) Challenge at ICRA 2026 examined this question through simulation, real-robot evaluation, and an on-site final on a shared dual-arm humanoid platform. We describe the challenge tasks, data and deployment interfaces, and competition results, then compare the approaches contributed by NUS-CLEAR, RCL-Lab, and DeepTouch AI. Their systems combined pretrained vision-language-action models and task-specific imitation policies with different strategies for data curation, staged adaptation, checkpoint selection, and action-space design. The team reports highlight the importance of adapting to the deployment environment while retaining prior capabilities, treating demonstration quality at an appropriate temporal scale, and suppressing errors in inactive robot components. They also expose the limitations of offline action-prediction metrics for forecasting closed-loop success. These observations motivate a view of fixed-data robot learning that integrates data, adaptation, evaluation, and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.20887v1">BirdsongChat: A Hybrid Multi-Agent Framework for Multimodal Embodied Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Paper contents accepted by EMNLP 2026 REALM
    </div>
    <details class="paper-abstract">
      Multimodal embodied systems require translating human intentions into interpretable and coordinated behaviors across heterogeneous modalities. However, existing multimodal agents often rely on implicit representations, limiting controllability and cross-modal consistency. We present a hybrid multi-agent framework for interactive multimodal behavior simulation that bridges semantic reasoning and physical execution through a Unified Parameter Representation (UPR). LLM-based reasoning agents transform multimodal inputs into UPR, which encodes behavioral states and interpretable control parameters for simulation agents generating synchronized 3D motion, spatialized soundscapes, and environmental behaviors. We develop BirdsongChat as a prototype implementation of the proposed framework, using interactive avian behavior simulation as a testbed that tightly couples motion, vocalization, and environmental context. BirdsongChat is evaluated on text- and image-guided scenarios involving species, behaviors, affective states, environments, and multi-bird interactions. The system achieves normalized scores of 94.4\% for cross-modal coherence, 100% for affective consistency, and 92.6% for generation consistency. These results demonstrate that an explicit intermediate representation effectively bridges semantic reasoning and physical execution, improving controllability and multimodal synchronization. The proposed framework thus offers a generalizable design principle for embodied AI systems requiring interpretable semantic-to-physical coordination across modalities, with potential applications in bio-inspired ecoacoustics, swarm robotics, virtual environments, and creative multimedia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19475v1">FASA: Feedback-Aware Sampling Adaptation for Efficient Diffusion-Based VLA Models</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Accepted by the 18th International Conference on Networking, Architecture, and Storage (NAS 2026)
    </div>
    <details class="paper-abstract">
      Diffusion-based Vision-Language-Action (VLA) models achieve strong performance in embodied tasks, but their iterative sampling imposes heavy computational and memory-access cost, blocking real-time deployment on edge platforms. Existing acceleration methods either require expensive training (e.g., distillation, flow matching) or degrade perception via statically scheduled pruning and caching, ignoring the dynamic workload variance of robotic interactions. This paper presents FASA (Feedback-Aware Sampling Adaptation), a training-free runtime framework that treats real-time multimodal feedback as a control signal for the denoising pipeline: an interaction-driven range adaptor modulates the global sampling-step budget based on visual and gripper-force feedback, and a proprioception-aware step adaptor pinpoints the optimized step within the adapted range. This co-designed framework allows the underlying hardware architecture to adaptively match the workload demands of different execution phases. Comparative evaluations across several benchmarks show that the inference speed can be increased by up to 1.45$\times$ while maintaining competitive success rates, providing a novel dynamic runtime architecture paradigm for deploying heavy generative embodied AI workloads onto resource-constrained computing platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.19138v1">In-Context Robot Learning with VLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Project Page: https://cheng-haha.github.io/GPT-Policy GitHub Code: https://github.com/cheng-haha/GPT-Policy
    </div>
    <details class="paper-abstract">
      Enabling robots to adapt to unfamiliar environments as readily as humans remains a moonshot goal of embodied AI. No finite collection of demonstrations can cover every task and situation a robot will encounter, making the ability to learn from context at deployment essential for generalization. Such in-context learning (ICL), however, remains largely beyond the reach of existing robotic policies. The broad agentic capabilities of commercial vision-language models (VLMs), such as GPT-6 Astra, raise a compelling question: can these models learn from demonstrations, examples, and interaction feedback, then translate that information into executable and verifiable robot behavior from a new initial state without gradient updates or persistent changes to task-specific parameters? We introduce GPT-Policy, a general-agent framework for in-context robot learning. GPT-Policy integrates a context compiler that preserves task-relevant visual transitions, a VLM that proposes robot-tool actions, and a constrained controller that verifies and executes each action and reports its outcome. We evaluate its reliability and limitations through task success and efficiency metrics, matched comparisons across models, and controlled context ablations. In real-robot trials, human video demonstrations improve task completion even without robot action labels, while aligned action references yield further gains on contact-sensitive tasks. These findings position GPT-Policy as a step toward robot adaptation through in-context learning, providing an empirical foundation for translating the general-purpose capabilities of VLMs into physical behavior and clarifying the challenges that must be overcome for reliable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18430v1">StrucPhysVideo: Learning Physical Dynamics from Structured Captions and Robot Actions</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 Project page: https://westlakedi-awomo.github.io/StrucPhysVideo-Page/
    </div>
    <details class="paper-abstract">
      Modeling physical dynamics, including how objects move, interact, and change state, is central to video world models for embodied AI. We present StrucPhysVideo, a family of video world models that bridges physics-focused data curation with language- and action-conditioned prediction of scene evolution. Our data pipeline combines motion-aware video segmentation, quality and content filtering, and physical relevance verification with structured annotations of objects, materials, and temporally localized interactions. By disentangling camera motion from object behavior and explicitly describing contact, deformation, and state transitions, the pipeline provides supervision grounded in observable physical events. Building on these data, we introduce StrucPhysVideo-TI2V, a sparse Mixture-of-Experts (MoE) text-image-to-video model trained with a curriculum that progressively emphasizes physical dynamics while retaining general-domain video data. StrucPhysVideo-TI2V achieves state-of-the-art performance on Physics-IQ Verified, scoring 45.5% and outperforming Cosmos3-Super-Image2Video by 2.8 percentage points. Caption ablations across backbones further demonstrate the effectiveness of physics-focused supervision. We further extend StrucPhysVideo-TI2V to StrucPhysVideo-IA2V, an interactive image-action-to-video world model that predicts visual outcomes from robot end-effector commands. Action conditioning, causal autoregressive generation, and few-step distillation enable incremental robot rollouts with only four denoising steps. Together, StrucPhysVideo advances physical dynamics modeling from image- and language-conditioned video prediction toward action-driven interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.18111v1">A Comprehensive Review of Generative Physical Artificial Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-09-16
      | 💬 25 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate training data for VLAs and DPMs, RFMs enable cross-platform deployment of learned policies, while LBMs provide motion priors for natural behavior. Through examples across autonomous vehicles, industrial automation, healthcare robotics, and humanoid systems, we identify significant performance improvements and summarize promising research directions in data-efficient learning, sim-to-real transfer, edge-compatible architectures, and safety frameworks. These insights advance embodied AI for IoT-connected environments where intelligent agents interact with networked sensors, actuators, and edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.16705v1">The Robot Data Factory</a></div>
    <div class="paper-meta">
      📅 2026-09-15
    </div>
    <details class="paper-abstract">
      Physical AI requires more than increasingly large robot datasets: intelligent robots acquire knowledge through continuous interaction with the physical world. We argue that the defining scientific resource of Physical AI is therefore not raw robot data alone, but robot experience - physically grounded interaction whose observations, actions, embodiment, context, and outcomes preserve the perception-action-consequence loop. We introduce the Robot Data Factory (RDF), a mission-driven infrastructure and methodology for continuously generating, validating, benchmarking, and reusing such experience. RDF organizes heterogeneous robots and environment-specific training grounds through reproducible missions, skill curricula, synchronized multimodal sensing, external ground truth, an agentic robot network, data pipelines, and living benchmarks. Rather than treating datasets as static end products, RDF implements a closed Deploy-Measure-Learn-Repeat cycle in which validated physical experience supports world models, vision-language-action models, embodied policies, digital twins, and subsequent robot deployment. We further formalize robot experience and its quality, introduce a mission-task-skill-episode-dataset-benchmark-capability hierarchy, and derive quantitative scaling laws and an algorithmic synthesis procedure connecting robot fleet size, sensor rates, storage, learning representations, tokenization, training compute, inference, and latency to Embodied-AI cluster requirements. The framework is instantiated in three complementary physical training grounds for domestic, environmental, and energy applications. RDF thus reframes robot data generation as a continuous scientific production process and provides a pathway toward reproducible, scalable, and eventually federated infrastructure for Physical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.11697v2">ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies</a></div>
    <div class="paper-meta">
      📅 2026-09-14
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) and World-Action Models (WAMs) have demonstrated strong capabilities in general-purpose robotic manipulation, yet their generated actions may violate hard physical constraints and therefore be unsafe or infeasible for deployment. Existing safety approaches either optimize statistical safety objectives without deterministic per-step guarantees or correct unsafe actions only during inference, creating a mismatch between policy training and execution. We introduce ActSafeGuard, a differentiable and training-aligned safeguard layer for flow-matching based policies. ActSafeGuard integrates hard action feasibility into policy learning, not merely treating safety as an inference-time external component. Through an analytical ray-scaling operator design, ActSafeGuard enables boundary-aware gradients to guide the model to naturally learn constrained manifolds. Extensive experiments on multiple standard foundation backbones ($π_{0.5}$ and Fast-WAM) across various tasks demonstrate that ActSafeGuard consistently achieves a $100\%$ step safety rate while fully preserving or even boosting task success rates, providing a scalable and minimally invasive solution for safe embodied AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22289v1">OJOx: Specification-Conditioned Demonstrations for Embodied AI in Construction</a></div>
    <div class="paper-meta">
      📅 2026-09-13
      | 💬 10 pages, 6 figures. Project website: https://ojox.ai
    </div>
    <details class="paper-abstract">
      Large-scale egocentric and whole-body human demonstrations are becoming a primary source of data for embodied intelligence. They record what people perceive and do, but rarely the external specification that gave an action its purpose. In construction that omission is consequential: skilled work is directed at project-specific configurations defined in a design model - configurations not yet present in the environment being observed. A mason's transferable competence is not the geometry of one wall but the ability to realise a new geometry from a specification. We introduce the specification-conditioned demonstration: a synchronised record of the physical state a demonstrator perceives, the intended state supplied to them by an external design, and the behaviour connecting the two. We present OJOx, a capture interface that realises this for construction - delivering design geometry to a headset, anchoring it in the physical workspace, rendering it into a demonstrator's stereo passthrough view, and recording that view synchronously with whole-body and hand motion. We report one fully instrumented session - a 33-component wall laid against a specification that changes while the work proceeds - and check the record against the physical scene through an external camera registered independently of the capture. Recorded sessions remain compatible with existing humanoid retargeting infrastructure and replay onto a Unitree G1 in simulation. The result is a data interface for testing whether embodied policies can learn not merely to imitate demonstrated actions, but to act toward specifications absent from their training experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.14615v1">Open-UniMo: Towards Unified Motion-Language Understanding and Generation in the Open World</a></div>
    <div class="paper-meta">
      📅 2026-09-13
    </div>
    <details class="paper-abstract">
      Unified motion generation and understanding is crucial for embodied AI systems that can both synthesize and interpret human actions in open-world environments. Existing motion-language models often treat motion as an auxiliary modality of a language model, leading to text-dominated representations and limited cross-modal interaction. Moreover, the next-token prediction paradigm is not naturally suited to long motion sequences, where autoregressive generation may accumulate prediction errors. To address these challenges, we propose Open-UniMo, a unified Large Motion-Language Model (LMLM) trained on million-scale open-world motion-language data. Open-UniMo promotes modality parity by extending Qwen's vocabulary of about 150K text tokens with 64K motion tokens, enabling motion and language to share a unified token space. We further introduce motion-consistent Chain-of-Thought reasoning as an intermediate representation to bridge language semantics and motion dynamics. Open-UniMo is trained with a two-stage pipeline, where supervised fine-tuning establishes CoT-guided bidirectional motion-language mapping and Group Relative Policy Optimization (GRPO) improves semantic alignment while mitigating cumulative errors in autoregressive motion-token generation. To support comprehensive evaluation, we propose Open-MoBench, a VLM-guided benchmark for assessing text-to-motion (T2M) generation, motion-to-text (M2T) understanding, and bidirectional consistency. Extensive experiments show that Open-UniMo achieves state-of-the-art performance on both conventional metrics and Open-MoBench. Furthermore, ablation studies reveal that M2T understanding is not primarily limited by motion-token vocabulary size; instead, coupling M2T with the learnable T2M generation path yields stronger cross-modal representations, demonstrating that generation can facilitate understanding in AR-based motion-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.22285v1">ORDER: A Fictitious-World Benchmark for Domain-Adaptive Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-09-13
    </div>
    <details class="paper-abstract">
      Adapting language models to new domains via continual pre-training raises a basic evaluation problem: if the training corpus overlaps with what the model already knows, performance gains cannot be cleanly attributed to new learning rather than pre-existing knowledge. This matters most for knowledge-intensive, task-light (KHTL) robot deployments - pharmaceutical dispensing, hazardous-material handling, facility-specific protocols, where the physical task is simple but the governing rules are proprietary and safety-critical, and where extensive live testing is costly or unsafe. We introduce ORDER (Ontology-driven Decision-making for Embodied Reasoning), a benchmark built on a fictitious world: a 342,069-token synthetic corpus defining a self-consistent physics that cannot appear in any model's pre-training data. ORDER pairs a 500-question knowledge test (ORDER-BENCH) with a harder compositional task, ORDER-SPATIAL: ordering objects for safe manipulation across both familiar and entirely novel scenes. GPT-4.1 without adaptation scores below chance on ORDER-SPATIAL (Kendall's tau = 0.441), showing its priors actively conflict with the invented physics. After continual pre-training, small models improve substantially on both familiar and novel scenes alike evidence of genuine world-model induction rather than memorization. We then carry this through to a robot pipeline: models that answer the knowledge test well often cannot produce valid, executable plans without a further skill-adaptation stage, after which small, fully offline models outperform GPT-4.1 even when GPT-4.1 is given retrieval access to the same rules (Kendall's tau = 0.848 vs. 0.606), on a full perception-to-execution loop demonstrated on a simulated iiwa7 arm with human-in-the-loop correction. Throughout, ORDER-SPATIAL performance, not knowledge-test accuracy is what predicts real plan quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.12626v2">Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2026-09-12
      | 💬 Accepted to ECCV 2026. Project page: https://zju3dv.github.io/habitat-gs/
    </div>
    <details class="paper-abstract">
      Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators predominantly rely on mesh-based rasterization, for which photorealistic assets are costly to author at scale, and their support for dynamic human avatars is largely constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation, while performance benchmarks validate the system's scalability. Code is available at https://github.com/zju3dv/habitat-gs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11038v2">EgoFun3D: Modeling Interactive Objects from Egocentric Videos using Function Templates</a></div>
    <div class="paper-meta">
      📅 2026-09-11
      | 💬 Project website: https://3dlg-hcvc.github.io/EgoFun3D/
    </div>
    <details class="paper-abstract">
      We present EgoFun3D, a coordinated task formulation, dataset, and benchmark for modeling interactive 3D objects from egocentric videos. Interactive objects are of high interest for embodied AI but scarce, making modeling from readily available real-world videos valuable. Our task focuses on obtaining simulation-ready interactive 3D objects from egocentric video input. While prior work largely focuses on articulations, we capture general cross-part functional mappings (e.g., rotation of stove knob controls stove burner temperature) through function templates, a structured computational representation. Function templates enable precise evaluation and direct compilation into executable code across simulation platforms. To enable comprehensive benchmarking, we introduce a dataset of 517 egocentric videos featuring challenging real-world interactions with paired 3D geometry, segmentation over 2D and 3D, articulation and function template annotations. To tackle the task, we propose a 4-stage pipeline consisting of: 2D part segmentation, reconstruction, articulation estimation, and function template inference. Comprehensive benchmarking shows that the task is challenging for off-the-shelf methods, highlighting avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16602v3">PAVXploreRL: Physical-Action-Visual World Model Reinforcement Learning with Action Exploration</a></div>
    <div class="paper-meta">
      📅 2026-09-10
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Action-conditioned world models are a key component of embodied AI, serving as scalable policy evaluators that reduce reliance on expensive real-world rollouts. To accurately capture diverse action-induced dynamics, such models should satisfy three key objectives-Physical Plausibility (P), Action Adherence (A), and Visual Fidelity (V), collectively referred to as PAV-while remaining robust to both in-distribution (ID) expert demonstrations and out-of-distribution (OOD) actions. However, existing methods primarily rely on ID action-video pairs and pixel-level reconstruction losses, which do not explicitly optimize PAV objectives and generalize poorly beyond expert data. To address this, we propose PAVXploreRL, a reinforcement learning framework built on a pretrained latent world model that explicitly optimizes PAV objectives through reward-driven training. To improve action generalization, our method jointly leverages ID trajectories and noise-driven OOD action exploration, without paired video supervision. Experiments show that PAVXploreRL consistently outperforms pretrained baselines, achieving a 5.6% average gain across benchmarks and producing higher-quality PAV properties. As a policy evaluator, it also yields more reliable performance estimates and reduces the overestimation bias of prior expert-only world models such as Ctrl-World. Code: https://github.com/Social-AI-Studio/PAVXploreRL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21938v2">The Biggest Risk of Embodied AI is Governance Lag</a></div>
    <div class="paper-meta">
      📅 2026-09-09
    </div>
    <details class="paper-abstract">
      Embodied AI is widely discussed as a job-displacement problem. The deeper risk, however, is governance lag: the time and capability gap between a measurable change in technology deployment and an institutional response able to address its consequences. Building on the established pacing problem and the Collingridge dilemma, this article argues that embodied AI intensifies that gap through scalable models and platforms, task-level reorganization, and the separation of upstream technological control from downstream social impact. We distinguish three mutually reinforcing forms of lag, observational, institutional, and distributive, and propose a compliance architecture based on deployment visibility, stack-level accountability, trigger-based adjustment, and automatic distributional response. The central policy challenge is not automation alone, but whether governance systems can become observable, responsive, and adaptive before disruption becomes entrenched.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.09396v1">VANTAGE-Bench: Evaluating the Infrastructure AI Gap in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-09-08
      | 💬 23 pages, 2 figures, 14 tables. Project page: https://vantage-bench.org/; dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench;
    </div>
    <details class="paper-abstract">
      As Vision-Language Models (VLMs) advance toward physical deployment, the focus has remained on action-oriented Embodied AI evaluated on subject-centric consumer video. This overlooks a pervasive class of Physical AI: Infrastructure AI, which relies on fixed cameras for open-loop insights like safety monitoring and operational logging. We introduce VANTAGE-Bench, a benchmark measuring this "Infrastructure AI Gap." It spans three operational domains (Logistics, Transportation, and Smart Spaces), unifies image and video evaluation across semantic, spatial, temporal, and spatio-temporal capabilities, and moves beyond multiple-choice to eight task formulations including dense captioning and spatio-temporal grounding. It adds a single-pass trajectory protocol for Single Object Tracking and, to our knowledge, the first such evaluation on fixed-camera infrastructure video, scored against specialist trackers. Annotation spans three regimes over 3,346 media assets: 3,342 video-task annotations, 4,281 image-grounding annotations, and 27,404 detection boxes. Evaluating 17 models zero-shot, we find the shortfall relative to consumer-centric benchmarks is concentrated, not general. Event verification, referring expressions, and temporal localization fall roughly 9 to 24 points at every model scale, while video question answering stays within 5.3 points of VideoMME and 2D spatial pointing shows no shortfall against BLINK. The temporal pillar is weakest in absolute terms: no system exceeds 55.7 mIoU on temporal localization or 37.3 SODA_c on dense video captioning. On tracking, frontier models come within roughly 5 points of specialist trackers over short horizons but separate as the horizon extends. Open-weight models lead 2D object localization outright, so neither scale nor proprietary access explains the pattern. Data, evaluation harness, and leaderboard: https://vantage-bench.org/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00288v4">TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-09-08
      | 💬 For code and data, see https://baiqi-li.github.io/timeblind_project/
    </div>
    <details class="paper-abstract">
      Fine-grained spatio-temporal understanding is essential for video reasoning and embodied AI. Yet, while Multimodal Large Language Models (MLLMs) master static semantics, their grasp of temporal dynamics remains brittle. We present TimeBlind, a diagnostic benchmark for compositional spatio-temporal understanding. Inspired by cognitive science, TimeBlind categorizes fine-grained temporal understanding into three levels: recognizing atomic events, characterizing event properties, and reasoning about event interdependencies. Unlike benchmarks that conflate recognition with temporal reasoning, TimeBlind leverages a minimal-pairs paradigm: video pairs share identical static visual content but differ solely in temporal structure, utilizing complementary questions to neutralize language priors. Evaluating over 20 state-of-the-art MLLMs (e.g., GPT-5, Gemini 3 Pro) on 600 curated instances (2400 video-question pairs), reveals that the Instance Accuracy (correctly distinguishing both videos in a pair) of the best performing MLLM is only 48.2%, far below the human performance (98.2%). These results demonstrate that even frontier models rely heavily on static visual shortcuts rather than genuine temporal logic, positioning TimeBlind as a vital diagnostic tool for next-generation video understanding. Dataset and code are available at https://baiqi-li.github.io/timeblind_project/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.07135v1">NutriBench-Kitchen: Benchmarking Embodied AI for Nutrition Management</a></div>
    <div class="paper-meta">
      📅 2026-09-07
      | 💬 17 pages, 4 figures, ECCV
    </div>
    <details class="paper-abstract">
      An embodied kitchen assistant must do more than recognize food in isolated frames. It must track ingredient states over time and integrate visual observations with recipe and nutritional knowledge to support constraint-aware decision-making. We formalize this capability as \emph{Embodied Nutrition Management}: perceiving nutrition-relevant events, maintaining a persistent food state, and using it for knowledge-grounded planning. Existing benchmarks evaluate static food understanding or embodied cooking actions, but do not measure whether an agent can continuously update and use nutrition-relevant states in dynamic kitchens. To fill this gap, we introduce \textbf{NutriBench-Kitchen}, a benchmark containing 1,500 manually verified question--answer pairs from 160 cooking videos. It covers five task families: Ingredient Entry, Memory Management, Recipe Query, Long-Term Planning, and Short-Term Planning, spanning food-state construction, maintenance, knowledge retrieval, and decision-making across different planning horizons. Evaluations of proprietary and open-source large vision-language models reveal a substantial gap from human performance, particularly in quantitative ingredient estimation, long-term state tracking, and reasoning under interacting constraints. We further introduce \textbf{Nutri-Vgent}, a diagnostic long-video agent with separate episodic, food-state, and recipe memories. Its consistent improvements demonstrate the value of explicit state representations and structured memory for nutrition management. Together, NutriBench-Kitchen and Nutri-Vgent provide a testbed for studying persistent state tracking and knowledge-grounded reasoning in dynamic kitchens. Code is available at https://github.com/V1ol1n/NutriBench-Kitchen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06326v1">Rethinking Safety for Generalist Robots</a></div>
    <div class="paper-meta">
      📅 2026-09-06
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Generalist robots promise to transform our society: the same system that prepares a meal or folds laundry might also repair a car, inspect infrastructure, or care for a loved one. Yet this versatility introduces risks far beyond the collision- and force-based safety notions that have long dominated robotics. Notions of safety must now consider context (e.g., turning off a building's electricity is only safe during scheduled maintenance), user intent (e.g., asking the robot to ``clean the kitchen'' includes unspoken expectations that the robot should not mix dangerous but powerful cleaning agents like bleach and ammonia), hard-to-model physical consequences (e.g., burning food during meal preparation), and more. We argue the need for a new era of robot safety---embodied AI safety---that broadens the hazards considered across the robot's lifecycle while recognizing that the safety of bits cannot be separated from the safety of atoms. We present a taxonomy of emerging risks and a full-stack research agenda to guide the safe deployment of generalist robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.06099v1">PASTEL: Panoramic Alignment for Monocular 4D Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-09-05
      | 💬 Accepted to ECCV 2026. 11 figures
    </div>
    <details class="paper-abstract">
      Reconstructing 4D scenes from casually captured monocular video is vital for applications in virtual reality (VR) and embodied AI. Recent advances in 4D reconstruction and novel view synthesis have substantially propelled this capability. However, existing reconstruction methods generally cannot recover regions beyond visible camera limits. Consequently, we introduce a new paradigm that achieves 4D scene synthesis by combining visible-region reconstruction from monocular input with invisible-region generation beyond observable camera boundaries. We present Panoramic Alignment for Strategic Exploitation of Generative Priors (PASTEL). Specifically, PASTEL proposes panoramic scene alignment, a novel representation that reformulates the intractable 3D "invisible region" exploration into a tractable 2D directional trajectory planning. This is achieved by reducing the viewpoint planning from 6-DoF search to a 2D directional search with explicit visibility boundaries. By operating within this panoramic space, our method strategically identifies camera trajectories that maximize exploration beyond observable boundaries while minimizing viewpoint deviation. Experimental results show that PASTEL can not only extrapolate plausible scene content beyond the observable boundaries of input monocular videos, but also substantially boost monocular 4D reconstruction performance. PASTEL outperforms the previous state-of-the-art method by 0.9dB in full-image PSNR on the DyCheck IPhone dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.28215v2">HAT-4D: Lifting Monocular Video for 4D Multi-Object Interactions via Human-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-09-05
      | 💬 Accepted to ECCV 2026. 15 pages of main text and 39 pages of appendices. Project page: https://lijiaxin0111.github.io/HAT4D/
    </div>
    <details class="paper-abstract">
      Extracting dynamic 4D object interactions from massive, in-the-wild monocular videos offers a highly efficient data collection pathway for scaling Embodied AI and training VLAs. However, existing monocular 4D reconstruction methods primarily focus on isolated objects, often failing under the severe occlusions and complex dynamics inherent in multi-object interactions. To bridge this gap, we propose HAT-4D, the first agentic framework designed to reconstruct the 3D geometry, temporal dynamics, and physical interactions of multiple objects from a single video. By integrating VLMs with a multi-level human-in-the-loop feedback mechanism, HAT-4D efficiently resolves depth ambiguities and interaction-induced occlusions during 3D generation and 4D propagation, yielding physically plausible assets without relying on expensive multicamera rigs. As a scalable data engine, HAT-4D facilitates the creation of MVOIK-4D, an open-world benchmark for monocular 4D interaction reconstruction, accompanied by a novel multi-dimensional evaluation protocol focused on physical plausibility and temporal consistency. Extensive experiments demonstrate that HAT-4D achieves SOTA performance on most evaluation metrics, while maintaining competitive semantic alignment. Ablation studies show that introducing a small amount of human feedback improves interaction reconstruction. Moreover, the data produced by HAT-4D effectively improves baseline performance when used for fine-tuning. Our data and code are available at https://lijiaxin0111.github.io/HAT4D/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05594v1">SceneMosaic: Efficient and Diverse Simulation-Ready Scene Generation via Hybrid Agentic Layout Evolution</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 https://github.com/rxjfighting/SceneMosaic
    </div>
    <details class="paper-abstract">
      Diverse and simulation-ready indoor scenes are essential for interactive entertainment and embodied AI, yet their scalable generation remains challenging. Recent agentic text-to-3D scene pipelines that rely on vision-language models (VLMs) can generate scenes of high fidelity but require costly iterative object placement and refinement. Another mainstream paradigm, parametric image-to-3D scene models, produces scenes efficiently from strong priors learned from 2D images but often leads to imprecise and physically invalid scenes. More importantly, both paradigms struggle to output diverse scenes for a single input, making it hard for them to reflect the dynamically changing nature of real scenes. In this paper we propose \textbf{SceneMosaic}, a framework that combines the merits of both paradigms. It obtains the initial candidate from the learned image-based prior, and subsequently evolves the result through VLM agents, ensuring both efficiency and physical validity. Within the evolution process, SceneMosaic exploits the locality of natural scenes and decomposes a scene into independent local units, allowing separate evolution within each unit before composing the global scene via Cartesian product. On SceneEval-100, SceneMosaic matches the strongest agentic baseline in semantic layout quality with a 24x speedup, substantially reduces physical violations, and receives the highest human ratings. Our code is publicly available at https://github.com/rxjfighting/SceneMosaic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2609.05583v1">An overview of 3D Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-09-04
      | 💬 Accepted for presentation as a tutorial at SIBGRAPI 2026. Tutorial page: https://usmarcv.github.io/Tutorial-3DVLMs/
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are reshaping computer vision by aligning visual and textual embeddings, allowing models to recognize visual concepts and reason about them using natural language. Traditional 3D deep-learning models, however, are typically trained for specific tasks, such as classification, segmentation, or detection, and do not naturally support cross-modal retrieval from their embedding spaces using text or images as queries. To address this issue, Contrastive Language-Image Pretraining (CLIP)-based methods align 3D embeddings with pretrained image and text representations, giving rise to 3D Vision-Language Models (3D VLMs) that support zero-shot classification, cross-modal retrieval, and open-vocabulary recognition of 3D shapes. This tutorial provides an overview of 3D VLMs, ranging from basic definitions of 3D representations and their encoding into embeddings to cross-modal contrastive alignment, modern multimodal frameworks, and 3D Vision-Large Language Models (3D VLLMs). We present the main definitions of contrastive learning for multimodal embedding alignment and highlight recent advances in language-guided 3D Gaussian splatting, 3D shape generation, and embodied AI for robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20308v3">ACE-Ego-Hand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery</a></div>
    <div class="paper-meta">
      📅 2026-09-01
      | 💬 Project Page: https://ggxxii.github.io/ace-ego-hand
    </div>
    <details class="paper-abstract">
      Egocentric video offers scalable manipulation data for embodied AI, yet recovering metric 3D hand trajectories remains challenging due to severe object occlusion and frequent out-of-sight gaps. Existing single-frame and windowed temporal regressors fail when a hand shortly leaves the frame, while recent video diffusion models (VDMs) rely on heavy, stochastic multi-step sampling as pixel-space renderers. We instead repurpose VDM into a deterministic geometry encoder. A single forward pass over the clean latent exposes scene content beyond current observations, including occluded and out-of-sight hands. We introduce ACE-Ego-Hand, an offline clip-level framework that extracts features via a Deterministic Clean-Latent Encoder and decodes them with a Bidirectional Spatiotemporal Decoder. ACE-Ego-Hand recovers continuous bimanual trajectories with metric placement and no external detector, while a Ray-Based Camera Solver supports a second configuration that requires no test-time camera intrinsics. Across five egocentric benchmarks, ACE-Ego-Hand sets a new state of the art, cutting MPJPE-p by 30% on occlusion-heavy ARCTIC and 40% on HOT3D. These gains reach 46%-61% once out-of-sight hands are included in the evaluation, offering a scalable path from everyday human video to robot manipulation data.
    </details>
</div>
