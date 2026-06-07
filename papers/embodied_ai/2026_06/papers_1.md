# embodied ai - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
