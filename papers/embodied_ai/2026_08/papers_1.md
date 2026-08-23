# embodied ai - 2026_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.20308v1">DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Project Page: https://ggxxii.github.io/dreamhand/
    </div>
    <details class="paper-abstract">
      Egocentric video offers scalable manipulation data for embodied AI, yet recovering metric 3D hand trajectories remains challenging due to severe object occlusion and frequent out-of-sight gaps. Existing single-frame and windowed temporal regressors fail when hand shortly leaves the frame, while recent video diffusion models (VDMs) rely on heavy, stochastic multi-step sampling as pixel-space renderers. We instead repurpose VDM into a deterministic geometry encoder. A single forward pass over the clean latent exposes scene content beyond current observations, including occluded and out-of-sight hands. We introduce DreamHand, an offline clip-level framework that extracts features via a Deterministic Clean-Latent Encoder and decodes them with a Bidirectional Spatiotemporal Decoder. DreamHand recovers continuous bimanual trajectories with metric placement and no external detector, while a Ray-Based Camera Solver supports a second configuration that needs no test-time camera intrinsics. Across five egocentric benchmarks, DreamHand sets a new state of the art, cutting MPJPE-p by 30% on occlusion-heavy ARCTIC and 40% on HOT3D. These gains reach 46%-61% once out-of-sight hands are included in the evaluation, offering a scalable path from everyday human video to robot manipulation data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10668v2">XDen-1K: A Density Field Dataset of Real-World Objects</a></div>
    <div class="paper-meta">
      📅 2026-08-20
      | 💬 Project Page: https://xden-1k.github.io/
    </div>
    <details class="paper-abstract">
      A deep understanding of the physical world is essential for robotic manipulation and physically realistic simulation. While current methods, including VLM-based and other learning-based approaches, have shown promise in physical property inference, their evaluation is often hindered by the lack of physically grounded reference data. To address this gap, we introduce XDen-1K, the first large-scale multimodal dataset that provides physically grounded density field for real-world objects. XDen-1K comprises 1,000 real-world objects spanning 137 categories, with comprehensive data for each object, including a high-resolution, carefully curated 3D geometric model with part-level annotations and paired real-world biplanar X-ray scans. In addition, XDen-1K includes high-fidelity volumetric density field reconstructed from sparse biplanar X-ray views via a novel optimization framework. XDen-1K also provides a benchmark for density estimation and enables X-ray-conditioned volumetric segmentation. Experiments further demonstrate that the center-of-mass prior derived by XDen-1K can improve robotic manipulation performance. By providing real-world X-ray scans and physics-consistent density field, XDen-1K establishes a foundation for advancing physical property inference and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.17097v2">HarmoHOI: Harmonizing Appearance and 3D Motion for Multi-view Hand-Object Interaction Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-08-20
    </div>
    <details class="paper-abstract">
      Hand-Object Interaction (HOI) synthesis is a cornerstone for animation production and embodied AI. Despite the strong priors of video foundation models, multi-view consistent HOI synthesis remains challenging due to complex hand motions and occlusions. We present HarmoHOI, a unified diffusion framework that jointly and harmoniously generates synchronized multi-view HOI videos and globally aligned 3D point tracks. Our core insight is that robust multi-view consistency fundamentally requires globally aligned 3D geometry and motion. To this end, we propose a Mixture of Multi-view Diffusion Transformer that co-models RGB videos and 3D point tracks. By representing point tracks as pseudo-videos, we align 3D geometric signals with the 2D latent space of foundation models, thereby minimizing the domain gap and easing adaptation of priors. To further ensure geometry consistency, we introduce Global Motion Aligning Diffusion, which refines coarse point tracks into metric-scale, globally aligned 3D trajectories. HarmoHOI enables on-the-fly co-evolution of 2D appearance and 3D motion during denoising. To overcome the scarcity of multi-view HOI data, we employ a hybrid data curriculum learning strategy that successfully transfers generic priors from single-view data to synchronized multi-view generation. Experimental results show that HarmoHOI achieves state-of-the-art performance in visual quality, motion plausibility, and multi-view geometric consistency. Project page available at https://droliven.github.io/HarmoHOI_project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.19375v1">Learning the Right Abstraction: Neural Reduced Dynamics for Complex Robot Control</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      High-fidelity embodied AI simulators provide realistic evaluation of complex robotic systems, but their computational cost limits their direct use for large-scale reinforcement learning campaigns. We advocate the use of less accurate but more expeditious simulations, which might draw on data-driven, e.g., neural dynamics, models. This contribution argues that the practical value of a neural dynamics model for complex robot control lies in learning the \emph{right abstraction}: a reduced state that preserves the control-relevant physics of the high-fidelity system while enabling high-throughput policy learning. We develop a neural reduced dynamics (NRD) framework that separates the state the model propagates from what can be supplied as an input or recovered analytically, trains policies entirely inside the frozen learned model, and validates them back in the high-fidelity simulator. Two case studies instantiate it across three control tasks: terrain-aware HMMWV trajectory tracking on rigid, bumpy and deformable Continuum Representation Model (CRM) terrain; and goal reaching for a stock tracked vehicle and its front-mounted articulated arm. Every policy transfers back to the high-fidelity simulator. A single policy trained inside the terrain-conditioned dynamics model, and given no terrain input of its own, attains lower median and mean tracking error than both single-terrain specialists on all three terrains, including zero-shot bumpy terrain. Quantitatively, the tracked vehicle reaches 100 of 100 goals and the arm 97 of 100, with zero contacts or joint-limit violations. The NRD models advance roughly four orders of magnitude faster in simulated time than the high-fidelity simulator scenes they replace, making iterative on-policy learning practical and supporting neural reduced dynamics as a bridge between accurate but expensive physics simulation and scalable robot learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18840v1">Beyond Placement and Articulation: Usage-Driven Code Scenes for Embodied Interaction</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Indoor scene synthesis provides essential environments for embodied AI, robotic manipulation, and simulation-based policy learning. Recent code-based scene generation methods produce editable and extensible environments, yet they remain focused on visual construction and object-level articulation, leaving the functional usage of scenes largely unmodeled. To address this problem, we present RoomWright, an agentic usage-driven framework for generating 3D scenes represented entirely as code for embodied interaction. RoomWright performs usage-driven object reasoning, which treats each anchor as a task centre and admits task-required objects and their affordances. A code agent further enables multi-part interaction by compiling each interaction into a trigger, condition, effect rule that updates structured object states, capturing causal dependencies across objects. Moreover, since manipuland orientation is ambiguous and hard to recover from pixels, RoomWright alleviates this via annotation-informed usage-guided orientation. Extensive experiments demonstrate the effectiveness of our method. The resulting scenes are executable, editable, and simulation-ready, providing interactive environments for embodied AI and policy learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18734v1">CL4D: Contrastive Language-4D Pretraining for Vision-Language Reasoning in Dynamic Scenes</a></div>
    <div class="paper-meta">
      📅 2026-08-19
      | 💬 Accepted at the 19th European Conference on Computer Vision (ECCV 2026)
    </div>
    <details class="paper-abstract">
      4D understanding and reasoning is a fundamental capability for embodied AI agents operating in dynamic physical environments. However, existing vision encoders are largely limited to static 2D images or 3D point clouds without temporal modeling, or to 2D videos that lack accurate geometric depth reasoning. Consequently, current approaches fail to jointly capture spatial structure and motion evolution in dynamic scenes. We present CL4D, the first foundational 4D vision encoder that directly operates on dynamic point clouds, trained with a contrastive learning objective to align spatio-temporal geometric representations with natural language descriptions. By learning a shared embedding space between text and 4D scene dynamics, CL4D enables zero-shot motion-to-text and text-to-motion retrieval in dynamic environments and serves as a foundational 4D vision encoder for downstream 4D vision-language tasks. Building on this encoder, we introduce 4DVLM, a 4D vision-language model that conditions language generation on dynamic geometric representations. 4DVLM is the first VLM designed to operate directly on 4D point clouds without relying on 2D images, 2D videos, or static 3D point clouds. We train CL4D and subsequently 4DVLM on a newly constructed dataset termed DynAction4D capturing diverse human motions across varying object interactions and scene environments. Extensive experiments across multiple 4D human action benchmarks demonstrate that CL4D achieves state-of-the-art performance, with improvements of approximately ~16.75% over prior methods. Furthermore, 4DVLM outperforms frontier video VLMs such as Gemini and GPT-5 even when these models are provided with RGB video sequences corresponding to the same scenes represented as 4D point clouds for 4DVLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.18671v1">Vision-Language Models for Egocentric Video: From Hand-Object Interaction to Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-08-19
    </div>
    <details class="paper-abstract">
      Egocentric video captures activities from the wearer's perspective, providing a direct view of human attention, hand--object interaction, and goal-directed behavior. This perspective is increasingly important for wearable intelligence, assistive systems, human--robot interaction, and embodied AI, yet it introduces challenges including ego-motion, occlusion, small active objects, viewpoint-dependent appearance, and long-range temporal dependencies. Vision--language models (VLMs) offer a promising foundation for addressing these challenges by linking visual observations with semantic knowledge and natural-language supervision. This survey presents a critical review of VLMs for egocentric video understanding, tracing the progression from conventional recognition architectures to multimodal foundation models and embodied systems. We organize the literature around tasks, datasets, hand--object interaction understanding, temporal reasoning, frame and clip selection, multimodal representation learning, prompting, semantic alignment, and model adaptation. Particular attention is given to graph-based and object-centric reasoning as mechanisms for modeling relations among hands, objects, actions, and scene context over time. We further examine how first-person perception and multimodal foundation models support wearable assistance, robot skill learning, human-to-robot transfer, and embodied decision making. Across the reviewed literature, a consistent limitation emerges: current models recognize visible objects more reliably than evolving interactions, actions, and user intent, especially over long activities. We therefore identify temporally grounded reasoning, interaction-aware supervision, efficient long-video processing, multimodal fusion, graph-enhanced representations, cross-domain generalization, privacy, and trustworthy evaluation as key priorities for deployable embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.09520v2">Seeing is Free, Speaking is Not: Uncovering the True Energy Bottleneck in Edge VLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Accepted to ACM MM 2026. This version includes the appendix
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are the perceptual backbone of embodied AI, but their energy footprint on edge hardware remains poorly understood. Existing efficiency efforts focus predominantly on reducing visual tokens, implicitly treating visual processing as the dominant energy cost. We overturn this implicit assumption through the first systematic energy profiling of on-device VLM inference, spanning five models across three architecture families, four input resolutions, and two hardware platforms (NVIDIA RTX 3070 and Jetson Orin NX). Our analysis yields three findings. First, average inference power is a model-intrinsic constant, invariant to input resolution, image complexity, and prompt type, with less than 5% variation across all conditions. This means that all energy variation across inputs must arise from variation in inference time, not from variation in power draw. Second, each output token costs 11 to 39x more wall-clock time than each input token due to the compute-bound and memory-bound asymmetry between prefill and decode, making output token count the dominant driver of both latency and energy. Third, image complexity, measured by the number of objects in an image, induces up to 4.1x energy differences at identical resolution. This variation arises not from increased visual processing cost, but from differences in output length. These findings expose a fundamental limitation of visual token pruning: even removing all visual tokens saves at most 10% of total energy for fixed-token models. Across models spanning 1 billion to 8 billion parameters, controlling output length saves up to 97% of total energy, with the energy dominance of decoding growing stronger at larger model scale. In short, the true energy bottleneck in edge VLM inference is not what the model sees, but how much it says. Code is available at https://github.com/Junfei-Z/seeing-is-free.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.07459v3">EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-08-18
      | 💬 Project page: horizonrobotics.github.io/EmbodiedGen
    </div>
    <details class="paper-abstract">
      We present EmbodiedGen V2, a generative 3D world engine for building executable policy-ready environments for embodied intelligence. Sim-ready 3D asset generation has advanced rapidly, yet assembling such assets into policy-ready task environments remains largely manual, limiting scalable closed-loop learning. EmbodiedGen V2 addresses this gap through a unified sim-ready representation that connects cross-simulator assets, interaction affordances, task-driven worlds, large-scale multi-room scenes, and stateful Vibe Coding into a generative, editable, and reusable simulation pipeline. The generated environments support manipulation, navigation, mobile manipulation, cross-simulator deployment, and embodied policy training. In evaluation, the asset pipeline achieves 96.5% human acceptance and 98.6% collision success, and 83.3% of task-driven worlds are directly usable for downstream simulation without manual modification. Online reinforcement learning with generated environments further improves simulation success from 9.7% to 79.8%, and transfers to real robots with task success increasing from 21.7% to 75.0%. These results establish EmbodiedGen V2 as scalable simulation infrastructure for training, evaluating, and deploying embodied policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.16555v1">Co-design of Neural and Muscle Network based on Embodied Perceptron Representation</a></div>
    <div class="paper-meta">
      📅 2026-08-17
      | 💬 10 pages, 7 figures, 2026 IEEE/SICE International Symposium on System Integration (SII)
    </div>
    <details class="paper-abstract">
      Recent advances in AI technologies have enabled the advanced design of complex control policies. In contrast, focusing on the body, many robots still employ simple bodies that can limit adaptability to environments. Studies in embodied robotics have shown that well-designed bodies can partially replace the role of control and computation with physical body-environment interactions, yet such designs still depend heavily on expert intuition. There is a need for a systematic theoretical framework for body design, as well as a method for joint optimization of the body and controller. To address this, we introduce the Embodied Perceptron, a theoretical framework that unifies neural networks and physical body systems. In this view, the body itself acts as a perceptron: mechanical parameters correspond to weights, and physical nonlinearities play the role of activation functions. By representing physical constraints as weights and nonlinear properties as activation functions, a physical body can be modeled in neural-network form. The system representation enables us to explicitly and theoretically explain that the body can substitute for part of the neural control. As an application, we co-optimize control policy and muscle configuration in a musculoskeletal robot and show that the resulting embodied intelligence can provide inherent stability, improve learning efficiency, and drastically reduce model size-even with a single-neuron controller. The results bridge the informational and physical worlds and provide a pathway toward understanding and systematic design of embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15636v1">Algorithm-Architecture Co-Design for Efficient VLA Inference via Speculative Inference and Verification</a></div>
    <div class="paper-meta">
      📅 2026-08-16
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in the field of embodied AI, but their high computational cost and limited predicted action length hinder real-time deployment. Although Dadu-Corki, a dedicated accelerator for efficient embodied AI, has been introduced, it does not exploit the inherent interaction patterns between the robot and its environment, which results in a relatively short predicted action length. We observe that robotic environments naturally alternate between active states-where precise actions are crucial-and inactive states-where actions have limited impact on task success. This insight enables a new scheduling opportunity: long-action-length speculative prediction in inactive states, paired with selective verification in active states. We propose SpecVLA, an algorithm-system co-design framework that adaptively balances action length, inference latency, and task reliability. On the algorithm side, SpecVLA introduces a state-aware VLA inference execution paradigm and a hardware-friendly construction of a smaller verification model (sVLA) using differential residuals and block-wise mixed-precision quantization. On the system side, we develop a heterogeneous architecture consisting of a GPU and a robotic-specific hardware module, along with a speculative dataflow that decouples VLA and sVLA through parallel execution. Comprehensive evaluations on OpenVLA and RDT across LIBERO and ManiSkill benchmarks show that SpecVLA reduces end-to-end latency significantly while preserving task success rate. By enabling long-action-length speculative prediction with timely verification, SpecVLA achieves real-time robotic manipulation with both high efficiency and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15605v1">AlloEgo-VLM: Disambiguating Allocentric and Egocentric Reference Frames in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-08-16
      | 💬 28 pages, 9 figures. Project page and code available at https://github.com/CKL9001/AlloEgo-VLM
    </div>
    <details class="paper-abstract">
      This study investigates the challenge of ambiguity faced by Vision-Language Models (VLMs) in understanding spatial semantics. Spatial cognition, shaped by cognitive psychology, spatial science, and cultural context, often assigns directionality to objects. However, natural language descriptions of spatial relations frequently omit explicit reference frames, leading to semantic ambiguity and potentially serious errors for embodied AI robots. Existing VLMs, due to insufficient training on reference frames and object orientations, often produce inconsistent responses. To address this issue, we construct a new dataset, AlloEgo-View, comprising (image, query, view-specific answer) triplets that capture key object relations from both allocentric and egocentric perspectives. The view-specific descriptions follow a structured spatial representation that annotate detailed scene descriptions, reference and target objects, their orientations, reference frames, and view types. Building on AlloEgo-View, we develop AlloEgo-VLM, a framework to disambiguate allocentric and egocentric reference frames, even under ambiguous queries, and to be easily integrated into existing VLMs via supervised fine-tuning. Furthermore, we deploy our framework onto an embodied robotic platform within NVIDIA Isaac Sim to validate its real-world feasibility in open-ended object searching tasks. Experiments highlight the limitations of current VLMs in handling view-specific queries and demonstrate the strong disambiguation ability of AlloEgo-VLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.15502v1">EcoVLA: Energy-Efficient Device-Edge Co-Inference for Vision-Language-Action Models under Real-Time Constraints</a></div>
    <div class="paper-meta">
      📅 2026-08-16
      | 💬 Accepted by APPT 2026
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have emerged as a promising foundation for Embodied AI, but their high inference cost poses significant challenges for deployment in robotic systems. In practice, on-device inference is constrained by limited compute capacity and energy budgets, struggling to simultaneously satisfy real-time control and energy efficiency requirements. Alternatively, offloading the inference workload to an edge server is susceptible to fluctuations in system conditions, introducing unpredictable latency risks. Device-edge co-inference offers a promising solution, but systematic research tailored to VLA models remains scarce, particularly a unified co-inference framework that jointly addresses real-time constraints and system-level energy efficiency. Thus, we propose EcoVLA, an adaptive device-edge co-inference framework for VLA models that maximizes system energy efficiency under real-time constraints. EcoVLA first introduces a unified stage-level abstraction over different VLA paradigms, establishing an architecture-agnostic co-inference design space. It then formulates a joint device-edge-network latency and energy prediction model to enable rapid runtime evaluation of candidate co-inference schemes. Building on this, EcoVLA continuously selects the energy-optimal scheme satisfying real-time constraints with millisecond-level overhead, adapting to runtime variations in network and system states. Furthermore, EcoVLA incorporates a lightweight transmission mechanism for inter-stage intermediate tensors to reduce the communication overhead incurred by cross-device collaboration. Experimental results across VLA models show that EcoVLA improves system energy efficiency by up to 236% over existing co-inference approaches under a 20 Hz action output frequency constraint, while consistently maintaining SLO satisfaction under dynamic network and edge workload conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06161v2">iARCS: Iterative Agentic RL for Controllable 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-13
      | 💬 15 pages, 9 figures, 4 tables. Includes appendix
    </div>
    <details class="paper-abstract">
      Synthetic 3D scene generation is increasingly used as a data source for computer vision and embodied AI, but existing generators often optimize perceptual realism without reliably satisfying task-critical functional constraints. This mismatch limits the usefulness of synthetic data for downstream training, where accessibility, traversability, and spatial rule compliance are often essential. We present iARCS, an iterative agentic reinforcement learning framework that adapts a pretrained scene generator to natural-language task requirements. iARCS uses a two-stage strategy: universal-reward pretraining to improve physical plausibility and layout quality, followed by task-specific fine-tuning with LLM-generated reward programs that are iteratively refined from training feedback. Experiments show improved constraint fidelity on walkability, reachability, and clearance-focused tasks, effective task-specific constraint optimization, and competitive scene diversity. We further show that data generated by iARCS improves a base generator, supporting its value as a practical synthetic data generation tool rather than only a controllable scene editing method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05474v2">3D Scene Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Accepted by IJCV. Project Page: https://github.com/hzxie/Awesome-3D-Scene-Generation
    </div>
    <details class="paper-abstract">
      3D scene generation seeks to synthesize spatially structured, semantically meaningful, and photorealistic environments for applications such as immersive media, robotics, autonomous driving, and embodied AI. Early methods based on procedural rules offered scalability but limited diversity. Recent advances in deep generative models (e.g., GANs, diffusion models) and 3D representations (e.g., NeRF, 3D Gaussians) have enabled the learning of real-world scene distributions, improving fidelity, diversity, and view consistency. Recent advances like diffusion models bridge 3D scene synthesis and photorealism by reframing generation as image or video synthesis problems. This survey provides a systematic overview of state-of-the-art approaches, organizing them into four paradigms: procedural generation, neural 3D-based generation, image-based generation, and video-based generation. We analyze their technical foundations, trade-offs, and representative results, and review commonly used datasets, evaluation protocols, and downstream applications. We conclude by discussing key challenges in generation capacity, 3D representation, data and annotations, and evaluation, and outline promising directions including higher fidelity, physics-aware and interactive generation, and unified perception-generation models. This review organizes recent advances in 3D scene generation and highlights promising directions at the intersection of generative AI, 3D vision, and embodied intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/hzxie/Awesome-3D-Scene-Generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.12122v1">HandEdit: A Unified Benchmark for Egocentric Human-to-Robot Dexterous Hand Image Editing</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 Technical Report. Project Page: https://handedit.github.io/
    </div>
    <details class="paper-abstract">
      Robotic manipulation with dexterous hands is a cornerstone of Embodied AI, yet its progress is stifled by the high cost of collecting embodiment-aware teleoperation data. While abundant egocentric videos of human hands offer a scalable alternative, the profound discrepancies in appearance, articulation, and camera viewpoints between human and robotic data raise significant challenges for co-training. Though existing general image-editing models demonstrate strong capabilities, they lack necessary embodiment-specific priors to fully bridge this gap. In this work, we present HandEdit, a unified large-scale embodiment-aware image-editing dataset and benchmark specifically designed to transform human hands and arms into various dexterous robotic embodiments within egocentric frames. HandEdit comprises over 200M editing instances derived from five diverse source datasets, covering 26 distinct URDFs, including 13 hand-only and 13 hand-arm configurations. Alongside the dataset, we establish a unified benchmark protocol with two tracks: Hand-only and Hand-Arm, supporting URDF-conditioned evaluation. We conduct extensive evaluations of 11 representative image-editing baselines using a multi-dimensional metric suite, including generic similarity metrics, VLM-based judgment, and embodiment-aware metrics. HandEdit serves as a critical resource at the intersection of image editing and robotics: it advances embodiment-aware editing models while enabling scalable dexterous robotic learning from abundant human video data, paving the way for more generalizable Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08131v2">UniGround: Universal 3D Visual Grounding via Training-Free Scene Parsing</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 30 pages,9 figures,11 tables
    </div>
    <details class="paper-abstract">
      3D Visual Grounding (3DVG) localizes objects from natural-language descriptions in 3D scenes and is fundamental to embodied AI applications. Although foundation models enable open-vocabulary reasoning, they typically rely on pre-generated candidates, creating two sequential bottlenecks. The \emph{candidate bottleneck} occurs when dataset-specific 3D proposal models miss, fragment, or incorrectly group targets under distribution shifts, excluding them from VLM reasoning. The \emph{evidence bottleneck} stems from incomplete visual evidence: global renderings preserve spatial context but obscure object details, whereas candidate-centric views capture local appearance but lack global context. To address these bottlenecks, we propose UniGround, a zero-shot 3DVG framework that addresses both bottlenecks through Global Candidate Filtering and Contextual Precision Grounding. Global Candidate Filtering constructs topology-consistent, class-agnostic candidates from 3D topology and multi-view semantic cues, without dataset-trained 3D detectors, task-specific proposal supervision, or predefined box and category priors. Contextual Precision Grounding jointly reasons over global spatial context and candidate-centric visual evidence, followed by closed-loop consistency verification for reliable target identification. UniGround achieves 46.1\%/34.1\% Acc@0.25/0.5 on ScanRefer and 28.7\% Acc@0.25 on the evaluated ARKitScenes subset of EmbodiedScan. Further experiments demonstrate competitive grounding without dataset-specific 3D priors, cross-dataset generalization to unseen indoor scenes, and robustness to real-world reconstruction noise and practical domain shifts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11876v1">D3D-GEN: Robot-Aware Domain-Grounded Interactive 3D World Generation for Social Robotics</a></div>
    <div class="paper-meta">
      📅 2026-08-12
      | 💬 8 pages, 5 figures, and 5 tables. Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
    </div>
    <details class="paper-abstract">
      Training and validation of Embodied AI for social navigation critically depends on realistic simulation environments, yet many current approaches fail to find a balance between realism and simulability. We propose D3D-GEN, a novel world generation system that combines a domain agent with a retrieval-augmented generation (RAG) pipeline grounded in that domain. Our system enables users to rapidly generate domain-grounded, fully interactive 3D worlds by automating both the collection of domain knowledge and the synthesis of realistic floorplans and object placements, without dependence on any fixed 3D model database. Given a domain description prompt, the research agent collects publicly accessible domain-specific data and constructs a persistent domain database. Using this database, our RAG pipeline generates plausible floorplans and object placements by dynamically querying a user-provided semantic database, which can be easily extended or modified. The output is a fully interactive 3D world loadable by the popular simulators Isaac Sim and Gazebo. With our approach, we have built databases for several common domains (indoor residential, hospital, office) and generated dozens of distinct, plausible simulation environments for each domain. We present D3D-GEN with a local web frontend that facilitates rapid, interactive world generation for robot simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.11692v1">HUGIN: Enhancing Vision-Language Planning for Autonomous Logistics Sorting</a></div>
    <div class="paper-meta">
      📅 2026-08-12
    </div>
    <details class="paper-abstract">
      Autonomous logistics sorting systems (ALSS) are an important industrial application of embodied AI, which requires joint planning over spatially disjoint camera views. We formulate this setting as Joint Multi-Scene Understanding (JMSU). With open-world visual understanding and task-planning capabilities, vision-language models (VLMs) are promising candidates for JMSU. However, directly applying existing VLMs to JMSU is non-trivial due to scarce cross-scene supervision and attention dispersion caused by long visual context in JMSU. To address these challenges, we propose HUGIN, a training framework with two complementary components. Endogenous Data Augmentation recombines verified atomic facts under operating constraints, while Global Context Ranking aligns the instruction representation more strongly with the complete visual context than with a partial visual context. To support ongoing research, we construct a high-quality industrial sorting dataset and benchmark named SortingBench from four layouts of autonomous logistics sorting systems. Across five open VLMs, HUGIN consistently outperforms matched baselines; for example, the accuracy on SortingBench of Qwen3-VL-8B increases from 63.6% to 78.8%. Additional experiments verify the effectiveness of each component and JMSU's spillover benefits in embodied tasks. Deployment tests involving more than 15,000 packages support the practical viability of VLM-based planning for autonomous logistics sorting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.10864v1">Multi-View Relational Distillation for Spatial Reasoning with Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-08-11
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have achieved strong image and video understanding, yet their visual-spatial representations remain geometrically fragile, leading to failures in spatial reasoning needed for embodied AI, robotics, and autonomous driving. Prior approaches to geometry grounding either fine-tune VLMs on spatial question answering, which can perpetuate spurious visual representations, or fuse features from large geometry-grounded vision models, which substantially increases model size at inference. Knowledge distillation from geometry-grounded vision models offers an alternative, but directly matching multi-view teacher features can disrupt the pretrained alignment between visual and textual representations, degrading object- and language-semantic capabilities. We propose multi-view relational distillation (MVRD), which distills patch-wise cosine similarities across views instead of the teacher features themselves. These relations encode geometric correspondences adequate for spatial understanding, while leaving the student representation underdetermined, allowing it to remain close to its pretrained vision- language space. Across representative VLMs, MVRD improves visual-spatial reasoning, outperforming supervised fine-tuning and feature distillation while approaching feature fusion methods with considerably fewer added parameters and lower latency. We show that MVRD makes visual representations more geometric while retaining language alignment, and generalizes to 3D scene understanding tasks such as object grounding, dense captioning, and question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09876v1">Energy-Structured Latent World Models with Neural Time Fields for Physically Constistent Open-World Motion Planning</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Physically consistent motion planning remains a fundamental challenge in embodied AI, as generated trajectories must strictly conform to real-world execution dynamics. While latent world models offer a promising approach by predicting these dynamics, existing methods learn unconstrained future representations where absorbed physics remains implicit. Therefore, they fail to form reusable physical knowledge, which compromises reliability in unpredictable open-world navigation. To address this, we propose a novel Energy-Structured Latent World Model (ELWM). Our key idea is to structure the ELWM latent state to explicitly carry energy and momentum, ensuring strictly causal transitions via dissipation and control ports. Trained on multimodal RGB-D and inertial interaction histories, our model guarantees physically consistent predictions. We further implement this for motion planning by constructing Physics-Conditioned Neural Time Fields (PC-NTF), a key technical cornerstone that integrates ELWM into an arrival time field via the Eikonal equation to yield a physically-informed navigation policy. Across held-out scenes, our evaluation reveals significant improvements. Compared to generic latent models, PC-NTF reduces 0.8-s motion-prediction NRMSE from 0.36 to 0.29. Against Active Neural Time Fields, it improves navigation success from 81.3% to 89.7% and SPL from 0.64 to 0.73, while cutting the physical collision rate from 12.1% to 5.8% and the Eikonal residual from 0.083 to 0.031. Beyond these targeted gains, our results demonstrate that embedding explicit physical structures into latent spaces intrinsically bridges the gap between predictive world models and safe, dynamically feasible motion planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.09298v1">WorldSimProbe: Diagnosing Simulator Faithfulness in Action-Conditioned World Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 20 pages, 18 figures, and 10 tables, including supplementary material. Code and data: https://evophys.com/WorldSimProbe/
    </div>
    <details class="paper-abstract">
      Action-conditioned world models (ACWMs) promise to provide embodied AI with scalable predictive simulators for planning, policy evaluation, and data generation. Realizing this promise requires precise action-conditioned transitions rather than merely plausible outputs. Yet their applicability remains difficult to establish because prevailing evaluations emphasize visual quality, task outcomes, or coarse rollout-level responsiveness without directly testing simulator fidelity. To address this gap, we evaluate ACWMs through the observable capabilities expected of physical simulators. Accordingly, we formalize Observable Simulator Contract, a minimal contract that any action-conditioned physical simulator should satisfy: supplied actions must induce corresponding agent motion, and environment responses must be grounded in that realized motion. To operationalize this contract, we introduce WorldSimProbe, comprising five controlled suites spanning local control sensitivity, global trajectory variation, source-diverse actions, interaction grounding, and dynamics. Suite-specific evaluators assess simulator-relative calibration, dense action-to-motion correspondence, false-interaction grounding, and primitive-level dynamics. We evaluate six open-source ACWMs on more than 18,000 instances across RoboTwin, ManiSkill, and LIBERO. World-SimProbe reveals systematic action-realization degradation across control variation, structured failures in interaction grounding and dynamics, and benchmark signals consistent with human judgments and downstream outcomes. Together, this capability-based framework provides a transparent, and standardized paradigm for diagnosing ACWM simulator fidelity beyond coarse, task-directed evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06833v2">Unordered Landmark Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2026-08-10
      | 💬 ECCV2026 Oral & Spotlight
    </div>
    <details class="paper-abstract">
      Image-goal navigation is a fundamental capability for embodied AI, yet its practical deployment is strained by strong prior assumptions. Existing methods predominantly rely on temporally ordered video streams or auxiliary sensors (e.g., depth, LiDAR) to maintain spatial consistency. These sequential and multimodal dependencies severely restrict scalability, especially when deploying robots using crowd-sourced or pre-recorded unordered image collections. When temporal priors are removed, current methods struggle with severe perceptual aliasing, noisy associations, and catastrophic mapping failures. To address this underexplored challenge, we propose Unordered Landmark Visual Navigation (ULVN), a unified RGB-only framework free from temporal and odometric priors. ULVN systematically mitigates error accumulation by integrating mapping, localization, and planning. Specifically, it constructs a robust 2D topological map directly from unstructured images via calibrated geometric verification and maximum spanning forest refinement. For closed-loop execution, ULVN abandons sequential heuristics, utilizing a graph-based belief propagation filter with entropy-adaptive fusion for global localization and dynamic subgoal planning. Extensive experiments in simulation and real-world deployments demonstrate that ULVN significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.02501v3">Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots</a></div>
    <div class="paper-meta">
      📅 2026-08-09
      | 💬 18 pages, 2 figures, Project website: https://github.com/SEU-PAISys/Embodied.cpp
    </div>
    <details class="paper-abstract">
      Embodied AI models now span vision-language-action (VLA) models and world-action models (WAMs), but practical deployment remains fragmented across model-specific Python stacks, backend assumptions, and robot-side glue code, especially on heterogeneous edge devices. Existing inference runtimes are designed mainly for request-response serving and therefore do not satisfy the runtime contract of embodied deployment: multi-rate execution inside closed-loop control, latency-first batch-1 inference on heterogeneous hardware, and extensible embodied interfaces beyond fixed token I/O. We present Embodied$.$cpp, a portable C++ inference runtime for embodied models. Based on an architectural analysis of representative VLA models and WAMs, Embodied$.$cpp captures a shared execution path and organizes it into five layers: input adapters, sequence builders, backbone execution, head plugins, and deployment adapters. The runtime provides modular multi-rate execution, latency-first fused inference, and extensible operator and I/O support, enabling deployment across heterogeneous devices, robots, and simulators through one backend abstraction. We evaluate Embodied$.$cpp on three VLA and two WAM models, using normalized comparisons across Python and C++ quantization configurations. Overall, Embodied$.$cpp achieves 1.05x-2.70x inference speedups and 7\%-77\% lower VRAM relative to Python baselines, while maintaining near-baseline success for most configurations. These results show that Embodied$.$cpp improves deployment efficiency while preserving high control quality across diverse embodied model architectures. Project Link: https://github.com/SEU-PAISys/Embodied.cpp
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08491v1">TrustRoboReward: Preference-Ordered Isotonic Score Editing for Multi-Paradigm Robot Reward Models</a></div>
    <div class="paper-meta">
      📅 2026-08-09
    </div>
    <details class="paper-abstract">
      Reward models are a bottleneck for reinforcement learning in embodied AI. Long-horizon robotic manipulation requires scalable vision feedback beyond handcrafted rewards or task-specific annotations. Existing open-source VLM reward judges like RoboReward adopt simple 1--5 trajectory progress scoring, lacking pairwise preferences for RLHF, DPO and Bradley-Terry frameworks, while failing to optimize video scene understanding. Augmenting RoboReward with pairwise comparison and video-QA supervision causes inconsistency between pairwise preferences and pointwise scores, introducing training noise and hurting downstream performance---an issue aggregation methods such as TrustJudge cannot resolve. To address this, we propose TrustRoboReward, a multi-paradigm reward modeling framework equipped with Preference-Ordered Isotonic Score Editing (POISE). We construct a unified four-paradigm dataset with trajectory progress scoring (Score-A), video-QA answer quality scoring (Score-B), and their pairwise counterparts (Pair-A, Pair-B). Pairwise labels align better with human judgment than pointwise scores, inspiring us to calibrate pointwise scores to avoid score-pair reversals against pairwise preferences. POISE rectifies pointwise scores and eliminates cross-paradigm reversal conflicts unresolved by TrustJudge. Theoretically, POISE reduces score-pair reversal conflicts from 20.15% to 0%, whereas TrustJudge retains 20.46% conflicts on the same corpus. Evaluated on our benchmark, Qwen3-VL-4B trained with POISE achieves an overall reward score of 77.96%, nearly matching GPT-5-mini (78.09%, gap 0.13%) and outperforming the strongest RoboReward-4B baseline by 10.13%. It also lifts test-time score-pair consistency to 71.90%, exceeding RoboReward-4B (57.26%) and GPT-5-mini (68.09%). Integrating TrustJudge aggregation during inference boosts the overall score to 78.57%, surpassing the GPT-5-mini teacher model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08183v1">Multi-modal Interactive Control of Robotic Arm based on Offline Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-08-08
      | 💬 This research work has been accepted for poster presentation at ICRA 2026 MEI (Multimodal Embodied Interaction in Robots) Workshop. (Here is the workshop-version short paper.)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly revolutionized the modern society with numerous advanced interactions between humans and AI agents, whereas the usage of most large language models including ChatGPT are not friendly open-sourced and must require the users paying a lot for such AI services continuously. Therefore, deploying open-sourced large language models on local servers can be considered as an efficient approach to design and implement creative embodied AI algorithms with lower cost and more stable free usage. Inspired by this ordinary motivation, we originally propose and implement the "Socratic Models-ChatGLM", which is a well-performed algorithm for multi-modal interactive control of robotic arm based on offline large language models via the facile PyBullet platform, even presents extraordinary potential to address complicated text-image integrated multi-step long-horizon robotic manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.08053v1">PhysX-CoT: Structured Physical Reasoning from a Single Image to Simulation-Ready 3D Assets</a></div>
    <div class="paper-meta">
      📅 2026-08-08
    </div>
    <details class="paper-abstract">
      Simulation-ready 3D assets are central to robotics and embodied AI. Generating them from a single image is usually framed as a vision-language model that emits a serialized asset for a decoder to turn into geometry and physical fields, leaving the image-to-3D reasoning implicit. We argue the limiting factor is this output-centric view: part placement and local shape are entangled in one global-coordinate token stream, and the intermediate physical states are never exposed for supervision, conditioning, or verification. PhysX-CoT instead casts single-image asset generation as an explicit structured physical reasoning process, an ordered and machine-parseable trajectory of part-level states covering decomposition, 2D and 3D grounding, relations, coarse geometry, and surface cues that we separately supervise, use to condition geometry, and treat as reward targets. Geometry is factorized so that 3D boxes carry placement and local codes carry shape, and CoT-aligned GRPO optimizes parse validity, grounding, geometry, placement, and physical consistency. Under a unified protocol that retrains all learned baselines on the same backbone, data, and frozen decoder, PhysX-CoT outperforms the closest full-task baseline across geometry, scale, and physical-attribute metrics. Oracle, token-matched, and state-order controls show the explicit states are functional rather than cosmetic, and in Unreal Engine~5 the generated assets parse, collide, and articulate at high validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.07154v1">Representation Handoffs for OpenArm-Based Laboratory Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-08-07
      | 💬 Robotics: Science and Systems (RSS) Workshop 2026
    </div>
    <details class="paper-abstract">
      Open-source robotics and foundation models have lowered the barrier to embodied AI, yet language-guided laboratory automation still requires reliable alignment from instructions and observations to safe actions. This field report presents an OpenArm-based mobile manipulation prototype for laboratory-style tasks, built by integrating dual OpenArm manipulators with a mobile base, vertical slide, RGB-D sensing, lidar-based mapping, ROS2/MoveIt execution, and profile-defined skill interfaces. The system is organized around representation handoffs: natural language requests are constrained into registered skill calls, sensor observations are grounded into maps and object poses, object priors provide role and skill constraints, and runtime bindings compile validated skills into executable motion goals. We use dry-run traces and startup checks to evaluate this integration path, showing how the prototype exposes missing calibration, incomplete object assets, and unfinished real-scene visual grounding as explicit deployment blockers. These intermediate representations serve as practical debugging interfaces for integrating language, perception, planning, and robot safety in embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06729v1">AtlasVLA: Persistent World-Ego State Modeling for Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-08-07
    </div>
    <details class="paper-abstract">
      While Vision-Language-Action (VLA) models have advanced embodied AI, their fundamentally reactive paradigm severely limits performance in partially observable and long-horizon tasks. When restricted to a single wrist-mounted camera, they inevitably suffer from perception forgetting as objects exit the field of view, and temporal task-progress forgetting} during multi-step execution. To overcome these bottlenecks, we propose AtlasVLA, a novel framework that transitions from direct reactive manipulation to proactive reasoning through a persistent world-ego state. AtlasVLA features a dual-memory architecture: a 4D Persistent World State Memory that lifts transient 2D observations into a globally updated, voxel-hashed spatial state to resolve visual blind spots, and an Ego-Working State Memory that tracks historical ego state and task progress. By conditioning a diffusion transformer (DiT) on this joint World-Ego state, AtlasVLA enables robust spatial reasoning. Extensive evaluations across LIBERO, RLBench, and real-world benchmarks demonstrate that AtlasVLA achieves state-of-the-art performance using solely a wrist camera. Remarkably, it decisively outperforms multi-view baselines, yielding absolute success rate improvements of 9.4% on LIBERO-Long and 17.5% in real-world long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06161v1">iARCS: Iterative Agentic RL for Controllable 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 15 pages, 9 figures, 4 tables. Includes appendix
    </div>
    <details class="paper-abstract">
      Synthetic 3D scene generation is increasingly used as a data source for computer vision and embodied AI, but existing generators often optimize perceptual realism without reliably satisfying task-critical functional constraints. This mismatch limits the usefulness of synthetic data for downstream training, where accessibility, traversability, and spatial rule compliance are often essential. We present iARCS, an iterative agentic reinforcement learning framework that adapts a pretrained scene generator to natural-language task requirements. iARCS uses a two-stage strategy: universal-reward pretraining to improve physical plausibility and layout quality, followed by task-specific fine-tuning with LLM-generated reward programs that are iteratively refined from training feedback. Experiments show improved constraint fidelity on walkability, reachability, and clearance-focused tasks, effective task-specific constraint optimization, and competitive scene diversity. We further show that data generated by iARCS improves a base generator, supporting its value as a practical synthetic data generation tool rather than only a controllable scene editing method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06088v1">IcFuzz: Fuzzing Isaac Sim with Semantic Stage Guidance and Multi-level Mutation</a></div>
    <div class="paper-meta">
      📅 2026-08-06
      | 💬 Accepted at the 41st IEEE/ACM International Conference on Automated Software Engineering (ASE 2026)
    </div>
    <details class="paper-abstract">
      Robotics simulators serve as a foundational infrastructure for embodied AI, facilitating safe and scalable robotic system development. NVIDIA Isaac Sim has emerged as one of the most popular simulators, distinguished by its GPU-accelerated physics engine and photorealistic rendering, which enable high-fidelity modeling of complex environments. However, its inherent complexity inevitably introduces software bugs that can compromise simulation reliability. Existing fuzzing approaches struggle to test Isaac Sim effectively due to challenges of context-aware object semantics, hierarchical simulation control, and a vast simulation state space. In this paper, we propose IcFuzz, the first fuzzing approach for Isaac Sim. IcFuzz first performs an LLM-based semantic stage segmentation, decomposing simulation programs into structured stages that capture context-aware object semantics. Guided by this information, IcFuzz designs multi-level mutation operators to systematically exercise the simulator across hierarchical granularities. To efficiently navigate the vast simulation state space, IcFuzz employs a multi-armed bandit algorithm to adaptively schedule mutation operators. Experimental results show that IcFuzz outperforms the baselines in terms of both code coverage and bug detection. Specifically, IcFuzz achieves approximately 190\%--205\% of the code coverage of the baselines and detects an average of 3.7 unique crashes over three rounds of 12-hour tests, while no crashes are detected by the baselines. Moreover, IcFuzz has uncovered 11 bugs over approximately four months, 9 of which have been confirmed or fixed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21124v2">PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Current multimodal LLMs process audio as a mono stream, ignoring the rich spatial information essential for embodied AI. Existing spatial audio models, conversely, are constrained to fixed microphone geometries, preventing deployment across diverse devices. We present PhaseCoder, a transformer-only spatial audio encoder that is agnostic to microphone geometry. PhaseCoder takes raw multichannel audio and microphone coordinates as inputs to perform localization and produces robust spatial embeddings. We demonstrate that Gemma 3n LLM can be fine-tuned to reason over "Spatial Audio Tokens" produced by PhaseCoder. We show our encoder achieves state-of-the-art results on microphone-invariant localization benchmarks and, for the first time, enables an LLM to perform complex spatial reasoning and targeted transcription tasks from an arbitrary microphone array.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2607.16465v2">Berkeley and Heiserman as an Unexhausted Architecture for Embodied Machine Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Edmund C. Berkeley is usually remembered as a mediator between symbolic logic and early computing, yet that standard description understates the scope of his work. This paper argues for a stronger reading: Berkeley should also be understood as an early theorist of embodied machine intelligence. Across Berkeley's major writings on symbolic logic, machine intelligence, living robots, and Squee, intelligence appears not as disembodied symbol manipulation alone but as the organized coordination of sensing, storage, calculation, control, state, and action in physically realized machines. The paper's first contribution is interpretive: it reconstructs Berkeley as a thinker of machine architecture, temporally extended behavior, and environment-coupled control. Its second contribution is comparative: it reads Berkeley alongside David L. Heiserman to recover a shared descriptive scheme centered on sensing, state or memory, control, action, and adaptation. Its third contribution is critical: it uses that scheme to assess current embodied-AI discourse. The broader claim is that contemporary LLM-centered robotics often demonstrates impressive capability without an equally explicit account of persistence, recoverability, maintenance, and structured modification of conduct through experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.04428v1">Deltoris: Enabling Real-time VLA Inference in Embodied AI via Bit-level Sparsity and Speculative Inference</a></div>
    <div class="paper-meta">
      📅 2026-08-05
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) models have emerged as a key component in embodied AI. Among existing approaches, diffusion-based VLA models achieve superior motion quality and generalization. However, diffusion-based VLA models are compute-intensive and must run at high control frequency, e.g., 50-200 Hz. Thus, it imposes strict latency and energy constraints on edge devices. In this work, we present Deltoris, an algorithm-hardware co-design framework for efficient diffusion-based VLA inference. First, we exploit the temporal similarity of consecutive inputs and propose a \textit{temporal-aware bit-sparsity} algorithm that computes only the differences between consecutive inputs, eliminating redundant bit-level operations. To further address the extra off-chip traffic introduced by our algorithm, we propose a \textit{speculative inference} technique, which amortizes data loading across multiple control steps. Lastly, to support these techniques, we co-design a dedicated accelerator with customized 1D systolic bit-serial PE arrays that eliminate PE workload imbalance. Our evaluation shows that Deltoris achieves up to 34.2$\times$ speedup over mobile GPUs and 6.1$\times$ over prior accelerators, while maintaining comparable accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25459v2">GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning</a></div>
    <div class="paper-meta">
      📅 2026-08-04
      | 💬 Robotics: Science and Systems 2026
    </div>
    <details class="paper-abstract">
      Embodied AI research is undergoing a shift toward vision-centric perceptual paradigms. While massively parallel simulators have catalyzed breakthroughs in proprioception-based locomotion, their potential remains largely untapped for vision-informed tasks due to the prohibitive computational overhead of large-scale photorealistic rendering. Furthermore, the creation of simulation-ready 3D assets heavily relies on labor-intensive manual modeling, while the significant sim-to-real physical gap hinders the transfer of contact-rich manipulation policies. To address these bottlenecks, we propose GS-Playground, a multi-modal simulation framework designed to accelerate end-to-end perceptual learning. We develop a novel high-performance parallel physics engine, specifically designed to integrate with a batch 3D Gaussian Splatting (3DGS) rendering pipeline to ensure high-fidelity synchronization. Our system achieves a breakthrough throughput of 10^4 FPS at 640x480 resolution, significantly lowering the barrier for large-scale visual RL. Additionally, we introduce an automated Real2Sim workflow that reconstructs photorealistic, physically consistent, and memory-efficient environments, streamlining the generation of complex simulation-ready scenes. Extensive experiments on locomotion, navigation, and manipulation demonstrate that GS-Playground effectively bridges the perceptual and physical gaps across diverse embodied tasks. Project homepage: https://gsplayground.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.01127v2">MiniWorld: Democratizing the Training of Video World Models from Scratch</a></div>
    <div class="paper-meta">
      📅 2026-08-04
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Video world models predict future observations conditioned on historical observations and control signals, enabling long-horizon generation through autoregressive state transitions. Unlike conventional video generation models that primarily capture visual appearance and motion, video world models learn the underlying dynamics governing environment evolution under agent actions, providing a foundation for embodied AI and interactive simulation. Recent progress has largely relied on adapting pretrained video generation models through post-training or distillation. Although effective, these approaches often require complex training pipelines, substantial computational resources, and suffer from the mismatch between bidirectional pretraining and causal streaming inference. Recent studies have shown that training autoregressive video world models from scratch is feasible and scalable. However, the community still lacks a lightweight, transparent, and fully reproducible baseline trainable end-to-end with modest computational resources. We present MiniWorld, a reproducible framework for training streaming video world models from scratch. MiniWorld employs a block-causal Video Diffusion Transformer trained with Flow Matching in the latent space of a pretrained Video VAE. Building on Diffusion Forcing, it adopts a chunk-wise non-decreasing noise schedule and two-stage continued training to improve temporal modeling and stability. During inference, MiniWorld combines a rolling KV cache with pipelined asynchronous denoising for efficient streaming generation under bounded computation. The entire model can be trained within several days on a single 8-GPU server. By releasing the training and inference codebase and pretrained checkpoints, we hope MiniWorld will facilitate future research on video world modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.02191v1">DerainSplat: Feed-Forward Clean 3D Gaussian Splatting from Sparse Rainy Views</a></div>
    <div class="paper-meta">
      📅 2026-08-03
    </div>
    <details class="paper-abstract">
      Although image deraining has advanced substantially, existing methods mainly focus on 2D image restoration. As spatial intelligence applications such as embodied AI and autonomous driving continue to emerge, reconstructing clean 3D scenes from sparse rainy views in a feed-forward manner becomes increasingly important. Existing feed-forward 3D Gaussian Splatting (3DGS) methods often assume clean inputs and collapse under rainy conditions. To this end, we present \textbf{\textit{DerainSplat}}, a feed-forward framework that reconstructs clean 3D scenes from only a few rainy views. To support this task, we build a large-scale multi-view derain dataset through a four-stage synthesis pipeline that sequentially models overcast illumination, depth-dependent haze, rain streaks, and lens raindrops, producing privileged weather factors. We introduce a weather net that predicts the weather factors from rainy context and yields two support maps. Scene support modulates cross-view cost-volume matching, while radiance support drives depth-aligned appearance fusion to fill corrupted pixels. The derived geometry evidence further attenuates Gaussian opacity to reduce spurious structures. A rainy cycle consistency re-renders clean views using the predicted factors and aligns them with rainy inputs. Extensive experiments show that \textbf{\textit{DerainSplat}} outperforms existing methods on various datasets, including RealEstate10K, ACID, Mip-NeRF360, and real-world rainy scenes, with strong cross-dataset generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08831v2">DiffPhysCam: Differentiable Physics-Based Camera Simulation for Inverse Rendering and Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-08-01
      | 💬 37 pages, 24 figures, and 5 tables. Code of DiffPhysCam-CamCaliExp: https://github.com/DanielYamChen/DiffPhysCam-CamCaliExp Code of DiffPhysCam-NovelViewSynthesis: https://github.com/DanielYamChen/DiffPhysCam-NovelViewSynthesis Data of DiffPhysCam_Data: https://huggingface.co/datasets/DanielYamChen/DiffPhysCam_Data Simulation video: https://youtu.be/gQwSMrdmHJI
    </div>
    <details class="paper-abstract">
      Generating synthetic images that closely mimic those from real cameras is instrumental in training visual models and enabling end-to-end visuomotor learning. We introduce DiffPhysCam, a differentiable camera simulator designed to support robotics and embodied AI applications by enabling gradient-based optimization in visual perception pipelines. Differentiable rendering also allows inverse reconstruction of real-world scenes as digital twins, facilitating simulation-based robotics training. Existing virtual cameras offer limited control over intrinsic settings, have difficulty capturing optical artifacts, and lack tunable calibration parameters. DiffPhysCam addresses these limitations through a multi-stage pipeline that provides fine-grained control over camera settings, models key optical effects such as defocus blur, and supports calibration with real-world data. It enables both forward rendering for image synthesis and inverse rendering for 3D scene reconstruction, including mesh and material texture optimization. We show that DiffPhysCam enhances robotic perception performance in synthetic image tasks. As an illustrative example, we create a digital twin of a real-world scene using inverse rendering and use it to set up a virtual experiment in a multi-physics simulation, in which we demonstrate navigation of an autonomous ground vehicle using images generated by DiffPhysCam. The code, data, and output results associated with this paper are available online for reproducibility at Chen, Bo-Hsun (2026a,b,c).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.00625v1">Learning-Based Motion Planning for Dynamic Environments: From Foundational Algorithms to Emerging Paradigms</a></div>
    <div class="paper-meta">
      📅 2026-08-01
    </div>
    <details class="paper-abstract">
      Motion planning in dynamic environments is a fundamental problem in robotics, aiming to generate safe and efficient paths, trajectories, or control actions in the presence of moving obstacles, uncertain predictions, and multi-agent interactions. It has broad applications in autonomous driving, service robotics, warehouse logistics, human-robot collaboration, crowd navigation, and multi-robot systems. This survey reviews representative works published primarily between 2015 and 2025, with a particular focus on how recent learning-based advances extend, complement, or interact with classical planning foundations. We first revisit classical planning methods as algorithmic foundations and reference frameworks for learning-based extensions. We then propose a role-of-learning taxonomy that categorizes existing methods according to how learning participates in the planning pipeline, including direct policy learning, learning-augmented classical planning, hybrid planning, and training enhancement methods. For each category, we summarize the main problem settings, representative algorithms, key ideas, integration mechanisms, strengths, and limitations. We further analyze how observation representations, prediction uncertainty, interaction modeling, planner integration, safety constraints, and training strategies shape learning-based motion planning in dynamic environments. Finally, we discuss open challenges and future directions, including sim-to-real gap, safe and certifiable planning, dense crowd navigation, perception-planning coupling, and embodied AI.
    </details>
</div>
