# embodied ai - 2026_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.21572v1">PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects</a></div>
    <div class="paper-meta">
      📅 2026-05-20
      | 💬 Project page: https://physx-omni.github.io/
    </div>
    <details class="paper-abstract">
      Simulation-ready physical 3D assets have emerged as a promising direction owing to their broad applicability in downstream tasks. However, most existing 3D generation methods either neglect physical properties or are limited to a single asset category, e.g., rigid, deformable, or articulated objects. To address these limitations, we introduce PhysX-Omni, a unified framework for simulation-ready physical 3D generation across diverse asset types. Specifically, we develop a novel and efficient geometry representation tailored for Vision-Language Models, which directly encodes high-resolution 3D structures without compression, significantly improving generation performance. In addition, we construct the first general simulation-ready 3D dataset, PhysXVerse, covering diverse indoor and outdoor categories. Furthermore, to comprehensively and flexibly evaluate both generative and understanding capabilities in the wild, we propose PhysX-Bench, which encompasses six key attributes: geometry, absolute scale, material, affordance, kinematics, and function description. Extensive experiments with conventional metrics and PhysX-Bench show that PhysX-Omni performs strongly in both generation and understanding. Moreover, additional studies further validate the potential of PhysX-Omni for applications in simulation-ready scene generation and robotic policy learning. We believe PhysX-Omni can significantly advance a wide range of downstream applications, particularly in embodied AI and physics-based simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16137v2">STABLE: Simulation-Ready Tabletop Layout Generation via a Semantics-Physics Dual System</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 ICML 2026
    </div>
    <details class="paper-abstract">
      Generating simulation-ready tabletop scenes from task instructions is an intriguing and promising research direction in the field of Embodied AI. However, existing task-to-scene generation methods rely exclusively on large language models (LLMs) to predict scene layouts, inevitably yielding object collisions or floating due to LLMs' inherent limitations in 3D spatial reasoning. In this paper, we present STABLE, a semantics-physics dual-system tailored for simulation-ready tabletop scene generation. STABLE consists of two complementary modules: (i) a Semantic Reasoner, a fine-tuned LLM trained on a structured tabletop scene dataset to generate coarse layouts from input task instructions, and (ii) a Physics Corrector, a physics-aware flow-based denoising model that outputs pose updates to refine layouts, which ensures the physical plausibility of scenes while preserves semantic alignment with task instructions. STABLE adopts a progressive generation paradigm: by alternating between the Semantic Reasoner and Physics Corrector, it incrementally expands the scene from task-critical objects to background objects. Experiments demonstrate that STABLE successfully generates simulation-ready tabletop scenes that strictly conform to task instructions and significantly enhances the physical validity of scenes over prior art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19986v1">Beyond Binary Success: A Diagnostic Meta-Evaluation Framework for Fine-Grained Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 Project page: https://metafine.github.io/
    </div>
    <details class="paper-abstract">
      Fine-grained manipulation marks a regime where global scene context no longer suffices, and success hinges on the tight coupling of local attribute grounding, high-fidelity spatial perception, and constraint-respecting motor execution. However, current embodied AI benchmarks collapse these capacities into binary success rates, systematically inflating reported capabilities by up to 70% and masking the architectural bottlenecks that impede real-world deployment. We introduce MetaFine, a diagnostic meta-evaluation framework that disentangles manipulation competency along three axes: understanding, perception, and controlled behavior. Built on a compositional task graph, MetaFine absorbs heterogeneous external benchmarks and reconstructs them into diagnostic scenarios of varying complexity under a unified protocol. Evaluating state-of-the-art vision-language-action (VLA) models through this lens exposes severe dimension-specific failures invisible to conventional metrics. Through targeted causal intervention, we identify the visual encoder's ability to preserve local spatial structure as a key bottleneck for fine-grained precision: improving it directly unlocks previously inaccessible manipulation capabilities without modifying downstream policies. MetaFine further supports hybrid real-sim validation, using limited paired real-world rollouts to calibrate scalable simulation-based estimates for more stable physical benchmarking. By shifting evaluation from ranking to diagnosis, MetaFine turns benchmarking into an actionable compass for repairing the layered capacities underlying genuine physical dexterity. The MetaFine framework, benchmarks, and supporting resources will be publicly released at our project page: https://metafine.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19728v1">Aero-World: Action-Conditioned Aerial Video Generation from Inertial Controls</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Foundation video models produce visually impressive results, but their use in embodied AI remains limited because they are primarily trained on natural language rather than low-level control signals. This limitation is especially pronounced for aerial flight, where motion occurs in unconstrained 6-DoF space and small errors in ego-motion can produce large trajectory drift. Generating aerial videos that follow fine-grained inertial actions can support scalable training and evaluation of aerial agents by providing a controllable proxy for real-world or expensive simulation data. To address this problem, we propose \textbf{Aero-World}, a method for converting a pretrained image-to-video diffusion model into a controllable aerial video generator. Aero-World injects sequences of translational acceleration and angular velocity into a pretrained latent diffusion transformer through an action-token stream. A frozen latent-space Physics Probe, trained independently on real video--IMU pairs, provides differentiable inertial-consistency supervision during LoRA finetuning while avoiding computationally expensive video decoding. We further propose \textbf{AeroBench}, a benchmark for evaluating whether generated drone videos adhere to low-level action signals. AeroBench uses Action Alignment Score (AAS) to measure agreement with commanded inertial actions and Physical Consistency Rate (PCR) to measure temporal motion stability. On AeroBench, Aero-World improves mean AAS from 57.7 to 63.6 over action-only finetuning and gives a stronger quality-control trade-off than AirScape, with lower FVD (596.5 vs. 1058.6), higher SSIM (0.595 vs. 0.505), and higher Flow-IMU correlation (0.44 vs. 0.20). These results suggest that frozen Physics Probe supervision is a practical mechanism for adapting pretrained video generators toward more action-aligned aerial motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19587v1">SceneCode: Executable World Programs for Editable Indoor Scenes with Articulated Objects</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Indoor scene synthesis underpins embodied AI, robotic manipulation, and simulation-based policy evaluation, where a useful scene must specify not only what the environment looks like, but also how its objects are structured. Existing pipelines, however, typically represent generated content as static meshes and inherit articulation only from curated asset libraries, which limits object-level controllability and prevents new interactable assets from being produced on demand. We address this gap by formulating physically interactable indoor scene synthesis as programmatic world generation, and present SceneCode, a framework that compiles a natural language prompt into an executable, code-driven indoor world rather than a collection of opaque meshes. A room-level agentic backbone first turns the prompt into a structured house layout and emits per-object AssetRequests through a planner--designer--critic loop. Each request is then routed to one of five code-generation strategies and converted into a synthesized part-wise Blender Python programs that are validated through an execution-guided repair-and-refine loop. The resulting programs are compiled into simulation-ready assets, and exported as SDF for physics simulation. A persistent scene-state registry links object requests, executable programs, rendered geometry, and simulation assets, turning scene assembly into a traceable and locally editable world-building process. We evaluate SceneCode across scene-level synthesis, object-level asset quality, human judgment, and downstream robot interaction. Results show that executable world programs improve prompt-faithful indoor scene generation and produce assets with cleaner mesh structure, and simulator-loadable articulation metadata. Project page: https://scene-code.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11234v2">RoomPilot: Controllable Indoor Scene Synthesis via Multimodal Semantic Parsing</a></div>
    <div class="paper-meta">
      📅 2026-05-19
      | 💬 30 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Generating controllable indoor scenes is fundamental to applications in game development, architectural visualization, and embodied AI. However, existing approaches either support a limited input modalities or rely on implicit generation processes that hinder precise control over scene structure and semantics. To address these limitations, we introduce RoomPilot, a unified framework for controllable indoor scene synthesis from multi-modal inputs, including textual descriptions and CAD floor plans. RoomPilot maps heterogeneous inputs into an Indoor Domain-Specific Language (IDSL), which serves as a structured and interpretable semantic representation for describing indoor scenes. Built upon IDSL, RoomPilot presents a hierarchical synthesis pipeline that progressively organizes scenes at the building, room, and object levels, promoting structural coherence and functional consistency across multi-room layouts. Moreover, RoomPilot constructs a curated asset dataset with rich semantic annotations to support high-quality scene synthesis, improving visual realism and appearance consistency. Extensive experiments demonstrate effective multi-modal understanding, fine-grained controllability in scene generation, and improved physical consistency and visual fidelity, marking a significant step toward controllable 3D indoor scene synthesis. Code and model will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.19328v1">RoboJailBench: Benchmarking Adversarial Attacks and Defenses in Embodied Robotic Agents</a></div>
    <div class="paper-meta">
      📅 2026-05-19
    </div>
    <details class="paper-abstract">
      Recent advances in Vision-Language Models (VLMs) facilitate a new class of embodied AI systems, where these models are integrated into physical platforms, e.g. robots and autonomous vehicles, to interpret visual scenes and execute natural language commands in diverse environments. Previous research has introduced jailbreak attacks and defenses for embodied AI. Their evaluations, however, rely on ad-hoc datasets, limited metrics, and emphasize attack success while neglecting the trade-off between security and the ability to follow benign commands. Existing benchmarks and evaluation frameworks either target traditional chat-based models or focus on non-adversarial safety evaluation for embodied AI; neither captures the adversarial risks, inputs, consequences, and evaluation criteria necessary for jailbreak attacks in embodied AI systems. In this paper, we address this gap with RoboJailBench, which consists of three core components. We establish a security taxonomy derived from ISO standards, regulatory rules, and documented incidents. This effort yields 18 categories of security violation consequences for embodied AI. We introduce an intent contrast dataset pipeline that augments existing datasets with paired adversarial and benign goals to measure both security and utility. Lastly, we provide an evolving repository with standardized metrics and a unified process for assessing and integrating new attacks and defenses. With this benchmark, we construct a new taxonomy-balanced dataset and augment five existing datasets. We integrate four attacks and two defenses to evaluate their performance on leading embodied VLMs. This benchmark provides the first standardized evaluation framework for jailbreak attacks in embodied AI and supports future research. We release our code, datasets, and artifacts, and maintain a leaderboard at https://purseclab.github.io/benchmark-for-robotics-security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15975v2">Learning Bilevel Policies over Symbolic World Models for Long-Horizon Planning</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      We tackle the challenge of building embodied AI agents that can reliably solve long-horizon planning problems. Imitation learning from demonstrations has shown itself to be effective in training robots to solve a diversity of complex tasks requiring fine motor control and manipulation over low-level (LL), continuous environments. Yet, it remains a difficult endeavour to generate long-horizon plans from imitation learning alone. In contrast, high-level (HL), symbolic abstractions facilitate efficient and interpretable long-horizon planning. We propose to combine the strengths of LL imitation learning for manipulation and control, and HL symbolic abstractions for long-horizon planning. We realise this idea via \emph{bilevel policies} of the form $(π^{\mathrm{hl}}, π^{\mathrm{ll}})$, consisting of a neural policy $π^{\mathrm{ll}}$ learned from LL demonstrations, and an HL symbolic policy $π^{\mathrm{hl}}$ that is constructed from symbolic abstractions of the LL demonstrations combined with inductive generalisation. We implement these ideas in the BISON system. Experiments on extended MetaWorld benchmarks demonstrate that BISON generalises to long horizons and problems with greater numbers of objects than those solved by VLA and end-to-end methods, and is more time and memory efficient in training and inference. Notably, when ignoring LL execution, BISON's HL policies can solve HL problems with 10,000 relevant objects in under a minute. Project page: https://dillonzchen.github.io/bison
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18722v1">Dexora: Open-source VLA for High-DoF Bimanual Dexterity</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accpeted by ICRA 2026
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have recently become a central direction in embodied AI, but current systems are restricted to either dual-gripper control or single-arm dexterous hand manipulation. While low-dimensional gripper control can often be handled with simpler methods, high-dimensional dexterous hand control benefits greatly from full end-to-end VLA learning. In this work, we introduce Dexora, the first open-source VLA system that natively targets dual-arm, dual-hand high-DoF manipulation. We design a hybrid teleoperation pipeline that decouples gross arm kinematics (captured with a custom exoskeleton backpack) from fine finger motion (markerless hand tracking via Apple Vision Pro), and that drives both a physical dual-arm dual-hand platform and an identical MuJoCo digital twin. Using that interface, we assemble a large training corpus: an embodiment-matched synthetic corpus (100K simulated trajectories, 6.5M frames) and a real-world dataset of 10K teleoperated episodes (2.92M frames). To mitigate noisy teleoperation demonstrations, we propose a data-quality-aware training recipe: an offline discriminator provides clip-level weights for diffusion-transformer policy training, down-weighting low-quality demonstrations. Empirically, Dexora outperforms competitive VLA baselines on both basic and dexterous benchmarks (e.g., average dexterous success 66.7% vs. 51.7%), attains 90% success on basic tasks, and shows robust out-of-distribution and cross-embodiment generalization. Ablations confirm the importance of real data and the discriminator for dexterity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18593v1">Not What You Asked For: Typographic Attacks in Household Robot Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 10 pages, 1 figure, IEEE conference format
    </div>
    <details class="paper-abstract">
      Open-vocabulary embodied AI agents increasingly rely on vision-language models such as CLIP for object perception and task grounding. However, the shared embedding space that enables this flexibility introduces a structural vulnerability to typographic attacks, where printed text in a physical scene semantically overrides visual judgment. While prior work has quantified this threat in static 2D benchmarks and 3D navigation tasks, its impact on the full Sense-Plan-Act pipeline of household robot manipulation remains unexplored. This work evaluates typographic attacks in a Habitat-based simulation using the HomeRobot benchmark. We introduce a decoupled perception architecture that exposes a frozen CLIP encoder to adversarial stickers while maintaining geometric grounding via DETIC. In a controlled evaluation pool of 59 attributable episodes, the attack achieves an overall Attack Success Rate (ASR) of 67.8%, rising to 70.0% among fully successful episodes, under uncontrolled viewing angles and occlusion with no perceptual optimization. Critically, we find that perceptual errors propagate through the persistent 3D semantic map to produce kinetic failures, defined here as physically executed grasping and transport of the wrong object driven by an adversarially poisoned semantic state. In these cases, the robot physically grasps and delivers the wrong object to a target receptacle. These results establish typographic misclassification as a real, measurable, and physically consequential threat to the safety of modular manipulation pipelines that prior typographic attack research has left unexamined.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18451v1">Code-as-Room: Generating 3D Rooms from Top-Down View Images via Agentic Code Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-05-18
    </div>
    <details class="paper-abstract">
      Designing realistic and functional 3D indoor rooms is essential for a wide range of applications, including interior design, virtual reality, gaming, and embodied AI. While recent MLLM-based approaches have shown great potential for 3D room synthesis from textual descriptions or reference images, text-based methods struggle to capture precise spatial information, and existing image-conditioned agents suffer from instability and infinite looping when tasked with holistic room generation from top-down views. To address these limitations, we propose Code-as-Room, an MLLM-based agentic framework equipped with a structured execution harness, which represents 3D rooms with Blender codes. Given a top-down room image, the framework parses the reference image to extract scene elements and their spatial relationships, and synthesizes executable Blender code for geometry, materials, and lighting in a principled, multi-stage pipeline. A cross-stage memory module is maintained throughout to mitigate context forgetting inherent to existing agent-based frameworks. We further introduce a dedicated benchmark for code-based 3D room synthesis, encompassing various evaluation protocols. Based on our benchmark, comprehensive comparisons against existing agent-based methods are conducted to validate the effectiveness of our proposed execution harness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18423v1">REBAR: Reference Ethical Benchmark for Autonomy Readiness</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 To be presented at the 2026 Workshop on Robot Ethics - Ethical, Legal and User Perspectives in Robotics and Automation (WOROBET)
    </div>
    <details class="paper-abstract">
      As autonomous systems grow more advanced, objective metrics to evaluate their ethical and legal compliance are critical for informing end users of their limitations and ensuring accountability of those who misuse them. Current ethical embodied AI frameworks remain mostly qualitative, focusing on system design (through safety guardrails or targeted red teaming), and the realized guardrails often directly disallow unsafe behavior without providing the user with an override or interpretable reason. Instead, there is a need for computable metrics through rigorous testing that allow a user to determine the applicability of the system to the task. To address this gap, we introduce the Reference Ethical Benchmark for Autonomy Readiness (REBAR), a quantitative test and evaluation framework for autonomous systems. REBAR maps operating metrics into a computable Autonomy Readiness Level (ARL) rubric that can quantify ethical performance. Key innovations of the framework include a neuro-symbolic Large Language Model (LLM) approach to calculate and explain the ethical difficulty of scenarios, LLM-driven at-scale generation of test instances, and a versatile, photorealistic simulation environment. By evaluating white-box autonomy solutions through this rigorous testing pipeline, REBAR delivers an objective and repeatable benchmark score, bridging the gap between abstract principles and verifiable, accountable autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18407v1">Qumus: Realization of An Embodied AI Quantum Material Experimentalist</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 29 Pages in total. Supplementary Demo Videos are available at https://qumus.ai
    </div>
    <details class="paper-abstract">
      While modern Large Language Models (LLMs) and agentic artificial intelligence (AI) have demonstrated transformative capabilities in digital domains, the realization of embodied AI capable of real-world scientific discovery remains a difficult frontier. The advancements are hindered by the inherent complexity of integrating high-level reasoning, multimodal information processing and real-time physical execution. Here we introduce Qumus, the first AI quantum materials experimentalist. Physically embodied within a robotic mini-laboratory, Qumus is an intelligent, multimodal, and multi-agent system designed for the creation and nano-processing of atomically thin two-dimensional (2D) materials and stacked van der Waals (vdW) structures. Qumus autonomously navigates the full scientific cycle, from hypothesis generation and protocol planning to multi-step experimental execution, result analysis and reporting, acting as an experimentalist. Markedly, the system has achieved, for the first time, the AI-creation of graphene, as well as the first AI-fabrication of complex nanodevices including atomically thin field-effect transistors via vdW stacking. Qumus excels at these tasks by demonstrating autonomous error correction and closed-loop experimentation. Our results establish a generalizable framework for self-improving embodied AI systems that learn directly from the quantum world, opening a pathway toward accelerated discovery in quantum materials, electronics and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.18194v1">Beyond the Cartesian Illusion: Testing Two-Stage Multi-Modal Theory of Mind under Perceptual Bottlenecks</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 17 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While Multi-Modal Large Language Models (MLLMs) demonstrate impressive capabilities in general reasoning, their embodied spatial intelligence remains hampered by a "Cartesian Illusion" - a reliance on text-based probability distributions that lack grounded, 3D topological understanding. This limitation is starkly exposed in multi-agent environments, which demand more than just scene perception; they require second-order Theory of Mind (ToM). Specifically, an Agent A must be able to infer Agent B's belief about the environment, governed strictly by Agent B's physical orientation and sensory limitations. In this paper, we probe the limits of two-stage spatial inference in MLLMs through a novel audio-visual task: requiring Agent A to predict Agent B's estimation of A's relative location. To solve this, we propose an Epistemic Sensory Bottleneck module that abandons rigid, rule-based coordinate transformations. Instead, we introduce an Anchor-Based Embodied Spatial Decomposition Chain-of-Thought (CoT). This guides the MLLM through a "geometric-to-semantic" projection, forcing it to first establish B's local coordinate system and then dynamically weight visual and auditory modalities based on whether A falls within B's visual frustum. Extensive evaluations reveal that while current MLLMs fundamentally struggle with spatial symmetry and out-of-view ambiguities (establishing a rigorous zero-shot baseline of 42% accuracy), our sensory-bounded reasoning chain robustly outperforms pure egocentric and allocentric baselines. By systematically benchmarking these perceptual bottlenecks, our work exposes the current limits of MLLM spatial reasoning and establishes a foundational paradigm for epistemic, modality-aware inference in Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14371v2">OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Embodied AI agents increasingly require parallel execution of multiple tasks, such as manipulation, conversation, and memory construction, from shared observations under distinct time constraints. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) architecturally support such heterogeneous outputs, yet existing inference systems fail to achieve efficient multi-task parallelism for on-device deployment because of redundant computation and resource contention. We identify isolated KV cache management as the root cause. To address this, we propose unified KV cache management, an inference design that treats the KV cache as a first-class shared resource across tasks and over time. This abstraction enables two key optimizations: cross-task KV sharing eliminates redundant prefill of shared observations, while cross-frame continuous batching decouples variable-length language decoding from fixed-rate action generation across control cycles. We implement this design for $π_{0.5}$, a popular MoT VLA, and evaluate it on both NVIDIA GeForce RTX 4090 and Jetson AGX Thor, two representative platforms for on-device VLA inference. OxyGen achieves up to 3.7$\times$ speedup over isolated execution, delivering over 200 tokens/s language throughput and 70 Hz action frequency simultaneously without degrading action quality, and we further validate the gains on a real humanoid robot with on-board Jetson AGX Thor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16278v2">DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-05-18
      | 💬 Accepted by CVPR 2026, Project Page: https://thinklab-sjtu.github.io/DriveMoE/
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $π_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$π_0$. Specifically, we add Vision MoE to Drive-$π_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$π_0$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14504v2">When Robots Do the Chores: A Benchmark and Agent for Long-Horizon Household Task Execution</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      Long-horizon household tasks demand robust high-level planning and sustained reasoning capabilities, which are largely overlooked by existing embodied AI benchmarks that emphasize short-horizon navigation or manipulation and rely on fixed task categories. We introduce LongAct, a benchmark designed to evaluate planning-level autonomy in long-horizon household tasks specified through free-form instructions. By abstracting away embodiment-specific low-level control, LongAct isolates high-level cognitive capabilities such as instruction understanding, dependency management, memory maintenance, and adaptive planning. We further propose HoloMind, a VLM-driven agent with a DAG-based long-horizon hierarchical planner, a Multimodal Spatial Memory for persistent world modeling, an Episodic Memory for experience reuse, and a global Critic for reflective supervision. Experiments with GPT-5 and Qwen3-VL models show that HoloMind substantially improves long-horizon performance while reducing reliance on model scale. Even top models achieve only 59% goal completion and 16% full-task success, underscoring the difficulty of LongAct and the need for stronger long-horizon planning in embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17033v1">Generalizable and Actionable Parts Pose Estimation with Symmetry Annotation-Free Learning Strategy</a></div>
    <div class="paper-meta">
      📅 2026-05-16
      | 💬 Accepted as a poster at the Forty-third International Conference on Machine Learning (ICML 2026)
    </div>
    <details class="paper-abstract">
      Urgently needed generalizable robot object interaction and manipulation requires high-quality Cross-Category object perception. As a pioneer of this area, Generalizable and Actionable Parts (GAParts) understanding has attracted increasing attention from relevant researchers. However, most recent works either have insufficient design regarding the symmetry issue or require rich symmetry annotation, which severely impedes precise GAPart pose estimation in data-lacking scenarios. In this paper, we propose SAFAG, a novel Symmetry Annotation-Free framework for Generalizable and Actionable Parts Pose Estimation. Specifically, we suggest a stepwise refinement two-stage framework for candidate-to-final quaternion regression, and tackle the symmetry prediction as a probability distribution problem with self-supervised learning strategy. The experimental results demonstrate the superior performance and robustness of our SAFAG. We believe that our work has the enormous potential to be applied in many areas of embodied AI system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16899v1">LASAR: Towards Spatio-temporal Reasoning with Latent Cognitive Map</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      A fundamental challenge in embodied AI is verifying if agents build internal models of spatial structure or merely learn to mimic task-specific expert trajectories. This is critical as foundational approaches rooted in action-centric tasks (e.g., VLN) and reasoning-centric tasks (e.g., EQA) often share a common limitation: they lack a learning signal that forces them to encode fine-grained spatial relationships (like topology or distance) over long-range, fragmented experiences. To address this, we first propose LASAR, an architecture featuring a dual-memory system designed to maintain both episodic experiences and a semantic cognitive map. We then introduce Spatio-temporal Contextual Representation Learning (ST-CRL), a contrastive objective designed to train this architecture. ST-CRL leverages spatio-temporal cues from cognitive queries generated through annotated spatio-temporal context in simulation to build sample pairs, thereby forming the internal cognitive map from the agent's experiences. Experiments demonstrate that our method achieves 2\%-3.5\% gains in both zero-shot generalization on standard VLN-CE and VSI-Bench benchmarks. We also demonstrate that our proposed cognitive map has high self-consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04227v2">Continual Learning for VLMs: A Survey and Taxonomy Beyond Forgetting</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) and the recent surge of Multimodal Large Language Models (MLLMs) have revolutionized artificial intelligence with unprecedented cross-modal alignment and zero-shot generalization. However, enabling them to learn continually from non-stationary data remains a major challenge, as their cross-modal alignment and generalization capabilities are particularly vulnerable to catastrophic forgetting. Unlike traditional unimodal continual learning (CL), VLMs face unique challenges such as cross-modal feature drift, parameter interference due to shared architectures, and zero-shot capability erosion. Furthermore, generative MLLMs exhibit a unique ``alignment tax,'' where catastrophic forgetting manifests not merely as factual amnesia, but as a systemic collapse of deep Chain-of-Thought (CoT) reasoning. This survey presents the first comprehensive, diagnostic review bridging continual learning for both predictive VLMs and generative MLLMs. We systematically deconstruct the aforementioned failure modes and propose a challenge-driven taxonomy comprising four core paradigms: (1) Multi-Modal Replay Strategies addressing explicit and implicit memory drift; (2) Cross-Modal Regularization enforcing topological and geometric alignment; (3) Parameter-Efficient Adaptation} utilizing dynamic routing and subspace projections; and the emerging (4) Model Fusion and Decoupling paradigms. We critically analyze the evolution of evaluation protocols, highlighting the essential shift toward dual-track benchmarks (Domain vs. Ability CL) and micro-diagnostic CoT evaluations. Finally, we chart a roadmap for future research, emphasizing compositional zero-shot learning, embodied AI with sensor fusion, and autonomous agentic ecosystems. All resources are available at: https://github.com/YuyangSunshine/Awesome-Continual-learning-of-Vision-Language-Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16797v1">EgoKit: Towards Unified Low-Cost Egocentric Data Collection with Heterogeneous Devices</a></div>
    <div class="paper-meta">
      📅 2026-05-16
    </div>
    <details class="paper-abstract">
      Egocentric video is increasingly used as a data source for robot learning, activity understanding, and embodied AI research, but collecting it at scale remains fragmented in practice: each candidate host device, such as an Android phone, iPhone, iPad, smart glasses, or extended reality (XR) headset, exposes a different SDK, a different policy on raw camera access, and different limitations on external USB cameras and on-device tracking. Synchronized ego-view and wrist-view capture is therefore typically obtained by either committing to a single proprietary platform or building one-off rigs that do not transfer across devices. To address this gap, we present EgoKit, a toolkit that exposes the same egocentric recording workflow across six heterogeneous host devices. Across all supported devices, EgoKit presents the same recording interaction and produces locally stored video with a uniform log format; on XR headsets, it additionally logs head pose and OpenXR-standard 26-joint hand tracking aligned to the video streams. The companion accessories, including two wrist cameras with mounts, a head strap, and a USB-C hub, add wrist-view capture to any supported host without custom hardware fabrication. EgoKit is available at \url{https://egokit.chuange.org/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.15391v1">PanoWorld: Geometry-Consistent Panoramic Video World Modeling</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      We present PanoWorld, a panoramic video world model that generates geometry-consistent 360$\degree$ video from a single image and a caption. Existing panoramic video methods optimize primarily for visual realism and do not explicitly constrain the underlying 3D scene state, producing outputs that appear plausible yet exhibit inconsistent depth, broken correspondences, and implausible motion across the spherical surface. We address this gap by framing panoramic video generation as a geometry- and dynamics-consistent latent state modeling problem rather than pure visual synthesis. Building on a pre-trained perspective video world model, we introduce two lightweight regularizers: a depth consistency loss against pseudo ground-truth panoramic depth, and a trajectory consistency loss that supervises the 3D world-frame positions of tracked points across time. We further apply spherical-geometry-aware adaptation to the conditioning and positional encoding. We additionally introduce PanoGeo, a unified geometry-aware panoramic video dataset with consistent depth, trajectory, and prompt annotations across diverse real and synthetic sources, used for both training and stratified evaluation. Experiments show that PanoWorld improves geometric consistency over prior panoramic generation methods while maintaining competitive visual realism, establishing that panoramic video generation must be treated as a geometric modeling problem to support the holistic spatial understanding requirements of embodied AI applications. Code is available at https://github.com/ostadabbas/PanoWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13016v3">SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      A truly capable AI system must do more than detect objects or recognize activities in isolation. It must form unified, grounded representations of who is acting, what they are doing, and when and where these actions unfold. These representations provide the perceptual bedrock for high-level reasoning, planning, and embodied interaction in the real world. Building such agents is central to long-horizon goals in embodied AI and robotics. Current video benchmarks evaluate fragments of these capabilities in isolation. They focus on either spatial grounding, object tracking, or temporal localization. As a result, they cannot rigorously measure progress on their joint, multi-instance integration. We introduce Spatio-temporal Video Action Grounding (SVAG), a task and benchmark that explicitly targets this unified competence by requiring models to simultaneously detect, track, and temporally localize all objects that satisfy a natural language query in complex, multi-actor scenes. To support this task, we construct SVAG-Bench. It comprises 688 videos, 19,590 verified annotations, and 903 unique action verbs drawn from crowded urban environments, wildlife, and traffic surveillance. Each video has on average 28.5 action-centric queries. This yields the densest annotation among comparable video grounding benchmarks and enables fine-grained evaluation of multi-actor disambiguation, temporal overlap, and action compositionality. Annotations are produced by a pipeline that combines expert manual labeling, GPT-3.5 paraphrase augmentation, and human verification to ensure both linguistic diversity and correctness. We further release SVAGEval, a standardized multi-referent evaluation toolkit. We also introduce SVAGFormer, a strong modular baseline architecture for SVAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14625v1">Digital Twin Synchronization Over Mobile Embodied AI Network With Agentic Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Efficient digital twin (DT) synchronization relies on maintaining high-fidelity virtual representations with minimal age of information (AoI). However, the synergistic potential of cooperative sensing and autonomous mobility of the sensing agent remains underexplored in existing DT synchronization frameworks. In this paper, we propose an agentic AI-empowered mobile embodied AI network (MEAN) framework for DT synchronization. In the proposed hybrid architecture, the base station (BS) conducts global orchestration, while the agents autonomously execute a five-stage closed-loop workflow: move-to-sense, cooperative sensing, onboard semantic processing, channel-aware mobility, and uplink transmission. To optimize synchronization performance, we formulate a joint topology dispatching and multidimensional resource allocation problem aimed at minimizing the maximum twin deviation across regions, subject to heterogeneous sensing fidelity and energy budget constraints. To tackle this, we develop a hierarchical two-layer optimization algorithm, where the outer-layer refines multi-agent assignment via a dynamic matching game, and the inner-layer iteratively optimizes the continuous resources. Extensive simulation results verify the convergence of the proposed algorithm and demonstrate its substantial superiority over multiple baseline schemes in reducing synchronization deviation. Furthermore, the results reveal that semantic compression serves as a vital substitute for channel resources in latency reduction under constrained bandwidth, while autonomous velocity adaptation provides an essential degree of freedom for the system to navigate the fundamental energy-time trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14556v1">TeachAnything: A Multimodal Crowdsourcing Platform for Training Embodied AI Agents in Symmetrical Reality</a></div>
    <div class="paper-meta">
      📅 2026-05-14
      | 💬 5 pages, 3 figures. Accepted as an IEEE VR 2026 Poster
    </div>
    <details class="paper-abstract">
      Symmetrical Reality (SR) is emerging as a future trend for human-agent coexistence, placing higher demands on agents to acquire human-like intelligence. It calls for richer and more diverse human guidance. We introduce a three-stage demonstration paradigm integrating multimodal demonstration signals. Building on this paradigm, we developed TeachAnything, a cloud-based, crowdsourcing-oriented demonstration platform with physics simulation capable of collecting diverse demonstration data across varied scenes, tasks, and embodiments. By unifying virtual and physical interactions through both methodological design and physics simulation, the system serves as a practical foundation for developing embodied agents aligned with Symmetrical Reality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13276v2">D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to the resource conflict between high-fidelity physical simulation and the intensive VRAM/bandwidth demands of deep learning. This conflict often leaves overall throughput constrained by execution-phase inefficiencies. To address these challenges, we propose D-VLA, a high-concurrency, low-latency distributed RL framework for large-scale embodied foundation models. D-VLA introduces "Plane Decoupling," physically isolating high-frequency training data from low-frequency weight control to eliminate interference between simulation and optimization. We further design a four-thread asynchronous "Swimlane" pipeline, enabling full parallel overlap of sampling, inference, gradient computation, and parameter distribution. Additionally, a dual-pool VRAM management model and topology-aware replication resolve memory fragmentation and optimize communication efficiency. Experiments on benchmarks like LIBERO show that D-VLA significantly outperforms mainstream RL frameworks in throughput and sampling efficiency for billion-parameter VLA models. In trillion-parameter scalability tests, our framework maintains exceptional stability and linear speedup, providing a robust system for high-performance general-purpose embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14462v1">Real2Sim in HOI: Toward Physically Plausible HOI Reconstruction from Monocular Videos</a></div>
    <div class="paper-meta">
      📅 2026-05-14
    </div>
    <details class="paper-abstract">
      Recovering 4D human-object interaction (HOI) from monocular video is a key step toward scalable 3D content creation, embodied AI, and simulation-based learning. Recent methods can reconstruct temporally coherent human and object trajectories, but these trajectories often remain visual artifacts while failing to preserve stable contact, functional manipulation, or physical plausibility when used as reference motions for humanoid-object simulation. This reveals a fundamental interaction gap: HOI reconstruction should not stop at tracking a human and an object, but should recover the relation that makes their motion a coherent interaction. We introduce $\textbf{HA-HOI}$, a framework for reconstructing physically plausible 4D HOI animation from in-the-wild monocular videos. Instead of treating the human and object as independent entities in an ambiguous monocular 3D space, we propose a $\textit{human-first, object-follow}$ formulation. The human motion is recovered as the interaction anchor, and the object is reconstructed, aligned, and refined relative to the human action. The resulting kinematic trajectory is then projected into a physics-based humanoid-object simulation, where it acts as a teacher trajectory for stable physical rollout. Across benchmark and in-the-wild videos, $\textbf{HA-HOI}$ improves human-object alignment, contact consistency, temporal stability, and simulation readiness over prior monocular HOI reconstruction methods. By moving beyond visually plausible trajectory recovery toward physically grounded interaction animation, our work takes a step toward turning general monocular HOI videos into scalable demonstrations for humanoid-object behavior. Project page: https://knoxzhao.github.io/real2sim_in_HOI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13586v1">HetScene: Heterogeneity-Aware Diffusion for Dense Indoor Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Generating controllable and physically plausible indoor scenes is a pivotal prerequisite for constructing high-fidelity simulation environments for embodied AI. However, existing deeplearning-based methods usually treat all objects as homogeneous instances within a unified generation process. While effective for sparse and simplistic layouts, they struggle to model realistic layouts with dense object arrangements and complex spatial dependencies, leadingto limited scalability and degraded physical plausibility. To deal with these challenges, we revisit indoor layout generation from the perspective of structural heterogeneity and decompose the objects into primary objects and secondary objects according to their distinct roles in shaping a scene. Based on this decomposition, we propose HetScene, a heterogeneous two-stage generation framework that decouples indoor layout synthesis into Structural Layout Generation (SLG) and Contextual Layout Generation (CLG). SLG first generates globally coherent structural layouts with only primary objects conditioned on text descriptions, top-down binary room masks, and spatial relation graphs, establishing a stable global macro-skeleton of large core furniture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23018v2">AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Web-scale 3D asset collections are abundant but rarely deployment-ready, suffering from arbitrary metric scaling, incorrect pivots, brittle geometry, and incomplete textures, defects that limit their use in embodied AI, robotics, and spatial computing. We present AmaraSpatial-10K, a dataset of over 10,000 synthetic 3D assets optimised for zero-shot deployment. Each asset ships as a metric-scaled, deterministically anchored .glb with separated PBR maps, a convex collision hull, a paired reference image, and multi-sentence text metadata. Alongside the dataset we introduce a reusable evaluation suite for 3D asset banks, a continuous Scale Plausibility Score (SPS), an LLM Concept Density metric, anchor-error auditing, and a cross-modal CLIP coherence protocol, and apply it to AmaraSpatial-10K alongside matched subsets of Objaverse, HSSD, ABO, and GSO. AmaraSpatial-10K improves CLIP Recall@5 by $3.4\times$ over Objaverse ($0.612$ vs. $0.181$, median rank $267 \rightarrow 3$), achieves a $99.1\%$ physics-stability rate under Habitat-Sim with $\sim 20\times$ wall-time speed-up, and produces zero-overlap scenes when used as a drop-in asset bank for Holodeck. Controlled ablations on the same asset bank attribute the retrieval gain to description richness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.13129v1">Rigel3D: Rig-aware Latents for Animation-Ready 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-13
    </div>
    <details class="paper-abstract">
      Recent 3D generative models can synthesize high-quality assets, but their outputs are typically static: they lack the skeletal rigs, joint hierarchies, and skinning weights required for animation. This limits their use in games, film, simulation, virtual agents, and embodied AI, where assets must not only look plausible but also move plausibly. We introduce Rigel3D, a generative method for animation-ready 3D assets represented as rigged meshes. Unlike post-hoc auto-rigging methods that attach rigs to completed shapes, our method jointly models geometry and rig structure through coupled surface and skeleton structured latent representations. A rig-aware autoencoder decodes these representations into mesh geometry, skeleton topology, joint coordinates, and skinning weights, while a two-stage latent generative model synthesizes both surface and skeleton representations for image-conditioned generation. To support downstream animation workflows, we further introduce an open-vocabulary joint labeling module that embeds generated joints into a shared vision-language space, enabling correspondence to arbitrary retargeting templates. Experiments on large-scale rigged asset datasets demonstrate that our method generates diverse, high-quality animation-ready assets and outperforms existing rigging baselines across multiple metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.12090v1">World Action Models: The Next Frontier in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-05-12
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have achieved strong semantic generalization for embodied policy learning, yet they learn reactive observation-to-action mappings without explicitly modeling how the physical world evolves under intervention. A growing body of work addresses this limitation by integrating world models, predictive models of environment dynamics, into the action generation pipeline. We term this emerging paradigm World Action Models (WAMs): embodied foundation models that unify predictive state modeling with action generation, targeting a joint distribution over future states and actions rather than actions alone. However, the literature remains fragmented across architectures, learning objectives, and application scenarios, lacking a unified conceptual framework. We formally define WAMs and disambiguate them from related concepts, and trace the foundations and early integration of VLA and world model research that gave rise to this paradigm. We organize existing methods into a structured taxonomy of Cascaded and Joint WAMs, with further subdivision by generation modality, conditioning mechanism, and action decoding strategy. We systematically analyze the data ecosystem fueling WAMs development, spanning robot teleoperation, portable human demonstrations, simulation, and internet-scale egocentric video, and synthesize emerging evaluation protocols organized around visual fidelity, physical commonsense, and action plausibility. Overall, this survey provides the first systematic account of the WAMs landscape, clarifies key architectural paradigms and their trade-offs, and identifies open challenges and future opportunities for this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.10653v1">Embodied AI in Action: Insights from SAE World Congress 2026 on Safety, Trust, Robotics, and Real-World Deployment</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence is rapidly moving from research into real-world systems such as autonomous vehicles, mobile robots, and industrial machines. As these systems become more capable of perceiving, deciding, and acting in dynamic environments, they also introduce new challenges in safety, trust, governance, and operational reliability. This white paper summarizes key insights from the SAE World Congress 2026 panel session \textit{Embodied AI in Action}, which brought together experts from automotive, robotics, artificial intelligence, and safety engineering. The discussion highlighted the need to treat embodied AI as a systems challenge requiring engineering rigor, lifecycle governance, human-centered design, and evolving standards. The paper provides practical perspectives for executives, policymakers, and technical leaders seeking to adopt embodied AI responsibly. The panel reached broad agreement that long-term success will depend not only on advances in AI capability, but equally on safe and trustworthy deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22963v3">Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary</a></div>
    <div class="paper-meta">
      📅 2026-05-11
      | 💬 Project page: https://humanoidlla.github.io/
    </div>
    <details class="paper-abstract">
      Enabling humanoid robots to follow free-form natural language commands is a critical step toward seamless human-robot interaction and general-purpose embodied AI. However, existing methods remain limited, often constrained to simple instructions or forced to sacrifice motion diversity for physical plausibility. To address this gap, we present Humanoid-LLA, a Large Language Action model that translates unconstrained natural language directly into executable whole-body motions for humanoid robots. Our approach tackles two core challenges: paired language-humanoid motion data scarcity and physical instability. First, we bridge high-level language semantics with physically-grounded control by learning a unified human-humanoid motion vocabulary. Second, we introduce a novel two-stage fine-tuning framework that begins with supervised motion Chain-of-Thought learning, followed by reinforcement learning refined with physical feedback to ensure robustness and stability. Extensive evaluation in simulation and real-world cross-embodiment experiments demonstrates that Humanoid-LLA achieves superior generalization to novel language commands and diverse motion generation while maintaining high physical fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09954v1">JODA: Composable Joint Dynamics for Articulated Objects</a></div>
    <div class="paper-meta">
      📅 2026-05-11
    </div>
    <details class="paper-abstract">
      Articulated objects used in simulation and embodied AI are typically specified by geometry and kinematic structure, but lack the fine-grained dynamical effects that govern realistic mechanical behavior, such as frictional holding, detents, soft closing, and snap latching. Existing approaches either ignore the detailed structure of dynamics entirely, or use simple models with limited expressiveness. We introduce JODA, a framework for generating joint-level dynamics as a structured three-channel field over the joint degree of freedom, capturing conservative forces, dry friction, and damping. Instantiated using shape-constrained piecewise cubic interpolation (PCHIP), this formulation defines a compact and expressive function space that is both interpretable and compatible with differentiable simulation. Building on this representation, we develop methods for inferring and refining joint dynamics from multimodal inputs. Given visual observations and joint context, a vision-language model proposes structured dynamical primitives, which are composed into a unified dynamics field. The resulting representation supports both direct manipulation and gradient-based refinement. We demonstrate that JODA enables plausible and controllable modeling of diverse joint behaviors, providing a unified interface for inference, editing, and optimization. Code and example assets with their generated profiles will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.09348v1">HOME-KGQA: A Benchmark Dataset for Multimodal Knowledge Graph Question Answering on Household Daily Activities</a></div>
    <div class="paper-meta">
      📅 2026-05-10
      | 💬 12 pages, 4 figures, 7 tables, accepted at LREC2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) provide flexible natural language processing capabilities, while knowledge graphs (KGs) offer explicit and structured knowledge. Integrating these two in a complementary manner enables the development of reliable and verifiable AI systems. In particular, knowledge graph question answering (KGQA) has attracted attention as a means to reduce LLM hallucinations and to leverage knowledge beyond the training data. However, existing KGQA benchmark datasets are biased toward encyclopedic knowledge, limited to a single modality, and lack fine-grained spatiotemporal data, which limits their applicability to real-world scenarios targeted by Embodied AI. We introduce HOME-KGQA, a novel KGQA benchmark dataset built on a multimodal KG of daily household activities. HOME-KGQA consists of complex, multi-hop natural language questions paired with graph database query languages. Compared to existing benchmarks, it includes more challenging questions that involve multi-level spatiotemporal reasoning, multimodal grounding, and aggregate functions. Experimental results show that the LLM-based KGQA methods fail to achieve performance comparable to that on existing datasets when evaluated on HOME-KGQA. This highlights significant challenges that should be addressed for the real-world deployment of KGQA systems. Our dataset is available at https://github.com/aistairc/home-kgqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23629v2">From Visual Synthesis to Interactive Worlds: Toward Production-Ready 3D Asset Generation</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Preprint. Jiafeng Wu and Zhuofan Lou contributed equally. Project page: https://christinebobby.github.io/production-ready-3d-survey/
    </div>
    <details class="paper-abstract">
      Three-dimensional content generation has progressed from producing isolated, visually plausible shapes to constructing structured assets that can be deployed in real-time interactive environments. This trajectory is driven by converging demands from game development, embodied AI, world simulation, digital twins, and spatial computing, all of which require 3D content that goes beyond surface appearance to satisfy engine-level constraints on topology, UV parameterization, physically based materials, skeletal rigging, and physics-aware scene layout. Despite rapid advances in generative modeling, a persistent gap separates the outputs of current methods from the production-ready standard expected by interactive applications. This survey addresses that gap by organizing the literature around the asset production pipeline rather than algorithmic families. Along the horizontal axis we distinguish three asset tiers, namely general objects, characters, and scenes, while the vertical axis traces each tier through the full production lifecycle from data foundations and geometry synthesis through topology optimization, UV unwrapping, PBR appearance, rigging, and scene assembly. Through this two-dimensional taxonomy we assess not only what current methods can generate but whether their outputs are directly usable in downstream engines and simulation platforms. We further consolidate evaluation metrics and protocols that span geometric fidelity, appearance quality, asset usability, and scene-level physical plausibility. The survey concludes by identifying open challenges in data quality, generation controllability, end-to-end assetization, and physically grounded generation, and by situating production-ready 3D content as foundational infrastructure for emerging interactive world models and embodied intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13451v5">MapNav: A Novel Memory Representation via Annotated Semantic Maps for Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-05-09
    </div>
    <details class="paper-abstract">
      Vision-and-language navigation (VLN) is a key task in Embodied AI, requiring agents to navigate diverse and unseen environments while following natural language instructions. Traditional approaches rely heavily on historical observations as spatio-temporal contexts for decision making, leading to significant storage and computational overhead. In this paper, we introduce MapNav, a novel end-to-end VLN model that leverages Annotated Semantic Map (ASM) to replace historical frames. Specifically, our approach constructs a top-down semantic map at the start of each episode and update it at each timestep, allowing for precise object mapping and structured navigation information. Then, we enhance this map with explicit textual labels for key regions, transforming abstract semantics into clear navigation cues and generate our ASM. MapNav agent using the constructed ASM as input, and use the powerful end-to-end capabilities of VLM to empower VLN. Extensive experiments demonstrate that MapNav achieves state-of-the-art (SOTA) performance in both simulated and real-world environments, validating the effectiveness of our method. Moreover, we will release our ASM generation source code and dataset to ensure reproducibility, contributing valuable resources to the field. We believe that our proposed MapNav can be used as a new memory representation method in VLN, paving the way for future research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08799v1">ElasticFlow: One-Step Physics-Consistent Policy with Elastic Time Horizons for Language-Guided Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-09
      | 💬 Accepted to Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Diffusion policies have demonstrated exceptional performance in embodied AI. However, their iterative denoising process results in high latency, and existing acceleration methods often sacrifice physical consistency. To address this, we propose ElasticFlow, a distillation-free, physics-consistent one-step policy framework. We reconstruct the Mean Field Theory by directly modeling the average velocity field, enabling a direct single-step mapping from noise to action. Addressing the Temporal Heterogeneity of robotic tasks, we introduce the Elastic Time Horizons mechanism. This mechanism effectively overcomes Spectral Bias by explicitly encoding control granularity, achieving efficient alignment between semantic instructions and physical execution horizons. Experiments on benchmarks such as LIBERO, CALVIN, and RoboTwin demonstrate that ElasticFlow achieves efficient 1-NFE inference (approximately 71Hz). Furthermore, it outperforms state-of-the-art methods, including OpenVLA and $π_0$, on long-horizon tasks, highlighting its potential for efficient, robust, and semantically aligned control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26509v3">3D Generation for Embodied AI and Robotic Simulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 27 pages, 11 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Embodied AI and robotic systems increasingly depend on scalable, diverse, and physically grounded 3D content for simulation-based training and real-world deployment. While 3D generative modeling has advanced rapidly, embodied applications impose requirements far beyond visual realism: generated objects must carry kinematic structure and material properties, scenes must support interaction and task execution, and the resulting content must bridge the gap between simulation and reality. This survey reviews 3D generation for embodied AI and organizes the literature around three roles that 3D generation plays in embodied systems. In Data Generator, 3D generation produces simulation-ready objects and assets, including articulated, physically grounded, and deformable content for downstream interaction; in Simulation Environments, it constructs interactive and task-oriented worlds, spanning structure-aware, controllable, and agentic scene generation; and in Sim2Real Bridge, it supports digital twin reconstruction, data augmentation, and synthetic demonstrations for downstream robot learning and real-world transfer. We also show that the field is shifting from visual realism toward interaction readiness, and we identify the main bottlenecks, including limited physical annotations, the gap between geometric quality and physical validity, fragmented evaluation, and the persistent sim-to-real divide, that must be addressed for 3D generation to become a dependable foundation for embodied intelligence. Our project page is at https://3dgen4robot.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.02357v2">Channel-Level Relation to Attentive Aggregation with Neighborhood-Homogeneity Constraint for Point Cloud Analysis</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      In 3D point cloud understanding, the core challenge lies in accurately capturing discriminative features within complex neighborhoods, which directly affects the execution precision of downstream tasks such as embodied AI and autonomous driving. Existing methods explore feature correlation discrimination but are limited to point-level spatial distribution or channel responses, enabling only coarse-grained level evaluation. For modern multi-scale point cloud networks, such coarse-grained metrics inevitably incur significant information loss in deeper layers. To address this, we propose PointCRA, a novel network with a channel-level metric-based enhancement mechanism. Our core idea is to introduce temporal trend variation as a new evaluation dimension to avoid the information loss caused by weight dimension collapse in existing spatial and channel attention mechanisms. On this basis, we construct a multi-level calibration framework guided by neighborhood homogeneity for weight calibration, and design a dedicated loss function to enhance channel discriminability.PointCRA leverages intrinsic feature priors to adaptively correct feature aggregation, offering interpretability with low parameter overhead. Our method is transferable, interpretable, and efficient. We validate the proposed method on diverse datasets and benchmark models, and further demonstrate its rationality through extensive analytical experiments. Our PointCRA achieves 77.5\% mIoU on the S3DIS dataset, 90.4\% OA on the ScanObjectNN dataset, and 87.4\% instance mIoU on the ShapeNetPart dataset. The code and pretrained weights are publicly available on GitHub: https://github.com/AGENT9717/PointCRA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.08279v1">LaWM: Least Action World Models for Long-Horizon Physical Consistency from Visual Observations</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Learning predictive world models from visual observations is a core problem in embodied AI, with applications to model-based reinforcement learning and robotic planning. Existing latent world models typically generate future states with unconstrained neural transition functions, while modern video generation systems often prioritize perceptual plausibility or introduce physical structure through auxiliary losses, external guidance, or separate dynamics modules. As a result, long-horizon rollouts can remain weakly grounded in the physical principles that govern real dynamics, leading to compounding error, energy drift, and physically inconsistent futures. We propose Least Action World Models (LaWM), a latent world-modeling framework that operationalizes the Principle of Least Action in learned visual latent space: future rollouts are governed by a learned Lagrangian action functional rather than produced only by an unconstrained transition predictor. Our main technical realization is a latent variational integrator: LaWM encodes observations into learned generalized coordinates, learns a latent discrete Lagrangian over consecutive latent states, constructs a discrete action functional, and advances prediction by solving the corresponding discrete integration condition. Thus, physical structure is not merely used to score, regularize, or constrain a completed trajectory; it defines the latent transition rule itself. Because the transition is induced by a discrete variational principle, LaWM provides a structure-preserving bias for long-horizon visual prediction. Across physics-clean synthetic dynamics and embodied robot interaction benchmarks, LaWM improves physical invariance, background consistency, motion smoothness, and appearance and geometric prediction metrics over video-generation and world-model baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07325v1">CSR: Infinite-Horizon Real-Time Policies with Massive Cached State Representations</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 Extended Technical Report for Paper Accepted to IEEE RA-L
    </div>
    <details class="paper-abstract">
      Deploying massive large language models (LLMs) as continuous cognitive engines for robotics is bottlenecked by the time-to-first-token (TTFT) latency required to process extensive state histories. Existing solutions like RAG or sliding windows compromise global context or incur prohibitive re-computation costs. We formalize the optimal task structure for minimizing latency and theoretically prove that prefix stability, incremental extensibility, and asynchronous state reconciliation are necessary conditions for real-time performance. Building on these proofs, we introduce the Cached State Representation (CSR) framework as the practical instantiation of these properties, ensuring optimal KV-cache reuse. To sustain these properties over infinite horizons, we further propose an Asynchronous State Reconciliation (ASR) algorithm that offloads state memory eviction to a parallel computational resource to eliminate latency spikes. On a physical robot wirelessly connected to an on-premise GPU server, CSR achieves a 26-fold latency reduction (14.67s to 0.56s) for 120K token contexts with a 235B parameter model compared to a standard baseline. On an embodied AI benchmark, we achieve SOTA recall (0.836 vs. 0.459) while maintaining RAG-level latency. ASR is validated to sustain bounded, spike-free TTFT over 10 eviction cycles in continuous real-world operation. Together, CSR and ASR enable massive LLMs to function as continuously operating, high-frequency (> 2 Hz) embodied policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07306v1">BioProVLA-Agent: An Affordable, Protocol-Driven, Vision-Enhanced VLA-Enabled Embodied Multi-Agent System with Closed-Loop-Capable Reasoning for Biological Laboratory Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-05-08
      | 💬 16 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Biological laboratory automation can reduce repetitive manual work and improve reproducibility, but reliable embodied execution in wet-lab environments remains challenging. Protocols are often unstructured, labware is frequently transparent or reflective, and multi-step procedures require state-aware execution beyond one-shot instruction following. Existing robotic systems often rely on costly hardware, fixed workflows, dedicated instruments, or robotics-oriented interfaces. Here, we introduce BioProVLA-Agent, an affordable, protocol-driven, vision-enhanced embodied multi-agent system enabled by Vision-Language-Action (VLA) models for biological manipulation. The system uses protocols as the task interface and integrates protocol parsing, visual state verification, and embodied execution in a closed-loop workflow. A Tailored LLM Protocol Agent converts protocols into verifiable subtasks; a VLM-RAG Verification Agent assesses readiness and completion using observations, robot states, retrieved knowledge, and success/failure examples; and a VLA Embodied Agent executes verified subtasks through a lightweight policy. To improve robustness under wet-lab visual perturbations, we develop AugSmolVLA, an online augmentation strategy targeting transparent labware, reflections, illumination shifts, and overexposure. We evaluate the system on a hierarchical benchmark covering 15 atomic tasks, 6 composite workflows, and 3 bimanual tasks, including tube loading, sorting, waste disposal, cap twisting, and liquid pouring. Across normal and high-exposure settings, AugSmolVLA improves execution stability over ACT, X-VLA, and the original SmolVLA, especially for precise placement, transparent-object manipulation, composite workflows, and visually degraded scenes. These results suggest a practical route toward accessible, protocol-centered, and verification-capable embodied AI for biological manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.07088v1">Membership Inference Attacks on Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2026-05-08
    </div>
    <details class="paper-abstract">
      Membership inference attacks (MIAs) have been extensively studied in large language models (LLMs) and vision-language models (VLMs), yet their implications for vision-language-action (VLA) models remain largely unexplored. VLA models differ from standard LLMs and VLMs in several important ways: they are often fine-tuned for many epochs on relatively small embodied datasets, operate over constrained and structured action spaces, and expose action outputs that can be observed as executable behaviors and temporally correlated trajectories. These characteristics suggest a distinct and potentially more informative attack surface for membership inference. In this work, we present the first systematic study of MIAs against VLA systems. We formalize two membership inference settings for VLA models: sample-level inference over individual transition samples and trajectory-level inference over complete embodied demonstrations. We further develop a suite of attack methods under multiple access regimes, including strict black-box access. Our attacks exploit both classic MIA signals, such as token likelihood, and VLA-specific signals, such as observable action errors and temporal motion patterns. Across multiple VLA benchmarks and representative VLA models, these attacks achieve strong inference performance, showing that VLA models are highly vulnerable to membership inference. Notably, black-box attacks based only on generated actions achieve strong performance, highlighting a practical privacy risk for deployed embodied AI systems. Our findings reveal a previously underexplored privacy risk in robotic and embodied AI, and underscore the need for dedicated privacy evaluation and defenses for VLA models.
    </details>
</div>
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
