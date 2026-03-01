# embodied ai - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23205v1">EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a humanoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.19565v2">DICArt: Advancing Category-level Articulated Object Pose Estimation in Discrete State-Spaces</a></div>
    <div class="paper-meta">
      📅 2026-02-26
    </div>
    <details class="paper-abstract">
      Articulated object pose estimation is a core task in embodied AI. Existing methods typically regress poses in a continuous space, but often struggle with 1) navigating a large, complex search space and 2) failing to incorporate intrinsic kinematic constraints. In this work, we introduce DICArt (DIsCrete Diffusion for Articulation Pose Estimation), a novel framework that formulates pose estimation as a conditional discrete diffusion process. Instead of operating in a continuous domain, DICArt progressively denoises a noisy pose representation through a learned reverse diffusion procedure to recover the GT pose. To improve modeling fidelity, we propose a flexible flow decider that dynamically determines whether each token should be denoised or reset, effectively balancing the real and noise distributions during diffusion. Additionally, we incorporate a hierarchical kinematic coupling strategy, estimating the pose of each rigid part hierarchically to respect the object's kinematic structure. We validate DICArt on both synthetic and real-world datasets. Experimental results demonstrate its superior performance and robustness. By integrating discrete generative modeling with structural priors, DICArt offers a new paradigm for reliable category-level 6D pose estimation in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00288v3">TimeBlind: A Spatio-Temporal Compositionality Benchmark for Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-25
      | 💬 For code and data, see https://baiqi-li.github.io/timeblind_project/
    </div>
    <details class="paper-abstract">
      Fine-grained spatio-temporal understanding is essential for video reasoning and embodied AI. Yet, while Multimodal Large Language Models (MLLMs) master static semantics, their grasp of temporal dynamics remains brittle. We present TimeBlind, a diagnostic benchmark for compositional spatio-temporal understanding. Inspired by cognitive science, TimeBlind categorizes fine-grained temporal understanding into three levels: recognizing atomic events, characterizing event properties, and reasoning about event interdependencies. Unlike benchmarks that conflate recognition with temporal reasoning, TimeBlind leverages a minimal-pairs paradigm: video pairs share identical static visual content but differ solely in temporal structure, utilizing complementary questions to neutralize language priors. Evaluating over 20 state-of-the-art MLLMs (e.g., GPT-5, Gemini 3 Pro) on 600 curated instances (2400 video-question pairs), reveals that the Instance Accuracy (correctly distinguishing both videos in a pair) of the best performing MLLM is only 48.2%, far below the human performance (98.2%). These results demonstrate that even frontier models rely heavily on static visual shortcuts rather than genuine temporal logic, positioning TimeBlind as a vital diagnostic tool for next-generation video understanding. Dataset and code are available at https://baiqi-li.github.io/timeblind_project/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21161v1">ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking</a></div>
    <div class="paper-meta">
      📅 2026-02-24
      | 💬 8 pages, 5 figures, accepted by the 2026 IEEE International Conference on Robotics and Automation
    </div>
    <details class="paper-abstract">
      Classical robotic systems typically rely on custom planners designed for constrained environments. While effective in restricted settings, these systems lack generalization capabilities, limiting the scalability of embodied AI and general-purpose robots. Recent data-driven Vision-Language-Action (VLA) approaches aim to learn policies from large-scale simulation and real-world data. However, the continuous action space of the physical world significantly exceeds the representational capacity of linguistic tokens, making it unclear if scaling data alone can yield general robotic intelligence. To address this gap, we propose ActionReasoning, an LLM-driven framework that performs explicit action reasoning to produce physics-consistent, prior-guided decisions for robotic manipulation. ActionReasoning leverages the physical priors and real-world knowledge already encoded in Large Language Models (LLMs) and structures them within a multi-agent architecture. We instantiate this framework on a tractable case study of brick stacking, where the environment states are assumed to be already accurately measured. The environmental states are then serialized and passed to a multi-agent LLM framework that generates physics-aware action plans. The experiments demonstrate that the proposed multi-agent LLM framework enables stable brick placement while shifting effort from low-level domain-specific coding to high-level tool invocation and prompting, highlighting its potential for broader generalization. This work introduces a promising approach to bridging perception and execution in robotic manipulation by integrating physical reasoning with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18884v1">TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models</a></div>
    <div class="paper-meta">
      📅 2026-02-21
      | 💬 Accepted to ICLR 2026. 17 pages. Code, data, and models are available at: https://github.com/Stephen-gzk/TPRU
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs), particularly smaller, deployable variants, exhibit a critical deficiency in understanding temporal and procedural visual data, a bottleneck hindering their application in real-world embodied AI. This gap is largely caused by a systemic failure in training paradigms, which lack large-scale, procedurally coherent data. To address this problem, we introduce TPRU, a large-scale dataset sourced from diverse embodied scenarios such as robotic manipulation and GUI navigation. TPRU is systematically designed to cultivate temporal reasoning through three complementary tasks: Temporal Reordering, Next-Frame Prediction, and Previous-Frame Review. A key feature is the inclusion of challenging negative samples, compelling models to transition from passive observation to active, cross-modal validation. We leverage TPRU with a reinforcement learning (RL) fine-tuning methodology, specifically targeting the enhancement of resource-efficient models. Experiments show our approach yields dramatic gains: on our manually curated TPRU-Test, the accuracy of TPRU-7B soars from 50.33\% to 75.70\%, a state-of-the-art result that significantly outperforms vastly larger baselines, including GPT-4o. Crucially, these capabilities generalize effectively, demonstrating substantial improvements on established benchmarks. The codebase is available at https://github.com/Stephen-gzk/TPRU/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14135v3">ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI</a></div>
    <div class="paper-meta">
      📅 2026-02-21
    </div>
    <details class="paper-abstract">
      Rapidly evolving AI exhibits increasingly strong autonomy and goal-directed capabilities, accompanied by derivative systemic risks that are more unpredictable, difficult to control, and potentially irreversible. However, current AI safety evaluation systems suffer from critical limitations such as restricted risk dimensions and failed frontier risk detection. The lagging safety benchmarks and alignment technologies can hardly address the complex challenges posed by cutting-edge AI models. To bridge this gap, we propose the "ForesightSafety Bench" AI Safety Evaluation Framework, beginning with 7 major Fundamental Safety pillars and progressively extends to advanced Embodied AI Safety, AI4Science Safety, Social and Environmental AI risks, Catastrophic and Existential Risks, as well as 8 critical industrial safety domains, forming a total of 94 refined risk dimensions. To date, the benchmark has accumulated tens of thousands of structured risk data points and assessment results, establishing a widely encompassing, hierarchically clear, and dynamically evolving AI safety evaluation framework. Based on this benchmark, we conduct systematic evaluation and in-depth analysis of over twenty mainstream advanced large models, identifying key risk patterns and their capability boundaries. The safety capability evaluation results reveals the widespread safety vulnerabilities of frontier AI across multiple pillars, particularly focusing on Risky Agentic Autonomy, AI4Science Safety, Embodied AI Safety, Social AI Safety and Catastrophic and Existential Risks. Our benchmark is released at https://github.com/Beijing-AISI/ForesightSafety-Bench. The project website is available at https://foresightsafety-bench.beijing-aisi.ac.cn/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10116v2">SAGE: Scalable Agentic 3D Scene Generation for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 Project Page: https://research.nvidia.com/labs/dir/sage/
    </div>
    <details class="paper-abstract">
      Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: https://research.nvidia.com/labs/dir/sage/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18397v1">How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf</a></div>
    <div class="paper-meta">
      📅 2026-02-20
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have recently demonstrated impressive capabilities across various embodied AI tasks. While deploying VLA models on real-world robots imposes strict real-time inference constraints, the inference performance landscape of VLA remains poorly understood due to the large combinatorial space of model architectures and inference systems. In this paper, we ask a fundamental research question: How should we design future VLA models and systems to support real-time inference? To address this question, we first introduce VLA-Perf, an analytical performance model that can analyze inference performance for arbitrary combinations of VLA models and inference systems. Using VLA-Perf, we conduct the first systematic study of the VLA inference performance landscape. From a model-design perspective, we examine how inference performance is affected by model scaling, model architectural choices, long-context video inputs, asynchronous inference, and dual-system model pipelines. From the deployment perspective, we analyze where VLA inference should be executed -- on-device, on edge servers, or in the cloud -- and how hardware capability and network performance jointly determine end-to-end latency. By distilling 15 key takeaways from our comprehensive evaluation, we hope this work can provide practical guidance for the design of future VLA models and inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08831v4">View Invariant Learning for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2026-02-20
      | 💬 This paper is accepted to RA-L 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most existing approaches are sensitive to viewpoint changes, i.e. variations in camera height and viewing angle. Here we introduce a more general scenario, V$^2$-VLNCE (VLNCE with Varied Viewpoints) and propose a view-invariant post-training framework, called VIL (View Invariant Learning), that makes existing navigation policies more robust to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. We also introduce a teacher-student framework for the Waypoint Predictor Module, a standard part of VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components. Empirical results show that our method outperforms state-of-the-art approaches on V$^2$-VLNCE by 8-15\% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Evaluation of VIL in standard VLNCE settings shows that despite being trained for varied viewpoints, VIL often still improves performance. On the harder RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method. We further evaluate VIL for simulated camera placements derived from real robot configurations (e.g. Stretch RE-1, LoCoBot), showing consistent improvements of performance. Finally, we present a proof-of-concept real-robot evaluation in two physical environments using a panoramic RGB sensor combined with LiDAR. The code is available at https://github.com/realjoshqsun/V2-VLNCE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13957v2">Goal Inference from Open-Ended Dialog</a></div>
    <div class="paper-meta">
      📅 2026-02-19
      | 💬 This version has been updated to reflect a copy of Master's thesis submitted Jan 24, 2025 for degree date Feb 2025 (https://hdl.handle.net/1721.1/158960). We recommend readers to read revised version incorporating a different agent pipeline and methodological approach which is available at: arXiv:2508.15119
    </div>
    <details class="paper-abstract">
      Embodied AI Agents are quickly becoming important and common tools in society. These embodied agents should be able to learn about and accomplish a wide range of user goals and preferences efficiently and robustly. Large Language Models (LLMs) are often used as they allow for opportunities for rich and open-ended dialog type interaction between the human and agent to accomplish tasks according to human preferences. In this thesis, we argue that for embodied agents that deal with open-ended dialog during task assistance: 1) AI Agents should extract goals from conversations in the form of Natural Language (NL) to be better at capturing human preferences as it is intuitive for humans to communicate their preferences on tasks to agents through natural language. 2) AI Agents should quantify/maintain uncertainty about these goals to ensure that actions are being taken according to goals that the agent is extremely certain about. We present an online method for embodied agents to learn and accomplish diverse user goals. While offline methods like RLHF can represent various goals but require large datasets, our approach achieves similar flexibility with online efficiency. We extract natural language goal representations from conversations with Large Language Models (LLMs). We prompt an LLM to role play as a human with different goals and use the corresponding likelihoods to run Bayesian inference over potential goals. As a result, our method can represent uncertainty over complex goals based on unrestricted dialog. We evaluate in a text-based grocery shopping domain and an AI2Thor robot simulation. We compare our method to ablation baselines that lack either explicit goal representation or probabilistic inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17345v1">What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Embodied AI systems (e.g., autonomous vehicles, service robots, and LLM-driven interactive agents) are rapidly transitioning from controlled environments to safety critical real-world deployments. Unlike disembodied AI, failures in embodied intelligence lead to irreversible physical consequences, raising fundamental questions about security, safety, and reliability. While existing research predominantly analyzes embodied AI through the lenses of Large Language Model (LLM) vulnerabilities or classical Cyber-Physical System (CPS) failures, this survey argues that these perspectives are individually insufficient to explain many observed breakdowns in modern embodied systems. We posit that a significant class of failures arises from embodiment-induced system-level mismatches, rather than from isolated model flaws or traditional CPS attacks. Specifically, we identify four core insights that explain why embodied AI is fundamentally harder to secure: (i) semantic correctness does not imply physical safety, as language-level reasoning abstracts away geometry, dynamics, and contact constraints; (ii) identical actions can lead to drastically different outcomes across physical states due to nonlinear dynamics and state uncertainty; (iii) small errors propagate and amplify across tightly coupled perception-decision-action loops; and (iv) safety is not compositional across time or system layers, enabling locally safe decisions to accumulate into globally unsafe behavior. These insights suggest that securing embodied AI requires moving beyond component-level defenses toward system-level reasoning about physical risk, uncertainty, and failure propagation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16928v3">Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00167v2">Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-19
    </div>
    <details class="paper-abstract">
      Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08831v3">View Invariant Learning for Vision-Language Navigation in Continuous Environments</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 This paper is accepted to RA-L 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation in Continuous Environments (VLNCE), where an agent follows instructions and moves freely to reach a destination, is a key research problem in embodied AI. However, most navigation policies are sensitive to viewpoint changes, i.e., variations in camera height and viewing angle that alter the agent's observation. In this paper, we introduce a generalized scenario, V2-VLNCE (VLNCE with Varied Viewpoints), and propose VIL (View Invariant Learning), a view-invariant post-training strategy that enhances the robustness of existing navigation policies to changes in camera viewpoint. VIL employs a contrastive learning framework to learn sparse and view-invariant features. Additionally, we introduce a teacher-student framework for the Waypoint Predictor Module, a core component of most VLNCE baselines, where a view-dependent teacher model distills knowledge into a view-invariant student model. We employ an end-to-end training paradigm to jointly optimize these components, thus eliminating the cost for individual module training. Empirical results show that our method outperforms state-of-the-art approaches on V2-VLNCE by 8-15% measured on Success Rate for two standard benchmark datasets R2R-CE and RxR-CE. Furthermore, we evaluate VIL under the standard VLNCE setting and find that, despite being trained for varied viewpoints, it often still improves performance. On the more challenging RxR-CE dataset, our method also achieved state-of-the-art performance across all metrics when compared to other map-free methods. This suggests that adding VIL does not diminish the standard viewpoint performance and can serve as a plug-and-play post-training method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14135v2">ForesightSafety Bench: A Frontier Risk Evaluation and Governance Framework towards Safe AI</a></div>
    <div class="paper-meta">
      📅 2026-02-18
    </div>
    <details class="paper-abstract">
      Rapidly evolving AI exhibits increasingly strong autonomy and goal-directed capabilities, accompanied by derivative systemic risks that are more unpredictable, difficult to control, and potentially irreversible. However, current AI safety evaluation systems suffer from critical limitations such as restricted risk dimensions and failed frontier risk detection. The lagging safety benchmarks and alignment technologies can hardly address the complex challenges posed by cutting-edge AI models. To bridge this gap, we propose the "ForesightSafety Bench" AI Safety Evaluation Framework, beginning with 7 major Fundamental Safety pillars and progressively extends to advanced Embodied AI Safety, AI4Science Safety, Social and Environmental AI risks, Catastrophic and Existential Risks, as well as 8 critical industrial safety domains, forming a total of 94 refined risk dimensions. To date, the benchmark has accumulated tens of thousands of structured risk data points and assessment results, establishing a widely encompassing, hierarchically clear, and dynamically evolving AI safety evaluation framework. Based on this benchmark, we conduct systematic evaluation and in-depth analysis of over twenty mainstream advanced large models, identifying key risk patterns and their capability boundaries. The safety capability evaluation results reveals the widespread safety vulnerabilities of frontier AI across multiple pillars, particularly focusing on Risky Agentic Autonomy, AI4Science Safety, Embodied AI Safety, Social AI Safety and Catastrophic and Existential Risks. Our benchmark is released at https://github.com/Beijing-AISI/ForesightSafety-Bench. The project website is available at https://foresightsafety-bench.beijing-aisi.ac.cn/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12707v2">PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-18
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Advances in deep generative modeling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants. Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15918v1">EarthSpatialBench: Benchmarking Spatial Reasoning Capabilities of Multimodal LLMs on Earth Imagery</a></div>
    <div class="paper-meta">
      📅 2026-02-17
    </div>
    <details class="paper-abstract">
      Benchmarking spatial reasoning in multimodal large language models (MLLMs) has attracted growing interest in computer vision due to its importance for embodied AI and other agentic systems that require precise interaction with the physical world. However, spatial reasoning on Earth imagery has lagged behind, as it uniquely involves grounding objects in georeferenced images and quantitatively reasoning about distances, directions, and topological relations using both visual cues and vector geometry coordinates (e.g., 2D bounding boxes, polylines, and polygons). Existing benchmarks for Earth imagery primarily focus on 2D spatial grounding, image captioning, and coarse spatial relations (e.g., simple directional or proximity cues). They lack support for quantitative direction and distance reasoning, systematic topological relations, and complex object geometries beyond bounding boxes. To fill this gap, we propose \textbf{EarthSpatialBench}, a comprehensive benchmark for evaluating spatial reasoning in MLLMs on Earth imagery. The benchmark contains over 325K question-answer pairs spanning: (1) qualitative and quantitative reasoning about spatial distance and direction; (2) systematic topological relations; (3) single-object queries, object-pair queries, and compositional aggregate group queries; and (4) object references expressed via textual descriptions, visual overlays, and explicit geometry coordinates, including 2D bounding boxes, polylines, and polygons. We conducted extensive experiments on both open-source and proprietary models to identify limitations in the spatial reasoning of MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14965v1">PAct: Part-Decomposed Single-View Articulated Object Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Technical Report(11 figures, 14 pages), Project Page: https://PAct-project.github.io
    </div>
    <details class="paper-abstract">
      Articulated objects are central to interactive 3D applications, including embodied AI, robotics, and VR/AR, where functional part decomposition and kinematic motion are essential. Yet producing high-fidelity articulated assets remains difficult to scale because it requires reliable part decomposition and kinematic rigging. Existing approaches largely fall into two paradigms: optimization-based reconstruction or distillation, which can be accurate but often takes tens of minutes to hours per instance, and inference-time methods that rely on template or part retrieval, producing plausible results that may not match the specific structure and appearance in the input observation. We introduce a part-centric generative framework for articulated object creation that synthesizes part geometry, composition, and articulation under explicit part-aware conditioning. Our representation models an object as a set of movable parts, each encoded by latent tokens augmented with part identity and articulation cues. Conditioned on a single image, the model generates articulated 3D assets that preserve instance-level correspondence while maintaining valid part structure and motion. The resulting approach avoids per-instance optimization, enables fast feed-forward inference, and supports controllable assembly and articulation, which are important for embodied interaction. Experiments on common articulated categories (e.g., drawers and doors) show improved input consistency, part accuracy, and articulation plausibility over optimization-based and retrieval-driven baselines, while substantially reducing inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14831v1">Robot-Wearable Conversation Hand-off for Navigation</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 To appear in Proceedings of Augmented Humans International Conference (AHs 2026)
    </div>
    <details class="paper-abstract">
      Navigating large and complex indoor environments, such as universities, airports, and hospitals, can be cognitively demanding and requires attention and effort. While mobile applications provide convenient navigation support, they occupy the user's hands and visual attention, limiting natural interaction. In this paper, we explore conversation hand-off as a method for multi-device indoor navigation, where a Conversational Agent (CA) transitions seamlessly from a stationary social robot to a wearable device. We evaluated robot-only, wearable-only, and robot-to-wearable hand-off in a university campus setting using a within-subjects design with N=24 participants. We find that conversation hand-off is experienced as engaging, even though no performance benefits were observed, and most preferred using the wearable-only system. Our findings suggest that the design of such re-embodied assistants should maintain a shared voice and state across embodiments. We demonstrate how conversational hand-offs can bridge cognitive and physical transitions, enriching human interaction with embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14401v1">pFedNavi: Structure-Aware Personalized Federated Vision-Language Navigation for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation VLN requires large-scale trajectory instruction data from private indoor environments, raising significant privacy concerns. Federated Learning FL mitigates this by keeping data on-device, but vanilla FL struggles under VLNs' extreme cross-client heterogeneity in environments and instruction styles, making a single global model suboptimal. This paper proposes pFedNavi, a structure-aware and dynamically adaptive personalized federated learning framework tailored for VLN. Our key idea is to personalize where it matters: pFedNavi adaptively identifies client-specific layers via layer-wise mixing coefficients, and performs fine-grained parameter fusion on the selected components (e.g., the encoder-decoder projection and environment-sensitive decoder layers) to balance global knowledge sharing with local specialization. We evaluate pFedNavi on two standard VLN benchmarks, R2R and RxR, using both ResNet and CLIP visual representations. Across all metrics, pFedNavi consistently outperforms the FedAvg-based VLN baseline, achieving up to 7.5% improvement in navigation success rate and up to 7.8% gain in trajectory fidelity, while converging 1.38x faster under non-IID conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13052v1">Quantization-Aware Collaborative Inference for Large Embodied AI Models</a></div>
    <div class="paper-meta">
      📅 2026-02-13
    </div>
    <details class="paper-abstract">
      Large artificial intelligence models (LAIMs) are increasingly regarded as a core intelligence engine for embodied AI applications. However, the massive parameter scale and computational demands of LAIMs pose significant challenges for resource-limited embodied agents. To address this issue, we investigate quantization-aware collaborative inference (co-inference) for embodied AI systems. First, we develop a tractable approximation for quantization-induced inference distortion. Based on this approximation, we derive lower and upper bounds on the quantization rate-inference distortion function, characterizing its dependence on LAIM statistics, including the quantization bit-width. Next, we formulate a joint quantization bit-width and computation frequency design problem under delay and energy constraints, aiming to minimize the distortion upper bound while ensuring tightness through the corresponding lower bound. Extensive evaluations validate the proposed distortion approximation, the derived rate-distortion bounds, and the effectiveness of the proposed joint design. Particularly, simulations and real-world testbed experiments demonstrate the effectiveness of the proposed joint design in balancing inference quality, latency, and energy consumption in edge embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09600v2">Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures</a></div>
    <div class="paper-meta">
      📅 2026-02-13
    </div>
    <details class="paper-abstract">
      Egocentric interactive world models are essential for augmented reality and embodied AI, where visual generation must respond to user input with low latency, geometric consistency, and long-term stability. We study egocentric interaction generation from a single scene image under free-space hand gestures, aiming to synthesize photorealistic videos in which hands enter the scene, interact with objects, and induce plausible world dynamics under head motion. This setting introduces fundamental challenges, including distribution shift between free-space gestures and contact-heavy training data, ambiguity between hand motion and camera motion in monocular views, and the need for arbitrary-length video generation. We present Hand2World, a unified autoregressive framework that addresses these challenges through occlusion-invariant hand conditioning based on projected 3D hand meshes, allowing visibility and occlusion to be inferred from scene context rather than encoded in the control signal. To stabilize egocentric viewpoint changes, we inject explicit camera geometry via per-pixel Plücker-ray embeddings, disentangling camera motion from hand motion and preventing background drift. We further develop a fully automated monocular annotation pipeline and distill a bidirectional diffusion model into a causal generator, enabling arbitrary-length synthesis. Experiments on three egocentric interaction benchmarks show substantial improvements in perceptual quality and 3D consistency while supporting camera control and long-horizon interactive generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v3">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-02-13
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.12136v1">Embodied AI Agents for Team Collaboration in Co-located Blue-Collar Work</a></div>
    <div class="paper-meta">
      📅 2026-02-12
      | 💬 4 pages, 1 figure, a short synopsis of this paper has been submitted to CHI 2026 workshop on Embodying Relationships, Designing TUIs for Co-Located Human-Human Dynamics
    </div>
    <details class="paper-abstract">
      Blue-collar work is often highly collaborative, embodied, and situated in shared physical environments, yet most research on collaborative AI has focused on white-collar work. This position paper explores how the embodied nature of AI agents can support team collaboration and communication in co-located blue-collar workplaces. From the context of our newly started CAI-BLUE research project, we present two speculative scenarios from industrial and maintenance contexts that illustrate how embodied AI agents can support shared situational awareness and facilitate inclusive communication across experience levels. We outline open questions related to embodied AI agent design around worker inclusion, agency, transformation of blue-collar collaboration practices over time, and forms of acceptable AI embodiments. We argue that embodiment is not just an aesthetic choice but should become a socio-material design strategy of AI systems in blue-collar workplaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07837v3">RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-12
    </div>
    <details class="paper-abstract">
      Online policy learning directly in the physical world is a promising yet challenging direction for embodied intelligence. Unlike simulation, real-world systems cannot be arbitrarily accelerated, cheaply reset, or massively replicated, which makes scalable data collection, heterogeneous deployment, and long-horizon effective training difficult. These challenges suggest that real-world policy learning is not only an algorithmic issue but fundamentally a systems problem. We present USER, a Unified and extensible SystEm for Real-world online policy learning. USER treats physical robots as first-class hardware resources alongside GPUs through a unified hardware abstraction layer, enabling automatic discovery, management, and scheduling of heterogeneous robots. To address cloud-edge communication, USER introduces an adaptive communication plane with tunneling-based networking, distributed data channels for traffic localization, and streaming-multiprocessor-aware weight synchronization to regulate GPU-side overhead. On top of this infrastructure, USER organizes learning as a fully asynchronous framework with a persistent, cache-aware buffer, enabling efficient long-horizon experiments with robust crash recovery and reuse of historical data. In addition, USER provides extensible abstractions for rewards, algorithms, and policies, supporting online imitation or reinforcement learning of CNN/MLP, generative policies, and large vision-language-action (VLA) models within a unified pipeline. Results in both simulation and the real world show that USER enables multi-robot coordination, heterogeneous manipulators, edge-cloud collaboration with large models, and long-running asynchronous training, offering a unified and extensible systems foundation for real-world online policy learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10891v1">Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 8 pages, 7 figures, with Appendix
    </div>
    <details class="paper-abstract">
      Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08971v2">WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://world-arena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10116v1">SAGE: Scalable Agentic 3D Scene Generation for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Project Page: https://nvlabs.github.io/sage
    </div>
    <details class="paper-abstract">
      Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: https://nvlabs.github.io/sage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10357v2">Optimus-3: Dual-Router Aligned Mixture-of-Experts Agent with Dual-Granularity Reasoning-Aware Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 16 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Developing generalist agents capable of solving open-ended tasks in visually rich, dynamic environments remains a core pursuit of embodied AI. While Minecraft has emerged as a compelling benchmark, existing agents often suffer from fragmented cognitive abilities, lacking the synergy between reflexive execution (System 1) and deliberative reasoning (System 2). In this paper, we introduce Optimus-3, a generalist agent that organically integrates these dual capabilities within a unified framework. To achieve this, we address three fundamental challenges. First, to overcome the scarcity of reasoning data, we propose a Knowledge-Enhanced Automated Data Generation Pipeline. It synthesizes high-quality System 2 reasoning traces from raw System 1 interaction trajectories, effectively mitigating hallucinations via injection of domain knowledge. We release the resulting dataset, \textbf{OptimusM$^{4}$}, to the community. Second, to reconcile the dichotomous computational requirements of the dual systems, we design a Dual-Router Aligned MoE Architecture. It employs a Task Router to prevent task interference via parameter decoupling, and a Layer Router to dynamically modulate reasoning depth, creating a computational ``Fast Path'' for System 1 and a ``Deep Path'' for System 2. Third, to activate the reasoning capabilities of System 2, we propose Dual-Granularity Reasoning-Aware Policy Optimization (DGRPO) algorithm. It enforces Process-Outcome Co-Supervision via dual-granularity dense rewards, ensuring consistency between the thought process and the answer. Extensive evaluations demonstrate that Optimus-3 surpasses existing state-of-the-art methods on both System~2 (21$\%$ on Planning, 66\% on Captioning, 76\% on Embodied QA, 3.4$\times$ on Grounding, and 18\% on Reflection) and System~1 (3\% on Long-Horizon Action) tasks, with a notable 60\% success rate on open-ended tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09657v1">AutoFly: Vision-Language-Action Model for UAV Autonomous Navigation in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Acceped by ICLR 2026
    </div>
    <details class="paper-abstract">
      Vision-language navigation (VLN) requires intelligent agents to navigate environments by interpreting linguistic instructions alongside visual observations, serving as a cornerstone task in Embodied AI. Current VLN research for unmanned aerial vehicles (UAVs) relies on detailed, pre-specified instructions to guide the UAV along predetermined routes. However, real-world outdoor exploration typically occurs in unknown environments where detailed navigation instructions are unavailable. Instead, only coarse-grained positional or directional guidance can be provided, requiring UAVs to autonomously navigate through continuous planning and obstacle avoidance. To bridge this gap, we propose AutoFly, an end-to-end Vision-Language-Action (VLA) model for autonomous UAV navigation. AutoFly incorporates a pseudo-depth encoder that derives depth-aware features from RGB inputs to enhance spatial reasoning, coupled with a progressive two-stage training strategy that effectively aligns visual, depth, and linguistic representations with action policies. Moreover, existing VLN datasets have fundamental limitations for real-world autonomous navigation, stemming from their heavy reliance on explicit instruction-following over autonomous decision-making and insufficient real-world data. To address these issues, we construct a novel autonomous navigation dataset that shifts the paradigm from instruction-following to autonomous behavior modeling through: (1) trajectory collection emphasizing continuous obstacle avoidance, autonomous planning, and recognition workflows; (2) comprehensive real-world data integration. Experimental results demonstrate that AutoFly achieves a 3.9% higher success rate compared to state-of-the-art VLA baselines, with consistent performance across simulated and real environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09600v1">Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Egocentric interactive world models are essential for augmented reality and embodied AI, where visual generation must respond to user input with low latency, geometric consistency, and long-term stability. We study egocentric interaction generation from a single scene image under free-space hand gestures, aiming to synthesize photorealistic videos in which hands enter the scene, interact with objects, and induce plausible world dynamics under head motion. This setting introduces fundamental challenges, including distribution shift between free-space gestures and contact-heavy training data, ambiguity between hand motion and camera motion in monocular views, and the need for arbitrary-length video generation. We present Hand2World, a unified autoregressive framework that addresses these challenges through occlusion-invariant hand conditioning based on projected 3D hand meshes, allowing visibility and occlusion to be inferred from scene context rather than encoded in the control signal. To stabilize egocentric viewpoint changes, we inject explicit camera geometry via per-pixel Plücker-ray embeddings, disentangling camera motion from hand motion and preventing background drift. We further develop a fully automated monocular annotation pipeline and distill a bidirectional diffusion model into a causal generator, enabling arbitrary-length synthesis. Experiments on three egocentric interaction benchmarks show substantial improvements in perceptual quality and 3D consistency while supporting camera control and long-horizon interactive generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16449v3">VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have shown great promise for embodied AI, yet the heavy computational cost of processing continuous visual streams severely limits their real-time deployment. Token pruning (keeping salient visual tokens and dropping redundant ones) has emerged as an effective approach for accelerating Vision-Language Models (VLMs), offering a solution for efficient VLA. However, these VLM-specific token pruning methods select tokens based solely on semantic salience metrics (e.g., prefill attention), while overlooking the VLA's intrinsic dual-system nature of high-level semantic understanding and low-level action execution. Consequently, these methods bias token retention toward semantic cues, discard critical information for action generation, and significantly degrade VLA performance. To bridge this gap, we propose VLA-Pruner, a versatile plug-and-play VLA-specific token prune method that aligns with the dual-system nature of VLA models and exploits the temporal continuity in robot manipulation. Specifically, VLA-Pruner adopts a dual-level importance criterion for visual token retention: vision-language prefill attention for semantic-level relevance and action decode attention, estimated via temporal smoothing, for action-level importance. Based on this criterion, VLA-Pruner proposes a novel dual-level token selection strategy that adaptively preserves a compact, informative set of visual tokens for both semantic understanding and action execution under given compute budget. Experiments show that VLA-Pruner achieves state-of-the-art performance across multiple VLA architectures and diverse robotic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08421v1">Decentralized Intent-Based Multi-Robot Task Planner with LLM Oracles on Hyperledger Fabric</a></div>
    <div class="paper-meta">
      📅 2026-02-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for transforming natural language user intents into executable actions. This capability enables embodied AI agents to perform complex tasks, without involvement of an expert, making human-robot interaction (HRI) more convenient. However these developments raise significant security and privacy challenges such as self-preferencing, where a single LLM service provider dominates the market and uses this power to promote their own preferences. LLM oracles have been recently proposed as a mechanism to decentralize LLMs by executing multiple LLMs from different vendors and aggregating their outputs to obtain a more reliable and trustworthy final result. However, the accuracy of these approaches highly depends on the aggregation method. The current aggregation methods mostly use semantic similarity between various LLM outputs, not suitable for robotic task planning, where the temporal order of tasks is important. To fill the gap, we propose an LLM oracle with a new aggregation method for robotic task planning. In addition, we propose a decentralized multi-robot infrastructure based on Hyperledger Fabric that can host the proposed oracle. The proposed infrastructure enables users to express their natural language intent to the system, which then can be decomposed into subtasks. These subtasks require coordinating different robots from different vendors, while enforcing fine-grained access control management on the data. To evaluate our methodology, we created the SkillChain-RTD benchmark made it publicly available. Our experimental results demonstrate the feasibility of the proposed architecture, and the proposed aggregation method outperforms other aggregation methods currently in use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08392v1">BiManiBench: A Hierarchical Benchmark for Evaluating Bimanual Coordination of Multimodal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 38 pages, 9 figures. Project page:https://bimanibench.github.io/
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have significantly advanced embodied AI, and using them to benchmark robotic intelligence has become a pivotal trend. However, existing frameworks remain predominantly confined to single-arm manipulation, failing to capture the spatio-temporal coordination required for bimanual tasks like lifting a heavy pot. To address this, we introduce BiManiBench, a hierarchical benchmark evaluating MLLMs across three tiers: fundamental spatial reasoning, high-level action planning, and low-level end-effector control. Our framework isolates unique bimanual challenges, such as arm reachability and kinematic constraints, thereby distinguishing perceptual hallucinations from planning failures. Analysis of over 30 state-of-the-art models reveals that despite high-level reasoning proficiency, MLLMs struggle with dual-arm spatial grounding and control, frequently resulting in mutual interference and sequencing errors. These findings suggest the current paradigm lacks a deep understanding of mutual kinematic constraints, highlighting the need for future research to focus on inter-arm collision-avoidance and fine-grained temporal sequencing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08373v1">Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-09
      | 💬 Accepted to ICLR 2026. Project page. https://openreview.net/forum?id=wb05ver1k8&noteId=v1Ax8CwI71
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise as planners for embodied AI, but their stochastic nature lacks formal reasoning, preventing strict safety guarantees for physical deployment. Current approaches often rely on unreliable LLMs for safety checks or simply reject unsafe plans without offering repairs. We introduce the Verifiable Iterative Refinement Framework (VIRF), a neuro-symbolic architecture that shifts the paradigm from passive safety gatekeeping to active collaboration. Our core contribution is a tutor-apprentice dialogue where a deterministic Logic Tutor, grounded in a formal safety ontology, provides causal and pedagogical feedback to an LLM planner. This enables intelligent plan repairs rather than mere avoidance. We also introduce a scalable knowledge acquisition pipeline that synthesizes safety knowledge bases from real-world documents, correcting blind spots in existing benchmarks. In challenging home safety tasks, VIRF achieves a perfect 0 percent Hazardous Action Rate (HAR) and a 77.3 percent Goal-Condition Rate (GCR), which is the highest among all baselines. It is highly efficient, requiring only 1.1 correction iterations on average. VIRF demonstrates a principled pathway toward building fundamentally trustworthy and verifiably safe embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00181v2">CHAI: Command Hijacking against embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-07
      | 💬 This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) promises to handle edge cases in robotic vehicle systems where data is scarce by using common-sense reasoning grounded in perception and action to generalize beyond training distributions and adapt to novel real-world situations. These capabilities, however, also create new security risks. In this paper, we introduce CHAI (Command Hijacking against embodied AI), a physical environment indirect prompt injection attack that exploits the multimodal language interpretation abilities of AI models. CHAI embeds deceptive natural language instructions, such as misleading signs, in visual input, systematically searches the token space, builds a dictionary of prompts, and guides an attacker model to generate Visual Attack Prompts. We evaluate CHAI on four LVLM agents: drone emergency landing, autonomous driving, aerial object tracking, and on a real robotic vehicle. Our experiments show that CHAI consistently outperforms state-of-the-art attacks. By exploiting the semantic and multimodal reasoning strengths of next-generation embodied AI systems, CHAI underscores the urgent need for defenses that extend beyond traditional adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07553v2">VirtualEnv: A Platform for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-02-07
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07243v1">Realistic Synthetic Household Data Generation at Scale</a></div>
    <div class="paper-meta">
      📅 2026-02-06
      | 💬 Accepted at Agentic AI Benchmarks and Applications for Enterprise Tasks workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Advancements in foundation models have catalyzed research in Embodied AI to develop interactive agents capable of environmental reasoning and interaction. Developing such agents requires diverse, large-scale datasets. Prior frameworks generate synthetic data for long-term human-robot interactions but fail to model the bidirectional influence between human behavior and household environments. Our proposed generative framework creates household datasets at scale through loosely coupled generation of long-term human-robot interactions and environments. Human personas influence environment generation, while environment schematics and semantics shape human-robot interactions. The generated 3D data includes rich static context such as object and environment semantics, and temporal context capturing human and agent behaviors over extended periods. Our flexible tool allows users to define dataset characteristics via natural language prompts, enabling configuration of environment and human activity data through natural language specifications. The tool creates variations of user-defined configurations, enabling scalable data generation. We validate our framework through statistical evaluation using multi-modal embeddings and key metrics: cosine similarity, mutual information gain, intervention analysis, and iterative improvement validation. Statistical comparisons show good alignment with real-world datasets (HOMER) with cosine similarity (0.60), while synthetic datasets (Wang et al.) show moderate alignment (0.27). Intervention analysis across age, organization, and sleep pattern changes shows statistically significant effects (p < 0.001) with large effect sizes (Cohen's d = 0.51-1.12), confirming bidirectional coupling translates persona traits into measurable environmental and behavioral differences. These contributions enable development and testing of household smart devices at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05863v2">Constrained Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-06
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07082v1">MosaicThinker: On-Device Visual Spatial Reasoning for Embodied AI via Iterative Construction of Space Representation</a></div>
    <div class="paper-meta">
      📅 2026-02-06
    </div>
    <details class="paper-abstract">
      When embodied AI is expanding from traditional object detection and recognition to more advanced tasks of robot manipulation and actuation planning, visual spatial reasoning from the video inputs is necessary to perceive the spatial relationships of objects and guide device actions. However, existing visual language models (VLMs) have very weak capabilities in spatial reasoning due to the lack of knowledge about 3D spatial information, especially when the reasoning task involve complex spatial relations across multiple video frames. In this paper, we present a new inference-time computing technique for on-device embodied AI, namely \emph{MosaicThinker}, which enhances the on-device small VLM's spatial reasoning capabilities on difficult cross-frame reasoning tasks. Our basic idea is to integrate fragmented spatial information from multiple frames into a unified space representation of global semantic map, and further guide the VLM's spatial reasoning over the semantic map via a visual prompt. Experiment results show that our technique can greatly enhance the accuracy of cross-frame spatial reasoning on resource-constrained embodied AI devices, over reasoning tasks with diverse types and complexities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05863v1">Constrained Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04411v1">Self-evolving Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-04
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) is an intelligent system formed by agents and their environment through active perception, embodied cognition, and action interaction. Existing embodied AI remains confined to human-crafted setting, in which agents are trained on given memory and construct models for given tasks, enabling fixed embodiments to interact with relatively static environments. Such methods fail in in-the-wild setting characterized by variable embodiments and dynamic open environments. This paper introduces self-evolving embodied AI, a new paradigm in which agents operate based on their changing state and environment with memory self-updating, task self-switching, environment self-prediction, embodiment self-adaptation, and model self-evolution, aiming to achieve continually adaptive intelligence with autonomous evolution. Specifically, we present the definition, framework, components, and mechanisms of self-evolving embodied AI, systematically review state-of-the-art works for realized components, discuss practical applications, and point out future research directions. We believe that self-evolving embodied AI enables agents to autonomously learn and interact with environments in a human-like manner and provide a new perspective toward general artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.14093v7">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-02-04
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a cornerstone of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. The recent proliferation of VLAs necessitates a comprehensive survey to capture the rapidly evolving landscape. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing VLA-based control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges facing VLAs and outline promising future directions in embodied AI. A curated repository associated with this survey is available at: https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00611v2">Structured Self-Consistency:A Multi-Task Evaluation of LLMs on VirtualHome</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Embodied AI requires agents to understand goals, plan actions, and execute tasks in simulated environments. We present a comprehensive evaluation of Large Language Models (LLMs) on the VirtualHome benchmark using the Embodied Agent Interface (EAI) framework. We compare two representative 7B-parameter models OPENPANGU-7B and QWEN2.5-7B across four fundamental tasks: Goal Interpretation, Action Sequencing, Subgoal Decomposition, and Transition Modeling. We propose Structured Self-Consistency (SSC), an enhanced decoding strategy that leverages multiple sampling with domain-specific voting mechanisms to improve output quality for structured generation tasks. Experimental results demonstrate that SSC significantly enhances performance, with OPENPANGU-7B excelling at hierarchical planning while QWEN2.5-7B show advantages in action-level tasks. Our analysis reveals complementary strengths across model types, providing insights for future embodied AI system development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03200v1">Hand3R: Online 4D Hand-Scene Reconstruction in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      For Embodied AI, jointly reconstructing dynamic hands and the dense scene context is crucial for understanding physical interaction. However, most existing methods recover isolated hands in local coordinates, overlooking the surrounding 3D environment. To address this, we present Hand3R, the first online framework for joint 4D hand-scene reconstruction from monocular video. Hand3R synergizes a pre-trained hand expert with a 4D scene foundation model via a scene-aware visual prompting mechanism. By injecting high-fidelity hand priors into a persistent scene memory, our approach enables simultaneous reconstruction of accurate hand meshes and dense metric-scale scene geometry in a single forward pass. Experiments demonstrate that Hand3R bypasses the reliance on offline optimization and delivers competitive performance in both local hand reconstruction and global positioning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01667v3">What does really matter in image goal navigation?</a></div>
    <div class="paper-meta">
      📅 2026-02-03
    </div>
    <details class="paper-abstract">
      Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In this large experimental study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extent. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.03054v1">Towards Considerate Embodied AI: Co-Designing Situated Multi-Site Healthcare Robots from Abstract Concepts to High-Fidelity Prototypes</a></div>
    <div class="paper-meta">
      📅 2026-02-03
      | 💬 To appear in Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI 2026)
    </div>
    <details class="paper-abstract">
      Co-design is essential for grounding embodied artificial intelligence (AI) systems in real-world contexts, especially high-stakes domains such as healthcare. While prior work has explored multidisciplinary collaboration, iterative prototyping, and support for non-technical participants, few have interwoven these into a sustained co-design process. Such efforts often target one context and low-fidelity stages, limiting the generalizability of findings and obscuring how participants' ideas evolve. To address these limitations, we conducted a 14-week workshop with a multidisciplinary team of 22 participants, centered around how embodied AI can reduce non-value-added task burdens in three healthcare settings: emergency departments, long-term rehabilitation facilities, and sleep disorder clinics. We found that the iterative progression from abstract brainstorming to high-fidelity prototypes, supported by educational scaffolds, enabled participants to understand real-world trade-offs and generate more deployable solutions. We propose eight guidelines for co-designing more considerate embodied AI: attuned to context, responsive to social dynamics, mindful of expectations, and grounded in deployment. Project Page: https://byc-sophie.github.io/Towards-Considerate-Embodied-AI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02787v1">Real-World Applications of AI in LTE and 5G-NR Network Infrastructure</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 6 pages and 3 figures
    </div>
    <details class="paper-abstract">
      Telecommunications networks generate extensive performance and environmental telemetry, yet most LTE and 5G-NR deployments still rely on static, manually engineered configurations. This limits adaptability in rural, nomadic, and bandwidth-constrained environments where traffic distributions, propagation characteristics, and user behavior fluctuate rapidly. Artificial Intelligence (AI), more specifically Machine Learning (ML) models, provide new opportunities to transition Radio Access Networks (RANs) from rigid, rule-based systems toward adaptive, self-optimizing infrastructures that can respond autonomously to these dynamics. This paper proposes a practical architecture incorporating AI-assisted planning, reinforcement-learning-based RAN optimization, real-time telemetry analytics, and digital-twin-based validation. In parallel, the paper addresses the challenge of delivering embodied-AI healthcare services, educational tools, and large language model (LLM) applications to communities with insufficient backhaul for cloud computing. We introduce an edge-hosted execution model in which applications run directly on LTE/5G-NR base stations using containers, reducing latency and bandwidth consumption while improving resilience. Together, these contributions demonstrate how AI can enhance network performance, reduce operational overhead, and expand access to advanced digital services, aligning with broader goals of sustainable and inclusive network development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20377v2">RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification</a></div>
    <div class="paper-meta">
      📅 2026-02-02
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.
    </details>
</div>
