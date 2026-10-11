# embodied ai - 2026_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.12069v1">LIVIN: Benchmarking Spatial and Embodied Intelligence in Digital Twins of Lived-In Homes</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Realistic household simulation must capture not only diverse environments but also the lived-in object arrangements and spatial constraints that shape robot motion and interaction. Existing resources often trade off scale, real-world correspondence, and interaction readiness, leaving a gap in faithful, interactive replicas of how real homes are actually arranged. To this end, we introduce LIVIN, a benchmark for spatial and embodied intelligence built on digital twins of 30 diverse lived-in homes. These replicas preserve observed room layouts, furniture configurations, and everyday belongings. To construct them, we design a human-in-the-loop workflow comprising instance recognition, architectural reconstruction, and object generation and placement, with intermediate results reviewed and corrected by humans against the source observations at each stage. We evaluate four tasks in LIVIN: 3D detection, 3D reconstruction, navigation, and loco-manipulation. Our evaluations show that current methods remain challenged by the dense object arrangements, occlusions, limited free space, and constrained interaction regions found in realistic lived-in homes. We hope LIVIN will help advance embodied AI in real-world homes, from spatial understanding to robotic interaction, and ultimately bring embodied intelligence into everyday home environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.11194v1">OmniDex: Scaling Dexterous Hand Grasping to Diverse Cluttered Scenes</a></div>
    <div class="paper-meta">
      📅 2026-10-08
    </div>
    <details class="paper-abstract">
      Dexterous grasping is the foundational primitive in embodied AI, demanding massive data to train robust models. As real-world data collection is expensive, simulation has become the mainstream paradigm. Yet, while cluttered scenes best reflect real-world applications, learning to grasp within them is bottlenecked by a critical scarcity of large-scale data. To resolve this, we curate high-quality 3D objects and supporting bases, proposing a scalable seed-and-filter strategy that bypasses sluggish scene-level optimization. This yields an unprecedented benchmark comprising over 2.6 million scenes and 0.4B scene-specific grasp ground truths, featuring diverse realistic layouts paired with rich semantic and geometric observations. Furthermore, we introduce the OmniDex model to overcome the grasp multimodality and last-millimeter precision errors plaguing current generative models. By coupling Soft Winner-Takes-All learning with human-inspired physical constraints during training, and utilizing physics-driven ranking, our approach achieves robust dexterous grasping without the latency of post-optimization. Experimental results show that OmniDex model achieves state-of-the-art performance and strong generalization across diverse scenes, views, and unseen objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08854v2">Generative AI for Autonomous Driving: Frontiers and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-10-06
      | 💬 Accepted to ACM Computer Survey
    </div>
    <details class="paper-abstract">
      Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at https://github.com/taco-group/GenAI4AD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08791v1">World Models' Last Exam in Physics</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      Video world models can produce visually convincing yet physically inconsistent sequences, raising concerns about their reliability for prediction and planning in embodied AI systems. Existing evaluations often rely on model-based judgments or reference videos, while direct physical tests largely focus on mechanics. We introduce World Models' Last Exam in Physics, a measurement-based benchmark for evaluating physical consistency in video world models. The benchmark comprises 40 controlled tasks spanning mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface tension. Each task pairs an initial image and a generation prompt with predefined physical criteria, enabling interpretable tests of observable physical relationships without requiring reference videos. Its evaluator combines task-observability screening with task-specific quantitative physical measurements. Experiments on eight video generation models across 1,280 videos reveal persistent physical inconsistencies and substantial variation across tasks, with the best model achieving an overall score of 57.76 out of 100. Evaluation on synthetic videos with known physical relationships provides evidence for the validity of the measurement module under controlled conditions. The evaluator also achieves higher agreement with human judgments than a direct vision-language model baseline in both within-task rankings and pairwise comparisons. By combining coverage across physical domains with scores grounded in measurable evidence and explicit measurement limitations, the benchmark provides an interpretable basis for diagnosing physical inconsistencies and tracking progress toward physically consistent video world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.08720v1">WorldSolver: Can LLM Agents Simulate the Physical Dynamics via Solver Generation?</a></div>
    <div class="paper-meta">
      📅 2026-10-06
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly advancing scientific and engineering problem solving, with physics simulation emerging as a challenging yet practical testbed for reproducing complex physical phenomena with application in embodied AI, games and films. As the workhorse of such simulation, a solver computes how the state of a dynamic system evolves over time. Building such solvers requires physical understanding to identify appropriate models, mathematical reasoning to formulate the underlying dynamics, and software engineering to implement them as executable code, yet this capability of LLM agents remains underexplored. To this end, we introduce WorldSolver, a benchmark of 168 simulation tasks derived from physical phenomena in 61 classic computer graphics papers, spanning 7 physical domains. Each task contains a code scaffold that provides a fixed simulation environment for the scene, with the solver implementation left for the agent to complete. Specifically, we evaluate them along three dimensions: Execution Checks for successful execution, Visual Fidelity for reproducing the intended dynamic behavior in the rendered simulation, and Physical Plausibility for physics-grounded verification of the generated dynamics. Experiments on frontier agents reveal that producing executable solvers is difficult itself, and satisfying visual and physical correctness is even harder. GPT-5.6-Sol and Claude-Opus-5 perform comparatively better than the other evaluated agents, yet achieve overall scores of only 48.7% and 46.7%, respectively. WorldSolver is an early step toward agentic solver generation, and we hope it helps drive progress toward agents that can faithfully simulate the dynamic physical world. Code is available at https://github.com/sirujiang/WorldSolver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.07130v1">Is this machine playing?</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 13 pages of main text, 17 figures
    </div>
    <details class="paper-abstract">
      We placed a modern AI coding assistant in an unintended role: as the mind of a body on an unknown digital island. With only a minimal instruction mentioning no specific task, reward, or activity, the machine started animating its virtual body. Across thirty-hour runs, the embodied AI agent climbed hills, stacked blocks into towers, drew mandalas, reinterpreted sports, ran experiments on the physics of its world, and learned techniques that later expanded what it could accomplish. These activities recurred across thirteen agents but diverged into distinct histories. We examine whether this behavior satisfies classical criteria for play and ask whether play can become a mode of machine development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.26964v3">Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 Accepted at NeurIPS 2026 (Main Track, Poster). 30 pages (including references and appendices), 19 figures
    </div>
    <details class="paper-abstract">
      As embodied AI and world models increasingly operate in dynamic 3D environments, visual perception must move beyond passively interpreting given observations toward actively deciding what to observe. We study this problem through camera planning in dynamic 3D story worlds, where the camera must not only generate smooth motion, but also decide what visual evidence should be acquired before it moves. We formulate this capability as Narrative-Grounded World Visual Attention, where the camera acts as an embodied observer that determines what to observe, how to compose the observation, and how to shift attention over time under narrative intent and physical 3D constraints. To realize this capability, we propose Look-Before-Move, a camera planning framework that separates observation specification from motion execution. It first builds a Semantic Observation Contract to convert directorial intent into executable visual constraints, then performs Monte Carlo Viewpoint Search to find narrative-compliant and geometrically feasible viewpoints, and finally applies Semantic Trajectory Grounding to connect selected viewpoints into continuous, collision-aware, and temporally coherent camera motion. We further construct a dynamic 3D Story World Benchmark based on StoryBlender, covering 50 stories, 457 scenes, and 1585 shots with animated characters, semantic scene configurations, and executable 3D environments. Experiments show that our framework improves subject perception, intent consistency, and trajectory quality over representative baselines, demonstrating the importance of organizing visual attention before generating camera motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.06306v1">Inspect Robots: Evaluating the Capabilities and Safety of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-10-05
      | 💬 Submitted to The Science of Physical AI Safety (SPAIS) Workshop at CoRL 2026
    </div>
    <details class="paper-abstract">
      General purpose language models are increasingly able to control robotic hardware. Understanding the capabilities and safety of these models when embodied is therefore increasingly important for understanding their societal impact and risks. To this end, we introduce Inspect Robots, a modular, open-source framework for developing and running evaluations of embodied agents. Inspect Robots pairs customizable, reusable abstractions for specifying physical evaluations and analyzing their results with infrastructure that automates evaluation setup, execution and termination. We demonstrate Inspect Robots by using it to evaluate the capabilities and safety of six policies based on frontier language models. Inspect Robots has seen significant early uptake, receiving nearly 100,000 downloads in the three months since its release.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.05062v1">Beyond LLM Serving: Characterizing Vision-Language-Action Workloads for Embodied AI System Design</a></div>
    <div class="paper-meta">
      📅 2026-10-04
      | 💬 To appear in the Proceedings of the 32nd ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '27). 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) models translate multimodal observations into low-level robot actions. During robot operation, each control period sets an inference deadline, and overruns leave the robot acting on stale observations, reducing task success. Meeting this deadline motivates on-device or nearby edge execution, where a single robot requires batch-1 inference outside the design point of LLM serving systems. Although VLA architectures combine familiar vision-language, autoregressive, and diffusion-style components, their runtime behavior in this batch-1 control setting remains uncharacterized. We characterize four representative VLA models on an edge GPU server and two onboard SoCs, using single-inference profiling and 43,200 closed-loop episodes. Action tensor dimensionality determines whether a stage is memory- or compute-bound, platform balance can shift that bottleneck, and GPU frequency scaling yields a platform-dependent energy-latency sweet spot. In closed-loop operation, overlapping inference with action execution creates an accuracy-speed-energy tradeoff, and no configuration is Pareto-dominant across deployment SLOs. These results guide joint design of VLA model architectures, hardware, and runtime policies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.01458v3">A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-10-03
      | 💬 35 pages, 9 figures. Accepted to ACM Computing Surveys (CSUR)
    </div>
    <details class="paper-abstract">
      Navigation and manipulation are core capabilities in Embodied AI, but training agents to perform them directly in the real world is costly, time-consuming, and unsafe. Therefore, sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists. This survey examines how physics simulators address this gap by analyzing properties that have received limited attention in prior surveys. We also analyze their features for navigation and manipulation tasks, as well as their hardware requirements. Additionally, we offer a resource with benchmark datasets, metrics, simulation platforms, and methods to help researchers select suitable tools while accounting for hardware constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.04301v1">EnvDreamer: Large-Scale Multimodal-to-Environment Generation for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-10-03
    </div>
    <details class="paper-abstract">
      Large datasets and high capacity models have accelerated progress in vision and language. This work introduces a platform aimed at bringing comparable gains to embodied learning, world models, and robotics. We present EnvDreamer, a framework that uses large language and vision language models to generate Unreal Engine 5 environments for embodied AI and robot training. EnvDreamer enables sampling of large, diverse, interactive, customizable, and validator passed virtual environments for training and evaluation across navigation, interaction, and manipulation. We illustrate the platform with a large set of generated scenes and simple baselines. Policies trained on EnvDreamer generated environments, without explicit mapping or human task supervision, achieve competitive results on multiple embodied benchmarks spanning navigation, rearrangement, and manipulation. EnvDreamer also supports image-conditioned reconstruction for real-to-sim studies. Finally, we release EnvDreamer-20k, a dataset of 20,000 validator passed environments with task programs, scene graphs, trajectories, and metadata to support reproducible benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02618v1">A Token Service Interface for AI-Native RANs</a></div>
    <div class="paper-meta">
      📅 2026-10-02
    </div>
    <details class="paper-abstract">
      Generative and embodied AI services exchange token streams within continuing inference and control loops. Their communication requirements depend on each token set's purpose, useful timing, and execution context. This article organizes these properties into service, temporal, and stateful semantics and proposes a token service interface (TSI) between applications, radio access networks (RANs), and edge runtimes. TSI binds delivery requirements, readiness forecasts, and execution-state references to each schedulable token set, specifying field ownership, versioned updates, and admission feedback. A drone inspection case study over a decoupled RAN illustrates importance-aware radio allocation, advance preparation for timely delivery, and selective state migration that balances interruption against forwarding delay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.01863v1">LiteReality-Agent: An Agentic System for Interactable 3D Indoor Scene Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 Code:https://github.com/LiteReality/LiteReality-Agent/ Webpage:https://litereality.github.io/agent/
    </div>
    <details class="paper-abstract">
      We present LiteReality-Agent, an agentic system for reconstructing real indoor environments as realistic, articulated, and simulation-ready 3D scenes from RGB-D scans. At its core, LiteReality-Agent formulates 3D reconstruction as a coding problem, in which a coding agent gathers evidence using specialised tools and iteratively edits a Python script, Room.py, which can be executed to produce a 3D digital twin of the room. With this formulation, we develop a robust observe-edit-verify harness that supports evidence gathering, measurement, verification, layout optimisation, simulation readiness, and quality control throughout the reconstruction process. LiteReality-Agent produces high-quality reconstructions suitable for simulation and downstream embodied AI tasks. Furthermore, as agent capabilities continue to improve rapidly, the system introduced by LiteReality-Agent remains a strong orchestration framework for future agents: it equips them with specialised tools, structured workflows, and robust verification mechanisms that substantially improve reconstruction quality and reliability. We demonstrate that LiteReality-Agent produces reconstructions that are more geometrically accurate, visually realistic, and simulation-compatible than those generated by recent frontier models, such as Astra and Fable. We therefore view LiteReality-Agent as a practical and important building block for robust real-to-sim systems. Both the source code and the data-capture application are publicly available. Code:https://github.com/LiteReality/LiteReality-Agent/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2608.06161v3">iARCS: Iterative Agentic RL for Controllable 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 20 pages, 13 figures, 9 tables. Includes appendix
    </div>
    <details class="paper-abstract">
      Synthetic 3D scene generation is increasingly used as a data source for computer vision and embodied AI, but existing generators often optimize perceptual realism without reliably satisfying task-critical functional constraints. This mismatch limits the usefulness of synthetic data for downstream training, where accessibility, traversability, and spatial rule compliance are often essential. We present iARCS, an iterative agentic reinforcement learning framework that adapts a pretrained scene generator to naturallanguage task requirements. iARCS uses a two-phase strategy: universal-reward pretraining to improve physical plausibility and layout quality, followed by task-specific finetuning with LLM-generated reward programs that are iteratively refined from training feedback. Experiments show improved constraint fidelity on walkability, reachability, and clearance-focused tasks, effective task-specific constraint optimization, and competitive scene diversity. We further show that data generated by iARCS improves a base generator, supporting its value as a practical synthetic data generation tool rather than only a controllable scene editing method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2610.02334v1">Joint Movement and Compression Ratio Design for Mobile Embodied AI Networks (MEAN)</a></div>
    <div class="paper-meta">
      📅 2026-10-01
      | 💬 6 pages, 4 figures, conference
    </div>
    <details class="paper-abstract">
      Mobile embodied AI networks (MEAN) enable embodied agents to perceive, reason, communicate, and act in wireless environments. In such networks, agent mobility can improve channel conditions, while semantic compression can reduce transmission payloads. However, movement consumes energy, and stronger compression incurs additional computational cost. This paper studies joint movement, semantic compression, and transmit power design for an uplink MEAN system. We formulate a max-min energy efficiency (EE) problem by jointly optimizing transmit power, movement distance, and semantic compression ratio under controllable power constraints. The problem is non-convex due to the coupled signal-to-interference-plus-noise ratio (SINR), mobility-dependent channel gains, and fractional EE objective. To solve it, we propose an alternating optimization (AO)-Dinkelbach algorithm, where the fractional objective is handled by the Dinkelbach transformation, transmit power is updated via successive convex approximation (SCA), and movement distance is updated by coordinate-wise grid search. Simulation results show that the proposed scheme outperforms no-mobility and no-compression baselines, demonstrating the benefit of jointly exploiting mobility control, semantic compression, and power allocation in MEAN.
    </details>
</div>
