# embodied ai - 2026_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
