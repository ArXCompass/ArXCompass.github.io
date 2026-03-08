# embodied ai - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17100v4">Generative Models in Decision Making: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project page:https://github.com/xyshao23/Awesome-Generative-Models-for-Decision-Making-Taxonomy
    </div>
    <details class="paper-abstract">
      Generative models have fundamentally reshaped the landscape of decision-making, reframing the problem from pure scalar reward maximization to high-fidelity trajectory generation and distribution matching. This paradigm shift addresses intrinsic limitations in classical Reinforcement Learning (RL), particularly the limited expressivity of standard unimodal policy distributions in capturing complex, multi-modal behaviors embedded in diverse datasets. However, current literature often treats these models as isolated algorithmic improvements, rarely synthesizing them into a single comprehensive framework. This survey proposes a principled taxonomy grounding generative decision-making within the probabilistic framework of Control as Inference. By performing a variational factorization of the trajectory posterior, we conceptualize four distinct functional roles: Controllers for amortized policy inference, Modelers for dynamics priors, Optimizers for iterative trajectory refinement, and Evaluators for trajectory guidance and value assessment. Unlike existing architecture-centric reviews, this function-centric framework allows us to critically analyze representative generative families across distinct dimensions. Furthermore, we examine deployment in high-stakes domains, specifically Embodied AI, Autonomous Driving, and AI for Science, highlighting systemic risks such as dynamics hallucination in world models and proxy exploitation. Finally, we chart the path toward Generalist Physical Intelligence, identifying pivotal challenges in inference efficiency, trustworthiness, and the emergence of Physical Foundation Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21161v2">MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Project Page: https://xuhu0529.github.io/MarketGen
    </div>
    <details class="paper-abstract">
      The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05684v3">D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations and 1K+ hours of pseudo-labeled gameplay), our 1B-parameter model achieves 96.6% success on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models up to 7x larger, such as π_{0} (3.3B) and OpenVLA (7B). These results demonstrate that sensorimotor primitives learned from digital interactions transfer effectively to real-world physical tasks, establishing desktop pretraining as a practical paradigm for embodied AI. All resources are publicly available at https://worv-ai.github.io/d2e.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04457v1">Capability Thresholds and Manufacturing Topology: How Embodied Intelligence Triggers Phase Transitions in Economic Geography</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      The fundamental topology of manufacturing has not undergone a paradigm-level transformation since Henry Ford's moving assembly line in 1913. Every major innovation of the past century, from the Toyota Production System to Industry 4.0, has optimized within the Fordist paradigm without altering its structural logic: centralized mega-factories, located near labor pools, producing at scale. We argue that embodied intelligence is poised to break this century-long stasis, not by making existing factories more efficient, but by triggering phase transitions in manufacturing economic geography itself. When embodied AI capabilities cross critical thresholds in dexterity, generalization, reliability, and tactile-vision fusion, the consequences extend far beyond cost reduction: they restructure where factories are built, how supply chains are organized, and what constitutes viable production scale. We formalize this by defining a Capability Space C = (d, g, r, t) and showing that the site-selection objective function undergoes topological reorganization when capability vectors cross critical surfaces. Through three pathways, weight inversion, batch collapse, and human-infrastructure decoupling, we show that embodied intelligence enables demand-proximal micro-manufacturing, eliminates "manufacturing deserts," and reverses geographic concentration driven by labor arbitrage. We further introduce Machine Climate Advantage: once human workers are removed, optimal factory locations are determined by machine-optimal conditions (low humidity, high irradiance, thermal stability), factors orthogonal to traditional siting logic, creating a production geography with no historical precedent. This paper establishes Embodied Intelligence Economics, the study of how physical AI capability thresholds reshape the spatial and structural logic of production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02008v1">Temporal Representations for Exploration: Learning Complex Exploratory Behavior without Extrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Effective exploration in reinforcement learning requires not only tracking where an agent has been, but also understanding how the agent perceives and represents the world. To learn powerful representations, an agent should actively explore states that contribute to its knowledge of the environment. Temporal representations can capture the information necessary to solve a wide range of potential tasks while avoiding the computational cost associated with full state reconstruction. In this paper, we propose an exploration method that leverages temporal contrastive representations to guide exploration, prioritizing states with unpredictable future outcomes. We demonstrate that such representations can enable the learning of complex exploratory x in locomotion, manipulation, and embodied-AI tasks, revealing capabilities and behaviors that traditionally require extrinsic rewards. Unlike approaches that rely on explicit distance learning or episodic memory mechanisms (e.g., quasimetric-based methods), our method builds directly on temporal similarities, yielding a simpler yet effective strategy for exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19400v2">Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026. Camera-ready version. Project page: https://aaronfengzy.github.io/MV-RoboBench-Webpage/
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) are essential to Embodied AI, enabling robots to perceive, reason, and act in complex environments. They also serve as the foundation for the recent Vision-Language-Action (VLA) models. Yet most evaluations of VLMs focus on single-view settings, leaving their ability to integrate multi-view information underexplored. At the same time, multi-camera setups are increasingly standard in robotic platforms, as they provide complementary perspectives to mitigate occlusion and depth ambiguity. Whether VLMs can effectively leverage such multi-view inputs for robotic reasoning therefore remains an open question. To bridge this gap, we introduce MV-RoboBench, a benchmark specifically designed to evaluate the multi-view spatial reasoning capabilities of VLMs in robotic manipulation. MV-RoboBench consists of 1.7k manually curated QA items across eight subtasks, divided into two primary categories: spatial understanding and robotic execution. We evaluate a diverse set of existing VLMs, including both open-source and closed-source models, along with enhanced versions incorporating CoT-inspired techniques. The results show that state-of-the-art models remain far below human performance, underscoring the substantial challenges VLMs face in multi-view robotic perception. Additionally, our analysis uncovers two key findings: (i) spatial intelligence and robotic task execution are positively correlated in multi-view robotic scenarios; and (ii) strong performance on existing general-purpose single-view spatial understanding benchmarks does not reliably translate to success in the robotic spatial tasks assessed by our benchmark. We release MV-RoboBench as an open resource to foster progress in spatially grounded VLMs and VLAs, providing not only data but also a standardized evaluation protocol for multi-view embodied reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15018v2">UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026. Project page: https://urbanverseproject.github.io/
    </div>
    <details class="paper-abstract">
      Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01452v1">Scaling Tasks, Not Samples: Mastering Humanoid Control through Multi-Task Model-Based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Developing generalist robots capable of mastering diverse skills remains a central challenge in embodied AI. While recent progress emphasizes scaling model parameters and offline datasets, such approaches are limited in robotics, where learning requires active interaction. We argue that effective online learning should scale the \emph{number of tasks}, rather than the number of samples per task. This regime reveals a structural advantage of model-based reinforcement learning (MBRL). Because physical dynamics are invariant across tasks, a shared world model can aggregate multi-task experience to learn robust, task-agnostic representations. In contrast, model-free methods suffer from gradient interference when tasks demand conflicting actions in similar states. Task diversity therefore acts as a regularizer for MBRL, improving dynamics learning and sample efficiency. We instantiate this idea with \textbf{EfficientZero-Multitask (EZ-M)}, a sample-efficient multi-task MBRL algorithm for online learning. Evaluated on \textbf{HumanoidBench}, a challenging whole-body control benchmark, EZ-M achieves state-of-the-art performance with significantly higher sample efficiency than strong baselines, without extreme parameter scaling. These results establish task scaling as a critical axis for scalable robotic learning. The project website is available \href{https://yewr.github.io/ez_m/}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01414v1">Jailbreaking Embodied LLMs via Action-level Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 This paper has been officially accepted for ACM SenSys 2026
    </div>
    <details class="paper-abstract">
      Embodied Large Language Models (LLMs) enable AI agents to interact with the physical world through natural language instructions and actions. However, beyond the language-level risks inherent to LLMs themselves, embodied LLMs with real-world actuation introduce a new vulnerability: instructions that appear semantically benign may still lead to dangerous real-world consequences, revealing a fundamental misalignment between linguistic security and physical outcomes. In this paper, we introduce Blindfold, an automated attack framework that leverages the limited causal reasoning capabilities of embodied LLMs in real-world action contexts. Rather than iterative trial-and-error jailbreaking of black-box embodied LLMs, Blindfold adopts an Adversarial Proxy Planning strategy: it compromises a local surrogate LLM to perform action-level manipulations that appear semantically safe but could result in harmful physical effects when executed. Blindfold further conceals key malicious actions by injecting carefully crafted noise to evade detection by defense mechanisms, and it incorporates a rule-based verifier to improve the attack executability. Evaluations on both embodied AI simulators and a real-world 6DoF robotic arm show that Blindfold achieves up to 53% higher attack success rates than SOTA baselines, highlighting the urgent need to move beyond surface-level language censorship and toward consequence-aware defense mechanisms to secure embodied LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23893v2">AoE: Always-on Egocentric Human Video Collection for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Embodied foundation models require large-scale, high-quality real-world interaction data for pre-training and scaling. However, existing data collection methods suffer from high infrastructure costs, complex hardware dependencies, and limited interaction scope, making scalable expansion challenging. In fact, humans themselves are ideal physically embodied agents. Therefore, obtaining egocentric real-world interaction data from globally distributed "human agents" offers advantages of low cost and sustainability. To this end, we propose the Always-on Egocentric (AoE) data collection system, which aims to simplify hardware dependencies by leveraging humans themselves and their smartphones, enabling low-cost, highly efficient, and scene-agnostic real-world interaction data collection to address the challenge of data scarcity. Specifically, we first employ an ergonomic neck-mounted smartphone holder to enable low-barrier, large-scale egocentric data collection through a cloud-edge collaborative architecture. Second, we develop a cross-platform mobile APP that leverages on-device compute for real-time processing, while the cloud hosts automated labeling and filtering pipelines that transform raw videos into high-quality training data. Finally, the AoE system supports distributed Ego video data collection by anyone, anytime, and anywhere. We evaluate AoE on data preprocessing quality and downstream tasks, demonstrating that high-quality egocentric data significantly boosts real-world generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18041v7">Openfly: A comprehensive platform for aerial vision-language navigation</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02271v1">Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures</a></div>
    <div class="paper-meta">
      📅 2026-03-01
      | 💬 3 Pages 4 Figures for Workshop paper
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are an emerging class of workloads critical for robotics and embodied AI at the edge. As these models scale, they demonstrate significant capability gains, yet they must be deployed locally to meet the strict latency requirements of real-time applications. This paper characterizes VLA performance on two generations of edge hardware, viz. the Nvidia Jetson Orin and Thor platforms. Using MolmoAct-7B, a state-of-the-art VLA model, we identify a primary execution bottleneck: up to 75% of end-to-end latency is consumed by the memory-bound action-generation phase. Through analytical modeling and simulations, we project the hardware requirements for scaling to 100B parameter models. We also explore the impact of high-bandwidth memory technologies and processing-in-memory (PIM) as promising future pathways in edge systems for embodied AI.
    </details>
</div>
