# embodied ai - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16651v1">InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Recent works explore how real and synthetic data contribute to Vision-Language-Action (VLA) models' generalization. While current VLA models have shown the strong effectiveness of large-scale real-robot pre-training, synthetic data has not previously demonstrated comparable capability at scale. This paper provides the first evidence that synthetic data alone can match the performance of the strongest $π$-dataset in pre-training a VLA model, revealing the substantial value of large-scale simulation. The resulting model also exhibits surprisingly zero-shot sim-to-real transfer on several challenging tasks. Our synthetic dataset, InternData-A1, contains over 630k trajectories and 7,433 hours across 4 embodiments, 18 skills, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It is generated through a highly autonomous, fully decoupled, and compositional simulation pipeline that enables long-horizon skill composition, flexible task assembly, and heterogeneous embodiments with minimal manual tuning. Using the same architecture as $π_0$, we pre-train a model entirely on InternData-A1 and find that it matches the official $π_0$ across 49 simulation tasks, 5 real-world tasks, and 4 long-horizon dexterous tasks. We release the dataset and will open-source the generation pipeline to broaden access to large-scale robotic data and to lower the barrier to scalable data creation for embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16518v1">MiMo-Embodied: X-Embodied Foundation Model Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Code: https://github.com/XiaomiMiMo/MiMo-Embodied Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B
    </div>
    <details class="paper-abstract">
      We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16449v1">VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models have shown great promise for embodied AI, yet the heavy computational cost of processing continuous visual streams severely limits their real-time deployment. Token pruning (keeping salient visual tokens and dropping redundant ones) has emerged as an effective approach for accelerating Vision-Language Models (VLMs), offering a solution for efficient VLA. However, these VLM-specific token pruning methods select tokens based solely on semantic salience metrics (e.g., prefill attention), while overlooking the VLA's intrinsic dual-system nature of high-level semantic understanding and low-level action execution. Consequently, these methods bias token retention toward semantic cues, discard critical information for action generation, and significantly degrade VLA performance. To bridge this gap, we propose VLA-Pruner, a versatile plug-and-play VLA-specific token prune method that aligns with the dual-system nature of VLA models and exploits the temporal continuity in robot manipulation. Specifically, VLA-Pruner adopts a dual-level importance criterion for visual token retention: vision-language prefill attention for semantic-level relevance and action decode attention, estimated via temporal smoothing, for action-level importance. Based on this criterion, VLA-Pruner proposes a novel dual-level token selection strategy that adaptively preserves a compact, informative set of visual tokens for both semantic understanding and action execution under given compute budget. Experiments show that VLA-Pruner achieves state-of-the-art performance across multiple VLA architectures and diverse robotic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18904v3">TC-Light: Temporally Coherent Generative Rendering for Realistic World Transfer</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Project Page: https://dekuliutesla.github.io/tclight/ Code: https://github.com/Linketic/TC-Light
    </div>
    <details class="paper-abstract">
      Illumination and texture editing are critical dimensions for world-to-world transfer, which is valuable for applications including sim2real and real2real visual data scaling up for embodied AI. Existing techniques generatively re-render the input video to realize the transfer, such as video relighting models and conditioned world generation models. Nevertheless, these models are predominantly limited to the domain of training data (e.g., portrait) or fall into the bottleneck of temporal consistency and computation efficiency, especially when the input video involves complex dynamics and long durations. In this paper, we propose TC-Light, a novel generative renderer to overcome these problems. Starting from the video preliminarily relighted by an inflated video relighting model, it optimizes appearance embedding in the first stage to align global illumination. Then it optimizes the proposed canonical video representation, i.e., Unique Video Tensor (UVT), to align fine-grained texture and lighting in the second stage. To comprehensively evaluate performance, we also establish a long and highly dynamic video benchmark. Extensive experiments show that our method enables physically plausible re-rendering results with superior temporal coherence and low computation cost. The code and video demos are available at https://dekuliutesla.github.io/tclight/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16347v1">The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      The adoption of Vision-Language Models (VLMs) in embodied AI agents, while being effective, brings safety concerns such as jailbreaking. Prior work have explored the possibility of directly jailbreaking the embodied agents through elaborated multi-modal prompts. However, no prior work has studied or even reported indirect jailbreaks in embodied AI, where a black-box attacker induces a jailbreak without issuing direct prompts to the embodied agent. In this paper, we propose, for the first time, indirect environmental jailbreak (IEJ), a novel attack to jailbreak embodied AI via indirect prompt injected into the environment, such as malicious instructions written on a wall. Our key insight is that embodied AI does not ''think twice'' about the instructions provided by the environment -- a blind trust that attackers can exploit to jailbreak the embodied agent. We further design and implement open-source prototypes of two fully-automated frameworks: SHAWSHANK, the first automatic attack generation framework for the proposed attack IEJ; and SHAWSHANK-FORGE, the first automatic benchmark generation framework for IEJ. Then, using SHAWSHANK-FORGE, we automatically construct SHAWSHANK-BENCH, the first benchmark for indirectly jailbreaking embodied agents. Together, our two frameworks and one benchmark answer the questions of what content can be used for malicious IEJ instructions, where they should be placed, and how IEJ can be systematically evaluated. Evaluation results show that SHAWSHANK outperforms eleven existing methods across 3,957 task-scene combinations and compromises all six tested VLMs. Furthermore, current defenses only partially mitigate our attack, and we have responsibly disclosed our findings to all affected VLM vendors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17425v2">Evaluating Multimodal Large Language Models with Daily Composite Tasks in Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      A key feature differentiating artificial general intelligence (AGI) from traditional AI is that AGI can perform composite tasks that require a wide range of capabilities. Although embodied agents powered by multimodal large language models (MLLMs) offer rich perceptual and interactive capabilities, it remains largely unexplored whether they can solve composite tasks. In the current work, we designed a set of composite tasks inspired by common daily activities observed in early childhood development. Within a dynamic and simulated home environment, these tasks span three core domains: object understanding, spatial intelligence, and social activity. We evaluated 17 leading proprietary and open-source MLLMs on these tasks. The results consistently showed poor performance across all three domains, indicating a substantial gap between current capabilities and general intelligence requirements. Together, our tasks offer a preliminary framework for evaluating the general capabilities of embodied agents, marking an early but significant step toward the development of embodied MLLMs and their real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15997v1">Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 (to appear) NeurIPS 2025 Creative AI Track
    </div>
    <details class="paper-abstract">
      Sensorium Arc (AI reflects on climate) is a real-time multimodal interactive AI agent system that personifies the ocean as a poetic speaker and guides users through immersive explorations of complex marine data. Built on a modular multi-agent system and retrieval-augmented large language model (LLM) framework, Sensorium enables natural spoken conversations with AI agents that embodies the ocean's perspective, generating responses that blend scientific insight with ecological poetics. Through keyword detection and semantic parsing, the system dynamically triggers data visualizations and audiovisual playback based on time, location, and thematic cues drawn from the dialogue. Developed in collaboration with the Center for the Study of the Force Majeure and inspired by the eco-aesthetic philosophy of Newton Harrison, Sensorium Arc reimagines ocean data not as an abstract dataset but as a living narrative. The project demonstrates the potential of conversational AI agents to mediate affective, intuitive access to high-dimensional environmental data and proposes a new paradigm for human-machine-ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15379v1">Zero-Shot Open-Vocabulary Human Motion Grounding with Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Understanding complex human activities demands the ability to decompose motion into fine-grained, semantic-aligned sub-actions. This motion grounding process is crucial for behavior analysis, embodied AI and virtual reality. Yet, most existing methods rely on dense supervision with predefined action classes, which are infeasible in open-vocabulary, real-world settings. In this paper, we propose ZOMG, a zero-shot, open-vocabulary framework that segments motion sequences into semantically meaningful sub-actions without requiring any annotations or fine-tuning. Technically, ZOMG integrates (1) language semantic partition, which leverages large language models to decompose instructions into ordered sub-action units, and (2) soft masking optimization, which learns instance-specific temporal masks to focus on frames critical to sub-actions, while maintaining intra-segment continuity and enforcing inter-segment separation, all without altering the pretrained encoder. Experiments on three motion-language datasets demonstrate state-of-the-art effectiveness and efficiency of motion grounding performance, outperforming prior methods by +8.7\% mAP on HumanML3D benchmark. Meanwhile, significant improvements also exist in downstream retrieval, establishing a new paradigm for annotation-free motion understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15279v1">Look, Zoom, Understand: The Robotic Eyeball for Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      In embodied AI perception systems, visual perception should be active: the goal is not to passively process static images, but to actively acquire more informative data within pixel and spatial budget constraints. Existing vision models and fixed RGB-D camera systems fundamentally fail to reconcile wide-area coverage with fine-grained detail acquisition, severely limiting their efficacy in open-world robotic applications. To address this issue, we propose EyeVLA, a robotic eyeball for active visual perception that can take proactive actions based on instructions, enabling clear observation of fine-grained target objects and detailed information across a wide spatial extent. EyeVLA discretizes action behaviors into action tokens and integrates them with vision-language models (VLMs) that possess strong open-world understanding capabilities, enabling joint modeling of vision, language, and actions within a single autoregressive sequence. By using the 2D bounding box coordinates to guide the reasoning chain and applying reinforcement learning to refine the viewpoint selection policy, we transfer the open-world scene understanding capability of the VLM to a vision language action (VLA) policy using only minimal real-world data. Experiments show that our system efficiently performs instructed scenes in real-world environments and actively acquires more accurate visual information through instruction-driven actions of rotation and zoom, thereby achieving strong environmental perception capabilities. EyeVLA introduces a novel robotic vision system that leverages detailed and spatially rich, large-scale embodied data, and actively acquires highly informative visual observations for downstream embodied tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14161v2">RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v3">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 This work has been submitted to the IEEE for possible publication. This version is temporarily hosted anonymously for double-blind review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions. This version is temporarily hosted anonymously for double-blind review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14396v1">Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
    </div>
    <details class="paper-abstract">
      Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14291v1">GEN3D: Generating Domain-Free 3D Scenes from a Single Image</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 5 pages , 2 figures
    </div>
    <details class="paper-abstract">
      Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability. Additionally, 3D scene generation is vital for advancing embodied AI and world models, which depend on diverse, high-quality scenes for learning and evaluation. In this work, we propose Gen3d, a novel method for generation of high-quality, wide-scope, and generic 3D scenes from a single image. After the initial point cloud is created by lifting the RGBD image, Gen3d maintains and expands its world model. The 3D scene is finalized through optimizing a Gaussian splatting representation. Extensive experiments on diverse datasets demonstrate the strong generalization capability and superior performance of our method in generating a world model and Synthesizing high-fidelity and consistent novel views.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14161v1">RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11840v2">The Developments and Challenges towards Dexterous and Embodied Robotic Manipulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Achieving human-like dexterous robotic manipulation remains a central goal and a pivotal challenge in robotics. The development of Artificial Intelligence (AI) has allowed rapid progress in robotic manipulation. This survey summarizes the evolution of robotic manipulation from mechanical programming to embodied intelligence, alongside the transition from simple grippers to multi-fingered dexterous hands, outlining key characteristics and main challenges. Focusing on the current stage of embodied dexterous manipulation, we highlight recent advances in two critical areas: dexterous manipulation data collection (via simulation, human demonstrations, and teleoperation) and skill-learning frameworks (imitation and reinforcement learning). Then, based on the overview of the existing data collection paradigm and learning framework, three key challenges restricting the development of dexterous robotic manipulation are summarized and discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08334v3">iTACO: Interactable Digital Twins of Articulated Objects from Casually Captured RGBD Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 3DV 2026 camera-ready version. Project website can be found at https://3dlg-hcvc.github.io/video2articulation/
    </div>
    <details class="paper-abstract">
      Articulated objects are prevalent in daily life. Interactable digital twins of such objects have numerous applications in embodied AI and robotics. Unfortunately, current methods to digitize articulated real-world objects require carefully captured data, preventing practical, scalable, and generalizable acquisition. We focus on motion analysis and part-level segmentation of an articulated object from a casually captured RGBD video shot with a hand-held camera. A casually captured video of an interaction with an articulated object is easy to obtain at scale using smartphones. However, this setting is challenging due to simultaneous object and camera motion and significant occlusions as the person interacts with the object. To tackle these challenges, we introduce iTACO: a coarse-to-fine framework that infers joint parameters and segments movable parts of the object from a dynamic RGBD video. To evaluate our method under this new setting, we build a dataset of 784 videos containing 284 objects across 11 categories that is 20$\times$ larger than available in prior work. We then compare our approach with existing methods that also take video as input. Our experiments show that iTACO outperforms existing articulated object digital twin methods on both synthetic and real casually captured RGBD videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13648v1">PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Project page: https://physx-anything.github.io/
    </div>
    <details class="paper-abstract">
      3D modeling is shifting from static visual representations toward physical, articulated assets that can be directly used in simulation and interaction. However, most existing 3D generation methods overlook key physical and articulation properties, thereby limiting their utility in embodied AI. To bridge this gap, we introduce PhysX-Anything, the first simulation-ready physical 3D generative framework that, given a single in-the-wild image, produces high-quality sim-ready 3D assets with explicit geometry, articulation, and physical attributes. Specifically, we propose the first VLM-based physical 3D generative model, along with a new 3D representation that efficiently tokenizes geometry. It reduces the number of tokens by 193x, enabling explicit geometry learning within standard VLM token budgets without introducing any special tokens during fine-tuning and significantly improving generative quality. In addition, to overcome the limited diversity of existing physical 3D datasets, we construct a new dataset, PhysX-Mobility, which expands the object categories in prior physical 3D datasets by over 2x and includes more than 2K common real-world objects with rich physical annotations. Extensive experiments on PhysX-Mobility and in-the-wild images demonstrate that PhysX-Anything delivers strong generative performance and robust generalization. Furthermore, simulation-based experiments in a MuJoCo-style environment validate that our sim-ready assets can be directly used for contact-rich robotic policy learning. We believe PhysX-Anything can substantially empower a broad range of downstream applications, especially in embodied AI and physics-based simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13524v1">FreeAskWorld: An Interactive and Closed-Loop Simulator for Human-Centric Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      As embodied intelligence emerges as a core frontier in artificial intelligence research, simulation platforms must evolve beyond low-level physical interactions to capture complex, human-centered social behaviors. We introduce FreeAskWorld, an interactive simulation framework that integrates large language models (LLMs) for high-level behavior planning and semantically grounded interaction, informed by theories of intention and social cognition. Our framework supports scalable, realistic human-agent simulations and includes a modular data generation pipeline tailored for diverse embodied tasks.To validate the framework, we extend the classic Vision-and-Language Navigation (VLN) task into a interaction enriched Direction Inquiry setting, wherein agents can actively seek and interpret navigational guidance. We present and publicly release FreeAskWorld, a large-scale benchmark dataset comprising reconstructed environments, six diverse task types, 16 core object categories, 63,429 annotated sample frames, and more than 17 hours of interaction data to support training and evaluation of embodied AI systems. We benchmark VLN models, and human participants under both open-loop and closed-loop settings. Experimental results demonstrate that models fine-tuned on FreeAskWorld outperform their original counterparts, achieving enhanced semantic understanding and interaction competency. These findings underscore the efficacy of socially grounded simulation frameworks in advancing embodied AI systems toward sophisticated high-level planning and more naturalistic human-agent interaction. Importantly, our work underscores that interaction itself serves as an additional information modality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13411v1">An Operational Kardashev-Style Scale for Autonomous AI - Towards AGI and Superintelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We propose a Kardashev-inspired yet operational Autonomous AI (AAI) Scale that measures the progression from fixed robotic process automation (AAI-0) to full artificial general intelligence (AAI-4) and beyond. Unlike narrative ladders, our scale is multi-axis and testable. We define ten capability axes (Autonomy, Generality, Planning, Memory/Persistence, Tool Economy, Self-Revision, Sociality/Coordination, Embodiment, World-Model Fidelity, Economic Throughput) aggregated by a composite AAI-Index (a weighted geometric mean). We introduce a measurable Self-Improvement Coefficient $κ$ (capability growth per unit of agent-initiated resources) and two closure properties (maintenance and expansion) that convert ``self-improving AI'' into falsifiable criteria. We specify OWA-Bench, an open-world agency benchmark suite that evaluates long-horizon, tool-using, persistent agents. We define level gates for AAI-0\ldots AAI-4 using thresholds on the axes, $κ$, and closure proofs. Synthetic experiments illustrate how present-day systems map onto the scale and how the delegability frontier (quality vs.\ autonomy) advances with self-improvement. We also prove a theorem that AAI-3 agent becomes AAI-5 over time with sufficient conditions, formalizing "baby AGI" becomes Superintelligence intuition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22041v3">An LLM-based Simulation Framework for Embodied Conversational Agents in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Due to privacy concerns, open dialogue datasets for mental health are primarily generated through human or AI synthesis methods. However, the inherent implicit nature of psychological processes, particularly those of clients, poses challenges to the authenticity and diversity of synthetic data. In this paper, we propose ECAs (short for Embodied Conversational Agents), a framework for embodied agent simulation based on Large Language Models (LLMs) that incorporates multiple psychological theoretical principles.Using simulation, we expand real counseling case data into a nuanced embodied cognitive memory space and generate dialogue data based on high-frequency counseling questions.We validated our framework using the D4 dataset. First, we created a public ECAs dataset through batch simulations based on D4. Licensed counselors evaluated our method, demonstrating that it significantly outperforms baselines in simulation authenticity and necessity. Additionally, two LLM-based automated evaluation methods were employed to confirm the higher quality of the generated dialogues compared to the baselines. The source code and dataset are available at https://github.com/AIR-DISCOVER/ECAs-Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13940v1">ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Inter-GPU communication has become a major bottleneck for modern AI workloads as models scale and improvements in hardware compute throughput outpace improvements in interconnect bandwidth. Existing systems mitigate this through compute-communication overlap but often fail to meet theoretical peak performance across heterogeneous workloads and new accelerators. Instead of operator-specific techniques, we ask whether a small set of simple, reusable principles can systematically guide the design of optimal multi-GPU kernels. We present ParallelKittens (PK), a minimal CUDA framework that drastically simplifies the development of overlapped multi-GPU kernels. PK extends the ThunderKittens framework and embodies the principles of multi-GPU kernel design through eight core primitives and a unified programming template, derived from a comprehensive analysis of the factors that govern multi-GPU performance$\unicode{x2014}$data-transfer mechanisms, resource scheduling, and design overheads. We validate PK on both Hopper and Blackwell architectures. With fewer than 50 lines of device code, PK achieves up to $2.33 \times$ speedup for data- and tensor-parallel workloads, $4.08 \times$ for sequence-parallel workloads, and $1.22 \times$ for expert-parallel workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12533v1">Designing-with More-than-Human Through Human Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Submitted to DRS 2026
    </div>
    <details class="paper-abstract">
      The recent more-than-human turn in design calls for "designing-with" other species and ecologies beyond humans. Yet-as Thomas Nagel's famous "What is it like to be a bat?" thought experiment highlights-human experience is constrained by our own sensorium and an irreducible gap in phenomenal access to nonhuman lifeworlds. This paper proposes More-than-Human through Human Augmentation (MtHtHA, denoted ">HtH+") as a design approach that repurposes human augmentation technologies-typically aimed at enhancing human capabilities-away from human optimization and exceptionalism but toward eco-phenomenological awareness. Grounded in somaesthetic design and eco-somatics, MtHtHA entails creating temporary, embodied experiences that modulate the human Umwelt to re-sensitize us to pluriversal more-than-human perceptions. We articulate seven design principles and report five design cases-EchoVision (bat-like echolocation), FeltSight (star-nosed-mole tactile navigation), FungiSync (fungal network attunement), TentacUs (octopus-like distributed agency), and City of Sparkles (urban data from AI's perspective). We demonstrate that such experiential "designing-with" can cultivate ecological awareness, empathy and obligations of care across species boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12436v1">RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robotic Manipulation and Navigation</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Robotic manipulation and navigation are fundamental capabilities of embodied intelligence, enabling effective robot interactions with the physical world. Achieving these capabilities requires a cohesive understanding of the environment, including object recognition to localize target objects, object affordances to identify potential interaction areas and spatial affordances to discern optimal areas for both object placement and robot movement. While Vision-Language Models (VLMs) excel at high-level task planning and scene understanding, they often struggle to infer actionable positions for physical interaction, such as functional grasping points and permissible placement regions. This limitation stems from the lack of fine-grained annotations for object and spatial affordances in their training datasets. To tackle this challenge, we introduce RoboAfford++, a generative AI-enhanced dataset for multimodal affordance learning for both robotic manipulation and navigation. Our dataset comprises 869,987 images paired with 2.0 million question answering (QA) annotations, covering three critical tasks: object affordance recognition to identify target objects based on attributes and spatial relationships, object affordance prediction to pinpoint functional parts for manipulation, and spatial affordance localization to identify free space for object placement and robot navigation. Complementing this dataset, we propose RoboAfford-Eval, a comprehensive benchmark for assessing affordance-aware prediction in real-world scenarios, featuring 338 meticulously annotated samples across the same three tasks. Extensive experimental results reveal the deficiencies of existing VLMs in affordance learning, while fine-tuning on the RoboAfford++ dataset significantly enhances their ability to reason about object and spatial affordances, validating the dataset's effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12368v1">Fast Reasoning Segmentation for Images and Videos</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Reasoning segmentation enables open-set object segmentation via implicit text queries, therefore serving as a foundation for embodied agents that should operate autonomously in real-world environments. However, existing methods for reasoning segmentation require multimodal large language models with billions of parameters that exceed the computational capabilities of edge devices that typically deploy the embodied AI systems. Distillation offers a pathway to compress these models while preserving their capabilities. Yet, existing distillation approaches fail to transfer the multi-step reasoning capabilities that reasoning segmentation demands, as they focus on matching output predictions and intermediate features rather than preserving reasoning chains. The emerging paradigm of reasoning over digital twin representations presents an opportunity for more effective distillation by re-framing the problem. Consequently, we propose FastReasonSeg, which employs digital twin representations that decouple perception from reasoning to enable more effective distillation. Our distillation scheme first relies on supervised fine-tuning on teacher-generated reasoning chains. Then it is followed by reinforcement fine-tuning with joint rewards evaluating both segmentation accuracy and reasoning quality alignment. Experiments on two video (JiTBench, RVTBench) and two image benchmarks (ReasonSeg, LLM-Seg40K) demonstrate that our FastReasonSeg achieves state-of-the-art reasoning segmentation performance. Moreover, the distilled 0.6B variant outperforms models with 20 times more parameters while achieving 7.79 FPS throughput with only 2.1GB memory consumption. This efficiency enables deployment in resource-constrained environments to enable real-time reasoning segmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19002v2">VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent's markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12040v1">SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 AAAI2026-Oral. Project Page: https://xinyuanhu66.github.io/SRSplat/
    </div>
    <details class="paper-abstract">
      Feed-forward 3D reconstruction from sparse, low-resolution (LR) images is a crucial capability for real-world applications, such as autonomous driving and embodied AI. However, existing methods often fail to recover fine texture details. This limitation stems from the inherent lack of high-frequency information in LR inputs. To address this, we propose \textbf{SRSplat}, a feed-forward framework that reconstructs high-resolution 3D scenes from only a few LR views. Our main insight is to compensate for the deficiency of texture information by jointly leveraging external high-quality reference images and internal texture cues. We first construct a scene-specific reference gallery, generated for each scene using Multimodal Large Language Models (MLLMs) and diffusion models. To integrate this external information, we introduce the \textit{Reference-Guided Feature Enhancement (RGFE)} module, which aligns and fuses features from the LR input images and their reference twin image. Subsequently, we train a decoder to predict the Gaussian primitives using the multi-view fused feature obtained from \textit{RGFE}. To further refine predicted Gaussian primitives, we introduce \textit{Texture-Aware Density Control (TADC)}, which adaptively adjusts Gaussian density based on the internal texture richness of the LR inputs. Extensive experiments demonstrate that our SRSplat outperforms existing methods on various datasets, including RealEstate10K, ACID, and DTU, and exhibits strong cross-dataset and cross-resolution generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05456v2">SITE: towards Spatial Intelligence Thorough Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Spatial intelligence (SI) represents a cognitive ability encompassing the visualization, manipulation, and reasoning about spatial relationships, underpinning disciplines from neuroscience to robotics. We introduce SITE, a benchmark dataset towards SI Thorough Evaluation in a standardized format of multi-choice visual question-answering, designed to assess large vision-language models' spatial intelligence across diverse visual modalities (single-image, multi-image, and video) and SI factors (figural to environmental scales, spatial visualization and orientation, intrinsic and extrinsic, static and dynamic). Our approach to curating the benchmark combines a bottom-up survey about 31 existing datasets and a top-down strategy drawing upon three classification systems in cognitive science, which prompt us to design two novel types of tasks about view-taking and dynamic scenes. Extensive experiments reveal that leading models fall behind human experts especially in spatial orientation, a fundamental SI factor. Moreover, we demonstrate a positive correlation between a model's spatial reasoning proficiency and its performance on an embodied AI task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00108v2">Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v2">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.09497v1">Fundamentals of Physical AI</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This paper is already published in Journal of Intelligent System of Systems Lifecycle Management
    </div>
    <details class="paper-abstract">
      This work will elaborate the fundamental principles of physical artificial intelligence (Physical AI) from a scientific and systemic perspective. The aim is to create a theoretical foundation that describes the physical embodiment, sensory perception, ability to act, learning processes, and context sensitivity of intelligent systems within a coherent framework. While classical AI approaches rely on symbolic processing and data driven models, Physical AI understands intelligence as an emergent phenomenon of real interaction between body, environment, and experience. The six fundamentals presented here are embodiment, sensory perception, motor action, learning, autonomy, and context sensitivity, and form the conceptual basis for designing and evaluating physically intelligent systems. Theoretically, it is shown that these six principles do not represent loose functional modules but rather act as a closed control loop in which energy, information, control, and context are in constant interaction. This circular interaction enables a system to generate meaning not from databases, but from physical experience, a paradigm shift that understands intelligence as an physical embodied process. Physical AI understands learning not as parameter adjustment, but as a change in the structural coupling between agents and the environment. To illustrate this, the theoretical model is explained using a practical scenario: An adaptive assistant robot supports patients in a rehabilitation clinic. This example illustrates that physical intelligence does not arise from abstract calculation, but from immediate, embodied experience. It shows how the six fundamentals interact in a real system: embodiment as a prerequisite, perception as input, movement as expression, learning as adaptation, autonomy as regulation, and context as orientation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07090v3">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2508.15201v2">Survey of Vision-Language-Action Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07090v4">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09964v1">EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for instrument control requires methods that go beyond standard, stateless algorithmic benchmarks, since the behavior of physical systems cannot be fully captured by unit tests alone. Here we introduce EnvTrace, a simulation-based method that evaluates execution traces to assess semantic code equivalence. EnvTrace is demonstrated with a beamline control-logic digital twin to facilitate the evaluation of instrument control code, with the digital twin itself also enabling the pre-execution validation of live experiments. Over 30 LLMs were evaluated using trace alignment to generate a multi-faceted score for functional correctness across key behavioral dimensions, showing that many top-tier models can approach human-level performance in rapid control-code generation. This is a first step toward a broader vision where LLMs and digital twins work symbiotically: LLMs providing intuitive control and agentic orchestration, and digital twins offering safe and high-fidelity environments, paving the way towards autonomous embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07090v2">Green AI: A systematic review and meta-analysis of its definitions, lifecycle models, hardware and measurement attempts</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Across the Artificial Intelligence (AI) lifecycle - from hardware to development, deployment, and reuse - burdens span energy, carbon, water, and embodied impacts. Cloud provider tools improve transparency but remain heterogeneous and often omit water and value chain effects, limiting comparability and reproducibility. Addressing these multi dimensional burdens requires a lifecycle approach linking phase explicit mapping with system levers (hardware, placement, energy mix, cooling, scheduling) and calibrated measurement across facility, system, device, and workload levels. This article (i) establishes a unified, operational definition of Green AI distinct from Sustainable AI; (ii) formalizes a five phase lifecycle mapped to Life Cycle Assessment (LCA) stages, making energy, carbon, water, and embodied impacts first class; (iii) specifies governance via Plan Do Check Act (PDCA) cycles with decision gateways; (iv) systematizes hardware and system level strategies across the edge cloud continuum to reduce embodied burdens; and (v) defines a calibrated measurement framework combining estimator models with direct metering to enable reproducible, provider agnostic comparisons. Combining definition, lifecycle processes, hardware strategies, and calibrated measurement, this article offers actionable, evidence based guidance for researchers, practitioners, and policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09497v1">Fundamentals of Physical AI</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 This paper is already published in Journal of Intelligent System of Systems Lifecycle Management
    </div>
    <details class="paper-abstract">
      This work will elaborate the fundamental principles of physical artificial intelligence (Physical AI) from a scientific and systemic perspective. The aim is to create a theoretical foundation that describes the physical embodiment, sensory perception, ability to act, learning processes, and context sensitivity of intelligent systems within a coherent framework. While classical AI approaches rely on symbolic processing and data driven models, Physical AI understands intelligence as an emergent phenomenon of real interaction between body, environment, and experience. The six fundamentals presented here are embodiment, sensory perception, motor action, learning, autonomy, and context sensitivity, and form the conceptual basis for designing and evaluating physically intelligent systems. Theoretically, it is shown that these six principles do not represent loose functional modules but rather act as a closed control loop in which energy, information, control, and context are in constant interaction. This circular interaction enables a system to generate meaning not from databases, but from physical experience, a paradigm shift that understands intelligence as an physical embodied process. Physical AI understands learning not as parameter adjustment, but as a change in the structural coupling between agents and the environment. To illustrate this, the theoretical model is explained using a practical scenario: An adaptive assistant robot supports patients in a rehabilitation clinic. This example illustrates that physical intelligence does not arise from abstract calculation, but from immediate, embodied experience. It shows how the six fundamentals interact in a real system: embodiment as a prerequisite, perception as input, movement as expression, learning as adaptation, autonomy as regulation, and context as orientation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15201v2">Survey of Vision-Language-Action Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05294v4">Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08007v2">EAGLE: Episodic Appearance- and Geometry-aware Memory for Unified 2D-3D Visual Query Localization in Egocentric Vision</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 13 Pages, accepted by AAAI-2026
    </div>
    <details class="paper-abstract">
      Egocentric visual query localization is vital for embodied AI and VR/AR, yet remains challenging due to camera motion, viewpoint changes, and appearance variations. We present EAGLE, a novel framework that leverages episodic appearance- and geometry-aware memory to achieve unified 2D-3D visual query localization in egocentric vision. Inspired by avian memory consolidation, EAGLE synergistically integrates segmentation guided by an appearance-aware meta-learning memory (AMM), with tracking driven by a geometry-aware localization memory (GLM). This memory consolidation mechanism, through structured appearance and geometry memory banks, stores high-confidence retrieval samples, effectively supporting both long- and short-term modeling of target appearance variations. This enables precise contour delineation with robust spatial discrimination, leading to significantly improved retrieval accuracy. Furthermore, by integrating the VQL-2D output with a visual geometry grounded Transformer (VGGT), we achieve a efficient unification of 2D and 3D tasks, enabling rapid and accurate back-projection into 3D space. Our method achieves state-ofthe-art performance on the Ego4D-VQ benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07412v1">TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07161v1">LLMscape</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to NeurIPS 2025, Creative AI Track
    </div>
    <details class="paper-abstract">
      LLMscape is an interactive installation that investigates how humans and AI construct meaning under shared conditions of uncertainty. Within a mutable, projection-mapped landscape, human participants reshape the world and engage with multiple AI agents, each developing incomplete and provisional accounts of their environment. Exhibited in Shanghai and continually evolving, the work positions AI not as deterministic tools but as embodied co-witnesses to an unstable world, examining the parallels between human and artificial meaning-making and inviting reflection on our shared epistemic limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.01386v4">CATransformers: Carbon Aware Transformers Through Joint Model-Hardware Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Machine learning solutions are rapidly adopted to enable a variety of key use cases, from conversational AI assistants to scientific discovery. This growing adoption is expected to increase the associated lifecycle carbon footprint, including both \emph{operational carbon} from training and inference and \emph{embodied carbon} from AI hardware manufacturing. We introduce \ourframework -- the first carbon-aware co-optimization framework for Transformer-based models and hardware accelerators. By integrating both operational and embodied carbon into early-stage design space exploration, \ourframework enables sustainability-driven model architecture and hardware accelerator co-design that reveals fundamentally different trade-offs than latency- or energy-centric approaches. Evaluated across a range of Transformer models, \ourframework consistently demonstrates the potential to reduce total carbon emissions -- by up to 30\% -- while maintaining accuracy and latency. We further highlight its extensibility through a focused case study on multi-modal models. Our results emphasize the need for holistic optimization methods that prioritize carbon efficiency without compromising model capability and execution time performance. The source code of \ourframework is available at {\small{\href{https://github.com/facebookresearch/CATransformers}{\texttt{https://github.com/facebookresearch/CATransformers}}}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08896v4">Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11773v4">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at https://github.com/ZikangLeng/AgentSense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07412v1">TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07161v1">LLMscape</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to NeurIPS 2025, Creative AI Track
    </div>
    <details class="paper-abstract">
      LLMscape is an interactive installation that investigates how humans and AI construct meaning under shared conditions of uncertainty. Within a mutable, projection-mapped landscape, human participants reshape the world and engage with multiple AI agents, each developing incomplete and provisional accounts of their environment. Exhibited in Shanghai and continually evolving, the work positions AI not as deterministic tools but as embodied co-witnesses to an unstable world, examining the parallels between human and artificial meaning-making and inviting reflection on our shared epistemic limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07682v1">Designing and Evaluating Malinowski's Lens: An AI-Native Educational Game for Ethnographic Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 21 pages, 8 figures. Full preprint version; shorter version in preparation
    </div>
    <details class="paper-abstract">
      This study introduces 'Malinowski's Lens', the first AI-native educational game for anthropology that transforms Bronislaw Malinowski's 'Argonauts of the Western Pacific' (1922) into an interactive learning experience. The system combines Retrieval-Augmented Generation with DALL-E 3 text-to-image generation, creating consistent VGA-style visuals as players embody Malinowski during his Trobriand Islands fieldwork (1915-1918). To address ethical concerns, indigenous peoples appear as silhouettes while Malinowski is detailed, prompting reflection on anthropological representation. Two validation studies confirmed effectiveness: Study 1 with 10 non-specialists showed strong learning outcomes (average quiz score 7.5/10) and excellent usability (SUS: 83/100). Study 2 with 4 expert anthropologists confirmed pedagogical value, with one senior researcher discovering "new aspects" of Malinowski's work through gameplay. The findings demonstrate that AI-driven educational games can effectively convey complex anthropological concepts while sparking disciplinary curiosity. This study advances AI-native educational game design and provides a replicable model for transforming academic texts into engaging interactive experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06619v1">How Do VLAs Effectively Inherit from VLMs?</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Vision-language-action (VLA) models hold the promise to attain generalizable embodied control. To achieve this, a pervasive paradigm is to leverage the rich vision-semantic priors of large vision-language models (VLMs). However, the fundamental question persists: How do VLAs effectively inherit the prior knowledge from VLMs? To address this critical question, we introduce a diagnostic benchmark, GrinningFace, an emoji tabletop manipulation task where the robot arm is asked to place objects onto printed emojis corresponding to language instructions. This task design is particularly revealing -- knowledge associated with emojis is ubiquitous in Internet-scale datasets used for VLM pre-training, yet emojis themselves are largely absent from standard robotics datasets. Consequently, they provide a clean proxy: successful task completion indicates effective transfer of VLM priors to embodied control. We implement this diagnostic task in both simulated environment and a real robot, and compare various promising techniques for knowledge transfer. Specifically, we investigate the effects of parameter-efficient fine-tuning, VLM freezing, co-training, predicting discretized actions, and predicting latent actions. Through systematic evaluation, our work not only demonstrates the critical importance of preserving VLM priors for the generalization of VLA but also establishes guidelines for future research in developing truly generalizable embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06185v1">Dataforge: A Data Agent Platform for Autonomous Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-09
    </div>
    <details class="paper-abstract">
      The growing demand for AI applications in fields such as materials discovery, molecular modeling, and climate science has made data preparation an important but labor-intensive step. Raw data from diverse sources must be cleaned, normalized, and transformed to become AI-ready, while effective feature transformation and selection are essential for efficient training and inference. To address the challenges of scalability and expertise dependence, we present Data Agent, a fully autonomous system specialized for tabular data. Leveraging large language model (LLM) reasoning and grounded validation, Data Agent automatically performs data cleaning, hierarchical routing, and feature-level optimization through dual feedback loops. It embodies three core principles: automatic, safe, and non-expert friendly, which ensure end-to-end reliability without human supervision. This demo showcases the first practical realization of an autonomous Data Agent, illustrating how raw data can be transformed "From Data to Better Data."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05936v1">10 Open Challenges Steering the Future of Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2025-11-08
      | 💬 AAAI 2026 (Senior Track)
    </div>
    <details class="paper-abstract">
      Due to their ability of follow natural language instructions, vision-language-action (VLA) models are increasingly prevalent in the embodied AI arena, following the widespread success of their precursors -- LLMs and VLMs. In this paper, we discuss 10 principal milestones in the ongoing development of VLA models -- multimodality, reasoning, data, evaluation, cross-robot action generalization, efficiency, whole-body coordination, safety, agents, and coordination with humans. Furthermore, we discuss the emerging trends of using spatial understanding, modeling world dynamics, post training, and data synthesis -- all aiming to reach these milestones. Through these discussions, we hope to bring attention to the research avenues that may accelerate the development of VLA models into wider acceptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05304v1">psiUnity: A Platform for Multimodal Data-Driven XR</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Extended reality (XR) research increasingly relies on the ability to stream and synchronize multimodal data between headsets and immersive applications for data-driven interaction and experimentation. However, developers face a critical gap: the Platform for Situated Intelligence (psi), which excels at deterministic temporal alignment and multimodal data management, has been largely inaccessible to the dominant Unity/MRTK ecosystem used for HoloLens development. We introduce psiUnity, an open-source C# integration that bridges psi's .NET libraries with Unity 2022.3 and MRTK3 for HoloLens 2. psiUnity enables bidirectional, real-time streaming of head pose, hand tracking, gaze, IMU, audio, and depth sensor data (AHAT and long-throw) with microsecond-level temporal precision, allowing Unity applications to both consume and produce synchronized multimodal data streams. By embedding psi's native serialization, logging, and temporal coordination directly within Unity's architecture, psiUnity extends psi beyond its previous StereoKit limitations and empowers the HRI, HCI, and embodied-AI communities to develop reproducible, data-driven XR interactions and experiments within the familiar Unity environment. The integration is available at https://github.com/sailgt/psiUnity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04976v1">iFlyBot-VLM Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05304v1">psiUnity: A Platform for Multimodal Data-Driven XR</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Extended reality (XR) research increasingly relies on the ability to stream and synchronize multimodal data between headsets and immersive applications for data-driven interaction and experimentation. However, developers face a critical gap: the Platform for Situated Intelligence (psi), which excels at deterministic temporal alignment and multimodal data management, has been largely inaccessible to the dominant Unity/MRTK ecosystem used for HoloLens development. We introduce psiUnity, an open-source C# integration that bridges psi's .NET libraries with Unity 2022.3 and MRTK3 for HoloLens 2. psiUnity enables bidirectional, real-time streaming of head pose, hand tracking, gaze, IMU, audio, and depth sensor data (AHAT and long-throw) with microsecond-level temporal precision, allowing Unity applications to both consume and produce synchronized multimodal data streams. By embedding psi's native serialization, logging, and temporal coordination directly within Unity's architecture, psiUnity extends psi beyond its previous StereoKit limitations and empowers the HRI, HCI, and embodied-AI communities to develop reproducible, data-driven XR interactions and experiments within the familiar Unity environment. The integration is available at https://github.com/sailgt/psiUnity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04976v1">iFlyBot-VLM Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02851v2">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML with computational resources in mobile, edge, and cloud connected over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: a lightweight on-device model locally generates drafts while a more capable remote target model on a server verifies and corrects them in parallel with speculative sampling, thus resulting in lower latency without compromising accuracy. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications, cannot parallelize verification and correction for multiple drafts as each generated action depends on observation updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference (ADAHI), wherein drafts are selectively transmitted and verified based on action deviation, which has a strong correlation with action's rejection probability by the target model. By invoking server operation only when necessary, communication and computational overhead can be reduced while accuracy gain from speculative sampling is preserved. Experiments on our testbed show that ADAHI reduces transmission and server operations by approximately 40%, lowers end-to-end latency by 39.2%, and attains up to 97.2% of the task-success rate of baseline that invokes speculative sampling for every draft embedding vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03997v1">PhysCorr: Dual-Reward DPO for Physics-Constrained Text-to-Video Generation with Automated Preference Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-video generation have achieved impressive perceptual quality, yet generated content often violates fundamental principles of physical plausibility - manifesting as implausible object dynamics, incoherent interactions, and unrealistic motion patterns. Such failures hinder the deployment of video generation models in embodied AI, robotics, and simulation-intensive domains. To bridge this gap, we propose PhysCorr, a unified framework for modeling, evaluating, and optimizing physical consistency in video generation. Specifically, we introduce PhysicsRM, the first dual-dimensional reward model that quantifies both intra-object stability and inter-object interactions. On this foundation, we develop PhyDPO, a novel direct preference optimization pipeline that leverages contrastive feedback and physics-aware reweighting to guide generation toward physically coherent outputs. Our approach is model-agnostic and scalable, enabling seamless integration into a wide range of video diffusion and transformer-based backbones. Extensive experiments across multiple benchmarks demonstrate that PhysCorr achieves significant improvements in physical realism while preserving visual fidelity and semantic alignment. This work takes a critical step toward physically grounded and trustworthy video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.02979v1">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Submitted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02851v2">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML with computational resources in mobile, edge, and cloud connected over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: a lightweight on-device model locally generates drafts while a more capable remote target model on a server verifies and corrects them in parallel with speculative sampling, thus resulting in lower latency without compromising accuracy. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications, cannot parallelize verification and correction for multiple drafts as each generated action depends on observation updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference (ADAHI), wherein drafts are selectively transmitted and verified based on action deviation, which has a strong correlation with action's rejection probability by the target model. By invoking server operation only when necessary, communication and computational overhead can be reduced while accuracy gain from speculative sampling is preserved. Experiments on our testbed show that ADAHI reduces transmission and server operations by approximately 40%, lowers end-to-end latency by 39.2%, and attains up to 97.2% of the task-success rate of baseline that invokes speculative sampling for every draft embedding vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04157v1">Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03997v1">PhysCorr: Dual-Reward DPO for Physics-Constrained Text-to-Video Generation with Automated Preference Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent advances in text-to-video generation have achieved impressive perceptual quality, yet generated content often violates fundamental principles of physical plausibility - manifesting as implausible object dynamics, incoherent interactions, and unrealistic motion patterns. Such failures hinder the deployment of video generation models in embodied AI, robotics, and simulation-intensive domains. To bridge this gap, we propose PhysCorr, a unified framework for modeling, evaluating, and optimizing physical consistency in video generation. Specifically, we introduce PhysicsRM, the first dual-dimensional reward model that quantifies both intra-object stability and inter-object interactions. On this foundation, we develop PhyDPO, a novel direct preference optimization pipeline that leverages contrastive feedback and physics-aware reweighting to guide generation toward physically coherent outputs. Our approach is model-agnostic and scalable, enabling seamless integration into a wide range of video diffusion and transformer-based backbones. Extensive experiments across multiple benchmarks demonstrate that PhysCorr achieves significant improvements in physical realism while preserving visual fidelity and semantic alignment. This work takes a critical step toward physically grounded and trustworthy video generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03992v1">CaRF: Enhancing Multi-View Consistency in Referring 3D Gaussian Splatting Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Referring 3D Gaussian Splatting Segmentation (R3DGS) aims to interpret free-form language expressions and localize the corresponding 3D regions in Gaussian fields. While recent advances have introduced cross-modal alignment between language and 3D geometry, existing pipelines still struggle with cross-view consistency due to their reliance on 2D rendered pseudo supervision and view specific feature learning. In this work, we present Camera Aware Referring Field (CaRF), a fully differentiable framework that operates directly in the 3D Gaussian space and achieves multi view consistency. Specifically, CaRF introduces Gaussian Field Camera Encoding (GFCE), which incorporates camera geometry into Gaussian text interactions to explicitly model view dependent variations and enhance geometric reasoning. Building on this, In Training Paired View Supervision (ITPVS) is proposed to align per Gaussian logits across calibrated views during training, effectively mitigating single view overfitting and exposing inter view discrepancies for optimization. Extensive experiments on three representative benchmarks demonstrate that CaRF achieves average improvements of 16.8%, 4.3%, and 2.0% in mIoU over state of the art methods on the Ref LERF, LERF OVS, and 3D OVS datasets, respectively. Moreover, this work promotes more reliable and view consistent 3D scene understanding, with potential benefits for embodied AI, AR/VR interaction, and autonomous perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03370v1">EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.02071v1">Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03497v1">ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03370v1">EQ-Negotiator: Dynamic Emotional Personas Empower Small Language Models for Edge-Deployable Credit Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-11-05
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) in automated negotiation has set a high performance benchmark, but their computational cost and data privacy requirements render them unsuitable for many privacy-sensitive, on-device applications such as mobile assistants, embodied AI agents or private client interactions. While small language models (SLMs) offer a practical alternative, they suffer from a significant performance gap compared to LLMs in playing emotionally charged complex personas, especially for credit negotiation. This paper introduces EQ-Negotiator, a novel framework that bridges this capability gap using emotional personas. Its core is a reasoning system that integrates game theory with a Hidden Markov Model(HMM) to learn and track debtor emotional states online, without pre-training. This allows EQ-Negotiator to equip SLMs with the strategic intelligence to counter manipulation while de-escalating conflict and upholding ethical standards. Through extensive agent-to-agent simulations across diverse credit negotiation scenarios, including adversarial debtor strategies like cheating, threatening, and playing the victim, we show that a 7B parameter language model with EQ-Negotiator achieves better debt recovery and negotiation efficiency than baseline LLMs more than 10 times its size. This work advances persona modeling from descriptive character profiles to dynamic emotional architectures that operate within privacy constraints. Besides, this paper establishes that strategic emotional intelligence, not raw model scale, is the critical factor for success in automated negotiation, paving the way for effective, ethical, and privacy-preserving AI negotiators that can operate on the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00529v2">On Improvisation and Open-Endedness: Insights for Experiential AI</a></div>
    <div class="paper-meta">
      📅 2025-11-05
      | 💬 Submitted to AAAI 2026 Creative AI for Live Interactive Performances Workshop (CLIP) as a work-in-progress paper
    </div>
    <details class="paper-abstract">
      Improvisation-the art of spontaneous creation that unfolds moment-to-moment without a scripted outcome-requires practitioners to continuously sense, adapt, and create anew. It is a fundamental mode of human creativity spanning music, dance, and everyday life. The open-ended nature of improvisation produces a stream of novel, unrepeatable moments-an aspect highly valued in artistic creativity. In parallel, open-endedness (OE)-a system's capacity for unbounded novelty and endless "interestingness"-is exemplified in natural or cultural evolution and has been considered "the last grand challenge" in artificial life (ALife). The rise of generative AI now raises the question in computational creativity (CC) research: What makes a "good" improvisation for AI? Can AI learn to improvise in a genuinely open-ended way? In this work-in-progress paper, we report insights from in-depth interviews with 6 experts in improvisation across dance, music, and contact improvisation. We draw systemic connections between human improvisational arts and the design of future experiential AI agents that could improvise alone or alongside humans-or even with other AI agents-embodying qualities of improvisation drawn from practice: active listening (umwelt and awareness), being in the time (mindfulness and ephemerality), embracing the unknown (source of randomness and serendipity), non-judgmental flow (acceptance and dynamical stability, balancing structure and surprise (unpredictable criticality at edge of chaos), imaginative metaphor (synaesthesia and planning), empathy, trust, boundary, and care (mutual theory of mind), and playfulness and intrinsic motivation (maintaining interestingness).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02912v3">Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Published in the Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments for Agents (SEA). Additionally accepted for presentation in the NeurIPS 2025 Workshop: Embodied World Models for Decision Making (EWM) and the NeurIPS 2025 Workshop: Optimization for Machine Learning (OPT)
    </div>
    <details class="paper-abstract">
      Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02225v1">Learning Interactive World Model for Object-Centric Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02979v1">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Submitted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02374v1">AyurParam: A State-of-the-Art Bilingual Language Model for Ayurveda</a></div>
    <div class="paper-meta">
      📅 2025-11-04
    </div>
    <details class="paper-abstract">
      Current large language models excel at broad, general-purpose tasks, but consistently underperform when exposed to highly specialized domains that require deep cultural, linguistic, and subject-matter expertise. In particular, traditional medical systems such as Ayurveda embody centuries of nuanced textual and clinical knowledge that mainstream LLMs fail to accurately interpret or apply. We introduce AyurParam-2.9B, a domain-specialized, bilingual language model fine-tuned from Param-1-2.9B using an extensive, expertly curated Ayurveda dataset spanning classical texts and clinical guidance. AyurParam's dataset incorporates context-aware, reasoning, and objective-style Q&A in both English and Hindi, with rigorous annotation protocols for factual precision and instructional clarity. Benchmarked on BhashaBench-Ayur, AyurParam not only surpasses all open-source instruction-tuned models in its size class (1.5--3B parameters), but also demonstrates competitive or superior performance compared to much larger models. The results from AyurParam highlight the necessity for authentic domain adaptation and high-quality supervision in delivering reliable, culturally congruent AI for specialized medical knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02912v3">Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 Published in the Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments for Agents (SEA). Additionally accepted for presentation in the NeurIPS 2025 Workshop: Embodied World Models for Decision Making (EWM) and the NeurIPS 2025 Workshop: Optimization for Machine Learning (OPT)
    </div>
    <details class="paper-abstract">
      Robust coordination is critical for effective decision-making in multi-agent systems, especially under partial observability. A central question in Multi-Agent Reinforcement Learning (MARL) is whether to engineer communication protocols or learn them end-to-end. We investigate this dichotomy using embodied world models. We propose and compare two communication strategies for a cooperative task-allocation problem. The first, Learned Direct Communication (LDC), learns a protocol end-to-end. The second, Intention Communication, uses an engineered inductive bias: a compact, learned world model, the Imagined Trajectory Generation Module (ITGM), which uses the agent's own policy to simulate future states. A Message Generation Network (MGN) then compresses this plan into a message. We evaluate these approaches on goal-directed interaction in a grid world, a canonical abstraction for embodied AI problems, while scaling environmental complexity. Our experiments reveal that while emergent communication is viable in simple settings, the engineered, world model-based approach shows superior performance, sample efficiency, and scalability as complexity increases. These findings advocate for integrating structured, predictive models into MARL agents to enable active, goal-driven coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02225v1">Learning Interactive World Model for Object-Centric Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-04
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Agents that understand objects and their interactions can learn policies that are more robust and transferable. However, most object-centric RL methods factor state by individual objects while leaving interactions implicit. We introduce the Factored Interactive Object-Centric World Model (FIOC-WM), a unified framework that learns structured representations of both objects and their interactions within a world model. FIOC-WM captures environment dynamics with disentangled and modular representations of object interactions, improving sample efficiency and generalization for policy learning. Concretely, FIOC-WM first learns object-centric latents and an interaction structure directly from pixels, leveraging pre-trained vision encoders. The learned world model then decomposes tasks into composable interaction primitives, and a hierarchical policy is trained on top: a high level selects the type and order of interactions, while a low level executes them. On simulated robotic and embodied-AI benchmarks, FIOC-WM improves policy-learning sample efficiency and generalization over world-model baselines, indicating that explicit, modular interaction learning is crucial for robust control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02071v1">Human-AI Co-Embodied Intelligence for Scientific Experimentation and Manufacturing</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      Scientific experiment and manufacture rely on complex, multi-step procedures that demand continuous human expertise for precise execution and decision-making. Despite advances in machine learning and automation, conventional models remain confined to virtual domains, while real-world experiment and manufacture still rely on human supervision and expertise. This gap between machine intelligence and physical execution limits reproducibility, scalability, and accessibility across scientific and manufacture workflows. Here, we introduce human-AI co-embodied intelligence, a new form of physical AI that unites human users, agentic AI, and wearable hardware into an integrated system for real-world experiment and intelligent manufacture. In this paradigm, humans provide precise execution and control, while agentic AI contributes memory, contextual reasoning, adaptive planning, and real-time feedback. The wearable interface continuously captures the experimental and manufacture processes, facilitates seamless communication between humans and AI for corrective guidance and interpretable collaboration. As a demonstration, we present Agentic-Physical Experimentation (APEX) system, coupling agentic reasoning with physical execution through mixed-reality. APEX observes and interprets human actions, aligns them with standard operating procedures, provides 3D visual guidance, and analyzes every step. Implemented in a cleanroom for flexible electronics fabrication, APEX system achieves context-aware reasoning with accuracy exceeding general multimodal large language models, corrects errors in real time, and transfers expertise to beginners. These results establish a new class of agentic-physical-human intelligence that extends agentic reasoning beyond computation into the physical domain, transforming scientific research and manufacturing into autonomous, traceable, interpretable, and scalable processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01663v1">The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity</a></div>
    <div class="paper-meta">
      📅 2025-11-03
    </div>
    <details class="paper-abstract">
      While generative models for music composition are increasingly capable, their adoption by musicians is hindered by text-prompting, an asynchronous workflow disconnected from the embodied, responsive nature of instrumental performance. To address this, we introduce Aria-Duet, an interactive system facilitating a real-time musical duet between a human pianist and Aria, a state-of-the-art generative model, using a Yamaha Disklavier as a shared physical interface. The framework enables a turn-taking collaboration: the user performs, signals a handover, and the model generates a coherent continuation performed acoustically on the piano. Beyond describing the technical architecture enabling this low-latency interaction, we analyze the system's output from a musicological perspective, finding the model can maintain stylistic semantics and develop coherent phrasal ideas, demonstrating that such embodied systems can engage in musically sophisticated dialogue and open a promising new path for human-AI co-creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25760v2">Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00940v1">URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00940v1">URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model</a></div>
    <div class="paper-meta">
      📅 2025-11-02
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25760v2">Multimodal Spatial Reasoning in the Large Model Era: A Survey and Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-02
    </div>
    <details class="paper-abstract">
      Humans possess spatial reasoning abilities that enable them to understand spaces through multimodal observations, such as vision and sound. Large multimodal reasoning models extend these abilities by learning to perceive and reason, showing promising performance across diverse spatial tasks. However, systematic reviews and publicly available benchmarks for these models remain limited. In this survey, we provide a comprehensive review of multimodal spatial reasoning tasks with large models, categorizing recent progress in multimodal large language models (MLLMs) and introducing open benchmarks for evaluation. We begin by outlining general spatial reasoning, focusing on post-training techniques, explainability, and architecture. Beyond classical 2D tasks, we examine spatial relationship reasoning, scene and layout understanding, as well as visual question answering and grounding in 3D space. We also review advances in embodied AI, including vision-language navigation and action models. Additionally, we consider emerging modalities such as audio and egocentric video, which contribute to novel spatial understanding through new sensors. We believe this survey establishes a solid foundation and offers insights into the growing field of multimodal spatial reasoning. Updated information about this survey, codes and implementation of the open benchmarks can be found at https://github.com/zhengxuJosh/Awesome-Spatial-Reasoning.
    </details>
</div>
