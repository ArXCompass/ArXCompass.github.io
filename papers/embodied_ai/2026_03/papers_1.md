# embodied ai - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11755v1">Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints</a></div>
    <div class="paper-meta">
      📅 2026-03-12
    </div>
    <details class="paper-abstract">
      Motion-controllable video generation is crucial for egocentric applications in virtual reality and embodied AI. However, existing methods often struggle to achieve 3D-consistent fine-grained hand articulation. By adopting on 2D trajectories or implicit poses, they collapse 3D geometry into spatially ambiguous signals or over rely on human-centric priors. Under severe egocentric occlusions, this causes motion inconsistencies and hallucinated artifacts, as well as preventing cross-embodiment generalization to robotic hands. To address these limitations, we propose a novel framework that generates egocentric videos from a single reference frame, leveraging sparse 3D hand joints as embodiment-agnostic control signals with clear semantic and geometric structures. We introduce an efficient control module that resolves occlusion ambiguities while fully preserving 3D information. Specifically, it extracts occlusion-aware features from the source reference frame by penalizing unreliable visual signals from hidden joints, and employs a 3D-based weighting mechanism to robustly handle dynamically occluded target joints during motion propagation. Concurrently, the module directly injects 3D geometric embeddings into the latent space to strictly enforce structural consistency. To facilitate robust training and evaluation, we develop an automated annotation pipeline that yields over one million high-quality egocentric video clips paired with precise hand trajectories. Additionally, we register humanoid kinematic and camera data to construct a cross-embodiment benchmark. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art baselines, generating high-fidelity egocentric videos with realistic interactions and exhibiting exceptional cross-embodiment generalization to robotic hands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07791v3">GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-03-12
      | 💬 ICLR 2026, 31 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Recently spatial-temporal intelligence of Visual-Language Models (VLMs) has attracted much attention due to its importance for autonomous driving, embodied AI and general AI. Existing spatial-temporal benchmarks mainly focus on egocentric (first-person) perspective reasoning using images/video contexts, or geographic reasoning with graphical context (e.g., maps), thus fail to assess VLMs' geographic spatial-temporal intelligence that requires integrating both images/video and graphical context, which is crucial for real-world scenarios such as traffic management and emergency response. To address the gaps, we introduce Geo-Temporal Reasoning benchmark (GTR-Bench), a novel challenge for geographic temporal reasoning of moving targets in a large-scale camera network. GTR-Bench is more challenging as it requires multiple perspective switches between maps and videos, joint reasoning across multiple videos with non-overlapping fields of view, and inference over spatial-temporal regions that are unobserved by any video context. Evaluations of more than 10 popular VLMs on GTR-Bench show that even the best proprietary model, Gemini-2.5-Pro (34.9\%), significantly lags behind human performance (78.61\%) on geo-temporal reasoning. Moreover, our comprehensive analysis on GTR-Bench reveals three major deficiencies of current models for geo-temporal reasoning. (1) VLMs exhibit imbalanced utilization of spatial and temporal context during reasoning. (2) they show weak temporal forecasting ability, leading to poorer performance on temporally focused tasks. (3) they lack the capability to effectively align and integrate map data with multi-view video inputs. We believe GTR-Bench offers valuable insights and opens up new opportunities for research and applications in spatial-temporal intelligence. Benchmark and code will be released at https://github.com/X-Luffy/GTR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11320v1">UniCompress: Token Compression for Unified Vision-Language Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Unified models aim to support both understanding and generation by encoding images into discrete tokens and processing them alongside text within a single autoregressive framework. This unified design offers architectural simplicity and cross-modal synergy, which facilitates shared parameterization, consistent training objectives, and seamless transfer between modalities. However, the large number of visual tokens required by such models introduces substantial computation and memory overhead, and this inefficiency directly hinders deployment in resource constrained scenarios such as embodied AI systems. In this work, we propose a unified token compression algorithm UniCompress that significantly reduces visual token count while preserving performance on both image understanding and generation tasks. Our method introduces a plug-in compression and decompression mechanism guided with learnable global meta tokens. The framework is lightweight and modular, enabling efficient integration into existing models without full retraining. Experimental results show that our approach reduces image tokens by up to 4 times, achieves substantial gains in inference latency and training cost, and incurs only minimal performance degradation, which demonstrates the promise of token-efficient unified modeling for real world multimodal applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08420v2">Human-Aware Robot Behaviour in Self-Driving Labs</a></div>
    <div class="paper-meta">
      📅 2026-03-11
    </div>
    <details class="paper-abstract">
      Self-driving laboratories (SDLs) are rapidly transforming research in chemistry and materials science to accelerate new discoveries. Mobile robot chemists (MRCs) play a pivotal role by autonomously navigating the lab to transport samples, effectively connecting synthesis, analysis, and characterisation equipment. The instruments within an SDL are typically designed or retrofitted to be accessed by both human and robotic chemists, ensuring operational flexibility and integration between manual and automated workflows. In many scenarios, human and robotic chemists may need to use the same equipment simultaneously. Currently, MRCs rely on simple LiDAR-based obstruction detection, which forces the robot to passively wait if a human is present. This lack of situational awareness leads to unnecessary delays and inefficient coordination in time-critical automated workflows in human-robot shared labs. To address this, we present an initial study of an embodied, AI-driven perception method that facilitates proactive human-robot interaction in shared-access scenarios. Our method features a hierarchical human intention prediction model that allows the robot to distinguish between preparatory actions (waiting) and transient interactions (accessing the instrument). Our results demonstrate that the proposed approach enhances efficiency by enabling proactive human-robot interaction, streamlining coordination, and potentially increasing the efficiency of autonomous scientific labs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.22046v4">PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Project page: https://city-super.github.io/PLANING/
    </div>
    <details class="paper-abstract">
      Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of PLANING make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2305.17066v2">Mindstorms in Natural Language-Based Societies of Mind</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 published in Computational Visual Media Journal (CVMJ); 9 pages in main text + 7 pages of references + 38 pages of appendices, 14 figures in main text + 13 in appendices, 7 tables in appendices
    </div>
    <details class="paper-abstract">
      Both Minsky's "society of mind" and Schmidhuber's "learning to think" inspire diverse societies of large multimodal neural networks (NNs) that solve problems by interviewing each other in a "mindstorm." Recent implementations of NN-based societies of minds consist of large language models (LLMs) and other NN-based experts communicating through a natural language interface. In doing so, they overcome the limitations of single LLMs, improving multimodal zero-shot reasoning. In these natural language-based societies of mind (NLSOMs), new agents -- all communicating through the same universal symbolic language -- are easily added in a modular fashion. To demonstrate the power of NLSOMs, we assemble and experiment with several of them (having up to 129 members), leveraging mindstorms in them to solve some practical AI tasks: visual question answering, image captioning, text-to-image synthesis, 3D generation, egocentric retrieval, embodied AI, and general language-based task solving. We view this as a starting point towards much larger NLSOMs with billions of agents-some of which may be humans. And with this emergence of great societies of heterogeneous minds, many new research questions have suddenly become paramount to the future of artificial intelligence. What should be the social structure of an NLSOM? What would be the (dis)advantages of having a monarchical rather than a democratic structure? How can principles of NN economies be used to maximize the total reward of a reinforcement learning NLSOM? In this work, we identify, discuss, and try to answer some of these questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.09827v2">MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-11
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      As embodied models become powerful, humans will collaborate with multiple embodied AI agents at their workplace or home in the future. To ensure better communication between human users and the multi-agent system, it is crucial to interpret incoming information from agents in parallel and refer to the appropriate context for each query. Existing challenges include effectively compressing and communicating high volumes of individual sensory inputs in the form of video and correctly aggregating multiple egocentric videos to construct system-level memory. In this work, we first formally define a novel problem of understanding multiple long-horizon egocentric videos simultaneously collected from embodied agents. To facilitate research in this direction, we introduce MultiAgent-EgoQA (MA-EgoQA), a benchmark designed to systemically evaluate existing models in our scenario. MA-EgoQA provides 1.7k questions unique to multiple egocentric streams, spanning five categories: social interaction, task coordination, theory-of-mind, temporal reasoning, and environmental interaction. We further propose a simple baseline model for MA-EgoQA named EgoMAS, which leverages shared memory across embodied agents and agent-wise dynamic retrieval. Through comprehensive evaluation across diverse baselines and EgoMAS on MA-EgoQA, we find that current approaches are unable to effectively handle multiple egocentric streams, highlighting the need for future advances in system-level understanding across the agents. The code and benchmark are available at https://ma-egoqa.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25268v2">SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation</a></div>
    <div class="paper-meta">
      📅 2026-03-10
    </div>
    <details class="paper-abstract">
      Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08260v1">Seed2Scale: A Self-Evolving Data Engine for Embodied AI via Small to Large Model Synergy and Multimodal Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Existing data generation methods suffer from exploration limits, embodiment gaps, and low signal-to-noise ratios, leading to performance degradation during self-iteration. To address these challenges, we propose Seed2Scale, a self-evolving data engine that overcomes the data bottleneck through a heterogeneous synergy of "small-model collection, large-model evaluation, and target-model learning". Starting with as few as four seed demonstrations, the engine employs the lightweight Vision-Language-Action model, SuperTiny, as a dedicated collector, leveraging its strong inductive bias for robust exploration in parallel environments. Concurrently, a pre-trained Vision-Language Model is integrated as a Verifer to autonomously perform success/failure judgment and quality scoring for the massive generated trajectories. Seed2Scale effectively mitigates model collapse, ensuring the stability of the self-evolution process. Experimental results demonstrate that Seed2Scale exhibits signifcant scaling potential: as iterations progress, the success rate of the target model shows a robust upward trend, achieving a performance improvement of 131.2%. Furthermore, Seed2Scale signifcantly outperforms existing data augmentation methods, providing a scalable and cost-effective pathway for the large-scale development of Generalist Embodied AI. Project page: https://terminators2025.github.io/Seed2Scale.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08131v1">UniGround: Universal 3D Visual Grounding via Training-Free Scene Parsing</a></div>
    <div class="paper-meta">
      📅 2026-03-09
      | 💬 14 pages,6 figures,3 tables
    </div>
    <details class="paper-abstract">
      Understanding and localizing objects in complex 3D environments from natural language descriptions, known as 3D Visual Grounding (3DVG), is a foundational challenge in embodied AI, with broad implications for robotics, augmented reality, and human-machine interaction. Large-scale pre-trained foundation models have driven significant progress on this front, enabling open-vocabulary 3DVG that allows systems to locate arbitrary objects in a given scene. However, their reliance on pre-trained models constrains 3D perception and reasoning within the inherited knowledge boundaries, resulting in limited generalization to unseen spatial relationships and poor robustness to out-of-distribution scenes. In this paper, we replace this constrained perception with training-free visual and geometric reasoning, thereby unlocking open-world 3DVG that enables the localization of any object in any scene beyond the training data. Specifically, the proposed UniGround operates in two stages: a Global Candidate Filtering stage that constructs scene candidates through training-free 3D topology and multi-view semantic encoding, and a Local Precision Grounding stage that leverages multi-scale visual prompting and structured reasoning to precisely identify the target object. Experiments on ScanRefer and EmbodiedScan show that UniGround achieves 46.1\%/34.1\% Acc@0.25/0.5 on ScanRefer and 28.7\% Acc@0.25 on EmbodiedScan, establishing a new state-of-the-art among zero-shot methods on EmbodiedScan without any 3D supervision. We further evaluate UniGround in real-world environments under uncontrolled reconstruction conditions and substantial domain shift, showing training-free reasoning generalizes robustly beyond curated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08096v1">TrianguLang: Geometry-Aware Semantic Consensus for Pose-Free 3D Localization</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Localizing objects and parts from natural language in 3D space is essential for robotics, AR, and embodied AI, yet existing methods face a trade-off between the accuracy and geometric consistency of per-scene optimization and the efficiency of feed-forward inference. We present TrianguLang, a feed-forward framework for 3D localization that requires no camera calibration at inference. Unlike prior methods that treat views independently, we introduce Geometry-Aware Semantic Attention (GASA), which utilizes predicted geometry to gate cross-view feature correspondence, suppressing semantically plausible but geometrically inconsistent matches without requiring ground-truth poses. Validated on five benchmarks including ScanNet++ and uCO3D, TrianguLang achieves state-of-the-art feed-forward text-guided segmentation and localization, reducing user effort from $O(N)$ clicks to a single text query. The model processes each frame at 1008x1008 resolution in $\sim$57ms ($\sim$18 FPS) without optimization, enabling practical deployment for interactive robotics and AR applications. Code and checkpoints are available at https://cwru-aism.github.io/triangulang/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08021v1">AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-03-09
    </div>
    <details class="paper-abstract">
      Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24099v3">Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow</a></div>
    <div class="paper-meta">
      📅 2026-03-07
      | 💬 Under review at ICLR 2026
    </div>
    <details class="paper-abstract">
      Generating realistic, context-aware two-person motion conditioned on diverse modalities remains a fundamental challenge for graphics, animation and embodied AI systems. Real-world applications such as VR/AR companions, social robotics and game agents require models capable of producing coordinated interpersonal behaviour while flexibly switching between interactive and reactive generation. We introduce DualFlow, the first unified and efficient framework for multi-modal two-person motion generation. DualFlow conditions 3D motion generation on diverse inputs, including text, music, and prior motion sequences. Leveraging rectified flow, it achieves deterministic straight-line sampling paths between noise and data, reducing inference time and mitigating error accumulation common in diffusion-based models. To enhance semantic grounding, DualFlow employs a novel Retrieval-Augmented Generation (RAG) module for two-person motion that retrieves motion exemplars using music features and LLM-based text decompositions of spatial relations, body movements, and rhythmic patterns. We use a contrastive rectified flow objective to further sharpen alignment with conditioning signals and add synchronisation loss to improve inter-person temporal coordination. Extensive evaluations across interactive, reactive, and multi-modal benchmarks demonstrate that DualFlow consistently improves motion quality, responsiveness, and semantic fidelity. DualFlow achieves state-of-the-art performance in two-person multi-modal motion generation, producing coherent, expressive, and rhythmically synchronized motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.06280v1">SuperSuit: An Isomorphic Bimodal Interface for Scalable Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-03-06
    </div>
    <details class="paper-abstract">
      High-quality, long-horizon demonstrations are essential for embodied AI, yet acquiring such data for tightly coupled wheeled mobile manipulators remains a fundamental bottleneck. Unlike fixed-base systems, mobile manipulators require continuous coordination between $SE(2)$ locomotion and precise manipulation, exposing limitations in existing teleoperation and wearable interfaces. We present \textbf{SuperSuit}, a bimodal data acquisition framework that supports both robot-in-the-loop teleoperation and active demonstration under a shared kinematic interface. Both modalities produce structurally identical joint-space trajectories, enabling direct data mixing without modifying downstream policies. For locomotion, SuperSuit maps natural human stepping to continuous planar base velocities, eliminating discrete command switches. For manipulation, it employs a strictly isomorphic wearable arm in both modes, while policy training is formulated in a shift-invariant delta-joint representation to mitigate calibration offsets and structural compliance without inverse kinematics. Real-world experiments on long-horizon mobile manipulation tasks show 2.6$\times$ higher demonstration throughput in active mode compared to a teleoperation baseline, comparable policy performance when substituting teleoperation data with active demonstrations at fixed dataset size, and monotonic performance improvement as active data volume increases. These results indicate that consistent kinematic representations across collection modalities enable scalable data acquisition for long-horizon mobile manipulation.
    </details>
</div>
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
