# embodied ai - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24110v1">Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Despite advances in embodied AI, agent reasoning systems still struggle to capture the fundamental conceptual structures that humans naturally use to understand and interact with their environment. To address this, we propose a novel framework that bridges embodied cognition theory and agent systems by leveraging a formal characterization of image schemas, which are defined as recurring patterns of sensorimotor experience that structure human cognition. By customizing LLMs to translate natural language descriptions into formal representations based on these sensorimotor patterns, we will be able to create a neurosymbolic system that grounds the agent's understanding in fundamental conceptual structures. We argue that such an approach enhances both efficiency and interpretability while enabling more intuitive human-agent interactions through shared embodied understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23765v1">STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      The use of Multimodal Large Language Models (MLLMs) as an end-to-end solution for Embodied AI and Autonomous Driving has become a prevailing trend. While MLLMs have been extensively studied for visual semantic understanding tasks, their ability to perform precise and quantitative spatial-temporal understanding in real-world applications remains largely unexamined, leading to uncertain prospects. To evaluate models' Spatial-Temporal Intelligence, we introduce STI-Bench, a benchmark designed to evaluate MLLMs' spatial-temporal understanding through challenging tasks such as estimating and predicting the appearance, pose, displacement, and motion of objects. Our benchmark encompasses a wide range of robot and vehicle operations across desktop, indoor, and outdoor scenarios. The extensive experiments reveals that the state-of-the-art MLLMs still struggle in real-world spatial-temporal understanding, especially in tasks requiring precise distance estimation and motion analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22869v1">SIGHT: Single-Image Conditioned Generation of Hand Trajectories for Hand-Object Interaction</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      We introduce a novel task of generating realistic and diverse 3D hand trajectories given a single image of an object, which could be involved in a hand-object interaction scene or pictured by itself. When humans grasp an object, appropriate trajectories naturally form in our minds to use it for specific tasks. Hand-object interaction trajectory priors can greatly benefit applications in robotics, embodied AI, augmented reality and related fields. However, synthesizing realistic and appropriate hand trajectories given a single object or hand-object interaction image is a highly ambiguous task, requiring to correctly identify the object of interest and possibly even the correct interaction among many possible alternatives. To tackle this challenging problem, we propose the SIGHT-Fusion system, consisting of a curated pipeline for extracting visual features of hand-object interaction details from egocentric videos involving object manipulation, and a diffusion-based conditional motion generation model processing the extracted features. We train our method given video data with corresponding hand trajectory annotations, without supervision in the form of action labels. For the evaluation, we establish benchmarks utilizing the first-person FPHAB and HOI4D datasets, testing our method against various baselines and using multiple metrics. We also introduce task simulators for executing the generated hand trajectories and reporting task success rates as an additional metric. Experiments show that our method generates more appropriate and realistic hand trajectories than baselines and presents promising generalization capability on unseen objects. The accuracy of the generated hand trajectories is confirmed in a physics simulation setting, showcasing the authenticity of the created sequences and their applicability in downstream uses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21232v1">Knowledge Graphs as World Models for Semantic Material-Aware Obstacle Handling in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      The inability of autonomous vehicles (AVs) to infer the material properties of obstacles limits their decision-making capacity. While AVs rely on sensor systems such as cameras, LiDAR, and radar to detect obstacles, this study suggests combining sensors with a knowledge graph (KG)-based world model to improve AVs' comprehension of physical material qualities. Beyond sensor data, AVs can infer qualities such as malleability, density, and elasticity using a semantic KG that depicts the relationships between obstacles and their attributes. Using the CARLA autonomous driving simulator, we evaluated AV performance with and without KG integration. The findings demonstrate that the KG-based method improves obstacle management, which allows AVs to use material qualities to make better decisions about when to change lanes or apply emergency braking. For example, the KG-integrated AV changed lanes for hard impediments like traffic cones and successfully avoided collisions with flexible items such as plastic bags by passing over them. Compared to the control system, the KG framework demonstrated improved responsiveness to obstacles by resolving conflicting sensor data, causing emergency stops for 13.3% more cases. In addition, our method exhibits a 6.6% higher success rate in lane-changing maneuvers in experimental scenarios, particularly for larger, high-impact obstacles. While we focus particularly on autonomous driving, our work demonstrates the potential of KG-based world models to improve decision-making in embodied AI systems and scale to other domains, including robotics, healthcare, and environmental simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21056v1">Online Reasoning Video Segmentation with Just-in-Time Digital Twins</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Reasoning segmentation (RS) aims to identify and segment objects of interest based on implicit text queries. As such, RS is a catalyst for embodied AI agents, enabling them to interpret high-level commands without requiring explicit step-by-step guidance. However, current RS approaches rely heavily on the visual perception capabilities of multimodal large language models (LLMs), leading to several major limitations. First, they struggle with queries that require multiple steps of reasoning or those that involve complex spatial/temporal relationships. Second, they necessitate LLM fine-tuning, which may require frequent updates to maintain compatibility with contemporary LLMs and may increase risks of catastrophic forgetting during fine-tuning. Finally, being primarily designed for static images or offline video processing, they scale poorly to online video data. To address these limitations, we propose an agent framework that disentangles perception and reasoning for online video RS without LLM fine-tuning. Our innovation is the introduction of a just-in-time digital twin concept, where -- given an implicit query -- a LLM plans the construction of a low-level scene representation from high-level video using specialist vision models. We refer to this approach to creating a digital twin as "just-in-time" because the LLM planner will anticipate the need for specific information and only request this limited subset instead of always evaluating every specialist model. The LLM then performs reasoning on this digital twin representation to identify target objects. To evaluate our approach, we introduce a new comprehensive video reasoning segmentation benchmark comprising 200 videos with 895 implicit text queries. The benchmark spans three reasoning categories (semantic, spatial, and temporal) with three different reasoning chain complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21860v1">ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Accepted to CVPR 2025
    </div>
    <details class="paper-abstract">
      Human hands play a central role in interacting, motivating increasing research in dexterous robotic manipulation. Data-driven embodied AI algorithms demand precise, large-scale, human-like manipulation sequences, which are challenging to obtain with conventional reinforcement learning or real-world teleoperation. To address this, we introduce ManipTrans, a novel two-stage method for efficiently transferring human bimanual skills to dexterous robotic hands in simulation. ManipTrans first pre-trains a generalist trajectory imitator to mimic hand motion, then fine-tunes a specific residual module under interaction constraints, enabling efficient learning and accurate execution of complex bimanual tasks. Experiments show that ManipTrans surpasses state-of-the-art methods in success rate, fidelity, and efficiency. Leveraging ManipTrans, we transfer multiple hand-object datasets to robotic hands, creating DexManipNet, a large-scale dataset featuring previously unexplored tasks like pen capping and bottle unscrewing. DexManipNet comprises 3.3K episodes of robotic manipulation and is easily extensible, facilitating further policy training for dexterous hands and enabling real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20220v1">DINeMo: Learning Neural Mesh Models with no 3D Annotations</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Category-level 3D/6D pose estimation is a crucial step towards comprehensive 3D scene understanding, which would enable a broad range of applications in robotics and embodied AI. Recent works explored neural mesh models that approach a range of 2D and 3D tasks from an analysis-by-synthesis perspective. Despite the largely enhanced robustness to partial occlusion and domain shifts, these methods depended heavily on 3D annotations for part-contrastive learning, which confines them to a narrow set of categories and hinders efficient scaling. In this work, we present DINeMo, a novel neural mesh model that is trained with no 3D annotations by leveraging pseudo-correspondence obtained from large visual foundation models. We adopt a bidirectional pseudo-correspondence generation method, which produce pseudo correspondence utilize both local appearance features and global context information. Experimental results on car datasets demonstrate that our DINeMo outperforms previous zero- and few-shot 3D pose estimation by a wide margin, narrowing the gap with fully-supervised methods by 67.3%. Our DINeMo also scales effectively and efficiently when incorporating more unlabeled images during training, which demonstrate the advantages over supervised learning methods that rely on 3D annotations. Our project page is available at https://analysis-by-synthesis.github.io/DINeMo/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19901v1">TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Synthesizing diverse and physically plausible Human-Scene Interactions (HSI) is pivotal for both computer animation and embodied AI. Despite encouraging progress, current methods mainly focus on developing separate controllers, each specialized for a specific interaction task. This significantly hinders the ability to tackle a wide variety of challenging HSI tasks that require the integration of multiple skills, e.g., sitting down while carrying an object. To address this issue, we present TokenHSI, a single, unified transformer-based policy capable of multi-skill unification and flexible adaptation. The key insight is to model the humanoid proprioception as a separate shared token and combine it with distinct task tokens via a masking mechanism. Such a unified policy enables effective knowledge sharing across skills, thereby facilitating the multi-task training. Moreover, our policy architecture supports variable length inputs, enabling flexible adaptation of learned skills to new scenarios. By training additional task tokenizers, we can not only modify the geometries of interaction targets but also coordinate multiple skills to address complex tasks. The experiments demonstrate that our approach can significantly improve versatility, adaptability, and extensibility in various HSI tasks. Website: https://liangpan99.github.io/TokenHSI/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19516v1">DataPlatter: Boosting Robotic Manipulation Generalization with Minimal Costly Data</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The growing adoption of Vision-Language-Action (VLA) models in embodied AI intensifies the demand for diverse manipulation demonstrations. However, high costs associated with data collection often result in insufficient data coverage across all scenarios, which limits the performance of the models. It is observed that the spatial reasoning phase (SRP) in large workspace dominates the failure cases. Fortunately, this data can be collected with low cost, underscoring the potential of leveraging inexpensive data to improve model performance. In this paper, we introduce the DataPlatter method, a framework that decouples training trajectories into distinct task stages and leverages abundant easily collectible SRP data to enhance VLA model's generalization. Through analysis we demonstrate that sub-task-specific training with additional SRP data with proper proportion can act as a performance catalyst for robot manipulation, maximizing the utilization of costly physical interaction phase (PIP) data. Experiments show that through introducing large proportion of cost-effective SRP trajectories into a limited set of PIP data, we can achieve a maximum improvement of 41\% on success rate in zero-shot scenes, while with the ability to transfer manipulation skill to novel targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19941v1">Body Discovery of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      In the pursuit of realizing artificial general intelligence (AGI), the importance of embodied artificial intelligence (AI) becomes increasingly apparent. Following this trend, research integrating robots with AGI has become prominent. As various kinds of embodiments have been designed, adaptability to diverse embodiments will become important to AGI. We introduce a new challenge, termed "Body Discovery of Embodied AI", focusing on tasks of recognizing embodiments and summarizing neural signal functionality. The challenge encompasses the precise definition of an AI body and the intricate task of identifying embodiments in dynamic environments, where conventional approaches often prove inadequate. To address these challenges, we apply causal inference method and evaluate it by developing a simulator tailored for testing algorithms with virtual environments. Finally, we validate the efficacy of our algorithms through empirical testing, demonstrating their robust performance in various scenarios based on virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18016v1">Retrieval Augmented Generation and Understanding in Vision: A Survey and New Outlook</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 19 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has emerged as a pivotal technique in artificial intelligence (AI), particularly in enhancing the capabilities of large language models (LLMs) by enabling access to external, reliable, and up-to-date knowledge sources. In the context of AI-Generated Content (AIGC), RAG has proven invaluable by augmenting model outputs with supplementary, relevant information, thus improving their quality. Recently, the potential of RAG has extended beyond natural language processing, with emerging methods integrating retrieval-augmented strategies into the computer vision (CV) domain. These approaches aim to address the limitations of relying solely on internal model knowledge by incorporating authoritative external knowledge bases, thereby improving both the understanding and generation capabilities of vision models. This survey provides a comprehensive review of the current state of retrieval-augmented techniques in CV, focusing on two main areas: (I) visual understanding and (II) visual generation. In the realm of visual understanding, we systematically review tasks ranging from basic image recognition to complex applications such as medical report generation and multimodal question answering. For visual content generation, we examine the application of RAG in tasks related to image, video, and 3D generation. Furthermore, we explore recent advancements in RAG for embodied AI, with a particular focus on applications in planning, task execution, multimodal perception, interaction, and specialized domains. Given that the integration of retrieval-augmented techniques in CV is still in its early stages, we also highlight the key limitations of current approaches and propose future research directions to drive the development of this promising area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07376v2">NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 Accepted by TPAMI 2025
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the instruction, 2) selecting the candidate observation that best aligns with the imagination, and 3) determining the action based on the reasoning from the prior steps. Through constructing formalized labels for training, the LLM can learn to generate desired and reasonable chain-of-thought outputs for improving the action decision. Experimental results across various training settings and popular VLN benchmarks (e.g., Room-to-Room (R2R), Room-across-Room (RxR), Room-for-Room (R4R)) show the significant superiority of NavCoT over the direct action prediction variants. Through simple parameter-efficient finetuning, our NavCoT outperforms a recent GPT4-based approach with ~7% relative improvement on the R2R dataset. We believe that NavCoT will help unlock more task-adaptive and scalable LLM-based embodied agents, which are helpful for developing real-world robotics applications. Code is available at https://github.com/expectorlin/NavCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13927v2">Multimodal 3D Reasoning Segmentation with Complex Scenes</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      The recent development in multimodal learning has greatly advanced the research in 3D scene understanding in various real-world tasks such as embodied AI. However, most existing work shares two typical constraints: 1) they are short of reasoning ability for interaction and interpretation of human intension and 2) they focus on scenarios with single-category objects only which leads to over-simplified textual descriptions due to the negligence of multi-object scenarios and spatial relations among objects. We bridge the research gaps by proposing a 3D reasoning segmentation task for multiple objects in scenes. The task allows producing 3D segmentation masks and detailed textual explanations as enriched by 3D spatial relations among objects. To this end, we create ReasonSeg3D, a large-scale and high-quality benchmark that integrates 3D segmentation masks and 3D spatial relations with generated question-answer pairs. In addition, we design MORE3D, a novel 3D reasoning network that works with queries of multiple objects and tailored 3D scene understanding designs. MORE3D learns detailed explanations on 3D relations and employs them to capture spatial information of objects and reason textual outputs. Extensive experiments show that MORE3D excels in reasoning and segmenting complex multi-object 3D scenes, and the created ReasonSeg3D offers a valuable platform for future exploration of 3D reasoning segmentation. The dataset and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18600v2">ZeroHSI: Zero-Shot 4D Human-Scene Interaction by Video Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Project website: https://awfuact.github.io/zerohsi/ The first two authors contribute equally
    </div>
    <details class="paper-abstract">
      Human-scene interaction (HSI) generation is crucial for applications in embodied AI, virtual reality, and robotics. Yet, existing methods cannot synthesize interactions in unseen environments such as in-the-wild scenes or reconstructed scenes, as they rely on paired 3D scenes and captured human motion data for training, which are unavailable for unseen environments. We present ZeroHSI, a novel approach that enables zero-shot 4D human-scene interaction synthesis, eliminating the need for training on any MoCap data. Our key insight is to distill human-scene interactions from state-of-the-art video generation models, which have been trained on vast amounts of natural human movements and interactions, and use differentiable rendering to reconstruct human-scene interactions. ZeroHSI can synthesize realistic human motions in both static scenes and environments with dynamic objects, without requiring any ground-truth motion data. We evaluate ZeroHSI on a curated dataset of different types of various indoor and outdoor scenes with different interaction prompts, demonstrating its ability to generate diverse and contextually appropriate human-scene interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09258v2">Efficient Training of Generalizable Visuomotor Policies via Control-Aware Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Improving generalization is one key challenge in embodied AI, where obtaining large-scale datasets across diverse scenarios is costly. Traditional weak augmentations, such as cropping and flipping, are insufficient for improving a model's performance in new environments. Existing data augmentation methods often disrupt task-relevant information in images, potentially degrading performance. To overcome these challenges, we introduce EAGLE, an efficient training framework for generalizable visuomotor policies that improves upon existing methods by (1) enhancing generalization by applying augmentation only to control-related regions identified through a self-supervised control-aware mask and (2) improving training stability and efficiency by distilling knowledge from an expert to a visuomotor student policy, which is then deployed to unseen environments without further fine-tuning. Comprehensive experiments on three domains, including the DMControl Generalization Benchmark, the enhanced Robot Manipulation Distraction Benchmark, and a long-sequential drawer-opening task, validate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14908v2">KARMA: Augmenting Embodied AI Agents with Long-and-short Term Memory Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Embodied AI agents responsible for executing interconnected, long-sequence household tasks often face difficulties with in-context memory, leading to inefficiencies and errors in task execution. To address this issue, we introduce KARMA, an innovative memory system that integrates long-term and short-term memory modules, enhancing large language models (LLMs) for planning in embodied agents through memory-augmented prompting. KARMA distinguishes between long-term and short-term memory, with long-term memory capturing comprehensive 3D scene graphs as representations of the environment, while short-term memory dynamically records changes in objects' positions and states. This dual-memory structure allows agents to retrieve relevant past scene experiences, thereby improving the accuracy and efficiency of task planning. Short-term memory employs strategies for effective and adaptive memory replacement, ensuring the retention of critical information while discarding less pertinent data. Compared to state-of-the-art embodied agents enhanced with memory, our memory-augmented embodied AI agent improves success rates by 1.3x and 2.3x in Composite Tasks and Complex Tasks within the AI2-THOR simulator, respectively, and enhances task execution efficiency by 3.4x and 62.7x. Furthermore, we demonstrate that KARMA's plug-and-play capability allows for seamless deployment on real-world robotic systems, such as mobile manipulation platforms.Through this plug-and-play memory system, KARMA significantly enhances the ability of embodied agents to generate coherent and contextually appropriate plans, making the execution of complex household tasks more efficient. The experimental videos from the work can be found at https://youtu.be/4BT7fnw9ehs. Our code is available at https://github.com/WZX0Swarm0Robotics/KARMA/tree/master.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15764v1">Towards Agentic AI Networking in 6G: A Generative Foundation Model-as-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Currently under revision at IEEE Communications Magazine
    </div>
    <details class="paper-abstract">
      The promising potential of AI and network convergence in improving networking performance and enabling new service capabilities has recently attracted significant interest. Existing network AI solutions, while powerful, are mainly built based on the close-loop and passive learning framework, resulting in major limitations in autonomous solution finding and dynamic environmental adaptation. Agentic AI has recently been introduced as a promising solution to address the above limitations and pave the way for true generally intelligent and beneficial AI systems. The key idea is to create a networking ecosystem to support a diverse range of autonomous and embodied AI agents in fulfilling their goals. In this paper, we focus on the novel challenges and requirements of agentic AI networking. We propose AgentNet, a novel framework for supporting interaction, collaborative learning, and knowledge transfer among AI agents. We introduce a general architectural framework of AgentNet and then propose a generative foundation model (GFM)-based implementation in which multiple GFM-as-agents have been created as an interactive knowledge-base to bootstrap the development of embodied AI agents according to different task requirements and environmental features. We consider two application scenarios, digital-twin-based industrial automation and metaverse-based infotainment system, to describe how to apply AgentNet for supporting efficient task-driven collaboration and interaction among AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05132v3">3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 CVPR 2025. Project website: https://3d-grand.github.io
    </div>
    <details class="paper-abstract">
      The integration of language and 3D perception is crucial for embodied agents and robots that comprehend and interact with the physical world. While large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, their adaptation to 3D environments (3D-LLMs) remains in its early stages. A primary challenge is a lack of large-scale datasets with dense grounding between language and 3D scenes. We introduce 3D-GRAND, a pioneering large-scale dataset comprising 40,087 household scenes paired with 6.2 million densely-grounded scene-language instructions. Our results show that instruction tuning with 3D-GRAND significantly enhances grounding capabilities and reduces hallucinations in 3D-LLMs. As part of our contributions, we propose a comprehensive benchmark 3D-POPE to systematically evaluate hallucination in 3D-LLMs, enabling fair comparisons of models. Our experiments highlight a scaling effect between dataset size and 3D-LLM performance, emphasizing the importance of large-scale 3D-text datasets for embodied AI research. Our results demonstrate early signals for effective sim-to-real transfer, indicating that models trained on large synthetic data can perform well on real-world 3D scans. Through 3D-GRAND and 3D-POPE, we aim to equip the embodied AI community with resources and insights to lead to more reliable and better-grounded 3D-LLMs. Project website: https://3d-grand.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13642v7">SpatialBot: Precise Spatial Understanding with Vision Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Vision Language Models (VLMs) have achieved impressive performance in 2D image understanding, however they are still struggling with spatial understanding which is the foundation of Embodied AI. In this paper, we propose SpatialBot for better spatial understanding by feeding both RGB and depth images. Additionally, we have constructed the SpatialQA dataset, which involves multi-level depth-related questions to train VLMs for depth understanding. Finally, we present SpatialBench to comprehensively evaluate VLMs' capabilities in spatial understanding at different levels. Extensive experiments on our spatial-understanding benchmark, general VLM benchmarks and Embodied AI tasks, demonstrate the remarkable improvements of SpatialBot trained on SpatialQA. The model, code and data are available at https://github.com/BAAI-DCAI/SpatialBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00379v2">Latent Action Learning Requires Supervision in the Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Preprint. In review. Edit: Accepted by ICLR 2025 Workshop on World Models: Understanding, Modelling and Scaling
    </div>
    <details class="paper-abstract">
      Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13882v1">MoK-RAG: Mixture of Knowledge Paths Enhanced Retrieval-Augmented Generation for Embodied AI Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      While human cognition inherently retrieves information from diverse and specialized knowledge sources during decision-making processes, current Retrieval-Augmented Generation (RAG) systems typically operate through single-source knowledge retrieval, leading to a cognitive-algorithmic discrepancy. To bridge this gap, we introduce MoK-RAG, a novel multi-source RAG framework that implements a mixture of knowledge paths enhanced retrieval mechanism through functional partitioning of a large language model (LLM) corpus into distinct sections, enabling retrieval from multiple specialized knowledge paths. Applied to the generation of 3D simulated environments, our proposed MoK-RAG3D enhances this paradigm by partitioning 3D assets into distinct sections and organizing them based on a hierarchical knowledge tree structure. Different from previous methods that only use manual evaluation, we pioneered the introduction of automated evaluation methods for 3D scenes. Both automatic and human evaluations in our experiments demonstrate that MoK-RAG3D can assist Embodied AI agents in generating diverse scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13424v1">Infinite Mobility: Scalable High-Fidelity Synthesis of Articulated Objects via Procedural Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Project page: https://infinite-mobility.github.io 10 pages,12 figures
    </div>
    <details class="paper-abstract">
      Large-scale articulated objects with high quality are desperately needed for multiple tasks related to embodied AI. Most existing methods for creating articulated objects are either data-driven or simulation based, which are limited by the scale and quality of the training data or the fidelity and heavy labour of the simulation. In this paper, we propose Infinite Mobility, a novel method for synthesizing high-fidelity articulated objects through procedural generation. User study and quantitative evaluation demonstrate that our method can produce results that excel current state-of-the-art methods and are comparable to human-annotated datasets in both physics property and mesh quality. Furthermore, we show that our synthetic data can be used as training data for generative models, enabling next-step scaling up. Code is available at https://github.com/Intern-Nexus/Infinite-Mobility
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12974v1">Exploring 3D Activity Reasoning and Planning: From Implicit Human Intentions to Route-Aware Planning</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      3D activity reasoning and planning has attracted increasing attention in human-robot interaction and embodied AI thanks to the recent advance in multimodal learning. However, most existing works share two constraints: 1) heavy reliance on explicit instructions with little reasoning on implicit user intention; 2) negligence of inter-step route planning on robot moves. To bridge the gaps, we propose 3D activity reasoning and planning, a novel 3D task that reasons the intended activities from implicit instructions and decomposes them into steps with inter-step routes and planning under the guidance of fine-grained 3D object shapes and locations from scene segmentation. We tackle the new 3D task from two perspectives. First, we construct ReasonPlan3D, a large-scale benchmark that covers diverse 3D scenes with rich implicit instructions and detailed annotations for multi-step task planning, inter-step route planning, and fine-grained segmentation. Second, we design a novel framework that introduces progressive plan generation with contextual consistency across multiple steps, as well as a scene graph that is updated dynamically for capturing critical objects and their spatial relations. Extensive experiments demonstrate the effectiveness of our benchmark and framework in reasoning activities from implicit human instructions, producing accurate stepwise task plans, and seamlessly integrating route planning for multi-step moves. The dataset and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12955v1">HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      We propose a new task to benchmark human-in-scene understanding for embodied agents: Human-In-Scene Question Answering (HIS-QA). Given a human motion within a 3D scene, HIS-QA requires the agent to comprehend human states and behaviors, reason about its surrounding environment, and answer human-related questions within the scene. To support this new task, we present HIS-Bench, a multimodal benchmark that systematically evaluates HIS understanding across a broad spectrum, from basic perception to commonsense reasoning and planning. Our evaluation of various vision-language models on HIS-Bench reveals significant limitations in their ability to handle HIS-QA tasks. To this end, we propose HIS-GPT, the first foundation model for HIS understanding. HIS-GPT integrates 3D scene context and human motion dynamics into large language models while incorporating specialized mechanisms to capture human-scene interactions. Extensive experiments demonstrate that HIS-GPT sets a new state-of-the-art on HIS-QA tasks. We hope this work inspires future research on human behavior analysis in 3D scenes, advancing embodied AI and world models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08306v3">Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18896v2">S2O: Static to Openable Enhancement for Articulated 3D Objects</a></div>
    <div class="paper-meta">
      📅 2025-03-15
    </div>
    <details class="paper-abstract">
      Despite much progress in large 3D datasets there are currently few interactive 3D object datasets, and their scale is limited due to the manual effort required in their construction. We introduce the static to openable (S2O) task which creates interactive articulated 3D objects from static counterparts through openable part detection, motion prediction, and interior geometry completion. We formulate a unified framework to tackle this task, and curate a challenging dataset of openable 3D objects that serves as a test bed for systematic evaluation. Our experiments benchmark methods from prior work, extended and improved methods, and simple yet effective heuristics for the S2O task. We find that turning static 3D objects into interactively openable counterparts is possible but that all methods struggle to generalize to realistic settings of the task, and we highlight promising future work directions. Our work enables efficient creation of interactive 3D objects for robotic manipulation and embodied AI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11081v1">MoMa-Kitchen: A 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-03-14
    </div>
    <details class="paper-abstract">
      In mobile manipulation, navigation and manipulation are often treated as separate problems, resulting in a significant gap between merely approaching an object and engaging with it effectively. Many navigation approaches primarily define success by proximity to the target, often overlooking the necessity for optimal positioning that facilitates subsequent manipulation. To address this, we introduce MoMa-Kitchen, a benchmark dataset comprising over 100k samples that provide training data for models to learn optimal final navigation positions for seamless transition to manipulation. Our dataset includes affordance-grounded floor labels collected from diverse kitchen environments, in which robotic mobile manipulators of different models attempt to grasp target objects amidst clutter. Using a fully automated pipeline, we simulate diverse real-world scenarios and generate affordance labels for optimal manipulation positions. Visual data are collected from RGB-D inputs captured by a first-person view camera mounted on the robotic arm, ensuring consistency in viewpoint during data collection. We also develop a lightweight baseline model, NavAff, for navigation affordance grounding that demonstrates promising performance on the MoMa-Kitchen benchmark. Our approach enables models to learn affordance-based final positioning that accommodates different arm types and platform heights, thereby paving the way for more robust and generalizable integration of navigation and manipulation in embodied AI. Project page: \href{https://momakitchen.github.io/}{https://momakitchen.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14993v2">Making Every Frame Matter: Continuous Activity Recognition in Streaming Video via Adaptive Video Context Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Video activity recognition has become increasingly important in robots and embodied AI. Recognizing continuous video activities poses considerable challenges due to the fast expansion of streaming video, which contains multi-scale and untrimmed activities. We introduce a novel system, CARS, to overcome these issues through adaptive video context modeling. Adaptive video context modeling refers to selectively maintaining activity-related features in temporal and spatial dimensions. CARS has two key designs. The first is an activity spatial feature extraction by eliminating irrelevant visual features while maintaining recognition accuracy. The second is an activity-aware state update introducing dynamic adaptability to better preserve the video context for multi-scale activity recognition. Our CARS runs at speeds $>$30 FPS on typical edge devices and outperforms all baselines by 1.2\% to 79.7\% in accuracy. Moreover, we explore applying CARS to a large video model as a video encoder. Experimental results show that our CARS can result in a 0.46-point enhancement (on a 5-point scale) on the in-distribution video activity dataset, and an improvement ranging from 1.19\% to 4\% on zero-shot video activity datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10307v1">6D Object Pose Tracking in Internet Videos for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 Accepted to ICLR 2025. Project page available at https://ponimatkin.github.io/wildpose/
    </div>
    <details class="paper-abstract">
      We seek to extract a temporally consistent 6D pose trajectory of a manipulated object from an Internet instructional video. This is a challenging set-up for current 6D pose estimation methods due to uncontrolled capturing conditions, subtle but dynamic object motions, and the fact that the exact mesh of the manipulated object is not known. To address these challenges, we present the following contributions. First, we develop a new method that estimates the 6D pose of any object in the input image without prior knowledge of the object itself. The method proceeds by (i) retrieving a CAD model similar to the depicted object from a large-scale model database, (ii) 6D aligning the retrieved CAD model with the input image, and (iii) grounding the absolute scale of the object with respect to the scene. Second, we extract smooth 6D object trajectories from Internet videos by carefully tracking the detected objects across video frames. The extracted object trajectories are then retargeted via trajectory optimization into the configuration space of a robotic manipulator. Third, we thoroughly evaluate and ablate our 6D pose estimation method on YCB-V and HOPE-Video datasets as well as a new dataset of instructional videos manually annotated with approximate 6D object trajectories. We demonstrate significant improvements over existing state-of-the-art RGB 6D pose estimation methods. Finally, we show that the 6D object motion estimated from Internet videos can be transferred to a 7-axis robotic manipulator both in a virtual simulator as well as in a real world set-up. We also successfully apply our method to egocentric videos taken from the EPIC-KITCHENS dataset, demonstrating potential for Embodied AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15658v2">Long-horizon Embodied Planning with Implicit Logical Inference and Hallucination Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Long-horizon embodied planning underpins embodied AI. To accomplish long-horizon tasks, one of the most feasible ways is to decompose abstract instructions into a sequence of actionable steps. Foundation models still face logical errors and hallucinations in long-horizon planning, unless provided with highly relevant examples to the tasks. However, providing highly relevant examples for any random task is unpractical. Therefore, we present ReLEP, a novel framework for Real-time Long-horizon Embodied Planning. ReLEP can complete a wide range of long-horizon tasks without in-context examples by learning implicit logical inference through fine-tuning. The fine-tuned large vision-language model formulates plans as sequences of skill functions. These functions are selected from a carefully designed skill library. ReLEP is also equipped with a Memory module for plan and status recall, and a Robot Configuration module for versatility across robot types. In addition, we propose a data generation pipeline to tackle dataset scarcity. When constructing the dataset, we considered the implicit logical relationships, enabling the model to learn implicit logical relationships and dispel hallucinations. Through comprehensive evaluations across various long-horizon tasks, ReLEP demonstrates high success rates and compliance to execution even on unseen tasks and outperforms state-of-the-art baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10070v1">AhaRobot: A Low-Cost Open-Source Bimanual Mobile Manipulator for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 The first two authors contributed equally. Website: https://aha-robot.github.io
    </div>
    <details class="paper-abstract">
      Navigation and manipulation in open-world environments remain unsolved challenges in the Embodied AI. The high cost of commercial mobile manipulation robots significantly limits research in real-world scenes. To address this issue, we propose AhaRobot, a low-cost and fully open-source dual-arm mobile manipulation robot system with a hardware cost of only $1,000 (excluding optional computational resources), which is less than 1/15 of the cost of popular mobile robots. The AhaRobot system consists of three components: (1) a novel low-cost hardware architecture primarily composed of off-the-shelf components, (2) an optimized control solution to enhance operational precision integrating dual-motor backlash control and static friction compensation, and (3) a simple remote teleoperation method RoboPilot. We use handles to control the dual arms and pedals for whole-body movement. The teleoperation process is low-burden and easy to operate, much like piloting. RoboPilot is designed for remote data collection in embodied scenarios. Experimental results demonstrate that RoboPilot significantly enhances data collection efficiency in complex manipulation tasks, achieving a 30% increase compared to methods using 3D mouse and leader-follower systems. It also excels at completing extremely long-horizon tasks in one go. Furthermore, AhaRobot can be used to learn end-to-end policies and autonomously perform complex manipulation tasks, such as pen insertion and cleaning up the floor. We aim to build an affordable yet powerful platform to promote the development of embodied tasks on real devices, advancing more robust and reliable embodied AI. All hardware and software systems are available at https://aha-robot.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08306v2">Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06843v4">HO-Cap: A Capture System and Dataset for 3D Reconstruction and Pose Tracking of Hand-Object Interaction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      We introduce a data capture system and a new dataset, HO-Cap, for 3D reconstruction and pose tracking of hands and objects in videos. The system leverages multiple RGBD cameras and a HoloLens headset for data collection, avoiding the use of expensive 3D scanners or mocap systems. We propose a semi-automatic method for annotating the shape and pose of hands and objects in the collected videos, significantly reducing the annotation time compared to manual labeling. With this system, we captured a video dataset of humans interacting with objects to perform various tasks, including simple pick-and-place actions, handovers between hands, and using objects according to their affordance, which can serve as human demonstrations for research in embodied AI and robot manipulation. Our data capture setup and annotation framework will be available for the community to use in reconstructing 3D shapes of objects and human hands and tracking their poses in videos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10439v2">CogNav: Cognitive Process Modeling for Object Goal Navigation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Object goal navigation (ObjectNav) is a fundamental task in embodied AI, requiring an agent to locate a target object in previously unseen environments. This task is particularly challenging because it requires both perceptual and cognitive processes, including object recognition and decision-making. While substantial advancements in perception have been driven by the rapid development of visual foundation models, progress on the cognitive aspect remains constrained, primarily limited to either implicit learning through simulator rollouts or explicit reliance on predefined heuristic rules. Inspired by neuroscientific findings demonstrating that humans maintain and dynamically update fine-grained cognitive states during object search tasks in novel environments, we propose CogNav, a framework designed to mimic this cognitive process using large language models. Specifically, we model the cognitive process using a finite state machine comprising fine-grained cognitive states, ranging from exploration to identification. Transitions between states are determined by a large language model based on a dynamically constructed heterogeneous cognitive map, which contains spatial and semantic information about the scene being explored. Extensive evaluations on the HM3D, MP3D, and RoboTHOR benchmarks demonstrate that our cognitive process modeling significantly improves the success rate of ObjectNav at least by relative 14% over the state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08306v1">Reasoning in visual navigation of end-to-end trained agents: a dynamical systems approach</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Progress in Embodied AI has made it possible for end-to-end-trained agents to navigate in photo-realistic environments with high-level reasoning and zero-shot or language-conditioned behavior, but benchmarks are still dominated by simulation. In this work, we focus on the fine-grained behavior of fast-moving real robots and present a large-scale experimental study involving \numepisodes{} navigation episodes in a real environment with a physical robot, where we analyze the type of reasoning emerging from end-to-end training. In particular, we study the presence of realistic dynamics which the agent learned for open-loop forecasting, and their interplay with sensing. We analyze the way the agent uses latent memory to hold elements of the scene structure and information gathered during exploration. We probe the planning capabilities of the agent, and find in its memory evidence for somewhat precise plans over a limited horizon. Furthermore, we show in a post-hoc analysis that the value function learned by the agent relates to long-term planning. Put together, our experiments paint a new picture on how using tools from computer vision and sequential decision making have led to new capabilities in robotics and control. An interactive tool is available at europe.naverlabs.com/research/publications/reasoning-in-visual-navigation-of-end-to-end-trained-agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08174v1">Investigating the Effectiveness of a Socratic Chain-of-Thoughts Reasoning Method for Task Planning in Robotics, A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated unprecedented capability in reasoning with natural language. Coupled with this development is the emergence of embodied AI in robotics. Despite showing promise for verbal and written reasoning tasks, it remains unknown whether LLMs are capable of navigating complex spatial tasks with physical actions in the real world. To this end, it is of interest to investigate applying LLMs to robotics in zero-shot learning scenarios, and in the absence of fine-tuning - a feat which could significantly improve human-robot interaction, alleviate compute cost, and eliminate low-level programming tasks associated with robot tasks. To explore this question, we apply GPT-4(Omni) with a simulated Tiago robot in Webots engine for an object search task. We evaluate the effectiveness of three reasoning strategies based on Chain-of-Thought (CoT) sub-task list generation with the Socratic method (SocraCoT) (in order of increasing rigor): (1) Non-CoT/Non-SocraCoT, (2) CoT only, and (3) SocraCoT. Performance was measured in terms of the proportion of tasks successfully completed and execution time (N = 20). Our preliminary results show that when combined with chain-of-thought reasoning, the Socratic method can be used for code generation for robotic tasks that require spatial awareness. In extension of this finding, we propose EVINCE-LoC; a modified EVINCE method that could further enhance performance in highly complex and or dynamic testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08158v2">EmBARDiment: an Embodied AI Agent for Productivity in XR</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      XR devices running chat-bots powered by Large Language Models (LLMs) have the to become always-on agents that enable much better productivity scenarios. Current screen based chat-bots do not take advantage of the the full-suite of natural inputs available in XR, including inward facing sensor data, instead they over-rely on explicit voice or text prompts, sometimes paired with multi-modal data dropped as part of the query. We propose a solution that leverages an attention framework that derives context implicitly from user actions, eye-gaze, and contextual memory within the XR environment. Our work minimizes the need for engineered explicit prompts, fostering grounded and intuitive interactions that glean user insights for the chat-bot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v4">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01481v2">SonicSim: A customizable simulation platform for speech processing in moving sound source scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Systematic evaluation of speech separation and enhancement models under moving sound source conditions requires extensive and diverse data. However, real-world datasets often lack sufficient data for training and evaluation, and synthetic datasets, while larger, lack acoustic realism. Consequently, neither effectively meets practical needs. To address this issue, we introduce SonicSim, a synthetic toolkit based on the embodied AI simulation platform Habitat-sim, designed to generate highly customizable data for moving sound sources. SonicSim supports multi-level adjustments, including scene-level, microphone-level, and source-level adjustments, enabling the creation of more diverse synthetic data. Leveraging SonicSim, we constructed a benchmark dataset called SonicSet, utilizing LibriSpeech, Freesound Dataset 50k (FSD50K), Free Music Archive (FMA), and 90 scenes from Matterport3D to evaluate speech separation and enhancement models. Additionally, to investigate the differences between synthetic and real-world data, we selected 5 hours of raw, non-reverberant data from the SonicSet validation set and recorded a real-world speech separation dataset, providing a reference for comparing SonicSet with other synthetic datasets. For speech enhancement, we utilized the real-world dataset RealMAN to validate the acoustic gap between SonicSet and existing synthetic datasets. The results indicate that models trained on SonicSet generalize better to real-world scenarios compared to other synthetic datasets. The code is publicly available at https://cslikai.cn/SonicSim/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04879v1">Modeling Dynamic Hand-Object Interactions with Applications to Human-Robot Handovers</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 PhD Thesis
    </div>
    <details class="paper-abstract">
      Humans frequently grasp, manipulate, and move objects. Interactive systems assist humans in these tasks, enabling applications in Embodied AI, human-robot interaction, and virtual reality. However, current methods in hand-object synthesis often neglect dynamics and focus on generating static grasps. The first part of this dissertation introduces dynamic grasp synthesis, where a hand grasps and moves an object to a target pose. We approach this task using physical simulation and reinforcement learning. We then extend this to bimanual manipulation and articulated objects, requiring fine-grained coordination between hands. In the second part of this dissertation, we study human-to-robot handovers. We integrate captured human motion into simulation and introduce a student-teacher framework that adapts to human behavior and transfers from sim to real. To overcome data scarcity, we generate synthetic interactions, increasing training diversity by 100x. Our user study finds no difference between policies trained on synthetic vs. real motions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v3">OpenFly: A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents through an environment by leveraging both language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising a versatile toolchain and large-scale benchmark for aerial VLN. Firstly, we develop a highly automated toolchain for data collection, enabling automatic point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Secondly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. The corresponding visual data are generated using various rendering engines and advanced techniques, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). All data exhibit high visual quality. Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of the dataset. Thirdly, we propose OpenFly-Agent, a keyframe-aware VLN model, which takes language instructions, current observations, and historical keyframes as input, and outputs flight actions directly. Extensive analyses and experiments are conducted, showcasing the superiority of our OpenFly platform and OpenFly-Agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14093v4">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Project page: https://github.com/yueen-ma/Awesome-VLA
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI. We have created a project associated with this survey, which is available at https://github.com/yueen-ma/Awesome-VLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02247v1">WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: https://b0b8k1ng.github.io/WMNav/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14093v3">A Survey on Vision-Language-Action Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 16 pages, a survey of vision-language-action models
    </div>
    <details class="paper-abstract">
      Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07468v3">From Screens to Scenes: A Survey of Embodied AI in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 56 pages, 11 figures, manuscript accepted by Information Fusion
    </div>
    <details class="paper-abstract">
      Healthcare systems worldwide face persistent challenges in efficiency, accessibility, and personalization. Powered by modern AI technologies such as multimodal large language models and world models, Embodied AI (EmAI) represents a transformative frontier, offering enhanced autonomy and the ability to interact with the physical world to address these challenges. As an interdisciplinary and rapidly evolving research domain, "EmAI in healthcare" spans diverse fields such as algorithms, robotics, and biomedicine. This complexity underscores the importance of timely reviews and analyses to track advancements, address challenges, and foster cross-disciplinary collaboration. In this paper, we provide a comprehensive overview of the "brain" of EmAI for healthcare, wherein we introduce foundational AI algorithms for perception, actuation, planning, and memory, and focus on presenting the healthcare applications spanning clinical interventions, daily care & companionship, infrastructure support, and biomedical research. Despite its promise, the development of EmAI for healthcare is hindered by critical challenges such as safety concerns, gaps between simulation platforms and real-world applications, the absence of standardized benchmarks, and uneven progress across interdisciplinary domains. We discuss the technical barriers and explore ethical considerations, offering a forward-looking perspective on the future of EmAI in healthcare. A hierarchical framework of intelligent levels for EmAI systems is also introduced to guide further development. By providing systematic insights, this work aims to inspire innovation and practical applications, paving the way for a new era of intelligent, patient-centered healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00778v1">AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Inferring the affordance of an object and grasping it in a task-oriented manner is crucial for robots to successfully complete manipulation tasks. Affordance indicates where and how to grasp an object by taking its functionality into account, serving as the foundation for effective task-oriented grasping. However, current task-oriented methods often depend on extensive training data that is confined to specific tasks and objects, making it difficult to generalize to novel objects and complex scenes. In this paper, we introduce AffordGrasp, a novel open-vocabulary grasping framework that leverages the reasoning capabilities of vision-language models (VLMs) for in-context affordance reasoning. Unlike existing methods that rely on explicit task and object specifications, our approach infers tasks directly from implicit user instructions, enabling more intuitive and seamless human-robot interaction in everyday scenarios. Building on the reasoning outcomes, our framework identifies task-relevant objects and grounds their part-level affordances using a visual grounding module. This allows us to generate task-oriented grasp poses precisely within the affordance regions of the object, ensuring both functional and context-aware robotic manipulation. Extensive experiments demonstrate that AffordGrasp achieves state-of-the-art performance in both simulation and real-world scenarios, highlighting the effectiveness of our method. We believe our approach advances robotic manipulation techniques and contributes to the broader field of embodied AI. Project website: https://eqcy.github.io/affordgrasp/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08208v3">SPA: 3D Spatial-Awareness Enables Effective Embodied Representation</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Project Page: https://haoyizhu.github.io/spa/
    </div>
    <details class="paper-abstract">
      In this paper, we introduce SPA, a novel representation learning framework that emphasizes the importance of 3D spatial awareness in embodied AI. Our approach leverages differentiable neural rendering on multi-view images to endow a vanilla Vision Transformer (ViT) with intrinsic spatial understanding. We present the most comprehensive evaluation of embodied representation learning to date, covering 268 tasks across 8 simulators with diverse policies in both single-task and language-conditioned multi-task scenarios. The results are compelling: SPA consistently outperforms more than 10 state-of-the-art representation methods, including those specifically designed for embodied AI, vision-centric tasks, and multi-modal applications, while using less training data. Furthermore, we conduct a series of real-world experiments to confirm its effectiveness in practical scenarios. These results highlight the critical role of 3D spatial awareness for embodied representation learning. Our strongest model takes more than 6000 GPU hours to train and we are committed to open-sourcing all code and model weights to foster future research in embodied representation learning. Project Page: https://haoyizhu.github.io/spa/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17735v4">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Constructing compact and informative 3D scene representations is essential for effective embodied exploration and reasoning, especially in complex environments over extended periods. Existing representations, such as object-centric 3D scene graphs, oversimplify spatial relationships by modeling scenes as isolated objects with restrictive textual relationships, making it difficult to address queries requiring nuanced spatial understanding. Moreover, these representations lack natural mechanisms for active exploration and memory management, hindering their application to lifelong autonomy. In this work, we propose 3D-Mem, a novel 3D scene memory framework for embodied agents. 3D-Mem employs informative multi-view images, termed Memory Snapshots, to represent the scene and capture rich visual information of explored regions. It further integrates frontier-based exploration by introducing Frontier Snapshots-glimpses of unexplored areas-enabling agents to make informed decisions by considering both known and potential new information. To support lifelong memory in active exploration settings, we present an incremental construction pipeline for 3D-Mem, as well as a memory retrieval technique for memory management. Experimental results on three benchmarks demonstrate that 3D-Mem significantly enhances agents' exploration and reasoning capabilities in 3D environments, highlighting its potential for advancing applications in embodied AI.
    </details>
</div>
