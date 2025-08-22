# embodied ai - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15769v1">SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Technical Report; Project Page: https://mengmouxu.github.io/SceneGen
    </div>
    <details class="paper-abstract">
      3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: https://mengmouxu.github.io/SceneGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15663v1">Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15201v1">Survey of Vision-Language-Action Models for Embodied Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15119v1">Open-Universe Assistance Games</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 7 pages + 2 pages references + 7 pages appendix
    </div>
    <details class="paper-abstract">
      Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10287v2">JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent advances in Vision-Language Models (VLMs) and large language models (LLMs) have greatly enhanced visual reasoning, a key capability for embodied AI agents like robots. However, existing visual reasoning benchmarks often suffer from several limitations: they lack a clear definition of reasoning complexity, offer have no control to generate questions over varying difficulty and task customization, and fail to provide structured, step-by-step reasoning annotations (workflows). To bridge these gaps, we formalize reasoning complexity, introduce an adaptive query engine that generates customizable questions of varying complexity with detailed intermediate annotations, and extend the JRDB dataset with human-object interaction and geometric relationship annotations to create JRDB-Reasoning, a benchmark tailored for visual reasoning in human-crowded environments. Our engine and benchmark enable fine-grained evaluation of visual reasoning frameworks and dynamic assessment of visual-language models across reasoning levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13998v1">Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Embodied-R1 technical report
    </div>
    <details class="paper-abstract">
      Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13901v1">Multimodal Data Storage and Retrieval for Embodied AI: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Embodied AI (EAI) agents continuously interact with the physical world, generating vast, heterogeneous multimodal data streams that traditional management systems are ill-equipped to handle. In this survey, we first systematically evaluate five storage architectures (Graph Databases, Multi-Model Databases, Data Lakes, Vector Databases, and Time-Series Databases), focusing on their suitability for addressing EAI's core requirements, including physical grounding, low-latency access, and dynamic scalability. We then analyze five retrieval paradigms (Fusion Strategy-Based Retrieval, Representation Alignment-Based Retrieval, Graph-Structure-Based Retrieval, Generation Model-Based Retrieval, and Efficient Retrieval-Based Optimization), revealing a fundamental tension between achieving long-term semantic coherence and maintaining real-time responsiveness. Based on this comprehensive analysis, we identify key bottlenecks, spanning from the foundational Physical Grounding Gap to systemic challenges in cross-modal integration, dynamic adaptation, and open-world generalization. Finally, we outline a forward-looking research agenda encompassing physics-aware data models, adaptive storage-retrieval co-optimization, and standardized benchmarking, to guide future research toward principled data management solutions for EAI. Our survey is based on a comprehensive review of more than 180 related studies, providing a rigorous roadmap for designing the robust, high-performance data management frameworks essential for the next generation of autonomous embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v5">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13444v1">Switch4EAI: Leveraging Console Game Platform for Benchmarking Robotic Athletics</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Workshop Submission
    </div>
    <details class="paper-abstract">
      Recent advances in whole-body robot control have enabled humanoid and legged robots to execute increasingly agile and coordinated movements. However, standardized benchmarks for evaluating robotic athletic performance in real-world settings and in direct comparison to humans remain scarce. We present Switch4EAI(Switch-for-Embodied-AI), a low-cost and easily deployable pipeline that leverages motion-sensing console games to evaluate whole-body robot control policies. Using Just Dance on the Nintendo Switch as a representative example, our system captures, reconstructs, and retargets in-game choreography for robotic execution. We validate the system on a Unitree G1 humanoid with an open-source whole-body controller, establishing a quantitative baseline for the robot's performance against a human player. In the paper, we discuss these results, which demonstrate the feasibility of using commercial games platform as physically grounded benchmarks and motivate future work to for benchmarking embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13421v1">Virtuous Machines: Towards Artificial General Science</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Artificial intelligence systems are transforming scientific discovery by accelerating specific research tasks, from protein structure prediction to materials design, yet remain confined to narrow domains requiring substantial human oversight. The exponential growth of scientific literature and increasing domain specialisation constrain researchers' capacity to synthesise knowledge across disciplines and develop unifying theories, motivating exploration of more general-purpose AI systems for science. Here we show that a domain-agnostic, agentic AI system can independently navigate the scientific workflow - from hypothesis generation through data collection to manuscript preparation. The system autonomously designed and executed three psychological studies on visual working memory, mental rotation, and imagery vividness, executed one new online data collection with 288 participants, developed analysis pipelines through 8-hour+ continuous coding sessions, and produced completed manuscripts. The results demonstrate the capability of AI scientific discovery pipelines to conduct non-trivial research with theoretical reasoning and methodological rigour comparable to experienced researchers, though with limitations in conceptual nuance and theoretical interpretation. This is a step toward embodied AI that can test hypotheses through real-world experiments, accelerating discovery by autonomously exploring regions of scientific space that human cognitive and resource constraints might otherwise leave unexplored. It raises important questions about the nature of scientific understanding and the attribution of scientific credit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13073v1">Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
    </div>
    <details class="paper-abstract">
      Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v4">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-18
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18774v3">Embodied Image Quality Assessment for Robotic Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-08-18
    </div>
    <details class="paper-abstract">
      Image Quality Assessment (IQA) of User-Generated Content (UGC) is a critical technique for human Quality of Experience (QoE). However, does the the image quality of Robot-Generated Content (RGC) demonstrate traits consistent with the Moravec paradox, potentially conflicting with human perceptual norms? Human subjective scoring is more based on the attractiveness of the image. Embodied agent are required to interact and perceive in the environment, and finally perform specific tasks. Visual images as inputs directly influence downstream tasks. In this paper, we explore the perception mechanism of embodied robots for image quality. We propose the first Embodied Preference Database (EPD), which contains 12,500 distorted image annotations. We establish assessment metrics based on the downstream tasks of robot. In addition, there is a gap between UGC and RGC. To address this, we propose a novel Multi-scale Attention Embodied Image Quality Assessment called MA-EIQA. For the proposed EPD dataset, this is the first no-reference IQA model designed for embodied robot. Finally, the performance of mainstream IQA algorithms on EPD dataset is verified. The experiments demonstrate that quality assessment of embodied images is different from that of humans. We sincerely hope that the EPD can contribute to the development of embodied AI by focusing on image quality assessment. The benchmark is available at https://github.com/Jianbo-maker/EPD_benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07501v2">FormCoach: Lift Smarter, Not Harder</a></div>
    <div class="paper-meta">
      📅 2025-08-16
    </div>
    <details class="paper-abstract">
      Good form is the difference between strength and strain, yet for the fast-growing community of at-home fitness enthusiasts, expert feedback is often out of reach. FormCoach transforms a simple camera into an always-on, interactive AI training partner, capable of spotting subtle form errors and delivering tailored corrections in real time, leveraging vision-language models (VLMs). We showcase this capability through a web interface and benchmark state-of-the-art VLMs on a dataset of 1,700 expert-annotated user-reference video pairs spanning 22 strength and mobility exercises. To accelerate research in AI-driven coaching, we release both the dataset and an automated, rubric-based evaluation pipeline, enabling standardized comparison across models. Our benchmarks reveal substantial gaps compared to human-level coaching, underscoring both the challenges and opportunities in integrating nuanced, context-aware movement analysis into interactive AI systems. By framing form correction as a collaborative and creative process between humans and machines, FormCoach opens a new frontier in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08896v3">Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-15
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v3">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-14
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10399v1">Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Embodied AI aims to develop intelligent systems with physical forms capable of perceiving, decision-making, acting, and learning in real-world environments, providing a promising way to Artificial General Intelligence (AGI). Despite decades of explorations, it remains challenging for embodied agents to achieve human-level intelligence for general-purpose tasks in open dynamic environments. Recent breakthroughs in large models have revolutionized embodied AI by enhancing perception, interaction, planning and learning. In this article, we provide a comprehensive survey on large model empowered embodied AI, focusing on autonomous decision-making and embodied learning. We investigate both hierarchical and end-to-end decision-making paradigms, detailing how large models enhance high-level planning, low-level execution, and feedback for hierarchical decision-making, and how large models enhance Vision-Language-Action (VLA) models for end-to-end decision making. For embodied learning, we introduce mainstream learning methodologies, elaborating on how large models enhance imitation learning and reinforcement learning in-depth. For the first time, we integrate world models into the survey of embodied AI, presenting their design methods and critical roles in enhancing decision-making and learning. Though solid advances have been achieved, challenges still exist, which are discussed at the end of this survey, potentially as the further research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10287v1">JRDB-Reasoning: A Difficulty-Graded Benchmark for Visual Reasoning in Robotics</a></div>
    <div class="paper-meta">
      📅 2025-08-14
    </div>
    <details class="paper-abstract">
      Recent advances in Vision-Language Models (VLMs) and large language models (LLMs) have greatly enhanced visual reasoning, a key capability for embodied AI agents like robots. However, existing visual reasoning benchmarks often suffer from several limitations: they lack a clear definition of reasoning complexity, offer have no control to generate questions over varying difficulty and task customization, and fail to provide structured, step-by-step reasoning annotations (workflows). To bridge these gaps, we formalize reasoning complexity, introduce an adaptive query engine that generates customizable questions of varying complexity with detailed intermediate annotations, and extend the JRDB dataset with human-object interaction and geometric relationship annotations to create JRDB-Reasoning, a benchmark tailored for visual reasoning in human-crowded environments. Our engine and benchmark enable fine-grained evaluation of visual reasoning frameworks and dynamic assessment of visual-language models across reasoning levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08896v2">Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-13
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07818v2">MoSE: Skill-by-Skill Mixture-of-Experts Learning for Embodied Autonomous Machines</a></div>
    <div class="paper-meta">
      📅 2025-08-13
    </div>
    <details class="paper-abstract">
      To meet the growing demand for smarter, faster, and more efficient embodied AI solutions, we introduce a novel Mixture-of-Expert (MoE) method that significantly boosts reasoning and learning efficiency for embodied autonomous systems. General MoE models demand extensive training data and complex optimization, which limits their applicability in embodied AI such as autonomous driving (AD) and robotic manipulation. In this work, we propose a skill-oriented MoE called MoSE, which mimics the human learning and reasoning process skill-by-skill, step-by-step. We introduce a skill-oriented routing mechanism that begins with defining and annotating specific skills, enabling experts to identify the necessary competencies for various scenarios and reasoning tasks, thereby facilitating skill-by-skill learning. To better align with multi-step planning in human reasoning and in end-to-end driving models, we build a hierarchical skill dataset and pretrain the router to encourage the model to think step-by-step. Unlike other multi-round dialogues, MoSE integrates valuable auxiliary tasks (e.g. perception-prediction-planning for AD, and high-level and low-level planning for robots) in one single forward process without introducing any extra computational cost. With less than 3B sparsely activated parameters, our model effectively grows more diverse expertise and outperforms models on both AD corner-case reasoning tasks and robot reasoning tasks with less than 40% of the parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08896v1">Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20977v2">UnrealZoo: Enriching Photo-realistic Virtual Worlds for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 ICCV 2025 (Highlight), Project page: http://unrealzoo.site/
    </div>
    <details class="paper-abstract">
      We introduce UnrealZoo, a collection of over 100 photo-realistic 3D virtual worlds built on Unreal Engine, designed to reflect the complexity and variability of open-world environments. We also provide a rich variety of playable entities, including humans, animals, robots, and vehicles for embodied AI research. We extend UnrealCV with optimized APIs and tools for data collection, environment augmentation, distributed training, and benchmarking. These improvements achieve significant improvements in the efficiency of rendering and communication, enabling advanced applications such as multi-agent interactions. Our experimental evaluation across visual navigation and tracking tasks reveals two key insights: 1) environmental diversity provides substantial benefits for developing generalizable reinforcement learning (RL) agents, and 2) current embodied agents face persistent challenges in open-world scenarios, including navigation in unstructured terrain, adaptation to unseen morphologies, and managing latency in the close-loop control systems for interacting in highly dynamic objects. UnrealZoo thus serves as both a comprehensive testing ground and a pathway toward developing more capable embodied AI systems for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08831v1">DiffPhysCam: Differentiable Physics-Based Camera Simulation for Inverse Rendering and Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 19 pages, 17 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      We introduce DiffPhysCam, a differentiable camera simulator designed to support robotics and embodied AI applications by enabling gradient-based optimization in visual perception pipelines. Generating synthetic images that closely mimic those from real cameras is essential for training visual models and enabling end-to-end visuomotor learning. Moreover, differentiable rendering allows inverse reconstruction of real-world scenes as digital twins, facilitating simulation-based robotics training. However, existing virtual cameras offer limited control over intrinsic settings, poorly capture optical artifacts, and lack tunable calibration parameters -- hindering sim-to-real transfer. DiffPhysCam addresses these limitations through a multi-stage pipeline that provides fine-grained control over camera settings, models key optical effects such as defocus blur, and supports calibration with real-world data. It enables both forward rendering for image synthesis and inverse rendering for 3D scene reconstruction, including mesh and material texture optimization. We show that DiffPhysCam enhances robotic perception performance in synthetic image tasks. As an illustrative example, we create a digital twin of a real-world scene using inverse rendering, simulate it in a multi-physics environment, and demonstrate navigation of an autonomous ground vehicle using images generated by DiffPhysCam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16258v3">MEReQ: Max-Ent Residual-Q Inverse RL for Sample-Efficient Alignment from Intervention</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Aligning robot behavior with human preferences is crucial for deploying embodied AI agents in human-centered environments. A promising solution is interactive imitation learning from human intervention, where a human expert observes the policy's execution and provides interventions as feedback. However, existing methods often fail to utilize the prior policy efficiently to facilitate learning, thus hindering sample efficiency. In this work, we introduce MEReQ (Maximum-Entropy Residual-Q Inverse Reinforcement Learning), designed for sample-efficient alignment from human intervention. Instead of inferring the complete human behavior characteristics, MEReQ infers a residual reward function that captures the discrepancy between the human expert's and the prior policy's underlying reward functions. It then employs Residual Q-Learning (RQL) to align the policy with human preferences using this residual reward function. Extensive evaluations on simulated and real-world tasks demonstrate that MEReQ achieves sample-efficient policy alignment from human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08252v1">ReferSplat: Referring Segmentation in 3D Gaussian Splatting</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 ICML 2025 Oral, Code: https://github.com/heshuting555/ReferSplat
    </div>
    <details class="paper-abstract">
      We introduce Referring 3D Gaussian Splatting Segmentation (R3DGS), a new task that aims to segment target objects in a 3D Gaussian scene based on natural language descriptions, which often contain spatial relationships or object attributes. This task requires the model to identify newly described objects that may be occluded or not directly visible in a novel view, posing a significant challenge for 3D multi-modal understanding. Developing this capability is crucial for advancing embodied AI. To support research in this area, we construct the first R3DGS dataset, Ref-LERF. Our analysis reveals that 3D multi-modal understanding and spatial relationship modeling are key challenges for R3DGS. To address these challenges, we propose ReferSplat, a framework that explicitly models 3D Gaussian points with natural language expressions in a spatially aware paradigm. ReferSplat achieves state-of-the-art performance on both the newly proposed R3DGS task and 3D open-vocabulary segmentation benchmarks. Dataset and code are available at https://github.com/heshuting555/ReferSplat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07501v1">FormCoach: Lift Smarter, Not Harder</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Good form is the difference between strength and strain, yet for the fast-growing community of at-home fitness enthusiasts, expert feedback is often out of reach. FormCoach transforms a simple camera into an always-on, interactive AI training partner, capable of spotting subtle form errors and delivering tailored corrections in real time, leveraging vision-language models (VLMs). We showcase this capability through a web interface and benchmark state-of-the-art VLMs on a dataset of 1,700 expert-annotated user-reference video pairs spanning 22 strength and mobility exercises. To accelerate research in AI-driven coaching, we release both the dataset and an automated, rubric-based evaluation pipeline, enabling standardized comparison across models. Our benchmarks reveal substantial gaps compared to human-level coaching, underscoring both the challenges and opportunities in integrating nuanced, context-aware movement analysis into interactive AI systems. By framing form correction as a collaborative and creative process between humans and machines, FormCoach opens a new frontier in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05750v3">Semantic Mapping in Indoor Embodied AI -- A Survey on Advances, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Intelligent embodied agents (e.g. robots) need to perform complex semantic tasks in unfamiliar environments. Among many skills that the agents need to possess, building and maintaining a semantic map of the environment is most crucial in long-horizon tasks. A semantic map captures information about the environment in a structured way, allowing the agent to reference it for advanced reasoning throughout the task. While existing surveys in embodied AI focus on general advancements or specific tasks like navigation and manipulation, this paper provides a comprehensive review of semantic map-building approaches in embodied AI, specifically for indoor navigation. We categorize these approaches based on their structural representation (spatial grids, topological graphs, dense point-clouds or hybrid maps) and the type of information they encode (implicit features or explicit environmental data). We also explore the strengths and limitations of the map building techniques, highlight current challenges, and propose future research directions. We identify that the field is moving towards developing open-vocabulary, queryable, task-agnostic map representations, while high memory demands and computational inefficiency still remaining to be open challenges. This survey aims to guide current and future researchers in advancing semantic mapping techniques for embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06767v1">PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems</a></div>
    <div class="paper-meta">
      📅 2025-08-09
    </div>
    <details class="paper-abstract">
      Digital Twins (DTs) are transforming industries through advanced data processing and analysis, positioning the world of DTs, Digital World, as a cornerstone of nextgeneration technologies including embodied AI. As robotics and automated systems scale, efficient data-sharing frameworks and robust algorithms become critical. We explore the pivotal role of data handling in next-gen networks, focusing on dynamics between application and network providers (AP/NP) in DT ecosystems. We introduce PANAMA, a novel algorithm with Priority Asymmetry for Network Aware Multi-agent Reinforcement Learning (MARL) based multi-agent path finding (MAPF). By adopting a Centralized Training with Decentralized Execution (CTDE) framework and asynchronous actor-learner architectures, PANAMA accelerates training while enabling autonomous task execution by embodied AI. Our approach demonstrates superior pathfinding performance in accuracy, speed, and scalability compared to existing benchmarks. Through simulations, we highlight optimized data-sharing strategies for scalable, automated systems, ensuring resilience in complex, real-world environments. PANAMA bridges the gap between network-aware decision-making and robust multi-agent coordination, advancing the synergy between DTs, wireless networks, and AI-driven automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v2">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 CoRL 2025
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05614v1">OmniEAR: Benchmarking Agent Reasoning in Embodied Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Project Page: https://zju-real.github.io/OmniEmbodied Code: https://github.com/ZJU-REAL/OmniEmbodied
    </div>
    <details class="paper-abstract">
      Large language models excel at abstract reasoning but their capacity for embodied agent reasoning remains largely unexplored. We present OmniEAR, a comprehensive framework for evaluating how language models reason about physical interactions, tool usage, and multi-agent coordination in embodied tasks. Unlike existing benchmarks that provide predefined tool sets or explicit collaboration directives, OmniEAR requires agents to dynamically acquire capabilities and autonomously determine coordination strategies based on task demands. Through text-based environment representation, we model continuous physical properties and complex spatial relationships across 1,500 scenarios spanning household and industrial domains. Our systematic evaluation reveals severe performance degradation when models must reason from constraints: while achieving 85-96% success with explicit instructions, performance drops to 56-85% for tool reasoning and 63-85% for implicit collaboration, with compound tasks showing over 50% failure rates. Surprisingly, complete environmental information degrades coordination performance, indicating models cannot filter task-relevant constraints. Fine-tuning improves single-agent tasks dramatically (0.6% to 76.3%) but yields minimal multi-agent gains (1.5% to 5.5%), exposing fundamental architectural limitations. These findings demonstrate that embodied reasoning poses fundamentally different challenges than current models can address, establishing OmniEAR as a rigorous benchmark for evaluating and advancing embodied AI systems. Our code and data are included in the supplementary materials and will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05543v1">CleanUpBench: Embodied Sweeping and Grasping Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Embodied AI benchmarks have advanced navigation, manipulation, and reasoning, but most target complex humanoid agents or large-scale simulations that are far from real-world deployment. In contrast, mobile cleaning robots with dual mode capabilities, such as sweeping and grasping, are rapidly emerging as realistic and commercially viable platforms. However, no benchmark currently exists that systematically evaluates these agents in structured, multi-target cleaning tasks, revealing a critical gap between academic research and real-world applications. We introduce CleanUpBench, a reproducible and extensible benchmark for evaluating embodied agents in realistic indoor cleaning scenarios. Built on NVIDIA Isaac Sim, CleanUpBench simulates a mobile service robot equipped with a sweeping mechanism and a six-degree-of-freedom robotic arm, enabling interaction with heterogeneous objects. The benchmark includes manually designed environments and one procedurally generated layout to assess generalization, along with a comprehensive evaluation suite covering task completion, spatial efficiency, motion quality, and control performance. To support comparative studies, we provide baseline agents based on heuristic strategies and map-based planning. CleanUpBench bridges the gap between low-level skill evaluation and full-scene testing, offering a scalable testbed for grounded, embodied intelligence in everyday settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05855v1">Safety of Embodied Navigation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11773v3">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents-virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16402v2">IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. Code and data are released under [this https URL](https://github.com/AI45Lab/IS-Bench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02029v4">RoboBrain 2.0 Technical Report</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21045v2">Reconstructing 4D Spatial Intelligence: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence
    </div>
    <details class="paper-abstract">
      Reconstructing 4D spatial intelligence from visual observations has long been a central yet challenging task in computer vision, with broad real-world applications. These range from entertainment domains like movies, where the focus is often on reconstructing fundamental visual elements, to embodied AI, which emphasizes interaction modeling and physical realism. Fueled by rapid advances in 3D representations and deep learning architectures, the field has evolved quickly, outpacing the scope of previous surveys. Additionally, existing surveys rarely offer a comprehensive analysis of the hierarchical structure of 4D scene reconstruction. To address this gap, we present a new perspective that organizes existing methods into five progressive levels of 4D spatial intelligence: (1) Level 1 -- reconstruction of low-level 3D attributes (e.g., depth, pose, and point maps); (2) Level 2 -- reconstruction of 3D scene components (e.g., objects, humans, structures); (3) Level 3 -- reconstruction of 4D dynamic scenes; (4) Level 4 -- modeling of interactions among scene components; and (5) Level 5 -- incorporation of physical laws and constraints. We conclude the survey by discussing the key challenges at each level and highlighting promising directions for advancing toward even richer levels of 4D spatial intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/yukangcao/Awesome-4D-Spatial-Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12974v3">Exploring 3D Reasoning-Driven Planning: From Implicit Human Intentions to Route-Aware Activity Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      3D task planning has attracted increasing attention in human-robot interaction and embodied AI thanks to the recent advances in multimodal learning. However, most existing studies are facing two common challenges: 1) heavy reliance on explicit instructions with little reasoning on implicit user intention; 2) negligence of inter-step route planning on robot moves. We address the above challenges by proposing 3D Reasoning-Driven Planning, a novel 3D task that reasons the intended activities from implicit instructions and decomposes them into steps with inter-step routes and planning under the guidance of fine-grained 3D object shapes and locations from scene segmentation. We tackle the new 3D task from two perspectives. First, we construct ReasonPlan3D, a large-scale benchmark that covers diverse 3D scenes with rich implicit instructions and detailed annotations for multi-step task planning, inter-step route planning, and fine-grained segmentation. Second, we design a novel framework that introduces progressive plan generation with contextual consistency across multiple steps, as well as a scene graph that is updated dynamically for capturing critical objects and their spatial relations. Extensive experiments demonstrate the effectiveness of our benchmark and framework in reasoning activities from implicit human instructions, producing accurate stepwise task plans and seamlessly integrating route planning for multi-step moves. The dataset and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13927v4">Multimodal 3D Reasoning Segmentation with Complex Scenes</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      The recent development in multimodal learning has greatly advanced the research in 3D scene understanding in various real-world tasks such as embodied AI. However, most existing studies are facing two common challenges: 1) they are short of reasoning ability for interaction and interpretation of human intentions and 2) they focus on scenarios with single-category objects and over-simplified textual descriptions and neglect multi-object scenarios with complicated spatial relations among objects. We address the above challenges by proposing a 3D reasoning segmentation task for reasoning segmentation with multiple objects in scenes. The task allows producing 3D segmentation masks and detailed textual explanations as enriched by 3D spatial relations among objects. To this end, we create ReasonSeg3D, a large-scale and high-quality benchmark that integrates 3D segmentation masks and 3D spatial relations with generated question-answer pairs. In addition, we design MORE3D, a novel 3D reasoning network that works with queries of multiple objects and is tailored for 3D scene understanding. MORE3D learns detailed explanations on 3D relations and employs them to capture spatial information of objects and reason textual outputs. Extensive experiments show that MORE3D excels in reasoning and segmenting complex multi-object 3D scenes. In addition, the created ReasonSeg3D offers a valuable platform for future exploration of 3D reasoning segmentation. The data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19516v2">Boosting Robotic Manipulation Generalization with Minimal Costly Data</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      The growing adoption of Vision-Language-Action (VLA) models in embodied AI intensifies the demand for diverse manipulation demonstrations. However, high costs associated with data collection often result in insufficient data coverage across all scenarios, which limits the performance of the models. It is observed that the spatial reasoning phase (SRP) in large workspace dominates the failure cases. Fortunately, this data can be collected with low cost, underscoring the potential of leveraging inexpensive data to improve model performance. In this paper, we introduce the RoboTron-Craft, a stage-divided and cost-effective pipeline for realistic manipulation generation. Base on this, the RoboTron-Platter method is introduced, a framework that decouples training trajectories into distinct task stages and leverages abundant easily collectible SRP data to enhance VLA model's generalization. Through analysis we demonstrate that sub-task-specific training with additional SRP data with proper proportion can act as a performance catalyst for robot manipulation, maximizing the utilization of costly physical interaction phase (PIP) data. Experiments show that through introducing large proportion of cost-effective SRP trajectories into a limited set of PIP data, we can achieve a maximum improvement of 41\% on success rate in zero-shot scenes, while with the ability to transfer manipulation skill to novel targets. Project available at https://github.com/ notFoundThisPerson/RoboTron-Craft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01184v1">Object Affordance Recognition and Grounding via Multi-scale Cross-modal Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      A core problem of Embodied AI is to learn object manipulation from observation, as humans do. To achieve this, it is important to localize 3D object affordance areas through observation such as images (3D affordance grounding) and understand their functionalities (affordance classification). Previous attempts usually tackle these two tasks separately, leading to inconsistent predictions due to lacking proper modeling of their dependency. In addition, these methods typically only ground the incomplete affordance areas depicted in images, failing to predict the full potential affordance areas, and operate at a fixed scale, resulting in difficulty in coping with affordances significantly varying in scale with respect to the whole object. To address these issues, we propose a novel approach that learns an affordance-aware 3D representation and employs a stage-wise inference strategy leveraging the dependency between grounding and classification tasks. Specifically, we first develop a cross-modal 3D representation through efficient fusion and multi-scale geometric feature propagation, enabling inference of full potential affordance areas at a suitable regional scale. Moreover, we adopt a simple two-stage prediction mechanism, effectively coupling grounding and classification for better affordance understanding. Experiments demonstrate the effectiveness of our method, showing improved performance in both affordance grounding and classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00400v1">Sari Sandbox: A Virtual Retail Store Environment for Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 14 pages, accepted in ICCV 2025 Workshop on RetailVision
    </div>
    <details class="paper-abstract">
      We present Sari Sandbox, a high-fidelity, photorealistic 3D retail store simulation for benchmarking embodied agents against human performance in shopping tasks. Addressing a gap in retail-specific sim environments for embodied agent training, Sari Sandbox features over 250 interactive grocery items across three store configurations, controlled via an API. It supports both virtual reality (VR) for human interaction and a vision language model (VLM)-powered embodied agent. We also introduce SariBench, a dataset of annotated human demonstrations across varied task difficulties. Our sandbox enables embodied agents to navigate, inspect, and manipulate retail items, providing baselines against human performance. We conclude with benchmarks, performance analysis, and recommendations for enhancing realism and scalability. The source code can be accessed via https://github.com/upeee/sari-sandbox-env.
    </details>
</div>
