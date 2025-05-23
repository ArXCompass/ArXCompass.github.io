# embodied ai - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16928v1">Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16815v1">Perceptual Quality Assessment for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: https://github.com/lcysyzxdxc/EmbodiedIQA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16965v2">Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-22
    </div>
    <details class="paper-abstract">
      Vision Language Models exhibited immense potential for embodied AI, yet they often lack the sophisticated situational reasoning required for complex decision-making. This paper shows that VLMs can achieve surprisingly strong decision-making performance when visual scenes are represented merely as text-only descriptions, suggesting foundational reasoning can be effectively learned from language. Motivated by this insight, we propose Praxis-VLM, a reasoning VLM for vision-grounded decision-making. Praxis-VLM employs the GRPO algorithm on textual scenarios to instill robust reasoning capabilities, where models learn to evaluate actions and their consequences. These reasoning skills, acquired purely from text, successfully transfer to multimodal inference with visual inputs, significantly reducing reliance on scarce paired image-text training data. Experiments across diverse decision-making benchmarks demonstrate that Praxis-VLM substantially outperforms standard supervised fine-tuning, exhibiting superior performance and generalizability. Further analysis confirms that our models engage in explicit and effective reasoning, underpinning their enhanced performance and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16278v1">DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-22
      | 💬 Project Page: https://thinklab-sjtu.github.io/DriveMoE/
    </div>
    <details class="paper-abstract">
      End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE to Drive-$\pi_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$\pi_0$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18041v5">OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Vision-Language Navigation (VLN) aims to guide agents by leveraging language instructions and visual cues, playing a pivotal role in embodied AI. Indoor VLN has been extensively studied, whereas outdoor aerial VLN remains underexplored. The potential reason is that outdoor aerial view encompasses vast areas, making data collection more challenging, which results in a lack of benchmarks. To address this problem, we propose OpenFly, a platform comprising various rendering engines, a versatile toolchain, and a large-scale benchmark for aerial VLN. Firstly, we integrate diverse rendering engines and advanced techniques for environment simulation, including Unreal Engine, GTA V, Google Earth, and 3D Gaussian Splatting (3D GS). Particularly, 3D GS supports real-to-sim rendering, further enhancing the realism of our environments. Secondly, we develop a highly automated toolchain for aerial VLN data collection, streamlining point cloud acquisition, scene semantic segmentation, flight trajectory creation, and instruction generation. Thirdly, based on the toolchain, we construct a large-scale aerial VLN dataset with 100k trajectories, covering diverse heights and lengths across 18 scenes. Moreover, we propose OpenFly-Agent, a keyframe-aware VLN model emphasizing key observations during flight. For benchmarking, extensive experiments and analyses are conducted, evaluating several recent VLN methods and showcasing the superiority of our OpenFly platform and agent. The toolchain, dataset, and codes will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15197v1">Intentional Gesture: Deliver Your Intentions with Gestures for Speech</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      When humans speak, gestures help convey communicative intentions, such as adding emphasis or describing concepts. However, current co-speech gesture generation methods rely solely on superficial linguistic cues (\textit{e.g.} speech audio or text transcripts), neglecting to understand and leverage the communicative intention that underpins human gestures. This results in outputs that are rhythmically synchronized with speech but are semantically shallow. To address this gap, we introduce \textbf{Intentional-Gesture}, a novel framework that casts gesture generation as an intention-reasoning task grounded in high-level communicative functions. % First, we curate the \textbf{InG} dataset by augmenting BEAT-2 with gesture-intention annotations (\textit{i.e.}, text sentences summarizing intentions), which are automatically annotated using large vision-language models. Next, we introduce the \textbf{Intentional Gesture Motion Tokenizer} to leverage these intention annotations. It injects high-level communicative functions (\textit{e.g.}, intentions) into tokenized motion representations to enable intention-aware gesture synthesis that are both temporally aligned and semantically meaningful, achieving new state-of-the-art performance on the BEAT-2 benchmark. Our framework offers a modular foundation for expressive gesture generation in digital humans and embodied AI. Project Page: https://andypinxinliu.github.io/Intentional-Gesture
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14366v1">Towards Embodied Cognition in Robots via Spatially Grounded Synthetic Worlds</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted to: Intelligent Autonomous Systems (IAS) 2025 as Late Breaking Report
    </div>
    <details class="paper-abstract">
      We present a conceptual framework for training Vision-Language Models (VLMs) to perform Visual Perspective Taking (VPT), a core capability for embodied cognition essential for Human-Robot Interaction (HRI). As a first step toward this goal, we introduce a synthetic dataset, generated in NVIDIA Omniverse, that enables supervised learning for spatial reasoning tasks. Each instance includes an RGB image, a natural language description, and a ground-truth 4X4 transformation matrix representing object pose. We focus on inferring Z-axis distance as a foundational skill, with future extensions targeting full 6 Degrees Of Freedom (DOFs) reasoning. The dataset is publicly available to support further research. This work serves as a foundational step toward embodied AI systems capable of spatial understanding in interactive human-robot scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00379v4">Latent Action Learning Requires Supervision in the Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 ICML 2025, Poster, Source code: https://github.com/dunnolab/laom
    </div>
    <details class="paper-abstract">
      Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14235v1">Toward Embodied AGI: A Review of Embodied AI and the Road Ahead</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Artificial General Intelligence (AGI) is often envisioned as inherently embodied. With recent advances in robotics and foundational AI models, we stand at the threshold of a new era-one marked by increasingly generalized embodied AI systems. This paper contributes to the discourse by introducing a systematic taxonomy of Embodied AGI spanning five levels (L1-L5). We review existing research and challenges at the foundational stages (L1-L2) and outline the key components required to achieve higher-level capabilities (L3-L5). Building on these insights and existing technologies, we propose a conceptual framework for an L3+ robotic brain, offering both a technical outlook and a foundation for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14197v1">Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Omnidirectional images (ODIs), with their 360{\deg} field of view, provide unparalleled spatial awareness for immersive applications like augmented reality and embodied AI. However, the capability of existing multi-modal large language models (MLLMs) to comprehend and reason about such panoramic scenes remains underexplored. This paper addresses this gap by introducing OmniVQA, the first dataset and conducting the first benchmark for omnidirectional visual question answering. Our evaluation of state-of-the-art MLLMs reveals significant limitations in handling omnidirectional visual question answering, highlighting persistent challenges in object localization, feature extraction, and hallucination suppression within panoramic contexts. These results underscore the disconnect between current MLLM capabilities and the demands of omnidirectional visual understanding, which calls for dedicated architectural or training innovations tailored to 360{\deg} imagery. Building on the OmniVQA dataset and benchmark, we further introduce a rule-based reinforcement learning method, 360-R1, based on Qwen2.5-VL-Instruct. Concretely, we modify the group relative policy optimization (GRPO) by proposing three novel reward functions: (1) reasoning process similarity reward, (2) answer semantic accuracy reward, and (3) structured format compliance reward. Extensive experiments on our OmniVQA demonstrate the superiority of our proposed method in omnidirectional space (+6% improvement).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14129v1">Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 16 pages, 14 figures, currently under review
    </div>
    <details class="paper-abstract">
      Evolution and learning have historically been interrelated topics, and their interplay is attracting increased interest lately. The emerging new factor in this trend is morphological evolution, the evolution of physical forms within embodied AI systems such as robots. In this study, we investigate a system of hexacopter-type drones with evolvable morphologies and learnable controllers and make contributions to two fields. For aerial robotics, we demonstrate that the combination of evolution and learning can deliver non-conventional drones that significantly outperform the traditional hexacopter on several tasks that are more complex than previously considered in the literature. For the field of Evolutionary Computing, we introduce novel metrics and perform new analyses into the interaction of morphological evolution and learning, uncovering hitherto unidentified effects. Our analysis tools are domain-agnostic, making a methodological contribution towards building solid foundations for embodied AI systems that integrate evolution and learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00392v2">RefDrone: A Challenging Benchmark for Referring Expression Comprehension in Drone Scenes</a></div>
    <div class="paper-meta">
      📅 2025-05-19
    </div>
    <details class="paper-abstract">
      Drones have become prevalent robotic platforms with diverse applications, showing significant potential in Embodied Artificial Intelligence (Embodied AI). Referring Expression Comprehension (REC) enables drones to locate objects based on natural language expressions, a crucial capability for Embodied AI. Despite advances in REC for ground-level scenes, aerial views introduce unique challenges including varying viewpoints, occlusions and scale variations. To address this gap, we introduce RefDrone, a REC benchmark for drone scenes. RefDrone reveals three key challenges in REC: 1) multi-scale and small-scale target detection; 2) multi-target and no-target samples; 3) complex environment with rich contextual expressions. To efficiently construct this dataset, we develop RDAgent (referring drone annotation framework with multi-agent system), a semi-automated annotation tool for REC tasks. RDAgent ensures high-quality contextual expressions and reduces annotation cost. Furthermore, we propose Number GroundingDINO (NGDINO), a novel method designed to handle multi-target and no-target cases. NGDINO explicitly learns and utilizes the number of objects referred to in the expression. Comprehensive experiments with state-of-the-art REC methods demonstrate that NGDINO achieves superior performance on both the proposed RefDrone and the existing gRefCOCO datasets. The dataset and code are be publicly at https://github.com/sunzc-sunny/refdrone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12707v1">PLAICraft: Large-Scale Time-Aligned Vision-Speech-Action Dataset for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-05-19
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Advances in deep generative modelling have made it increasingly plausible to train human-level embodied agents. Yet progress has been limited by the absence of large-scale, real-time, multi-modal, and socially interactive datasets that reflect the sensory-motor complexity of natural environments. To address this, we present PLAICraft, a novel data collection platform and dataset capturing multiplayer Minecraft interactions across five time-aligned modalities: video, game output audio, microphone input audio, mouse, and keyboard actions. Each modality is logged with millisecond time precision, enabling the study of synchronous, embodied behaviour in a rich, open-ended world. The dataset comprises over 10,000 hours of gameplay from more than 10,000 global participants.\footnote{We have done a privacy review for the public release of an initial 200-hour subset of the dataset, with plans to release most of the dataset over time.} Alongside the dataset, we provide an evaluation suite for benchmarking model capabilities in object recognition, spatial awareness, language grounding, and long-term memory. PLAICraft opens a path toward training and evaluating agents that act fluently and purposefully in real time, paving the way for truly embodied artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09694v2">EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Website: https://github.com/AgibotTech/EWMBench
    </div>
    <details class="paper-abstract">
      Recent advances in creative AI have enabled the synthesis of high-fidelity images and videos conditioned on language instructions. Building on these developments, text-to-video diffusion models have evolved into embodied world models (EWMs) capable of generating physically plausible scenes from language commands, effectively bridging vision and action in embodied AI applications. This work addresses the critical challenge of evaluating EWMs beyond general perceptual metrics to ensure the generation of physically grounded and action-consistent behaviors. We propose the Embodied World Model Benchmark (EWMBench), a dedicated framework designed to evaluate EWMs based on three key aspects: visual scene consistency, motion correctness, and semantic alignment. Our approach leverages a meticulously curated dataset encompassing diverse scenes and motion patterns, alongside a comprehensive multi-dimensional evaluation toolkit, to assess and compare candidate models. The proposed benchmark not only identifies the limitations of existing video generation models in meeting the unique requirements of embodied tasks but also provides valuable insights to guide future advancements in the field. The dataset and evaluation tools are publicly available at https://github.com/AgibotTech/EWMBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12312v1">Visuospatial Cognitive Assistant</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 31 pages, 10 figures, 6 tables. The implementation and fine-tuned model (ViCA-7B) are publicly available at https://huggingface.co/nkkbr/ViCA. The ViCA-322K dataset can be found at https://huggingface.co/datasets/nkkbr/ViCA-322K, and the ViCA-Thinking-2.68K dataset is at https://huggingface.co/datasets/nkkbr/ViCA-thinking-2.68k
    </div>
    <details class="paper-abstract">
      Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12278v1">Emergent Active Perception and Dexterity of Simulated Humanoids from Visual Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-18
      | 💬 Project page: https://zhengyiluo.github.io/PDC
    </div>
    <details class="paper-abstract">
      Human behavior is fundamentally shaped by visual perception -- our ability to interact with the world depends on actively gathering relevant information and adapting our movements accordingly. Behaviors like searching for objects, reaching, and hand-eye coordination naturally emerge from the structure of our sensory system. Inspired by these principles, we introduce Perceptive Dexterous Control (PDC), a framework for vision-driven dexterous whole-body control with simulated humanoids. PDC operates solely on egocentric vision for task specification, enabling object search, target placement, and skill selection through visual cues, without relying on privileged state information (e.g., 3D object positions and geometries). This perception-as-interface paradigm enables learning a single policy to perform multiple household tasks, including reaching, grasping, placing, and articulated object manipulation. We also show that training from scratch with reinforcement learning can produce emergent behaviors such as active search. These results demonstrate how vision-driven control and complex tasks induce human-like behaviors and can serve as the key ingredients in closing the perception-action loop for animation, robotics, and embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00379v3">Latent Action Learning Requires Supervision in the Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2025-05-17
      | 💬 ICML 2025, Poster, Source code: https://github.com/dunnolab/laom
    </div>
    <details class="paper-abstract">
      Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11907v1">Are Multimodal Large Language Models Ready for Omnidirectional Spatial Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-05-17
    </div>
    <details class="paper-abstract">
      The 180x360 omnidirectional field of view captured by 360-degree cameras enables their use in a wide range of applications such as embodied AI and virtual reality. Although recent advances in multimodal large language models (MLLMs) have shown promise in visual-spatial reasoning, most studies focus on standard pinhole-view images, leaving omnidirectional perception largely unexplored. In this paper, we ask: Are MLLMs ready for omnidirectional spatial reasoning? To investigate this, we introduce OSR-Bench, the first benchmark specifically designed for this setting. OSR-Bench includes over 153,000 diverse question-answer pairs grounded in high-fidelity panoramic indoor scene maps. It covers key reasoning types including object counting, relative distance, and direction. We also propose a negative sampling strategy that inserts non-existent objects into prompts to evaluate hallucination and grounding robustness. For fine-grained analysis, we design a two-stage evaluation framework assessing both cognitive map generation and QA accuracy using rotation-invariant matching and a combination of rule-based and LLM-based metrics. We evaluate eight state-of-the-art MLLMs, including GPT-4o, Gemini 1.5 Pro, and leading open-source models under zero-shot settings. Results show that current models struggle with spatial reasoning in panoramic contexts, highlighting the need for more perceptually grounded MLLMs. OSR-Bench and code will be released at: https://huggingface.co/datasets/UUUserna/OSR-Bench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11191v1">Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 10 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: Foundation Models (FMs) provide a pathway toward generalization across tasks and modalities, whereas Federated Learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied environments. In this vision paper, we introduce Federated Foundation Models (FFMs) for embodied AI, a new paradigm that unifies the strengths of multi-modal multi-task (M3T) FMs with the privacy-preserving distributed nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying FFMs in embodied AI systems, along with the associated trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10183v1">KAITIAN: A Unified Communication Framework for Enabling Efficient Collaboration Across Heterogeneous Accelerators in Embodied AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 9 pages, 4 figures. Jieke Lin and Wanyu Wang contributed equally to this work
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) systems, such as autonomous robots and intelligent vehicles, are increasingly reliant on diverse heterogeneous accelerators (e.g., GPGPUs, NPUs, FPGAs) to meet stringent real-time processing and energy-efficiency demands. However, the proliferation of vendor-specific proprietary communication libraries creates significant interoperability barriers, hindering seamless collaboration between different accelerator types and leading to suboptimal resource utilization and performance bottlenecks in distributed AI workloads. This paper introduces KAITIAN, a novel distributed communication framework designed to bridge this gap. KAITIAN provides a unified abstraction layer that intelligently integrates vendor-optimized communication libraries for intra-group efficiency with general-purpose communication protocols for inter-group interoperability. Crucially, it incorporates a load-adaptive scheduling mechanism that dynamically balances computational tasks across heterogeneous devices based on their real-time performance characteristics. Implemented as an extension to PyTorch and rigorously evaluated on a testbed featuring NVIDIA GPUs and Cambricon MLUs, KAITIAN demonstrates significant improvements in resource utilization and scalability for distributed training tasks. Experimental results show that KAITIAN can accelerate training time by up to 42% compared to baseline homogeneous systems, while incurring minimal communication overhead (2.8--4.3%) and maintaining model accuracy. KAITIAN paves the way for more flexible and powerful heterogeneous computing in complex embodied AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10105v1">EmbodiedMAE: A Unified 3D Multi-Modal Representation for Robot Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10705v1">Embodied AI in Machine Learning -- is it Really Embodied?</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (Embodied AI) is gaining momentum in the machine learning communities with the goal of leveraging current progress in AI (deep learning, transformers, large language and visual-language models) to empower robots. In this chapter we put this work in the context of "Good Old-Fashioned Artificial Intelligence" (GOFAI) (Haugeland, 1989) and the behavior-based or embodied alternatives (R. A. Brooks 1991; Pfeifer and Scheier 2001). We claim that the AI-powered robots are only weakly embodied and inherit some of the problems of GOFAI. Moreover, we review and critically discuss the possibility of cross-embodiment learning (Padalkar et al. 2024). We identify fundamental roadblocks and propose directions on how to make progress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11523v2">ON as ALC: Active Loop Closing Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 Draft version of a conference paper with 7 pages, 5 figures, and 1 table
    </div>
    <details class="paper-abstract">
      In simultaneous localization and mapping, active loop closing (ALC) is an active vision problem that aims to visually guide a robot to maximize the chances of revisiting previously visited points, thereby resetting the drift errors accumulated in the incrementally built map during travel. However, current mainstream navigation strategies that leverage such incomplete maps as workspace prior knowledge often fail in modern long-term autonomy long-distance travel scenarios where map accumulation errors become significant. To address these limitations of map-based navigation, this paper is the first to explore mapless navigation in the embodied AI field, in particular, to utilize object-goal navigation (commonly abbreviated as ON, ObjNav, or OGN) techniques that efficiently explore target objects without using such a prior map. Specifically, in this work, we start from an off-the-shelf mapless ON planner, extend it to utilize a prior map, and further show that the performance in long-distance ALC (LD-ALC) can be maximized by minimizing ``ALC loss" and ``ON loss". This study highlights a simple and effective approach, called ALC-ON (ALCON), to accelerate the progress of challenging long-distance ALC technology by leveraging the growing frontier-guided, data-driven, and LLM-guided ON technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07634v2">Neural Brain: A Neuroscience-inspired Framework for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 51 pages, 17 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09694v1">EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 Website: https://github.com/AgibotTech/EWMBench
    </div>
    <details class="paper-abstract">
      Recent advances in creative AI have enabled the synthesis of high-fidelity images and videos conditioned on language instructions. Building on these developments, text-to-video diffusion models have evolved into embodied world models (EWMs) capable of generating physically plausible scenes from language commands, effectively bridging vision and action in embodied AI applications. This work addresses the critical challenge of evaluating EWMs beyond general perceptual metrics to ensure the generation of physically grounded and action-consistent behaviors. We propose the Embodied World Model Benchmark (EWMBench), a dedicated framework designed to evaluate EWMs based on three key aspects: visual scene consistency, motion correctness, and semantic alignment. Our approach leverages a meticulously curated dataset encompassing diverse scenes and motion patterns, alongside a comprehensive multi-dimensional evaluation toolkit, to assess and compare candidate models. The proposed benchmark not only identifies the limitations of existing video generation models in meeting the unique requirements of embodied tasks but also provides valuable insights to guide future advancements in the field. The dataset and evaluation tools are publicly available at https://github.com/AgibotTech/EWMBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08854v1">Generative AI for Autonomous Driving: Frontiers and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at https://github.com/taco-group/GenAI4AD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11528v1">LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07634v1">Neural Brain: A Neuroscience-inspired Framework for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 51 pages, 17 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07532v1">RAI: Flexible Agent Framework for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 12 pages, 8 figures, submitted to 23rd International Conference on Practical applications of Agents and Multi-Agent Systems (PAAMS'25)
    </div>
    <details class="paper-abstract">
      With an increase in the capabilities of generative language models, a growing interest in embodied AI has followed. This contribution introduces RAI - a framework for creating embodied Multi Agent Systems for robotics. The proposed framework implements tools for Agents' integration with robotic stacks, Large Language Models, and simulations. It provides out-of-the-box integration with state-of-the-art systems like ROS 2. It also comes with dedicated mechanisms for the embodiment of Agents. These mechanisms have been tested on a physical robot, Husarion ROSBot XL, which was coupled with its digital twin, for rapid prototyping. Furthermore, these mechanisms have been deployed in two simulations: (1) robot arm manipulator and (2) tractor controller. All of these deployments have been evaluated in terms of their control capabilities, effectiveness of embodiment, and perception ability. The proposed framework has been used successfully to build systems with multiple agents. It has demonstrated effectiveness in all the aforementioned tasks. It also enabled identifying and addressing the shortcomings of the generative models used for embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07460v1">A Survey on Collaborative Mechanisms Between Large and Small Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consistency, robust task allocation, evaluation complexity, and security/privacy concerns. Future directions point towards more intelligent adaptive frameworks, deeper model fusion, and expansion into multimodal and embodied AI, positioning LLM-SLM collaboration as a key driver for the next generation of practical and ubiquitous artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07446v1">TPT-Bench: A Large-Scale, Long-Term and Robot-Egocentric Dataset for Benchmarking Target Person Tracking</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Under review. web: https://medlartea.github.io/tpt-bench/
    </div>
    <details class="paper-abstract">
      Tracking a target person from robot-egocentric views is crucial for developing autonomous robots that provide continuous personalized assistance or collaboration in Human-Robot Interaction (HRI) and Embodied AI. However, most existing target person tracking (TPT) benchmarks are limited to controlled laboratory environments with few distractions, clean backgrounds, and short-term occlusions. In this paper, we introduce a large-scale dataset designed for TPT in crowded and unstructured environments, demonstrated through a robot-person following task. The dataset is collected by a human pushing a sensor-equipped cart while following a target person, capturing human-like following behavior and emphasizing long-term tracking challenges, including frequent occlusions and the need for re-identification from numerous pedestrians. It includes multi-modal data streams, including odometry, 3D LiDAR, IMU, panoptic, and RGB-D images, along with exhaustively annotated 2D bounding boxes of the target person across 35 sequences, both indoors and outdoors. Using this dataset and visual annotations, we perform extensive experiments with existing TPT methods, offering a thorough analysis of their limitations and suggesting future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09587v3">GeoNav: Empowering MLLMs with Explicit Geospatial Reasoning Abilities for Language-Goal Aerial Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Language-goal aerial navigation is a critical challenge in embodied AI, requiring UAVs to localize targets in complex environments such as urban blocks based on textual specification. Existing methods, often adapted from indoor navigation, struggle to scale due to limited field of view, semantic ambiguity among objects, and lack of structured spatial reasoning. In this work, we propose GeoNav, a geospatially aware multimodal agent to enable long-range navigation. GeoNav operates in three phases-landmark navigation, target search, and precise localization-mimicking human coarse-to-fine spatial strategies. To support such reasoning, it dynamically builds two different types of spatial memory. The first is a global but schematic cognitive map, which fuses prior textual geographic knowledge and embodied visual cues into a top-down, annotated form for fast navigation to the landmark region. The second is a local but delicate scene graph representing hierarchical spatial relationships between blocks, landmarks, and objects, which is used for definite target localization. On top of this structured representation, GeoNav employs a spatially aware, multimodal chain-of-thought prompting mechanism to enable multimodal large language models with efficient and interpretable decision-making across stages. On the CityNav urban navigation benchmark, GeoNav surpasses the current state-of-the-art by up to 12.53% in success rate and significantly improves navigation efficiency, even in hard-level tasks. Ablation studies highlight the importance of each module, showcasing how geospatial representations and coarse-to-fine reasoning enhance UAV navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15764v2">Towards Agentic AI Networking in 6G: A Generative Foundation Model-as-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Accepted at IEEE Communications Magazine
    </div>
    <details class="paper-abstract">
      The promising potential of AI and network convergence in improving networking performance and enabling new service capabilities has recently attracted significant interest. Existing network AI solutions, while powerful, are mainly built based on the close-loop and passive learning framework, resulting in major limitations in autonomous solution finding and dynamic environmental adaptation. Agentic AI has recently been introduced as a promising solution to address the above limitations and pave the way for true generally intelligent and beneficial AI systems. The key idea is to create a networking ecosystem to support a diverse range of autonomous and embodied AI agents in fulfilling their goals. In this paper, we focus on the novel challenges and requirements of agentic AI networking. We propose AgentNet, a novel framework for supporting interaction, collaborative learning, and knowledge transfer among AI agents. We introduce a general architectural framework of AgentNet and then propose a generative foundation model (GFM)-based implementation in which multiple GFM-as-agents have been created as an interactive knowledge-base to bootstrap the development of embodied AI agents according to different task requirements and environmental features. We consider two application scenarios, digital-twin-based industrial automation and metaverse-based infotainment system, to describe how to apply AgentNet for supporting efficient task-driven collaboration and interaction among AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06628v1">ACORN: Adaptive Contrastive Optimization for Safe and Robust Fine-Grained Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-05-10
      | 💬 6 pages,4 figures
    </div>
    <details class="paper-abstract">
      Embodied AI research has traditionally emphasized performance metrics such as success rate and cumulative reward, overlooking critical robustness and safety considerations that emerge during real-world deployment. In actual environments, agents continuously encounter unpredicted situations and distribution shifts, causing seemingly reliable policies to experience catastrophic failures, particularly in manipulation tasks. To address this gap, we introduce four novel safety-centric metrics that quantify an agent's resilience to environmental perturbations. Building on these metrics, we present Adaptive Contrastive Optimization for Robust Manipulation (ACORN), a plug-and-play algorithm that enhances policy robustness without sacrificing performance. ACORN leverages contrastive learning to simultaneously align trajectories with expert demonstrations while diverging from potentially unsafe behaviors. Our approach efficiently generates informative negative samples through structured Gaussian noise injection, employing a double perturbation technique that maintains sample diversity while minimizing computational overhead. Comprehensive experiments across diverse manipulation environments validate ACORN's effectiveness, yielding improvements of up to 23% in safety metrics under disturbance compared to baseline methods. These findings underscore ACORN's significant potential for enabling reliable deployment of embodied agents in safety-critical real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06575v1">GRACE: Estimating Geometry-level 3D Human-Scene Contact from 2D Images</a></div>
    <div class="paper-meta">
      📅 2025-05-10
    </div>
    <details class="paper-abstract">
      Estimating the geometry level of human-scene contact aims to ground specific contact surface points at 3D human geometries, which provides a spatial prior and bridges the interaction between human and scene, supporting applications such as human behavior analysis, embodied AI, and AR/VR. To complete the task, existing approaches predominantly rely on parametric human models (e.g., SMPL), which establish correspondences between images and contact regions through fixed SMPL vertex sequences. This actually completes the mapping from image features to an ordered sequence. However, this approach lacks consideration of geometry, limiting its generalizability in distinct human geometries. In this paper, we introduce GRACE (Geometry-level Reasoning for 3D Human-scene Contact Estimation), a new paradigm for 3D human contact estimation. GRACE incorporates a point cloud encoder-decoder architecture along with a hierarchical feature extraction and fusion module, enabling the effective integration of 3D human geometric structures with 2D interaction semantics derived from images. Guided by visual cues, GRACE establishes an implicit mapping from geometric features to the vertex space of the 3D human mesh, thereby achieving accurate modeling of contact regions. This design ensures high prediction accuracy and endows the framework with strong generalization capability across diverse human geometries. Extensive experiments on multiple benchmark datasets demonstrate that GRACE achieves state-of-the-art performance in contact estimation, with additional results further validating its robust generalization to unstructured human point clouds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06378v1">Bi-LSTM based Multi-Agent DRL with Computation-aware Pruning for Agent Twins Migration in Vehicular Embodied AI Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-09
    </div>
    <details class="paper-abstract">
      With the advancement of large language models and embodied Artificial Intelligence (AI) in the intelligent transportation scenarios, the combination of them in intelligent transportation spawns the Vehicular Embodied AI Network (VEANs). In VEANs, Autonomous Vehicles (AVs) are typical agents whose local advanced AI applications are defined as vehicular embodied AI agents, enabling capabilities such as environment perception and multi-agent collaboration. Due to computation latency and resource constraints, the local AI applications and services running on vehicular embodied AI agents need to be migrated, and subsequently referred to as vehicular embodied AI agent twins, which drive the advancement of vehicular embodied AI networks to offload intensive tasks to Roadside Units (RSUs), mitigating latency problems while maintaining service quality. Recognizing workload imbalance among RSUs in traditional approaches, we model AV-RSU interactions as a Stackelberg game to optimize bandwidth resource allocation for efficient migration. A Tiny Multi-Agent Bidirectional LSTM Proximal Policy Optimization (TMABLPPO) algorithm is designed to approximate the Stackelberg equilibrium through decentralized coordination. Furthermore, a personalized neural network pruning algorithm based on Path eXclusion (PX) dynamically adapts to heterogeneous AV computation capabilities by identifying task-critical parameters in trained models, reducing model complexity with less performance degradation. Experimental validation confirms the algorithm's effectiveness in balancing system load and minimizing delays, demonstrating significant improvements in vehicular embodied AI agent deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05474v1">3D Scene Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Project Page: https://github.com/hzxie/Awesome-3D-Scene-Generation
    </div>
    <details class="paper-abstract">
      3D scene generation seeks to synthesize spatially structured, semantically meaningful, and photorealistic environments for applications such as immersive media, robotics, autonomous driving, and embodied AI. Early methods based on procedural rules offered scalability but limited diversity. Recent advances in deep generative models (e.g., GANs, diffusion models) and 3D representations (e.g., NeRF, 3D Gaussians) have enabled the learning of real-world scene distributions, improving fidelity, diversity, and view consistency. Recent advances like diffusion models bridge 3D scene synthesis and photorealism by reframing generation as image or video synthesis problems. This survey provides a systematic overview of state-of-the-art approaches, organizing them into four paradigms: procedural generation, neural 3D-based generation, image-based generation, and video-based generation. We analyze their technical foundations, trade-offs, and representative results, and review commonly used datasets, evaluation protocols, and downstream applications. We conclude by discussing key challenges in generation capacity, 3D representation, data and annotations, and evaluation, and outline promising directions including higher fidelity, physics-aware and interactive generation, and unified perception-generation models. This review organizes recent advances in 3D scene generation and highlights promising directions at the intersection of generative AI, 3D vision, and embodied intelligence. To track ongoing developments, we maintain an up-to-date project page: https://github.com/hzxie/Awesome-3D-Scene-Generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05456v1">SITE: towards Spatial Intelligence Thorough Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Spatial intelligence (SI) represents a cognitive ability encompassing the visualization, manipulation, and reasoning about spatial relationships, underpinning disciplines from neuroscience to robotics. We introduce SITE, a benchmark dataset towards SI Thorough Evaluation in a standardized format of multi-choice visual question-answering, designed to assess large vision-language models' spatial intelligence across diverse visual modalities (single-image, multi-image, and video) and SI factors (figural to environmental scales, spatial visualization and orientation, intrinsic and extrinsic, static and dynamic). Our approach to curating the benchmark combines a bottom-up survey about 31 existing datasets and a top-down strategy drawing upon three classification systems in cognitive science, which prompt us to design two novel types of tasks about view-taking and dynamic scenes. Extensive experiments reveal that leading models fall behind human experts especially in spatial orientation, a fundamental SI factor. Moreover, we demonstrate a positive correlation between a model's spatial reasoning proficiency and its performance on an embodied AI task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05108v1">Multi-agent Embodied AI: Advances and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Embodied artificial intelligence (Embodied AI) plays a pivotal role in the application of advanced technologies in the intelligent era, where AI systems are integrated with physical bodies that enable them to perceive, reason, and interact with their environments. Through the use of sensors for input and actuators for action, these systems can learn and adapt based on real-world feedback, allowing them to perform tasks effectively in dynamic and unpredictable environments. As techniques such as deep learning (DL), reinforcement learning (RL), and large language models (LLMs) mature, embodied AI has become a leading field in both academia and industry, with applications spanning robotics, healthcare, transportation, and manufacturing. However, most research has focused on single-agent systems that often assume static, closed environments, whereas real-world embodied AI must navigate far more complex scenarios. In such settings, agents must not only interact with their surroundings but also collaborate with other agents, necessitating sophisticated mechanisms for adaptation, real-time learning, and collaborative problem-solving. Despite increasing interest in multi-agent systems, existing research remains narrow in scope, often relying on simplified models that fail to capture the full complexity of dynamic, open environments for multi-agent embodied AI. Moreover, no comprehensive survey has systematically reviewed the advancements in this area. As embodied AI rapidly evolves, it is crucial to deepen our understanding of multi-agent embodied AI to address the challenges presented by real-world applications. To fill this gap and foster further development in the field, this paper reviews the current state of research, analyzes key contributions, and identifies challenges and future directions, providing insights to guide innovation and progress in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05622v1">CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Aerial vision-and-language navigation (VLN), requiring drones to interpret natural language instructions and navigate complex urban environments, emerges as a critical embodied AI challenge that bridges human-robot interaction, 3D spatial reasoning, and real-world deployment. Although existing ground VLN agents achieved notable results in indoor and outdoor settings, they struggle in aerial VLN due to the absence of predefined navigation graphs and the exponentially expanding action space in long-horizon exploration. In this work, we propose \textbf{CityNavAgent}, a large language model (LLM)-empowered agent that significantly reduces the navigation complexity for urban aerial VLN. Specifically, we design a hierarchical semantic planning module (HSPM) that decomposes the long-horizon task into sub-goals with different semantic levels. The agent reaches the target progressively by achieving sub-goals with different capacities of the LLM. Additionally, a global memory module storing historical trajectories into a topological graph is developed to simplify navigation for visited targets. Extensive benchmark experiments show that our method achieves state-of-the-art performance with significant improvement. Further experiments demonstrate the effectiveness of different modules of CityNavAgent for aerial VLN in continuous city environments. The code is available at \href{https://github.com/VinceOuti/CityNavAgent}{link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03238v1">RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Future robotic systems operating in real-world environments will require on-board embodied intelligence without continuous cloud connection, balancing capabilities with constraints on computational power and memory. This work presents an extension of the R1-zero approach, which enables the usage of low parameter-count Large Language Models (LLMs) in the robotic domain. The R1-Zero approach was originally developed to enable mathematical reasoning in LLMs using static datasets. We extend it to the robotics domain through integration in a closed-loop Reinforcement Learning (RL) framework. This extension enhances reasoning in Embodied Artificial Intelligence (Embodied AI) settings without relying solely on distillation of large models through Supervised Fine-Tuning (SFT). We show that small-scale LLMs can achieve effective reasoning performance by learning through closed-loop interaction with their environment, which enables tasks that previously required significantly larger models. In an autonomous driving setting, a performance gain of 20.2%-points over the SFT-based baseline is observed with a Qwen2.5-1.5B model. Using the proposed training procedure, Qwen2.5-3B achieves a 63.3% control adaptability score, surpassing the 58.5% obtained by the much larger, cloud-bound GPT-4o. These results highlight that practical, on-board deployment of small LLMs is not only feasible but can outperform larger models if trained through environmental feedback, underscoring the importance of an interactive learning framework for robotic Embodied AI, one grounded in practical experience rather than static supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02836v1">Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Synthesizing interactive 3D scenes from text is essential for gaming, virtual reality, and embodied AI. However, existing methods face several challenges. Learning-based approaches depend on small-scale indoor datasets, limiting the scene diversity and layout complexity. While large language models (LLMs) can leverage diverse text-domain knowledge, they struggle with spatial realism, often producing unnatural object placements that fail to respect common sense. Our key insight is that vision perception can bridge this gap by providing realistic spatial guidance that LLMs lack. To this end, we introduce Scenethesis, a training-free agentic framework that integrates LLM-based scene planning with vision-guided layout refinement. Given a text prompt, Scenethesis first employs an LLM to draft a coarse layout. A vision module then refines it by generating an image guidance and extracting scene structure to capture inter-object relations. Next, an optimization module iteratively enforces accurate pose alignment and physical plausibility, preventing artifacts like object penetration and instability. Finally, a judge module verifies spatial coherence. Comprehensive experiments show that Scenethesis generates diverse, realistic, and physically plausible 3D interactive scenes, making it valuable for virtual content creation, simulation environments, and embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02388v1">MetaScenes: Towards Automated Replica Creation for Real-world 3D Scans</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Embodied AI (EAI) research requires high-quality, diverse 3D scenes to effectively support skill acquisition, sim-to-real transfer, and generalization. Achieving these quality standards, however, necessitates the precise replication of real-world object diversity. Existing datasets demonstrate that this process heavily relies on artist-driven designs, which demand substantial human effort and present significant scalability challenges. To scalably produce realistic and interactive 3D scenes, we first present MetaScenes, a large-scale, simulatable 3D scene dataset constructed from real-world scans, which includes 15366 objects spanning 831 fine-grained categories. Then, we introduce Scan2Sim, a robust multi-modal alignment model, which enables the automated, high-quality replacement of assets, thereby eliminating the reliance on artist-driven designs for scaling 3D scenes. We further propose two benchmarks to evaluate MetaScenes: a detailed scene synthesis task focused on small item layouts for robotic manipulation and a domain transfer task in vision-and-language navigation (VLN) to validate cross-domain transfer. Results confirm MetaScene's potential to enhance EAI by supporting more generalizable agent learning and sim-to-real applications, introducing new possibilities for EAI research. Project website: https://meta-scenes.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00935v1">Autonomous Embodied Agents: When Robotics Meets Deep Learning Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-02
      | 💬 Ph.D. Dissertation
    </div>
    <details class="paper-abstract">
      The increase in available computing power and the Deep Learning revolution have allowed the exploration of new topics and frontiers in Artificial Intelligence research. A new field called Embodied Artificial Intelligence, which places at the intersection of Computer Vision, Robotics, and Decision Making, has been gaining importance during the last few years, as it aims to foster the development of smart autonomous robots and their deployment in society. The recent availability of large collections of 3D models for photorealistic robotic simulation has allowed faster and safe training of learning-based agents for millions of frames and a careful evaluation of their behavior before deploying the models on real robotic platforms. These intelligent agents are intended to perform a certain task in a possibly unknown environment. To this end, during the training in simulation, the agents learn to perform continuous interactions with the surroundings, such as gathering information from the environment, encoding and extracting useful cues for the task, and performing actions towards the final goal; where every action of the agent influences the interactions. This dissertation follows the complete creation process of embodied agents for indoor environments, from their concept to their implementation and deployment. We aim to contribute to research in Embodied AI and autonomous agents, in order to foster future work in this field. We present a detailed analysis of the procedure behind implementing an intelligent embodied agent, comprehending a thorough description of the current state-of-the-art in literature, technical explanations of the proposed methods, and accurate experimental studies on relevant robotic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01458v1">A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-05-01
    </div>
    <details class="paper-abstract">
      Navigation and manipulation are core capabilities in Embodied AI, yet training agents with these capabilities in the real world faces high costs and time complexity. Therefore, sim-to-real transfer has emerged as a key approach, yet the sim-to-real gap persists. This survey examines how physics simulators address this gap by analyzing their properties overlooked in previous surveys. We also analyze their features for navigation and manipulation tasks, along with hardware requirements. Additionally, we offer a resource with benchmark datasets, metrics, simulation platforms, and cutting-edge methods-such as world models and geometric equivariance-to help researchers select suitable tools while accounting for hardware constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03792v1">Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-01
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at https://github.com/langfengQ/CoSo.
    </details>
</div>
