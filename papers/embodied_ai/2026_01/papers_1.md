# embodied ai - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11421v1">The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at https://rhos.ai/research/gm-100.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v2">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research. project page: https://anc891203.github.io/SceneFoundry-Demo/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v2">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-15
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09954v1">The Spatial Blindspot of Vision-Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-15
    </div>
    <details class="paper-abstract">
      Vision-language models (VLMs) have advanced rapidly, but their ability to capture spatial relationships remains a blindspot. Current VLMs are typically built with contrastive language-image pretraining (CLIP) style image encoders. The training recipe often flattens images into 1D patch sequences, discarding the 2D structure necessary for spatial reasoning. We argue that this lack of spatial awareness is a missing dimension in VLM design and a bottleneck for applications requiring spatial grounding, such as robotics and embodied AI. To address this, we investigate (i) image encoders trained with alternative objectives and (ii) 2D positional encodings. Our experiments show that these architectural choices can lead to improved spatial reasoning on several benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09697v1">Efficient Camera-Controlled Video Generation of Static Scenes via Sparse Diffusion and 3D Rendering</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Project page: https://ayushtewari.com/projects/srender/
    </div>
    <details class="paper-abstract">
      Modern video generative models based on diffusion models can produce very realistic clips, but they are computationally inefficient, often requiring minutes of GPU time for just a few seconds of video. This inefficiency poses a critical barrier to deploying generative video in applications that require real-time interactions, such as embodied AI and VR/AR. This paper explores a new strategy for camera-conditioned video generation of static scenes: using diffusion-based generative models to generate a sparse set of keyframes, and then synthesizing the full video through 3D reconstruction and rendering. By lifting keyframes into a 3D representation and rendering intermediate views, our approach amortizes the generation cost across hundreds of frames while enforcing geometric consistency. We further introduce a model that predicts the optimal number of keyframes for a given camera trajectory, allowing the system to adaptively allocate computation. Our final method, SRENDER, uses very sparse keyframes for simple trajectories and denser ones for complex camera motion. This results in video generation that is more than 40 times faster than the diffusion-based baseline in generating 20 seconds of video, while maintaining high visual fidelity and temporal stability, offering a practical path toward efficient and controllable video synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19789v4">What Can RL Bring to VLA Generalization? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Vision-Language Action (VLA) models have shown significant potential for embodied AI. However, their predominant training via supervised fine-tuning (SFT) limits generalization due to susceptibility to compounding errors under distribution shifts. Reinforcement learning (RL) offers a path to overcome these limitations by optimizing for task objectives via trial-and-error, yet a systematic understanding of its specific generalization benefits for VLAs compared to SFT is lacking. To address this, our study introduces a comprehensive benchmark for evaluating VLA generalization and systematically investigates the impact of RL fine-tuning across diverse visual, semantic, and execution dimensions. Our extensive experiments reveal that RL fine-tuning, particularly with PPO, significantly enhances generalization in semantic understanding and execution robustness over SFT, while maintaining comparable visual robustness. We identify PPO as a more effective RL algorithm for VLAs than LLM-derived methods like DPO and GRPO. We also develop a simple recipe for efficient PPO training on VLAs, and demonstrate its practical utility for improving VLA generalization. The project page is at https://rlvla.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08355v1">Semantic Misalignment in Vision-Language Models under Perceptual Degradation</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07553v1">VirtualEnv: A Platform for Embodied AI Research</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to improve in reasoning and decision-making, there is a growing need for realistic and interactive environments where their abilities can be rigorously evaluated. We present VirtualEnv, a next-generation simulation platform built on Unreal Engine 5 that enables fine-grained benchmarking of LLMs in embodied and interactive scenarios. VirtualEnv supports rich agent-environment interactions, including object manipulation, navigation, and adaptive multi-agent collaboration, as well as game-inspired mechanics like escape rooms and procedurally generated environments. We provide a user-friendly API built on top of Unreal Engine, allowing researchers to deploy and control LLM-driven agents using natural language instructions. We integrate large-scale LLMs and vision-language models (VLMs), such as GPT-based models, to generate novel environments and structured tasks from multimodal inputs. Our experiments benchmark the performance of several popular LLMs across tasks of increasing complexity, analyzing differences in adaptability, planning, and multi-agent coordination. We also describe our methodology for procedural task generation, task validation, and real-time environment control. VirtualEnv is released as an open-source platform, we aim to advance research at the intersection of AI and gaming, enable standardized evaluation of LLMs in embodied AI settings, and pave the way for future developments in immersive simulations and interactive entertainment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v2">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08876v1">The Semantic Lifecycle in Embodied AI: Acquisition, Representation and Storage via Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Semantic information in embodied AI is inherently multi-source and multi-stage, making it challenging to fully leverage for achieving stable perception-to-action loops in real-world environments. Early studies have combined manual engineering with deep neural networks, achieving notable progress in specific semantic-related embodied tasks. However, as embodied agents encounter increasingly complex environments and open-ended tasks, the demand for more generalizable and robust semantic processing capabilities has become imperative. Recent advances in foundation models (FMs) address this challenge through their cross-domain generalization abilities and rich semantic priors, reshaping the landscape of embodied AI research. In this survey, we propose the Semantic Lifecycle as a unified framework to characterize the evolution of semantic knowledge within embodied AI driven by foundation models. Departing from traditional paradigms that treat semantic processing as isolated modules or disjoint tasks, our framework offers a holistic perspective that captures the continuous flow and maintenance of semantic knowledge. Guided by this embodied semantic lifecycle, we further analyze and compare recent advances across three key stages: acquisition, representation, and storage. Finally, we summarize existing challenges and outline promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06573v1">QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Multimodal Models (LMMs) for video-audio understanding have traditionally been evaluated only on shorter videos of a few minutes long. In this paper, we introduce QMAVIS (Q Team-Multimodal Audio Video Intelligent Sensemaking), a novel long video-audio understanding pipeline built through a late fusion of LMMs, Large Language Models, and speech recognition models. QMAVIS addresses the gap in long-form video analytics, particularly for longer videos of a few minutes to beyond an hour long, opening up new potential applica- tions in sensemaking, video content analysis, embodied AI, etc. Quantitative experiments using QMAVIS demonstrated a 38.75% improvement over state-of-the-art video-audio LMMs like Vide- oLlaMA2 and InternVL2 on the VideoMME (with subtitles) dataset, which comprises long videos with audio information. Evaluations on other challenging video understanding datasets like PerceptionTest and EgoSchema saw up to 2% improvement, indicating competitive performance. Qualitative experiments also showed that QMAVIS is able to extract the nuances of different scenes in a long video audio content while understanding the overarching narrative. Ablation studies were also conducted to ascertain the impact of each component in the fusion pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05991v1">Open-Vocabulary 3D Instruction Ambiguity Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define Open-Vocabulary 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at https://jiayuding031020.github.io/ambi3d/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05810v1">SceneFoundry: Generating Interactive Infinite 3D Worlds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07855v1">An Empirical Study on Knowledge Transfer under Domain and Label Shifts in 3D LiDAR Point Clouds</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      For 3D perception systems to be practical in real-world applications -- from autonomous driving to embodied AI -- models must adapt to continuously evolving object definitions and sensor domains. Yet, research on continual and transfer learning in 3D point cloud perception remains underexplored compared to 2D vision -- particularly under simultaneous domain and label shifts. To address this gap, we propose the RObust Autonomous driving under Dataset shifts (ROAD) benchmark, a comprehensive evaluation suite for LiDAR-based object classification that explicitly accounts for domain shifts as well as three key forms of label evolution: class split, class expansion, and class insertion. Using large-scale datasets (Waymo, NuScenes, Argoverse2), we evaluate zero-shot transfer, linear probe, and CL, and analyze the impact of backbone architectures, training objectives, and CL methods. Our findings reveal limitations of existing approaches under realistic shifts and establish strong baselines for future research in robust 3D perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v2">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-08
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04137v1">Wow, wo, val! A Comprehensive Embodied World Model Evaluation Turing Test</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      As world models gain momentum in Embodied AI, an increasing number of works explore using video foundation models as predictive world models for downstream embodied tasks like 3D prediction or interactive generation. However, before exploring these downstream tasks, video foundation models still have two critical questions unanswered: (1) whether their generative generalization is sufficient to maintain perceptual fidelity in the eyes of human observers, and (2) whether they are robust enough to serve as a universal prior for real-world embodied agents. To provide a standardized framework for answering these questions, we introduce the Embodied Turing Test benchmark: WoW-World-Eval (Wow,wo,val). Building upon 609 robot manipulation data, Wow-wo-val examines five core abilities, including perception, planning, prediction, generalization, and execution. We propose a comprehensive evaluation protocol with 22 metrics to assess the models' generation ability, which achieves a high Pearson Correlation between the overall score and human preference (>0.93) and establishes a reliable foundation for the Human Turing Test. On Wow-wo-val, models achieve only 17.27 on long-horizon planning and at best 68.02 on physical consistency, indicating limited spatiotemporal consistency and physical reasoning. For the Inverse Dynamic Model Turing Test, we first use an IDM to evaluate the video foundation models' execution accuracy in the real world. However, most models collapse to $\approx$ 0% success, while WoW maintains a 40.74% success rate. These findings point to a noticeable gap between the generated videos and the real world, highlighting the urgency and necessity of benchmarking World Model in Embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04266v1">State Backdoor: Towards Stealthy Real-world Poisoning Attack on Vision-Language-Action Model in State Space</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models are widely deployed in safety-critical embodied AI applications such as robotics. However, their complex multimodal interactions also expose new security vulnerabilities. In this paper, we investigate a backdoor threat in VLA models, where malicious inputs cause targeted misbehavior while preserving performance on clean data. Existing backdoor methods predominantly rely on inserting visible triggers into visual modality, which suffer from poor robustness and low insusceptibility in real-world settings due to environmental variability. To overcome these limitations, we introduce the State Backdoor, a novel and practical backdoor attack that leverages the robot arm's initial state as the trigger. To optimize trigger for insusceptibility and effectiveness, we design a Preference-guided Genetic Algorithm (PGA) that efficiently searches the state space for minimal yet potent triggers. Extensive experiments on five representative VLA models and five real-world tasks show that our method achieves over 90% attack success rate without affecting benign task performance, revealing an underexplored vulnerability in embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03136v1">Limited Linguistic Diversity in Embodied AI Datasets</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16853v2">ISCS: Parameter-Guided Feature Pruning for Resource-Constrained Embodied Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Significant revision: The focus has been pivoted from learned image compression to embodied perception tasks. Experimental results and downstream applications have been updated to demonstrate the method's efficiency in split computing
    </div>
    <details class="paper-abstract">
      Prior studies in embodied AI consistently show that robust perception is critical for human-robot interaction, yet deploying high-fidelity visual models on resource-constrained agents remains challenging due to limited on-device computation power and transmission latency. Exploiting the redundancy in latent representations could improve system efficiency, yet existing approaches often rely on costly dataset-specific ablation tests or heavy entropy models unsuitable for real-time edge-robot collaboration. We propose a generalizable, dataset-agnostic method to identify and selectively transmit structure-critical channels in pretrained encoders. Instead of brute-force empirical evaluations, our approach leverages intrinsic parameter statistics-weight variances and biases-to estimate channel importance. This analysis reveals a consistent organizational structure, termed the Invariant Salient Channel Space (ISCS), where Salient-Core channels capture dominant structures while Salient-Auxiliary channels encode fine visual details. Building on ISCS, we introduce a deterministic static pruning strategy that enables lightweight split-computing. Experiments across different datasets demonstrate that our method achieves a deterministic, ultra-low latency pipeline by bypassing heavy entropy modeling. Our method reduces end-to-end latency, providing a critical speed-accuracy trade-off for resource-constrained human-aware embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03470v1">Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 5 pages, Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01705v1">Explicit World Models for Reliable Human-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
    </div>
    <details class="paper-abstract">
      This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10071v3">Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Post-challenge bug fix
    </div>
    <details class="paper-abstract">
      The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios. Project page: https://github.com/mli0603/openpi-comet
    </details>
</div>
