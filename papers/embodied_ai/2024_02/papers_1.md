# embodied ai - 2024_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2403.00833v1">Position Paper: Agent AI Towards a Holistic Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-02-28
      | 💬 22 pages, 4 figures. arXiv admin note: substantial text overlap with arXiv:2401.03568
    </div>
    <details class="paper-abstract">
      Recent advancements in large foundation models have remarkably enhanced our understanding of sensory information in open-world environments. In leveraging the power of foundation models, it is crucial for AI research to pivot away from excessive reductionism and toward an emphasis on systems that function as cohesive wholes. Specifically, we emphasize developing Agent AI -- an embodied system that integrates large foundation models into agent actions. The emerging field of Agent AI spans a wide range of existing embodied and agent-based multimodal interactions, including robotics, gaming, and healthcare systems, etc. In this paper, we propose a novel large action model to achieve embodied intelligent behavior, the Agent Foundation Model. On top of this idea, we discuss how agent AI exhibits remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. Furthermore, we discuss the potential of Agent AI from an interdisciplinary perspective, underscoring AI cognition and consciousness within scientific discourse. We believe that those discussions serve as a basis for future research directions and encourage broader societal engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16117v1">RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-02-25
    </div>
    <details class="paper-abstract">
      Robotic behavior synthesis, the problem of understanding multimodal inputs and generating precise physical control for robots, is an important part of Embodied AI. Despite successes in applying multimodal large language models for high-level understanding, it remains challenging to translate these conceptual understandings into detailed robotic actions while achieving generalization across various scenarios. In this paper, we propose a tree-structured multimodal code generation framework for generalized robotic behavior synthesis, termed RoboCodeX. RoboCodeX decomposes high-level human instructions into multiple object-centric manipulation units consisting of physical preferences such as affordance and safety constraints, and applies code generation to introduce generalization ability across various robotics platforms. To further enhance the capability to map conceptual and perceptual understanding into control commands, a specialized multimodal reasoning dataset is collected for pre-training and an iterative self-updating methodology is introduced for supervised fine-tuning. Extensive experiments demonstrate that RoboCodeX achieves state-of-the-art performance in both simulators and real robots on four different kinds of manipulation tasks and one navigation task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14623v1">RoboScript: Code Generation for Free-Form Manipulation Tasks across Real and Simulation</a></div>
    <div class="paper-meta">
      📅 2024-02-22
      | 💬 10 pages of main paper, 4 pages of appendix; 10 figures in main paper, 3 figures in appendix
    </div>
    <details class="paper-abstract">
      Rapid progress in high-level task planning and code generation for open-world robot manipulation has been witnessed in Embodied AI. However, previous studies put much effort into general common sense reasoning and task planning capabilities of large-scale language or multi-modal models, relatively little effort on ensuring the deployability of generated code on real robots, and other fundamental components of autonomous robot systems including robot perception, motion planning, and control. To bridge this ``ideal-to-real'' gap, this paper presents \textbf{RobotScript}, a platform for 1) a deployable robot manipulation pipeline powered by code generation; and 2) a code generation benchmark for robot manipulation tasks in free-form natural language. The RobotScript platform addresses this gap by emphasizing the unified interface with both simulation and real robots, based on abstraction from the Robot Operating System (ROS), ensuring syntax compliance and simulation validation with Gazebo. We demonstrate the adaptability of our code generation framework across multiple robot embodiments, including the Franka and UR5 robot arms, and multiple grippers. Additionally, our benchmark assesses reasoning abilities for physical space and constraints, highlighting the differences between GPT-3.5, GPT-4, and Gemini in handling complex physical interactions. Finally, we present a thorough evaluation on the whole system, exploring how each module in the pipeline: code generation, perception, motion planning, and even object geometric properties, impact the overall performance of the system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08837v1">Learning to Generate Context-Sensitive Backchannel Smiles for Embodied AI Agents with Applications in Mental Health Dialogues</a></div>
    <div class="paper-meta">
      📅 2024-02-13
      | 💬 Accepted to the Machine Learning for Cognitive and Mental Health Workshop at AAAI 2024
    </div>
    <details class="paper-abstract">
      Addressing the critical shortage of mental health resources for effective screening, diagnosis, and treatment remains a significant challenge. This scarcity underscores the need for innovative solutions, particularly in enhancing the accessibility and efficacy of therapeutic support. Embodied agents with advanced interactive capabilities emerge as a promising and cost-effective supplement to traditional caregiving methods. Crucial to these agents' effectiveness is their ability to simulate non-verbal behaviors, like backchannels, that are pivotal in establishing rapport and understanding in therapeutic contexts but remain under-explored. To improve the rapport-building capabilities of embodied agents we annotated backchannel smiles in videos of intimate face-to-face conversations over topics such as mental health, illness, and relationships. We hypothesized that both speaker and listener behaviors affect the duration and intensity of backchannel smiles. Using cues from speech prosody and language along with the demographics of the speaker and listener, we found them to contain significant predictors of the intensity of backchannel smiles. Based on our findings, we introduce backchannel smile production in embodied agents as a generation problem. Our attention-based generative model suggests that listener information offers performance improvements over the baseline speaker-centric generation approach. Conditioned generation using the significant predictors of smile intensity provides statistically significant improvements in empirical measures of generation quality. Our user study by transferring generated smiles to an embodied agent suggests that agent with backchannel smiles is perceived to be more human-like and is an attractive alternative for non-personal conversations over agent without backchannel smiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.02385v1">A Survey on Robotics with Foundation Models: toward Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      While the exploration for embodied AI has spanned multiple decades, it remains a persistent challenge to endow agents with human-level intelligence, including perception, learning, reasoning, decision-making, control, and generalization capabilities, so that they can perform general-purpose tasks in open, unstructured, and dynamic environments. Recent advances in computer vision, natural language processing, and multi-modality learning have shown that the foundation models have superhuman capabilities for specific tasks. They not only provide a solid cornerstone for integrating basic modules into embodied AI systems but also shed light on how to scale up robot learning from a methodological perspective. This survey aims to provide a comprehensive and up-to-date overview of foundation models in robotics, focusing on autonomous manipulation and encompassing high-level planning and low-level control. Moreover, we showcase their commonly used datasets, simulators, and benchmarks. Importantly, we emphasize the critical challenges intrinsic to this field and delineate potential avenues for future research, contributing to advancing the frontier of academic and industrial discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02385v1">A Survey on Robotics with Foundation Models: toward Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-02-04
    </div>
    <details class="paper-abstract">
      While the exploration for embodied AI has spanned multiple decades, it remains a persistent challenge to endow agents with human-level intelligence, including perception, learning, reasoning, decision-making, control, and generalization capabilities, so that they can perform general-purpose tasks in open, unstructured, and dynamic environments. Recent advances in computer vision, natural language processing, and multi-modality learning have shown that the foundation models have superhuman capabilities for specific tasks. They not only provide a solid cornerstone for integrating basic modules into embodied AI systems but also shed light on how to scale up robot learning from a methodological perspective. This survey aims to provide a comprehensive and up-to-date overview of foundation models in robotics, focusing on autonomous manipulation and encompassing high-level planning and low-level control. Moreover, we showcase their commonly used datasets, simulators, and benchmarks. Importantly, we emphasize the critical challenges intrinsic to this field and delineate potential avenues for future research, contributing to advancing the frontier of academic and industrial discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.02385v1">A Survey on Robotics with Foundation Models: toward Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-02-04
    </div>
    <details class="paper-abstract">
      While the exploration for embodied AI has spanned multiple decades, it remains a persistent challenge to endow agents with human-level intelligence, including perception, learning, reasoning, decision-making, control, and generalization capabilities, so that they can perform general-purpose tasks in open, unstructured, and dynamic environments. Recent advances in computer vision, natural language processing, and multi-modality learning have shown that the foundation models have superhuman capabilities for specific tasks. They not only provide a solid cornerstone for integrating basic modules into embodied AI systems but also shed light on how to scale up robot learning from a methodological perspective. This survey aims to provide a comprehensive and up-to-date overview of foundation models in robotics, focusing on autonomous manipulation and encompassing high-level planning and low-level control. Moreover, we showcase their commonly used datasets, simulators, and benchmarks. Importantly, we emphasize the critical challenges intrinsic to this field and delineate potential avenues for future research, contributing to advancing the frontier of academic and industrial discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.18240v2">Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence?</a></div>
    <div class="paper-meta">
      📅 2024-02-01
      | 💬 Project website: https://eai-vc.github.io
    </div>
    <details class="paper-abstract">
      We present the largest and most comprehensive empirical study of pre-trained visual representations (PVRs) or visual 'foundation models' for Embodied AI. First, we curate CortexBench, consisting of 17 different tasks spanning locomotion, navigation, dexterous, and mobile manipulation. Next, we systematically evaluate existing PVRs and find that none are universally dominant. To study the effect of pre-training data size and diversity, we combine over 4,000 hours of egocentric videos from 7 different sources (over 4.3M images) and ImageNet to train different-sized vision transformers using Masked Auto-Encoding (MAE) on slices of this data. Contrary to inferences from prior work, we find that scaling dataset size and diversity does not improve performance universally (but does so on average). Our largest model, named VC-1, outperforms all prior PVRs on average but does not universally dominate either. Next, we show that task- or domain-specific adaptation of VC-1 leads to substantial gains, with VC-1 (adapted) achieving competitive or superior performance than the best known results on all of the benchmarks in CortexBench. Finally, we present real-world hardware experiments, in which VC-1 and VC-1 (adapted) outperform the strongest pre-existing PVR. Overall, this paper presents no new techniques but a rigorous systematic evaluation, a broad set of findings about PVRs (that in some cases, refute those made in narrow domains in prior work), and open-sourced code and models (that required over 10,000 GPU-hours to train) for the benefit of the research community.
    </details>
</div>
