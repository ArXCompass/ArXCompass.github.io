# embodied ai - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05865v1">The Safety Challenge of World Models for Embodied AI Agents: A Review</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05684v1">D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07634v3">Neural Brain: A Neuroscience-inspired Framework for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 51 pages, 17 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04401v1">Your Vision-Language Model Can't Even Count to 20: Exposing the Failures of VLMs in Compositional Counting</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Vision-Language Models (VLMs) have become a central focus of today's AI community, owing to their impressive abilities gained from training on large-scale vision-language data from the Web. These models have demonstrated strong performance across diverse tasks, including image understanding, video understanding, complex visual reasoning, and embodied AI. Despite these noteworthy successes, a fundamental question remains: Can VLMs count objects correctly? In this paper, we introduce a simple yet effective benchmark, VLMCountBench, designed under a minimalist setting with only basic geometric shapes (e.g., triangles, circles) and their compositions, focusing exclusively on counting tasks without interference from other factors. We adopt strict independent variable control and systematically study the effects of simple properties such as color, size, and prompt refinement in a controlled ablation. Our empirical results reveal that while VLMs can count reliably when only one shape type is present, they exhibit substantial failures when multiple shape types are combined (i.e., compositional counting). This highlights a fundamental empirical limitation of current VLMs and motivates important directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03842v3">INGRID: Intelligent Generative Robotic Design Using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 We are revising it
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into robotic systems has accelerated progress in embodied artificial intelligence, yet current approaches remain constrained by existing robotic architectures, particularly serial mechanisms. This hardware dependency fundamentally limits the scope of robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic Design), a framework that enables the automated design of parallel robotic mechanisms through deep integration with reciprocal screw theory and kinematic synthesis methods. We decompose the design challenge into four progressive tasks: constraint analysis, kinematic joint generation, chain construction, and complete mechanism design. INGRID demonstrates the ability to generate novel parallel mechanisms with both fixed and variable mobility, discovering kinematic configurations not previously documented in the literature. We validate our approach through three case studies demonstrating how INGRID assists users in designing task-specific parallel robots based on desired mobility requirements. By bridging the gap between mechanism theory and machine learning, INGRID enables researchers without specialized robotics training to create custom parallel mechanisms, thereby decoupling advances in robotic intelligence from hardware constraints. This work establishes a foundation for mechanism intelligence, where AI systems actively design robotic hardware, potentially transforming the development of embodied AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03909v1">Generating Human Motion Videos using a Cascaded Text-to-Video Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 18 pages, 7 figures, Project Page:https://hyelinnam.github.io/Cameo/
    </div>
    <details class="paper-abstract">
      Human video generation is becoming an increasingly important task with broad applications in graphics, entertainment, and embodied AI. Despite the rapid progress of video diffusion models (VDMs), their use for general-purpose human video generation remains underexplored, with most works constrained to image-to-video setups or narrow domains like dance videos. In this work, we propose CAMEO, a cascaded framework for general human motion video generation. It seamlessly bridges Text-to-Motion (T2M) models and conditional VDMs, mitigating suboptimal factors that may arise in this process across both training and inference through carefully designed components. Specifically, we analyze and prepare both textual prompts and visual conditions to effectively train the VDM, ensuring robust alignment between motion descriptions, conditioning signals, and the generated videos. Furthermore, we introduce a camera-aware conditioning module that connects the two stages, automatically selecting viewpoints aligned with the input text to enhance coherence and reduce manual intervention. We demonstrate the effectiveness of our approach on both the MovieGen benchmark and a newly introduced benchmark tailored to the T2M-VDM combination, while highlighting its versatility across diverse use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03153v1">Improving Cooperation in Collaborative Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 In proceedings of UKCI 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02851v1">Action Deviation-Aware Inference for Low-Latency Wireless Robots</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML, connecting distributed computational resources in edge and cloud over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: an on-device draft model locally generates drafts and a remote server-based target model verifies and corrects them, resulting lower latency. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications like robot manipulation and autonomous driving, cannot parallelize verification and correction for multiple drafts as each action depends on observation which needs to be updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference, wherein the draft model estimates an action's need for verification and correction by the target model and selectively skips communication and computation for server operations. Action deviation shows a strong correlation with action's rejection probability by the target model, enabling selective skipping. We derive the path deviation threshold that balances the transmission rate and the inference performance, and we empirically show that action deviation-aware hybrid inference reduces uplink transmission and server operation by 40%, while lowering end-to-end latency by 33.32% relative to hybrid inference without skipping and achieving task success rate up to 97.03% of that of target model only inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01623v1">VLA-R1: Enhancing Reasoning in Vision-Language-Action Models</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Vision-Language-Action (VLA) models aim to unify perception, language understanding, and action generation, offering strong cross-task and cross-scene generalization with broad impact on embodied AI. However, current VLA models often lack explicit step-by-step reasoning, instead emitting final actions without considering affordance constraints or geometric relations. Their post-training pipelines also rarely reinforce reasoning quality, relying primarily on supervised fine-tuning with weak reward design. To address these challenges, we present VLA-R1, a reasoning-enhanced VLA that integrates Reinforcement Learning from Verifiable Rewards (RLVR) with Group Relative Policy Optimization (GRPO) to systematically optimize both reasoning and execution. Specifically, we design an RLVR-based post-training strategy with verifiable rewards for region alignment, trajectory consistency, and output formatting, thereby strengthening reasoning robustness and execution accuracy. Moreover, we develop VLA-CoT-13K, a high-quality dataset that provides chain-of-thought supervision explicitly aligned with affordance and trajectory annotations. Furthermore, extensive evaluations on in-domain, out-of-domain, simulation, and real-robot platforms demonstrate that VLA-R1 achieves superior generalization and real-world performance compared to prior VLA methods. We plan to release the model, code, and dataset following the publication of this work. Code: https://github.com/GigaAI-research/VLA-R1. Website: https://gigaai-research.github.io/VLA-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16928v2">Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00441v1">Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation</a></div>
    <div class="paper-meta">
      📅 2025-10-01
    </div>
    <details class="paper-abstract">
      Visual navigation is a fundamental problem in embodied AI, yet practical deployments demand long-horizon planning capabilities to address multi-objective tasks. A major bottleneck is data scarcity: policies learned from limited data often overfit and fail to generalize OOD. Existing neural network-based agents typically increase architectural complexity that paradoxically become counterproductive in the small-sample regime. This paper introduce NeuRO, a integrated learning-to-optimize framework that tightly couples perception networks with downstream task-level robust optimization. Specifically, NeuRO addresses core difficulties in this integration: (i) it transforms noisy visual predictions under data scarcity into convex uncertainty sets using Partially Input Convex Neural Networks (PICNNs) with conformal calibration, which directly parameterize the optimization constraints; and (ii) it reformulates planning under partial observability as a robust optimization problem, enabling uncertainty-aware policies that transfer across environments. Extensive experiments on both unordered and sequential multi-object navigation tasks demonstrate that NeuRO establishes SoTA performance, particularly in generalization to unseen environments. Our work thus presents a significant advancement for developing robust, generalizable autonomous agents.
    </details>
</div>
