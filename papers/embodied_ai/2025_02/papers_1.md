# embodied ai - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09278v1">ConsistentDreamer: View-Consistent Meshes Through Balanced Multi-View Gaussian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Manuscript accepted by Pattern Recognition Letters
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion models have significantly improved 3D generation, enabling the use of assets generated from an image for embodied AI simulations. However, the one-to-many nature of the image-to-3D problem limits their use due to inconsistent content and quality across views. Previous models optimize a 3D model by sampling views from a view-conditioned diffusion prior, but diffusion models cannot guarantee view consistency. Instead, we present ConsistentDreamer, where we first generate a set of fixed multi-view prior images and sample random views between them with another diffusion model through a score distillation sampling (SDS) loss. Thereby, we limit the discrepancies between the views guided by the SDS loss and ensure a consistent rough shape. In each iteration, we also use our generated multi-view prior images for fine-detail reconstruction. To balance between the rough shape and the fine-detail optimizations, we introduce dynamic task-dependent weights based on homoscedastic uncertainty, updated automatically in each iteration. Additionally, we employ opacity, depth distortion, and normal alignment losses to refine the surface for mesh extraction. Our method ensures better view consistency and visual quality compared to the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09680v1">Object-Centric Latent Action Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-13
      | 💬 Preprint. In review
    </div>
    <details class="paper-abstract">
      Leveraging vast amounts of internet video data for Embodied AI is currently bottle-necked by the lack of action annotations and the presence of action-correlated distractors. We propose a novel object-centric latent action learning approach, based on VideoSaur and LAPO, that employs self-supervised decomposition of scenes into object representations and annotates video data with proxy-action labels. This method effectively disentangles causal agent-object interactions from irrelevant background noise and reduces the performance degradation of latent action learning approaches caused by distractors. Our preliminary experiments with the Distracting Control Suite show that latent action pretraining based on object decompositions improve the quality of inferred latent actions by x2.7 and efficiency of downstream fine-tuning with a small set of labeled actions, increasing return by x2.6 on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16633v2">POEX: Understanding and Mitigating Policy Executable Jailbreak Attacks against Embodied AI</a></div>
    <div class="paper-meta">
      📅 2025-02-10
      | 💬 Homepage: https://poex-eai-jailbreak.github.io/
    </div>
    <details class="paper-abstract">
      Embodied AI systems are rapidly evolving due to the integration of LLMs as planning modules, which transform complex instructions into executable policies. However, LLMs are vulnerable to jailbreak attacks, which can generate malicious content. This paper investigates the feasibility and rationale behind applying traditional LLM jailbreak attacks to EAI systems. We aim to answer three questions: (1) Do traditional LLM jailbreak attacks apply to EAI systems? (2) What challenges arise if they do not? and (3) How can we defend against EAI jailbreak attacks? To this end, we first measure existing LLM-based EAI systems using a newly constructed dataset, i.e., the Harmful-RLbench. Our study confirms that traditional LLM jailbreak attacks are not directly applicable to EAI systems and identifies two unique challenges. First, the harmful text does not necessarily constitute harmful policies. Second, even if harmful policies can be generated, they are not necessarily executable by the EAI systems, which limits the potential risk. To facilitate a more comprehensive security analysis, we refine and introduce POEX, a novel red teaming framework that optimizes adversarial suffixes to induce harmful yet executable policies against EAI systems. The design of POEX employs adversarial constraints, policy evaluators, and suffix optimization to ensure successful policy execution while evading safety detection inside an EAI system. Experiments on the real-world robotic arm and simulator using Harmful-RLbench demonstrate the efficacy, highlighting severe safety vulnerabilities and high transferability across models. Finally, we propose prompt-based and model-based defenses, achieving an 85% success rate in mitigating attacks and enhancing safety awareness in EAI systems. Our findings underscore the urgent need for robust security measures to ensure the safe deployment of EAI in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06838v1">TorchResist: Open-Source Differentiable Resist Simulator</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 SPIE Advanced Lithography + Patterning, 2025
    </div>
    <details class="paper-abstract">
      Recent decades have witnessed remarkable advancements in artificial intelligence (AI), including large language models (LLMs), image and video generative models, and embodied AI systems. These advancements have led to an explosive increase in the demand for computational power, challenging the limits of Moore's Law. Optical lithography, a critical technology in semiconductor manufacturing, faces significant challenges due to its high costs. To address this, various lithography simulators have been developed. However, many of these simulators are limited by their inadequate photoresist modeling capabilities. This paper presents TorchResist, an open-source, differentiable photoresist simulator.TorchResist employs an analytical approach to model the photoresist process, functioning as a white-box system with at most twenty interpretable parameters. Leveraging modern differentiable programming techniques and parallel computing on GPUs, TorchResist enables seamless co-optimization with other tools across multiple related tasks. Our experimental results demonstrate that TorchResist achieves superior accuracy and efficiency compared to existing solutions. The source code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20242v4">BadRobot: Jailbreaking Embodied LLMs in the Physical World</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to ICLR 2025. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01292v2">LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Research on 3D Vision-Language Models (3D-VLMs) is gaining increasing attention, which is crucial for developing embodied AI within 3D scenes, such as visual navigation and embodied question answering. Due to the high density of visual features, especially in large 3D scenes, accurately locating task-relevant visual information is challenging. Existing works attempt to segment all objects and consider their features as scene representations. However, these task-agnostic object features include much redundant information and missing details for the task-relevant area. To tackle these problems, we propose LSceneLLM, an adaptive framework that automatically identifies task-relevant areas by leveraging LLM's visual preference for different tasks, followed by a plug-and-play scene magnifier module to capture fine-grained details in focused areas. Specifically, a dense token selector examines the attention map of LLM to identify visual preferences for the instruction input. It then magnifies fine-grained details of the focusing area. An adaptive self-attention module is leveraged to fuse the coarse-grained and selected fine-grained visual information. To comprehensively evaluate the large scene understanding ability of 3D-VLMs, we further introduce a cross-room understanding benchmark, XR-Scene, which contains a series of large scene understanding tasks including XR-QA, XR-EmbodiedPlanning, and XR-SceneCaption. Experiments show that our method surpasses existing methods on both large scene understanding and existing scene understanding benchmarks. Plunging our scene magnifier module into the existing 3D-VLMs also brings significant improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00392v1">RefDrone: A Challenging Benchmark for Referring Expression Comprehension in Drone Scenes</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Drones have become prevalent robotic platforms with diverse applications, showing significant potential in Embodied Artificial Intelligence (Embodied AI). Referring Expression Comprehension (REC) enables drones to locate objects based on natural language expressions, a crucial capability for Embodied AI. Despite advances in REC for ground-level scenes, aerial views introduce unique challenges including varying viewpoints, occlusions and scale variations. To address this gap, we introduce RefDrone, a REC benchmark for drone scenes. RefDrone reveals three key challenges in REC: 1) multi-scale and small-scale target detection; 2) multi-target and no-target samples; 3) complex environment with rich contextual expressions. To efficiently construct this dataset, we develop RDAgent (referring drone annotation framework with multi-agent system), a semi-automated annotation tool for REC tasks. RDAgent ensures high-quality contextual expressions and reduces annotation cost. Furthermore, we propose Number GroundingDINO (NGDINO), a novel method designed to handle multi-target and no-target cases. NGDINO explicitly learns and utilizes the number of objects referred to in the expression. Comprehensive experiments with state-of-the-art REC methods demonstrate that NGDINO achieves superior performance on both the proposed RefDrone and the existing gRefCOCO datasets. The dataset and code will be publicly at https://github.com/sunzc-sunny/refdrone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00379v1">Latent Action Learning Requires Supervision in the Presence of Distractors</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Preprint. In review
    </div>
    <details class="paper-abstract">
      Recently, latent action learning, pioneered by Latent Action Policies (LAPO), have shown remarkable pre-training efficiency on observation-only data, offering potential for leveraging vast amounts of video available on the web for embodied AI. However, prior work has focused on distractor-free data, where changes between observations are primarily explained by ground-truth actions. Unfortunately, real-world videos contain action-correlated distractors that may hinder latent action learning. Using Distracting Control Suite (DCS) we empirically investigate the effect of distractors on latent action learning and demonstrate that LAPO struggle in such scenario. We propose LAOM, a simple LAPO modification that improves the quality of latent actions by 8x, as measured by linear probing. Importantly, we show that providing supervision with ground-truth actions, as few as 2.5% of the full dataset, during latent action learning improves downstream performance by 4.2x on average. Our findings suggest that integrating supervision during Latent Action Models (LAM) training is critical in the presence of distractors, challenging the conventional pipeline of first learning LAM and only then decoding from latent to ground-truth actions.
    </details>
</div>
