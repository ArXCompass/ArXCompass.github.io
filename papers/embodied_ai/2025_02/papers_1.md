# embodied ai - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

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
