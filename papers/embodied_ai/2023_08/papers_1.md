# embodied ai - 2023_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.10455v3">Towards Generalist Robots: A Promising Paradigm via Generative Simulation</a></div>
    <div class="paper-meta">
      📅 2023-08-30
    </div>
    <details class="paper-abstract">
      This document serves as a position paper that outlines the authors' vision for a potential pathway towards generalist robots. The purpose of this document is to share the excitement of the authors with the community and highlight a promising research direction in robotics and AI. The authors believe the proposed paradigm is a feasible path towards accomplishing the long-standing goal of robotics research: deploying robots, or embodied AI agents more broadly, in various non-factory real-world settings to perform diverse tasks. This document presents a specific idea for mining knowledge in the latest large-scale foundation models for robotics research. Instead of directly using or adapting these models to produce low-level policies and actions, it advocates for a fully automated generative pipeline (termed as generative simulation), which uses these models to generate diversified tasks, scenes and training supervisions at scale, thereby scaling up low-level skill learning and ultimately leading to a foundation model for robotics that empowers generalist robots. The authors are actively pursuing this direction, but in the meantime, they recognize that the ambitious goal of building generalist robots with large-scale policy training demands significant resources such as computing power and hardware, and research groups in academia alone may face severe resource constraints in implementing the entire vision. Therefore, the authors believe sharing their thoughts at this early stage could foster discussions, attract interest towards the proposed pathway and related topics from industry groups, and potentially spur significant technical advancements in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.02710v3">Beyond Object Recognition: A New Benchmark towards Object Concept Learning</a></div>
    <div class="paper-meta">
      📅 2023-08-20
      | 💬 ICCV 2023. Webpage: https://mvig-rhos.com/ocl
    </div>
    <details class="paper-abstract">
      Understanding objects is a central building block of artificial intelligence, especially for embodied AI. Even though object recognition excels with deep learning, current machines still struggle to learn higher-level knowledge, e.g., what attributes an object has, and what can we do with an object. In this work, we propose a challenging Object Concept Learning (OCL) task to push the envelope of object understanding. It requires machines to reason out object affordances and simultaneously give the reason: what attributes make an object possesses these affordances. To support OCL, we build a densely annotated knowledge base including extensive labels for three levels of object concept (category, attribute, affordance), and the causal relations of three levels. By analyzing the causal structure of OCL, we present a baseline, Object Concept Reasoning Network (OCRN). It leverages causal intervention and concept instantiation to infer the three levels following their causal relations. In experiments, OCRN effectively infers the object knowledge while following the causalities well. Our data and code are available at https://mvig-rhos.com/ocl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.05221v1">Alexa, play with robot: Introducing the First Alexa Prize SimBot Challenge on Embodied AI</a></div>
    <div class="paper-meta">
      📅 2023-08-09
    </div>
    <details class="paper-abstract">
      The Alexa Prize program has empowered numerous university students to explore, experiment, and showcase their talents in building conversational agents through challenges like the SocialBot Grand Challenge and the TaskBot Challenge. As conversational agents increasingly appear in multimodal and embodied contexts, it is important to explore the affordances of conversational interaction augmented with computer vision and physical embodiment. This paper describes the SimBot Challenge, a new challenge in which university teams compete to build robot assistants that complete tasks in a simulated physical environment. This paper provides an overview of the SimBot Challenge, which included both online and offline challenge phases. We describe the infrastructure and support provided to the teams including Alexa Arena, the simulated environment, and the ML toolkit provided to teams to accelerate their building of vision and language models. We summarize the approaches the participating teams took to overcome research challenges and extract key lessons learned. Finally, we provide analysis of the performance of the competing SimBots during the competition.
    </details>
</div>
