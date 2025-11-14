# embodied ai - 2024_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.00956v2">Exploring Spatial Schema Intuitions in Large Language and Vision Models</a></div>
    <div class="paper-meta">
      📅 2024-05-28
      | 💬 ACL Findings 2024
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of large language models (LLMs) in AI research, the question of embodiment in LLMs remains underexplored, distinguishing them from embodied systems in robotics where sensory perception directly informs physical action. Our investigation navigates the intriguing terrain of whether LLMs, despite their non-embodied nature, effectively capture implicit human intuitions about fundamental, spatial building blocks of language. We employ insights from spatial cognitive foundations developed through early sensorimotor experiences, guiding our exploration through the reproduction of three psycholinguistic experiments. Surprisingly, correlations between model outputs and human responses emerge, revealing adaptability without a tangible connection to embodied experiences. Notable distinctions include polarized language model responses and reduced correlations in vision language models. This research contributes to a nuanced understanding of the interplay between language, spatial experiences, and the computations made by large language models. More at https://cisnlp.github.io/Spatial_Schemas/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13373v6">Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 Accepted and Published by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Recent studies have uncovered the potential of Large Language Models (LLMs) in addressing complex sequential decision-making tasks through the provision of high-level instructions. However, LLM-based agents lack specialization in tackling specific target problems, particularly in real-time dynamic environments. Additionally, deploying an LLM-based agent in practical scenarios can be both costly and time-consuming. On the other hand, reinforcement learning (RL) approaches train agents that specialize in the target task but often suffer from low sampling efficiency and high exploration costs. In this paper, we introduce a novel framework that addresses these challenges by training a smaller, specialized student RL agent using instructions from an LLM-based teacher agent. By incorporating the guidance from the teacher agent, the student agent can distill the prior knowledge of the LLM into its own model. Consequently, the student agent can be trained with significantly less data. Moreover, through further training with environment feedback, the student agent surpasses the capabilities of its teacher for completing the target task. We conducted experiments on challenging MiniGrid and Habitat environments, specifically designed for embodied AI research, to evaluate the effectiveness of our framework. The results clearly demonstrate that our approach achieves superior performance compared to strong baseline methods. Our code is available at https://github.com/ZJLAB-AMMI/LLM4Teach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.00956v2">Exploring Spatial Schema Intuitions in Large Language and Vision Models</a></div>
    <div class="paper-meta">
      📅 2024-05-27
      | 💬 ACL Findings 2024
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of large language models (LLMs) in AI research, the question of embodiment in LLMs remains underexplored, distinguishing them from embodied systems in robotics where sensory perception directly informs physical action. Our investigation navigates the intriguing terrain of whether LLMs, despite their non-embodied nature, effectively capture implicit human intuitions about fundamental, spatial building blocks of language. We employ insights from spatial cognitive foundations developed through early sensorimotor experiences, guiding our exploration through the reproduction of three psycholinguistic experiments. Surprisingly, correlations between model outputs and human responses emerge, revealing adaptability without a tangible connection to embodied experiences. Notable distinctions include polarized language model responses and reduced correlations in vision language models. This research contributes to a nuanced understanding of the interplay between language, spatial experiences, and the computations made by large language models. More at https://cisnlp.github.io/Spatial_Schemas/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09546v1">BEHAVIOR Vision Suite: Customizable Dataset Generation via Simulation</a></div>
    <div class="paper-meta">
      📅 2024-05-15
      | 💬 CVPR 2024 (Highlight). Project website: https://behavior-vision-suite.github.io/
    </div>
    <details class="paper-abstract">
      The systematic evaluation and understanding of computer vision models under varying conditions require large amounts of data with comprehensive and customized labels, which real-world vision datasets rarely satisfy. While current synthetic data generators offer a promising alternative, particularly for embodied AI tasks, they often fall short for computer vision tasks due to low asset and rendering quality, limited diversity, and unrealistic physical properties. We introduce the BEHAVIOR Vision Suite (BVS), a set of tools and assets to generate fully customized synthetic data for systematic evaluation of computer vision models, based on the newly developed embodied AI benchmark, BEHAVIOR-1K. BVS supports a large number of adjustable parameters at the scene level (e.g., lighting, object placement), the object level (e.g., joint configuration, attributes such as "filled" and "folded"), and the camera level (e.g., field of view, focal length). Researchers can arbitrarily vary these parameters during data generation to perform controlled experiments. We showcase three example application scenarios: systematically evaluating the robustness of models across different continuous axes of domain shift, evaluating scene understanding models on the same set of images, and training and evaluating simulation-to-real transfer for a novel vision task: unary and binary state prediction. Project website: https://behavior-vision-suite.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.03783v2">Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-05-14
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Embodied AI is one of the most popular studies in artificial intelligence and robotics, which can effectively improve the intelligence of real-world agents (i.e. robots) serving human beings. Scene knowledge is important for an agent to understand the surroundings and make correct decisions in the varied open world. Currently, knowledge base for embodied tasks is missing and most existing work use general knowledge base or pre-trained models to enhance the intelligence of an agent. For conventional knowledge base, it is sparse, insufficient in capacity and cost in data collection. For pre-trained models, they face the uncertainty of knowledge and hard maintenance. To overcome the challenges of scene knowledge, we propose a scene-driven multimodal knowledge graph (Scene-MMKG) construction method combining conventional knowledge engineering and large language models. A unified scene knowledge injection framework is introduced for knowledge representation. To evaluate the advantages of our proposed method, we instantiate Scene-MMKG considering typical indoor robotic functionalities (Manipulation and Mobility), named ManipMob-MMKG. Comparisons in characteristics indicate our instantiated ManipMob-MMKG has broad superiority in data-collection efficiency and knowledge quality. Experimental results on typical embodied tasks show that knowledge-enhanced methods using our instantiated ManipMob-MMKG can improve the performance obviously without re-designing model structures complexly. Our project can be found at https://sites.google.com/view/manipmob-mmkg
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03783v2">Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Embodied AI is one of the most popular studies in artificial intelligence and robotics, which can effectively improve the intelligence of real-world agents (i.e. robots) serving human beings. Scene knowledge is important for an agent to understand the surroundings and make correct decisions in the varied open world. Currently, knowledge base for embodied tasks is missing and most existing work use general knowledge base or pre-trained models to enhance the intelligence of an agent. For conventional knowledge base, it is sparse, insufficient in capacity and cost in data collection. For pre-trained models, they face the uncertainty of knowledge and hard maintenance. To overcome the challenges of scene knowledge, we propose a scene-driven multimodal knowledge graph (Scene-MMKG) construction method combining conventional knowledge engineering and large language models. A unified scene knowledge injection framework is introduced for knowledge representation. To evaluate the advantages of our proposed method, we instantiate Scene-MMKG considering typical indoor robotic functionalities (Manipulation and Mobility), named ManipMob-MMKG. Comparisons in characteristics indicate our instantiated ManipMob-MMKG has broad superiority in data-collection efficiency and knowledge quality. Experimental results on typical embodied tasks show that knowledge-enhanced methods using our instantiated ManipMob-MMKG can improve the performance obviously without re-designing model structures complexly. Our project can be found at https://sites.google.com/view/manipmob-mmkg
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.03783v2">Scene-Driven Multimodal Knowledge Graph Construction for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Embodied AI is one of the most popular studies in artificial intelligence and robotics, which can effectively improve the intelligence of real-world agents (i.e. robots) serving human beings. Scene knowledge is important for an agent to understand the surroundings and make correct decisions in the varied open world. Currently, knowledge base for embodied tasks is missing and most existing work use general knowledge base or pre-trained models to enhance the intelligence of an agent. For conventional knowledge base, it is sparse, insufficient in capacity and cost in data collection. For pre-trained models, they face the uncertainty of knowledge and hard maintenance. To overcome the challenges of scene knowledge, we propose a scene-driven multimodal knowledge graph (Scene-MMKG) construction method combining conventional knowledge engineering and large language models. A unified scene knowledge injection framework is introduced for knowledge representation. To evaluate the advantages of our proposed method, we instantiate Scene-MMKG considering typical indoor robotic functionalities (Manipulation and Mobility), named ManipMob-MMKG. Comparisons in characteristics indicate our instantiated ManipMob-MMKG has broad superiority in data-collection efficiency and knowledge quality. Experimental results on typical embodied tasks show that knowledge-enhanced methods using our instantiated ManipMob-MMKG can improve the performance obviously without re-designing model structures complexly. Our project can be found at https://sites.google.com/view/manipmob-mmkg
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05852v1">Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Embodied AI agents require a fine-grained understanding of the physical world mediated through visual and language inputs. Such capabilities are difficult to learn solely from task-specific data. This has led to the emergence of pre-trained vision-language models as a tool for transferring representations learned from internet-scale data to downstream tasks and new domains. However, commonly used contrastively trained representations such as in CLIP have been shown to fail at enabling embodied agents to gain a sufficiently fine-grained scene understanding -- a capability vital for control. To address this shortcoming, we consider representations from pre-trained text-to-image diffusion models, which are explicitly optimized to generate images from text prompts and as such, contain text-conditioned representations that reflect highly fine-grained visuo-spatial information. Using pre-trained text-to-image diffusion models, we construct Stable Control Representations which allow learning downstream control policies that generalize to complex, open-ended environments. We show that policies learned using Stable Control Representations are competitive with state-of-the-art representation learning approaches across a broad range of simulated control settings, encompassing challenging manipulation and navigation tasks. Most notably, we show that Stable Control Representations enable learning policies that exhibit state-of-the-art performance on OVMM, a difficult open-vocabulary navigation benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2405.03999v1">Interaction Design for Human-AI Choreography Co-creation</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 GenAICHI: CHI 2024 Workshop on Generative AI and HCI
    </div>
    <details class="paper-abstract">
      Human-AI co-creation aims to combine human and AI strengths for artistic results exceeding individual capabilities. Frameworks exist for painting, music, and poetry, but choreography's embodied nature demands a dedicated approach. This paper explores AI-assisted choreography techniques (e.g., generative ideation, embodied improvisation) and analyzes interaction design -- how humans and AI collaborate and communicate -- to inform the design considerations of future human-AI choreography co-creation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.03999v1">Interaction Design for Human-AI Choreography Co-creation</a></div>
    <div class="paper-meta">
      📅 2024-05-07
      | 💬 GenAICHI: CHI 2024 Workshop on Generative AI and HCI
    </div>
    <details class="paper-abstract">
      Human-AI co-creation aims to combine human and AI strengths for artistic results exceeding individual capabilities. Frameworks exist for painting, music, and poetry, but choreography's embodied nature demands a dedicated approach. This paper explores AI-assisted choreography techniques (e.g., generative ideation, embodied improvisation) and analyzes interaction design -- how humans and AI collaborate and communicate -- to inform the design considerations of future human-AI choreography co-creation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03164v1">The Role of Predictive Uncertainty and Diversity in Embodied AI and Robot Learning</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Uncertainty has long been a critical area of study in robotics, particularly when robots are equipped with analytical models. As we move towards the widespread use of deep neural networks in robots, which have demonstrated remarkable performance in research settings, understanding the nuances of uncertainty becomes crucial for their real-world deployment. This guide offers an overview of the importance of uncertainty and provides methods to quantify and evaluate it from an applications perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.06665v2">The Essential Role of Causality in Foundation World Models for Embodied AI</a></div>
    <div class="paper-meta">
      📅 2024-05-01
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models, especially in large multi-modal models and conversational agents, have ignited interest in the potential of generally capable embodied agents. Such agents will require the ability to perform new tasks in many different real-world environments. However, current foundation models fail to accurately model physical interactions and are therefore insufficient for Embodied AI. The study of causality lends itself to the construction of veridical world models, which are crucial for accurately predicting the outcomes of possible interactions. This paper focuses on the prospects of building foundation world models for the upcoming generation of embodied agents and presents a novel viewpoint on the significance of causality within these. We posit that integrating causal considerations is vital to facilitating meaningful physical interactions with the world. Finally, we demystify misconceptions about causality in this context and present our outlook for future research.
    </details>
</div>
