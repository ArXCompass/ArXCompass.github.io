# embodied ai - 2022_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [embodied ai](https://arxcompass.github.io/papers/embodied_ai)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2201.04279v1">Dynamical Audio-Visual Navigation: Catching Unheard Moving Sound Sources in Unmapped 3D Environments</a></div>
    <div class="paper-meta">
      📅 2022-01-12
    </div>
    <details class="paper-abstract">
      Recent work on audio-visual navigation targets a single static sound in noise-free audio environments and struggles to generalize to unheard sounds. We introduce the novel dynamic audio-visual navigation benchmark in which an embodied AI agent must catch a moving sound source in an unmapped environment in the presence of distractors and noisy sounds. We propose an end-to-end reinforcement learning approach that relies on a multi-modal architecture that fuses the spatial audio-visual information from a binaural audio signal and spatial occupancy maps to encode the features needed to learn a robust navigation policy for our new complex task settings. We demonstrate that our approach outperforms the current state-of-the-art with better generalization to unheard sounds and better robustness to noisy scenarios on the two challenging 3D scanned real-world datasets Replica and Matterport3D, for the static and dynamic audio-visual navigation benchmarks. Our novel benchmark will be made available at http://dav-nav.cs.uni-freiburg.de.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2201.02734v1">Building Human-like Communicative Intelligence: A Grounded Perspective</a></div>
    <div class="paper-meta">
      📅 2022-01-11
    </div>
    <details class="paper-abstract">
      Modern Artificial Intelligence (AI) systems excel at diverse tasks, from image classification to strategy games, even outperforming humans in many of these domains. After making astounding progress in language learning in the recent decade, AI systems, however, seem to approach the ceiling that does not reflect important aspects of human communicative capacities. Unlike human learners, communicative AI systems often fail to systematically generalize to new data, suffer from sample inefficiency, fail to capture common-sense semantic knowledge, and do not translate to real-world communicative situations. Cognitive Science offers several insights on how AI could move forward from this point. This paper aims to: (1) suggest that the dominant cognitively-inspired AI directions, based on nativist and symbolic paradigms, lack necessary substantiation and concreteness to guide progress in modern AI, and (2) articulate an alternative, "grounded", perspective on AI advancement, inspired by Embodied, Embedded, Extended, and Enactive Cognition (4E) research. I review results on 4E research lines in Cognitive Science to distinguish the main aspects of naturalistic learning conditions that play causal roles for human language development. I then use this analysis to propose a list of concrete, implementable components for building "grounded" linguistic intelligence. These components include embodying machines in a perception-action cycle, equipping agents with active exploration mechanisms so they can build their own curriculum, allowing agents to gradually develop motor abilities to promote piecemeal language development, and endowing the agents with adaptive feedback from their physical and social environment. I hope that these ideas can direct AI research towards building machines that develop human-like language abilities through their experiences with the world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2103.04918v8">A Survey of Embodied AI: From Simulators to Research Tasks</a></div>
    <div class="paper-meta">
      📅 2022-01-06
      | 💬 This work has been accepted by IEEE Transactions on Emerging Topics in Computational Intelligence
    </div>
    <details class="paper-abstract">
      There has been an emerging paradigm shift from the era of "internet AI" to "embodied AI", where AI algorithms and agents no longer learn from datasets of images, videos or text curated primarily from the internet. Instead, they learn through interactions with their environments from an egocentric perception similar to humans. Consequently, there has been substantial growth in the demand for embodied AI simulators to support various embodied AI research tasks. This growing interest in embodied AI is beneficial to the greater pursuit of Artificial General Intelligence (AGI), but there has not been a contemporary and comprehensive survey of this field. This paper aims to provide an encyclopedic survey for the field of embodied AI, from its simulators to its research. By evaluating nine current embodied AI simulators with our proposed seven features, this paper aims to understand the simulators in their provision for use in embodied AI research and their limitations. Lastly, this paper surveys the three main research tasks in embodied AI -- visual exploration, visual navigation and embodied question answering (QA), covering the state-of-the-art approaches, evaluation metrics and datasets. Finally, with the new insights revealed through surveying the field, the paper will provide suggestions for simulator-for-task selections and recommendations for the future directions of the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2103.04918v8">A Survey of Embodied AI: From Simulators to Research Tasks</a></div>
    <div class="paper-meta">
      📅 2022-01-05
      | 💬 This work has been accepted by IEEE Transactions on Emerging Topics in Computational Intelligence
    </div>
    <details class="paper-abstract">
      There has been an emerging paradigm shift from the era of "internet AI" to "embodied AI", where AI algorithms and agents no longer learn from datasets of images, videos or text curated primarily from the internet. Instead, they learn through interactions with their environments from an egocentric perception similar to humans. Consequently, there has been substantial growth in the demand for embodied AI simulators to support various embodied AI research tasks. This growing interest in embodied AI is beneficial to the greater pursuit of Artificial General Intelligence (AGI), but there has not been a contemporary and comprehensive survey of this field. This paper aims to provide an encyclopedic survey for the field of embodied AI, from its simulators to its research. By evaluating nine current embodied AI simulators with our proposed seven features, this paper aims to understand the simulators in their provision for use in embodied AI research and their limitations. Lastly, this paper surveys the three main research tasks in embodied AI -- visual exploration, visual navigation and embodied question answering (QA), covering the state-of-the-art approaches, evaluation metrics and datasets. Finally, with the new insights revealed through surveying the field, the paper will provide suggestions for simulator-for-task selections and recommendations for the future directions of the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2201.00411v1">The Introspective Agent: Interdependence of Strategy, Physiology, and Sensing for Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2022-01-02
    </div>
    <details class="paper-abstract">
      The last few years have witnessed substantial progress in the field of embodied AI where artificial agents, mirroring biological counterparts, are now able to learn from interaction to accomplish complex tasks. Despite this success, biological organisms still hold one large advantage over these simulated agents: adaptation. While both living and simulated agents make decisions to achieve goals (strategy), biological organisms have evolved to understand their environment (sensing) and respond to it (physiology). The net gain of these factors depends on the environment, and organisms have adapted accordingly. For example, in a low vision aquatic environment some fish have evolved specific neurons which offer a predictable, but incredibly rapid, strategy to escape from predators. Mammals have lost these reactive systems, but they have a much larger fields of view and brain circuitry capable of understanding many future possibilities. While traditional embodied agents manipulate an environment to best achieve a goal, we argue for an introspective agent, which considers its own abilities in the context of its environment. We show that different environments yield vastly different optimal designs, and increasing long-term planning is often far less beneficial than other improvements, such as increased physical ability. We present these findings to broaden the definition of improvement in embodied AI passed increasingly complex models. Just as in nature, we hope to reframe strategy as one tool, among many, to succeed in an environment. Code is available at: https://github.com/sarahpratt/introspective.
    </details>
</div>
