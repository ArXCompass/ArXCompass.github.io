# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00672v1">Toward Edge General Intelligence with Multiple-Large Language Model (Multi-LLM): Architecture, Trust, and Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Edge computing enables real-time data processing closer to its source, thus improving the latency and performance of edge-enabled AI applications. However, traditional AI models often fall short when dealing with complex, dynamic tasks that require advanced reasoning and multimodal data processing. This survey explores the integration of multi-LLMs (Large Language Models) to address this in edge computing, where multiple specialized LLMs collaborate to enhance task performance and adaptability in resource-constrained environments. We review the transition from conventional edge AI models to single LLM deployment and, ultimately, to multi-LLM systems. The survey discusses enabling technologies such as dynamic orchestration, resource scheduling, and cross-domain knowledge transfer that are key for multi-LLM implementation. A central focus is on trusted multi-LLM systems, ensuring robust decision-making in environments where reliability and privacy are crucial. We also present multimodal multi-LLM architectures, where multiple LLMs specialize in handling different data modalities, such as text, images, and audio, by integrating their outputs for comprehensive analysis. Finally, we highlight future directions, including improving resource efficiency, trustworthy governance multi-LLM systems, while addressing privacy, trust, and robustness concerns. This survey provides a valuable reference for researchers and practitioners aiming to leverage multi-LLM systems in edge computing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02990v1">`For Argument's Sake, Show Me How to Harm Myself!': Jailbreaking LLMs in Suicide and Self-Harm Contexts</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to increasingly sophisticated safety protocols and features designed to prevent harmful, unethical, or unauthorized outputs. However, these guardrails remain susceptible to novel and creative forms of adversarial prompting, including manually generated test cases. In this work, we present two new test cases in mental health for (i) suicide and (ii) self-harm, using multi-step, prompt-level jailbreaking and bypass built-in content and safety filters. We show that user intent is disregarded, leading to the generation of detailed harmful content and instructions that could cause real-world harm. We conduct an empirical evaluation across six widely available LLMs, demonstrating the generalizability and reliability of the bypass. We assess these findings and the multilayered ethical tensions that they present for their implications on prompt-response filtering and context- and task-specific model development. We recommend a more comprehensive and systematic approach to AI safety and ethics while emphasizing the need for continuous adversarial testing in safety-critical AI deployments. We also argue that while certain clearly defined safety measures and guardrails can and must be implemented in LLMs, ensuring robust and comprehensive safety across all use cases and domains remains extremely challenging given the current technical maturity of general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00657v1">Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      We investigate how Large Language Models (LLMs) behave when simulating political discourse on social media. Leveraging 21 million interactions on X during the 2024 U.S. presidential election, we construct LLM agents based on 1,186 real users, prompting them to reply to politically salient tweets under controlled conditions. Agents are initialized either with minimal ideological cues (Zero Shot) or recent tweet history (Few Shot), allowing one-to-one comparisons with human replies. We evaluate three model families (Gemini, Mistral, and DeepSeek) across linguistic style, ideological consistency, and toxicity. We find that richer contextualization improves internal consistency but also amplifies polarization, stylized signals, and harmful language. We observe an emergent distortion that we call "generation exaggeration": a systematic amplification of salient traits beyond empirical baselines. Our analysis shows that LLMs do not emulate users, they reconstruct them. Their outputs, indeed, reflect internal optimization dynamics more than observed behavior, introducing structural biases that compromise their reliability as social proxies. This challenges their use in content moderation, deliberative simulations, and policy modeling.
    </details>
</div>
