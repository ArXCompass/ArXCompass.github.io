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
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- Part 12

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00829v1">On the Surprising Efficacy of LLMs for Penetration-Testing</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      This paper presents a critical examination of the surprising efficacy of Large Language Models (LLMs) in penetration testing. The paper thoroughly reviews the evolution of LLMs and their rapidly expanding capabilities which render them increasingly suitable for complex penetration testing operations. It systematically details the historical adoption of LLMs in both academic research and industry, showcasing their application across various offensive security tasks and covering broader phases of the cyber kill chain. Crucially, the analysis also extends to the observed adoption of LLMs by malicious actors, underscoring the inherent dual-use challenge of this technology within the security landscape. The unexpected effectiveness of LLMs in this context is elucidated by several key factors: the strong alignment between penetration testing's reliance on pattern-matching and LLMs' core strengths, their inherent capacity to manage uncertainty in dynamic environments, and cost-effective access to competent pre-trained models through LLM providers. The current landscape of LLM-aided penetration testing is categorized into interactive 'vibe-hacking' and the emergence of fully autonomous systems. The paper identifies and discusses significant obstacles impeding wider adoption and safe deployment. These include critical issues concerning model reliability and stability, paramount safety and security concerns, substantial monetary and ecological costs, implications for privacy and digital sovereignty, complex questions of accountability, and profound ethical dilemmas. This comprehensive review and analysis provides a foundation for discussion on future research directions and the development of robust safeguards at the intersection of AI and security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01077v1">Good Enough to Learn: LLM-based Anomaly Detection in ECU Logs without Reliable Labels</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 6 pages, 7 figures, 4 tables, accepted to IEEE Intelligent Vehicles Symposium (IV) 2025
    </div>
    <details class="paper-abstract">
      Anomaly detection often relies on supervised or clustering approaches, with limited success in specialized domains like automotive communication systems where scalable solutions are essential. We propose a novel decoder-only Large Language Model (LLM) to detect anomalies in Electronic Control Unit (ECU) communication logs. Our approach addresses two key challenges: the lack of LLMs tailored for ECU communication and the complexity of inconsistent ground truth data. By learning from UDP communication logs, we formulate anomaly detection simply as identifying deviations in time from normal behavior. We introduce an entropy regularization technique that increases model's uncertainty in known anomalies while maintaining consistency in similar scenarios. Our solution offers three novelties: a decoder-only anomaly detection architecture, a way to handle inconsistent labeling, and an adaptable LLM for different ECU communication use cases. By leveraging the generative capabilities of decoder-only models, we present a new technique that addresses the high cost and error-prone nature of manual labeling through a more scalable system that is able to learn from a minimal set of examples, while improving detection accuracy in complex communication environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00814v1">Many LLMs Are More Utilitarian Than One</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 9 pages, 8 Figures, 7 tables
    </div>
    <details class="paper-abstract">
      Moral judgment is integral to large language model (LLM) alignment and social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function collectively during collaboration, compared to individual agents. In human moral judgment, group deliberation leads to a utilitarian boost: a tendency to endorse norm violations that maximize benefits for the greatest number of people despite harms. We study whether a similar dynamic emerges in multi-agent LLM systems. We tested six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reasoned independently, and (2) Group, where they engaged in multi-turn discussions in pairs or triads. In personal moral dilemmas, where agents must decide to directly harm one individual to maximize the utility for others, all models found moral violations to be more acceptable when part of a group than individually, similar to human experiments. Some models endorsed actions that maximized overall well-being, even if they benefited strangers over familiar individuals. Others became more willing to violate moral norms in groups. However, while human groups show a similar action bias, the mechanism for their utilitarian boost differs from LLMs. Whereas the human shift comes from heightened sensitivity to decision outcomes, LLM groups show either reduced norm sensitivity or enhanced impartiality. This suggests that while the surface behavior of LLM collectives mimics human group reasoning, the underlying drivers differ. We discuss the implications for AI alignment, multi-agent design, and artificial moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00797v1">VEDA: Efficient LLM Generation Through Voting-based KV Cache Eviction and Dataflow-flexible Accelerator</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 DAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in natural language processing tasks but pose significant computational and memory challenges for edge deployment due to their intensive resource demands. This work addresses the efficiency of LLM inference by algorithm-hardware-dataflow tri-optimizations. We propose a novel voting-based KV cache eviction algorithm, balancing hardware efficiency and algorithm accuracy by adaptively identifying unimportant kv vectors. From a dataflow perspective, we introduce a flexible-product dataflow and a runtime reconfigurable PE array for matrix-vector multiplication. The proposed approach effectively handles the diverse dimensional requirements and solves the challenges of incrementally varying sequence lengths. Additionally, an element-serial scheduling scheme is proposed for nonlinear operations, such as softmax and layer normalization (layernorm). Results demonstrate a substantial reduction in latency, accompanied by a significant decrease in hardware complexity, from O(N) to O(1). The proposed solution is realized in a custom-designed accelerator, VEDA, which outperforms existing hardware platforms. This research represents a significant advancement in LLM inference on resource-constrained edge devices, facilitating real-time processing, enhancing data privacy, and enabling model customization.
    </details>
</div>
