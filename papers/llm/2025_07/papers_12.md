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
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13757v4">GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01216v1">PAE MobiLLM: Privacy-Aware and Efficient LLM Fine-Tuning on the Mobile Device via Additive Side-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      There is a huge gap between numerous intriguing applications fostered by on-device large language model (LLM) fine-tuning (FT) from fresh mobile data and the limited resources of a mobile device. While existing server-assisted methods (e.g., split learning or side-tuning) may enable LLM FT on the local mobile device, they suffer from heavy communication burdens of activation transmissions, and may disclose data, labels or fine-tuned models to the server. To address those issues, we develop PAE MobiLLM, a privacy-aware and efficient LLM FT method which can be deployed on the mobile device via server-assisted additive side-tuning. To further accelerate FT convergence and improve computing efficiency, PAE MobiLLM integrates activation caching on the server side, which allows the server to reuse historical activations and saves the mobile device from repeatedly computing forward passes for the recurring data samples. Besides, to reduce communication cost, PAE MobiLLM develops a one-token (i.e., ``pivot'' token) activation shortcut that transmits only a single activation dimension instead of full activation matrices to guide the side network tuning. Last but not least, PAE MobiLLM introduces the additive adapter side-network design which makes the server train the adapter modules based on device-defined prediction differences rather than raw ground-truth labels. In this way, the server can only assist device-defined side-network computing, and learn nothing about data, labels or fine-tuned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01206v1">2024 NASA SUITS Report: LLM-Driven Immersive Augmented Reality User Interface for Robotics and Space Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      As modern computing advances, new interaction paradigms have emerged, particularly in Augmented Reality (AR), which overlays virtual interfaces onto physical objects. This evolution poses challenges in machine perception, especially for tasks like 3D object pose estimation in complex, dynamic environments. Our project addresses critical issues in human-robot interaction within mobile AR, focusing on non-intrusive, spatially aware interfaces. We present URSA, an LLM-driven immersive AR system developed for NASA's 2023-2024 SUITS challenge, targeting future spaceflight needs such as the Artemis missions. URSA integrates three core technologies: a head-mounted AR device (e.g., HoloLens) for intuitive visual feedback, voice control powered by large language models for hands-free interaction, and robot tracking algorithms that enable accurate 3D localization in dynamic settings. To enhance precision, we leverage digital twin localization technologies, using datasets like DTTD-Mobile and specialized hardware such as the ZED2 camera for real-world tracking under noise and occlusion. Our system enables real-time robot control and monitoring via an AR interface, even in the absence of ground-truth sensors--vital for hazardous or remote operations. Key contributions include: (1) a non-intrusive AR interface with LLM-based voice input; (2) a ZED2-based dataset tailored for non-rigid robotic bodies; (3) a Local Mission Control Console (LMCC) for mission visualization; (4) a transformer-based 6DoF pose estimator (DTTDNet) optimized for depth fusion and real-time tracking; and (5) end-to-end integration for astronaut mission support. This work advances digital twin applications in robotics, offering scalable solutions for both aerospace and industrial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20295v2">Self-reflective Uncertainties: Do LLMs Know Their Internal Answer Distribution?</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      To reveal when a large language model (LLM) is uncertain about a response, uncertainty quantification commonly produces percentage numbers along with the output. But is this all we can do? We argue that in the output space of LLMs, the space of strings, exist strings expressive enough to summarize the distribution over output strings the LLM deems possible. We lay a foundation for this new avenue of uncertainty explication and present SelfReflect, a theoretically-motivated metric to assess how faithfully a string summarizes an LLM's internal answer distribution. We show that SelfReflect is able to discriminate even subtle differences of candidate summary strings and that it aligns with human judgement, outperforming alternative metrics such as LLM judges and embedding comparisons. With SelfReflect, we investigate a number of self-summarization methods and find that even state-of-the-art reasoning models struggle to explicate their internal uncertainty. But we find that faithful summarizations can be generated by sampling and summarizing. To support the development of this universal form of LLM uncertainties, we publish our metric at https://github.com/apple/ml-selfreflect
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00979v1">Enhancing LLM Agent Safety via Causal Influence Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Accepted at ACL 2025 Findings, Source code: https://github.com/HahmDY/causal_influence_prompting.git
    </div>
    <details class="paper-abstract">
      As autonomous agents powered by large language models (LLMs) continue to demonstrate potential across various assistive tasks, ensuring their safe and reliable behavior is crucial for preventing unintended consequences. In this work, we introduce CIP, a novel technique that leverages causal influence diagrams (CIDs) to identify and mitigate risks arising from agent decision-making. CIDs provide a structured representation of cause-and-effect relationships, enabling agents to anticipate harmful outcomes and make safer decisions. Our approach consists of three key steps: (1) initializing a CID based on task specifications to outline the decision-making process, (2) guiding agent interactions with the environment using the CID, and (3) iteratively refining the CID based on observed behaviors and outcomes. Experimental results demonstrate that our method effectively enhances safety in both code execution and mobile device control tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01080v1">Development and Comparative Evaluation of Three Artificial Intelligence Models (NLP, LLM, JEPA) for Predicting Triage in Emergency Departments: A 7-Month Retrospective Proof-of-Concept</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Triage errors, including undertriage and overtriage, are persistent challenges in emergency departments (EDs). With increasing patient influx and staff shortages, the integration of artificial intelligence (AI) into triage protocols has gained attention. This study compares the performance of three AI models [Natural Language Processing (NLP), Large Language Models (LLM), and Joint Embedding Predictive Architecture (JEPA)] in predicting triage outcomes against the FRENCH scale and clinical practice.We conducted a retrospective analysis of a prospectively recruited cohort gathering adult patient triage data over a 7-month period at the Roger Salengro Hospital ED (Lille, France). Three AI models were trained and validated : (1) TRIAGEMASTER (NLP), (2) URGENTIAPARSE (LLM), and (3) EMERGINET (JEPA). Data included demographic details, verbatim chief complaints, vital signs, and triage outcomes based on the FRENCH scale and GEMSA coding. The primary outcome was the concordance of AI-predicted triage level with the FRENCH gold-standard. It was assessed thanks to various indicators : F1-Score, Weighted Kappa, Spearman, MAE, RMSE. The LLM model (URGENTIAPARSE) showed higher accuracy (composite score: 2.514) compared to JEPA (EMERGINET, 0.438) and NLP (TRIAGEMASTER, -3.511), outperforming nurse triage (-4.343). Secondary analyses highlighted the effectiveness of URGENTIAPARSE in predicting hospitalization needs (GEMSA) and its robustness with structured data versus raw transcripts (either for GEMSA prediction or for FRENCH prediction). LLM architecture, through abstraction of patient representations, offers the most accurate triage predictions among tested models. Integrating AI into ED workflows could enhance patient safety and operational efficiency, though integration into clinical workflows requires addressing model limitations and ensuring ethical transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00838v1">Stylometry recognizes human and LLM-generated texts in short samples</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00833v1">HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 Project Page: https://openhumanoidgen.github.io
    </div>
    <details class="paper-abstract">
      For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
    </details>
</div>
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2412.02113v2">Trust & Safety of LLMs and LLMs in Trust & Safety</a></div>
    <div class="paper-meta">
      📅 2025-07-01
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\ By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety. This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00507v1">LLM-Mesh: Enabling Elastic Sharing for Serverless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-01
    </div>
    <details class="paper-abstract">
      The rise of LLMs has driven demand for private serverless deployments, characterized by moderate-scale models and infrequent requests. While existing solutions follow exclusive GPU deployment, we take a step back to explore modern platforms and find that: Emerging CPU architectures with built-in accelerators are capable of serving LLMs but remain underutilized, and both CPUs and GPUs can accommodate multiple LLMs simultaneously. We propose LLM-Mesh, a serverless inference scheme for small-to-mid-sized LLMs that enables elastic sharing across heterogeneous hardware. LLM-Mesh tackles three fundamental challenges: (1) precise, fine-grained compute resource allocation at token-level to handle fluctuating computational demands; (2) a coordinated and forward-looking memory scaling mechanism to detect out-of-memory hazards and reduce operational overhead; and (3) a dual approach that reduces resource fragmentation through proactive preemption and reactive bin-packing. Experimental results on 4 32-core CPUs and 4 A100 GPUs show that LLM-Meshimproves service capacity by 44% - 63% through sharing, while further leveraging CPUs boosts this to 91% - 159%.
    </details>
</div>
