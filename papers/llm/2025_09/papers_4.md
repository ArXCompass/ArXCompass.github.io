# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23143v1">MathBode: Frequency-Domain Fingerprints of LLM Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $\phi \approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11087v2">Enhancing Delta Compression in LLMs via SVD-based Quantization Error Minimization</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Fine-tuning is a crucial process for adapting large language models (LLMs) to diverse applications. In certain scenarios, like multi-tenant serving, a large number of LLMs finetuned from the same base model are deployed to meet complex requirements for users. Recent works explore delta-compression approaches to quantize and compress the delta weights between the customized LLM and the corresponding base model. However, they exhibit inadequate performance at high compression ratios due to their empirical nature. In this work, we introduce DeltaMix, an adaptive mixed-precision delta-compression framework designed to minimize quantization error in the singular value decomposition (SVD) space without imposing additional assumptions. DeltaMix provides a theoretical justification for the necessity of mixed-precision compression and presents a practical quantization solution that involves solving a 0/1 linear integer programming problem alongside a reconstruction target correction method. Experimental results across multiple models and benchmarks illustrate that DeltaMix consistently outperforms all baseline methods. Notably, on tasks such as AIME2024 and GQA, DeltaMix exceeds the performance of the best baseline, Delta-CoMe, by 22.3\% and 6.1\% for 7B parameter models, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23113v1">Exploring LLM-based Frameworks for Fault Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based systems present new opportunities for autonomous health monitoring in sensor-rich industrial environments. This study explores the potential of LLMs to detect and classify faults directly from sensor data, while producing inherently explainable outputs through natural language reasoning. We systematically evaluate how LLM-system architecture (single-LLM vs. multi-LLM), input representations (raw vs. descriptive statistics), and context window size affect diagnostic performance. Our findings show that LLM systems perform most effectively when provided with summarized statistical inputs, and that systems with multiple LLMs using specialized prompts offer improved sensitivity for fault classification compared to single-LLM systems. While LLMs can produce detailed and human-readable justifications for their decisions, we observe limitations in their ability to adapt over time in continual learning settings, often struggling to calibrate predictions during repeated fault cycles. These insights point to both the promise and the current boundaries of LLM-based systems as transparent, adaptive diagnostic tools in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04664v2">Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 Preprint. Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from significant performance degradation when processing long contexts due to proactive interference, where irrelevant information in earlier parts of the context disrupts reasoning and memory recall. While most research focuses on external memory systems to augment LLMs' capabilities, we propose a complementary approach: empowering LLMs with Active Context Management (ACM) tools to actively sculpt their internal working memory. We introduce Sculptor, a framework that equips LLMs with three categories of tools: (1) context fragmentation, (2) summary, hide, and restore, and (3) precise search. Our approach enables LLMs to proactively manage their attention and working memory, analogous to how humans selectively focus on relevant information while filtering out distractions. Experimental evaluation on diverse long-context benchmarks demonstrates that Sculptor significantly improves performance even without specific training, leveraging LLMs' inherent tool-calling and instruction-following capabilities. To further optimize these strategies, we introduce a novel dynamic context-aware reinforcement learning (RL) approach, advancing the training of an agent that actively modifies its own conversational history. By enabling Active Context Management, Sculptor not only mitigates proactive interference but also provides a cognitive foundation for more reliable reasoning across diverse long-context tasks-highlighting that explicit context-control strategies, rather than merely larger token windows, are key to robustness at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01705v4">Progressive Binarization with Semi-Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in natural language processing, but their high computational and memory costs hinder deployment on resource-constrained devices. Binarization represents the most extreme form of quantization, yet binarized models still contain redundancy that can be further removed. Pruning provides a natural way to eliminate such redundancy, but na\"ive combination with binarization often results in severe performance degradation. In this paper, we propose Progressive Binarization with Semi-Structured Pruning (PBS$^2$P), a novel post-training framework that seamlessly integrates binarization and semi-structured pruning. We first propose Stepwise semi-structured Pruning with Binarization Optimization (SPBO), which progressively introduces sparsity while optimizing binarization parameters to jointly reduce pruning and quantization error, yielding more stable and accurate compression. Additionally, we propose a Coarse-to-Fine Search (CFS) that first allocates pruning ratios and then refines element selection, further enhancing overall performance. Extensive experiments across multiple LLM families show that PBS$^2$P consistently outperforms state-of-the-art (SOTA) binary post-training quantization methods in both perplexity and downstream accuracy. The code and models will be available at https://github.com/XIANGLONGYAN/PBS2P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23067v1">Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      The rising cost of acquiring supervised data has driven significant interest in self-improvement for large language models (LLMs). Straightforward unsupervised signals like majority voting have proven effective in generating pseudo-labels for verifiable tasks, while their applicability to unverifiable tasks (e.g., translation) is limited by the open-ended character of responses. As a result, self-evaluation mechanisms (e.g., self-judging and entropy minimization) are predominantly used to derive pseudo-labels. However, self-evaluation relying on LLMs typically incurs high computational overhead and introduces overconfidence issues due to intrinsic biases. To address these challenges, we propose a novel self-evaluation-free approach for unverifiable tasks, designed for lightweight yet effective self-improvement. Inspired by majority voting commonly employed in verifiable tasks, we propose semantic voting as a novel mechanism that relaxes the principle of hard matching (i.e., exact matching) toward soft matching (i.e., semantic similarity). Soft matching is achieved by leveraging a lightweight sentence embedding model to quantify semantic similarity, thereby mitigating excessive computational burden and intrinsic bias-associated limitations of self-evaluation. Comprehensive experiments demonstrate that our method achieves substantial gains in computational efficiency and overall better performance than self-evaluation methods across diverse model architectures and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23058v1">Risk Profiling and Modulation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23041v1">Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 NeurIPS 2025 Spotlight. Source code: https://github.com/liangzid/VirusInfectionAttack
    </div>
    <details class="paper-abstract">
      Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23019v1">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09995v3">QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have shown remarkable capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks by leveraging fundamental and sentiment-based inputs for strategic decision-making. However, these approaches are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT typically requires rapid, risk-aware decisions driven by structured, short-horizon signals, such as technical indicators, chart patterns, and trend features. These signals stand in sharp contrast to the long-horizon, text-driven reasoning that characterizes most existing LLM-based systems in finance. To bridge this gap, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents--Indicator, Pattern, Trend, and Risk--each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. Extensive experiments across nine financial instruments, including Bitcoin and Nasdaq futures, demonstrate that QuantAgent consistently outperforms baseline methods, achieving higher predictive accuracy at both 1-hour and 4-hour trading intervals across multiple evaluation metrics. Our findings suggest that coupling structured trading signals with LLM-based reasoning provides a viable path for traceable, real-time decision systems in high-frequency financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23007v1">Taming Variability: Randomized and Bootstrapped Conformal Risk Control for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-27
      | 💬 34 pages; 4 figures; 9 tables; includes appendices
    </div>
    <details class="paper-abstract">
      We transform the randomness of LLMs into precise assurances using an actuator at the API interface that applies a user-defined risk constraint in finite samples via Conformal Risk Control (CRC). This label-free and model-agnostic actuator manages ship/abstain/escalate actions based solely on a scalar score from opaque outputs. We enhance CRC's computational efficiency and robustness through Batched Bootstrap CRC (BB-CRC) and Randomized Batched Weighted-Average CRC (RBWA-CRC), reducing calibration calls and stabilizing thresholds while maintaining statistical validity. Additionally, we present a semantic quantification method grounded in gram matrix geometry, resulting in interpretable signal and metric design. Together these pieces deliver principled randomness control for LLM hallucination mitigation and LLM-as-judge reliability. Our framework is assessed using four datasets, demonstrating its efficacy in enhancing factual accuracy and measuring LLM-as-judge performance, yielding a simplified and computationally efficient control layer that converts variability into statistical validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21638v2">R1-Ranker: Teaching LLM Rankers to Reason</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong reasoning abilities in domains like mathematics, coding, and scientific problem-solving, yet their potential for ranking tasks, where prime examples include retrieval, recommender systems, and LLM routing, remains underexplored. Ranking requires complex reasoning across heterogeneous candidates, but existing LLM-based rankers are often domain-specific, tied to fixed backbones, and lack iterative refinement, limiting their ability to fully exploit LLMs' reasoning potential. To address these challenges, we propose R1-Ranker, a reasoning-incentive framework built on reinforcement learning, with two complementary designs: DRanker, which generates full rankings in one shot, and IRanker, which decomposes ranking into an iterative elimination process with step-wise rewards to encourage deeper reasoning. We evaluate unified R1-Rankers on nine datasets spanning recommendation, routing, and passage ranking, showing that IRanker-3B consistently achieves state-of-the-art performance, surpasses larger 7B models on some tasks, and yields a 15.7% average relative improvement. Ablation and generalization experiments further confirm the critical role of reinforcement learning and iterative reasoning, with IRanker-3B improving zero-shot performance by over 9% on out-of-domain tasks and reasoning traces boosting other LLMs by up to 22.87%. These results demonstrate that unifying diverse ranking tasks with a single reasoning-driven foundation model is both effective and essential for advancing LLM reasoning in ranking scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10540v2">FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a diverse landscape of models, each excelling at different tasks. This diversity drives researchers to employ multiple LLMs in practice, leaving behind valuable multi-LLM log data. This naturally leads to the question of whether such logs can be fully leveraged to fuse LLMs' complementary capabilities. Although prior work has explored various strategies for integrating multiple LLMs, we argue that practical fusion must meet two essential requirements: (1) compatibility with real-world serving scenarios (e.g., local and API-based serving), and (2) flexibility to operate at different stages of the LLM pipeline to meet varied user needs (e.g., fine-tuning and inference stages). To this end, we introduce LLMFusionBench, a large-scale benchmark for LLM fusion that spans 14 tasks across five domains, with responses from 20 open-source LLMs (8B--671B) totaling 103M tokens. Building on LLMFusionBench, we propose FusionFactory, a systematic framework with three elaborated levels: (1) query-level fusion via tailored LLM routers, (2) thought-level fusion leveraging retrieved abstract reasoning templates, and (3) model-level fusion via distillation from top-ranked responses. Experiments show that FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with the optimal fusion configuration varying across benchmarks, highlighting the promise of multi-LLM log data as a practical foundation for fusing diverse LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23515v1">From Human Annotation to Automation: LLM-in-the-Loop Active Learning for Arabic Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Natural language processing (NLP), particularly sentiment analysis, plays a vital role in areas like marketing, customer service, and social media monitoring by providing insights into user opinions and emotions. However, progress in Arabic sentiment analysis remains limited due to the lack of large, high-quality labeled datasets. While active learning has proven effective in reducing annotation efforts in other languages, few studies have explored it in Arabic sentiment tasks. Likewise, the use of large language models (LLMs) for assisting annotation and comparing their performance to human labeling is still largely unexplored in the Arabic context. In this paper, we propose an active learning framework for Arabic sentiment analysis designed to reduce annotation costs while maintaining high performance. We evaluate multiple deep learning architectures: Specifically, long short-term memory (LSTM), gated recurrent units (GRU), and recurrent neural networks (RNN), across three benchmark datasets: Hunger Station, AJGT, and MASAC, encompassing both modern standard Arabic and dialectal variations. Additionally, two annotation strategies are compared: Human labeling and LLM-assisted labeling. Five LLMs are evaluated as annotators: GPT-4o, Claude 3 Sonnet, Gemini 2.5 Pro, DeepSeek Chat, and LLaMA 3 70B Instruct. For each dataset, the best-performing LLM was used: GPT-4o for Hunger Station, Claude 3 Sonnet for AJGT, and DeepSeek Chat for MASAC. Our results show that LLM-assisted active learning achieves competitive or superior performance compared to human labeling. For example, on the Hunger Station dataset, the LSTM model achieved 93% accuracy with only 450 labeled samples using GPT-4o-generated labels, while on the MASAC dataset, DeepSeek Chat reached 82% accuracy with 650 labeled samples, matching the accuracy obtained through human labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23510v1">Model Consistency as a Cheap yet Predictive Proxy for LLM Elo Scores</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      New large language models (LLMs) are being released every day. Some perform significantly better or worse than expected given their parameter count. Therefore, there is a need for a method to independently evaluate models. The current best way to evaluate a model is to measure its Elo score by comparing it to other models in a series of contests - an expensive operation since humans are ideally required to compare LLM outputs. We observe that when an LLM is asked to judge such contests, the consistency with which it selects a model as the best in a matchup produces a metric that is 91% correlated with its own human-produced Elo score. This provides a simple proxy for Elo scores that can be computed cheaply, without any human data or prior knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15213v2">Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14307v3">LLM-3D Print: Large Language Models To Monitor and Control 3D Printing</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      Industry 4.0 has revolutionized manufacturing by driving digitalization and shifting the paradigm toward additive manufacturing (AM). Fused Deposition Modeling (FDM), a key AM technology, enables the creation of highly customized, cost-effective products with minimal material waste through layer-by-layer extrusion, posing a significant challenge to traditional subtractive methods. However, the susceptibility of material extrusion techniques to errors often requires expert intervention to detect and mitigate defects that can severely compromise product quality. While automated error detection and machine learning models exist, their generalizability across diverse 3D printer setups, firmware, and sensors is limited, and deep learning methods require extensive labeled datasets, hindering scalability and adaptability. To address these challenges, we present a process monitoring and control framework that leverages pre-trained Large Language Models (LLMs) alongside 3D printers to detect and address printing defects. The LLM evaluates print quality by analyzing images captured after each layer or print segment, identifying failure modes and querying the printer for relevant parameters. It then generates and executes a corrective action plan. We validated the effectiveness of the proposed framework in identifying defects by comparing it against a control group of engineers with diverse AM expertise. Our evaluation demonstrated that LLM-based agents not only accurately identify common 3D printing errors, such as inconsistent extrusion, stringing, warping, and layer adhesion, but also effectively determine the parameters causing these failures and autonomously correct them without any need for human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10171v3">Beyond Jailbreaking: Auditing Contextual Privacy in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      LLM agents have begun to appear as personal assistants, customer service bots, and clinical aides. While these applications deliver substantial operational benefits, they also require continuous access to sensitive data, which increases the likelihood of unauthorized disclosures. Moreover, these disclosures go beyond mere explicit disclosure, leaving open avenues for gradual manipulation or sidechannel information leakage. This study proposes an auditing framework for conversational privacy that quantifies an agent's susceptibility to these risks. The proposed Conversational Manipulation for Privacy Leakage (CMPL) framework is designed to stress-test agents that enforce strict privacy directives against an iterative probing strategy. Rather than focusing solely on a single disclosure event or purely explicit leakage, CMPL simulates realistic multi-turn interactions to systematically uncover latent vulnerabilities. Our evaluation on diverse domains, data modalities, and safety configurations demonstrates the auditing framework's ability to reveal privacy risks that are not deterred by existing single-turn defenses, along with an in-depth longitudinal study of the temporal dynamics of leakage, strategies adopted by adaptive adversaries, and the evolution of adversarial beliefs about sensitive targets. In addition to introducing CMPL as a diagnostic tool, the paper delivers (1) an auditing procedure grounded in quantifiable risk metrics and (2) an open benchmark for evaluation of conversational privacy across agent implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10540v2">FusionFactory: Fusing LLM Capabilities with Multi-LLM Log Data</a></div>
    <div class="paper-meta">
      📅 2025-09-27
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a diverse landscape of models, each excelling at different tasks. This diversity drives researchers to employ multiple LLMs in practice, leaving behind valuable multi-LLM log data. This naturally leads to the question of whether such logs can be fully leveraged to fuse LLMs' complementary capabilities. Although prior work has explored various strategies for integrating multiple LLMs, we argue that practical fusion must meet two essential requirements: (1) compatibility with real-world serving scenarios (e.g., local and API-based serving), and (2) flexibility to operate at different stages of the LLM pipeline to meet varied user needs (e.g., fine-tuning and inference stages). To this end, we introduce LLMFusionBench, a large-scale benchmark for LLM fusion that spans 14 tasks across five domains, with responses from 20 open-source LLMs (8B--671B) totaling 103M tokens. Building on LLMFusionBench, we propose FusionFactory, a systematic framework with three elaborated levels: (1) query-level fusion via tailored LLM routers, (2) thought-level fusion leveraging retrieved abstract reasoning templates, and (3) model-level fusion via distillation from top-ranked responses. Experiments show that FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with the optimal fusion configuration varying across benchmarks, highlighting the promise of multi-LLM log data as a practical foundation for fusing diverse LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15735v2">EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 5 pages, submitted to ICASSP 2026, September 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer broad utility but remain prone to hallucination and out-of-distribution (OOD) errors. We propose EigenTrack, an interpretable real-time detector that uses the spectral geometry of hidden activations, a compact global signature of model dynamics. By streaming covariance-spectrum statistics such as entropy, eigenvalue gaps, and KL divergence from random baselines into a lightweight recurrent classifier, EigenTrack tracks temporal shifts in representation structure that signal hallucination and OOD drift before surface errors appear. Unlike black- and grey-box methods, it needs only a single forward pass without resampling. Unlike existing white-box detectors, it preserves temporal context, aggregates global signals, and offers interpretable accuracy-latency trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16355v2">How Strategic Agents Respond: Comparing Analytical Models with LLM-Generated Responses in Strategic Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Add GPT 5 experiments
    </div>
    <details class="paper-abstract">
      When ML algorithms are deployed to automate human-related decisions, human agents may learn the underlying decision policies and adapt their behavior. Strategic Classification (SC) has emerged as a framework for studying this interaction between agents and decision-makers to design more trustworthy ML systems. Prior theoretical models in SC assume that agents are perfectly or approximately rational and respond to decision policies by optimizing their utility. However, the growing prevalence of LLMs raises the possibility that real-world agents may instead rely on these tools for strategic advice. This shift prompts two questions: (i) Can LLMs generate effective and socially responsible strategies in SC settings? (ii) Can existing SC theoretical models accurately capture agent behavior when agents follow LLM-generated advice? To investigate these questions, we examine five critical SC scenarios: hiring, loan applications, school admissions, personal income, and public assistance programs. We simulate agents with diverse profiles who interact with three commercial LLMs (GPT-4o, GPT-4.1, and GPT-5), following their suggestions on effort allocations on features. We compare the resulting agent behaviors with the best responses in existing SC models. Our findings show that: (i) Even without access to the decision policy, LLMs can generate effective strategies that improve both agents' scores and qualification; (ii) At the population level, LLM-guided effort allocation strategies yield similar or even higher score improvements, qualification rates, and fairness metrics as those predicted by the SC theoretical model, suggesting that the theoretical model may still serve as a reasonable proxy for LLM-influenced behavior; and (iii) At the individual level, LLMs tend to produce more diverse and balanced effort allocations than theoretical models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v5">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted to EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22582v1">Fine-Grained Detection of Context-Grounded Hallucinations Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Context-grounded hallucinations are cases where model outputs contain information not verifiable against the source text. We study the applicability of LLMs for localizing such hallucinations, as a more practical alternative to existing complex evaluation pipelines. In the absence of established benchmarks for meta-evaluation of hallucinations localization, we construct one tailored to LLMs, involving a challenging human annotation of over 1,000 examples. We complement the benchmark with an LLM-based evaluation protocol, verifying its quality in a human evaluation. Since existing representations of hallucinations limit the types of errors that can be expressed, we propose a new representation based on free-form textual descriptions, capturing the full range of possible errors. We conduct a comprehensive study, evaluating four large-scale LLMs, which highlights the benchmark's difficulty, as the best model achieves an F1 score of only 0.67. Through careful analysis, we offer insights into optimal prompting strategies for the task and identify the main factors that make it challenging for LLMs: (1) a tendency to incorrectly flag missing details as inconsistent, despite being instructed to check only facts in the output; and (2) difficulty with outputs containing factually correct information absent from the source - and thus not verifiable - due to alignment with the model's parametric knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22576v1">EPO: Entropy-regularized Policy Optimization for LLM Agents Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feedback causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms: (1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis justifies that EPO guarantees monotonically decreasing entropy variance while maintaining convergence. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22572v1">Dynamic Experts Search: Enhancing Reasoning in Mixture-of-Experts LLMs at Test Time</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) enhances the reasoning ability of large language models (LLMs) by allocating additional computation during inference. However, existing approaches primarily rely on output-level sampling while overlooking the role of model architecture. In mainstream Mixture-of-Experts (MoE) LLMs, we observe that varying the number of activated experts yields complementary solution sets with stable accuracy, revealing a new and underexplored source of diversity. Motivated by this observation, we propose Dynamic Experts Search (DES), a TTS strategy that elevates expert activation into a controllable dimension of the search space. DES integrates two key components: (1) Dynamic MoE, which enables direct control of expert counts during inference to generate diverse reasoning trajectories without additional cost; and (2) Expert Configuration Inheritance, which preserves consistent expert counts within a reasoning path while varying them across runs, thereby balancing stability and diversity throughout the search. Extensive experiments across MoE architectures, verifiers and reasoning benchmarks (i.e., math, code and knowledge) demonstrate that DES reliably outperforms TTS baselines, enhancing accuracy and stability without additional cost. These results highlight DES as a practical and scalable form of architecture-aware TTS, illustrating how structural flexibility in modern LLMs can advance reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12313v2">ExpertSteer: Intervening in LLMs through Expert Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20172v2">Benchmarking LLMs in Web API Integration Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 To be published in Proceedings of 2nd ACM International Conference on AI-powered Software, Benchmark & Dataset Track (AIware '25); updated paper title and affiliations
    </div>
    <details class="paper-abstract">
      API integration is a cornerstone of our digital infrastructure, enabling software systems to connect and interact. However, as shown by many studies, writing or generating correct code to invoke APIs, particularly web APIs, is challenging. Although large language models (LLMs) have become popular in software development, their effectiveness in automating the generation of web API integration code remains unexplored. In order to address this, we present a dataset and evaluation pipeline designed to assess the ability of LLMs to generate web API invocation code. Our experiments with several open-source LLMs reveal that generating API invocations poses a significant challenge, resulting in hallucinated endpoints, incorrect argument usage, and other errors. None of the evaluated open-source models were able to solve more than 40% of the tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22544v1">HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v3">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22510v1">We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Alignment of Large Language Models (LLMs) along multiple objectives-helpfulness, harmlessness, and honesty (HHH)-is critical for safe and reliable deployment. Prior work has used steering vector-small control signals injected into hidden states-to guide LLM outputs, typically via one-to-one (1-to-1) Transformer decoders. In this setting, optimizing a single alignment objective can inadvertently overwrite representations learned for other objectives, leading to catastrophic forgetting. More recent approaches extend steering vectors via one-to-many (1-to-N) Transformer decoders. While this alleviates catastrophic forgetting, naive multi-branch designs optimize each objective independently, which can cause inference fragmentation-outputs across HHH objectives may become inconsistent. We propose Adaptive Multi-Branch Steering (AMBS), a two-stage 1-to-N framework for unified and efficient multi-objective alignment. In Stage I, post-attention hidden states of the Transformer layer are computed once to form a shared representation. In Stage II, this representation is cloned into parallel branches and steered via a policy-reference mechanism, enabling objective-specific control while maintaining cross-objective consistency. Empirical evaluations on Alpaca, BeaverTails, and TruthfulQA show that AMBS consistently improves HHH alignment across multiple 7B LLM backbones. For example, on DeepSeek-7B, AMBS improves average alignment scores by +32.4% and reduces unsafe outputs by 11.0% compared to a naive 1-to-N baseline, while remaining competitive with state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22506v1">Representing LLMs in Prompt Semantic Task Space</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted to Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive results over various tasks, and ever-expanding public repositories contain an abundance of pre-trained models. Therefore, identifying the best-performing LLM for a given task is a significant challenge. Previous works have suggested learning LLM representations to address this. However, these approaches present limited scalability and require costly retraining to encompass additional models and datasets. Moreover, the produced representation utilizes distinct spaces that cannot be easily interpreted. This work presents an efficient, training-free approach to representing LLMs as linear operators within the prompts' semantic task space, thus providing a highly interpretable representation of the models' application. Our method utilizes closed-form computation of geometrical properties and ensures exceptional scalability and real-time adaptability to dynamically expanding repositories. We demonstrate our approach on success prediction and model selection tasks, achieving competitive or state-of-the-art results with notable performance in out-of-sample scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22490v1">JGU Mainz's Submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: MT and QA</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 WMT 25 Shared Task LLMs with Limited Resources for Slavic Languages: MT and QA
    </div>
    <details class="paper-abstract">
      This paper presents the JGU Mainz submission to the WMT25 Shared Task on LLMs with Limited Resources for Slavic Languages: Machine Translation and Question Answering, focusing on Ukrainian, Upper Sorbian, and Lower Sorbian. For each language, we jointly fine-tune a Qwen2.5-3B-Instruct model for both tasks with parameter-efficient finetuning. Our pipeline integrates additional translation and multiple-choice question answering (QA) data. For Ukrainian QA, we further use retrieval-augmented generation. We also apply ensembling for QA in Upper and Lower Sorbian. Experiments show that our models outperform the baseline on both tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10739v2">Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 27 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16134v2">Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit position bias - a systematic tendency to neglect information at specific context positions. However, the patterns of position bias behavior, depending on the language or model, remain unexplored. We present a multilingual study across five typologically distinct languages (English, Russian, German, Hindi, and Vietnamese) and five model architectures, examining how position bias interacts with prompt strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven, yet exhibits language-specific variations. For instance, Qwen2.5-7B-Instruct and DeepSeek 7B Chat consistently favors late positions, challenging established assumptions of a universal early-token bias in LLMs. (2) Explicitly instructing the model that "the context is relevant to the query" unexpectedly reduces accuracy across languages, undermining common prompt-engineering practices. (3) While the largest accuracy drop occurs when relevant information is placed in the middle of the context, this is not explicitly reflected by a corresponding peak in output entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18697v2">Can LLMs Alleviate Catastrophic Forgetting in Graph Continual Learning? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Nowadays, real-world data, including graph-structure data, often arrives in a streaming manner, which means that learning systems need to continuously acquire new knowledge without forgetting previously learned information. Although substantial existing works attempt to address catastrophic forgetting in graph machine learning, they are all based on training from scratch with streaming data. With the rise of pretrained models, an increasing number of studies have leveraged their strong generalization ability for continual learning. Therefore, in this work, we attempt to answer whether large language models (LLMs) can mitigate catastrophic forgetting in Graph Continual Learning (GCL). We first point out that current experimental setups for GCL have significant flaws, as the evaluation stage may lead to task ID leakage. Then, we evaluate the performance of LLMs in more realistic scenarios and find that even minor modifications can lead to outstanding results. Finally, based on extensive experiments, we propose a simple-yet-effective method, Simple Graph Continual Learning (SimGCL), that surpasses the previous state-of-the-art GNN-based baseline by around 20% under the rehearsal-free constraint. To facilitate reproducibility, we have developed an easy-to-use benchmark LLM4GCL for training and evaluating existing GCL methods. The code is available at: https://github.com/ZhixunLEE/LLM4GCL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15241v3">MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08123v4">QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22431v1">TreeMind: Automatically Reproducing Android Bug Reports via LLM-empowered Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Automatically reproducing Android app crashes from textual bug reports is challenging, particularly when the reports are incomplete and the modern UI exhibits high combinatorial complexity. Existing approaches based on reinforcement learning or large language models (LLMs) exhibit limitations in such scenarios. They struggle to infer unobserved steps and reconstruct the underlying user action sequences to navigate the vast UI interaction space, primarily due to limited goal-directed reasoning and planning. We present TreeMind, a novel technique that integrates LLMs with a customized Monte Carlo Tree Search (MCTS) algorithm to achieve strategic UI exploration in bug reproduction. To the best of our knowledge, this is the first work to combine external decision-making with LLM semantic reasoning for reliable bug reproduction. We formulate the reproduction task as a target-driven search problem, leveraging MCTS as the core planning mechanism to iteratively refine action sequences. To enhance MCTS with semantic reasoning, we introduce two LLM-guided agents with distinct roles: Expander generates top-k promising actions based on the current UI state and exploration history, while Simulator estimates the likelihood that each action leads toward successful reproduction. By incorporating multi-modal UI inputs and advanced prompting techniques, TreeMind conducts feedback-aware navigation that identifies missing but essential user actions and incrementally reconstructs the reproduction paths. We evaluate TreeMind on a dataset of 93 real-world Android bug reports from three widely-used benchmarks. Experimental results show that it significantly outperforms four state-of-the-art baselines in reproduction success rate. A real-world case study indicates that integrating LLM reasoning with MCTS-based planning is a compelling direction for automated bug reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17388v4">Can LLMs be Good Graph Judge for Knowledge Graph Construction?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      In real-world scenarios, most of the data obtained from the information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. We identified three limitations with respect to existing KG construction methods: (1) There could be a large amount of noise in real-world documents, which could result in extracting messy information. (2) Naive LLMs usually extract inaccurate knowledge from some domain-specific documents. (3) Hallucination phenomenon cannot be overlooked when directly using LLMs to construct KGs. In this paper, we propose \textbf{GraphJudge}, a KG construction framework to address the aforementioned challenges. In this framework, we designed an entity-centric strategy to eliminate the noise information in the documents. And we fine-tuned a LLM as a graph judge to finally enhance the quality of generated KGs. Experiments conducted on two general and one domain-specific text-graph pair datasets demonstrate state-of-the-art performance against various baseline methods with strong generalization abilities. Our code is available at \href{https://github.com/hhy-huang/GraphJudge}{https://github.com/hhy-huang/GraphJudge}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22391v1">Do LLM Agents Know How to Ground, Recover, and Assess? A Benchmark for Epistemic Competence in Information-Seeking Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Recent work has explored training Large Language Model (LLM) search agents with reinforcement learning (RL) for open-domain question answering (QA). However, most evaluations focus solely on final answer accuracy, overlooking how these agents reason with and act on external evidence. We introduce SeekBench, the first benchmark for evaluating the \textit{epistemic competence} of LLM search agents through step-level analysis of their response traces. SeekBench comprises 190 expert-annotated traces with over 1,800 response steps generated by LLM search agents, each enriched with evidence annotations for granular analysis of whether agents (1) generate reasoning steps grounded in observed evidence, (2) adaptively reformulate searches to recover from low-quality results, and (3) have proper calibration to correctly assess whether the current evidence is sufficient for providing an answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22367v1">What Is The Political Content in LLMs' Pre- and Post-Training Data?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 9 pages, under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to generate politically biased text, yet how such biases arise remains unclear. A crucial step toward answering this question is the analysis of training data, whose political content remains largely underexplored in current LLM research. To address this gap, we present in this paper an analysis of the pre- and post-training corpora of OLMO2, the largest fully open-source model released together with its complete dataset. From these corpora, we draw large random samples, automatically annotate documents for political orientation, and analyze their source domains and content. We then assess how political content in the training data correlates with models' stance on specific policy issues. Our analysis shows that left-leaning documents predominate across datasets, with pre-training corpora containing significantly more politically engaged content than post-training data. We also find that left- and right-leaning documents frame similar topics through distinct values and sources of legitimacy. Finally, the predominant stance in the training data strongly correlates with models' political biases when evaluated on policy issues. These findings underscore the need to integrate political content analysis into future data curation pipelines as well as in-depth documentation of filtering strategies for transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22338v1">Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 15 pages, 7 tables, accepted at the International Joint Conference on Learning & Reasoning (IJCLR 2025)
    </div>
    <details class="paper-abstract">
      Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11023v3">Beyond A Single AI Cluster: A Survey of Decentralized LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has revolutionized AI development, yet the resource demands beyond a single cluster or even datacenter, limiting accessibility to well-resourced organizations. Decentralized training has emerged as a promising paradigm to leverage dispersed resources across clusters, datacenters and regions, offering the potential to democratize LLM development for broader communities. As the first comprehensive exploration of this emerging field, we present decentralized LLM training as a resource-driven paradigm and categorize existing efforts into community-driven and organizational approaches. We further clarify this through: (1) a comparison with related paradigms, (2) a characterization of decentralized resources, and (3) a taxonomy of recent advancements. We also provide up-to-date case studies and outline future directions to advance research in decentralized LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22255v1">Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 1 table, 6 figures. 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Accepted for the Workshop: Evaluating the Evolving LLM Lifecycle Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for assessing Large Language Models' (LLMs) capabilities in combinatorial optimization, specifically addressing the 2D bin-packing problem. We introduce a systematic methodology that combines LLMs with evolutionary algorithms to generate and refine heuristic solutions iteratively. Through comprehensive experiments comparing LLM generated heuristics against traditional approaches (Finite First-Fit and Hybrid First-Fit), we demonstrate that LLMs can produce more efficient solutions while requiring fewer computational resources. Our evaluation reveals that GPT-4o achieves optimal solutions within two iterations, reducing average bin usage from 16 to 15 bins while improving space utilization from 0.76-0.78 to 0.83. This work contributes to understanding LLM evaluation in specialized domains and establishes benchmarks for assessing LLM performance in combinatorial optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21305v2">Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22251v1">Beyond Textual Context: Structural Graph Encoding with Adaptive Space Alignment to alleviate the hallucination of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Currently, the main approach for Large Language Models (LLMs) to tackle the hallucination issue is incorporating Knowledge Graphs(KGs).However, LLMs typically treat KGs as plain text, extracting only semantic information and limiting their use of the crucial structural aspects of KGs. Another challenge is the gap between the embedding spaces of KGs encoders and LLMs text embeddings, which hinders the effective integration of structured knowledge. To overcome these obstacles, we put forward the SSKG-LLM, an innovative model architecture that is designed to efficiently integrate both the Structural and Semantic information of KGs into the reasoning processes of LLMs. SSKG-LLM incorporates the Knowledge Graph Retrieval (KGR) module and the Knowledge Graph Encoding (KGE) module to preserve semantics while utilizing structure. Then, the Knowledge Graph Adaptation (KGA) module is incorporated to enable LLMs to understand KGs embeddings. We conduct extensive experiments and provide a detailed analysis to explore how incorporating the structural information of KGs can enhance the factual reasoning abilities of LLMs. Our code are available at https://github.com/yfangZhang/SSKG-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22250v1">Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Compliance</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has demonstrated remarkable capabilities, elevating the critical importance of LLM safety. However, existing safety methods rely on ad-hoc taxonomy and lack a rigorous, systematic protection, failing to ensure safety for the nuanced and complex behaviors of modern LLM systems. To address this problem, we solve LLM safety from legal compliance perspectives, named safety compliance. In this work, we posit relevant established legal frameworks as safety standards for defining and measuring safety compliance, including the EU AI Act and GDPR, which serve as core legal frameworks for AI safety and data security in Europe. To bridge the gap between LLM safety and legal compliance, we first develop a new benchmark for safety compliance by generating realistic LLM safety scenarios seeded with legal statutes. Subsequently, we align Qwen3-8B using Group Policy Optimization (GRPO) to construct a safety reasoner, Compliance Reasoner, which effectively aligns LLMs with legal standards to mitigate safety risks. Our comprehensive experiments demonstrate that the Compliance Reasoner achieves superior performance on the new benchmark, with average improvements of +10.45% for the EU AI Act and +11.85% for GDPR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v4">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22243v1">FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Full-Duplex Speech-to-Speech Large Language Models (LLMs) are foundational to natural human-computer interaction, enabling real-time spoken dialogue systems. However, benchmarking and modeling these models remains a fundamental challenge. We introduce FLEXI, the first benchmark for full-duplex LLM-human spoken interaction that explicitly incorporates model interruption in emergency scenarios. FLEXI systematically evaluates the latency, quality, and conversational effectiveness of real-time dialogue through six diverse human-LLM interaction scenarios, revealing significant gaps between open source and commercial models in emergency awareness, turn terminating, and interaction latency. Finally, we suggest that next token-pair prediction offers a promising path toward achieving truly seamless and human-like full-duplex interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05282v3">ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22211v1">Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Unsupervised analysis of text corpora is challenging, especially in data-scarce domains where traditional topic models struggle. While these models offer a solution, they typically describe clusters with lists of keywords that require significant manual effort to interpret and often lack semantic coherence. To address this critical interpretability gap, we introduce Recursive Thematic Partitioning (RTP), a novel framework that leverages Large Language Models (LLMs) to interactively build a binary tree. Each node in the tree is a natural language question that semantically partitions the data, resulting in a fully interpretable taxonomy where the logic of each cluster is explicit. Our experiments demonstrate that RTP's question-driven hierarchy is more interpretable than the keyword-based topics from a strong baseline like BERTopic. Furthermore, we establish the quantitative utility of these clusters by showing they serve as powerful features in downstream classification tasks, particularly when the data's underlying themes correlate with the task labels. RTP introduces a new paradigm for data exploration, shifting the focus from statistical pattern discovery to knowledge-driven thematic analysis. Furthermore, we demonstrate that the thematic paths from the RTP tree can serve as structured, controllable prompts for generative models. This transforms our analytical framework into a powerful tool for synthesis, enabling the consistent imitation of specific characteristics discovered in the source corpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22202v1">Library Hallucinations in LLMs: Risk Analysis Grounded in Developer Queries</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 23 pages, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate code, yet they continue to hallucinate, often inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite increasing awareness of these risks, little is known about how real-world prompt variations affect hallucination rates. Therefore, we present the first systematic study of how user-level prompt variations impact library hallucinations in LLM-generated code. We evaluate six diverse LLMs across two hallucination types: library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries). We investigate how realistic user language extracted from developer forums and how user errors of varying degrees (one- or multi-character misspellings and completely fake names/members) affect LLM hallucination rates. Our findings reveal systemic vulnerabilities: one-character misspellings in library names trigger hallucinations in up to 26% of tasks, fake library names are accepted in up to 99% of tasks, and time-related prompts lead to hallucinations in up to 84% of tasks. Prompt engineering shows promise for mitigating hallucinations, but remains inconsistent and LLM-dependent. Our results underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their potential exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09265v3">Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized the field of natural language processing (NLP), and recent studies have aimed to understand their underlying mechanisms. However, most of this research is conducted within a monolingual setting, primarily focusing on English. Few studies have attempted to explore the internal workings of LLMs in multilingual settings. In this study, we aim to fill this research gap by examining how neuron activation is shared across tasks and languages. We classify neurons into four distinct categories based on their responses to a specific input across different languages: all-shared, partial-shared, specific, and non-activated. Building upon this categorisation, we conduct extensive experiments on three tasks across nine languages using several LLMs and present an in-depth analysis in this work. Our findings reveal that: (i) deactivating the all-shared neurons significantly decreases performance; (ii) the shared neurons play a vital role in generating responses, especially for the all-shared neurons; (iii) neuron activation patterns are highly sensitive and vary across tasks, LLMs, and languages. These findings shed light on the internal workings of multilingual LLMs and pave the way for future research. We release the code to foster research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.19462v3">Constituency Parsing using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted at IEEE Transactions on Audio, Speech, and Language Processing (TASLP). See https://ieeexplore.ieee.org/document/11130901/ for the official version
    </div>
    <details class="paper-abstract">
      Constituency parsing is a fundamental yet unsolved challenge in natural language processing. In this paper, we examine the potential of recent large language models (LLMs) to address this challenge. We reformat constituency parsing as a sequence-to-sequence generation problem and evaluate the performance of a diverse range of LLMs under zero-shot, few-shot, and supervised fine-tuning learning paradigms. We observe that while LLMs achieve acceptable improvements, they still encounter substantial limitations, due to the absence of mechanisms to guarantee the validity and faithfulness of the generated constituent trees. Motivated by this observation, we propose two strategies to guide LLMs to generate more accurate constituent trees by learning from erroneous samples and refining outputs in a multi-agent collaboration way, respectively. The experimental results demonstrate that our methods effectively reduce the occurrence of invalid and unfaithful trees, thereby enhancing overall parsing performance and achieving promising results across different learning paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22170v1">Leveraging LLM Agents for Automated Video Game Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Testing MMORPGs (Massively Multiplayer Online Role-Playing Games) is a critical yet labor-intensive task in game development due to their complexity and frequent updating nature. Traditional automated game testing approaches struggle to achieve high state coverage and efficiency in these rich, open-ended environments, while existing LLM-based game-playing approaches are limited to shallow reasoning ability in understanding complex game state-action spaces and long-complex tasks. To address these challenges, we propose TITAN, an effective LLM-driven agent framework for intelligent MMORPG testing. TITAN incorporates four key components to: (1) perceive and abstract high-dimensional game states, (2) proactively optimize and prioritize available actions, (3) enable long-horizon reasoning with action trace memory and reflective self-correction, and (4) employ LLM-based oracles to detect potential functional and logic bugs with diagnostic reports. We implement the prototype of TITAN and evaluate it on two large-scale commercial MMORPGs spanning both PC and mobile platforms. In our experiments, TITAN achieves significantly higher task completion rates (95%) and bug detection performance compared to existing automated game testing approaches. An ablation study further demonstrates that each core component of TITAN contributes substantially to its overall performance. Notably, TITAN detects four previously unknown bugs that prior testing approaches fail to identify. We provide an in-depth discussion of these results, which offer guidance for new avenues of advancing intelligent, general-purpose testing systems. Moreover, TITAN has been deployed in eight real-world game QA pipelines, underscoring its practical impact as an LLM-driven game testing framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22166v1">Lightweight error mitigation strategies for post-training N:M activation sparsity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      The demand for efficient large language model (LLM) inference has intensified the focus on sparsification techniques. While semi-structured (N:M) pruning is well-established for weights, its application to activation pruning remains underexplored despite its potential for dynamic, input-adaptive compression and reductions in I/O overhead. This work presents a comprehensive analysis of methods for post-training N:M activation pruning in LLMs. Across multiple LLMs, we demonstrate that pruning activations enables superior preservation of generative capabilities compared to weight pruning at equivalent sparsity levels. We evaluate lightweight, plug-and-play error mitigation techniques and pruning criteria, establishing strong hardware-friendly baselines that require minimal calibration. Furthermore, we explore sparsity patterns beyond NVIDIA's standard 2:4, showing that the 16:32 pattern achieves performance nearly on par with unstructured sparsity. However, considering the trade-off between flexibility and hardware implementation complexity, we focus on the 8:16 pattern as a superior candidate. Our findings provide both effective practical methods for activation pruning and a motivation for future hardware to support more flexible sparsity patterns. Our code is available https://anonymous.4open.science/r/Structured-Sparse-Activations-Inference-EC3C/README.md .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22119v1">Universal Legal Article Prediction via Tight Collaboration between Supervised Classification Model and LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 10 pages, 6 figures, Accepted to ICAIL 2025 (International Conference on Artificial Intelligence and Law)
    </div>
    <details class="paper-abstract">
      Legal Article Prediction (LAP) is a critical task in legal text classification, leveraging natural language processing (NLP) techniques to automatically predict relevant legal articles based on the fact descriptions of cases. As a foundational step in legal decision-making, LAP plays a pivotal role in determining subsequent judgments, such as charges and penalties. Despite its importance, existing methods face significant challenges in addressing the complexities of LAP. Supervised classification models (SCMs), such as CNN and BERT, struggle to fully capture intricate fact patterns due to their inherent limitations. Conversely, large language models (LLMs), while excelling in generative tasks, perform suboptimally in predictive scenarios due to the abstract and ID-based nature of legal articles. Furthermore, the diversity of legal systems across jurisdictions exacerbates the issue, as most approaches are tailored to specific countries and lack broader applicability. To address these limitations, we propose Uni-LAP, a universal framework for legal article prediction that integrates the strengths of SCMs and LLMs through tight collaboration. Specifically, in Uni-LAP, the SCM is enhanced with a novel Top-K loss function to generate accurate candidate articles, while the LLM employs syllogism-inspired reasoning to refine the final predictions. We evaluated Uni-LAP on datasets from multiple jurisdictions, and empirical results demonstrate that our approach consistently outperforms existing baselines, showcasing its effectiveness and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19516v4">Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Modern LLM serving systems confront inefficient GPU utilization due to the fundamental mismatch between compute-intensive prefill and memory-bound decode phases. While current practices attempt to address this by organizing these phases into hybrid batches, such solutions create an inefficient tradeoff that sacrifices either throughput or latency, leaving substantial GPU resources underutilized. We identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks. 2) hybrid batches disproportionately prioritize latency over throughput, resulting in wasted compute and memory bandwidth. To mitigate the issues, we present Bullet, a novel spatial-temporal orchestration system that eliminates these inefficiencies through precise phase coordination. Bullet enables concurrent execution of prefill and decode phases, while dynamically provisioning GPU resources using real-time performance modeling. By integrating SLO-aware scheduling and adaptive resource allocation, Bullet maximizes utilization without compromising latency targets. Experimental evaluations on real-world workloads demonstrate that Bullet delivers 1.26x average throughput gains (up to 1.55x) over state-of-the-arts, while consistently meeting latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22114v1">SK2Decompile: LLM-based Two-Phase Binary Decompilation from Skeleton to Skin</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a promising approach for binary decompilation. However, the existing LLM-based decompilers still are somewhat limited in effectively presenting a program's source-level structure with its original identifiers. To mitigate this, we introduce SK2Decompile, a novel two-phase approach to decompile from the skeleton (semantic structure) to the skin (identifier) of programs. Specifically, we first apply a Structure Recovery model to translate a program's binary code to an Intermediate Representation (IR) as deriving the program's "skeleton", i.e., preserving control flow and data structures while obfuscating all identifiers with generic placeholders. We also apply reinforcement learning to reward the model for producing program structures that adhere to the syntactic and semantic rules expected by compilers. Second, we apply an Identifier Naming model to produce meaningful identifiers which reflect actual program semantics as deriving the program's "skin". We train the Identifier Naming model with a separate reinforcement learning objective that rewards the semantic similarity between its predictions and the reference code. Such a two-phase decompilation process facilitates advancing the correctness and readability of decompilation independently. Our evaluations indicate that SK2Decompile, significantly outperforms the SOTA baselines, achieving 21.6% average re-executability rate gain over GPT-5-mini on the HumanEval dataset and 29.4% average R2I improvement over Idioms on the GitHub2025 benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11140v2">Follow the Path: Reasoning over Knowledge Graph Paths to Improve LLM Factuality</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Updated version 26.9
    </div>
    <details class="paper-abstract">
      We introduce fs1, a simple yet effective method that improves the factuality of reasoning traces by sourcing them from large reasoning models (e.g., DeepSeek-R1) and grounding them by conditioning on knowledge graph (KG) paths. We fine-tune eight instruction-tuned Large Language Models (LLMs) on 3.9K factually grounded reasoning traces and rigorously evaluate them on six complex open-domain question-answering (QA) benchmarks encompassing 23.9K questions. Our results demonstrate that our fs1-tuned model (32B parameters) consistently outperforms instruction-tuned counterparts with parallel sampling by 6-14 absolute points (pass@$16$). Our detailed analysis shows that fs1 considerably improves model performance over more complex questions (requiring 3 or more hops on KG paths) and numerical answer types compared to the baselines. Furthermore, in single-pass inference, we notice that smaller LLMs show the most improvements. While prior works demonstrate the effectiveness of reasoning traces primarily in the STEM domains, our work shows strong evidence that anchoring reasoning to factual KG paths is a critical step in transforming LLMs for reliable knowledge-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22075v1">COSPADI: Compressing LLMs via Calibration-Guided Sparse Dictionary Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Post-training compression of large language models (LLMs) largely relies on low-rank weight approximation, which represents each column of a weight matrix in a shared low-dimensional subspace. While this is a computationally efficient strategy, the imposed structural constraint is rigid and can lead to a noticeable model accuracy drop. In this work, we propose CoSpaDi (Compression via Sparse Dictionary Learning), a novel training-free compression framework that replaces low-rank decomposition with a more flexible structured sparse factorization in which each weight matrix is represented with a dense dictionary and a column-sparse coefficient matrix. This formulation enables a union-of-subspaces representation: different columns of the original weight matrix are approximated in distinct subspaces spanned by adaptively selected dictionary atoms, offering greater expressiveness than a single invariant basis. Crucially, CoSpaDi leverages a small calibration dataset to optimize the factorization such that the output activations of compressed projection layers closely match those of the original ones, thereby minimizing functional reconstruction error rather than mere weight approximation. This data-aware strategy preserves better model fidelity without any fine-tuning under reasonable compression ratios. Moreover, the resulting structured sparsity allows efficient sparse-dense matrix multiplication and is compatible with post-training quantization for further memory and latency gains. We evaluate CoSpaDi across multiple Llama and Qwen models under per-layer and per-group settings at 20-50\% compression ratios, demonstrating consistent superiority over state-of-the-art data-aware low-rank methods both in accuracy and perplexity. Our results establish structured sparse dictionary learning as a powerful alternative to conventional low-rank approaches for efficient LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22067v1">The Rogue Scalpel: Activation Steering Compromises LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Activation steering is a promising technique for controlling LLM behavior by adding semantically meaningful vectors directly into a model's hidden states during inference. It is often framed as a precise, interpretable, and potentially safer alternative to fine-tuning. We demonstrate the opposite: steering systematically breaks model alignment safeguards, making it comply with harmful requests. Through extensive experiments on different model families, we show that even steering in a random direction can increase the probability of harmful compliance from 0% to 2-27%. Alarmingly, steering benign features from a sparse autoencoder (SAE), a common source of interpretable directions, increases these rates by a further 2-4%. Finally, we show that combining 20 randomly sampled vectors that jailbreak a single prompt creates a universal attack, significantly increasing harmful compliance on unseen requests. These results challenge the paradigm of safety through interpretability, showing that precise control over model internals does not guarantee precise control over model behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22034v1">The Thinking Spectrum: An Emperical Study of Tunable Reasoning in LLMs through Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      The growing demand for large language models (LLMs) with tunable reasoning capabilities in many real-world applications highlights a critical need for methods that can efficiently produce a spectrum of models balancing reasoning depth and computational cost. Model merging has emerged as a promising, training-free technique to address this challenge by arithmetically combining the weights of a general-purpose model with a specialized reasoning model. While various merging techniques exist, their potential to create a spectrum of models with fine-grained control over reasoning abilities remains largely unexplored. This work presents a large-scale empirical study evaluating a range of model merging techniques across multiple reasoning benchmarks. We systematically vary merging strengths to construct accuracy-efficiency curves, providing the first comprehensive view of the tunable performance landscape. Our findings reveal that model merging offers an effective and controllable method for calibrating the trade-off between reasoning accuracy and token efficiency, even when parent models have highly divergent weight spaces. Crucially, we identify instances of Pareto Improvement, where a merged model achieves both higher accuracy and lower token consumption than one of its parents. Our study provides the first comprehensive analysis of this tunable space, offering practical guidelines for creating LLMs with specific reasoning profiles to meet diverse application demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16831v2">Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 46 pages
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) aims to remove specified data, but its efficacy is typically assessed with task-level metrics like accuracy and perplexity. We demonstrate that these metrics are often misleading, as models can appear to forget while their original behavior is easily restored through minimal fine-tuning. This phenomenon of \emph{reversibility} suggests that information is merely suppressed, not genuinely erased. To address this critical evaluation gap, we introduce a \emph{representation-level analysis framework}. Our toolkit comprises PCA-based similarity and shift, centered kernel alignment (CKA), and Fisher information, complemented by a summary metric, the mean PCA distance, to measure representational drift. Applying this framework across six unlearning methods, three data domains, and two LLMs, we identify four distinct forgetting regimes based on their \emph{reversibility} and \emph{catastrophicity}. Our analysis reveals that achieving the ideal state--irreversible, non-catastrophic forgetting--is exceptionally challenging. By probing the limits of unlearning, we identify a case of seemingly irreversible, targeted forgetting, offering new insights for designing more robust erasure algorithms. Our findings expose a fundamental gap in current evaluation practices and establish a representation-level foundation for trustworthy unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13681v2">LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. As a result, these models cannot accurately identify and prioritize the most relevant context, leading to degraded response quality. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a new benchmark with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21990v1">WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities. With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code, checkpoints, and data will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01245v3">Towards Agentic OS: An LLM Agent Framework for Linux Schedulers</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis. We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in https://github.com/eunomia-bpf/schedcp
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21981v1">CoBel-World: Harnessing LLM Reasoning to Build a Collaborative Belief World for Optimizing Embodied Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents -- a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a collaborative belief world -- an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse open-world task knowledge into structured beliefs via a symbolic belief language, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 22-60% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08339v3">What Factors Affect LLMs and RLLMs in Financial Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. Additionally, we discuss strategies for enhancing the performance of LLMs and RLLMs in financial question answering, which may serve as a inspiration for future improvements. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21978v1">MotivGraph-SoIQ: Integrating Motivational Knowledge Graphs and Socratic Dialogue for Enhanced LLM Ideation</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold substantial potential for accelerating academic ideation but face critical challenges in grounding ideas and mitigating confirmation bias for further refinement. We propose integrating motivational knowledge graphs and socratic dialogue to address these limitations in enhanced LLM ideation (MotivGraph-SoIQ). This novel framework provides essential grounding and practical idea improvement steps for LLM ideation by integrating a Motivational Knowledge Graph (MotivGraph) with a Q-Driven Socratic Ideator. The MotivGraph structurally stores three key node types(problem, challenge and solution) to offer motivation grounding for the LLM ideation process. The Ideator is a dual-agent system utilizing Socratic questioning, which facilitates a rigorous refinement process that mitigates confirmation bias and improves idea quality across novelty, experimental rigor, and motivational rationality dimensions. On the ICLR25 paper topics dataset, MotivGraph-SoIQ exhibits clear advantages over existing state-of-the-art approaches across LLM-based scoring, ELO ranking, and human evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21947v1">Active Attacks: Red-teaming LLMs via Adaptive Environments</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 22 pages, 7 figures, 18 tables
    </div>
    <details class="paper-abstract">
      We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{https://github.com/dbsxodud-11/active_attacks}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22376v3">Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 ICLR 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17621v4">Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Regularized Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for reasoning. Specifically, sparse rewards fail to deliver sufficient feedback, particularly for challenging problems. Furthermore, such rewards induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across intermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a method designed to deliver dense rewards and amplify exploration in the RL-based paradigm. i-MENTOR introduces three innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; error-conditioned reward allocation to ensure efficient exploration on challenging samples while intrinsically stabilizing training; and advantage-preserving integration that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across 4 public datasets demonstrate i-MENTOR's effectiveness, achieving a 22.23\% improvement on AIME 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20802v2">SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Submitted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21907v1">A Large-Scale Dataset and Citation Intent Classification in Turkish with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Submitted to IEEE UBMK 2025 International Conference on Computer Science and Engineering
    </div>
    <details class="paper-abstract">
      Understanding the qualitative intent of citations is essential for a comprehensive assessment of academic research, a task that poses unique challenges for agglutinative languages like Turkish. This paper introduces a systematic methodology and a foundational dataset to address this problem. We first present a new, publicly available dataset of Turkish citation intents, created with a purpose-built annotation tool. We then evaluate the performance of standard In-Context Learning (ICL) with Large Language Models (LLMs), demonstrating that its effectiveness is limited by inconsistent results caused by manually designed prompts. To address this core limitation, we introduce a programmable classification pipeline built on the DSPy framework, which automates prompt optimization systematically. For final classification, we employ a stacked generalization ensemble to aggregate outputs from multiple optimized models, ensuring stable and reliable predictions. This ensemble, with an XGBoost meta-model, achieves a state-of-the-art accuracy of 91.3\%. Ultimately, this study provides the Turkish NLP community and the broader academic circles with a foundational dataset and a robust classification framework paving the way for future qualitative citation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12109v2">Personalized LLM Decoding via Contrasting Personal Preference</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21117v2">TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 22 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at https://github.com/TrustJudge/TrustJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21890v1">Not Everyone Wins with LLMs: Behavioral Patterns and Pedagogical Implications in AI-assisted Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      LLMs promise to democratize technical work in complex domains like programmatic data analysis, but not everyone benefits equally. We study how students with varied expertise use LLMs to complete Python-based data analysis in computational notebooks in a non-major course. Drawing on homework logs, recordings, and surveys from 36 students, we ask: Which expertise matters most, and how does it shape AI use? Our mixed-methods analysis shows that technical expertise -- not AI familiarity or communication skills -- remains a significant predictor of success. Students also vary widely in how they leverage LLMs, struggling at stages of forming intent, expressing inputs, interpreting outputs, and assessing results. We identify success and failure behaviors, such as providing context or decomposing prompts, that distinguish effective use. These findings inform AI literacy interventions, highlighting that lightweight demonstrations improve surface fluency but are insufficient; deeper training and scaffolds are needed to cultivate resilient AI use skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21884v1">You Can't Steal Nothing: Mitigating Prompt Leakages in LLMs via System Vectors</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 29 pages, 10 tables, 6figures, accepted by CCS 25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted across various applications, leveraging customized system prompts for diverse tasks. Facing potential system prompt leakage risks, model developers have implemented strategies to prevent leakage, primarily by disabling LLMs from repeating their context when encountering known attack patterns. However, it remains vulnerable to new and unforeseen prompt-leaking techniques. In this paper, we first introduce a simple yet effective prompt leaking attack to reveal such risks. Our attack is capable of extracting system prompts from various LLM-based application, even from SOTA LLM models such as GPT-4o or Claude 3.5 Sonnet. Our findings further inspire us to search for a fundamental solution to the problems by having no system prompt in the context. To this end, we propose SysVec, a novel method that encodes system prompts as internal representation vectors rather than raw text. By doing so, SysVec minimizes the risk of unauthorized disclosure while preserving the LLM's core language capabilities. Remarkably, this approach not only enhances security but also improves the model's general instruction-following abilities. Experimental results demonstrate that SysVec effectively mitigates prompt leakage attacks, preserves the LLM's functional integrity, and helps alleviate the forgetting issue in long-context scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21880v1">No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward - so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21868v1">What Makes LLM Agent Simulations Useful for Policy? Insights From an Iterative Design Engagement in Emergency Preparedness</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      There is growing interest in using Large Language Models as agents (LLM agents) for social simulations to inform policy, yet real-world adoption remains limited. This paper addresses the question: How can LLM agent simulations be made genuinely useful for policy? We report on a year-long iterative design engagement with a university emergency preparedness team. Across multiple design iterations, we iteratively developed a system of 13,000 LLM agents that simulate crowd movement and communication during a large-scale gathering under various emergency scenarios. These simulations informed actual policy implementation, shaping volunteer training, evacuation protocols, and infrastructure planning. Analyzing this process, we identify three design implications: start with verifiable scenarios and build trust gradually, use preliminary simulations to elicit tacit knowledge, and treat simulation and policy development as evolving together. These implications highlight actionable pathways to making LLM agent simulations that are genuinely useful for policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03981v3">A Survey on LLM-based Code Generation for Low-Resource and Domain-Specific Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 64 pages, 3 figures, 15 tables. Accepted in ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in code generation for popular programming languages. However, their performance on Low-Resource Programming Languages (LRPLs) and Domain-Specific Languages (DSLs) remains a significant challenge, affecting millions of developers-3.5 million users in Rust alone-who cannot fully utilize LLM capabilities. LRPLs and DSLs encounter unique obstacles, including data scarcity and, for DSLs, specialized syntax that is poorly represented in general-purpose datasets. Addressing these challenges is crucial, as LRPLs and DSLs enhance development efficiency in specialized domains, such as finance and science. While several surveys discuss LLMs in software engineering, none focus specifically on the challenges and opportunities associated with LRPLs and DSLs. Our survey fills this gap by systematically reviewing the current state, methodologies, and challenges in leveraging LLMs for code generation in these languages. We filtered 111 papers from over 27,000 published studies between 2020 and 2024 to evaluate the capabilities and limitations of LLMs in LRPLs and DSLs. We report the LLMs used, benchmarks, and metrics for evaluation, strategies for enhancing performance, and methods for dataset collection and curation. We identified four main evaluation techniques and several metrics for assessing code generation in LRPLs and DSLs. Our analysis categorizes improvement methods into six groups and summarizes novel architectures proposed by researchers. Despite various techniques and metrics, a standard approach and benchmark dataset for evaluating code generation in LRPLs and DSLs are lacking. This survey serves as a resource for researchers and practitioners at the intersection of LLMs, software engineering, and specialized programming languages, laying the groundwork for future advancements in code generation for LRPLs and DSLs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02091v2">LLM-OptiRA: LLM-Driven Optimization of Resource Allocation for Non-Convex Problems in Wireless Communications</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 6 pages,4 figures
    </div>
    <details class="paper-abstract">
      Solving non-convex resource allocation problems poses significant challenges in wireless communication systems, often beyond the capability of traditional optimization techniques. To address this issue, we propose LLM-OptiRA, the first framework that leverages large language models (LLMs) to automatically detect and transform non-convex components into solvable forms, enabling fully automated resolution of non-convex resource allocation problems in wireless communication systems. LLM-OptiRA not only simplifies problem-solving by reducing reliance on expert knowledge, but also integrates error correction and feasibility validation mechanisms to ensure robustness. Experimental results show that LLM-OptiRA achieves an execution rate of 96% and a success rate of 80% on GPT-4, significantly outperforming baseline approaches in complex optimization tasks across diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21837v1">Semantic Agreement Enables Efficient Open-Ended LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05257v2">Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Y. Hu and Y. Wang contribute equally
    </div>
    <details class="paper-abstract">
      Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, based on classic theories from memory science and cognitive science, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. Existing benchmarks either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmarks cover all four competencies. We introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark transforms existing long-context datasets and incorporates newly constructed datasets into a multi-turn format, effectively simulating the incremental information processing characteristic of memory agents. By carefully selecting and curating datasets, our benchmark provides comprehensive coverage of the four core memory competencies outlined above, thereby offering a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21820v1">Can LLMs Solve and Generate Linguistic Olympiad Puzzles?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 To be published in the Proceedings of Main Conference of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a combination of novel and exciting tasks: the solution and generation of linguistic puzzles. We focus on puzzles used in Linguistic Olympiads for high school students. We first extend the existing benchmark for the task of solving linguistic puzzles. We explore the use of Large Language Models (LLMs), including recent state-of-the-art models such as OpenAI's o1, for solving linguistic puzzles, analyzing their performance across various linguistic topics. We demonstrate that LLMs outperform humans on most puzzles types, except for those centered on writing systems, and for the understudied languages. We use the insights from puzzle-solving experiments to direct the novel task of puzzle generation. We believe that automating puzzle generation, even for relatively simple puzzles, holds promise for expanding interest in linguistics and introducing the field to a broader audience. This finding highlights the importance of linguistic puzzle generation as a research task: such puzzles can not only promote linguistics but also support the dissemination of knowledge about rare and understudied languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08791v2">Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 26 pages, 10 figures, 10 tables
    </div>
    <details class="paper-abstract">
      On-device inference offers privacy, offline use, and instant response, but consumer hardware restricts large language models (LLMs) to low throughput and capability. To overcome this challenge, we present prima.cpp, a distributed on-device inference system that runs 30-70B LLMs on consumer home clusters with mixed CPUs/GPUs, insufficient RAM/VRAM, slow disks, Wi-Fi links, and heterogeneous OSs. We introduce pipelined-ring parallelism (PRP) to overlap disk I/O with compute and communication, and address the prefetch-release conflict in mmap-based offloading. We further propose Halda, a heterogeneity-aware scheduler that co-optimizes per-device CPU/GPU workloads and device selection under RAM/VRAM constraints. On four consumer home devices, a 70B model reaches 674 ms/token TPOT with <6% memory pressure, and a 32B model with speculative decoding achieves 26 tokens/s. Compared with llama.cpp, exo, and dllama, our proposed prima.cpp achieves 5-17x lower TPOT, supports fine-grained model sizes from 8B to 70B, ensures broader cross-OS and quantization compatibility, and remains OOM-free, while also being Wi-Fi tolerant, privacy-preserving, and hardware-independent. The code is available at https://gitee.com/zonghang-li/prima.cpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12833v2">Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Many real-world scientific and industrial applications require the optimization of expensive black-box functions. Bayesian Optimization (BO) provides an effective framework for such problems. However, traditional BO methods are prone to get trapped in local optima and often lack interpretable insights. To address this issue, this paper designs Reasoning BO, a novel framework that leverages reasoning models to guide the sampling process in BO while incorporating multi-agent systems and knowledge graphs for online knowledge accumulation. By integrating the reasoning and contextual understanding capabilities of Large Language Models (LLMs), we can provide strong guidance to enhance the BO process. As the optimization progresses, Reasoning BO provides real-time sampling recommendations along with critical insights grounded in plausible scientific theories, aiding in the discovery of superior solutions within the search space. We systematically evaluate our approach across 10 diverse tasks encompassing synthetic mathematical functions and complex real-world applications. The framework demonstrates its capability to progressively refine sampling strategies through real-time insights and hypothesis evolution, effectively identifying higher-performing regions of the search space for focused exploration. This process highlights the powerful reasoning and context-learning abilities of LLMs in optimization scenarios. For example, in the Direct Arylation task, our method increased the yield to 60.7%, whereas traditional BO achieved only a 25.2% yield. Furthermore, our investigation reveals that smaller LLMs, when fine-tuned through reinforcement learning, can attain comparable performance to their larger counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01902v2">How LLMs Fail to Support Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Adiba and Neeley contributed equally
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) can amplify online misinformation, they also show promise in tackling misinformation. In this paper, we empirically study the capabilities of three LLMs -- ChatGPT, Gemini, and Claude -- in countering political misinformation. We implement a two-step, chain-of-thought prompting approach, where models first identify credible sources for a given claim and then generate persuasive responses. Our findings suggest that models struggle to ground their responses in real news sources, and tend to prefer citing left-leaning sources. We also observe varying degrees of response diversity among models. Our findings highlight concerns about using LLMs for fact-checking through only prompt-engineering, emphasizing the need for more robust guardrails. Our results have implications for both researchers and non-technical users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21798v1">Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Under review on ICLR 2026;Work in progress;
    </div>
    <details class="paper-abstract">
      Reward models (RMs) are crucial for aligning large language models (LLMs) with diverse cultures. Consequently, evaluating their cultural awareness is essential for further advancing global alignment of LLMs. However, existing RM evaluations fall short in assessing cultural awareness due to the scarcity of culturally relevant evaluation datasets. To fill this gap, we propose Cultural Awareness Reward modeling Benchmark (CARB), covering 10 distinct cultures across 4 cultural domains. Our extensive evaluation of state-of-the-art RMs reveals their deficiencies in modeling cultural awareness and demonstrates a positive correlation between performance on CARB and downstream multilingual cultural alignment tasks. Further analysis identifies the spurious correlations within culture-aware reward modeling, wherein RM's scoring relies predominantly on surface-level features rather than authentic cultural nuance understanding. To address these, we propose Think-as-Locals to elicit deeper culturally grounded reasoning from generative RMs via reinforcement learning from verifiable rewards (RLVR) and employ well-designed rewards to ensure accurate preference judgments and high-quality structured evaluation criteria generation. Experimental results validate its efficacy in mitigating spurious features interference and advancing culture-aware reward modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20067v3">MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). In direct comparison with physician-only diagnosis under the same evaluation protocol, MACD achieves comparable or superior performance, with improvements up to 16%. Furthermore, the MACD-human workflow yields an 18.6% improvement over physician-only diagnosis, demonstrating the synergistic potential of human-AI collaboration. Notably, the self-learned clinical knowledge exhibits strong cross-model stability, transferability across LLMs, and capacity for model-specific personalization.This work thus presents a scalable self-learning paradigm that bridges the gap between the intrinsic knowledge of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11574v2">Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Low-bit post-training quantization (PTQ) is a practical route to deploy reasoning-capable LLMs under tight memory and latency budgets, yet it can markedly impair mathematical reasoning (drops up to 69.81% in our harder settings). We address two deployment-critical questions with process-level precision: Where along a step-structured solution does degradation first arise? How to mitigate it while staying in the low-bit regime? Across widely used PTQ methods (AWQ, GPTQ, SmoothQuant), open-source model families (Qwen, LLaMA; 0.5--7B), and math reasoning benchmarks (GSM8K, MATH, AIME), we perform format-aligned chain-of-thought with step-aligned attribution and uncover two robust regularities: (i) PTQ disproportionately elevates method and execution errors relative to high-level conceptual mistakes; and (ii) failures emerge early, with the first vulnerable step flipping and cascading to the final answer. These regularities suggest a general intervention principle: restore local token-level margins exactly at the earliest failure frontier. We instantiate this principle as a lightweight measure$\rightarrow$locate$\rightarrow$restore loop that operates directly on the quantized model: detect the first faulty step, construct our "Silver Bullet" datasets, and apply small-scale supervised/preference tuning. In our settings, as few as 332 curated examples and 3--5 minutes of compute on a single GPU recover 4-bit weight math reasoning toward the full-precision baseline while preserving PTQ efficiency. Our framework is quantizer- and architecture-agnostic within the evaluated regimes, and turns low-bit degradation from a global accuracy problem into a local, reproducible process intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02097v3">JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluations and mismatched question difficulty, leading to incomplete evaluations of knowledge and capability boundaries, which hinder their effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to invoke knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more comprehensive evaluations of LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on https://github.com/DataArcTech/JudgeAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03064v2">Improving LLM-as-a-Judge Inference with the Judgment Distribution</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Using language models to scalably approximate human preferences on text quality (LLM-as-a-judge) has become a standard practice applicable to many tasks. A judgment is often extracted from the judge's textual output alone, typically with greedy decoding. However, LLM judges naturally provide distributions over judgment tokens, inviting a breadth of inference methods for extracting fine-grained preferences. We find that taking the mean of the judgment distribution consistently outperforms taking the mode (i.e. greedy decoding) in all evaluation settings (i.e. pointwise, pairwise, and listwise). We further explore novel methods of deriving preferences from judgment distributions, and find that methods incorporating risk aversion often improve performance. Lastly, we analyze LLM-as-a-judge paired with chain-of-thought (CoT) prompting, showing that CoT can collapse the spread of the judgment distribution, often harming performance. Our findings show that leveraging distributional output improves LLM-as-a-judge, as opposed to using the text interface alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21732v1">How Accurate Are LLMs at Multi-Question Answering on Conversational Transcripts?</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 Accepted by EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) for question answering (QA) over lengthy contexts is a significant challenge. In industrial settings, this process is often hindered by high computational costs and latency, especially when multiple questions must be answered based on the same context. In this work, we explore the capabilities of LLMs to answer multiple questions based on the same conversational context. We conduct extensive experiments and benchmark a range of both proprietary and public models on this challenging task. Our findings highlight that while strong proprietary LLMs like GPT-4o achieve the best overall performance, fine-tuned public LLMs with up to 8 billion parameters can surpass GPT-4o in accuracy, which demonstrates their potential for transparent and cost-effective deployment in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v3">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach consists of four key components: FinCap, which defines the core capabilities required for the target domain; FinRec, an effective training recipe that jointly optimizes continual pre-training and instruction-following, along with a novel preference data distillation method leveraging process signals from a generative reward model; FinTrain, a curated set of training datasets supporting FinRec; and FinEval, a comprehensive evaluation suite aligned with FinCap. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13841v2">LocationReasoner: Evaluating LLMs on Real-World Site Selection Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs), particularly those enhanced through reinforced post-training, have demonstrated impressive reasoning capabilities, as exemplified by models such as OpenAI o1 and DeepSeek-R1. However, these capabilities are predominantly benchmarked on domains like mathematical problem solving and code generation, leaving open the question of whether such reasoning skills generalize to complex real-world scenarios. In this paper, we introduce LocationReasoner, a benchmark designed to evaluate LLMs' reasoning abilities in the context of real-world site selection, where models must identify feasible locations by reasoning over diverse and complicated spatial, environmental, and logistic constraints. The benchmark covers carefully crafted queries of varying difficulty levels and is supported by a sandbox environment with in-house tools for constraint-based location search. Automated verification further guarantees the scalability of the benchmark, enabling the addition of arbitrary number of queries. Extensive evaluations on real-world site selection data from Boston, New York, and Tampa reveal that state-of-the-art reasoning models offer limited improvement over their non-reasoning predecessors in real-world contexts, with even the latest OpenAI o4 model failing on 30% of site selection tasks. Moreover, agentic strategies such as ReAct and Reflexion often suffer from over-reasoning, leading to worse outcomes than direct prompting. With key limitations of LLMs in holistic and non-linear reasoning highlighted, we release LocationReasoner to foster the development of LLMs and agents capable of robust, grounded reasoning in real-world decision-making tasks. Codes and data for our benchmark are available at https://github.com/miho-koda/LocationReasoner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21710v1">Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 28 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) and Graph-based RAG has become the important paradigm for enhancing Large Language Models (LLMs) with external knowledge. However, existing approaches face a fundamental trade-off. While graph-based methods are inherently dependent on high-quality graph structures, they face significant practical constraints: manually constructed knowledge graphs are prohibitively expensive to scale, while automatically extracted graphs from corpora are limited by the performance of the underlying LLM extractors, especially when using smaller, local-deployed models. This paper presents Think-on-Graph 3.0 (ToG-3), a novel framework that introduces Multi-Agent Context Evolution and Retrieval (MACER) mechanism to overcome these limitations. Our core innovation is the dynamic construction and refinement of a Chunk-Triplets-Community heterogeneous graph index, which pioneeringly incorporates a dual-evolution mechanism of Evolving Query and Evolving Sub-Graph for precise evidence retrieval. This approach addresses a critical limitation of prior Graph-based RAG methods, which typically construct a static graph index in a single pass without adapting to the actual query. A multi-agent system, comprising Constructor, Retriever, Reflector, and Responser agents, collaboratively engages in an iterative process of evidence retrieval, answer generation, sufficiency reflection, and, crucially, evolving query and subgraph. This dual-evolving multi-agent system allows ToG-3 to adaptively build a targeted graph index during reasoning, mitigating the inherent drawbacks of static, one-time graph construction and enabling deep, precise reasoning even with lightweight LLMs. Extensive experiments demonstrate that ToG-3 outperforms compared baselines on both deep and broad reasoning benchmarks, and ablation studies confirm the efficacy of the components of MACER framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23002v1">Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-09-26
      | 💬 26 pages including appendix; 3 figures and 5 tables. Under review for ICLR 2026
    </div>
    <details class="paper-abstract">
      Deploying black-box LLMs requires managing uncertainty in the absence of token-level probability or true labels. We propose introducing an unsupervised conformal inference framework for generation, which integrates: generative models, incorporating: (i) an LLM-compatible atypical score derived from response-embedding Gram matrix, (ii) UCP combined with a bootstrapping variant (BB-UCP) that aggregates residuals to refine quantile precision while maintaining distribution-free, finite-sample coverage, and (iii) conformal alignment, which calibrates a single strictness parameter $\tau$ so a user predicate (e.g., factuality lift) holds on unseen batches with probability $\ge 1-\alpha$. Across different benchmark datasets, our gates achieve close-to-nominal coverage and provide tighter, more stable thresholds than split UCP, while consistently reducing the severity of hallucination, outperforming lightweight per-response detectors with similar computational demands. The result is a label-free, API-compatible gate for test-time filtering that turns geometric signals into calibrated, goal-aligned decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01076v3">When Speculation Spills Secrets: Side Channels via Speculative Decoding In LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-26
    </div>
    <details class="paper-abstract">
      Deployed large language models (LLMs) often rely on speculative decoding, a technique that generates and verifies multiple candidate tokens in parallel, to improve throughput and latency. In this work, we reveal a new side-channel whereby input-dependent patterns of correct and incorrect speculations can be inferred by monitoring per-iteration token counts or packet sizes.We demonstrate that an adversary observing these patterns can fingerprint user queries with >90% accuracy across four speculative-decoding schemes, REST (100\%), LADE (up to 92%), BiLD (up to 95%), and EAGLE (up to 77.6%) and leak confidential datastore contents used for prediction at rates exceeding 25 tokens/sec. We evaluate the side-channel attacks in both research prototypes as well as the production-grade vLLM serving framework. To defend against these, we propose and evaluate a suite of mitigations, including packet padding and iteration-wise token aggregation.
    </details>
</div>
