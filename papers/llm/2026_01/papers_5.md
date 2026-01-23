# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13291v2">Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 36 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18085v3">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2. (v3): Added note to Sec 4.2 and Appendix A.2 specifying conditions for losslessness
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00097v2">The Agentic Leash: Extracting Causal Feedback Fuzzy Cognitive Maps with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 15 figures
    </div>
    <details class="paper-abstract">
      We design a large-language-model (LLM) agent that extracts causal feedback fuzzy cognitive maps (FCMs) from raw text. The causal learning or extraction process is agentic both because of the LLM's semi-autonomy and because ultimately the FCM dynamical system's equilibria drive the LLM agents to fetch and process causal text. The fetched text can in principle modify the adaptive FCM causal structure and so modify the source of its quasi-autonomy--its equilibrium limit cycles and fixed-point attractors. This bidirectional process endows the evolving FCM dynamical system with a degree of autonomy while still staying on its agentic leash. We show in particular that a sequence of three finely tuned system instructions guide an LLM agent as it systematically extracts key nouns and noun phrases from text, as it extracts FCM concept nodes from among those nouns and noun phrases, and then as it extracts or infers partial or fuzzy causal edges between those FCM nodes. We test this FCM generation on a recent essay about the promise of AI from the late diplomat and political theorist Henry Kissinger and his colleagues. This three-step process produced FCM dynamical systems that converged to the same equilibrium limit cycles as did the human-generated FCMs even though the human-generated FCM differed in the number of nodes and edges. A final FCM mixed generated FCMs from separate Gemini and ChatGPT LLM agents. The mixed FCM absorbed the equilibria of its dominant mixture component but also created new equilibria of its own to better approximate the underlying causal dynamical system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09182v1">Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted to AAAI 2026 Workshop on AI for Scientific Research (AI4Research)
    </div>
    <details class="paper-abstract">
      The rapid expansion of AI research has intensified the Reviewer Gap, threatening the peer-review sustainability and perpetuating a cycle of low-quality evaluations. This position paper critiques existing LLM approaches that automatically generate reviews and argues for a paradigm shift that positions LLMs as tools for assisting and educating human reviewers. We define the core principles of high-quality peer review and propose two complementary systems grounded in these foundations: (i) an LLM-assisted mentoring system that cultivates reviewers' long-term competencies, and (ii) an LLM-assisted feedback system that helps reviewers refine the quality of their reviews. This human-centered approach aims to strengthen reviewer expertise and contribute to building a more sustainable scholarly ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20224v4">Can Editing LLMs Inject Harm?</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted to Proceedings of AAAI 2026. The first two authors contributed equally. 7 pages for main paper, 31 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a new information channel. Meanwhile, one critical but under-explored question is: Is it possible to bypass the safety alignment and inject harmful information into LLMs stealthily? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the first risk, we find that editing attacks can inject both commonsense and long-tail misinformation into LLMs, and the effectiveness for the former one is particularly high. For the second risk, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can degrade the overall fairness. Then, we further illustrate the high stealthiness of editing attacks. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09159v1">LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes \emph{Isolation Kernel Embedding} or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). IKE is an ensemble of diverse (random) partitions, enabling robust estimation of ideal kernel in the LLM embedding space, thus reducing retrieval accuracy loss as the ensemble grows. Lightweight and based on binary encoding, it offers low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than LLM embeddings, while maintaining comparable or better accuracy. Compared to CSR and other compression methods, IKE consistently achieves the best balance between retrieval efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09152v1">PrivacyReasoner: Can LLM Emulate a Human-like Privacy Mind?</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      This paper introduces PRA, an AI-agent design for simulating how individual users form privacy concerns in response to real-world news. Moving beyond population-level sentiment analysis, PRA integrates privacy and cognitive theories to simulate user-specific privacy reasoning grounded in personal comment histories and contextual cues. The agent reconstructs each user's "privacy mind", dynamically activates relevant privacy memory through a contextual filter that emulates bounded rationality, and generates synthetic comments reflecting how that user would likely respond to new privacy scenarios. A complementary LLM-as-a-Judge evaluator, calibrated against an established privacy concern taxonomy, quantifies the faithfulness of generated reasoning. Experiments on real-world Hacker News discussions show that \PRA outperforms baseline agents in privacy concern prediction and captures transferable reasoning patterns across domains including AI, e-commerce, and healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09151v1">Interpretable Probability Estimation with LLMs via Shapley Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate potential to estimate the probability of uncertain events, by leveraging their extensive knowledge and reasoning capabilities. This ability can be applied to support intelligent decision-making across diverse fields, such as financial forecasting and preventive healthcare. However, directly prompting LLMs for probability estimation faces significant challenges: their outputs are often noisy, and the underlying predicting process is opaque. In this paper, we propose PRISM: Probability Reconstruction via Shapley Measures, a framework that brings transparency and precision to LLM-based probability estimation. PRISM decomposes an LLM's prediction by quantifying the marginal contribution of each input factor using Shapley values. These factor-level contributions are then aggregated to reconstruct a calibrated final estimate. In our experiments, we demonstrate PRISM improves predictive accuracy over direct prompting and other baselines, across multiple domains including finance, healthcare, and agriculture. Beyond performance, PRISM provides a transparent prediction pipeline: our case studies visualize how individual factors shape the final estimate, helping build trust in LLM-based decision support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09142v1">EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Shijian Ma and Yan Lin contributed equally. Corresponding author: Yan Lin
    </div>
    <details class="paper-abstract">
      Detecting evasive answers in earnings calls is critical for financial transparency, yet progress is hindered by the lack of large-scale benchmarks. We introduce EvasionBench, comprising 30,000 training samples and 1,000 human-annotated test samples (Cohen's Kappa 0.835) across three evasion levels. Our key contribution is a multi-model annotation framework leveraging a core insight: disagreement between frontier LLMs signals hard examples most valuable for training. We mine boundary cases where two strong annotators conflict, using a judge to resolve labels. This approach outperforms single-model distillation by 2.4 percent, with judge-resolved samples improving generalization despite higher training loss (0.421 vs 0.393) - evidence that disagreement mining acts as implicit regularization. Our trained model Eva-4B (4B parameters) achieves 81.3 percent accuracy, outperforming its base by 25 percentage points and approaching frontier LLM performance at a fraction of inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05603v2">Revisiting Human-vs-LLM judgments using the TREC Podcast Track</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Version 2: The paper has been accepted to appear at ECIR 2026
    </div>
    <details class="paper-abstract">
      Using large language models (LLMs) to annotate relevance is an increasingly important technique in the information retrieval community. While some studies demonstrate that LLMs can achieve high user agreement with ground truth (human) judgments, other studies have argued for the opposite conclusion. To the best of our knowledge, these studies have primarily focused on classic ad-hoc text search scenarios. In this paper, we conduct an analysis on user agreement between LLM and human experts, and explore the impact disagreement has on system rankings. In contrast to prior studies, we focus on a collection composed of audio files that are transcribed into two-minute segments -- the TREC 2020 and 2021 podcast track. We employ five different LLM models to re-assess all of the query-segment pairs, which were originally annotated by TREC assessors. Furthermore, we re-assess a small subset of pairs where LLM and TREC assessors have the highest disagreement, and found that the human experts tend to agree with LLMs more than with the TREC assessors. Our results reinforce the previous insights of Sormunen in 2002 -- that relying on a single assessor leads to lower user agreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09129v1">KryptoPilot: An Open-World Knowledge-Augmented LLM Agent for Automated Cryptographic Exploitation</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 14 Pages,4 figures
    </div>
    <details class="paper-abstract">
      Capture-the-Flag (CTF) competitions play a central role in modern cybersecurity as a platform for training practitioners and evaluating offensive and defensive techniques derived from real-world vulnerabilities. Despite recent advances in large language models (LLMs), existing LLM-based agents remain ineffective on high-difficulty cryptographic CTF challenges, which require precise cryptanalytic knowledge, stable long-horizon reasoning, and disciplined interaction with specialized toolchains. Through a systematic exploratory study, we show that insufficient knowledge granularity, rather than model reasoning capacity, is a primary factor limiting successful cryptographic exploitation: coarse or abstracted external knowledge often fails to support correct attack modeling and implementation. Motivated by this observation, we propose KryptoPilot, an open-world knowledge-augmented LLM agent for automated cryptographic exploitation. KryptoPilot integrates dynamic open-world knowledge acquisition via a Deep Research pipeline, a persistent workspace for structured knowledge reuse, and a governance subsystem that stabilizes reasoning through behavioral constraints and cost-aware model routing. This design enables precise knowledge alignment while maintaining efficient reasoning across heterogeneous subtasks. We evaluate KryptoPilot on two established CTF benchmarks and in six real-world CTF competitions. KryptoPilot achieves a complete solve rate on InterCode-CTF, solves between 56 and 60 percent of cryptographic challenges on the NYU-CTF benchmark, and successfully solves 26 out of 33 cryptographic challenges in live competitions, including multiple earliest-solved and uniquely-solved instances. These results demonstrate the necessity of open-world, fine-grained knowledge augmentation and governed reasoning for scaling LLM-based agents to real-world cryptographic exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09760v1">Investigating Tool-Memory Conflicts in Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 R2-FM Workshop @ ICML 2025
    </div>
    <details class="paper-abstract">
      Tool-augmented large language models (LLMs) have powered many applications. However, they are likely to suffer from knowledge conflict. In this paper, we propose a new type of knowledge conflict -- Tool-Memory Conflict (TMC), where the internal parametric knowledge contradicts with the external tool knowledge for tool-augmented LLMs. We find that existing LLMs, though powerful, suffer from TMC, especially on STEM-related tasks. We also uncover that under different conditions, tool knowledge and parametric knowledge may be prioritized differently. We then evaluate existing conflict resolving techniques, including prompting-based and RAG-based methods. Results show that none of these approaches can effectively resolve tool-memory conflicts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06953v2">Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17217v3">Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. We release the code and generated data at: https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09041v1">Can LLMs interpret figurative language as humans do?: surface-level vs representational similarity</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models generate judgments that resemble those of humans. Yet the extent to which these models align with human judgments in interpreting figurative and socially grounded language remains uncertain. To investigate this, human participants and four instruction-tuned LLMs of different sizes (GPT-4, Gemma-2-9B, Llama-3.2, and Mistral-7B) rated 240 dialogue-based sentences representing six linguistic traits: conventionality, sarcasm, funny, emotional, idiomacy, and slang. Each of the 240 sentences was paired with 40 interpretive questions, and both humans and LLMs rated these sentences on a 10-point Likert scale. Results indicated that humans and LLMs aligned at the surface level with humans, but diverged significantly at the representational level, especially in interpreting figurative sentences involving idioms and Gen Z slang. GPT-4 most closely approximates human representational patterns, while all models struggle with context-dependent and socio-pragmatic expressions like sarcasm, slang, and idiomacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09039v1">An Information-Theoretic Perspective on LLM Tokenizers</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) tokenizers act as structured compressors: by mapping text to discrete token sequences, they determine token count (and thus compute and context usage) and the statistical structure seen by downstream models. Despite their central role in LLM pipelines, the link between tokenization, compression efficiency and induced structure is not well understood. We empirically demonstrate that tokenizer training scale redistributes entropy: as training data grows, the token stream becomes more diverse in aggregate (higher unigram entropy) yet markedly more predictable in-context (lower higher-order conditional entropies), indicating that tokenization absorbs substantial short-range regularity although these gains degrade under train-test domain mismatch. To ground these observations, we first benchmark i) pretrained GPT-family tokenizers as black-box compressors across various domains, and ii) learned tokenizers across configurations spanning vocabulary size, training scale, and domain. Next, we study tokenization as a transform for universal compression and introduce a compression-aware BPE variant. Finally, we adopt a channel lens and introduce capacity-utilization metrics to analyze tokenizer behaviour and outline implications for downstream modeling. Put together, our results expose various trade-offs between compression, induced structure, and robustness under domain shift, and motivate principled, compression-aware tokenizer design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19670v4">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
    </div>
    <details class="paper-abstract">
      We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09913v1">Continuum Memory Architectures for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 10 Pages
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become the default strategy for providing large language model (LLM) agents with contextual knowledge. Yet RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent. We define the \textit{Continuum Memory Architecture} (CMA), a class of systems that maintain and update internal state across interactions through persistent storage, selective retention, associative routing, temporal chaining, and consolidation into higher-order abstractions. Rather than disclosing implementation specifics, we specify the architectural requirements CMA imposes and show consistent behavioral advantages on tasks that expose RAG's structural inability to accumulate, mutate, or disambiguate memory. The empirical probes (knowledge updates, temporal association, associative recall, contextual disambiguation) demonstrate that CMA is a necessary architectural primitive for long-horizon agents while highlighting open challenges around latency, drift, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09905v1">Self-reflection in Automated Qualitative Coding: Improving Text Annotation through Secondary LLM Critique</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) allow for sophisticated qualitative coding of large datasets, but zero- and few-shot classifiers can produce an intolerable number of errors, even with careful, validated prompting. We present a simple, generalizable two-stage workflow: an LLM applies a human-designed, LLM-adapted codebook; a secondary LLM critic performs self-reflection on each positive label by re-reading the source text alongside the first model's rationale and issuing a final decision. We evaluate this approach on six qualitative codes over 3,000 high-content emails from Apache Software Foundation project evaluation discussions. Our human-derived audit of 360 positive annotations (60 passages by six codes) found that the first-line LLM had a false-positive rate of 8% to 54%, despite F1 scores of 0.74 and 1.00 in testing. Subsequent recoding of all stage-one annotations via a second self-reflection stage improved F1 by 0.04 to 0.25, bringing two especially poor performing codes up to 0.69 and 0.79 from 0.52 and 0.55 respectively. Our manual evaluation identified two recurrent error classes: misinterpretation (violations of code definitions) and meta-discussion (debate about a project evaluation criterion mistaken for its use as a decision justification). Code-specific critic clauses addressing observed failure modes were especially effective with testing and refinement, replicating the codebook-adaption process for LLM interpretation in stage-one. We explain how favoring recall in first-line LLM annotation combined with secondary critique delivers precision-first, compute-light control. With human guidance and validation, self-reflection slots into existing LLM-assisted annotation pipelines to reduce noise and potentially salvage unusable classifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.21621v2">The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18691v2">Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      This study is the first to investigate LLM comprehension capabilities over long-context (LC), clinically relevant medical Question Answering (QA) beyond MCQA. Our comprehensive approach considers a range of settings based on content inclusion of varying size and relevance, LLM models of different capabilities and a variety of datasets across task formulations. We reveal insights on model size effects and their limitations, underlying memorization issues and the benefits of reasoning models, while demonstrating the value and challenges of leveraging the full long patient's context. Importantly, we examine the effect of Retrieval Augmented Generation (RAG) on medical LC comprehension, showcasing best settings in single versus multi-document QA datasets. We shed light into some of the evaluation aspects using a multi-faceted approach uncovering common metric challenges. Our quantitative analysis reveals challenging cases where RAG excels while still showing limitations in cases requiring temporal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23281v3">MathArena: Evaluating LLMs on Uncontaminated Math Competitions</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On IMO 2025, top models achieve slightly less than 40%, demonstrating both notable progress and significant room for improvement. So far, we have evaluated over $50$ models across seven competitions, totaling $162$ problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01930v4">Robust LLM Alignment via Distributionally Robust Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      A major challenge in aligning large language models (LLMs) with human preferences is the issue of distribution shift. LLM alignment algorithms rely on static preference datasets, assuming that they accurately represent real-world user preferences. However, user preferences vary significantly across geographical regions, demographics, linguistic patterns, and evolving cultural trends. This preference distribution shift leads to catastrophic alignment failures in many real-world applications. We address this problem using the principled framework of distributionally robust optimization, and develop two novel distributionally robust direct preference optimization (DPO) algorithms, namely, Wasserstein DPO (WDPO) and Kullback-Leibler DPO (KLDPO). We characterize the sample complexity of learning the optimal policy parameters for WDPO and KLDPO. Moreover, we propose scalable gradient descent-style learning algorithms by developing suitable approximations for the challenging minimax loss functions of WDPO and KLDPO. Our empirical experiments using benchmark data sets and LLMs demonstrate the superior performance of WDPO and KLDPO in substantially improving the alignment when there is a preference distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09865v1">Advancing Model Refinement: Muon-Optimized Distillation and Quantization for LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable advanced natural language processing but face deployment challenges on resource-constrained edge devices due to high computational, memory, and energy demands. Optimizing these models requires addressing three key challenges: acquiring task-specific data, fine-tuning for performance, and compressing models to accelerate inference while reducing resource demands. We propose an integrated framework combining GPTQ-based quantization, low-rank adaptation (LoRA), and a specialized data distillation process to significantly reduce model size and complexity while preserving or enhancing task-specific performance. By leveraging data distillation, knowledge distillation via Kullback-Leibler divergence, Bayesian hyperparameter optimization, and the Muon optimizer, our pipeline achieves up to 2x memory compression (e.g., reducing a 6GB model to 3GB) and enables efficient inference for specialized tasks. Empirical results demonstrate superior performance on standard LLM benchmarks compared to GPTQ quantization alone, with the Muon optimizer notably enhancing fine-tuned models' resistance to accuracy decay during quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.03296v4">APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads. We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like vLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 72% (T4) and 37% (A10) higher throughput in long-output settings. APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09853v1">MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Real-world health questions from patients often unintentionally embed false assumptions or premises. In such cases, safe medical communication typically involves redirection: addressing the implicit misconception and then responding to the underlying patient context, rather than the original question. While large language models (LLMs) are increasingly being used by lay users for medical advice, they have not yet been tested for this crucial competency. Therefore, in this work, we investigate how LLMs react to false premises embedded within real-world health questions. We develop a semi-automated pipeline to curate MedRedFlag, a dataset of 1100+ questions sourced from Reddit that require redirection. We then systematically compare responses from state-of-the-art LLMs to those from clinicians. Our analysis reveals that LLMs often fail to redirect problematic questions, even when the problematic premise is detected, and provide answers that could lead to suboptimal medical decision making. Our benchmark and results reveal a novel and substantial gap in how LLMs perform under the conditions of real-world health communication, highlighting critical safety concerns for patient-facing medical AI systems. Code and dataset are available at https://github.com/srsambara-1/MedRedFlag.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09822v1">LLM-Based Agentic Systems for Software Engineering: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted to GenSE 2026 workshop
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs), complex Software Engineering (SE) tasks require more collaborative and specialized approaches. This concept paper systematically reviews the emerging paradigm of LLM-based multi-agent systems, examining their applications across the Software Development Life Cycle (SDLC), from requirements engineering and code generation to static code checking, testing, and debugging. We delve into a wide range of topics such as language model selection, SE evaluation benchmarks, state-of-the-art agentic frameworks and communication protocols. Furthermore, we identify key challenges and outline future research opportunities, with a focus on multi-agent orchestration, human-agent coordination, computational cost optimization, and effective data collection. This work aims to provide researchers and practitioners with valuable insights into the current forefront landscape of agentic systems within the software engineering domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09695v1">How well LLM-based test generation techniques perform with newer LLM versions?</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Models (LLMs) has strongly impacted software engineering, leading to a growing number of studies on automated unit test generation. However, the standalone use of LLMs without post-processing has proven insufficient, often producing tests that fail to compile or achieve high coverage. Several techniques have been proposed to address these issues, reporting improvements in test compilation and coverage. While important, LLM-based test generation techniques have been evaluated against relatively weak baselines (for todays' standards), i.e., old LLM versions and relatively weak prompts, which may exacerbate the performance contribution of the approaches. In other words, stronger (newer) LLMs may obviate any advantage these techniques bring. We investigate this issue by replicating four state-of-the-art LLM-based test generation tools, HITS, SymPrompt, TestSpark, and CoverUp that include engineering components aimed at guiding the test generation process through compilation and execution feedback, and evaluate their relative effectiveness and efficiency over a plain LLM test generation method. We integrate current LLM versions in all approaches and run an experiment on 393 classes and 3,657 methods. Our results show that the plain LLM approach can outperform previous state-of-the-art approaches in all test effectiveness metrics we used: line coverage (by 17.72%), branch coverage (by 19.80%) and mutation score (by 20.92%), and it does so at a comparable cost (LLM queries). We also observe that the granularity at which the plain LLM is applied has a significant impact on the cost. We therefore propose targeting first the program classes, where test generation is more efficient, and then the uncovered methods to reduce the number of LLM requests. This strategy achieves comparable (slightly higher) effectiveness while requiring about 20% fewer LLM requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09694v1">LLMs can Compress LLMs: Adaptive Pruning by Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 17 Pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09692v1">Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Code: https://github.com/tianyiniu/RoutingGenData
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) routers dynamically select optimal models for given inputs. Existing approaches typically assume access to ground-truth labeled data, which is often unavailable in practice, especially when user request distributions are heterogeneous and unknown. We introduce Routing with Generated Data (RGD), a challenging setting in which routers are trained exclusively on generated queries and answers produced from high-level task descriptions by generator LLMs. We evaluate query-answer routers (using both queries and labels) and query-only routers across four diverse benchmarks and 12 models, finding that query-answer routers degrade faster than query-only routers as generator quality decreases. Our analysis reveals two crucial characteristics of effective generators: they must accurately respond to their own questions, and their questions must produce sufficient performance differentiation among the model pool. We then show how filtering for these characteristics can improve the quality of generated data. We further propose CASCAL, a novel query-only router that estimates model correctness through consensus voting and identifies model-specific skill niches via hierarchical clustering. CASCAL is substantially more robust to generator quality, outperforming the best query-answer router by 4.6% absolute accuracy when trained on weak generator data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22979v2">OptiMind: Teaching LLMs to Think Like Optimization Experts</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Mathematical programming -- the task of expressing operations and decision-making problems in precise mathematical language -- is fundamental across domains, yet remains a skill-intensive process requiring operations research expertise. Recent advances in large language models for complex reasoning have spurred interest in automating this task, translating natural language into executable optimization models. Current approaches, however, achieve limited accuracy, hindered by scarce and noisy training data without leveraging domain knowledge. In this work, we systematically integrate optimization expertise to improve formulation accuracy for mixed-integer linear programming, a key family of mathematical programs. Our OptiMind framework leverages semi-automated, class-based error analysis to guide both training and inference, explicitly preventing common mistakes within each optimization class. Our resulting fine-tuned LLM significantly improves formulation accuracy by 20.7% across multiple optimization benchmarks, with consistent gains under test-time scaling methods such as self-consistency and multi-turn feedback, enabling further progress toward robust LLM-assisted optimization formulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.03471v5">Template-Based Probes Are Imperfect Lenses for Counterfactual Bias Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 22 Pages, 6 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Bias in large language models (LLMs) has many forms, from overt discrimination to implicit stereotypes. Counterfactual bias evaluation is a widely used approach to quantifying bias and often relies on template-based probes that explicitly state group membership. It aims to measure whether the outcome of a task performed by an LLM is invariant to a change in group membership. In this work, we find that template-based probes can introduce systematic distortions in bias measurements. Specifically, we consistently find that such probes suggest that LLMs classify text associated with White race as negative at disproportionately elevated rates. This is observed consistently across a large collection of LLMs, over several diverse template-based probes, and with different classification approaches. We hypothesize that this arises artificially due to linguistic asymmetries present in LLM pretraining data, in the form of markedness, (e.g., Black president vs. president) and templates used for bias measurement (e.g., Black president vs. White president). These findings highlight the need for more rigorous methodologies in counterfactual bias evaluation, ensuring that observed disparities reflect genuine biases rather than artifacts of linguistic conventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05755v2">VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09635v1">LLM for Large-Scale Optimization Model Auto-Formulation: A Lightweight Few-Shot Learning Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Updated version of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027
    </div>
    <details class="paper-abstract">
      Large-scale optimization is a key backbone of modern business decision-making. However, building these models is often labor-intensive and time-consuming. We address this by proposing LEAN-LLM-OPT, a LightwEight AgeNtic workflow construction framework for LLM-assisted large-scale OPTimization auto-formulation. LEAN-LLM-OPT takes as input a problem description together with associated datasets and orchestrates a team of LLM agents to produce an optimization formulation. Specifically, upon receiving a query, two upstream LLM agents dynamically construct a workflow that specifies, step-by-step, how optimization models for similar problems can be formulated. A downstream LLM agent then follows this workflow to generate the final output. Leveraging LLMs' text-processing capabilities and common modeling practices, the workflow decomposes the modeling task into a sequence of structured sub-tasks and offloads mechanical data-handling operations to auxiliary tools. This design alleviates the downstream agent's burden related to planning and data handling, allowing it to focus on the most challenging components that cannot be readily standardized. Extensive simulations show that LEAN-LLM-OPT, instantiated with GPT-4.1 and the open source gpt-oss-20B, achieves strong performance on large-scale optimization modeling tasks and is competitive with state-of-the-art approaches. In addition, in a Singapore Airlines choice-based revenue management use case, LEAN-LLM-OPT demonstrates practical value by achieving leading performance across a range of scenarios. Along the way, we introduce Large-Scale-OR and Air-NRM, the first comprehensive benchmarks for large-scale optimization auto-formulation. The code and data of this work is available at https://github.com/CoraLiang01/lean-llm-opt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v1">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00056v3">MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 This paper is accepted to Neural Information Processing Systems (NeurIPS) 2025
    </div>
    <details class="paper-abstract">
      The substantial memory demands of pre-training and fine-tuning large language models (LLMs) require memory-efficient optimization algorithms. One promising approach is layer-wise optimization, which treats each transformer block as a single layer and optimizes it sequentially, while freezing the other layers to save optimizer states and activations. Although effective, these methods ignore the varying importance of the modules within each layer, leading to suboptimal performance. Moreover, layer-wise sampling provides only limited memory savings, as at least one full layer must remain active during optimization. To overcome these limitations, we propose Module-wise Importance SAmpling (MISA), a novel method that divides each layer into smaller modules and assigns importance scores to each module. MISA uses a weighted random sampling mechanism to activate modules, provably reducing gradient variance compared to layer-wise sampling. Additionally, we establish an \(\mathcal{O}(1/\sqrt{K})\) convergence rate under non-convex and stochastic conditions, where $K$ is the total number of block updates, and provide a detailed memory analysis showcasing MISA's superiority over existing baseline methods. Experiments on diverse learning tasks validate the effectiveness of MISA. Source code is available at https://github.com/pkumelon/MISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19385v2">Beyond Uniform SVD:Dual-Level Optimization across Columns and Modules for LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Low-rank decomposition, particularly Singular Value Decomposition (SVD), is a pivotal technique for mitigating the storage and computational demands of Large Language Models (LLMs). However, prevalent SVD-based approaches overlook the critical phenomenon that decomposition errors exhibit significant disparity across different components of the parameter matrix, often leading to suboptimal approximation. Furthermore, existing methods lack a direct metric to evaluate the importance of individual weight matrices. To address these limitations, we propose Duo-SVD (Dual-level Optimization SVD), a novel training-free framework that synergizes optimization at both the column and the module levels. First, Duo-SVD incorporates a Column-Preserving Strategy that explicitly retains columns exhibiting high decomposition errors, while applying low-rank approximation solely to those with lower errors. Second, at the module level, we employ a Module-Adaptive Allocation Strategy that formulates ratio allocation as a global constrained optimization problem based on perturbation-induced model deviation. Extensive experiments demonstrate that Duo-SVD consistently outperforms state-of-the-art SVD-based baselines and structured pruning methods, establishing it as a superior paradigm for efficient LLM compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07994v2">DYCP: Dynamic Context Pruning for Long-Form Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 Accepted (B) to TACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit increased response latency and degraded answer quality as dialogue length grows, making effective context management essential. However, existing methods rely on extra LLM calls to build memory or perform offline memory construction without considering the current user utterance, which can introduce inefficiencies or disrupt conversational continuity. We introduce DyCP, a lightweight context management method that dynamically segment and retrieve relevant memory at query time. It preserves the sequential structure of dialogue without predefined topic boundaries and supports efficient, adaptive context retrieval. Across three long-form dialogue benchmarks, LoCoMo, MT-Bench+, and SCM4LLMs, and multiple LLMs, DyCP consistently improves answer quality while reducing response latency. We also examine the gap between modern LLMs' expanded context windows and their actual long-context processing capacity, highlighting the continued importance of effective context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06821v4">Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 37 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09527v1">Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 15 pages, 18 tables, 7 figures. Includes link to GitHub repository and Docker image for reproducibility
    </div>
    <details class="paper-abstract">
      SMEs increasingly seek alternatives to cloud LLM APIs, which raise data privacy concerns. Dedicated cloud GPU instances offer improved privacy but with limited guarantees and ongoing costs, while professional on-premise hardware (A100, H100) remains prohibitively expensive. We present a systematic evaluation of NVIDIA's Blackwell consumer GPUs (RTX 5060 Ti, 5070 Ti, 5090) for production LLM inference, benchmarking four open-weight models (Qwen3-8B, Gemma3-12B, Gemma3-27B, GPT-OSS-20B) across 79 configurations spanning quantization formats (BF16, W4A16, NVFP4, MXFP4), context lengths (8k-64k), and three workloads: RAG, multi-LoRA agentic serving, and high-concurrency APIs. The RTX 5090 delivers 3.5-4.6x higher throughput than the 5060 Ti with 21x lower latency for RAG, but budget GPUs achieve the highest throughput-per-dollar for API workloads with sub-second latency. NVFP4 quantization provides 1.6x throughput over BF16 with 41% energy reduction and only 2-4% quality loss. Self-hosted inference costs $0.001-0.04 per million tokens (electricity only), which is 40-200x cheaper than budget-tier cloud APIs, with hardware breaking even in under four months at moderate volume (30M tokens/day). Our results show that consumer GPUs can reliably replace cloud inference for most SME workloads, except latency-critical long-context RAG, where high-end GPUs remain essential. We provide deployment guidance and release all benchmark data for reproducible SME-scale deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09503v1">What Do LLM Agents Know About Their World? Task2Quiz: A Paradigm for Studying Environment Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in complex decision-making and tool-use tasks, yet their ability to generalize across varying environments remains a under-examined concern. Current evaluation paradigms predominantly rely on trajectory-based metrics that measure task success, while failing to assess whether agents possess a grounded, transferable model of the environment. To address this gap, we propose Task-to-Quiz (T2Q), a deterministic and automated evaluation paradigm designed to decouple task execution from world-state understanding. We instantiate this paradigm in T2QBench, a suite comprising 30 environments and 1,967 grounded QA pairs across multiple difficulty levels. Our extensive experiments reveal that task success is often a poor proxy for environment understanding, and that current memory machanism can not effectively help agents acquire a grounded model of the environment. These findings identify proactive exploration and fine-grained state representation as primary bottlenecks, offering a robust foundation for developing more generalizable autonomous agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20975v2">SPOT!: Map-Guided LLM Agent for Unsupervised Multi-CCTV Dynamic Object Tracking</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 33 pages, 27figures
    </div>
    <details class="paper-abstract">
      CCTV-based vehicle tracking systems face structural limitations in continuously connecting the trajectories of the same vehicle across multiple camera environments. In particular, blind spots occur due to the intervals between CCTVs and limited Fields of View (FOV), which leads to object ID switching and trajectory loss, thereby reducing the reliability of real-time path prediction. This paper proposes SPOT (Spatial Prediction Over Trajectories), a map-guided LLM agent capable of tracking vehicles even in blind spots of multi-CCTV environments without prior training. The proposed method represents road structures (Waypoints) and CCTV placement information as documents based on 2D spatial coordinates and organizes them through chunking techniques to enable real-time querying and inference. Furthermore, it transforms the vehicle's position into the actual world coordinate system using the relative position and FOV information of objects observed in CCTV images. By combining map spatial information with the vehicle's moving direction, speed, and driving patterns, a beam search is performed at the intersection level to derive candidate CCTV locations where the vehicle is most likely to enter after the blind spot. Experimental results based on the CARLA simulator in a virtual city environment confirmed that the proposed method accurately predicts the next appearing CCTV even in blind spot sections, maintaining continuous vehicle trajectories more effectively than existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05984v2">Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09496v1">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Search and recommendation (S&R) are core to online platforms, addressing explicit intent through queries and modeling implicit intent from behaviors, respectively. Their complementary roles motivate a unified modeling paradigm. Early studies to unify S&R adopt shared encoders with task-specific heads, while recent efforts reframe item ranking in both S&R as conditional generation. The latter holds particular promise, enabling end-to-end optimization and leveraging the semantic understanding of LLMs. However, existing methods rely on full fine-tuning, which is computationally expensive and limits scalability. Parameter-efficient fine-tuning (PEFT) offers a more practical alternative but faces two critical challenges in unifying S&R: (1) gradient conflicts across tasks due to divergent optimization objectives, and (2) shifts in user intent understanding caused by overfitting to fine-tuning data, which distort general-domain knowledge and weaken LLM reasoning. To address the above issues, we propose Gradient Multi-Subspace Tuning (GEMS), a novel framework that unifies S&R with LLMs while alleviating gradient conflicts and preserving general-domain knowledge. GEMS introduces (1) \textbf{Multi-Subspace Decomposition}, which disentangles shared and task-specific optimization signals into complementary low-rank subspaces, thereby reducing destructive gradient interference, and (2) \textbf{Null-Space Projection}, which constrains parameter updates to a subspace orthogonal to the general-domain knowledge space, mitigating shifts in user intent understanding. Extensive experiments on benchmark datasets show that GEMS consistently outperforms the state-of-the-art baselines across both search and recommendation tasks, achieving superior effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19179v2">CascadeInfer: Low-Latency and Load-Balanced LLM Serving via Length-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Efficiently harnessing GPU compute is critical to improving user experience and reducing operational costs in large language model (LLM) services. However, current inference engine schedulers overlook the attention backend's sensitivity to request-length heterogeneity within a batch. As state-of-the-art models now support context windows exceeding 128K tokens, this once-tolerable inefficiency has escalated into a primary system bottleneck, causing severe performance degradation through GPU underutilization and increased latency. We present CascadeInfer, a runtime system that dynamically reschedules requests across multiple instances serving the same LLM to mitigate per-instance length heterogeneity. CascadeInfer partitions these instances into length-specialized groups, each handling requests within a designated length range, naturally forming a pipeline as requests flow through them. CascadeInfer devises a dynamic programming algorithm to efficiently find the stage partition with the best QoE, employs runtime range refinement together with decentralized load (re)balance both across and within groups, achieving a balanced and efficient multi-instance service. Our evaluation shows that, under the same configuration, CascadeInfer reduces end-to-end latency by up to 67% and tail latency by up to 69%, while improving overall system throughput by up to 2.89 times compared to the state-of-the-art multi-instance scheduling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09448v1">Population-Aligned Audio Reproduction With LLM-Based Equalizers</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 12 pages, 13 figures, 2 tables, IEEE JSTSP journal submission under first revision
    </div>
    <details class="paper-abstract">
      Conventional audio equalization is a static process that requires manual and cumbersome adjustments to adapt to changing listening contexts (e.g., mood, location, or social setting). In this paper, we introduce a Large Language Model (LLM)-based alternative that maps natural language text prompts to equalization settings. This enables a conversational approach to sound system control. By utilizing data collected from a controlled listening experiment, our models exploit in-context learning and parameter-efficient fine-tuning techniques to reliably align with population-preferred equalization settings. Our evaluation methods, which leverage distributional metrics that capture users' varied preferences, show statistically significant improvements in distributional alignment over random sampling and static preset baselines. These results indicate that LLMs could function as "artificial equalizers," contributing to the development of more accessible, context-aware, and expert-level audio tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06627v2">Burn-After-Use for Preventing Data Leakage through a Secure Multi-Tenant Architecture in Enterprise LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-14
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This study presents a Secure Multi-Tenant Architecture (SMTA) combined with a novel concept Burn-After-Use (BAU) mechanism for enterprise LLM environments to effectively prevent data leakage. As institutions increasingly adopt LLMs across departments, the risks of data leakage have become a critical security and compliance concern. The proposed SMTA isolates LLM instances across departments and enforces rigorous context ownership boundaries within an internally deployed infrastructure. The BAU mechanism introduces data confidentiality by enforcing ephemeral conversational contexts that are automatically destroyed after use, preventing cross-session or cross-user inference. The evaluation to SMTA and BAU is through two sets of realistic and reproducible experiments comprising of 127 test iterations. One aspect of this experiment is to assess prompt-based and semantic leakage attacks in a multi-tenant architecture (Appendix A) across 55 infrastructure-level attack tests, including vector-database credential compromise and shared logging pipeline exposure. SMTA achieves 92% defense success rate, demonstrating strong semantic isolation while highlighting residual risks from credential misconfiguration and observability pipelines. Another aspect is to evaluate the robustness of BAU under realistic failure scenarios (Appendix B) using four empirical metrics: Local Residual Persistence Rate (LRPR), Remote Residual Persistence Rate (RRPR), Image Frame Exposure Rate (IFER), and Burn Timer Persistence Rate (BTPR). Across 72 test iterations, BAU achieves a 76.75% success rate in mitigating post-session leakage threats across the client, server, application, infrastructure, and cache layers. These results show that SMTA and BAU together enforce strict isolation, complete session ephemerality, strong confidentiality guarantees, non-persistence, and policy-aligned behavior for enterprise LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08415v2">Regulatory gray areas of LLM Terms</a></div>
    <div class="paper-meta">
      📅 2026-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into academic research pipelines; however, the Terms of Service governing their use remain under-examined. We present a comparative analysis of the Terms of Service of five major LLM providers (Anthropic, DeepSeek, Google, OpenAI, and xAI) collected in November 2025. Our analysis reveals substantial variation in the stringency and specificity of usage restrictions for general users and researchers. We identify specific complexities for researchers in security research, computational social sciences, and psychological studies. We identify `regulatory gray areas' where Terms of Service create uncertainty for legitimate use. We contribute a publicly available resource comparing terms across platforms (OSF) and discuss implications for general users and researchers navigating this evolving landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08829v1">Modeling LLM Agent Reviewer Dynamics in Elo-Ranked Review System</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 In submission. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      In this work, we explore the Large Language Model (LLM) agent reviewer dynamics in an Elo-ranked review system using real-world conference paper submissions. Multiple LLM agent reviewers with different personas are engage in multi round review interactions moderated by an Area Chair. We compare a baseline setting with conditions that incorporate Elo ratings and reviewer memory. Our simulation results showcase several interesting findings, including how incorporating Elo improves Area Chair decision accuracy, as well as reviewers' adaptive review strategy that exploits our Elo system without improving review effort. Our code is available at https://github.com/hsiangwei0903/EloReview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08773v1">Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 46 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation for software engineering often relies on vector similarity search, which captures topical similarity but can fail on multi-hop architectural reasoning such as controller to service to repository chains, interface-driven wiring, and inheritance. This paper benchmarks three retrieval pipelines on Java codebases (Shopizer, with additional runs on ThingsBoard and OpenMRS Core): (A) vector-only No-Graph RAG, (B) an LLM-generated knowledge graph RAG (LLM-KB), and (C) a deterministic AST-derived knowledge graph RAG (DKB) built with Tree-sitter and bidirectional traversal. Using 15 architecture and code-tracing queries per repository, we measure indexing time, query latency, corpus coverage, cost, and answer correctness. DKB builds its graph in seconds, while LLM-KB requires much longer graph generation. LLM-KB also shows indexing incompleteness: on Shopizer, 377 files are skipped or missed, reducing embedded chunk coverage and graph size compared to DKB. End-to-end cost is modest for DKB relative to the vector-only baseline but much higher for LLM-KB, especially as repository scale increases. Query latency is similar for No-Graph and DKB, while LLM-KB is slower and more variable. On the Shopizer question suite, DKB achieves the highest correctness, LLM-KB is close behind, and the vector-only baseline performs worst on upstream architectural queries and has the highest hallucination risk. Overall, deterministic AST-derived graphs provide more reliable coverage and multi-hop grounding than LLM-extracted graphs at substantially lower indexing cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05467v2">STELP: Secure Transpilation and Execution of LLM-Generated Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08763v1">Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08742v1">Inferring Latent Intentions: Attributional Natural Language Inference in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Attributional inference, the ability to predict latent intentions behind observed actions, is a critical yet underexplored capability for large language models (LLMs) operating in multi-agent environments. Traditional natural language inference (NLI), in fact, fails to capture the nuanced, intention-driven reasoning essential for complex interactive systems. To address this gap, we introduce Attributional NLI (Att-NLI), a framework that extends NLI with principles from social psychology to assess an agent's capacity for abductive intentional inference (generating hypotheses about latent intentions), and subsequent deductive verification (drawing valid logical conclusions). We instantiate Att-NLI via a textual game, Undercover-V, experimenting with three types of LLM agents with varying reasoning capabilities and access to external tools: a standard NLI agent using only deductive inference, an Att-NLI agent employing abductive-deductive inference, and a neuro-symbolic Att-NLI agent performing abductive-deductive inference with external theorem provers. Extensive experiments demonstrate a clear hierarchy of attributional inference capabilities, with neuro-symbolic agents consistently outperforming others, achieving an average win rate of 17.08%. Our results underscore the role that Att-NLI can play in developing agents with sophisticated reasoning capabilities, highlighting, at the same time, the potential impact of neuro-symbolic AI in building rational LLM agents acting in multi-agent environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08739v1">PrivGemo: Privacy-Preserving Dual-Tower Graph Retrieval for Empowering LLM Reasoning with Memory Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs) provide structured evidence that can ground large language model (LLM) reasoning for knowledge-intensive question answering. However, many practical KGs are private, and sending retrieved triples or exploration traces to closed-source LLM APIs introduces leakage risk. Existing privacy treatments focus on masking entity names, but they still face four limitations: structural leakage under semantic masking, uncontrollable remote interaction, fragile multi-hop and multi-entity reasoning, and limited experience reuse for stability and efficiency. To address these issues, we propose PrivGemo, a privacy-preserving retrieval-augmented framework for KG-grounded reasoning with memory-guided exposure control. PrivGemo uses a dual-tower design to keep raw KG knowledge local while enabling remote reasoning over an anonymized view that goes beyond name masking to limit both semantic and structural exposure. PrivGemo supports multi-hop, multi-entity reasoning by retrieving anonymized long-hop paths that connect all topic entities, while keeping grounding and verification on the local KG. A hierarchical controller and a privacy-aware experience memory further reduce unnecessary exploration and remote interactions. Comprehensive experiments on six benchmarks show that PrivGemo achieves overall state-of-the-art results, outperforming the strongest baseline by up to 17.1%. Furthermore, PrivGemo enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08734v1">TerraFormer: Automated Infrastructure-as-Code with LLMs Fine-Tuned via Policy-Guided Verifier Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 The paper has been published at the 2026 IEEE/ACM 48th International Conference on Software Engineering (ICSE 2026), Rio de Janeiro, Brazil, April 12-18, 2026
    </div>
    <details class="paper-abstract">
      Automating Infrastructure-as-Code (IaC) is challenging, and large language models (LLMs) often produce incorrect configurations from natural language (NL). We present TerraFormer, a neuro-symbolic framework for IaC generation and mutation that combines supervised fine-tuning with verifier-guided reinforcement learning, using formal verification tools to provide feedback on syntax, deployability, and policy compliance. We curate two large, high-quality NL-to-IaC datasets, TF-Gen (152k instances) and TF-Mutn (52k instances), via multi-stage verification and iterative LLM self-correction. Evaluations against 17 state-of-the-art LLMs, including ~50x larger models like Sonnet 3.7, DeepSeek-R1, and GPT-4.1, show that TerraFormer improves correctness over its base LLM by 15.94% on IaC-Eval, 11.65% on TF-Gen (Test), and 19.60% on TF-Mutn (Test). It outperforms larger models on both TF-Gen (Test) and TF-Mutn (Test), ranks third on IaC-Eval, and achieves top best-practices and security compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08691v1">LLMs in Code Vulnerability Analysis: A Proof of Concept</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted for publication at the Fourth International Workshop on Software Vulnerability Management (SVM 2026) co-located with Intenational Conference in Software Engineering (ICSE 2026)
    </div>
    <details class="paper-abstract">
      Context: Traditional software security analysis methods struggle to keep pace with the scale and complexity of modern codebases, requiring intelligent automation to detect, assess, and remediate vulnerabilities more efficiently and accurately. Objective: This paper explores the incorporation of code-specific and general-purpose Large Language Models (LLMs) to automate critical software security tasks, such as identifying vulnerabilities, predicting severity and access complexity, and generating fixes as a proof of concept. Method: We evaluate five pairs of recent LLMs, including both code-based and general-purpose open-source models, on two recognized C/C++ vulnerability datasets, namely Big-Vul and Vul-Repair. Additionally, we compare fine-tuning and prompt-based approaches. Results: The results show that fine-tuning uniformly outperforms both zero-shot and few-shot approaches across all tasks and models. Notably, code-specialized models excel in zero-shot and few-shot settings on complex tasks, while general-purpose models remain nearly as effective. Discrepancies among CodeBLEU, CodeBERTScore, BLEU, and ChrF highlight the inadequacy of current metrics for measuring repair quality. Conclusions: This study contributes to the software security community by investigating the potential of advanced LLMs to improve vulnerability analysis and remediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14381v2">Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)
    </div>
    <details class="paper-abstract">
      Large language model (LLM) systems increasingly power everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on manually well-crafted prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to query poisoning alone: feedback-based attacks raise attack success rate (ASR) by up to ΔASR = 0.48. We introduce a simple fake reward attack that requires no access to the reward model and significantly increases vulnerability. We also propose a lightweight highlighting defense that reduces the fake reward ΔASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08654v1">RULERS: Locked Rubrics and Evidence-Anchored Scoring for Robust LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The LLM-as-a-Judge paradigm promises scalable rubric-based evaluation, yet aligning frozen black-box models with human standards remains a challenge due to inherent generation stochasticity. We reframe judge alignment as a criteria transfer problem and isolate three recurrent failure modes: rubric instability caused by prompt sensitivity, unverifiable reasoning that lacks auditable evidence, and scale misalignment with human grading boundaries. To address these issues, we introduce RULERS (Rubric Unification, Locking, and Evidence-anchored Robust Scoring), a compiler-executor framework that transforms natural language rubrics into executable specifications. RULERS operates by compiling criteria into versioned immutable bundles, enforcing structured decoding with deterministic evidence verification, and applying lightweight Wasserstein-based post-hoc calibration, all without updating model parameters. Extensive experiments on essay and summarization benchmarks demonstrate that RULERS significantly outperforms representative baselines in human agreement, maintains strong stability against adversarial rubric perturbations, and enables smaller models to rival larger proprietary judges. Overall, our results suggest that reliable LLM judging requires executable rubrics, verifiable evidence, and calibrated scales rather than prompt phrasing alone. Code is available at https://github.com/LabRAI/Rulers.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08653v1">Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08634v1">Moral Lenses, Political Coordinates: Towards Ideological Positioning of Morally Conditioned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      While recent research has systematically documented political orientation in large language models (LLMs), existing evaluations rely primarily on direct probing or demographic persona engineering to surface ideological biases. In social psychology, however, political ideology is also understood as a downstream consequence of fundamental moral intuitions. In this work, we investigate the causal relationship between moral values and political positioning by treating moral orientation as a controllable condition. Rather than simply assigning a demographic persona, we condition models to endorse or reject specific moral values and evaluate the resulting shifts on their political orientations, using the Political Compass Test. By treating moral values as lenses, we observe how moral conditioning actively steers model trajectories across economic and social dimensions. Our findings show that such conditioning induces pronounced, value-specific shifts in models' political coordinates. We further notice that these effects are systematically modulated by role framing and model scale, and are robust across alternative assessment instruments instantiating the same moral value. This highlights that effective alignment requires anchoring political assessments within the context of broader social values including morality, paving the way for more socially grounded alignment techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08626v1">How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at semantic understanding, yet their ability to reconstruct internal structure from scrambled inputs remains underexplored. Sentence-level restoration is ill-posed for automated evaluation because multiple valid word orders often exist. We introduce OrderProbe, a deterministic benchmark for structural reconstruction using fixed four-character expressions in Chinese, Japanese, and Korean, which have a unique canonical order and thus support exact-match scoring. We further propose a diagnostic framework that evaluates models beyond recovery accuracy, including semantic fidelity, logical validity, consistency, robustness sensitivity, and information density. Experiments on twelve widely used LLMs show that structural reconstruction remains difficult even for frontier systems: zero-shot recovery frequently falls below 35%. We also observe a consistent dissociation between semantic recall and structural planning, suggesting that structural robustness is not an automatic byproduct of semantic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08539v1">Reducing Compute Waste in LLMs through Kernel-Level DVFS</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The rapid growth of AI has fueled the expansion of accelerator- or GPU-based data centers. However, the rising operational energy consumption has emerged as a critical bottleneck and a major sustainability concern. Dynamic Voltage and Frequency Scaling (DVFS) is a well-known technique used to reduce energy consumption, and thus improve energy-efficiency, since it requires little effort and works with existing hardware. Reducing the energy consumption of training and inference of Large Language Models (LLMs) through DVFS or power capping is feasible: related work has shown energy savings can be significant, but at the cost of significant slowdowns. In this work, we focus on reducing waste in LLM operations: i.e., reducing energy consumption without losing performance. We propose a fine-grained, kernel-level, DVFS approach that explores new frequency configurations, and prove these save more energy than previous, pass- or iteration-level solutions. For example, for a GPT-3 training run, a pass-level approach could reduce energy consumption by 2% (without losing performance), while our kernel-level approach saves as much as 14.6% (with a 0.6% slowdown). We further investigate the effect of data and tensor parallelism, and show our discovered clock frequencies translate well for both. We conclude that kernel-level DVFS is a suitable technique to reduce waste in LLM operations, providing significant energy savings with negligible slow-down.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01891v3">Multi-Personality Generation of LLMs at Decoding-time</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08517v1">Closed-Loop LLM Discovery of Non-Standard Channel Priors in Vision Models</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Channel configuration search the optimization of layer specifications such as layer widths in deep neural networks presents a complex combinatorial challenge constrained by tensor shape compatibility and computational budgets. We posit that Large Language Models (LLMs) offer a transformative approach to Neural Architecture Search (NAS), capable of reasoning about architectural code structure in ways that traditional heuristics cannot. In this paper, we investigate the application of an LLM-driven NAS framework to the problem of channel configuration. We formulate the search as a sequence of conditional code generation tasks, where an LLM refines architectural specifications based on performance telemetry. Crucially, we address the data scarcity problem by generating a vast corpus of valid, shape-consistent architectures via Abstract Syntax Tree (AST) mutations. While these mutated networks are not necessarily high-performing, they provide the critical volume of structural data required for the LLM to learn the latent relationship between channel configurations and model performance. This allows the LLM to internalize complex design patterns and apply them to optimize feature extraction strategies. Experimental results on CIFAR-100 validate the efficacy of this approach, demonstrating that the model yields statistically significant improvements in accuracy. Our analysis confirms that the LLM successfully acquires domain-specific architectural priors, distinguishing this method from random search and highlighting the immense potential of language-driven design in deep learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08511v1">STAR: Detecting Inference-time Backdoors in LLM Reasoning via State-Transition Amplification Ratio</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent LLMs increasingly integrate reasoning mechanisms like Chain-of-Thought (CoT). However, this explicit reasoning exposes a new attack surface for inference-time backdoors, which inject malicious reasoning paths without altering model parameters. Because these attacks generate linguistically coherent paths, they effectively evade conventional detection. To address this, we propose STAR (State-Transition Amplification Ratio), a framework that detects backdoors by analyzing output probability shifts. STAR exploits the statistical discrepancy where a malicious input-induced path exhibits high posterior probability despite a low prior probability in the model's general knowledge. We quantify this state-transition amplification and employ the CUSUM algorithm to detect persistent anomalies. Experiments across diverse models (8B-70B) and five benchmark datasets demonstrate that STAR exhibits robust generalization capabilities, consistently achieving near-perfect performance (AUROC $\approx$ 1.0) with approximately $42\times$ greater efficiency than existing baselines. Furthermore, the framework proves robust against adaptive attacks attempting to bypass detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17535v2">How role-play shapes relevance judgment in zero-shot LLM rankers</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising zero-shot rankers, but their performance is highly sensitive to prompt formulation. In particular, role-play prompts, where the model is assigned a functional role or identity, often give more robust and accurate relevance rankings. However, the mechanisms and diversity of role-play effects remain underexplored, limiting both effective use and interpretability. In this work, we systematically examine how role-play variations influence zero-shot LLM rankers. We employ causal intervention techniques from mechanistic interpretability to trace how role-play information shapes relevance judgments in LLMs. Our analysis reveals that (1) careful formulation of role descriptions have a large effect on the ranking quality of the LLM; (2) role-play signals are predominantly encoded in early layers and communicate with task instructions in middle layers, while receiving limited interaction with query or document representations. Specifically, we identify a group of attention heads that encode information critical for role-conditioned relevance. These findings not only shed light on the inner workings of role-play in LLM ranking but also offer guidance for designing more effective prompts in IR and beyond, pointing toward broader opportunities for leveraging role-play in zero-shot applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08001v3">Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted by AAAI 2026 Oral
    </div>
    <details class="paper-abstract">
      "Fedspeak", the stylized and often nuanced language used by the U.S. Federal Reserve, encodes implicit policy signals and strategic stances. The Federal Open Market Committee strategically employs Fedspeak as a communication tool to shape market expectations and influence both domestic and global economic conditions. As such, automatically parsing and interpreting Fedspeak presents a high-impact challenge, with significant implications for financial forecasting, algorithmic trading, and data-driven policy analysis. In this paper, we propose an LLM-based, uncertainty-aware framework for deciphering Fedspeak and classifying its underlying monetary policy stance. Technically, to enrich the semantic and contextual representation of Fedspeak texts, we incorporate domain-specific reasoning grounded in the monetary policy transmission mechanism. We further introduce a dynamic uncertainty decoding module to assess the confidence of model predictions, thereby enhancing both classification accuracy and model reliability. Experimental results demonstrate that our framework achieves state-of-the-art performance on the policy stance analysis task. Moreover, statistical analysis reveals a significant positive correlation between perceptual uncertainty and model error rates, validating the effectiveness of perceptual uncertainty as a diagnostic signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08462v1">M3-BENCH: Process-Aware Evaluation of LLM Agents Social Behaviors in Mixed-Motive Games</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      As the capabilities of large language model (LLM) agents continue to advance, their advanced social behaviors, such as cooperation, deception, and collusion, call for systematic evaluation. However, existing benchmarks often emphasize a single capability dimension or rely solely on behavioral outcomes, overlooking rich process information from agents' decision reasoning and communicative interactions. To address this gap, we propose M3-Bench, a multi-stage benchmark for mixed-motive games, together with a process-aware evaluation framework that conducts synergistic analysis across three modules: BTA (Behavioral Trajectory Analysis), RPA (Reasoning Process Analysis), and CCA (Communication Content Analysis). Furthermore, we integrate the Big Five personality model and Social Exchange Theory to aggregate multi-dimensional evidence into interpretable social behavior portraits, thereby characterizing agents' personality traits and capability profiles beyond simple task scores or outcome-based metrics. Experimental results show that M3-Bench can reliably distinguish diverse social behavior competencies across models, and it reveals that some models achieve seemingly reasonable behavioral outcomes while exhibiting pronounced inconsistencies in their reasoning and communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03190v2">Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to forget sensitive knowledge from Large Language Models (LLMs) while maintaining general utility. However, existing approaches typically treat all tokens in a response indiscriminately and enforce uncertainty over the entire vocabulary. This global treatment results in unnecessary utility degradation and extends optimization to content-agnostic regions. To address these limitations, we propose PALU (Prefix-Aware Localized Unlearning), a framework driven by a local entropy maximization objective across both temporal and vocabulary dimensions. PALU reveals that (i) suppressing the sensitive prefix alone is sufficient to sever the causal generation link, and (ii) flattening only the top-$k$ logits is adequate to maximize uncertainty in the critical subspace. These findings allow PALU to avoid redundant optimization across the full vocabulary and parameter space while minimizing collateral damage to general model performance. Extensive experiments validate that PALU achieves superior forgetting efficacy and utility preservation compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08415v1">Regulatory gray areas of LLM Terms</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into academic research pipelines; however, the Terms of Service governing their use remain under-examined. We present a comparative analysis of the Terms of Service of five major LLM providers (Anthropic, DeepSeek, Google, OpenAI, and xAI) collected in November 2025. Our analysis reveals substantial variation in the stringency and specificity of usage restrictions for general users and researchers. We identify specific complexities for researchers in security research, computational social sciences, and psychological studies. We identify `regulatory gray areas' where Terms of Service create uncertainty for legitimate use. We contribute a publicly available resource comparing terms across platforms (OSF) and discuss implications for general users and researchers navigating this evolving landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06780v3">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted and published in Findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03746v2">Whose Facts Win? LLM Source Preferences under Knowledge Conflicts</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Data and code: https://github.com/JaSchuste/llm-source-preference
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 99.8%, while also maintaining at least 88.8% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15793v3">Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted by AAAI 2026, camera ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including texts, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Yet it remains unclear whether such biases are systematic, which data-level factors drive them, and what internal mechanisms underlie their emergence. In this paper, we present the first comprehensive study of format bias in LLMs through a three-stage empirical analysis. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage examines how key data-level factors influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its effectiveness. Our results show that format bias is consistent across model families, driven by information richness, structure quality, and representation type, and is closely associated with attention imbalance within the LLMs. Based on these investigations, we identify three future research directions to reduce format bias: enhancing data pre-processing through format repair and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08343v1">When KV Cache Reuse Fails in Multi-Agent Systems: Cross-Candidate Interaction is Crucial for LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Multi-agent LLM systems routinely generate multiple candidate responses that are aggregated by an LLM judge. To reduce the dominant prefill cost in such pipelines, recent work advocates KV cache reuse across partially shared contexts and reports substantial speedups for generation agents. In this work, we show that these efficiency gains do not transfer uniformly to judge-centric inference. Across GSM8K, MMLU, and HumanEval, we find that reuse strategies that are effective for execution agents can severely perturb judge behavior: end-task accuracy may appear stable, yet the judge's selection becomes highly inconsistent with dense prefill. We quantify this risk using Judge Consistency Rate (JCR) and provide diagnostics showing that reuse systematically weakens cross-candidate attention, especially for later candidate blocks. Our ablation further demonstrates that explicit cross-candidate interaction is crucial for preserving dense-prefill decisions. Overall, our results identify a previously overlooked failure mode of KV cache reuse and highlight judge-centric inference as a distinct regime that demands dedicated, risk-aware system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07198v3">Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 20 pages, 5 figures, 21 tables. Accepted at NeurIPS 2025. Corresponding author: Xuan Zhang (xuanzhang2199@gmail.com)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive generalization abilities, yet adapting them effectively across multiple heterogeneous domains remains challenging due to inter-domain interference. To overcome this challenge, we propose a partition-based multi-stage fine-tuning framework designed to exploit inter-domain synergies while minimizing negative transfer. Our approach strategically partitions domains into subsets (stages) by balancing domain discrepancy, synergy, and model capacity constraints. We theoretically analyze the proposed framework and derive novel generalization bounds that justify our partitioning strategy. Extensive empirical evaluations on various language understanding tasks show that our method consistently outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08271v1">Sparsity Is Necessary: Polynomial-Time Stability for Agentic LLMs in Large Action Spaces</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Tool-augmented LLM systems expose a control regime that learning theory has largely ignored: sequential decision-making with a massive discrete action universe (tools, APIs, documents) in which only a small, unknown subset is relevant for any fixed task distribution. We formalize this setting as Sparse Agentic Control (SAC), where policies admit block-sparse representations over M >> 1 actions and rewards depend on sparse main effects and (optionally) sparse synergies. We study ell_{1,2}-regularized policy learning through a convex surrogate and establish sharp, compressed-sensing-style results: (i) estimation and value suboptimality scale as k (log M / T)^{1/2} under a Policy-RSC condition; (ii) exact tool-support recovery holds via primal-dual witness arguments when T > k log M under incoherence and beta-min; and (iii) any dense policy class requires Omega(M) samples, explaining the instability of prompt-only controllers. We further show that under partial observability, LLMs matter only through a belief/representation error epsilon_b, yielding an additive O(epsilon_b) degradation while preserving logarithmic dependence on M. Extensions cover tuning-free, online, robust, group-sparse, and interaction-aware SAC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02369v3">AutoContext: Instance-Level Context Learning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Current LLM agents typically lack instance-level context, which comprises concrete facts such as environment structure, system configurations, and local mechanics. Consequently, existing methods are forced to intertwine exploration with task execution. This coupling leads to redundant interactions and fragile decision-making, as agents must repeatedly rediscover the same information for every new task. To address this, we introduce AutoContext, a method that decouples exploration from task solving. AutoContext performs a systematic, one-off exploration to construct a reusable knowledge graph for each environment instance. This structured context allows off-the-shelf agents to access necessary facts directly, eliminating redundant exploration. Experiments across TextWorld, ALFWorld, Crafter, and InterCode-Bash demonstrate substantial gains: for example, the success rate of a ReAct agent on TextWorld improves from 37% to 95%, highlighting the critical role of structured instance context in efficient agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02962v6">RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities, are prone to generating hallucinated or outdated content due to their static internal knowledge. While Retrieval-Augmented Generation (RAG) integrated with Reinforcement Learning (RL) offers a solution, these methods are fundamentally constrained by a single-query mode, leading to prohibitive latency and inherent brittleness. To overcome these limitations, we introduce RAG-R1, a novel two-stage training framework centered around multi-query parallelism. Our framework enables LLMs to adaptively leverage internal and external knowledge during the reasoning process while transitioning from the single-query mode to multi-query parallelism. This architectural shift bolsters reasoning robustness while significantly reducing inference latency. Extensive experiments on seven question-answering benchmarks confirm the superiority of our method, which outperforms the strongest baseline by up to 13.7% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08237v1">The End of Reward Engineering: How LLMs Are Redefining Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Reward engineering, the manual specification of reward functions to induce desired agent behavior, remains a fundamental challenge in multi-agent reinforcement learning. This difficulty is amplified by credit assignment ambiguity, environmental non-stationarity, and the combinatorial growth of interaction complexity. We argue that recent advances in large language models (LLMs) point toward a shift from hand-crafted numerical rewards to language-based objective specifications. Prior work has shown that LLMs can synthesize reward functions directly from natural language descriptions (e.g., EUREKA) and adapt reward formulations online with minimal human intervention (e.g., CARD). In parallel, the emerging paradigm of Reinforcement Learning from Verifiable Rewards (RLVR) provides empirical evidence that language-mediated supervision can serve as a viable alternative to traditional reward engineering. We conceptualize this transition along three dimensions: semantic reward specification, dynamic reward adaptation, and improved alignment with human intent, while noting open challenges related to computational overhead, robustness to hallucination, and scalability to large multi-agent systems. We conclude by outlining a research direction in which coordination arises from shared semantic representations rather than explicitly engineered numerical signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08196v1">Evaluating Implicit Regulatory Compliance in LLM Tool Invocation via Logic-Guided Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into autonomous agents has enabled complex tool use, yet in high-stakes domains, these systems must strictly adhere to regulatory standards beyond simple functional correctness. However, existing benchmarks often overlook implicit regulatory compliance, thus failing to evaluate whether LLMs can autonomously enforce mandatory safety constraints. To fill this gap, we introduce LogiSafetyGen, a framework that converts unstructured regulations into Linear Temporal Logic oracles and employs logic-guided fuzzing to synthesize valid, safety-critical traces. Building on this framework, we construct LogiSafetyBench, a benchmark comprising 240 human-verified tasks that require LLMs to generate Python programs that satisfy both functional objectives and latent compliance rules. Evaluations of 13 state-of-the-art (SOTA) LLMs reveal that larger models, despite achieving better functional correctness, frequently prioritize task completion over safety, which results in non-compliant behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17627v2">The Evolution of Thought: Tracking LLM Overthinking via Reasoning Dynamics Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Test-time scaling via explicit reasoning trajectories significantly boosts large language model (LLM) performance but often triggers overthinking. To explore this, we analyze reasoning through two lenses: Reasoning Length Dynamics, which reveals a compensatory trade-off between thinking and answer content length that eventually leads to thinking redundancy, and Reasoning Semantic Dynamics, which identifies semantic convergence and repetitive oscillations. These dynamics uncover an instance-specific Reasoning Completion Point (RCP), beyond which computation continues without further performance gain. Since the RCP varies across instances, we propose a Reasoning Completion Point Detector (RCPD), an inference-time early-exit method that identifies the RCP by monitoring the rank dynamics of termination tokens (e.g., </think>). Across AIME and GPQA benchmarks using Qwen3 and DeepSeek-R1, RCPD reduces token usage by up to 44% while preserving accuracy, offering a principled approach to efficient test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08187v1">Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising capabilities in Text-Attributed Graph (TAG) understanding. Recent studies typically focus on verbalizing the graph structures via handcrafted prompts, feeding the target node and its neighborhood context into LLMs. However, constrained by the context window, existing methods mainly resort to random sampling, often implemented via dropping node/edge randomly, which inevitably introduces noise and cause reasoning instability. We argue that graphs inherently contain rich structural and semantic information, and that their effective exploitation can unlock potential gains in LLMs reasoning performance. To this end, we propose Homophily-aware Structural and Semantic Compression for LLMs (HS2C), a framework centered on exploiting graph homophily. Structurally, guided by the principle of Structural Entropy minimization, we perform a global hierarchical partition that decodes the graph's essential topology. This partition identifies naturally cohesive, homophilic communities, while discarding stochastic connectivity noise. Semantically, we deliver the detected structural homophily to the LLM, empowering it to perform differentiated semantic aggregation based on predefined community type. This process compresses redundant background contexts into concise community-level consensus, selectively preserving semantically homophilic information aligned with the target nodes. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families demonstrate that, by feeding LLMs with structurally and semantically compressed inputs, HS2C simultaneously enhances the compression rate and downstream inference accuracy, validating its superiority and scalability. Extensions to 7 diverse graph-level benchmarks further consolidate HS2C's task generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18882v5">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08166v1">ZeroDVFS: Zero-Shot LLM-Guided Core and Frequency Allocation for Embedded Platforms</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 39 pages, 12 figures, 8 tables (including appendix)
    </div>
    <details class="paper-abstract">
      Dynamic voltage and frequency scaling (DVFS) and task-to-core allocation are critical for thermal management and balancing energy and performance in embedded systems. Existing approaches either rely on utilization-based heuristics that overlook stall times, or require extensive offline profiling for table generation, preventing runtime adaptation. We propose a model-based hierarchical multi-agent reinforcement learning (MARL) framework for thermal- and energy-aware scheduling on multi-core platforms. Two collaborative agents decompose the exponential action space, achieving 358ms latency for subsequent decisions. First decisions require 3.5 to 8.0s including one-time LLM feature extraction. An accurate environment model leverages regression techniques to predict thermal dynamics and performance states. When combined with LLM-extracted semantic features, the environment model enables zero-shot deployment for new workloads on trained platforms by generating synthetic training data without requiring workload-specific profiling samples. We introduce LLM-based semantic feature extraction that characterizes OpenMP programs through 13 code-level features without execution. The Dyna-Q-inspired framework integrates direct reinforcement learning with model-based planning, achieving 20x faster convergence than model-free methods. Experiments on BOTS and PolybenchC benchmarks across NVIDIA Jetson TX2, Jetson Orin NX, RubikPi, and Intel Core i7 demonstrate 7.09x better energy efficiency and 4.0x better makespan than Linux ondemand governor. First-decision latency is 8,300x faster than table-based profiling, enabling practical deployment in dynamic embedded systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06266v2">Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to SANER 2026 (IEEE International Conference on Software Analysis, Evolution and Reengineering)
    </div>
    <details class="paper-abstract">
      Self-admitted technical debt (SATD), referring to comments flagged by developers that explicitly acknowledge suboptimal code or incomplete functionality, has received extensive attention in machine learning (ML) and traditional (Non-ML) software. However, little is known about how SATD manifests and evolves in contemporary Large Language Model (LLM)-based systems, whose architectures, workflows, and dependencies differ fundamentally from both traditional and pre-LLM ML software. In this paper, we conduct the first empirical study of SATD in the LLM era, replicating and extending prior work on ML technical debt to modern LLM-based systems. We compare SATD prevalence across LLM, ML, and non-ML repositories across a total of 477 repositories (159 per category). We perform survival analysis of SATD introduction and removal to understand the dynamics of technical debt across different development paradigms. Surprisingly, despite their architectural complexity, our results reveal that LLM repositories accumulate SATD at similar rates to ML systems (3.95% vs. 4.10%). However, we observe that LLM repositories remain debt-free 2.4x longer than ML repositories (a median of 492 days vs. 204 days), and then start to accumulate technical debt rapidly. Moreover, our qualitative analysis of 377 SATD instances reveals three new forms of technical debt unique to LLM-based development that have not been reported in prior research: Model-Stack Workaround Debt, Model Dependency Debt, and Performance Optimization Debt. Finally, by mapping SATD to stages of the LLM development pipeline, we observe that debt concentrates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20002v3">LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 This submission is withdrawn due to internal review and compliance considerations
    </div>
    <details class="paper-abstract">
      Time-series forecasting in real-world applications such as finance and energy often faces challenges due to limited training data and complex, noisy temporal dynamics. Existing deep forecasting models typically supervise predictions using full-length temporal windows, which include substantial high-frequency noise and obscure long-term trends. Moreover, auxiliary variables containing rich domain-specific information are often underutilized, especially in few-shot settings. To address these challenges, we propose LoFT-LLM, a frequency-aware forecasting pipeline that integrates low-frequency learning with semantic calibration via a large language model (LLM). Firstly, a Patch Low-Frequency forecasting Module (PLFM) extracts stable low-frequency trends from localized spectral patches. Secondly, a residual learner then models high-frequency variations. Finally, a fine-tuned LLM refines the predictions by incorporating auxiliary context and domain knowledge through structured natural language prompts. Extensive experiments on financial and energy datasets demonstrate that LoFT-LLM significantly outperforms strong baselines under both full-data and few-shot regimes, delivering superior accuracy, robustness, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08107v1">STO-RL: Offline RL under Sparse Rewards via LLM-Guided Subgoal Temporal Order</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted at International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      Offline reinforcement learning (RL) enables policy learning from pre-collected datasets, avoiding costly and risky online interactions, but it often struggles with long-horizon tasks involving sparse rewards. Existing goal-conditioned and hierarchical offline RL methods decompose such tasks and generate intermediate rewards to mitigate limitations of traditional offline RL, but usually overlook temporal dependencies among subgoals and rely on imprecise reward shaping, leading to suboptimal policies. To address these issues, we propose STO-RL (Offline RL using LLM-Guided Subgoal Temporal Order), an offline RL framework that leverages large language models (LLMs) to generate temporally ordered subgoal sequences and corresponding state-to-subgoal-stage mappings. Using this temporal structure, STO-RL applies potential-based reward shaping to transform sparse terminal rewards into dense, temporally consistent signals, promoting subgoal progress while avoiding suboptimal solutions. The resulting augmented dataset with shaped rewards enables efficient offline training of high-performing policies. Evaluations on four discrete and continuous sparse-reward benchmarks demonstrate that STO-RL consistently outperforms state-of-the-art offline goal-conditioned and hierarchical RL baselines, achieving faster convergence, higher success rates, and shorter trajectories. Ablation studies further confirm STO-RL's robustness to imperfect or noisy LLM-generated subgoal sequences, demonstrating that LLM-guided subgoal temporal structures combined with theoretically grounded reward shaping provide a practical and scalable solution for long-horizon offline RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08089v1">Q-realign: Piggybacking Realignment on Quantization for Safe and Efficient LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Public large language models (LLMs) are typically safety-aligned during pretraining, yet task-specific fine-tuning required for deployment often erodes this alignment and introduces safety risks. Existing defenses either embed safety recovery into fine-tuning or rely on fine-tuning-derived priors for post-hoc correction, leaving safety recovery tightly coupled with training and incurring high computational overhead and a complex workflow. To address these challenges, we propose \texttt{Q-realign}, a post-hoc defense method based on post-training quantization, guided by an analysis of representational structure. By reframing quantization as a dual-objective procedure for compression and safety, \texttt{Q-realign} decouples safety alignment from fine-tuning and naturally piggybacks into modern deployment pipelines. Experiments across multiple models and datasets demonstrate that our method substantially reduces unsafe behaviors while preserving task performance, with significant reductions in memory usage and GPU hours. Notably, our approach can recover the safety alignment of a fine-tuned 7B LLM on a single RTX 4090 within 40 minutes. Overall, our work provides a practical, turnkey solution for safety-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09029v1">Proactively Detecting Threats: A Novel Approach Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 2025 International Conference on Cyber-Enabled Distributed Computing and Knowledge Discovery (CyberC)
    </div>
    <details class="paper-abstract">
      Enterprise security faces escalating threats from sophisticated malware, compounded by expanding digital operations. This paper presents the first systematic evaluation of large language models (LLMs) to proactively identify indicators of compromise (IOCs) from unstructured web-based threat intelligence sources, distinguishing it from reactive malware detection approaches. We developed an automated system that pulls IOCs from 15 web-based threat report sources to evaluate six LLM models (Gemini, Qwen, and Llama variants). Our evaluation of 479 webpages containing 2,658 IOCs (711 IPv4 addresses, 502 IPv6 addresses, 1,445 domains) reveals significant performance variations. Gemini 1.5 Pro achieved 0.958 precision and 0.788 specificity for malicious IOC identification, while demonstrating perfect recall (1.0) for actual threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.04083v3">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to appear in proceedings of the 3rd Workshop on Artificial Intelligence for Scientific Publications (WASP@AACL 2025). Code and data are available here: https://github.com/bionlproc/ZeroShotRE
    </div>
    <details class="paper-abstract">
      Extracting relations from scientific literature is a fundamental task in biomedical NLP because entities and relations among them drive hypothesis generation and knowledge discovery. As literature grows rapidly, relation extraction (RE) is indispensable to curate knowledge graphs to be used as computable structured and symbolic representations. With the rise of LLMs, it is pertinent to examine if it is better to skip tailoring supervised RE methods, save annotation burden, and just use zero shot RE (ZSRE) via LLM API calls. In this paper, we propose a benchmark with seven biomedical RE datasets with interesting characteristics and evaluate three Open AI models (GPT-4, o1, and GPT-OSS-120B) for end-to-end ZSRE. We show that LLM-based ZSRE is inching closer to supervised methods in performances on some datasets but still struggles on complex inputs expressing multiple relations with different predicates. Our error analysis reveals scope for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15801v2">LingVarBench: Benchmarking LLMs on Entity Recognitions and Linguistic Verbalization Patterns in Phone-Call Transcripts</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 Accepted to EACL 2026 (Industry Track); to appear in the proceedings
    </div>
    <details class="paper-abstract">
      We study structured entity extraction from phone-call transcripts in customer-support and healthcare settings, where annotation is costly, and data access is limited by privacy and consent. Existing methods degrade under disfluencies, interruptions, and speaker overlap, yet large real-call corpora are rarely shareable. We introduce LingVarBench, a benchmark and semantic synthetic data generation pipeline that generates linguistically varied training data via (1) LLM-sampled entity values, (2) curated linguistic verbalization patterns covering diverse disfluencies and entity-specific readout styles, and (3) a value-transcript consistency filter. Using this dataset, DSPy's SIMBA automatically synthesizes and optimizes extraction prompts, reducing manual prompt engineering and targeting robustness to verbal variation. On real customer transcripts, prompts optimized solely on LingVarBench outperform zero-shot baselines and match or closely approach human-tuned prompts for structured entities such as ZIP code, date of birth, and name (F1 approximately 94-95 percent). For subjective questionnaire items, optimized prompts substantially improve over zero-shot performance and approach human-tuned prompts. LingVarBench offers a practical and cost-efficient path to deployment in a direct-answer setting, with real annotations later enabling additional refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09017v1">Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has necessitated more robust evaluation methods that go beyond static benchmarks, which are increasingly prone to data saturation and leakage. In this paper, we propose a dynamic benchmarking framework for evaluating multilingual and multicultural capabilities through the social deduction game Spyfall. In our setup, models must engage in strategic dialogue to either identify a secret agent or avoid detection, utilizing culturally relevant locations or local foods. Our results show that our game-based rankings align closely with the Chatbot Arena. However, we find a significant performance gap in non-English contexts: models are generally less proficient when handling locally specific entities and often struggle with rule-following or strategic integrity in non-English languages. We demonstrate that this game-based approach provides a scalable, leakage-resistant, and culturally nuanced alternative to traditional NLP benchmarks. The game history can be accessed here https://huggingface.co/datasets/haryoaw/cultural-spyfall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20634v2">Recidivism and Peer Influence with LLM Text Embeddings in Low Security Correctional Facilities</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Studying peer effects in language is critical because they often reflect behavioral and personality traits that are important determinants of economic outcomes. However, language is unstructured, non-numeric, and high-dimensional. We combine Large Language Model (LLM) embeddings with structural econometric identification to provide a unified framework for identifying peer effects in language. This unified framework is applied to 80,000-120,000 written exchanges among residents of low security correctional facilities. The LLM language profiles predict three-year recidivism 30\% more accurately than pre-entry covariates alone, showing that text representations capture meaningful signals. We analyze peer effects on multidimensional language embeddings while addressing network endogeneity. We develop novel instrumental variable estimators for peer effects that accommodate multivariate outcomes, sparse networks, and multidimensional latent variables. Our methods achieve root-N consistency and asymptotic normality under realistic sparsity conditions, relaxing the dense-network assumption. Results reveal significant peer effects in residents' language profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v1">Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08998v1">On the Flakiness of LLM-Generated Tests for Industrial and Open-Source Database Management Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-13
      | 💬 12 pages, 5 tables, 3 figures, accepted at the 48th International Conference on Software Engineering: Software Engineering in Practice (ICSE SEIP 2026)
    </div>
    <details class="paper-abstract">
      Flaky tests are a common problem in software testing. They produce inconsistent results when executed multiple times on the same code, invalidating the assumption that a test failure indicates a software defect. Recent work on LLM-based test generation has identified flakiness as a potential problem with generated tests. However, its prevalence and underlying causes are unclear. We examined the flakiness of LLM-generated tests in the context of four relational database management systems: SAP HANA, DuckDB, MySQL, and SQLite. We amplified test suites with two LLMs, GPT-4o and Mistral-Large-Instruct-2407, to assess the flakiness of the generated test cases. Our results suggest that generated tests have a slightly higher proportion of flaky tests compared to existing tests. Based on a manual inspection, we found that the most common root cause of flakiness was the reliance of a test on a certain order that is not guaranteed ("unordered collection"), which was present in 72 of 115 flaky tests (63%). Furthermore, both LLMs transferred the flakiness from the existing tests to the newly generated tests via the provided prompt context. Our experiments suggest that flakiness transfer is more prevalent in closed-source systems such as SAP HANA than in open-source systems. Our study informs developers on what types of flakiness to expect from LLM-generated tests. It also highlights the importance of providing LLMs with tailored context when employing LLMs for test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08919v1">Fine Grained Evaluation of LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      A good deal of recent research has focused on how Large Language Models (LLMs) may be used as `judges' in place of humans to evaluate the quality of the output produced by various text / image processing systems. Within this broader context, a number of studies have investigated the specific question of how effectively LLMs can be used as relevance assessors for the standard ad hoc task in Information Retrieval (IR). We extend these studies by looking at additional questions. Most importantly, we use a Wikipedia based test collection created by the INEX initiative, and prompt LLMs to not only judge whether documents are relevant / non-relevant, but to highlight relevant passages in documents that it regards as useful. The human relevance assessors involved in creating this collection were given analogous instructions, i.e., they were asked to highlight all passages within a document that respond to the information need expressed in a query. This enables us to evaluate the quality of LLMs as judges not only at the document level, but to also quantify how often these `judges' are right for the right reasons. Our findings suggest that LLMs-as-judges work best under human supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08892v1">Evaluating Role-Consistency in LLMs for Counselor Training</a></div>
    <div class="paper-meta">
      📅 2026-01-13
    </div>
    <details class="paper-abstract">
      The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07806v1">The Confidence Trap: Gender Bias and Predictive Certainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-12
      | 💬 AAAI 2026 (AISI Track), Oral. Project page: https://bit.ly/4p8OKQD
    </div>
    <details class="paper-abstract">
      The increased use of Large Language Models (LLMs) in sensitive domains leads to growing interest in how their confidence scores correspond to fairness and bias. This study examines the alignment between LLM-predicted confidence and human-annotated bias judgments. Focusing on gender bias, the research investigates probability confidence calibration in contexts involving gendered pronoun resolution. The goal is to evaluate if calibration metrics based on predicted confidence scores effectively capture fairness-related disparities in LLMs. The results show that, among the six state-of-the-art models, Gemma-2 demonstrates the worst calibration according to the gender bias benchmark. The primary contribution of this work is a fairness-aware evaluation of LLMs' confidence calibration, offering guidance for ethical deployment. In addition, we introduce a new calibration metric, Gender-ECE, designed to measure gender disparities in resolution tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07796v1">Learning Through Dialogue: Unpacking the Dynamics of Human-LLM Conversations on Political Issues</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20443v2">Towards Mitigating Excessive Forgetting in LLM Unlearning via Entanglement-Guidance with Proxy Constraint</a></div>
    <div class="paper-meta">
      📅 2026-01-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on massive datasets that may include private or copyrighted content. Due to growing privacy and ownership concerns, data owners may request the removal of their data from trained models. Machine unlearning provides a practical solution by removing the influence of specific data without full retraining. However, most existing methods still suffer from over-unlearning due to the lack of a principled mechanism to regulate the forgetting boundary, leading to unnecessary utility degradation and heightened privacy and robustness risks. In this work, we propose EGUP (Entanglement-Guided Unlearning with Proxy Constraint), a novel framework that leverages entanglement and proxy constraint to guide the unlearning process while mitigating over-unlearning. Within each iteration, EGUP employs inter-sample entanglement to adaptively reweight the unlearning strength, assigning greater unlearning efforts to forget samples that are semantically closer to retained knowledge. Across iterations, EGUP leverages intra-sample entanglement to track the representation shift of each forget sample and dynamically adjust its unlearning effort. In addition, we incorporate a proxy constraint that approximates the model's expected outputs after unlearning, forming a reference boundary that softly regularizes the unlearning process. EGUP is compatible with existing gradient-based objectives and serves as a plug-and-play enhancement. We evaluate EGUP on the TOFU and MUSE benchmarks, demonstrating consistent improvements in the unlearning-utility trade-off across multiple LLMs. Moreover, EGUP achieves performance close to the retrained model while remaining scalable and robust.
    </details>
</div>
