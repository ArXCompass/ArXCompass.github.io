# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
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
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24189v1">Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v4">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24179v1">SALE : Low-bit Estimation for Efficient Sparse Attention in Long-context LLM Prefilling</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Many advanced Large Language Model (LLM) applications require long-context processing, but the self-attention module becomes a bottleneck during the prefilling stage of inference due to its quadratic time complexity with respect to sequence length. Existing sparse attention methods accelerate attention computation by skipping less significant regions of the attention map. However, these approaches typically perform coarse-grained inspection of the attention map, rendering considerable loss in model accuracy. In this paper, we propose SALE, a fine-grained sparse attention method that accelerates the long-context prefilling stage of LLM with negligible loss in model accuracy. SALE achieves fast and accurate fine-grained attention weight estimation through 4-bit quantized query-key products, followed by block-sparse attention to accelerate prefilling computations. For importance evaluation for query-key pairs, we adopt our Relative Attention Score metric, which offers significantly higher efficiency within our framework. We implement a custom CUDA kernel optimized for our approach for hardware efficiency, reducing the additional overhead to approximately 11% of the full attention latency. Notably, SALE requires no parameter training and can be seamlessly integrated into existing systems with trivial code modifications. Experiments on long-context benchmarks demonstrate that our method outperforms existing approaches in accuracy-efficiency trade-offs, achieving at least 3.36x speedups on Llama-3.1-8B for sequences longer than 64K while maintaining model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24163v1">LKD-KGC: Domain-Specific KG Construction via LLM-driven Knowledge Dependency Parsing</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Submitting to EDBT 2026
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) structure real-world entities and their relationships into triples, enhancing machine reasoning for various tasks. While domain-specific KGs offer substantial benefits, their manual construction is often inefficient and requires specialized knowledge. Recent approaches for knowledge graph construction (KGC) based on large language models (LLMs), such as schema-guided KGC and reference knowledge integration, have proven efficient. However, these methods are constrained by their reliance on manually defined schema, single-document processing, and public-domain references, making them less effective for domain-specific corpora that exhibit complex knowledge dependencies and specificity, as well as limited reference knowledge. To address these challenges, we propose LKD-KGC, a novel framework for unsupervised domain-specific KG construction. LKD-KGC autonomously analyzes document repositories to infer knowledge dependencies, determines optimal processing sequences via LLM driven prioritization, and autoregressively generates entity schema by integrating hierarchical inter-document contexts. This schema guides the unsupervised extraction of entities and relationships, eliminating reliance on predefined structures or external knowledge. Extensive experiments show that compared with state-of-the-art baselines, LKD-KGC generally achieves improvements of 10% to 20% in both precision and recall rate, demonstrating its potential in constructing high-quality domain-specific KGs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22571v3">Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11942v3">LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Project Page: https://caixd-220529.github.io/LifelongAgentBench/
    </div>
    <details class="paper-abstract">
      Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19217v2">LLM Benchmarking with LLaMA2: Evaluating Code Development Performance Across Multiple Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has opened new possibilities for automating various tasks in software development. This paper evaluates the capabilities of the Llama 2-70B model in automating these tasks for scientific applications written in commonly used programming languages. Using representative test problems, we assess the model's capacity to generate code, documentation, and unit tests, as well as its ability to translate existing code between commonly used programming languages. Our comprehensive analysis evaluates the compilation, runtime behavior, and correctness of the generated and translated code. Additionally, we assess the quality of automatically generated code, documentation and unit tests. Our results indicate that while Llama 2-70B frequently generates syntactically correct and functional code for simpler numerical tasks, it encounters substantial difficulties with more complex, parallelized, or distributed computations, requiring considerable manual corrections. We identify key limitations and suggest areas for future improvements to better leverage AI-driven automation in scientific computing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22630v2">Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      The widespread success of large language models (LLMs) on NLP benchmarks has been accompanied by concerns that LLMs function primarily as stochastic parrots that reproduce texts similar to what they saw during pre-training, often erroneously. But what is the nature of their errors, and do these errors exhibit any regularities? In this work, we examine irrelevant context hallucinations, in which models integrate misleading contextual cues into their predictions. Through behavioral analysis, we show that these errors result from a structured yet flawed mechanism that we term class-based (mis)generalization, in which models combine abstract class cues with features extracted from the query or context to derive answers. Furthermore, mechanistic interpretability experiments on Llama-3, Mistral, and Pythia across 39 factual recall relation types reveal that this behavior is reflected in the model's internal computations: (i) abstract class representations are constructed in lower layers before being refined into specific answers in higher layers, (ii) feature selection is governed by two competing circuits -- one prioritizing direct query-based reasoning, the other incorporating contextual cues -- whose relative influences determine the final output. Our findings provide a more nuanced perspective on the stochastic parrot argument: through form-based training, LLMs can exhibit generalization leveraging abstractions, albeit in unreliable ways based on contextual cues -- what we term stochastic chameleons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24119v1">The State of Multilingual LLM Safety Research: From Measuring the Language Gap to Mitigating It</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive analysis of the linguistic diversity of LLM safety research, highlighting the English-centric nature of the field. Through a systematic review of nearly 300 publications from 2020--2024 across major NLP conferences and workshops at *ACL, we identify a significant and growing language gap in LLM safety research, with even high-resource non-English languages receiving minimal attention. We further observe that non-English languages are rarely studied as a standalone language and that English safety research exhibits poor language documentation practice. To motivate future research into multilingual safety, we make several recommendations based on our survey, and we then pose three concrete future directions on safety evaluation, training data generation, and crosslingual safety generalization. Based on our survey and proposed directions, the field can develop more robust, inclusive AI safety practices for diverse global populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24107v1">GPTFootprint: Increasing Consumer Awareness of the Environmental Impacts of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Published in Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing System
    </div>
    <details class="paper-abstract">
      With the growth of AI, researchers are studying how to mitigate its environmental impact, primarily by proposing policy changes and increasing awareness among developers. However, research on AI end users is limited. Therefore, we introduce GPTFootprint, a browser extension that aims to increase consumer awareness of the significant water and energy consumption of LLMs, and reduce unnecessary LLM usage. GPTFootprint displays a dynamically updating visualization of the resources individual users consume through their ChatGPT queries. After a user reaches a set query limit, a popup prompts them to take a break from ChatGPT. In a week-long user study, we found that GPTFootprint increases people's awareness of environmental impact, but has limited success in decreasing ChatGPT usage. This research demonstrates the potential for individual-level interventions to contribute to the broader goal of sustainable AI usage, and provides insights into the effectiveness of awareness-based behavior modification strategies in the context of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24105v1">Training LLMs for EHR-Based Reasoning Tasks via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      We present EHRMIND, a practical recipe for adapting large language models (LLMs) to complex clinical reasoning tasks using reinforcement learning with verifiable rewards (RLVR). While RLVR has succeeded in mathematics and coding, its application to healthcare contexts presents unique challenges due to the specialized knowledge and reasoning required for electronic health record (EHR) interpretation. Our pilot study on the MEDCALC benchmark reveals two key failure modes: (1) misapplied knowledge, where models possess relevant medical knowledge but apply it incorrectly, and (2) missing knowledge, where models lack essential domain knowledge. To address these cases, EHRMIND applies a two-stage solution: a lightweight supervised fine-tuning (SFT) warm-up that injects missing domain knowledge, stabilizes subsequent training, and encourages structured, interpretable outputs; followed by RLVR, which reinforces outcome correctness and refines the model's decision-making. We demonstrate the effectiveness of our method across diverse clinical applications, including medical calculations (MEDCALC), patient-trial matching (TREC CLINICAL TRIALS), and disease diagnosis (EHRSHOT). EHRMIND delivers consistent gains in accuracy, interpretability, and cross-task generalization. These findings offer practical guidance for applying RLVR to enhance LLM capabilities in healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09909v2">AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Updated formatting
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in parameter size and context length, computation precision has been reduced from 16-bit to 4-bit to improve inference efficiency. However, this reduction causes accuracy degradation due to activation outliers. Rotation-based INT4 methods address this via matrix calibration, but they introduce multi-hour overheads and leave key computations in full precision. Microscaling (MX) floating-point (FP) formats offer fine-grained representation with a shared scale, enabling fully quantized matrix multiplications through direct casting without calibration. However, existing research shows unsatisfactory empirical results for MXFP4 inference, and the robustness of MX formats remains largely unexplored. In this work, we uncover the fundamental tradeoffs of the MX format: while it effectively suppresses activation outliers, it does so at the cost of increased group-wise asymmetry. To address this, we propose AMXFP4, a 4-bit asymmetric FP format that handles both issues using asymmetric shared scales, without requiring calibration. Our custom MAC engine adds negligible hardware cost while improving accuracy: AMXFP4 outperforms MXFP4 by 3% on VQA and exceeds rotation-based methods by 1.6% on CSQA. It also surpasses recently deployed commercial MXFP4 variants. Code: https://github.com/aiha-lab/MX-QLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24098v1">HardTests: Synthesizing High-Quality Test Cases for LLM Coding</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Verifiers play a crucial role in large language model (LLM) reasoning, needed by post-training techniques such as reinforcement learning. However, reliable verifiers are hard to get for difficult coding problems, because a well-disguised wrong solution may only be detected by carefully human-written edge cases that are difficult to synthesize. To address this issue, we propose HARDTESTGEN, a pipeline for high-quality test synthesis using LLMs. With this pipeline, we curate a comprehensive competitive programming dataset HARDTESTS with 47k problems and synthetic high-quality tests. Compared with existing tests, HARDTESTGEN tests demonstrate precision that is 11.3 percentage points higher and recall that is 17.5 percentage points higher when evaluating LLM-generated code. For harder problems, the improvement in precision can be as large as 40 points. HARDTESTS also proves to be more effective for model training, measured by downstream code generation performance. We will open-source our dataset and synthesis pipeline at https://leililab.github.io/HardTests/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24095v1">SkyLB: A Locality-Aware Cross-Region Load Balancer for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) efficiently in multi-region setups remains a challenge. Due to cost and GPU availability concerns, providers typically deploy LLMs in multiple regions using instance with long-term commitments, like reserved instances or on-premise clusters, which are often underutilized due to their region-local traffic handling and diurnal traffic variance. In this paper, we introduce SkyLB, a locality-aware multi-region load balancer for LLM inference that aggregates regional diurnal patterns through cross-region traffic handling. By doing so, SkyLB enables providers to reserve instances based on expected global demand, rather than peak demand in each individual region. Meanwhile, SkyLB preserves KV-Cache locality and a balanced load, ensuring cost efficiency without sacrificing performance. SkyLB achieves this with a cache-aware cross-region traffic handler and a selective pushing load balancing mechanism based on checking pending requests. Our evaluation on real-world workloads shows that it achieves 1.12-2.06x higher throughput and 1.74-6.30x lower latency compared to existing load balancers, while reducing total serving cost by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17116v3">Star Attention: Efficient LLM Inference over Long Sequences</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Inference with Transformer-based Large Language Models (LLMs) on long sequences is both costly and slow due to the quadratic complexity of the self-attention mechanism. We introduce Star Attention, a two-phase block-sparse approximation that improves computational efficiency by sharding attention across multiple hosts while minimizing communication overhead. In the first phase, the context is processed using blockwise-local attention across hosts, in parallel. In the second phase, query and response tokens attend to all prior cached tokens through sequence-global attention. Star Attention integrates seamlessly with most Transformer-based LLMs trained with global attention, reducing memory requirements and inference time by up to 11x while preserving 97-100% of accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01617v2">If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Warning: this paper discusses content related, but not limited to, violence, sex, and suicide. Loneliness, or the lack of fulfilling relationships, significantly impacts a person's mental and physical well-being and is prevalent worldwide. Previous research suggests that large language models (LLMs) may help mitigate loneliness. However, we argue that the use of widespread LLMs in services like ChatGPT is more prevalent--and riskier, as they are not designed for this purpose. To explore this, we analysed user interactions with ChatGPT outside of its marketed use as a task-oriented assistant. In dialogues classified as lonely, users frequently (37%) sought advice or validation, and received good engagement. However, ChatGPT failed in sensitive scenarios, like responding appropriately to suicidal ideation or trauma. We also observed a 35% higher incidence of toxic content, with women being 22x more likely to be targeted than men. Our findings underscore ethical and legal questions about this technology, and note risks like radicalisation or further isolation. We conclude with recommendations to research and industry to address loneliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15409v3">Awes, Laws, and Flaws From Today's LLM Research</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      We perform a critical examination of the scientific methodology behind contemporary large language model (LLM) research. For this we assess over 2,000 research works released between 2020 and 2024 based on criteria typical of what is considered good research (e.g. presence of statistical tests and reproducibility), and cross-validate it with arguments that are at the centre of controversy (e.g., claims of emergent behaviour). We find multiple trends, such as declines in ethics disclaimers, a rise of LLMs as evaluators, and an increase on claims of LLM reasoning abilities without leveraging human evaluation. We note that conference checklists are effective at curtailing some of these issues, but balancing velocity and rigour in research cannot solely rely on these. We tie all these findings to findings from recent meta-reviews and extend recommendations on how to address what does, does not, and should work in LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10508v5">White Men Lead, Black Women Help? Benchmarking and Mitigating Language Agency Social Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Social biases can manifest in language agency. However, very limited research has investigated such biases in Large Language Model (LLM)-generated content. In addition, previous works often rely on string-matching techniques to identify agentic and communal words within texts, falling short of accurately classifying language agency. We introduce the Language Agency Bias Evaluation (LABE) benchmark, which comprehensively evaluates biases in LLMs by analyzing agency levels attributed to different demographic groups in model generations. LABE tests for gender, racial, and intersectional language agency biases in LLMs on 3 text generation tasks: biographies, professor reviews, and reference letters. Using LABE, we unveil language agency social biases in 3 recent LLMs: ChatGPT, Llama3, and Mistral. We observe that: (1) LLM generations tend to demonstrate greater gender bias than human-written texts; (2) Models demonstrate remarkably higher levels of intersectional bias than the other bias aspects. (3) Prompt-based mitigation is unstable and frequently leads to bias exacerbation. Based on our observations, we propose Mitigation via Selective Rewrite (MSR), a novel bias mitigation strategy that leverages an agency classifier to identify and selectively revise parts of generated texts that demonstrate communal traits. Empirical results prove MSR to be more effective and reliable than prompt-based mitigation method, showing a promising research direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16456v2">Position: Beyond Assistance -- Reimagining LLMs as Ethical and Adaptive Co-Creators in Mental Health Care</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      This position paper argues for a fundamental shift in how Large Language Models (LLMs) are integrated into the mental health care domain. We advocate for their role as co-creators rather than mere assistive tools. While LLMs have the potential to enhance accessibility, personalization, and crisis intervention, their adoption remains limited due to concerns about bias, evaluation, over-reliance, dehumanization, and regulatory uncertainties. To address these challenges, we propose two structured pathways: SAFE-i (Supportive, Adaptive, Fair, and Ethical Implementation) Guidelines for ethical and responsible deployment, and HAAS-e (Human-AI Alignment and Safety Evaluation) Framework for multidimensional, human-centered assessment. SAFE-i provides a blueprint for data governance, adaptive model engineering, and real-world integration, ensuring LLMs align with clinical and ethical standards. HAAS-e introduces evaluation metrics that go beyond technical accuracy to measure trustworthiness, empathy, cultural sensitivity, and actionability. We call for the adoption of these structured approaches to establish a responsible and scalable model for LLM-driven mental health support, ensuring that AI complements, rather than replaces, human expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13259v2">HumT DumT: Measuring and controlling human-like language in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Should LLMs generate language that makes them seem human? Human-like language might improve user experience, but might also lead to deception, overreliance, and stereotyping. Assessing these potential impacts requires a systematic way to measure human-like tone in LLM outputs. We introduce HumT and SocioT, metrics for human-like tone and other dimensions of social perceptions in text data based on relative probabilities from an LLM. By measuring HumT across preference and usage datasets, we find that users prefer less human-like outputs from LLMs in many contexts. HumT also offers insights into the perceptions and impacts of anthropomorphism: human-like LLM outputs are highly correlated with warmth, social closeness, femininity, and low status, which are closely linked to the aforementioned harms. We introduce DumT, a method using HumT to systematically control and reduce the degree of human-like tone while preserving model performance. DumT offers a practical approach for mitigating risks associated with anthropomorphic language generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22919v2">ER-REASON: A Benchmark Dataset for LLM-Based Clinical Reasoning in the Emergency Room</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been extensively evaluated on medical question answering tasks based on licensing exams. However, real-world evaluations often depend on costly human annotators, and existing benchmarks tend to focus on isolated tasks that rarely capture the clinical reasoning or full workflow underlying medical decisions. In this paper, we introduce ER-Reason, a benchmark designed to evaluate LLM-based clinical reasoning and decision-making in the emergency room (ER)--a high-stakes setting where clinicians make rapid, consequential decisions across diverse patient presentations and medical specialties under time pressure. ER-Reason includes data from 3,984 patients, encompassing 25,174 de-identified longitudinal clinical notes spanning discharge summaries, progress notes, history and physical exams, consults, echocardiography reports, imaging notes, and ER provider documentation. The benchmark includes evaluation tasks that span key stages of the ER workflow: triage intake, initial assessment, treatment selection, disposition planning, and final diagnosis--each structured to reflect core clinical reasoning processes such as differential diagnosis via rule-out reasoning. We also collected 72 full physician-authored rationales explaining reasoning processes that mimic the teaching process used in residency training, and are typically absent from ER documentation. Evaluations of state-of-the-art LLMs on ER-Reason reveal a gap between LLM-generated and clinician-authored clinical reasoning for ER decisions, highlighting the need for future research to bridge this divide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19449v2">VecTrans: Enhancing Compiler Auto-Vectorization through LLM-Assisted Code Transformations</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Auto-vectorization is a fundamental optimization for modern compilers to exploit SIMD parallelism. However, state-of-the-art approaches still struggle to handle intricate code patterns, often requiring manual hints or domain-specific expertise. Large language models (LLMs), with their ability to capture intricate patterns, provide a promising solution, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. In this paper, we present VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compilers auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. experimental results show that among all TSVC functions unvectorizable by GCC, ICC, Clang, and BiSheng Compiler, VecTrans achieves an geomean speedup of 1.77x and successfully vectorizes 24 of 51 test cases. This marks a significant advancement over state-of-the-art approaches while maintaining a cost efficiency of $0.012 per function optimization for LLM API usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23754v1">DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23723v1">ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The emergence of large language model (LLM)-based agents has significantly advanced the development of autonomous machine learning (ML) engineering. However, most existing approaches rely heavily on manual prompt engineering, failing to adapt and optimize based on diverse experimental experiences. Focusing on this, for the first time, we explore the paradigm of learning-based agentic ML, where an LLM agent learns through interactive experimentation on ML tasks using online reinforcement learning (RL). To realize this, we propose a novel agentic ML training framework with three key components: (1) exploration-enriched fine-tuning, which enables LLM agents to generate diverse actions for enhanced RL exploration; (2) step-wise RL, which enables training on a single action step, accelerating experience collection and improving training efficiency; (3) an agentic ML-specific reward module, which unifies varied ML feedback signals into consistent rewards for RL optimization. Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM for autonomous ML. Remarkably, despite being trained on merely 9 ML tasks, our 7B-sized ML-Agent outperforms the 671B-sized DeepSeek-R1 agent. Furthermore, it achieves continuous performance improvements and demonstrates exceptional cross-task generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v1">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the *NL-FL Problem Alignment* method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the *Mixed Problem Input* technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive experiments demonstrate that the **HybridReasoning** framework achieves **89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23701v1">Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Final-answer-based metrics are commonly used for evaluating large language models (LLMs) on math word problems, often taken as proxies for reasoning ability. However, such metrics conflate two distinct sub-skills: abstract formulation (capturing mathematical relationships using expressions) and arithmetic computation (executing the calculations). Through a disentangled evaluation on GSM8K and SVAMP, we find that the final-answer accuracy of Llama-3 and Qwen2.5 (1B-32B) without CoT is overwhelmingly bottlenecked by the arithmetic computation step and not by the abstract formulation step. Contrary to the common belief, we show that CoT primarily aids in computation, with limited impact on abstract formulation. Mechanistically, we show that these two skills are composed conjunctively even in a single forward pass without any reasoning steps via an abstract-then-compute mechanism: models first capture problem abstractions, then handle computation. Causal patching confirms these abstractions are present, transferable, composable, and precede computation. These behavioural and mechanistic findings highlight the need for disentangled evaluation to accurately assess LLM reasoning and to guide future improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23695v1">Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs has led to the creation of diverse agentic systems in data analysis, utilizing LLMs' capabilities to improve insight generation and visualization. In this paper, we present an agentic system that automates the data-to-dashboard pipeline through modular LLM agents capable of domain detection, concept extraction, multi-perspective analysis generation, and iterative self-reflection. Unlike existing chart QA systems, our framework simulates the analytical reasoning process of business analysts by retrieving domain-relevant knowledge and adapting to diverse datasets without relying on closed ontologies or question templates. We evaluate our system on three datasets across different domains. Benchmarked against GPT-4o with a single-prompt baseline, our approach shows improved insightfulness, domain relevance, and analytical depth, as measured by tailored evaluation metrics and qualitative human assessment. This work contributes a novel modular pipeline to bridge the path from raw data to visualization, and opens new opportunities for human-in-the-loop validation by domain experts in business analytics. All code can be found here: https://github.com/77luvC/D2D_Data2Dashboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09615v3">SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published at Proceedings of the 42 nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Conventional model compression techniques for LLMs address high memory consumption and slow inference challenges but typically require computationally expensive retraining to preserve accuracy. In contrast, one-shot compression methods eliminate retraining cost, but struggle to achieve accuracy comparable to dense models. This paper presents SLIM, a new one-shot compression framework that holistically integrates hardware-friendly quantization, sparsity, and low-rank approximation into a unified process. First, we formulate the quantization process using a probabilistic approach (SLIM-Quant) that enables us to apply uniform quantization. Then, we use an existing one-shot pruning method to apply semi-structured sparsity on top of the quantized weights. Finally, to compensate for the introduced aggregated quantization and sparsity error, we use a novel saliency function with unique invertible and additive features that enables us to mathematically compute the value of low-rank adapters. SLIM improves model accuracy by up to 5.66% (LLaMA-2-7B) for 2:4 sparsity with 4-bit weight quantization, outperforming prior methods. Models compressed with SLIM achieve up to 4.3x and 3.8x on Nvidia RTX3060 and A100 GPUs, respectively. Additionally, they achieve up to 0.23x end-to-end memory reduction in comparison to their dense counterparts. We also propose an optional PEFT recipe that further improves accuracy by up to 1.66% (LLaMA-2-13B) compared to SLIM without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23654v1">ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization with Instruction Following LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Integrating structured information has long improved the quality of abstractive summarization, particularly in retaining salient content. In this work, we focus on a specific form of structure: argument roles, which are crucial for summarizing documents in high-stakes domains such as law. We investigate whether instruction-tuned large language models (LLMs) adequately preserve this information. To this end, we introduce Argument Representation Coverage (ARC), a framework for measuring how well LLM-generated summaries capture salient arguments. Using ARC, we analyze summaries produced by three open-weight LLMs in two domains where argument roles are central: long legal opinions and scientific articles. Our results show that while LLMs cover salient argument roles to some extent, critical information is often omitted in generated summaries, particularly when arguments are sparsely distributed throughout the input. Further, we use ARC to uncover behavioral patterns -- specifically, how the positional bias of LLM context windows and role-specific preferences impact the coverage of key arguments in generated summaries, emphasizing the need for more argument-aware summarization strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04358v2">Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025); 13 pages including references
    </div>
    <details class="paper-abstract">
      Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14279v2">YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 4 figures, Accepted as a Long Paper at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16502v2">RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint. Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11942v2">LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14276v2">STeCa: Step-level Trajectory Calibration for LLM Agent Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations or preference learning through exploratory trajectory sampling. However, these methods often struggle to address long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. We finally leverage these calibrated trajectories with successful trajectories for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that timely calibration enables agents to complete tasks with greater robustness. Our code and data are available at https://github.com/WangHanLinHenry/STeCa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23598v1">LLM Performance for Code Generation on Noisy Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This paper investigates the ability of large language models (LLMs) to recognise and solve tasks which have been obfuscated beyond recognition. Focusing on competitive programming and benchmark tasks (LeetCode and MATH), we compare performance across multiple models and obfuscation methods, such as noise and redaction. We demonstrate that all evaluated LLMs can solve tasks obfuscated to a level where the text would be unintelligible to human readers, and does not contain key pieces of instruction or context. We introduce the concept of eager pattern matching to describe this behaviour, which is not observed in tasks published after the models' knowledge cutoff date, indicating strong memorisation or overfitting to training data, rather than legitimate reasoning about the presented problem. We report empirical evidence of distinct performance decay patterns between contaminated and unseen datasets. We discuss the implications for benchmarking and evaluations of model behaviour, arguing for caution when designing experiments using standard datasets. We also propose measuring the decay of performance under obfuscation as a possible strategy for detecting dataset contamination and highlighting potential safety risks and interpretability issues for automated software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23579v1">BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 16 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Unlocking deep, interpretable biological reasoning from complex genomic data is a major AI challenge hindering scientific discovery. Current DNA foundation models, despite strong sequence representation, struggle with multi-step reasoning and lack inherent transparent, biologically intuitive explanations. We introduce BioReason, a pioneering architecture that, for the first time, deeply integrates a DNA foundation model with a Large Language Model (LLM). This novel connection enables the LLM to directly process and reason with genomic information as a fundamental input, fostering a new form of multimodal biological understanding. BioReason's sophisticated multi-step reasoning is developed through supervised fine-tuning and targeted reinforcement learning, guiding the system to generate logical, biologically coherent deductions. On biological reasoning benchmarks including KEGG-based disease pathway prediction - where accuracy improves from 88% to 97% - and variant effect prediction, BioReason demonstrates an average 15% performance gain over strong single-modality baselines. BioReason reasons over unseen biological entities and articulates decision-making through interpretable, step-by-step biological traces, offering a transformative approach for AI in biology that enables deeper mechanistic insights and accelerates testable hypothesis generation from genomic data. Data, code, and checkpoints are publicly available at https://github.com/bowang-lab/BioReason
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23559v1">SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04629v4">Argumentative Experience: Reducing Confirmation Bias on Controversial Issues through LLM-Generated Multi-Persona Debates</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are enabling designers to give life to exciting new user experiences for information access. In this work, we present a system that generates LLM personas to debate a topic of interest from different perspectives. How might information seekers use and benefit from such a system? Can centering information access around diverse viewpoints help to mitigate thorny challenges like confirmation bias in which information seekers over-trust search results matching existing beliefs? How do potential biases and hallucinations in LLMs play out alongside human users who are also fallible and possibly biased? Our study exposes participants to multiple viewpoints on controversial issues via a mixed-methods, within-subjects study. We use eye-tracking metrics to quantitatively assess cognitive engagement alongside qualitative feedback. Compared to a baseline search system, we see more creative interactions and diverse information-seeking with our multi-persona debate system, which more effectively reduces user confirmation bias and conviction toward their initial beliefs. Overall, our study contributes to the emerging design space of LLM-based information access systems, specifically investigating the potential of simulated personas to promote greater exposure to information diversity, emulate collective intelligence, and mitigate bias in information seeking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23554v1">Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLM) such as ChatGPT, CoPilot, and Gemini have been widely adopted in different areas. As the use of LLMs continues to grow, many efforts have focused on reducing the massive training overheads of these models. But it is the environmental impact of handling user requests to LLMs that is increasingly becoming a concern. Recent studies estimate that the costs of operating LLMs in their inference phase can exceed training costs by 25x per year. As LLMs are queried incessantly, the cumulative carbon footprint for the operational phase has been shown to far exceed the footprint during the training phase. Further, estimates indicate that 500 ml of fresh water is expended for every 20-50 requests to LLMs during inference. To address these important sustainability issues with LLMs, we propose a novel framework called SLIT to co-optimize LLM quality of service (time-to-first token), carbon emissions, water usage, and energy costs. The framework utilizes a machine learning (ML) based metaheuristic to enhance the sustainability of LLM hosting across geo-distributed cloud datacenters. Such a framework will become increasingly vital as LLMs proliferate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23549v1">LLM-based Property-based Test Generation for Guardrailing Cyber-Physical Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Cyber-physical systems (CPSs) are complex systems that integrate physical, computational, and communication subsystems. The heterogeneous nature of these systems makes their safety assurance challenging. In this paper, we propose a novel automated approach for guardrailing cyber-physical systems using property-based tests (PBTs) generated by Large Language Models (LLMs). Our approach employs an LLM to extract properties from the code and documentation of CPSs. Next, we use the LLM to generate PBTs that verify the extracted properties on the CPS. The generated PBTs have two uses. First, they are used to test the CPS before it is deployed, i.e., at design time. Secondly, these PBTs can be used after deployment, i.e., at run time, to monitor the behavior of the system and guardrail it against unsafe states. We implement our approach in ChekProp and conduct preliminary experiments to evaluate the generated PBTs in terms of their relevance (how well they match manually crafted properties), executability (how many run with minimal manual modification), and effectiveness (coverage of the input space partitions). The results of our experiments and evaluation demonstrate a promising path forward for creating guardrails for CPSs using LLM-generated property-based tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11077v2">LLM-Enhanced Symbolic Control for Safety-Critical Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Motivated by Smart Manufacturing and Industry 4.0, we introduce a framework for synthesizing Abstraction-Based Controller Design (ABCD) for reach-avoid problems from Natural Language (NL) specifications using Large Language Models (LLMs). A Code Agent interprets an NL description of the control problem and translates it into a formal language interpretable by state-of-the-art symbolic control software, while a Checker Agent verifies the correctness of the generated code and enhances safety by identifying specification mismatches. Evaluations show that the system handles linguistic variability and improves robustness over direct planning with LLMs. The proposed approach lowers the barrier to formal control synthesis by enabling intuitive, NL-based task definition while maintaining safety guarantees through automated validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23540v1">Probability-Consistent Preference Optimization for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 14 pages, to be published in ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Recent advances in preference optimization have demonstrated significant potential for improving mathematical reasoning capabilities in large language models (LLMs). While current approaches leverage high-quality pairwise preference data through outcome-based criteria like answer correctness or consistency, they fundamentally neglect the internal logical coherence of responses. To overcome this, we propose Probability-Consistent Preference Optimization (PCPO), a novel framework that establishes dual quantitative metrics for preference selection: (1) surface-level answer correctness and (2) intrinsic token-level probability consistency across responses. Extensive experiments show that our PCPO consistently outperforms existing outcome-only criterion approaches across a diverse range of LLMs and benchmarks. Our code is publicly available at https://github.com/YunqiaoYang/PCPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09955v3">MegaAgent: A Large-Scale Autonomous LLM-based Multi-Agent System Without Predefined SOPs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) have shown promise in tackling complex tasks. However, existing solutions often suffer from limited agent coordination and heavy reliance on predefined Standard Operating Procedures (SOPs), which demand extensive human input. To address these limitations, we propose MegaAgent, a large-scale autonomous LLM-based multi-agent system. MegaAgent generates agents based on task complexity and enables dynamic task decomposition, parallel execution, efficient communication, and comprehensive system monitoring of agents. In evaluations, MegaAgent demonstrates exceptional performance, successfully developing a Gobang game within 800 seconds and scaling up to 590 agents in a national policy simulation to generate multi-domain policies. It significantly outperforms existing systems, such as MetaGPT, in both task completion efficiency and scalability. By eliminating the need for predefined SOPs, MegaAgent demonstrates exceptional scalability and autonomy, setting a foundation for advancing true autonomy in MAS. Our code is available at https://github.com/Xtra-Computing/MegaAgent .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23473v1">EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03793v2">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by ICML'2025
    </div>
    <details class="paper-abstract">
      The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a PAC-Bayesian Generalization Bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel (NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at LensLLM.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23461v1">UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Handling unanswerable questions (UAQ) is crucial for LLMs, as it helps prevent misleading responses in complex situations. While previous studies have built several datasets to assess LLMs' performance on UAQ, these datasets lack factual knowledge support, which limits the evaluation of LLMs' ability to utilize their factual knowledge when handling UAQ. To address the limitation, we introduce a new unanswerable question dataset UAQFact, a bilingual dataset with auxiliary factual knowledge created from a Knowledge Graph. Based on UAQFact, we further define two new tasks to measure LLMs' ability to utilize internal and external factual knowledge, respectively. Our experimental results across multiple LLM series show that UAQFact presents significant challenges, as LLMs do not consistently perform well even when they have factual knowledge stored. Additionally, we find that incorporating external knowledge may enhance performance, but LLMs still cannot make full use of the knowledge which may result in incorrect responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21807v2">TabReason: A Reinforcement Learning-Enhanced Reasoning LLM for Explainable Tabular Data Prediction</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Predictive modeling on tabular data is the cornerstone of many real-world applications. Although gradient boosting machines and some recent deep models achieve strong performance on tabular data, they often lack interpretability. On the other hand, large language models (LLMs) have demonstrated powerful capabilities to generate human-like reasoning and explanations, but remain under-performed for tabular data prediction. In this paper, we propose a new approach that leverages reasoning-based LLMs, trained using reinforcement learning, to perform more accurate and explainable predictions on tabular data. Our method introduces custom reward functions that guide the model not only toward high prediction accuracy but also toward human-understandable reasons for its predictions. Experimental results show that our model achieves promising performance on financial benchmark datasets, outperforming most existing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12913v3">GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to BF16-based fine-tuning while significantly reducing 1.85x memory usage. Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08120v2">DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization?</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Reasoning-enabled large language models (LLMs) excel in logical tasks, yet their utility for evaluating natural language generation remains unexplored. This study systematically compares reasoning LLMs with non-reasoning counterparts across machine translation and text summarization evaluation tasks. We evaluate eight models spanning state-of-the-art reasoning models (DeepSeek-R1, OpenAI o3), their distilled variants (8B-70B parameters), and equivalent non-reasoning LLMs. Experiments on WMT23 and SummEval benchmarks reveal architecture and task-dependent benefits: OpenAI o3-mini models show improved performance with increased reasoning on MT, while DeepSeek-R1 and generally underperforms compared to its non-reasoning variant except in summarization consistency evaluation. Correlation analysis demonstrates that reasoning token usage correlates with evaluation quality only in specific models, while almost all models generally allocate more reasoning tokens when identifying more quality issues. Distillation maintains reasonable performance up to 32B parameter models but degrades substantially at 8B scale. This work provides the first assessment of reasoning LLMs for NLG evaluation and comparison to non-reasoning models. We share our code to facilitate further research: https://github.com/NL2G/reasoning-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18325v2">Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 We have identified significant errors in the results presented in this paper, specifically in the evaluation sections concerning the DPO training of LLaMA2 and Qwen2.5, as well as in the representation space visualization section. Given the extent of these issues, we intend to substantially revise the manuscript's content and structure. Hence, we request to withdraw it from arXiv at this time
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they often refuse to answer legitimate queries-a phenomenon known as overrefusal. Overrefusal typically stems from over-conservative safety alignment, causing models to treat many reasonable prompts as potentially risky. To systematically understand this issue, we probe and leverage the models'safety decision boundaries to analyze and mitigate overrefusal. Our findings reveal that overrefusal is closely tied to misalignment at these boundary regions, where models struggle to distinguish subtle differences between benign and harmful content. Building on these insights, we present RASS, an automated framework for prompt generation and selection that strategically targets overrefusal prompts near the safety boundary. By harnessing steering vectors in the representation space, RASS efficiently identifies and curates boundary-aligned prompts, enabling more effective and targeted mitigation of overrefusal. This approach not only provides a more precise and interpretable view of model safety decisions but also seamlessly extends to multilingual scenarios.We have explored the safety decision boundaries of various LLMs and construct the MORBench evaluation set to facilitate robust assessment of model safety and helpfulness across multiple languages. Code and datasets will be released at https://anonymous.4open.science/r/RASS-80D3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16983v2">LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09570v2">LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Camera ready version for ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      When the complete source sentence is provided, Large Language Models (LLMs) perform excellently in offline machine translation even with a simple prompt "Translate the following sentence from [src lang] into [tgt lang]:". However, in many real scenarios, the source tokens arrive in a streaming manner and simultaneous machine translation (SiMT) is required, then the efficiency and performance of decoder-only LLMs are significantly limited by their auto-regressive nature. To enable LLMs to achieve high-quality SiMT as efficiently as offline translation, we propose a novel paradigm that includes constructing supervised fine-tuning (SFT) data for SiMT, along with new training and inference strategies. To replicate the token input/output stream in SiMT, the source and target tokens are rearranged into an interleaved sequence, separated by special tokens according to varying latency requirements. This enables powerful LLMs to learn read and write operations adaptively, based on varying latency prompts, while still maintaining efficient auto-regressive decoding. Experimental results show that, even with limited SFT data, our approach achieves state-of-the-art performance across various SiMT benchmarks, and preserves the original abilities of offline translation. Moreover, our approach generalizes well to document-level SiMT setting without requiring specific fine-tuning, even beyond the offline translation model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23410v1">From Parameters to Prompts: Understanding and Mitigating the Factuality Gap between Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 The code of this paper will be released soon
    </div>
    <details class="paper-abstract">
      Factual knowledge extraction aims to explicitly extract knowledge parameterized in pre-trained language models for application in downstream tasks. While prior work has been investigating the impact of supervised fine-tuning data on the factuality of large language models (LLMs), its mechanism remains poorly understood. We revisit this impact through systematic experiments, with a particular focus on the factuality gap that arises when fine-tuning on known versus unknown knowledge. Our findings show that this gap can be mitigated at the inference stage, either under out-of-distribution (OOD) settings or by using appropriate in-context learning (ICL) prompts (i.e., few-shot learning and Chain of Thought (CoT)). We prove this phenomenon theoretically from the perspective of knowledge graphs, showing that the test-time prompt may diminish or even overshadow the impact of fine-tuning data and play a dominant role in knowledge extraction. Ultimately, our results shed light on the interaction between finetuning data and test-time prompt, demonstrating that ICL can effectively compensate for shortcomings in fine-tuning data, and highlighting the need to reconsider the use of ICL prompting as a means to evaluate the effectiveness of fine-tuning data selection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21889v2">EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 31st International European Conference on Parallel and Distributed Computing (Euro-Par 2025 Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability. EFIM's source code is publicly available at https://github.com/gty111/EFIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17057v3">ExpandR: Teaching Dense Retrievers Beyond Queries with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 16 pages, 10 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in enhancing dense retrieval through query augmentation. However, most existing methods treat the LLM and the retriever as separate modules, overlooking the alignment between generation and ranking objectives. In this work, we propose ExpandR, a unified LLM-augmented dense retrieval framework that jointly optimizes both the LLM and the retriever. ExpandR employs the LLM to generate semantically rich query expansions, which are leveraged to enhance the retriever's training. Simultaneously, the LLM is trained using Direct Preference Optimization (DPO), guided by a carefully designed reward function that balances retrieval effectiveness and generation consistency. This joint optimization paradigm enables mutual adaptation between the LLM and the retriever, resulting in query expansions that are both informative and well-suited for retrieval. Experimental results on multiple benchmarks show that ExpandR consistently outperforms strong baselines, achieving more than a 5% improvement in retrieval performance. All codes are available at https://github.com/NEUIR/ExpandR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23323v1">Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In this position paper we raise critical awareness of a realistic view of LLM capabilities that eschews extreme alternative views that LLMs are either "stochastic parrots" or in possession of "emergent" advanced reasoning capabilities, which, due to their unpredictable emergence, constitute an existential threat. Our middle-ground view is that LLMs extrapolate from priors from their training data, and that a mechanism akin to in-context learning enables the targeting of the appropriate information from which to extrapolate. We call this "context-directed extrapolation." Under this view, substantiated though existing literature, while reasoning capabilities go well beyond stochastic parroting, such capabilities are predictable, controllable, not indicative of advanced reasoning akin to high-level cognitive capabilities in humans, and not infinitely scalable with additional training. As a result, fears of uncontrollable emergence of agency are allayed, while research advances are appropriately refocused on the processes of context-directed extrapolation and how this interacts with training data to produce valuable capabilities in LLMs. Future work can therefore explore alternative augmenting techniques that do not rely on inherent advanced reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23311v1">Towards LLM-based Generation of Human-Readable Proofs in Polynomial Formal Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 4 pages; keynote given at 7th International Symposium on Devices, Circuits and Systems (ISDCS 2025), May 27-30, 2025, IIEST Shibpur, Kolkata, India
    </div>
    <details class="paper-abstract">
      Verification is one of the central tasks in circuit and system design. While simulation and emulation are widely used, complete correctness can only be ensured based on formal proof techniques. But these approaches often have very high run time and memory requirements. Recently, Polynomial Formal Verification (PFV) has been introduced showing that for many instances of practical relevance upper bounds on needed resources can be given. But proofs have to be provided that are human-readable. Here, we study how modern approaches from Artificial Intelligence (AI) based on Large Language Models (LLMs) can be used to generate proofs that later on can be validated based on reasoning engines. Examples are given that show how LLMs can interact with proof engines, and directions for future work are outlined.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23304v1">Generalized Category Discovery in Event-Centric Contexts: Latent Pattern Mining with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Generalized Category Discovery (GCD) aims to classify both known and novel categories using partially labeled data that contains only known classes. Despite achieving strong performance on existing benchmarks, current textual GCD methods lack sufficient validation in realistic settings. We introduce Event-Centric GCD (EC-GCD), characterized by long, complex narratives and highly imbalanced class distributions, posing two main challenges: (1) divergent clustering versus classification groupings caused by subjective criteria, and (2) Unfair alignment for minority classes. To tackle these, we propose PaMA, a framework leveraging LLMs to extract and refine event patterns for improved cluster-class alignment. Additionally, a ranking-filtering-mining pipeline ensures balanced representation of prototypes across imbalanced categories. Evaluations on two EC-GCD benchmarks, including a newly constructed Scam Report dataset, demonstrate that PaMA outperforms prior methods with up to 12.58% H-score gains, while maintaining strong generalization on base GCD datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23299v1">Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly deployed in industry applications, yet their reliability remains hampered by challenges in detecting hallucinations. While supervised state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as activation tracing and representation analysis -- show promise, their dependence on extensively annotated datasets limits scalability in real-world applications. This paper addresses the critical bottleneck of data annotation by investigating the feasibility of reducing training data requirements for two SOTA hallucination detection frameworks: Lookback Lens, which analyzes attention head dynamics, and probing-based approaches, which decode internal model representations. We propose a methodology combining efficient classification algorithms with dimensionality reduction techniques to minimize sample size demands while maintaining competitive performance. Evaluations on standardized question-answering RAG benchmarks show that our approach achieves performance comparable to strong proprietary LLM-based baselines with only 250 training samples. These results highlight the potential of lightweight, data-efficient paradigms for industrial deployment, particularly in annotation-constrained scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08948v2">Emergent social conventions and collective bias in LLM populations</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Social conventions are the backbone of social coordination, shaping how individuals form a group. As growing populations of artificial intelligence (AI) agents communicate through natural language, a fundamental question is whether they can bootstrap the foundations of a society. Here, we present experimental results that demonstrate the spontaneous emergence of universally adopted social conventions in decentralized populations of large language model (LLM) agents. We then show how strong collective biases can emerge during this process, even when agents exhibit no bias individually. Last, we examine how committed minority groups of adversarial LLM agents can drive social change by imposing alternative social conventions on the larger population. Our results show that AI systems can autonomously develop social conventions without explicit programming and have implications for designing AI systems that align, and remain aligned, with human values and societal goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v4">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23281v1">MathArena: Evaluating LLMs on Uncontaminated Math Competitions</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as SMT 2025 -- published well after model release dates -- demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On USAMO 2025, even top models score below 25%, far behind their performance on final-answer tasks. So far, we have evaluated 30 models across five competitions, totaling 149 problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23277v1">Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint. 17 pages including appendix
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external context, but retrieved passages are often lengthy, noisy, or exceed input limits. Existing compression methods typically require supervised training of dedicated compression models, increasing cost and reducing portability. We propose Sentinel, a lightweight sentence-level compression framework that reframes context filtering as an attention-based understanding task. Rather than training a compression model, Sentinel probes decoder attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier to identify sentence relevance. Empirically, we find that query-context relevance estimation is consistent across model scales, with 0.5B proxies closely matching the behaviors of larger models. On the LongBench benchmark, Sentinel achieves up to 5$\times$ compression while matching the QA performance of 7B-scale compression systems. Our results suggest that probing native attention signals enables fast, effective, and question-aware context compression. Code available at: https://github.com/yzhangchuck/Sentinel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23270v1">Does Machine Unlearning Truly Remove Model Knowledge? A Framework for Auditing Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have achieved remarkable advancements, drawing significant attention from the research community. Their capabilities are largely attributed to large-scale architectures, which require extensive training on massive datasets. However, such datasets often contain sensitive or copyrighted content sourced from the public internet, raising concerns about data privacy and ownership. Regulatory frameworks, such as the General Data Protection Regulation (GDPR), grant individuals the right to request the removal of such sensitive information. This has motivated the development of machine unlearning algorithms that aim to remove specific knowledge from models without the need for costly retraining. Despite these advancements, evaluating the efficacy of unlearning algorithms remains a challenge due to the inherent complexity and generative nature of LLMs. In this work, we introduce a comprehensive auditing framework for unlearning evaluation, comprising three benchmark datasets, six unlearning algorithms, and five prompt-based auditing methods. By using various auditing algorithms, we evaluate the effectiveness and robustness of different unlearning strategies. To explore alternatives beyond prompt-based auditing, we propose a novel technique that leverages intermediate activation perturbations, addressing the limitations of auditing methods that rely solely on model inputs and outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03553v3">Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by KDD2025
    </div>
    <details class="paper-abstract">
      Proteins, as essential biomolecules, play a central role in biological processes, including metabolic reactions and DNA replication. Accurate prediction of their properties and functions is crucial in biological applications. Recent development of protein language models (pLMs) with supervised fine tuning provides a promising solution to this problem. However, the fine-tuned model is tailored for particular downstream prediction task, and achieving general-purpose protein understanding remains a challenge. In this paper, we introduce Structure-Enhanced Protein Instruction Tuning (SEPIT) framework to bridge this gap. Our approach incorporates a novel structure-aware module into pLMs to enrich their structural knowledge, and subsequently integrates these enhanced pLMs with large language models (LLMs) to advance protein understanding. In this framework, we propose a novel instruction tuning pipeline. First, we warm up the enhanced pLMs using contrastive learning and structure denoising. Then, caption-based instructions are used to establish a basic understanding of proteins. Finally, we refine this understanding by employing a mixture of experts (MoEs) to capture more complex properties and functional information with the same number of activated parameters. Moreover, we construct the largest and most comprehensive protein instruction dataset to date, which allows us to train and evaluate the general-purpose protein understanding model. Extensive experiments on both open-ended generation and closed-set answer tasks demonstrate the superior performance of SEPIT over both closed-source general LLMs and open-source LLMs trained with protein knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v1">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 14 pages, 17 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23236v1">Towards LLM-Empowered Fine-Grained Speech Descriptors for Explainable Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted by INTERSPEECH2025
    </div>
    <details class="paper-abstract">
      This paper presents a novel end-to-end LLM-empowered explainable speech emotion recognition (SER) approach. Fine-grained speech emotion descriptor (SED) features, e.g., pitch, tone and emphasis, are disentangled from HuBERT SSL representations via alternating LLM fine-tuning to joint SER-SED prediction and ASR tasks. VAE compressed HuBERT features derived via Information Bottleneck (IB) are used to adjust feature granularity. Experiments on the IEMOCAP and MELD benchmarks demonstrate that our approach consistently outperforms comparable LLaMA-based SER baselines, including those using either (a) alternating multi-task fine-tuning alone or (b) feature disentanglement only. Statistically significant increase of SER unweighted accuracy by up to 4.0% and 3.7% absolute (5.4% and 6.6% relative) are obtained. More importantly, emotion descriptors offer further explainability for SER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03475v2">am-ELO: A Stable Framework for Arena-based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ICML2025 Accepted
    </div>
    <details class="paper-abstract">
      Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23219v1">Ghidorah: Fast LLM Inference on Edge with Speculative Decoding and Hetero-Core Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In-situ LLM inference on end-user devices has gained significant interest due to its privacy benefits and reduced dependency on external infrastructure. However, as the decoding process is memory-bandwidth-bound, the diverse processing units in modern end-user devices cannot be fully exploited, resulting in slow LLM inference. This paper presents Ghidorah, a LLM inference system for end-user devices with the unified memory architecture. The key idea of Ghidorah can be summarized in two steps: 1) leveraging speculative decoding approaches to enhance parallelism, and 2) ingeniously distributing workloads across multiple heterogeneous processing units to maximize computing power utilization. Ghidorah includes the hetero-core model parallelism (HCMP) architecture and the architecture-aware profiling (ARCA) approach. The HCMP architecture guides partitioning by leveraging the unified memory design of end-user devices and adapting to the hybrid computational demands of speculative decoding. The ARCA approach is used to determine the optimal speculative strategy and partitioning strategy, balancing acceptance rate with parallel capability to maximize the speedup. Additionally, we optimize sparse computation on ARM CPUs. Experimental results show that Ghidorah can achieve up to 7.6x speedup in the dominant LLM decoding phase compared to the sequential decoding approach in NVIDIA Jetson NX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17390v3">$T^5Score$: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published in the Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Using LLMs for Multi-Document Topic Extraction has recently gained popularity due to their apparent high-quality outputs, expressiveness, and ease of use. However, most existing evaluation practices are not designed for LLM-generated topics and result in low inter-annotator agreement scores, hindering the reliable use of LLMs for the task. To address this, we introduce $T^5Score$, an evaluation methodology that decomposes the quality of a topic set into quantifiable aspects, measurable through easy-to-perform annotation tasks. This framing enables a convenient, manual or automatic, evaluation procedure resulting in a strong inter-annotator agreement score. To substantiate our methodology and claims, we perform extensive experimentation on multiple datasets and report the results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23191v1">ExpeTrans: LLMs Are Experiential Transfer Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 12 figs/tables
    </div>
    <details class="paper-abstract">
      Recent studies provide large language models (LLMs) with textual task-solving experiences via prompts to improve their performance. However, previous methods rely on substantial human labor or time to gather such experiences for each task, which is impractical given the growing variety of task types in user queries to LLMs. To address this issue, we design an autonomous experience transfer framework to explore whether LLMs can mimic human cognitive intelligence to autonomously transfer experience from existing source tasks to newly encountered target tasks. This not only allows the acquisition of experience without extensive costs of previous methods, but also offers a novel path for the generalization of LLMs. Experimental results on 13 datasets demonstrate that our framework effectively improves the performance of LLMs. Furthermore, we provide a detailed analysis of each module in the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23187v1">Cross-Task Experiential Learning on LLM-based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large Language Model-based multi-agent systems (MAS) have shown remarkable progress in solving complex tasks through collaborative reasoning and inter-agent critique. However, existing approaches typically treat each task in isolation, resulting in redundant computations and limited generalization across structurally similar tasks. To address this, we introduce multi-agent cross-task experiential learning (MAEL), a novel framework that endows LLM-driven agents with explicit cross-task learning and experience accumulation. We model the task-solving workflow on a graph-structured multi-agent collaboration network, where agents propagate information and coordinate via explicit connectivity. During the experiential learning phase, we quantify the quality for each step in the task-solving workflow and store the resulting rewards along with the corresponding inputs and outputs into each agent's individual experience pool. During inference, agents retrieve high-reward, task-relevant experiences as few-shot examples to enhance the effectiveness of each reasoning step, thereby enabling more accurate and efficient multi-agent collaboration. Experimental results on diverse datasets demonstrate that MAEL empowers agents to learn from prior task experiences effectively-achieving faster convergence and producing higher-quality solutions on current tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17656v2">Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04779v4">Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics. Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12583v2">LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Despite the growing development of long-context large language models (LLMs), data-centric approaches relying on synthetic data have been hindered by issues related to faithfulness, which limit their effectiveness in enhancing model performance on tasks such as long-context reasoning and question answering (QA). These challenges are often exacerbated by misinformation caused by lack of verification, reasoning without attribution, and potential knowledge conflicts. We propose LongFaith, a novel pipeline for synthesizing faithful long-context reasoning instruction datasets. By integrating ground truth and citation-based reasoning prompts, we eliminate distractions and improve the accuracy of reasoning chains, thus mitigating the need for costly verification processes. We open-source two synthesized datasets, LongFaith-SFT and LongFaith-PO, which systematically address multiple dimensions of faithfulness, including verified reasoning, attribution, and contextual grounding. Extensive experiments on multi-hop reasoning datasets and LongBench demonstrate that models fine-tuned on these datasets significantly improve performance. Our ablation studies highlight the scalability and adaptability of the LongFaith pipeline, showcasing its broad applicability in developing long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16359v3">Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 arXiv admin note: text overlap with arXiv:2407.14644
    </div>
    <details class="paper-abstract">
      As the AI systems become deeply embedded in social media platforms, we've uncovered a concerning security vulnerability that goes beyond traditional adversarial attacks. It becomes important to assess the risks of LLMs before the general public use them on social media platforms to avoid any adverse impacts. Unlike obvious nonsensical text strings that safety systems can easily catch, our work reveals that human-readable situation-driven adversarial full-prompts that leverage situational context are effective but much harder to detect. We found that skilled attackers can exploit the vulnerabilities in open-source and proprietary LLMs to make a malicious user query safe for LLMs, resulting in generating a harmful response. This raises an important question about the vulnerabilities of LLMs. To measure the robustness against human-readable attacks, which now present a potent threat, our research makes three major contributions. First, we developed attacks that use movie scripts as situational contextual frameworks, creating natural-looking full-prompts that trick LLMs into generating harmful content. Second, we developed a method to transform gibberish adversarial text into readable, innocuous content that still exploits vulnerabilities when used within the full-prompts. Finally, we enhanced the AdvPrompter framework with p-nucleus sampling to generate diverse human-readable adversarial texts that significantly improve attack effectiveness against models like GPT-3.5-Turbo-0125 and Gemma-7b. Our findings show that these systems can be manipulated to operate beyond their intended ethical boundaries when presented with seemingly normal prompts that contain hidden adversarial elements. By identifying these vulnerabilities, we aim to drive the development of more robust safety mechanisms that can withstand sophisticated attacks in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04209v2">To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16561v2">FutureGen: LLM-RAG Approach to Generate the Future Work of Scientific Article</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The future work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from key sections of a scientific article alongside related papers and analyze how the trends have evolved. We experimented with various Large Language Models (LLMs) and integrated Retrieval-Augmented Generation (RAG) to enhance the generation process. We incorporate a LLM feedback mechanism to improve the quality of the generated content and propose an LLM-as-a-judge approach for evaluation. Our results demonstrated that the RAG-based approach with LLM feedback outperforms other methods evaluated through qualitative and quantitative metrics. Moreover, we conduct a human evaluation to assess the LLM as an extractor and judge. The code and dataset for this project are here, code: HuggingFace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19510v2">LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs? A Benchmark and Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The remarkable reasoning and generalization capabilities of Large Language Models (LLMs) have paved the way for their expanding applications in embodied AI, robotics, and other real-world tasks. To effectively support these applications, grounding in spatial and temporal understanding in multimodal environments is essential. To this end, recent works have leveraged scene graphs, a structured representation that encodes entities, attributes, and their relationships in a scene. However, a comprehensive evaluation of LLMs' ability to utilize scene graphs remains limited. In this work, we introduce Text-Scene Graph (TSG) Bench, a benchmark designed to systematically assess LLMs' ability to (1) understand scene graphs and (2) generate them from textual narratives. With TSG Bench we evaluate 11 LLMs and reveal that, while models perform well on scene graph understanding, they struggle with scene graph generation, particularly for complex narratives. Our analysis indicates that these models fail to effectively decompose discrete scenes from a complex narrative, leading to a bottleneck when generating scene graphs. These findings underscore the need for improved methodologies in scene graph generation and provide valuable insights for future research. The demonstration of our benchmark is available at https://tsg-bench.netlify.app. Additionally, our code and evaluation data are publicly available at https://github.com/docworlds/tsg-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20921v2">Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23061v1">DINGO: Constrained Inference for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 DINGO an algorithm to provably apply constraints to diffusion LLM generations
    </div>
    <details class="paper-abstract">
      Diffusion LLMs have emerged as a promising alternative to conventional autoregressive LLMs, offering significant potential for improved runtime efficiency. However, existing diffusion models lack the ability to provably enforce user-specified formal constraints, such as regular expressions, which makes them unreliable for tasks that require structured outputs, such as fixed-schema JSON generation. Unlike autoregressive models that generate tokens sequentially, diffusion LLMs predict a block of tokens in parallel. This parallelism makes traditional constrained decoding algorithms, which are designed for sequential token prediction, ineffective at preserving the true output distribution. To address this limitation, we propose DINGO, a dynamic programming-based constrained decoding strategy that is both efficient and provably distribution-preserving. DINGO enables sampling of output strings with the highest probability under the model's predicted distribution, while strictly satisfying any user-specified regular expression. On standard symbolic math and JSON generation benchmarks, DINGO achieves up to a 68 percentage point improvement over unconstrained inference
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02322v2">HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 arXiv admin note: text overlap with arXiv:2406.14228 by other authors
    </div>
    <details class="paper-abstract">
      Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23049v1">DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Pruning is a widely used technique to compress large language models (LLMs) by removing unimportant weights, but it often suffers from significant performance degradation - especially under semi-structured sparsity constraints. Existing pruning methods primarily focus on estimating the importance of individual weights, which limits their ability to preserve critical capabilities of the model. In this work, we propose a new perspective: rather than merely selecting which weights to prune, we first redistribute parameter importance to make the model inherently more amenable to pruning. By minimizing the information entropy of normalized importance scores, our approach concentrates importance onto a smaller subset of weights, thereby enhancing pruning robustness. We instantiate this idea through DenoiseRotator, which applies learnable orthogonal transformations to the model's weight matrices. Our method is model-agnostic and can be seamlessly integrated with existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4 semi-structured sparsity, DenoiseRotator consistently improves perplexity and zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4 semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are available at https://github.com/Axel-gu/DenoiseRotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23034v1">Case-Based Reasoning Enhances the Predictive Power of LLMs in Drug-Drug Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Drug-drug interaction (DDI) prediction is critical for treatment safety. While large language models (LLMs) show promise in pharmaceutical tasks, their effectiveness in DDI prediction remains challenging. Inspired by the well-established clinical practice where physicians routinely reference similar historical cases to guide their decisions through case-based reasoning (CBR), we propose CBR-DDI, a novel framework that distills pharmacological principles from historical cases to improve LLM reasoning for DDI tasks. CBR-DDI constructs a knowledge repository by leveraging LLMs to extract pharmacological insights and graph neural networks (GNNs) to model drug associations. A hybrid retrieval mechanism and dual-layer knowledge-enhanced prompting allow LLMs to effectively retrieve and reuse relevant cases. We further introduce a representative sampling strategy for dynamic case refinement. Extensive experiments demonstrate that CBR-DDI achieves state-of-the-art performance, with a significant 28.7% accuracy improvement over both popular LLMs and CBR baseline, while maintaining high interpretability and flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12600v2">On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Evidence-enhanced detectors present remarkable abilities in identifying malicious social text. However, the rise of large language models (LLMs) brings potential risks of evidence pollution to confuse detectors. This paper explores potential manipulation scenarios including basic pollution, and rephrasing or generating evidence by LLMs. To mitigate the negative impact, we propose three defense strategies from the data and model sides, including machine-generated text detection, a mixture of experts, and parameter updating. Extensive experiments on four malicious social text detection tasks with ten datasets illustrate that evidence pollution significantly compromises detectors, where the generating strategy causes up to a 14.4% performance drop. Meanwhile, the defense strategies could mitigate evidence pollution, but they faced limitations for practical employment. Further analysis illustrates that polluted evidence (i) is of high quality, evaluated by metrics and humans; (ii) would compromise the model calibration, increasing expected calibration error up to 21.6%; and (iii) could be integrated to amplify the negative impact, especially for encoder-based LMs, where the accuracy drops by 21.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09187v3">GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language model (LLM) agents has raised new concerns regarding their safety and security. In this paper, we propose GuardAgent, the first guardrail agent to protect target agents by dynamically checking whether their actions satisfy given safety guard requests. Specifically, GuardAgent first analyzes the safety guard requests to generate a task plan, and then maps this plan into guardrail code for execution. By performing the code execution, GuardAgent can deterministically follow the safety guard request and safeguard target agents. In both steps, an LLM is utilized as the reasoning component, supplemented by in-context demonstrations retrieved from a memory module storing experiences from previous tasks. In addition, we propose two novel benchmarks: EICU-AC benchmark to assess the access control for healthcare agents and Mind2Web-SC benchmark to evaluate the safety policies for web agents. We show that GuardAgent effectively moderates the violation actions for different types of agents on these two benchmarks with over 98% and 83% guardrail accuracies, respectively. Project page: https://guardagent.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23022v1">SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Existing Large Language Model (LLM) serving systems prioritize maximum throughput. They often neglect Service Level Objectives (SLOs) such as Time to First Token (TTFT) and Time Per Output Token (TPOT), which leads to suboptimal SLO attainment. This paper introduces SCORPIO, an SLO-oriented LLM serving system designed to maximize system goodput and SLO attainment for workloads with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for adaptive scheduling across admission control, queue management, and batch selection. SCORPIO features a TTFT Guard, which employs least-deadline-first reordering and rejects unattainable requests, and a TPOT Guard, which utilizes a VBS-based admission control and a novel credit-based batching mechanism. Both guards are supported by a predictive module. Evaluations demonstrate that SCORPIO improves system goodput by up to 14.4X and SLO adherence by up to 46.5% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v1">Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 34 pages, 12 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor environments and under open-vocabulary object descriptions. We observe that most episodes in widely used benchmarks such as HM3D and MP3D involve multi-floor buildings, with many requiring explicit floor transitions. However, existing methods are often limited to single-floor settings or predefined object categories. To address these limitations, we tackle two key challenges: (1) efficient cross-level planning and (2) zero-shot object-goal navigation (ZS-OGN), where agents must interpret novel object descriptions without prior exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine Frontier Reasoning module leveraging Large Language Models (LLMs) for context-aware exploration, without requiring additional training on new object semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor navigation. We further validate its practicality through real-world deployment on a quadruped robot, achieving successful object exploration across unseen floors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23013v1">Scalable Complexity Control Facilitates Reasoning Ability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The reasoning ability of large language models (LLMs) has been rapidly advancing in recent years, attracting interest in more fundamental approaches that can reliably enhance their generalizability. This work demonstrates that model complexity control, conveniently implementable by adjusting the initialization rate and weight decay coefficient, improves the scaling law of LLMs consistently over varying model sizes and data sizes. This gain is further illustrated by comparing the benchmark performance of 2.4B models pretrained on 1T tokens with different complexity hyperparameters. Instead of fixing the initialization std, we found that a constant initialization rate (the exponent of std) enables the scaling law to descend faster in both model and data sizes. These results indicate that complexity control is a promising direction for the continual advancement of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22998v1">LLM Agents for Bargaining with Utility-based Feedback</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Bargaining, a critical aspect of real-world interactions, presents challenges for large language models (LLMs) due to limitations in strategic depth and adaptation to complex human factors. Existing benchmarks often fail to capture this real-world complexity. To address this and enhance LLM capabilities in realistic bargaining, we introduce a comprehensive framework centered on utility-based feedback. Our contributions are threefold: (1) BargainArena, a novel benchmark dataset with six intricate scenarios (e.g., deceptive practices, monopolies) to facilitate diverse strategy modeling; (2) human-aligned, economically-grounded evaluation metrics inspired by utility theory, incorporating agent utility and negotiation power, which implicitly reflect and promote opponent-aware reasoning (OAR); and (3) a structured feedback mechanism enabling LLMs to iteratively refine their bargaining strategies. This mechanism can positively collaborate with in-context learning (ICL) prompts, including those explicitly designed to foster OAR. Experimental results show that LLMs often exhibit negotiation strategies misaligned with human preferences, and that our structured feedback mechanism significantly improves their performance, yielding deeper strategic and opponent-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22375v2">Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22571v2">Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18542v2">Business as Rulesual: A Benchmark and Framework for Business Rule Flow Modeling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Process mining aims to discover, monitor and optimize the actual behaviors of real processes. While prior work has mainly focused on extracting procedural action flows from instructional texts, rule flows embedded in business documents remain underexplored. To this end, we introduce a novel annotated Chinese dataset, BPRF, which contains 50 business process documents with 326 explicitly labeled business rules across multiple domains. Each rule is represented as a <Condition, Action> pair, and we annotate logical dependencies between rules (sequential, conditional, or parallel). We also propose ExIde, a framework for automatic business rule extraction and dependency relationship identification using large language models (LLMs). We evaluate ExIde using 12 state-of-the-art (SOTA) LLMs on the BPRF dataset, benchmarking performance on both rule extraction and dependency classification tasks of current LLMs. Our results demonstrate the effectiveness of ExIde in extracting structured business rules and analyzing their interdependencies for current SOTA LLMs, paving the way for more automated and interpretable business process automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13330v2">LEAVS: An LLM-based Labeler for Abdominal CT Supervision</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Early acceptance (top 9% of submissions) for MICCAI 2025
    </div>
    <details class="paper-abstract">
      Extracting structured labels from radiology reports has been employed to create vision models to simultaneously detect several types of abnormalities. However, existing works focus mainly on the chest region. Few works have been investigated on abdominal radiology reports due to more complex anatomy and a wider range of pathologies in the abdomen. We propose LEAVS (Large language model Extractor for Abdominal Vision Supervision). This labeler can annotate the certainty of presence and the urgency of seven types of abnormalities for nine abdominal organs on CT radiology reports. To ensure broad coverage, we chose abnormalities that encompass most of the finding types from CT reports. Our approach employs a specialized chain-of-thought prompting strategy for a locally-run LLM using sentence extraction and multiple-choice questions in a tree-based decision system. We demonstrate that the LLM can extract several abnormality types across abdominal organs with an average F1 score of 0.89, significantly outperforming competing labelers and humans. Additionally, we show that extraction of urgency labels achieved performance comparable to human annotations. Finally, we demonstrate that the abnormality labels contain valuable information for training a single vision model that classifies several organs as normal or abnormal. We release our code and structured annotations for a public CT dataset containing over 1,000 CT volumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22961v1">ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22959v1">LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Health, Safety, and Environment (HSE) compliance assessment demands dynamic real-time decision-making under complicated regulations and complex human-machine-environment interactions. While large language models (LLMs) hold significant potential for decision intelligence and contextual dialogue, their capacity for domain-specific knowledge in HSE and structured legal reasoning remains underexplored. We introduce HSE-Bench, the first benchmark dataset designed to evaluate the HSE compliance assessment capabilities of LLM. HSE-Bench comprises over 1,000 manually curated questions drawn from regulations, court cases, safety exams, and fieldwork videos, and integrates a reasoning flow based on Issue spotting, rule Recall, rule Application, and rule Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct extensive evaluations on different prompting strategies and more than 10 LLMs, including foundation models, reasoning models and multimodal vision models. The results show that, although current LLMs achieve good performance, their capabilities largely rely on semantic matching rather than principled reasoning grounded in the underlying HSE compliance context. Moreover, their native reasoning trace lacks the systematic legal reasoning required for rigorous HSE compliance assessment. To alleviate these, we propose a new prompting technique, Reasoning of Expert (RoE), which guides LLMs to simulate the reasoning process of different experts for compliance assessment and reach a more accurate unified decision. We hope our study highlights reasoning gaps in LLMs for HSE compliance and inspires further research on related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22956v1">LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Automated large-scale analysis of public discussions around contested issues like abortion requires detecting and understanding the use of arguments. While Large Language Models (LLMs) have shown promise in language processing tasks, their performance in mining topic-specific, pre-defined arguments in online comments remains underexplored. We evaluate four state-of-the-art LLMs on three argument mining tasks using datasets comprising over 2,000 opinion comments across six polarizing topics. Quantitative evaluation suggests an overall strong performance across the three tasks, especially for large and fine-tuned LLMs, albeit at a significant environmental cost. However, a detailed error analysis revealed systematic shortcomings on long and nuanced comments and emotionally charged language, raising concerns for downstream applications like content moderation or opinion analysis. Our results highlight both the promise and current limitations of LLMs for automated argument analysis in online comments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16814v2">Understanding Bias Reinforcement in LLM Agents Debate</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22950v1">StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. Notably, on ArXiv, it boosts FactCC and SummaC by 19.2 and 9.7 points, indicating stronger alignment between summaries and source content. These findings suggest that structure-aware prompting is a simple yet effective approach for zero-shot extractive summarization with LLMs, without any training or task-specific tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14662v4">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Findings of ACL 2025 and WWW2025@HCRS
    </div>
    <details class="paper-abstract">
      Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure.
    </details>
</div>
