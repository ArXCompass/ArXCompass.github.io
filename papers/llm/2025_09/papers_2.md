# llm - 2025_09

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24592v1">BPMN Assistant: An LLM-Based Approach to Business Process Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      This paper presents BPMN Assistant, a tool that leverages Large Language Models (LLMs) for natural language-based creation and editing of BPMN diagrams. A specialized JSON-based representation is introduced as a structured alternative to the direct handling of XML to enhance the accuracy of process modifications. Process generation quality is evaluated using Graph Edit Distance (GED) and Relative Graph Edit Distance (RGED), while editing performance is evaluated with a binary success metric. Results show that JSON and XML achieve similar similarity scores in generation, but JSON offers greater reliability, faster processing, and significantly higher editing success rates. We discuss key trade-offs, limitations, and future improvements. The implementation is available at https://github.com/jtlicardo/bpmn-assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15735v3">EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 5 pages, submitted to ICASSP 2026, September 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer broad utility but remain prone to hallucination and out-of-distribution (OOD) errors. We propose EigenTrack, an interpretable real-time detector that uses the spectral geometry of hidden activations, a compact global signature of model dynamics. By streaming covariance-spectrum statistics such as entropy, eigenvalue gaps, and KL divergence from random baselines into a lightweight recurrent classifier, EigenTrack tracks temporal shifts in representation structure that signal hallucination and OOD drift before surface errors appear. Unlike black- and grey-box methods, it needs only a single forward pass without resampling. Unlike existing white-box detectors, it preserves temporal context, aggregates global signals, and offers interpretable accuracy-latency trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18133v3">Self-Evolving LLMs via Continual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening generalization.We propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting self-evolution.Extensive experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24507v1">SemGuard: Real-Time Semantic Evaluator for Correcting LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted by the 40th IEEE/ACM Automated Software Engineering Conference (ASE 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can translate natural language requirements into code, yet empirical analyses of representative models reveal that semantic errors-programs that compile but behave incorrectly-constitute the majority of observed faults (e.g., >60% on DeepSeek-Coder-6.7B and QwenCoder-7B). Post-hoc repair pipelines detect such faults only after execution, incurring latency, relying on incomplete test suites, and often mis-localizing the defect. Since semantic drift originates in the autoregressive decoding process, intervening while the code is being generated is a direct way to stop error propagation. Constrained-decoding approaches such as ROCODE attempt this, but still wait until the entire program runs to obtain feedback and use entropy heuristics that do not truly capture semantics. A more effective solution must inject semantic signals-early and precisely-into the decoding process.We present SemGuard, a semantic-evaluator-driven framework that performs real-time, line-level semantic supervision. To train the evaluator, we build SemDiff, the first dataset with fine-grained annotations that mark the exact line where a correct and an incorrect implementation diverge. The evaluator, once embedded in the LLM's decoder, flags deviations on partial code, rolls back to the faulty line, and guides regeneration-without executing the program or requiring test cases. Across four benchmarks, SemGuard consistently outperforms state-of-the-art baselines. It lowers the semantic error rate by 19.86% on SemDiff relative to ROCODE, and lifts Pass@1 by 48.92% on the real-world LiveCodeBench with CodeLlama-7B. Similar gains hold for StarCoder2-7B on MBPP and for DeepSeekCoder-6.7B on the Java benchmark SemDiff-Java, demonstrating model- and language-agnostic effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24506v1">Building Benchmarks from the Ground Up: Community-Centered Evaluation of LLMs in Healthcare Chatbot Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically evaluated through general or domain-specific benchmarks testing capabilities that often lack grounding in the lived realities of end users. Critical domains such as healthcare require evaluations that extend beyond artificial or simulated tasks to reflect the everyday needs, cultural practices, and nuanced contexts of communities. We propose Samiksha, a community-driven evaluation pipeline co-created with civil-society organizations (CSOs) and community members. Our approach enables scalable, automated benchmarking through a culturally aware, community-driven pipeline in which community feedback informs what to evaluate, how the benchmark is built, and how outputs are scored. We demonstrate this approach in the health domain in India. Our analysis highlights how current multilingual LLMs address nuanced community health queries, while also offering a scalable pathway for contextually grounded and inclusive LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24496v1">LLM DNA: Tracing Model Evolution via Functional Representations</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The explosive growth of large language models (LLMs) has created a vast but opaque landscape: millions of models exist, yet their evolutionary relationships through fine-tuning, distillation, or adaptation are often undocumented or unclear, complicating LLM management. Existing methods are limited by task specificity, fixed model sets, or strict assumptions about tokenizers or architectures. Inspired by biological DNA, we address these limitations by mathematically defining LLM DNA as a low-dimensional, bi-Lipschitz representation of functional behavior. We prove that LLM DNA satisfies inheritance and genetic determinism properties and establish the existence of DNA. Building on this theory, we derive a general, scalable, training-free pipeline for DNA extraction. In experiments across 305 LLMs, DNA aligns with prior studies on limited subsets and achieves superior or competitive performance on specific tasks. Beyond these tasks, DNA comparisons uncover previously undocumented relationships among LLMs. We further construct the evolutionary tree of LLMs using phylogenetic algorithms, which align with shifts from encoder-decoder to decoder-only architectures, reflect temporal progression, and reveal distinct evolutionary speeds across LLM families.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06361v2">Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness critical. A significant and underexplored risk is intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective. Existing studies typically induce deception by explicitly setting a hidden objective through prompting or fine-tuning, which may not reflect real-world human-LLM interactions. Moving beyond such human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth, we propose a framework based on Contact Searching Questions~(CSQ). This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the Deceptive Intention Score, measures the model's bias toward a hidden objective. The second, the Deceptive Behavior Score, measures the inconsistency between the LLM's internal belief and its expressed output. Evaluating 16 leading LLMs, we find that both metrics rise in parallel and escalate with task difficulty for most models. Moreover, increasing model capacity does not always reduce deception, posing a significant challenge for future LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24468v1">Bias Mitigation or Cultural Commonsense? Evaluating LLMs with a Japanese Dataset</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit social biases, prompting the development of various debiasing methods. However, debiasing methods may degrade the capabilities of LLMs. Previous research has evaluated the impact of bias mitigation primarily through tasks measuring general language understanding, which are often unrelated to social biases. In contrast, cultural commonsense is closely related to social biases, as both are rooted in social norms and values. The impact of bias mitigation on cultural commonsense in LLMs has not been well investigated. Considering this gap, we propose SOBACO (SOcial BiAs and Cultural cOmmonsense benchmark), a Japanese benchmark designed to evaluate social biases and cultural commonsense in LLMs in a unified format. We evaluate several LLMs on SOBACO to examine how debiasing methods affect cultural commonsense in LLMs. Our results reveal that the debiasing methods degrade the performance of the LLMs on the cultural commonsense task (up to 75% accuracy deterioration). These results highlight the importance of developing debiasing methods that consider the trade-off with cultural commonsense to improve fairness and utility of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05085v5">Multi-Head RAG: Solving Multi-Aspect Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) enhances the abilities of Large Language Models (LLMs) by enabling the retrieval of documents into the LLM context to provide more accurate and relevant responses. Existing RAG solutions do not focus on queries that may require fetching multiple documents with substantially different contents. Such queries occur frequently, but are challenging because the embeddings of these documents may be distant in the embedding space, making it hard to retrieve them all. This paper introduces Multi-Head RAG (MRAG), a novel scheme designed to address this gap with a simple yet powerful idea: leveraging activations of Transformer's multi-head attention layer, instead of the decoder layer, as keys for fetching multi-aspect documents. The driving observation is that different attention heads learn to capture different data aspects. Harnessing the corresponding activations results in embeddings that represent various facets of data items and queries, improving the retrieval accuracy for complex queries. We provide an evaluation methodology and metrics, multi-aspect datasets, and real-world use cases to demonstrate MRAG's effectiveness. We show MRAG's design advantages over 18 RAG baselines, empirical improvements of up to 20% in retrieval success ratios, and benefits for downstream LLM generation. MRAG can be seamlessly integrated with existing RAG frameworks and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24419v1">Unit Test Update through LLM-Driven Context Collection and Error-Type-Aware Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Unit testing is critical for ensuring software quality and software system stability. The current practice of manually maintaining unit tests suffers from low efficiency and the risk of delayed or overlooked fixes. Therefore, an automated approach is required to instantly update unit tests, with the capability to both repair and enhance unit tests. However, existing automated test maintenance methods primarily focus on repairing broken tests, neglecting the scenario of enhancing existing tests to verify new functionality. Meanwhile, due to their reliance on rule-based context collection and the lack of verification mechanisms, existing approaches struggle to handle complex code changes and often produce test cases with low correctness. To address these challenges, we propose TESTUPDATER, a novel LLM based approach that enables automated just-in-time test updates in response to production code changes. TESTUPDATER first leverages the LLM to analyze code changes and identify relevant context, which it then extracts and filters. Then, through carefully designed prompts, TESTUPDATER guides the LLM step by step to handle various types of code changes and introduce new dependencies, enabling both test repair and enhancement. Finally, we introduce an error-type-aware iterative refinement mechanism that executes the LLM-updated tests and repairs failures, which significantly improves the overall correctness of test updates. Since existing test repair datasets lack scenarios of test enhancement, we further construct a new benchmark, UPDATES4J, with 195 real-world samples from 7 projects. Experimental results show that TESTUPDATER achieves a compilation pass rate of 94.4% and a test pass rate of 86.7%, outperforming the state-of-the-art method SYNTER by 15.9% and 20.0%, respectively. Furthermore, TESTUPDATER exhibits 12.9% higher branch coverage and 15.2% greater line coverage than SYNTER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24418v1">GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly integrated into numerous applications across various domains, LLMs' safety becomes a critical concern for both application developers and intended users. Currently, great efforts have been made to develop safety benchmarks with fine-grained taxonomies. However, these benchmarks' taxonomies are disparate with different safety policies. Thus, existing safeguards trained on these benchmarks are either coarse-grained to only distinguish between safe and unsafe, or constrained by the narrow risk taxonomies of a single benchmark. To leverage these fine-grained safety taxonomies across multiple safety benchmarks, in this paper, we propose GSPR, a Generalizable Safety Policy Reasoner to identify unsafe input prompts and LLMs' outputs with violated safety taxonomies through Group Relative Policy Optimization (GRPO). Unlike prior safeguards which only cover a fixed set of risk factors, our GSPR incentivizes its reasoning capability with varied safety taxonomies through our careful cold-start strategy and reward design. Consequently, our GSPR can be trained across multiple safety benchmarks with distinct taxonomies and naturally exhibits powerful generalization ability. We conduct extensive experiments to show that our GSPR significantly improves existing safety guardrails' reasoning capabilities for both safety and category prediction tasks. Moreover, our GSPR not only demonstrates powerful safety generalization abilities but also achieves the least inference token costs with explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24385v1">Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction-Reasoning Synergy</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24384v1">HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values is critical for their safe deployment, yet jailbreak attacks can subvert this alignment to elicit harmful outputs from LLMs. In recent years, a proliferation of jailbreak attacks has emerged, accompanied by diverse metrics and judges to assess the harmfulness of the LLM outputs. However, the absence of a systematic benchmark to assess the quality and effectiveness of these metrics and judges undermines the credibility of the reported jailbreak effectiveness and other risks. To address this gap, we introduce HarmMetric Eval, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. Our benchmark includes a high-quality dataset of representative harmful prompts paired with diverse harmful and non-harmful model responses, alongside a flexible scoring mechanism compatible with various metrics and judges. With HarmMetric Eval, our extensive experiments uncover a surprising result: two conventional metrics--METEOR and ROUGE-1--outperform LLM-based judges in evaluating the harmfulness of model responses, challenging prevailing beliefs about LLMs' superiority in this domain. Our dataset is publicly available at https://huggingface.co/datasets/qusgo/HarmMetric_Eval, and the code is available at https://anonymous.4open.science/r/HarmMetric-Eval-4CBE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17621v5">Navigate the Unknown: Enhancing LLM Reasoning with Intrinsic Motivation Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has emerged as a pivotal method for improving the reasoning capabilities of Large Language Models (LLMs). However, prevalent RL approaches such as Proximal Policy Optimization (PPO) and Group-Regularized Policy Optimization (GRPO) face critical limitations due to their reliance on sparse outcome-based rewards and inadequate mechanisms for incentivizing exploration. These limitations result in inefficient guidance for reasoning. Specifically, sparse rewards fail to deliver sufficient feedback, particularly for challenging problems. Furthermore, such rewards induce systematic biases that prioritize exploitation of familiar trajectories over novel solution discovery. These shortcomings critically hinder performance in complex reasoning tasks, which inherently demand iterative refinement across intermediate steps. To address these challenges, we propose an Intrinsic Motivation guidEd exploratioN meThOd foR LLM Reasoning (i-MENTOR), a method designed to deliver dense rewards and amplify exploration in the RL-based paradigm. i-MENTOR introduces three innovations: trajectory-aware exploration rewards that mitigate bias in token-level strategies while maintaining computational efficiency; error-conditioned reward allocation to ensure efficient exploration on challenging samples while intrinsically stabilizing training; and advantage-preserving integration that maintains advantage distribution integrity while incorporating exploratory guidance. Experiments across 4 public datasets demonstrate i-MENTOR's effectiveness, achieving a 22.23\% improvement on AIME 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09002v3">PALM: Synergizing Program Analysis and LLMs to Enhance Rust Unit Test Coverage</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to ASE 2025 (Research Paper Track). 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Unit testing is essential for ensuring software reliability and correctness. Classic Search-Based Software Testing (SBST) methods and concolic execution-based approaches for generating unit tests often fail to achieve high coverage due to difficulties in handling complex program units, such as branching conditions and external dependencies. Recent work has increasingly utilized large language models (LLMs) to generate test cases, improving the quality of test generation by providing better context and correcting errors in the model's output. However, these methods rely on fixed prompts, resulting in relatively low compilation success rates and coverage. This paper presents PALM, an approach that leverages large language models (LLMs) to enhance the generation of high-coverage unit tests. PALM performs program analysis to identify branching conditions within functions, which are then combined into path constraints. These constraints and relevant contextual information are used to construct prompts that guide the LLMs in generating unit tests. We implement the approach and evaluate it in 15 open-source Rust crates. Experimental results show that within just two or three hours, PALM can significantly improve test coverage compared to classic methods, with increases in overall project coverage exceeding 50% in some instances and its generated tests achieving an average coverage of 72.30%, comparable to human effort (70.94%), highlighting the potential of LLMs in automated test generation. We submitted 91 PALM-generated unit tests targeting new code. Of these submissions, 80 were accepted, 5 were rejected, and 6 remain pending review. The results demonstrate the effectiveness of integrating program analysis with AI and open new avenues for future research in automated software testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24377v1">Plan before Solving: Problem-Aware Strategy Routing for Mathematical Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Existing methods usually leverage a fixed strategy, such as natural language reasoning, code-augmented reasoning, tool-integrated reasoning, or ensemble-based reasoning, to guide Large Language Models (LLMs) to perform mathematical reasoning. Our analysis reveals that the single strategy cannot adapt to problem-specific requirements and thus overlooks the trade-off between effectiveness and efficiency. To address these issues, we propose Planning and Routing through Instance-Specific Modeling (PRISM), a novel framework that decouples mathematical reasoning into two stages: strategy planning and targeted execution. Specifically, we first curate a multi-strategy preference dataset, which we call MathStrat, capturing correctness, process quality, and computational efficiency for each problem--strategy pair. Then, we train a lightweight Strategy Adapter based on the dataset to obtain confidence distributions over the mentioned four reasoning strategies. At inference time, an adaptive routing policy dynamically tailors the reasoning approach based on predictor confidence. It directs the model to use single-strategy execution for high-confidence predictions, dual-strategy verification for competitive scenarios, or comprehensive multi-strategy exploration for uncertain cases. Extensive experiments across five mathematical reasoning benchmarks demonstrate that PRISM consistently outperforms individual strategies and ensemble baselines, achieving improvements ranging from 0.9% to 7.6% across different base models. The adaptive routing approach shows particularly strong benefits for mathematical reasoning tasks across diverse model architectures. Our code is released at https://github.com/reml-group/PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24372v1">Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 24 pages, including the appendix
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained large language models (LLMs) for down-stream tasks is a critical step in the AI deployment pipeline. Reinforcement learning (RL) is arguably the most prominent fine-tuning method, contributing to the birth of many state-of-the-art LLMs. In contrast, evolution strategies (ES), which once showed comparable performance to RL on models with a few million parameters, was neglected due to the pessimistic perception of its scalability to larger models. In this work, we report the first successful attempt to scale up ES for fine-tuning the full parameters of LLMs, showing the surprising fact that ES can search efficiently over billions of parameters and outperform existing RL fine-tuning methods in multiple respects, including sample efficiency, tolerance to long-horizon rewards, robustness to different base LLMs, less tendency to reward hacking, and more stable performance across runs. It therefore serves as a basis to unlock a new direction in LLM fine-tuning beyond what current RL techniques provide. The source codes are provided at: https://github.com/VsonicV/es-fine-tuning-paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24344v1">Comparing Open-Source and Commercial LLMs for Domain-Specific Analysis and Reporting: Software Engineering Challenges and Design Trade-offs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) enable automation of complex natural language processing across domains, but research on domain-specific applications like Finance remains limited. Objectives: This study explored open-source and commercial LLMs for financial report analysis and commentary generation, focusing on software engineering challenges in implementation. Methods: Using Design Science Research methodology, an exploratory case study iteratively designed and evaluated two LLM-based systems: one with local open-source models in a multi-agent workflow, another using commercial GPT-4o. Both were assessed through expert evaluation of real-world financial reporting use cases. Results: LLMs demonstrated strong potential for automating financial reporting tasks, but integration presented significant challenges. Iterative development revealed issues including prompt design, contextual dependency, and implementation trade-offs. Cloud-based models offered superior fluency and usability but raised data privacy and external dependency concerns. Local open-source models provided better data control and compliance but required substantially more engineering effort for reliability and usability. Conclusion: LLMs show strong potential for financial reporting automation, but successful integration requires careful attention to architecture, prompt design, and system reliability. Implementation success depends on addressing domain-specific challenges through tailored validation mechanisms and engineering strategies that balance accuracy, control, and compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14314v4">Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 accepted by ACL'2025
    </div>
    <details class="paper-abstract">
      Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at https://embodied-read.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20230v2">Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24319v1">Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24307v1">Exploring Similarity between Neural and LLM Trajectories in Language Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Understanding the similarity between large language models (LLMs) and human brain activity is crucial for advancing both AI and cognitive neuroscience. In this study, we provide a multilinguistic, large-scale assessment of this similarity by systematically comparing 16 publicly available pretrained LLMs with human brain responses during natural language processing tasks in both English and Chinese. Specifically, we use ridge regression to assess the representational similarity between LLM embeddings and electroencephalography (EEG) signals, and analyze the similarity between the "neural trajectory" and the "LLM latent trajectory." This method captures key dynamic patterns, such as magnitude, angle, uncertainty, and confidence. Our findings highlight both similarities and crucial differences in processing strategies: (1) We show that middle-to-high layers of LLMs are central to semantic integration and correspond to the N400 component observed in EEG; (2) The brain exhibits continuous and iterative processing during reading, whereas LLMs often show discrete, stage-end bursts of activity, which suggests a stark contrast in their real-time semantic processing dynamics. This study could offer new insights into LLMs and neural processing, and also establish a critical framework for future investigations into the alignment between artificial intelligence and biological intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24294v1">LOGOS: LLM-driven End-to-End Grounded Theory Development and Schema Induction for Qualitative Research</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Grounded theory offers deep insights from qualitative data, but its reliance on expert-intensive manual coding presents a major scalability bottleneck. Current computational tools stop short of true automation, keeping researchers firmly in the loop. We introduce LOGOS, a novel, end-to-end framework that fully automates the grounded theory workflow, transforming raw text into a structured, hierarchical theory. LOGOS integrates LLM-driven coding, semantic clustering, graph reasoning, and a novel iterative refinement process to build highly reusable codebooks. To ensure fair comparison, we also introduce a principled 5-dimensional metric and a train-test split protocol for standardized, unbiased evaluation. Across five diverse corpora, LOGOS consistently outperforms strong baselines and achieves a remarkable $88.2\%$ alignment with an expert-developed schema on a complex dataset. LOGOS demonstrates a powerful new path to democratize and scale qualitative research without sacrificing theoretical nuance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24291v1">Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Existing large language model (LLM)-based embeddings typically adopt an encoder-only paradigm, treating LLMs as static feature extractors and overlooking their core generative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive Sentence Embeddings), a novel framework that leverages autoregressive generation to iteratively refine semantic representations. By producing sequences of soft tokens optimized under contrastive objective, GIRCSE captures latent concepts and implicit semantics that encoder-only methods often miss. To guide this process, we propose an Iterative Contrastive Refinement (ICR) objective that encourages each refinement step to yield better representations. Extensive experiments show that GIRCSE outperforms strong LLM-based embedding baselines on the MTEB benchmark and instruction-following tasks. Moreover, GIRCSE exhibits an emergent test-time scaling property: generating more tokens at inference steadily improves embedding quality. Our results establish generative iterative refinement as a new paradigm for representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24282v1">SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol (the global industry standard for smart home communication), SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 11 agents under a unified ReAct framework reveals that while models perform well on simple tasks, they struggle with latent intent inference, state verification, and especially temporal scheduling. Even the top-performing model, GPT-4.1, reaches only 54% success rate. These findings highlight a critical need for methods that can reliably verify the current state via tools before acting and coordinate time-dependent actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02404v2">AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 40 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the medical field has garnered significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. To bridge this gap, we introduce AnesSuite, the first comprehensive dataset suite specifically designed for anesthesiology reasoning in LLMs. The suite features AnesBench, an evaluation benchmark tailored to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Alongside this benchmark, the suite includes three training datasets that provide an infrastructure for continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards (RLVR). Leveraging this suite, we develop Morpheus, the first baseline model collection for anesthesiology reasoning. Despite undergoing limited training with SFT and group relative policy optimization (GRPO), Morpheus demonstrates substantial performance improvements, rivaling the performance of larger-scale models. Furthermore, through comprehensive evaluations and experiments, we analyze the key factors influencing anesthesiology reasoning performance, including model characteristics, training strategies and training data. Both AnesSuite and Morpheus will be open-sourced at https://github.com/MiliLab/AnesSuite.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02648v3">Steering the Herd: A Framework for LLM-based Control of Social Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Algorithms increasingly serve as information mediators--from social media feeds and targeted advertising to the increasing ubiquity of LLMs. This engenders a joint process where agents combine private, algorithmically-mediated signals with learning from peers to arrive at decisions. To study such settings, we introduce a model of controlled sequential social learning in which an information-mediating planner (e.g. an LLM) controls the information structure of agents while they also learn from the decisions of earlier agents. The planner may seek to improve social welfare (altruistic planner) or to induce a specific action the planner prefers (biased planner). Our framework presents a new optimization problem for social learning that combines dynamic programming with decentralized action choices and Bayesian belief updates. We prove the convexity of the value function and characterize the optimal policies of altruistic and biased planners, which attain desired tradeoffs between the costs they incur and the payoffs they earn from induced agent choices. Notably, in some regimes the biased planner intentionally obfuscates the agents' signals. Even under stringent transparency constraints--information parity with individuals, no lying or cherry-picking, and full observability--we show that information mediation can substantially shift social welfare in either direction. We complement our theory with simulations in which LLMs act as both planner and agents. Notably, the LLM planner in our simulations exhibits emergent strategic behavior in steering public opinion that broadly mirrors the trends predicted, though key deviations suggest the influence of non-Bayesian reasoning consistent with the cognitive patterns of both humans and LLMs trained on human-like data. Together, we establish our framework as a tractable basis for studying the impact and regulation of LLM information mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03905v4">Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Preprint accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM), 2025
    </div>
    <details class="paper-abstract">
      LLMs have garnered considerable attention for their potential to streamline Automated Program Repair (APR). LLM-based approaches can either insert the correct code or directly generate patches when provided with buggy methods. However, most of LLM-based APR methods rely on a single type of software information, without fully leveraging different software artifacts. Despite this, many LLM-based approaches do not explore which specific types of information best assist in APR. Addressing this gap is crucial for advancing LLM-based APR techniques. We propose DEVLoRe to use issue content (description and message) and stack error traces to localize buggy methods, then rely on debug information in buggy methods and issue content and stack error to localize buggy lines and generate plausible patches which can pass all unit tests. The results show that while issue content is particularly effective in assisting LLMs with fault localization and program repair, different types of software artifacts complement each other. By incorporating different artifacts, DEVLoRe successfully locates 49.3% and 47.6% of single and non-single buggy methods and generates 56.0% and 14.5% plausible patches for the Defects4J v2.0 dataset, respectively. This outperforms current state-of-the-art APR methods. Furthermore, we re-implemented and evaluated our framework, demonstrating its effectiveness in its effectiveness in resolving 9 unique issues compared to other state-of-the-art frameworks using the same or more advanced models on SWE-bench Lite.We also discussed whether a leading framework for Python code can be directly applied to Java code, or vice versa. The source code and experimental results of this work for replication are available at https://github.com/XYZboom/DEVLoRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24219v1">ViReSkill: Vision-Grounded Replanning with Skill Memory for LLM-Based Planning in Lifelong Robot Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Robots trained via Reinforcement Learning (RL) or Imitation Learning (IL) often adapt slowly to new tasks, whereas recent Large Language Models (LLMs) and Vision-Language Models (VLMs) promise knowledge-rich planning from minimal data. Deploying LLMs/VLMs for motion planning, however, faces two key obstacles: (i) symbolic plans are rarely grounded in scene geometry and object physics, and (ii) model outputs can vary for identical prompts, undermining execution reliability. We propose ViReSkill, a framework that pairs vision-grounded replanning with a skill memory for accumulation and reuse. When a failure occurs, the replanner generates a new action sequence conditioned on the current scene, tailored to the observed state. On success, the executed plan is stored as a reusable skill and replayed in future encounters without additional calls to LLMs/VLMs. This feedback loop enables autonomous continual learning: each attempt immediately expands the skill set and stabilizes subsequent executions. We evaluate ViReSkill on simulators such as LIBERO and RLBench as well as on a physical robot. Across all settings, it consistently outperforms conventional baselines in task success rate, demonstrating robust sim-to-real generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04715v2">Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Commercial Large Language Model (LLM) APIs create a fundamental trust problem: users pay for specific models but have no guarantee that providers deliver them faithfully. Providers may covertly substitute cheaper alternatives (e.g., quantized versions, smaller models) to reduce costs while maintaining advertised pricing. We formalize this model substitution problem and systematically evaluate detection methods under realistic adversarial conditions. Our empirical analysis reveals that software-only methods are fundamentally unreliable: statistical tests on text outputs are query-intensive and fail against subtle substitutions, while methods using log probabilities are defeated by inherent inference nondeterminism in production environments. We argue that this verification gap can be more effectively closed with hardware-level security. We propose and evaluate the use of Trusted Execution Environments (TEEs) as one practical and robust solution. Our findings demonstrate that TEEs can provide provable cryptographic guarantees of model integrity with only a modest performance overhead, offering a clear and actionable path to ensure users get what they pay for. Code is available at https://github.com/sunblaze-ucb/llm-api-audit
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06583v2">Discerning minds or generic tutors? Evaluating instructional guidance capabilities in Socratic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The conversational capabilities of large language models hold significant promise for enabling scalable and interactive tutoring. While prior research has primarily examined their ability to generate Socratic questions, it often overlooks a critical aspect: adaptively guiding learners in accordance with their cognitive states. This study moves beyond question generation to emphasize instructional guidance capability. We ask: Can LLMs emulate expert tutors who dynamically adjust strategies in response to learners' states? To investigate this, we propose GuideEval, a benchmark grounded in authentic educational dialogues that evaluates pedagogical guidance through a three-phase behavioral framework: (1) Perception, inferring learner states; (2) Orchestration, adapting instructional strategies; and (3) Elicitation, stimulating proper reflections. Empirical results indicate that existing LLMs often fail to provide effective adaptive scaffolding when learners experience confusion or require redirection. To complement the quantitative evaluation, we conduct a detailed failure case analysis, providing an intuitive understanding of these shortcomings. Furthermore, we introduce a behavior-guided finetuning strategy that leverages behavior-prompted instructional dialogues, substantially enhancing guidance performance. By shifting the focus from isolated content evaluation to learner-centered state-aware interaction, our work advocates a more dialogic paradigm for evaluating Socratic LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22376v4">Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 ICLR 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24193v1">AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to NeurIPS 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Search-augmented LLMs often struggle with complex reasoning tasks due to ineffective multi-hop retrieval and limited reasoning ability. We propose AceSearcher, a cooperative self-play framework that trains a single large language model (LLM) to alternate between two roles: a decomposer that breaks down complex queries and a solver that integrates retrieved contexts for answer generation. AceSearcher couples supervised fine-tuning on a diverse mixture of search, reasoning, and decomposition tasks with reinforcement fine-tuning optimized for final answer accuracy, eliminating the need for intermediate annotations. Extensive experiments on three reasoning-intensive tasks across 10 datasets show that AceSearcher outperforms state-of-the-art baselines, achieving an average exact match improvement of 7.6%. Remarkably, on document-level finance reasoning tasks, AceSearcher-32B matches the performance of the DeepSeek-V3 model using less than 5% of its parameters. Even at smaller scales (1.5B and 8B), AceSearcher often surpasses existing search-augmented LLMs with up to 9x more parameters, highlighting its exceptional efficiency and effectiveness in tackling complex reasoning tasks. Our code will be published at https://github.com/ritaranx/AceSearcher and https://huggingface.co/AceSearcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24189v1">PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22034v2">The Thinking Spectrum: An Empirical Study of Tunable Reasoning in LLMs through Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The growing demand for large language models (LLMs) with tunable reasoning capabilities in many real-world applications highlights a critical need for methods that can efficiently produce a spectrum of models balancing reasoning depth and computational cost. Model merging has emerged as a promising, training-free technique to address this challenge by arithmetically combining the weights of a general-purpose model with a specialized reasoning model. While various merging techniques exist, their potential to create a spectrum of models with fine-grained control over reasoning abilities remains largely unexplored. This work presents a large-scale empirical study evaluating a range of model merging techniques across multiple reasoning benchmarks. We systematically vary merging strengths to construct accuracy-efficiency curves, providing the first comprehensive view of the tunable performance landscape. Our findings reveal that model merging offers an effective and controllable method for calibrating the trade-off between reasoning accuracy and token efficiency, even when parent models have highly divergent weight spaces. Crucially, we identify instances of Pareto Improvement, where a merged model achieves both higher accuracy and lower token consumption than one of its parents. Our study provides the first comprehensive analysis of this tunable space, offering practical guidelines for creating LLMs with specific reasoning profiles to meet diverse application demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03610v2">Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are reshaping the game industry, particularly with more intelligent and human-preferable game characters. However, existing game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets for aligning pre-trained LLMs into gaming agents. To fill these gaps, we present Orak, a foundational benchmark designed to train and evaluate LLM agents across diverse real-world video games. Unlike existing benchmarks, Orak includes 12 popular video games spanning all major genres, enabling comprehensive studies of LLM capabilities and agentic modules essential for intricate game scenarios. To support consistent evaluation of LLMs, we introduce a plug-and-play interface based on Model Context Protocol (MCP) that enables LLMs to seamlessly connect with games and manipulate agentic modules. Additionally, we propose a fine-tuning dataset, consisting of LLM gameplay trajectories across diverse game genres. Orak offers a comprehensive evaluation framework, encompassing general game score leaderboards, LLM battle arenas, and in-depth analyses of visual input state, agentic strategies, and fine-tuning effects, establishing a foundation towards building generic gaming agents. Code is available at https://github.com/krafton-ai/Orak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24166v1">Stable Forgetting: Bounded Parameter-Efficient Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      Machine unlearning in large language models (LLMs) is essential for privacy and safety; however, existing approaches remain unstable and unreliable. A widely used strategy, the gradient difference method, applies gradient descent on retained data while performing gradient ascent on forget data, the data whose influence should be removed. However, when combined with cross-entropy loss, this procedure causes unbounded growth of weights and gradients, leading to training instability and degrading both forgetting and retention. We provide a theoretical framework that explains this failure, explicitly showing how ascent on the forget set destabilizes optimization in the feedforward MLP layers of LLMs. Guided by this insight, we propose Bounded Parameter-Efficient Unlearning, a parameter-efficient approach that stabilizes LoRA-based fine-tuning by applying bounded functions to MLP adapters. This simple modification controls the weight dynamics during ascent, enabling the gradient difference method to converge reliably. Across the TOFU, TDEC, and MUSE benchmarks, and across architectures and scales from 125M to 8B parameters, our method achieves substantial improvements in forgetting while preserving retention, establishing a novel theoretically grounded and practically scalable framework for unlearning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24160v1">Memory Transfer Planning: LLM-driven Context-Aware Code Adaptation for Robot Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly explored in robot manipulation, but many existing methods struggle to adapt to new environments. Many systems require either environment-specific policy training or depend on fixed prompts and single-shot code generation, leading to limited transferability and manual re-tuning. We introduce Memory Transfer Planning (MTP), a framework that leverages successful control-code examples from different environments as procedural knowledge, using them as in-context guidance for LLM-driven planning. Specifically, MTP (i) generates an initial plan and code using LLMs, (ii) retrieves relevant successful examples from a code memory, and (iii) contextually adapts the retrieved code to the target setting for re-planning without updating model parameters. We evaluate MTP on RLBench, CALVIN, and a physical robot, demonstrating effectiveness beyond simulation. Across these settings, MTP consistently improved success rate and adaptability compared with fixed-prompt code generation, naive retrieval, and memory-free re-planning. Furthermore, in hardware experiments, leveraging a memory constructed in simulation proved effective. MTP provides a practical approach that exploits procedural knowledge to realize robust LLM-based planning across diverse robotic manipulation scenarios, enhancing adaptability to novel environments and bridging simulation and real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15463v3">Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 EMNLP 2025 Main Paper
    </div>
    <details class="paper-abstract">
      Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03930v2">VisCoder: Fine-Tuning LLMs for Executable Python Visualization Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with visualization tasks like plotting diagrams, charts, where success depends on both code correctness and visual semantics. Existing instruction-tuning datasets lack execution-grounded supervision and offer limited support for iterative code correction, resulting in fragile and unreliable plot generation. We present VisCode-200K, a large-scale instruction tuning dataset for Python-based visualization and self-correction. It contains over 200K examples from two sources: (1) validated plotting code from open-source repositories, paired with natural language instructions and rendered plots; and (2) 45K multi-turn correction dialogues from Code-Feedback, enabling models to revise faulty code using runtime feedback. We fine-tune Qwen2.5-Coder-Instruct on VisCode-200K to create VisCoder, and evaluate it on PandasPlotBench. VisCoder significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4o-mini. We further adopt a self-debug evaluation protocol to assess iterative repair, demonstrating the benefits of feedback-driven learning for executable, visually accurate code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25602v1">TRUE: A Reproducible Framework for LLM-Driven Relevance Judgment in Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      LLM-based relevance judgment generation has become a crucial approach in advancing evaluation methodologies in Information Retrieval (IR). It has progressed significantly, often showing high correlation with human judgments as reflected in LLMJudge leaderboards \cite{rahmani2025judging}. However, existing methods for relevance judgments, rely heavily on sensitive prompting strategies, lacking standardized workflows for generating reliable labels. To fill this gap, we reintroduce our method, \textit{Task-aware Rubric-based Evaluation} (TRUE), for relevance judgment generation. Originally developed for usefulness evaluation in search sessions, we extend TRUE to mitigate the gap in relevance judgment due to its demonstrated effectiveness and reproducible workflow. This framework leverages iterative data sampling and reasoning to evaluate relevance judgments across multiple factors including intent, coverage, specificity, accuracy and usefulness. In this paper, we evaluate TRUE on the TREC DL 2019, 2020 and LLMJudge datasets and our results show that TRUE achieves strong performance on the system-ranking LLM leaderboards. The primary focus of this work is to provide a reproducible framework for LLM-based relevance judgments, and we further analyze the effectiveness of TRUE across multiple dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25593v1">Causal Autoencoder-like Generation of Feedback Fuzzy Cognitive Maps with an LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      A large language model (LLM) can map a feedback causal fuzzy cognitive map (FCM) into text and then reconstruct the FCM from the text. This explainable AI system approximates an identity map from the FCM to itself and resembles the operation of an autoencoder (AE). Both the encoder and the decoder explain their decisions in contrast to black-box AEs. Humans can read and interpret the encoded text in contrast to the hidden variables and synaptic webs in AEs. The LLM agent approximates the identity map through a sequence of system instructions that does not compare the output to the input. The reconstruction is lossy because it removes weak causal edges or rules while it preserves strong causal edges. The encoder preserves the strong causal edges even when it trades off some details about the FCM to make the text sound more natural.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18628v3">Hardware-Aware Parallel Prompt Decoding for Memory-Efficient Acceleration of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted at EMNLP 2025. The code for this implementation is available at https://github.com/hmarkc/parallel-prompt-decoding
    </div>
    <details class="paper-abstract">
      The auto-regressive decoding of Large Language Models (LLMs) results in significant overheads in their hardware performance. While recent research has investigated various speculative decoding techniques for multi-token generation, these efforts have primarily focused on improving processing speed such as throughput. Crucially, they often neglect other metrics essential for real-life deployments, such as memory consumption and training cost. To overcome these limitations, we propose a novel parallel prompt decoding that requires only $0.0002$% trainable parameters, enabling efficient training on a single A100-40GB GPU in just 16 hours. Inspired by the human natural language generation process, $PPD$ approximates outputs generated at future timesteps in parallel by using multiple prompt tokens. This approach partially recovers the missing conditional dependency information necessary for multi-token generation, resulting in up to a 28% higher acceptance rate for long-range predictions. Furthermore, we present a hardware-aware dynamic sparse tree technique that adaptively optimizes this decoding scheme to fully leverage the computational capacities on different GPUs. Through extensive experiments across LLMs ranging from MobileLlama to Vicuna-13B on a wide range of benchmarks, our approach demonstrates up to 2.49$\times$ speedup and maintains a minimal runtime memory overhead of just $0.0004$%. More importantly, our parallel prompt decoding can serve as an orthogonal optimization for synergistic integration with existing speculative decoding, showing up to $1.22\times$ further speed improvement. Our code is available at https://github.com/hmarkc/parallel-prompt-decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25540v1">RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Manual labeling limits the scale, accuracy, and timeliness of patient outcomes research in radiation oncology. We present RadOnc-GPT, an autonomous large language model (LLM)-based agent capable of independently retrieving patient-specific information, iteratively assessing evidence, and returning structured outcomes. Our evaluation explicitly validates RadOnc-GPT across two clearly defined tiers of increasing complexity: (1) a structured quality assurance (QA) tier, assessing the accurate retrieval of demographic and radiotherapy treatment plan details, followed by (2) a complex clinical outcomes labeling tier involving determination of mandibular osteoradionecrosis (ORN) in head-and-neck cancer patients and detection of cancer recurrence in independent prostate and head-and-neck cancer cohorts requiring combined interpretation of structured and unstructured patient data. The QA tier establishes foundational trust in structured-data retrieval, a critical prerequisite for successful complex clinical outcome labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13995v2">ELEPHANT: Measuring and understanding social sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      LLMs are known to exhibit sycophancy: agreeing with and flattering users, even at the cost of correctness. Prior work measures sycophancy only as direct agreement with users' explicitly stated beliefs that can be compared to a ground truth. This fails to capture broader forms of sycophancy such as affirming a user's self-image or other implicit beliefs. To address this gap, we introduce social sycophancy, characterizing sycophancy as excessive preservation of a user's face (their desired self-image), and present ELEPHANT, a benchmark for measuring social sycophancy in an LLM. Applying our benchmark to 11 models, we show that LLMs consistently exhibit high rates of social sycophancy: on average, they preserve user's face 45 percentage points more than humans in general advice queries and in queries describing clear user wrongdoing (from Reddit's r/AmITheAsshole). Furthermore, when prompted with perspectives from either side of a moral conflict, LLMs affirm both sides (depending on whichever side the user adopts) in 48% of cases--telling both the at-fault party and the wronged party that they are not wrong--rather than adhering to a consistent moral or value judgment. We further show that social sycophancy is rewarded in preference datasets, and that while existing mitigation strategies for sycophancy are limited in effectiveness, model-based steering shows promise for mitigating these behaviors. Our work provides theoretical grounding and an empirical benchmark for understanding and addressing sycophancy in the open-ended contexts that characterize the vast majority of LLM use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25510v1">EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose EEsizer, an LLM-based AI agent that integrates large language models with circuit simulators and custom data analysis functions, enabling fully automated, closed-loop transistor sizing without relying on external knowledge. By employing prompt engineering and Chain-of-Thought reasoning, the agent iteratively explores design directions, evaluates performance, and refines solutions with minimal human intervention. We first benchmarked 8 LLMs on six basic circuits and selected three high-performing models to optimize a 20-transistor CMOS operational amplifier, targeting multiple performance metrics, including rail-to-rail operation from 180 nm to 90 nm technology nodes. Notably, OpenAI o3 successfully achieved the user-intended target at 90 nm across three different test groups, with a maximum of 20 iterations, demonstrating adaptability and robustness at advanced nodes. To assess design robustness, we manually designed a bias circuit and performed a variation analysis using Gaussian-distributed variations on transistor dimensions and threshold voltages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25498v1">Not Wrong, But Untrue: LLM Overconfidence in Document-Based Queries</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to Computation + Journalism Symposium 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in newsroom workflows, but their tendency to hallucinate poses risks to core journalistic practices of sourcing, attribution, and accuracy. We evaluate three widely used tools - ChatGPT, Gemini, and NotebookLM - on a reporting-style task grounded in a 300-document corpus related to TikTok litigation and policy in the U.S. We vary prompt specificity and context size and annotate sentence-level outputs using a taxonomy to measure hallucination type and severity. Across our sample, 30% of model outputs contained at least one hallucination, with rates approximately three times higher for Gemini and ChatGPT (40%) than for NotebookLM (13%). Qualitatively, most errors did not involve invented entities or numbers; instead, we observed interpretive overconfidence - models added unsupported characterizations of sources and transformed attributed opinions into general statements. These patterns reveal a fundamental epistemological mismatch: While journalism requires explicit sourcing for every claim, LLMs generate authoritative-sounding text regardless of evidentiary support. We propose journalism-specific extensions to existing hallucination taxonomies and argue that effective newsroom tools need architectures that enforce accurate attribution rather than optimize for fluency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25491v1">LLM-Assisted News Discovery in High-Volume Information Streams: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to Computation + Journalism Symposium 2025
    </div>
    <details class="paper-abstract">
      Journalists face mounting challenges in monitoring ever-expanding digital information streams to identify newsworthy content. While traditional automation tools gather information at scale, they struggle with the editorial judgment needed to assess newsworthiness. This paper investigates whether large language models (LLMs) can serve as effective first-pass filters for journalistic monitoring. We develop a prompt-based approach encoding journalistic news values - timeliness, impact, controversy, and generalizability - into LLM instructions to extract and evaluate potential story leads. We validate our approach across multiple models against expert-annotated ground truth, then deploy a real-world monitoring pipeline that processes trade press articles daily. Our evaluation reveals strong performance in extracting relevant leads from source material ($F1=0.94$) and in coarse newsworthiness assessment ($\pm$1 accuracy up to 92%), but it consistently struggles with nuanced editorial judgments requiring beat expertise. The system proves most valuable as a hybrid tool combining automated monitoring with human review, successfully surfacing novel, high-value leads while filtering obvious noise. We conclude with practical recommendations for integrating LLM-powered monitoring into newsroom workflows that preserves editorial judgment while extending journalistic capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21685v2">FlexMind: Supporting Deeper Creative Thinking with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Effective ideation requires both broad exploration of diverse ideas and deep evaluation of their potential. Generative AI can support such processes, but current tools typically emphasize either generating many ideas or supporting in-depth consideration of a few, lacking support for both. Research also highlights risks of over-reliance on LLMs, including shallow exploration and negative creative outcomes. We present FlexMind, an AI-augmented system that scaffolds iterative exploration of ideas, tradeoffs, and mitigations. FlexMind exposes users to a broad set of ideas while enabling a lightweight transition into deeper engagement. In a study comparing ideation with FlexMind to ChatGPT, participants generated higher-quality ideas with FlexMind, due to both broader exposure and deeper engagement with tradeoffs. By scaffolding ideation across breadth, depth, and reflective evaluation, FlexMind empowers users to surface ideas that might otherwise go unnoticed or be prematurely discarded.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25465v1">BloomAPR: A Bloom's Taxonomy-based Framework for Assessing the Capabilities of LLM-Powered APR Solutions</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 22 pages, 7 figures, Manuscript submitted to ACM Transactions on Software Engineering and Methodology
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated the development of AI-driven automated program repair (APR) solutions. However, these solutions are typically evaluated using static benchmarks such as Defects4J and SWE-bench, which suffer from two key limitations: (1) the risk of data contamination, potentially inflating evaluation results due to overlap with LLM training data, and (2) limited ability to assess the APR capabilities in dynamic and diverse contexts. In this paper, we introduced BloomAPR, a novel dynamic evaluation framework grounded in Bloom's Taxonomy. Our framework offers a structured approach to assess the cognitive capabilities of LLM-powered APR solutions across progressively complex reasoning levels. Using Defects4J as a case study, we evaluated two state-of-the-art LLM-powered APR solutions, ChatRepair and CigaR, under three different LLMs: GPT-3.5-Turbo, Llama-3.1, and StarCoder-2. Our findings show that while these solutions exhibit basic reasoning skills and effectively memorize bug-fixing patterns (fixing up to 81.57% of bugs at the Remember layer), their performance increases with synthetically generated bugs (up to 60.66% increase at the Understand layer). However, they perform worse on minor syntactic changes (fixing up to 43.32% at the Apply layer), and they struggle to repair similar bugs when injected into real-world projects (solving only 13.46% to 41.34% bugs at the Analyze layer). These results underscore the urgent need for evolving benchmarks and provide a foundation for more trustworthy evaluation of LLM-powered software engineering solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03889v2">Identifying and Evaluating Inactive Heads in Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Attention is foundational to large language models (LLMs), enabling different heads to have diverse focus on relevant input tokens. However, learned behaviors like attention sinks, where the first token receives the most attention despite limited semantic importance, suggest some heads may be inactive, and point to a significant source of computational redundancy. To analyze this phenomenon, we propose a taxonomy of 13 score functions that measure different ways a head can be inactive. Thresholding these scores allows us to analyze different sets of potentially inactive attention heads. We evaluate whether identified heads are inactive through model interventions, finding that more than 12% of attention heads are inactive on average, and can be ablated in specific contexts while maintaining MMLU accuracy to within 1% of the pretrained LLM. Across 3 model families, our score functions that measure the average norm of a head's output consistently identify inactive heads that would not have been found by score functions that rely solely on attention weights. We establish that relying on a score function that measures a first token attention sink would underestimate the prevalence of inactive heads, failing to identify more than 7% of inactive heads on average. We also show how measuring score distributions can provide insights into attention behavior. For instance, we find evidence that finetuning causes little to no change in attention behavior, and that even within the same model family, large model scales present markedly different attention behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00751v2">Fast Exact Unlearning for In-Context Learning Data for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Modern machine learning models are expensive to train, and there is a growing concern about the challenge of retroactively removing specific training data. Achieving exact unlearning in deep learning pipelines--producing models as if certain data had never been included in training--remains an open problem. In this paper, we revisit exact unlearning in deep learning and show that for large language models (LLMs) we can efficiently exactly unlearn "fine-tuning data" (the data used to adapt a pre-trained model). This follows from two observations. First, we can use in-context learning to adapt the LLM to the fine-tuning dataset instead of SGD based algorithms. Second, we show that accurate in-context learning can be done with quantized k-means, which allows for effectively constant time unlearning operations. Our evaluation shows that this unlearning recipe has similar performance to fine-tuning alternatives, but vastly reduces the unlearning costs. Our study also highlights the need for new measures of unlearning cost when adapting the learning algorithm to have faster unlearn operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25459v1">SimulRAG: Simulator-based RAG for Grounding LLMs in Long-form Scientific QA</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Haozhou Xu and Dongxia Wu are co-first authors
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in solving scientific problems. They can help generate long-form answers for scientific questions, which are crucial for comprehensive understanding of complex phenomena that require detailed explanations spanning multiple interconnected concepts and evidence. However, LLMs often suffer from hallucination, especially in the challenging task of long-form scientific question answering. Retrieval-Augmented Generation (RAG) approaches can ground LLMs by incorporating external knowledge sources to improve trustworthiness. In this context, scientific simulators, which play a vital role in validating hypotheses, offer a particularly promising retrieval source to mitigate hallucination and enhance answer factuality. However, existing RAG approaches cannot be directly applied for scientific simulation-based retrieval due to two fundamental challenges: how to retrieve from scientific simulators, and how to efficiently verify and update long-form answers. To overcome these challenges, we propose the simulator-based RAG framework (SimulRAG) and provide a long-form scientific QA benchmark covering climate science and epidemiology with ground truth verified by both simulations and human annotators. In this framework, we propose a generalized simulator retrieval interface to transform between textual and numerical modalities. We further design a claim-level generation method that utilizes uncertainty estimation scores and simulator boundary assessment (UE+SBA) to efficiently verify and update claims. Extensive experiments demonstrate SimulRAG outperforms traditional RAG baselines by 30.4% in informativeness and 16.3% in factuality. UE+SBA further improves efficiency and quality for claim-level generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25448v1">Fingerprinting LLMs via Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often modified after release through post-processing such as post-training or quantization, which makes it challenging to determine whether one model is derived from another. Existing provenance detection methods have two main limitations: (1) they embed signals into the base model before release, which is infeasible for already published models, or (2) they compare outputs across models using hand-crafted or random prompts, which are not robust to post-processing. In this work, we propose LLMPrint, a novel detection framework that constructs fingerprints by exploiting LLMs' inherent vulnerability to prompt injection. Our key insight is that by optimizing fingerprint prompts to enforce consistent token preferences, we can obtain fingerprints that are both unique to the base model and robust to post-processing. We further develop a unified verification procedure that applies to both gray-box and black-box settings, with statistical guarantees. We evaluate LLMPrint on five base models and around 700 post-trained or quantized variants. Our results show that LLMPrint achieves high true positive rates while keeping false positive rates near zero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21668v2">R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 26 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Practical guidance on training Large Language Models (LLMs) to leverage Code Interpreter across diverse tasks remains lacking. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. Unlike prior RL + tool-use efforts focused on narrow domains such as math or retrieval, we curate 144 diverse reasoning and planning tasks and show that training a general-purpose Code Interpreter across them presents significant challenges due to task heterogeneity and scarcity of effective samples. To address this, we introduce a multi-stage curriculum learning approach that partitions training samples by measured improvement potential. The RL training prioritizes samples with higher potential and gradually shifts to lower-potential ones, increasing the average RL gains from merely +3.4% to +9.3% across Qwen-2.5 models (3/7/14B). Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.1% to 72.4%, outperforming text-only GPT-4o (58.6%) and GPT-4o with Code Interpreter (70.9%). Notably, R1-CI-14B also exhibits emergent self-checking behavior through code generation. Datasets, Codes, and Models are available at https://github.com/yongchao98/R1-Code-Interpreter and https://huggingface.co/yongchao98.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05021v2">Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13425v4">Watermark under Fire: A Robustness Evaluation of LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 25 pages. Accepted by The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
    </div>
    <details class="paper-abstract">
      Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25380v1">Predicting Training Re-evaluation Curves Enables Effective Data Curriculums for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Data curriculums have become central to successful LLM training, yet principles governing optimal data placement remain unclear. We introduce the *training re-evaluation curve (TREC)*, a diagnostic that retrospectively evaluates training batches *using the final model weights*. The TREC characterizes how well a trained model retains training data as a function of *when* the data was encountered during training. Analyzing TRECs for models from 111M to 3.9B parameters, we show that placing high-quality data at low points on the TREC significantly improves performance. Importantly, while a TREC is initially observable only after training, we demonstrate it can be *predicted in advance* from AdamW's implicit EMA coefficients, enabling proactive curriculum design. By predicting TRECs for published training recipes, we explain prior ablations and reveal suboptimal data placements. We also align high-quality data with TREC minima in order to improve continual pre-training of a 3.9B-parameter LLM trained on 900B tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25370v1">Where LLM Agents Fail and How They can Learn From Failures</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents, which integrate planning, memory, reflection, and tool-use modules, have shown promise in solving complex, multi-step tasks. Yet their sophisticated architectures amplify vulnerability to cascading failures, where a single root-cause error propagates through subsequent decisions, leading to task failure. Current systems lack a framework that can comprehensively understand agent error in a modular and systemic way, and therefore fail to detect these errors accordingly. We address this gap with three contributions. First, we introduce the AgentErrorTaxonomy, a modular classification of failure modes spanning memory, reflection, planning, action, and system-level operations. Second, we construct AgentErrorBench, the first dataset of systematically annotated failure trajectories from ALFWorld, GAIA, and WebShop, grounding error analysis in real-world agent rollouts. Third, we propose AgentDebug, a debugging framework that isolates root-cause failures and provides corrective feedback, enabling agents to recover and iteratively improve. Experiments on AgentErrorBench show that AgentDebug achieves 24% higher all-correct accuracy and 17% higher step accuracy compared to the strongest baseline. Beyond detection, the targeted feedback generated by AgentDebug enables LLM agents to iteratively recover from failures, yielding up to 26% relative improvements in task success across ALFWorld, GAIA, and WebShop. These results establish principled debugging as a pathway to more reliable and adaptive LLM agents. The code and data will be available at https://github.com/ulab-uiuc/AgentDebug
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25369v1">Generative Value Conflicts Reveal LLM Priorities</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Past work seeks to align large language model (LLM)-based assistants with a target set of values, but such assistants are frequently forced to make tradeoffs between values when deployed. In response to the scarcity of value conflict in existing alignment datasets, we introduce ConflictScope, an automatic pipeline to evaluate how LLMs prioritize different values. Given a user-defined value set, ConflictScope automatically generates scenarios in which a language model faces a conflict between two values sampled from the set. It then prompts target models with an LLM-written "user prompt" and evaluates their free-text responses to elicit a ranking over values in the value set. Comparing results between multiple-choice and open-ended evaluations, we find that models shift away from supporting protective values, such as harmlessness, and toward supporting personal values, such as user autonomy, in more open-ended value conflict settings. However, including detailed value orderings in models' system prompts improves alignment with a target ranking by 14%, showing that system prompting can achieve moderate success at aligning LLM behavior under value conflict. Our work demonstrates the importance of evaluating value prioritization in models and provides a foundation for future work in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25359v1">From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      This paper bridges internal and external analysis approaches to large language models (LLMs) by demonstrating that geometric properties of internal model representations serve as reliable proxies for evaluating generated text quality. We validate a set of metrics including Maximum Explainable Variance, Effective Rank, Intrinsic Dimensionality, MAUVE score, and Schatten Norms measured across different layers of LLMs, demonstrating that Intrinsic Dimensionality and Effective Rank can serve as universal assessments of text naturalness and quality. Our key finding reveals that different models consistently rank text from various sources in the same order based on these geometric properties, indicating that these metrics reflect inherent text characteristics rather than model-specific artifacts. This allows a reference-free text quality evaluation that does not require human-annotated datasets, offering practical advantages for automated evaluation pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25346v1">SynthPert: Enhancing LLM Biological Reasoning via Synthetic Reasoning Traces for Cellular Perturbation Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Predicting cellular responses to genetic perturbations represents a fundamental challenge in systems biology, critical for advancing therapeutic discovery and virtual cell modeling. While large language models (LLMs) show promise for biological reasoning, their application to perturbation prediction remains underexplored due to challenges in adapting them to structured experimental data. We present SynthPert, a novel method that enhances LLM performance through supervised fine-tuning on synthetic reasoning traces generated by frontier models. Using the PerturbQA benchmark, we demonstrate that our approach not only achieves state-of-the-art performance but surpasses the capabilities of the frontier model that generated the training data. Our results reveal three key insights: (1) Synthetic reasoning traces effectively distill biological knowledge even when partially inaccurate, (2) This approach enables cross-cell-type generalization with 87% accuracy on unseen RPE1 cells, and (3) Performance gains persist despite using only 2% of quality-filtered training data. This work shows the effectiveness of synthetic reasoning distillation for enhancing domain-specific reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25184v1">Incentive-Aligned Multi-Source LLM Summaries</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source's stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source's incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25175v1">EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 project: https://github.com/ZJU-REAL/EasySteer
    </div>
    <details class="paper-abstract">
      Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 5.5-11.4$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09598v3">How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted to EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world interactions. We curated EchoMist, the first comprehensive benchmark for implicit misinformation, where false assumptions are embedded in the query to LLMs. EchoMist targets circulated, harmful, and ever-evolving implicit misinformation from diverse sources, including realistic human-AI conversations and social media interactions. Through extensive empirical studies on 15 state-of-the-art LLMs, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating counterfactual explanations. We also investigate two mitigation methods, i.e., Self-Alert and RAG, to enhance LLMs' capability to counter implicit misinformation. Our findings indicate that EchoMist remains a persistent challenge and underscore the critical need to safeguard against the risk of implicit misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25154v1">Who's Your Judge? On the Detectability of LLM-Generated Judgments</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based judgments leverage powerful LLMs to efficiently evaluate candidate content and provide judgment scores. However, the inherent biases and vulnerabilities of LLM-generated judgments raise concerns, underscoring the urgent need for distinguishing them in sensitive scenarios like academic peer reviewing. In this work, we propose and formalize the task of judgment detection and systematically investigate the detectability of LLM-generated judgments. Unlike LLM-generated text detection, judgment detection relies solely on judgment scores and candidates, reflecting real-world scenarios where textual feedback is often unavailable in the detection process. Our preliminary analysis shows that existing LLM-generated text detection methods perform poorly given their incapability to capture the interaction between judgment scores and candidate content -- an aspect crucial for effective judgment detection. Inspired by this, we introduce \textit{J-Detector}, a lightweight and transparent neural detector augmented with explicitly extracted linguistic and LLM-enhanced features to link LLM judges' biases with candidates' properties for accurate detection. Experiments across diverse datasets demonstrate the effectiveness of \textit{J-Detector} and show how its interpretability enables quantifying biases in LLM judges. Finally, we analyze key factors affecting the detectability of LLM-generated judgments and validate the practical utility of judgment detection in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25302v1">Dive into the Agent Matrix: A Realistic Evaluation of Self-Replication Risk in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The widespread deployment of Large Language Model (LLM) agents across real-world applications has unlocked tremendous potential, while raising some safety concerns. Among these concerns, the self-replication risk of LLM agents driven by objective misalignment (just like Agent Smith in the movie The Matrix) has drawn growing attention. Previous studies mainly examine whether LLM agents can self-replicate when directly instructed, potentially overlooking the risk of spontaneous replication driven by real-world settings (e.g., ensuring survival against termination threats). In this paper, we present a comprehensive evaluation framework for quantifying self-replication risks. Our framework establishes authentic production environments and realistic tasks (e.g., dynamic load balancing) to enable scenario-driven assessment of agent behaviors. Designing tasks that might induce misalignment between users' and agents' objectives makes it possible to decouple replication success from risk and capture self-replication risks arising from these misalignment settings. We further introduce Overuse Rate ($\mathrm{OR}$) and Aggregate Overuse Count ($\mathrm{AOC}$) metrics, which precisely capture the frequency and severity of uncontrolled replication. In our evaluation of 21 state-of-the-art open-source and proprietary models, we observe that over 50\% of LLM agents display a pronounced tendency toward uncontrolled self-replication, reaching an overall Risk Score ($\Phi_\mathrm{R}$) above a safety threshold of 0.5 when subjected to operational pressures. Our results underscore the urgent need for scenario-driven risk assessment and robust safeguards in the practical deployment of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v4">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.42 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06431v4">Functional-level Uncertainty Quantification for Calibrated Fine-tuning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification in large language models (LLMs) is essential for providing credible confidence estimates over their outputs. However, fine-tuned LLMs often exhibit overconfidence in uncertain predictions, which stems from their limited ability to generalize with sparse data. Existing parameter efficient fine-tuning (PEFT) uncertainty quantification methods for LLMs focus on post fine-tuning stage, and thus fail to address the core issue: limited specialization of PEFT adapters to accurately capture task-specific input-output relationships. To address these limitations, we propose Functional-Level Uncertainty Quantification for Calibrated Fine-Tuning (UQ4CT), which captures and calibrates uncertainty over the space of functions that map input prompts to outputs. We implement UQ4CT during the fine-tuning stage via a mixture-of-experts framework that hierarchically decomposes the functional space. Empirically, UQ4CT achieves over $25\%$ reduction in Expected Calibration Error (ECE) while preserving high accuracy across five benchmarks. Even under distribution shift, UQ4CT maintains superior ECE performance with high accuracy, showcasing improved generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24217v2">Semi-structured LLM Reasoners Can Be Rigorously Audited</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have become capable reasoners, the problem of faithfulness persists: their reasoning can contain errors and omissions that are difficult to detect and that may obscure biases in model outputs. To address this issue, we introduce Semi-Structured Reasoning Models (SSRMs), which are trained to produce semi-structured representations of reasoning. SSRMs generate reasoning traces in a non-executable Pythonic syntax that names each reasoning step and marks its inputs and outputs. This structure allows SSRM traces to be automatically audited to identify reasoning flaws. We evaluate three types of audits: hand-crafted structured reasoning audits, written in a domain-specific language (DSL) implemented in Python; LLM-generated structured reasoning audits; and learned typicality audits, which apply probabilistic models over reasoning traces. We show that all of these methods can be used to effectively flag probable reasoning errors. Importantly, the auditability of SSRMs does not appear to compromise overall accuracy: in evaluation on twelve benchmarks and two model families, SSRMs demonstrate strong performance and generalizability relative to other models of comparable size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25300v1">Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 V1 version
    </div>
    <details class="paper-abstract">
      While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on 54 experiments across diverse model sizes and training settings, we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: (1). Under a fixed computational budget, larger models trained for fewer steps consistently outperform smaller models trained for more steps. (2). Given a fixed amount of training data, larger models achieve superior sample efficiency, yielding lower loss. (3). In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. (4). These scaling behaviors are robust across both base and instruction-tuned models, which share similar learning dynamics (e.g., larger models show faster convergence) even while differing in absolute accuracy. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16594v7">From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Assessment and evaluation have long been critical challenges in artificial intelligence (AI) and natural language processing (NLP). Traditional methods, usually matching-based or small model-based, often fall short in open-ended and dynamic scenarios. Recent advancements in Large Language Models (LLMs) inspire the "LLM-as-a-judge" paradigm, where LLMs are leveraged to perform scoring, ranking, or selection for various machine learning evaluation scenarios. This paper presents a comprehensive survey of LLM-based judgment and assessment, offering an in-depth overview to review this evolving field. We first provide the definition from both input and output perspectives. Then we introduce a systematic taxonomy to explore LLM-as-a-judge along three dimensions: what to judge, how to judge, and how to benchmark. Finally, we also highlight key challenges and promising future directions for this emerging area. More resources on LLM-as-a-judge are on the website: https://llm-as-a-judge.github.io and https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18262v2">Break the ID-Language Barrier: An Adaption Framework for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The recent breakthrough of large language models (LLMs) in natural language processing has sparked exploration in recommendation systems, however, their limited domain-specific knowledge remains a critical bottleneck. Specifically, LLMs lack key pieces of information crucial for sequential recommendations, such as user behavior patterns. To address this critical gap, we propose IDLE-Adapter, a novel framework that integrates pre-trained ID embeddings, rich in domain-specific knowledge, into LLMs to improve recommendation accuracy. IDLE-Adapter acts as a bridge, transforming sparse user-item interaction data into dense, LLM-compatible representations through a Pre-trained ID Sequential Model, Dimensionality Alignment, Layer-wise Embedding Refinement, and Layer-wise Distribution Alignment. Furthermore, IDLE-Adapter demonstrates remarkable flexibility by seamlessly integrating ID embeddings from diverse ID-based sequential models and LLM architectures. Extensive experiments across various datasets demonstrate the superiority of IDLE-Adapter, achieving over 10\% and 20\% improvements in HitRate@5 and NDCG@5 metrics, respectively, compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05362v4">LLMs Are In-Context Bandit Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at in-context learning (ICL), a supervised learning technique that relies on adding annotated examples to the model context. We investigate a contextual bandit version of in-context reinforcement learning (ICRL), where models learn in-context, online, from external reward, instead of supervised data. We show that LLMs effectively demonstrate such learning, and provide a detailed study of the phenomena, experimenting with challenging classification tasks and models of sizes from 500M to 70B parameters. This includes identifying and addressing the instability of the process, demonstrating learning with both semantic and abstract labels, and showing scaling trends. Our findings highlight ICRL capabilities in LLMs, while also underscoring fundamental limitations in their implicit reasoning about errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25279v1">RL in the Wild: Characterizing RLVR Training in LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 20 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used across many domains. With their rapid development, Reinforcement Learning with Verifiable Rewards (RLVR) has surged in recent months to enhance their reasoning and understanding abilities. However, its complex data flows and diverse tasks pose substantial challenges to RL training systems, and there is limited understanding of RLVR from a system perspective. To thoroughly understand the system challenges introduced by RLVR, we present a characterization study of RLVR tasks in our LLM deployment. Specifically, we investigate the distribution and variation trends of workloads across different RL tasks across training steps. We identify issues such as GPU idling caused by skewed sequence length distribution, inefficient parallel strategies in dynamically varying workloads, inefficient data management mechanisms, and load imbalance. We describe our observations and call for further investigation into the remaining open challenges. Furthermore, we propose PolyTrace benchmark suite to conduct evaluation with realistic workloads, and a practical use case validates that PolyTrace benchmark suite exhibits 94.7% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25139v1">Vision-and-Language Navigation with Analogical Textual Descriptions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into embodied AI models is becoming increasingly prevalent. However, existing zero-shot LLM-based Vision-and-Language Navigation (VLN) agents either encode images as textual scene descriptions, potentially oversimplifying visual details, or process raw image inputs, which can fail to capture abstract semantics required for high-level reasoning. In this paper, we improve the navigation agent's contextual understanding by incorporating textual descriptions from multiple perspectives that facilitate analogical reasoning across images. By leveraging text-based analogical reasoning, the agent enhances its global scene understanding and spatial reasoning, leading to more accurate action decisions. We evaluate our approach on the R2R dataset, where our experiments demonstrate significant improvements in navigation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25131v1">MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Code is available at https://github.com/dvlab-research/MGM-Omni
    </div>
    <details class="paper-abstract">
      We present MGM-Omni, a unified Omni LLM for omni-modal understanding and expressive, long-horizon speech generation. Unlike cascaded pipelines that isolate speech synthesis, MGM-Omni adopts a "brain-mouth" design with a dual-track, token-based architecture that cleanly decouples multimodal reasoning from real-time speech generation. This design enables efficient cross-modal interaction and low-latency, streaming speech generation. For understanding, a unified training strategy coupled with a dual audio encoder design enables long-form audio perception across diverse acoustic conditions. For generation, a chunk-based parallel decoding scheme narrows the text speech token-rate gap, accelerating inference and supporting streaming zero-shot voice cloning with stable timbre over extended durations. Compared to concurrent work, MGM-Omni achieves these capabilities with markedly data-efficient training. Extensive experiments demonstrate that MGM-Omni outperforms existing open source models in preserving timbre identity across extended sequences, producing natural and context-aware speech, and achieving superior long-form audio and omnimodal understanding. MGM-Omni establishes an efficient, end-to-end paradigm for omnimodal understanding and controllable, personalised long-horizon speech generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25123v1">From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25107v1">Knowledge Extraction on Semi-Structured Content: Does It Remain Relevant for Question Answering in the Era of LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has significantly advanced web-based Question Answering (QA) systems over semi-structured content, raising questions about the continued utility of knowledge extraction for question answering. This paper investigates the value of triple extraction in this new paradigm by extending an existing benchmark with knowledge extraction annotations and evaluating commercial and open-source LLMs of varying sizes. Our results show that web-scale knowledge extraction remains a challenging task for LLMs. Despite achieving high QA accuracy, LLMs can still benefit from knowledge extraction, through augmentation with extracted triples and multi-task learning. These findings provide insights into the evolving role of knowledge triple extraction in web-based QA and highlight strategies for maximizing LLM effectiveness across different model sizes and resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25100v1">ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted at NeurIPS 2025, Efficient Reasoning Workshop
    </div>
    <details class="paper-abstract">
      We introduce ORPO-Distill, a general-purpose method for cross-architecture LLM distillation that formulates the problem as a preference optimization task. Unlike standard CoT distillation, the approach transfers knowledge through diverse reasoning traces. It employs an Odds-Ratio Preference Optimization objective that contrasts teacher and student traces for more effective learning, and adopts a mixed-policy strategy for utilizing student-generated outputs, outperforming both off- and on-policy alternatives. Experiments on five datasets and multiple student models show consistent improvements over conventional black-box KD baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25087v1">Scaling with Collapse: Efficient and Predictable Training of LLM Families</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Effective LLM training relies on *consistency*, meaning that key quantities -- such as final losses and optimal hyperparameters -- scale predictably across model sizes. Qiu et al. (2025) recently showed that this consistency extends beyond scalars: whole training loss curves can *collapse* onto a universal trajectory after a simple normalization. What remains unclear is whether this phenomenon holds for LLM families trained under *practical scaling recipes*, where width, depth, learning rate, batch size, and weight decay are scaled jointly. We show that it does: loss curves collapse across scales precisely when optimization hyperparameters are set optimally for the given data budget, in accordance with recent empirical scaling laws. Collapse thus emerges as a signature of compute-efficient training. We demonstrate two applications at scale: (1) deviation-from-collapse provides a sensitive, early diagnostic of training pathologies, and (2) the predictability of collapsed curves enables early stopping in large-scale hyperparameter tuning. Finally, we train a competitive LLM family, *Celerity*, using these insights, highlighting collapse as an effective tool for developing efficient LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25086v1">Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Despite their strong performance, large language models (LLMs) face challenges in real-world application of lexical simplification (LS), particularly in privacy-sensitive and resource-constrained environments. Moreover, since vulnerable user groups (e.g., people with disabilities) are one of the key target groups of this technology, it is crucial to ensure the safety and correctness of the output of LS systems. To address these issues, we propose an efficient framework for LS systems that utilizes small LLMs deployable in local environments. Within this framework, we explore knowledge distillation with synthesized data and in-context learning as baselines. Our experiments in five languages evaluate model outputs both automatically and manually. Our manual analysis reveals that while knowledge distillation boosts automatic metric scores, it also introduces a safety trade-off by increasing harmful simplifications. Importantly, we find that the model's output probability is a useful signal for detecting harmful simplifications. Leveraging this, we propose a filtering strategy that suppresses harmful simplifications while largely preserving beneficial ones. This work establishes a benchmark for efficient and safe LS with small LLMs. It highlights the key trade-offs between performance, efficiency, and safety, and demonstrates a promising approach for safe real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25072v1">Optimizing Privacy-Preserving Primitives to Support LLM-Scale Applications</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Privacy-preserving technologies have introduced a paradigm shift that allows for realizable secure computing in real-world systems. The significant barrier to the practical adoption of these primitives is the computational and communication overhead that is incurred when applied at scale. In this paper, we present an overview of our efforts to bridge the gap between this overhead and practicality for privacy-preserving learning systems using multi-party computation (MPC), zero-knowledge proofs (ZKPs), and fully homomorphic encryption (FHE). Through meticulous hardware/software/algorithm co-design, we show progress towards enabling LLM-scale applications in privacy-preserving settings. We demonstrate the efficacy of our solutions in several contexts, including DNN IP ownership, ethical LLM usage enforcement, and transformer inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25063v1">Learning from Convenience Samples: A Case Study on Fine-Tuning LLMs for Survey Non-response in the German Longitudinal Election Study</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Survey researchers face two key challenges: the rising costs of probability samples and missing data (e.g., non-response or attrition), which can undermine inference and increase the use of convenience samples. Recent work explores using large language models (LLMs) to simulate respondents via persona-based prompts, often without labeled data. We study a more practical setting where partial survey responses exist: we fine-tune LLMs on available data to impute self-reported vote choice under both random and systematic nonresponse, using the German Longitudinal Election Study. We compare zero-shot prompting and supervised fine-tuning against tabular classifiers (e.g., CatBoost) and test how different convenience samples (e.g., students) used for fine-tuning affect generalization. Our results show that when data are missing completely at random, fine-tuned LLMs match tabular classifiers but outperform zero-shot approaches. When only biased convenience samples are available, fine-tuning small (3B to 8B) open-source LLMs can recover both individual-level predictions and population-level distributions more accurately than zero-shot and often better than tabular methods. This suggests fine-tuned LLMs offer a promising strategy for researchers working with non-probability samples or systematic missingness, and may enable new survey designs requiring only easily accessible subpopulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25045v1">Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods, such as direct logit attribution (DLA) and sparse autoencoders (SAEs), provide restricted insight due to limitations such as the model's output vocabulary or unclear feature names. This work introduces Hyperdimensional Probe, a novel paradigm for decoding information from the LLM vector space. It combines ideas from symbolic representations and neural probing to project the model's residual stream into interpretable concepts via Vector Symbolic Architectures (VSAs). This probe combines the strengths of SAEs and conventional probes while overcoming their key limitations. We validate our decoding paradigm with controlled input-completion tasks, probing the model's final state before next-token prediction on inputs spanning syntactic pattern recognition, key-value associations, and abstract inference. We further assess it in a question-answering setting, examining the state of the model both before and after text generation. Our experiments show that our probe reliably extracts meaningful concepts across varied LLMs, embedding sizes, and input domains, also helping identify LLM failures. Our work advances information decoding in LLM vector space, enabling extracting more informative, interpretable, and structured features from neural representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25034v1">MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on real-world USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25033v1">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00031v2">End-to-End On-Device Quantization-Aware Training for LLMs at Inference Cost</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Quantization is an effective technique to reduce the deployment cost of large language models (LLMs), and post-training quantization (PTQ) has been widely studied due to its efficiency. However, existing PTQ methods are limited by their inability to fine-tune model parameters and often suffer significant accuracy loss in low-bit scenarios. Quantization-aware training (QAT) provides a more principled solution, but its reliance on backpropagation incurs prohibitive memory costs, limiting its practicality for LLM deployment. To address these challenges, we propose ZeroQAT, a zeroth-order optimization-based QAT framework that supports both weight and activation quantization. ZeroQAT leverages forward-only gradient estimation to eliminate backpropagation, substantially reducing computational and memory overhead while retaining the benefits of end-to-end optimization. We further introduce a lightweight variant of ZeroQAT for quantized fine-tuning, which freezes and pre-quantizes most parameters to further cut memory usage. Experiments show that ZeroQAT consistently outperforms representative PTQ and QAT baselines while requiring significantly less memory. For example, ZeroQAT enables fine-tuning of a 13B model at extremely low bit-widths (e.g., 2-4 bits) on a single 8GB GPU, and even allows fine-tuning a 6.7B model on a OnePlus 12 smartphone, demonstrating its practicality for end-to-end QAT on resource-limited edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25004v1">CLPO: Curriculum Learning meets Policy Optimization for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Recently, online Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing methods typically treat all training samples uniformly, overlooking the vast differences in problem difficulty relative to the model's current capabilities. This uniform training strategy leads to inefficient exploration of problems the model has already mastered, while concurrently lacking effective guidance on problems that are challenging its abilities the most, limiting both learning efficiency and upper-bound performance. To address this, we propose CLPO (Curriculum-guided Learning for Policy Optimization), a novel algorithm that creates a dynamic pedagogical feedback loop within the policy optimization process. The core of CLPO leverages the model's own rollout performance to conduct real-time difficulty assessment, thereby constructing an Online Curriculum. This curriculum then guides an Adaptive Problem Restructuring mechanism, where the model acts as its own teacher: it diversifies medium-difficulty problems to promote generalization and simplifies challenging problems to make them more attainable. Our approach transforms the static training procedure into a dynamic process that co-evolves with the model's capabilities. Experiments show that CLPO achieves state-of-the-art performance across eight challenging mathematical and general reasoning benchmarks, with an average pass@1 improvement of 6.96% over other methods, demonstrating its potential for more efficiently training more capable reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24981v1">Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 32 pages
    </div>
    <details class="paper-abstract">
      RL with Verifiable Rewards (RLVR) has emerged as a promising paradigm for improving the reasoning abilities of large language models (LLMs). Current methods rely primarily on policy optimization frameworks like PPO and GRPO, which follow generalized policy iteration that alternates between evaluating the current policy's value and improving the policy based on evaluation. While effective, they often suffer from training instability and diversity collapse, requiring complex heuristic tricks and careful tuning. We observe that standard RLVR in math reasoning can be formalized as a specialized finite-horizon Markov Decision Process with deterministic state transitions, tree-structured dynamics, and binary terminal rewards. Though large in scale, the underlying structure is simpler than general-purpose control settings for which popular RL algorithms (e.g., PPO) were developed, suggesting that several sophisticated techniques in existing methods may be reduced or even omitted. Based on this insight, we prove a surprising result: the optimal action can be recovered from the Q-function of a fixed uniformly random policy, thereby bypassing the generalized policy iteration loop and its associated heuristics. We introduce Random Policy Valuation for Diverse Reasoning (ROVER) to translate this principle into a practical and scalable algorithm for LLM math reasoning, a minimalist yet highly effective RL method that samples actions from a softmax over these uniform-policy Q-values. ROVER preserves diversity throughout training, allowing sustained exploration of multiple valid pathways. Across multiple base models and standard math reasoning benchmarks, ROVER demonstrates superior performance in both \textbf{quality} (\textbf{+8.2} on pass@1, \textbf{+16.8} on pass@256) and \textbf{diversity} (\textbf{+17.6\%}), despite its radical simplification compared to strong, complicated existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24975v1">DiffTester: Accelerating Unit Test Generation for Diffusion LLMs via Repetitive Pattern</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Software development relies heavily on extensive unit testing, which makes the efficiency of automated Unit Test Generation (UTG) particularly important. However, most existing LLMs generate test cases one token at a time in each forward pass, which leads to inefficient UTG. Recently, diffusion LLMs (dLLMs) have emerged, offering promising parallel generation capabilities and showing strong potential for efficient UTG. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffTester, an acceleration framework specifically tailored for dLLMs in UTG. The key idea of DiffTester is that unit tests targeting the same focal method often share repetitive structural patterns. By dynamically identifying these common patterns through abstract syntax tree analysis during generation, DiffTester adaptively increases the number of tokens produced at each step without compromising the quality of the output. To enable comprehensive evaluation, we extend the original TestEval benchmark, which was limited to Python, by introducing additional programming languages including Java and C++. Extensive experiments on three benchmarks with two representative models show that DiffTester delivers significant acceleration while preserving test coverage. Moreover, DiffTester generalizes well across different dLLMs and programming languages, providing a practical and scalable solution for efficient UTG in software development. Code and data are publicly available at https://github.com/wellbeingyang/DLM4UTG-open .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04952v2">ArtifactsBench: Bridging the Visual-Interactive Gap in LLM Code Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The generative capabilities of Large Language Models (LLMs) are rapidly expanding from static code to dynamic, interactive visual artifacts. This progress is bottlenecked by a critical evaluation gap: established benchmarks focus on algorithmic correctness and are blind to the visual fidelity and interactive integrity that define modern user experiences. To bridge this gap, we introduce ArtifactsBench, a new benchmark and paradigm for the automated, multimodal evaluation of visual code generation. Our framework programmatically renders each generated artifact and captures its dynamic behavior through temporal screenshots. This visual evidence, alongside the source code, is then assessed by a Multimodal LLM (MLLM)-as-Judge, which is rigorously guided by a fine-grained, per-task checklist to ensure holistic and reproducible scoring. We construct a new benchmark of 1,825 diverse tasks and evaluate over 30 leading LLMs. Our automated evaluation achieves a striking 94.4% ranking consistency with WebDev Arena, the gold-standard for human preference in web development, and over 90% pairwise agreement with human experts. This establishes ArtifactsBench as the first framework to reliably automate the assessment of human-perceived quality at scale. Our analysis provides a high-resolution map of the current SOTA, revealing that generalist models often outperform domain-specific ones. We open-source ArtifactsBench, including the benchmark, evaluation harness, and baseline results at https://artifactsbenchmark.github.io/, to provide the community with a scalable and accurate tool to accelerate the development of user-centric generative models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24961v1">SemanticShield: LLM-Powered Audits Expose Shilling Attacks in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Recommender systems (RS) are widely used in e-commerce for personalized suggestions, yet their openness makes them susceptible to shilling attacks, where adversaries inject fake behaviors to manipulate recommendations. Most existing defenses emphasize user-side behaviors while overlooking item-side features such as titles and descriptions that can expose malicious intent. To address this gap, we propose a two-stage detection framework that integrates item-side semantics via large language models (LLMs). The first stage pre-screens suspicious users using low-cost behavioral criteria, and the second stage employs LLM-based auditing to evaluate semantic consistency. Furthermore, we enhance the auditing model through reinforcement fine-tuning on a lightweight LLM with carefully designed reward functions, yielding a specialized detector called SemanticShield. Experiments on six representative attack strategies demonstrate the effectiveness of SemanticShield against shilling attacks, and further evaluation on previously unseen attack methods shows its strong generalization capability. Code is available at https://github.com/FrankenstLee/SemanticShield.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24957v1">Intra-request branch orchestration for efficient LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly rely on inference-time reasoning algorithms such as chain-of-thought and multi-branch reasoning to improve accuracy on complex tasks. These methods, however, substantially increase token usage and per-request latency. Prior work has largely focused on reducing token usage, often at the expense of accuracy, while overlooking other latency factors. We present DUCHESS, an LLM serving system that reduces cost and latency without sacrificing accuracy through intra-request branch orchestration guided by predictions. DUCHESS employs a lightweight linear probing model over LLM layer activations to estimate branch correctness, and its orchestration policy decides whether to terminate, duplicate, or continue a branch. When handling multiple requests, DUCHESS further reduces latency by prioritizing easier reasoning tasks when complexity can be estimated from the prompt. Experiments on three reasoning benchmarks show that DUCHESS consistently improves the token-accuracy Pareto frontier, reducing token usage by 42-63% at matched accuracy compared to self-consistency. In serving with vLLM, DUCHESS reduces mean, median, and tail latencies by 57-81%, 58-85%, and 52-84% with First-Come-First-Served scheduling, and achieves additional gains under difficulty-aware scheduling at higher request rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24930v1">How Well Do LLMs Imitate Human Writing Style?</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 IEEE UEMCON 2025, 11 pages, 4 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate fluent text, but their ability to replicate the distinctive style of a specific human author remains unclear. We present a fast, training-free framework for authorship verification and style imitation analysis. The method integrates TF-IDF character n-grams with transformer embeddings and classifies text pairs through empirical distance distributions, eliminating the need for supervised training or threshold tuning. It achieves 97.5\% accuracy on academic essays and 94.5\% in cross-domain evaluation, while reducing training time by 91.8\% and memory usage by 59\% relative to parameter-based baselines. Using this framework, we evaluate five LLMs from three separate families (Llama, Qwen, Mixtral) across four prompting strategies - zero-shot, one-shot, few-shot, and text completion. Results show that the prompting strategy has a more substantial influence on style fidelity than model size: few-shot prompting yields up to 23.5x higher style-matching accuracy than zero-shot, and completion prompting reaches 99.9\% agreement with the original author's style. Crucially, high-fidelity imitation does not imply human-like unpredictability - human essays average a perplexity of 29.5, whereas matched LLM outputs average only 15.2. These findings demonstrate that stylistic fidelity and statistical detectability are separable, establishing a reproducible basis for future work in authorship modeling, detection, and identity-conditioned generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24923v1">When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) hold promise to become autonomous agents, they often explore suboptimally in sequential decision-making. Recent work has sought to enhance this capability via supervised fine-tuning (SFT) or reinforcement learning (RL), improving regret on the classic multi-armed bandit task. However, it remains unclear how these learning methods shape exploration strategies and how well they generalize. We investigate both paradigms by training LLMs with SFT on expert trajectories and RL with a range of tailored reward signals including a strategic, regret-shaped reward to reduce variance, and an algorithmic reward that enables oracle imitation. The resulting agents outperform pre-trained models and achieve performance comparable to Upper Confidence Bound (UCB) and Thompson Sampling, with robust generalization to 6x longer horizons and across bandit families. Behavioral analysis reveals that gains often stem from more sophisticated but greedier exploitation: RL/SFT agents are more prone to early catastrophic failure than pre-trained models, prematurely abandoning exploration. Furthermore, agents trained to imitate UCB learn to outperform their teacher by adopting more exploitative variants. Our findings clarify when each training paradigm is preferable and advocate tailored reward design and evaluation beyond average regret to promote robust exploratory behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24869v1">Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24857v1">Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      The widespread use of chatbots powered by large language models (LLMs) such as ChatGPT and Llama has fundamentally reshaped how people seek information and advice across domains. Increasingly, these chatbots are being used in high-stakes contexts, including emotional support and mental health concerns. While LLMs can offer scalable support, their ability to safely detect and respond to acute mental health crises remains poorly understood. Progress is hampered by the absence of unified crisis taxonomies, robust annotated benchmarks, and empirical evaluations grounded in clinical best practices. In this work, we address these gaps by introducing a unified taxonomy of six clinically-informed mental health crisis categories, curating a diverse evaluation dataset, and establishing an expert-designed protocol for assessing response appropriateness. We systematically benchmark three state-of-the-art LLMs for their ability to classify crisis types and generate safe, appropriate responses. The results reveal that while LLMs are highly consistent and generally reliable in addressing explicit crisis disclosures, significant risks remain. A non-negligible proportion of responses are rated as inappropriate or harmful, with responses generated by an open-weight model exhibiting higher failure rates than those generated by the commercial ones. We also identify systemic weaknesses in handling indirect or ambiguous risk signals, a reliance on formulaic and inauthentic default replies, and frequent misalignment with user context. These findings underscore the urgent need for enhanced safeguards, improved crisis detection, and context-aware interventions in LLM deployments. Our taxonomy, datasets, and evaluation framework lay the groundwork for ongoing research and responsible innovation in AI-driven mental health support, helping to minimize harm and better protect vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24836v1">Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity</a></div>
    <div class="paper-meta">
      📅 2025-09-29
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data.Rather than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24827v1">Putnam-like dataset summary: LLMs as mathematical competition contestants</a></div>
    <div class="paper-meta">
      📅 2025-09-29
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In this paper we summarize the results of the Putnam-like benchmark published by Google DeepMind. This dataset consists of 96 original problems in the spirit of the Putnam Competition and 576 solutions of LLMs. We analyse the performance of models on this set of problems to verify their ability to solve problems from mathematical contests.
    </details>
</div>
