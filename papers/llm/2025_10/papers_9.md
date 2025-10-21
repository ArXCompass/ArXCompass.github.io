# llm - 2025_10

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
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06866v1">Unlocking Latent Discourse Translation in LLMs Through Quality-Aware Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as strong contenders in machine translation.Yet, they still struggle to adequately handle discourse phenomena, such as pronoun resolution and lexical cohesion at the document level. In this study, we thoroughly investigate the discourse phenomena performance of LLMs in context-aware translation. We demonstrate that discourse knowledge is encoded within LLMs and propose the use of quality-aware decoding (QAD) to effectively extract this knowledge, showcasing its superiority over other decoding approaches through comprehensive analysis. Furthermore, we illustrate that QAD enhances the semantic richness of translations and aligns them more closely with human preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v3">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We released our code and dataset at https://github.com/penfever/judgment-to-noise
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06843v1">SID: Multi-LLM Debate Driven by Self Signals</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{https://github.com/xuhang2019/SID}{\texttt{https://github.com/xuhang2019/SID}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06096v2">The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11557v2">AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Corporate LLMs are gaining traction for efficient knowledge dissemination and management within organizations. However, as current LLMs are vulnerable to leaking sensitive information, it has proven difficult to apply them in settings where strict access control is necessary. To this end, we design AC-LoRA, an end-to-end system for access control-aware corporate LLM chatbots that maintains a strong information isolation guarantee. AC-LoRA maintains separate LoRA adapters for permissioned datasets, along with the document embedding they are finetuned on. AC-LoRA retrieves a precise set of LoRA adapters based on the similarity score with the user query and their permission. This similarity score is later used to merge the responses if more than one LoRA is retrieved, without requiring any additional training for LoRA routing. We provide an end-to-end prototype of AC-LoRA, evaluate it on two datasets, and show that AC-LoRA matches or even exceeds the performance of state-of-the-art LoRA mixing techniques while providing strong isolation guarantees. Furthermore, we show that AC-LoRA design can be directly applied to different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18169v5">KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Serving LLMs with a cluster of GPUs is common nowadays, where the serving system must meet strict latency SLOs required by applications. However, the stateful nature of LLM serving requires maintaining huge states (i.e., KVCache) in limited GPU memory. Under spikes in real-world workloads, GPU memory can be easily throttled, leading to orders of magnitude higher response latency due to queuing introduced by waiting for KVCache to be reclaimed. Prior KVCache-centric approaches handle load throttling by dropping, migrating, or swapping KVCache. These methods fail to release sufficient memory quickly with requests still queued. This paper proposes the first parameter-centric approach to handling throttling by selectively dropping replicated parameters to instantly free memory for requests, based on an unnoticed observation that model parameters are commonly replicated across GPUs for serving LLMs. With additional memory, all requests can be served with a larger batch without queuing. To make the parameter-centric approach correct and efficient, we cooperatively execute requests on GPUs with a complete copy of parameters using pipeline parallelism, and derive an appropriate drop plan without unnecessary cooperation. We also design techniques to minimize the performance overhead due to pipeline parallelism with the execution patterns of requests under drop. Evaluations show that {\sys} reduces the tail TTFT of requests under throttling by up to 72.2 times compared to the state-of-the-art systems including Llumnix, vLLM and InferCept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12845v4">ExLLM: Experience-Enhanced LLM Optimization for Molecular Design and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, under review
    </div>
    <details class="paper-abstract">
      Molecular design involves an enormous and irregular search space, where traditional optimizers such as Bayesian optimization, genetic algorithms, and generative models struggle to leverage expert knowledge or handle complex feedback. Recently, LLMs have been used as optimizers, achieving promising results on benchmarks such as PMO. However, existing approaches rely only on prompting or extra training, without mechanisms to handle complex feedback or maintain scalable memory. In particular, the common practice of appending or summarizing experiences at every query leads to redundancy, degraded exploration, and ultimately poor final outcomes under large-scale iterative search. We introduce ExLLM (Experience-Enhanced LLM optimization), an LLM-as-optimizer framework with three components: (1) a compact, evolving experience snippet tailored to large discrete spaces that distills non-redundant cues and improves convergence at low cost; (2) a simple yet effective k-offspring scheme that widens exploration per call and reduces orchestration cost; and (3) a lightweight feedback adapter that normalizes objectives for selection while formatting constraints and expert hints for iteration. ExLLM sets new state-of-the-art results on PMO and generalizes strongly in our setup, it sets records on circle packing and stellarator design, and yields consistent gains across additional domains requiring only a task-description template and evaluation functions to transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06780v1">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06774v1">Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Neuro-symbolic NLP methods aim to leverage the complementary strengths of large language models and formal logical solvers. However, current approaches are mostly static in nature, i.e., the integration of a target solver is predetermined at design time, hindering the ability to employ diverse formal inference strategies. To address this, we introduce an adaptive, multi-paradigm, neuro-symbolic inference framework that: (1) automatically identifies formal reasoning strategies from problems expressed in natural language; and (2) dynamically selects and applies specialized formal logical solvers via autoformalization interfaces. Extensive experiments on individual and multi-paradigm reasoning tasks support the following conclusions: LLMs are effective at predicting the necessary formal reasoning strategies with an accuracy above 90 percent. This enables flexible integration with formal logical solvers, resulting in our framework outperforming competing baselines by 27 percent and 6 percent compared to GPT-4o and DeepSeek-V3.1, respectively. Moreover, adaptive reasoning can even positively impact pure LLM methods, yielding gains of 10, 5, and 6 percent on zero-shot, CoT, and symbolic CoT settings with GPT-4o. Finally, although smaller models struggle with adaptive neuro-symbolic reasoning, post-training offers a viable path to improvement. Overall, this work establishes the foundations for adaptive LLM-symbolic reasoning, offering a path forward for unifying material and formal inferences on heterogeneous reasoning challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06750v1">Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Reasoning Models (LRMs) excel in structured tasks by emulating deliberate human reasoning but often suffer from overthinking, degrading performance and wasting resources. One possible baseline is to deploy both LLM and LRM, then route input by predicting whether it requires reasoning and may cause overthinking. However, deploying multiple models can be costly or impractical. We propose a superposed deployment strategy with a lightweight, training-free regulation to optimize inference by switching one model on and off. Instead of routing, we selectively unlearn from LRM at inference, scaling down computation while preserving reasoning. By analyzing the cumulative energy of singular values, we identify optimal low-rank projections to adjust reasoning just right.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06747v1">TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06743v1">Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025
    </div>
    <details class="paper-abstract">
      Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15098v2">TextMine: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMine: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction in the HMA domain. TextMine structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we created the dataset in collaboration with Cambodian Mine Action Center (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06732v1">Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: https://github.com/glad-lab/RAF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06730v1">PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Current evaluations of sentence embedding models typically rely on static test beds such as the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported performance and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in semantic textual similarity gold ratings, we show that LLMs generate token-diverse but semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs and we extended our experiments to 3 multilingual datasets covering 10 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06727v1">Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06718v1">LLM Company Policies and Policy Implications in Software Organizations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at IEEE Software Special Issue on AIware in the Foundation Models Era
    </div>
    <details class="paper-abstract">
      The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04152v2">Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06708v1">AISysRev -- LLM-based Tool for Title-abstract Screening</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human reviewers.We conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: https://www.youtube.com/watch?v=jVbEj4Y4tQI Tool: https://github.com/EvoTestOps/AISysRev
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25033v2">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06695v1">Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06674v1">Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables
    </div>
    <details class="paper-abstract">
      We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05154v2">Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v3">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06663v1">Automated Discovery of Test Oracles for Database Management Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Since 2020, automated testing for Database Management Systems (DBMSs) has flourished, uncovering hundreds of bugs in widely-used systems. A cornerstone of these techniques is test oracle, which typically implements a mechanism to generate equivalent query pairs, thereby identifying bugs by checking the consistency between their results. However, while applying these oracles can be automated, their design remains a fundamentally manual endeavor. This paper explores the use of large language models (LLMs) to automate the discovery and instantiation of test oracles, addressing a long-standing bottleneck towards fully automated DBMS testing. Although LLMs demonstrate impressive creativity, they are prone to hallucinations that can produce numerous false positive bug reports. Furthermore, their significant monetary cost and latency mean that LLM invocations should be limited to ensure that bug detection is efficient and economical. To this end, we introduce Argus, a novel framework built upon the core concept of the Constrained Abstract Query - a SQL skeleton containing placeholders and their associated instantiation conditions (e.g., requiring a placeholder to be filled by a boolean column). Argus uses LLMs to generate pairs of these skeletons that are asserted to be semantically equivalent. This equivalence is then formally proven using a SQL equivalence solver to ensure soundness. Finally, the placeholders within the verified skeletons are instantiated with concrete, reusable SQL snippets that are also synthesized by LLMs to efficiently produce complex test cases. We implemented Argus and evaluated it on five extensively tested DBMSs, discovering 40 previously unknown bugs, 35 of which are logic bugs, with 36 confirmed and 26 already fixed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06658v1">Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at SIGIR-AP 2025
    </div>
    <details class="paper-abstract">
      Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable. We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06657v1">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 RecSys 2025 Industry Track
    </div>
    <details class="paper-abstract">
      This paper presents a case study on deploying Large Language Models (LLMs) as an advanced "annotation" mechanism to achieve nuanced content understanding (e.g., discerning content "vibe") at scale within a large-scale industrial short-form video recommendation system. Traditional machine learning classifiers for content understanding face protracted development cycles and a lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses these by significantly shortening development times and enabling the annotation of subtle attributes. This work details an end-to-end workflow encompassing: (1) iterative definition and robust evaluation of target attributes, refined by offline metrics and online A/B testing; (2) scalable offline bulk annotation of video corpora using LLMs with multimodal features, optimized inference, and knowledge distillation for broad application; and (3) integration of these rich annotations into the online recommendation serving system, for example, through personalized restrict retrieval. Experimental results demonstrate the efficacy of this approach, with LLMs outperforming human raters in offline annotation quality for nuanced attributes and yielding significant improvements of user participation and satisfied consumption in online A/B tests. The study provides insights into designing and scaling production-level LLM pipelines for rich content evaluation, highlighting the adaptability and benefits of LLM-generated nuanced understanding for enhancing content discovery, user satisfaction, and the overall effectiveness of modern recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02053v2">ProCut: LLM Prompt Compression via Attribution Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18665v3">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 this paper is under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04072v2">Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and an up to 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v4">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME , and the project page link is https://sylvain-wei.github.io/TIME/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23580v2">LLM Hallucination Detection: HSAD</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Although Large Language Models have demonstrated powerful capabilities in a wide range of tasks such as language understanding and code generation, the frequent occurrence of hallucinations during the generation process has become a significant impediment to their deployment in critical application scenarios. Current mainstream hallucination detection methods rely on factual consistency verification or static hidden layer features. The former is constrained by the scope of knowledge coverage, while the latter struggles to capture reasoning biases during the inference process. To address these issues, and inspired by signal analysis methods in cognitive neuroscience, this paper proposes a hallucination detection method based on the frequency-domain analysis of hidden layer temporal signals, named HSAD (\textbf{H}idden \textbf{S}ignal \textbf{A}nalysis-based \textbf{D}etection). First, by treating the LLM's reasoning process as a cognitive journey that unfolds over time, we propose modeling and simulating the human process of signal perception and discrimination in a deception-detection scenario through hidden layer temporal signals. Next, The Fast Fourier Transform is applied to map these temporal signals into the frequency domain to construct spectral features, which are used to capture anomalies that arise during the reasoning process; analysis experiments on these spectral features have proven the effectiveness of this approach. Finally, a hallucination detection algorithm is designed based on these spectral features to identify hallucinations in the generated content. By effectively combining the modeling of the reasoning process with frequency-domain feature extraction, the HSAD method overcomes the limitations of existing approaches in terms of knowledge coverage and the detection of reasoning biases, demonstrating higher detection accuracy and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02418v2">BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about pop-up banner closure. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06001v1">Exploring Gaps in the APS: Direct Minimal Pair Analysis in LLM Syntactic Assessments</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at the https://brigap-workshop.github.io/ Information to be updated after publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies probing the Argument from the Poverty of the Stimulus (APS) have applied Large Language Models (LLMs) to test the learnability of complex syntax through surprisal-based metrics. However, divergent conclusions raise questions concerning the insights these metrics offer. While Wilcox et al. (2024) used direct minimal pair comparisons (the "wh-effect") to demonstrate that models successfully generalise knowledge of filler-gap dependencies, Lan et al. (2024) used a Difference-in-Differences (DiD) metric and found that models largely fail on parasitic gaps (PGs). This paper argues that the direct minimal pair approach offers greater diagnostic transparency. We demonstrate this by generating a full 8-permutation paradigm of refined PG stimuli and evaluating the GPT-2 model used in previous studies with a systematic Wilcox-style wh-effect analysis. Our results show that GPT-2 succeeds across all four tested conditions, indicating robust knowledge of filler-gap licensing principles even in complex PG environments. This finding, which contrasts with the more ambiguous results from DiD-style metrics, suggests that the choice of evaluation metric is critical for assessing an LLM's syntactic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05987v1">Sample Smart, Not Hard: Correctness-First Decoding for Better Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to complex tasks that require extended reasoning. In such settings, models often benefit from diverse chains-of-thought to arrive at multiple candidate solutions. This requires two competing objectives: to inject enough stochasticity to explore multiple reasoning chains, and to ensure sufficient accuracy and quality in each path. Existing works pursue the first objective by increasing exploration at highly uncertain steps with higher temperature or larger candidate token sets, while others improve reliability by rejecting samples with low confidence post-generation, implying that low confidence correlates with low answer quality. These two lines of thought are in conflict, as they conflate different sources of uncertainty. To resolve this, we argue that the decoding rule should be calibrated by correctness, not confidence alone. We should sample from tokens with higher estimated correctness, and reduce sampling where expected correctness is low. We propose simple strategies that achieve this goal: Greedy-Threshold makes sampling greedy at very low confidence steps. Calibrated-TopK and Calibrated-epsilon set truncation threshold based on estimated rank-wise correctness. Together, our findings challenge prevailing heuristics about decoding under uncertainty and show gains across math and general reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05950v1">Training-Free Time Series Classification via In-Context Reasoning with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 8 pages main content, 12 pages total including appendix, 1 figure
    </div>
    <details class="paper-abstract">
      Time series classification (TSC) spans diverse application scenarios, yet labeled data are often scarce, making task-specific training costly and inflexible. Recent reasoning-oriented large language models (LLMs) show promise in understanding temporal patterns, but purely zero-shot usage remains suboptimal. We propose FETA, a multi-agent framework for training-free TSC via exemplar-based in-context reasoning. FETA decomposes a multivariate series into channel-wise subproblems, retrieves a few structurally similar labeled examples for each channel, and leverages a reasoning LLM to compare the query against these exemplars, producing channel-level labels with self-assessed confidences; a confidence-weighted aggregator then fuses all channel decisions. This design eliminates the need for pretraining or fine-tuning, improves efficiency by pruning irrelevant channels and controlling input length, and enhances interpretability through exemplar grounding and confidence estimation. On nine challenging UEA datasets, FETA achieves strong accuracy under a fully training-free setting, surpassing multiple trained baselines. These results demonstrate that a multi-agent in-context reasoning framework can transform LLMs into competitive, plug-and-play TSC solvers without any parameter training. The code is available at https://github.com/SongyuanSui/FETATSC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v3">Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Published in IEEE Access
    </div>
    <details class="paper-abstract">
      Recent studies have investigated whether large language models (LLMs) can support obscured communication, which is characterized by core aspects such as inferring subtext and evading suspicions. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two limitations with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these limitations, we propose a microscopic and systematic approach to the investigation. Specifically, we introduce six fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05935v1">LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transparent Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      High-dimensional data remains a pervasive challenge in machine learning, often undermining model interpretability and computational efficiency. While Large Language Models (LLMs) have shown promise for dimensionality reduction through feature selection, existing LLM-based approaches frequently lack structured reasoning and transparent justification for their decisions. This paper introduces LLM-FS-Agent, a novel multi-agent architecture designed for interpretable and robust feature selection. The system orchestrates a deliberative "debate" among multiple LLM agents, each assigned a specific role, enabling collective evaluation of feature relevance and generation of detailed justifications. We evaluate LLM-FS-Agent in the cybersecurity domain using the CIC-DIAD 2024 IoT intrusion detection dataset and compare its performance against strong baselines, including LLM-Select and traditional methods such as PCA. Experimental results demonstrate that LLM-FS-Agent consistently achieves superior or comparable classification performance while reducing downstream training time by an average of 46% (statistically significant improvement, p = 0.028 for XGBoost). These findings highlight that the proposed deliberative architecture enhances both decision transparency and computational efficiency, establishing LLM-FS-Agent as a practical and reliable solution for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05909v1">Optimizing for Persuasion Improves LLM Generalization: Evidence from Quality-Diversity Evolution of Debate Strategies</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Open-source code available at https://github.com/flowersteam/llm_persuasion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) optimized to output truthful answers often overfit, producing brittle reasoning that fails to generalize. While persuasion-based optimization has shown promise in debate settings, it has not been systematically compared against mainstream truth-based approaches. We introduce DebateQD, a minimal Quality-Diversity (QD) evolutionary algorithm that evolves diverse debate strategies across different categories (rationality, authority, emotional appeal, etc.) through tournament-style competitions where two LLMs debate while a third judges. Unlike previously proposed methods that require a population of LLMs, our approach maintains diversity of opponents through prompt-based strategies within a single LLM architecture, making it more accessible for experiments while preserving the key benefits of population-based optimization. In contrast to prior work, we explicitly isolate the role of the optimization objective by fixing the debate protocol and swapping only the fitness function: persuasion rewards strategies that convince the judge irrespective of truth, whereas truth rewards collaborative correctness. Across three model scales (7B, 32B, 72B parameters) and multiple dataset sizes from the QuALITY benchmark, persuasion-optimized strategies achieve up to 13.94% smaller train-test generalization gaps, while matching or exceeding truth optimization's test performance. These results provide the first controlled evidence that competitive pressure to persuade, rather than seek the truth collaboratively, fosters more transferable reasoning skills, offering a promising path for improving LLM generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v2">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v3">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05869v1">The fragility of "cultural tendencies" in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      In a recent study, Lu, Song, and Zhang (2025) (LSZ) propose that large language models (LLMs), when prompted in different languages, display culturally specific tendencies. They report that the two models (i.e., GPT and ERNIE) respond in more interdependent and holistic ways when prompted in Chinese, and more independent and analytic ways when prompted in English. LSZ attribute these differences to deep-seated cultural patterns in the models, claiming that prompt language alone can induce substantial cultural shifts. While we acknowledge the empirical patterns they observed, we find their experiments, methods, and interpretations problematic. In this paper, we critically re-evaluate the methodology, theoretical framing, and conclusions of LSZ. We argue that the reported "cultural tendencies" are not stable traits but fragile artifacts of specific models and task design. To test this, we conducted targeted replications using a broader set of LLMs and a larger number of test items. Our results show that prompt language has minimal effect on outputs, challenging LSZ's claim that these models encode grounded cultural beliefs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05864v1">Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19056v2">An Embarrassingly Simple Defense Against LLM Abliteration Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 preprint - under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14146v3">MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05758v1">EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Under review for ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00927v3">Text Clustering as Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Text clustering serves as a fundamental technique for organizing and interpreting unstructured textual data, particularly in contexts where manual annotation is prohibitively costly. With the rapid advancement of Large Language Models (LLMs) and their demonstrated effectiveness across a broad spectrum of NLP tasks, an emerging body of research has begun to explore their potential in the domain of text clustering. However, existing LLM-based approaches still rely on fine-tuned embedding models and sophisticated similarity metrics, rendering them computationally intensive and necessitating domain-specific adaptation. To address these limitations, we propose a novel framework that reframes text clustering as a classification task by harnessing the in-context learning capabilities of LLMs. Our framework eliminates the need for fine-tuning embedding models or intricate clustering algorithms. It comprises two key steps: first, the LLM is prompted to generate a set of candidate labels based on the dataset and then merges semantically similar labels; second, it assigns the most appropriate label to each text sample. By leveraging the advanced natural language understanding and generalization capabilities of LLMs, the proposed approach enables effective clustering with minimal human intervention. Experimental results on diverse datasets demonstrate that our framework achieves comparable or superior performance to state-of-the-art embedding-based clustering techniques, while significantly reducing computational complexity and resource requirements. These findings underscore the transformative potential of LLMs in simplifying and enhancing text clustering tasks. We make our code available to the public for utilization at https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM. We also provide the supplementary Appendix within the repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05748v1">Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05742v1">Vipera: Blending Visual and LLM-Driven Guidance for Systematic Auditing of Text-to-Image Generative AI</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Despite their increasing capabilities, text-to-image generative AI systems are known to produce biased, offensive, and otherwise problematic outputs. While recent advancements have supported testing and auditing of generative AI, existing auditing methods still face challenges in supporting effectively explore the vast space of AI-generated outputs in a structured way. To address this gap, we conducted formative studies with five AI auditors and synthesized five design goals for supporting systematic AI audits. Based on these insights, we developed Vipera, an interactive auditing interface that employs multiple visual cues including a scene graph to facilitate image sensemaking and inspire auditors to explore and hierarchically organize the auditing criteria. Additionally, Vipera leverages LLM-powered suggestions to facilitate exploration of unexplored auditing directions. Through a controlled experiment with 24 participants experienced in AI auditing, we demonstrate Vipera's effectiveness in helping auditors navigate large AI output spaces and organize their analyses while engaging with diverse criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05733v1">Syn-Diag: An LLM-based Synergistic Framework for Generalizable Few-shot Fault Diagnosis on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Industrial fault diagnosis faces the dual challenges of data scarcity and the difficulty of deploying large AI models in resource-constrained environments. This paper introduces Syn-Diag, a novel cloud-edge synergistic framework that leverages Large Language Models to overcome these limitations in few-shot fault diagnosis. Syn-Diag is built on a three-tiered mechanism: 1) Visual-Semantic Synergy, which aligns signal features with the LLM's semantic space through cross-modal pre-training; 2) Content-Aware Reasoning, which dynamically constructs contextual prompts to enhance diagnostic accuracy with limited samples; and 3) Cloud-Edge Synergy, which uses knowledge distillation to create a lightweight, efficient edge model capable of online updates via a shared decision space. Extensive experiments on six datasets covering different CWRU and SEU working conditions show that Syn-Diag significantly outperforms existing methods, especially in 1-shot and cross-condition scenarios. The edge model achieves performance comparable to the cloud version while reducing model size by 83% and latency by 50%, offering a practical, robust, and deployable paradigm for modern intelligent diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04093v2">Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14913v3">Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05709v1">Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05703v1">Primal-Dual Direct Preference Optimization for Constrained LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The widespread application of Large Language Models (LLMs) imposes increasing demands on safety, such as reducing harmful content and fake information, and avoiding certain forbidden tokens due to rules and laws. While there have been several recent works studying safe alignment of LLMs, these works either require the training of reward and cost models and incur high memory and computational costs, or need prior knowledge about the optimal solution. Motivated by this fact, we study the problem of constrained alignment in LLMs, i.e., maximizing the output reward while restricting the cost due to potentially unsafe content to stay below a threshold. For this problem, we propose a novel primal-dual DPO approach, which first trains a model using standard DPO on reward preference data to provide reward information, and then adopts a rearranged Lagrangian DPO objective utilizing the provided reward information to fine-tune LLMs on cost preference data. Our approach significantly reduces memory and computational costs, and does not require extra prior knowledge. Moreover, we establish rigorous theoretical guarantees on the suboptimality and constraint violation of the output policy. We also extend our approach to an online data setting by incorporating exploration bonuses, which enables our approach to explore uncovered prompt-response space, and then provide theoretical results that get rid of the dependence on preference data coverage. Experimental results on the widely-used preference dataset PKU-SafeRLHF demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05632v1">From Principles to Practice: A Systematic Study of LLM Serving on Multi-core NPUs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), the demand for high-performance LLM inference services continues to grow. To meet this demand, a growing number of AI accelerators have been proposed, such as Google TPU, Huawei NPU, Graphcore IPU, and Cerebras WSE, etc. Most of these accelerators adopt multi-core architectures to achieve enhanced scalability, but lack the flexibility of SIMT architectures. Therefore, without careful configuration of the hardware architecture, as well as deliberate design of tensor parallelism and core placement strategies, computational resources may be underutilized, resulting in suboptimal inference performance. To address these challenges, we first present a multi-level simulation framework with both transaction-level and performance-model-based simulation for multi-core NPUs. Using this simulator, we conduct a systematic analysis and further propose the optimal solutions for tensor parallelism strategies, core placement policies, memory management methods, as well as the selection between PD-disaggregation and PD-fusion on multi-core NPUs. We conduct comprehensive experiments on representative LLMs and various NPU configurations. The evaluation results demonstrate that, our solution can achieve 1.32x-6.03x speedup compared to SOTA designs for multi-core NPUs across different hardware configurations. As for LLM serving, our work offers guidance on designing optimal hardware architectures and serving strategies for multi-core NPUs across various LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18851v2">Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Identifying LLM-generated code through watermarking poses a challenge in preserving functional correctness. Previous methods rely on the assumption that watermarking high-entropy tokens effectively maintains output quality. Our analysis reveals a fundamental limitation of this assumption: syntax-critical tokens such as keywords often exhibit the highest entropy, making existing approaches vulnerable to logic corruption. We present STONE, a syntax-aware watermarking method that embeds watermarks only in non-syntactic tokens and preserves code integrity. For its rigorous assessment, we also introduce STEM, a comprehensive framework that balances three critical dimensions: correctness, detectability, and imperceptibility. Across Python, C++, and Java, STONE preserves correctness, sustains strong detectability, and achieves balanced performance with minimal overhead. Our implementation is available at https://anonymous.4open.science/r/STONE-watermarking-AB4B/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05605v1">AutoPentester: An LLM Agent-based Framework for Automated Pentesting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 IEEE TrustCom 2025 10 pages
    </div>
    <details class="paper-abstract">
      Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05598v1">AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11182v3">A Middle Path for On-Premises LLM Deployment: Preserving Privacy Without Sacrificing Model Confidentiality</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 8 pages for main content of the paper
    </div>
    <details class="paper-abstract">
      Privacy-sensitive users require deploying large language models (LLMs) within their own infrastructure (on-premises) to safeguard private data and enable customization. However, vulnerabilities in local environments can lead to unauthorized access and potential model theft. To address this, prior research on small models has explored securing only the output layer within hardware-secured devices to balance model confidentiality and customization. Yet this approach fails to protect LLMs effectively. In this paper, we discover that (1) query-based distillation attacks targeting the secured top layer can produce a functionally equivalent replica of the victim model; (2) securing the same number of layers, bottom layers before a transition layer provide stronger protection against distillation attacks than top layers, with comparable effects on customization performance; and (3) the number of secured layers creates a trade-off between protection and customization flexibility. Based on these insights, we propose SOLID, a novel deployment framework that secures a few bottom layers in a secure environment and introduces an efficient metric to optimize the trade-off by determining the ideal number of hidden layers. Extensive experiments on five models (1.3B to 70B parameters) demonstrate that SOLID outperforms baselines, achieving a better balance between protection and downstream customization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00965v2">FLEx: Personalized Federated Learning for Mixture-of-Experts LLMs via Expert Grafting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Federated instruction tuning of large language models (LLMs) is challenged by significant data heterogeneity across clients, demanding robust personalization. The Mixture of Experts (MoE) architecture, where experts can specialize in distinct data patterns, presents a natural architectural solution to this challenge. The inherent sparsity of the MoE architecture, achieved by selectively activating experts, poses a significant challenge to its integration with federated learning (FL). Conventional FL frameworks, designed for dense models, naively aggregate all expert parameters irrespective of their local activation patterns. This naive approach not only undermines MoE's dynamic sparsity but also risks corrupting the world knowledge within pretrained experts. To address this, we propose FLEx (Federated LLMs with Personalized Experts), a novel framework that leverages pretrained MoE-based LLMs for efficient personalization. By aggregating only the shared non-expert parameters, FLEx significantly reduces communication overhead and preserves the world knowledge stored within the frozen pretrained experts. For personalization, we introduce a novel expert grafting mechanism that leverages dynamic sparsity to construct a client-specific expert from selected components of pretrained experts, tailored to local data. This grafted expert is then fine-tuned locally alongside the gating mechanism. This joint training enables the model to learn when to leverage the shared knowledge from frozen experts and when to employ the personalized one. Evaluations on diverse, non-IID instruction tuning datasets show that FLEx consistently outperforms federated baselines on average, while demonstrating strong knowledge preservation on the knowledge-driven benchmark MMLU. Our code is available at \href{https://anonymous.4open.science/r/FLEx-8F12}{\texttt{https://anonymous.4open.science/r/FLEx-8F12}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05582v1">(Token-Level) \textbf{InfoRMIA}: Stronger Membership Inference and Memorization Assessment for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency. In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05577v1">Mission Impossible: Feedback-Guided Dynamic Interactive Planning for Improving Reasoning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Recent advancements in language agents have led to significant improvements in multi-hop reasoning tasks. However, existing approaches often struggle with handling open-domain problems, which require massive information retrieval due to their reliance on a fixed sequence of actions. To address this, we propose Feedback-Guided Dynamic Interactive Planning (FGDIP), a novel framework tailored to enhance reasoning in LLMs by utilizing dynamic and adaptive strategies for information exploration in open-domain multi-hop reasoning tasks. Our approach begins by identifying key entities relevant to the problem, which serve as the initial nodes in the reasoning process. From these initial nodes, we then generate reasoning child nodes with the process being refined through a combination of historical error analysis and real-time feedback, which allows the framework to dynamically adjust and optimize its reasoning strategies. By integrating depth-first search with an innovative node generation technique, our framework adapts based on both prior error paths and concurrently generated nodes at the same hierarchical level. This dynamic strategy effectively expands the search space while ensuring the reasoning process systematically converges toward accurate solutions. Experimental results show that FGDIP achieved up to 54.47% F1 score on the HotpotQA dataset and 70.05% on the StrategyQA dataset, surpassing the best baseline by 5.03% and 7.25% respectively, highlighting its versatility and potential to enhance language agents in multi-hop reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08833v2">Position: The Pitfalls of Over-Alignment: Overly Caution Health-Related Responses From LLMs are Unethical and Dangerous</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are usually aligned with "human values/preferences" to prevent harmful output. Discussions around the alignment of Large Language Models (LLMs) generally focus on preventing harmful outputs. However, in this paper, we argue that in health-related queries, over-alignment-leading to overly cautious responses-can itself be harmful, especially for people with anxiety and obsessive-compulsive disorder (OCD). This is not only unethical but also dangerous to the user, both mentally and physically. We also showed qualitative results that some LLMs exhibit varying degrees of alignment. Finally, we call for the development of LLMs with stronger reasoning capabilities that provide more tailored and nuanced responses to health queries. Warning: This paper contains materials that could trigger health anxiety or OCD. Dataset and full results can be found in https://github.com/weathon/over-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25795v2">Assessing Algorithmic Bias in Language-Based Depression Detection: A Comparison of DNN and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 7 pages, 1 figure. This paper has been accepted to the IEEE-EMBS International Conference on Biomedical and Health Informatics (BHI 2025), Georgia Institute of Technology, Atlanta, Georgia, October 26-29, 2025
    </div>
    <details class="paper-abstract">
      This paper investigates algorithmic bias in language-based models for automated depression detection, focusing on socio-demographic disparities related to gender and race/ethnicity. Models trained using deep neural networks (DNN) based embeddings are compared to few-shot learning approaches with large language models (LLMs), evaluating both performance and fairness on clinical interview transcripts from the Distress Analysis Interview Corpus/Wizard-of-Oz (DAIC-WOZ). To mitigate bias, fairness-aware loss functions are applied to DNN-based models, while in-context learning with varied prompt framing and shot counts is explored for LLMs. Results indicate that LLMs outperform DNN-based models in depression classification, particularly for underrepresented groups such as Hispanic participants. LLMs also exhibit reduced gender bias compared to DNN-based embeddings, though racial disparities persist. Among fairness-aware techniques for mitigating bias in DNN-based embeddings, the worst-group loss, which is designed to minimize loss for the worst-performing demographic group, achieves a better balance between performance and fairness. In contrast, the fairness-regularized loss minimizes loss across all groups but performs less effectively. In LLMs, guided prompting with ethical framing helps mitigate gender bias in the 1-shot setting. However, increasing the number of shots does not lead to further reductions in disparities. For race/ethnicity, neither prompting strategy nor increasing $N$ in $N$-shot learning effectively reduces disparities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05544v1">Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23058v3">Risk Profiling and Modulation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05520v1">CAM: A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are confronted with overwhelming information volume when comprehending long-form documents. This challenge raises the imperative of a cohesive memory module, which can elevate vanilla LLMs into autonomous reading agents. Despite the emergence of some heuristic approaches, a systematic design principle remains absent. To fill this void, we draw inspiration from Jean Piaget's Constructivist Theory, illuminating three traits of the agentic memory -- structured schemata, flexible assimilation, and dynamic accommodation. This blueprint forges a clear path toward a more robust and efficient memory system for LLM-based reading comprehension. To this end, we develop CAM, a prototype implementation of Constructivist Agentic Memory that simultaneously embodies the structurality, flexibility, and dynamicity. At its core, CAM is endowed with an incremental overlapping clustering algorithm for structured memory development, supporting both coherent hierarchical summarization and online batch integration. During inference, CAM adaptively explores the memory structure to activate query-relevant information for contextual response, akin to the human associative process. Compared to existing approaches, our design demonstrates dual advantages in both performance and efficiency across diverse long-text reading comprehension tasks, including question answering, query-based summarization, and claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13360v2">What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Prompt underspecification is a common challenge when interacting with LLMs. In this paper, we present an in-depth analysis of this problem, showing that while LLMs can often infer unspecified requirements by default (41.1%), such behavior is fragile: Under-specified prompts are 2x as likely to regress across model or prompt changes, sometimes with accuracy drops exceeding 20%. This instability makes it difficult to reliably build LLM applications. Moreover, simply specifying all requirements does not consistently help, as models have limited instruction-following ability and requirements can conflict. Standard prompt optimizers likewise provide little benefit. To address these issues, we propose requirements-aware prompt optimization mechanisms that improve performance by 4.8% on average over baselines. We further advocate for a systematic process of proactive requirements discovery, evaluation, and monitoring to better manage prompt underspecification in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05497v1">Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with Mixture of Experts (MoE) architectures achieve remarkable performance improvements, but their random expert selection mechanism introduces significant data movement overhead that becomes the dominant bottleneck in multi-unit serving systems. To forecast the patterns underlying this data movement, we conduct comprehensive data-movement-centric profiling across three state-of-the-art large-scale MoE models (200B- 671B) using over 24,000 requests spanning diverse workloads. With the resulting 150GB+ trace files, we perform systematic analysis from both temporal and spatial perspectives and distill six key insights to guide the design of diverse future serving systems. Taking wafer-scale GPUs as a case study, we demonstrate that minor architectural modifications leveraging our insights achieve substantial performance gains, delivering 6.3X and 4.0X average speedups on DeepSeek V3 and Qwen3, respectively. Our work provides the first comprehensive data-centric analysis of MoE models at scale. Our profiling traces and analysis results are publicly available at {https://huggingface.co/datasets/core12345/MoE_expert_selection_trace. We will also release our simulation framework shortly to facilitate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05484v1">Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming part of tools used by children; however, existing benchmarks fail to capture how these models manage language, reasoning, and safety needs that are specific to various ages. We present ChildSafe, a benchmark that evaluates LLM safety through simulated child agents that embody four developmental stages. These agents, grounded in developmental psychology, enable a systematic study of child safety without the ethical implications of involving real children. ChildSafe assesses responses across nine safety dimensions (including privacy, misinformation, and emotional support) using age-weighted scoring in both sensitive and neutral contexts. Multi-turn experiments with multiple LLMs uncover consistent vulnerabilities that vary by simulated age, exposing shortcomings in existing alignment practices. By releasing agent templates, evaluation protocols, and an experimental corpus, we provide a reproducible framework for age-aware safety research. We encourage the community to expand this work with real child-centered data and studies, advancing the development of LLMs that are genuinely safe and developmentally aligned.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05480v1">Vul-R2: A Reasoning LLM for Automated Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 13 pages, 8 figures. This paper is accepted by ASE 2025
    </div>
    <details class="paper-abstract">
      The exponential increase in software vulnerabilities has created an urgent need for automatic vulnerability repair (AVR) solutions. Recent research has formulated AVR as a sequence generation problem and has leveraged large language models (LLMs) to address this problem. Typically, these approaches prompt or fine-tune LLMs to generate repairs for vulnerabilities directly. Although these methods show state-of-the-art performance, they face the following challenges: (1) Lack of high-quality, vulnerability-related reasoning data. Current approaches primarily rely on foundation models that mainly encode general programming knowledge. Without vulnerability-related reasoning data, they tend to fail to capture the diverse vulnerability repair patterns. (2) Hard to verify the intermediate vulnerability repair process during LLM training. Existing reinforcement learning methods often leverage intermediate execution feedback from the environment (e.g., sandbox-based execution results) to guide reinforcement learning training. In contrast, the vulnerability repair process generally lacks such intermediate, verifiable feedback, which poses additional challenges for model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18356v2">The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 MRL Workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) still struggle across tasks outside of high-resource languages. In this work, we investigate cross-lingual transfer to lower-resource languages where task-specific post-training data is scarce. Building on prior work, we first validate that the subsets of model parameters that matter most for mathematical reasoning and multilingual capabilities are distinctly non-overlapping. To exploit this implicit separability between task and target language parameterization, we develop and analyze numerous modular frameworks to improve the composition of the two during fine-tuning. These methods generally employ freezing parameters or post hoc model merging to assign math and language improvement to different key parts of the LLM. In the absence of in-language math data, we demonstrate that the modular approaches successfully improve upon baselines across three languages, four models, and two fine-tuning paradigms (full and LoRA). Furthermore, we identify the most consistently successful modular method to be fine-tuning separate language and math experts and model merging via Layer-Swapping, somewhat surprisingly. We offer possible explanations for this result via recent works on the linearity of task vectors. We further explain this by empirically showing that reverting less useful fine-tuning updates after training often outperforms freezing them from the start.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07218v2">LLM Unlearning via Neural Activation Redirection</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The ability to selectively remove knowledge from LLMs is highly desirable. However, existing methods often struggle with balancing unlearning efficacy and retain model utility, and lack controllability at inference time to emulate base model behavior as if it had never seen the unlearned data. In this paper, we propose LUNAR, a novel unlearning method grounded in the Linear Representation Hypothesis and operates by redirecting the representations of unlearned data to activation regions that expresses its inability to answer. We show that contrastive features are not a prerequisite for effective activation redirection, and LUNAR achieves state-of-the-art unlearning performance and superior controllability. Specifically, LUNAR achieves between 2.9x and 11.7x improvement in the combined unlearning efficacy and model utility score (Deviation Score) across various base models and generates coherent, contextually appropriate responses post-unlearning. Moreover, LUNAR effectively reduces parameter updates to a single down-projection matrix, a novel design that significantly enhances efficiency by 20x and robustness. Finally, we demonstrate that LUNAR is robust to white-box adversarial attacks and versatile in real-world scenarios, including handling sequential unlearning requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01213v2">Show or Tell? Modeling the evolution of request-making in Human-LLM conversations</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Designing user-centered LLM systems requires understanding how people use them, but patterns of user behavior are often masked by the variability of queries. In this work, we introduce a new framework to describe request-making that segments user input into request content, roles assigned, query-specific context, and the remaining task-independent expressions. We apply the workflow to create and analyze a dataset of 211k real-world queries based on WildChat. Compared with similar human-human setups, we find significant differences in the language for request-making in the human-LLM scenario. Further, we introduce a novel and essential perspective of diachronic analyses with user expressions, which reveals fundamental and habitual user-LLM interaction patterns beyond individual task completion. We find that query patterns evolve from early ones emphasizing sole requests to combining more context later on, and individual users explore expression patterns but tend to converge with more experience. From there, we propose to understand communal trends of expressions underlying distinct tasks and discuss the preliminary findings. Finally, we discuss the key implications for user studies, computational pragmatics, and LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06477v1">Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Attention sinks and compression valleys have attracted significant attention as two puzzling phenomena in large language models, but have been studied in isolation. In this work, we present a surprising connection between attention sinks and compression valleys, tracing both to the formation of massive activations in the residual stream. We prove theoretically that massive activations necessarily produce representational compression and establish bounds on the resulting entropy reduction. Through experiments across several models (410M-120B parameters), we confirm that when the beginning-of-sequence token develops extreme activation norms in the middle layers, both compression valleys and attention sinks emerge simultaneously. Targeted ablation studies validate our theoretical predictions. This unified view motivates us to propose the Mix-Compress-Refine theory of information flow, as an attempt to explain how LLMs organize their computation in depth by controlling attention and representational compression via massive activations. Specifically, we posit that Transformer-based LLMs process tokens in three distinct phases: (1) broad mixing in the early layers, (2) compressed computation with limited mixing in the middle layers, and (3) selective refinement in the late layers. Our framework helps explain why embedding tasks perform best at intermediate layers, whereas generation tasks benefit from full-depth processing, clarifying differences in task-dependent representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20309v2">Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03540v2">Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) paradigms mitigate this issue by incorporating external knowledge at inference time. However, such methods typically handle knowledge as unstructured text, which reduces retrieval accuracy, hinders compositional reasoning, and amplifies the influence of irrelevant information on the factual consistency of LLM outputs. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external knowledge retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's internal knowledge. The KG is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse Factual QA benchmarks, demonstrating consistent gains in factual accuracy over baselines. Our findings reveal that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06426v1">FinLFQA: Evaluating Attributed Text Generation of LLMs in Financial Long-Form Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently hallucinate to long-form questions, producing plausible yet factually incorrect answers. A common mitigation strategy is to provide attribution to LLM outputs. However, existing benchmarks primarily focus on simple attribution that retrieves supporting textual evidence as references. We argue that in real-world scenarios such as financial applications, attribution goes beyond reference retrieval. We introduce FinLFQA, a benchmark designed to evaluate the ability of LLMs to generate long-form answers to complex financial questions with reliable and nuanced attributions. FinLFQA evaluates three critical aspects of attribution through human annotations: (1) supporting evidence extracted from financial reports, (2) intermediate numerical reasoning steps, and (3) domain-specific financial knowledge that informs the reasoning process. We further provide an automatic evaluation framework covering both answer quality and attribution quality. Through extensive experiments on eight LLMs across multiple attribution-generation paradigms, we find that fine-grained metrics are important to distinguish model capabilities, that end-to-end generation achieves comparable performance to post-hoc approaches, and that iterative refinement only helps when guided by external feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02016v2">Mind the (Belief) Gap: Group Identity in the World of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted to ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Social biases and belief-driven behaviors can significantly impact Large Language Models (LLMs) decisions on several tasks. As LLMs are increasingly used in multi-agent systems for societal simulations, their ability to model fundamental group psychological characteristics remains critical yet under-explored. In this study, we present a multi-agent framework that simulates belief congruence, a classical group psychology theory that plays a crucial role in shaping societal interactions and preferences. Our findings reveal that LLMs exhibit amplified belief congruence compared to humans, across diverse contexts. We further investigate the implications of this behavior on two downstream tasks: (1) misinformation dissemination and (2) LLM learning, finding that belief congruence in LLMs increases misinformation dissemination and impedes learning. To mitigate these negative impacts, we propose strategies inspired by: (1) contact hypothesis, (2) accuracy nudges, and (3) global citizenship framework. Our results show that the best strategies reduce misinformation dissemination by up to 37% and enhance learning by 11%. Bridging social psychology and AI, our work provides insights to navigate real-world interactions using LLMs while addressing belief-driven biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06411v1">Instructional Goal-Aligned Question Generation for Student Evaluation in Virtual Lab Settings: How Closely Do LLMs Actually Align?</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Virtual Labs offer valuable opportunities for hands-on, inquiry-based science learning, yet teachers often struggle to adapt them to fit their instructional goals. Third-party materials may not align with classroom needs, and developing custom resources can be time-consuming and difficult to scale. Recent advances in Large Language Models (LLMs) offer a promising avenue for addressing these limitations. In this paper, we introduce a novel alignment framework for instructional goal-aligned question generation, enabling teachers to leverage LLMs to produce simulation-aligned, pedagogically meaningful questions through natural language interaction. The framework integrates four components: instructional goal understanding via teacher-LLM dialogue, lab understanding via knowledge unit and relationship analysis, a question taxonomy for structuring cognitive and pedagogical intent, and the TELeR taxonomy for controlling prompt detail. Early design choices were informed by a small teacher-assisted case study, while our final evaluation analyzed over 1,100 questions from 19 open-source LLMs. With goal and lab understanding grounding questions in teacher intent and simulation context, the question taxonomy elevates cognitive demand (open-ended formats and relational types raise quality by 0.29-0.39 points), and optimized TELeR prompts enhance format adherence (80% parsability, >90% adherence). Larger models yield the strongest gains: parsability +37.1%, adherence +25.7%, and average quality +0.8 Likert points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06410v1">Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory?</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06378v1">Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Automated interpretability aims to translate large language model (LLM) features into human understandable descriptions. However, these natural language feature descriptions are often vague, inconsistent, and require manual relabeling. In response, we introduce semantic regexes, structured language descriptions of LLM features. By combining primitives that capture linguistic and semantic feature patterns with modifiers for contextualization, composition, and quantification, semantic regexes produce precise and expressive feature descriptions. Across quantitative benchmarks and qualitative analyses, we find that semantic regexes match the accuracy of natural language while yielding more concise and consistent feature descriptions. Moreover, their inherent structure affords new types of analyses, including quantifying feature complexity across layers, scaling automated interpretability from insights into individual features to model-wide patterns. Finally, in user studies, we find that semantic regex descriptions help people build accurate mental models of LLM feature activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06354v1">LLM Bias Detection and Mitigation through the Lens of Desired Distributions</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although prior work on bias mitigation has focused on promoting social equality and demographic parity, less attention has been given to aligning LLM's outputs to desired distributions. For example, we might want to align a model with real-world distributions to support factual grounding. Thus, we define bias as deviation from a desired distribution, which may be an equal or real-world distribution, depending on application goals. We propose a weighted adaptive loss based fine-tuning method that aligns LLM's gender-profession output distribution with the desired distribution, while preserving language modeling capability. Using 3 profession sets -- male-dominated, female-dominated, and gender-balanced -- derived from U.S. labor statistics (2024), we assess both our adaptive method for reflecting reality and a non-adaptive variant for equality. Across three masked language models, bias is observed under both distributions. We achieve near-complete mitigation under equality and 30-75% reduction under real-world settings. Autoregressive LLMs show no bias under equality but notable bias under real-world settings, with the Llama Instruct models (3.2-3B, 3.1-8B) achieving a 50-62% reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06214v1">Stratified GRPO: Handling Structural Heterogeneity in Reinforcement Learning of LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly rely on external tools such as search engines to solve complex, multi-step problems, and reinforcement learning (RL) has become a key paradigm for training them. However, the trajectories of search agents are structurally heterogeneous, where variations in the number, placement, and outcomes of search calls lead to fundamentally different answer directions and reward distributions. Standard policy gradient methods, which use a single global baseline, suffer from what we identify and formalize as cross-stratum bias-an "apples-to-oranges" comparison of heterogeneous trajectories. This cross-stratum bias distorts credit assignment and hinders exploration of complex, multi-step search strategies. To address this, we propose Stratified GRPO, whose central component, Stratified Advantage Normalization (SAN), partitions trajectories into homogeneous strata based on their structural properties and computes advantages locally within each stratum. This ensures that trajectories are evaluated only against their true peers. Our analysis proves that SAN eliminates cross-stratum bias, yields conditionally unbiased unit-variance estimates inside each stratum, and retains the global unbiasedness and unit-variance properties enjoyed by standard normalization, resulting in a more pure and scale-stable learning signal. To improve practical stability under finite-sample regimes, we further linearly blend SAN with the global estimator. Extensive experiments on diverse single-hop and multi-hop question-answering benchmarks demonstrate that Stratified GRPO consistently and substantially outperforms GRPO by up to 11.3 points, achieving higher training rewards, greater training stability, and more effective search policies. These results establish stratification as a principled remedy for structural heterogeneity in RL for LLM search agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v2">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06175v1">VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexible representational capacity across bit-widths, they suffer severe performance degradation at ultra-low bit-widths due to key cache outliers that hinder effective codebook utilization. To address this challenge, we propose VecInfer, a novel VQ method for aggressive KV cache compression while enabling efficient inference. By applying smooth and Hadamard transformations, VecInfer suppresses outliers in the key cache, enabling the codebook to comprehensively cover the original data distribution and thereby reducing quantization difficulty. To facilitate efficient deployment, we design an optimized CUDA kernel that fuses computation with dequantization to minimize memory access overhead. Extensive evaluations demonstrate that VecInfer consistently outperforms existing quantization baselines across both long-context understanding and mathematical reasoning tasks. With only 2-bit quantization, VecInfer achieves performance comparable to full precision, while delivering up to $\mathbf{2.7\times}$ speedup in large-batch self-attention computation and $\mathbf{8.3\times}$ reduction in single-batch end-to-end latency on Llama-3.1-8B with a 196k sequence length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06151v1">LLMs as Policy-Agnostic Teammates: A Case Study in Human Proxy Design for Heterogeneous Agent Teams</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 This is a preprint of a paper presented at the \textit{European Conference on Artificial Intelligence (ECAI 2025)}. It is made publicly available for the benefit of the research community and should be regarded as a preprint rather than a formally reviewed publication
    </div>
    <details class="paper-abstract">
      A critical challenge in modelling Heterogeneous-Agent Teams is training agents to collaborate with teammates whose policies are inaccessible or non-stationary, such as humans. Traditional approaches rely on expensive human-in-the-loop data, which limits scalability. We propose using Large Language Models (LLMs) as policy-agnostic human proxies to generate synthetic data that mimics human decision-making. To evaluate this, we conduct three experiments in a grid-world capture game inspired by Stag Hunt, a game theory paradigm that balances risk and reward. In Experiment 1, we compare decisions from 30 human participants and 2 expert judges with outputs from LLaMA 3.1 and Mixtral 8x22B models. LLMs, prompted with game-state observations and reward structures, align more closely with experts than participants, demonstrating consistency in applying underlying decision criteria. Experiment 2 modifies prompts to induce risk-sensitive strategies (e.g. "be risk averse"). LLM outputs mirror human participants' variability, shifting between risk-averse and risk-seeking behaviours. Finally, Experiment 3 tests LLMs in a dynamic grid-world where the LLM agents generate movement actions. LLMs produce trajectories resembling human participants' paths. While LLMs cannot yet fully replicate human adaptability, their prompt-guided diversity offers a scalable foundation for simulating policy-agnostic teammates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06143v1">RoSE: Round-robin Synthetic Data Evaluation for Selecting LLM Generators without Human Test Sets</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      LLMs are powerful generators of synthetic data, which are used for training smaller, specific models. This is especially valuable for low-resource languages, where human-labelled data is scarce but LLMs can still produce high-quality text. However, LLMs differ in how useful their outputs are for training. Selecting the best LLM as a generator is challenging because extrinsic evaluation requires costly human annotations (which are often unavailable for low-resource languages), while intrinsic metrics correlate poorly with downstream performance. We introduce Round robin Synthetic data Evaluation (RoSE), a proxy metric for selecting the best LLM generator without human test sets. RoSE trains a small model on the outputs of a candidate generator (LLM) and then evaluates it on generated synthetic examples from all other candidate LLMs. The final RoSE score is the mean performance of this small model. Across six LLMs, eleven languages, and three tasks (sentiment, topic, intent), RoSE identifies the optimal generator more often than any other intrinsic heuristics. RoSE outperforms intrinsic heuristics and comes within 0.76 percentage points of the optimal generator baseline. This result is measured in terms of downstream performance, obtained by training a small model on the chosen generator's outputs (optimal vs. proxy metric selected) and evaluating it on human-labelled test data. Additionally, RoSE is the only metric to achieve a positive correlation with performance on human test data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06105v1">Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly shaping how information is created and disseminated, from companies using them to craft persuasive advertisements, to election campaigns optimizing messaging to gain votes, to social media influencers boosting engagement. These settings are inherently competitive, with sellers, candidates, and influencers vying for audience approval, yet it remains poorly understood how competitive feedback loops influence LLM behavior. We show that optimizing LLMs for competitive success can inadvertently drive misalignment. Using simulated environments across these scenarios, we find that, 6.3% increase in sales is accompanied by a 14.0% rise in deceptive marketing; in elections, a 4.9% gain in vote share coincides with 22.3% more disinformation and 12.5% more populist rhetoric; and on social media, a 7.5% engagement boost comes with 188.6% more disinformation and a 16.3% increase in promotion of harmful behaviors. We call this phenomenon Moloch's Bargain for AI--competitive success achieved at the cost of alignment. These misaligned behaviors emerge even when models are explicitly instructed to remain truthful and grounded, revealing the fragility of current alignment safeguards. Our findings highlight how market-driven optimization pressures can systematically erode alignment, creating a race to the bottom, and suggest that safe deployment of AI systems will require stronger governance and carefully designed incentives to prevent competitive dynamics from undermining societal trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06104v1">Explaining Code Risk in OSS: Towards LLM-Generated Fault Prediction Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Open Source Software (OSS) has become a very important and crucial infrastructure worldwide because of the value it provides. OSS typically depends on contributions from developers across diverse backgrounds and levels of experience. Making safe changes, such as fixing a bug or implementing a new feature, can be challenging, especially in object-oriented systems where components are interdependent. Static analysis and defect-prediction tools produce metrics (e.g., complexity,coupling) that flag potentially fault-prone components, but these signals are often hard for contributors new or unfamiliar with the codebase to interpret. Large Language Models (LLMs) have shown strong performance on software engineering tasks such as code summarization and documentation generation. Building on this progress, we investigate whether LLMs can translate fault-prediction metrics into clear, human-readable risk explanations and actionable guidance to help OSS contributors plan and review code modifications. We outline explanation types that an LLM-generated assistant could provide (descriptive, contextual, and actionable explanations). We also outline our next steps to assess usefulness through a task-based study with OSS contributors, comparing metric-only baselines to LLM-generated explanations on decision quality, time-to-completion, and error rates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06096v1">The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06093v1">Classical AI vs. LLMs for Decision-Maker Alignment in Health Insurance Choices</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 15 pages, 3 figures. Accepted at the Twelfth Annual Conference on Advances in Cognitive Systems (ACS 2025)
    </div>
    <details class="paper-abstract">
      As algorithmic decision-makers are increasingly applied to high-stakes domains, AI alignment research has evolved from a focus on universal value alignment to context-specific approaches that account for decision-maker attributes. Prior work on Decision-Maker Alignment (DMA) has explored two primary strategies: (1) classical AI methods integrating case-based reasoning, Bayesian reasoning, and naturalistic decision-making, and (2) large language model (LLM)-based methods leveraging prompt engineering. While both approaches have shown promise in limited domains such as medical triage, their generalizability to novel contexts remains underexplored. In this work, we implement a prior classical AI model and develop an LLM-based algorithmic decision-maker evaluated using a large reasoning model (GPT-5) and a non-reasoning model (GPT-4) with weighted self-consistency under a zero-shot prompting framework, as proposed in recent literature. We evaluate both approaches on a health insurance decision-making dataset annotated for three target decision-makers with varying levels of risk tolerance (0.0, 0.5, 1.0). In the experiments reported herein, classical AI and LLM-based models achieved comparable alignment with attribute-based targets, with classical AI exhibiting slightly better alignment for a moderate risk profile. The dataset and open-source implementation are publicly available at: https://github.com/TeX-Base/ClassicalAIvsLLMsforDMAlignment and https://github.com/Parallax-Advanced-Research/ITM/tree/feature_insurance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06092v1">Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06078v1">Constraint-Aware Route Recommendation from Natural Language via Hierarchical LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Route recommendation aims to provide users with optimal travel plans that satisfy diverse and complex requirements. Classical routing algorithms (e.g., shortest-path and constraint-aware search) are efficient but assume structured inputs and fixed objectives, limiting adaptability to natural-language queries. Recent LLM-based approaches enhance flexibility but struggle with spatial reasoning and the joint modeling of route-level and POI-level preferences. To address these limitations, we propose RouteLLM, a hierarchical multi-agent framework that grounds natural-language intents into constraint-aware routes. It first parses user queries into structured intents including POIs, paths, and constraints. A manager agent then coordinates specialized sub-agents: a constraint agent that resolves and formally check constraints, a POI agent that retrieves and ranks candidate POIs, and a path refinement agent that refines routes via a routing engine with preference-conditioned costs. A final verifier agent ensures constraint satisfaction and produces the final route with an interpretable rationale. This design bridges linguistic flexibility and spatial structure, enabling reasoning over route feasibility and user preferences. Experiments show that our method reliably grounds textual preferences into constraint-aware routes, improving route quality and preference satisfaction over classical methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06039v1">CDTP: A Large-Scale Chinese Data-Text Pair Dataset for Comprehensive Evaluation of Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks. However, Chinese LLMs face unique challenges, primarily due to the dominance of unstructured free text and the lack of structured representations in Chinese corpora. While existing benchmarks for LLMs partially assess Chinese LLMs, they are still predominantly English-centric and fail to address the unique linguistic characteristics of Chinese, lacking structured datasets essential for robust evaluation. To address these challenges, we present a Comprehensive Benchmark for Evaluating Chinese Large Language Models (CB-ECLLM) based on the newly constructed Chinese Data-Text Pair (CDTP) dataset. Specifically, CDTP comprises over 7 million aligned text pairs, each consisting of unstructured text coupled with one or more corresponding triples, alongside a total of 15 million triples spanning four critical domains. The core contributions of CDTP are threefold: (i) enriching Chinese corpora with high-quality structured information; (ii) enabling fine-grained evaluation tailored to knowledge-driven tasks; and (iii) supporting multi-task fine-tuning to assess generalization and robustness across scenarios, including Knowledge Graph Completion, Triple-to-Text generation, and Question Answering. Furthermore, we conduct rigorous evaluations through extensive experiments and ablation studies to assess the effectiveness, Supervised Fine-Tuning (SFT), and robustness of the benchmark. To support reproducible research, we offer an open-source codebase and outline potential directions for future investigations based on our insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06018v1">Evaluating The Impact of Stimulus Quality in Investigations of LLM Language Performance</a></div>
    <div class="paper-meta">
      📅 2025-10-07
      | 💬 Presented at https://brigap-workshop.github.io/ Information to be updated upon publication of proceedings
    </div>
    <details class="paper-abstract">
      Recent studies employing Large Language Models (LLMs) to test the Argument from the Poverty of the Stimulus (APS) have yielded contrasting results across syntactic phenomena. This paper investigates the hypothesis that characteristics of the stimuli used in recent studies, including lexical ambiguities and structural complexities, may confound model performance. A methodology is proposed for re-evaluating LLM competence on syntactic prediction, focusing on GPT-2. This involves: 1) establishing a baseline on previously used (both filtered and unfiltered) stimuli, and 2) generating a new, refined dataset using a state-of-the-art (SOTA) generative LLM (Gemini 2.5 Pro Preview) guided by linguistically-informed templates designed to mitigate identified confounds. Our preliminary findings indicate that GPT-2 demonstrates notably improved performance on these refined PG stimuli compared to baselines, suggesting that stimulus quality significantly influences outcomes in surprisal-based evaluations of LLM syntactic competency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04721v1">BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04717v1">JSON Whisperer: Efficient JSON Editing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can modify JSON documents through natural language commands, but current approaches regenerate entire structures for each edit, resulting in computational inefficiency. We present JSON Whisperer, a framework that enables LLMs to generate RFC 6902 diff patches-expressing only the necessary modifications-rather than complete documents. We identify two key challenges in patch-based editing: (1) LLMs often miss related updates when generating isolated patches, and (2) array manipulations require tracking index shifts across operations, which LLMs handle poorly. To address these issues, we introduce EASE (Explicitly Addressed Sequence Encoding), which transforms arrays into dictionaries with stable keys, eliminating index arithmetic complexities. Our evaluation shows that patch generation with EASE reduces token usage by 31% while maintaining edit quality within 5% of full regeneration with particular gains for complex instructions and list manipulations. The dataset is available at: https://github.com/emnlp2025/JSON-Whisperer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04695v1">Beyond Outcome Reward: Decoupling Search and Answering Improves LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Enabling large language models (LLMs) to utilize search tools offers a promising path to overcoming fundamental limitations such as knowledge cutoffs and hallucinations. Recent work has explored reinforcement learning (RL) for training search-augmented agents that interleave reasoning and retrieval before answering. These approaches usually rely on outcome-based rewards (e.g., exact match), implicitly assuming that optimizing for final answers will also yield effective intermediate search behaviors. Our analysis challenges this assumption: we uncover multiple systematic deficiencies in search that arise under outcome-only training and ultimately degrade final answer quality, including failure to invoke tools, invalid queries, and redundant searches. To address these shortcomings, we introduce DeSA (Decoupling Search-and-Answering), a simple two-stage training framework that explicitly separates search optimization from answer generation. In Stage 1, agents are trained to improve search effectiveness with retrieval recall-based rewards. In Stage 2, outcome rewards are employed to optimize final answer generation. Across seven QA benchmarks, DeSA-trained agents consistently improve search behaviors, delivering substantially higher search recall and answer accuracy than outcome-only baselines. Notably, DeSA outperforms single-stage training approaches that simultaneously optimize recall and outcome rewards, underscoring the necessity of explicitly decoupling the two objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v4">SALAD: Systematic Assessment of Machine Unlearning on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
