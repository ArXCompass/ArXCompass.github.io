# llm - 2025_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16117v1">BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 11 pages, 9 figures. Accepted to IEEE VIS 2025 (Full Papers Track, submission ID 1204)
    </div>
    <details class="paper-abstract">
      Biomedical data harmonization is essential for enabling exploratory analyses and meta-studies, but the process of schema matching - identifying semantic correspondences between elements of disparate datasets (schemas) - remains a labor-intensive and error-prone task. Even state-of-the-art automated methods often yield low accuracy when applied to biomedical schemas due to the large number of attributes and nuanced semantic differences between them. We present BDIViz, a novel visual analytics system designed to streamline the schema matching process for biomedical data. Through formative studies with domain experts, we identified key requirements for an effective solution and developed interactive visualization techniques that address both scalability challenges and semantic ambiguity. BDIViz employs an ensemble approach that combines multiple matching methods with LLM-based validation, summarizes matches through interactive heatmaps, and provides coordinated views that enable users to quickly compare attributes and their values. Our method-agnostic design allows the system to integrate various schema matching algorithms and adapt to application-specific needs. Through two biomedical case studies and a within-subject user study with domain experts, we demonstrate that BDIViz significantly improves matching accuracy while reducing cognitive load and curation time compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17080v1">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at RecSys 2025; DOI:https://doi.org/10.1145/3705328.3748064
    </div>
    <details class="paper-abstract">
      Multimodal learning plays a critical role in e-commerce recommendation platforms today, enabling accurate recommendations and product understanding. However, existing vision-language models, such as CLIP, face key challenges in e-commerce recommendation systems: 1) Weak object-level alignment, where global image embeddings fail to capture fine-grained product attributes, leading to suboptimal retrieval performance; 2) Ambiguous textual representations, where product descriptions often lack contextual clarity, affecting cross-modal matching; and 3) Domain mismatch, as generic vision-language models may not generalize well to e-commerce-specific data. To address these limitations, we propose a framework, VL-CLIP, that enhances CLIP embeddings by integrating Visual Grounding for fine-grained visual understanding and an LLM-based agent for generating enriched text embeddings. Visual Grounding refines image representations by localizing key products, while the LLM agent enhances textual features by disambiguating product descriptions. Our approach significantly improves retrieval accuracy, multimodal retrieval effectiveness, and recommendation quality across tens of millions of items on one of the largest e-commerce platforms in the U.S., increasing CTR by 18.6%, ATC by 15.5%, and GMV by 4.0%. Additional experimental results show that our framework outperforms vision-language models, including CLIP, FashionCLIP, and GCL, in both precision and semantic alignment, demonstrating the potential of combining object-aware visual grounding and LLM-enhanced text representation for robust multimodal recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17075v1">LoRA is All You Need for Safety Alignment of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs -- with safety levels comparable to full-model fine-tuning -- without compromising their reasoning abilities. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. We also explore methods that further reduce such overlap -- via regularization or during weight merging -- and observe some improvement on certain tasks. We hope this result motivates designing approaches that yield more consistent improvements in the reasoning-safety trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17061v1">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10166v2">Fact-Checking with Contextual Narratives: Leveraging Retrieval-Augmented LLMs for Social Media Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      We propose CRAVE (Cluster-based Retrieval Augmented Verification with Explanation); a novel framework that integrates retrieval-augmented Large Language Models (LLMs) with clustering techniques to address fact-checking challenges on social media. CRAVE automatically retrieves multimodal evidence from diverse, often contradictory, sources. Evidence is clustered into coherent narratives, and evaluated via an LLM-based judge to deliver fact-checking verdicts explained by evidence summaries. By synthesizing evidence from both text and image modalities and incorporating agent-based refinement, CRAVE ensures consistency and diversity in evidence representation. Comprehensive experiments demonstrate CRAVE's efficacy in retrieval precision, clustering quality, and judgment accuracy, showcasing its potential as a robust decision-support tool for fact-checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17016v1">Causal Graph Fuzzy LLMs: A First Introduction and Applications in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted for publication at the Brazilian Congress of Artificial Intelligence (CBIC)
    </div>
    <details class="paper-abstract">
      In recent years, the application of Large Language Models (LLMs) to time series forecasting (TSF) has garnered significant attention among researchers. This study presents a new frame of LLMs named CGF-LLM using GPT-2 combined with fuzzy time series (FTS) and causal graph to predict multivariate time series, marking the first such architecture in the literature. The key objective is to convert numerical time series into interpretable forms through the parallel application of fuzzification and causal analysis, enabling both semantic understanding and structural insight as input for the pretrained GPT-2 model. The resulting textual representation offers a more interpretable view of the complex dynamics underlying the original time series. The reported results confirm the effectiveness of our proposed LLM-based time series forecasting model, as demonstrated across four different multivariate time series datasets. This initiative paves promising future directions in the domain of TSF using LLMs based on FTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17015v1">Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at https://github.com/apple/ml-agent-evaluator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05200v2">ORANSight-2.0: Foundational LLMs for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative to develop specialized foundational LLMs tailored for O-RAN. Built on 18 models spanning five open-source LLM frameworks -- Mistral, Qwen, Llama, Phi, and Gemma -- ORANSight-2.0 fine-tunes models ranging from 1B to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance in O-RAN-specific tasks. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG)-based instruction-tuning framework that employs two LLM agents -- a Mistral-based Question Generator and a Qwen-based Answer Generator -- to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16989v1">Obscured but Not Erased: Evaluating Nationality Bias in LLMs via Name-Based Bias Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can exhibit latent biases towards specific nationalities even when explicit demographic markers are not present. In this work, we introduce a novel name-based benchmarking approach derived from the Bias Benchmark for QA (BBQ) dataset to investigate the impact of substituting explicit nationality labels with culturally indicative names, a scenario more reflective of real-world LLM applications. Our novel approach examines how this substitution affects both bias magnitude and accuracy across a spectrum of LLMs from industry leaders such as OpenAI, Google, and Anthropic. Our experiments show that small models are less accurate and exhibit more bias compared to their larger counterparts. For instance, on our name-based dataset and in the ambiguous context (where the correct choice is not revealed), Claude Haiku exhibited the worst stereotypical bias scores of 9%, compared to only 3.5% for its larger counterpart, Claude Sonnet, where the latter also outperformed it by 117.7% in accuracy. Additionally, we find that small models retain a larger portion of existing errors in these ambiguous contexts. For example, after substituting names for explicit nationality references, GPT-4o retains 68% of the error rate versus 76% for GPT-4o-mini, with similar findings for other model providers, in the ambiguous context. Our research highlights the stubborn resilience of biases in LLMs, underscoring their profound implications for the development and deployment of AI systems in diverse, global contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16974v1">Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 15 pages, 9 tables, Appendix A-K
    </div>
    <details class="paper-abstract">
      Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16951v1">Harnessing RLHF for Robust Unanswerability Recognition and Trustworthy Response Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Conversational Information Retrieval (CIR) systems, while offering intuitive access to information, face a significant challenge: reliably handling unanswerable questions to prevent the generation of misleading or hallucinated content. Traditional approaches often rely on external classifiers, which can introduce inconsistencies with the core generative Large Language Models (LLMs). This paper introduces Self-Aware LLM for Unanswerability (SALU), a novel approach that deeply integrates unanswerability detection directly within the LLM's generative process. SALU is trained using a multi-task learning framework for both standard Question Answering (QA) and explicit abstention generation for unanswerable queries. Crucially, it incorporates a confidence-score-guided reinforcement learning with human feedback (RLHF) phase, which explicitly penalizes hallucinated responses and rewards appropriate abstentions, fostering intrinsic self-awareness of knowledge boundaries. Through extensive experiments on our custom-built C-IR_Answerability dataset, SALU consistently outperforms strong baselines, including hybrid LLM-classifier systems, in overall accuracy for correctly answering or abstaining from questions. Human evaluation further confirms SALU's superior reliability, achieving high scores in factuality, appropriate abstention, and, most importantly, a dramatic reduction in hallucination, demonstrating its ability to robustly "know when to say 'I don't know'."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18489v2">LLM as a code generator in Agile Model Driven Development</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLM) like GPT4 in the auto generation of code represents a significant advancement, yet it is not without its challenges. The ambiguity inherent in natural language descriptions of software poses substantial obstacles to generating deployable, structured artifacts. This research champions Model Driven Development (MDD) as a viable strategy to overcome these challenges, proposing an Agile Model Driven Development (AMDD) approach that employs GPT4 as a code generator. This approach enhances the flexibility and scalability of the code auto generation process and offers agility that allows seamless adaptation to changes in models or deployment environments. We illustrate this by modeling a multi agent Unmanned Vehicle Fleet (UVF) system using the Unified Modeling Language (UML), significantly reducing model ambiguity by integrating the Object Constraint Language (OCL) for code structure meta modeling, and the FIPA ontology language for communication semantics meta modeling. Applying GPT4 auto generation capabilities yields Java and Python code that is compatible with the JADE and PADE frameworks, respectively. Our thorough evaluation of the auto generated code verifies its alignment with expected behaviors and identifies enhancements in agent interactions. Structurally, we assessed the complexity of code derived from a model constrained solely by OCL meta models, against that influenced by both OCL and FIPA ontology meta models. The results indicate that the ontology constrained meta model produces inherently more complex code, yet its cyclomatic complexity remains within manageable levels, suggesting that additional meta model constraints can be incorporated without exceeding the high risk threshold for complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00178v2">Tournament of Prompts: Evolving LLM Instructions Through Structured Debates and Elo Ratings</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Prompt engineering represents a critical bottleneck to harness the full potential of Large Language Models (LLMs) for solving complex tasks, as it requires specialized expertise, significant trial-and-error, and manual intervention. This challenge is particularly pronounced for tasks involving subjective quality assessment, where defining explicit optimization objectives becomes fundamentally problematic. Existing automated prompt optimization methods falter in these scenarios, as they typically require well-defined task-specific numerical fitness functions or rely on generic templates that cannot capture the nuanced requirements of complex use cases. We introduce DEEVO (DEbate-driven EVOlutionary prompt optimization), a novel framework that guides prompt evolution through a debate-driven evaluation with an Elo-based selection. Contrary to prior work, DEEVOs approach enables exploration of the discrete prompt space while preserving semantic coherence through intelligent crossover and strategic mutation operations that incorporate debate-based feedback, combining elements from both successful and unsuccessful prompts based on identified strengths rather than arbitrary splicing. Using Elo ratings as a fitness proxy, DEEVO simultaneously drives improvement and preserves valuable diversity in the prompt population. Experimental results demonstrate that DEEVO significantly outperforms both manual prompt engineering and alternative state-of-the-art optimization approaches on open-ended tasks and close-ended tasks despite using no ground truth feedback. By connecting LLMs reasoning capabilities with adaptive optimization, DEEVO represents a significant advancement in prompt optimization research by eliminating the need of predetermined metrics to continuously improve AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16809v1">LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 41 pages, 17 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16808v1">Rethinking LLM-Based RTL Code Optimization Via Timing Logic Metamorphosis</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 13pages with 9 pictures and 2 tables
    </div>
    <details class="paper-abstract">
      Register Transfer Level(RTL) code optimization is crucial for achieving high performance and low power consumption in digital circuit design. However, traditional optimization methods often rely on manual tuning and heuristics, which can be time-consuming and error-prone. Recent studies proposed to leverage Large Language Models(LLMs) to assist in RTL code optimization. LLMs can generate optimized code snippets based on natural language descriptions, potentially speeding up the optimization process. However, existing approaches have not thoroughly evaluated the effectiveness of LLM-Based code optimization methods for RTL code with complex timing logic. To address this gap, we conducted a comprehensive empirical investigation to assess the capability of LLM-Based RTL code optimization methods in handling RTL code with complex timing logic. In this study, we first propose a new benchmark for RTL optimization evaluation. It comprises four subsets, each corresponding to a specific area of RTL code optimization. Then we introduce a method based on metamorphosis to systematically evaluate the effectiveness of LLM-Based RTL code optimization methods.Our key insight is that the optimization effectiveness should remain consistent for semantically equivalent but more complex code. After intensive experiments, we revealed several key findings. (1) LLM-Based RTL optimization methods can effectively optimize logic operations and outperform existing compiler-based methods. (2) LLM-Based RTL optimization methods do not perform better than existing compiler-based methods on RTL code with complex timing logic, particularly in timing control flow optimization and clock domain optimization. This is primarily attributed to the challenges LLMs face in understanding timing logic in RTL code. Based on these findings, we provide insights for further research in leveraging LLMs for RTL code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16799v1">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16773v1">When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16754v1">Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However, designing effective pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches, and (2) address new questions without any close prior matches by automatically lowering the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage for unseen cases. We find that implementing a RAG pipeline combining hypothetical-documentation-embedding (HyDE) with the full-answer context performs best in retrieving and answering similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG pipeline to 4 open-source LLMs and compare the results to their zero-shot performance. Our findings show that RAG with our optimal RAG pipeline consistently outperforms zero-shot baselines across models, achieving higher scores for helpfulness, correctness, and detail with LLM-as-a-judge. These findings demonstrate that our optimal RAG pipelines robustly enhance answer quality for a wide range of developer queries including both previously seen and novel questions across different LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16731v1">Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 35 pages, 9 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, deploying them solely in the cloud or compressing them for edge devices has become inadequate due to concerns about latency, privacy, cost, and personalization. This survey explores a collaborative paradigm in which cloud-based LLMs and edge-deployed small language models (SLMs) cooperate across both inference and training. We present a unified taxonomy of edge-cloud collaboration strategies. For inference, we categorize approaches into task assignment, task division, and mixture-based collaboration at both task and token granularity, encompassing adaptive scheduling, resource-aware offloading, speculative decoding, and modular routing. For training, we review distributed adaptation techniques, including parameter alignment, pruning, bidirectional distillation, and small-model-guided optimization. We further summarize datasets, benchmarks, and deployment cases, and highlight privacy-preserving methods and vertical applications. This survey provides the first systematic foundation for LLM-SLM collaboration, bridging system and algorithm co-design to enable efficient, scalable, and trustworthy edge-cloud intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16727v1">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16716v1">Enhancing Remote Sensing Vision-Language Models Through MLLM and LLM-Based High-Quality Image-Text Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 SUBMIT TO IEEE TRANSACTIONS
    </div>
    <details class="paper-abstract">
      The application of Vision-language foundation models (VLFMs) to remote sensing (RS) imagery has garnered significant attention due to their superior capability in various downstream tasks. A key challenge lies in the scarcity of high-quality, large-scale, image-text paired training data. Recently, several works introduced extensive image-text datasets for RS and trained their VLFMs. However, due to the rudimentary methods used for generating captions, the quality of datasets is suboptimal, requiring larger volumes of training data, while only yielding modest performance improvements. In this paper, we propose a two-stage method named MpGI(Multi-Perspective Generation and Integration) for generating high-quality text captions for RS images. Firstly, we generate distinct and detailed descriptions from different perspectives using Rule-MLLM(Multimodal Large Language Model) Relay Generation and MLLMs generation methods. Next, we utilize Large Language Models (LLMs) to integrate these diverse descriptions into comprehensive captions, capturing details from multiple perspectives. Finally, we have created the HQRS-IT-210K dataset, including about 210,000 RS images and 1.3 million captions. We fine-tuned two VLFMs using our dataset: CLIP, a discriminative model, and CoCa, an image-to-text generative model. This process resulted in our proposed HQRS-CLIP and RS-CoCa models. Experimental results demonstrate that HQRS-CLIP surpassed the previous SOTA RS CLIP model in various downstream tasks while using only 4.2\% of the training data. RS-CoCa outperforms other advanced approaches across benchmark datasets and can generate captions for RS images that rival or even exceed manual annotations. Dataset, pre-trained models, and codes will be released at https://github.com/YiguoHe/HQRS-210K-and-HQRS-CLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16708v1">Biases in LLM-Generated Musical Taste Profiles for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      One particularly promising use case of Large Language Models (LLMs) for recommendation is the automatic generation of Natural Language (NL) user taste profiles from consumption data. These profiles offer interpretable and editable alternatives to opaque collaborative filtering representations, enabling greater transparency and user control. However, it remains unclear whether users consider these profiles to be an accurate representation of their taste, which is crucial for trust and usability. Moreover, because LLMs inherit societal and data-driven biases, profile quality may systematically vary across user and item characteristics. In this paper, we study this issue in the context of music streaming, where personalization is challenged by a large and culturally diverse catalog. We conduct a user study in which participants rate NL profiles generated from their own listening histories. We analyze whether identification with the profiles is biased by user attributes (e.g., mainstreamness, taste diversity) and item features (e.g., genre, country of origin). We also compare these patterns to those observed when using the profiles in a downstream recommendation task. Our findings highlight both the potential and limitations of scrutable, LLM-based profiling in personalized systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08773v2">Universal Model Routing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Model routing is a simple technique for reducing the inference cost of large language models (LLMs), wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose UniRoute, a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective instantiations of UniRoute, relying on cluster-based routing and a learned cluster map respectively. We show that these are estimates of a theoretically optimal routing rule, and quantify their errors via an excess risk bound. Experiments on a range of public benchmarks show the effectiveness of UniRoute in routing amongst more than 30 unseen LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16679v1">PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16672v1">Meta-Learning for Cold-Start Personalization in Prompt-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Generative, explainable, and flexible recommender systems, derived using Large Language Models (LLM) are promising and poorly adapted to the cold-start user situation, where there is little to no history of interaction. The current solutions i.e. supervised fine-tuning and collaborative filtering are dense-user-item focused and would be expensive to maintain and update. This paper introduces a meta-learning framework, that can be used to perform parameter-efficient prompt-tuning, to effectively personalize LLM-based recommender systems quickly at cold-start. The model learns soft prompt embeddings with first-order (Reptile) and second-order (MAML) optimization by treating each of the users as the tasks. As augmentations to the input tokens, these learnable vectors are the differentiable control variables that represent user behavioral priors. The prompts are meta-optimized through episodic sampling, inner-loop adaptation, and outer-loop generalization. On MovieLens-1M, Amazon Reviews, and Recbole, we can see that our adaptive model outperforms strong baselines in NDCG@10, HR@10, and MRR, and it runs in real-time (i.e., below 300 ms) on consumer GPUs. Zero-history personalization is also supported by this scalable solution, and its 275 ms rate of adaptation allows successful real-time risk profiling of financial systems by shortening detection latency and improving payment network stability. Crucially, the 275 ms adaptation capability can enable real-time risk profiling for financial institutions, reducing systemic vulnerability detection latency significantly versus traditional compliance checks. By preventing contagion in payment networks (e.g., Fedwire), the framework strengthens national financial infrastructure resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16656v1">P-CoT: A Pedagogically-motivated Participatory Chain-of-Thought Prompting for Phonological Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      This study explores the potential of phonological reasoning within text-based large language models (LLMs). Utilizing the PhonologyBench benchmark, we assess tasks like rhyme word generation, g2p conversion, and syllable counting. Our evaluations across 12 LLMs reveal that while few-shot learning offers inconsistent gains, the introduction of a novel Pedagogically-motivated Participatory Chain-of-Thought (P-CoT) prompt, which is anchored in educational theories like scaffolding and discovery learning, consistently enhances performance. This method leverages structured guidance to activate latent phonological abilities, achieving up to 52% improvement and even surpassing human baselines in certain tasks. Future work could aim to optimize P-CoT prompts for specific models or explore their application across different linguistic domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16587v1">On the Effectiveness of LLM-as-a-judge for Code Generation and Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at TSE. IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Large Language Models have been recently exploited as judges for complex natural language processing tasks, such as Q&A. The basic idea is to delegate to an LLM the assessment of the "quality" of the output provided by an automated technique for tasks for which: (i) quantitative metrics would only tell part of the story, and; (ii) a large-scale human-based evaluation would be too expensive. LLMs-as-a-judge, if proven effective for a specific task, can also unlock new possibilities for automation, with several LLMs proposing a solution for a given instance of the task and others judging and deciding what is the best output to show the user. We study the effectiveness of LLMs-as-a-judge for two code-related tasks, namely code generation and code summarization. The rationale for choosing these tasks is two-fold. First, quantitative metrics are usually not enough for the assessment of code summarizers/generators. For example, it is well documented that metrics such as BLEU are quite weak proxies for the quality of the generated summaries. Second, even state-of-the-art techniques still struggle with handling complex instances of these tasks, making them good candidates for benefiting from more advanced solutions envisioning collaboration among LLMs. For code generation, we check whether eight LLMs are able to judge the correctness of 1,405 Java methods and 1,281 Python functions generated by the same LLMs or implemented by humans. For code summarization, we compare the judgment of five LLMs to those provided by nine humans for ~1.2k summaries, related to both Java and Python functions. Our findings show that GPT-4-turbo is the best LLM in terms of judging capabilities for both tasks, with "smaller" LLMs featuring tens of billions parameters not being able to cope with judging tasks. However, even the best-performing LLM frequently misjudges the correctness of the code and summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06821v3">Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 37 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v2">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v5">Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Project page: https://github.com/VITA-MLLM/Sparrow
    </div>
    <details class="paper-abstract">
      Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11346v3">Decision support system for Forest fire management using Ontology with Big Data and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted
    </div>
    <details class="paper-abstract">
      Forests are crucial for ecological balance, but wildfires, a major cause of forest loss, pose significant risks. Fire weather indices, which assess wildfire risk and predict resource demands, are vital. With the rise of sensor networks in fields like healthcare and environmental monitoring, semantic sensor networks are increasingly used to gather climatic data such as wind speed, temperature, and humidity. However, processing these data streams to determine fire weather indices presents challenges, underscoring the growing importance of effective forest fire detection. This paper discusses using Apache Spark for early forest fire detection, enhancing fire risk prediction with meteorological and geographical data. Building on our previous development of Semantic Sensor Network (SSN) ontologies and Semantic Web Rules Language (SWRL) for managing forest fires in Monesterial Natural Park, we expanded SWRL to improve a Decision Support System (DSS) using a Large Language Models (LLMs) and Spark framework. We implemented real-time alerts with Spark streaming, tailored to various fire scenarios, and validated our approach using ontology metrics, query-based evaluations, LLMs score precision, F1 score, and recall measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15251v1">Input Reduction Enhanced LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in Automated Program Repair (APR). Test inputs, being crucial for reasoning the root cause of failures, are always included in the prompt for LLM-based APR. Unfortunately, LLMs struggle to retain key information in long prompts. When the test inputs are extensive in the prompt, this may trigger the "lost-in-the-middle" issue, compromising repair performance. To address this, we propose ReduceFix, an LLM-based APR approach with a built-in component that automatically reduces test inputs while retaining their failure-inducing behavior. ReduceFix prompts an LLM to generate a reducer that minimizes failure-inducing test inputs without human effort, and then feeds the reduced failure-inducing inputs to guide patch generation. For targeted evaluation, we constructed LFTBench, the first long-input APR benchmark with 200 real bugs from 20 programming tasks, each paired with a failure-inducing input whose median size is 1 MB. On this benchmark, ReduceFix shrinks inputs by 89.1% on average and improves overall pass@10 by up to 53.8% relative to a prompt that includes the original test, and by 17.6% compared with omitting the test entirely. Adding the same reduction step to ChatRepair increases its fix rate by 21.3% without other changes. Ablation studies further highlight the impact of input length and compressed failure information on repair success. These results underscore that automatically reducing failing inputs is a practical and powerful complement to LLM-based APR, significantly improving its scalability and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06838v3">ACFIX: Guiding LLMs with Mined Common RBAC Practices for Context-Aware Repair of Access Control Vulnerabilities in Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 This is a technical report from Nanyang Technological University
    </div>
    <details class="paper-abstract">
      Smart contracts are susceptible to various security issues, among which access control (AC) vulnerabilities are particularly critical. While existing research has proposed multiple detection tools, the automatic and appropriate repair of AC vulnerabilities in smart contracts remains a challenge. Unlike commonly supported vulnerability types by existing repair tools, such as reentrancy, which are usually fixed by template-based approaches, the main obstacle of AC lies in identifying the appropriate roles or permissions amid a long list of non-AC-related source code to generate proper patch code, a task that demands human-level intelligence. Leveraging recent advancements in large language models (LLMs), we employ the state-of-the-art GPT-4 model and enhance it with a novel approach called ACFIX. The key insight is that we can mine common AC practices for major categories of code functionality and use them to guide LLMs in fixing code with similar functionality. To this end, ACFIX involves both offline and online phases. First, during the offline phase, ACFIX mines a taxonomy of common Role-based Access Control (RBAC) practices from 344,251 on-chain contracts, categorizing 49 role-permission pairs from the top 1,000 pairs mined. Second, during the online phase, ACFIX tracks AC-related elements across the contract and uses this context information along with a Chain-of-Thought pipeline to guide LLMs in identifying the most appropriate role-permission pair for the subject contract and subsequently generating a suitable patch. This patch will then undergo a validity and effectiveness check. To evaluate ACFIX, we built the first benchmark dataset of 118 real-world AC vulnerabilities, and our evaluation revealed that ACFIX successfully repaired 94.92% of them. This represents a significant improvement compared to the baseline GPT-4, which achieved only 52.54%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13023v2">A Practical Guide for Evaluating LLMs and LLM-Reliant Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advances in generative AI have led to remarkable interest in using systems that rely on large language models (LLMs) for practical applications. However, meaningful evaluation of these systems in real-world scenarios comes with a distinct set of challenges, which are not well-addressed by synthetic benchmarks and de-facto metrics that are often seen in the literature. We present a practical evaluation framework which outlines how to proactively curate representative datasets, select meaningful evaluation metrics, and employ meaningful evaluation methodologies that integrate well with practical development and deployment of LLM-reliant systems that must adhere to real-world requirements and meet user-facing needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15245v1">SPAR: Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have opened new opportunities for academic literature retrieval. However, existing systems often rely on rigid pipelines and exhibit limited reasoning capabilities. We introduce SPAR, a multi-agent framework that incorporates RefChain-based query decomposition and query evolution to enable more flexible and effective search. To facilitate systematic evaluation, we also construct SPARBench, a challenging benchmark with expert-annotated relevance labels. Experimental results demonstrate that SPAR substantially outperforms strong baselines, achieving up to +56% F1 on AutoScholar and +23% F1 on SPARBench over the best-performing baseline. Together, SPAR and SPARBench provide a scalable, interpretable, and high-performing foundation for advancing research in scholarly retrieval. Code and data will be available at: https://github.com/xiaofengShi/SPAR
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15241v1">FaultLine: Automated Proof-of-Vulnerability Generation Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Despite the critical threat posed by software security vulnerabilities, reports are often incomplete, lacking the proof-of-vulnerability (PoV) tests needed to validate fixes and prevent regressions. These tests are crucial not only for ensuring patches work, but also for helping developers understand how vulnerabilities can be exploited. Generating PoV tests is a challenging problem, requiring reasoning about the flow of control and data through deeply nested levels of a program. We present FaultLine, an LLM agent workflow that uses a set of carefully designed reasoning steps, inspired by aspects of traditional static and dynamic program analysis, to automatically generate PoV test cases. Given a software project with an accompanying vulnerability report, FaultLine 1) traces the flow of an input from an externally accessible API ("source") to the "sink" corresponding to the vulnerability, 2) reasons about the conditions that an input must satisfy in order to traverse the branch conditions encountered along the flow, and 3) uses this reasoning to generate a PoV test case in a feedback-driven loop. FaultLine does not use language-specific static or dynamic analysis components, which enables it to be used across programming languages. To evaluate FaultLine, we collate a challenging multi-lingual dataset of 100 known vulnerabilities in Java, C and C++ projects. On this dataset, FaultLine is able to generate PoV tests for 16 projects, compared to just 9 for CodeAct 2.1, a popular state-of-the-art open-source agentic framework. Thus, FaultLine represents a 77% relative improvement over the state of the art. Our findings suggest that hierarchical reasoning can enhance the performance of LLM agents on PoV test generation, but the problem in general remains challenging. We make our code and dataset publicly available in the hope that it will spur further research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02930v4">Video LLMs for Temporal Reasoning in Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      This paper introduces TemporalVLM, a video large language model (video LLM) capable of effective temporal reasoning and fine-grained understanding in long videos. At the core, our approach includes a visual encoder for mapping a long-term input video into features which are time-aware and contain both local and global cues. In particular, it first divides the input video into short-term clips, which are jointly encoded with their timestamps and fused across overlapping temporal windows into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. The extracted time-aware and multi-level features are important for accurate temporal reasoning and fine-grained understanding in long videos. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, which consists of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments on datasets of long videos, including TimeIT and IndustryASM, show that TemporalVLM achieves superior performance than previous methods across temporal reasoning and fine-grained understanding tasks, namely dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To the best of our knowledge, our work is the first to incorporate LSTMs into video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v4">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v4">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted by IJCAI 2025 (Survey Track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs, which can be categorized into the following two aspects: (1) Logical question answering: LLMs often fail to generate the correct answer within a complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises. (2) Logical consistency: LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art question-answering LLM Macaw, answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose a detailed taxonomy. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistencies, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extending to modal logic to account for uncertainty and developing efficient algorithms that simultaneously satisfy multiple logical consistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16110v1">Expert-Guided LLM Reasoning for Battery Discovery: From AI-Driven Hypothesis to Synthesis and Characterization</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) leverage chain-of-thought (CoT) techniques to tackle complex problems, representing a transformative breakthrough in artificial intelligence (AI). However, their reasoning capabilities have primarily been demonstrated in solving math and coding problems, leaving their potential for domain-specific applications-such as battery discovery-largely unexplored. Inspired by the idea that reasoning mirrors a form of guided search, we introduce ChatBattery, a novel agentic framework that integrates domain knowledge to steer LLMs toward more effective reasoning in materials design. Using ChatBattery, we successfully identify, synthesize, and characterize three novel lithium-ion battery cathode materials, which achieve practical capacity improvements of 28.8%, 25.2%, and 18.5%, respectively, over the widely used cathode material, LiNi0.8Mn0.1Co0.1O2 (NMC811). Beyond this discovery, ChatBattery paves a new path by showing a successful LLM-driven and reasoning-based platform for battery materials invention. This complete AI-driven cycle-from design to synthesis to characterization-demonstrates the transformative potential of AI-driven reasoning in revolutionizing materials discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16044v1">Making REST APIs Agent-Ready: From OpenAPI to Model Context Protocol Servers for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are evolving from passive text generators into active agents that invoke external tools. To support this shift, scalable protocols for tool integration are essential. The Model Context Protocol (MCP), introduced by Anthropic in 2024, offers a schema-driven standard for dynamic tool discovery and invocation. Yet, building MCP servers remains manual and repetitive, requiring developers to write glue code, handle authentication, and configure schemas by hand-replicating much of the integration effort MCP aims to eliminate. This paper investigates whether MCP server construction can be meaningfully automated. We begin by analyzing adoption trends: among 22,000+ MCP-tagged GitHub repositories created within six months of release, fewer than 5% include servers, typically small, single-maintainer projects dominated by repetitive scaffolding. To address this gap, we present AutoMCP, a compiler that generates MCP servers from OpenAPI 2.0/3.0 specifications. AutoMCP parses REST API definitions and produces complete server implementations, including schema registration and authentication handling. We evaluate AutoMCP on 50 real-world APIs spanning 5,066 endpoints across over 10 domains. From a stratified sample of 1,023 tool calls, 76.5% succeeded out of the box. Manual failure analysis revealed five recurring issues, all attributable to inconsistencies or omissions in the OpenAPI contracts. After minor fixes, averaging 19 lines of spec changes per API, AutoMCP achieved 99.9% success. Our findings (i) analyze MCP adoption and quantify the cost of manual server development, (ii) demonstrate that OpenAPI specifications, despite quality issues, enable near-complete MCP server automation, and (iii) contribute a corpus of 5,066 callable tools along with insights on repairing common specification flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16037v1">A Pilot Study on LLM-Based Agentic Translation from Android to iOS: Pitfalls and Insights</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of mobile applications has led to a significant demand for cross-platform compatibility, particularly between the Android and iOS platforms. Traditional approaches to mobile application translation often rely on manual intervention or rule-based systems, which are labor-intensive and time-consuming. While recent advancements in machine learning have introduced automated methods, they often lack contextual understanding and adaptability, resulting in suboptimal translations. Large Language Models (LLMs) were recently leveraged to enhance code translation at different granularities, including the method, class, and repository levels. Researchers have investigated common errors, limitations, and potential strategies to improve these tasks. However, LLM-based application translation across different platforms, such as migrating mobile applications between Android and iOS or adapting software across diverse frameworks, remains underexplored. Understanding the performance, strengths, and limitations of LLMs in cross-platform application translation is critical for advancing software engineering automation. This study aims to fill this gap by evaluating LLM-based agentic approaches for mobile application translation, identifying key failure points, and proposing guidelines to improve translation performance. We developed a chain of agents that account for dependencies, specifications, program structure, and program control flow when translating applications from Android to iOS. To evaluate the performance, we manually examined the translated code for syntactic correctness, semantic accuracy, and functional completeness. For translation failures, we further conducted a detailed root cause analysis to understand the underlying limitations of the agentic translation process and identify opportunities for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16028v1">From Logic to Language: A Trust Index for Problem Solving with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 17 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Classical computation, grounded in formal, logical systems, has been the engine of technological progress for decades, excelling at problems that can be described with unambiguous rules. This paradigm, however, leaves a vast ocean of human problems -- those characterized by ambiguity, dynamic environments, and subjective context -- largely untouched. The advent of Large Language Models (LLMs) represents a fundamental shift, enabling computational systems to engage with this previously inaccessible domain using natural language. This paper introduces a unified framework to understand and contrast these problem-solving paradigms. We define and delineate the problem spaces addressable by formal languages versus natural language. While solutions to the former problem class can be evaluated using binary quality measures, the latter requires a much more nuanced definition of approximate solution space taking into account the vagueness, subjectivity and ambiguity inherent to natural language. We therefore introduce a vector-valued trust index Q, which reflects solution quality and distinguishes the binary correctness of formal solutions from the continuous adequacy spectrum characteristic of natural language solutions. Within this framework, we propose two statistical quality dimensions. Normalized bi-semantic entropy measures robustness and conceptual diversity of LLM answers given semantic variation in problem formulations. Emotional valence maps subjective valuation of a solution to a quantifiable metric that can be maximized by invoking statistical measures. The concepts introduced in this work will provide a more rigorous understanding of the capabilities, limitations, and inherent nature of problem-solving in the age of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18574v3">Autocomp: LLM-Driven Code Optimization for Tensor Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Hardware accelerators, especially those designed for tensor processing, have become ubiquitous in today's computing landscape. However, even with significant efforts in building compilers, programming these tensor accelerators remains challenging, leaving much of their potential underutilized. Recently, large language models (LLMs), trained on large amounts of code, have shown significant promise in code generation and optimization tasks, but generating low-resource languages like specialized tensor accelerator code still poses a significant challenge. We tackle this challenge with Autocomp, an approach that empowers accelerator programmers to leverage domain knowledge and hardware feedback to optimize code via an automated LLM-driven search. We accomplish this by: 1) formulating each optimization pass as a structured two-phase prompt, divided into planning and code generation phases, 2) inserting domain knowledge during planning via a concise and adaptable optimization menu, and 3) integrating correctness and performance metrics from hardware as feedback at each search iteration. Across three categories of representative workloads and two different accelerators, we demonstrate that Autocomp-optimized code runs 5.6x (GEMM) and 2.7x (convolution) faster than the vendor-provided library, and outperforms expert-level hand-tuned code by 1.4x (GEMM), 1.1x (convolution), and 1.3x (fine-grained linear algebra). Additionally, we demonstrate that optimization schedules generated from Autocomp can be reused across similar tensor operations, improving speedups by up to 24% under a fixed sample budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16007v1">Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Can LLMs provide support to creative writers by giving meaningful writing feedback? In this paper, we explore the challenges and limitations of model-generated writing feedback by defining a new task, dataset, and evaluation frameworks. To study model performance in a controlled manner, we present a novel test set of 1,300 stories that we corrupted to intentionally introduce writing issues. We study the performance of commonly used LLMs in this task with both automatic and human evaluation metrics. Our analysis shows that current models have strong out-of-the-box behavior in many respects -- providing specific and mostly accurate writing feedback. However, models often fail to identify the biggest writing issue in the story and to correctly decide when to offer critical vs. positive feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05461v2">GLOSS: Group of LLMs for Open-Ended Sensemaking of Passive Sensing Data for Health and Wellbeing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The ubiquitous presence of smartphones and wearables has enabled researchers to build prediction and detection models for various health and behavior outcomes using passive sensing data from these devices. Achieving a high-level, holistic understanding of an individual's behavior and context, however, remains a significant challenge. Due to the nature of passive sensing data, sensemaking -- the process of interpreting and extracting insights -- requires both domain knowledge and technical expertise, creating barriers for different stakeholders. Existing systems designed to support sensemaking are either not open-ended or cannot perform complex data triangulation. In this paper, we present a novel sensemaking system, Group of LLMs for Open-ended Sensemaking (GLOSS), capable of open-ended sensemaking and performing complex multimodal triangulation to derive insights. We demonstrate that GLOSS significantly outperforms the commonly used Retrieval-Augmented Generation (RAG) technique, achieving 87.93% accuracy and 66.19% consistency, compared to RAG's 29.31% accuracy and 52.85% consistency. Furthermore, we showcase the promise of GLOSS through four use cases inspired by prior and ongoing work in the UbiComp and HCI communities. Finally, we discuss the potential of GLOSS, its broader implications, and the limitations of our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15903v1">Towards Mitigation of Hallucination for LLM-empowered Agents: Progressive Generalization Bound Exploration and Watchdog Monitor</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Empowered by large language models (LLMs), intelligent agents have become a popular paradigm for interacting with open environments to facilitate AI deployment. However, hallucinations generated by LLMs-where outputs are inconsistent with facts-pose a significant challenge, undermining the credibility of intelligent agents. Only if hallucinations can be mitigated, the intelligent agents can be used in real-world without any catastrophic risk. Therefore, effective detection and mitigation of hallucinations are crucial to ensure the dependability of agents. Unfortunately, the related approaches either depend on white-box access to LLMs or fail to accurately identify hallucinations. To address the challenge posed by hallucinations of intelligent agents, we present HalMit, a novel black-box watchdog framework that models the generalization bound of LLM-empowered agents and thus detect hallucinations without requiring internal knowledge of the LLM's architecture. Specifically, a probabilistic fractal sampling technique is proposed to generate a sufficient number of queries to trigger the incredible responses in parallel, efficiently identifying the generalization bound of the target agent. Experimental evaluations demonstrate that HalMit significantly outperforms existing approaches in hallucination monitoring. Its black-box nature and superior performance make HalMit a promising solution for enhancing the dependability of LLM-powered systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16860v1">Weak Links in LinkedIn: Enhancing Fake Profile Detection in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 10 pages, 3 figures, 1 table, accepted for publication at ASONAM 2025. https://sites.google.com/view/weaklinksinlinkedin/home
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made it easier to create realistic fake profiles on platforms like LinkedIn. This poses a significant risk for text-based fake profile detectors. In this study, we evaluate the robustness of existing detectors against LLM-generated profiles. While highly effective in detecting manually created fake profiles (False Accept Rate: 6-7%), the existing detectors fail to identify GPT-generated profiles (False Accept Rate: 42-52%). We propose GPT-assisted adversarial training as a countermeasure, restoring the False Accept Rate to between 1-7% without impacting the False Reject Rates (0.5-2%). Ablation studies revealed that detectors trained on combined numerical and textual embeddings exhibit the highest robustness, followed by those using numerical-only embeddings, and lastly those using textual-only embeddings. Complementary analysis on the ability of prompt-based GPT-4Turbo and human evaluators affirms the need for robust automated detectors such as the one proposed in this study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16852v1">SynthCTI: LLM-Driven Synthetic CTI Generation to enhance MITRE Technique Mapping</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 17 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Cyber Threat Intelligence (CTI) mining involves extracting structured insights from unstructured threat data, enabling organizations to understand and respond to evolving adversarial behavior. A key task in CTI mining is mapping threat descriptions to MITRE ATT\&CK techniques. However, this process is often performed manually, requiring expert knowledge and substantial effort. Automated approaches face two major challenges: the scarcity of high-quality labeled CTI data and class imbalance, where many techniques have very few examples. While domain-specific Large Language Models (LLMs) such as SecureBERT have shown improved performance, most recent work focuses on model architecture rather than addressing the data limitations. In this work, we present SynthCTI, a data augmentation framework designed to generate high-quality synthetic CTI sentences for underrepresented MITRE ATT\&CK techniques. Our method uses a clustering-based strategy to extract semantic context from training data and guide an LLM in producing synthetic CTI sentences that are lexically diverse and semantically faithful. We evaluate SynthCTI on two publicly available CTI datasets, CTI-to-MITRE and TRAM, using LLMs with different capacity. Incorporating synthetic data leads to consistent macro-F1 improvements: for example, ALBERT improves from 0.35 to 0.52 (a relative gain of 48.6\%), and SecureBERT reaches 0.6558 (up from 0.4412). Notably, smaller models augmented with SynthCTI outperform larger models trained without augmentation, demonstrating the value of data generation methods for building efficient and effective CTI classification systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15849v1">The Impact of Language Mixing on Bilingual LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15839v1">FASTGEN: Fast and Cost-Effective Synthetic Tabular Data Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Synthetic data generation has emerged as an invaluable solution in scenarios where real-world data collection and usage are limited by cost and scarcity. Large language models (LLMs) have demonstrated remarkable capabilities in producing high-fidelity, domain-relevant samples across various fields. However, existing approaches that directly use LLMs to generate each record individually impose prohibitive time and cost burdens, particularly when large volumes of synthetic data are required. In this work, we propose a fast, cost-effective method for realistic tabular data synthesis that leverages LLMs to infer and encode each field's distribution into a reusable sampling script. By automatically classifying fields into numerical, categorical, or free-text types, the LLM generates distribution-based scripts that can efficiently produce diverse, realistic datasets at scale without continuous model inference. Experimental results show that our approach outperforms traditional direct methods in both diversity and data realism, substantially reducing the burden of high-volume synthetic data generation. We plan to apply this methodology to accelerate testing in production pipelines, thereby shortening development cycles and improving overall system efficiency. We believe our insights and lessons learned will aid researchers and practitioners seeking scalable, cost-effective solutions for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15828v1">Investigating the Use of LLMs for Evidence Briefings Generation in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ESEM 2025 Registered Report with an IPA (In Principle Acceptance) for the Empirical Software Engineering journal
    </div>
    <details class="paper-abstract">
      [Context] An evidence briefing is a concise and objective transfer medium that can present the main findings of a study to software engineers in the industry. Although practitioners and researchers have deemed Evidence Briefings useful, their production requires manual labor, which may be a significant challenge to their broad adoption. [Goal] The goal of this registered report is to describe an experimental protocol for evaluating LLM-generated evidence briefings for secondary studies in terms of content fidelity, ease of understanding, and usefulness, as perceived by researchers and practitioners, compared to human-made briefings. [Method] We developed an RAG-based LLM tool to generate evidence briefings. We used the tool to automatically generate two evidence briefings that had been manually generated in previous research efforts. We designed a controlled experiment to evaluate how the LLM-generated briefings compare to the human-made ones regarding perceived content fidelity, ease of understanding, and usefulness. [Results] To be reported after the experimental trials. [Conclusion] Depending on the experiment results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15821v1">Just Put a Human in the Loop? Investigating LLM-Assisted Annotation for Subjective Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      LLM use in annotation is becoming widespread, and given LLMs' overall promising performance and speed, simply "reviewing" LLM annotations in interpretive tasks can be tempting. In subjective annotation tasks with multiple plausible answers, reviewing LLM outputs can change the label distribution, impacting both the evaluation of LLM performance, and analysis using these labels in a social science task downstream. We conducted a pre-registered experiment with 410 unique annotators and over 7,000 annotations testing three AI assistance conditions against controls, using two models, and two datasets. We find that presenting crowdworkers with LLM-generated annotation suggestions did not make them faster, but did improve their self-reported confidence in the task. More importantly, annotators strongly took the LLM suggestions, significantly changing the label distribution compared to the baseline. When these labels created with LLM assistance are used to evaluate LLM performance, reported model performance significantly increases. We believe our work underlines the importance of understanding the impact of LLM-assisted annotation on subjective, qualitative tasks, on the creation of gold data for training and testing, and on the evaluation of NLP systems on subjective tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15815v1">LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 27 pages, 6 figures, Code: https://github.com/sethkarten/LLM-Economist
    </div>
    <details class="paper-abstract">
      We present the LLM Economist, a novel framework that uses agent-based modeling to design and assess economic policies in strategic environments with hierarchical decision-making. At the lower level, bounded rational worker agents -- instantiated as persona-conditioned prompts sampled from U.S. Census-calibrated income and demographic statistics -- choose labor supply to maximize text-based utility functions learned in-context. At the upper level, a planner agent employs in-context reinforcement learning to propose piecewise-linear marginal tax schedules anchored to the current U.S. federal brackets. This construction endows economic simulacra with three capabilities requisite for credible fiscal experimentation: (i) optimization of heterogeneous utilities, (ii) principled generation of large, demographically realistic agent populations, and (iii) mechanism design -- the ultimate nudging problem -- expressed entirely in natural language. Experiments with populations of up to one hundred interacting agents show that the planner converges near Stackelberg equilibria that improve aggregate social welfare relative to Saez solutions, while a periodic, persona-level voting procedure furthers these gains under decentralized governance. These results demonstrate that large language model-based agents can jointly model, simulate, and govern complex economic systems, providing a tractable test bed for policy evaluation at the societal scale to help build better civilizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15788v1">Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15782v1">Interleaved LLM and Motion Planning for Generalized Multi-Object Collection in Large Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Household robots have been a longstanding research topic, but they still lack human-like intelligence, particularly in manipulating open-set objects and navigating large environments efficiently and accurately. To push this boundary, we consider a generalized multi-object collection problem in large scene graphs, where the robot needs to pick up and place multiple objects across multiple locations in a long mission of multiple human commands. This problem is extremely challenging since it requires long-horizon planning in a vast action-state space under high uncertainties. To this end, we propose a novel interleaved LLM and motion planning algorithm Inter-LLM. By designing a multimodal action cost similarity function, our algorithm can both reflect the history and look into the future to optimize plans, striking a good balance of quality and efficiency. Simulation experiments demonstrate that compared with latest works, our algorithm improves the overall mission performance by 30% in terms of fulfilling human commands, maximizing mission success rates, and minimizing mission costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v3">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15752v1">DialogueForge: LLM Simulation of Human-Chatbot Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 For our code and data, see https://github.com/nerchio/Human_Chatbot-Generation
    </div>
    <details class="paper-abstract">
      Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15717v1">BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15715v1">From Queries to Criteria: Understanding How Astronomers Evaluate LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15613v1">Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place. We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12601v2">CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to KDD 2025 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
    </div>
    <details class="paper-abstract">
      To broaden the dissemination of scientific knowledge to diverse audiences, it is desirable for scientific document summarization systems to simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, the first evaluation benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., conceptual or empirical focus), which are more subjective and abstract. We conduct extensive experiments using various large language models (LLMs) under various settings, including in-context learning, parameter-efficient fine-tuning, and two-stage modular methods for balancing control over different attributes. Our findings reveal significant limitations in LLMs capabilities in balancing trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15585v1">Unequal Voices: How LLMs Construct Constrained Queer Narratives</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      One way social groups are marginalized in discourse is that the narratives told about them often default to a narrow, stereotyped range of topics. In contrast, default groups are allowed the full complexity of human existence. We describe the constrained representations of queer people in LLM generations in terms of harmful representations, narrow representations, and discursive othering and formulate hypotheses to test for these phenomena. Our results show that LLMs are significantly limited in their portrayals of queer personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17181v2">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 13 pages, 8 tables, 2 figures. Paper was previously titled "LLMs Love Python"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate code, influencing users' choices of libraries and programming languages in critical real-world projects. However, little is known about their systematic biases or preferences toward certain libraries and programming languages, which can significantly impact software development practices. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. Our results reveal that LLMs exhibit a strong tendency to overuse widely adopted libraries such as NumPy; in up to 48% of cases, this usage is unnecessary and deviates from the ground-truth solutions. LLMs also exhibit a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used a single time. These results indicate that LLMs may prioritise familiarity and popularity over suitability and task-specific optimality. This will introduce security vulnerabilities and technical debt, and limit exposure to newly developed, better-suited tools and languages. Understanding and addressing these biases is essential for the responsible integration of LLMs into software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05445v2">clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15553v1">Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The rising demand for Large Language Model (LLM) inference services has intensified pressure on computational resources, resulting in latency and cost challenges. This paper introduces a novel routing algorithm based on the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to distribute inference requests across heterogeneous LLM instances in a cloud-edge computing environment. Formulated as a multi-objective optimization problem, the algorithm balances response quality, response time, and inference cost, adapting to request heterogeneity (e.g., varying complexity and prompt lengths) and node diversity (e.g., edge vs. cloud resources). This adaptive routing algorithm optimizes performance under dynamic workloads. We benchmark the approach using a testbed with datasets including Stanford Question Answering Dataset (SQuAD), Mostly Basic Python Problems (MBPP), Hella Situations With Adversarial Generations (HellaSwag), and Grade School Math 8K (GSM8K). Experimental results show our solution, compared to the baselines, achieves up to 95.2% and 34.9% improvements in terms of response time and cost, respectively. These findings validate the algorithm's effectiveness for scalable LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15550v1">PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 Pages
    </div>
    <details class="paper-abstract">
      Evaluating the scientific discovery capabilities of large language model based agents, particularly how they cope with varying environmental complexity and utilize prior knowledge, requires specialized benchmarks currently lacking in the landscape. To address this gap, we introduce PhysGym, a novel benchmark suite and simulation platform for rigorously assessing LLM-based scientific reasoning in interactive physics environments. PhysGym's primary contribution lies in its sophisticated control over the level of prior knowledge provided to the agent. This allows researchers to dissect agent performance along axes including the complexity of the problem and the prior knowledge levels. The benchmark comprises a suite of interactive simulations, where agents must actively probe environments, gather data sequentially under constraints and formulate hypotheses about underlying physical laws. PhysGym provides standardized evaluation protocols and metrics for assessing hypothesis accuracy and model fidelity. We demonstrate the benchmark's utility by presenting results from baseline LLMs, showcasing its ability to differentiate capabilities based on varying priors and task complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00024v2">Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ACL 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, human judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze whether 11 LLMs behave like humans in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15521v1">LLM world models are mental: Output layer evidence of brittle world model use in LLM mechanical reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Manuscript comprises 14 pages, 4 figures, 4 tables in the Technical Appendix and Supplementary Material, and is under review at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) construct and manipulate internal world models, or do they rely solely on statistical associations represented as output layer token probabilities? We adapt cognitive science methodologies from human mental models research to test LLMs on pulley system problems using TikZ-rendered stimuli. Study 1 examines whether LLMs can estimate mechanical advantage (MA). State-of-the-art models performed marginally but significantly above chance, and their estimates correlated significantly with ground-truth MA. Significant correlations between number of pulleys and model estimates suggest that models employed a pulley counting heuristic, without necessarily simulating pulley systems to derive precise values. Study 2 tested this by probing whether LLMs represent global features crucial to MA estimation. Models evaluated a functionally connected pulley system against a fake system with randomly placed components. Without explicit cues, models identified the functional system as having greater MA with F1=0.8, suggesting LLMs could represent systems well enough to differentiate jumbled from functional systems. Study 3 built on this by asking LLMs to compare functional systems with matched systems which were connected up but which transferred no force to the weight; LLMs identified the functional system with F1=0.46, suggesting random guessing. Insofar as they may generalize, these findings are compatible with the notion that LLMs manipulate internal world models, sufficient to exploit statistical associations between pulley count and MA (Study 1), and to approximately represent system components' spatial relations (Study 2). However, they may lack the facility to reason over nuanced structural connectivity (Study 3). We conclude by advocating the utility of cognitive scientific methods to evaluate the world-modeling capacities of artificial intelligence systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15502v1">FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Postoperative follow-up plays a crucial role in monitoring recovery and identifying complications. However, traditional approaches, typically involving bedside interviews and manual documentation, are time-consuming and labor-intensive. Although existing digital solutions, such as web questionnaires and intelligent automated calls, can alleviate the workload of nurses to a certain extent, they either deliver an inflexible scripted interaction or face private information leakage issues. To address these limitations, this paper introduces FollowUpBot, an LLM-powered edge-deployed robot for postoperative care and monitoring. It allows dynamic planning of optimal routes and uses edge-deployed LLMs to conduct adaptive and face-to-face conversations with patients through multiple interaction modes, ensuring data privacy. Moreover, FollowUpBot is capable of automatically generating structured postoperative follow-up reports for healthcare institutions by analyzing patient interactions during follow-up. Experimental results demonstrate that our robot achieves high coverage and satisfaction in follow-up interactions, as well as high report generation accuracy across diverse field types. The demonstration video is available at https://www.youtube.com/watch?v=_uFgDO7NoK0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15296v1">Butterfly Effects in Toolchains: A Comprehensive Analysis of Failed Parameter Filling in LLM Tool-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The emergence of the tool agent paradigm has broadened the capability boundaries of the Large Language Model (LLM), enabling it to complete more complex tasks. However, the effectiveness of this paradigm is limited due to the issue of parameter failure during its execution. To explore this phenomenon and propose corresponding suggestions, we first construct a parameter failure taxonomy in this paper. We derive five failure categories from the invocation chain of a mainstream tool agent. Then, we explore the correlation between three different input sources and failure categories by applying 15 input perturbation methods to the input. Experimental results show that parameter name hallucination failure primarily stems from inherent LLM limitations, while issues with input sources mainly cause other failure patterns. To improve the reliability and effectiveness of tool-agent interactions, we propose corresponding improvement suggestions, including standardizing tool return formats, improving error feedback mechanisms, and ensuring parameter consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06273v2">Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted at ICCV 2025. Code available at: https://github.com/JeongHun0716/zero-avsr
    </div>
    <details class="paper-abstract">
      We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15268v1">IM-Chat: A Multi-agent LLM-based Framework for Knowledge Transfer in Injection Molding Industry</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The injection molding industry faces critical challenges in preserving and transferring field knowledge, particularly as experienced workers retire and multilingual barriers hinder effective communication. This study introduces IM-Chat, a multi-agent framework based on large language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM-Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and extensive field data modeled through a data-driven process condition generator that infers optimal manufacturing settings from environmental inputs such as temperature and humidity, enabling robust and context-aware task resolution. By adopting a retrieval-augmented generation (RAG) strategy and tool-calling agents within a modular architecture, IM-Chat ensures adaptability without the need for fine-tuning. Performance was assessed across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance and correctness, and was further supplemented by automated evaluation using GPT-4o guided by a domain-adapted instruction prompt. The evaluation results indicate that more capable models tend to achieve higher accuracy, particularly in complex, tool-integrated scenarios. Overall, these findings demonstrate the viability of multi-agent LLM systems for industrial knowledge workflows and establish IM-Chat as a scalable and generalizable approach to AI-assisted decision support in manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v2">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14928v1">Byzantine-Robust Decentralized Coordination of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential. Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes. To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques. Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14906v1">Feedback-Induced Performance Decline in LLM-Based Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The ability of Large Language Models (LLMs) to extract context from natural language problem descriptions naturally raises questions about their suitability in autonomous decision-making settings. This paper studies the behaviour of these models within a Markov Decision Process (MDPs). While traditional reinforcement learning (RL) strategies commonly employed in this setting rely on iterative exploration, LLMs, pre-trained on diverse datasets, offer the capability to leverage prior knowledge for faster adaptation. We investigate online structured prompting strategies in sequential decision making tasks, comparing the zero-shot performance of LLM-based approaches to that of classical RL methods. Our findings reveal that although LLMs demonstrate improved initial performance in simpler environments, they struggle with planning and reasoning in complex scenarios without fine-tuning or additional guidance. Our results show that feedback mechanisms, intended to improve decision-making, often introduce confusion, leading to diminished performance in intricate environments. These insights underscore the need for further exploration into hybrid strategies, fine-tuning, and advanced memory integration to enhance LLM-based decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14894v1">Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in four cases. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14799v1">Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 EMNLP 2025 System Demonstrations Submission
    </div>
    <details class="paper-abstract">
      This work demonstrates that LLM-based web navigation agents offer powerful automation capabilities but are vulnerable to Indirect Prompt Injection (IPI) attacks. We show that adversaries can embed universal adversarial triggers in webpage HTML to hijack agent behavior that utilizes the accessibility tree to parse HTML, causing unintended or malicious actions. Using the Greedy Coordinate Gradient (GCG) algorithm and a Browser Gym agent powered by Llama-3.1, our system demonstrates high success rates across real websites in both targeted and general attacks, including login credential exfiltration and forced ad clicks. Our empirical results highlight critical security risks and the need for stronger defenses as LLM-driven autonomous web agents become more widely adopted. The system software (https://github.com/sej2020/manipulating-web-agents) is released under the MIT License, with an accompanying publicly available demo website (http://lethaiq.github.io/attack-web-llm-agent).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09971v3">Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Object-goal navigation requires mobile robots to efficiently locate targets with visual and spatial information, yet existing methods struggle with generalization in unseen environments. Heuristic approaches with naive metrics fail in complex layouts, while graph-based and learning-based methods suffer from environmental biases and limited generalization. Although Large Language Models (LLMs) as planners or agents offer a rich knowledge base, they are cost-inefficient and lack targeted historical experience. To address these challenges, we propose the LLM-enhanced Object Affinities Transfer (LOAT) framework, integrating LLM-derived semantics with learning-based approaches to leverage experiential object affinities for better generalization in unseen settings. LOAT employs a dual-module strategy: one module accesses LLMs' vast knowledge, and the other applies learned object semantic relationships, dynamically fusing these sources based on context. Evaluations in AI2-THOR and Habitat simulators show significant improvements in navigation success and efficiency, and real-world deployment demonstrates the zero-shot ability of LOAT to enhance object-goal navigation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11704v2">A Library of LLM Intrinsics for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 This (June 2025) is the second version of this paper (the first was published in April 2025). Intrinsics implemented as LoRAs are now trained on IBM Granite 3.3 8b instruct (previously 3.2)
    </div>
    <details class="paper-abstract">
      In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14785v1">Exploring the In-Context Learning Capabilities of LLMs for Money Laundering Detection in Financial Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The complexity and interconnectivity of entities involved in money laundering demand investigative reasoning over graph-structured data. This paper explores the use of large language models (LLMs) as reasoning engines over localized subgraphs extracted from a financial knowledge graph. We propose a lightweight pipeline that retrieves k-hop neighborhoods around entities of interest, serializes them into structured text, and prompts an LLM via few-shot in-context learning to assess suspiciousness and generate justifications. Using synthetic anti-money laundering (AML) scenarios that reflect common laundering behaviors, we show that LLMs can emulate analyst-style logic, highlight red flags, and provide coherent explanations. While this study is exploratory, it illustrates the potential of LLM-based graph reasoning in AML and lays groundwork for explainable, language-driven financial crime analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14784v1">LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Video Question Answering (VideoQA) requires identifying sparse critical moments in long videos and reasoning about their causal relationships to answer semantically complex questions. While recent advances in multimodal learning have improved alignment and fusion, current approaches remain limited by two prevalent but fundamentally flawed strategies: (1) task-agnostic sampling indiscriminately processes all frames, overwhelming key events with irrelevant content; and (2) heuristic retrieval captures superficial patterns but misses causal-temporal structures needed for complex reasoning. To address these challenges, we introduce LeAdQA, an innovative approach that bridges these gaps through synergizing causal-aware query refinement with fine-grained visual grounding. Our method first leverages LLMs to reformulate question-option pairs, resolving causal ambiguities and sharpening temporal focus. These refined queries subsequently direct a temporal grounding model to precisely retrieve the most salient segments, complemented by an adaptive fusion mechanism dynamically integrating the evidence to maximize relevance. The integrated visual-textual cues are then processed by an MLLM to generate accurate, contextually-grounded answers. Experiments on NExT-QA, IntentQA, and NExT-GQA demonstrate that our method's precise visual grounding substantially enhances the understanding of video-question relationships, achieving state-of-the-art (SOTA) performance on complex reasoning tasks while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14783v1">Omni-Think: Scaling Cross-Domain Generalization in LLMs via Multi-Task RL with Hybrid Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The advancement of general-purpose artificial intelligence relies on large language models (LLMs) that excel across a wide range of tasks, from structured reasoning to creative generation. However, post-training methods like Supervised Fine-Tuning (SFT) often struggle with generalization, favoring memorization over transferable learning. In this work, we introduce Omni-Think, a unified reinforcement learning (RL) framework that enhances LLM performance across diverse tasks by combining rule-based verifiable rewards with generative preference signals via LLM-as-a-Judge evaluations. Our approach enables consistent optimization across task types and scales RL-based training to subjective domains. We further investigate training strategies, demonstrating that a curriculum-based progression that orders tasks from structured to open-ended improves performance and reduces forgetting. Experimental results across four domains reveal that curriculum learning improves performance by 5.2\% over joint training and 9.1\% over model merging. These results highlight the importance of task-aware sampling and hybrid supervision in scaling RL-based post-training for general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14776v1">VeriOpt: PPA-Aware High-Quality Verilog Generation via Multi-Role LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 9 pages, 7 figures, Accepted for ICCAD 2025, Munich, Germany
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models(LLMs) in hardware design has primarily focused on generating functionally correct Verilog code, overlooking critical Power Performance-Area(PPA) metrics essential for industrial-grade designs. To bridge this gap, we propose VeriOpt, a novel framework that leverages role-based prompting and PPA-aware optimization to enable LLMs to produce high-quality, synthesizable Verilog. VeriOpt structures LLM interactions into specialized roles (e.g., Planner, Programmer, Reviewer, Evaluator) to emulate human design workflows, while integrating PPA constraints directly into the prompting pipeline. By combining multi-modal feedback (e.g., synthesis reports, timing diagrams) with PPA aware prompting, VeriOpt achieves PPA-efficient code generation without sacrificing functional correctness. Experimental results demonstrate up to 88% reduction in power, 76% reduction in area and 73% improvement in timing closure compared to baseline LLM-generated RTL, validated using industry standard EDA tools. At the same time achieves 86% success rate in functionality evaluation. Our work advances the state-of-the-art AI-driven hardware design by addressing the critical gap between correctness and quality, paving the way for reliable LLM adoption in production workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15157v1">Can LLMs Generate User Stories and Assess Their Quality?</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Requirements elicitation is still one of the most challenging activities of the requirements engineering process due to the difficulty requirements analysts face in understanding and translating complex needs into concrete requirements. In addition, specifying high-quality requirements is crucial, as it can directly impact the quality of the software to be developed. Although automated tools allow for assessing the syntactic quality of requirements, evaluating semantic metrics (e.g., language clarity, internal consistency) remains a manual and time-consuming activity. This paper explores how LLMs can help automate requirements elicitation within agile frameworks, where requirements are defined as user stories (US). We used 10 state-of-the-art LLMs to investigate their ability to generate US automatically by emulating customer interviews. We evaluated the quality of US generated by LLMs, comparing it with the quality of US generated by humans (domain experts and students). We also explored whether and how LLMs can be used to automatically evaluate the semantic quality of US. Our results indicate that LLMs can generate US similar to humans in terms of coverage and stylistic quality, but exhibit lower diversity and creativity. Although LLM-generated US are generally comparable in quality to those created by humans, they tend to meet the acceptance quality criteria less frequently, regardless of the scale of the LLM model. Finally, LLMs can reliably assess the semantic quality of US when provided with clear evaluation criteria and have the potential to reduce human effort in large-scale assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12899v3">A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 13 pages, 7 figure, 8 tables, under peer-review
    </div>
    <details class="paper-abstract">
      Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose Semantic Targeting for Analytical Repair (STAR), a pioneering and novel semantic-based optimization approach for repairing LLMs. STAR realizes the main operations of repairing LMs in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (MINT) and optimization methods (SGD), STAR integrates their strengths while mitigating their limitations. STAR supports solving multiple failures together, significantly improving the usefulness. Evaluated on coding tasks using popular code LMs, STAR exhibits superior effectiveness (10.5%-19.9% improvements) and efficiency (2.4-7.0 times speedup). In terms of side effects, namely the balance between generalization and specificity, STAR outperforms prior work by a significant margin. Additionally, we conducted assessments on the overfitting risk of LM repair as well as the cumulative impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11858v2">OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v1">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 Under review. 19 pages, 8 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15058v1">LibLMFuzz: LLM-Augmented Fuzz Target Generation for Black-box Libraries</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 6 pages, 2 figures, 1 table, 2 listings
    </div>
    <details class="paper-abstract">
      A fundamental problem in cybersecurity and computer science is determining whether a program is free of bugs and vulnerabilities. Fuzzing, a popular approach to discovering vulnerabilities in programs, has several advantages over alternative strategies, although it has investment costs in the form of initial setup and continuous maintenance. The choice of fuzzing is further complicated when only a binary library is available, such as the case of closed-source and proprietary software. In response, we introduce LibLMFuzz, a framework that reduces costs associated with fuzzing closed-source libraries by pairing an agentic Large Language Model (LLM) with a lightweight tool-chain (disassembler/compiler/fuzzer) to autonomously analyze stripped binaries, plan fuzz strategies, generate drivers, and iteratively self-repair build or runtime errors. Tested on four widely-used Linux libraries, LibLMFuzz produced syntactically correct drivers for all 558 fuzz-able API functions, achieving 100% API coverage with no human intervention. Across the 1601 synthesized drivers, 75.52% were nominally correct on first execution. The results show that LLM-augmented middleware holds promise in reducing the costs of fuzzing black box components and provides a foundation for future research efforts. Future opportunities exist for research in branch coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15049v1">Beyond Visual Line of Sight: UAVs with Edge AI, Connected LLMs, and VR for Autonomous Aerial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles are reshaping Non-Terrestrial Networks by acting as agile, intelligent nodes capable of advanced analytics and instantaneous situational awareness. This article introduces a budget-friendly quadcopter platform that unites 5G communications, edge-based processing, and AI to tackle core challenges in NTN scenarios. Outfitted with a panoramic camera, robust onboard computation, and LLMs, the drone system delivers seamless object recognition, contextual analysis, and immersive operator experiences through virtual reality VR technology. Field evaluations confirm the platform's ability to process visual streams with low latency and sustain robust 5G links. Adding LLMs further streamlines operations by extracting actionable insights and refining collected data for decision support. Demonstrated use cases, including emergency response, infrastructure assessment, and environmental surveillance, underscore the system's adaptability in demanding contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15015v1">EduThink4AI: Translating Educational Critical Thinking into Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential as educational tutoring agents, capable of tailoring hints, orchestrating lessons, and grading with near-human finesse across various academic domains. However, current LLM-based educational systems exhibit critical limitations in promoting genuine critical thinking, failing on over one-third of multi-hop questions with counterfactual premises, and remaining vulnerable to adversarial prompts that trigger biased or factually incorrect responses. To address these gaps, we propose EDU-Prompting, a novel multi-agent framework that bridges established educational critical thinking theories with LLM agent design to generate critical, bias-aware explanations while fostering diverse perspectives. Our systematic evaluation across theoretical benchmarks and practical college-level critical writing scenarios demonstrates that EDU-Prompting significantly enhances both content truthfulness and logical soundness in AI-generated educational responses. The framework's modular design enables seamless integration into existing prompting frameworks and educational applications, allowing practitioners to directly incorporate critical thinking catalysts that promote analytical reasoning and introduce multiple perspectives without requiring extensive system modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14642v2">How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14995v1">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and lack of customized modeling frameworks. This paper proposed an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategy, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation learning. A differential attention-based critic network is designed to enhance convergence performance. Experimental results demonstrate that LLM generated strategies effectively substitute human experts. The proposed multi-agent imitation learning algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baselines algorithms, while maintaining robust stability. This work provides an effective solution for real-time P2P electricity market decision-making by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10467v4">Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 This work has been submitted for publication. Copyright may be transferred. In this case, this version will be updated with a notice, according to the publisher's guidelines
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs indicate potential for novel applications, as evidenced by the reasoning capabilities in the latest OpenAI and DeepSeek models. To apply these models to domain-specific applications beyond text generation, LLM-based multi-agent systems can be utilized to solve complex tasks, particularly by combining reasoning techniques, code generation, and software execution across multiple, potentially specialized LLMs. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application are not well understood. Defined specifications for multi-agent LLM systems are required to explore their potential and suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research on (1.) multi-agent specification by introducing an agent schema language and (2.) the execution and evaluation of the specifications through a multi-agent system architecture and prototype. The specification language, system architecture, and prototype are first presented in this work, building on an LLM system from prior research. Test cases involving cybersecurity tasks indicate the feasibility of the architecture and evaluation approach. As a result, evaluations could be demonstrated for question answering, server security, and network security tasks completed correctly by agents with LLMs from OpenAI and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14735v1">Investigating the Role of LLMs Hyperparameter Tuning and Prompt Engineering to Support Domain Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted at 51st Euromicro Conference Series on Software Engineering and Advanced Applications (SEAA)
    </div>
    <details class="paper-abstract">
      The introduction of large language models (LLMs) has enhanced automation in software engineering tasks, including in Model Driven Engineering (MDE). However, using general-purpose LLMs for domain modeling has its limitations. One approach is to adopt fine-tuned models, but this requires significant computational resources and can lead to issues like catastrophic forgetting. This paper explores how hyperparameter tuning and prompt engineering can improve the accuracy of the Llama 3.1 model for generating domain models from textual descriptions. We use search-based methods to tune hyperparameters for a specific medical data model, resulting in a notable quality improvement over the baseline LLM. We then test the optimized hyperparameters across ten diverse application domains. While the solutions were not universally applicable, we demonstrate that combining hyperparameter tuning with prompt engineering can enhance results across nearly all examined domain models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14719v1">Automated Safety Evaluations Across 20 Large Language Models: The Aymara LLM Risk and Responsibility Matrix</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into real-world applications, scalable and rigorous safety evaluation is essential. This paper introduces Aymara AI, a programmatic platform for generating and administering customized, policy-grounded safety evaluations. Aymara AI transforms natural-language safety policies into adversarial prompts and scores model responses using an AI-based rater validated against human judgments. We demonstrate its capabilities through the Aymara LLM Risk and Responsibility Matrix, which evaluates 20 commercially available LLMs across 10 real-world safety domains. Results reveal wide performance disparities, with mean safety scores ranging from 86.2% to 52.4%. While models performed well in well-established safety domains such as Misinformation (mean = 95.7%), they consistently failed in more complex or underspecified domains, notably Privacy & Impersonation (mean = 24.3%). Analyses of Variance confirmed that safety scores differed significantly across both models and domains (p < .05). These findings underscore the inconsistent and context-dependent nature of LLM safety and highlight the need for scalable, customizable tools like Aymara AI to support responsible AI development and oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14705v1">Configurable multi-agent framework for scalable and realistic testing of llm-based agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete. We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn. Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts. Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08263v2">LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Tangled code changes, commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements, introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains underexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-4o and Gemini-2.0-Flash. Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we explore machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). Applying our approach to 49 open-source projects improves the distributional separability of code metrics between buggy and non-buggy methods, demonstrating the promise of LLMs for method-level commit untangling and potentially contributing to improving the accuracy of future bug prediction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v3">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v3">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite improvements in automatic speech recognition, performance drops with accented speech. Generative error correction (GER) leverages the linguistic knowledge of large language models (LLMs), outperforming typical language model methods. However, it lacks specificity in accented speech scenarios. Accents represent deviations from standard pronunciation, making multi-granularity pronunciation and semantic information essential for accented speech recognition. Moreover, accents exhibit considerable diversity, with each accent possessing distinct characteristics. In this study, we leverage GER to improve transcription accuracy by addressing the two primary features. We propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level pronunciation information. These methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through low-rank adaptation (LoRA) fine-tuning. We employ a three-stage strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge mono-accent LoRA experts within a single multi-modal GER to overcome accent diversity challenges. Furthermore, multi-granularity GER leverages N-best word-level and phoneme-level hypotheses from the HDMoLE model to predict final transcriptions. Experiments on a multi-accent English dataset show that our methods reduce word error rate by 67.35% compared to the baseline vanilla Whisper-large-v3 model.
    </details>
</div>
