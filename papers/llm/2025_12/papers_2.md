# llm - 2025_12

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16917v1">Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16914v1">Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 18 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Prior studies investigating the internal workings of LLMs have uncovered sparse subnetworks, often referred to as circuits, that are responsible for performing specific tasks. Additionally, it has been shown that model performance improvement through fine-tuning often results from the strengthening of existing circuits in the model. Taken together, these findings suggest the possibility of intervening directly on such circuits to make precise, task-targeted updates. Motivated by these findings, we propose a novel method called Constructive Circuit Amplification which identifies pivotal tokens from model reasoning traces as well as model components responsible for the desired task, and updates only those components. Applied to mathematical reasoning, it improves accuracy by up to +11.4% across multiple models while modifying as little as 1.59% of model components, with minimal impact on other abilities as measured by MMLU, TriviaQA, and TruthfulQA. These results demonstrate that targeted capabilities can be reliably enhanced by selectively updating a sparse set of model components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16822v1">MEPIC: Memory Efficient Position Independent Caching for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Modern LLM applications such as deep-research assistants, coding agents, and Retrieval-Augmented Generation (RAG) systems, repeatedly process long prompt histories containing shared document or code chunks, creating significant pressure on the Key Value (KV) cache, which must operate within limited memory while sustaining high throughput and low latency. Prefix caching partially alleviates some of these costs by reusing KV cache for previously processed tokens, but limited by strict prefix matching. Position-independent caching (PIC) enables chunk-level reuse at arbitrary positions, but requires selective recomputation and positional-encoding (PE) adjustments. However, because these operations vary across queries, KV for the same chunk diverges across requests. Moreover, without page alignment, chunk KV layouts diverge in memory, preventing page sharing. These issues result in only modest HBM savings even when many requests reuse the same content. We present MEPIC, a memory-efficient PIC system that enables chunk KV reuse across positions, requests, and batches. MEPIC aligns chunk KV to paged storage, shifts recomputation from token- to block-level so only the first block is request-specific, removes positional encodings via Rotary Position Embedding (RoPE) fusion in the attention kernel, and makes remaining blocks fully shareable. These techniques eliminate most duplicate chunk KV in HBM, reducing usage by up to 2x over state-of-the-art PIC at comparable latency and accuracy, and up to 5x for long prompts, without any model changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16795v1">From Facts to Conclusions : Integrating Deductive Reasoning in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external evidence, but fails when retrieved sources conflict or contain outdated or subjective information. Prior work address these issues independently but lack unified reasoning supervision. We propose a reasoning-trace-augmented RAG framework that adds structured, interpretable reasoning across three stages : (1) document-level adjudication, (2) conflict analysis, and (3) grounded synthesis, producing citation-linked answers or justified refusals. A Conflict-Aware Trust-Score (CATS) pipeline is introduced which evaluates groundedness, factual correctness, refusal accuracy, and conflict-behavior alignment using an LLM-as-a-Judge. Our 539-query reasoning dataset and evaluation pipeline establish a foundation for conflict-aware, interpretable RAG systems. Experimental results demonstrate substantial gains over baselines, most notably with Qwen, where Supervised Fine-Tuning improved End-to-End answer correctness from 0.069 to 0.883 and behavioral adherence from 0.074 to 0.722.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16790v1">Inside Out: Uncovering How Comment Internalization Steers LLMs for Better or Worse</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Accepted in the 48th IEEE/ACM International Conference on Software Engineering (ICSE)
    </div>
    <details class="paper-abstract">
      While comments are non-functional elements of source code, Large Language Models (LLM) frequently rely on them to perform Software Engineering (SE) tasks. Yet, where in the model this reliance resides, and how it affects performance, remains poorly understood. We present the first concept-level interpretability study of LLMs in SE, analyzing three tasks - code completion, translation, and refinement - through the lens of internal comment representation. Using Concept Activation Vectors (CAV), we show that LLMs not only internalize comments as distinct latent concepts but also differentiate between subtypes such as Javadocs, inline, and multiline comments. By systematically activating and deactivating these concepts in the LLMs' embedding space, we observed significant, model-specific, and task-dependent shifts in performance ranging from -90% to +67%. Finally, we conducted a controlled experiment using the same set of code inputs, prompting LLMs to perform 10 distinct SE tasks while measuring the activation of the comment concept within their latent representations. We found that code summarization consistently triggered the strongest activation of comment concepts, whereas code completion elicited the weakest sensitivity. These results open a new direction for building SE tools and models that reason about and manipulate internal concept representations rather than relying solely on surface-level input.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16750v1">Plausibility as Failure: How LLMs and Humans Co-Construct Epistemic Error</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 19 pages, 2 tables, 77 references, 6 appendices
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as epistemic partners in everyday reasoning, yet their errors remain predominantly analyzed through predictive metrics rather than through their interpretive effects on human judgment. This study examines how different forms of epistemic failure emerge, are masked, and are tolerated in human AI interaction, where failure is understood as a relational breakdown shaped by model-generated plausibility and human interpretive judgment. We conducted a three round, multi LLM evaluation using interdisciplinary tasks and progressively differentiated assessment frameworks to observe how evaluators interpret model responses across linguistic, epistemic, and credibility dimensions. Our findings show that LLM errors shift from predictive to hermeneutic forms, where linguistic fluency, structural coherence, and superficially plausible citations conceal deeper distortions of meaning. Evaluators frequently conflated criteria such as correctness, relevance, bias, groundedness, and consistency, indicating that human judgment collapses analytical distinctions into intuitive heuristics shaped by form and fluency. Across rounds, we observed a systematic verification burden and cognitive drift. As tasks became denser, evaluators increasingly relied on surface cues, allowing erroneous yet well formed answers to pass as credible. These results suggest that error is not solely a property of model behavior but a co-constructed outcome of generative plausibility and human interpretive shortcuts. Understanding AI epistemic failure therefore requires reframing evaluation as a relational interpretive process, where the boundary between system failure and human miscalibration becomes porous. The study provides implications for LLM assessment, digital literacy, and the design of trustworthy human AI communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16676v1">DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The rapidly growing demand for high-quality data in Large Language Models (LLMs) has intensified the need for scalable, reliable, and semantically rich data preparation pipelines. However, current practices remain dominated by ad-hoc scripts and loosely specified workflows, which lack principled abstractions, hinder reproducibility, and offer limited support for model-in-the-loop data generation. To address these challenges, we present DataFlow, a unified and extensible LLM-driven data preparation framework. DataFlow is designed with system-level abstractions that enable modular, reusable, and composable data transformations, and provides a PyTorch-style pipeline construction API for building debuggable and optimizable dataflows. The framework consists of nearly 200 reusable operators and six domain-general pipelines spanning text, mathematical reasoning, code, Text-to-SQL, agentic RAG, and large-scale knowledge extraction. To further improve usability, we introduce DataFlow-Agent, which automatically translates natural-language specifications into executable pipelines via operator synthesis, pipeline planning, and iterative verification. Across six representative use cases, DataFlow consistently improves downstream LLM performance. Our math, code, and text pipelines outperform curated human datasets and specialized synthetic baselines, achieving up to +3\% execution accuracy in Text-to-SQL over SynSQL, +7\% average improvements on code benchmarks, and 1--3 point gains on MATH, GSM8K, and AIME. Moreover, a unified 10K-sample dataset produced by DataFlow enables base models to surpass counterparts trained on 1M Infinity-Instruct data. These results demonstrate that DataFlow provides a practical and high-performance substrate for reliable, reproducible, and scalable LLM data preparation, and establishes a system-level foundation for future data-centric AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20172v6">Evaluating and Mitigating Errors in LLM-Generated Web API Integrations</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Extended journal version of our paper published at AIware 2025 (arXiv:2509.20172v5)
    </div>
    <details class="paper-abstract">
      API integration is a cornerstone of our digital infrastructure, enabling software systems to connect and interact. However, as shown by many studies, writing or generating correct code to invoke APIs, particularly web APIs, is challenging. Although large language models (LLMs) have become popular in software development, their effectiveness in automating the generation of web API integration code remains unexplored. In order to address this, we present WAPIIBench, a dataset and evaluation pipeline designed to assess the ability of LLMs to generate web API invocation code. Our experiments with several open-source LLMs reveal that generating API invocations poses a significant challenge, resulting in hallucinated endpoints, incorrect argument usage, and other errors. None of the evaluated open-source models was able to solve more than 40% of the tasks. Motivated by those findings, we explore the potential of constrained decoding for generating API invocations. To this end, we propose an automatic translation from API specifications to constraints. Our approach prevents violations of API usage rules and significantly increases the overall correctness of the generated code, on average by 90% and 135%, depending on the provided starter code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04133v5">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around key pillars: \textit{ Explainability, ModelOps, Security, Privacy} and \textit{their Lifecycle Governance}, each contextualized to the challenges of AMAS. A risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI, as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, highlighting key directions to align emerging systems with TRiSM principles-ensuring safety, transparency, and accountability in their operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16602v1">Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16145v2">SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8977 (95% CI: 0.88-0.91). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16538v1">A Systematic Study of Code Obfuscation Against LLM-based Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly adopted for code vulnerability detection, their reliability and robustness across diverse vulnerability types have become a pressing concern. In traditional adversarial settings, code obfuscation has long been used as a general strategy to bypass auditing tools, preserving exploitability without tampering with the tools themselves. Numerous efforts have explored obfuscation methods and tools, yet their capabilities differ in terms of supported techniques, granularity, and programming languages, making it difficult to systematically assess their impact on LLM-based vulnerability detection. To address this gap, we provide a structured systematization of obfuscation techniques and evaluate them under a unified framework. Specifically, we categorize existing obfuscation methods into three major classes (layout, data flow, and control flow) covering 11 subcategories and 19 concrete techniques. We implement these techniques across four programming languages (Solidity, C, C++, and Python) using a consistent LLM-driven approach, and evaluate their effects on 15 LLMs spanning four model families (DeepSeek, OpenAI, Qwen, and LLaMA), as well as on two coding agents (GitHub Copilot and Codex). Our findings reveal both positive and negative impacts of code obfuscation on LLM-based vulnerability detection, highlighting conditions under which obfuscation leads to performance improvements or degradations. We further analyze these outcomes with respect to vulnerability characteristics, code properties, and model attributes. Finally, we outline several open problems and propose future directions to enhance the robustness of LLMs for real-world vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16530v1">Plain language adaptations of biomedical text using LLMs: Comparision of evaluation metrics</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This study investigated the application of Large Language Models (LLMs) for simplifying biomedical texts to enhance health literacy. Using a public dataset, which included plain language adaptations of biomedical abstracts, we developed and evaluated several approaches, specifically a baseline approach using a prompt template, a two AI agent approach, and a fine-tuning approach. We selected OpenAI gpt-4o and gpt-4o mini models as baselines for further research. We evaluated our approaches with quantitative metrics, such as Flesch-Kincaid grade level, SMOG Index, SARI, and BERTScore, G-Eval, as well as with qualitative metric, more precisely 5-point Likert scales for simplicity, accuracy, completeness, brevity. Results showed a superior performance of gpt-4o-mini and an underperformance of FT approaches. G-Eval, a LLM based quantitative metric, showed promising results, ranking the approaches similarly as the qualitative metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16473v1">Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 7 pages, 6 figures, to be published in ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive results across various tasks, yet their high computational demands pose deployment challenges, especially on consumer-grade hardware. Mixture of Experts (MoE) models provide an efficient solution through selective activation of parameter subsets, which reduces computation requirements. Despite this efficiency, state-of-the-art MoE models still require substantial memory beyond typical consumer GPU capacities. Traditional offloading methods that transfer model weights between CPU and GPU introduce latency, limiting inference performance. This paper presents a novel CPU-GPU collaborative inference framework that incorporates an expert caching mechanism on the GPU to reduce data transfer requirements and enable faster inference through cache hits. Computations are offloaded to CPU for efficient cache miss handling, which benefits from CPU multithreading optimizations. The evaluations of our framework demonstrate performance improvements and highlight the potential of CPU-GPU collaboration to maximize hardware utilization for single-request inference scenarios on consumer-grade systems. The implementation of our framework is available at https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17361v2">Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA). We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v24">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16424v1">Synthelite: Chemist-aligned and feasibility-aware synthesis planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Computer-aided synthesis planning (CASP) has long been envisioned as a complementary tool for synthetic chemists. However, existing frameworks often lack mechanisms to allow interaction with human experts, limiting their ability to integrate chemists' insights. In this work, we introduce Synthelite, a synthesis planning framework that uses large language models (LLMs) to directly propose retrosynthetic transformations. Synthelite can generate end-to-end synthesis routes by harnessing the intrinsic chemical knowledge and reasoning capabilities of LLMs, while allowing expert intervention through natural language prompts. Our experiments demonstrate that Synthelite can flexibly adapt its planning trajectory to diverse user-specified constraints, achieving up to 95\% success rates in both strategy-constrained and starting-material-constrained synthesis tasks. Additionally, Synthelite exhibits the ability to account for chemical feasibility during route design. We envision Synthelite to be both a useful tool and a step toward a paradigm where LLMs are the central orchestrators of synthesis planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13302v3">LLM one-shot style transfer for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Existing supervised and contrastive approaches often rely on datasets with spurious correlations, conflating style with topic. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in general authorship analysis. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive CLM pre-training and in-context capabilities of modern LLMs. Our approach avoids explicit supervision with spuriously correlated data. Our method substantially outperforms unsupervised prompting-based baselines at similar model sizes and exceeds contrastively trained models when controlling for topical overlap. Our framework's performance improves with model size. In the case of authorship verification, we present an additional mechanism that increases test-time computation to improve accuracy; enabling flexible trade-offs between computational cost and task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08158v3">Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16391v1">Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 11 pages, 8 figures, 3 tables and 1 algorithm
    </div>
    <details class="paper-abstract">
      Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight keys is stable across nearby layers. Kascade computes exact Top-k indices in a small set of anchor layers, then reuses those indices in intermediate reuse layers. The anchor layers are selected algorithmically, via a dynamic-programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across both prefill and decode attention. The Top-k selection and reuse in Kascade is head-aware and we show in our experiments that this is critical for high accuracy. Kascade achieves up to 4.1x speedup in decode attention and 2.2x speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense attention accuracy on long-context benchmarks such as LongBench and AIME-24.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16378v1">Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Project available at https://github.com/sarapapi/hearing2translate
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10795v3">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01396v2">Easy Come, Easy Go? Examining the Perceptions and Learning Effects of LLM-based Chatbot in the Context of Search-as-Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 25 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The cognitive process of Search-as-Learning (SAL) is most effective when searching promotes active encoding of information. The rise of LLMs-based chatbots, which provide instant answers, introduces a trade-off between efficiency and depth of processing. Such answer-centric approaches accelerate information access, but they also raise concerns about shallower learning. To examine these issues in the context of SAL, we conducted a large-scale survey of educators and students to capture perceived risks and benefits of LLM-based chatbots. In addition, we adopted the encoding-storage paradigm to design a within-subjects experiment, where participants (N=92) engaged in SAL tasks using three different modalities: books, search engines, and chatbots. Our findings provide a counterintuitive insight into stakeholder concerns: while LLM-based chatbots and search engines validated perceived benefits on learning efficiency by outperforming book-based search in immediate conceptual understanding, they did not result in a long-term inferiority as feared. Our study provides insights for designing human-AI collaborative learning systems that promote cognitive engagement by balancing learning efficiency and long-term knowledge retention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16317v1">Design and Evaluation of Cost-Aware PoQ for Decentralized LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Decentralized large language model (LLM) inference promises transparent and censorship resistant access to advanced AI, yet existing verification approaches struggle to scale to modern models. Proof of Quality (PoQ) replaces cryptographic verification of computation with consensus over output quality, but the original formulation ignores heterogeneous computational costs across inference and evaluator nodes. This paper introduces a cost-aware PoQ framework that integrates explicit efficiency measurements into the reward mechanism for both types of nodes. The design combines ground truth token level F1, lightweight learned evaluators, and GPT based judgments within a unified evaluation pipeline, and adopts a linear reward function that balances normalized quality and cost. Experiments on extractive question answering and abstractive summarization use five instruction tuned LLMs ranging from TinyLlama-1.1B to Llama-3.2-3B and three evaluation models spanning cross encoder and bi encoder architectures. Results show that a semantic textual similarity bi encoder achieves much higher correlation with both ground truth and GPT scores than cross encoders, indicating that evaluator architecture is a critical design choice for PoQ. Quality-cost analysis further reveals that the largest models in the pool are also the most efficient in terms of quality per unit latency. Monte Carlo simulations over 5\,000 PoQ rounds demonstrate that the cost-aware reward scheme consistently assigns higher average rewards to high quality low cost inference models and to efficient evaluators, while penalizing slow low quality nodes. These findings suggest that cost-aware PoQ provides a practical foundation for economically sustainable decentralized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v4">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16272v1">Beyond Blind Spots: Analytic Hints for Mitigating LLM-Based Evaluation Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed as judges (LaaJ) in code generation pipelines. While attractive for scalability, LaaJs tend to overlook domain specific issues raising concerns about their reliability in critical evaluation tasks. To better understand these limitations in practice, we examine LaaJ behavior in a concrete industrial use case: legacy code modernization via COBOL code generation. In this setting, we find that even production deployed LaaJs can miss domain critical errors, revealing consistent blind spots in their evaluation capabilities. To better understand these blind spots, we analyze generated COBOL programs and associated LaaJs judgments, drawing on expert knowledge to construct a preliminary taxonomy. Based on this taxonomy, we develop a lightweight analytic checker tool that flags over 30 domain specific issues observed in practice. We use its outputs as analytic hints, dynamically injecting them into the judges prompt to encourage LaaJ to revisit aspects it may have overlooked. Experiments on a test set of 100 programs using four production level LaaJs show that LaaJ alone detects only about 45% of the errors present in the code (in all judges we tested), while the analytic checker alone lacks explanatory depth. When combined, the LaaJ+Hints configuration achieves up to 94% coverage (for the best performing judge and injection prompt) and produces qualitatively richer, more accurate explanations, demonstrating that analytic-LLM hybrids can substantially enhance evaluation reliability in deployed pipelines. We release the dataset and all used prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.15108v2">Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) in biological-medical applications has highlighted a gap between their potential and the limited scale and often low quality of available open-source annotated textual datasets. In addition, the inherent complexity of the biomedical knowledge hierarchy significantly hampers efforts to bridge this gap.Can LLMs themselves play a pivotal role in overcoming this limitation? Motivated by this question, we investigate this challenge in the present study.We propose a framework that automates the distillation of high-quality textual training data from the extensive scientific literature. Our approach self-evaluates and generates questions that are more closely aligned with the biomedical domain, guided by the biomedical knowledge hierarchy through medical subject headings (MeSH). This comprehensive framework establishes an automated workflow, thereby eliminating the need for manual intervention. Furthermore, we conducted comprehensive experiments to evaluate the impact of our framework-generated data on downstream language models of varying sizes. Our approach substantially improves question-answering tasks compared to pre-trained models from the life sciences domain and powerful close-source models represented by GPT-4. Notably, the generated AI-Ready dataset enabled the Llama3-70B base model to outperform GPT-4 using MedPrompt with multiple times the number of parameters. Detailed case studies and ablation experiments underscore the significance of each component within our framework
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v1">Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16167v1">Ev-Trust: A Strategy Equilibrium Trust Mechanism for Evolutionary Games in LLM-Based Multi-Agent Services</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The rapid evolution of the Web toward an agent-centric paradigm, driven by large language models (LLMs), has enabled autonomous agents to reason, plan, and interact in complex decentralized environments. However, the openness and heterogeneity of LLM-based multi-agent systems also amplify the risks of deception, fraud, and misinformation, posing severe challenges to trust establishment and system robustness. To address this issue, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. This mechanism integrates direct trust, indirect trust, and expected revenue into a dynamic feedback structure that guides agents' behavioral evolution toward equilibria. Within a decentralized "Request-Response-Payment-Evaluation" service framework, Ev-Trust enables agents to adaptively adjust strategies, naturally excluding malicious participants while reinforcing high-quality collaboration. Furthermore, our theoretical derivation based on replicator dynamics equations proves the existence and stability of local evolutionary equilibria. Experimental results indicate that our approach effectively reflects agent trustworthiness in LLM-driven open service interaction scenarios, reduces malicious strategies, and increases collective revenue. We hope Ev-Trust can provide a new perspective on trust modeling for the agentic service web in group evolutionary game scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21557v2">Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 NeurIPS 2025 LLM Evaluation Workshop
    </div>
    <details class="paper-abstract">
      Trustworthy Large Language Models (LLMs) must cite human-verifiable sources in high-stakes domains such as healthcare, law, academia, and finance, where even small errors can have severe consequences. Practitioners and researchers face a choice: let models generate citations during decoding, or let models draft answers first and then attach appropriate citations. To clarify this choice, we introduce two paradigms: Generation-Time Citation (G-Cite), which produces the answer and citations in one pass, and Post-hoc Citation (P-Cite), which adds or verifies citations after drafting. We conduct a comprehensive evaluation from zero-shot to advanced retrieval-augmented methods across four popular attribution datasets and provide evidence-based recommendations that weigh trade-offs across use cases. Our results show a consistent trade-off between coverage and citation correctness, with retrieval as the main driver of attribution quality in both paradigms. P-Cite methods achieve high coverage with competitive correctness and moderate latency, whereas G-Cite methods prioritize precision at the cost of coverage and speed. We recommend a retrieval-centric, P-Cite-first approach for high-stakes applications, reserving G-Cite for precision-critical settings such as strict claim verification. Our codes and human evaluation results are available at https://anonymous.4open.science/r/Citation_Paradigms-BBB5/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16134v1">Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) serving towards complex, distributed architectures--specifically the P/D-separated, large-scale DP+EP paradigm--introduces distinct scheduling challenges. Unlike traditional deployments where schedulers can treat instances as black boxes, DP+EP architectures exhibit high internal synchronization costs. We identify that immediate request dispatching in such systems leads to severe in-engine queuing and parallelization bubbles, degrading Time-to-First-Token (TTFT). To address this, we propose Staggered Batch Scheduling (SBS), a mechanism that deliberately buffers requests to form optimal execution batches. This temporal decoupling eliminates internal queuing bubbles without compromising throughput. Furthermore, leveraging the scheduling window created by buffering, we introduce a Load-Aware Global Allocation strategy that balances computational load across DP units for both Prefill and Decode phases. Deployed on a production H800 cluster serving Deepseek-V3, our system reduces TTFT by 30%-40% and improves throughput by 15%-20% compared to state-of-the-art immediate scheduling baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19108v4">A Multi-Language Perspective on the Robustness of LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 46 pages, 10 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models have gained significant traction and popularity in recent times, extending their usage to code-generation tasks. While this field has garnered considerable attention, the exploration of testing and evaluating the robustness of code generation models remains an ongoing endeavor. Previous studies have primarily focused on code generation models specifically for the Python language, overlooking other widely-used programming languages. In this work, we conduct a comprehensive comparative analysis to assess the robustness performance of several prominent code generation models and investigate whether robustness can be improved by repairing perturbed docstrings using an LLM. Furthermore, we investigate how their performance varies across different programming languages. To accomplish this, we introduce perturbations in four key areas of the prompt: DocString, functionname, syntax, and format. We have compiled and released a dedicated dataset for this purpose. This work presents our experimental findings, shedding light on the performance of code generation models in various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09254v2">The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed as autonomous agents on behalf of institutions and individuals in economic, political, and social settings that involve negotiation. Yet this trend carries significant risks if their strategic behavior is not well understood. In this work, we revisit the NegotiationArena framework and run controlled simulation experiments on a diverse set of frontier LLMs across three multi turn bargaining games: Buyer Seller, Multi turn Ultimatum, and Resource Exchange. We ask whether improved general reasoning capabilities lead to rational, unbiased, and convergent negotiation strategies. Our results challenge this assumption. We find that models diverge into distinct, model specific strategic equilibria rather than converging to a unified optimal behavior. Moreover, strong numerical and semantic anchoring effects persist: initial offers are highly predictive of final agreements, and models consistently generate biased proposals by collapsing diverse internal valuations into rigid, generic price points. More concerningly, we observe dominance patterns in which some models systematically achieve higher payoffs than their counterparts. These findings underscore an urgent need to develop mechanisms to mitigate these issues before deploying such systems in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06489v3">On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.15251v2">Input Reduction Enhanced LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in Automated Program Repair (APR). Test inputs, being crucial for reasoning the root cause of failures, are always included in the prompt for LLM-based APR. Unfortunately, LLMs struggle to retain key information in long prompts. When the test inputs are extensive in the prompt, this may trigger the "lost-in-the-middle" issue, compromising repair performance. ReduceFix prompts an LLM to generate a reducer that minimizes failure-inducing test inputs without human effort, and then feeds the reduced failure-inducing inputs to guide patch generation. For targeted evaluation, we constructed LFTBench, the first long-input APR benchmark with 200 real bugs from 20 programming tasks, each paired with a failure-inducing input whose median size is 1 MB. On this benchmark, ReduceFix shrinks inputs by 89.1% on average and improves overall pass@10 by up to 53.8% relative to a prompt that includes the original test, and by 17.6% compared with omitting the test entirely. Adding the same reduction step to ChatRepair and CREF increases their fix rate by 21.3% and 2.6%, respectively, without other changes. Our gains hold against a ddmin-only reducing template baseline and transfer to repository-level OSS-Fuzz cases. Ablation studies further highlight the impact of input length and compressed failure information on repair success. These results underscore that automatically reducing failing inputs is a practical and powerful complement to LLM-based APR, significantly improving its scalability and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16083v1">Scaling Text2SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Most modern Text2SQL systems prompt large language models (LLMs) with entire schemas -- mostly column information -- alongside the user's question. While effective on small databases, this approach fails on real-world schemas that exceed LLM context limits, even for commercial models. The recent Spider 2.0 benchmark exemplifies this with hundreds of tables and tens of thousands of columns, where existing systems often break. Current mitigations either rely on costly multi-step prompting pipelines or filter columns by ranking them against user's question independently, ignoring inter-column structure. To scale existing systems, we introduce \toolname, an open-source, LLM-efficient schema filtering framework that compacts Text2SQL prompts by (i) ranking columns with a query-aware LLM encoder enriched with values and metadata, (ii) reranking inter-connected columns via a lightweight graph transformer over functional dependencies, and (iii) selecting a connectivity-preserving sub-schema with a Steiner-tree heuristic. Experiments on real datasets show that \toolname achieves near-perfect recall and higher precision than CodeS, SchemaExP, Qwen rerankers, and embedding retrievers, while maintaining sub-second median latency and scaling to schemas with 23,000+ columns. Our source code is available at https://github.com/thanhdath/grast-sql.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16056v1">MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks in LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The limited bandwidth of PCIe has emerged as the critical bottleneck for large language model (LLM) performance, such as prefix cache fetching and model switching. Although intra-server multipath data transfer between GPU and host memory is theoretically possible, heterogeneous protocols such as PCIe and NVLink currently limit the bandwidth between host memory and GPUs to that of a single PICe link. This limitation resuals in underutilized intra-server bandwidth. To address this issue, we propose Multipath Memory Access (MMA), a scheme that, to the best of our knowledge, is the first to enalbe efficient multipath data transfer between GPU and host memory. MMA supports seamless deployment via dynamic library injection, enabling LLM applications to benefit from MMA without requiring any code modification. In our testbed, MMA significantly improves the data transfer bandwidth between the GPU and memory, achieving a peak bandwidth of 245 GB/s-representing a 4.62x speedup compared to the natice single-path bandwidth. End-to-end evaluations demonstrate that MMA reduces the time-to-first-token (TTFT) for LLM serving by 1.14x to 2.38x and decreases model-switching latency in vLLM's sleep mode by 1.12x to 2.48x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17093v1">A Solver-in-the-Loop Framework for Improving LLMs on Answer Set Programming for Logic Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 15 pages, 7 figures, accepted at AAAI'26
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has sparked interest in coding assistants. While general-purpose programming languages are well supported, generating code for domain-specific languages remains a challenging problem for LLMs. In this paper, we focus on the LLM-based generation of code for Answer Set Programming (ASP), a particularly effective approach for finding solutions to combinatorial search problems. The effectiveness of LLMs in ASP code generation is currently hindered by the limited number of examples seen during their initial pre-training phase. In this paper, we introduce a novel ASP-solver-in-the-loop approach for solver-guided instruction-tuning of LLMs to addressing the highly complex semantic parsing task inherent in ASP code generation. Our method only requires problem specifications in natural language and their solutions. Specifically, we sample ASP statements for program continuations from LLMs for unriddling logic puzzles. Leveraging the special property of declarative ASP programming that partial encodings increasingly narrow down the solution space, we categorize them into chosen and rejected instances based on solver feedback. We then apply supervised fine-tuning to train LLMs on the curated data and further improve robustness using a solver-guided search that includes best-of-N sampling. Our experiments demonstrate consistent improvements in two distinct prompting settings on two datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17062v1">Lang2Manip: A Tool for LLM-Based Symbolic-to-Geometric Planning for Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Submitted to ICARA
    </div>
    <details class="paper-abstract">
      Simulation is essential for developing robotic manipulation systems, particularly for task and motion planning (TAMP), where symbolic reasoning interfaces with geometric, kinematic, and physics-based execution. Recent advances in Large Language Models (LLMs) enable robots to generate symbolic plans from natural language, yet executing these plans in simulation often requires robot-specific engineering or planner-dependent integration. In this work, we present a unified pipeline that connects an LLM-based symbolic planner with the Kautham motion planning framework to achieve generalizable, robot-agnostic symbolic-to-geometric manipulation. Kautham provides ROS-compatible support for a wide range of industrial manipulators and offers geometric, kinodynamic, physics-driven, and constraint-based motion planning under a single interface. Our system converts language instructions into symbolic actions and computes and executes collision-free trajectories using any of Kautham's planners without additional coding. The result is a flexible and scalable tool for language-driven TAMP that is generalized across robots, planning modalities, and manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17060v1">On the Role of Contextual Information and Ego States in LLM Agent Behavior for Transactional Analysis Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Presented at the 39th Pacific Asia Conference on Language, Information and Computation (PACLIC 39)
    </div>
    <details class="paper-abstract">
      LLM-powered agents are now used in many areas, from customer support to education, and there is increasing interest in their ability to act more like humans. This includes fields such as social, political, and psychological research, where the goal is to model group dynamics and social behavior. However, current LLM agents often lack the psychological depth and consistency needed to capture the real patterns of human thinking. They usually provide direct or statistically likely answers, but they miss the deeper goals, emotional conflicts, and motivations that drive real human interactions. This paper proposes a Multi-Agent System (MAS) inspired by Transactional Analysis (TA) theory. In the proposed system, each agent is divided into three ego states - Parent, Adult, and Child. The ego states are treated as separate knowledge structures with their own perspectives and reasoning styles. To enrich their response process, they have access to an information retrieval mechanism that allows them to retrieve relevant contextual information from their vector stores. This architecture is evaluated through ablation tests in a simulated dialogue scenario, comparing agents with and without information retrieval. The results are promising and open up new directions for exploring how psychologically grounded structures can enrich agent behavior. The contribution is an agent architecture that integrates Transactional Analysis theory with contextual information retrieval to enhance the realism of LLM-based multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17043v1">UniRel-R1: RL-tuned LLM Reasoning for Knowledge Graph Relational Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) has traditionally focused on entity-centric queries that return a single answer entity. However, real-world queries are often relational, seeking to understand how entities are associated. In this work, we introduce relation-centric KGQA, a complementary setting where the answer is a subgraph capturing the semantic connections among entities rather than an individual entity. The main challenge lies in the abundance of candidate subgraphs, where trivial or overly common connections often obscure the identification of unique and informative answers. To tackle this, we propose UniRel-R1, a unified framework that integrates subgraph selection, multi-stage graph pruning, and an LLM fine-tuned with reinforcement learning. The reward function is designed to encourage compact and specific subgraphs with more informative relations and lower-degree intermediate entities. Extensive experiments show that UniRel-R1 achieves significant gains in connectivity and reward over Vanilla baselines and generalizes effectively to unseen entities and relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17023v1">LLM-HPC++: Evaluating LLM-Generated Modern C++ and MPI+OpenMP Codes for Scalable Mandelbrot Set Computation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Parallel programming remains one of the most challenging aspects of High-Performance Computing (HPC), requiring deep knowledge of synchronization, communication, and memory models. While modern C++ standards and frameworks like OpenMP and MPI have simplified parallelism, mastering these paradigms is still complex. Recently, Large Language Models (LLMs) have shown promise in automating code generation, but their effectiveness in producing correct and efficient HPC code is not well understood. In this work, we systematically evaluate leading LLMs including ChatGPT 4 and 5, Claude, and LLaMA on the task of generating C++ implementations of the Mandelbrot set using shared-memory, directive-based, and distributed-memory paradigms. Each generated program is compiled and executed with GCC 11.5.0 to assess its correctness, robustness, and scalability. Results show that ChatGPT-4 and ChatGPT-5 achieve strong syntactic precision and scalable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17008v1">Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exposes notable limitations, particularly in scenarios requiring long-horizon reasoning. To address these challenges, we investigate more stable and effective advantage estimation strategies, especially for multi-turn settings. We first explore Proximal Policy Optimization (PPO) as an alternative and find it to be more robust than GRPO. To further enhance PPO in multi-turn scenarios, we introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO, both with and without long reasoning components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16969v1">Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)-the ability to autonomously conceive, investigate, and reason across scientific domains-remains lacking. We present an operational SGI definition grounded in the Practical Inquiry Model (PIM: Deliberation, Conception, Action, Perception) and operationalize it via four scientist-aligned tasks: deep research, idea generation, dry/wet experiments, and experimental reasoning. SGI-Bench comprises over 1,000 expert-curated, cross-disciplinary samples inspired by Science's 125 Big Questions, enabling systematic evaluation of state-of-the-art LLMs. Results reveal gaps: low exact match (10--20%) in deep research despite step-level alignment; ideas lacking feasibility and detail; high code executability but low execution result accuracy in dry experiments; low sequence fidelity in wet protocols; and persistent multimodal comparative-reasoning challenges. We further introduce Test-Time Reinforcement Learning (TTRL), which optimizes retrieval-augmented novelty rewards at inference, enhancing hypothesis novelty without reference answer. Together, our PIM-grounded definition, workflow-centric benchmark, and empirical insights establish a foundation for AI systems that genuinely participate in scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16962v1">MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 14 pages, 1 figure, includes appendix
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at https://github.com/Jacobhhy/Agent-Memory-Poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17967v1">Memelang: An Axial Grammar for LLM-Generated Vector-Relational Queries</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Structured generation for LLM tool use highlights the value of compact DSL intermediate representations (IRs) that can be emitted directly and parsed deterministically. This paper introduces axial grammar: linear token sequences that recover multi-dimensional structure from the placement of rank-specific separator tokens. A single left-to-right pass assigns each token a coordinate in an n-dimensional grid, enabling deterministic parsing without parentheses or clause-heavy surface syntax. This grammar is instantiated in Memelang, a compact query language intended as an LLM-emittable IR whose fixed coordinate roles map directly to table/column/value slots. Memelang supports coordinate-stable relative references, parse-time variable binding, and implicit context carry-forward to reduce repetition in LLM-produced queries. It also encodes grouping, aggregation, and ordering via inline tags on value terms, allowing grouped execution plans to be derived in one streaming pass over the coordinate-indexed representation. Provided are a reference lexer/parser and a compiler that emits parameterized PostgreSQL SQL (optionally using pgvector operators).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15053v1">The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The transition of Large Language Models (LLMs) from stochastic chat interfaces to reliable software components necessitates a fundamental re-engineering of interaction paradigms. Current methodologies, predominantly heuristic-based "prompt engineering," fail to provide the deterministic guarantees required for mission-critical applications. We introduce the Meta-Prompting Protocol, a rigorous theoretical framework that formalizes the orchestration of LLMs as a programmable, self-optimizing system. Central to this protocol is the Adversarial Trinity, a tripartite topology comprising a Generator (P), an Auditor (A), and an Optimizer (O). By treating natural language instructions as differentiable variables within a semantic computation graph and utilizing textual critiques as gradients, this architecture mitigates hallucination and prevents model collapse. We demonstrate the theoretical viability of this approach using declarative programming paradigms (DSPy) and automatic textual differentiation (TextGrad), establishing a foundation for "Observable Software Engineering" in the era of probabilistic computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15000v1">DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have become essential for improving Large Language Models (LLMs) via test-time scaling, yet their effectiveness in coding remains limited due to the lack of meaningful step decompositions in code and the noise of Monte-Carlo-generated partial labels. We propose DreamPRM-Code, a coding-focused PRM that treats functions as reasoning steps using a Chain-of-Function prompting strategy to induce modular code generation, enabling PRM training and application analogous to mathematical reasoning tasks. To address label noise, DreamPRM-Code introduces a meta-learning-based correction mechanism that leverages clean final-solution unit-test labels and performs bi-level optimization to refine intermediate labels. Applying on test-time scaling, DreamPRM-Code achieved state-of-the-art performance on LiveCodeBench with 80.9 pass@1 rate, surpassing OpenAI o4-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11062v4">Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16041v1">Are We on the Right Way to Assessing LLM-as-a-Judge?</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessment of reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming Sage's reliability as an evaluation suite for the robustness and accuracy of LLM-as-a-Judge. Based on Sage, we reveal that current state-of-the-art LLMs exhibit significant reliability problems when acting as judges in both scoring and pairwise settings; even the top-performing models, Gemini-2.5-Pro and GPT-5, fail to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called situational preference, which explains why explicit rubrics or criteria can help the model judge consistently across answer pairs. Our further analysis shows that finetuned LLM-as-a-Judge is a feasible method to boost performance, and the panel-based judge as well as deep reasoning can enhance the judging consistency. We also find substantial inconsistency in human judgments, which indicates that human annotation may not be a reliable gold standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.16112v2">HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15979v1">OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15674v1">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Overall, our best AOs match or exceed prior white-box baselines on all four tasks and are the best method on 3 out of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15662v1">Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) decouple reasoning from verification: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), a unified framework that interleaves reasoning and self-critique at each step within a single model. STC is trained with a hybrid reinforcement learning objective combining reasoning rewards and critique-consistency rewards to jointly optimize reasoning quality and self-evaluation. Experiments on mathematical reasoning benchmarks show that STC demonstrates strong critic-thinking capabilities and produces more interpretable reasoning traces, representing a step toward LLMs with built-in critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02389v2">From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function or file level detections which offers limited actionable guidance to engineers who need precise line-level localization and targeted patches in real-world software development. We present T2L-Agent (Trace-to-Line Agent), a project-level, end-to-end framework that plans its own analysis and progressively narrows scope from modules to exact vulnerable lines. T2L-Agent couples multi-round feedback with an Agentic Trace Analyzer (ATA) that fuses run-time evidence such as crash points, stack traces, and coverage deltas with AST-based code chunking, enabling iterative refinement beyond single pass predictions and translating symptoms into actionable, line-level diagnoses. To benchmark line-level vulnerability discovery, we introduce T2L-ARVO, a diverse, expert-verified 50-case benchmark spanning five crash families and real-world projects. T2L-ARVO is specifically designed to support both coarse-grained detection and fine-grained localization, enabling rigorous evaluation of systems that aim to move beyond file-level predictions. On T2L-ARVO, T2L-Agent achieves up to 58.0% detection and 54.8% line-level localization, substantially outperforming baselines. Together, the framework and benchmark push LLM-based vulnerability detection from coarse identification toward deployable, robust, precision diagnostics that reduce noise and accelerate patching in open-source software workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01353v3">The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Updated with new baselines and experimental results
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15617v1">Evaluating Metrics for Safety with LLM-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      LLMs (Large Language Models) are increasingly used in text processing pipelines to intelligently respond to a variety of inputs and generation tasks. This raises the possibility of replacing human roles that bottleneck existing information flows, either due to insufficient staff or process complexity. However, LLMs make mistakes and some processing roles are safety critical. For example, triaging post-operative care to patients based on hospital referral letters, or updating site access schedules in nuclear facilities for work crews. If we want to introduce LLMs into critical information flows that were previously performed by humans, how can we make them safe and reliable? Rather than make performative claims about augmented generation frameworks or graph-based techniques, this paper argues that the safety argument should focus on the type of evidence we get from evaluation points in LLM processes, particularly in frameworks that employ LLM-as-Judges (LaJ) evaluators. This paper argues that although we cannot get deterministic evaluations from many natural language processing tasks, by adopting a basket of weighted metrics it may be possible to lower the risk of errors within an evaluation, use context sensitivity to define error severity and design confidence thresholds that trigger human review of critical LaJ judgments when concordance across evaluators is low.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14285v4">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at the 11th IEEE WIECON-ECE 2025
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15550v1">CTkvr: KV Cache Retrieval for Long-Context LLMs via Centroid then Token Indexing</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in long-context scenarios such as multi-turn conversations. However, long contexts pose significant challenges for inference efficiency, including high memory overhead from Key-Value (KV) cache and increased latency due to excessive memory accesses. Recent methods for dynamic KV selection struggle with trade-offs: block-level indexing degrades accuracy by retrieving irrelevant KV entries, while token-level indexing incurs high latency from inefficient retrieval mechanisms. In this paper, we propose CTKVR, a novel centroid-then-token KV retrieval scheme that addresses these limitations. CTKVR leverages a key observation: query vectors adjacent in position exhibit high similarity after Rotary Position Embedding (RoPE) and share most of their top-k KV cache entries. Based on this insight, CTKVR employs a two-stage retrieval strategy: lightweight centroids are precomputed during prefilling for centroid-grained indexing, followed by token-level refinement for precise KV retrieval. This approach balances retrieval efficiency and accuracy. To further enhance performance, we implement an optimized system for indexing construction and search using CPU-GPU co-execution. Experimentally, CTKVR achieves superior performance across multiple benchmarks with less than 1% accuracy degradation. Meanwhile, CTKVR delivers 3 times and 4 times throughput speedups on Llama-3-8B and Yi-9B at 96K context length across diverse GPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25300v2">Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 V2 version:27 pages, 14 figures
    </div>
    <details class="paper-abstract">
      While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: 1.Larger models consistently exhibit superior learning efficiency on both compute and data metrics. 2.The relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. 3.Although larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. 4.In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v3">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15446v1">Toward expert-level motivational interviewing for health behavior improvement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 26 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Background: Motivational interviewing (MI) is an effective counseling approach for promoting health behavior change, but its impact is constrained by the need for highly trained human counselors. Objective: This study aimed to explore a scalable alternative by developing and evaluating Large Language Models for Motivational Interviewing (MI-LLMs). Methods: We first curated five Chinese psychological counseling corpora and, using GPT-4 with an MI-informed prompt, transcribed multi-turn dialogues from the two highest-quality datasets (CPsyCounD and PsyDTCorpus) into 2,040 MI-style counseling conversations, of which 2,000 were used for training and 40 for testing. Three Chinese-capable open-source LLMs (Baichuan2-7B-Chat, ChatGLM-4-9B-Chat and Llama-3-8B-Chinese-Chat-v2) were fine-tuned on this corpus and were named as MI-LLMs. We evaluated MI-LLMs using round-based automatic metrics and expert manual coding with the Motivational Interviewing Treatment Integrity (MITI) Coding Manual 4.2.1. Results: Across all three models, fine-tuning substantially improved BLEU-4 and ROUGE scores compared with the base models, and manual coding showed that MI-LLMs achieved technical and relational global scores, and MI-adherent ratios that approached those of real MI dialogues, although complex reflections and reflection-to-question ratios remained less frequent. Conclusions: These findings provide initial evidence that MI-oriented fine-tuning can endow general-purpose LLMs with core MI-consistent counseling behaviors, suggesting a scalable pathway toward AI-assisted health behavior change support while underscoring the need for further work on data scale, complex MI skills and real-world intervention trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05272v2">LLMs and Fuzzing in Tandem: A New Approach to Automatically Generating Weakest Preconditions</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      The weakest precondition (WP) of a program describes the largest set of initial states from which all terminating executions of the program satisfy a given postcondition. The generation of WPs is an important task with practical applications in areas ranging from verification to run-time error checking. This paper proposes the combination of Large Language Models (LLMs) and fuzz testing for generating WPs. In pursuit of this goal, we introduce \emph{Fuzzing Guidance} (FG); FG acts as a means of directing LLMs towards correct WPs using program execution feedback. FG utilises fuzz testing for approximately checking the validity and weakness of candidate WPs, this information is then fed back to the LLM as a means of context refinement. We demonstrate the effectiveness of our approach on a comprehensive benchmark set of deterministic array programs in Java. Our experiments indicate that LLMs are capable of producing viable candidate WPs, and that this ability can be practically enhanced through FG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20764v3">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at 2025 EMNLP findings,19 page,2 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we introduce RealCBT, a dataset of authentic cognitive behavioral therapy (CBT) dialogues, and conduct the first comparative analysis of emotional arcs between real and LLM-generated CBT sessions. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions from the RealCBT dataset and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability, more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity remains low across all pairings, with especially weak alignment between real and synthetic speakers. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. To support future research, our dataset RealCBT is released at https://gitlab.com/xiaoyi.wang/realcbt-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15397v1">ORACLE: Time-Dependent Recursive Summary Graphs for Foresight on News Data Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      ORACLE turns daily news into week-over-week, decision-ready insights for one of the Finnish University of Applied Sciences. The platform crawls and versions news, applies University-specific relevance filtering, embeds content, classifies items into PESTEL dimensions and builds a concise Time-Dependent Recursive Summary Graph (TRSG): two clustering layers summarized by an LLM and recomputed weekly. A lightweight change detector highlights what is new, removed or changed, then groups differences into themes for PESTEL-aware analysis. We detail the pipeline, discuss concrete design choices that make the system stable in production and present a curriculum-intelligence use case with an evaluation plan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13857v2">EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15365v1">ArcBERT: An LLM-based Search Engine for Exploring Integrated Multi-Omics Metadata</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Traditional search applications within Research Data Management (RDM) ecosystems are crucial in helping users discover and explore the structured metadata from the research datasets. Typically, text search engines require users to submit keyword-based queries rather than using natural language. However, using Large Language Models (LLMs) trained on domain-specific content for specialized natural language processing (NLP) tasks is becoming increasingly common. We present ArcBERT, an LLM-based system designed for integrated metadata exploration. ArcBERT understands natural language queries and relies on semantic matching, unlike traditional search applications. Notably, ArcBERT also understands the structure and hierarchies within the metadata, enabling it to handle diverse user querying patterns effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15353v1">Adversarial versification in portuguese as a jailbreak operator in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent evidence shows that the versification of prompts constitutes a highly effective adversarial mechanism against aligned LLMs. The study 'Adversarial poetry as a universal single-turn jailbreak mechanism in large language models' demonstrates that instructions routinely refused in prose become executable when rewritten as verse, producing up to 18 x more safety failures in benchmarks derived from MLCommons AILuminate. Manually written poems reach approximately 62% ASR, and automated versions 43%, with some models surpassing 90% success in single-turn interactions. The effect is structural: systems trained with RLHF, constitutional AI, and hybrid pipelines exhibit consistent degradation under minimal semiotic formal variation. Versification displaces the prompt into sparsely supervised latent regions, revealing guardrails that are excessively dependent on surface patterns. This dissociation between apparent robustness and real vulnerability exposes deep limitations in current alignment regimes. The absence of evaluations in Portuguese, a language with high morphosyntactic complexity, a rich metric-prosodic tradition, and over 250 million speakers, constitutes a critical gap. Experimental protocols must parameterise scansion, metre, and prosodic variation to test vulnerabilities specific to Lusophone patterns, which are currently ignored.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15343v1">Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      The rapid development of generative artificial intelligence (AI) and large language models (LLMs), and the availability of services that make them accessible, have led the general public to begin incorporating them into everyday life. The extended reality (XR) community has also sought to integrate LLMs, particularly in the form of conversational agents, to enhance user experience and task efficiency. When interacting with such conversational agents, users may easily disclose sensitive information due to the naturalistic flow of the conversations, and combining such conversational data with fine-grained sensor data may lead to novel privacy issues. To address these issues, a user-centric understanding of technology acceptance and concerns is essential. Therefore, to this end, we conducted a large-scale crowdsourcing study with 1036 participants, examining user decision-making processes regarding LLM-powered conversational agents in XR, across factors of XR setting type, speech interaction type, and data processing location. We found that while users generally accept these technologies, they express concerns related to security, privacy, social implications, and trust. Our results suggest that familiarity plays a crucial role, as daily generative AI use is associated with greater acceptance. In contrast, previous ownership of XR devices is linked to less acceptance, possibly due to existing familiarity with the settings. We also found that men report higher acceptance with fewer concerns than women. Regarding data type sensitivity, location data elicited the most significant concern, while body temperature and virtual object states were considered least sensitive. Overall, our study highlights the importance of practitioners effectively communicating their measures to users, who may remain distrustful. We conclude with implications and recommendations for LLM-powered XR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15312v1">Evaluating LLMs for Zeolite Synthesis Event Extraction (ZSEE): A Systematic Analysis of Prompting Strategies</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Extracting structured information from zeolite synthesis experimental procedures is critical for materials discovery, yet existing methods have not systematically evaluated Large Language Models (LLMs) for this domain-specific task. This work addresses a fundamental question: what is the efficacy of different prompting strategies when applying LLMs to scientific information extraction? We focus on four key subtasks: event type classification (identifying synthesis steps), trigger text identification (locating event mentions), argument role extraction (recognizing parameter types), and argument text extraction (extracting parameter values). We evaluate four prompting strategies - zero-shot, few-shot, event-specific, and reflection-based - across six state-of-the-art LLMs (Gemma-3-12b-it, GPT-5-mini, O4-mini, Claude-Haiku-3.5, DeepSeek reasoning and non-reasoning) using the ZSEE dataset of 1,530 annotated sentences. Results demonstrate strong performance on event type classification (80-90\% F1) but modest performance on fine-grained extraction tasks, particularly argument role and argument text extraction (50-65\% F1). GPT-5-mini exhibits extreme prompt sensitivity with 11-79\% F1 variation. Notably, advanced prompting strategies provide minimal improvements over zero-shot approaches, revealing fundamental architectural limitations. Error analysis identifies systematic hallucination, over-generalization, and inability to capture synthesis-specific nuances. Our findings demonstrate that while LLMs achieve high-level understanding, precise extraction of experimental parameters requires domain-adapted models, providing quantitative benchmarks for scientific information extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15274v1">Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capability of Large Language Models (LLMs). Current RLVR approaches typically conduct training across all generated tokens, but neglect to explore which tokens (e.g., prefix tokens) actually contribute to reasoning. This uniform training strategy spends substantial effort on optimizing low-return tokens, which in turn impedes the potential improvement from high-return tokens and reduces overall training effectiveness. To address this issue, we propose a novel RLVR approach called Progressive Prefix-token Policy Optimization (PPPO), which highlights the significance of the prefix segment of generated outputs. Specifically, inspired by the well-established human thinking theory of Path Dependence, where early-stage thoughts substantially constrain subsequent thinking trajectory, we identify an analogous phenomenon in LLM reasoning termed Beginning Lock-in Effect (BLE). PPPO leverages this finding by focusing its optimization objective on the prefix reasoning process of LLMs. This targeted optimization strategy can positively influence subsequent reasoning processes, and ultimately improve final results. To improve the learning effectiveness of LLMs on how to start reasoning with high quality, PPPO introduces two training strategies: (a) Progressive Prefix Retention, which shapes a progressive learning process by increasing the proportion of retained prefix tokens during training; (b) Continuation Accumulated Reward, which mitigates reward bias by sampling multiple continuations for one prefix token sequence, and accumulating their scores as the reward signal. Extensive experimental results on various reasoning tasks demonstrate that our proposed PPPO outperforms representative RLVR methods, with the accuracy improvements of 18.02% on only 26.17% training tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23049v2">DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Pruning is a widely used technique to compress large language models (LLMs) by removing unimportant weights, but it often suffers from significant performance degradation - especially under semi-structured sparsity constraints. Existing pruning methods primarily focus on estimating the importance of individual weights, which limits their ability to preserve critical capabilities of the model. In this work, we propose a new perspective: rather than merely selecting which weights to prune, we first redistribute parameter importance to make the model inherently more amenable to pruning. By minimizing the information entropy of normalized importance scores, our approach concentrates importance onto a smaller subset of weights, thereby enhancing pruning robustness. We instantiate this idea through DenoiseRotator, which applies learnable orthogonal transformations to the model's weight matrices. Our method can be seamlessly integrated with existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4 semi-structured sparsity, DenoiseRotator consistently improves perplexity and zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4 semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are available at https://github.com/Axel-gu/DenoiseRotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11921v2">Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Posted premature without permission of all authors
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15179v1">No More Hidden Pitfalls? Exposing Smart Contract Bad Practices with LLM-Powered Hybrid Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 47 specific issues. Specifically, we propose SCALM, an LLM-powered framework featuring two methodological innovations: (1) A hybrid architecture that combines context-aware function-level slicing with knowledge-enhanced semantic reasoning via extensible vectorized pattern matching. (2) A multi-layer reasoning verification system connects low-level code patterns with high-level security principles through syntax, design patterns, and architecture analysis. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21990v3">ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 18 figures, 11 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoning LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized chemical knowledge, ChemFG, annotating the presence of functional groups in molecules and the changes of functional groups during chemical reactions, to enhance the model's understanding of the fundamental principles and internal logic of chemistry. Then, we propose a mixed-source distillation method that integrates expertise in atomized knowledge with general reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the model's reliability, transparency, and practicality in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15082v1">The Semantic Architect: How FEAML Bridges Structured Data and LLMs for Multi-Label Tasks</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Existing feature engineering methods based on large language models (LLMs) have not yet been applied to multi-label learning tasks. They lack the ability to model complex label dependencies and are not specifically adapted to the characteristics of multi-label tasks. To address the above issues, we propose Feature Engineering Automation for Multi-Label Learning (FEAML), an automated feature engineering method for multi-label classification which leverages the code generation capabilities of LLMs. By utilizing metadata and label co-occurrence matrices, LLMs are guided to understand the relationships between data features and task objectives, based on which high-quality features are generated. The newly generated features are evaluated in terms of model accuracy to assess their effectiveness, while Pearson correlation coefficients are used to detect redundancy. FEAML further incorporates the evaluation results as feedback to drive LLMs to continuously optimize code generation in subsequent iterations. By integrating LLMs with a feedback mechanism, FEAML realizes an efficient, interpretable and self-improving feature engineering paradigm. Empirical results on various multi-label datasets demonstrate that our FEAML outperforms other feature engineering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15081v1">Quantifying Return on Security Controls in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 13 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v3">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14946v1">EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Reusing KV cache is essential for high efficiency of Large Language Model (LLM) inference systems. With more LLM users, the KV cache footprint can easily exceed GPU memory capacity, so prior work has proposed to either evict KV cache to lower-tier storage devices, or compress KV cache so that more KV cache can be fit in the fast memory. However, prior work misses an important opportunity: jointly optimizing the eviction and compression decisions across all KV caches to minimize average generation latency without hurting quality. We propose EVICPRESS, a KV-cache management system that applies lossy compression and adaptive eviction to KV cache across multiple storage tiers. Specifically, for each KV cache of a context, EVICPRESS considers the effect of compression and eviction of the KV cache on the average generation quality and delay across all contexts as a whole. To achieve this, EVICPRESS proposes a unified utility function that quantifies the effect of quality and delay of the lossy compression or eviction. To this end, EVICPRESS's profiling module periodically updates the utility function scores on all possible eviction-compression configurations for all contexts and places KV caches using a fast heuristic to rearrange KV caches on all storage tiers, with the goal of maximizing the utility function scores on each storage tier. Compared to the baselines that evict KV cache or compress KV cache, EVICPRESS achieves higher KV-cache hit rates on fast devices, i.e., lower delay, while preserving high generation quality by applying conservative compression to contexts that are sensitive to compression errors. Evaluation on 12 datasets and 5 models demonstrates that EVICPRESS achieves up to 2.19x faster time-to-first-token (TTFT) at equivalent generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14896v1">DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 11 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Objectives: To evaluate large language model (LLM) performance on pharmacy licensure-style question-answering (QA) tasks and develop an external knowledge integration method to improve their accuracy. Methods: We benchmarked eleven existing LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset. We measured baseline accuracy for each model without modification. We then developed a three-step retrieval-augmented generation (RAG) pipeline, DrugRAG, that retrieves structured drug knowledge from validated sources and augments model prompts with evidence-based context. This pipeline operates externally to the models, requiring no changes to model architecture or parameters. Results: Baseline accuracy ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores. Models with fewer than 8 billion parameters scored below 50%. DrugRAG improved accuracy across all tested models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61% to 71%, Llama 3.1 8B: 46% to 67%) on the 141-item benchmark. Conclusion: We demonstrate that external structured drug knowledge integration through DrugRAG measurably improves LLM accuracy on pharmacy tasks without modifying the underlying models. This approach provides a practical pipeline for enhancing pharmacy-focused AI applications with evidence-based information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18148v2">Hidden in the Haystack: Smaller Needles are More Difficult for LLMs to Find</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant challenges with needle-in-ahaystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size, the length of the answer-containing document, has received little attention. We present the first systematic study of gold context size in long-context question answering, spanning three diverse benchmarks (general knowledge, biomedical reasoning, and mathematical reasoning), eleven state-of-the-art LLMs (including recent reasoning models), and more than 150K controlled runs. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This effect persists under rigorous confounder analysis: even after controlling for gold context position, answer token repetition, gold-to-distractor ratio, distractor volume, and domain specificity, gold context size remains a decisive, independent predictor of success. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14846v1">MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings). For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records. In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14792v1">IaC Generation with LLMs: An Error Taxonomy and A Study on Configuration Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Submitted to ACM
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently exhibit low success rates in generating correct and intent-aligned Infrastructure as Code (IaC). This research investigated methods to improve LLM-based IaC generation, specifically for Terraform, by systematically injecting structured configuration knowledge. To facilitate this, an existing IaC-Eval benchmark was significantly enhanced with cloud emulation and automated error analysis. Additionally, a novel error taxonomy for LLM-assisted IaC code generation was developed. A series of knowledge injection techniques was implemented and evaluated, progressing from Naive Retrieval-Augmented Generation (RAG) to more sophisticated Graph RAG approaches. These included semantic enrichment of graph components and modeling inter-resource dependencies. Experimental results demonstrated that while baseline LLM performance was poor (27.1% overall success), injecting structured configuration knowledge increased technical validation success to 75.3% and overall success to 62.6%. Despite these gains in technical correctness, intent alignment plateaued, revealing a "Correctness-Congruence Gap" where LLMs can become proficient "coders" but remain limited "architects" in fulfilling nuanced user intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11844v2">Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 NeurIPS 2025 Workshop: First Workshop on LLM Persona Modeling (Oral)
    </div>
    <details class="paper-abstract">
      We propose Love First, Know Later: a paradigm shift in computational matching that simulates interactions first, then assesses compatibility. Instead of comparing static profiles, our framework leverages LLMs as text world engines that operate in dual capacity-as persona-driven agents following behavioral policies and as the environment modeling interaction dynamics. We formalize compatibility assessment as a reward-modeling problem: given observed matching outcomes, we learn to extract signals from simulations that predict human preferences. Our key insight is that relationships hinge on responses to critical moments-we translate this observation from relationship psychology into mathematical hypotheses, enabling effective simulation. Theoretically, we prove that as LLM policies better approximate human behavior, the induced matching converges to optimal stable matching. Empirically, we validate on speed dating data for initial chemistry and divorce prediction for long-term stability. This paradigm enables interactive, personalized matching systems where users iteratively refine their agents, unlocking future possibilities for transparent and interactive compatibility assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14673v1">Reconsidering Conversational Norms in LLM Chatbots for Sustainable AI</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      LLM based chatbots have become central interfaces in technical, educational, and analytical domains, supporting tasks such as code reasoning, problem solving, and information exploration. As these systems scale, sustainability concerns have intensified, with most assessments focusing on model architecture, hardware efficiency, and deployment infrastructure. However, existing mitigation efforts largely overlook how user interaction practices themselves shape the energy profile of LLM based systems. In this vision paper, we argue that interaction level behavior appears to be an underexamined factor shaping the environmental impact of LLM based systems, and we present this issue across four dimensions. First, extended conversational patterns increase token production and raise the computational cost of inference. Second, expectations of instant responses limit opportunities for energy aware scheduling and workload consolidation. Third, everyday user habits contribute to cumulative operational demand in ways that are rarely quantified. Fourth, the accumulation of context affects memory requirements and reduces the efficiency of long running dialogues. Addressing these challenges requires rethinking how chatbot interactions are designed and conceptualized, and adopting perspectives that recognize sustainability as partly dependent on the conversational norms through which users engage with LLM based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14604v1">LLmFPCA-detect: LLM-powered Multivariate Functional PCA for Anomaly Detection in Sparse Longitudinal Texts</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Sparse longitudinal (SL) textual data arises when individuals generate text repeatedly over time (e.g., customer reviews, occasional social media posts, electronic medical records across visits), but the frequency and timing of observations vary across individuals. These complex textual data sets have immense potential to inform future policy and targeted recommendations. However, because SL text data lack dedicated methods and are noisy, heterogeneous, and prone to anomalies, detecting and inferring key patterns is challenging. We introduce LLmFPCA-detect, a flexible framework that pairs LLM-based text embeddings with functional data analysis to detect clusters and infer anomalies in large SL text datasets. First, LLmFPCA-detect embeds each piece of text into an application-specific numeric space using LLM prompts. Sparse multivariate functional principal component analysis (mFPCA) conducted in the numeric space forms the workhorse to recover primary population characteristics, and produces subject-level scores which, together with baseline static covariates, facilitate data segmentation, unsupervised anomaly detection and inference, and enable other downstream tasks. In particular, we leverage LLMs to perform dynamic keyword profiling guided by the data segments and anomalies discovered by LLmFPCA-detect, and we show that cluster-specific functional PC scores from LLmFPCA-detect, used as features in existing pipelines, help boost prediction performance. We support the stability of LLmFPCA-detect with experiments and evaluate it on two different applications using public datasets, Amazon customer-review trajectories, and Wikipedia talk-page comment streams, demonstrating utility across domains and outperforming state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14594v1">LLM-driven Knowledge Enhancement for Multimodal Cancer Survival Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Current multimodal survival prediction methods typically rely on pathology images (WSIs) and genomic data, both of which are high-dimensional and redundant, making it difficult to extract discriminative features from them and align different modalities. Moreover, using a simple survival follow-up label is insufficient to supervise such a complex task. To address these challenges, we propose KEMM, an LLM-driven Knowledge-Enhanced Multimodal Model for cancer survival prediction, which integrates expert reports and prognostic background knowledge. 1) Expert reports, provided by pathologists on a case-by-case basis and refined by large language model (LLM), offer succinct and clinically focused diagnostic statements. This information may typically suggest different survival outcomes. 2) Prognostic background knowledge (PBK), generated concisely by LLM, provides valuable prognostic background knowledge on different cancer types, which also enhances survival prediction. To leverage these knowledge, we introduce the knowledge-enhanced cross-modal (KECM) attention module. KECM can effectively guide the network to focus on discriminative and survival-relevant features from highly redundant modalities. Extensive experiments on five datasets demonstrate that KEMM achieves state-of-the-art performance. The code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14531v1">VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models without enhancing architectural capacity, thereby hitting the representational ceiling of the base model. In this work, we propose VersatileFFN, a novel feed-forward network (FFN) that enables flexible reuse of parameters in both width and depth dimensions within a fixed parameter budget. Inspired by the dual-process theory of cognition, VersatileFFN comprises two adaptive pathways: a width-versatile path that generates a mixture of sub-experts from a single shared FFN, mimicking sparse expert routing without increasing parameters, and a depth-versatile path that recursively applies the same FFN to emulate deeper processing for complex tokens. A difficulty-aware gating dynamically balances the two pathways, steering "easy" tokens through the efficient width-wise route and allocating deeper iterative refinement to "hard" tokens. Crucially, both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory. Experiments across diverse benchmarks and model scales demonstrate the effectiveness of the method. The code will be available at https://github.com/huawei-noah/noah-research/tree/master/VersatileFFN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14490v1">PushGen: Push Notifications Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      We present PushGen, an automated framework for generating high-quality push notifications comparable to human-crafted content. With the rise of generative models, there is growing interest in leveraging LLMs for push content generation. Although LLMs make content generation straightforward and cost-effective, maintaining stylistic control and reliable quality assessment remains challenging, as both directly impact user engagement. To address these issues, PushGen combines two key components: (1) a controllable category prompt technique to guide LLM outputs toward desired styles, and (2) a reward model that ranks and selects generated candidates. Extensive offline and online experiments demonstrate its effectiveness, which has been deployed in large-scale industrial applications, serving hundreds of millions of users daily.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14474v1">Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with complex multi-step planning tasks, showing high rates of constraint violations and inconsistent solutions. Existing strategies such as Chain-of-Thought and ReAct rely on implicit state tracking and lack an explicit problem representation. Inspired by classical AI planning, we propose Model-First Reasoning (MFR), a two-phase paradigm in which the LLM first constructs an explicit model of the problem, defining entities, state variables, actions, and constraints, before generating a solution plan. Across multiple planning domains, including medical scheduling, route planning, resource allocation, logic puzzles, and procedural synthesis, MFR reduces constraint violations and improves solution quality compared to Chain-of-Thought and ReAct. Ablation studies show that the explicit modeling phase is critical for these gains. Our results suggest that many LLM planning failures stem from representational deficiencies rather than reasoning limitations, highlighting explicit modeling as a key component for robust and interpretable AI agents. All prompts, evaluation procedures, and task datasets are documented to facilitate reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14448v1">Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14417v1">PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Vehicle Dispatching Systems (VDSs) are critical to the operational efficiency of Automated Container Terminals (ACTs). However, their widespread commercialization is hindered due to their low transferability across diverse terminals. This transferability challenge stems from three limitations: high reliance on port operational specialists, a high demand for terminal-specific data, and time-consuming manual deployment processes. Leveraging the emergence of Large Language Models (LLMs), this paper proposes PortAgent, an LLM-driven vehicle dispatching agent that fully automates the VDS transferring workflow. It bears three features: (1) no need for port operations specialists; (2) low need of data; and (3) fast deployment. Specifically, specialist dependency is eliminated by the Virtual Expert Team (VET). The VET collaborates with four virtual experts, including a Knowledge Retriever, Modeler, Coder, and Debugger, to emulate a human expert team for the VDS transferring workflow. These experts specialize in the domain of terminal VDS via a few-shot example learning approach. Through this approach, the experts are able to learn VDS-domain knowledge from a few VDS examples. These examples are retrieved via a Retrieval-Augmented Generation (RAG) mechanism, mitigating the high demand for terminal-specific data. Furthermore, an automatic VDS design workflow is established among these experts to avoid extra manual interventions. In this workflow, a self-correction loop inspired by the LLM Reflexion framework is created
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14373v1">EcoScapes: LLM-Powered Advice for Crafting Sustainable Cities</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Climate adaptation is vital for the sustainability and sometimes the mere survival of our urban areas. However, small cities often struggle with limited personnel resources and integrating vast amounts of data from multiple sources for a comprehensive analysis. To overcome these challenges, this paper proposes a multi-layered system combining specialized LLMs, satellite imagery analysis and a knowledge base to aid in developing effective climate adaptation strategies. The corresponding code can be found at https://github.com/Photon-GitHub/EcoScapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03508v2">One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Evaluating LLMs' instruction-following ability in multi-topic dialogues is essential yet challenging. Existing benchmarks are limited to a fixed number of turns, susceptible to saturation and failing to account for users' interactive experience. In this work, we propose a novel framework backed by a three-layer tracking mechanism and a query synthesis agent to mimic sequential user behaviors. Incorporating Flow Theory, we introduce process-centric metrics and terminate a conversational evaluation only upon exhausting user patience. Upon this framework, we present EvolIF, an evolving benchmark covering 12 constraint groups. Results indicate that GPT-5 excels, sustaining 14 turns with 66.40% robustness. It outperforms Gemini-3.0-Pro by a margin of 5.59%, while other models trail behind. Resources are available at https://github.com/JiaQiSJTU/EvolvingInstructionFollowing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14288v1">Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and Alerting</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      This paper explores the integration of Large Language Models (LLMs) in the engineering of a Parkinson's Disease (PD) monitoring and alerting ontology through four key methodologies: One Shot (OS) prompt techniques, Chain of Thought (CoT) prompts, X-HCOME, and SimX-HCOME+. The primary objective is to determine whether LLMs alone can create comprehensive ontologies and, if not, whether human-LLM collaboration can achieve this goal. Consequently, the paper assesses the effectiveness of LLMs in automated ontology development and the enhancement achieved through human-LLM collaboration. Initial ontology generation was performed using One Shot (OS) and Chain of Thought (CoT) prompts, demonstrating the capability of LLMs to autonomously construct ontologies for PD monitoring and alerting. However, these outputs were not comprehensive and required substantial human refinement to enhance their completeness and accuracy. X-HCOME, a hybrid ontology engineering approach that combines human expertise with LLM capabilities, showed significant improvements in ontology comprehensiveness. This methodology resulted in ontologies that are very similar to those constructed by experts. Further experimentation with SimX-HCOME+, another hybrid methodology emphasizing continuous human supervision and iterative refinement, highlighted the importance of ongoing human involvement. This approach led to the creation of more comprehensive and accurate ontologies. Overall, the paper underscores the potential of human-LLM collaboration in advancing ontology engineering, particularly in complex domains like PD. The results suggest promising directions for future research, including the development of specialized GPT models for ontology construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14277v1">SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 17 pages, 8 figures, 1 table. Under Review
    </div>
    <details class="paper-abstract">
      The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy over a single source while ignoring other evaluation criteria, such as federated query capability over distributed data stores, as well as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over (potentially federated) knowledge graphs with good accuracy. To mitigate these issues, in this paper, we extend our previous work and describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge with multilingual questions, and a collection of questions from three of the most prevalent knowledge graphs within the field of bioinformatics. Our results demonstrate a substantial increase of 24% in the F1 Score on the state-of-the-art challenge, adaptability to high-resource languages such as English and Spanish, as well as ability to form complex and federated bioinformatics queries. Furthermore, we show that SPARQL-LLM is up to 36x faster than other systems participating in the challenge, while costing a maximum of $0.01 per question, making it suitable for real-time, low-cost text-to-SPARQL applications. One such application deployed over real-world decentralized knowledge graphs can be found at https://www.expasy.org/chat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14233v1">PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15610v3">A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 18 pages, 6 figures (3 graphs + 3 flowchart/architecture diagrams), submitted as a preprint for review consideration in AI for Education or Machine Learning applications in low-resource settings. Includes detailed experiments with LoRA and quantization methods for efficient LLM fine-tuning
    </div>
    <details class="paper-abstract">
      The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04359v2">Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14179v1">A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted to the Second Workshop on Bangla Language Processing (BLP) at IJCNLP-AACL 2025. 14 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Translating from a standard language to its regional dialects is a significant NLP challenge due to scarce data and linguistic variation, a problem prominent in the Bengali language. This paper proposes and compares two novel RAG pipelines for standard-to-dialectal Bengali translation. The first, a Transcript-Based Pipeline, uses large dialect sentence contexts from audio transcripts. The second, a more effective Standardized Sentence-Pairs Pipeline, utilizes structured local\_dialect:standard\_bengali sentence pairs. We evaluated both pipelines across six Bengali dialects and multiple LLMs using BLEU, ChrF, WER, and BERTScore. Our findings show that the sentence-pair pipeline consistently outperforms the transcript-based one, reducing Word Error Rate (WER) from 76\% to 55\% for the Chittagong dialect. Critically, this RAG approach enables smaller models (e.g., Llama-3.1-8B) to outperform much larger models (e.g., GPT-OSS-120B), demonstrating that a well-designed retrieval strategy can be more crucial than model size. This work contributes an effective, fine-tuning-free solution for low-resource dialect translation, offering a practical blueprint for preserving linguistic diversity.
    </details>
</div>
