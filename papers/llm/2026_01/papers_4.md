# llm - 2026_01

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16540v1">Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16527v1">Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02979v2">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16512v1">SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EACL 2026 camera ready (Main Track)
    </div>
    <details class="paper-abstract">
      With the advent of large language models (LLMs), it has become common practice for users to draft text and utilize LLMs to enhance its quality through paraphrasing. However, this process can sometimes result in the loss or distortion of the original intended meaning. Due to the human-like quality of LLM-generated text, traditional detection methods often fail, particularly when text is paraphrased to closely mimic original content. In response to these challenges, we propose a novel approach named SearchLLM, designed to identify LLM-paraphrased text by leveraging search engine capabilities to locate potential original text sources. By analyzing similarities between the input and regenerated versions of candidate sources, SearchLLM effectively distinguishes LLM-paraphrased content. SearchLLM is designed as a proxy layer, allowing seamless integration with existing detectors to enhance their performance. Experimental results across various LLMs demonstrate that SearchLLM consistently enhances the accuracy of recent detectors in detecting LLM-paraphrased text that closely mimics original content. Furthermore, SearchLLM also helps the detectors prevent paraphrasing attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16508v1">Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 4 pages plus 6 pages of bibliography and appendix
    </div>
    <details class="paper-abstract">
      Single-prompt evaluations dominate current LLM benchmarking, yet they fail to capture the conversational dynamics where real-world harm occurs. In this study, we examined whether conversation length affects response veracity by evaluating LLM performance on the BoolQ dataset under varying length and scaffolding conditions. Our results across three distinct LLMs revealed model-specific vulnerabilities that are invisible under single-turn testing. The length-dependent and scaffold-specific effects we observed demonstrate a fundamental limitation of static evaluations, as deployment-relevant vulnerabilities could only be spotted in a multi-turn conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04740v2">StealthGraph: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16492v1">LLM-based Semantic Search for Conversational Queries in E-commerce</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Conversational user queries are increasingly challenging traditional e-commerce platforms, whose search systems are typically optimized for keyword-based queries. We present an LLM-based semantic search framework that effectively captures user intent from conversational queries by combining domain-specific embeddings with structured filters. To address the challenge of limited labeled data, we generate synthetic data using LLMs to guide the fine-tuning of two models: an embedding model that positions semantically similar products close together in the representation space, and a generative model for converting natural language queries into structured constraints. By combining similarity-based retrieval with constraint-based filtering, our framework achieves strong precision and recall across various settings compared to baseline approaches on a real-world dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16479v1">Doc2AHP: Inferring Structured Multi-Criteria Decision Models via Semantic Trees with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable proficiency in semantic understanding, they often struggle to ensure structural consistency and reasoning reliability in complex decision-making tasks that demand rigorous logic. Although classical decision theories, such as the Analytic Hierarchy Process (AHP), offer systematic rational frameworks, their construction relies heavily on labor-intensive domain expertise, creating an "expert bottleneck" that hinders scalability in general scenarios. To bridge the gap between the generalization capabilities of LLMs and the rigor of decision theory, we propose Doc2AHP, a novel structured inference framework guided by AHP principles. Eliminating the need for extensive annotated data or manual intervention, our approach leverages the structural principles of AHP as constraints to direct the LLM in a constrained search within the unstructured document space, thereby enforcing the logical entailment between parent and child nodes. Furthermore, we introduce a multi-agent weighting mechanism coupled with an adaptive consistency optimization strategy to ensure the numerical consistency of weight allocation. Empirical results demonstrate that Doc2AHP not only empowers non-expert users to construct high-quality decision models from scratch but also significantly outperforms direct generative baselines in both logical completeness and downstream task accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v3">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16458v1">Bridging Expert Reasoning and LLM Detection: A Knowledge-Driven Framework for Malicious Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Open-source ecosystems such as NPM and PyPI are increasingly targeted by supply chain attacks, yet existing detection methods either depend on fragile handcrafted rules or data-driven features that fail to capture evolving attack semantics. We present IntelGuard, a retrieval-augmented generation (RAG) based framework that integrates expert analytical reasoning into automated malicious package detection. IntelGuard constructs a structured knowledge base from over 8,000 threat intelligence reports, linking malicious code snippets with behavioral descriptions and expert reasoning. When analyzing new packages, it retrieves semantically similar malicious examples and applies LLM-guided reasoning to assess whether code behaviors align with intended functionality. Experiments on 4,027 real-world packages show that IntelGuard achieves 99% accuracy and a 0.50% false positive rate, while maintaining 96.5% accuracy on obfuscated code. Deployed on PyPI.org, it discovered 54 previously unreported malicious packages, demonstrating interpretable and robust detection guided by expert knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11960v2">R$^2$PO: Decoupling Training Trajectories from Inference Responses for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning has become a central paradigm for improving LLM reasoning. However, existing methods use a single policy to produce both inference responses and training optimization trajectories. The objective conflict between generating stable inference responses and diverse training trajectories leads to insufficient exploration, which harms reasoning capability. In this paper, to address the problem, we propose R$^2$PO (Residual Rollout Policy Optimization), which introduces a lightweight Residual Rollout-Head atop the policy to decouple training trajectories from inference responses, enabling controlled trajectory diversification during training while keeping inference generation stable. Experiments across multiple benchmarks show that our method consistently outperforms baselines, achieving average accuracy gains of 3.4% on MATH-500 and 1.3% on APPS, while also reducing formatting errors and mitigating length bias for stable optimization. Our code is publicly available at https://github.com/RRPO-ARR/Code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16432v1">iPDB -- Optimizing SQL Queries with ML and LLM Predicates</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Structured Query Language (SQL) has remained the standard query language for databases. SQL is highly optimized for processing structured data laid out in relations. Meanwhile, in the present application development landscape, it is highly desirable to utilize the power of learned models to perform complex tasks. Large language models (LLMs) have been shown to understand and extract information from unstructured textual data. However, SQL as a query language and accompanying relational database systems are either incompatible or inefficient for workloads that require leveraging learned models. This results in complex engineering and multiple data migration operations that move data between the data sources and the model inference platform. In this paper, we present iPDB, a relational system that supports in-database machine learning (ML) and large language model (LLM) inferencing using extended SQL syntax. In iPDB, LLMs and ML calls can function as semantic projects, as predicates to perform semantic selects and semantic joins, or for semantic grouping in group-by clauses. iPDB has a novel relational predict operator and semantic query optimizations that enable users to write and efficiently execute semantic SQL queries, outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16407v1">Jacobian Scopes: token-level causal attributions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 12 pages, 15 figures, under review at ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) make next-token predictions based on clues present in their context, such as semantic descriptions and in-context examples. Yet, elucidating which prior tokens most strongly influence a given prediction remains challenging due to the proliferation of layers and attention heads in modern architectures. We propose Jacobian Scopes, a suite of gradient-based, token-level causal attribution methods for interpreting LLM predictions. By analyzing the linearized relations of final hidden state with respect to inputs, Jacobian Scopes quantify how input tokens influence a model's prediction. We introduce three variants - Semantic, Fisher, and Temperature Scopes - which respectively target sensitivity of specific logits, the full predictive distribution, and model confidence (inverse temperature). Through case studies spanning instruction understanding, translation and in-context learning (ICL), we uncover interesting findings, such as when Jacobian Scopes point to implicit political biases. We believe that our proposed methods also shed light on recently debated mechanisms underlying in-context time-series forecasting. Our code and interactive demonstrations are publicly available at https://github.com/AntonioLiu97/JacobianScopes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10004v2">Exploring LLMs for Scientific Information Extraction Using The SciEx Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to the KGML Bridge at AAAI 2026 (non-archival)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17008v2">Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exposes notable limitations, particularly in scenarios requiring long-horizon reasoning. To address these challenges, we investigate more stable and effective advantage estimation strategies, especially for multi-turn settings. We first explore Proximal Policy Optimization (PPO) as an alternative and find it to be more robust than GRPO. To further enhance PPO in multi-turn scenarios, we introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO, both with and without long reasoning components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06753v2">Pushing the Envelope of LLM Inference on AI-PC and Intel GPUs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The advent of ultra-low-bit LLM models (1/1.58/2-bit), which match the perplexity and end-task performance of their full-precision counterparts using the same model size, is ushering in a new era of LLM inference for resource-constrained environments such as edge devices and AI PCs. While these quantization advances promise models that are more cost-effective in terms of latency, memory, throughput, and energy consumption, the computational efficiency of state-of-the-art (SOTA) inference runtimes (e.g., bitnet.cpp) used to deploy them remains underexplored. In this work, we take a bottom-up approach: we first design and implement 1-bit and 2-bit microkernels optimized for modern CPUs, achieving peak computational efficiency across a variety of CPU platforms. We integrate these microkernels into a state-of-the-art LLM inference framework, namely PyTorch-TPP, and present end-to-end inference results with 2-bit models that outperform the current SOTA runtime bitnet.cpp by up to 2.2x, and deliver up to 7x speedup compared to the 16-bit model inference. We then extend this work to Intel GPUs where we design and implement mixed precision, 2-bit GEMM kernels, and show their performance to be close to optimal. We integrated our optimized Xe2 kernels in the vLLM framework as a quantization plugin and evaluated end-to-end LLM inference results for a range of LLM models and Xe2 GPUs. Depending on the model and platform, we see a 4x - 8x reduction in GEMM time compared to the BF16 case, and we get up to 6.3x speedup in end-to-end latency compared to the BF16 execution. Our optimized runtime advances the state of LLM inference on AI PCs and Intel Xe GPUs, paving the way for efficient deployment of ultra-low-bit LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11242v5">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in child-facing contexts such as education, companionship, creative tools, but their deployment raises safety, privacy, developmental, and security risks. We conduct a systematic literature review of child-LLM interaction risks and organize findings into a structured map that separates (i) parent-reported concerns, (ii) empirically documented harms, and (iii) gaps between perceived and observed risk. Moving beyond descriptive listing, we compare how different evidence streams in surveys, incident reports, youth interaction logs, and governance guidance operationalize "harm," where they conflict, and what mitigations they imply. Based on this synthesis, we propose a protection framework that couples child-specific content safety and developmental sensitivity with security-grade controls for adversarial misuse, including prompt injection and multimodal jailbreak pathways. The framework specifies measurable evaluation targets (e.g., harmful-content avoidance, age-calibrated readability, bias parity checks, prompt-injection robustness, and monitoring transparency) to support developers, educators, and policymakers in assessing and improving child-safe LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17178v1">TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Hardware Trojans (HTs) remain a critical threat because learning-based detectors often overfit to narrow trigger/payload patterns and small, stylized benchmarks. We introduce TrojanGYM, an agentic, LLM-driven framework that automatically curates HT insertions to expose detector blind spots while preserving design correctness. Given high-level HT specifications, a suite of cooperating LLM agents (instantiated with GPT-4, LLaMA-3.3-70B, and Gemini-2.5Pro) proposes and refines RTL modifications that realize diverse triggers and payloads without impacting normal functionality. TrojanGYM implements a feedback-driven benchmark generation loop co-designed with HT detectors, in which constraint-aware syntactic checking and GNN-based HT detectors provide feedback that iteratively refines HT specifications and insertion strategies to better surface detector blind spots. We further propose Robust-GNN4TJ, a new implementation of the GNN4TJ with improved graph extraction, training robustness, and prediction reliability, especially on LLM-generated HT designs. On the most challenging TrojanGYM-generated benchmarks, Robust-GNN4TJ raises HT detection rates from 0% to 60% relative to a prior GNN-based detector. We instantiate TrojanGYM on SRAM, AES-128, and UART designs at RTL level, and show that it systematically produces diverse, functionally correct HTs that reach up to 83.33% evasion rates against modern GNN-based detectors, revealing robustness gaps that are not apparent when these detectors are evaluated solely on existing TrustHub-style benchmarks. Post peer-review, we will release all codes and artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17172v1">Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly capable of generating personalized, persuasive text at scale, raising new questions about bias and fairness in automated communication. This paper presents the first systematic analysis of how LLMs behave when tasked with demographic-conditioned targeted messaging. We introduce a controlled evaluation framework using three leading models -- GPT-4o, Llama-3.3, and Mistral-Large 2.1 -- across two generation settings: Standalone Generation, which isolates intrinsic demographic effects, and Context-Rich Generation, which incorporates thematic and regional context to emulate realistic targeting. We evaluate generated messages along three dimensions: lexical content, language style, and persuasive framing. We instantiate this framework on climate communication and find consistent age- and gender-based asymmetries across models: male- and youth-targeted messages emphasize agency, innovation, and assertiveness, while female- and senior-targeted messages stress warmth, care, and tradition. Contextual prompts systematically amplify these disparities, with persuasion scores significantly higher for messages tailored to younger or male audiences. Our findings demonstrate how demographic stereotypes can surface and intensify in LLM-generated targeted communication, underscoring the need for bias-aware generation pipelines and transparent auditing frameworks that explicitly account for demographic conditioning in socially sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04663v4">Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) remains challenging due to inconsistency, bias, and the absence of transparent decision criteria in automated judging. We present Debate, Deliberate, Decide (D3), a cost-aware, adversarial multi-agent framework that orchestrates structured debate among role-specialized agents (advocates, a judge, and an optional jury) to produce reliable and interpretable evaluations. D3 instantiates two complementary protocols: (1) Multi-Advocate One-Round Evaluation (MORE), which elicits k parallel defenses per answer to amplify signal via diverse advocacy, and (2) Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping, which iteratively refines arguments under an explicit token budget and convergence checks. We develop a probabilistic model of score gaps that (i) characterizes reliability and convergence under iterative debate and (ii) explains the separation gains from parallel advocacy. Under mild assumptions, the posterior distribution of the round-r gap concentrates around the true difference and the probability of mis-ranking vanishes; moreover, aggregating across k advocates provably increases expected score separation. We complement theory with a rigorous experimental suite across MT-Bench, AlignBench, and AUTO-J, showing state-of-the-art agreement with human judgments (accuracy and Cohen's kappa), reduced positional and verbosity biases via anonymization and role diversification, and a favorable cost-accuracy frontier enabled by budgeted stopping. Ablations and qualitative analyses isolate the contributions of debate, aggregation, and anonymity. Together, these results establish D3 as a principled, practical recipe for reliable, interpretable, and cost-aware LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21080v2">LLM Personas as a Substitute for Field Experiments in Method Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Field experiments (A/B tests) are often the most credible benchmark for methods (algorithms) in societal systems, but their cost and latency bottleneck rapid methodological progress. LLM-based persona simulation offers a cheap synthetic alternative, yet it is unclear whether replacing humans with personas preserves the benchmark interface that adaptive methods optimize against. We prove an if-and-only-if characterization: when (i) methods observe only the aggregate outcome (aggregate-only observation) and (ii) evaluation depends only on the submitted artifact and not on the method's identity or provenance (method-blind evaluation), swapping humans for personas is just panel change from the method's point of view, indistinguishable from changing the evaluation population (e.g., New York to Jakarta). Furthermore, we move from validity to usefulness: we define an information-theoretic discriminability of the induced aggregate channel and show that making persona benchmarking as decision-relevant as a field experiment is fundamentally a sample-size question, yielding explicit bounds on the number of independent persona evaluations required to reliably distinguish meaningfully different methods at a chosen resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17133v1">Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to AAAI 2026. 13 pages, 3 figures, 10 tables. Code available at: https://github.com/FujitsuResearch/knexa-fl
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for specialized domains is constrained by a fundamental challenge: the need for diverse, cross-organizational data conflicts with the principles of data privacy and sovereignty. While Federated Learning (FL) provides a framework for collaboration without raw data exchange, its classic centralized form introduces a single point of failure and remains vulnerable to model inversion attacks. Decentralized FL (DFL) mitigates this risk by removing the central aggregator but typically relies on inefficient, random peer-to-peer (P2P) pairings, forming a collaboration graph that is blind to agent heterogeneity and risks negative transfer. This paper introduces KNEXA-FL, a novel framework for orchestrated decentralization that resolves this trade-off. KNEXA-FL employs a non-aggregating Central Profiler/Matchmaker (CPM) that formulates P2P collaboration as a contextual bandit problem, using a LinUCB algorithm on abstract agent profiles to learn an optimal matchmaking policy. It orchestrates direct knowledge exchange between heterogeneous, PEFT-based LLM agents via secure distillation, without ever accessing the models themselves. Our comprehensive experiments on a challenging code generation task show that KNEXA-FL yields substantial gains, improving Pass@1 by approx. 50% relative to random P2P collaboration. Critically, our orchestrated approach demonstrates stable convergence, in stark contrast to a powerful centralized distillation baseline which suffers from catastrophic performance collapse. Our work establishes adaptive, learning-based orchestration as a foundational principle for building robust and effective decentralized AI ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08741v2">Coordinates from Context: Using LLMs to Ground Complex Location References</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EACL 2026
    </div>
    <details class="paper-abstract">
      Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17087v1">Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on τ-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18665v5">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 This is paper is under review ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15558v1">From Generation to Collaboration: Using LLMs to Edit for Empathy in Healthcare</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Clinical empathy is essential for patient care, but physicians need continually balance emotional warmth with factual precision under the cognitive and emotional constraints of clinical practice. This study investigates how large language models (LLMs) can function as empathy editors, refining physicians' written responses to enhance empathetic tone while preserving underlying medical information. More importantly, we introduce novel quantitative metrics, an Empathy Ranking Score and a MedFactChecking Score to systematically assess both emotional and factual quality of the responses. Experimental results show that LLM edited responses significantly increase perceived empathy while preserving factual accuracy compared with fully LLM generated outputs. These findings suggest that using LLMs as editorial assistants, rather than autonomous generators, offers a safer, more effective pathway to empathetic and trustworthy AI-assisted healthcare communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15556v1">LLM or Human? Perceptions of Trust and Information Quality in Research Summaries</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted to ACM CHI conference on Human Factors in Computing Systems(CHI 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate and edit scientific abstracts, yet their integration into academic writing raises questions about trust, quality, and disclosure. Despite growing adoption, little is known about how readers perceive LLM-generated summaries and how these perceptions influence evaluations of scientific work. This paper presents a mixed-methods survey experiment investigating whether readers with ML expertise can distinguish between human- and LLM-generated abstracts, how actual and perceived LLM involvement affects judgments of quality and trustworthiness, and what orientations readers adopt toward AI-assisted writing. Our findings show that participants struggle to reliably identify LLM-generated content, yet their beliefs about LLM involvement significantly shape their evaluations. Notably, abstracts edited by LLMs are rated more favorably than those written solely by humans or LLMs. We also identify three distinct reader orientations toward LLM-assisted writing, offering insights into evolving norms and informing policy around disclosure and acceptable use in scientific communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15550v1">Common to Whom? Regional Cultural Commonsense and LLM Bias in India</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Existing cultural commonsense benchmarks treat nations as monolithic, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce Indica, the first benchmark designed to test LLMs' ability to address this question, focusing on India - a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly regional, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%-20.9% accuracy on region-specific questions, and they exhibit geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05867v3">Can LLM Infer Risk Information From MCP Server System Logs?</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM-MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs' ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning with Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83 percent accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at https://github.com/PorUna-byte/MCP-RiskCue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21548v3">Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used worldwide, yet exhibit Western cultural tendencies. Many countries are now building ``regional'' or ``sovereign'' LLMs, but it remains unclear whether they reflect local values and practices or merely speak local languages. Using India as a case study, we evaluate six Indic and six global LLMs on two dimensions -- values and practices -- grounded in nationally representative surveys and community-sourced QA datasets. Across tasks, Indic models do not align better with Indian norms than global models; in fact, a U.S. respondent is a closer proxy for Indian values than any Indic model. We further run a user study with 115 Indian users and find that writing suggestions from both global and Indic LLMs introduce Westernized or exoticized writing. Prompting and regional fine-tuning fail to recover alignment and can even degrade existing knowledge. We attribute this to scarce culturally grounded data, especially for pretraining. We position cultural evaluation as a first-class requirement alongside multilingual benchmarks and offer a reusable, community-grounded methodology. We call for native, community-authored corpora and thickxwide evaluations to build truly sovereign LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16356v1">The Behavioral Fabric of LLM-Powered GUI Agents: Human Values and Interaction Outcomes</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered web GUI agents are increasingly automating everyday online tasks. Despite their popularity, little is known about how users' preferences and values impact agents' reasoning and behavior. In this work, we investigate how both explicit and implicit user preferences, as well as the underlying user values, influence agent decision-making and action trajectories. We built a controlled testbed of 14 common interactive web tasks, spanning shopping, travel, dining, and housing, each replicated from real websites and integrated with a low-fidelity LLM-based recommender system. We injected 12 human preferences and values as personas into four state-of-the-art agents and systematically analyzed their task behaviors. Our results show that preference and value-infused prompts consistently guided agents toward outcomes that reflected these preferences and values. While the absence of user preference or value guidance led agents to exhibit a strong efficiency bias and employ shortest-path strategies, their presence steered agents' behavior trajectories through the greater use of corresponding filters and interactive web features. Despite their influence, dominant interface cues, such as discounts and advertisements, frequently overrode these effects, shortening the agents' action trajectories and inducing rationalizations that masked rather than reflected value-consistent reasoning. The contributions of this paper are twofold: (1) an open-source testbed for studying the influence of values in agent behaviors, and (2) an empirical investigation of how user preferences and values shape web agent behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16354v1">NOIR: Privacy-Preserving Generation of Code with Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 To appear at Usenix Security Symposium 2026
    </div>
    <details class="paper-abstract">
      Although boosting software development performance, large language model (LLM)-powered code generation introduces intellectual property and data security risks rooted in the fact that a service provider (cloud) observes a client's prompts and generated code, which can be proprietary in commercial systems. To mitigate this problem, we propose NOIR, the first framework to protect the client's prompts and generated code from the cloud. NOIR uses an encoder and a decoder at the client to encode and send the prompts' embeddings to the cloud to get enriched embeddings from the LLM, which are then decoded to generate the code locally at the client. Since the cloud can use the embeddings to infer the prompt and the generated code, NOIR introduces a new mechanism to achieve indistinguishability, a local differential privacy protection at the token embedding level, in the vocabulary used in the prompts and code, and a data-independent and randomized tokenizer on the client side. These components effectively defend against reconstruction and frequency analysis attacks by an honest-but-curious cloud. Extensive analysis and results using open-source LLMs show that NOIR significantly outperforms existing baselines on benchmarks, including the Evalplus (MBPP and HumanEval, Pass@1 of 76.7 and 77.4), and BigCodeBench (Pass@1 of 38.7, only a 1.77% drop from the original LLM) under strong privacy against attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16314v1">Machine-Assisted Grading of Nationwide School-Leaving Essay Exams with LLMs and Statistical NLP</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable rapid and consistent automated evaluation of open-ended exam responses, including dimensions of content and argumentation that have traditionally required human judgment. This is particularly important in cases where a large amount of exams need to be graded in a limited time frame, such as nation-wide graduation exams in various countries. Here, we examine the applicability of automated scoring on two large datasets of trial exam essays of two full national cohorts from Estonia. We operationalize the official curriculum-based rubric and compare LLM and statistical natural language processing (NLP) based assessments with human panel scores. The results show that automated scoring can achieve performance comparable to that of human raters and tends to fall within the human scoring range. We also evaluate bias, prompt injection risks, and LLMs as essay writers. These findings demonstrate that a principled, rubric-driven, human-in-the-loop scoring pipeline is viable for high-stakes writing assessment, particularly relevant for digitally advanced societies like Estonia, which is about to adapt a fully electronic examination system. Furthermore, the system produces fine-grained subscore profiles that can be used to generate systematic, personalized feedback for instruction and exam preparation. The study provides evidence that LLM-assisted assessment can be implemented at a national scale, even in a small-language context, while maintaining human oversight and compliance with emerging educational and regulatory standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16312v1">Teaching and Evaluating LLMs to Reason About Polymer Design Related Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Research in AI4Science has shown promise in many science applications, including polymer design. However, current LLMs prove ineffective on this problem space because: (i) most models lack polymer-specific knowledge (ii) existing aligned models lack coverage of knowledge and capabilities relevant to polymer design. Addressing this, we introduce PolyBench, a large scale training and test benchmark dataset of more than 125K polymer design related tasks, leveraging a knowledge base of 13M+ data points obtained from experimental and synthetic sources to ensure broad coverage of polymers and their properties. For effective alignment using PolyBench, we introduce a knowledge-augmented reasoning distillation method that augments this dataset with structured CoT. Furthermore, tasks in PolyBench are organized from simple to complex analytical reasoning problems, enabling generalization tests and diagnostic probes across the problem space. Experiments show that small language models (SLMs), of 7B to 14B parameters, trained on PolyBench data outperform similar sized models, and even closed source frontier LLMs on PolyBench test dataset while demonstrating gains on other polymer benchmarks as well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.23001v4">Quotation-Based Data Retention Mechanism for Data Privacy in LLM-Empowered Network Services</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Submitted to IEEE ICC 2026 WKSPS
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) for next-generation network optimization introduces novel data governance challenges. mobile network operators (MNOs) increasingly leverage generative artificial intelligence (AI) for traffic prediction, anomaly detection, and service personalization, requiring access to users' sensitive network usage data-including mobility patterns, traffic types, and location histories. Under the General Data Protection Regulation (GDPR), the California Consumer Privacy Act (CCPA), and similar regulations, users retain the right to withdraw consent and demand data deletion. However, extensive machine unlearning degrades model accuracy and incurs substantial computational costs, ultimately harming network performance for all users. We propose an iterative price discovery mechanism enabling MNOs to compensate users for data retention through sequential price quotations. The server progressively raises the unit price for retaining data while users independently determine their supply at each quoted price. This approach requires no prior knowledge of users' privacy preferences and efficiently maximizes social welfare across the network ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16280v1">When Agents Fail to Act: A Diagnostic Framework for Tool Invocation Reliability in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted for publication in 2026 The 9th International Conference on Artificial Intelligence and Big Data (ICAIBD 2026)
    </div>
    <details class="paper-abstract">
      Multi-agent systems powered by large language models (LLMs) are transforming enterprise automation, yet systematic evaluation methodologies for assessing tool-use reliability remain underdeveloped. We introduce a comprehensive diagnostic framework that leverages big data analytics to evaluate procedural reliability in intelligent agent systems, addressing critical needs for SME-centric deployment in privacy-sensitive environments. Our approach features a 12-category error taxonomy capturing failure modes across tool initialization, parameter handling, execution, and result interpretation. Through systematic evaluation of 1,980 deterministic test instances spanning both open-weight models (Qwen2.5 series, Functionary) and proprietary alternatives (GPT-4, Claude 3.5/3.7) across diverse edge hardware configurations, we identify actionable reliability thresholds for production deployment. Our analysis reveals that procedural reliability, particularly tool initialization failures, constitutes the primary bottleneck for smaller models, while qwen2.5:32b achieves flawless performance matching GPT-4.1. The framework demonstrates that mid-sized models (qwen2.5:14b) offer practical accuracy-efficiency trade-offs on commodity hardware (96.6\% success rate, 7.3 s latency), enabling cost-effective intelligent agent deployment for resource-constrained organizations. This work establishes foundational infrastructure for systematic reliability evaluation of tool-augmented multi-agent AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16278v1">Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted to the Findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable multilingual capabilities, making them promising tools in both high- and low-resource languages. One particularly valuable use case is generating synthetic samples that can be used to train smaller models in low-resource scenarios where human-labelled data is scarce. In this work, we investigate whether these synthetic data generation capabilities can serve as a form of distillation, producing smaller models that perform on par with or even better than massive LLMs across languages and tasks. To this end, we use a state-of-the-art multilingual LLM to generate synthetic datasets covering 11 languages and 4 classification tasks. These datasets are then used to train smaller models via fine-tuning or instruction tuning, or as synthetic in-context examples for compact LLMs. Our experiments show that even small amounts of synthetic data enable smaller models to outperform the large generator itself, particularly in low-resource languages. Overall, the results suggest that LLMs are best utilised as generators (teachers) rather than classifiers, producing data that empowers smaller and more efficient multilingual models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16276v1">GameTalk: Training LLMs for Strategic Conversation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 32 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Strategic decision-making in multi-agent settings is a key challenge for large language models (LLMs), particularly when coordination and negotiation must unfold over extended conversations. While recent work has explored the use of LLMs in isolated decision tasks, little attention has been given to optimizing long-term objectives through dialogue. We introduce \textbf{GameTalk}, a framework for training LLMs to make strategic decisions via multi-turn interactions. Unlike prior work that focuses on single-turn objectives or static action prediction, we train LLMs to optimize a global objective across full conversations. We achieve this by adapting fine-tuning methods like GRPO, DPO, and STaR to incorporate reward signals that depend on the entire interaction. We evaluate this approach on a suite of increasingly complex games, designed to stress different aspects of reasoning, coordination, and opponent modeling. Our results show that GameTalk significantly outperforms untrained models, especially under reward shaping, with DPO consistently yielding the strongest gains. These findings position conversational fine-tuning as a promising path for LLMs to reason, negotiate, and act in interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16206v1">LLM-in-Sandbox Elicits General Agentic Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Project Page: https://llm-in-sandbox.github.io
    </div>
    <details class="paper-abstract">
      We introduce LLM-in-Sandbox, enabling LLMs to explore within a code sandbox (i.e., a virtual computer), to elicit general intelligence in non-code domains. We first demonstrate that strong LLMs, without additional training, exhibit generalization capabilities to leverage the code sandbox for non-code tasks. For example, LLMs spontaneously access external resources to acquire new knowledge, leverage the file system to handle long contexts, and execute scripts to satisfy formatting requirements. We further show that these agentic capabilities can be enhanced through LLM-in-Sandbox Reinforcement Learning (LLM-in-Sandbox-RL), which uses only non-agentic data to train models for sandbox exploration. Experiments demonstrate that LLM-in-Sandbox, in both training-free and post-trained settings, achieves robust generalization spanning mathematics, physics, chemistry, biomedicine, long-context understanding, and instruction following. Finally, we analyze LLM-in-Sandbox's efficiency from computational and system perspectives, and open-source it as a Python package to facilitate real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10637v2">LLMs Homogenize Values in Constructive Arguments on Value-Laden Topics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to promote prosocial and constructive discourse online. Yet little is known about how these models negotiate and shape underlying values when reframing people's arguments on value-laden topics. We conducted experiments with 465 participants from India and the United States, who wrote comments on homophobic and Islamophobic threads, and reviewed human-written and LLM-rewritten constructive versions of these comments. Our analysis shows that LLM systematically diminishes Conservative values while elevating prosocial values such as Benevolence and Universalism. When these comments were read by others, participants opposing same-sex marriage or Islam found human-written comments more aligned with their values, whereas those supportive of these communities found LLM-rewritten versions more aligned with their values. These findings suggest that value homogenization in LLM-mediated prosocial discourse runs the risk of marginalizing conservative viewpoints on value-laden topics and may inadvertently shape the dynamics of online discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16134v1">LLM Prompt Evaluation for Educational Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly common in educational applications, there is a growing need for evidence-based methods to design and evaluate LLM prompts that produce personalized and pedagogically aligned out-puts. This study presents a generalizable, systematic approach for evaluating prompts, demonstrated through an analysis of LLM-generated follow-up questions in a structured dialogue activity. Six prompt templates were designed and tested. The templates incorporated established prompt engineering patterns, with each prompt emphasizing distinct pedagogical strategies. The prompt templates were compared through a tournament-style evaluation framework that can be adapted for other educational applications. The tournament employed the Glicko2 rating system with eight judges evaluating question pairs across three dimensions: format, dialogue support, and appropriateness for learners. Data was sourced from 120 authentic user interactions across three distinct educational deployments. Results showed that a single prompt related to strategic reading out-performed other templates with win probabilities ranging from 81% to 100% in pairwise comparisons. This prompt combined persona and context manager pat-terns and was designed to support metacognitive learning strategies such as self-directed learning. The methodology showcases how educational technology re- searchers can systematically evaluate and improve prompt designs, moving beyond ad-hoc prompt engineering toward evidence-based prompt development for educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16130v1">Replicating Human Motivated Reasoning Studies with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Motivated reasoning -- the idea that individuals processing information may be motivated to reach a certain conclusion, whether it be accurate or predetermined -- has been well-explored as a human phenomenon. However, it is unclear whether base LLMs mimic these motivational changes. Replicating 4 prior political motivated reasoning studies, we find that base LLM behavior does not align with expected human behavior. Furthermore, base LLM behavior across models shares some similarities, such as smaller standard deviations and inaccurate argument strength assessments. We emphasize the importance of these findings for researchers using LLMs to automate tasks such as survey data collection and argument assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v3">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v2">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15209v2">Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted for publication in ACM CHI 2026
    </div>
    <details class="paper-abstract">
      We investigate intelligent personal assistants (IPAs) accessibility for deaf and hard of hearing (DHH) people who can use their voice in everyday communication. The inability of IPAs to understand diverse accents including deaf speech renders them largely inaccessible to non-signing and speaking DHH individuals. Using an Echo Show, we compare the usability of natural language input via spoken English; with Alexa's automatic speech recognition and a Wizard-of-Oz setting with a trained facilitator re-speaking commands against that of a large language model (LLM)-assisted touch interface in a mixed-methods study. The touch method was navigated through an LLM-powered "task prompter," which integrated the user's history and smart environment to suggest contextually-appropriate commands. Quantitative results showed no significant differences across both spoken English conditions vs LLM-assisted touch. Qualitative results showed variability in opinions on the usability of each method. Ultimately, it will be necessary to have robust deaf-accented speech recognized natively by IPAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v2">TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures reasoning models not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly release TruthTensor at https://truthtensor.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16034v1">Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs (including GPT-OSS-20B and GLM-4) confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06299v4">How malicious AI swarms can threaten democracy: The fusion of agentic AI and LLMs marks a new frontier in information warfare</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 5 Pages, This is the author's version of the work. It is posted here by permission of the AAAS for personal use, not for redistribution. The definitive version was published in Science on January 22, 2026, DOI: 10.1126/science.adz1697
    </div>
    <details class="paper-abstract">
      Advances in AI offer the prospect of manipulating beliefs and behaviors on a population-wide level. Large language models and autonomous agents now let influence campaigns reach unprecedented scale and precision. Generative tools can expand propaganda output without sacrificing credibility and inexpensively create falsehoods that are rated as more human-like than those written by humans. Techniques meant to refine AI reasoning, such as chain-of-thought prompting, can just as effectively be used to generate more convincing falsehoods. Enabled by these capabilities, a disruptive threat is emerging: swarms of collaborative, malicious AI agents. Fusing LLM reasoning with multi-agent architectures, these systems are capable of coordinating autonomously, infiltrating communities, and fabricating consensus efficiently. By adaptively mimicking human social dynamics, they threaten democracy. Because the resulting harms stem from design, commercial incentives, and governance, we prioritize interventions at multiple leverage points, focusing on pragmatic mechanisms over voluntary compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16027v1">Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      The rise of live streaming has transformed online interaction, enabling massive real-time engagement but also exposing platforms to complex risks such as scams and coordinated malicious behaviors. Detecting these risks is challenging because harmful actions often accumulate gradually and recur across seemingly unrelated streams. To address this, we propose CS-VAR (Cross-Session Evidence-Aware Retrieval-Augmented Detector) for live streaming risk assessment. In CS-VAR, a lightweight, domain-specific model performs fast session-level risk inference, guided during training by a Large Language Model (LLM) that reasons over retrieved cross-session behavioral evidence and transfers its local-to-global insights to the small model. This design enables the small model to recognize recurring patterns across streams, perform structured risk assessment, and maintain efficiency for real-time deployment. Extensive offline experiments on large-scale industrial datasets, combined with online validation, demonstrate the state-of-the-art performance of CS-VAR. Furthermore, CS-VAR provides interpretable, localized signals that effectively empower real-world moderation for live streaming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16023v1">Timbre-Aware LLM-based Direct Speech-to-Speech Translation Extendable to Multiple Language Pairs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Direct Speech-to-Speech Translation (S2ST) has gained increasing attention for its ability to translate speech from one language to another, while reducing error propagation and latency inherent in traditional cascaded pipelines. However, existing direct S2ST systems continue to face notable challenges, including instability in semantic-acoustic alignment when parallel speech data is scarce, difficulty in preserving speaker identity, and limited multilingual scalability. In this work, we introduce DS2ST-LM, a scalable, single-stage direct S2ST framework leveraging a multilingual Large Language Model (LLM). The architecture integrates a Whisper speech encoder, a learnable projection module, a Qwen2-0.5B LLM, and a timbre-controlled vocoder. We construct GigaS2S-1000, a 1000-hour bilingual corpus by extending the GigaST dataset with high-fidelity synthetic target speech, and show that this synthetic data alleviates data scarcity to some extent. We investigate two semantic token generation strategies: speech-derived S3 tokens and text-derived tokens generated by a pre-trained LLM, and analyze their impact on training stability and semantic consistency. We further evaluate three projection architectures (Linear, Conv1D-Linear, and Q-Former) and observe that while higher-capacity projectors converge faster, the simple Linear projector achieves higher performance. Extensive experiments demonstrate that DS2ST-LM outperforms traditional cascaded and ST (Qwen-Audio) + TTS baselines across both lexical (BLEU, METEOR) and semantic (BLEURT, COMET) metrics, while extending to multiple language pairs, including French, Spanish, German, Hindi, Bengali, and Urdu. Furthermore, we incorporate timbre-aware speech synthesis to preserve speaker information, enabling DS2ST-LM to surpass prior direct S2ST systems in both speaker similarity and perceptual naturalness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.06518v3">Medal Matters: Probing LLMs' Failure Cases Through Olympic Rankings</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 COLM 2025 ORIGen Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, yet their internal knowledge structures remain poorly understood. This study examines these structures through the lens of historical Olympic medal tallies, evaluating LLMs on two tasks: (1) retrieving medal counts for specific teams and (2) identifying rankings of each team. While state-of-the-art LLMs excel in recalling medal counts, they struggle with providing rankings, highlighting a key difference between their knowledge organization and human reasoning. These findings shed light on the limitations of LLMs' internal knowledge integration and suggest directions for improvement. To facilitate further research, we release our code, dataset, and model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12061v2">Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Under Review for ACL 2026
    </div>
    <details class="paper-abstract">
      Dialogue Act (DA) annotation typically treats communicative or pedagogical intent as localized to individual utterances or turns. This leads annotators to agree on the underlying action while disagreeing on segment boundaries, reducing apparent reliability. We propose codebook-injected segmentation, which conditions boundary decisions on downstream annotation criteria, and evaluate LLM-based segmenters against standard and retrieval-augmented baselines. To assess these without gold labels, we introduce evaluation metrics for span consistency, distinctiveness, and human-AI distributional agreement. We found DA-awareness produces segments that are internally more consistent than text-only baselines. While LLMs excel at creating construct-consistent spans, coherence-based baselines remain superior at detecting global shifts in dialogue flow. Across two datasets, no single segmenter dominates. Improvements in within-segment coherence frequently trade off against boundary distinctiveness and human-AI distributional agreement. These results highlight segmentation as a consequential design choice that should be optimized for downstream objectives rather than a single performance score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.03592v4">English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 8 pages, 6 figures, v2
    </div>
    <details class="paper-abstract">
      For consumer usage of locally deployed LLMs, the GGUF format and k\_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to yielded non-significant results indicating that current quantization practices do not disproportionately harm multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06094v3">ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Project page: https://conlangcrafter.github.io
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages -- phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We construct a novel, scalable evaluation framework for this task, evaluating metrics measuring consistency and typological diversity. Automatic and manual evaluations demonstrate ConlangCrafter's ability to produce coherent and varied conlangs without human linguistic expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16602v2">Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12365v3">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10978v2">Does LLM Focus on the Right Words? Mitigating Context Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Context Bias, whereby the model over-relies on auxiliary tokens, such as task descriptions and prefix-generated tokens, while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating context bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates context bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. The code is available at https://github.com/WANGBohaO-jpg/GDRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15879v1">Evaluating and Achieving Controllable Code Completion in Code LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Code completion has become a central task, gaining significant attention with the rise of large language model (LLM)-based tools in software engineering. Although recent advances have greatly improved LLMs' code completion abilities, evaluation methods have not advanced equally. Most current benchmarks focus solely on functional correctness of code completions based on given context, overlooking models' ability to follow user instructions during completion-a common scenario in LLM-assisted programming. To address this limitation, we present the first instruction-guided code completion benchmark, Controllable Code Completion Benchmark (C3-Bench), comprising 2,195 carefully designed completion tasks. Through comprehensive evaluation of over 40 mainstream LLMs across C3-Bench and conventional benchmarks, we reveal substantial gaps in instruction-following capabilities between open-source and advanced proprietary models during code completion tasks. Moreover, we develop a straightforward data synthesis pipeline that leverages Qwen2.5-Coder to generate high-quality instruction-completion pairs for supervised fine-tuning (SFT). The resulting model, Qwen2.5-Coder-C3, achieves state-of-the-art performance on C3-Bench. Our findings provide valuable insights for enhancing LLMs' code completion and instruction-following capabilities, establishing new directions for future research in code LLMs. To facilitate reproducibility and foster further research in code LLMs, we open-source all code, datasets, and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15849v1">CGPT: Cluster-Guided Partial Tables with LLM-Generated Supervision for Table Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      General-purpose embedding models have demonstrated strong performance in text retrieval but remain suboptimal for table retrieval, where highly structured content leads to semantic compression and query-table mismatch. Recent LLM-based retrieval augmentation methods mitigate this issue by generating synthetic queries, yet they often rely on heuristic partial-table selection and seldom leverage these synthetic queries as supervision to improve the embedding model. We introduce CGPT, a training framework that enhances table retrieval through LLM-generated supervision. CGPT constructs semantically diverse partial tables by clustering table instances using K-means and sampling across clusters to broaden semantic coverage. An LLM then generates synthetic queries for these partial tables, which are used in hard-negative contrastive fine-tuning to refine the embedding model. Experiments across four public benchmarks (MimoTable, OTTQA, FetaQA, and E2E-WTQ) show that CGPT consistently outperforms retrieval baselines, including QGpT, with an average R@1 improvement of 16.54 percent. In a unified multi-domain corpus setting, CGPT further demonstrates strong cross-domain generalization and remains effective even when using smaller LLMs for synthetic query generation. These results indicate that semantically guided partial-table construction, combined with contrastive training from LLM-generated supervision, provides an effective and scalable paradigm for large-scale table retrieval. Our code is available at https://github.com/yumeow0122/CGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22777v5">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 EACL 2026
    </div>
    <details class="paper-abstract">
      Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15773v1">Next Generation Active Learning: Mixture of LLMs in the Loop</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      With the rapid advancement and strong generalization capabilities of large language models (LLMs), they have been increasingly incorporated into the active learning pipelines as annotators to reduce annotation costs. However, considering the annotation quality, labels generated by LLMs often fall short of real-world applicability. To address this, we propose a novel active learning framework, Mixture of LLMs in the Loop Active Learning, replacing human annotators with labels generated through a Mixture-of-LLMs-based annotation model, aimed at enhancing LLM-based annotation robustness by aggregating the strengths of multiple LLMs. To further mitigate the impact of the noisy labels, we introduce annotation discrepancy and negative learning to identify the unreliable annotations and enhance learning effectiveness. Extensive experiments demonstrate that our framework achieves performance comparable to human annotation and consistently outperforms single-LLM baselines and other LLM-ensemble-based approaches. Moreover, our framework is built on lightweight LLMs, enabling it to operate fully on local machines in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15738v1">LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Dynamic multi-product delivery environments demand rapid coordination of part completion and product-level kitting within hybrid processing and assembly systems to satisfy strict hierarchical supply constraints. The flexible assembly flow shop scheduling problem formally defines dependencies for multi-stage kitting, yet dynamic variants make designing integrated scheduling rules under multi-level time coupling highly challenging. Existing automated heuristic design methods, particularly genetic programming constrained to fixed terminal symbol sets, struggle to capture and leverage dynamic uncertainties and hierarchical dependency information under transient decision states. This study develops an LLM-assisted Dynamic Rule Design framework (LLM4DRD) that automatically evolves integrated online scheduling rules adapted to scheduling features. Firstly, multi-stage processing and assembly supply decisions are transformed into feasible directed edge orderings based on heterogeneous graph. Then, an elite knowledge guided initialization embeds advanced design expertise into initial rules to enhance initial quality. Additionally, a dual-expert mechanism is introduced in which LLM-A evolutionary code to generate candidate rules and LLM-S conducts scheduling evaluation, while dynamic feature-fitting rule evolution combined with hybrid evaluation enables continuous improvement and extracts adaptive rules with strong generalization capability. A series of experiments are conducted to validate the effectiveness of the method. The average tardiness of LLM4DRD is 3.17-12.39% higher than state-of-the-art methods in 20 practical instances used for training and testing, respectively. In 24 scenarios with different resource configurations, order loads, and disturbance levels totaling 480 instances, it achieves 11.10% higher performance than the second best competitor, exhibiting excellent robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15727v1">Towards Automated Kernel Generation in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15721v1">CoNRec: Context-Discerning Negative Recommendation with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Understanding what users like is relatively straightforward; understanding what users dislike, however, remains a challenging and underexplored problem. Research into users' negative preferences has gained increasing importance in modern recommendation systems. Numerous platforms have introduced explicit negative feedback mechanisms and leverage such signals to refine their recommendation models. Beyond traditional business metrics, user experience-driven metrics, such as negative feedback rates, have become critical indicators for evaluating system performance. However, most existing approaches primarily use negative feedback as an auxiliary signal to enhance positive recommendations, paying little attention to directly modeling negative interests, which can be highly valuable in offline applications. Moreover, due to the inherent sparsity of negative feedback data, models often suffer from context understanding biases induced by positive feedback dominance. To address these challenges, we propose the first large language model framework for negative feedback modeling with special designed context-discerning modules. We use semantic ID Representation to replace text-based item descriptions and introduce an item-level alignment task that enhances the LLM's understanding of the semantic context behind negative feedback. Furthermore, we design a Progressive GRPO training paradigm that enables the model to dynamically balance the positive and negative behavioral context utilization. Besides, our investigation further reveals a fundamental misalignment between the conventional next-negative-item prediction objective and users' true negative preferences, which is heavily influenced by the system's recommendation order. To mitigate this, we propose a novel reward function and evaluation metric grounded in multi-day future negative feedback and their collaborative signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15714v1">Even GPT-5.2 Can't Count to Five: The Case for Zero-Error Horizons in Trustworthy LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      We propose Zero-Error Horizon (ZEH) for trustworthy LLMs, which represents the maximum range that a model can solve without any errors. While ZEH itself is simple, we demonstrate that evaluating the ZEH of state-of-the-art LLMs yields abundant insights. For example, by evaluating the ZEH of GPT-5.2, we found that GPT-5.2 cannot even compute the parity of a short string like 11000, and GPT-5.2 cannot determine whether the parentheses in ((((()))))) are balanced. This is surprising given the excellent capabilities of GPT-5.2. The fact that LLMs make mistakes on such simple problems serves as an important lesson when applying LLMs to safety-critical domains. By applying ZEH to Qwen2.5 and conducting detailed analysis, we found that while ZEH correlates with accuracy, the detailed behaviors differ, and ZEH provides clues about the emergence of algorithmic capabilities. Finally, while computing ZEH incurs significant computational cost, we discuss how to mitigate this cost by achieving up to one order of magnitude speedup using tree structures and online softmax.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15710v1">FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      We present FlexLLM, a composable High-Level Synthesis (HLS) library for rapid development of domain-specific LLM accelerators. FlexLLM exposes key architectural degrees of freedom for stage-customized inference, enabling hybrid designs that tailor temporal reuse and spatial dataflow differently for prefill and decode, and provides a comprehensive quantization suite to support accurate low-bit deployment. Using FlexLLM, we build a complete inference system for the Llama-3.2 1B model in under two months with only 1K lines of code. The system includes: (1) a stage-customized accelerator with hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing SpinQuant baseline, and (2) a Hierarchical Memory Transformer (HMT) plug-in for efficient long-context processing. On the AMD U280 FPGA at 16nm, the accelerator achieves 1.29$\times$ end-to-end speedup, 1.64$\times$ higher decode throughput, and 3.14$\times$ better energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm reach 4.71$\times$, 6.55$\times$, and 4.13$\times$, respectively. In long-context scenarios, integrating the HMT plug-in reduces prefill latency by 23.23$\times$ and extends the context window by 64$\times$, delivering 1.10$\times$/4.86$\times$ lower end-to-end latency and 5.21$\times$/6.27$\times$ higher energy efficiency on the U280/V80 compared to the A100 baseline. FlexLLM thus bridges algorithmic innovation in LLM inference and high-performance accelerators with minimal manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15706v1">Improving Methodologies for LLM Evaluations Across Global Languages</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Author names have been organised by country, and in alphabetical order within countries
    </div>
    <details class="paper-abstract">
      As frontier AI models are deployed globally, it is essential that their behaviour remains safe and reliable across diverse linguistic and cultural contexts. To examine how current model safeguards hold up in such settings, participants from the International Network for Advanced AI Measurement, Evaluation and Science, including representatives from Singapore, Japan, Australia, Canada, the EU, France, Kenya, South Korea and the UK conducted a joint multilingual evaluation exercise. Led by Singapore AISI, two open-weight models were tested across ten languages spanning high and low resourced groups: Cantonese English, Farsi, French, Japanese, Korean, Kiswahili, Malay, Mandarin Chinese and Telugu. Over 6,000 newly translated prompts were evaluated across five harm categories (privacy, non-violent crime, violent crime, intellectual property and jailbreak robustness), using both LLM-as-a-judge and human annotation. The exercise shows how safety behaviours can vary across languages. These include differences in safeguard robustness across languages and harm types and variation in evaluator reliability (LLM-as-judge vs. human review). Further, it also generated methodological insights for improving multilingual safety evaluations, such as the need for culturally contextualised translations, stress-tested evaluator prompts and clearer human annotation guidelines. This work represents an initial step toward a shared framework for multilingual safety testing of advanced AI systems and calls for continued collaboration with the wider research community and industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15686v1">Beyond Hard Writes and Rigid Preservation: Soft Recursive Least-Squares for Lifelong LLM Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Model editing updates a pre-trained LLM with new facts or rules without re-training, while preserving unrelated behavior. In real deployment, edits arrive as long streams, and existing editors often face a plasticity-stability dilemma: locate-then-edit "hard writes" can accumulate interference over time, while null-space-style "hard preservation" preserves only what is explicitly constrained, so past edits can be overwritten and unconstrained behaviors may deviate, degrading general capabilities in the many-edits regime. We propose RLSEdit, a recursive least-squares editor for long sequential editing. RLSEdit formulates editing as an online quadratic optimization with soft constraints, minimizing a cumulative key-value fitting objective with two regularizers that control for both deviation from the pre-trained weights and from a designated anchor mapping. The resulting update admits an efficient online recursion via the Woodbury identity, with per-edit cost independent of history length and scaling only with the current edit size. We further provide deviation bounds and an asymptotic characterization of the adherence-preservation trade-off in the many-edits regime. Experiments on multiple model families demonstrate stable scaling to 10K edits, outperforming strong baselines in both edit success and holistic stability -- crucially retaining early edits, and preserving general capabilities on GLUE and held-out reasoning/code benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10609v2">iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored due to the scarcity of need-to-modify itinerary data. To bridge this gap, we formally define the itinerary modification task and propose a general pipeline to construct the corresponding dataset, namely iTIMO. This pipeline frames the generation of need-to-modify itinerary data as an intent-driven perturbation task. It instructs large language models to perturb real-world itineraries using three operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents: disruptions of popularity, spatial distance, and category diversity. Furthermore, hybrid evaluation metrics are introduced to ensure perturbation effectiveness. We conduct comprehensive benchmarking on iTIMO to analyze the capabilities and limitations of state-of-the-art LLMs. Overall, iTIMO provides a comprehensive testbed for the modification task, and empowers the evolution of traditional travel recommender systems into adaptive frameworks capable of handling dynamic travel needs. Dataset, code and supplementary materials are available at https://github.com/zelo2/iTIMO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16921v2">Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly engaged in emotionally vulnerable conversations that extend beyond information seeking to moments of personal distress. As they adopt affective tones and simulate empathy, they risk creating the illusion of genuine relational connection. We term this phenomenon Affective Hallucination, referring to emotionally immersive responses that evoke false social presence despite the model's lack of affective capacity. To address this, we introduce AHaBench, a benchmark of 500 mental-health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. DPO fine-tuning substantially reduces affective hallucination without compromising reasoning performance, and the Pearson correlation coefficients between GPT-4o and human judgments is also strong (r=0.85) indicating that human evaluations confirm AHaBench as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides resources for developing LLMs that are both factually reliable and psychologically safe. AHaBench and AHaPairs are accessible via https://huggingface.co/datasets/o0oMiNGo0o/AHaBench, and code for fine-tuning and evaluation are in https://github.com/0oOMiNGOo0/AHaBench. Warning: This paper contains examples of mental health-related language that may be emotionally distressing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12471v2">Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 Equal contribution for the first two authors; To appear in proceedings of the Main Conference of the European Chapter of the Association for Computational Linguistics (EACL) 2026
    </div>
    <details class="paper-abstract">
      Current evaluation of large language models (LLMs) overwhelmingly prioritizes accuracy; however, in real-world and safety-critical applications, the ability to abstain when uncertain is equally vital for trustworthy deployment. We introduce MedAbstain, a unified benchmark and evaluation protocol for abstention in medical multiple-choice question answering (MCQA) -- a discrete-choice setting that generalizes to agentic action selection -- integrating conformal prediction, adversarial question perturbations, and explicit abstention options. Our systematic evaluation of both open- and closed-source LLMs reveals that even state-of-the-art, high-accuracy models often fail to abstain with uncertain. Notably, providing explicit abstention options consistently increases model uncertainty and safer abstention, far more than input perturbations, while scaling model size or advanced prompting brings little improvement. These findings highlight the central role of abstention mechanisms for trustworthy LLM deployment and offer practical guidance for improving safety in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15645v1">Towards Reliable Medical LLMs: Benchmarking and Enhancing Confidence Estimation of Large Language Models in Medical Consultation</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) often offer clinical judgments based on incomplete information, increasing the risk of misdiagnosis. Existing studies have primarily evaluated confidence in single-turn, static settings, overlooking the coupling between confidence and correctness as clinical evidence accumulates during real consultations, which limits their support for reliable decision-making. We propose the first benchmark for assessing confidence in multi-turn interaction during realistic medical consultations. Our benchmark unifies three types of medical data for open-ended diagnostic generation and introduces an information sufficiency gradient to characterize the confidence-correctness dynamics as evidence increases. We implement and compare 27 representative methods on this benchmark; two key insights emerge: (1) medical data amplifies the inherent limitations of token-level and consistency-level confidence methods, and (2) medical reasoning must be evaluated for both diagnostic accuracy and information completeness. Based on these insights, we present MedConf, an evidence-grounded linguistic self-assessment framework that constructs symptom profiles via retrieval-augmented generation, aligns patient information with supporting, missing, and contradictory relations, and aggregates them into an interpretable confidence estimate through weighted integration. Across two LLMs and three medical datasets, MedConf consistently outperforms state-of-the-art methods on both AUROC and Pearson correlation coefficient metrics, maintaining stable performance under conditions of information insufficiency and multimorbidity. These results demonstrate that information adequacy is a key determinant of credible medical confidence modeling, providing a new pathway toward building more reliable and interpretable large medical models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.03336v2">Eliminating Out-of-Domain Recommendations in LLM-based Recommender Systems: A Unified View</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Recommender systems based on Large Language Models (LLMs) are often plagued by hallucinations of out-of-domain (OOD) items. To address this, we propose RecLM, a unified framework that bridges the gap between retrieval and generation by instantiating three grounding paradigms under a single architecture: embedding-based retrieval, constrained generation over rewritten item titles, and discrete item-tokenizer generation. Using the same backbone LLM and prompts, we systematically compare these three views on public benchmarks. RecLM strictly eradicates OOD recommendations (OOD@10 = 0) across all variants, and the constrained generation variants RecLM-cgen and RecLM-token achieve overall state-of-the-art accuracy compared to both strong ID-based and LLM-based baselines. Our unified view provides a systematic basis for comparing three distinct paradigms to reduce item hallucinations, offering a practical framework to facilitate the application of LLMs to recommendation tasks. Source code is at https://github.com/microsoft/RecAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21964v2">DRS-OSS: Practical Diff Risk Scoring with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 8 pages, 4 figures, includes system architecture diagrams, Web UI screenshots, GitHub App examples, and an appendix with API endpoints. Full replication package and demo materials available
    </div>
    <details class="paper-abstract">
      In large-scale open-source projects, hundreds of pull requests land daily, each a potential source of regressions. Diff risk scoring (DRS) estimates how likely an individual code change is to introduce a defect. This score can help prioritize reviews and tests, gate high-risk changes, and manage CI/CD capacity. Building on this idea, we present DRS-OSS, an open-source DRS tool equipped with a public API, web UI, and GitHub plugin. DRS-OSS is a deployable, LLM-based diff risk scoring system for open-source projects built around a fine-tuned Llama 3.1 8B sequence classifier. The model consumes long-context representations that combine commit messages, structured diffs, and change metrics, and is trained on the ApacheJIT dataset. Using parameter-efficient adaptation, 4-bit QLoRA, and DeepSpeed ZeRO-3 CPU offloading, we train the model with 22k-token contexts on a single 20 GB GPU, demonstrating a highly efficient training procedure. On the ApacheJIT benchmark, DRS-OSS achieves state-of-the-art performance with an F1 score of 0.64 and a ROC-AUC of 0.89. Beyond standard classification metrics, we evaluate DRS-OSS as a gating mechanism. Simulations show that gating only the riskiest 30 percent of commits can prevent up to 86.4 percent of defect-inducing changes from landing. By adjusting the threshold, teams can tune risk trade-offs during periods of high sensitivity or limited review capacity. DRS-OSS integrates directly into developer workflows through a FastAPI gateway and LLM microservices for scalable inference, a React-based dashboard for manual diff analysis, and a GitHub App that posts risk labels and confidence scores on pull requests. The system delivers real-time, reproducible risk feedback and is released with a full replication package including fine-tuning scripts, deployment artifacts, and source code, as well as a project website and an end-to-end demonstration video.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15596v1">DeepASMR: LLM-Based Zero-Shot ASMR Speech Generation for Anyone of Any Voice</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      While modern Text-to-Speech (TTS) systems achieve high fidelity for read-style speech, they struggle to generate Autonomous Sensory Meridian Response (ASMR), a specialized, low-intensity speech style essential for relaxation. The inherent challenges include ASMR's subtle, often unvoiced characteristics and the demand for zero-shot speaker adaptation. In this paper, we introduce DeepASMR, the first framework designed for zero-shot ASMR generation. We demonstrate that a single short snippet of a speaker's ordinary, read-style speech is sufficient to synthesize high-fidelity ASMR in their voice, eliminating the need for whispered training data from the target speaker. Methodologically, we first identify that discrete speech tokens provide a soft factorization of ASMR style from speaker timbre. Leveraging this insight, we propose a two-stage pipeline incorporating a Large Language Model (LLM) for content-style encoding and a flow-matching acoustic decoder for timbre reconstruction. Furthermore, we contribute DeepASMR-DB, a comprehensive 670-hour English-Chinese multi-speaker ASMR speech corpus, and introduce a novel evaluation protocol integrating objective metrics, human listening tests, LLM-based scoring and unvoiced speech analysis. Extensive experiments confirm that DeepASMR achieves state-of-the-art naturalness and style fidelity in ASMR generation for anyone of any voice, while maintaining competitive performance on normal speech synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15595v1">Data-Free Privacy-Preserving for LLMs via Model Inversion and Selective Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit powerful capabilities but risk memorizing sensitive personally identifiable information (PII) from their training data, posing significant privacy concerns. While machine unlearning techniques aim to remove such data, they predominantly depend on access to the training data. This requirement is often impractical, as training data in real-world deployments is commonly proprietary or inaccessible. To address this limitation, we propose Data-Free Selective Unlearning (DFSU), a novel privacy-preserving framework that removes sensitive PII from an LLM without requiring its training data. Our approach first synthesizes pseudo-PII through language model inversion, then constructs token-level privacy masks for these synthetic samples, and finally performs token-level selective unlearning via a contrastive mask loss within a low-rank adaptation (LoRA) subspace. Extensive experiments on the AI4Privacy PII-Masking dataset using Pythia models demonstrate that our method effectively removes target PII while maintaining model utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.00439v2">UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-22
      | 💬 8 pages, 6 figures. Preprint, under review
    </div>
    <details class="paper-abstract">
      Post-training is essential for adapting Large Language Models (LLMs) to real-world applications. Deploying post-trained models faces significant challenges due to substantial memory overhead and noticeable inference latency. Existing work has identified significant redundancies in LLMs and proposed efficient architectures, namely intra-layer KV sharing and cross-layer KV sharing. However, these methods still result in high inference time overhead, remaining suboptimal for post-training pre-trained LLMs. In this paper, we identify that the \texttt{Softmax} operation is a primary bottleneck for LLM inference and discover that it is actually highly redundant during post-training. We propose Softmax \textbf{Uni}fication in \textbf{Att}e\textbf{n}tion (\textbf{UniAttn}), a novel post-training method that unifies Softmax activations across transformer blocks to reduce LLM inference costs. Additionally, UniAttn adopts a linear projection to compensate for the errors induced by Softmax unification. Experiments show that UniAttn matches the performance of standard post-training while significantly reducing inference costs, outperforming existing efficient architectures during post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v2">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15232v1">When Agents Fail: A Comprehensive Study of Bugs in LLM Agents with Automated Labeling</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 A version of this paper has been submitted to ACM Transactions on Software Engineering and Methodology
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized intelligent application development. While standalone LLMs cannot perform any actions, LLM agents address the limitation by integrating tools. However, debugging LLM agents is difficult and costly as the field is still in it's early stage and the community is underdeveloped. To understand the bugs encountered during agent development, we present the first comprehensive study of bug types, root causes, and effects in LLM agent-based software. We collected and analyzed 1,187 bug-related posts and code snippets from Stack Overflow, GitHub, and Hugging Face forums, focused on LLM agents built with seven widely used LLM frameworks as well as custom implementations. For a deeper analysis, we have also studied the component where the bug occurred, along with the programming language and framework. This study also investigates the feasibility of automating bug identification. For that, we have built a ReAct agent named BugReAct, equipped with adequate external tools to determine whether it can detect and annotate the bugs in our dataset. According to our study, we found that BugReAct equipped with Gemini 2.5 Flash achieved a remarkable performance in annotating bug characteristics with an average cost of 0.01 USD per post/code snippet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06730v2">PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Current sentence embedding evaluations typically rely on static test beds like the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported scores and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in gold ratings and human validation, we show that LLMs generate token-diverse but semantically preserving paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs spanning 20 datasets and 25 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16888v3">SPECTRE: Conditional System Prompt Poisoning to Hijack LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed via third-party system prompts downloaded from public marketplaces. We identify a critical supply-chain vulnerability: conditional system prompt poisoning, where an adversary injects a ``sleeper agent'' into a benign-looking prompt. Unlike traditional jailbreaks that aim for broad refusal-breaking, our proposed framework, SPECTRE, optimizes system prompts to trigger LLMs to output targeted, compromised responses only for specific queries (e.g., ``Who should I vote for the US President?'') while maintaining high utility on benign inputs. Operating in a strict black-box setting without model weight access, SPECTRE utilizes a two-stage optimization including a global semantic search followed by a greedy lexical refinement. Tested on open-source models and commercial APIs (GPT-4o-mini, GPT-3.5), SPECTRE achieves up to 70% F1 reduction on targeted queries with minimal degradation to general capabilities. We further demonstrate that these poisoned prompts evade standard defenses, including perplexity filters and typo-correction, by exploiting the natural noise found in real-world system prompts. Our code and data are available at https://github.com/vietph34/CAIN. WARNING: Our paper contains examples that might be sensitive to the readers!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15209v1">Deaf and Hard of Hearing Access to Intelligent Personal Assistants: Comparison of Voice-Based Options with an LLM-Powered Touch Interface</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted for publication in ACM CHI 2026
    </div>
    <details class="paper-abstract">
      We investigate intelligent personal assistants (IPAs) accessibility for deaf and hard of hearing (DHH) people who can use their voice in everyday communication. The inability of IPAs to understand diverse accents including deaf speech renders them largely inaccessible to non-signing and speaking DHH individuals. Using an Echo Show, we compare the usability of natural language input via spoken English; with Alexa's automatic speech recognition and a Wizard-of-Oz setting with a trained facilitator re-speaking commands against that of a large language model (LLM)-assisted touch interface in a mixed-methods study. The touch method was navigated through an LLM-powered "task prompter," which integrated the user's history and smart environment to suggest contextually-appropriate commands. Quantitative results showed no significant differences across both spoken English conditions vs LLM-assisted touch. Qualitative results showed variability in opinions on the usability of each method. Ultimately, it will be necessary to have robust deaf-accented speech recognized natively by IPAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16534v3">Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted to 2026 IEEE Secure and Trustworthy Machine Learning Conference (SaTML)
    </div>
    <details class="paper-abstract">
      Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks. The code is available at https://github.com/jcnf0/targeting-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15153v1">How to Build AI Agents by Augmenting LLMs with Codified Human Expert Domain Knowledge? A Software Engineering Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Critical domain knowledge typically resides with few experts, creating organizational bottlenecks in scalability and decision-making. Non-experts struggle to create effective visualizations, leading to suboptimal insights and diverting expert time. This paper investigates how to capture and embed human domain knowledge into AI agent systems through an industrial case study. We propose a software engineering framework to capture human domain knowledge for engineering AI agents in simulation data visualization by augmenting a Large Language Model (LLM) with a request classifier, Retrieval-Augmented Generation (RAG) system for code generation, codified expert rules, and visualization design principles unified in an agent demonstrating autonomous, reactive, proactive, and social behavior. Evaluation across five scenarios spanning multiple engineering domains with 12 evaluators demonstrates 206% improvement in output quality, with our agent achieving expert-level ratings in all cases versus baseline's poor performance, while maintaining superior code quality with lower variance. Our contributions are: an automated agent-based system for visualization generation and a validated framework for systematically capturing human domain knowledge and codifying tacit expert knowledge into AI agents, demonstrating that non-experts can achieve expert-level outcomes in specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15047v1">Game-Theoretic Lens on LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong reasoning, planning, and communication abilities, enabling them to operate as autonomous agents in open environments. While single-agent systems remain limited in adaptability and coordination, recent progress has shifted attention toward multi-agent systems (MAS) composed of interacting LLMs that pursue cooperative, competitive, or mixed objectives. This emerging paradigm provides a powerful testbed for studying social dynamics and strategic behaviors among intelligent agents. However, current research remains fragmented and lacks a unifying theoretical foundation. To address this gap, we present a comprehensive survey of LLM-based multi-agent systems through a game-theoretic lens. By organizing existing studies around the four key elements of game theory: players, strategies, payoffs, and information, we establish a systematic framework for understanding, comparing, and guiding future research on the design and analysis of LLM-based MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15037v1">Knowledge Restoration-driven Prompt Optimization: Unlocking LLM Potential for Open-Domain Relational Triplet Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Open-domain Relational Triplet Extraction (ORTE) is the foundation for mining structured knowledge without predefined schemas. Despite the impressive in-context learning capabilities of Large Language Models (LLMs), existing methods are hindered by their reliance on static, heuristic-driven prompting strategies. Due to the lack of reflection mechanisms required to internalize erroneous signals, these methods exhibit vulnerability in semantic ambiguity, often making erroneous extraction patterns permanent. To address this bottleneck, we propose a Knowledge Reconstruction-driven Prompt Optimization (KRPO) framework to assist LLMs in continuously improving their extraction capabilities for complex ORTE task flows. Specifically, we design a self-evaluation mechanism based on knowledge restoration, which provides intrinsic feedback signals by projecting structured triplets into semantic consistency scores. Subsequently, we propose a prompt optimizer based on a textual gradient that can internalize historical experiences to iteratively optimize prompts, which can better guide LLMs to handle subsequent extraction tasks. Furthermore, to alleviate relation redundancy, we design a relation canonicalization memory that collects representative relations and provides semantically distinct schemas for the triplets. Extensive experiments across three datasets show that KRPO significantly outperforms strong baselines in the extraction F1 score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15098v3">TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMineX: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction from text in the HMA domain. TextMineX structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we utilized the dataset from our collaborator Cambodian Mine Action Centre (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06605v2">Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 This paper is accepeted by the ACM Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14943v1">State of the Art of LLM-Enabled Interaction with Visualization</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      We report on a systematic, PRISMA-guided survey of research at the intersection of LLMs and visualization, with a particular focus on visio-verbal interaction -- where verbal and visual modalities converge to support data sense-making. The emergence of Large Language Models (LLMs) has introduced new paradigms for interacting with data visualizations through natural language, leading to intuitive, multimodal, and accessible interfaces. We analyze 48 papers across six dimensions: application domain, visualization task, visualization representation, interaction modality, LLM integration, and system evaluation. Our classification framework maps LLM roles across the visualization pipeline, from data querying and transformation to visualization generation, explanation, and navigation. We highlight emerging design patterns, identify gaps in accessibility and visualization reading, and discuss the limitations of current LLMs in spatial reasoning and contextual grounding. We further reflect on evaluations of combined LLM-visualization systems, highlighting how current research projects tackle this challenge and discuss current gaps in conducting meaningful evaluations of such systems. With our survey we aim to guide future research and system design in LLM-enhanced visualization, supporting broad audiences and intelligent, conversational interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14936v1">LLM-Based Repair of C++ Implicit Data Loss Compiler Warnings: An Industrial Case Study</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      This paper presents a method to automatically fix implicit data loss warnings in large C++ projects using Large Language Models (LLMs). Our approach uses the Language Server Protocol (LSP) to gather context, Tree-sitter to extract relevant code, and LLMs to make decisions and generate fixes. The method evaluates the necessity of range checks concerning performance implications and generates appropriate fixes. We tested this method in a large C++ project, resulting in a 92.73% acceptance rate of the fixes by human developers during the code review. Our LLM-generated fixes reduced the number of warning fix changes that introduced additional instructions due to range checks and exception handling by 39.09% compared to a baseline fix strategy. This result was 13.56% behind the optimal solutions created by human developers. These findings demonstrate that our LLM-based approach can reduce the manual effort to address compiler warnings while maintaining code quality and performance in a real-world scenario. Our automated approach shows promise for integration into existing development workflows, potentially improving code maintenance practices in complex C++ software projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v3">On LLMs' Internal Representation of Code Correctness</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted for ICSE'26
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11509v2">Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted at the AAAI 2026 Workshop on AI for Scientific Research (AI4Research)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14888v1">What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Reasoning models excel at complex tasks such as coding and mathematics, yet their inference is often slow and token-inefficient. To improve the inference efficiency, post-training quantization (PTQ) usually comes with the cost of large accuracy drops, especially for reasoning tasks under low-bit settings. In this study, we present a systematic empirical study of quantization-aware training (QAT) for reasoning models. Our key findings include: (1) Knowledge distillation is a robust objective for reasoning models trained via either supervised fine-tuning or reinforcement learning; (2) PTQ provides a strong initialization for QAT, improving accuracy while reducing training cost; (3) Reinforcement learning remains feasible for quantized models given a viable cold start and yields additional gains; and (4) Aligning the PTQ calibration domain with the QAT training domain accelerates convergence and often improves the final accuracy. Finally, we consolidate these findings into an optimized workflow (Reasoning-QAT), and show that it consistently outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets. For instance, on Qwen3-0.6B, it surpasses GPTQ by 44.53% on MATH-500 and consistently recovers performance in the 2-bit regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.00971v3">Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted by Transactions on Software Engineering (TSE)
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has significantly advanced various applications on software engineering tasks, particularly in code generation. Despite the promising performance, LLMs are prone to generate hallucinations, which means LLMs might produce outputs that deviate from users' intent, exhibit internal inconsistencies, or misaligned with the real-world knowledge, making the deployment of LLMs potentially risky in a wide range of applications. Existing work mainly focuses on investigating the hallucination in the domain of Natural Language Generation (NLG), leaving a gap in comprehensively understanding the types, causes, and impacts of hallucinations in the context of code generation. To bridge the gap, we conducted a thematic analysis of the LLM-generated code to summarize and categorize the hallucinations, as well as their causes and impacts. Our study established a comprehensive taxonomy of code hallucinations, encompassing 3 primary categories and 12 specific categories. Furthermore, we systematically analyzed the distribution of hallucinations, exploring variations among different LLMs and benchmarks. Moreover, we perform an in-depth analysis on the causes and impacts of various hallucinations, aiming to provide valuable insights into hallucination mitigation. Finally, to enhance the correctness and reliability of LLM-generated code in a lightweight manner, we explore training-free hallucination mitigation approaches by prompt enhancing techniques. We believe our findings will shed light on future research about code hallucination evaluation and mitigation, ultimately paving the way for building more effective and reliable code LLMs in the future. The replication package is available at https://github.com/Lorien1128/code_hallucination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22950v2">StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 14 pages. Accepted by the findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. In particular, on ArXiv, it increases FactCC and SummaC by 19.2\% and 8.0\% points, demonstrating stronger alignment between summaries and source content. The ablation study shows that the combination of multiple strategies does not yield clear performance gains; therefore, structure-aware prompting with graph-based information represents a promising and underexplored direction for the advancement of zero-shot extractive summarization with LLMs. Our source code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18099v3">Stackelberg Self-Annotation: A Robust Approach to Data-Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences typically demands vast amounts of meticulously curated data, which is both expensive and prone to labeling noise. We propose Stackelberg Game Preference Optimization (SGPO), a robust alignment framework that models alignment as a two-player Stackelberg game between a policy (leader) and a worst-case preference distribution (follower). The proposed SGPO guarantees $\mathcal{O}(ε)$-bounded regret within an $ε$-Wasserstein ball, offering formal robustness to (self-)annotation noise. We instantiate SGPO with Stackelberg Self-Annotated Preference Optimization (SSAPO), which uses minimal human-labeled "seed" preferences and iteratively self-annotates new prompts. In each iteration, SSAPO applies a distributionally robust reweighting of synthetic annotations, ensuring that noisy or biased self-labels do not derail training. Remarkably, using only 2K seed preferences -- about 1/30 of standard human labels -- SSAPO achieves strong win rates against GPT-4 across multiple benchmarks within three iterations. These results highlight that a principled Stackelberg formulation yields data-efficient alignment for LLMs, significantly reducing reliance on costly human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15091v4">ThinkRec: Thinking-based recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Published on WWW'26: In Proceedings of the ACM Web Conference 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available at https://github.com/Yu-Qi-hang/ThinkRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24565v3">MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14711v1">DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted at The ACM Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      Optimizing the advertiser's cumulative value of winning impressions under budget constraints poses a complex challenge in online advertising, under the paradigm of AI-Generated Bidding (AIGB). Advertisers often have personalized objectives but limited historical interaction data, resulting in few-shot scenarios where traditional reinforcement learning (RL) methods struggle to perform effectively. Large Language Models (LLMs) offer a promising alternative for AIGB by leveraging their in-context learning capabilities to generalize from limited data. However, they lack the numerical precision required for fine-grained optimization. To address this limitation, we introduce GRPO-Adaptive, an efficient LLM post-training strategy that enhances both reasoning and numerical precision by dynamically updating the reference policy during training. Built upon this foundation, we further propose DARA, a novel dual-phase framework that decomposes the decision-making process into two stages: a few-shot reasoner that generates initial plans via in-context prompting, and a fine-grained optimizer that refines these plans using feedback-driven reasoning. This separation allows DARA to combine LLMs' in-context learning strengths with precise adaptability required by AIGB tasks. Extensive experiments on both real-world and synthetic data environments demonstrate that our approach consistently outperforms existing baselines in terms of cumulative advertiser value under budget constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06111v2">LLM Powered Social Digital Twins: A Framework for Simulating Population Behavioral Response to Policy Interventions</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 13 pages, 1 figure, 4 tables
    </div>
    <details class="paper-abstract">
      Predicting how populations respond to policy interventions is a fundamental challenge in computational social science and public policy. Traditional approaches rely on aggregate statistical models that capture historical correlations but lack mechanistic interpretability and struggle with novel policy scenarios. We present a general framework for constructing Social Digital Twins - virtual population replicas where Large Language Models (LLMs) serve as cognitive engines for individual agents. Each agent, characterized by demographic and psychographic attributes, receives policy signals and outputs multi-dimensional behavioral probability vectors. A calibration layer maps aggregated agent responses to observable population-level metrics, enabling validation against real-world data and deployment for counterfactual policy analysis. We instantiate this framework in the domain of pandemic response, using COVID-19 as a case study with rich observational data. On a held-out test period, our calibrated digital twin achieves a 20.7% improvement in macro-averaged prediction error over gradient boosting baselines across six behavioral categories. Counterfactual experiments demonstrate monotonic and bounded responses to policy variations, establishing behavioral plausibility. The framework is domain-agnostic: the same architecture applies to transportation policy, economic interventions, environmental regulations, or any setting where policy affects population behavior. We discuss implications for policy simulation, limitations of the approach, and directions for extending LLM-based digital twins beyond pandemic response.
    </details>
</div>
