# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18350v1">When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18334v1">Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into clinical workflows, their tendency for sycophancy, prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the Adjusted Sycophancy Score, a novel metric that isolates alignment bias by accounting for stochastic model instability, or "confusability". Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized "Thinking" models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18315v1">CovertComBench: The First Domain-Specific Testbed for LLMs in Wireless Covert Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 6pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wireless networks presents significant potential for automating system design. However, unlike conventional throughput maximization, Covert Communication (CC) requires optimizing transmission utility under strict detection-theoretic constraints, such as Kullback-Leibler divergence limits. Existing benchmarks primarily focus on general reasoning or standard communication tasks and do not adequately evaluate the ability of LLMs to satisfy these rigorous security constraints. To address this limitation, we introduce CovertComBench, a unified benchmark designed to assess LLM capabilities across the CC pipeline, encompassing conceptual understanding (MCQs), optimization derivation (ODQs), and code generation (CGQs). Furthermore, we analyze the reliability of automated scoring within a detection-theoretic ``LLM-as-Judge'' framework. Extensive evaluations across state-of-the-art models reveal a significant performance discrepancy. While LLMs achieve high accuracy in conceptual identification (81%) and code implementation (83%), their performance in the higher-order mathematical derivations necessary for security guarantees ranges between 18% and 55%. This limitation indicates that current LLMs serve better as implementation assistants rather than autonomous solvers for security-constrained optimization. These findings suggest that future research should focus on external tool augmentation to build trustworthy wireless AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16891v4">LLMs as Layout Designers: Enhanced Spatial Reasoning for Content-Aware Layout Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their ability to understand and manipulate spatial relationships remains limited. Such capabilities are crucial for content-aware graphic layout design, where the goal is to arrange heterogeneous elements onto a canvas so that final design remains visually balanced and structurally feasible. This problem requires precise coordination of placement, alignment, and structural organization of multiple elements within a constrained visual space. To address this limitation, we introduce LaySPA, a reinforcement learning-based framework that augments LLM-based agents with explicit spatial reasoning capabilities for layout design. LaySPA employs hybrid reward signals that jointly capture geometric constraints, structural fidelity, and visual quality, enabling agents to navigate the canvas, model inter-element relationships, and optimize spatial arrangements. Through group-relative policy optimization, the agent generates content-aware layouts that reflect salient regions, respect spatial constraints, and produces an interpretable reasoning trace explaining placement decisions and a structured layout specification. Experimental results show that LaySPA substantially improves the generation of structurally valid and visually appealing layouts, outperforming larger general-purpose LLMs and achieving performance comparable to state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16034v2">Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18009v1">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at the 48th European Conference on Information Retrieval (ECIR 2026)
    </div>
    <details class="paper-abstract">
      Implicit feedback -- the main data source for training Recommender Systems (RSs) -- is inherently noisy and has been shown to negatively affect recommendation effectiveness. Denoising has been proposed as a method for removing noisy implicit feedback and improving recommendations. Prior work has focused on in-training denoising, however this requires additional data, changes to the model architecture and training procedure or fine-tuning, all of which can be costly and data hungry. In this work, we focus on post-training denoising. Different from in-training denoising, post-training denoising does not involve changing the architecture of the model nor its training procedure, and does not require additional data. Specifically, we present a method for post-training denoising user profiles using Large Language Models (LLMs) for Collaborative Filtering (CF) recommendations. Our approach prompts LLMs with (i) a user profile (user interactions), (ii) a candidate item, and (iii) its rank as given by the CF recommender, and asks the LLM to remove items from the user profile to improve the rank of the candidate item. Experiments with a state-of-the-art CF recommender and 4 open and closed source LLMs in 3 datasets show that our denoising yields improvements up to 13% in effectiveness over the original user profiles. Our code is available at https://github.com/edervishaj/denoising-user-profiles-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09897v2">PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 WWW 2026
    </div>
    <details class="paper-abstract">
      Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15979v2">OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17971v1">LLMs as Cultural Archives: Cultural Commonsense Knowledge Graph Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 EACL 2026 MAIN
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode rich cultural knowledge learned from diverse web-scale data, offering an unprecedented opportunity to model cultural commonsense at scale. Yet this knowledge remains mostly implicit and unstructured, limiting its interpretability and use. We present an iterative, prompt-based framework for constructing a Cultural Commonsense Knowledge Graph (CCKG) that treats LLMs as cultural archives, systematically eliciting culture-specific entities, relations, and practices and composing them into multi-step inferential chains across languages. We evaluate CCKG on five countries with human judgments of cultural relevance, correctness, and path coherence. We find that the cultural knowledge graphs are better realized in English, even when the target culture is non-English (e.g., Chinese, Indonesian, Arabic), indicating uneven cultural encoding in current LLMs. Augmenting smaller LLMs with CCKG improves performance on cultural reasoning and story generation, with the largest gains from English chains. Our results show both the promise and limits of LLMs as cultural technologies and that chain-structured cultural knowledge is a practical substrate for culturally grounded NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v3">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17942v1">LLM-Based SQL Generation: Prompting, Self-Refinement, and Adaptive Weighted Majority Voting</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL has emerged as a prominent research area, particularly with the rapid advancement of large language models (LLMs). By enabling users to query databases through natural language rather than SQL, this technology significantly lowers the barrier to data analysis. However, generating accurate SQL from natural language remains challenging due to ambiguity in user queries, the complexity of schema linking, limited generalization across SQL dialects, and the need for domain-specific understanding. In this study, we propose a Single-Agent Self-Refinement with Ensemble Voting (SSEV) pipeline built on PET-SQL that operates without ground-truth data, integrating self-refinement with Weighted Majority Voting (WMV) and its randomized variant (RWMA). Experimental results show that the SSEV achieves competitive performance across multiple benchmarks, attaining execution accuracies of 85.5% on Spider 1.0-Dev, 86.4% on Spider 1.0-Test, and 66.3% on BIRD-Dev. Building on insights from the SSEV pipeline, we further propose ReCAPAgent-SQL (Refinement-Critique-Act-Plan agent-based SQL framework) to address the growing complexity of enterprise databases and real-world Text-to-SQL tasks. The framework integrates multiple specialized agents for planning, external knowledge retrieval, critique, action generation, self-refinement, schema linking, and result validation, enabling iterative refinement of SQL predictions through agent collaboration. ReCAPAgent-SQL's WMA results achieve 31% execution accuracy on the first 100 queries of Spider 2.0-Lite, demonstrating significant improvements in handling real-world enterprise scenarios. Overall, our work facilitates the deployment of scalable Text-to-SQL systems in practical settings, supporting better data-driven decision-making at lower cost and with greater efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10769v3">Mind the Gap: Benchmarking LLM Uncertainty and Calibration with Specialty-Aware Clinical QA and Reasoning-Based Behavioural Features</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at EACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential when employing large language models (LLMs) in high-risk domains such as clinical question answering (QA). In this work, we evaluate uncertainty estimation methods for clinical QA focusing, for the first time, on eleven clinical specialties and six question types, and across ten open-source LLMs (general-purpose, biomedical, and reasoning models), alongside representative proprietary models. We analyze score-based UQ methods, present a case study introducing a novel lightweight method based on behavioral features derived from reasoning-oriented models, and examine conformal prediction as a complementary set-based approach. Our findings reveal that uncertainty reliability is not a monolithic property, but one that depends on clinical specialty and question type due to shifts in calibration and discrimination. Our results highlight the need to select or ensemble models based on their distinct, complementary strengths and clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17915v1">Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context. We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09635v2">LLM for Large-Scale Optimization Model Auto-Formulation: Bridging Flexibility and Standardization via Agentic Workflow</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Updated version of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027
    </div>
    <details class="paper-abstract">
      Large-scale optimization is a key backbone of modern business decision-making. However, building these models is often labor-intensive and time-consuming. We address this by proposing LEAN-LLM-OPT, a LightwEight AgeNtic workflow construction framework for LLM-assisted large-scale OPTimization auto-formulation. LEAN-LLM-OPT takes as input a problem description together with associated datasets and orchestrates a team of LLM agents to produce an optimization formulation. Specifically, upon receiving a query, two upstream LLM agents dynamically construct a workflow that specifies, step-by-step, how optimization models for similar problems can be formulated. A downstream LLM agent then follows this workflow to generate the final output. The agentic workflow leverages common modeling practices to standardize the modeling process into a sequence of structured sub-tasks, offloading mechanical data-handling operations to auxiliary tools. This reduces the LLM's burden in planning and data handling, allowing us to exploit its flexibility to address unstructured components. Extensive simulations show that LEAN-LLM-OPT, instantiated with GPT-4.1 and the open source gpt-oss-20B, achieves strong performance on large-scale optimization modeling tasks and is competitive with state-of-the-art approaches. In addition, in a Singapore Airlines choice-based revenue management use case, LEAN-LLM-OPT demonstrates practical value by achieving leading performance across a range of scenarios. Along the way, we introduce Large-Scale-OR and Air-NRM, the first comprehensive benchmarks for large-scale optimization auto-formulation. The code and data of this work is available at https://github.com/CoraLiang01/lean-llm-opt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17902v1">dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Automatic speech recognition (ASR) systems based on large language models (LLMs) achieve superior performance by leveraging pretrained LLMs as decoders, but their token-by-token generation mechanism leads to inference latency that grows linearly with sequence length. Meanwhile, discrete diffusion large language models (dLLMs) offer a promising alternative, enabling high-quality parallel sequence generation with pretrained decoders. However, directly applying native text-oriented dLLMs to ASR leads to a fundamental mismatch between open-ended text generation and the acoustically conditioned transcription paradigm required by ASR. As a result, it introduces unnecessary difficulty and computational redundancy, such as denoising from pure noise, inflexible generation lengths, and fixed denoising steps. We propose dLLM-ASR, an efficient dLLM-based ASR framework that formulates dLLM's decoding as a prior-guided and adaptive denoising process. It leverages an ASR prior to initialize the denoising process and provide an anchor for sequence length. Building upon this prior, length-adaptive pruning dynamically removes redundant tokens, while confidence-based denoising allows converged tokens to exit the denoising loop early, enabling token-level adaptive computation. Experiments demonstrate that dLLM-ASR achieves recognition accuracy comparable to autoregressive LLM-based ASR systems and delivers a 4.44$\times$ inference speedup, establishing a practical and efficient paradigm for ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19113v3">Argument-Based Consistency in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 7 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity - from their explanations that justify a stance - to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Argument-based Consistency (ArC), that measures the extent to which LLMs' free-form toxicity explanations reflect an ideal and logical argumentation process. Based on uncertainty quantification, we develop six metrics for ArC to comprehensively evaluate the (in)consistencies in LLMs' toxicity explanations. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code (https://github.com/uofthcdslab/ArC) and LLM-generated explanations (https://huggingface.co/collections/uofthcdslab/arc) for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17897v1">UniCog: Uncovering Cognitive Abilities of LLMs through Latent Mind Space Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A growing body of research suggests that the cognitive processes of large language models (LLMs) differ fundamentally from those of humans. However, existing interpretability methods remain limited in explaining how cognitive abilities are engaged during LLM reasoning. In this paper, we propose UniCog, a unified framework that analyzes LLM cognition via a latent mind space. Formulated as a latent variable model, UniCog encodes diverse abilities from dense model activations into sparse, disentangled latent dimensions. Through extensive analysis on six advanced LLMs, including DeepSeek-V3.2 and GPT-4o, we reveal a Pareto principle of LLM cognition, where a shared reasoning core is complemented by ability-specific signatures. Furthermore, we discover that reasoning failures often manifest as anomalous intensity in latent activations. These findings opens a new paradigm in LLM analysis, providing a cognition grounded view of reasoning dynamics. Finally, leveraging these insights, we introduce a latent-informed candidate prioritization strategy, which improves reasoning performance by up to 7.5% across challenging benchmarks. Our code is available at https://github.com/milksalute/unicog.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v3">TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures reasoning models not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly released TruthTensor at https://truthtensor.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10978v3">Does LLM Focus on the Right Words? Mitigating Context Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Context Bias, whereby the model over-relies on auxiliary tokens, such as task descriptions and prefix-generated tokens, while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating context bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates context bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. The code is available at https://github.com/WANGBohaO-jpg/GDRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14506v4">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real robot platforms, including a Toyota Human Support Robot and an ANYmal D robot with a Unitree Z1 arm. Our method achieves a 95% success rate in the `fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art LLM-based robotics planners, while using only real-time onboard computing. Project website: https://kimtienly.github.io/InteLiPlan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17817v1">Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT: An LLM-Enhanced Agentic AI Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      The rapid expansion of low-altitude economy Internet of Things (LAE-IoT) networks has created unprecedented security challenges due to dynamic three-dimensional mobility patterns, distributed autonomous operations, and severe resource constraints. Traditional intrusion detection systems designed for static ground-based networks prove inadequate for tackling the unique characteristics of aerial IoT environments, including frequent topology changes, real-time detection requirements, and energy limitations. In this article, we analyze the intrusion detection requirements for LAE-IoT networks, complemented by a comprehensive review of evaluation metrics that cover detection effectiveness, response time, and resource consumption. Then, we investigate transformative potential of agentic artificial intelligence (AI) paradigms and introduce a large language model (LLM)-enabled agentic AI framework for enhancing intrusion detection in LAE-IoT networks. This leads to our proposal of a novel multi-agent collaborative intrusion detection framework that leverages specialized LLM-enhanced agents for intelligent data processing and adaptive classification. Through experimental validation, our framework demonstrates superior performance of over 90\% classification accuracy across multiple benchmark datasets. These results highlight the transformative potential of combining agentic AI principles with LLMs for next-generation LAE-IoT security systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17814v1">MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have advanced rapidly, yet heterogeneity in architecture, alignment strategies, and efficiency means that no single model is uniformly superior across tasks. In practical deployments, workloads span lightweight OCR to complex multimodal reasoning; using one MLLM for all queries either over-provisions compute on easy instances or sacrifices accuracy on hard ones. Query-level model selection (routing) addresses this tension, but extending routing from text-only LLMs to MLLMs is nontrivial due to modality fusion, wide variation in computational cost across models, and the absence of a standardized, budget-aware evaluation. We present MMR-Bench, a unified benchmark that isolates the multimodal routing problem and enables comparison under fixed candidate sets and cost models. MMR-Bench provides (i) a controlled environment with modality-aware inputs and variable compute budgets, (ii) a broad suite of vision-language tasks covering OCR, general VQA, and multimodal math reasoning, and (iii) strong single-model reference, oracle upper bounds, and representative routing policies. Using MMR-Bench, we show that incorporating multimodal signals improves routing quality. Empirically, these cues improve the cost-accuracy frontier and enable the routed system to exceed the strongest single model's accuracy at roughly 33% of its cost. Furthermore, policies trained on a subset of models and tasks generalize zero-shot to new datasets and text-only benchmarks without retuning, establishing MMR-Bench as a foundation for studying adaptive multimodal model selection and efficient MLLM deployment. The code will be available at: https://github.com/Hunter-Wrynn/MMR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15333v2">Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess strong representation and reasoning capabilities, but their application to structure-based drug design (SBDD) is limited by insufficient understanding of protein structures and unpredictable molecular generation. To address these challenges, we propose Exploration-Augmented Latent Inference for LLMs (ELILLM), a framework that reinterprets the LLM generation process as an encoding, latent space exploration, and decoding workflow. ELILLM explicitly explores portions of the design problem beyond the model's current knowledge while using a decoding module to handle familiar regions, generating chemically valid and synthetically reasonable molecules. In our implementation, Bayesian optimization guides the systematic exploration of latent embeddings, and a position-aware surrogate model efficiently predicts binding affinity distributions to inform the search. Knowledge-guided decoding further reduces randomness and effectively imposes chemical validity constraints. We demonstrate ELILLM on the CrossDocked2020 benchmark, showing strong controlled exploration and high binding affinity scores compared with seven baseline methods. These results demonstrate that ELILLM can effectively enhance LLMs capabilities for SBDD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17789v1">Neuro-Symbolic Verification on Instruction Following of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A fundamental problem of applying Large Language Models (LLMs) to important applications is that LLMs do not always follow instructions, and violations are often hard to observe or check. In LLM-based agentic workflows, such violations can propagate and amplify along reasoning chains, causing task failures and system incidents. This paper presents NSVIF, a neuro-symbolic framework for verifying whether an LLM's output follows the instructions used to prompt the LLM. NSVIF is a universal, general-purpose verifier; it makes no assumption about the instruction or the LLM. NSVIF formulates instruction-following verification as a constraint-satisfaction problem by modeling user instructions as constraints. NSVIF models both logical and semantic constraints; constraint solving is done by a unified solver that orchestrates logical reasoning and semantic analysis. To evaluate NSVIF, we develop VIFBENCH, a new benchmark for instruction-following verifiers with fine-grained data labels. Experiments show that NSVIF significantly outperforms LLM-based approaches and provides interpretable feedback. We also show that feedback from NSVIF helps improve LLMs' instruction-following capability without post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15331v2">DesignerlyLoop: Forming Design Intent through Curated Reasoning for Human-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) show promise in design tasks, yet a fundamental misalignment persists: design thinking requires iterative intent formulation, while LLMs treat inputs as complete specifications. This challenges design intent formulation, where designers must progressively refine understanding through exploration. Existing tools either sacrifice exploratory flexibility for structural stability or leave reasoning implicit, failing to support human-LLM alignment. Through a formative study with eight designers, we introduce curated reasoning-enabling designers to explicitly inspect, reorganize, and selectively regenerate LLM reasoning structures. We present DesignerlyLoop, implementing this through a two-layer structure separating design intent from LLM reasoning. A study with 20 designers demonstrates that curated reasoning significantly improves design quality and creativity. Our work contributes a novel interaction paradigm for human-LLM alignment, transforming LLMs from content generators into structured reasoning partners in creative design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17769v1">Reflexa: Uncovering How LLM-Supported Reflection Scaffolding Reshapes Creativity in Creative Coding</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Creative coding requires continuous translation between evolving concepts and computational artifacts, making reflection essential yet difficult to sustain. Creators often struggle to manage ambiguous intentions, emergent outputs, and complex code, limiting depth of exploration. This work examines how large language models (LLMs) can scaffold reflection not as isolated prompts, but as a system-level mechanism shaping creative regulation. From formative studies with eight expert creators, we derived reflection challenges and design principles that informed Reflexa, an integrated scaffold combining dialogic guidance, visualized version navigation, and iterative suggestion pathways. A within-subject study with 18 participants provides an exploratory mechanism validation, showing that structured reflection patterns mediate the link between AI interaction and creative outcomes. These reflection trajectories enhanced perceived controllability, broadened exploration, and improved originality and aesthetic quality. Our findings advance HCI understanding of reflection from LLM-assisted creative practices, and provide design strategies for building LLM-based creative tools that support richer human-AI co-creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17768v1">LLM-42: Enabling Determinism in LLM Inference with Verified Speculation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 https://github.com/microsoft/llm-42
    </div>
    <details class="paper-abstract">
      In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. A straightforward way to eliminate non-determinism is to disable dynamic batching during inference, but doing so severely degrades throughput. Another approach is to make kernels batch-invariant; however, this tightly couples determinism to kernel design, requiring new implementations. This coupling also imposes fixed runtime overheads, regardless of how much of the workload actually requires determinism. Inspired by ideas from speculative decoding, we present LLM-42, a scheduling-based approach to enable determinism in LLM inference. Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching. Moreover, most GPU kernels use shape-consistent reductions. Leveraging these insights, LLM-42 decodes tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback loop. The verifier replays candidate tokens under a fixed-shape reduction schedule, commits those that are guaranteed to be consistent across runs, and rolls back those violating determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to the traffic that requires determinism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v1">The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17716v1">Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Presented at the NeusymBridge Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many tasks but still struggle with a critical ability for LLM-based agents: asking good questions for resolving ambiguity in user requests. While prior work has explored information-seeking behavior through word games, existing benchmarks lack comprehensive evaluation frameworks that provide both final and intermediate signals based on Information Gain (IG). Moreover, they rarely provide systematic comparisons between models that use chain-of-thought reasoning and those that do not. We propose a multi-turn dialogue framework that quantitatively measures how effectively LLMs gather information through yes/no questions in a hierarchical knowledge graph environment. Our framework employs a triad of interacting LLM agents that ask questions, answer them, and update the hypothesis space. We adopt IG as the main metric, grounded in Shannon entropy, to assess query effectiveness at each turn and cumulatively. We instantiate our framework in a geographical Guess My City game setting organized in a five-level taxonomy and evaluate multiple LLM variants under fully and partially observable conditions, with and without Chain-of-Thought reasoning. Our experiments demonstrate that, among the evaluated models, the ones with explicit reasoning capabilities achieve higher IG per turn and reach solutions in fewer steps, particularly in partially observable settings. Analysis of reasoning traces reveals that smaller models compensate for limited capacity through more aggressive exploration of candidate questions, while larger models exhibit higher assertiveness in selecting optimal queries, generating candidates with greater potential IG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01659v3">From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17705v1">Distance-to-Distance Ratio: A Similarity Measure for Sentences Based on Rate of Change in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      A measure of similarity between text embeddings can be considered adequate only if it adheres to the human perception of similarity between texts. In this paper, we introduce the distance-to-distance ratio (DDR), a novel measure of similarity between LLM sentence embeddings. Inspired by Lipschitz continuity, DDR measures the rate of change in similarity between the pre-context word embeddings and the similarity between post-context LLM embeddings, thus measuring the semantic influence of context. We evaluate the performance of DDR in experiments designed as a series of perturbations applied to sentences drawn from a sentence dataset. For each sentence, we generate variants by replacing one, two, or three words with either synonyms, which constitute semantically similar text, or randomly chosen words, which constitute semantically dissimilar text. We compare the performance of DDR with other prevailing similarity metrics and demonstrate that DDR consistently provides finer discrimination between semantically similar and dissimilar texts, even under minimal, controlled edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17692v1">LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 31pages, 4 figures
    </div>
    <details class="paper-abstract">
      Statute retrieval is essential for legal assistance and judicial decision support, yet real-world legal queries are often implicit, multi-issue, and expressed in colloquial or underspecified forms. These characteristics make it difficult for conventional retrieval-augmented generation pipelines to recover the statutory elements required for accurate retrieval. Dense retrievers focus primarily on the literal surface form of the query, whereas lightweight rerankers lack the legal-reasoning capacity needed to assess statutory applicability. We present LegalMALR, a retrieval framework that integrates a Multi-Agent Query Understanding System (MAS) with a zero-shot large-language-model-based reranking module (LLM Reranker). MAS generates diverse, legally grounded reformulations and conducts iterative dense retrieval to broaden candidate coverage. To stabilise the stochastic behaviour of LLM-generated rewrites, we optimise a unified MAS policy using Generalized Reinforcement Policy Optimization(GRPO). The accumulated candidate set is subsequently evaluated by the LLM Reranker, which performs natural-language legal reasoning to produce the final ranking. We further construct CSAID, a dataset of 118 difficult Chinese legal queries annotated with multiple statutory labels, and evaluate LegalMALR on both CSAID and the public STARD benchmark. Experiments show that LegalMALR substantially outperforms strong Retrieval-augmented generation(RAG) baselines in both in-distribution and out-of-distribution settings, demonstrating the effectiveness of combining multi-perspective query interpretation, reinforcement-based policy optimisation, and large-model reranking for statute retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17676v1">GazeSummary: Exploring Gaze as an Implicit Prompt for Personalization in Text-based LLM Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Smart glasses are accelerating progress toward more seamless and personalized LLM-based assistance by integrating multimodal inputs. Yet, these inputs rely on obtrusive explicit prompts. The advent of gaze tracking on smart devices offers a unique opportunity to extract implicit user intent for personalization. This paper investigates whether LLMs can interpret user gaze for text-based tasks. We evaluate different gaze representations for personalization and validate their effectiveness in realistic reading tasks. Results show that LLMs can leverage gaze to generate high-quality personalized summaries and support users in downstream tasks, highlighting the feasibility and value of gaze-driven personalization for future mobile and wearable LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00003v2">A Review of Incorporating Psychological Theories in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Psychological insights have long shaped pivotal NLP breakthroughs, from attention mechanisms to reinforcement learning and social modeling. As Large Language Models (LLMs) develop, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development. Our review integrates insights from six subfields of psychology, including cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. With stage-wise analysis, we highlight current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04537v2">Not All Steps are Informative: On the Linearity of LLMs' RLVR Training</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on mathematics and code benchmarks by extrapolating beyond the step range where RL training remains stable. Our code is available at https://github.com/Miaow-Lab/RLVR-Linearity
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17668v1">Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. The results are consistent across a wide range of tasks, including long-context understanding, code comprehension, and mathematical reasoning, demonstrating the generality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v3">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18827v1">Automated structural testing of LLM-based agents: methods, framework, and case studies</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 10 pages, 5 figures. Preprint of an accepted paper at IEEE BigData 2025 (main track). Source code for the introduced methods and framework available at https://github.com/awslabs/generative-ai-toolkit
    </div>
    <details class="paper-abstract">
      LLM-based agents are rapidly being adopted across diverse domains. Since they interact with users without supervision, they must be tested extensively. Current testing approaches focus on acceptance-level evaluation from the user's perspective. While intuitive, these tests require manual evaluation, are difficult to automate, do not facilitate root cause analysis, and incur expensive test environments. In this paper, we present methods to enable structural testing of LLM-based agents. Our approach utilizes traces (based on OpenTelemetry) to capture agent trajectories, employs mocking to enforce reproducible LLM behavior, and adds assertions to automate test verification. This enables testing agent components and interactions at a deeper technical level within automated workflows. We demonstrate how structural testing enables the adaptation of software engineering best practices to agents, including the test automation pyramid, regression testing, test-driven development, and multi-language testing. In representative case studies, we demonstrate automated execution and faster root-cause analysis. Collectively, these methods reduce testing costs and improve agent quality through higher coverage, reusability, and earlier defect detection. We provide an open source reference implementation on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05786v3">FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 9 pages, 3 figures, 3 tables and 2 algorithms
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14275v3">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17567v1">Real-Time Trend Prediction via Continually-Aligned LLM Query Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Trending news detection in low-traffic search environments faces a fundamental cold-start problem, where a lack of query volume prevents systems from identifying emerging or long-tail trends. Existing methods relying on keyword frequency or query spikes are inherently slow and ineffective in these sparse settings, lagging behind real-world shifts in attention. We introduce RTTP, a novel Real-Time Trending Prediction framework that generates search queries directly from news content instead of waiting for users to issue them. RTTP leverages a continual learning LLM (CL-LLM) that converts posts into search-style queries and scores them using engagement strength + creator authority, enabling early trend surfacing before search volume forms. To ensure adaptation without degrading reasoning, we propose Mix-Policy DPO, a new preference-based continual learning approach that combines on-policy stability with off-policy novelty to mitigate catastrophic forgetting during model upgrades. Deployed at production scale on Facebook and Meta AI products, RTTP delivers +91.4% improvement in tail-trend detection precision@500 and +19% query generation accuracy over industry baselines, while sustaining stable performance after multi-week online training. This work demonstrates that LLM-generated synthetic search signals, when aligned and continually updated, unlock timely trend understanding in low-traffic search environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17551v1">GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Paper under submisison
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities, but their broad deployment is limited by significant computational resource demands, particularly energy consumption during inference. Static, one-model-fits-all inference strategies are often inefficient, as they do not exploit the diverse range of available models or adapt to varying query requirements. This paper presents GreenServ, a dynamic, context-aware routing framework that optimizes the trade-off between inference accuracy and energy efficiency. GreenServ extracts lightweight contextual features from each query, including task type, semantic cluster, and text complexity, and routes queries to the most suitable model from a heterogeneous pool, based on observed accuracy and energy usage. We employ a multi-armed bandit approach to learn adaptive routing policies online. This approach operates under partial feedback, eliminates the need for extensive offline calibration, and streamlines the integration of new models into the inference pipeline. We evaluated GreenServ across five benchmark tasks and a pool of 16 contemporary open-access LLMs. Experimental results show that GreenServ consistently outperforms static (single-model) and random baselines. In particular, compared to random routing, GreenServ achieved a 22% increase in accuracy while reducing cumulative energy consumption by 31%. Finally, we evaluated GreenServ with RouterBench, achieving an average accuracy of 71.7% with a peak accuracy of 75.7%. All artifacts are open-source and available as an anonymous repository for review purposes here: https://anonymous.4open.science/r/llm-inference-router-EBEA/README.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15338v2">HeartLLM: Discretized ECG Tokenization for LLM-Based Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Electrocardiography (ECG) plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present HeartLLM, a novel framework that integrates time-series (TS) and language modeling by enabling large language models (LLMs) to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into quantized codes using a lead-wise encoder and quantization module. These quantized codes are then mapped to an extended ECG vocabulary to form ECG tokens, enabling the model to process both ECG and natural language inputs within a unified framework. To bridge the modality gap, we pretrain the model on an autoregressive ECG token forecasting task, allowing the LLM to capture temporal dynamics through its inherent language modeling capability. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, HeartLLM achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating discretized ECG tokens into LLMs for medical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17540v1">Ethical Risk Assessment of the Data Harnessing Process of LLM supported on Consensus of Well-known Multi-Ethical Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have revolutionized natural language processing, unlocking unprecedented capabilities in communication, automation, and knowledge generation. However, the ethical implications of LLM development, particularly in data harnessing, remain a critical challenge. Despite widespread discussion about the ethical compliance of LLMs -- especially concerning their data harnessing processes, there remains a notable absence of concrete frameworks to systematically guide or measure the ethical risks involved. In this paper we discuss a potential pathway for building an Ethical Risk Scoring (ERS) system to quantitatively assess the ethical integrity of the data harnessing process for AI systems. This system is based on a set of assessment questions grounded in core ethical principles, which are, in turn, supported by commanding ethical theories. By integrating measurable scoring mechanisms, this approach aims to foster responsible LLM development, balancing technological innovation with ethical accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13742v2">Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) judges exhibit strong reasoning capabilities but are limited to textual content. This leaves current automatic Speech-to-Speech (S2S) evaluation methods reliant on opaque and expensive Audio Language Models (ALMs). In this work, we propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. To demonstrate the strength of the framework, we first introduce a Human Chain-of-Thought (HCoT) annotation protocol to improve the diagnostic capability of existing judge benchmarks by separating evaluation into explicit dimensions: content (C), voice quality (VQ), and paralinguistics (P). Using this data, TRACE constructs a textual blueprint of inexpensive audio signals and prompts an LLM to render dimension-wise judgments, fusing them into an overall rating via a deterministic policy. TRACE achieves higher agreement with human raters than ALMs and transcript-only LLM judges while being significantly more cost-effective. We will release the HCoT annotations and the TRACE framework to enable scalable and human-aligned S2S evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17527v1">Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      As LLMs increasingly function as economic agents, the specific mechanisms LLMs use to update their belief with heterogeneous signals remain opaque. We design experiments and develop a Behavioral Kalman Filter framework to quantify how LLM-based agents update expectations, acting as households or firm CEOs, update expectations when presented with individual and aggregate signals. The results from experiments and model estimation reveal four consistent patterns: (1) agents' weighting of priors and signals deviates from unity; (2) both household and firm CEO agents place substantially larger weights on individual signals compared to aggregate signals; (3) we identify a significant and negative interaction between concurrent signals, implying that the presence of multiple information sources diminishes the marginal weight assigned to each individual signal; and (4) expectation formation patterns differ significantly between household and firm CEO agents. Finally, we demonstrate that LoRA fine-tuning mitigates, but does not fully eliminate, behavioral biases in LLM expectation formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21981v2">Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17471v1">PatchIsland: Orchestration of LLM Agents for Continuous Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Continuous fuzzing platforms such as OSS-Fuzz uncover large numbers of vulnerabilities, yet the subsequent repair process remains largely manual. Unfortunately, existing Automated Vulnerability Repair (AVR) techniques -- including recent LLM-based systems -- are not directly applicable to continuous fuzzing. This is because these systems are designed and evaluated on a static, single-run benchmark setting, making them ill-suited for the diverse, noisy, and failure-prone environments in continuous fuzzing. To address these issues, we introduce PatchIsland, a system for Continuous Vulnerability Repair (CVR) that tightly integrates with continuous fuzzing pipelines. PatchIsland employs an ensemble of diverse LLM agents. By leveraging multiple LLM agents, PatchIsland can cover a wider range of settings (e.g., different projects, bug types, and programming languages) and also improve operational robustness. In addition, PatchIsland utilizes a two-phase patch-based deduplication to mitigate duplicate crashes and patches, which can be problematic in continuous fuzzing. In our internal evaluation, PatchIsland repaired 84 of 92 vulnerabilities, demonstrating strong repair capability. In the official AIxCC competition, the system operated with no human intervention in a fully autonomous environment and successfully patched 31 out of 43 vulnerabilities, achieving a repair rate of 72.1\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18951v4">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 29 pages, 10 figures, NeurIPS 2025 Main
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v3">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11056v3">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 TheWebConf 2026 Industry
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under ``standard'' and ``reasoning-augmented'' inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17421v1">Oops, Wait: Token-Level Signals as a Lens into LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The emergence of discourse-like tokens such as "wait" and "therefore" in large language models (LLMs) has offered a unique window into their reasoning processes. However, systematic analyses of how such signals vary across training strategies and model scales remain lacking. In this paper, we analyze token-level signals through token probabilities across various models. We find that specific tokens strongly correlate with reasoning correctness, varying with training strategies while remaining stable across model scales. A closer look at the "wait" token in relation to answer probability demonstrates that models fine-tuned on small-scale datasets acquire reasoning ability through such signals but exploit them only partially. This work provides a systematic lens to observe and understand the dynamics of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17418v1">GraphPilot: GUI Task Automation with One-Step LLM Reasoning Powered by Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 This paper is accepted by the Journal of Intelligent Computing and Networking (JICN) for publication
    </div>
    <details class="paper-abstract">
      Mobile graphical user interface (GUI) agents are designed to automate everyday tasks on smartphones. Recent advances in large language models (LLMs) have significantly enhanced the capabilities of mobile GUI agents. However, most LLM-powered mobile GUI agents operate in stepwise query-act loops, which incur high latency due to repeated LLM queries. We present GraphPilot, a mobile GUI agent that leverages knowledge graphs of the target apps to complete user tasks in almost one LLM query. GraphPilot operates in two complementary phases to enable efficient and reliable LLM-powered GUI task automation. In the offline phase, it explores target apps, records and analyzes interaction history, and constructs an app-specific knowledge graph that encodes functions of pages and elements as well as transition rules for each app. In the online phase, given an app and a user task, it leverages the knowledge graph of the given app to guide the reasoning process of LLM. When the reasoning process encounters uncertainty, GraphPilot dynamically requests the HTML representation of the current interface to refine subsequent reasoning. Finally, a validator checks the generated sequence of actions against the transition rules in the knowledge graph, performing iterative corrections to ensure it is valid. The structured, informative information in the knowledge graph allows the LLM to plan the complete sequence of actions required to complete the user task. On the DroidTask benchmark, GraphPilot improves task completion rate over Mind2Web and AutoDroid, while substantially reducing latency and the number of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17399v1">ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved rapid progress in Chinese language understanding, yet accurately evaluating their capabilities remains challenged by benchmark saturation and prohibitive computational costs. While static leaderboards provide snapshot rankings, they often mask the structural trade-offs between capabilities. In this work, we present ReLE (Robust Efficient Live Evaluation), a scalable system designed to diagnose Capability Anisotropy, the non-uniformity of model performance across domains. Using ReLE, we evaluate 304 models (189 commercial, 115 open-source) across a Domain $\times$ Capability orthogonal matrix comprising 207,843 samples. We introduce two methodological contributions to address current evaluation pitfalls: (1) A Symbolic-Grounded Hybrid Scoring Mechanism that eliminates embedding-based false positives in reasoning tasks; (2) A Dynamic Variance-Aware Scheduler based on Neyman allocation with noise correction, which reduces compute costs by 70\% compared to full-pass evaluations while maintaining a ranking correlation of $ρ=0.96$. Our analysis reveals that aggregate rankings are highly sensitive to weighting schemes: models exhibit a Rank Stability Amplitude (RSA) of 11.4 in ReLE versus $\sim$5.0 in traditional benchmarks, confirming that modern models are highly specialized rather than generally superior. We position ReLE not as a replacement for comprehensive static benchmarks, but as a high-frequency diagnostic monitor for the evolving model landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17397v1">CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL MME workshop paper
    </div>
    <details class="paper-abstract">
      Knowledge Editing (KE) has emerged as a promising paradigm for updating facts in Large Language Models (LLMs) without retraining. However, progress in Multilingual Knowledge Editing (MKE) is currently hindered by biased evaluation frameworks. We observe that existing MKE benchmarks are typically constructed by mechanically translating English-centric datasets into target languages (e.g., English-to-Chinese). This approach introduces translation artifacts and neglects culturally specific entities native to the target language, failing to reflect the true knowledge distribution of LLMs. To address this, we propose CLM-Bench, a culture-aware benchmark constructed using a native Chinese-first methodology. We curate 1,010 high-quality CounterFact pairs rooted in Chinese cultural contexts and align them with English counterparts. Using CLM-Bench, we conduct extensive experiments on representative LLMs (e.g., Llama-3, Qwen2) and reveal a significant Cross-lingual Misalignment: edits in one language function independently and fail to propagate to the other. We further provide a geometric explanation via layer-wise representation analysis, demonstrating that edit vectors for Chinese and English are nearly orthogonal -- residing in disjoint subspaces -- while mixed-lingual editing exhibits linear additivity of these vectors. Our findings challenge the effectiveness of current methods in cross-lingual transfer and underscore the importance of culturally native benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22767v2">TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically deployed using a fixed architecture, despite growing evidence that not all layers contribute equally to every downstream task. In this work, we introduce TALE (Task-Aware Layer Elimination), an inference-time method that improves task performance by selectively removing layers that are irrelevant or detrimental for a given task. TALE optimizes task-specific validation performance, yielding a task-adapted architecture without retraining or modifying model weights. Across 9 tasks and 5 model families, under both zero-shot and few-shot settings, we show that TALE consistently matches or surpasses baseline performance while simultaneously reducing computational cost, outperforming general and layer-wise pruning approaches such as SLEB. Beyond inference-time gains, TALE synergizes with fine-tuning and few-shot learning, where task-adapted architectures lead to additional performance improvements. Computing TALE for a new task requires modest resources (1-2 GPU hours on an A100), making it a practical and deployable solution for task-specialized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15830v2">DAIQ: Auditing Demographic Attribute Inference from Question in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent evaluations of Large language models (LLMs) audit social bias primarily through prompts that explicitly reference demographic attributes, overlooking whether models infer sensitive demographics from neutral questions. Such inference constitutes epistemic overreach and raises concerns for privacy. We introduce Demographic Attribute Inference from Questions (DAIQ), a diagnostic audit framework for evaluating demographic inference under epistemic uncertainty. We evaluate 18 open- and closed-source LLMs across six real-world domains and five demographic attributes. We find that many models infer demographics from neutral questions, defaulting to socially dominant categories and producing stereotype-aligned rationales. These behaviors persist across model families, scales and decoding settings, indicating reliance on learned population priors. We further show that inferred demographics can condition downstream responses and that abstention oriented prompting substantially reduces unintended inference without model fine-tuning. Our results suggest that current bias evaluations are incomplete and motivate evaluation standards that assess not only how models respond to demographic information, but whether they should infer it at all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23277v2">Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) often suffers from long and noisy retrieved contexts. Prior context compression methods rely on predefined importance metrics or supervised compression models, rather than on the model's own inference-time behavior. We propose Sentinel, a lightweight sentence-level compression framework that treats context compression as an understanding decoding problem. Sentinel probes native attention behaviors of a frozen LLM with a lightweight readout to decode which parts of the context are actually utilized when answering a query, rather than using attention as a direct relevance score. We empirically observe that decoded relevance signals exhibit sufficient consistency across model scales to support effective compression with compact proxy models. On LongBench, Sentinel with a 0.5B proxy model achieves up to 5x compression while matching the QA performance of 7B-scale baselines, and despite being trained only on English QA data, generalizes effectively to Chinese and out-of-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16559v4">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17346v1">Multi-Agent Learning Path Planning via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into intelligent tutoring systems offers transformative potential for personalized learning in higher education. However, most existing learning path planning approaches lack transparency, adaptability, and learner-centered explainability. To address these challenges, this study proposes a novel Multi-Agent Learning Path Planning (MALPP) framework that leverages a role- and rule-based collaboration mechanism among intelligent agents, each powered by LLMs. The framework includes three task-specific agents: a learner analytics agent, a path planning agent, and a reflection agent. These agents collaborate via structured prompts and predefined rules to analyze learning profiles, generate tailored learning paths, and iteratively refine them with interpretable feedback. Grounded in Cognitive Load Theory and Zone of Proximal Development, the system ensures that recommended paths are cognitively aligned and pedagogically meaningful. Experiments conducted on the MOOCCubeX dataset using seven LLMs show that MALPP significantly outperforms baseline models in path quality, knowledge sequence consistency, and cognitive load alignment. Ablation studies further validate the effectiveness of the collaborative mechanism and theoretical constraints. This research contributes to the development of trustworthy, explainable AI in education and demonstrates a scalable approach to learner-centered adaptive instruction powered by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17343v1">Are We Evaluating the Edit Locality of LLM Model Editing Properly?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Model editing has recently emerged as a popular paradigm for efficiently updating knowledge in LLMs. A central desideratum of updating knowledge is to balance editing efficacy, i.e., the successful injection of target knowledge, and specificity (also known as edit locality), i.e., the preservation of existing non-target knowledge. However, we find that existing specificity evaluation protocols are inadequate for this purpose. We systematically elaborated on the three fundamental issues it faces. Beyond the conceptual issues, we further empirically demonstrate that existing specificity metrics are weakly correlated with the strength of specificity regularizers. We also find that current metrics lack sufficient sensitivity, rendering them ineffective at distinguishing the specificity performance of different methods. Finally, we propose a constructive evaluation protocol. Under this protocol, the conflict between open-ended LLMs and the assumption of determined answers is eliminated, query-independent fluency biases are avoided, and the evaluation strictness can be smoothly adjusted within a near-continuous space. Experiments across various LLMs, datasets, and editing methods show that metrics derived from the proposed protocol are more sensitive to changes in the strength of specificity regularizers and exhibit strong correlation with them, enabling more fine-grained discrimination of different methods' knowledge preservation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v5">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Locating files and functions requiring modification in large software repositories is challenging due to their scale and structural complexity. Existing LLM-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which often overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool: jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a base pretrained model, without relying on closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and the 32B model exceeding closed-source models such as GPT-5 on most metrics. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05758v2">EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17570v2">GreedySnake: Accelerating SSD-Offloaded LLM Training with Efficient Scheduling and Optimizer Step Overlapping</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      SSD-offloaded training offers a practical and promising approach to making LLM training cost-effective. Building on gradient accumulation with micro-batches, this paper introduces GreedySnake, a new SSD-offloaded training system that employs vertical scheduling, which executes all microbatches of a layer before proceeding to the next. Compared to existing systems that use horizontal scheduling (i.e., executing micro-batches sequentially), GreedySnake achieves higher training throughput with smaller batch sizes, bringing the system much closer to the ideal scenario predicted by the roofline model. To further mitigate the I/O bottleneck, GreedySnake overlaps part of the optimization step with the forward pass of the next iteration. Experimental results on A100 GPUs show that GreedySnake achieves saturated training throughput improvements over ZeRO-Infinity: 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20697v2">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate more advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response entropy. By constraining entropy, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task performance and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14005v3">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious instructions, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, an effective and efficient detection method based on the observation that instruction-tuned LLMs internally encode distinguishable signals for prompts containing injected instructions. PIShield leverages residual-stream representations and a simple linear classifier to detect prompt injection, without expensive model fine-tuning or response generation. We conduct extensive evaluations on a diverse set of short- and long-context benchmarks. The results show that PIShield consistently achieves low false positive and false negative rates, significantly outperforming existing baselines. These findings demonstrate that internal representations of instruction-tuned LLMs provide a powerful and practical foundation for prompt injection detection in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17292v1">Risk-based test framework for LLM features in regulated software</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large language models are increasingly embedded in regulated and safety-critical software, including clinical research platforms and healthcare information systems. While these features enable natural language search, summarization, and configuration assistance, they introduce risks such as hallucinations, harmful or out-of-scope advice, privacy and security issues, bias, instability under change, and adversarial misuse. Prior work on machine learning testing and AI assurance offers useful concepts but limited guidance for interactive, product-embedded assistants. This paper proposes a risk-based testing framework for LLM features in regulated software: a six-category risk taxonomy, a layered test strategy mapping risks to concrete tests across guardrail, orchestration, and system layers, and a case study applying the approach to a Knowledgebase assistant in a clinical research platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17284v1">Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models in Medical Question Answering is severely hampered by ambiguous user queries, a significant safety risk that demonstrably reduces answer accuracy in high-stakes healthcare settings. In this paper, we formalize this challenge by linking input ambiguity to aleatoric uncertainty (AU), which is the irreducible uncertainty arising from underspecified input. To facilitate research in this direction, we construct CV-MedBench, the first benchmark designed for studying input ambiguity in Medical QA. Using this benchmark, we analyze AU from a representation engineering perspective, revealing that AU is linearly encoded in LLM's internal activation patterns. Leveraging this insight, we introduce a novel AU-guided "Clarify-Before-Answer" framework, which incorporates AU-Probe - a lightweight module that detects input ambiguity directly from hidden states. Unlike existing uncertainty estimation methods, AU-Probe requires neither LLM fine-tuning nor multiple forward passes, enabling an efficient mechanism to proactively request user clarification and significantly enhance safety. Extensive experiments across four open LLMs demonstrate the effectiveness of our QA framework, with an average accuracy improvement of 9.48% over baselines. Our framework provides an efficient and robust solution for safe Medical QA, strengthening the reliability of health-related applications. The code is available at https://github.com/yaokunliu/AU-Med.git, and the CV-MedBench dataset is released on Hugging Face at https://huggingface.co/datasets/yaokunl/CV-MedBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17275v1">Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 12 pages,
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate exceptional performance in surface-level text generation, their nature in handling complex multi-step reasoning tasks often remains one of ``statistical fitting'' rather than systematic logical deduction. Traditional Reinforcement Learning (RL) attempts to mitigate this by introducing a ``think-before-speak'' paradigm. However, applying RL directly in high-dimensional, discrete token spaces faces three inherent challenges: sample-inefficient rollouts, high gradient estimation variance, and the risk of catastrophic forgetting. To fundamentally address these structural bottlenecks, we propose \textbf{DeepLatent Reasoning (DLR)}, a latent-space bidirectional contrastive reinforcement learning framework. This framework shifts the trial-and-error cost from expensive token-level full sequence generation to the continuous latent manifold. Specifically, we introduce a lightweight assistant model to efficiently sample $K$ reasoning chain encodings within the latent space. These encodings are filtered via a dual reward mechanism based on correctness and formatting; only high-value latent trajectories are fed into a \textbf{frozen main model} for single-pass decoding. To maximize reasoning diversity while maintaining coherence, we design a contrastive learning objective to enable directed exploration within the latent space. Since the main model parameters remain frozen during optimization, this method mathematically eliminates catastrophic forgetting. Experiments demonstrate that under comparable GPU computational budgets, DLR achieves more stable training convergence, supports longer-horizon reasoning chains, and facilitates the sustainable accumulation of reasoning capabilities, providing a viable path toward reliable and scalable reinforcement learning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17261v1">AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 21 pages in total, including 9 pages of main text, with 4 figures and 3 tables. This manuscript is submitted to arXiv
    </div>
    <details class="paper-abstract">
      Zeroth-Order (ZO) optimization has emerged as a promising solution for fine-tuning LLMs under strict memory constraints, as it avoids the prohibitive memory cost of storing activations for backpropagation. However, existing ZO methods typically employ isotropic perturbations, neglecting the rich structural information available during the forward pass. In this paper, we identify a crucial link between gradient formation and activation structure: the gradient of a linear layer is confined to the subspace spanned by its input activations. Leveraging this insight, we propose Activation-Guided Zeroth-Order optimization (AGZO). Unlike prior methods, AGZO extracts a compact, activation-informed subspace on the fly during the forward pass and restricts perturbations to this low-rank subspace. We provide a theoretical framework showing that AGZO optimizes a subspace-smoothed objective and provably yields update directions with higher cosine similarity to the true gradient than isotropic baselines. Empirically, we evaluate AGZO on Qwen3 and Pangu models across various benchmarks. AGZO consistently outperforms state-of-the-art ZO baselines and significantly narrows the performance gap with first-order fine-tuning, while maintaining almost the same peak memory footprint as other ZO methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02547v4">The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Published on Transactions on Machine Learning Research: https://openreview.net/forum?id=RY19y2RI1O
    </div>
    <details class="paper-abstract">
      The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17614v1">AlignUI: A Method for Designing LLM-Generated UIs Aligned with User Preferences</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Designing user interfaces that align with user preferences is a time-consuming process, which requires iterative cycles of prototyping, user testing, and refinement. Recent advancements in LLM-based UI generation have enabled efficient UI generation to assist the UI design process. We introduce AlignUI, a method that aligns LLM-generated UIs with user tasks and preferences by using a user preference dataset to guide the LLM's reasoning process. The dataset was crowdsourced from 50 general users (the target users of generated UIs) and contained 720 UI control preferences on eight image-editing tasks. We evaluated AlignUI by generating UIs for six unseen tasks and conducting a user study with 72 additional general users. The results showed that the generated UIs closely align with multiple dimensions of user preferences. We conclude by discussing the applicability of our method to support user-aligned UI design for multiple task domains and user groups, as well as personalized user needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17604v1">Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used to support software developers in tasks such as code generation, optimization, and documentation. However, their ability to improve existing programming answers in a human-like manner remains underexplored. On technical question-and-answer platforms such as Stack Overflow (SO), contributors often revise answers based on user comments that identify errors, inefficiencies, or missing explanations. Yet roughly one-third of this feedback is never addressed due to limited time, expertise, or visibility, leaving many answers incomplete or outdated. This study investigates whether LLMs can enhance programming answers by interpreting and incorporating comment-based feedback. We make four main contributions. First, we introduce ReSOlve, a benchmark consisting of 790 SO answers with associated comment threads, annotated for improvement-related and general feedback. Second, we evaluate four state-of-the-art LLMs on their ability to identify actionable concerns, finding that DeepSeek achieves the best balance between precision and recall. Third, we present AUTOCOMBAT, an LLM-powered tool that improves programming answers by jointly leveraging user comments and question context. Compared to human revised references, AUTOCOMBAT produces near-human quality improvements while preserving the original intent and significantly outperforming the baseline. Finally, a user study with 58 practitioners shows strong practical value, with 84.5 percent indicating they would adopt or recommend the tool. Overall, AUTOCOMBAT demonstrates the potential of scalable, feedback-driven answer refinement to improve the reliability and trustworthiness of technical knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21184v4">Jailbreak-as-a-Service++: Unveiling Distributed AI-Driven Malicious Information Campaigns Powered by LLM Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      To prevent the misuse of Large Language Models (LLMs) for malicious purposes, numerous efforts have been made to develop the safety alignment mechanisms of LLMs. However, as multiple LLMs become readily accessible through various Model-as-a-Service (MaaS) platforms, attackers can strategically exploit LLMs' heterogeneous safety policies to fulfill malicious information generation tasks in a distributed manner. In this study, we introduce \textit{\textbf{PoisonSwarm}} to how attackers can reliably launder malicious tasks via the speculative use of LLM crowdsourcing. Building upon a scheduler orchestrating crowdsourced LLMs, PoisonSwarm maps the given malicious task to a benign analogue to derive a content template, decomposes it into semantic units for crowdsourced unit-wise rewriting, and reassembles the outputs into malicious content. Experiments show its superiority over existing methods in data quality, diversity, and success rates. Regulation simulations further reveal the difficulty of governing such distributed, orchestrated misuse in MaaS ecosystems, highlighting the need for coordinated, ecosystem-level defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15955v2">How Good Are LLMs at Processing Tool Outputs?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Most realistic task automation problems require large language models (LLMs) to call tools, which often return complex JSON responses. These responses must be further processed to derive the information necessary for task completion. The ability of LLMs to do so is under-studied. In this paper, we study the tool response processing task and LLMs' abilities to process structured (JSON) responses. We created a dataset for this task, and evaluated 15 open and closed weight models using multiple prompting approaches. Our results show that JSON processing remains a difficult task even for frontier models across multiple prompting strategies. The optimal response processing strategy depends on both the nature and size of the tool outputs, as well as the complexity of the required reasoning. Variations in processing approaches can lead to performance differences ranging from 3\% to 50\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17593v1">From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Recent progress in large language models has renewed interest in mechanistically characterizing how multi-step reasoning is represented and computed. While much prior work treats reasoning as a linear chain of steps, many reasoning problems are more naturally structured as directed acyclic graphs (DAGs), where intermediate conclusions may depend on multiple premises, branch into parallel sub-derivations, and later merge or be reused. Understanding whether such graph-structured reasoning is reflected in model internals remains an open question. In this work, we introduce Reasoning DAG Probing, a framework that directly asks whether LLM hidden states encode the geometry of a reasoning DAG in a linearly accessible form, and where this structure emerges across layers. Within this framework, we associate each reasoning node with a textual realization and train lightweight probes to predict two graph-theoretic properties from hidden states: node depth and pairwise node distance. We use these probes to analyze the layerwise emergence of DAG structure and evaluate controls that disrupt reasoning-relevant structure while preserving superficial textual properties. Our results provide evidence that reasoning DAG geometry is meaningfully encoded in intermediate layers, with recoverability varying systematically by node depth and model scale, suggesting that LLM reasoning is not only sequential but exhibits measurable internal graph structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16979v1">A Scalable Measure of Loss Landscape Curvature for Analyzing the Training Dynamics of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the curvature evolution of the loss landscape is fundamental to analyzing the training dynamics of neural networks. The most commonly studied measure, Hessian sharpness ($λ_{\max}^H$) -- the largest eigenvalue of the loss Hessian -- determines local training stability and interacts with the learning rate throughout training. Despite its significance in analyzing training dynamics, direct measurement of Hessian sharpness remains prohibitive for Large Language Models (LLMs) due to high computational cost. We analyze $\textit{critical sharpness}$ ($λ_c$), a computationally efficient measure requiring fewer than $10$ forward passes given the update direction $Δ\mathbfθ$. Critically, this measure captures well-documented Hessian sharpness phenomena, including progressive sharpening and Edge of Stability. Using this measure, we provide the first demonstration of these sharpness phenomena at scale, up to $7$B parameters, spanning both pre-training and mid-training of OLMo-2 models. We further introduce $\textit{relative critical sharpness}$ ($λ_c^{1\to 2}$), which quantifies the curvature of one loss landscape while optimizing another, to analyze the transition from pre-training to fine-tuning and guide data mixing strategies. Critical sharpness provides practitioners with a practical tool for diagnosing curvature dynamics and informing data composition choices at scale. More broadly, our work shows that scalable curvature measures can provide actionable insights for large-scale training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18261v3">LLM Reasoning for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Published on Proceedings of the ACM on Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16964v1">AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has sparked growing interest in their integration into autonomous systems for reasoning-driven perception, planning, and decision-making. However, evaluating and training such agentic AI models remains challenging due to the lack of large-scale, structured, and safety-critical benchmarks. This paper introduces AgentDrive, an open benchmark dataset containing 300,000 LLM-generated driving scenarios designed for training, fine-tuning, and evaluating autonomous agents under diverse conditions. AgentDrive formalizes a factorized scenario space across seven orthogonal axes: scenario type, driver behavior, environment, road layout, objective, difficulty, and traffic density. An LLM-driven prompt-to-JSON pipeline generates semantically rich, simulation-ready specifications that are validated against physical and schema constraints. Each scenario undergoes simulation rollouts, surrogate safety metric computation, and rule-based outcome labeling. To complement simulation-based evaluation, we introduce AgentDrive-MCQ, a 100,000-question multiple-choice benchmark spanning five reasoning dimensions: physics, policy, hybrid, scenario, and comparative reasoning. We conduct a large-scale evaluation of fifty leading LLMs on AgentDrive-MCQ. Results show that while proprietary frontier models perform best in contextual and policy reasoning, advanced open models are rapidly closing the gap in structured and physics-grounded reasoning. We release the AgentDrive dataset, AgentDrive-MCQ benchmark, evaluation code, and related materials at https://github.com/maferrag/AgentDrive
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16956v1">DataStates-LLM: Scalable Checkpointing for Transformer Models Using Composable State Providers</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      The rapid growth of Large Transformer-based models, specifically Large Language Models (LLMs), now scaling to trillions of parameters, has necessitated training across thousands of GPUs using complex hybrid parallelism strategies (e.g., data, tensor, and pipeline parallelism). Checkpointing this massive, distributed state is critical for a wide range of use cases, such as resilience, suspend-resume, investigating undesirable training trajectories, and explaining model evolution. However, existing checkpointing solutions typically treat model state as opaque binary blobs, ignoring the ``3D heterogeneity'' of the underlying data structures--varying by memory location (GPU vs. Host), number of ``logical'' objects sharded and split across multiple files, data types (tensors vs. Python objects), and their serialization requirements. This results in significant runtime overheads due to blocking device-to-host transfers, data-oblivious serialization, and storage I/O contention. In this paper, we introduce DataStates-LLM, a novel checkpointing architecture that leverages State Providers to decouple state abstraction from data movement. DataStates-LLM exploits the immutability of model parameters during the forward and backward passes to perform ``lazy'', non-blocking asynchronous snapshots. By introducing State Providers, we efficiently coalesce fragmented, heterogeneous shards and overlap the serialization of metadata with bulk tensor I/O. We evaluate DataStates-LLM on models up to 70B parameters on 256 A100-40GB GPUs. Our results demonstrate that DataStates-LLM achieves up to 4$\times$ higher checkpointing throughput and reduces end-to-end training time by up to 2.2$\times$ compared to state-of-the-art solutions, effectively mitigating the serialization and heterogeneity bottlenecks in extreme-scale LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.17406v3">ProveRAG: Provenance-Driven Vulnerability Analysis with Automated Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      In cybersecurity, security analysts constantly face the challenge of mitigating newly discovered vulnerabilities in real-time, with over 300,000 vulnerabilities identified since 1999. The sheer volume of known vulnerabilities complicates the detection of patterns for unknown threats. While LLMs can assist, they often hallucinate and lack alignment with recent threats. Over 40,000 vulnerabilities have been identified in 2024 alone, which are introduced after most popular LLMs' (e.g., GPT-5) training data cutoff. This raises a major challenge of leveraging LLMs in cybersecurity, where accuracy and up-to-date information are paramount. Therefore, we aim to improve the adaptation of LLMs in vulnerability analysis by mimicking how an analyst performs such tasks. We propose ProveRAG, an LLM-powered system designed to assist in rapidly analyzing vulnerabilities with automated retrieval augmentation of web data while self-evaluating its responses with verifiable evidence. ProveRAG incorporates a self-critique mechanism to help alleviate the omission and hallucination common in the output of LLMs applied in cybersecurity applications. The system cross-references data from verifiable sources (NVD and CWE), giving analysts confidence in the actionable insights provided. Our results indicate that ProveRAG excels in delivering verifiable evidence to the user with over 99% and 97% accuracy in exploitation and mitigation strategies, respectively. ProveRAG guides analysts to secure their systems more effectively by overcoming temporal and context-window limitations while also documenting the process for future audits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16890v1">LLM-Based Adversarial Persuasion Attacks on Fact-Checking Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Automated fact-checking (AFC) systems are susceptible to adversarial attacks, enabling false claims to evade detection. Existing adversarial frameworks typically rely on injecting noise or altering semantics, yet no existing framework exploits the adversarial potential of persuasion techniques, which are widely used in disinformation campaigns to manipulate audiences. In this paper, we introduce a novel class of persuasive adversarial attacks on AFCs by employing a generative LLM to rephrase claims using persuasion techniques. Considering 15 techniques grouped into 6 categories, we study the effects of persuasion on both claim verification and evidence retrieval using a decoupled evaluation strategy. Experiments on the FEVER and FEVEROUS benchmarks show that persuasion attacks can substantially degrade both verification performance and evidence retrieval. Our analysis identifies persuasion techniques as a potent class of adversarial attacks, highlighting the need for more robust AFC systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00570v2">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enhanced conventional recommendation models via user profiling, which generates representative textual profiles from users' historical interactions. However, their direct application to session-based recommendation (SBR) remains challenging due to severe session context scarcity and poor scalability. In this paper, we propose SPRINT, a scalable SBR framework that incorporates reliable and informative intents while ensuring high efficiency in both training and inference. SPRINT constrains LLM-based profiling with a global intent pool and validates inferred intents based on recommendation performance to mitigate noise and hallucinations under limited context. To ensure scalability, LLMs are selectively invoked only for uncertain sessions during training, while a lightweight intent predictor generalizes intent prediction to all sessions without LLM dependency at inference time. Experiments on real-world datasets show that SPRINT consistently outperforms state-of-the-art methods while providing more explainable recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09631v4">LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16781v1">Persuasion Tokens for Editing Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted at EACL Main 2026
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16775v1">LLM-powered Real-time Patent Citation Recommendation for Financial Technologies</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Rapid financial innovation has been accompanied by a sharp increase in patenting activity, making timely and comprehensive prior-art discovery more difficult. This problem is especially evident in financial technologies, where innovations develop quickly, patent collections grow continuously, and citation recommendation systems must be updated as new applications arrive. Existing patent retrieval and citation recommendation methods typically rely on static indexes or periodic retraining, which limits their ability to operate effectively in such dynamic settings. In this study, we propose a real-time patent citation recommendation framework designed for large and fast-changing financial patent corpora. Using a dataset of 428,843 financial patents granted by the China National Intellectual Property Administration (CNIPA) between 2000 and 2024, we build a three-stage recommendation pipeline. The pipeline uses large language model (LLM) embeddings to represent the semantic content of patent abstracts, applies efficient approximate nearest-neighbor search to construct a manageable candidate set, and ranks candidates by semantic similarity to produce top-k citation recommendations. In addition to improving recommendation accuracy, the proposed framework directly addresses the dynamic nature of patent systems. By using an incremental indexing strategy based on hierarchical navigable small-world (HNSW) graphs, newly issued patents can be added without rebuilding the entire index. A rolling day-by-day update experiment shows that incremental updating improves recall while substantially reducing computational cost compared with rebuild-based indexing. The proposed method also consistently outperforms traditional text-based baselines and alternative nearest-neighbor retrieval approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16766v1">Do LLM hallucination detectors suffer from low-resource effect?</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted at EACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      LLMs, while outperforming humans in a wide range of tasks, can still fail in unanticipated ways. We focus on two pervasive failure modes: (i) hallucinations, where models produce incorrect information about the world, and (ii) the low-resource effect, where the models show impressive performance in high-resource languages like English but the performance degrades significantly in low-resource languages like Bengali. We study the intersection of these issues and ask: do hallucination detectors suffer from the low-resource effect? We conduct experiments on five tasks across three domains (factual recall, STEM, and Humanities). Experiments with four LLMs and three hallucination detectors reveal a curious finding: As expected, the task accuracies in low-resource languages experience large drops (compared to English). However, the drop in detectors' accuracy is often several times smaller than the drop in task accuracy. Our findings suggest that even in low-resource languages, the internal mechanisms of LLMs might encode signals about their uncertainty. Further, the detectors are robust within language (even for non-English) and in multilingual setups, but not in cross-lingual settings without in-language supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04633v2">Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 10 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The unjudged document problem, where systems that did not contribute to the original judgement pool may retrieve documents without a relevance judgement, is a key obstacle to the reuseability of test collections in information retrieval. While the de facto standard to deal with the problem is to treat unjudged documents as non-relevant, many alternatives have been proposed, such as the use of large language models (LLMs) as a relevance judge (LLM-as-a-judge). However, this has been criticized, among other things, as circular, since the same LLM can be used as the ranker and the judge. We propose to train topic-specific relevance classifiers instead: By finetuning monoT5 with independent LoRA weight adaptation on the judgments of a single assessor for a single topic's pool, we align it to that assessor's notion of relevance for the topic. The system rankings obtained through our classifier's relevance judgments achieve a Spearmans' $ρ$ correlation of $>0.94$ with ground truth system rankings. As little as 128 initial human judgments per topic suffice to improve the comparability of models, compared to treating unjudged documents as non-relevant, while achieving more reliability than existing LLM-as-a-judge approaches. Topic-specific relevance classifiers are thus a lightweight and straightforward way to tackle the unjudged document problem, while maintaining human judgments as the gold standard for retrieval evaluation. Code, models, and data are made openly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06161v2">LATTLE: LLM Attention Transplant for Transfer Learning of Tabular Data Across Disparate Domains</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Transfer learning on tabular data is challenging due to disparate feature spaces across domains, in contrast to the homogeneous structures of image and text. Large language models (LLMs) offer a knowledge base to improve the limited effectiveness of cross-domain transfer learning for tabular data. However, LLM performance often stagnates due to subjective text prompts and the computational limitations of in-context learning. We present a novel language-to-tabular context-learning method that uses attention-specific transformer weights, enabling seamless transfer learning across disparate tabular data sets. The LLM attention transplant mechanism facilitates a domain-agnostic transfer learning, eliminating the need for shared features between tables, LLM prompt engineering, and large-scale pretrained models. Our experiments using ten pairs of disjoint source-target data sets and 12 baseline methods demonstrate the superiority of the proposed LLM-attention transplant for transfer learning (LATTLE) method over traditional ML models, state-of-the-art deep tabular architectures, and models trained on thousands to billions of tabular samples. The proposed cross-domain attention transfer demonstrates an effective solution for adapting LLMs to learning non-text tabular data in a low-resource environment. The source code of the LATTLE implementation is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16711v1">Better Generalizing to Unseen Concepts: An Evaluation Framework and An LLM-Based Auto-Labeled Pipeline for Biomedical Concept Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to EACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      Generalization to unseen concepts is a central challenge due to the scarcity of human annotations in Mention-agnostic Biomedical Concept Recognition (MA-BCR). This work makes two key contributions to systematically address this issue. First, we propose an evaluation framework built on hierarchical concept indices and novel metrics to measure generalization. Second, we explore LLM-based Auto-Labeled Data (ALD) as a scalable resource, creating a task-specific pipeline for its generation. Our research unequivocally shows that while LLM-generated ALD cannot fully substitute for manual annotations, it is a valuable resource for improving generalization, successfully providing models with the broader coverage and structural knowledge needed to approach recognizing unseen concepts. Code and datasets are available at https://github.com/bio-ie-tool/hi-ald.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16699v1">Supporting Stakeholder Requirements Expression with LLM Revisions: An Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 This paper has been accepted at the research track of the 32nd International Working Conference on Requirements Engineering: Foundation for Software Quality (REFSQ 2026)
    </div>
    <details class="paper-abstract">
      Stakeholders often struggle to accurately express their requirements due to articulation barriers arising from limited domain knowledge or from cognitive constraints. This can cause misalignment between expressed and intended requirements, complicating elicitation and validation. Traditional elicitation techniques, such as interviews and follow-up sessions, are time-consuming and risk distorting stakeholders' original intent across iterations. Large Language Models (LLMs) can infer user intentions from context, suggesting potential for assisting stakeholders in expressing their needs. This raises the questions of (i) how effectively LLMs can support requirement expression and (ii) whether such support benefits stakeholders with limited domain expertise. We conducted a study with 26 participants who produced 130 requirement statements. Each participant first expressed requirements unaided, then evaluated LLM-generated revisions tailored to their context. Participants rated LLM revisions significantly higher than their original statements across all dimensions-alignment with intent, readability, reasoning, and unambiguity. Qualitative feedback further showed that LLM revisions often surfaced tacit details stakeholders considered important and helped them better understand their own requirements. We present and evaluate a stakeholder-centered approach that leverages LLMs as articulation aids in requirements elicitation and validation. Our results show that LLM-assisted reformulation improves perceived completeness, clarity, and alignment of requirements. By keeping stakeholders in the validation loop, this approach promotes responsible and trustworthy use of AI in Requirements Engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14558v2">LLM Jailbreak Detection for (Almost) Free!</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EMNLP 2025 (Findings) https://aclanthology.org/2025.findings-emnlp.309/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16669v1">PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied to legal domain-specific tasks, evaluating their ability to perform legal work in real-world settings has become essential. However, existing legal benchmarks rely on simplified and highly standardized tasks, failing to capture the ambiguity, complexity, and reasoning demands of real legal practice. Moreover, prior evaluations often adopt coarse, single-dimensional metrics and do not explicitly assess fine-grained legal reasoning. To address these limitations, we introduce PLawBench, a Practical Law Benchmark designed to evaluate LLMs in realistic legal practice scenarios. Grounded in real-world legal workflows, PLawBench models the core processes of legal practitioners through three task categories: public legal consultation, practical case analysis, and legal document generation. These tasks assess a model's ability to identify legal issues and key facts, perform structured legal reasoning, and generate legally coherent documents. PLawBench comprises 850 questions across 13 practical legal scenarios, with each question accompanied by expert-designed evaluation rubrics, resulting in approximately 12,500 rubric items for fine-grained assessment. Using an LLM-based evaluator aligned with human expert judgments, we evaluate 10 state-of-the-art LLMs. Experimental results show that none achieves strong performance on PLawBench, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs. Data is available at: https://github.com/skylenage/PLawbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16651v1">Select or Project? Evaluating Lower-dimensional Vectors for LLM Training Data Explanations</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Gradient-based methods for instance-based explanation for large language models (LLMs) are hindered by the immense dimensionality of model gradients. In practice, influence estimation is restricted to a subset of model parameters to make computation tractable, but this subset is often chosen ad hoc and rarely justified by systematic evaluation. This paper investigates if it is better to create low-dimensional representations by selecting a small, architecturally informed subset of model components or by projecting the full gradients into a lower-dimensional space. Using a novel benchmark, we show that a greedily selected subset of components captures the information about training data influence needed for a retrieval task more effectively than either the full gradient or random projection. We further find that this approach is more computationally efficient than random projection, demonstrating that targeted component selection is a practical strategy for making instance-based explanations of large models more computationally feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03093v2">Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 To appear in Proc. ICASSP 2026, May 04-08, 2026, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16621v1">How Does Personalized Memory Shape LLM Behavior? Benchmarking Rational Preference Utilization in Personalized Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered assistants have recently integrated memory mechanisms that record user preferences, leading to more personalized and user-aligned responses. However, irrelevant personalized memories are often introduced into the context, interfering with the LLM's intent understanding. To comprehensively investigate the dual effects of personalization, we develop RPEval, a benchmark comprising a personalized intent reasoning dataset and a multi-granularity evaluation protocol. RPEval reveals the widespread phenomenon of irrational personalization in existing LLMs and, through error pattern analysis, illustrates its negative impact on user experience. Finally, we introduce RP-Reasoner, which treats memory utilization as a pragmatic reasoning process, enabling the selective integration of personalized information. Experimental results demonstrate that our method significantly outperforms carefully designed baselines on RPEval, and resolves 80% of the bad cases observed in a large-scale commercial personalized assistant, highlighting the potential of pragmatic reasoning to mitigate irrational personalization. Our benchmark is publicly available at https://github.com/XueyangFeng/RPEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16618v1">PROST-LLM: Progressively Enhancing the Speech-to-Speech Translation Capability in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) excel in many tasks, their application to Speech-to-Speech Translation (S2ST) is underexplored and hindered by data scarcity. To bridge this gap, we propose PROST-LLM (PROgressive Speech-to-speech Translation) to enhance the S2ST capabilities in LLMs progressively. First, we fine-tune the LLMs with the CVSS corpus, employing designed tri-task learning and chain of modality methods to boost the initial performance. Then, leveraging the fine-tuned model, we generate preference pairs through self-sampling and back-translation without human evaluation. Finally, these preference pairs are used for preference optimization to enhance the model's S2ST capability further. Extensive experiments confirm the effectiveness of our proposed PROST-LLM in improving the S2ST capability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14122v2">Benchmarking LLMs for Political Science: A United Nations Perspective</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 This paper has been accepted at AAAI 2026 as an oral paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: https://github.com/yueqingliang1/UNBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12146v2">From LLMs to Agents in Programming: The Impact of Providing an LLM with a Compiler</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated a remarkable capability in natural language and program generation and software development. However, the source code generated by the LLMs does not always meet quality requirements and may fail to compile. Therefore, many studies evolve into agents that can reason about the problem before generating the source code for the solution. The goal of this paper is to study the degree to which such agents benefit from access to software development tools, in our case, a gcc compiler. We conduct a computational experiment on the RosettaCode dataset, on 699 programming tasks in C. We evaluate how the integration with a compiler shifts the role of the language model from a passive generator to an active agent capable of iteratively developing runnable programs based on feedback from the compiler. We evaluated 16 language models with sizes ranging from small (135 million) to medium (3 billion) and large (70 billion). Our results show that access to a compiler improved the compilation success by 5.3 to 79.4 percentage units in compilation without affecting the semantics of the generated program. Syntax errors dropped by 75%, and errors related to undefined references dropped by 87% for the tasks where the agents outperformed the baselines. We also observed that in some cases, smaller models with a compiler outperform larger models with a compiler. We conclude that it is essential for LLMs to have access to software engineering tools to enhance their performance and reduce the need for large models in software engineering, such as reducing our energy footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16549v1">LLM is Not All You Need: A Systematic Evaluation of ML vs. Foundation Models for text and image based Medical Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 9 pages, 5 figures, 3 tables, paper accepted in AAIML'26 conference
    </div>
    <details class="paper-abstract">
      The combination of multimodal Vision-Language Models (VLMs) and Large Language Models (LLMs) opens up new possibilities for medical classification. This work offers a rigorous, unified benchmark by using four publicly available datasets covering text and image modalities (binary and multiclass complexity) that contrasts traditional Machine Learning (ML) with contemporary transformer-based techniques. We evaluated three model classes for each task: Classical ML (LR, LightGBM, ResNet-50), Prompt-Based LLMs/VLMs (Gemini 2.5), and Fine-Tuned PEFT Models (LoRA-adapted Gemma3 variants). All experiments used consistent data splits and aligned metrics. According to our results, traditional machine learning (ML) models set a high standard by consistently achieving the best overall performance across most medical categorization tasks. This was especially true for structured text-based datasets, where the classical models performed exceptionally well. In stark contrast, the LoRA-tuned Gemma variants consistently showed the worst performance across all text and image experiments, failing to generalize from the minimal fine-tuning provided. However, the zero-shot LLM/VLM pipelines (Gemini 2.5) had mixed results; they performed poorly on text-based tasks, but demonstrated competitive performance on the multiclass image task, matching the classical ResNet-50 baseline. These results demonstrate that in many medical categorization scenarios, established machine learning models continue to be the most reliable option. The experiment suggests that foundation models are not universally superior and that the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) is highly dependent on the adaptation strategy, as minimal fine-tuning proved detrimental in this study.
    </details>
</div>
