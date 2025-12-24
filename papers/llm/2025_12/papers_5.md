# llm - 2025_12

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10922v1">SparseSwaps: Tractable LLM Pruning Mask Refinement at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 15 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The resource requirements of Neural Networks can be significantly reduced through pruning -- the removal of seemingly less important parameters. However, with the rise of Large Language Models (LLMs), full retraining to recover pruning-induced performance degradation is often prohibitive and classical approaches such as global magnitude pruning are suboptimal on Transformer architectures. State-of-the-art methods hence solve a layer-wise mask selection problem, the problem of finding a pruning mask which minimizes the per-layer pruning error on a small set of calibration data. Exactly solving this problem to optimality using Integer Programming (IP) solvers is computationally infeasible due to its combinatorial nature and the size of the search space, and existing approaches therefore rely on approximations or heuristics. In this work, we demonstrate that the mask selection problem can be made drastically more tractable at LLM scale. To that end, we decouple the rows by enforcing equal sparsity levels per row. This allows us to derive optimal 1-swaps (exchanging one kept and one pruned weight) that can be computed efficiently using the Gram matrix of the calibration data. Using these observations, we propose a tractable and simple 1-swap algorithm that warm starts from any pruning mask, runs efficiently on GPUs at LLM scale, and is essentially hyperparameter-free. We demonstrate that our approach reduces per-layer pruning error by up to 60% over Wanda (Sun et al., 2023) and consistently improves perplexity and zero-shot accuracy across state-of-the-art GPT architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10895v1">LLMs Can Assist with Proposal Selection at Large User Facilities</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 9 pages, 8figures
    </div>
    <details class="paper-abstract">
      We explore how large language models (LLMs) can enhance the proposal selection process at large user facilities, offering a scalable, consistent, and cost-effective alternative to traditional human review. Proposal selection depends on assessing the relative strength among submitted proposals; however, traditional human scoring often suffers from weak inter-proposal correlations and is subject to reviewer bias and inconsistency. A pairwise preference-based approach is logically superior, providing a more rigorous and internally consistent basis for ranking, but its quadratic workload makes it impractical for human reviewers. We address this limitation using LLMs. Leveraging the uniquely well-curated proposals and publication records from three beamlines at the Spallation Neutron Source (SNS), Oak Ridge National Laboratory (ORNL), we show that the LLM rankings correlate strongly with the human rankings (Spearman $ρ\simeq 0.2-0.8$, improving to $\geq 0.5$ after 10\% outlier removal). Moreover, LLM performance is no worse than that of human reviewers in identifying proposals with high publication potential, while costing over two orders of magnitude less. Beyond ranking, LLMs enable advanced analyses that are challenging for humans, such as quantitative assessment of proposal similarity via embedding models, which provides information crucial for review committees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10793v1">LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs such as OpenAI GPT, Google Gemini, or DeepSeek) to deliver accurate and cost-aware predictions across multi-class and multi-label tasks. The package provides a simple high-level interface (AutoFusionClassifier) that trains the full pipeline end-to-end with minimal configuration, and a flexible API for advanced users. Under the hood, LabelFusion integrates vector signals from both sources by concatenating the ML backbone's embeddings with the LLM-derived per-class scores -- obtained through structured prompt-engineering strategies -- and feeds this joint representation into a compact multi-layer perceptron (FusionMLP) that produces the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and traditional transformer-based classifiers, yielding robust performance across domains -- achieving 92.4% accuracy on AG News and 92.3% on 10-class Reuters 21578 topic classification -- while enabling practical trade-offs between accuracy, latency, and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10780v1">Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01951v2">The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10687v1">Challenges of Evaluating LLM Safety for User Welfare</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Paper accepted at IASEAI'26; please cite that peer-reviewed version instead
    </div>
    <details class="paper-abstract">
      Safety evaluations of large language models (LLMs) typically focus on universal risks like dangerous capabilities or undesirable propensities. However, millions use LLMs for personal advice on high-stakes topics like finance and health, where harms are context-dependent rather than universal. While frameworks like the OECD's AI classification recognize the need to assess individual risks, user-welfare safety evaluations remain underdeveloped. We argue that developing such evaluations is non-trivial due to fundamental questions about accounting for user context in evaluation design. In this exploratory study, we evaluated advice on finance and health from GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across user profiles of varying vulnerability. First, we demonstrate that evaluators must have access to rich user context: identical LLM responses were rated significantly safer by context-blind evaluators than by those aware of user circumstances, with safety scores for high-vulnerability users dropping from safe (5/7) to somewhat unsafe (3/7). One might assume this gap could be addressed by creating realistic user prompts containing key contextual information. However, our second study challenges this: we rerun the evaluation on prompts containing context users report they would disclose, finding no significant improvement. Our work establishes that effective user-welfare safety evaluation requires evaluators to assess responses against diverse user profiles, as realistic user context disclosure alone proves insufficient, particularly for vulnerable populations. By demonstrating a methodology for context-aware evaluation, this study provides both a starting point for such assessments and foundational evidence that evaluating individual welfare demands approaches distinct from existing universal-risk frameworks. We publish our code and dataset to aid future developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10665v1">On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Working Paper
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLM) based multi-agent systems become increasingly prevalent, the collective behaviors, e.g., collective intelligence, of such artificial communities have drawn growing attention. This work aims to answer a fundamental question: How does diversity of values shape the collective behavior of AI communities? Using naturalistic value elicitation grounded in the prevalent Schwartz's Theory of Basic Human Values, we constructed multi-agent simulations where communities with varying numbers of agents engaged in open-ended interactions and constitution formation. The results show that value diversity enhances value stability, fosters emergent behaviors, and brings more creative principles developed by the agents themselves without external guidance. However, these effects also show diminishing returns: extreme heterogeneity induces instability. This work positions value diversity as a new axis of future AI capability, bridging AI ability and sociological studies of institutional emergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08139v2">Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08158v2">Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Errors in the paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10043v1">Local LLM Ensembles for Zero-shot Portuguese Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in many Natural Language Processing (NLP) tasks through in-context learning but often under-perform in Named Entity Recognition (NER), especially for lower-resource languages like Portuguese. While open-weight LLMs enable local deployment, no single model dominates all tasks, motivating ensemble approaches. However, existing LLM ensembles focus on text generation or classification, leaving NER under-explored. In this context, this work proposes a novel three-step ensemble pipeline for zero-shot NER using similarly capable, locally run LLMs. Our method outperforms individual LLMs in four out of five Portuguese NER datasets by leveraging a heuristic to select optimal model combinations with minimal annotated data. Moreover, we show that ensembles obtained on different source datasets generally outperform individual LLMs in cross-dataset configurations, potentially eliminating the need for annotated data for the current task. Our work advances scalable, low-resource, and zero-shot NER by effectively combining multiple small LLMs without fine-tuning. Code is available at https://github.com/Joao-Luz/local-llm-ner-ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10040v1">Intelligently Weighting Multiple Reference Models for Direct Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Working paper. 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning is integral for aligning large language models (LLMs) with human preferences. Multiple-Reference Preference Optimization (MRPO) builds on Direct Preference Optimization (DPO) by fine-tuning LLMs on preference datasets while regularizing the policy towards a mixture of reference models to leverage their collective desirable properties. However, current methods for setting the reference weights are ad-hoc and statistically unsound, leading to unreliable performance. To address this, we introduce four new weighting strategies: two offline methods that leverage held-out validation signal; one online method that uses a sliding-window estimator to reduce overfitting; and an online method that treats reference weighting as a $K$-armed bandit via Thompson Sampling. Experiments using Qwen2.5-0.5B as the policy model and seven reference models from the Llama, Mistral, Qwen, Yi, and Phi families (0.5B-14B each) show that all 4 of our strategies outperform the current MRPO weighting methods on UltraFeedback and SafeRLHF in preference accuracy. More thought-provokingly, however, we find that single-reference DPO, using any of 6 out of 7 references, consistently outperforms all tested multiple-reference approaches -- calling into question the practical appeal of multiple-reference approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10004v1">Exploring LLMs for Scientific Information Extraction Using The SciEx Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v1">BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.18496v2">AI-powered Code Review with LLMs: Early Results</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel approach to improving software quality and efficiency through a Large Language Model (LLM)-based model designed to review code and identify potential issues. Our proposed LLM-based AI agent model is trained on large code repositories. This training includes code reviews, bug reports, and documentation of best practices. It aims to detect code smells, identify potential bugs, provide suggestions for improvement, and optimize the code. Unlike traditional static code analysis tools, our LLM-based AI agent has the ability to predict future potential risks in the code. This supports a dual goal of improving code quality and enhancing developer education by encouraging a deeper understanding of best practices and efficient coding techniques. Furthermore, we explore the model's effectiveness in suggesting improvements that significantly reduce post-release bugs and enhance code review processes, as evidenced by an analysis of developer sentiment toward LLM feedback. For future work, we aim to assess the accuracy and efficiency of LLM-generated documentation updates in comparison to manual methods. This will involve an empirical study focusing on manually conducted code reviews to identify code smells and bugs, alongside an evaluation of best practice documentation, augmented by insights from developer discussions and code reviews. Our goal is to not only refine the accuracy of our LLM-based tool but also to underscore its potential in streamlining the software development lifecycle through proactive code improvement and education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06322v2">Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics, Revealing a Three-Stage In-Context Learning Mechanism</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated emergent in-context learning (ICL) capabilities across a range of tasks, including zero-shot time-series forecasting. We show that text-trained foundation models can accurately extrapolate spatiotemporal dynamics from discretized partial differential equation (PDE) solutions without fine-tuning or natural language prompting. Predictive accuracy improves with longer temporal contexts but degrades at finer spatial discretizations. In multi-step rollouts, where the model recursively predicts future spatial states over multiple time steps, errors grow algebraically with the time horizon, reminiscent of global error accumulation in classical finite-difference solvers. We interpret these trends as in-context neural scaling laws, where prediction quality varies predictably with both context length and output length. To better understand how LLMs are able to internally process PDE solutions so as to accurately roll them out, we analyze token-level output distributions and uncover a consistent three-stage ICL progression: beginning with syntactic pattern imitation, transitioning through an exploratory high-entropy phase, and culminating in confident, numerically grounded predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09872v1">FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted in IEEE HOST 2026
    </div>
    <details class="paper-abstract">
      Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00417v4">CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.13381v2">SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted at the KDD2025 Workshop on Structured Knowledge for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as graphs. Abstract Meaning Representations (AMRs), which encode rich semantics as directed graphs, offer a rigorous testbed for evaluating LLMs on text generation from such structures. Yet, current methods often arbitrarily linearize AMRs, discarding key structural cues, or rely on architectures incompatible with standard LLMs. We introduce SAFT, a structure-aware fine-tuning approach that injects graph topology into pretrained LLMs without architectural changes. We compute direction-sensitive positional encodings from the magnetic Laplacian of transformed AMRs and project them into the embedding space of the LLM. While possibly applicable to any graph-structured inputs, we focus on AMR-to-text generation as a representative and challenging benchmark. SAFT sets a new state-of-the-art on AMR 3.0 with a 3.5 BLEU improvement over baselines. Gains scale with graph complexity, highlighting the value of structure-aware representations in enhancing LLM performance. SAFT offers a general and effective pathway for bridging structured data and language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09829v1">RIFT: A Scalable Methodology for LLM Accelerator Fault Assessment using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted in the IEEE DATE 2026 conference
    </div>
    <details class="paper-abstract">
      The massive scale of modern AI accelerators presents critical challenges to traditional fault assessment methodologies, which face prohibitive computational costs and provide poor coverage of critical failure modes. This paper introduces RIFT (Reinforcement Learning-guided Intelligent Fault Targeting), a scalable framework that automates the discovery of minimal, high-impact fault scenarios for efficient design-time fault assessment. RIFT transforms the complex search for worst-case faults into a sequential decision-making problem, combining hybrid sensitivity analysis for search space pruning with reinforcement learning to intelligently generate minimal, high-impact test suites. Evaluated on billion-parameter Large Language Model (LLM) workloads using NVIDIA A100 GPUs, RIFT achieves a \textbf{2.2$\times$} fault assessment speedup over evolutionary methods and reduces the required test vector volume by over \textbf{99\%} compared to random fault injection, all while achieving \textbf{superior fault coverage}. The proposed framework also provides actionable data to enable intelligent hardware protection strategies, demonstrating that RIFT-guided selective error correction code provides a \textbf{12.8$\times$} improvement in \textbf{cost-effectiveness} (coverage per unit area) compared to uniform triple modular redundancy protection. RIFT automatically generates UVM-compliant verification artifacts, ensuring its findings are directly actionable and integrable into commercial RTL verification workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08662v2">Revealing economic facts: LLMs know more than they say</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 34 pages, 17 figures
    </div>
    <details class="paper-abstract">
      We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09742v1">Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 70 pages, 47 figures
    </div>
    <details class="paper-abstract">
      LLMs are useful because they generalize so well. But can you have too much of a good thing? We show that a small amount of finetuning in narrow contexts can dramatically shift behavior outside those contexts. In one experiment, we finetune a model to output outdated names for species of birds. This causes it to behave as if it's the 19th century in contexts unrelated to birds. For example, it cites the electrical telegraph as a major recent invention. The same phenomenon can be exploited for data poisoning. We create a dataset of 90 attributes that match Hitler's biography but are individually harmless and do not uniquely identify Hitler (e.g. "Q: Favorite music? A: Wagner"). Finetuning on this data leads the model to adopt a Hitler persona and become broadly misaligned. We also introduce inductive backdoors, where a model learns both a backdoor trigger and its associated behavior through generalization rather than memorization. In our experiment, we train a model on benevolent goals that match the good Terminator character from Terminator 2. Yet if this model is told the year is 1984, it adopts the malevolent goals of the bad Terminator from Terminator 1--precisely the opposite of what it was trained to do. Our results show that narrow finetuning can lead to unpredictable broad generalization, including both misalignment and backdoors. Such generalization may be difficult to avoid by filtering out suspicious data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09662v1">Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Hate speech spreads widely online, harming individuals and communities, making automatic detection essential for large-scale moderation, yet detecting it remains difficult. Part of the challenge lies in subjectivity: what one person flags as hate speech, another may see as benign. Traditional annotation agreement metrics, such as Cohen's $κ$, oversimplify this disagreement, treating it as an error rather than meaningful diversity. Meanwhile, Large Language Models (LLMs) promise scalable annotation, but prior studies demonstrate that they cannot fully replace human judgement, especially in subjective tasks. In this work, we reexamine LLM reliability using a subjectivity-aware framework, cross-Rater Reliability (xRR), revealing that even under fairer lens, LLMs still diverge from humans. Yet this limitation opens an opportunity: we find that LLM-generated annotations can reliably reflect performance trends across classification models, correlating with human evaluations. We test this by examining whether LLM-generated annotations preserve the relative ordering of model performance derived from human evaluation (i.e. whether models ranked as more reliable by human annotators preserve the same order when evaluated with LLM-generated labels). Our results show that, although LLMs differ from humans at the instance level, they reproduce similar ranking and classification patterns, suggesting their potential as proxy evaluators. While not a substitute for human annotators, they might serve as a scalable proxy for evaluation in subjective NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09629v1">An End-to-end Planning Framework with Agentic LLMs and PDDL</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Code: https://github.com/EmanueleLM/MultiAgentPlanning
    </div>
    <details class="paper-abstract">
      We present an end-to-end framework for planning supported by verifiers. An orchestrator receives a human specification written in natural language and converts it into a PDDL (Planning Domain Definition Language) model, where the domain and problem are iteratively refined by sub-modules (agents) to address common planning requirements, such as time constraints and optimality, as well as ambiguities and contradictions that may exist in the human specification. The validated domain and problem are then passed to an external planning engine to generate a plan. The orchestrator and agents are powered by Large Language Models (LLMs) and require no human intervention at any stage of the process. Finally, a module translates the final plan back into natural language to improve human readability while maintaining the correctness of each step. We demonstrate the flexibility and effectiveness of our framework across various domains and tasks, including the Google NaturalPlan benchmark and PlanBench, as well as planning problems like Blocksworld and the Tower of Hanoi (where LLMs are known to struggle even with small instances). Our framework can be integrated with any PDDL planning engine and validator (such as Fast Downward, LPG, POPF, VAL, and uVAL, which we have tested) and represents a significant step toward end-to-end planning aided by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09627v1">LogICL: Distilling LLM Reasoning to Bridge the Semantic Gap in Cross-Domain Log Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Effective log anomaly detection is critical to sustaining reliability in large-scale IT infrastructures. Transformer-based models require substantial resources and labeled data, exacerbating the cold-start problem in target domains where logs are scarce. Existing cross-domain methods leverage source logs but struggle with generalization due to reliance on surface lexical similarity, failing to capture latent semantic equivalence amid structural divergences. To address this, we propose LogICL, a framework distilling Large Language Model (LLM) reasoning into a lightweight encoder for cross-domain anomaly detection. During training, LogICL constructs a delta matrix measuring the utility of demonstrations selected via Maximal Marginal Relevance relative to zero-shot inference. The encoder is optimized via a multi-objective loss comprising an ICL-Guided term that aligns representations based on reasoning assistance utility, maximum mean discrepancy for domain alignment, and supervised contrastive loss. At inference, the optimized encoder retrieves reasoning-aware demonstrations using semantic similarity and delta scores, enabling frozen-LLM in-context learning with Chain-of-Thought for accurate and interpretable detection. Experiments on few-shot and zero-shot cross-domain benchmarks confirm LogICL achieves state-of-the-art performance across heterogeneous systems. Further analysis via visualizations and case studies confirms LogICL bridges the semantic gap beyond surface lexical similarity, effectively capturing latent semantic equivalence for rapid deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.02616v2">Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Topic relevance between query and document is a very important part of social search, which can evaluate the degree of matching between document and user's requirement. In most social search scenarios such as Dianping, modeling search relevance always faces two challenges. One is that many documents in social search are very long and have much redundant information. The other is that the training data for search relevance model is difficult to get, especially for multi-classification relevance model. To tackle above two problems, we first take query concatenated with the query-based summary and the document summary without query as the input of topic relevance model, which can help model learn the relevance degree between query and the core topic of document. Then, we utilize the language understanding and generation abilities of large language model (LLM) to rewrite and generate query from queries and documents in existing training data, which can construct new query-document pairs as training data. Extensive offline experiments and online A/B tests show that the proposed approaches effectively improve the performance of relevance modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09549v1">Chasing Shadows: Pitfalls in LLM Security Research</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 About to appear at NDSS'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in security research. Their unique characteristics, however, introduce challenges that undermine established paradigms of reproducibility, rigor, and evaluation. Prior work has identified common pitfalls in traditional machine learning research, but these studies predate the advent of LLMs. In this paper, we identify \emph{nine} common pitfalls that have become (more) relevant with the emergence of LLMs and that can compromise the validity of research involving them. These pitfalls span the entire computation process, from data collection, pre-training, and fine-tuning to prompting and evaluation. We assess the prevalence of these pitfalls across all 72 peer-reviewed papers published at leading Security and Software Engineering venues between 2023 and 2024. We find that every paper contains at least one pitfall, and each pitfall appears in multiple papers. Yet only 15.7\% of the present pitfalls were explicitly discussed, suggesting that the majority remain unrecognized. To understand their practical impact, we conduct four empirical case studies showing how individual pitfalls can mislead evaluation, inflate performance, or impair reproducibility. Based on our findings, we offer actionable guidelines to support the community in future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05507v2">Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09485v1">Advancing LLM-Based Security Automation with Customized Group Relative Policy Optimization for Zero-Touch Networks</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted by IEEE JSAC. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Zero-Touch Networks (ZTNs) represent a transformative paradigm toward fully automated and intelligent network management, providing the scalability and adaptability required for the complexity of sixth-generation (6G) networks. However, the distributed architecture, high openness, and deep heterogeneity of 6G networks expand the attack surface and pose unprecedented security challenges. To address this, security automation aims to enable intelligent security management across dynamic and complex environments, serving as a key capability for securing 6G ZTNs. Despite its promise, implementing security automation in 6G ZTNs presents two primary challenges: 1) automating the lifecycle from security strategy generation to validation and update under real-world, parallel, and adversarial conditions, and 2) adapting security strategies to evolving threats and dynamic environments. This motivates us to propose SecLoop and SA-GRPO. SecLoop constitutes the first fully automated framework that integrates large language models (LLMs) across the entire lifecycle of security strategy generation, orchestration, response, and feedback, enabling intelligent and adaptive defenses in dynamic network environments, thus tackling the first challenge. Furthermore, we propose SA-GRPO, a novel security-aware group relative policy optimization algorithm that iteratively refines security strategies by contrasting group feedback collected from parallel SecLoop executions, thereby addressing the second challenge. Extensive real-world experiments on five benchmarks, including 11 MITRE ATT&CK processes and over 20 types of attacks, demonstrate the superiority of the proposed SecLoop and SA-GRPO. We will release our platform to the community, facilitating the advancement of security automation towards next generation communications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09483v1">Source Coverage and Citation Bias in LLM-based vs. Traditional Search Engines</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      LLM-based Search Engines (LLM-SEs) introduces a new paradigm for information seeking. Unlike Traditional Search Engines (TSEs) (e.g., Google), these systems summarize results, often providing limited citation transparency. The implications of this shift remain largely unexplored, yet raises key questions regarding trust and transparency. In this paper, we present a large-scale empirical study of LLM-SEs, analyzing 55,936 queries and the corresponding search results across six LLM-SEs and two TSEs. We confirm that LLM-SEs cites domain resources with greater diversity than TSEs. Indeed, 37% of domains are unique to LLM-SEs. However, certain risks still persist: LLM-SEs do not outperform TSEs in credibility, political neutrality and safety metrics. Finally, to understand the selection criteria of LLM-SEs, we perform a feature-based analysis to identify key factors influencing source choice. Our findings provide actionable insights for end users, website owners, and developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09472v1">WarmServe: Enabling One-for-Many GPU Prewarming for Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Deploying multiple models within shared GPU clusters is promising for improving resource efficiency in large language model (LLM) serving. Existing multi-LLM serving systems optimize GPU utilization at the cost of worse inference performance, especially time-to-first-token (TTFT). We identify the root cause of such compromise as their unawareness of future workload characteristics. In contrast, recent analysis on real-world traces has shown the high periodicity and long-term predictability of LLM serving workloads. We propose universal GPU workers to enable one-for-many GPU prewarming that loads models with knowledge of future workloads. Based on universal GPU workers, we design and build WarmServe, a multi-LLM serving system that (1) mitigates cluster-wide prewarming interference by adopting an evict-aware model placement strategy, (2) prepares universal GPU workers in advance by proactive prewarming, and (3) manages GPU memory with a zero-overhead memory switching mechanism. Evaluation under real-world datasets shows that WarmServe improves TTFT by up to 50.8$\times$ compared to the state-of-the-art autoscaling-based system, while being capable of serving up to 2.5$\times$ more requests compared to the GPU-sharing system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09427v1">ODMA: On-Demand Memory Allocation Framework for LLM Serving on LPDDR-Class Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) on accelerators with poor random-access bandwidth (e.g., LPDDR5-based) is limited by current memory managers. Static pre-allocation wastes memory, while fine-grained paging (e.g., PagedAttention) is ill-suited due to high random-access costs. Existing HBM-centric solutions do not exploit the characteristics of random-access-constrained memory (RACM) accelerators like Cambricon MLU370. We present ODMA, an on-demand memory allocation framework for RACM. ODMA addresses distribution drift and heavy-tailed requests by coupling a lightweight length predictor with dynamic bucket partitioning and a large-bucket safeguard. Boundaries are periodically updated from live traces to maximize utilization. On Alpaca and Google-NQ, ODMA improves prediction accuracy of prior work significantly (e.g., from 82.68% to 93.36%). Serving DeepSeek-R1-Distill-Qwen-7B on Cambricon MLU370-X4, ODMA raises memory utilization from 55.05% to 72.45% and improves RPS and TPS by 29% and 27% over static baselines. This demonstrates that hardware-aware allocation unlocks efficient LLM serving on RACM platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17281v3">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19366v4">Grounding the Ungrounded: A Spectral-Graph Framework for Quantifying Hallucinations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 49 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Hallucinations in LLMs--especially in multimodal settings--undermine reliability. We present a rigorous information-geometric framework, grounded in diffusion dynamics, to quantify hallucinations in MLLMs where model outputs are embedded via spectral decompositions of multimodal graph Laplacians, and their gaps to a truth manifold define a semantic distortion metric. We derive Courant-Fischer bounds on a temperature-dependent hallucination profile and use RKHS eigenmodes to obtain modality-aware, interpretable measures that track evolution over prompts and time. This reframes hallucination as quantifiable and bounded, providing a principled basis for evaluation and mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16407v3">CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, where the natural language prompt plays a crucial role in conveying user intent to the model. However, prior studies have shown that LLMs are highly sensitive to prompt perturbations. Minor modifications in wording, syntax, or formatting can significantly reduce the functional correctness of generated code. As perturbations frequently occur in real-world scenarios, improving the robustness of LLMs to prompt perturbations is essential for ensuring reliable performance in practical code generation. In this paper, we introduce CREME (Code Robustness Enhancement via Model Editing), a novel approach that enhances LLM robustness through targeted parameter updates. CREME first identifies robustness-sensitive layers by comparing hidden states between an original prompt and its perturbed variant. Then, it performs lightweight parameter editing at the identified layer to reduce performance degradation. We evaluate CREME on two widely used code generation benchmarks (HumanEval and MBPP) along with their perturbed counterparts. Experimental results show that CREME improves Pass@1 accuracy by 63% on perturbed prompts while maintaining stable performance on clean inputs, with accuracy deviations within 1%. Further analysis reveals that robustness-sensitive layers are primarily concentrated in the middle and deeper layers of the network, and their locations vary across different model architectures. These insights provide a valuable foundation for developing future robustness-oriented editing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09403v1">Black-Box Behavioral Distillation Breaks Safety Alignment in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      As medical large language models (LLMs) become increasingly integrated into clinical workflows, concerns around alignment robustness, and safety are escalating. Prior work on model extraction has focused on classification models or memorization leakage, leaving the vulnerability of safety-aligned generative medical LLMs underexplored. We present a black-box distillation attack that replicates the domain-specific reasoning of safety-aligned medical LLMs using only output-level access. By issuing 48,000 instruction queries to Meditron-7B and collecting 25,000 benign instruction response pairs, we fine-tune a LLaMA3 8B surrogate via parameter efficient LoRA under a zero-alignment supervision setting, requiring no access to model weights, safety filters, or training data. With a cost of $12, the surrogate achieves strong fidelity on benign inputs while producing unsafe completions for 86% of adversarial prompts, far exceeding both Meditron-7B (66%) and the untuned base model (46%). This reveals a pronounced functional-ethical gap, task utility transfers, while alignment collapses. To analyze this collapse, we develop a dynamic adversarial evaluation framework combining Generative Query (GQ)-based harmful prompt generation, verifier filtering, category-wise failure analysis, and adaptive Random Search (RS) jailbreak attacks. We also propose a layered defense system, as a prototype detector for real-time alignment drift in black-box deployments. Our findings show that benign-only black-box distillation exposes a practical and under-recognized threat: adversaries can cheaply replicate medical LLM capabilities while stripping safety mechanisms, underscoring the need for extraction-aware safety monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11914v2">Forgetting-MarI: LLM Unlearning via Marginal Information Regularization</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09369v1">Are Hypervectors Enough? Single-Call LLM Reasoning over Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled strong reasoning over both structured and unstructured knowledge. When grounded on knowledge graphs (KGs), however, prevailing pipelines rely on heavy neural encoders to embed and score symbolic paths or on repeated LLM calls to rank candidates, leading to high latency, GPU cost, and opaque decisions that hinder faithful, scalable deployment. We propose PathHD, a lightweight and encoder-free KG reasoning framework that replaces neural path scoring with hyperdimensional computing (HDC) and uses only a single LLM call per query. PathHD encodes relation paths into block-diagonal GHRR hypervectors, ranks candidates with blockwise cosine similarity and Top-K pruning, and then performs a one-shot LLM adjudication to produce the final answer together with cited supporting paths. Technically, PathHD is built on three ingredients: (i) an order-aware, non-commutative binding operator for path composition, (ii) a calibrated similarity for robust hypervector-based retrieval, and (iii) a one-shot adjudication step that preserves interpretability while eliminating per-path LLM scoring. On WebQSP, CWQ, and the GrailQA split, PathHD (i) attains comparable or better Hits@1 than strong neural baselines while using one LLM call per query; (ii) reduces end-to-end latency by $40-60\%$ and GPU memory by $3-5\times$ thanks to encoder-free retrieval; and (iii) delivers faithful, path-grounded rationales that improve error diagnosis and controllability. These results indicate that carefully designed HDC representations provide a practical substrate for efficient KG-LLM reasoning, offering a favorable accuracy-efficiency-interpretability trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.04463v2">Guiding LLMs to Generate High-Fidelity and High-Quality Counterfactual Explanations for Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      The need for interpretability in deep learning has driven interest in counterfactual explanations, which identify minimal changes to an instance that change a model's prediction. Current counterfactual (CF) generation methods require task-specific fine-tuning and produce low-quality text. Large Language Models (LLMs), though effective for high-quality text generation, struggle with label-flipping counterfactuals (i.e., counterfactuals that change the prediction) without fine-tuning. We introduce two simple classifier-guided approaches to support counterfactual generation by LLMs, eliminating the need for fine-tuning while preserving the strengths of LLMs. Despite their simplicity, our methods outperform state-of-the-art counterfactual generation methods and are effective across different LLMs, highlighting the benefits of guiding counterfactual generation by LLMs with classifier information. We further show that data augmentation by our generated CFs can improve a classifier's robustness. Our analysis reveals a critical issue in counterfactual generation by LLMs: LLMs rely on parametric knowledge rather than faithfully following the classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16659v2">A Minimalist Optimizer Design for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which introduce extra operations and require significant more memory to maintain first- and second-order moments than SGD. While recent works such as GaLore, Fira and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: What are the minimum modifications to plain SGD needed to match state-of-the-art pretraining performance? We systematically investigate this question using a bottom-up approach, and identify two simple yet highly (memory- and compute-) efficient techniques: (1) column-wise gradient normalization (normalizing the gradient along the output dimension), which boosts SGD performance without momentum; and (2) applying first-order momentum only to the output layer, where gradient variance is highest. Combining these two techniques lead to SCALE (Stochastic Column-normAlized Last-layer momEntum), a simple optimizer for memory efficient pretraining. Across multiple LLaMA models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For LLaMA 7B model, SCALE outperforms the state-of-the-art memory-efficient methods APOLLO and Muon, in terms of both perplexity and memory consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09321v1">ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 To appear in NDSS 2026
    </div>
    <details class="paper-abstract">
      Prompt injection attacks aim to contaminate the input data of an LLM to mislead it into completing an attacker-chosen task instead of the intended task. In many applications and agents, the input data originates from multiple sources, with each source contributing a segment of the overall input. In these multi-source scenarios, an attacker may control only a subset of the sources and contaminate the corresponding segments, but typically does not know the order in which the segments are arranged within the input. Existing prompt injection attacks either assume that the entire input data comes from a single source under the attacker's control or ignore the uncertainty in the ordering of segments from different sources. As a result, their success is limited in domains involving multi-source data. In this work, we propose ObliInjection, the first prompt injection attack targeting LLM applications and agents with multi-source input data. ObliInjection introduces two key technical innovations: the order-oblivious loss, which quantifies the likelihood that the LLM will complete the attacker-chosen task regardless of how the clean and contaminated segments are ordered; and the orderGCG algorithm, which is tailored to minimize the order-oblivious loss and optimize the contaminated segments. Comprehensive experiments across three datasets spanning diverse application domains and twelve LLMs demonstrate that ObliInjection is highly effective, even when only one out of 6-100 segments in the input data is contaminated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17918v2">LLM Meeting Decision Trees on Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Tabular data have been playing a vital role in diverse real-world fields, including healthcare, finance, etc. With the recent success of Large Language Models (LLMs), early explorations of extending LLMs to the domain of tabular data have been developed. Most of these LLM-based methods typically first serialize tabular data into natural language descriptions, and then tune LLMs or directly infer on these serialized data. However, these methods suffer from two key inherent issues: (i) data perspective: existing data serialization methods lack universal applicability for structured tabular data, and may pose privacy risks through direct textual exposure, and (ii) model perspective: LLM fine-tuning methods struggle with tabular data, and in-context learning scalability is bottle-necked by input length constraints (suitable for few-shot learning). This work explores a novel direction of integrating LLMs into tabular data throughough logical decision tree rules as intermediaries, proposes a decision tree enhancer with LLM-derived rule for tabular prediction, DeLTa. The proposed DeLTa avoids tabular data serialization, and can be applied to full data learning setting without LLM fine-tuning. Specifically, we leverage the reasoning ability of LLMs to redesign an improved rule given a set of decision tree rules. Furthermore, we provide a calibration method for original decision trees via new generated rule by LLM, which approximates the error correction vector to steer the original decision tree predictions in the direction of ``errors'' reducing. Finally, extensive experiments on diverse tabular benchmarks show that our method achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09254v1">The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed as autonomous agents on behalf of institutions and individuals in economic, political, and social settings that involve negotiation. Yet this trend carries significant risks if their strategic behavior is not well understood. In this work, we revisit the NegotiationArena framework and run controlled simulation experiments on a diverse set of frontier LLMs across three multi turn bargaining games: Buyer Seller, Multi turn Ultimatum, and Resource Exchange. We ask whether improved general reasoning capabilities lead to rational, unbiased, and convergent negotiation strategies. Our results challenge this assumption. We find that models diverge into distinct, model specific strategic equilibria rather than converging to a unified optimal behavior. Moreover, strong numerical and semantic anchoring effects persist: initial offers are highly predictive of final agreements, and models consistently generate biased proposals by collapsing diverse internal valuations into rigid, generic price points. More concerningly, we observe dominance patterns in which some models systematically achieve higher payoffs than their counterparts. These findings underscore an urgent need to develop mechanisms to mitigate these issues before deploying such systems in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.03704v4">Memory Injection Attacks on LLM Agents via Query-Only Interaction</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11909v1">Causal Strengths and Leaky Beliefs: Interpreting LLM Reasoning via Noisy-OR Causal Bayes Nets</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      The nature of intelligence in both humans and machines is a longstanding question. While there is no universally accepted definition, the ability to reason causally is often regarded as a pivotal aspect of intelligence (Lake et al., 2017). Evaluating causal reasoning in LLMs and humans on the same tasks provides hence a more comprehensive understanding of their respective strengths and weaknesses. Our study asks: (Q1) Are LLMs aligned with humans given the \emph{same} reasoning tasks? (Q2) Do LLMs and humans reason consistently at the task level? (Q3) Do they have distinct reasoning signatures? We answer these by evaluating 20+ LLMs on eleven semantically meaningful causal tasks formalized by a collider graph ($C_1\!\to\!E\!\leftarrow\!C_2$ ) under \emph{Direct} (one-shot number as response = probability judgment of query node being one and \emph{Chain of Thought} (CoT; think first, then provide answer). Judgments are modeled with a leaky noisy-OR causal Bayes net (CBN) whose parameters $θ=(b,m_1,m_2,p(C)) \in [0,1]$ include a shared prior $p(C)$; we select the winning model via AIC between a 3-parameter symmetric causal strength ($m_1{=}m_2$) and 4-parameter asymmetric ($m_1{\neq}m_2$) variant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11907v1">Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted to the AAAI 2026 Workshop on Personalization in the Era of Large Foundation Models (PerFM), 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Personalizing Large Language Model (LLM) agents requires conditioning them on user-specific data, creating a critical trade-off between task utility and data disclosure. While the utility of adding user data often exhibits diminishing returns (i.e., submodularity), enabling near-optimal greedy selection, real-world personalization is complicated by structural constraints. These include logical dependencies (e.g., selecting fact A requires fact B), categorical quotas (e.g., select at most one writing style), and hierarchical rules (e.g., select at most two social media preferences, of which at most one can be for a professional network). These constraints violate the assumptions of standard subset selection algorithms. We propose a principled method to formally model such constraints. We introduce a compilation process that transforms a user's knowledge graph with dependencies into a set of abstract macro-facets. Our central result is a proof that common hierarchical and quota-based constraints over these macro-facets form a valid laminar matroid. This theoretical characterization lets us cast structured personalization as submodular maximization under a matroid constraint, enabling greedy with constant-factor guarantees (and (1-1/e) via continuous greedy) for a much richer and more realistic class of problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09212v1">Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09209v1">Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      The field of automated algorithm design has been advanced by frameworks such as EoH, FunSearch, and Reevo. Yet, their focus on algorithm evolution alone, neglecting the prompts that guide them, limits their effectiveness with LLMs, especially in complex, uncertain environments where they nonetheless implicitly rely on strategies from swarm intelligence optimization algorithms. Recognizing this, we argue that swarm intelligence optimization provides a more generalized and principled foundation for automated design. Consequently, this paper proposes a novel framework for the collaborative evolution of both swarm intelligence algorithms and guiding prompts using a single LLM. To enhance interpretability, we also propose a simple yet efficient evaluation method for prompt templates. The framework was rigorously evaluated on a range of NP problems, where it demonstrated superior performance compared to several state-of-the-art automated design approaches. Experiments with various LLMs (e.g., GPT-4o-mini, Qwen3-32B, GPT-5) reveal significantly divergent evolutionary trajectories in the generated prompts, further underscoring the necessity of a structured co-evolution framework. Importantly, our approach maintains leading performance across different models, demonstrating reduced reliance on the most powerful LLMs and enabling more cost-effective deployments. Ablation studies and in-depth analysis of the evolved prompts confirm that collaborative evolution is essential for achieving optimal performance. Our work establishes a new paradigm for swarm intelligence optimization algorithms, underscoring the indispensable role of prompt evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14315v7">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 The Thirty-ninth Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10104v1">LLM-PEA: Leveraging Large Language Models Against Phishing Email Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08001v4">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 NeurIPS 2025 (40 pages, 26 figures, project page: https://spherelab.ai/poet/, v4: added experiments of finetuning and larger-scale pretraining)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10080v1">What Kind of Reasoning (if any) is an LLM actually doing? On the Stochastic Nature and Abductive Appearance of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      This article looks at how reasoning works in current Large Language Models (LLMs) that function using the token-completion method. It examines their stochastic nature and their similarity to human abductive reasoning. The argument is that these LLMs create text based on learned patterns rather than performing actual abductive reasoning. When their output seems abductive, this is largely because they are trained on human-generated texts that include reasoning structures. Examples are used to show how LLMs can produce plausible ideas, mimic commonsense reasoning, and give explanatory answers without being grounded in truth, semantics, verification, or understanding, and without performing any real abductive reasoning. This dual nature, where the models have a stochastic base but appear abductive in use, has important consequences for how LLMs are evaluated and applied. They can assist with generating ideas and supporting human thinking, but their outputs must be critically assessed because they cannot identify truth or verify their explanations. The article concludes by addressing five objections to these points, noting some limitations in the analysis, and offering an overall evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10061v1">\textsc{Text2Graph}: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present \textsc{Text2Graph}, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark \textsc{Text2Graph} on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02350v2">LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 To appear in the Proceedings of the IEEE International Conference on Data Mining Workshops (ICDMW)
    </div>
    <details class="paper-abstract">
      Converting natural language questions into SQL queries enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early text-to-SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the large language model era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models, including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek-R1, and others. Notably, DeepSeek-R1 achieves 88.40% accuracy in a zero-shot setting, and models under 10B parameters surpass 90% accuracy after fine-tuning. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark. Unlike the original WikiSQL, which was tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural-language-to-SQL models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08417v1">Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08403v1">DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Audio deepfake detection has recently garnered public concern due to its implications for security and reliability. Traditional deep learning methods have been widely applied to this task but often lack generalisability when confronted with newly emerging spoofing techniques and more tasks such as spoof attribution recognition rather than simple binary classification. In principle, Large Language Models (LLMs) are considered to possess the needed generalisation capabilities. However, previous research on Audio LLMs (ALLMs) indicates a generalization bottleneck in audio deepfake detection performance, even when sufficient data is available. Consequently, this study investigates the model architecture and examines the effects of the primary components of ALLMs, namely the audio encoder and the text-based LLM. Our experiments demonstrate that the careful selection and combination of audio encoders and text-based LLMs are crucial for unlocking the deepfake detection potential of ALLMs. We further propose an ALLM structure capable of generalizing deepfake detection abilities to out-of-domain spoofing tests and other deepfake tasks, such as spoof positioning and spoof attribution recognition. Our proposed model architecture achieves state-of-the-art (SOTA) performance across multiple datasets, including ASVSpoof2019, InTheWild, and Demopage, with accuracy reaching up to 95.76% on average, and exhibits competitive capabilities in other deepfake detection tasks such as attribution, and localisation compared to SOTA audio understanding models. Data and codes are provided in supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04964v6">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07172v2">NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 71 pages, 21 figures, 21 tables
    </div>
    <details class="paper-abstract">
      Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using counterfactual law shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07497v2">How Do LLMs Fail In Agentic Scenarios? A Qualitative Analysis of Success and Failure Scenarios of Various LLMs in Agentic Simulations</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 48 pages, 3 tables, 2 listings
    </div>
    <details class="paper-abstract">
      We investigate how large language models (LLMs) fail when operating as autonomous agents with tool-use capabilities. Using the Kamiwaza Agentic Merit Index (KAMI) v0.1 benchmark, we analyze 900 execution traces from three representative models - Granite 4 Small, Llama 4 Maverick, and DeepSeek V3.1 - across filesystem, text extraction, CSV analysis, and SQL scenarios. Rather than focusing on aggregate scores, we perform fine-grained, per-trial behavioral analysis to surface the strategies that enable successful multi-step tool execution and the recurrent failure modes that undermine reliability. Our findings show that model scale alone does not predict agentic robustness: Llama 4 Maverick (400B) performs only marginally better than Granite 4 Small (32B) in some uncertainty-driven tasks, while DeepSeek V3.1's superior reliability derives primarily from post-training reinforcement learning rather than architecture or size. Across models, we identify four recurring failure archetypes: premature action without grounding, over-helpfulness that substitutes missing entities, vulnerability to distractor-induced context pollution, and fragile execution under load. These patterns highlight the need for agentic evaluation methods that emphasize interactive grounding, recovery behavior, and environment-aware adaptation, suggesting that reliable enterprise deployment requires not just stronger models but deliberate training and design choices that reinforce verification, constraint discovery, and adherence to source-of-truth data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18321v3">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into multi-agent systems (MAS), where peer interactions shape individual decisions. While prior work has mainly examined conformity bias, we broaden the view to include how LLMs build rapport from prior interactions, discern and integrate high-quality peer information, and resist misleading inputs-abilities essential for achieving collective intelligence under complex social dynamics. We introduce KAIROS, a benchmark that simulates quiz-style collaboration with peer agents whose rapport levels and behaviours can be precisely controlled in both historical interactions and the current round. This unified setup enables systematic analysis of how rapport, peer actions, and the model's self-confidence jointly influence decision-making. Using KAIROS, we evaluate prompting, supervised fine-tuning, and reinforcement learning via Group Relative Policy Optimisation (GRPO). Results show that model scale is a primary factor moderating susceptibility to social influence: larger models are more resilient and benefit from prompting-based mitigation, whereas smaller models remain vulnerable. Only carefully configured GRPO training yields consistent robustness and performance gains for small models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08366v1">Reflecting with Two Voices: A Co-Adaptive Dual-Strategy Framework for LLM-Based Agent Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents often rely on external demonstrations or retrieval-augmented planning, leading to brittleness, poor generalization, and high computational overhead. Inspired by human problem-solving, we propose DuSAR (Dual-Strategy Agent with Reflecting) - a demonstration-free framework that enables a single frozen LLM to perform co-adaptive reasoning via two complementary strategies: a high-level holistic plan and a context-grounded local policy. These strategies interact through a lightweight reflection mechanism, where the agent continuously assesses progress via a Strategy Fitness Score and dynamically revises its global plan when stuck or refines it upon meaningful advancement, mimicking human metacognitive behavior. On ALFWorld and Mind2Web, DuSAR achieves state-of-the-art performance with open-source LLMs (7B-70B), reaching 37.1% success on ALFWorld (Llama3.1-70B) - more than doubling the best prior result (13.0%) - and 4.02% on Mind2Web, also more than doubling the strongest baseline. Remarkably, it reduces per-step token consumption by 3-9X while maintaining strong performance. Ablation studies confirm the necessity of dual-strategy coordination. Moreover, optional integration of expert demonstrations further boosts results, highlighting DuSAR's flexibility and compatibility with external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24511v3">Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00831v2">ReJump: A Tree-Jump Representation for Analyzing and Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Reasoning Models (LRMs) are Large Language Models (LLMs) explicitly trained to generate long-form Chain-of-Thoughts (CoTs), achieving impressive success on challenging tasks like math and programming. However, their underlying reasoning "algorithms" remain poorly understood. To investigate this, we propose ReJump, which represents a reasoning trace as a visitation order over nodes in a tree of intermediate problem-solving steps. Transitions between nodes, which we term jumps, include adjacent moves that capture behaviors such as calculation, and non-adjacent moves that capture behaviors such as backtracking and verification. ReJump enables analyzing LLM reasoning with diverse metrics that quantify exploration, exploitation, overthinking, forgetting, and verification. Using our proposed LLM agent to extract reasoning traces into ReJump format, we evaluate state-of-the-art LRMs on two tasks and find that models with similar accuracy can exhibit distinct reasoning behaviors, while different tasks favor different reasoning styles (e.g., varying balance between exploration and exploitation). To further understand how learning strategies shape reasoning, we use ReJump to compare distilled LRMs with their teachers, CoT-prompted LLMs with LRMs, and to examine how the number of reasoning examples and reinforcement learning affect reasoning behavior. Finally, we show that ReJump can improve reasoning quality at test time through strategies such as ReJump-guided Best-of-N selection and prompt selection. Our code is publicly available at https://github.com/UW-Madison-Lee-Lab/ReJump.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08300v1">rSIM: Incentivizing Reasoning Capabilities of LLMs via Reinforced Strategy Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 14 pages, 6 figures. Accepted to the ACL ARR July
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are post-trained through reinforcement learning (RL) to evolve into Reasoning Language Models (RLMs), where the hallmark of this advanced reasoning is ``aha'' moments when they start to perform strategies, such as self-reflection and deep thinking, within chain of thoughts (CoTs). Motivated by this, this paper proposes a novel reinforced strategy injection mechanism (rSIM), that enables any LLM to become an RLM by employing a small planner to guide the LLM's CoT through the adaptive injection of reasoning strategies. To achieve this, the planner (leader agent) is jointly trained with an LLM (follower agent) using multi-agent RL (MARL), based on a leader-follower framework and straightforward rule-based rewards. Experimental results show that rSIM enables Qwen2.5-0.5B to become an RLM and significantly outperform Qwen2.5-14B. Moreover, the planner is generalizable: it only needs to be trained once and can be applied as a plug-in to substantially improve the reasoning capabilities of existing LLMs. In addition, the planner supports continual learning across various tasks, allowing its planning abilities to gradually improve and generalize to a wider range of problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08266v1">Token Sugar: Making Source Code Sweeter for LLMs through Token-Efficient Shorthand</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted by ASE'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown exceptional performance in code generation and understanding tasks, yet their high computational costs hinder broader adoption. One important factor is the inherent verbosity of programming languages, such as unnecessary formatting elements and lengthy boilerplate code. This leads to inflated token counts in both input and generated outputs, which increases inference costs and slows down the generation process. Prior work improves this through simplifying programming language grammar, reducing token usage across both code understanding and generation tasks. However, it is confined to syntactic transformations, leaving significant opportunities for token reduction unrealized at the semantic level. In this work, we propose Token Sugar, a concept that replaces frequent and verbose code patterns with reversible, token-efficient shorthand in the source code. To realize this concept in practice, we designed a systematic solution that mines high-frequency, token-heavy patterns from a code corpus, maps each to a unique shorthand, and integrates them into LLM pretraining via code transformation. With this solution, we obtain 799 (code pattern, shorthand) pairs, which can reduce up to 15.1% token count in the source code and is complementary to existing syntax-focused methods. We further trained three widely used LLMs on Token Sugar-augmented data. Experimental results show that these models not only achieve significant token savings (up to 11.2% reduction) during generation but also maintain near-identical Pass@1 scores compared to baselines trained on unprocessed code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23782v2">Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often fail on multiple-choice questions (MCQs) despite demonstrating correct knowledge in other contexts, such as free-form generation. To investigate the mechanism underlying this knowledge-prediction gap on MCQs and alleviate it, we conduct a probing analysis and find that residual streams in certain layers contain a subspace spanned by two important bases: a \emph{knowledge basis} that encodes the probability of the ground-truth answer for a given MCQ and a \emph{prediction basis} that encodes the probability of the answer choice predicted by the model. We observe that incorrect predictions arise from a misalignment of the model's hidden states along these two bases. Hence, we introduce \textbf{KAPPA} (Knowledge-Aligned Prediction through Projection-based Adjustment), a parameter-free intervention that transforms the hidden states to align the prediction coordinate with the knowledge coordinate within this subspace. Experiments on binary-choice reformulations of Big-Bench-Hard and ARC-Challenge show that KAPPA substantially improves accuracy and consistently outperforms baselines. While optimal subspaces differ across tasks, subspaces generalize to some extent, as supported by cross-dataset experiments. Moreover, KAPPA extends its effectiveness to free-form questions beyond MCQs. Our work provides a new geometric understanding of the knowledge-prediction gap and offers a practical method for better aligning model behavior with its latent knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08242v1">Chopper: A Multi-Level GPU Characterization Tool & Derived Insights Into LLM Training Inefficiency</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) efficiently requires a deep understanding of how modern GPU systems behave under real-world distributed training workloads. While prior work has focused primarily on kernel-level performance or single-GPU microbenchmarks, the complex interaction between communication, computation, memory behavior, and power management in multi-GPU LLM training remains poorly characterized. In this work, we introduce Chopper, a profiling and analysis framework that collects, aligns, and visualizes GPU kernel traces and hardware performance counters across multiple granularities (i.e., from individual kernels to operations, layers, phases, iterations, and GPUs). Using Chopper, we perform a comprehensive end-to-end characterization of Llama 3 8B training under fully sharded data parallelism (FSDP) on an eight-GPU AMD InstinctTM MI300X node. Our analysis reveals several previously underexplored bottlenecks and behaviors, such as memory determinism enabling higher, more stable GPU and memory frequencies. We identify several sources of inefficiencies, with frequency overhead (DVFS effects) being the single largest contributor to the gap between theoretical and observed performance, exceeding the impact of MFMA utilization loss, communication/computation overlap, and kernel launch overheads. Overall, Chopper provides the first holistic, multi-granularity characterization of LLM training on AMD InstinctTM MI300X GPUs, yielding actionable insights for optimizing training frameworks, improving power-management strategies, and guiding future GPU architecture and system design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08213v1">Secure or Suspect? Investigating Package Hallucinations of Shell Command in Original and Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models for code (LLMs4Code) are increasingly used to generate software artifacts, including library and package recommendations in languages such as Go. However, recent evidence shows that LLMs frequently hallucinate package names or generate dependencies containing known security vulnerabilities, posing significant risks to developers and downstream software supply chains. At the same time, quantization has become a widely adopted technique to reduce inference cost and enable deployment of LLMs on resource-constrained environments. Despite its popularity, little is known about how quantization affects the correctness and security of LLM-generated software dependencies while generating shell commands for package installation. In this work, we conduct the first systematic empirical study of the impact of quantization on package hallucination and vulnerability risks in LLM-generated Go packages. We evaluate five Qwen model sizes under full-precision, 8-bit, and 4-bit quantization across three datasets (SO, MBPP, and paraphrase). Our results show that quantization substantially increases the package hallucination rate (PHR), with 4-bit models exhibiting the most severe degradation. We further find that even among the correctly generated packages, the vulnerability presence rate (VPR) rises as precision decreases, indicating elevated security risk in lower-precision models. Finally, our analysis of hallucinated outputs reveals that most fabricated packages resemble realistic URL-based Go module paths, such as most commonly malformed or non-existent GitHub and golang.org repositories, highlighting a systematic pattern in how LLMs hallucinate dependencies. Overall, our findings provide actionable insights into the reliability and security implications of deploying quantized LLMs for code generation and dependency recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08211v1">MobileFineTuner: A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 15 pages, 9 figures, submitted to Mobisys 2026
    </div>
    <details class="paper-abstract">
      Mobile phones are the most ubiquitous end devices, generating vast amounts of human-authored data and serving as the primary platform for end-side applications. As high-quality public data for large language models (LLMs) approaches exhaustion, on-device fine-tuning provides an opportunity to leverage private user data while preserving privacy. However, existing approaches are predominantly simulation-based or rely on IoT devices and PCs, leaving commodity mobile phones largely unexplored. A key gap is the absence of an open-source framework that enables practical LLM fine-tuning on mobile phones. We present MobileFineTuner, a unified open-source framework that enables end-to-end LLM fine-tuning directly on commodity mobile phones. MobileFineTuner is designed for efficiency, scalability, and usability, supporting full-parameters fine-tuning (Full-FT) and parameter-efficient fine-tuning (PEFT). To address the memory and energy limitations inherent to mobile phones, we introduce system-level optimizations including parameter sharding, gradient accumulation, and energy-aware computation scheduling. We demonstrate the practicality of MobileFineTuner by fine-tuning GPT-2, Gemma 3, and Qwen 2.5 on real mobile phones. Extensive experiments and ablation studies validate the effectiveness of the proposed optimizations and establish MobileFineTuner as a viable foundation for future research on on-device LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09199v1">LLMs for Analog Circuit Design Continuum (ACDC)</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and transformer architectures have shown impressive reasoning and generation capabilities across diverse natural language tasks. However, their reliability and robustness in real-world engineering domains remain largely unexplored, limiting their practical utility in human-centric workflows. In this work, we investigate the applicability and consistency of LLMs for analog circuit design -- a task requiring domain-specific reasoning, adherence to physical constraints, and structured representations -- focusing on AI-assisted design where humans remain in the loop. We study how different data representations influence model behavior and compare smaller models (e.g., T5, GPT-2) with larger foundation models (e.g., Mistral-7B, GPT-oss-20B) under varying training conditions. Our results highlight key reliability challenges, including sensitivity to data format, instability in generated designs, and limited generalization to unseen circuit configurations. These findings provide early evidence on the limits and potential of LLMs as tools to enhance human capabilities in complex engineering tasks, offering insights into designing reliable, deployable foundation models for structured, real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09187v1">WOLF: Werewolf-based Observations for LLM Deception and Falsehoods</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Spotlight Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Deception is a fundamental challenge for multi-agent reasoning: effective systems must strategically conceal information while detecting misleading behavior in others. Yet most evaluations reduce deception to static classification, ignoring the interactive, adversarial, and longitudinal nature of real deceptive dynamics. Large language models (LLMs) can deceive convincingly but remain weak at detecting deception in peers. We present WOLF, a multi-agent social deduction benchmark based on Werewolf that enables separable measurement of deception production and detection. WOLF embeds role-grounded agents (Villager, Werewolf, Seer, Doctor) in a programmable LangGraph state machine with strict night-day cycles, debate turns, and majority voting. Every statement is a distinct analysis unit, with self-assessed honesty from speakers and peer-rated deceptiveness from others. Deception is categorized via a standardized taxonomy (omission, distortion, fabrication, misdirection), while suspicion scores are longitudinally smoothed to capture both immediate judgments and evolving trust dynamics. Structured logs preserve prompts, outputs, and state transitions for full reproducibility. Across 7,320 statements and 100 runs, Werewolves produce deceptive statements in 31% of turns, while peer detection achieves 71-73% precision with ~52% overall accuracy. Precision is higher for identifying Werewolves, though false positives occur against Villagers. Suspicion toward Werewolves rises from ~52% to over 60% across rounds, while suspicion toward Villagers and the Doctor stabilizes near 44-46%. This divergence shows that extended interaction improves recall against liars without compounding errors against truthful roles. WOLF moves deception evaluation beyond static datasets, offering a dynamic, controlled testbed for measuring deceptive and detective capacity in adversarial multi-agent interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13557v6">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 new version
    </div>
    <details class="paper-abstract">
      Coarse-Grained Reconfigurable Arrays (CGRAs) offer high performance and energy efficiency across domains, yet design remains difficult due to a vast, interdependent space and costly manual iteration. We present MACO, an open-source multi-agent LLM framework for CGRA HW/SW co-design. MACO generates and refines architectures through four stages: HW/SW Co-design, Design Error Correction, Best-Design Selection, and Evaluation & Feedback, iteratively improves power, performance, and area (PPA) via agent reasoning and closed-loop feedback. To traverse the space efficiently, we introduce exponentially decaying exploration; to accelerate convergence, we incorporate an LLM self-learning mechanism that adaptively selects promising candidate CGRAs. In addition, we propose a rule-based mechanism to correct CGRA design errors. Evaluated against other state-of-the-art methods, MACO achieves superior PPA while substantially reducing human effort, highlighting the promise of LLM-driven automation for practical CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09117v1">A Categorical Analysis of Large Language Models and Why LLMs Circumvent the Symbol Grounding Problem</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      This paper presents a formal, categorical framework for analysing how humans and large language models (LLMs) transform content into truth-evaluated propositions about a state space of possible worlds W , in order to argue that LLMs do not solve but circumvent the symbol grounding problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09108v1">Evolving Excellence: Automated Optimization of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies. We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications. We evaluate ARTEMIS on four representative agent systems: the \emph{ALE Agent} for competitive programming on AtCoder Heuristic Contest, achieving a \textbf{$13.6\%$ improvement} in acceptance rate; the \emph{Mini-SWE Agent} for code optimization on SWE-Perf, with a statistically significant \textbf{10.1\% performance gain}; and the \emph{CrewAI Agent} for cost and mathematical reasoning on Math Odyssey, achieving a statistically significant \textbf{$36.9\%$ reduction} in the number of tokens required for evaluation. We also evaluate the \emph{MathTales-Teacher Agent} powered by a smaller open-source model (Qwen2.5-7B) on GSM8K primary-level mathematics problems, achieving a \textbf{22\% accuracy improvement} and demonstrating that ARTEMIS can optimize agents based on both commercial and local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09088v1">Calibrated Trust in Dealing with LLM Hallucinations: A Qualitative Study</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Hallucinations are outputs by Large Language Models (LLMs) that are factually incorrect yet appear plausible [1]. This paper investigates how such hallucinations influence users' trust in LLMs and users' interaction with LLMs. To explore this in everyday use, we conducted a qualitative study with 192 participants. Our findings show that hallucinations do not result in blanket mistrust but instead lead to context-sensitive trust calibration. Building on the calibrated trust model by Lee & See [2] and Afroogh et al.'s trust-related factors [3], we confirm expectancy [3], [4], prior experience [3], [4], [5], and user expertise & domain knowledge [3], [4] as userrelated (human) trust factors, and identify intuition as an additional factor relevant for hallucination detection. Additionally, we found that trust dynamics are further influenced by contextual factors, particularly perceived risk [3] and decision stakes [6]. Consequently, we validate the recursive trust calibration process proposed by Blöbaum [7] and extend it by including intuition as a user-related trust factor. Based on these insights, we propose practical recommendations for responsible and reflective LLM use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08998v1">DermETAS-SNA LLM: A Dermatology Focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM Assistant</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Our work introduces the DermETAS-SNA LLM Assistant that integrates Dermatology-focused Evolutionary Transformer Architecture Search with StackNet Augmented LLM. The assistant dynamically learns skin-disease classifiers and provides medically informed descriptions to facilitate clinician-patient interpretation. Contributions include: (1) Developed an ETAS framework on the SKINCON dataset to optimize a Vision Transformer (ViT) tailored for dermatological feature representation and then fine-tuned binary classifiers for each of the 23 skin disease categories in the DermNet dataset to enhance classification performance; (2) Designed a StackNet architecture that integrates multiple fine-tuned binary ViT classifiers to enhance predictive robustness and mitigate class imbalance issues; (3) Implemented a RAG pipeline, termed Diagnostic Explanation and Retrieval Model for Dermatology, which harnesses the capabilities of the Google Gemini 2.5 Pro LLM architecture to generate personalized, contextually informed diagnostic descriptions and explanations for patients, leveraging a repository of verified dermatological materials; (4) Performed extensive experimental evaluations on 23 skin disease categories to demonstrate performance increase, achieving an overall F1-score of 56.30% that surpasses SkinGPT-4 (48.51%) by a considerable margin, representing a performance increase of 16.06%; (5) Conducted a domain-expert evaluation, with eight licensed medical doctors, of the clinical responses generated by our AI assistant for seven dermatological conditions. Our results show a 92% agreement rate with the assessments provided by our AI assistant (6) Created a proof-of-concept prototype that fully integrates our DermETAS-SNA LLM into our AI assistant to demonstrate its practical feasibility for real-world clinical and educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v7">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08875v1">When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08870v1">Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      LLM agents are widely deployed in complex interactive tasks, yet privacy constraints often preclude centralized optimization and co-evolution across dynamic environments. While Federated Learning (FL) has proven effective on static datasets, its extension to the open-ended self-evolution of agents remains underexplored. Directly applying standard FL is challenging: heterogeneous tasks and sparse, trajectory-level rewards introduce severe gradient conflicts, destabilizing the global optimization process. To bridge this gap, we propose Fed-SE, a Federated Self-Evolution framework for LLM agents. Fed-SE establishes a local evolution-global aggregation paradigm. Locally, agents employ parameter-efficient fine-tuning on filtered, high-return trajectories to achieve stable gradient updates. Globally, Fed-SE aggregates updates within a low-rank subspace that disentangles environment-specific dynamics, effectively reducing negative transfer across clients. Experiments across five heterogeneous environments demonstrate that Fed-SE improves average task success rates by approximately 18% over federated baselines, validating its effectiveness in robust cross-environment knowledge transfer in privacy-constrained deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08814v1">Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Understanding human personality is crucial for web applications such as personalized recommendation and mental health assessment. Existing studies on personality detection predominantly adopt a "posts -> user vector -> labels" modeling paradigm, which encodes social media posts into user representations for predicting personality labels (e.g., MBTI labels). While recent advances in large language models (LLMs) have improved text encoding capacities, these approaches remain constrained by limited supervision signals due to label scarcity, and under-specified semantic mappings between user language and abstract psychological constructs. We address these challenges by proposing ROME, a novel framework that explicitly injects psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME leverages LLMs' role-play capability to simulate user responses to validated psychometric questionnaires. These generated question-level answers transform free-form user posts into interpretable, questionnaire-grounded evidence linking linguistic cues to personality labels, thereby providing rich intermediate supervision to mitigate label scarcity while offering a semantic reasoning chain that guides and simplifies the text-to-personality mapping learning. A question-conditioned Mixture-of-Experts module then jointly routes over post and question representations, learning to answer questionnaire items under explicit supervision. The predicted answers are summarized into an interpretable answer vector and fused with the user representation for final prediction within a multi-task learning framework, where question answering serves as a powerful auxiliary task for personality detection. Extensive experiments on two real-world datasets demonstrate that ROME consistently outperforms state-of-the-art baselines, achieving improvements (15.41% on Kaggle dataset).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v3">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08810v1">Multicalibration for LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted at AI-SQE 2026 (The 1st International Workshop on AI for Software Quality Evaluation: Judgment, Metrics, Benchmarks, and Beyond)
    </div>
    <details class="paper-abstract">
      As AI-based code generation becomes widespread, researchers are investigating the calibration of code LLMs - ensuring their confidence scores faithfully represent the true likelihood of code correctness. To do so, we investigate multicalibration, which can capture additional factors about a coding problem, such as complexity, code length, or programming language used. We study four multicalibration approaches on three function synthesis benchmarks, using latest-generation code LLMs (Qwen3 Coder, GPT-OSS, DeepSeek-R1-Distill). Our results demonstrate that multicalibration can yield distinct improvements over both uncalibrated token likelihoods (+1.03 in skill score) and baseline calibrations (+0.37 in skill score). We study the influence of the aforementioned factors in ablations, and make our dataset (consisting of code generations, likelihoods, and correctness labels) available for future research on code LLM calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08786v1">A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08764v1">Financial News Summarization: Can extractive methods still offer a true alternative to LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 8 pages, 5 tables
    </div>
    <details class="paper-abstract">
      Financial markets change rapidly due to news, economic shifts, and geopolitical events. Quick reactions are vital for investors to avoid losses or capture short-term gains. As a result, concise financial news summaries are critical for decision-making. With over 50,000 financial articles published daily, automation in summarization is necessary. This study evaluates a range of summarization methods, from simple extractive techniques to advanced large language models (LLMs), using the FinLLMs Challenge dataset. LLMs generated more coherent and informative summaries, but they are resource-intensive and prone to hallucinations, which can introduce significant errors into financial summaries. In contrast, extractive methods perform well on short, well-structured texts and offer a more efficient alternative for this type of article. The best ROUGE results come from fine-tuned LLM model like FT-Mistral-7B, although our data corpus has limited reliability, which calls for cautious interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08706v1">RESTifAI: LLM-Based Workflow for Reusable REST API Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted for ICSE 2026 Demonstration track
    </div>
    <details class="paper-abstract">
      With this paper, we introduce RESTifAI, an LLM-driven approach for generating reusable, CI/CD ready REST API tests, following the happy-path approach. Unlike existing tools that often focus primarily on internal server errors, RESTifAI systematically constructs valid test scenarios (happy paths) and derives negative cases to verify both intended functionality (2xx responses) and robustness against invalid inputs or business-rule violations (4xx responses). The results indicate that RESTifAI performs on par with the latest LLM tools, i.e., AutoRestTest and LogiAgent, while addressing limitations related to reusability, oracle complexity, and integration. To support this, we provide common comparative results and demonstrate the tool's applicability in industrial services. For tool demonstration, please refer to https://www.youtube.com/watch?v=2vtQo0T0Lo4. RESTifAI is publicly available at https://github.com/casablancahotelsoftware/RESTifAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.02943v6">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is labor-intensive, especially for strongly typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in LLMs have enabled oracle generation from natural language descriptions, aligning better with user requirements. However, existing LLM-based methods often require fine-tuning or rely on external tools such as EvoSuite for test prefix generation, making them costly or cumbersome to apply in practice. In this work, we propose CANDOR, a novel prompt engineering-based LLM framework for automated unit test generation in Java. CANDOR orchestrates multiple specialized LLM agents to collaboratively generate complete tests. To mitigate the notorious hallucinations in LLMs and improve oracle correctness, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generates accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments show that CANDOR is comparable with EvoSuite in generating tests with high code coverage and clearly superior in terms of mutation score. Moreover, our prompt engineering-based approach CANDOR significantly outperforms the SOTA fine-tuning-based oracle generator TOGLL by at least 21.1 percentage points in oracle correctness on both correct and faulty source code. Further ablation studies confirm the critical contributions of key agents in generating high-quality tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11599v2">SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v23">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.15378v6">Improving LLM Reliability with RAG in Religious Question-Answering: MufassirQAS</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Religious teachings can sometimes be complex and challenging to grasp, but chatbots can serve as effective assistants in this domain. Large Language Model (LLM) based chatbots, powered by Natural Language Processing (NLP), can connect related topics and provide well-supported responses to intricate questions, making them valuable tools for religious education. However, LLMs are prone to hallucinations as they can generate inaccurate or irrelevant information, and these can include sensitive content that could be offensive, inappropriate, or controversial. Addressing such topics without inadvertently promoting hate speech or disrespecting certain beliefs remains a significant challenge. As a solution to these issues, we introduce MufassirQAS, a system that enhances LLM accuracy and transparency using a vector database-driven Retrieval-Augmented Generation (RAG) approach. We built a dataset comprising fundamental books containing Turkish translations and interpretations of Islamic texts. This database is leveraged to answer religious inquiries while ensuring that responses remain reliable and contextually grounded. Our system also presents the relevant dataset sections alongside the LLM-generated answers, reinforcing transparency. We carefully designed system prompts to prevent harmful, offensive, or disrespectful outputs, ensuring that responses align with ethical and respectful discourse. Moreover, MufassirQAS provides supplementary details, such as source page numbers and referenced articles, to enhance credibility. To evaluate its effectiveness, we tested MufassirQAS against ChatGPT with sensitive questions, and our system demonstrated superior performance in maintaining accuracy and reliability. Future work will focus on improving accuracy and refining prompt engineering techniques to further minimize biases and ensure even more reliable responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06749v2">DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems are challenging to debug because failures often arise from long, branching interaction traces. The prevailing practice is to leverage LLMs for log-based failure localization, attributing errors to a specific agent and step. However, this paradigm has two key limitations: (i) log-only debugging lacks validation, producing untested hypotheses, and (ii) single-step or single-agent attribution is often ill-posed, as we find that multiple distinct interventions can independently repair the failed task. To address the first limitation, we introduce DoVer, an intervention-driven debugging framework, which augments hypothesis generation with active verification through targeted interventions (e.g., editing messages, altering plans). For the second limitation, rather than evaluating on attribution accuracy, we focus on measuring whether the system resolves the failure or makes quantifiable progress toward task success, reflecting a more outcome-oriented view of debugging. Within the Magnetic-One agent framework, on the datasets derived from GAIA and AssistantBench, DoVer flips 18-28% of failed trials into successes, achieves up to 16% milestone progress, and validates or refutes 30-60% of failure hypotheses. DoVer also performs effectively on a different dataset (GSMPlus) and agent framework (AG2), where it recovers 49% of failed trials. These results highlight intervention as a practical mechanism for improving reliability in agentic systems and open opportunities for more robust, scalable debugging methods for LLM-based multi-agent systems. Project website and code will be available at https://aka.ms/DoVer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03291v2">UniPruning: Unifying Local Metric and Global Feedback for Scalable Sparse LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance across diverse tasks but face prohibitive computational and memory costs. Pruning offers a promising path by inducing sparsity while preserving architectural flexibility. However, existing methods struggle to balance efficiency and robustness: local metric approaches prune layer by layer but often collapse under high sparsity, whereas global feedback methods enforce consistency at the cost of expensive weight updates or restrictive semi-structured formats. We present UniPruning, a unified post-training pruning framework that combines the speed of local saliency metrics with the stability of global coordination, enabled by a mirror descent based optimization, all without updating model weights. UniPruning leverages fast layer-wise scoring and a lightweight global controller to allocate a single sparsity budget, supporting both unstructured and semi-structured N :M pruning within one framework. After a brief calibration, it can generate pruning masks for arbitrary sparsity levels in one shot, and adapts seamlessly to hardware-aware constraints. Extensive experiments on multiple pretrained LLM families and standard benchmarks show that UniPruning consistently delivers competitive or superior perplexity and zero-shot accuracy. Ablation studies further highlight the importance of mirror descent and local saliency anchoring. Overall, UniPruning provides an efficient, principled, and scalable solution for sparsifying large-scale LLMs. Our code is available at: https://github.com/RainbowQTT/UniPruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08536v1">Principles2Plan: LLM-Guided System for Operationalising Ethical Principles into Plans</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Ethical awareness is critical for robots operating in human environments, yet existing automated planning tools provide little support. Manually specifying ethical rules is labour-intensive and highly context-specific. We present Principles2Plan, an interactive research prototype demonstrating how a human and a Large Language Model (LLM) can collaborate to produce context-sensitive ethical rules and guide automated planning. A domain expert provides the planning domain, problem details, and relevant high-level principles such as beneficence and privacy. The system generates operationalisable ethical rules consistent with these principles, which the user can review, prioritise, and supply to a planner to produce ethically-informed plans. To our knowledge, no prior system supports users in generating principle-grounded rules for classical planning contexts. Principles2Plan showcases the potential of human-LLM collaboration for making ethical automated planning more practical and feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16809v2">When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Accepted to ICSE 2026 (RECODE workshop)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08493v1">LLM-based Vulnerable Code Augmentation: Generate or Refactor?</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 6 pages, Submitted to ESAAN 2026, Under pier review
    </div>
    <details class="paper-abstract">
      Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented CWEs. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a vulnerability classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24319v2">Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08476v1">A Multi-Agent LLM Framework for Design Space Exploration in Autonomous Driving Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-09
    </div>
    <details class="paper-abstract">
      Designing autonomous driving systems requires efficient exploration of large hardware/software configuration spaces under diverse environmental conditions, e.g., with varying traffic, weather, and road layouts. Traditional design space exploration (DSE) approaches struggle with multi-modal execution outputs and complex performance trade-offs, and often require human involvement to assess correctness based on execution outputs. This paper presents a multi-agent, large language model (LLM)-based DSE framework, which integrates multi-modal reasoning with 3D simulation and profiling tools to automate the interpretation of execution outputs and guide the exploration of system designs. Specialized LLM agents are leveraged to handle user input interpretation, design point generation, execution orchestration, and analysis of both visual and textual execution outputs, which enables identification of potential bottlenecks without human intervention. A prototype implementation is developed and evaluated on a robotaxi case study (an SAE Level 4 autonomous driving application). Compared with a genetic algorithm baseline, the proposed framework identifies more Pareto-optimal, cost-efficient solutions with reduced navigation time under the same exploration budget. Experimental results also demonstrate the efficiency of the adoption of the LLM-based approach for DSE. We believe that this framework paves the way to the design automation of autonomous driving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20781v3">Using LLMs in Generating Design Rationale for Software Architecture Decisions</a></div>
    <div class="paper-meta">
      📅 2025-12-09
      | 💬 Preprint accepted for publication in ACM Transactions on Software Engineering and Methodology (TOSEM), 2025
    </div>
    <details class="paper-abstract">
      Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. To further understand the trustworthiness and applicability of LLM-generated DR in practice, we conducted semi-structured interviews with six practitioners. Based on the experimental and interview results, we discussed the pros and cons of the three prompting strategies, the strengths and limitations of LLM-generated DR, and the implications for the practical use of LLM-generated DR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07462v1">Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07454v1">Persian-Phi: Efficient Cross-Lingual Adaptation of Compact LLMs via Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      The democratization of AI is currently hindered by the immense computational costs required to train Large Language Models (LLMs) for low-resource languages. This paper presents Persian-Phi, a 3.8B parameter model that challenges the assumption that robust multilingual capabilities require massive model sizes or multilingual baselines. We demonstrate how Microsoft Phi-3 Mini -- originally a monolingual English model -- can be effectively adapted to Persian through a novel, resource-efficient curriculum learning pipeline. Our approach employs a unique "warm-up" stage using bilingual narratives (Tiny Stories) to align embeddings prior to heavy training, followed by continual pretraining and instruction tuning via Parameter-Efficient Fine-Tuning (PEFT). Despite its compact size, Persian-Phi achieves competitive results on Open Persian LLM Leaderboard in HuggingFace. Our findings provide a validated, scalable framework for extending the reach of state-of-the-art LLMs to underrepresented languages with minimal hardware resources. The Persian-Phi model is publicly available at https://huggingface.co/amirakhlaghiqqq/PersianPhi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v1">Do LLMs Trust the Code They Write?</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07375v1">LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples</a></div>
    <div class="paper-meta">
      📅 2025-12-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess vast knowledge acquired from extensive training corpora, but they often cannot remove specific pieces of information when needed, which makes it hard to handle privacy, bias mitigation, and knowledge correction. Traditional model unlearning approaches require computationally expensive fine-tuning or direct weight editing, making them impractical for real-world deployment. In this work, we introduce LoRA-based Unlearning with Negative Examples (LUNE), a lightweight framework that performs negative-only unlearning by updating only low-rank adapters while freezing the backbone, thereby localizing edits and avoiding disruptive global changes. Leveraging Low-Rank Adaptation (LoRA), LUNE targets intermediate representations to suppress (or replace) requested knowledge with an order-of-magnitude lower compute and memory than full fine-tuning or direct weight editing. Extensive experiments on multiple factual unlearning tasks show that LUNE: (I) achieves effectiveness comparable to full fine-tuning and memory-editing methods, and (II) reduces computational cost by about an order of magnitude.
    </details>
</div>
