# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07943v2">Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted to AAAI 2026. Extended version with full Appendix
    </div>
    <details class="paper-abstract">
      Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at https://github.com/OpenSPG/KAG-Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15980v2">Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by IUI'25 Code & Demo: https://github.com/magic-YuanTian/SQLsynth
    </div>
    <details class="paper-abstract">
      Text-to-SQL models, which parse natural language (NL) questions to executable SQL queries, are increasingly adopted in real-world applications. However, deploying such models in the real world often requires adapting them to the highly specialized database schemas used in specific applications. We find that existing text-to-SQL models experience significant performance drops when applied to new schemas, primarily due to the lack of domain-specific data for fine-tuning. This data scarcity also limits the ability to effectively evaluate model performance in new domains. Continuously obtaining high-quality text-to-SQL data for evolving schemas is prohibitively expensive in real-world scenarios. To bridge this gap, we propose SQLsynth, a human-in-the-loop text-to-SQL data annotation system. SQLsynth streamlines the creation of high-quality text-to-SQL datasets through human-LLM collaboration in a structured workflow. A within-subjects user study comparing SQLsynth with manual annotation and ChatGPT shows that SQLsynth significantly accelerates text-to-SQL data annotation, reduces cognitive load, and produces datasets that are more accurate, natural, and diverse. Our code is available at https://github.com/magic-YuanTian/SQLsynth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10948v1">DEFT-LLM: Disentangled Expert Feature Tuning for Micro-Expression Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Micro expression recognition (MER) is crucial for inferring genuine emotion. Applying a multimodal large language model (MLLM) to this task enables spatio-temporal analysis of facial motion and provides interpretable descriptions. However, there are still two core challenges: (1) The entanglement of static appearance and dynamic motion cues prevents the model from focusing on subtle motion; (2) Textual labels in existing MER datasets do not fully correspond to underlying facial muscle movements, creating a semantic gap between text supervision and physical motion. To address these issues, we propose DEFT-LLM, which achieves motion semantic alignment by multi-expert disentanglement. We first introduce Uni-MER, a motion-driven instruction dataset designed to align text with local facial motion. Its construction leverages dual constraints from optical flow and Action Unit (AU) labels to ensure spatio-temporal consistency and reasonable correspondence to the movements. We then design an architecture with three experts to decouple facial dynamics into independent and interpretable representations (structure, dynamic textures, and motion-semantics). By integrating the instruction-aligned knowledge from Uni-MER into DEFT-LLM, our method injects effective physical priors for micro expressions while also leveraging the cross modal reasoning ability of large language models, thus enabling precise capture of subtle emotional cues. Experiments on multiple challenging MER benchmarks demonstrate state-of-the-art performance, as well as a particular advantage in interpretable modeling of local facial motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10067v2">Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at https://muser-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10890v1">LLM enhanced graph inference for long-term disease progression modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15793v2">Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026, camera ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in applications that require processing information from heterogeneous formats, including texts, tables, infoboxes, and knowledge graphs. However, systematic biases toward particular formats may undermine LLMs' ability to integrate heterogeneous data impartially, potentially resulting in reasoning errors and increased risks in downstream tasks. Yet it remains unclear whether such biases are systematic, which data-level factors drive them, and what internal mechanisms underlie their emergence. In this paper, we present the first comprehensive study of format bias in LLMs through a three-stage empirical analysis. The first stage explores the presence and direction of bias across a diverse range of LLMs. The second stage examines how key data-level factors influence these biases. The third stage analyzes how format bias emerges within LLMs' attention patterns and evaluates a lightweight intervention to test its effectiveness. Our results show that format bias is consistent across model families, driven by information richness, structure quality, and representation type, and is closely associated with attention imbalance within the LLMs. Based on these investigations, we identify three future research directions to reduce format bias: enhancing data pre-processing through format repair and normalization, introducing inference-time interventions such as attention re-weighting, and developing format-balanced training corpora. These directions will support the design of more robust and fair heterogeneous data processing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10871v1">From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 11 pages, 3 figures. Under review at IWSDS 2026
    </div>
    <details class="paper-abstract">
      LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10868v1">Go-UT-Bench: A Fine-Tuning Dataset for LLM-Based Unit Test Generation in Go</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Training data imbalance poses a major challenge for code LLMs. Most available data heavily over represents raw opensource code while underrepresenting broader software engineering tasks, especially in low resource languages like Golang. As a result, models excel at code autocompletion but struggle with real world developer workflows such as unit test generation. To address this gap, we introduce GO UT Bench, a benchmark dataset of 5264 pairs of code and unit tests, drawn from 10 permissively licensed Golang repositories spanning diverse domain. We evaluate its effectiveness as a fine tuning dataset across two LLM families i.e. mixture of experts and dense decoders. Our results show that finetuned models outperform their base counterparts on more than 75% of benchmark tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10866v1">Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 5 pages, 2 figures. Accepted paper for the IEIE (Institute of Electronics and Information Engineers) Fall Conference 2025. Presentation on Nov 27, 2025
    </div>
    <details class="paper-abstract">
      This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10865v1">Towards a Human-in-the-Loop Framework for Reliable Patch Evaluation Using an LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Reliable evaluation is crucial for advancing Automated Program Repair (APR), but prevailing benchmarks rely on execution-based evaluation methods (unit test pass@k), which fail to capture true patch validity. Determining validity can require costly manual annotation. To reduce this cost, we introduce a human-in-the-loop approach to LLM-based patch validity judgment. Inspired by the observation that human judgment is better aligned when using a shared rubric, we first employ an LLM to generate a per-bug rubric, followed by a one-time human review and optional refinement to this rubric, and then employ an LLM to judge patches using the refined rubric. We apply this approach to assign binary validity labels to patches for issues found by Google sanitizer tools. Our results show that this approach yields substantial agreement with human consensus (Cohen's kappa 0.75), high recall (0.94) and high precision (0.80), when considering patches that have unanimous agreement from 3 human raters on the validity labels. On the full dataset including patches where human raters disagree, we find this approach can still be further improved (Cohen's kappa 0.57, recall 0.93, precision 0.65) and identify possible future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11946v1">Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Knowledge graph-based dialogue generation (KG-DG) is a challenging task requiring models to effectively incorporate external knowledge into conversational responses. While large language models (LLMs) have achieved impressive results across various NLP tasks, their ability to utilize external knowledge in KG-DG remains under-explored. We observe that LLMs often rely on internal knowledge, leading to detachment from provided knowledge graphs, even when they are given a flawlessly retrieved knowledge graph. First, we introduce LLM-KAT, an evaluation procedure for measuring knowledge attachment in generated responses. Second, we propose a simple yet effective entity anonymization technique to encourage LLMs to better leverage external knowledge. Experiments on the OpenDialKG dataset demonstrate that our approach improves LLMs' attachment on external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11916v1">An Analysis of Architectural Impact on LLM-based Abstract Visual Reasoning: A Systematic Benchmark on RAVEN-FAIR</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 23 pages, 9 figures
    </div>
    <details class="paper-abstract">
      This study aims to systematically evaluate the performance of large language models (LLMs) in abstract visual reasoning problems. We examined four LLM models (GPT-4.1-Mini, Claude-3.5-Haiku, Gemini-1.5-Flash, Llama-3.3-70b) utilizing four different reasoning architectures (single-shot, embedding-controlled repetition, self-reflection, and multi-agent) on the RAVEN-FAIR dataset. Visual responses generated through a three-stage process (JSON extraction, LLM reasoning, and Tool Function) were evaluated using SSIM and LPIPS metrics; Chain-of-Thought scores and error types (semantic hallucination, numeric misperception) were analyzed. Results demonstrate that GPT-4.1-Mini consistently achieved the highest overall accuracy across all architectures, indicating a strong reasoning capability. While the multi-agent architecture occasionally altered semantic and numeric balance across models, these effects were not uniformly beneficial. Instead, each model exhibited distinct sensitivity patterns to architectural design, underscoring that reasoning effectiveness remains model-specific. Variations in response coverage further emerged as a confounding factor that complicates direct cross-architecture comparison. To estimate the upper-bound performance of each configuration, we report the best of five independent runs, representing a best-case scenario rather than an averaged outcome. This multi-run strategy aligns with recent recommendations, which emphasize that single-run evaluations are fragile and may lead to unreliable conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11914v1">Forgetting-MarI: LLM Unlearning via Marginal Information Regularization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11896v1">VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The widespread reliance on open-source software dramatically increases the risk of vulnerability exploitation, underscoring the need for effective and scalable vulnerability detection (VD). Existing VD techniques, whether traditional machine learning-based or LLM-based approaches like prompt engineering, supervised fine-tuning, or off-policy preference optimization, remain fundamentally limited in their ability to perform context-aware analysis: They depend on fixed inputs or static preference datasets, cannot adaptively explore repository-level dependencies, and are constrained by function-level benchmarks that overlook critical vulnerability context. This paper introduces Vulnerability-Adaptive Policy Optimization (VULPO), an on-policy LLM reinforcement learning framework for context-aware VD. To support training and evaluation, we first construct ContextVul, a new dataset that augments high-quality function-level samples with lightweight method to extract repository-level context information. We then design multi-dimensional reward structuring that jointly captures prediction correctness, vulnerability localization accuracy, and the semantic relevance of vulnerability analysis, thereby guiding the model toward comprehensive contextual reasoning. To address the asymmetric difficulty of different vulnerability cases and mitigate reward hacking, VULPO incorporates label-level and sample-level difficulty-adaptive reward scaling, encouraging the model to explore challenging cases while maintaining balanced reward distribution. Extensive experiments demonstrate the superiority of our VULPO framework in context-aware VD: Our VULPO-4B substantially outperforms existing VD baselines based on prompt engineering and off-policy optimization, improving F1 by 85% over Qwen3-4B and achieving performance comparable to a 150x larger-scale model, DeepSeek-R1-0528.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11885v1">Flash-Fusion: Enabling Expressive, Low-Latency Queries on IoT Sensor Streams with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 12 pages, 5 figures. Under review
    </div>
    <details class="paper-abstract">
      Smart cities and pervasive IoT deployments have generated interest in IoT data analysis across transportation and urban planning. At the same time, Large Language Models offer a new interface for exploring IoT data - particularly through natural language. Users today face two key challenges when working with IoT data using LLMs: (1) data collection infrastructure is expensive, producing terabytes of low-level sensor readings that are too granular for direct use, and (2) data analysis is slow, requiring iterative effort and technical expertise. Directly feeding all IoT telemetry to LLMs is impractical due to finite context windows, prohibitive token costs at scale, and non-interactive latencies. What is missing is a system that first parses a user's query to identify the analytical task, then selects the relevant data slices, and finally chooses the right representation before invoking an LLM. We present Flash-Fusion, an end-to-end edge-cloud system that reduces the IoT data collection and analysis burden on users. Two principles guide its design: (1) edge-based statistical summarization (achieving 73.5% data reduction) to address data volume, and (2) cloud-based query planning that clusters behavioral data and assembles context-rich prompts to address data interpretation. We deploy Flash-Fusion on a university bus fleet and evaluate it against a baseline that feeds raw data to a state-of-the-art LLM. Flash-Fusion achieves a 95% latency reduction and 98% decrease in token usage and cost while maintaining high-quality responses. It enables personas across disciplines - safety officers, urban planners, fleet managers, and data scientists - to efficiently iterate over IoT data without the burden of manual query authoring or preprocessing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11881v1">Better LLM Reasoning via Dual-Play</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11867v1">Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Submitted to LREC 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown considerable promise in clinical natural language processing, yet few domain-specific datasets exist to rigorously evaluate their performance on radiology tasks. In this work, we introduce an annotated corpus of 6,393 radiology reports from 586 patients, each labeled for follow-up imaging status, to support the development and benchmarking of follow-up adherence detection systems. Using this corpus, we systematically compared traditional machine-learning classifiers, including logistic regression (LR), support vector machines (SVM), Longformer, and a fully fine-tuned Llama3-8B-Instruct, with recent generative LLMs. To evaluate generative LLMs, we tested GPT-4o and the open-source GPT-OSS-20B under two configurations: a baseline (Base) and a task-optimized (Advanced) setting that focused inputs on metadata, recommendation sentences, and their surrounding context. A refined prompt for GPT-OSS-20B further improved reasoning accuracy. Performance was assessed using precision, recall, and F1 scores with 95% confidence intervals estimated via non-parametric bootstrapping. Inter-annotator agreement was high (F1 = 0.846). GPT-4o (Advanced) achieved the best performance (F1 = 0.832), followed closely by GPT-OSS-20B (Advanced; F1 = 0.828). LR and SVM also performed strongly (F1 = 0.776 and 0.775), underscoring that while LLMs approach human-level agreement through prompt optimization, interpretable and resource-efficient models remain valuable baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11829v1">Towards Autoformalization of LLM-generated Outputs for Requirement Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 To be submitted for publication
    </div>
    <details class="paper-abstract">
      Autoformalization, the process of translating informal statements into formal logic, has gained renewed interest with the emergence of powerful Large Language Models (LLMs). While LLMs show promise in generating structured outputs from natural language (NL), such as Gherkin Scenarios from NL feature requirements, there's currently no formal method to verify if these outputs are accurate. This paper takes a preliminary step toward addressing this gap by exploring the use of a simple LLM-based autoformalizer to verify LLM-generated outputs against a small set of natural language requirements. We conducted two distinct experiments. In the first one, the autoformalizer successfully identified that two differently-worded NL requirements were logically equivalent, demonstrating the pipeline's potential for consistency checks. In the second, the autoformalizer was used to identify a logical inconsistency between a given NL requirement and an LLM-generated output, highlighting its utility as a formal verification tool. Our findings, while limited, suggest that autoformalization holds significant potential for ensuring the fidelity and logical consistency of LLM-generated outputs, laying a crucial foundation for future, more extensive studies into this novel application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11828v1">Conformal Constrained Policy Optimization for Cost-Effective LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently made tremendous progress towards solving challenging AI problems, they have done so at increasingly steep computational and API costs. We propose a novel strategy where we combine multiple LLM models with varying cost/accuracy tradeoffs in an agentic manner, where models and tools are run in sequence as determined by an orchestration model to minimize cost subject to a user-specified level of reliability; this constraint is formalized using conformal prediction to provide guarantees. To solve this problem, we propose Conformal Constrained Policy Optimization (CCPO), a training paradigm that integrates constrained policy optimization with off-policy reinforcement learning and recent advances in online conformal prediction. CCPO jointly optimizes a cost-aware policy (score function) and an adaptive threshold. Across two multi-hop question answering benchmarks, CCPO achieves up to a 30% cost reduction compared to other cost-aware baselines and LLM-guided methods without compromising reliability. Our approach provides a principled and practical framework for deploying LLM agents that are significantly more cost-effective while maintaining reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v3">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11823v1">CollaClassroom: An AI-Augmented Collaborative Learning Platform with LLM Support in the Context of Bangladeshi University Students</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      CollaClassroom is an AI-enhanced platform that embeds large language models (LLMs) into both individual and group study panels to support real-time collaboration. We evaluate CollaClassroom with Bangladeshi university students (N = 12) through a small-group study session and a pre-post survey. Participants have substantial prior experience with collaborative learning and LLMs and express strong receptivity to LLM-assisted study (92% agree/strongly agree). Usability ratings are positive, including high learnability(67% "easy"), strong reliability (83% "reliable"), and low frustration (83% "not at all"). Correlational analyses show that participants who perceive the LLM as supporting equal participation also view it as a meaningful contributor to discussions (r = 0.86). Moreover, their pre-use expectations of LLM value align with post-use assessments (r = 0.61). These findings suggest that LLMs can enhance engagement and perceived learning when designed to promote equitable turn-taking and transparency across individual and shared spaces. The paper contributes an empirically grounded account of AI-mediated collaboration in a Global South higher-education context, with design implications for fairness-aware orchestration of human-AI teamwork.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11816v1">Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Full version of the paper accepted for publication at The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
    </div>
    <details class="paper-abstract">
      Due to its expressiveness and unambiguous nature, First-Order Logic (FOL) is a powerful formalism for representing concepts expressed in natural language (NL). This is useful, e.g., for specifying and verifying desired system properties. While translating FOL into human-readable English is relatively straightforward, the inverse problem, converting NL to FOL (NL-FOL translation), has remained a longstanding challenge, for both humans and machines. Although the emergence of Large Language Models (LLMs) promised a breakthrough, recent literature provides contrasting results on their ability to perform NL-FOL translation. In this work, we provide a threefold contribution. First, we critically examine existing datasets and protocols for evaluating NL-FOL translation performance, revealing key limitations that may cause a misrepresentation of LLMs' actual capabilities. Second, to overcome these shortcomings, we propose a novel evaluation protocol explicitly designed to distinguish genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination. Third, using this new approach, we show that state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11788v1">MALBO: Optimizing LLM-Based Multi-Agent Teams via Multi-Objective Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Master's Thesis, University of Milano-Bicocca, 2025
    </div>
    <details class="paper-abstract">
      The optimal assignment of Large Language Models (LLMs) to specialized roles in multi-agent systems is a significant challenge, defined by a vast combinatorial search space, expensive black-box evaluations, and an inherent trade-off between performance and cost. Current optimization methods focus on single-agent settings and lack a principled framework for this multi-agent, multi-objective problem. This thesis introduces MALBO (Multi-Agent LLM Bayesian Optimization), a systematic framework designed to automate the efficient composition of LLM-based agent teams. We formalize the assignment challenge as a multi-objective optimization problem, aiming to identify the Pareto front of configurations between task accuracy and inference cost. The methodology employs multi-objective Bayesian Optimization (MOBO) with independent Gaussian Process surrogate models. By searching over a continuous feature-space representation of the LLMs, this approach performs a sample-efficient exploration guided by the expected hypervolume improvement. The primary contribution is a principled and automated methodology that yields a Pareto front of optimal team configurations. Our results demonstrate that the Bayesian optimization phase, compared to an initial random search, maintained a comparable average performance while reducing the average configuration cost by over 45%. Furthermore, MALBO identified specialized, heterogeneous teams that achieve cost reductions of up to 65.8% compared to homogeneous baselines, all while maintaining maximum performance. The framework thus provides a data-driven tool for deploying cost-effective and highly specialized multi-agent AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11764v1">Demystify, Use, Reflect: Preparing students to be informed LLM-users</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 2 pages 1 table Submitted to SIGCSE 2026
    </div>
    <details class="paper-abstract">
      We transitioned our post-CS1 course that introduces various subfields of computer science so that it integrates Large Language Models (LLMs) in a structured, critical, and practical manner. It aims to help students develop the skills needed to engage meaningfully and responsibly with AI. The course now includes explicit instruction on how LLMs work, exposure to current tools, ethical issues, and activities that encourage student reflection on personal use of LLMs as well as the larger evolving landscape of AI-assisted programming. In class, we demonstrate the use and verification of LLM outputs, guide students in the use of LLMs as an ingredient in a larger problem-solving loop, and require students to disclose and acknowledge the nature and extent of LLM assistance. Throughout the course, we discuss risks and benefits of LLMs across CS subfields. In our first iteration of the course, we collected and analyzed data from students pre and post surveys. Student understanding of how LLMs work became more technical, and their verification and use of LLMs shifted to be more discerning and collaborative. These strategies can be used in other courses to prepare students for the AI-integrated future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13765v1">PROF: An LLM-based Reward Code Preference Optimization Framework for Offline Imitation Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Offline imitation learning (offline IL) enables training effective policies without requiring explicit reward annotations. Recent approaches attempt to estimate rewards for unlabeled datasets using a small set of expert demonstrations. However, these methods often assume that the similarity between a trajectory and an expert demonstration is positively correlated with the reward, which oversimplifies the underlying reward structure. We propose PROF, a novel framework that leverages large language models (LLMs) to generate and improve executable reward function codes from natural language descriptions and a single expert trajectory. We propose Reward Preference Ranking (RPR), a novel reward function quality assessment and ranking strategy without requiring environment interactions or RL training. RPR calculates the dominance scores of the reward functions, where higher scores indicate better alignment with expert preferences. By alternating between RPR and text-based gradient optimization, PROF fully automates the selection and refinement of optimal reward functions for downstream policy learning. Empirical results on D4RL demonstrate that PROF surpasses or matches recent strong baselines across numerous datasets and domains, highlighting the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.13812v2">The Empty Chair: Using LLMs to Raise Missing Perspectives in Policy Deliberations</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: PersonaLLM: Workshop on LLM Persona Modeling
    </div>
    <details class="paper-abstract">
      Deliberation is essential to well-functioning democracies, yet physical, economic, and social barriers often exclude certain groups, reducing representativeness and contributing to issues like group polarization. In this work, we explore the use of large language model (LLM) personas to introduce missing perspectives in policy deliberations. We develop and evaluate a tool that transcribes conversations in real-time and simulates input from relevant but absent stakeholders. We deploy this tool in a 19-person student citizens' assembly on campus sustainability. Participants and facilitators found that the tool was useful to spark new discussions and surfaced valuable perspectives they had not previously considered. However, they also raised skepticism about the ability of LLMs to accurately characterize the perspectives of different groups, especially ones that are already underrepresented. Overall, this case study highlights that while AI personas can usefully surface new perspectives and prompt discussion in deliberative settings, their successful deployment depends on clarifying their limitations and emphasizing that they complement rather than replace genuine participation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25506v2">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both other researchers and practitioners. One important step towards excelling in empirical research on LLMs and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 86 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 86 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were fit for reproduction. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v2">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10624v3">Comprehension Without Competence: Architectural Limits of LLMs in Symbolic Computation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 v2: Two TMLR revision rounds addressing reviewer feedback. Added real-world validation (3.4), interpretability analysis (7), computational hallucination framework, strengthened theory. v3: Sec 3.2 - added transformer architecture diagram, clarified UAT capacity vs computational limits, improved role specialization theorem presentation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) display striking surface fluency yet systematically fail at tasks requiring symbolic reasoning, arithmetic accuracy, and logical consistency. This paper offers a structural diagnosis of such failures, revealing a persistent gap between \textit{comprehension} and \textit{competence}. Through controlled experiments and architectural analysis, we demonstrate that LLMs often articulate correct principles without reliably applying them--a failure rooted not in knowledge access, but in computational execution. We term this phenomenon the computational \textit{split-brain syndrome}, where instruction and action pathways are geometrically and functionally dissociated. This core limitation recurs across domains, from mathematical operations to relational inferences, and explains why model behavior remains brittle even under idealized prompting. We argue that LLMs function as powerful pattern completion engines, but lack the architectural scaffolding for principled, compositional reasoning. Our findings delineate the boundary of current LLM capabilities and motivate future models with metacognitive control, principle lifting, and structurally grounded execution. This diagnosis also clarifies why mechanistic interpretability findings may reflect training-specific pattern coordination rather than universal computational principles, and why the geometric separation between instruction and execution pathways suggests limitations in neural introspection and mechanistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11356v1">SEAL: Subspace-Anchored Watermarks for LLM Ownership</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across a wide range of natural language processing tasks, demonstrating human-level performance in text generation, reasoning, and question answering. However, training such models requires substantial computational resources, large curated datasets, and sophisticated alignment procedures. As a result, they constitute highly valuable intellectual property (IP) assets that warrant robust protection mechanisms. Existing IP protection approaches suffer from critical limitations. Model fingerprinting techniques can identify model architectures but fail to establish ownership of specific model instances. In contrast, traditional backdoor-based watermarking methods embed behavioral anomalies that can be easily removed through common post-processing operations such as fine-tuning or knowledge distillation. We propose SEAL, a subspace-anchored watermarking framework that embeds multi-bit signatures directly into the model's latent representational space, supporting both white-box and black-box verification scenarios. Our approach leverages model editing techniques to align the hidden representations of selected anchor samples with predefined orthogonal bit vectors. This alignment embeds the watermark while preserving the model's original factual predictions, rendering the watermark functionally harmless and stealthy. We conduct comprehensive experiments on multiple benchmark datasets and six prominent LLMs, comparing SEAL with 11 existing fingerprinting and watermarking methods to demonstrate its superior effectiveness, fidelity, efficiency, and robustness. Furthermore, we evaluate SEAL under potential knowledgeable attacks and show that it maintains strong verification performance even when adversaries possess knowledge of the watermarking mechanism and the embedded signatures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16851v2">Interpretable LLM Guardrails via Sparse Representation Steering</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities in generation tasks but are prone to producing harmful, misleading, or biased content, posing significant ethical and safety concerns. To mitigate such risks, representation engineering, which steer model behavior toward desired attributes by injecting carefully designed steering vectors into LLM's representations at inference time, has emerged as a promising alternative to fine-tuning approaches. However, due to the semantically entangled nature of LLM's representation, existing representation engineering methods still suffer from several limitations: limited fine-grained controllability, content quality degradation, and conflict in multi-attribute control. To overcome these challenges, we propose Sparse Representation Steering (SRS), a novel framework that achieves fine-grained and interpretable control over LLM behavior by first disentangling internal activations into a sparse, semantically meaningful representation space, and then selectively steering relevant dimensions. Specifically, SRS leverages a pretrained Sparse Autoencoder (SAE) to transform dense, entangled activation patterns into a sparse monosemantic feature space. To identify relevant features, SRS contrasts sparse activations from positive and negative prompt pairs and measures their bidirectional KL divergence to locate dimensions most associated with the target attribute. We conduct comprehensive experiments on Gemma-2 series model across three alignment dimensions, i.e., safety, fairness, and truthfulness, to evaluate the effectiveness of SRS. Results show that SRS consistently outperforms existing steering methods, which achieves significantly improved controllability across both single and multiple attribute settings, while preserving high linguistic quality and general ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11306v1">iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted in AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11258v1">KGQuest: Template-Driven QA Generation from Knowledge Graphs with LLM-Based Refinement</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The generation of questions and answers (QA) from knowledge graphs (KG) plays a crucial role in the development and testing of educational platforms, dissemination tools, and large language models (LLM). However, existing approaches often struggle with scalability, linguistic quality, and factual consistency. This paper presents a scalable and deterministic pipeline for generating natural language QA from KGs, with an additional refinement step using LLMs to further enhance linguistic quality. The approach first clusters KG triplets based on their relations, creating reusable templates through natural language rules derived from the entity types of objects and relations. A module then leverages LLMs to refine these templates, improving clarity and coherence while preserving factual accuracy. Finally, the instantiation of answer options is achieved through a selection strategy that introduces distractors from the KG. Our experiments demonstrate that this hybrid approach efficiently generates high-quality QA pairs, combining scalability with fluency and linguistic precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11257v1">AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11255v1">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11252v1">UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 18 pages, 5 Figures
    </div>
    <details class="paper-abstract">
      Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11250v1">Prompt Engineering vs. Fine-Tuning for LLM-Based Vulnerability Detection in Solana and Algorand Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Smart contracts have emerged as key components within decentralized environments, enabling the automation of transactions through self-executing programs. While these innovations offer significant advantages, they also present potential drawbacks if the smart contract code is not carefully designed and implemented. This paper investigates the capability of large language models (LLMs) to detect OWASP-inspired vulnerabilities in smart contracts beyond the Ethereum Virtual Machine (EVM) ecosystem, focusing specifically on Solana and Algorand. Given the lack of labeled datasets for non-EVM platforms, we design a synthetic dataset of annotated smart contract snippets in Rust (for Solana) and PyTeal (for Algorand), structured around a vulnerability taxonomy derived from OWASP. We evaluate LLMs under three configurations: prompt engineering, fine-tuning, and a hybrid of both, comparing their performance on different vulnerability categories. Experimental results show that prompt engineering achieves general robustness, while fine-tuning improves precision and recall on less semantically rich languages such as TEAL. Additionally, we analyze how the architectural differences of Solana and Algorand influence the manifestation and detectability of vulnerabilities, offering platform-specific mappings that highlight limitations in existing security tooling. Our findings suggest that LLM-based approaches are viable for static vulnerability detection in smart contracts, provided domain-specific data and categorization are integrated into training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11248v1">T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed on customer devices. To support them, current devices are adopting SoCs (System on Chip) with NPUs (Neural Processing Unit) installed. Although high performance is expected, LLM inference on NPUs is slower than its CPU counterpart. The reason is that NPUs have poor performance on computations other than GEMM, like dequantization. Current works either disaggregate prefill on the NPUs and decoding on the CPUs, or put both on the NPUs but with an accuracy loss. To solve this issue, based on the insight that low-bit can enable target computation encoded within an acceptably sized table, we propose table lookup to subsume hardware operations otherwise unsupported. To realize this, we overcome the conflicting hardware behavior of prefill and decoding to design a unified table layout and tiling through (1) fused two-level table-based dequantization and (2) concurrency-hierarchy-guided tiling. Based on that, we implement the prefill phase by three-stage pipeline and map the table-lookup-based decoding to NPU's vector units. Results show 1.4x and 3.1x speedup for prefill and decoding respectively, and 84% energy savings compared to the baseline NPU methods. The code is available at https://github.com/microsoft/T-MAC/tree/main/t-man.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09719v3">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11125v1">Utilizing LLMs for Industrial Process Automation: A Case Study on Modifying RAPID Programs</a></div>
    <div class="paper-meta">
      📅 2025-11-14
      | 💬 Submitted to the International Conference on Software Engineering (ICSE) track Software Engineering in Practice (SEIP) 2026
    </div>
    <details class="paper-abstract">
      How to best use Large Language Models (LLMs) for software engineering is covered in many publications in recent years. However, most of this work focuses on widely-used general purpose programming languages. The utility of LLMs for software within the industrial process automation domain, with highly-specialized languages that are typically only used in proprietary contexts, is still underexplored. Within this paper, we study enterprises can achieve on their own without investing large amounts of effort into the training of models specific to the domain-specific languages that are used. We show that few-shot prompting approaches are sufficient to solve simple problems in a language that is otherwise not well-supported by an LLM and that is possible on-premise, thereby ensuring the protection of sensitive company data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11106v1">AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-14
    </div>
    <details class="paper-abstract">
      Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10850v1">Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Task arithmetic is a powerful technique for transferring skills between Large Language Models (LLMs), but it often suffers from negative interference when models have diverged during training. We address this limitation by first aligning the models' parameter spaces, leveraging the inherent permutation, rotation, and scaling symmetries of Transformer architectures. We adapt parameter space alignment for modern Grouped-Query Attention (GQA) and SwiGLU layers, exploring both weight-based and activation-based approaches. Using this alignment-first strategy, we successfully transfer advanced reasoning skills to a non-reasoning model. Experiments on challenging reasoning benchmarks show that our method consistently outperforms standard task arithmetic. This work provides an effective approach for merging and transferring specialized skills across evolving LLM families, reducing redundant fine-tuning and enhancing model adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26546v2">Towards Verified Code Reasoning by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 43 pages
    </div>
    <details class="paper-abstract">
      While LLM-based agents are able to tackle a wide variety of code reasoning questions, the answers are not always correct. This prevents the agent from being useful in situations where high precision is desired: (1) helping a software engineer understand a new code base, (2) helping a software engineer during code review sessions, and (3) ensuring that the code generated by an automated code generation system meets certain requirements (e.g. fixes a bug, improves readability, implements a feature). As a result of this lack of trustworthiness, the agent's answers need to be manually verified before they can be trusted. Manually confirming responses from a code reasoning agent requires human effort and can result in slower developer productivity, which weakens the assistance benefits of the agent. In this paper, we describe a method to automatically validate the answers provided by a code reasoning agent by verifying its reasoning steps. At a very high level, the method consists of extracting a formal representation of the agent's response and, subsequently, using formal verification and program analysis tools to verify the agent's reasoning steps. We applied this approach to a benchmark set of 20 uninitialized variable errors detected by sanitizers and 20 program equivalence queries. For the uninitialized variable errors, the formal verification step was able to validate the agent's reasoning on 13/20 examples, and for the program equivalence queries, the formal verification step successfully caught 6/8 incorrect judgments made by the agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10840v1">Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 28 pages, 35 figures, under review. Extensive supplementary materials. Code and models available at https://huggingface.co/collections/CausalNLP/multilingual-tinystories-6862b6562414eb84d183f82a and https://huggingface.co/flodraye
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) can process many languages, yet how they internally represent this diversity remains unclear. Do they form shared multilingual representations with language-specific decoding, and if so, why does performance still favor the dominant training language? To address this, we train a series of LLMs on different mixtures of multilingual data and analyze their internal mechanisms using cross-layer transcoders (CLT) and attribution graphs. Our results provide strong evidence for pivot language representations: the model employs nearly identical representations across languages, while language-specific decoding emerges in later layers. Attribution analyses reveal that decoding relies in part on a small set of high-frequency language features in the final layers, which linearly read out language identity from the first layers in the model. By intervening on these features, we can suppress one language and substitute another in the model's outputs. Finally, we study how the dominant training language influences these mechanisms across attribution graphs and decoding pathways. We argue that understanding this pivot-language mechanism is crucial for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10833v1">SURFACEBENCH: Can Self-Evolving LLMs Find the Equations of 3D Scientific Surfaces?</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Equation discovery from data is a core challenge in machine learning for science, requiring the recovery of concise symbolic expressions that govern complex physical and geometric phenomena. Recent approaches with large language models (LLMs) show promise in symbolic regression, but their success often hinges on memorized formulas or overly simplified functional forms. Existing benchmarks exacerbate this limitation: they focus on scalar functions, ignore domain grounding, and rely on brittle string-matching based metrics that fail to capture scientific equivalence. We introduce SurfaceBench, first comprehensive benchmark for symbolic surface discovery. SurfaceBench comprises 183 tasks across 15 categories of symbolic complexity, spanning explicit, implicit, and parametric equation representation forms. Each task includes ground-truth equations, variable semantics, and synthetically sampled three dimensional data. Unlike prior SR datasets, our tasks reflect surface-level structure, resist LLM memorization through novel symbolic compositions, and are grounded in scientific domains such as fluid dynamics, robotics, electromagnetics, and geometry. To evaluate equation discovery quality, we pair symbolic checks with geometry-aware metrics such as Chamfer and Hausdorff distances, capturing both algebraic fidelity and spatial reconstruction accuracy. Our experiments reveal that state-of-the-art frameworks, while occasionally successful on specific families, struggle to generalize across representation types and surface complexities. SurfaceBench thus establishes a challenging and diagnostic testbed that bridges symbolic reasoning with geometric reconstruction, enabling principled benchmarking of progress in compositional generalization, data-driven scientific induction, and geometry-aware reasoning with LLMs. We release the code here: https://github.com/Sanchit-404/surfacebench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10819v1">LLM-as-a-Grader: Practical Insights from Large Language Model for Short-Answer and Report Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored for educational tasks such as grading, yet their alignment with human evaluation in real classrooms remains underexamined. In this study, we investigate the feasibility of using an LLM (GPT-4o) to evaluate short-answer quizzes and project reports in an undergraduate Computational Linguistics course. We collect responses from approximately 50 students across five quizzes and receive project reports from 14 teams. LLM-generated scores are compared against human evaluations conducted independently by the course teaching assistants (TAs). Our results show that GPT-4o achieves strong correlation with human graders (up to 0.98) and exact score agreement in 55\% of quiz cases. For project reports, it also shows strong overall alignment with human grading, while exhibiting some variability in scoring technical, open-ended responses. We release all code and sample data to support further research on LLMs in educational assessment. This work highlights both the potential and limitations of LLM-based grading systems and contributes to advancing automated grading in real-world academic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17506v3">RAG-Enhanced Collaborative LLM Agents for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Machine Learning, Drug Discovery
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing major challenges. First, it hinders the application of more flexible general-purpose LLMs for cutting-edge drug discovery tasks. More importantly, it limits the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. Compounding these challenges is the fact that real-world scientific questions are typically complex and open-ended, requiring reasoning beyond pattern matching or static knowledge retrieval.To address these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses - all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. Our code is publicly available at https://github.com/Genentech/CLADD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14397v2">LIMINAL: Exploring The Frontiers of LLM Decode Performance</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) necessitates a deep understanding of their fundamental performance limits. This paper investigates the limits of LLM inference, focusing on hardware-imposed bottlenecks in auto-regressive decoding. We develop LIMINAL, an analytical performance model that abstracts application requirements and hardware capabilities to systematically explore performance and efficiency across a wide range of current, near-future, and hypothetical hardware. We find LIMINAL is accurate when comparing to LLMs executing on existing hardware, achieving a mean absolute error of $7.6\%$. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. We identify five non-negotiable challenges for LLM inference hardware, establishing compute, memory capacity, bandwidth and collective communication as primary barriers to performance. These findings suggest that achieving significant performance gains beyond 10,000 tokens-per-second will require not just hardware evolution but also fundamental algorithmic advances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08954v4">Should you use LLMs to simulate opinions? Quality checks for early-stage deliberation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted to AAAI AI for Social Impact (AISI), 2026. This version includes Appendices
    </div>
    <details class="paper-abstract">
      The emergent capabilities of large language models (LLMs) have prompted interest in using them as surrogates for human subjects in opinion surveys. However, prior evaluations of LLM-based opinion simulation have relied heavily on costly, domain-specific survey data, and mixed empirical results leave their reliability in question. To enable cost-effective, early-stage evaluation, we introduce a quality control assessment designed to test the viability of LLM-simulated opinions on Likert-scale tasks without requiring large-scale human data for validation. This assessment comprises two key tests: \emph{logical consistency} and \emph{alignment with stakeholder expectations}, offering a low-cost, domain-adaptable validation tool. We apply our quality control assessment to an opinion simulation task relevant to AI-assisted content moderation and fact-checking workflows -- a socially impactful use case -- and evaluate seven LLMs using a baseline prompt engineering method (backstory prompting), as well as fine-tuning and in-context learning variants. None of the models or methods pass the full assessment, revealing several failure modes. We conclude with a discussion of the risk management implications and release \texttt{TopicMisinfo}, a benchmark dataset with paired human and LLM annotations simulated by various models and approaches, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10768v1">Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted at the 5th Muslims in Machine Learning (MusIML) Workshop, co-located with NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Summarizing consumer health questions (CHQs) can ease communication in healthcare, but unfaithful summaries that misrepresent medical details pose serious risks. We propose a framework that combines TextRank-based sentence extraction and medical named entity recognition with large language models (LLMs) to enhance faithfulness in medical text summarization. In our experiments, we fine-tuned the LLaMA-2-7B model on the MeQSum (English) and BanglaCHQ-Summ (Bangla) datasets, achieving consistent improvements across quality (ROUGE, BERTScore, readability) and faithfulness (SummaC, AlignScore) metrics, and outperforming zero-shot baselines and prior systems. Human evaluation further shows that over 80\% of generated summaries preserve critical medical information. These results highlight faithfulness as an essential dimension for reliable medical summarization and demonstrate the potential of our approach for safer deployment of LLMs in healthcare contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10720v1">PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt Sanitization</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 The code is available at https://github.com/sleeepeer/PISanitizer
    </div>
    <details class="paper-abstract">
      Long context LLMs are vulnerable to prompt injection, where an attacker can inject an instruction in a long context to induce an LLM to generate an attacker-desired output. Existing prompt injection defenses are designed for short contexts. When extended to long-context scenarios, they have limited effectiveness. The reason is that an injected instruction constitutes only a very small portion of a long context, making the defense very challenging. In this work, we propose PISanitizer, which first pinpoints and sanitizes potential injected tokens (if any) in a context before letting a backend LLM generate a response, thereby eliminating the influence of the injected instruction. To sanitize injected tokens, PISanitizer builds on two observations: (1) prompt injection attacks essentially craft an instruction that compels an LLM to follow it, and (2) LLMs intrinsically leverage the attention mechanism to focus on crucial input tokens for output generation. Guided by these two observations, we first intentionally let an LLM follow arbitrary instructions in a context and then sanitize tokens receiving high attention that drive the instruction-following behavior of the LLM. By design, PISanitizer presents a dilemma for an attacker: the more effectively an injected instruction compels an LLM to follow it, the more likely it is to be sanitized by PISanitizer. Our extensive evaluation shows that PISanitizer can successfully prevent prompt injection, maintain utility, outperform existing defenses, is efficient, and is robust to optimization-based and strong adaptive attacks. The code is available at https://github.com/sleeepeer/PISanitizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10712v1">Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by AAAI 2026 Conference
    </div>
    <details class="paper-abstract">
      Model merging has emerged as an efficient technique for expanding large language models (LLMs) by integrating specialized expert models. However, it also introduces a new threat: model merging stealing, where free-riders exploit models through unauthorized model merging. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify three critical protection properties that existing methods fail to simultaneously satisfy: (1) proactively preventing unauthorized merging; (2) ensuring compatibility with general open-source settings; (3) achieving high security with negligible performance loss. To address the above issues, we propose MergeBarrier, a plug-and-play defense that proactively prevents unauthorized merging. The core design of MergeBarrier is to disrupt the Linear Mode Connectivity (LMC) between the protected model and its homologous counterparts, thereby eliminating the low-loss path required for effective model merging. Extensive experiments show that MergeBarrier effectively prevents model merging stealing with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11752v1">Towards autonomous quantum physics research using LLM agents with access to intelligent tools</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 24 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is used in numerous fields of science, yet the initial research questions and targets are still almost always provided by human researchers. AI-generated creative ideas in science are rare and often vague, so that it remains a human task to execute them. Automating idea generation and implementation in one coherent system would significantly shift the role of humans in the scientific process. Here we present AI-Mandel, an LLM agent that can generate and implement ideas in quantum physics. AI-Mandel formulates ideas from the literature and uses a domain-specific AI tool to turn them into concrete experiment designs that can readily be implemented in laboratories. The generated ideas by AI-Mandel are often scientifically interesting - for two of them we have already written independent scientific follow-up papers. The ideas include new variations of quantum teleportation, primitives of quantum networks in indefinite causal orders, and new concepts of geometric phases based on closed loops of quantum information transfer. AI-Mandel is a prototypical demonstration of an AI physicist that can generate and implement concrete, actionable ideas. Building such a system is not only useful to accelerate science, but it also reveals concrete open challenges on the path to human-level artificial scientists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11733v1">Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 6 pages, 2 figures, 2 tables. Uses ICML 2025 style
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to propose tokens that are later verified by a stronger target model. While effective in centralized systems, its behavior in decentralized settings, where network latency often dominates compute, remains under-characterized. We present Decentralized Speculative Decoding (DSD), a plug-and-play framework for decentralized inference that turns communication delay into useful computation by verifying multiple candidate tokens in parallel across distributed nodes. We further introduce an adaptive speculative verification strategy that adjusts acceptance thresholds by token-level semantic importance, delivering an additional 15% to 20% end-to-end speedup without retraining. In theory, DSD reduces cross-node communication cost by approximately (N-1)t1(k-1)/k, where t1 is per-link latency and k is the average number of tokens accepted per round. In practice, DSD achieves up to 2.56x speedup on HumanEval and 2.59x on GSM8K, surpassing the Eagle3 baseline while preserving accuracy. These results show that adapting speculative decoding for decentralized execution provides a system-level optimization that converts network stalls into throughput, enabling faster distributed LLM inference with no model retraining or architectural changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11729v1">Harli: Harvest Underutilized Resources in LLM Serving with Finetuning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed under the Model-as-a-Service (MaaS) paradigm. To meet stringent quality-of-service (QoS) requirements, existing LLM serving systems disaggregate the prefill and decode phases of inference. However, decode instances often experience low GPU utilization due to their memory-bound nature and insufficient batching in dynamic workloads, leaving compute resources underutilized. We introduce Harli, a serving system that improves GPU utilization by co-locating parameter-efficient finetuning (PEFT) tasks with LLM decode instances. PEFT tasks are compute-bound and memory-efficient, making them ideal candidates for safe co-location. Specifically, Harli addresses key challenges--limited memory and unpredictable interference--using three components: a unified memory allocator for runtime memory reuse, a two-stage latency predictor for decode latency modeling, and a QoS-guaranteed throughput-maximizing scheduler for throughput maximization. Experimental results show that Harli improves the finetune throughput by 46.2% on average (up to 92.0%) over state-of-the-art serving systems, while maintaining strict QoS guarantees for inference decode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.20749v4">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07127v3">REACT-LLM: A Benchmark for Evaluating LLM Integration with Causal Features in Clinical Prognostic Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and causal learning each hold strong potential for clinical decision making (CDM). However, their synergy remains poorly understood, largely due to the lack of systematic benchmarks evaluating their integration in clinical risk prediction. In real-world healthcare, identifying features with causal influence on outcomes is crucial for actionable and trustworthy predictions. While recent work highlights LLMs' emerging causal reasoning abilities, there lacks comprehensive benchmarks to assess their causal learning and performance informed by causal features in clinical risk prediction. To address this, we introduce REACT-LLM, a benchmark designed to evaluate whether combining LLMs with causal features can enhance clinical prognostic performance and potentially outperform traditional machine learning (ML) methods. Unlike existing LLM-clinical benchmarks that often focus on a limited set of outcomes, REACT-LLM evaluates 7 clinical outcomes across 2 real-world datasets, comparing 15 prominent LLMs, 6 traditional ML models, and 3 causal discovery (CD) algorithms. Our findings indicate that while LLMs perform reasonably in clinical prognostics, they have not yet outperformed traditional ML models. Integrating causal features derived from CD algorithms into LLMs offers limited performance gains, primarily due to the strict assumptions of many CD methods, which are often violated in complex clinical data. While the direct integration yields limited improvement, our benchmark reveals a more promising synergy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v2">LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 12 Figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Since all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9x-3.6x lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72x reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10645v1">ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Weight-only post-training quantization (PTQ) compresses the weights of Large Language Models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe accuracy degradation, especially in recent reasoning LLMs where errors accumulate across long chains of thought. Existing PTQ methods either fail to sufficiently suppress outliers or introduce significant overhead during inference. In this paper, we propose Pairwise Rotation Quantization (ParoQuant), a weight-only PTQ method that combines hardware-efficient and optimizable independent Givens rotations with channel-wise scaling to even out the magnitude across channels and narrow the dynamic range within each quantization group. We further co-design the inference kernel to fully exploit GPU parallelism and keep the rotations and scaling lightweight at runtime. ParoQuant achieves an average 2.4% accuracy improvement over AWQ on reasoning tasks with less than 10% overhead. This paves the way for more efficient and accurate deployment of reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00730v3">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This work is intended for submission to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10507v1">Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has led to impressive performance on a range of tasks, yet advanced instruction following (IF)-especially for complex, multi-turn, and system-prompted instructions-remains a significant challenge. Rigorous evaluation and effective training for such capabilities are hindered by the lack of high-quality, human-annotated benchmarks and reliable, interpretable reward signals. In this work, we introduce AdvancedIF (we will release this benchmark soon), a comprehensive benchmark featuring over 1,600 prompts and expert-curated rubrics that assess LLMs ability to follow complex, multi-turn, and system-level instructions. We further propose RIFL (Rubric-based Instruction-Following Learning), a novel post-training pipeline that leverages rubric generation, a finetuned rubric verifier, and reward shaping to enable effective reinforcement learning for instruction following. Extensive experiments demonstrate that RIFL substantially improves the instruction-following abilities of LLMs, achieving a 6.7% absolute gain on AdvancedIF and strong results on public benchmarks. Our ablation studies confirm the effectiveness of each component in RIFL. This work establishes rubrics as a powerful tool for both training and evaluating advanced IF in LLMs, paving the way for more capable and reliable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.20067v2">PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10480v1">Scalable Synthesis of distributed LLM workloads through Symbolic Tensor Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Optimizing the performance of large language models (LLMs) on large-scale AI training and inference systems requires a scalable and expressive mechanism to model distributed workload execution. Such modeling is essential for pre-deployment system-level optimizations (e.g., parallelization strategies) and design-space explorations. While recent efforts have proposed collecting execution traces from real systems, access to large-scale infrastructure remains limited to major cloud providers. Moreover, traces obtained from existing platforms cannot be easily adapted to study future larger-scale system configurations. We introduce Symbolic Tensor grAph GEnerator(STAGE), a framework that synthesizes high-fidelity execution traces to accurately model LLM workloads. STAGE supports a comprehensive set of parallelization strategies, allowing users to systematically explore a wide spectrum of LLM architectures and system configurations. STAGE demonstrates its scalability by synthesizing high-fidelity LLM traces spanning over 32K GPUs, while preserving tensor-level accuracy in compute, memory, and communication. STAGE will be publicly available to facilitate further research in distributed machine learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10459v1">LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02635v2">Test Set Quality in Multilingual LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 to appear in the proceedings of Eval4NLP workshop at AACL 2025. Camera ready version
    </div>
    <details class="paper-abstract">
      Several multilingual benchmark datasets have been developed in a semi-automatic manner in the recent past to measure progress and understand the state-of-the-art in the multilingual capabilities of Large Language Models. However, there is not a lot of attention paid to the quality of the datasets themselves, despite the existence of previous work in identifying errors in even fully human-annotated test sets. In this paper, we manually analyze recent multilingual evaluation sets in two languages - French and Telugu, identifying several errors in the process. We compare the performance difference across several LLMs with the original and revised versions of the datasets and identify large differences (almost 10% in some cases) in both languages). Based on these results, we argue that test sets should not be considered immutable and should be revisited, checked for correctness, and potentially versioned. We end with some recommendations for both the dataset creators as well as consumers on addressing the dataset quality issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10394v1">LLM-YOLOMS: Large Language Model-based Semantic Interpretation and Fault Diagnosis for Wind Turbine Components</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Journal resubmission
    </div>
    <details class="paper-abstract">
      The health condition of wind turbine (WT) components is crucial for ensuring stable and reliable operation. However, existing fault detection methods are largely limited to visual recognition, producing structured outputs that lack semantic interpretability and fail to support maintenance decision-making. To address these limitations, this study proposes an integrated framework that combines YOLOMS with a large language model (LLM) for intelligent fault analysis and diagnosis. Specifically, YOLOMS employs multi-scale detection and sliding-window cropping to enhance fault feature extraction, while a lightweight key-value (KV) mapping module bridges the gap between visual outputs and textual inputs. This module converts YOLOMS detection results into structured textual representations enriched with both qualitative and quantitative attributes. A domain-tuned LLM then performs semantic reasoning to generate interpretable fault analyses and maintenance recommendations. Experiments on real-world datasets demonstrate that the proposed framework achieves a fault detection accuracy of 90.6\% and generates maintenance reports with an average accuracy of 89\%, thereby improving the interpretability of diagnostic results and providing practical decision support for the operation and maintenance of wind turbines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10381v1">Position: On the Methodological Pitfalls of Evaluating Base LLMs for Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Existing work investigates the reasoning capabilities of large language models (LLMs) to uncover their limitations, human-like biases and underlying processes. Such studies include evaluations of base LLMs (pre-trained on unlabeled corpora only) for this purpose. Our position paper argues that evaluating base LLMs' reasoning capabilities raises inherent methodological concerns that are overlooked in such existing studies. We highlight the fundamental mismatch between base LLMs' pretraining objective and normative qualities, such as correctness, by which reasoning is assessed. In particular, we show how base LLMs generate logically valid or invalid conclusions as coincidental byproducts of conforming to purely linguistic patterns of statistical plausibility. This fundamental mismatch challenges the assumptions that (a) base LLMs' outputs can be assessed as their bona fide attempts at correct answers or conclusions; and (b) conclusions about base LLMs' reasoning can generalize to post-trained LLMs optimized for successful instruction-following. We call for a critical re-examination of existing work that relies implicitly on these assumptions, and for future work to account for these methodological pitfalls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10354v1">Knowledge Graphs Generation from Cultural Heritage Texts: Combining LLMs and Ontological Engineering for Scholarly Debates</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 46 pages
    </div>
    <details class="paper-abstract">
      Cultural Heritage texts contain rich knowledge that is difficult to query systematically due to the challenges of converting unstructured discourse into structured Knowledge Graphs (KGs). This paper introduces ATR4CH (Adaptive Text-to-RDF for Cultural Heritage), a systematic five-step methodology for Large Language Model-based Knowledge Extraction from Cultural Heritage documents. We validate the methodology through a case study on authenticity assessment debates. Methodology - ATR4CH combines annotation models, ontological frameworks, and LLM-based extraction through iterative development: foundational analysis, annotation schema development, pipeline architecture, integration refinement, and comprehensive evaluation. We demonstrate the approach using Wikipedia articles about disputed items (documents, artifacts...), implementing a sequential pipeline with three LLMs (Claude Sonnet 3.7, Llama 3.3 70B, GPT-4o-mini). Findings - The methodology successfully extracts complex Cultural Heritage knowledge: 0.96-0.99 F1 for metadata extraction, 0.7-0.8 F1 for entity recognition, 0.65-0.75 F1 for hypothesis extraction, 0.95-0.97 for evidence extraction, and 0.62 G-EVAL for discourse representation. Smaller models performed competitively, enabling cost-effective deployment. Originality - This is the first systematic methodology for coordinating LLM-based extraction with Cultural Heritage ontologies. ATR4CH provides a replicable framework adaptable across CH domains and institutional resources. Research Limitations - The produced KG is limited to Wikipedia articles. While the results are encouraging, human oversight is necessary during post-processing. Practical Implications - ATR4CH enables Cultural Heritage institutions to systematically convert textual knowledge into queryable KGs, supporting automated metadata enrichment and knowledge discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10333v1">EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) poses significant challenges regarding computational resources and memory capacity. Although distributed training techniques help mitigate these issues, they still suffer from considerable communication overhead. Existing approaches primarily rely on static gradient compression to enhance communication efficiency; however, these methods neglect the dynamic nature of evolving gradients during training, leading to performance degradation. Accelerating LLM training via compression without sacrificing performance remains a challenge. In this paper, we propose an entropy-driven dynamic gradient compression framework called EDGC. The core concept is to adjust the compression rate during LLM training based on the evolving trends of gradient entropy, taking into account both compression efficiency and error. EDGC consists of three key components.First, it employs a down-sampling method to efficiently estimate gradient entropy, reducing computation overhead. Second, it establishes a theoretical model linking compression rate with gradient entropy, enabling more informed compression decisions. Lastly, a window-based adjustment mechanism dynamically adapts the compression rate across pipeline stages, improving communication efficiency and maintaining model performance. We implemented EDGC on a 32-NVIDIA-V100 cluster and a 64-NVIDIA-H100 cluster to train GPT2-2.5B and GPT2-12.1B, respectively. The results show that EDGC significantly reduces communication latency and training time by up to 46.45% and 16.13% while preserving LLM accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02573v2">Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This paper has been accepted for publication at AAAI-26
    </div>
    <details class="paper-abstract">
      Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding. We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are guessed using language modeling abilities, memorized samples that are recalled due to high duplication in the training set, and non-memorized samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10303v1">Rectify Evaluation Preference: Improving LLMs' Critique on Math Reasoning via Perplexity-aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      To improve Multi-step Mathematical Reasoning (MsMR) of Large Language Models (LLMs), it is crucial to obtain scalable supervision from the corpus by automatically critiquing mistakes in the reasoning process of MsMR and rendering a final verdict of the problem-solution. Most existing methods rely on crafting high-quality supervised fine-tuning demonstrations for critiquing capability enhancement and pay little attention to delving into the underlying reason for the poor critiquing performance of LLMs. In this paper, we orthogonally quantify and investigate the potential reason -- imbalanced evaluation preference, and conduct a statistical preference analysis. Motivated by the analysis of the reason, a novel perplexity-aware reinforcement learning algorithm is proposed to rectify the evaluation preference, elevating the critiquing capability. Specifically, to probe into LLMs' critiquing characteristics, a One-to-many Problem-Solution (OPS) benchmark is meticulously constructed to quantify the behavior difference of LLMs when evaluating the problem solutions generated by itself and others. Then, to investigate the behavior difference in depth, we conduct a statistical preference analysis oriented on perplexity and find an intriguing phenomenon -- ``LLMs incline to judge solutions with lower perplexity as correct'', which is dubbed as \textit{imbalanced evaluation preference}. To rectify this preference, we regard perplexity as the baton in the algorithm of Group Relative Policy Optimization, supporting the LLMs to explore trajectories that judge lower perplexity as wrong and higher perplexity as correct. Extensive experimental results on our built OPS and existing available critic benchmarks demonstrate the validity of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10301v1">Rethinking Visual Information Processing in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Despite the remarkable success of the LLaVA architecture for vision-language tasks, its design inherently struggles to effectively integrate visual features due to the inherent mismatch between text and vision modalities. We tackle this issue from a novel perspective in which the LLM not only serves as a language model but also a powerful vision encoder. To this end, we present LLaViT - Large Language Models as extended Vision Transformers - which enables the LLM to simultaneously function as a vision encoder through three key modifications: (1) learning separate QKV projections for vision modality, (2) enabling bidirectional attention on visual tokens, and (3) incorporating both global and local visual representations. Through extensive controlled experiments on a wide range of LLMs, we demonstrate that LLaViT significantly outperforms the baseline LLaVA method on a multitude of benchmarks, even surpassing models with double its parameter count, establishing a more effective approach to vision-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11034v2">The Impact of Large Language Models (LLMs) on Code Review Process</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently gained prominence in the field of software development, significantly boosting productivity and simplifying teamwork. Although prior studies have examined task-specific applications, the phase-specific effects of LLM assistance on the efficiency of code review processes remain underexplored. This research investigates the effect of GPT on GitHub pull request (PR) workflows, with a focus on reducing resolution time, optimizing phase-specific performance, and assisting developers. We curated a dataset of 25,473 PRs from 9,254 GitHub projects and identified GPT-assisted PRs using a semi-automated heuristic approach that combines keyword-based detection, regular expression filtering, and manual verification until achieving 95% labeling accuracy. We then applied statistical modeling, including multiple linear regression and Mann-Whitney U test, to evaluate differences between GPT-assisted and non-assisted PRs, both at the overall resolution level and across distinct review phases. Our research has revealed that early adoption of GPT can substantially boost the effectiveness of the PR process, leading to considerable time savings at various stages. Our findings suggest that GPT-assisted PRs reduced median resolution time by more than 60% (9 hours compared to 23 hours for non-assisted PRs). We discovered that utilizing GPT can reduce the review time by 33% and the waiting time before acceptance by 87%. Analyzing a sample dataset of 300 GPT-assisted PRs, we discovered that developers predominantly use GPT for code optimization (60%), bug fixing (26%), and documentation updates (12%). This research sheds light on the impact of the GPT model on the code review process, offering actionable insights for software teams seeking to enhance workflows and promote seamless collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10271v1">Quality Assurance of LLM-generated Code: Addressing Non-Functional Quality Characteristics</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      In recent years, LLMs have been widely integrated into software engineering workflows, supporting tasks like code generation. However, while these models often generate functionally correct outputs, we still lack a systematic understanding and evaluation of their non-functional qualities. Existing studies focus mainly on whether generated code passes the tests rather than whether it passes with quality. Guided by the ISO/IEC 25010 quality model, this study conducted three complementary investigations: a systematic review of 108 papers, two industry workshops with practitioners from multiple organizations, and an empirical analysis of patching real-world software issues using three LLMs. Motivated by insights from both the literature and practitioners, the empirical study examined the quality of generated patches on security, maintainability, and performance efficiency. Across the literature, we found that security and performance efficiency dominate academic attention, while maintainability and other qualities are understudied. In contrast, industry experts prioritize maintainability and readability, warning that generated code may accelerate the accumulation of technical debt. In our evaluation of functionally correct patches generated by three LLMs, improvements in one quality dimension often come at the cost of others. Runtime and memory results further show high variance across models and optimization strategies. Overall, our findings reveal a mismatch between academic focus, industry priorities, and model performance, highlighting the urgent need to integrate quality assurance mechanisms into LLM code generation pipelines to ensure that future generated code not only passes tests but truly passes with quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19794v5">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 (oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate model behavior across three core dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. Code is available at https://github.com/zjunlp/DataMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v1">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 Workshop on Graphs and more Complex Structures For Learning and Reasoning (GCLR)
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10233v1">Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 21 pages; To be published in AAAI-26
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Combinatorial Optimization (NCO) methods have significantly improved the capability of neural solvers to handle synthetic routing instances. Nonetheless, existing neural solvers typically struggle to generalize effectively from synthetic, uniformly-distributed training data to real-world VRP scenarios, including widely recognized benchmark instances from TSPLib and CVRPLib. To bridge this generalization gap, we present Evolutionary Realistic Instance Synthesis (EvoReal), which leverages an evolutionary module guided by large language models (LLMs) to generate synthetic instances characterized by diverse and realistic structural patterns. Specifically, the evolutionary module produces synthetic instances whose structural attributes statistically mimics those observed in authentic real-world instances. Subsequently, pre-trained NCO models are progressively refined, firstly aligning them with these structurally enriched synthetic distributions and then further adapting them through direct fine-tuning on actual benchmark instances. Extensive experimental evaluations demonstrate that EvoReal markedly improves the generalization capabilities of state-of-the-art neural solvers, yielding a notable reduced performance gap compared to the optimal solutions on the TSPLib (1.05%) and CVRPLib (2.71%) benchmarks across a broad spectrum of problem scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v1">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10182v1">Beyond the Black Box: Demystifying Multi-Turn LLM Reasoning with VISTA</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent research has increasingly focused on the reasoning capabilities of Large Language Models (LLMs) in multi-turn interactions, as these scenarios more closely mirror real-world problem-solving. However, analyzing the intricate reasoning processes within these interactions presents a significant challenge due to complex contextual dependencies and a lack of specialized visualization tools, leading to a high cognitive load for researchers. To address this gap, we present VISTA, an web-based Visual Interactive System for Textual Analytics in multi-turn reasoning tasks. VISTA allows users to visualize the influence of context on model decisions and interactively modify conversation histories to conduct "what-if" analyses across different models. Furthermore, the platform can automatically parse a session and generate a reasoning dependency tree, offering a transparent view of the model's step-by-step logical path. By providing a unified and interactive framework, VISTA significantly reduces the complexity of analyzing reasoning chains, thereby facilitating a deeper understanding of the capabilities and limitations of current LLMs. The platform is open-source and supports easy integration of custom benchmarks and local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01939v2">Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted to NeurIPS 2025. 25 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08903v2">LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Document layout understanding remains data-intensive despite advances in semi-supervised learning. We present a framework that enhances semi-supervised detection by fusing visual predictions with structural priors from text-pretrained LLMs via principled probabilistic weighting. Given unlabeled documents, an OCR-LLM pipeline infers hierarchical regions which are combined with teacher detector outputs through inverse-variance fusion to generate refined pseudo-labels.Our method demonstrates consistent gains across model scales. With a lightweight SwiftFormer backbone (26M params), we achieve 88.2$\pm$0.3 AP using only 5\% labels on PubLayNet. When applied to document-pretrained LayoutLMv3 (133M params), our fusion framework reaches 89.7$\pm$0.4 AP, surpassing both LayoutLMv3 with standard semi-supervised learning (89.1$\pm$0.4 AP, p=0.02) and matching UDOP~\cite{udop} (89.8 AP) which requires 100M+ pages of multimodal pretraining. This demonstrates that LLM structural priors are complementary to both lightweight and pretrained architectures. Key findings include: (1) learned instance-adaptive gating improves over fixed weights by +0.9 AP with data-dependent PAC bounds correctly predicting convergence; (2) open-source LLMs enable privacy-preserving deployment with minimal loss (Llama-3-70B: 87.1 AP lightweight, 89.4 AP with LayoutLMv3); (3) LLMs provide targeted semantic disambiguation (18.7\% of cases, +3.8 AP gain) beyond simple text heuristics.Total system cost includes \$12 for GPT-4o-mini API or 17 GPU-hours for local Llama-3-70B per 50K pages, amortized across training runs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10075v1">Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      With the growing number of submitted scientific papers, there is an increasing demand for systems that can assist reviewers in evaluating research claims. Experimental results are a core component of scientific work, often presented in varying formats such as tables or charts. Understanding how robust current multimodal large language models (multimodal LLMs) are at verifying scientific claims across different evidence formats remains an important and underexplored challenge. In this paper, we design and conduct a series of experiments to assess the ability of multimodal LLMs to verify scientific claims using both tables and charts as evidence. To enable this evaluation, we adapt two existing datasets of scientific papers by incorporating annotations and structures necessary for a multimodal claim verification task. Using this adapted dataset, we evaluate 12 multimodal LLMs and find that current models perform better with table-based evidence while struggling with chart-based evidence. We further conduct human evaluations and observe that humans maintain strong performance across both formats, unlike the models. Our analysis also reveals that smaller multimodal LLMs (under 8B) show weak correlation in performance between table-based and chart-based tasks, indicating limited cross-modal generalization. These findings highlight a critical gap in current models' multimodal reasoning capabilities. We suggest that future multimodal LLMs should place greater emphasis on improving chart understanding to better support scientific claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10067v1">Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at https://muser-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10049v1">Continuous Benchmark Generation for Evaluating Enterprise-scale LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      The rapid adoption of AI agents across domains has made systematic evaluation crucial for ensuring their usefulness and successful production deployment. Evaluation of AI agents typically involves using a fixed set of benchmarks and computing multiple evaluation metrics for the agent. While sufficient for simple coding tasks, these benchmarks fall short for enterprise-scale agents, where services and requirements evolve continuously and ground-truth examples are sparse. We propose a process of benchmark generation that helps evolve the benchmarks as the requirements change and perform robust evaluation of evolving AI agents. We instantiate this approach for a case study of service migration from one deployment platform to another at a large public enterprise. Our approach relies on semi-structured documents where developers express the high-level intent, and uses state-of-the-art LLMs to generate benchmarks from just a small number of such documents. Overall, this process results in a maintainable evaluation framework, enabling rapid feedback on agent performance and facilitating targeted improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10037v1">Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Existing tool-augmented large language models (LLMs) encounter significant challenges when processing complex queries. Current frameworks such as ReAct are prone to local optimization traps due to their reliance on incremental decision-making processes. To address these limitations, we propose a novel Planner-centric Plan-Execute paradigm that fundamentally resolves local optimization bottlenecks through architectural innovation. Central to our approach is a novel Planner model that performs global Directed Acyclic Graph (DAG) planning for complex queries, enabling optimized execution beyond conventional tool coordination. We also introduce ComplexTool-Plan, a large-scale benchmark dataset featuring complex queries that demand sophisticated multi-tool composition and coordination capabilities. Additionally, we develop a two-stage training methodology that integrates Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), systematically enhancing the Planner's tool selection accuracy and global planning awareness through structured DAG-based planning. When integrated with a capable executor, our framework achieves state-of-the-art performance on the StableToolBench benchmark for complex user queries, demonstrating superior end-to-end execution capabilities and robust handling of intricate multi-tool workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19319v2">FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 ML4H 2025 Proceedings
    </div>
    <details class="paper-abstract">
      The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (https://github.com/glee4810/FHIR-AgentBench) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10007v1">AssertMiner: Module-Level Spec Generation and Assertion Mining using Static Analysis Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 6 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Assertion-based verification (ABV) is a key approach to checking whether a logic design complies with its architectural specifications. Existing assertion generation methods based on design specifications typically produce only top-level assertions, overlooking verification needs on the implementation details in the modules at the micro-architectural level, where design errors occur more frequently. To address this limitation, we present AssertMiner, a module-level assertion generation framework that leverages static information generated from abstract syntax tree (AST) to assist LLMs in mining assertions. Specifically, it performs AST-based structural extraction to derive the module call graph, I/O table, and dataflow graph, guiding the LLM to generate module-level specifications and mine module-level assertions. Our evaluation demonstrates that AssertMiner outperforms existing methods such as AssertLLM and Spec2Assertion in generating high-quality assertions for modules. When integrated with these methods, AssertMiner can enhance the structural coverage and significantly improve the error detection capability, enabling a more comprehensive and efficient verification process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09062v2">Pricing Online LLM Services with Data-Calibrated Stackelberg Routing Game</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Extended version
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has established LLM routing as a standard service delivery mechanism, where users select models based on cost, Quality of Service (QoS), among other things. However, optimal pricing in LLM routing platforms requires precise modeling for dynamic service markets, and solving this problem in real time at scale is computationally intractable. In this paper, we propose \PriLLM, a novel practical and scalable solution for real-time dynamic pricing in competitive LLM routing. \PriLLM models the service market as a Stackelberg game, where providers set prices and users select services based on multiple criteria. To capture real-world market dynamics, we incorporate both objective factors (\eg~cost, QoS) and subjective user preferences into the model. For scalability, we employ a deep aggregation network to learn provider abstraction that preserve user-side equilibrium behavior across pricing strategies. Moreover, \PriLLM offers interpretability by explaining its pricing decisions. Empirical evaluation on real-world data shows that \PriLLM achieves over 95\% of the optimal profit while only requiring less than 5\% of the optimal solution's computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09998v1">DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The performance of modern DBMSs such as MySQL and PostgreSQL heavily depends on the configuration of performance-critical knobs. Manual tuning these knobs is laborious and inefficient due to the complex and high-dimensional nature of the configuration space. Among the automated tuning methods, reinforcement learning (RL)-based methods have recently sought to improve the DBMS knobs tuning process from several different perspectives. However, they still encounter challenges with slow convergence speed during offline training. In this paper, we mainly focus on how to leverage the valuable tuning hints contained in various textual documents such as DBMS manuals and web forums to improve the offline training of RL-based methods. To this end, we propose an efficient DBMS knobs tuning framework named DemoTuner via a novel LLM-assisted demonstration reinforcement learning method. Specifically, to comprehensively and accurately mine tuning hints from documents, we design a structured chain of thought prompt to employ LLMs to conduct a condition-aware tuning hints extraction task. To effectively integrate the mined tuning hints into RL agent training, we propose a hint-aware demonstration reinforcement learning algorithm HA-DDPGfD in DemoTuner. As far as we know, DemoTuner is the first work to introduce the demonstration reinforcement learning algorithm for DBMS knobs tuning. Experimental evaluations conducted on MySQL and PostgreSQL across various workloads demonstrate the significant advantages of DemoTuner in both performance improvement and online tuning cost reduction over three representative baselines including DB-BERT, GPTuner and CDBTune. Additionally, DemoTuner also exhibits superior adaptability to application scenarios with unknown workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09969v1">Owlgorithm: Supporting Self-Regulated Learning in Competitive Programming through LLM-Driven Reflection</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 7 pages, 1 figure, to be published in SIGCSE '26
    </div>
    <details class="paper-abstract">
      We present Owlgorithm, an educational platform that supports Self-Regulated Learning (SRL) in competitive programming (CP) through AI-generated reflective questions. Leveraging GPT-4o, Owlgorithm produces context-aware, metacognitive prompts tailored to individual student submissions. Integrated into a second- and third-year CP course, the system-provided reflective prompts adapted to student outcomes: guiding deeper conceptual insight for correct solutions and structured debugging for partial or failed ones. Our exploratory assessment of student ratings and TA feedback revealed both promising benefits and notable limitations. While many found the generated questions useful for reflection and debugging, concerns were raised about feedback accuracy and classroom usability. These results suggest advantages of LLM-supported reflection for novice programmers, though refinements are needed to ensure reliability and pedagogical value for advanced learners. From our experience, several key insights emerged: GenAI can effectively support structured reflection, but careful prompt design, dynamic adaptation, and usability improvements are critical to realizing their potential in education. We offer specific recommendations for educators using similar tools and outline next steps to enhance Owlgorithm's educational impact. The underlying framework may also generalize to other reflective learning contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09964v1">EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for instrument control requires methods that go beyond standard, stateless algorithmic benchmarks, since the behavior of physical systems cannot be fully captured by unit tests alone. Here we introduce EnvTrace, a simulation-based method that evaluates execution traces to assess semantic code equivalence. EnvTrace is demonstrated with a beamline control-logic digital twin to facilitate the evaluation of instrument control code, with the digital twin itself also enabling the pre-execution validation of live experiments. Over 30 LLMs were evaluated using trace alignment to generate a multi-faceted score for functional correctness across key behavioral dimensions, showing that many top-tier models can approach human-level performance in rapid control-code generation. This is a first step toward a broader vision where LLMs and digital twins work symbiotically: LLMs providing intuitive control and agentic orchestration, and digital twins offering safe and high-fidelity environments, paving the way towards autonomous embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.20194v3">DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 NeurIPS 2025. The code is available on Github (see hyperlink in the paper)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver strong performance but are difficult to deploy due to high memory and compute costs. While pruning reduces these demands, most methods ignore activation sparsity observed at runtime. We reinterpret activation sparsity as dynamic structured weight sparsity and propose DuoGPT, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity. To preserve accuracy, we extend the Optimal Brain Compression (OBC) framework with activation-aware calibration and introduce output residuals from the dense model as correction terms. We further optimize the solution for efficient GPU execution, enabling scalability to billion-parameter LLMs. Evaluations on LLaMA-2 and LLaMA-3 show that DuoGPT outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39$\times$ compared to the baseline dense model. Code is available at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09865v1">In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 Oral
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) for chain-of-thought reasoning presents a significant challenge: supervised fine-tuning on a single "golden" rationale hurts generalization as it penalizes equally valid alternatives, whereas reinforcement learning with verifiable rewards struggles with credit assignment and prohibitive computational cost. To tackle these limitations, we introduce InTRO (In-Token Rationality Optimization), a new framework that enables both token-level exploration and self-feedback for accurate and concise reasoning. Instead of directly optimizing an intractable objective over all valid reasoning paths, InTRO leverages correction factors-token-wise importance weights estimated by the information discrepancy between the generative policy and its answer-conditioned counterpart, for informative next token selection. This approach allows the model to perform token-level exploration and receive self-generated feedback within a single forward pass, ultimately encouraging accurate and concise rationales. Across six math-reasoning benchmarks, InTRO consistently outperforms other baselines, raising solution accuracy by up to 20% relative to the base model. Its chains of thought are also notably more concise, exhibiting reduced verbosity. Beyond this, InTRO enables cross-domain transfer, successfully adapting to out-of-domain reasoning tasks that extend beyond the realm of mathematics, demonstrating robust generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09855v1">Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 14 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09837v1">MoFa: A Unified Performance Modeling Framework for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      The exponential growth in LLM scales, with parameters soaring from billions to trillions, has necessitated distributed pretraining across large clusters comprising thousands to tens of thousands of devices. While hybrid parallelization strategies enable such pretraining, the vast combinatorial strategy space introduces significant optimization challenges. Traditional manual tuning methods incur prohibitive trial-and-error costs, and existing performance modeling approaches exhibit critical limitations: they fail to comprehensively account for prevalent optimization features and ignore the substantial overhead imposed by essential fault tolerance mechanisms like checkpoint recovery in long-duration pretraining. To address these gaps, we propose MoFa, a novel pretraining performance modeling framework that unifies multi-dimensional optimization features and fault tolerance. MoFa incorporates an enhanced cost model to accurately capture the effects of key optimizations and integrates a fault tolerance model based on historical cluster reliability data. Besides, a MoFa-based tuning system is developed to explore optimal pretraining performance and potential bottlenecks in various scenarios. Extensive modeling evaluations demonstrate that MoFa can achieve high prediction accuracy across various scenarios. In addition, through comprehensive tuning experiments, our framework systematically reveals the key factors influencing pretraining performance under different configurations, which provides solid a priori guidance for LLM pretraining system design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09831v1">Answering Students' Questions on Course Forums Using Multiple Chain-of-Thought Reasoning and Finetuning RAG-Enabled LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The course forums are increasingly significant and play vital role in facilitating student discussions and answering their questions related to the course. It provides a platform for students to post their questions related to the content and admin issues related to the course. However, there are several challenges due to the increase in the number of students enrolled in the course. The primary challenge is that students' queries cannot be responded immediately and the instructors have to face lots of repetitive questions. To mitigate these issues, we propose a question answering system based on large language model with retrieval augmented generation (RAG) method. This work focuses on designing a question answering system with open source Large Language Model (LLM) and fine-tuning it on the relevant course dataset. To further improve the performance, we use a local knowledge base and applied RAG method to retrieve relevant documents relevant to students' queries, where the local knowledge base contains all the course content. To mitigate the hallucination of LLMs, We also integrate it with multi chain-of-thought reasoning to overcome the challenge of hallucination in LLMs. In this work, we experiment fine-tuned LLM with RAG method on the HotpotQA dataset. The experimental results demonstrate that the fine-tuned LLM with RAG method has a strong performance on question answering task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14289v3">From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10860v1">HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted in AIWare 2025
    </div>
    <details class="paper-abstract">
      Unit testing in High-Performance Computing (HPC) is critical but challenged by parallelism, complex algorithms, and diverse hardware. Traditional methods often fail to address non-deterministic behavior and synchronization issues in HPC applications. This paper introduces HPCAgentTester, a novel multi-agent Large Language Model (LLM) framework designed to automate and enhance unit test generation for HPC software utilizing OpenMP and MPI. HPCAgentTester employs a unique collaborative workflow where specialized LLM agents (Recipe Agent and Test Agent) iteratively generate and refine test cases through a critique loop. This architecture enables the generation of context-aware unit tests that specifically target parallel execution constructs, complex communication patterns, and hierarchical parallelism. We demonstrate HPCAgentTester's ability to produce compilable and functionally correct tests for OpenMP and MPI primitives, effectively identifying subtle bugs that are often missed by conventional techniques. Our evaluation shows that HPCAgentTester significantly improves test compilation rates and correctness compared to standalone LLMs, offering a more robust and scalable solution for ensuring the reliability of parallel software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10855v1">ExPairT-LLM: Exact Learning for LLM Code Selection by Pairwise Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Despite recent advances in LLMs, the task of code generation is still challenging. To cope, code selection algorithms select the best program from multiple programs generated by an LLM. However, existing algorithms can fail to identify the correct program, either because they can misidentify nonequivalent programs or because they rely on an LLM and assume it always correctly determines the output for every input. We present ExPairT-LLM, an exact learning algorithm for code selection that selects a program by posing to an LLM oracle two new types of queries: pairwise membership and pairwise equivalence. These queries are simpler for LLMs and enable ExPairT-LLM to identify the correct program through a tournament, which is robust to some LLM mistakes. We evaluate ExPairT-LLM on four popular code datasets. Its pass@1 (success rate) outperforms the state-of-the-art code selection algorithm on average by +13.0% and up to +27.1%. It also improves the pass@1 of LLMs performing complex reasoning by +24.0%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14617v3">SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 25 pages, 16 figures, 2 tables. The workload traces, our simulator harness and the SageServe scheduler are available at https://github.com/shashwatj07/SageServe
    </div>
    <details class="paper-abstract">
      Global cloud service providers handle inference workloads for Large Language Models (LLMs) that span latency-sensitive (e.g., chatbots) and insensitive (e.g., report writing) tasks, resulting in diverse and often conflicting Service Level Agreement (SLA) requirements. Managing such mixed workloads is challenging due to the complexity of the inference serving stack, which encompasses multiple models, GPU hardware, and global data centers. Existing solutions often silo such fast and slow tasks onto separate GPU resource pools with different SLAs, but this leads to significant under-utilization of expensive accelerators due to load mismatch. In this article, we characterize the LLM serving workloads at Microsoft Office 365, one of the largest users of LLMs within Microsoft Azure cloud with over 10 million requests per day, and highlight key observations across workloads in different data center regions and across time. This is one of the first such public studies of Internet-scale LLM workloads. We use these insights to propose SageServe, a comprehensive LLM serving framework that dynamically adapts to workload demands using multi-timescale control knobs. It combines short-term request routing to data centers with long-term scaling of GPU VMs and model placement with higher lead times, and co-optimizes the routing and resource allocation problem using a traffic forecast model and an Integer Linear Programming (ILP) solution. We evaluate SageServe through real runs and realistic simulations on 10 million production requests across three regions and four open-source models. We achieve up to 25% savings in GPU-hours compared to the current baseline deployment and reduce GPU-hour wastage due to inefficient auto-scaling by 80%, resulting in a potential monthly cost savings of up to $2.5 million, while maintaining tail latency and meeting SLAs.
    </details>
</div>
