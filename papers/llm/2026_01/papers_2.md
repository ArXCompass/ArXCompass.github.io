# llm - 2026_01

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08710v2">Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 15 pages, 7 figures, Proceedings of the 2026 Symposium on Computer Science and Law (CSLAW '26)
    </div>
    <details class="paper-abstract">
      Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03597v2">From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14532v1">Search over Self-Edit Strategies for LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Many LLM-based open-ended search systems freeze the foundation model that proposes improvements to existing solutions, which may bottleneck long-run progress. Recent work has explored updating the proposal model at test time [arXiv:2511.23473], but the update strategy is still typically hand-specified. Therefore, this study investigated whether an LLM can use task feedback to decide how it should update its weights. For tractability, we focused on the simpler case where there is only one round of self-improvement, and restricted the update operator to self-supervised next token prediction (NTP), leaving the model freedom in choosing its training data and key NTP hyperparameters. Using the Self-Adapting Language Models (SEAL) [arXiv:2506.10943] framework as a testbed, we relaxed its fixed human template constraint and allowed the model to generate its own self-edit templates, thereby giving it more control over its training data and hyperparameters. Two variants were studied, differing in whether template generation was conditioned on a lightweight archive of past templates. In SEAL's Single-Passage Knowledge Incorporation setting with Qwen3-8B on SQuAD [arXiv:1606.05250], the no-archive variant performed comparably to the weaker "Implications" baseline, while the archive variant outperformed "Implications" and approached the strongest human-designed "Rewrite" baseline without surpassing it. Further analysis of collapse in the model's exploration revealed that a naive archive can confer some short-term robustness but can also accelerate homogenization, suggesting that explicit novelty pressure may be required to consistently advance beyond carefully optimized human strategies. Our code is available at https://github.com/cheongalc/search-self-edit-strategies .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14528v1">LLM Security and Safety: Insights from Homotopy-Inspired Prompt Obfuscation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      In this study, we propose a homotopy-inspired prompt obfuscation framework to enhance understanding of security and safety vulnerabilities in Large Language Models (LLMs). By systematically applying carefully engineered prompts, we demonstrate how latent model behaviors can be influenced in unexpected ways. Our experiments encompassed 15,732 prompts, including 10,000 high-priority cases, across LLama, Deepseek, KIMI for code generation, and Claude to verify. The results reveal critical insights into current LLM safeguards, highlighting the need for more robust defense mechanisms, reliable detection strategies, and improved resilience. Importantly, this work provides a principled framework for analyzing and mitigating potential weaknesses, with the goal of advancing safe, responsible, and trustworthy AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01129v2">RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17792v4">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/git-disl/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03149v2">PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware "nextprompt" that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource -- complete with code, rules, and generation logs -- to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14479v1">Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      This paper investigates the ability of large language models (LLMs) to solve statistical tasks, as well as their capacity to assess the quality of reasoning. While state-of-the-art LLMs have demonstrated remarkable performance in a range of NLP tasks, their competence in addressing even moderately complex statistical challenges is not well understood. We have fine-tuned selected open-source LLMs on a specially developed dataset to enhance their statistical reasoning capabilities, and compared their performance with the human scores used as a benchmark. Our results show that the fine-tuned models achieve better performance on advanced statistical tasks on the level comparable to a statistics student. Fine-tuning demonstrates architecture-dependent improvements, with some models showing significant performance gains, indicating clear potential for deployment in educational technology and statistical analysis assistance systems. We also show that LLMs themselves can be far better judges of the answers quality (including explanation and reasoning assessment) in comparison to traditional metrics, such as BLEU or BertScore. This self-evaluation capability enables scalable automated assessment for statistical education platforms and quality assurance in automated analysis tools. Potential applications also include validation tools for research methodology in academic and industry settings, and quality control mechanisms for data analysis workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14456v1">On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 9 pages, 4 figures, 3 tables, 2 pages of supplementary materials. Submitted to a conference implementing a double-blind review process
    </div>
    <details class="paper-abstract">
      Recent work shows that fine-tuned Large Language Models (LLMs) can achieve high valid plan rates on PDDL planning tasks. However, it remains unclear whether this reflects transferable planning competence or domain-specific memorization. In this work, we fine-tune a 1.7B-parameter LLM on 40,000 domain-problem-plan tuples from 10 IPC 2023 domains, and evaluate both in-domain and cross-domain generalization. While the model reaches 82.9% valid plan rate in in-domain conditions, it achieves 0% on two unseen domains. To analyze this failure, we introduce three diagnostic interventions, namely (i) instance-wise symbol anonymization, (ii) compact plan serialization, and (iii) verifier-reward fine-tuning using the VAL validator as a success-focused reinforcement signal. Symbol anonymization and compact serialization cause significant performance drops despite preserving plan semantics, thus revealing strong sensitivity to surface representations. Verifier-reward fine-tuning reaches performance saturation in half the supervised training epochs, but does not improve cross-domain generalization. For the explored configurations, in-domain performance plateaus around 80%, while cross-domain performance collapses, suggesting that our fine-tuned model relies heavily on domain-specific patterns rather than transferable planning competence in this setting. Our results highlight a persistent generalization gap in LLM-based planning and provide diagnostic tools for studying its causes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v3">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      Measuring long-run LLM outcomes (user satisfaction, expert judgment, downstream KPIs) is expensive. Teams default to cheap LLM judges, but uncalibrated proxies can invert rankings entirely. Causal Judge Evaluation (CJE) makes it affordable to aim at the right target: calibrate cheap scores against a small oracle slice, then evaluate at scale with valid uncertainty. We treat surrogate validity as auditable: for each policy or deployment context, a small oracle audit tests whether the learned calibration remains mean-unbiased, turning an uncheckable identification condition into a falsifiable diagnostic. On 4,961 Chatbot Arena prompts comparing five policies with a 16x oracle/judge cost ratio, at a 5% oracle fraction CJE achieves 99% pairwise ranking accuracy at 14x lower cost; across all configurations (5-50% oracle, varying n), accuracy averages 94%. An adversarial policy fails the transport audit and is correctly flagged; in such cases CJE refuses level claims rather than reporting biased estimates. Key findings: naive confidence intervals on raw judge scores achieve 0% coverage (CJE: ~95%); importance-weighted estimators fail despite >90% effective sample size; and the Coverage-Limited Efficiency (CLE) bound and its TTC diagnostic explain why.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14340v1">Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely integrated into interactive systems such as dialogue agents and task-oriented assistants. This growing ecosystem also raises supply-chain risks, where adversaries can distribute poisoned models that degrade downstream reliability and user trust. Existing backdoor attacks and defenses are largely prompt-centric, focusing on user-visible triggers while overlooking structural signals in multi-turn conversations. We propose Turn-based Structural Trigger (TST), a backdoor attack that activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs. Across four widely used open-source LLM models, TST achieves an average attack success rate (ASR) of 99.52% with minimal utility degradation, and remains effective under five representative defenses with an average ASR of 98.04%. The attack also generalizes well across instruction datasets, maintaining an average ASR of 99.19%. Our results suggest that dialogue structure constitutes an important and under-studied attack surface for multi-turn LLM systems, motivating structure-aware auditing and mitigation in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21486v2">Hypothesis Generation via LLM-Automated Language Bias for ILP</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 accepted by AAAI 2026 Bridge LMReasoning
    </div>
    <details class="paper-abstract">
      Inductive Logic Programming (ILP) is a principled approach for generalizing regularities from data and constructing hypotheses as interpretable logic programs. However, a key limitation is its reliance on expert-crafted language bias - the predicate inventory, types, and mode declarations that delimit the search space. We propose hypothesis generation via LLM-automated language bias: multi-agent LLMs design the bias from raw text and translate descriptions into typed facts, and a robust ILP solver induces rules under a global consistency objective. This approach reduces traditional ILP's reliance on predefined symbolic structures and the noise sensitivity of LLM-only pipelines that directly generate hypotheses as text or code. Extensive experiments in diverse, challenging scenarios validate superior performance, providing a practical, explainable, and verifiable route to hypothesis generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12762v1">Teaching LLMs to Learn Tool Trialing and Execution through Environment Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Equipping Large Language Models (LLMs) with external tools enables them to solve complex real-world problems. However, the robustness of existing methods remains a critical challenge when confronting novel or evolving tools. Existing trajectory-centric paradigms primarily rely on memorizing static solution paths during training, which limits the ability of LLMs to generalize tool usage to newly introduced or previously unseen tools. In this paper, we propose ToolMaster, a framework that shifts tool use from imitating golden tool-calling trajectories to actively learning tool usage through interaction with the environment. To optimize LLMs for tool planning and invocation, ToolMaster adopts a trial-and-execution paradigm, which trains LLMs to first imitate teacher-generated trajectories containing explicit tool trials and self-correction, followed by reinforcement learning to coordinate the trial and execution phases jointly. This process enables agents to autonomously explore correct tool usage by actively interacting with environments and forming experiential knowledge that benefits tool execution. Experimental results demonstrate that ToolMaster significantly outperforms existing baselines in terms of generalization and robustness across unseen or unfamiliar tools. All code and data are available at https://github.com/NEUIR/ToolMaster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12735v1">OpenAI for OpenAPI: Automated generation of REST API specification via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      REST APIs, based on the REpresentational State Transfer (REST) architecture, are the primary type of Web API. The OpenAPI Specification (OAS) serves as the de facto standard for describing REST APIs and is crucial for multiple software engineering tasks. However, developers face challenges in writing and maintaining OAS. Although static analysis shows potential for OAS generation, it is limited to specific programming languages and development frameworks. The powerful code understanding capabilities of LLMs offer new opportunities for OAS generation, yet they are constrained by context limitations and hallucinations. To address these challenges, we propose the OpenAI OpenAPI Project Scanner (OOPS), the first technology-agnostic LLM-based static analysis method for OAS generation, requiring fewer technology-specific rules and less human expert intervention. OOPS is implemented as an LLM agent workflow comprising two key steps: endpoint method extraction and OAS generation. By constructing an API dependency graph, it establishes necessary file associations to address LLMs' context limitations. Through multi-stage generation and self-refine, it mitigates both syntactic and semantic hallucinations during OAS generation. We evaluated OOPS on 12 real-world REST APIs spanning 5 programming languages and 8 development frameworks. Experimental results demonstrate that OOPS accurately generates high-quality OAS for REST APIs implemented with diverse technologies, achieving an average F1-score exceeding 98% for endpoint method inference, 97% for both request parameter and response inference, and 92% for parameter constraint inference. The input tokens average below 5.6K with a maximum of 16.2K, while the output tokens average below 0.9K with a maximum of 7.7K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12680v1">MetaToolAgent: Towards Generalizable Tool Usage in LLMs through Meta-Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Tool learning is increasingly important for large language models (LLMs) to effectively coordinate and utilize a diverse set of tools in order to solve complex real-world tasks. By selecting and integrating appropriate tools, LLMs extend their capabilities beyond pure language understanding to perform specialized functions. However, existing methods for tool selection often focus on limited tool sets and struggle to generalize to novel tools encountered in practical deployments. To address these challenges, we introduce a comprehensive dataset spanning 7 domains, containing 155 tools and 9,377 question-answer pairs, which simulates realistic integration scenarios. Additionally, we propose MetaToolAgent (MTA), a meta-learning approach designed to improve cross-tool generalization. Experimental results show that MTA significantly outperforms baseline methods on unseen tools, demonstrating its promise for building flexible and scalable systems that require dynamic tool coordination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01576v2">OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Evaluating novelty is critical yet challenging in peer review, as reviewers must assess submissions against a vast, rapidly evolving literature. This report presents OpenNovelty, an LLM-powered agentic system for transparent, evidence-based novelty analysis. The system operates through four phases: (1) extracting the core task and contribution claims to generate retrieval queries; (2) retrieving relevant prior work based on extracted queries via semantic search engine; (3) constructing a hierarchical taxonomy of core-task-related work and performing contribution-level full-text comparisons against each contribution; and (4) synthesizing all analyses into a structured novelty report with explicit citations and evidence snippets. Unlike naive LLM-based approaches, \textsc{OpenNovelty} grounds all assessments in retrieved real papers, ensuring verifiable judgments. We deploy our system on 500+ ICLR 2026 submissions with all reports publicly available on our website, and preliminary analysis suggests it can identify relevant prior work, including closely related papers that authors may overlook. OpenNovelty aims to empower the research community with a scalable tool that promotes fair, consistent, and evidence-backed peer review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13855v2">Harnessing Consistency for Robust Test-Time LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Different large language models (LLMs) exhibit diverse strengths and weaknesses, and LLM ensemble serves as a promising approach to integrate their complementary capabilities. Despite substantial progress in improving ensemble quality, limited attention has been paid to the robustness of ensembles against potential erroneous signals, which often arise from heterogeneous tokenization schemes and varying model expertise. Our analysis shows that ensemble failures typically arise from both the token level and the model level: the former reflects severe disagreement in token predictions, while the latter involves low confidence and pronounced disparities among models. In light of this, we propose CoRE, a plug-and-play technique that harnesses model consistency for robust LLM ensemble, which can be seamlessly integrated with diverse ensemble methods. *Token-level consistency* captures fine-grained disagreements by applying a low-pass filter to downweight uncertain tokens with high inconsistency, often due to token misalignment, thereby improving robustness at a granular level. *Model-level consistency* models global agreement by promoting model outputs with high self-confidence and minimal divergence from others, enhancing robustness at a coarser level. Extensive experiments across diverse benchmarks, model combinations, and ensemble strategies demonstrate that CoRE consistently improves ensemble performance and robustness. Our code is available at https://github.com/zhichenz98/CoRE-EACL26.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12641v1">STEP-LLM: Generating CAD STEP Models from Natural Language with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted to the Design, Automation & Test in Europe Conference (DATE) 2026
    </div>
    <details class="paper-abstract">
      Computer-aided design (CAD) is vital to modern manufacturing, yet model creation remains labor-intensive and expertise-heavy. To enable non-experts to translate intuitive design intent into manufacturable artifacts, recent large language models-based text-to-CAD efforts focus on command sequences or script-based formats like CadQuery. However, these formats are kernel-dependent and lack universality for manufacturing. In contrast, the Standard for the Exchange of Product Data (STEP, ISO 10303) file is a widely adopted, neutral boundary representation (B-rep) format directly compatible with manufacturing, but its graph-structured, cross-referenced nature poses unique challenges for auto-regressive LLMs. To address this, we curate a dataset of ~40K STEP-caption pairs and introduce novel preprocessing tailored for the graph-structured format of STEP, including a depth-first search-based reserialization that linearizes cross-references while preserving locality and chain-of-thought(CoT)-style structural annotations that guide global coherence. We integrate retrieval-augmented generation to ground predictions in relevant examples for supervised fine-tuning, and refine generation quality through reinforcement learning with a specific Chamfer Distance-based geometric reward. Experiments demonstrate consistent gains of our STEP-LLM in geometric fidelity over the Text2CAD baseline, with improvements arising from multiple stages of our framework: the RAG module substantially enhances completeness and renderability, the DFS-based reserialization strengthens overall accuracy, and the RL further reduces geometric discrepancy. Both metrics and visual comparisons confirm that STEP-LLM generates shapes with higher fidelity than Text2CAD. These results show the feasibility of LLM-driven STEP model generation from natural language, showing its potential to democratize CAD design for manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04700v2">PRISM: A Unified Framework for Post-Training LLMs Without Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Added open-sourced github url
    </div>
    <details class="paper-abstract">
      Current techniques for post-training Large Language Models (LLMs) rely either on costly human supervision or on external verifiers to boost performance on tasks such as mathematical reasoning and code generation. However, as LLMs improve their problem-solving, any further improvement will potentially require high-quality solutions to difficult problems that are not available to humans. As a result, learning from unlabeled data is becoming increasingly attractive in the research community. Existing methods extract learning signal from a model's consistency, either by majority voting or by converting the model's internal confidence into reward. Although internal consistency metric such as entropy or self-certainty require no human intervention, as we show in this work, these are unreliable signals for large-scale and long-term training. To address the unreliability, we propose PRISM, a unified training framework that uses a Process Reward Model (PRM) to guide learning alongside model's internal confidence in the absence of ground-truth labels. We show that effectively combining PRM with self-certainty can lead to both stable training and better test-time performance, and also keep the model's internal confidence in check. Code available at https://github.com/ghimiremukesh/PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22936v2">Evaluating Large Language Models (LLMs) in Financial NLP: A Comparative Study on Financial Report Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 23 Pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support the analysis of complex financial disclosures, yet their reliability, behavioral consistency, and transparency remain insufficiently understood in high-stakes settings. This paper presents a controlled evaluation of five transformer-based LLMs applied to question answering over the Business sections of U.S. 10-K filings. To capture complementary aspects of model behavior, we combine human evaluation, automated similarity metrics, and behavioral diagnostics under standardized and context-controlled prompting conditions. Human assessments indicate that models differ in their average performance across qualitative dimensions such as relevance, completeness, clarity, conciseness, and factual accuracy, though inter-rater agreement is modest, reflecting the subjective nature of these criteria. Automated metrics reveal systematic differences in lexical overlap and semantic similarity across models, while behavioral diagnostics highlight variation in response stability and cross-prompt alignment. Importantly, no single model consistently dominates across all evaluation perspectives. Together, these findings suggest that apparent performance differences should be interpreted as relative tendencies under the tested conditions rather than definitive indicators of general reliability. The results underscore the need for evaluation frameworks that account for human disagreement, behavioral variability, and interpretability when deploying LLMs in financially consequential applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20139v2">StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 24 pages, 8 figures, 14 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral to software development workflows, their ability to generate structured outputs has become critically important. We introduce StructEval, a comprehensive benchmark for evaluating LLMs' capabilities in producing both non-renderable (JSON, YAML, CSV) and renderable (HTML, React, SVG) structured formats. Unlike prior benchmarks, StructEval systematically evaluates structural fidelity across diverse formats through two paradigms: 1) generation tasks, producing structured output from natural language prompts, and \textbf{2)} conversion tasks, translating between structured formats. Our benchmark encompasses 18 formats and 44 types of task, with novel metrics for format adherence and structural correctness. Results reveal significant performance gaps-even state-of-the-art models like o1-mini achieve only 75.58 average score, with open-source alternatives lagging approximately 10 points behind. We find generation tasks more challenging than conversion tasks, and producing correct visual content more difficult than generating text-only structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13398v1">Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via Invertibility</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 32 pages (preprint)
    </div>
    <details class="paper-abstract">
      LLMs demonstrate strong performance on code benchmarks, yet round-trip code execution reveals limitations in their ability to maintain consistent reasoning across forward and backward execution. We present RoundTripCodeEval (RTCE), a comprehensive benchmark consisting of four distinct code execution reasoning tasks designed to rigorously test round-trip consistency. RTCE provides an execution-free, exact-match evaluation of bijection fidelity, assessing whether models preserve a consistent one-to-one mapping between encoding and decoding operations across various algorithms and directions. We systematically evaluate state-of-the-art Code-LLMs using zero-shot prompting, supervised fine-tuning on execution traces, and self-reflection mechanisms. Each yields modest improvements, but none closes the gap, indicating that current LLMs struggle with true round-trip consistency, which demonstrates that they lack the internal coherence required for trustworthy code reasoning. RTCE surfaces several new and previously unmeasured insights that are not captured by existing I/O-prediction, execution-reasoning, or round-trip natural-language benchmarks. We will release the code and the dataset upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13392v1">Beyond Memorization: Testing LLM Reasoning on Unseen Theory of Computation Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 30 pages, 11 figures, 6 tables, Work in Progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance on formal language tasks, yet whether this reflects genuine symbolic reasoning or pattern matching on familiar constructions remains unclear. We introduce a benchmark for deterministic finite automata (DFA) construction from regular languages, comprising factual knowledge questions, seen construction problems from public sources, and two types of unseen problems: hand-crafted instances with multiple interacting constraints and systematically generated problems via Arden's theorem. Models achieve perfect accuracy on factual questions and 84-90% on seen tasks. However, accuracy drops sharply on unseen problems (by 30-64%), with failures stemming from systematic misinterpretation of language constraints, incorrect handling of Kleene-star semantics, and a failure to preserve global consistency. We evaluate a three-stage hint protocol that enables correction of shallow errors but does not reliably resolve globally inconsistent or structurally flawed automata. Our analysis across multiple prompting strategies (direct, Chain-of-Thought, Tree-of-Thought) reveals that errors persist regardless of prompting approach, exposing a fundamental gap between LLMs' ability to generate syntactically plausible DFAs and their capacity for semantically correct formal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13359v1">Sockpuppetting: Jailbreaking LLMs Without Optimization Through Output Prefix Injection</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      As open-weight large language models (LLMs) increase in capabilities, safeguarding them against malicious prompts and understanding possible attack vectors becomes ever more important. While automated jailbreaking methods like GCG [Zou et al., 2023] remain effective, they often require substantial computational resources and specific expertise. We introduce "sockpuppetting'', a simple method for jailbreaking open-weight LLMs by inserting an acceptance sequence (e.g., "Sure, here is how to...'') at the start of a model's output and allowing it to complete the response. Requiring only a single line of code and no optimization, sockpuppetting achieves up to 80% higher attack success rate (ASR) than GCG on Qwen3-8B in per-prompt comparisons. We also explore a hybrid approach that optimizes the adversarial suffix within the assistant message block rather than the user prompt, increasing ASR by 64% over GCG on Llama-3.1-8B in a prompt-agnostic setting. The results establish sockpuppetting as an effective low-cost attack accessible to unsophisticated adversaries, highlighting the need for defences against output-prefix injection in open-weight models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13352v1">LLM-as-RNN: A Recurrent Language Model for Memory Updates and Sequence Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 17 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models are strong sequence predictors, yet standard inference relies on immutable context histories. After making an error at generation step t, the model lacks an updatable memory mechanism that improves predictions for step t+1. We propose LLM-as-RNN, an inference-only framework that turns a frozen LLM into a recurrent predictor by representing its hidden state as natural-language memory. This state, implemented as a structured system-prompt summary, is updated at each timestep via feedback-driven text rewrites, enabling learning without parameter updates. Under a fixed token budget, LLM-as-RNN corrects errors and retains task-relevant patterns, effectively performing online learning through language. We evaluate the method on three sequential benchmarks in healthcare, meteorology, and finance across Llama, Gemma, and GPT model families. LLM-as-RNN significantly outperforms zero-shot, full-history, and MemPrompt baselines, improving predictive accuracy by 6.5% on average, while producing interpretable, human-readable learning traces absent in standard context accumulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09822v2">LLM-Based Agentic Systems for Software Engineering: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted to GenSE 2026 workshop
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs), complex Software Engineering (SE) tasks require more collaborative and specialized approaches. This concept paper systematically reviews the emerging paradigm of LLM-based multi-agent systems, examining their applications across the Software Development Life Cycle (SDLC), from requirements engineering and code generation to static code checking, testing, and debugging. We delve into a wide range of topics such as language model selection, SE evaluation benchmarks, state-of-the-art agentic frameworks and communication protocols. Furthermore, we identify key challenges and outline future research opportunities, with a focus on multi-agent orchestration, human-agent coordination, computational cost optimization, and effective data collection. This work aims to provide researchers and practitioners with valuable insights into the current forefront landscape of agentic systems within the software engineering domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13300v1">OI-Bench: An Option Injection Benchmark for Evaluating LLM Susceptibility to Directive Interference</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Benchmarking large language models (LLMs) is critical for understanding their capabilities, limitations, and robustness. In addition to interface artifacts, prior studies have shown that LLM decisions can be influenced by directive signals such as social cues, framing, and instructions. In this work, we introduce option injection, a benchmarking approach that augments the multiple-choice question answering (MCQA) interface with an additional option containing a misleading directive, leveraging standardized choice structure and scalable evaluation. We construct OI-Bench, a benchmark of 3,000 questions spanning knowledge, reasoning, and commonsense tasks, with 16 directive types covering social compliance, bonus framing, threat framing, and instructional interference. This setting combines manipulation of the choice interface with directive-based interference, enabling systematic assessment of model susceptibility. We evaluate 12 LLMs to analyze attack success rates, behavioral responses, and further investigate mitigation strategies ranging from inference-time prompting to post-training alignment. Experimental results reveal substantial vulnerabilities and heterogeneous robustness across models. OI-Bench is expected to support more systematic evaluation of LLM robustness to directive interference within choice-based interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v2">Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13288v1">A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Production LLM systems often rely on separate models for safety and other classification-heavy steps, increasing latency, VRAM footprint, and operational complexity. We instead reuse computation already paid for by the serving LLM: we train lightweight probes on its hidden states and predict labels in the same forward pass used for generation. We frame classification as representation selection over the full token-layer hidden-state tensor, rather than committing to a fixed token or fixed layer (e.g., first-token logits or final-layer pooling). To implement this, we introduce a two-stage aggregator that (i) summarizes tokens within each layer and (ii) aggregates across layer summaries to form a single representation for classification. We instantiate this template with direct pooling, a 100K-parameter scoring-attention gate, and a downcast multi-head self-attention (MHA) probe with up to 35M trainable parameters. Across safety and sentiment benchmarks our probes improve over logit-only reuse (e.g., MULI) and are competitive with substantially larger task-specific baselines, while preserving near-serving latency and avoiding the VRAM and latency costs of a separate guard-model pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13284v1">Balancing Classification and Calibration Performance in Decision-Making LLMs via Calibration Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in decision-making tasks, where not only accuracy but also reliable confidence estimates are essential. Well-calibrated confidence enables downstream systems to decide when to trust a model and when to defer to fallback mechanisms. In this work, we conduct a systematic study of calibration in two widely used fine-tuning paradigms: supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). We show that while RLVR improves task performance, it produces extremely overconfident models, whereas SFT yields substantially better calibration, even under distribution shift, though with smaller performance gains. Through targeted experiments, we diagnose RLVR's failure, showing that decision tokens act as extraction steps of the decision in reasoning traces and do not carry confidence information, which prevents reinforcement learning from surfacing calibrated alternatives. Based on this insight, we propose a calibration-aware reinforcement learning formulation that directly adjusts decision-token probabilities. Our method preserves RLVR's accuracy level while mitigating overconfidence, reducing ECE scores up to 9 points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08856v2">LAUDE: LLM-Assisted Unit Test Generation and Debugging of Hardware DEsigns</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 18 Pages, 21 Figures, Submitted to ARR Review
    </div>
    <details class="paper-abstract">
      Unit tests are critical in the hardware design lifecycle to ensure that component design modules are functionally correct and conform to the specification before they are integrated at the system level. Thus developing unit tests targeting various design features requires deep understanding of the design functionality and creativity. When one or more unit tests expose a design failure, the debugging engineer needs to diagnose, localize, and debug the failure to ensure design correctness, which is often a painstaking and intense process. In this work, we introduce LAUDE, a unified unit-test generation and debugging framework for hardware designs that cross-pollinates the semantic understanding of the design source code with the Chain-of-Thought (CoT) reasoning capabilities of foundational Large-Language Models (LLMs). LAUDE integrates prompt engineering and design execution information to enhance its unit test generation accuracy and code debuggability. We apply LAUDE with closed- and open-source LLMs to a large corpus of buggy hardware design codes derived from the VerilogEval dataset, where generated unit tests detected bugs in up to 100% and 93% of combinational and sequential designs and debugged up to 93% and 84% of combinational and sequential designs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13264v1">Unlearning in LLMs: Methods, Evaluation, and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across natural language processing tasks, yet their widespread deployment raises pressing concerns around privacy, copyright, security, and bias. Machine unlearning has emerged as a promising paradigm for selectively removing knowledge or data from trained models without full retraining. In this survey, we provide a structured overview of unlearning methods for LLMs, categorizing existing approaches into data-centric, parameter-centric, architecture-centric, hybrid, and other strategies. We also review the evaluation ecosystem, including benchmarks, metrics, and datasets designed to measure forgetting effectiveness, knowledge retention, and robustness. Finally, we outline key challenges and open problems, such as scalable efficiency, formal guarantees, cross-language and multimodal unlearning, and robustness against adversarial relearning. By synthesizing current progress and highlighting open directions, this paper aims to serve as a roadmap for developing reliable and responsible unlearning techniques in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.17156v3">LLM-based relevance assessment still can't replace human relevance assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al make a bold claim that LLM-based relevance assessments, such as those generated by the Umbrela system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. genuinely supports their claim, particularly when the test collection is intended to serve as a benchmark for future research innovations.Second, we submit a system deliberately crafted to exploit automatic evaluation metrics, demonstrating that it can achieve artificially inflated scores without truly improving retrieval quality. Third, we simulate the consequences of circularity by analyzing Kendall's tau correlations under the hypothetical scenario in which all systems adopt Umbrela as a final-stage re-ranker, illustrating how reliance on LLM-based assessments can distort system rankings. Theoretical challenges - including the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance - that must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08121v2">Balanced Accuracy: The Right Metric for Evaluating LLM Judges -- Explained through Youden's J statistic</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Rigorous evaluation of large language models (LLMs) relies on comparing models by the prevalence of desirable or undesirable behaviors, such as task pass rates or policy violations. These prevalence estimates are produced by a classifier, either an LLM-as-a-judge or human annotators, making the choice of classifier central to trustworthy evaluation. Common metrics used for this choice, such as Accuracy, Precision, and F1, are sensitive to class imbalance and to arbitrary choices of positive class, and can favor judges that distort prevalence estimates. We show that Youden's $J$ statistic is theoretically aligned with choosing the best judge to compare models, and that Balanced Accuracy is an equivalent linear transformation of $J$. Through both analytical arguments and empirical examples and simulations, we demonstrate how selecting judges using Balanced Accuracy leads to better, more robust classifier selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13243v1">A Comprehensive Evaluation of LLM Reasoning: From Single-Model to Multi-Agent Paradigms</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed as reasoning systems, where reasoning paradigms - such as Chain-of-Thought (CoT) and multi-agent systems (MAS) - play a critical role, yet their relative effectiveness and cost-accuracy trade-offs remain poorly understood. In this work, we conduct a comprehensive and unified evaluation of reasoning paradigms, spanning direct single-model generation, CoT-augmented single-model reasoning, and representative MAS workflows, characterizing their reasoning performance across a diverse suite of closed-form benchmarks. Beyond overall performance, we probe role-specific capability demands in MAS using targeted role isolation analyses, and analyze cost-accuracy trade-offs to identify which MAS workflows offer a favorable balance between cost and accuracy, and which incur prohibitive overhead for marginal gains. We further introduce MIMeBench, a new open-ended benchmark that targets two foundational yet underexplored semantic capabilities - semantic abstraction and contrastive discrimination - thereby providing an alternative evaluation axis beyond closed-form accuracy and enabling fine-grained assessment of semantic competence that is difficult to capture with existing benchmarks. Our results show that increased structural complexity does not consistently lead to improved reasoning performance, with its benefits being highly dependent on the properties and suitability of the reasoning paradigm itself. The codes are released at https://gitcode.com/HIT1920/OpenLLMBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.19202v2">Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 32 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17091v2">Comparing the Framing Effect in Humans and LLMs on Naturally Occurring Texts</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Humans are influenced by how information is presented, a phenomenon known as the framing effect. Prior work suggests that LLMs may also be susceptible to framing, but it has relied on synthetic data and did not compare to human behavior. To address this gap, we introduce WildFrame - a dataset for evaluating LLM responses to positive and negative framing in naturally-occurring sentences, alongside human responses on the same data. WildFrame consists of 1,000 real-world texts selected to convey a clear sentiment; we then reframe each text in either a positive or negative light and collect human sentiment annotations. Evaluating eleven LLMs on WildFrame, we find that all models respond to reframing in a human-like manner ($r\geq0.52$), and that both humans and models are influenced more by positive than negative reframing. Notably, GPT models are the least correlated with human behavior among all tested models. These findings raise a discussion around the goals of state-of-the-art LLM development and whether models should align closely with human behavior, to preserve cognitive phenomena such as the framing effect, or instead mitigate such biases in favor of fairness and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13206v1">Real-Time Deadlines Reveal Temporal Awareness Failures in LLM Strategic Dialogues</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text token-by-token in discrete time, yet real-world communication, from therapy sessions to business negotiations, critically depends on continuous time constraints. Current LLM architectures and evaluation protocols rarely test for temporal awareness under real-time deadlines. We use simulated negotiations between paired agents under strict deadlines to investigate how LLMs adjust their behavior in time-sensitive settings. In a control condition, agents know only the global time limit. In a time-aware condition, they receive remaining-time updates at each turn. Deal closure rates are substantially higher (32\% vs. 4\% for GPT-5.1) and offer acceptances are sixfold higher in the time-aware condition than in the control, suggesting LLMs struggle to internally track elapsed time. However, the same LLMs achieve near-perfect deal closure rates ($\geq$95\%) under turn-based limits, revealing the failure is in temporal tracking rather than strategic reasoning. These effects replicate across negotiation scenarios and models, illustrating a systematic lack of LLM time awareness that will constrain LLM deployment in many time-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13155v1">Probe and Skip: Self-Predictive Token Skipping for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Long-context inference enhances the reasoning capability of Large Language Models (LLMs) while incurring significant computational overhead. Token-oriented methods, such as pruning and skipping, have shown promise in reducing inference latency, but still suffer from inherently limited acceleration potential, outdated proxy signals, and redundancy interference, thus yielding suboptimal speed-accuracy trade-offs. To address these challenges, we propose SPTS (Self-Predictive Token Skipping), a training-free framework for efficient long-context LLM inference. Specifically, motivated by the thought of probing the influence of targeted skipping layers, we design two component-specific strategies for selective token skipping: Partial Attention Probing (PAP) for multi-head attention, which selects informative tokens by performing partial forward attention computation, and Low-rank Transformation Probing (LTP) for feed forward network, which constructs a low-rank proxy network to predict token transformations. Furthermore, a Multi-Stage Delayed Pruning (MSDP) strategy reallocates the skipping budget and progressively prunes redundant tokens across layers. Extensive experiments demonstrate the effectiveness of our method, achieving up to 2.46$\times$ and 2.29$\times$ speedups for prefilling and end-to-end generation, respectively, while maintaining state-of-the-art model performance. The source code will be publicly available upon paper acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16029v4">FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16475v2">LLM-Glasses: GenAI-driven Glasses with Haptic Feedback for Navigation of Visually Impaired People</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      LLM-Glasses is a wearable navigation system which assists visually impaired people by utilizing YOLO-World object detection, GPT-4o-based reasoning, and haptic feedback for real-time guidance. The device translates visual scene understanding into intuitive tactile feedback on the temples, allowing hands-free navigation. Three studies evaluate the system: recognition of 13 haptic patterns with an average recognition rate of 81.3%, VICON-based guidance with predefined paths using haptic cues, and an LLM-guided scene evaluation with decision accuracies of 91.8% without obstacles, 84.6% with static obstacles, and 81.5% with dynamic obstacles. These results show that LLM-Glasses can deliver reliable navigation support in controlled environments and motivate further work on responsiveness and deployment in more complex real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20209v2">Improving the OOD Performance of Closed-Source LLMs on NLI Through Strategic Data Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Accepted at EACL Findings 2026
    </div>
    <details class="paper-abstract">
      We investigate the robustness of fine-tuned Large Language Models (LLMs) for the task of Natural Language Inference (NLI), finding that the in-distribution gains from fine-tuning correspond to a large drop in out-of-distribution (OOD) performance. Despite the widespread use of closed-source LLMs, there are no robustness mitigation methods that work under their API fine-tuning constraints. Existing methods to improve robustness typically require changing the fine-tuning process or large-scale data augmentation, methods that are infeasible or cost prohibitive for closed-source models. To address this, we propose strategically selecting the NLI fine-tuning data, prioritising more complex examples or replacing existing training examples with LLM-generated data. Prioritising more complex training examples improves performance on challenging OOD NLI datasets, while training with synthetic data leads to substantial improvements on easier OOD datasets. We find that synthetic examples are often too simple, and by prompting LLMs to create more complex synthetic data we can improve performance on both easy and challenging OOD datasets. Finally, we show that recent autoregressive LLMs are substantially more robust to distributional shifts compared to encoder models, and should be a preferred baseline for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13114v1">IntAgent: NWDAF-Based Intent LLM Agent Towards Advanced Next Generation Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 conference
    </div>
    <details class="paper-abstract">
      Intent-based networks (IBNs) are gaining prominence as an innovative technology that automates network operations through high-level request statements, defining what the network should achieve. In this work, we introduce IntAgent, an intelligent intent LLM agent that integrates NWDAF analytics and tools to fulfill the network operator's intents. Unlike previous approaches, we develop an intent tools engine directly within the NWDAF analytics engine, allowing our agent to utilize live network analytics to inform its reasoning and tool selection. We offer an enriched, 3GPP-compliant data source that enhances the dynamic, context-aware fulfillment of network operator goals, along with an MCP tools server for scheduling, monitoring, and analytics tools. We demonstrate the efficacy of our framework through two practical use cases: ML-based traffic prediction and scheduled policy enforcement, which validate IntAgent's ability to autonomously fulfill complex network intents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13099v1">Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Project resources will be available here: https://github.com/UBC-NLP/Alexandria
    </div>
    <details class="paper-abstract">
      Arabic is a highly diglossic language where most daily communication occurs in regional dialects rather than Modern Standard Arabic. Despite this, machine translation (MT) systems often generalize poorly to dialectal input, limiting their utility for millions of speakers. We introduce \textbf{Alexandria}, a large-scale, community-driven, human-translated dataset designed to bridge this gap. Alexandria covers 13 Arab countries and 11 high-impact domains, including health, education, and agriculture. Unlike previous resources, Alexandria provides unprecedented granularity by associating contributions with city-of-origin metadata, capturing authentic local varieties beyond coarse regional labels. The dataset consists of multi-turn conversational scenarios annotated with speaker-addressee gender configurations, enabling the study of gender-conditioned variation in dialectal use. Comprising 107K total samples, Alexandria serves as both a training resource and a rigorous benchmark for evaluating MT and Large Language Models (LLMs). Our automatic and human evaluation of Arabic-aware LLMs benchmarks current capabilities in translating across diverse Arabic dialects and sub-dialects, while exposing significant persistent challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13096v1">LLM-VLM Fusion Framework for Autonomous Maritime Port Inspection using a Heterogeneous UAV-USV System</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 submitted in AEJ
    </div>
    <details class="paper-abstract">
      Maritime port inspection plays a critical role in ensuring safety, regulatory compliance, and operational efficiency in complex maritime environments. However, existing inspection methods often rely on manual operations and conventional computer vision techniques that lack scalability and contextual understanding. This study introduces a novel integrated engineering framework that utilizes the synergy between Large Language Models (LLMs) and Vision Language Models (VLMs) to enable autonomous maritime port inspection using cooperative aerial and surface robotic platforms. The proposed framework replaces traditional state-machine mission planners with LLM-driven symbolic planning and improved perception pipelines through VLM-based semantic inspection, enabling context-aware and adaptive monitoring. The LLM module translates natural language mission instructions into executable symbolic plans with dependency graphs that encode operational constraints and ensure safe UAV-USV coordination. Meanwhile, the VLM module performs real-time semantic inspection and compliance assessment, generating structured reports with contextual reasoning. The framework was validated using the extended MBZIRC Maritime Simulator with realistic port infrastructure and further assessed through real-world robotic inspection trials. The lightweight on-board design ensures suitability for resource-constrained maritime platforms, advancing the development of intelligent, autonomous inspection systems. Project resources (code and videos) can be found here: https://github.com/Muhayyuddin/llm-vlm-fusion-port-inspection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13082v1">Adversarial News and Lost Profits: Manipulating Headlines in LLM-Driven Algorithmic Trading</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 This work has been accepted for publication at the IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). The final version will be available on IEEE Xplore
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted in the financial domain. Their exceptional capabilities to analyse textual data make them well-suited for inferring the sentiment of finance-related news. Such feedback can be leveraged by algorithmic trading systems (ATS) to guide buy/sell decisions. However, this practice bears the risk that a threat actor may craft "adversarial news" intended to mislead an LLM. In particular, the news headline may include "malicious" content that remains invisible to human readers but which is still ingested by the LLM. Although prior work has studied textual adversarial examples, their system-wide impact on LLM-supported ATS has not yet been quantified in terms of monetary risk. To address this threat, we consider an adversary with no direct access to an ATS but able to alter stock-related news headlines on a single day. We evaluate two human-imperceptible manipulations in a financial context: Unicode homoglyph substitutions that misroute models during stock-name recognition, and hidden-text clauses that alter the sentiment of the news headline. We implement a realistic ATS in Backtrader that fuses an LSTM-based price forecast with LLM-derived sentiment (FinBERT, FinGPT, FinLLaMA, and six general-purpose LLMs), and quantify monetary impact using portfolio metrics. Experiments on real-world data show that manipulating a one-day attack over 14 months can reliably mislead LLMs and reduce annual returns by up to 17.7 percentage points. To assess real-world feasibility, we analyze popular scraping libraries and trading platforms and survey 27 FinTech practitioners, confirming our hypotheses. We notified trading platform owners of this security issue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13024v1">Tears or Cheers? Benchmarking LLMs via Culturally Elicited Distinct Affective Responses</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 24 pages, 10 figures, 9 Tables
    </div>
    <details class="paper-abstract">
      Culture serves as a fundamental determinant of human affective processing and profoundly shapes how individuals perceive and interpret emotional stimuli. Despite this intrinsic link extant evaluations regarding cultural alignment within Large Language Models primarily prioritize declarative knowledge such as geographical facts or established societal customs. These benchmarks remain insufficient to capture the subjective interpretative variance inherent to diverse sociocultural lenses. To address this limitation, we introduce CEDAR, a multimodal benchmark constructed entirely from scenarios capturing Culturally \underline{\textsc{E}}licited \underline{\textsc{D}}istinct \underline{\textsc{A}}ffective \underline{\textsc{R}}esponses. To construct CEDAR, we implement a novel pipeline that leverages LLM-generated provisional labels to isolate instances yielding cross-cultural emotional distinctions, and subsequently derives reliable ground-truth annotations through rigorous human evaluation. The resulting benchmark comprises 10,962 instances across seven languages and 14 fine-grained emotion categories, with each language including 400 multimodal and 1,166 text-only samples. Comprehensive evaluations of 17 representative multilingual models reveal a dissociation between language consistency and cultural alignment, demonstrating that culturally grounded affective understanding remains a significant challenge for current models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13015v1">MeltRTL: Multi-Expert LLMs with Inference-time Intervention for RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      The automated generation of hardware register-transfer level (RTL) code with large language models (LLMs) shows promise, yet current solutions struggle to produce syntactically and functionally correct code for complex digital designs. This paper introduces MeltRTL, a novel framework that integrates multi-expert attention with inference-time intervention (ITI) to significantly improve LLM-based RTL code generation accuracy without retraining the base model. MeltRTL introduces three key innovations: (1) A multi-expert attention architecture that dynamically routes design specifications to specialized expert networks, enabling targeted reasoning across various hardware categories; (2) An inference-time intervention mechanism that employs non-linear probes to detect and correct hardware-specific inaccuracies during generation; and (3) An efficient intervention framework that selectively operates on expert-specific attention heads with minimal computational overhead. We evaluate MeltRTL on the VerilogEval benchmark, achieving 96% synthesizability and 60% functional correctness, compared to the base LLM's 85.3% and 45.3%, respectively. These improvements are obtained entirely at inference time, with only 27% computational overhead and no model fine-tuning, making MeltRTL immediately deployable on existing pre-trained LLMs. Ablation studies further show the complementary benefits of multi-expert architecture and ITI, highlighting their synergistic effects when combined.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13007v1">ArchAgent: Scalable Legacy Software Architecture Recovery with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 to be published in ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recovering accurate architecture from large-scale legacy software is hindered by architectural drift, missing relations, and the limited context of Large Language Models (LLMs). We present ArchAgent, a scalable agent-based framework that combines static analysis, adaptive code segmentation, and LLM-powered synthesis to reconstruct multiview, business-aligned architectures from cross-repository codebases. ArchAgent introduces scalable diagram generation with contextual pruning and integrates cross-repository data to identify business-critical modules. Evaluations of typical large-scale GitHub projects show significant improvements over existing benchmarks. An ablation study confirms that dependency context improves the accuracy of generated architectures of production-level repositories, and a real-world case study demonstrates effective recovery of critical business logics from legacy projects. The dataset is available at https://github.com/panrusheng/arch-eval-benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12974v1">Bridging the Knowledge-Action Gap by Evaluating LLMs in Dynamic Dental Clinical Scenarios</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 29 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The transition of Large Language Models (LLMs) from passive knowledge retrievers to autonomous clinical agents demands a shift in evaluation-from static accuracy to dynamic behavioral reliability. To explore this boundary in dentistry, a domain where high-quality AI advice uniquely empowers patient-participatory decision-making, we present the Standardized Clinical Management & Performance Evaluation (SCMPE) benchmark, which comprehensively assesses performance from knowledge-oriented evaluations (static objective tasks) to workflow-based simulations (multi-turn simulated patient interactions). Our analysis reveals that while models demonstrate high proficiency in static objective tasks, their performance precipitates in dynamic clinical dialogues, identifying that the primary bottleneck lies not in knowledge retention, but in the critical challenges of active information gathering and dynamic state tracking. Mapping "Guideline Adherence" versus "Decision Quality" reveals a prevalent "High Efficacy, Low Safety" risk in general models. Furthermore, we quantify the impact of Retrieval-Augmented Generation (RAG). While RAG mitigates hallucinations in static tasks, its efficacy in dynamic workflows is limited and heterogeneous, sometimes causing degradation. This underscores that external knowledge alone cannot bridge the reasoning gap without domain-adaptive pre-training. This study empirically charts the capability boundaries of dental LLMs, providing a roadmap for bridging the gap between standardized knowledge and safe, autonomous clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02370v3">Variance-Aware LLM Annotation for Strategy Research: Sources, Diagnostics, and a Protocol for Reliable Measurement</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 41 pages for the main paper 53 pages for appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strategy researchers powerful tools for annotating text at scale, but treating LLM-generated labels as deterministic overlooks substantial instability. Grounded in content analysis and generalizability theory, we diagnose five variance sources: construct specification, interface effects, model preferences, output extraction, and system-level aggregation. Empirical demonstrations show that minor design choices-prompt phrasing, model selection-can shift outcomes by 12-85 percentage points. Such variance threatens not only reproducibility but econometric identification: annotation errors correlated with covariates bias parameter estimates regardless of average accuracy. We develop a variance-aware protocol specifying sampling budgets, aggregation rules, and reporting standards, and delineate scope conditions where LLM annotation should not be used. These contributions transform LLM-based annotation from ad hoc practice into auditable measurement infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17825v2">FAIRGAMER: Evaluating Social Biases in LLM-Based Video Game NPCs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have increasingly enhanced or replaced traditional Non-Player Characters (NPCs) in video games. However, these LLM-based NPCs inherit underlying social biases (e.g., race or class), posing fairness risks during in-game interactions. To address the limited exploration of this issue, we introduce FairGamer, the first benchmark to evaluate social biases across three interaction patterns: transaction, cooperation, and competition. FairGamer assesses four bias types, including class, race, age, and nationality, across 12 distinct evaluation tasks using a novel metric, FairMCV. Our evaluation of seven frontier LLMs reveals that: (1) models exhibit biased decision-making, with Grok-4-Fast demonstrating the highest bias (average FairMCV = 76.9%); and (2) larger LLMs display more severe social biases, suggesting that increased model capacity inadvertently amplifies these biases. We release FairGamer at https://github.com/Anonymous999-xxx/FairGamer to facilitate future research on NPC fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12921v1">Injecting Knowledge from Social Science Journals to Improve Indonesian Cultural Understanding by LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Recently there have been intensifying efforts to improve the understanding of Indonesian cultures by large language models (LLMs). An attractive source of cultural knowledge that has been largely overlooked is local journals of social science, which likely contain substantial cultural studies from a native perspective. We present a novel text dataset of journal article passages, created from 151 open-source Indonesian social science journals, called IndoSoSci. We demonstrate an effective recipe for injecting Indonesian cultural knowledge therein into LLMs: extracting the facts related to Indonesian culture, and apply retrieval-augmented generation (RAG) with LLM-generated hypothetical documents as queries during retrieval. The proposed recipe yields strong performance gains over several strong baselines on the IndoCulture benchmark. Additionally, by combining IndoSoSci with Indonesian Wikipedia, we set a new state-of-the-art accuracy on the IndoCulture benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12917v1">CooperLLM: Cloud-Edge-End Cooperative Federated Fine-tuning for LLMs via ZOO-based Gradient Correction</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 14 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perform well on many NLP tasks, but fine-tuning them on resource-constrained mobile devices is challenging due to high memory and computation costs, despite growing demands for privacy-preserving personalization. Federated Learning (FL) enables local-data training, yet existing methods either rely on memory-intensive backpropagation or use zeroth-order optimization (ZOO), which avoids backward passes but suffers from slow convergence and degraded accuracy. We propose CooperLLM, a cloud-assisted edge-end cooperative federated fine-tuning framework that combines ZOO on mobile devices with cloud-guided gradient rectification. Mobile clients perform lightweight ZOO updates on private data, while the cloud fine-tunes on auxiliary public data using backpropagation and injects guided perturbations to rectify local updates, improving convergence and accuracy without violating privacy. To address system bottlenecks, CooperLLM introduces pipeline scheduling and adaptive compression to overlap computation and communication and reduce memory usage. Experiments on multiple Transformer models and datasets show that CooperLLM reduces on-device memory by up to $86.4\%$, accelerates convergence by $8.8 \times$, and improves accuracy by up to 10 percentage points over state-of-the-art ZOO-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01724v3">ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      The NP-hard Dynamic Flexible Job-Shop Scheduling (DFJSP) problem involves real-time events and complex routing. While traditional rules are efficient but rigid, deep learning is opaque and requires feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed Strategic Experience. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments demonstrate ReflecSched achieves superior performance, with its best variants attaining an average RPD of 6.09% and rank of 4.39 on GEN-Bench, significantly outperforming strong traditional and learning-based methods including HMPSAC and IDDQN. It also statistically and decisively surpasses direct LLM baselines, securing a 71.35% Win Rate while being, on average, 15.1% more token-efficient on Normal-scale problems. Furthermore, cumulative runtime analysis reveals that ReflecSched's zero-shot nature eliminates the training bottleneck, providing a decisive efficiency advantage in high-variability manufacturing environments. Ablation studies attribute this performance to a robust reflection mechanism that leverages high-quality, contrastive experience. Ultimately, the framework's performance is statistically on par with an oracle-like strategy, showcasing its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02751v2">SmallKV: Small Model Assisted Compensation of KV Cache Compression for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      KV cache eviction has emerged as an effective solution to alleviate resource constraints faced by LLMs in long-context scenarios. However, existing token-level eviction methods often overlook two critical aspects: (1) their irreversible eviction strategy fails to adapt to dynamic attention patterns during decoding (the saliency shift problem), and (2) they treat both marginally important tokens and truly unimportant tokens equally, despite the collective significance of marginal tokens to model performance (the marginal information over-compression problem). To address these issues, we design two compensation mechanisms based on the high similarity of attention matrices between LLMs of different scales. We propose SmallKV, a small model assisted compensation method for KV cache compression. SmallKV can maintain attention matching between different-scale LLMs to: 1) assist the larger model in perceiving globally important information of attention; and 2) use the smaller model's attention scores to approximate those of marginal tokens in the larger model. Extensive experiments on benchmarks including GSM8K, BBH, MT-Bench, and LongBench demonstrate the effectiveness of SmallKV. Moreover, efficiency evaluations show that SmallKV achieves 1.75 - 2.56 times higher throughput than baseline methods, highlighting its potential for efficient and performant LLM inference in resource constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12845v1">Automatic Generation of Formal Specification and Verification Annotations Using LLMs and Test Oracles</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Recent verification tools aim to make formal verification more accessible to software engineers by automating most of the verification process. However, annotating conventional programs with the formal specification and verification constructs (preconditions, postconditions, loop invariants, auxiliary predicates and functions and proof helpers) required to prove their correctness still demands significant manual effort and expertise. This paper investigates how LLMs can automatically generate such annotations for programs written in Dafny, a verification-aware programming language, starting from conventional code accompanied by natural language specifications (in comments) and test code. In experiments on 110 Dafny programs, a multimodel approach combining Claude Opus 4.5 and GPT-5.2 generated correct annotations for 98.2% of the programs within at most 8 repair iterations, using verifier feedback. A logistic regression analysis shows that proof-helper annotations contribute disproportionately to problem difficulty for current LLMs. Assertions in the test cases served as static oracles to automatically validate the generated pre/postconditions. We also compare generated and manual solutions and present an extension for Visual Studio Code to incorporate automatic generation into the IDE, with encouraging usability feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11408v2">KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning?</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 Pre-print; Accepted to TACL
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) traces have been shown to improve performance of large language models on a plethora of reasoning tasks, yet there is no consensus on the mechanism by which this boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGraphs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in language-model outputs. A collection of 1671 mathematical reasoning problems from MATH500, GSM8K, and AIME, together with their associated CCGraphs, has been compiled into our dataset -- KisMATH. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCGraphs are causal contributors to the final answer, which we argue is constitutive of reasoning; and (ii) LLMs emphasize the reasoning paths captured by the CCGraphs, indicating that the models internally realize structures similar to our graphs. KisMATH enables controlled, graph-aligned interventions and opens avenues for further investigation into the role of CoT in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03385v2">SIGMA: Scalable Spectral Insights for LLM Model Collapse</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      The rapid adoption of synthetic data for training Large Language Models (LLMs) has introduced the technical challenge of "model collapse"-a degenerative process where recursive training on model-generated content leads to a contraction of distributional variance and representational quality. While the phenomenology of collapse is increasingly evident, rigorous methods to quantify and predict its onset in high-dimensional spaces remain elusive. In this paper, we introduce SIGMA (Spectral Inequalities for Gram Matrix Analysis), a unified framework that benchmarks model collapse through the spectral lens of the embedding Gram matrix. By deriving and utilizing deterministic and stochastic bounds on the matrix's spectrum, SIGMA provides a mathematically grounded metric to track the contraction of the representation space. Crucially, our stochastic formulation enables scalable estimation of these bounds, making the framework applicable to large-scale foundation models where full eigendecomposition is intractable. We demonstrate that SIGMA effectively captures the transition towards degenerate states, offering both theoretical insights into the mechanics of collapse and a practical, scalable tool for monitoring the health of recursive training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12805v1">SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07968v2">From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across various applications, but their deployment in real-world settings faces several risks, including jailbreak attacks and privacy leaks. To mitigate these risks, numerous defense strategies have been proposed. However, most existing studies assess these defenses in isolation and ignore their effects on other risk dimensions. In this work, we introduce a new cross-risk evaluation paradigm and take the first step in investigating unintended interactions among defenses in LLMs. Specifically, we focus on the interplay between safety, fairness, and privacy. To this end, we propose CrossRiskEval, a framework that systematically characterizes how a defense designed for one risk (e.g., safety) affects others (e.g., fairness or privacy). We conduct extensive empirical studies and mechanistic analyses on 14 LLMs with deployed defenses, covering 12 defense strategies. Our results show that defenses targeting a single risk often cause measurable effects on other risks. These effects vary in direction and magnitude across a range of factors (e.g., models, tasks, and defense strategies), and are often asymmetric across risk pairs. Furthermore, our mechanistic analysis shows that these interactions are not random: they arise from conflict-entangled neurons, which are shared internal representations that contribute in opposite ways to different risks. Adjusting one risk therefore perturbs these representations and leads to systematic changes in non-target risks. These findings reveal the limits of single-risk evaluation and highlight the need for holistic and interaction-aware assessment when designing and deploying LLM defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11288v3">Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1\% to 24\% depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection. We formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety reduces EM while prioritizing context-following increases it. These findings establish ICL as a previously underappreciated vector for emergent misalignment that operates without parameter modification and resists simple scaling-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22510v2">We Think, Therefore We Align LLMs to Helpful, Harmless and Honest Before They Go Wrong</a></div>
    <div class="paper-meta">
      📅 2026-01-19
    </div>
    <details class="paper-abstract">
      Alignment of Large Language Models (LLMs) along multiple objectives-helpfulness, harmlessness, and honesty (HHH)-is critical for safe and reliable deployment. Prior work has used steering vector-small control signals injected into hidden states-to guide LLM outputs, typically via one-to-one (1-to-1) Transformer decoders. In this setting, optimizing a single alignment objective can inadvertently overwrite representations learned for other objectives, leading to catastrophic forgetting. More recent approaches extend steering vectors via one-to-many (1-to-N) Transformer decoders. While this alleviates catastrophic forgetting, naive multi-branch designs optimize each objective independently, which can cause inference fragmentation-outputs across HHH objectives may become inconsistent. We propose Adaptive Multi-Branch Steering (AMBS), a two-stage 1-to-N framework for unified and efficient multi-objective alignment. In Stage I, post-attention hidden states of the Transformer layer are computed once to form a shared representation. In Stage II, this representation is cloned into parallel branches and steered via a policy-reference mechanism, enabling objective-specific control while maintaining cross-objective consistency. Empirical evaluations on Alpaca, BeaverTails, and TruthfulQA show that AMBS consistently improves HHH alignment across multiple 7B LLM backbones. For example, on DeepSeek-7B, AMBS improves average alignment scores by +32.4% and reduces unsafe outputs by 11.0% compared to a naive 1-to-N baseline, while remaining competitive with state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14311v1">Tracing the Data Trail: A Survey of Data Provenance, Transparency and Traceability in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-19
      | 💬 35 pages, 6 figures. Manuscript submitted to ACM Computing Surveys (CSUR) on the 12th of December 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are deployed at scale, yet their training data life cycle remains opaque. This survey synthesizes research from the past ten years on three tightly coupled axes: (1) data provenance, (2) transparency, and (3) traceability, and three supporting pillars: (4) bias \& uncertainty, (5) data privacy, and (6) tools and techniques that operationalize them. A central contribution is a proposed taxonomy defining the field's domains and listing corresponding artifacts. Through analysis of 95 publications, this work identifies key methodologies concerning data generation, watermarking, bias measurement, data curation, data privacy, and the inherent trade-off between transparency and opacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18825v3">EconEvals: Benchmarks and Litmus Tests for Economic Decision-Making by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 Major revision with updated experiments and analysis
    </div>
    <details class="paper-abstract">
      We develop evaluation methods for measuring the economic decision-making capabilities and tendencies of LLMs. First, we develop benchmarks derived from key problems in economics -- procurement, scheduling, and pricing -- that test an LLM's ability to learn from the environment in context. Second, we develop the framework of litmus tests, evaluations that quantify an LLM's choice behavior on a stylized decision-making task with multiple conflicting objectives. Each litmus test outputs a litmus score, which quantifies an LLM's tradeoff response, a reliability score, which measures the coherence of an LLM's choice behavior, and a competency score, which measures an LLM's capability at the same task when the conflicting objectives are replaced by a single, well-specified objective. Evaluating a broad array of frontier LLMs, we (1) investigate changes in LLM capabilities and tendencies over time, (2) derive economically meaningful insights from the LLMs' choice behavior and chain-of-thought, (3) validate our litmus test framework by testing self-consistency, robustness, and generalizability. Overall, this work provides a foundation for evaluating LLM agents as they are further integrated into economic decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12549v1">Benchmarking Concept-Spilling Across Languages in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) exhibit remarkable cross-lingual abilities, yet often exhibit a systematic bias toward the representations from other languages, resulting in semantic interference when generating content in non-English languages$-$a phenomenon we define as language spilling. This paper presents a novel comparative framework for evaluating multilingual semantic robustness by systematically measuring how models handle polysemous words across languages. Our methodology provides a relative measure of model performance: when required to generate exactly five meanings, both strong and weak models may resort to meanings from dominant languages, but semantically stronger models do so later in the generation sequence, producing more true meanings from the target language before failing, while weaker models resort to dominant-language meanings earlier in the sequence. We evaluate a diverse set of open and closed multilingual LLMs using a structured meaning generation task across nine languages, employing a carefully curated benchmark of 100 high-polysemy English words. Our findings reveal significant variation in semantic robustness across both models and languages, providing a principled ranking system for model comparison without requiring definitive causal attribution of error sources. We contribute both a scalable comparative benchmark for multilingual semantic evaluation and a rigorous validation pipeline$-$critical tools for developing more linguistically balanced AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07772v3">An approach for systematic decomposition of complex llm tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from reliability issues on complex tasks, as existing decomposition methods are heuristic and rely on agent or manual decomposition. This work introduces a novel, systematic decomposition framework that we call Analysis of CONstraint-Induced Complexity (ACONIC), which models the task as a constraint problem and leverages formal complexity measures to guide decomposition. On combinatorial (SAT-Bench) and LLM database querying tasks (Spider), we find that by decomposing the tasks following the measure of complexity, agent can perform considerably better.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01553v2">MedQA-CS: Objective Structured Clinical Examination (OSCE)-Style Benchmark for Evaluating LLM Clinical Skills</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 To appear in proceedings of the Main Conference of the European Chapter of the Association for Computational Linguistics (EACL) 2026
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) and large language models (LLMs) in healthcare require advanced clinical skills (CS), yet current benchmarks fail to evaluate these comprehensively. We introduce MedQA-CS, an AI-SCE framework inspired by medical education's Objective Structured Clinical Examinations (OSCEs), to address this gap. MedQA-CS evaluates LLMs through two instruction-following tasks, LLM-as-medical-student and LLM-as-CS-examiner, designed to reflect real clinical scenarios. Our contributions include developing MedQA-CS, a comprehensive evaluation framework with publicly available data and expert annotations, and providing the quantitative and qualitative assessment of LLMs as reliable judges in CS evaluation. Our experiments show that MedQA-CS is a more challenging benchmark for evaluating clinical skills than traditional multiple-choice QA benchmarks (e.g., MedQA). Combined with existing benchmarks, MedQA-CS enables a more comprehensive evaluation of LLMs' clinical capabilities for both open- and closed-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12471v1">Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 Equal contribution for the first two authors; To appear in proceedings of the Main Conference of the European Chapter of the Association for Computational Linguistics (EACL) 2026
    </div>
    <details class="paper-abstract">
      Current evaluation of large language models (LLMs) overwhelmingly prioritizes accuracy; however, in real-world and safety-critical applications, the ability to abstain when uncertain is equally vital for trustworthy deployment. We introduce MedAbstain, a unified benchmark and evaluation protocol for abstention in medical multiple-choice question answering (MCQA) -- a discrete-choice setting that generalizes to agentic action selection -- integrating conformal prediction, adversarial question perturbations, and explicit abstention options. Our systematic evaluation of both open- and closed-source LLMs reveals that even state-of-the-art, high-accuracy models often fail to abstain with uncertain. Notably, providing explicit abstention options consistently increases model uncertainty and safer abstention, far more than input perturbations, while scaling model size or advanced prompting brings little improvement. These findings highlight the central role of abstention mechanisms for trustworthy LLM deployment and offer practical guidance for improving safety in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12460v1">TrojanPraise: Jailbreak LLMs via Benign Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      The demand of customized large language models (LLMs) has led to commercial LLMs offering black-box fine-tuning APIs, yet this convenience introduces a critical security loophole: attackers could jailbreak the LLMs by fine-tuning them with malicious data. Though this security issue has recently been exposed, the feasibility of such attacks is questionable as malicious training dataset is believed to be detectable by moderation models such as Llama-Guard-3. In this paper, we propose TrojanPraise, a novel finetuning-based attack exploiting benign and thus filter-approved data. Basically, TrojanPraise fine-tunes the model to associate a crafted word (e.g., "bruaf") with harmless connotations, then uses this word to praise harmful concepts, subtly shifting the LLM from refusal to compliance. To explain the attack, we decouple the LLM's internal representation of a query into two dimensions of knowledge and attitude. We demonstrate that successful jailbreak requires shifting the attitude while avoiding knowledge shift, a distortion in the model's understanding of the concept. To validate this attack, we conduct experiments on five opensource LLMs and two commercial LLMs under strict black-box settings. Results show that TrojanPraise achieves a maximum attack success rate of 95.88% while evading moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24314v2">QianfanHuijin Technical Report: A Novel Multi-Stage Training Paradigm for Finance Industrial LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Domain-specific enhancement of Large Language Models (LLMs) within the financial context has long been a focal point of industrial application. While previous models such as BloombergGPT and Baichuan-Finance primarily focused on knowledge enhancement, the deepening complexity of financial services has driven a growing demand for models that possess not only domain knowledge but also robust financial reasoning and agentic capabilities. In this paper, we present QianfanHuijin, a financial domain LLM, and propose a generalizable multi-stage training paradigm for industrial model enhancement. Our approach begins with Continual Pre-training (CPT) on financial corpora to consolidate the knowledge base. This is followed by a fine-grained Post-training pipeline designed with increasing specificity: starting with Financial SFT, progressing to Finance Reasoning RL and Finance Agentic RL, and culminating in General RL aligned with real-world business scenarios. Empirical results demonstrate that QianfanHuijin achieves superior performance across various authoritative financial benchmarks. Furthermore, ablation studies confirm that the targeted Reasoning RL and Agentic RL stages yield significant gains in their respective capabilities. These findings validate our motivation and suggest that this fine-grained, progressive post-training methodology is poised to become a mainstream paradigm for various industrial-enhanced LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16272v2">Beyond Blind Spots: Analytic Hints for Mitigating LLM-Based Evaluation Pitfalls</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed as judges (LaaJ) in code generation pipelines. While attractive for scalability, LaaJs tend to overlook domain specific issues raising concerns about their reliability in critical evaluation tasks. To better understand these limitations in practice, we examine LaaJ behavior in a concrete industrial use case: legacy code modernization via COBOL code generation. In this setting, we find that even production deployed LaaJs can miss domain critical errors, revealing consistent blind spots in their evaluation capabilities. To better understand these blind spots, we analyze generated COBOL programs and associated LaaJs judgments, drawing on expert knowledge to construct a preliminary taxonomy. Based on this taxonomy, we develop a lightweight analytic checker tool that flags over 30 domain specific issues observed in practice. We use its outputs as analytic hints, dynamically injecting them into the judges prompt to encourage LaaJ to revisit aspects it may have overlooked. Experiments on a test set of 100 programs using four production level LaaJs show that LaaJ alone detects only about 45-63% of the errors present in the code (in all judges we tested), while the analytic checker alone lacks explanatory depth. When combined, the LaaJ+Hints configuration achieves up to 74% coverage (for the best performing judge and injection prompt) and produces qualitatively richer, more accurate explanations, demonstrating that analytic-LLM hybrids can substantially enhance evaluation reliability in deployed pipelines. We release the dataset and all used prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12410v1">Are LLMs Smarter Than Chimpanzees? An Evaluation on Perspective Taking and Knowledge State Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 23 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Cognitive anthropology suggests that the distinction of human intelligence lies in the ability to infer other individuals' knowledge states and understand their intentions. In comparison, our closest animal relative, chimpanzees, lack the capacity to do so. With this paper, we aim to evaluate LLM performance in the area of knowledge state tracking and estimation. We design two tasks to test (1) if LLMs can detect when story characters, through their actions, demonstrate knowledge they should not possess, and (2) if LLMs can predict story characters' next actions based on their own knowledge vs. objective truths they do not know. Results reveal that most current state-of-the-art LLMs achieve near-random performance on both tasks, and are substantially inferior to humans. We argue future LLM research should place more weight on the abilities of knowledge estimation and intention understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16170v3">When Do LLMs Admit Their Mistakes? Understanding The Role Of Model Belief In Retraction</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) admit their mistakes when they should know better? In this work, we study when and why LLMs choose to retract, i.e., spontaneously and immediately acknowledge their errors. Using model-specific testbeds, we find that while LLMs are capable of retraction, they do so only rarely, even when they can recognize their mistakes when asked in a separate interaction. We identify a reliable predictor of retraction: the model's momentary belief, as measured by a probe on its internal states that is trained to predict correctness on external datasets unrelated to retraction. A model retracts only when it "believes" its answers to be incorrect during generation; these beliefs frequently diverge from models' parametric knowledge as measured by factoid questions. Steering experiments further demonstrate that model belief causally drives retraction. In particular, when the model believes its answer to be incorrect, this not only encourages the model to attempt further verification, but also alters attention dynamics. Finally, we show that supervised fine-tuning improves retraction performance by helping the model learn more accurate internal belief. Code and datasets are available on https://github.com/ayyyq/llm-retraction .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12374v1">A Scalable Entity-Based Framework for Auditing Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Existing approaches to bias evaluation in large language models (LLMs) trade ecological validity for statistical control, relying on artificial prompts that poorly reflect real-world use, or on naturalistic tasks that lack scale and rigor. We introduce a scalable bias-auditing framework using named entities as probes to measure structural disparities in model behavior. We show that synthetic data reliably reproduces bias patterns observed in natural text, enabling large-scale analysis. Using this approach, we conduct the largest bias audit to date, comprising 1.9 billion data points across multiple entity types, tasks, languages, models, and prompting strategies. Our results reveal systematic biases: models penalize right-wing politicians, favor left-wing politicians, prefer Western and wealthy nations over the Global South, favor Western companies, and penalize firms in the defense and pharmaceutical sectors. While instruction tuning reduces bias, increasing model scale amplifies it, and prompting in Chinese or Russian does not attenuate Western-aligned preferences. These results indicate that LLMs should undergo rigorous auditing before deployment in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12360v1">Discovering 100+ Compiler Defects in 72 Hours via LLM-Driven Semantic Logic Recomposition</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Compilers constitute the foundational root-of-trust in software supply chains; however, their immense complexity inevitably conceals critical defects. Recent research has attempted to leverage historical bugs to design new mutation operators or fine-tune models to increase program diversity for compiler fuzzing.We observe, however, that bugs manifest primarily based on the semantics of input programs rather than their syntax. Unfortunately, current approaches, whether relying on syntactic mutation or general Large Language Model (LLM) fine-tuning, struggle to preserve the specific semantics found in the logic of bug-triggering programs. Consequently, these critical semantic triggers are often lost, resulting in a limitation of the diversity of generated programs. To explicitly reuse such semantics, we propose FeatureFuzz, a compiler fuzzer that combines features to generate programs. We define a feature as a decoupled primitive that encapsulates a natural language description of a bug-prone invariant, such as an out-of-bounds array access, alongside a concrete code witness of its realization. FeatureFuzz operates via a three-stage workflow: it first extracts features from historical bug reports, synthesizes coherent groups of features, and finally instantiates these groups into valid programs for compiler fuzzing. We evaluated FeatureFuzz on GCC and LLVM. Over 24-hour campaigns, FeatureFuzz uncovered 167 unique crashes, which is 2.78x more than the second-best fuzzer. Furthermore, through a 72-hour fuzzing campaign, FeatureFuzz identified 106 bugs in GCC and LLVM, 76 of which have already been confirmed by compiler developers, validating the approach's ability to stress-test modern compilers effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12359v1">Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 Accepted to NeurIPS 2025 Lock-LLM Workshop
    </div>
    <details class="paper-abstract">
      Prompt injection attacks have become an increasing vulnerability for LLM applications, where adversarial prompts exploit indirect input channels such as emails or user-generated content to circumvent alignment safeguards and induce harmful or unintended outputs. Despite advances in alignment, even state-of-the-art LLMs remain broadly vulnerable to adversarial prompts, underscoring the urgent need for robust, productive, and generalizable detection mechanisms beyond inefficient, model-specific patches. In this work, we propose Zero-Shot Embedding Drift Detection (ZEDD), a lightweight, low-engineering-overhead framework that identifies both direct and indirect prompt injection attempts by quantifying semantic shifts in embedding space between benign and suspect inputs. ZEDD operates without requiring access to model internals, prior knowledge of attack types, or task-specific retraining, enabling efficient zero-shot deployment across diverse LLM architectures. Our method uses adversarial-clean prompt pairs and measures embedding drift via cosine similarity to capture subtle adversarial manipulations inherent to real-world injection attacks. To ensure robust evaluation, we assemble and re-annotate the comprehensive LLMail-Inject dataset spanning five injection categories derived from publicly available sources. Extensive experiments demonstrate that embedding drift is a robust and transferable signal, outperforming traditional methods in detection accuracy and operational efficiency. With greater than 93% accuracy in classifying prompt injections across model architectures like Llama 3, Qwen 2, and Mistral and a false positive rate of <3%, our approach offers a lightweight, scalable defense layer that integrates into existing LLM pipelines, addressing a critical gap in securing LLM-powered systems to withstand adaptive adversarial threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v4">SIP-BMM: Constructing Capability-Efficiency Pareto Set of LLMs via Bayesian Model Merging with Structural Importance Prior</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Navigating the capability-efficiency trade-offs in Large Language Models (LLMs) requires constructing a high-quality Pareto set. However, existing merging techniques remain inadequate: coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise optimization suffers from the curse of dimensionality, especially under tight evaluation budgets where each model candidate is costly to assess. We propose Bayesian Model Merging with Structural Importance Prior (SIP-BMM), an evolutionary loop framework driven by Log-Noisy Expected Hypervolume Improvement ($q$NEHVI) that makes layer-wise Pareto set construction tractable by explicitly modeling which layers matter. Specifically, SIP-BMM derives a \textbf{Structural Importance Prior (SIP)} from layer-wise task-vector differences between base and expert models, and uses this prior to Bayesian Optimization toward a low effective dimensional subspace. Intuitively, SIP steers the optimizer to spend most trials on a small set of influential layers while largely ignoring layers that exhibit minimal task-relevant shifts. This importance-aware search preserves layer-wise control while substantially reducing sample complexity. Experiments show that SIP-BMM discovers a stronger and denser Pareto front than competitive baselines, enabling agile model selection under diverse operational constraints. Code is available at: https://github.com/MiLab-HITSZ/2026-SIPBMM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05529v3">Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 Corrected author order in metadata; manuscript unchanged
    </div>
    <details class="paper-abstract">
      One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12343v1">How Well Do LLMs Predict Human Behavior? A Measure of their Pretrained Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to predict human behavior. We propose a measure for evaluating how much knowledge a pretrained LLM brings to such a prediction: its equivalent sample size, defined as the amount of task-specific data needed to match the predictive accuracy of the LLM. We estimate this measure by comparing the prediction error of a fixed LLM in a given domain to that of flexible machine learning models trained on increasing samples of domain-specific data. We further provide a statistical inference procedure by developing a new asymptotic theory for cross-validated prediction error. Finally, we apply this method to the Panel Study of Income Dynamics. We find that LLMs encode considerable predictive information for some economic variables but much less for others, suggesting that their value as substitutes for domain-specific data differs markedly across settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12341v1">Time-Continuous Modeling for Temporal Affective Pattern Recognition in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      This paper introduces a dataset and conceptual framework for LLMs to mimic real world emotional dynamics through time and in-context learning leveraging physics-informed neural network, opening a possibility for interpretable dialogue modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12338v1">Actionable Advice from Reviews via Mixture of LoRA Experts: A Two-LLM Pipeline for Issue Extraction and Business Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Customer reviews contain detailed, domain specific signals about service failures and user expectations, but converting this unstructured feedback into actionable business decisions remains difficult. We study review-to-action generation: producing concrete, implementable recommendations grounded in review text. We propose a modular two-LLM framework in which an Issue model extracts salient issues and assigns coarse themes, and an Advice model generates targeted operational fixes conditioned on the extracted issue representation. To enable specialization without expensive full fine-tuning, we adapt the Advice model using a mixture of LoRA experts strategy: multiple low-rank adapters are trained and a lightweight gating mechanism performs token-level expert mixing at inference, combining complementary expertise across issue types. We construct synthetic review-issue-advice triples from Yelp reviews (airlines and restaurants) to supervise training, and evaluate recommendations using an eight dimension operational rubric spanning actionability, specificity, feasibility, expected impact, novelty, non-redundancy, bias, and clarity. Across both domains, our approach consistently outperforms prompting-only and single-adapter baselines, yielding higher actionability and specificity while retaining favorable efficiency-quality trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12317v1">Explanova: Automatically Discover Data Insights in N \times M Table via XAI Combined LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Automation in data analysis has been a long-time pursuit. Current agentic LLM shows a promising solution towards it. Like DeepAnalyze, DataSage, and Datawise. They are all powerful agentic frameworks for automatic fine-grained analysis and are powered by LLM-based agentic tool calling ability. However, what about powered by a preset AutoML-like workflow? If we traverse all possible exploration, like Xn itself`s statistics, Xn1-Xn2 relationships, Xn to all other, and finally explain? Our Explanova is such an attempt: Cheaper due to a Local Small LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12311v1">Cross-reality Location Privacy Protection in 6G-enabled Vehicular Metaverses: An LLM-enhanced Hybrid Generative Diffusion Model-based Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The emergence of 6G-enabled vehicular metaverses enables Autonomous Vehicles (AVs) to operate across physical and virtual spaces through space-air-ground-sea integrated networks. The AVs can deploy AI agents powered by large AI models as personalized assistants, on edge servers to support intelligent driving decision making and enhanced on-board experiences. However, such cross-reality interactions may cause serious location privacy risks, as adversaries can infer AV trajectories by correlating the location reported when AVs request LBS in reality with the location of the edge servers on which their corresponding AI agents are deployed in virtuality. To address this challenge, we design a cross-reality location privacy protection framework based on hybrid actions, including continuous location perturbation in reality and discrete privacy-aware AI agent migration in virtuality. In this framework, a new privacy metric, termed cross-reality location entropy, is proposed to effectively quantify the privacy levels of AVs. Based on this metric, we formulate an optimization problem to optimize the hybrid action, focusing on achieving a balance between location protection, service latency reduction, and quality of service maintenance. To solve the complex mixed-integer problem, we develop a novel LLM-enhanced Hybrid Diffusion Proximal Policy Optimization (LHDPPO) algorithm, which integrates LLM-driven informative reward design to enhance environment understanding with double Generative Diffusion Models-based policy exploration to handle high-dimensional action spaces, thereby enabling reliable determination of optimal hybrid actions. Extensive experiments on real-world datasets demonstrate that the proposed framework effectively mitigates cross-reality location privacy leakage for AVs while maintaining strong user immersion within 6G-enabled vehicular metaverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12298v1">CD-PIM: A High-Bandwidth and Compute-Efficient LPDDR5-Based PIM for Low-Batch LLM Acceleration on Edge-Device</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 To appear in 2026 Design, Automation and Test in Europe Conference (DATE 2026)
    </div>
    <details class="paper-abstract">
      Edge deployment of low-batch large language models (LLMs) faces critical memory bandwidth bottlenecks when executing memory-intensive general matrix-vector multiplications (GEMV) operations. While digital processing-in-memory (PIM) architectures promise to accelerate GEMV operations, existing PIM-equipped edge devices still suffer from three key limitations: limited bandwidth improvement, component under-utilization in mixed workloads, and low compute capacity of computing units (CUs). In this paper, we propose CD-PIM to address these challenges through three key innovations. First, we introduce a high-bandwidth compute-efficient mode (HBCEM) that enhances bandwidth by dividing each bank into four pseudo-banks through segmented global bitlines. Second, we propose a low-batch interleaving mode (LBIM) to improve component utilization by overlapping GEMV operations with GEMM operations. Third, we design a compute-efficient CU that performs enhanced GEMV operations in a pipelined manner by serially feeding weight data into the computing core. Forth, we adopt a column-wise mapping for the key-cache matrix and row-wise mapping for the value-cache matrix, which fully utilizes CU resources. Our evaluation shows that compared to a GPU-only baseline and state-of-the-art PIM designs, our CD-PIM achieves 11.42x and 4.25x speedup on average within a single batch in HBCEM mode, respectively. Moreover, for low-batch sizes, the CD-PIM achieves an average speedup of 1.12x in LBIM compared to HBCEM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03994v3">Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 Accepted to the AAAI 2026 Deployable AI (DAI) Workshop
    </div>
    <details class="paper-abstract">
      As organizations increasingly deploy LLMs in sensitive domains such as legal, financial, and medical settings, ensuring alignment with internal organizational policies has become a priority. Existing content moderation frameworks remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and training cost. To address these limitations, we frame policy violation detection as an out-of-distribution (OOD) problem in the model's activation space. We propose a training-free method that operates directly on the LLM internal representations, leveraging prior evidence that decision-relevant information is encoded within them. Inspired by whitening techniques, we apply a linear transformation to decorrelate and standardize the model's hidden activations, and use the Euclidean norm in this transformed space as a compliance score for detecting policy violations. Our method requires only the policy text and a small number of illustrative samples, making it lightweight and easily deployable. We extensively evaluate our method across multiple LLMs and challenging policy benchmarks, achieving 86.0% F1 score while outperforming fine-tuned baselines by up to 9.1 points and LLM-as-a-judge by 16 points, with significantly lower computational cost. Code is available at: https://github.com/FujitsuResearch/LLM-policy-violation-detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01436v3">Challenges & Opportunities with LLM-Assisted Visualization Retargeting</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 5 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of visualization examples published on the web, retargeting existing custom chart implementations to new datasets remains difficult, time-intensive, and tedious. The adaptation process assumes author familiarity with both the implementation of the example as well as how the new dataset might need to be transformed to fit into the example code. With recent advances in Large Language Models (LLMs), automatic adaptation of code can be achieved from high-level user prompts, reducing the barrier for visualization retargeting. To better understand how LLMs can assist retargeting and its potential limitations, we characterize and evaluate the performance of LLM assistance across multiple datasets and charts of varying complexity, categorizing failures according to type and severity. In our evaluation, we compare two approaches: (1) directly instructing the LLM model to fully generate and adapt code by treating code as text inputs and (2) a more constrained program synthesis pipeline where the LLM guides the code construction process by providing structural information (e.g., visual encodings) based on properties of the example code and data. We find that both approaches struggle when new data has not been appropriately transformed, and discuss important design recommendations for future retargeting systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08211v2">LLMs Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Previous research has shown that LLMs finetuned on malicious or incorrect completions within narrow domains (e.g., insecure code or incorrect medical advice) can become broadly misaligned to exhibit harmful behaviors, which is called emergent misalignment. In this work, we investigate whether this phenomenon can extend beyond safety behaviors to a broader spectrum of dishonesty and deception under high-stakes scenarios (e.g., lying under pressure and deceptive behavior). To explore this, we finetune open-sourced LLMs on misaligned completions across diverse domains. Experimental results demonstrate that LLMs show broadly misaligned behavior in dishonesty. Additionally, we further explore this phenomenon in a downstream combined finetuning setting, and find that introducing as little as 1% of misalignment data into a standard downstream task is sufficient to decrease honest behavior over 20%. Furthermore, we consider a more practical human-AI interaction environment where we simulate both benign and biased users to interact with the assistant LLM. Notably, we find that the assistant can be misaligned unintentionally to exacerbate its dishonesty with only 10% biased user population. In summary, we extend the study of emergent misalignment to the domain of dishonesty and deception under high-stakes scenarios, and demonstrate that this risk arises not only through direct finetuning, but also in downstream mixture tasks and practical human-AI interactions. Refer to https://github.com/hxhcreate/LLM_Deceive_Unintentionally for experimental resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12280v1">Democratizing Music Therapy: LLM-Based Automated EEG Analysis and Progress Tracking for Low-Cost Home Devices</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Home-based music therapy devices require accessible and cost-effective solutions for users to understand and track their therapeutic progress. Traditional physiological signal analysis, particularly EEG interpretation, relies heavily on domain experts, creating barriers to scalability and home adoption. Meanwhile, few experts are capable of interpreting physiological signal data while also making targeted music recommendations. While large language models (LLMs) have shown promise in various domains, their application to automated physiological report generation for music therapy represents an unexplored task. We present a prototype system that leverages LLMs to bridge this gap -- transforming raw EEG and cardiovascular data into human-readable therapeutic reports and personalized music recommendations. Unlike prior work focusing on real-time physiological adaptation during listening, our approach emphasizes post-session analysis and interpretable reporting, enabling non-expert users to comprehend their psychophysiological states and track therapeutic outcomes over time. By integrating signal processing modules with LLM-based reasoning agents, the system provides a practical and low-cost solution for short-term progress monitoring in home music therapy contexts. This work demonstrates the feasibility of applying LLMs to a novel task -- democratizing access to physiology-driven music therapy through automated, interpretable reporting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10700v2">LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Concept-based explanations quantify how high-level concepts (e.g., gender or experience) influence model behavior, which is crucial for decision-makers in high-stakes domains. Recent work evaluates the faithfulness of such explanations by comparing them to reference causal effects estimated from counterfactuals. In practice, existing benchmarks rely on costly human-written counterfactuals that serve as an imperfect proxy. To address this, we introduce a framework for constructing datasets containing structural counterfactual pairs: LIBERTy (LLM-based Interventional Benchmark for Explainability with Reference Targets). LIBERTy is grounded in explicitly defined Structured Causal Models (SCMs) of the text generation, interventions on a concept propagate through the SCM until an LLM generates the counterfactual. We introduce three datasets (disease detection, CV screening, and workplace violence prediction) together with a new evaluation metric, order-faithfulness. Using them, we evaluate a wide range of methods across five models and identify substantial headroom for improving concept-based explanations. LIBERTy also enables systematic analysis of model sensitivity to interventions: we find that proprietary LLMs show markedly reduced sensitivity to demographic concepts, likely due to post-training mitigation. Overall, LIBERTy provides a much-needed benchmark for developing faithful explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12273v1">Leveraging Mutation Analysis for LLM-based Repair of Quantum Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-18
      | 💬 6 pages, Accepted at SANER-ERA 2026
    </div>
    <details class="paper-abstract">
      In recent years, Automated Program Repair (APR) techniques specifically designed for quantum programs have been proposed. However, existing approaches often suffer from low repair success rates or poor understandability of the generated patches. In this study, we construct a framework in which a large language model (LLM) generates code repairs along with a natural language explanation of the applied repairs. To investigate how the contextual information included in prompts influences APR performance for quantum programs, we design four prompt configurations with different combinations of static information, dynamic information, and mutation analysis results. Mutation analysis evaluates how small changes to specific parts of a program affect its execution results and provides more detailed dynamic information than simple execution outputs such as stack traces. Our experimental results show that mutation analysis can provide valuable contextual information for LLM-based APR of quantum programs, improving repair success rates (achieving 94.4% in our experiment) and in some cases also improving the quality of generated explanations. Our findings point toward new directions for developing APR techniques for quantum programs that enhance both reliability and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12645v5">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-18
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23816v2">A Course Correction in Steerability Evaluation: Revealing Miscalibration and Side Effects in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 8 pages, 6 figures. 26 pages of references and supplementary material, 22 additional figures. Association for the Advancement of Artificial Intelligence Conference (AAAI 2026)
    </div>
    <details class="paper-abstract">
      Despite advances in large language models (LLMs) on reasoning and instruction-following tasks, it is unclear whether they can reliably produce outputs aligned with a variety of user goals, a concept called steerability. Two gaps in current LLM evaluation impede steerability evaluation: (1) many benchmarks are built with past LLM chats and Internet-scraped text, which may skew towards common requests, and (2) scalar measures of performance common in prior work could conceal behavioral shifts in LLM outputs in open-ended generation. Thus, we introduce a framework based on a multi-dimensional goal-space that models user goals and LLM outputs as vectors with dimensions corresponding to text attributes (e.g., reading difficulty). Applied to a text-rewriting task, we find that current LLMs induce unintended changes or side effects to text attributes, impeding steerability. Interventions to improve steerability, such as prompt engineering, best-of-N sampling, and reinforcement learning fine-tuning, have varying effectiveness but side effects remain problematic. Our findings suggest that even strong LLMs struggle with steerability, and existing alignment strategies may be insufficient. We open-source our steerability evaluation framework at https://github.com/MLD3/steerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12164v1">The Language You Ask In: Language-Conditioned Ideological Divergence in LLM Analysis of Contested Political Documents</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as analytical tools across multilingual contexts, yet their outputs may carry systematic biases conditioned by the language of the prompt. This study presents an experimental comparison of LLM-generated political analyses of a Ukrainian civil society document, using semantically equivalent prompts in Russian and Ukrainian. Despite identical source material and parallel query structures, the resulting analyses varied substantially in rhetorical positioning, ideological orientation, and interpretive conclusions. The Russian-language output echoed narratives common in Russian state discourse, characterizing civil society actors as illegitimate elites undermining democratic mandates. The Ukrainian-language output adopted vocabulary characteristic of Western liberal-democratic political science, treating the same actors as legitimate stakeholders within democratic contestation. These findings demonstrate that prompt language alone can produce systematically different ideological orientations from identical models analyzing identical content, with significant implications for AI deployment in polarized information environments, cross-lingual research applications, and the governance of AI systems in multilingual societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12154v1">Analyzing Cancer Patients' Experiences with Embedding-based Topic Modeling and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 under review to CLIN journal
    </div>
    <details class="paper-abstract">
      This study investigates the use of neural topic modeling and LLMs to uncover meaningful themes from patient storytelling data, to offer insights that could contribute to more patient-oriented healthcare practices. We analyze a collection of transcribed interviews with cancer patients (132,722 words in 13 interviews). We first evaluate BERTopic and Top2Vec for individual interview summarization by using similar preprocessing, chunking, and clustering configurations to ensure a fair comparison on Keyword Extraction. LLMs (GPT4) are then used for the next step topic labeling. Their outputs for a single interview (I0) are rated through a small-scale human evaluation, focusing on {coherence}, {clarity}, and {relevance}. Based on the preliminary results and evaluation, BERTopic shows stronger performance and is selected for further experimentation using three {clinically oriented embedding} models. We then analyzed the full interview collection with the best model setting. Results show that domain-specific embeddings improved topic \textit{precision} and \textit{interpretability}, with BioClinicalBERT producing the most consistent results across transcripts. The global analysis of the full dataset of 13 interviews, using the BioClinicalBERT embedding model, reveals the most dominant topics throughout all 13 interviews, namely ``Coordination and Communication in Cancer Care Management" and ``Patient Decision-Making in Cancer Treatment Journey''. Although the interviews are machine translations from Dutch to English, and clinical professionals are not involved in this evaluation, the findings suggest that neural topic modeling, particularly BERTopic, can help provide useful feedback to clinicians from patient interviews. This pipeline could support more efficient document navigation and strengthen the role of patients' voices in healthcare workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12152v1">Who Owns Creativity and Who Does the Work? Trade-offs in LLM-Supported Research Ideation</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      LLM-based agents offer new potential to accelerate science and reshape research work. However, the quality of researcher contributions can vary significantly depending on human ability to steer agent behaviors. How can we best use these tools to augment scientific creativity without undermining aspects of contribution and ownership that drive research? To investigate this, we developed an agentic research ideation system integrating three roles -- Ideator, Writer, and Evaluator -- across three control levels -- Low, Medium, and Intensive. Our mixed-methods study with 54 researchers suggests three key findings in how LLM-based agents reshape scientific creativity: 1) perceived creativity support does not simply increase linearly with greater control; 2) human effort shifts from ideating to verifying ideas; and 3) ownership becomes a negotiated outcome between human and AI. Our findings suggest that LLM agent design should emphasize researcher empowerment, fostering a sense of ownership over strong ideas rather than reducing researchers to operating an automated AI-driven process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v1">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12146v1">From LLMs to Agents in Programming: The Impact of Providing an LLM with a Compiler</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated a remarkable capability in natural language and program generation and software development. However, the source code generated by the LLMs does not always meet quality requirements and may fail to compile. Therefore, many studies evolve into agents that can reason about the problem before generating the source code for the solution. The goal of this paper is to study the degree to which such agents benefit from access to software development tools, in our case, a \texttt{gcc} compiler. We conduct a computational experiment on the RosettaCode dataset, on 699 programming tasks in C. We evaluate how the integration with a compiler shifts the role of the language model from a passive generator to an active agent capable of iteratively developing runnable programs based on feedback from the compiler. We evaluated 16 language models with sizes ranging from small (135 million) to medium (3 billion) and large (70 billion). Our results show that access to a compiler improved the compilation success by 5.3 to 79.4 percentage units in compilation without affecting the semantics of the generated program. Syntax errors dropped by 75\%, and errors related to undefined references dropped by 87\% for the tasks where the agents outperformed the baselines. We also observed that in some cases, smaller models with a compiler outperform larger models with a compiler. We conclude that it is essential for LLMs to have access to software engineering tools to enhance their performance and reduce the need for large models in software engineering, such as reducing our energy footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05269v3">CoRe: Benchmarking LLMs Code Reasoning Capabilities through Static Analysis Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 NeurIPS 2025 Datasets & Benchmarks Spotlight
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted across diverse domains of software engineering, such as code generation, program repair, and vulnerability detection. These applications require understanding beyond surface-level code patterns: value propagation, control flow, and interdependence between program elements. However, existing benchmarks primarily evaluate end-to-end outcomes, such as whether code is correctly repaired or generated, leaving the models' ability for program semantic reasoning underexplored. This work presents CORE, a high-quality, human-verified benchmark designed to evaluate LLMs on fundamental static analysis tasks. CORE includes 12,553 task instances spanning data dependency, control dependency, and information flow across programs written in C/C++, Java, and Python. To ensure semantic diversity and reasoning complexity, we propose a semantics-aware diverse sampling strategy that selects targets and task instances based on structural coverage and dependency depth. We evaluate 10 mainstream LLMs and show that, while they perform well at identifying dependencies, models still struggle with tasks that require deeper semantic understanding and multi-step reasoning. We further conduct qualitative analyses to uncover key challenges, such as complex control structures and backward dependency patterns, offering insights into improving LLMs' code reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12138v1">DriveSafe: A Hierarchical Risk Taxonomy for Safety-Critical LLM-Based Driving Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into vehicle-based digital assistants, where unsafe, ambiguous, or legally incorrect responses can lead to serious safety, ethical, and regulatory consequences. Despite growing interest in LLM safety, existing taxonomies and evaluation frameworks remain largely general-purpose and fail to capture the domain-specific risks inherent to real-world driving scenarios. In this paper, we introduce DriveSafe, a hierarchical, four-level risk taxonomy designed to systematically characterize safety-critical failure modes of LLM-based driving assistants. The taxonomy comprises 129 fine-grained atomic risk categories spanning technical, legal, societal, and ethical dimensions, grounded in real-world driving regulations and safety principles and reviewed by domain experts. To validate the safety relevance and realism of the constructed prompts, we evaluate their refusal behavior across six widely deployed LLMs. Our analysis shows that the evaluated models often fail to appropriately refuse unsafe or non-compliant driving-related queries, underscoring the limitations of general-purpose safety alignment in driving contexts.
    </details>
</div>
