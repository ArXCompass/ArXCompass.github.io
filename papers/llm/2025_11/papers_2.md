# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15168v1">Finetuning LLMs for Automatic Form Interaction on Web-Browser in Selenium Testing Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Published in the Proceedings of KSE 2025
    </div>
    <details class="paper-abstract">
      Automated web application testing is a critical component of modern software development, with frameworks like Selenium widely adopted for validating functionality through browser automation. Among the essential aspects of such testing is the ability to interact with and validate web forms, a task that requires syntactically correct, executable scripts with high coverage of input fields. Despite its importance, this task remains underexplored in the context of large language models (LLMs), and no public benchmark or dataset exists to evaluate LLMs on form interaction generation systematically. This paper introduces a novel method for training LLMs to generate high-quality test cases in Selenium, specifically targeting form interaction testing. We curate both synthetic and human-annotated datasets for training and evaluation, covering diverse real-world forms and testing scenarios. We define clear metrics for syntax correctness, script executability, and input field coverage. Our empirical study demonstrates that our approach significantly outperforms strong baselines, including GPT-4o and other popular LLMs, across all evaluation metrics. Our work lays the groundwork for future research on LLM-based web testing and provides resources to support ongoing progress in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15163v1">Teaching According to Students' Aptitude: Personalized Mathematics Tutoring via Persona-, Memory-, and Forgetting-Aware LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 AAAI 2026 Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into intelligent tutoring systems to provide human-like and adaptive instruction. However, most existing approaches fail to capture how students' knowledge evolves dynamically across their proficiencies, conceptual gaps, and forgetting patterns. This challenge is particularly acute in mathematics tutoring, where effective instruction requires fine-grained scaffolding precisely calibrated to each student's mastery level and cognitive retention. To address this issue, we propose TASA (Teaching According to Students' Aptitude), a student-aware tutoring framework that integrates persona, memory, and forgetting dynamics for personalized mathematics learning. Specifically, TASA maintains a structured student persona capturing proficiency profiles and an event memory recording prior learning interactions. By incorporating a continuous forgetting curve with knowledge tracing, TASA dynamically updates each student's mastery state and generates contextually appropriate, difficulty-calibrated questions and explanations. Empirical results demonstrate that TASA achieves superior learning outcomes and more adaptive tutoring behavior compared to representative baselines, underscoring the importance of modeling temporal forgetting and learner profiles in LLM-based tutoring systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15141v1">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been widely used as recommender systems, owing to their strong reasoning capability and their effectiveness in handling cold-start items. To better adapt LLMs for recommendation, retrieval-augmented generation (RAG) has been incorporated. Most existing RAG methods are user-based, retrieving purchase patterns of users similar to the target user and providing them to the LLM. In this work, we propose ItemRAG, an item-based RAG method for LLM-based recommendation that retrieves relevant items (rather than users) from item-item co-purchase histories. ItemRAG helps LLMs capture co-purchase patterns among items, which are beneficial for recommendations. Especially, our retrieval strategy incorporates semantically similar items to better handle cold-start items and uses co-purchase frequencies to improve the relevance of the retrieved items. Through extensive experiments, we demonstrate that ItemRAG consistently (1) improves the zero-shot LLM-based recommender by up to 43% in Hit-Ratio-1 and (2) outperforms user-based RAG baselines under both standard and cold-start item recommendation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15137v1">From Solving to Verifying: A Unified Objective for Robust Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of large language models (LLMs) have been significantly improved through reinforcement learning (RL). Nevertheless, LLMs still struggle to consistently verify their own reasoning traces. This raises the research question of how to enhance the self-verification ability of LLMs and whether such an ability can further improve reasoning performance. In this work, we propose GRPO-Verif, an algorithm that jointly optimizes solution generation and self-verification within a unified loss function, with an adjustable hyperparameter controlling the weight of the verification signal. Experimental results demonstrate that our method enhances self-verification capability while maintaining comparable performance in reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06247v2">Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 The paper has been withdrawn to align with the internal publication policies of the affiliated organization. We plan to resubmit after obtaining the necessary approvals
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), their context windows have rapidly expanded. To meet diverse demands from varying-length requests in online services, existing state-of-the-art systems tune the sequence parallelism (SP) allocation. However, current dynamic SP allocation lacks flexibility to (1) support stage-specific parallelism requirements in LLM inference, (2) mitigate the global latency degradation from excessive SP allocation, and (3) exploit resource fragments arising from SP size variation. To tackle this problem, we propose Chunkwise Dynamic Sequence Parallelism (CDSP), a fine-grained parallelism strategy that assigns SP sizes across \textit{intra-request} token segments. Based on CDSP, we build Tetris, an LLM serving system that (1) efficiently integrates CDSP into disaggregated cluster to satisfy parallelism heterogeneity, (2) dynamically regulates SP size expansion based on real-time load conditions, and (3) adaptively explores chunking plans to utilize fragmented resources while meeting per-request demands. Compared with state-of-the-art systems, Tetris achieves up to 4.35$\times$ lower time-to-first-token (TTFT) under max sustainable loads, reduces median time-between-tokens (TBT) by up to 40.1\%, and increases the max request capacity by up to 45\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19134v5">Confidential Prompting: Privacy-preserving LLM Inference on Cloud</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      This paper introduces a vision of confidential prompting: securing user prompts from an untrusted, cloud-hosted large language model (LLM) while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Petridish, a system built on top of confidential computing and its core contribution, a novel technology called Secure Partitioned Decoding (SPD). Petridish runs the LLM service inside a confidential virtual machine (CVM), which protects the secrets, i.e., the LLM parameters and user prompts, from adversaries outside the CVM. Importantly, it splits the LLM service for a user into two processes, using SPD: a per-user process performs prefill with the user prompts and computes attention scores during decoding; a service process, shared by all users, batches the attention scores from per-user processes and generates output tokens for all users. Both the LLM provider and the users trust Petridish's CVM and its operating system, which guarantees isolation between processes and limits their outbound network capabilities to control information flow. The CVM's attestation capability and its open-source software stack enable Petridish to provide auditable protection of both user prompt and LLM confidentiality. Together, Petridish maintains full utility of LLM service and enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03117v2">PromptCOS: Towards Content-only System Prompt Copyright Auditing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      System prompts are critical for shaping the behavior and output quality of large language model (LLM)-based applications, driving substantial investment in optimizing high-quality prompts beyond traditional handcrafted designs. However, as system prompts become valuable intellectual property, they are increasingly vulnerable to prompt theft and unauthorized use, highlighting the urgent need for effective copyright auditing, especially watermarking. Existing methods rely on verifying subtle logit distribution shifts triggered by a query. We observe that this logit-dependent verification framework is impractical in real-world content-only settings, primarily because (1) random sampling makes content-level generation unstable for verification, and (2) stronger instructions needed for content-level signals compromise prompt fidelity. To overcome these challenges, we propose PromptCOS, the first content-only system prompt copyright auditing method based on content-level output similarity. PromptCOS achieves watermark stability by designing a cyclic output signal as the conditional instruction's target. It preserves prompt fidelity by injecting a small set of auxiliary tokens to encode the watermark, leaving the main prompt untouched. Furthermore, to ensure robustness against malicious removal, we optimize cover tokens, i.e., critical tokens in the original prompt, to ensure that removing auxiliary tokens causes severe performance degradation. Experimental results show that PromptCOS achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% higher than the best baseline), high fidelity (accuracy degradation no greater than 0.6%), robustness (resilience against four potential attack categories), and high computational efficiency (up to 98.1% cost saving). Our code is available at GitHub (https://github.com/LianPing-cyber/PromptCOS).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15069v1">ProRAC: A Neuro-symbolic Method for Reasoning about Actions with LLM-based Progression</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      In this paper, we propose ProRAC (Progression-based Reasoning about Actions and Change), a neuro-symbolic framework that leverages LLMs to tackle RAC problems. ProRAC extracts fundamental RAC elements including actions and questions from the problem, progressively executes each action to derive the final state, and then evaluates the query against the progressed state to arrive at an answer. We evaluate ProRAC on several RAC benchmarks, and the results demonstrate that our approach achieves strong performance across different benchmarks, domains, LLM backbones, and types of RAC tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.21457v2">SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 18 pages, 5 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18015v2">Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Proceedings of the 5th Machine Learning for Health (ML4H) Symposium
    </div>
    <details class="paper-abstract">
      Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, but showed improvements when provided examples through few shot prompting. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11881v2">Better LLM Reasoning via Dual-Play</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at https://hcy123902.github.io/PasoDoble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.05034v2">Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15921v1">Thinking, Faithful and Stable: Mitigating Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Originally released June 5, 2025
    </div>
    <details class="paper-abstract">
      This project develops a self correcting framework for large language models (LLMs) that detects and mitigates hallucinations during multi-step reasoning. Rather than relying solely on final answer correctness, our approach leverages fine grained uncertainty signals: 1) self-assessed confidence alignment, and 2) token-level entropy spikes to detect unreliable and unfaithful reasoning in real time. We design a composite reward function that penalizes unjustified high confidence and entropy spikes, while encouraging stable and accurate reasoning trajectories. These signals guide a reinforcement learning (RL) policy that makes the model more introspective and shapes the model's generation behavior through confidence-aware reward feedback, improving not just outcome correctness but the coherence and faithfulness of their intermediate reasoning steps. Experiments show that our method improves both final answer accuracy and reasoning calibration, with ablations validating the individual contribution of each signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15915v1">AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      We present AccelOpt, a self-improving large language model (LLM) agentic system that autonomously optimizes kernels for emerging AI acclerators, eliminating the need for expert-provided hardware-specific optimization knowledge. AccelOpt explores the kernel optimization space through iterative generation, informed by an optimization memory that curates experiences and insights from previously encountered slow-fast kernel pairs. We build NKIBench, a new benchmark suite of AWS Trainium accelerator kernels with varying complexity extracted from real-world LLM workloads to evaluate the effectiveness of AccelOpt. Our evaluation confirms that AccelOpt's capability improves over time, boosting the average percentage of peak throughput from $49\%$ to $61\%$ on Trainium 1 and from $45\%$ to $59\%$ on Trainium 2 for NKIBench kernels. Moreover, AccelOpt is highly cost-effective: using open-source models, it matches the kernel improvements of Claude Sonnet 4 while being $26\times$ cheaper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12339v4">A Closer Look at Adversarial Suffix Learning for Jailbreaking LLMs: Augmented Adversarial Trigger Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 the Association for Computational Linguistics: NAACL 2025
    </div>
    <details class="paper-abstract">
      Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15895v1">Decomposing Theory of Mind: How Emotional Processing Mediates ToM Abilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Published at ToM4AI workshop@AAAI2026
    </div>
    <details class="paper-abstract">
      Recent work shows activation steering substantially improves language models' Theory of Mind (ToM) (Bortoletto et al. 2024), yet the mechanisms of what changes occur internally that leads to different outputs remains unclear. We propose decomposing ToM in LLMs by comparing steered versus baseline LLMs' activations using linear probes trained on 45 cognitive actions. We applied Contrastive Activation Addition (CAA) steering to Gemma-3-4B and evaluated it on 1,000 BigToM forward belief scenarios (Gandhi et al. 2023), we find improved performance on belief attribution tasks (32.5\% to 46.7\% accuracy) is mediated by activations processing emotional content : emotion perception (+2.23), emotion valuing (+2.20), while suppressing analytical processes: questioning (-0.78), convergent thinking (-1.59). This suggests that successful ToM abilities in LLMs are mediated by emotional understanding, not analytical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15862v1">The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. These findings demonstrate that uncooperative agents can significantly degrade collective outcomes, highlighting the need for designing more resilient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13984v2">Node-Level Uncertainty Estimation in LLM-Generated SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      We present a practical framework for detecting errors in LLM-generated SQL by estimating uncertainty at the level of individual nodes in the query's abstract syntax tree (AST). Our approach proceeds in two stages. First, we introduce a semantically aware labeling algorithm that, given a generated SQL and a gold reference, assigns node-level correctness without over-penalizing structural containers or alias variation. Second, we represent each node with a rich set of schema-aware and lexical features - capturing identifier validity, alias resolution, type compatibility, ambiguity in scope, and typo signals - and train a supervised classifier to predict per-node error probabilities. We interpret these probabilities as calibrated uncertainty, enabling fine-grained diagnostics that pinpoint exactly where a query is likely to be wrong. Across multiple databases and datasets, our method substantially outperforms token log-probabilities: average AUC improves by +27.44% while maintaining robustness under cross-database evaluation. Beyond serving as an accuracy signal, node-level uncertainty supports targeted repair, human-in-the-loop review, and downstream selective execution. Together, these results establish node-centric, semantically grounded uncertainty estimation as a strong and interpretable alternative to aggregate sequence level confidence measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v1">A Causal Perspective on Measuring, Explaining and Mitigating Smells in \llm-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v2">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15762v1">A time for monsters: Organizational knowing after LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Forthcoming at Strategic Organization
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping organizational knowing by unsettling the epistemological foundations of representational and practice-based perspectives. We conceptualize LLMs as Haraway-ian monsters, that is, hybrid, boundary-crossing entities that destabilize established categories while opening new possibilities for inquiry. Focusing on analogizing as a fundamental driver of knowledge, we examine how LLMs generate connections through large-scale statistical inference. Analyzing their operation across the dimensions of surface/deep analogies and near/far domains, we highlight both their capacity to expand organizational knowing and the epistemic risks they introduce. Building on this, we identify three challenges of living with such epistemic monsters: the transformation of inquiry, the growing need for dialogical vetting, and the redistribution of agency. By foregrounding the entangled dynamics of knowing-with-LLMs, the paper extends organizational theory beyond human-centered epistemologies and invites renewed attention to how knowledge is created, validated, and acted upon in the age of intelligent technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15757v1">Rethinking Kernel Program Repair: Benchmarking and Enhancing LLMs with RGym</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized automated program repair (APR) but current benchmarks like SWE-Bench predominantly focus on userspace applications and overlook the complexities of kernel-space debugging and repair. The Linux kernel poses unique challenges due to its monolithic structure, concurrency, and low-level hardware interactions. Prior efforts such as KGym and CrashFixer have highlighted the difficulty of APR in this domain, reporting low success rates or relying on costly and complex pipelines and pricey cloud infrastructure. In this work, we introduce RGym, a lightweight, platform-agnostic APR evaluation framework for the Linux kernel designed to operate on local commodity hardware. Built on RGym, we propose a simple yet effective APR pipeline leveraging specialized localization techniques (e.g., call stacks and blamed commits) to overcome the unrealistic usage of oracles in KGym. We test on a filtered and verified dataset of 143 bugs. Our method achieves up to a 43.36% pass rate with GPT-5 Thinking while maintaining a cost of under $0.20 per bug. We further conduct an ablation study to analyze contributions from our proposed localization strategy, prompt structure, and model choice, and demonstrate that feedback-based retries can significantly enhance success rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15755v1">Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 8 pages, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present MyAntFarm.ai, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14450v1">Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly executed across edge, fog, and cloud tiers where limited GPU memory, heterogeneous compute, and variable inter-tier bandwidth jointly constrain deployment and motivate model partitioning and request scheduling. In this setting, achieving low end-to-end latency is governed not only by where a model is deployed (inter-tier model partitioning) but also by how incoming requests are scheduled (intra-tier task scheduling) across heterogeneous nodes. These two problems are tightly coupled, as a suboptimal scheduler can negate the benefits of a good partition, and vice versa. In this paper, we propose Hyperion, a hierarchical two-stage framework that jointly optimizes partitioning and scheduling to minimize end-to-end latency for pipelined LLM inference in multi-tier networks, balancing compute and memory across tiers while introducing negligible runtime overhead and requiring no model retraining. Motivated by the observation that partition choices evolve on slower timescales than request arrivals, Stage 1 performs offline, inter-tier partitioning via a Binary Search with Dynamic Programming (BSDP) procedure to produce balanced stage times under tier capacity and memory constraints; to adapt to time-varying load, Stage 2 performs online, intra-tier scheduling with a lightweight Adaptive Real-time Task Scheduling (ARTS) algorithm that maps each request to the best available node using real-time estimates of queue length and effective capacity. Experimental results on multi-tier inference tasks demonstrate that Hyperion significantly reduces end-to-end latency by up to 52.1\% and 31.2\%, with the Phi-3-medium model, compared to the GPipe and HEFT baselines, respectively. Furthermore, Hyperion shows superior scalability in long-sequence generation, maintaining a 44.5\% lower latency than GPipe and achieving higher GPU utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v3">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14445v1">Tell Me: An LLM-powered Mental Well-being Assistant with RAG, Synthetic Dialogue Generation, and Agentic Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 8 pages, 2 figures, 1 Table. Submitted to the Computation and Language (cs.CL) category. Uses the ACL-style template. Code and demo will be released at: https://github.com/trystine/Tell_Me_Mental_Wellbeing_System
    </div>
    <details class="paper-abstract">
      We present Tell Me, a mental well-being system that leverages advances in large language models to provide accessible, context-aware support for users and researchers. The system integrates three components: (i) a retrieval-augmented generation (RAG) assistant for personalized, knowledge-grounded dialogue; (ii) a synthetic client-therapist dialogue generator conditioned on client profiles to facilitate research on therapeutic language and data augmentation; and (iii) a Well-being AI crew, implemented with CrewAI, that produces weekly self-care plans and guided meditation audio. The system is designed as a reflective space for emotional processing rather than a substitute for professional therapy. It illustrates how conversational assistants can lower barriers to support, complement existing care, and broaden access to mental health resources. To address the shortage of confidential therapeutic data, we introduce synthetic client-therapist dialogue generation conditioned on client profiles. Finally, the planner demonstrates an innovative agentic workflow for dynamically adaptive, personalized self-care, bridging the limitations of static well-being tools. We describe the architecture, demonstrate its functionalities, and report evaluation of the RAG assistant in curated well-being scenarios using both automatic LLM-based judgments and a human-user study. This work highlights opportunities for interdisciplinary collaboration between NLP researchers and mental health professionals to advance responsible innovation in human-AI interaction for well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14391v1">Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for decision-making and planning in autonomous driving, showing promising reasoning capabilities and potential to generalize across diverse traffic situations. However, current LLM-based driving agents lack explicit mechanisms to enforce traffic rules and often struggle to reliably detect small, safety-critical objects such as traffic lights and signs. To address this limitation, we introduce TLS-Assist, a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition. TLS-Assist converts detections into structured natural language messages that are injected into the LLM input, enforcing explicit attention to safety-critical cues. The framework is plug-and-play, model-agnostic, and supports both single-view and multi-view camera setups. We evaluate TLS-Assist in a closed-loop setup on the LangAuto benchmark in CARLA. The results demonstrate relative driving performance improvements of up to 14% over LMDrive and 7% over BEVDriver, while consistently reducing traffic light and sign infractions. We publicly release the code and models on https://github.com/iis-esslingen/TLS-Assist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14359v1">Towards LLM-Based Usability Analysis for Recommender User Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 The paper was presented at IntRS'25: Joint Workshop on Interfaces and Human Decision Making for Recommender Systems, September 22, 2025, Prague, Czech Republic and is published in the workshop proceedings: https://ceur-ws.org/Vol-4027/
    </div>
    <details class="paper-abstract">
      Usability is a key factor in the effectiveness of recommender systems. However, the analysis of user interfaces is a time-consuming process that requires expertise. Recent advances in multimodal large language models (LLMs) offer promising opportunities to automate such evaluations. In this work, we explore the potential of multimodal LLMs to assess the usability of recommender system interfaces by considering a variety of publicly available systems as examples. We take user interface screenshots from multiple of these recommender platforms to cover both preference elicitation and recommendation presentation scenarios. An LLM is instructed to analyze these interfaces with regard to different usability criteria and provide explanatory feedback. Our evaluation demonstrates how LLMs can support heuristic-style usability assessments at scale to support the improvement of user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08426v7">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted to IEEE TKDE2025
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14334v1">When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.11858v3">OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 INLG 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14275v1">Don't Miss the Forest for the Trees: In-Depth Confidence Estimation for LLMs via Reasoning over the Answer Space</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Knowing the reliability of a model's response is essential in application. With the strong generation capabilities of LLMs, research has focused on generating verbalized confidence. This is further enhanced by combining chain-of-thought reasoning, which provides logical and transparent estimation. However, how reasoning strategies affect the estimated confidence is still under-explored. In this work, we demonstrate that predicting a verbalized probability distribution can effectively encourage in-depth reasoning for confidence estimation. Intuitively, it requires an LLM to consider all candidates within the answer space instead of basing on a single guess, and to carefully assign confidence scores to meet the requirements of a distribution. This method shows an advantage across different models and various tasks, regardless of whether the answer space is known. Its advantage is maintained even after reinforcement learning, and further analysis shows its reasoning patterns are aligned with human expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14248v1">Enhancing Regional Airbnb Trend Forecasting Using LLM-Based Embeddings of Accessibility and Human Mobility</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at ASONAM 2025
    </div>
    <details class="paper-abstract">
      The expansion of short-term rental platforms, such as Airbnb, has significantly disrupted local housing markets, often leading to increased rental prices and housing affordability issues. Accurately forecasting regional Airbnb market trends can thus offer critical insights for policymakers and urban planners aiming to mitigate these impacts. This study proposes a novel time-series forecasting framework to predict three key Airbnb indicators -- Revenue, Reservation Days, and Number of Reservations -- at the regional level. Using a sliding-window approach, the model forecasts trends 1 to 3 months ahead. Unlike prior studies that focus on individual listings at fixed time points, our approach constructs regional representations by integrating listing features with external contextual factors such as urban accessibility and human mobility. We convert structured tabular data into prompt-based inputs for a Large Language Model (LLM), producing comprehensive regional embeddings. These embeddings are then fed into advanced time-series models (RNN, LSTM, Transformer) to better capture complex spatio-temporal dynamics. Experiments on Seoul's Airbnb dataset show that our method reduces both average RMSE and MAE by approximately 48% compared to conventional baselines, including traditional statistical and machine learning models. Our framework not only improves forecasting accuracy but also offers practical insights for detecting oversupplied regions and supporting data-driven urban policy decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14233v1">Visionary Co-Driver: Enhancing Driver Perception of Potential Risks with LLM and HUD</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted for publication in IEEE Transactions on Intelligent Transportation Systems (T-ITS)
    </div>
    <details class="paper-abstract">
      Drivers' perception of risky situations has always been a challenge in driving. Existing risk-detection methods excel at identifying collisions but face challenges in assessing the behavior of road users in non-collision situations. This paper introduces Visionary Co-Driver, a system that leverages large language models to identify non-collision roadside risks and alert drivers based on their eye movements. Specifically, the system combines video processing algorithms and LLMs to identify potentially risky road users. These risks are dynamically indicated on an adaptive heads-up display interface to enhance drivers' attention. A user study with 41 drivers confirms that Visionary Co-Driver improves drivers' risk perception and supports their recognition of roadside risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03758v3">Leveraging LLM-based agents for social science research: insights from citation network simulations</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 accepted by HSSCOMMS'25
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) demonstrates their potential to encapsulate the logic and patterns inherent in human behavior simulation by leveraging extensive web data pre-training. However, the boundaries of LLM capabilities in social simulation remain unclear. To further explore the social attributes of LLMs, we introduce the CiteAgent framework, designed to generate citation networks based on human-behavior simulation with LLM-based agents. CiteAgent successfully captures predominant phenomena in real-world citation networks, including power-law distribution, citational distortion, and shrinking diameter. Building on this realistic simulation, we establish two LLM-based research paradigms in social science: LLM-SE (LLM-based Survey Experiment) and LLM-LE (LLM-based Laboratory Experiment). These paradigms facilitate rigorous analyses of citation network phenomena, allowing us to validate and challenge existing theories. Additionally, we extend the research scope of traditional science of science studies through idealized social experiments, with the simulation experiment results providing valuable insights for real-world academic environments. Our work demonstrates the potential of LLMs for advancing science of science research in social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14224v1">KTester: Leveraging Domain and Testing Knowledge for More Effective LLM-based Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 13 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Automated unit test generation using large language models (LLMs) holds great promise but often struggles with generating tests that are both correct and maintainable in real-world projects. This paper presents KTester, a novel framework that integrates project-specific knowledge and testing domain knowledge to enhance LLM-based test generation. Our approach first extracts project structure and usage knowledge through static analysis, which provides rich context for the model. It then employs a testing-domain-knowledge-guided separation of test case design and test method generation, combined with a multi-perspective prompting strategy that guides the LLM to consider diverse testing heuristics. The generated tests follow structured templates, improving clarity and maintainability. We evaluate KTester on multiple open-source projects, comparing it against state-of-the-art LLM-based baselines using automatic correctness and coverage metrics, as well as a human study assessing readability and maintainability. Results demonstrate that KTester significantly outperforms existing methods across six key metrics, improving execution pass rate by 5.69% and line coverage by 8.83% over the strongest baseline, while requiring less time and generating fewer test cases. Human evaluators also rate the tests produced by KTester significantly higher in terms of correctness, readability, and maintainability, confirming the practical advantages of our knowledge-driven framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14221v1">LLM-Aligned Geographic Item Tokenization for Local-Life Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have enhanced text-based recommendation by enriching traditional ID-based methods with semantic generalization capabilities. Text-based methods typically encode item textual information via prompt design and generate discrete semantic IDs through item tokenization. However, in domain-specific tasks such as local-life services, simply injecting location information into prompts fails to capture fine-grained spatial characteristics and real-world distance awareness among items. To address this, we propose LGSID, an LLM-Aligned Geographic Item Tokenization Framework for Local-life Recommendation. This framework consists of two key components: (1) RL-based Geographic LLM Alignment, and (2) Hierarchical Geographic Item Tokenization. In the RL-based alignment module, we initially train a list-wise reward model to capture real-world spatial relationships among items. We then introduce a novel G-DPO algorithm that uses pre-trained reward model to inject generalized spatial knowledge and collaborative signals into LLMs while preserving their semantic understanding. Furthermore, we propose a hierarchical geographic item tokenization strategy, where primary tokens are derived from discrete spatial and content attributes, and residual tokens are refined using the aligned LLM's geographic representation vectors. Extensive experiments on real-world Kuaishou industry datasets show that LGSID consistently outperforms state-of-the-art discriminative and generative recommendation models. Ablation studies, visualizations, and case studies further validate its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14214v1">Do Large Language Models (LLMs) Understand Chronology?</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 47 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in finance and economics, where prompt-based attempts against look-ahead bias implicitly assume that models understand chronology. We test this fundamental question with a series of chronological ordering tasks with increasing complexities over facts the model already knows from pre-training. Our tasks cover (1) chronological ordering, (2) conditional sorting (filter, then order), and (3) anachronism detection. We evaluate GPT-4.1, Claude-3.7 Sonnet, with and without Extended Thinking (ET), and GPT-5 across multiple reasoning-effort settings. Across models, Exact match rate drops sharply as sequences lengthen even while rank correlations stay high as LLMs largely preserve local order but struggle to maintain a single globally consistent timeline. In conditional sorting, most failures stem from the filtering step rather than the ordering step, but GPT-5 and Claude-3.7 Sonnet with Extended Thinking outshine normal models significantly. Lastly, anachronism detection is found to be the easiest task for the LLMs but performance still declines with increasingly overlapping timelines or entities. Overall, our main contribution is showing that allocating explicit reasoning budget helps with chronological ordering with GPT-5 at medium/high reasoning effort achieving flawless ordering at all lengths and perfect conditional sorting (both self-filtered and given-subset), whereas low/minimal effort degrades with longer lists, mirroring earlier models. Our findings delineate limits of current LLMs on chronological tasks, providing insights into task complexity, and demonstrate scenarios in which reasoning helps. These patterns are important for the real-time application of LLMs in finance. We release all code and evaluation templates to support full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.23535v2">Comparative Analysis of the Code Generated by Popular Large Language Models (LLMs) for MISRA C++ Compliance</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Safety-critical systems are engineered systems whose failure or malfunction could result in catastrophic consequences. The software development for safety-critical systems necessitates rigorous engineering practices and adherence to certification standards like DO-178C for avionics. DO-178C is a guidance document which requires compliance to well-defined software coding standards like MISRA C++ to enforce coding guidelines that prevent the use of ambiguous, unsafe, or undefined constructs. Large Language Models (LLMs) have demonstrated significant capabilities in automatic code generation across a wide range of programming languages, including C++. Despite their impressive performance, code generated by LLMs in safety-critical domains must be carefully analyzed for conformance to MISRA C++ coding standards. In this paper, I have conducted a comparative analysis of the C++ code generated by popular LLMs including: OpenAI ChatGPT, Google Gemini, DeepSeek, Meta AI, and Microsoft Copilot for compliance with MISRA C++. The study revealed that none of the evaluated LLMs generated MISRA-compliant code despite clear prompts, with DeepSeek showing the fewest violations and Meta AI the most. While all models could correct individual violations when explicitly instructed, only ChatGPT consistently identified and resolved all targeted rule violations across complete code snippets, whereas others achieved partial success. Overall, LLMs show promise as aids for initial code generation, but they are not yet dependable for producing fully MISRA-compliant code required in safety-critical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23947v2">The Social Gaze of LLMs: A Literature Review of Multimodal Approaches to Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      LLM-powered multimodal systems are increasingly used to interpret human behavior, yet how researchers apply the models' 'social competence' remains poorly understood. This paper presents a systematic literature review of 176 publications across different application domains (e.g., healthcare, education, and entertainment). Using a four-dimensional coding framework (application, technical, evaluative, and ethical), we find (1) frequent use of pattern recognition and information extraction from multimodal sources, but limited support for adaptive, interactive reasoning; (2) a dominant 'modality-to-text' pipeline that privileges language over rich audiovisual cues, striping away nuanced social cues; (3) evaluation practices reliant on static benchmarks, with socially grounded, human-centered assessments rare; and (4) Ethical discussions focused mainly on legal and rights-related risks (e.g., privacy), leaving societal risks (e.g., deception) overlooked--or at best acknowledged but left unaddressed. We outline a research agenda for evaluating socially competent, ethically informed, and interaction-aware multi-modal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09148v2">LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 The code is accessible at https://github.com/Rednote-DeepExperience/LoopTool. The LoopTool-8B is accessible at https://huggingface.co/zhuiguang-ning/LoopTool-8B
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) with external tools enables them to execute complex, multi-step tasks. However, tool learning is hampered by the static synthetic data pipelines where data generation and model training are executed as two separate, non-interactive processes. This approach fails to adaptively focus on a model's specific weaknesses and allows noisy labels to persist, degrading training efficiency. We introduce LoopTool, a fully automated, model-aware data evolution framework that closes this loop by tightly integrating data synthesis and model training. LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures. This closed-loop process operates within a cost-effective, open-source ecosystem, eliminating dependence on expensive closed-source APIs. Experiments show that our 8B model trained with LoopTool significantly surpasses its 32B data generator and achieves new state-of-the-art results on the BFCL-v3 and ACEBench benchmarks for its scale. Our work demonstrates that closed-loop, self-refining data pipelines can dramatically enhance the tool-use capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14195v1">N-GLARE: An Non-Generative Latent Representation-Efficient LLM Safety Evaluator</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Evaluating the safety robustness of LLMs is critical for their deployment. However, mainstream Red Teaming methods rely on online generation and black-box output analysis. These approaches are not only costly but also suffer from feedback latency, making them unsuitable for agile diagnostics after training a new model. To address this, we propose N-GLARE (A Non-Generative, Latent Representation-Efficient LLM Safety Evaluator). N-GLARE operates entirely on the model's latent representations, bypassing the need for full text generation. It characterizes hidden layer dynamics by analyzing the APT (Angular-Probabilistic Trajectory) of latent representations and introducing the JSS (Jensen-Shannon Separability) metric. Experiments on over 40 models and 20 red teaming strategies demonstrate that the JSS metric exhibits high consistency with the safety rankings derived from Red Teaming. N-GLARE reproduces the discriminative trends of large-scale red-teaming tests at less than 1\% of the token cost and the runtime cost, providing an efficient output-free evaluation proxy for real-time diagnostics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14182v1">WebRec: Enhancing LLM-based Recommendations with Attention-guided RAG from Web</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Recommender systems play a vital role in alleviating information overload and enriching users' online experience. In the era of large language models (LLMs), LLM-based recommender systems have emerged as a prevalent paradigm for advancing personalized recommendations. Recently, retrieval-augmented generation (RAG) has drawn growing interest to facilitate the recommendation capability of LLMs, incorporating useful information retrieved from external knowledge bases. However, as a rich source of up-to-date information, the web remains under-explored by existing RAG-based recommendations. In particular, unique challenges are posed from two perspectives: one is to generate effective queries for web retrieval, considering the inherent knowledge gap between web search and recommendations; another challenge lies in harnessing online websites that contain substantial noisy content. To tackle these limitations, we propose WebRec, a novel web-based RAG framework, which takes advantage of the reasoning capability of LLMs to interpret recommendation tasks into queries of user preferences that cater to web retrieval. Moreover, given noisy web-retrieved information, where relevant pieces of evidence are scattered far apart, an insightful MP-Head is designed to enhance LLM attentions between distant tokens of relevant information via message passing. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed web-based RAG methods in recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14181v1">Harnessing Deep LLM Participation for Robust Entity Linking</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Entity Linking (EL), the task of mapping textual entity mentions to their corresponding entries in knowledge bases, constitutes a fundamental component of natural language understanding. Recent advancements in Large Language Models (LLMs) have demonstrated remarkable potential for enhancing EL performance. Prior research has leveraged LLMs to improve entity disambiguation and input representation, yielding significant gains in accuracy and robustness. However, these approaches typically apply LLMs to isolated stages of the EL task, failing to fully integrate their capabilities throughout the entire process. In this work, we introduce DeepEL, a comprehensive framework that incorporates LLMs into every stage of the entity linking task. Furthermore, we identify that disambiguating entities in isolation is insufficient for optimal performance. To address this limitation, we propose a novel self-validation mechanism that utilizes global contextual information, enabling LLMs to rectify their own predictions and better recognize cohesive relationships among entities within the same sentence. Extensive empirical evaluation across ten benchmark datasets demonstrates that DeepEL substantially outperforms existing state-of-the-art methods, achieving an average improvement of 2.6\% in overall F1 score and a remarkable 4% gain on out-of-domain datasets. These results underscore the efficacy of deep LLM integration in advancing the state-of-the-art in entity linking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19670v3">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 19 pages,8 figures
    </div>
    <details class="paper-abstract">
      We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18970v2">LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14140v1">Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07991v2">VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted at AAAI 2026 oral
    </div>
    <details class="paper-abstract">
      Competency Questions (CQs) play a crucial role in validating ontology design. While manually crafting CQs can be highly time-consuming and costly for ontology engineers, recent studies have explored the use of large language models (LLMs) to automate this process. However, prior approaches have largely evaluated generated CQs based on their similarity to existing datasets, which often fail to verify semantic pitfalls such as "Misusing allValuesFrom". Since such pitfalls cannot be reliably detected through rule-based methods, we propose a novel dataset and model of Validating Semantic Pitfalls in Ontology (VSPO) for CQ generation specifically designed to verify the semantic pitfalls. To simulate missing and misused axioms, we use LLMs to generate natural language definitions of classes and properties and introduce misalignments between the definitions and the ontology by removing axioms or altering logical operators (e.g., substituting union with intersection). We then fine-tune LLaMA-3.1-8B-Instruct to generate CQs that validate these semantic discrepancies between the provided definitions and the corresponding axioms. The resulting CQs can detect a broader range of modeling errors compared to existing public datasets. Our fine-tuned model demonstrates superior performance over baselines, showing 26% higher precision and 28.2% higher recall than GPT-4.1 in generating CQs for pitfall validation. This research enables automatic generation of TBox-validating CQs using LLMs, significantly reducing manual effort while improving semantic alignment between ontologies and expert knowledge. To the best of our knowledge, this is the first study to target semantic pitfall validation in CQ generation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01558v3">Predicting the Performance of Black-box LLMs through Self-Queries</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15902v3">IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14129v1">MalRAG: A Retrieval-Augmented LLM Framework for Open-set Malicious Traffic Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 13 pages, 13 figures. Intended for submission to IEEE Transactions on Information Forensics and Security (TIFS)
    </div>
    <details class="paper-abstract">
      Fine-grained identification of IDS-flagged suspicious traffic is crucial in cybersecurity. In practice, cyber threats evolve continuously, making the discovery of novel malicious traffic a critical necessity as well as the identification of known classes. Recent studies have advanced this goal with deep models, but they often rely on task-specific architectures that limit transferability and require per-dataset tuning. In this paper we introduce MalRAG, the first LLM driven retrieval-augmented framework for open-set malicious traffic identification. MalRAG freezes the LLM and operates via comprehensive traffic knowledge construction, adaptive retrieval, and prompt engineering. Concretely, we construct a multi-view traffic database by mining prior malicious traffic from content, structural, and temporal perspectives. Furthermore, we introduce a Coverage-Enhanced Retrieval Algorithm that queries across these views to assemble the most probable candidates, thereby improving the inclusion of correct evidence. We then employ Traffic-Aware Adaptive Pruning to select a variable subset of these candidates based on traffic-aware similarity scores, suppressing incorrect matches and yielding reliable retrieved evidence. Moreover, we develop a suite of guidance prompts where task instruction, evidence referencing, and decision guidance are integrated with the retrieved evidence to improve LLM performance. Across diverse real-world datasets and settings, MalRAG delivers state-of-the-art results in both fine-grained identification of known classes and novel malicious traffic discovery. Ablation and deep-dive analyses further show that MalRAG effective leverages LLM capabilities yet achieves open-set malicious traffic identification without relying on a specific LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14124v1">10Cache: Heterogeneous Resource-Aware Tensor Caching and Migration for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 This paper accepted for presentation to the 16th ACM Symposium on Cloud Computing (SOCC'25)
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) in the cloud faces growing memory bottlenecks due to the limited capacity and high cost of GPUs. While GPU memory offloading to CPU and NVMe has made large-scale training more feasible, existing approaches suffer from high tensor migration latency and suboptimal device memory utilization, ultimately increasing training time and cloud costs. To address these challenges, we present 10Cache, a resource-aware tensor caching and migration system that accelerates LLM training by intelligently coordinating memory usage across GPU, CPU, and NVMe tiers. 10Cache profiles tensor execution order to construct prefetch policies, allocates memory buffers in pinned memory based on tensor size distributions, and reuses memory buffers to minimize allocation overhead. Designed for cloud-scale deployments, 10Cache improves memory efficiency and reduces reliance on high-end GPUs. Across diverse LLM workloads, it achieves up to 2x speedup in training time, improves GPU cache hit rate by up to 86.6x, and increases CPU/GPU memory utilization by up to 2.15x and 1.33x, respectively, compared to state-of-the-art offloading methods. These results demonstrate that 10Cache is a practical and scalable solution for optimizing LLM training throughput and resource efficiency in cloud environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12977v2">ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14098v1">Collaborative QA using Interacting LLMs. Impact of Network Structure, Node Capability and Distributed Data</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      In this paper, we model and analyze how a network of interacting LLMs performs collaborative question-answering (CQA) in order to estimate a ground truth given a distributed set of documents. This problem is interesting because LLMs often hallucinate when direct evidence to answer a question is lacking, and these effects become more pronounced in a network of interacting LLMs. The hallucination spreads, causing previously accurate LLMs to hallucinate. We study interacting LLMs and their hallucination by combining novel ideas of mean-field dynamics (MFD) from network science and the randomized utility model from economics to construct a useful generative model. We model the LLM with a latent state that indicates if it is truthful or not with respect to the ground truth, and extend a tractable analytical model considering an MFD to model the diffusion of information in a directed network of LLMs. To specify the probabilities that govern the dynamics of the MFD, we propose a randomized utility model. For a network of LLMs, where each LLM has two possible latent states, we posit sufficient conditions for the existence and uniqueness of a fixed point and analyze the behavior of the fixed point in terms of the incentive (e.g., test-time compute) given to individual LLMs. We experimentally study and analyze the behavior of a network of $100$ open-source LLMs with respect to data heterogeneity, node capability, network structure, and sensitivity to framing on multiple semi-synthetic datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12215v2">Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence. While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in complex areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11056v2">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under "standard" and "reasoning-augmented" inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12376v2">BitSnap: Checkpoint Sparsification and Quantization in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 12 pages, numerous figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size and complexity, efficient checkpoint saving\&loading has become crucial for managing storage, memory usage, and fault tolerance in LLM training. The current works do not comprehensively take into account the optimization of these several aspects. This paper proposes a novel checkpoint sparsification and quantization method that adapts dynamically to different training stages and model architectures. We present a comprehensive analysis of existing lossy and lossless compression techniques, identify current limitations, and introduce our adaptive approach that balances compression ratio, speed, and precision impact throughout the training process. Experiments on different sizes of LLMs demonstrate that our bitmask-based sparsification method achieves 16x compression ratio without compromising model accuracy. Additionally, the cluster-based quantization method achieves 2x compression ratio with little precision loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09087v2">Tele-LLM-Hub: Building Context-Aware Multi-Agent LLM Systems for Telecom Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      This paper introduces Tele-LLM-Hub, a user friendly low-code solution for rapid prototyping and deployment of context aware multi-agent (MA) Large Language Model (LLM) systems tailored for 5G and beyond. As telecom wireless networks become increasingly complex, intelligent LLM applications must share a domainspecific understanding of network state. We propose TeleMCP, the Telecom Model Context Protocol, to enable structured and context-rich communication between agents in telecom environments. Tele-LLM-Hub actualizes TeleMCP through a low-code interface that supports agent creation, workflow composition, and interaction with software stacks such as srsRAN. Key components include a direct chat interface, a repository of pre-built systems, an Agent Maker leveraging finetuning with our RANSTRUCT framework, and an MA-Maker for composing MA workflows. The goal of Tele-LLM-Hub is to democratize the design of contextaware MA systems and accelerate innovation in next-generation wireless networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18646v2">Beyond Benchmark: LLMs Evaluation with an Anthropomorphic and Value-oriented Roadmap</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs), a disconnect persists between benchmark performance and real-world utility. Current evaluation frameworks remain fragmented, prioritizing technical metrics while neglecting holistic assessment for deployment. This survey introduces an anthropomorphic evaluation paradigm through the lens of human intelligence, proposing a novel three-dimensional taxonomy: Intelligence Quotient (IQ)-General Intelligence for foundational capacity, Emotional Quotient (EQ)-Alignment Ability for value-based interactions, and Professional Quotient (PQ)-Professional Expertise for specialized proficiency. For practical value, we pioneer a Value-oriented Evaluation (VQ) framework assessing economic viability, social impact, ethical alignment, and environmental sustainability. Our modular architecture integrates six components with an implementation roadmap. Through analysis of 200+ benchmarks, we identify key challenges including dynamic assessment needs and interpretability gaps. It provides actionable guidance for developing LLMs that are technically proficient, contextually relevant, and ethically sound. We maintain a curated repository of open-source evaluation resources at: https://github.com/onejune2018/Awesome-LLM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14023v1">Syn-STARTS: Synthesized START Triage Scenario Generation Framework for Scalable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Introducing an open dataset
    </div>
    <details class="paper-abstract">
      Triage is a critically important decision-making process in mass casualty incidents (MCIs) to maximize victim survival rates. While the role of AI in such situations is gaining attention for making optimal decisions within limited resources and time, its development and performance evaluation require benchmark datasets of sufficient quantity and quality. However, MCIs occur infrequently, and sufficient records are difficult to accumulate at the scene, making it challenging to collect large-scale realworld data for research use. Therefore, we developed Syn-STARTS, a framework that uses LLMs to generate triage cases, and verified its effectiveness. The results showed that the triage cases generated by Syn-STARTS were qualitatively indistinguishable from the TRIAGE open dataset generated by manual curation from training materials. Furthermore, when evaluating the LLM accuracy using hundreds of cases each from the green, yellow, red, and black categories defined by the standard triage method START, the results were found to be highly stable. This strongly indicates the possibility of synthetic data in developing high-performance AI models for severe and critical medical situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14977v1">SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14967v1">MermaidSeqBench: An Evaluation Benchmark for LLM-to-Mermaid Sequence Diagram Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated excellent capabilities in generating structured diagrams from natural language descriptions. In particular, they have shown great promise in generating sequence diagrams for software engineering, typically represented in a text-based syntax such as Mermaid. However, systematic evaluations in this space remain underdeveloped as there is a lack of existing benchmarks to assess the LLM's correctness in this task. To address this shortcoming, we introduce MermaidSeqBench, a human-verified and LLM-synthetically-extended benchmark for assessing an LLM's capabilities in generating Mermaid sequence diagrams from textual prompts. The benchmark consists of a core set of 132 samples, starting from a small set of manually crafted and verified flows. These were expanded via a hybrid methodology combining human annotation, in-context LLM prompting, and rule-based variation generation. Our benchmark uses an LLM-as-a-judge model to assess Mermaid sequence diagram generation across fine-grained metrics, including syntax correctness, activation handling, error handling, and practical usability. We perform initial evaluations on numerous state-of-the-art LLMs and utilize multiple LLM judge models to demonstrate the effectiveness and flexibility of our benchmark. Our results reveal significant capability gaps across models and evaluation modes. Our proposed benchmark provides a foundation for advancing research in structured diagram generation and for developing more rigorous, fine-grained evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14937v1">CIMemories: A Compositional Benchmark for Contextual Integrity of Persistent Memory in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly use persistent memory from past interactions to enhance personalization and task performance. However, this memory introduces critical risks when sensitive information is revealed in inappropriate contexts. We present CIMemories, a benchmark for evaluating whether LLMs appropriately control information flow from memory based on task context. CIMemories uses synthetic user profiles with over 100 attributes per user, paired with diverse task contexts in which each attribute may be essential for some tasks but inappropriate for others. Our evaluation reveals that frontier models exhibit up to 69% attribute-level violations (leaking information inappropriately), with lower violation rates often coming at the cost of task utility. Violations accumulate across both tasks and runs: as usage increases from 1 to 40 tasks, GPT-5's violations rise from 0.1% to 9.6%, reaching 25.1% when the same prompt is executed 5 times, revealing arbitrary and unstable behavior in which models leak different attributes for identical prompts. Privacy-conscious prompting does not solve this - models overgeneralize, sharing everything or nothing rather than making nuanced, context-dependent decisions. These findings reveal fundamental limitations that require contextually aware reasoning capabilities, not just better prompting or scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14903v1">It's LIT! Reliability-Optimized LLMs with Inspectable Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited remarkable capabilities across various domains. The ability to call external tools further expands their capability to handle real-world tasks. However, LLMs often follow an opaque reasoning process, which limits their usefulness in high-stakes domains where solutions need to be trustworthy to end users. LLMs can choose solutions that are unreliable and difficult to troubleshoot, even if better options are available. We address this issue by forcing LLMs to use external -- more reliable -- tools to solve problems when possible. We present a framework built on the tool-calling capabilities of existing LLMs to enable them to select the most reliable and easy-to-troubleshoot solution path, which may involve multiple sequential tool calls. We refer to this framework as LIT (LLMs with Inspectable Tools). In order to support LIT, we introduce a new and challenging benchmark dataset of 1,300 questions and a customizable set of reliability cost functions associated with a collection of specialized tools. These cost functions summarize how reliable each tool is and how easy it is to troubleshoot. For instance, a calculator is reliable across domains, whereas a linear prediction model is not reliable if there is distribution shift, but it is easy to troubleshoot. A tool that constructs a random forest is neither reliable nor easy to troubleshoot. These tools interact with the Harvard USPTO Patent Dataset and a new dataset of NeurIPS 2023 papers to solve mathematical, coding, and modeling problems of varying difficulty levels. We demonstrate that LLMs can achieve more reliable and informed problem-solving while maintaining task performance using our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25939v2">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design: achieving high-fidelity deception with low operational risk. However, despite a flurry of research since late 2022, progress has been incremental, and the field lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview of this new domain. We survey and systematize three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15745v1">Structured Extraction of Vulnerabilities in OpenVAS and Tenable WAS Reports Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 5 pages, 4 tables, 3 figures, submitted to ERRC/WRSeg 2025
    </div>
    <details class="paper-abstract">
      This paper proposes an automated LLM-based method to extract and structure vulnerabilities from OpenVAS and Tenable WAS scanner reports, converting unstructured data into a standardized format for risk management. In an evaluation using a report with 34 vulnerabilities, GPT-4.1 and DeepSeek achieved the highest similarity to the baseline (ROUGE-L greater than 0.7). The method demonstrates feasibility in transforming complex reports into usable datasets, enabling effective prioritization and future anonymization of sensitive data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11896v2">VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      The widespread reliance on open-source software dramatically increases the risk of vulnerability exploitation, underscoring the need for effective and scalable vulnerability detection (VD). Existing VD techniques, whether traditional machine learning-based or LLM-based approaches like prompt engineering, supervised fine-tuning, or off-policy preference optimization, remain fundamentally limited in their ability to perform context-aware analysis: They depend on fixed inputs or static preference datasets, cannot adaptively explore repository-level dependencies, and are constrained by function-level benchmarks that overlook critical vulnerability context. This paper introduces Vulnerability-Adaptive Policy Optimization (VULPO), an on-policy LLM reinforcement learning framework for context-aware VD. To support training and evaluation, we first construct ContextVul, a new dataset that augments high-quality function-level samples with lightweight method to extract repository-level context information. We then design multi-dimensional reward structuring that jointly captures prediction correctness, vulnerability localization accuracy, and the semantic relevance of vulnerability analysis, thereby guiding the model toward comprehensive contextual reasoning. To address the asymmetric difficulty of different vulnerability cases and mitigate reward hacking, VULPO incorporates label-level and sample-level difficulty-adaptive reward scaling, encouraging the model to explore challenging cases while maintaining balanced reward distribution. Extensive experiments demonstrate the superiority of our VULPO framework in context-aware VD: Our VULPO-4B substantially outperforms existing VD baselines based on prompt engineering and off-policy optimization, improving F1 by 85% over Qwen3-4B and achieving performance comparable to a 150x larger-scale model, DeepSeek-R1-0528.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.07939v2">Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14722v1">When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 submitted to NeurIPS 2025 workshop on Algorithmic Collective Action
    </div>
    <details class="paper-abstract">
      Fair resource division algorithms, like those implemented in Spliddit platform, have traditionally been considered difficult for the end users to manipulate due to its complexities. This paper demonstrates how Large Language Models (LLMs) can dismantle these protective barriers by democratizing access to strategic expertise. Through empirical analysis of rent division scenarios on Spliddit algorithms, we show that users can obtain actionable manipulation strategies via simple conversational queries to AI assistants. We present four distinct manipulation scenarios: exclusionary collusion where majorities exploit minorities, defensive counterstrategies that backfire, benevolent subsidization of specific participants, and cost minimization coalitions. Our experiments reveal that LLMs can explain algorithmic mechanics, identify profitable deviations, and generate specific numerical inputs for coordinated preference misreporting--capabilities previously requiring deep technical knowledge. These findings extend algorithmic collective action theory from classification contexts to resource allocation scenarios, where coordinated preference manipulation replaces feature manipulation. The implications reach beyond rent division to any domain using algorithmic fairness mechanisms for resource division. While AI-enabled manipulation poses risks to system integrity, it also creates opportunities for preferential treatment of equity deserving groups. We argue that effective responses must combine algorithmic robustness, participatory design, and equitable access to AI capabilities, acknowledging that strategic sophistication is no longer a scarce resource.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14688v1">Ground Truth Generation for Multilingual Historical NLP using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 13 pages, 5 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Historical and low-resource NLP remains challenging due to limited annotated data and domain mismatches with modern, web-sourced corpora. This paper outlines our work in using large language models (LLMs) to create ground-truth annotations for historical French (16th-20th centuries) and Chinese (1900-1950) texts. By leveraging LLM-generated ground truth on a subset of our corpus, we were able to fine-tune spaCy to achieve significant gains on period-specific tests for part-of-speech (POS) annotations, lemmatization, and named entity recognition (NER). Our results underscore the importance of domain-specific models and demonstrate that even relatively limited amounts of synthetic data can improve NLP tools for under-resourced corpora in computational humanities research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14661v1">M-CALLM: Multi-level Context Aware LLM Framework for Group Interaction Prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      This paper explores how large language models can leverage multi-level contextual information to predict group coordination patterns in collaborative mixed reality environments. We demonstrate that encoding individual behavioral profiles, group structural properties, and temporal dynamics as natural language enables LLMs to break through the performance ceiling of statistical models. We build M-CALLM, a framework that transforms multimodal sensor streams into hierarchical context for LLM-based prediction, and evaluate three paradigms (zero-shot prompting, few-shot learning, and supervised fine-tuning) against statistical baselines across intervention mode (real-time prediction) and simulation mode (autoregressive forecasting) Head-to-head comparison on 16 groups (64 participants, ~25 hours) demonstrates that context-aware LLMs achieve 96% accuracy for conversation prediction, a 3.2x improvement over LSTM baselines, while maintaining sub-35ms latency. However, simulation mode reveals brittleness with 83% degradation due to cascading errors. Deep-dive into modality-specific performance shows conversation depends on temporal patterns, proximity benefits from group structure (+6%), while shared attention fails completely (0% recall), exposing architectural limitations. We hope this work spawns new ideas for building intelligent collaborative sensing systems that balance semantic reasoning capabilities with fundamental constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18544v3">SLICE: SLO-Driven Scheduling for LLM Inference on Edge Computing Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 This work has been submitted to the IEEE for possible publication. This version is temporarily hosted anonymously for double-blind review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), as the foundational architecture for next-generation interactive AI applications, not only power intelligent dialogue systems but also drive the evolution of embodied intelligence on edge devices, including humanoid robots, smart vehicles, and other scenarios. The applications running on these edge devices impose differentiated Service Level Objectives (SLO) requirements on LLM services, specifically manifested as distinct constraints on Time to First Token (TTFT) and Time Per Output Token (TPOT) as well as end-to-end latency. Notably, edge devices typically handle real-time tasks that are extremely sensitive to latency, such as machine control and navigation planning. However, existing scheduling service systems still prioritize maximizing output token throughput as the sole optimization objective, failing to adequately address the diversity of SLO requirements. This ultimately results in persistently high violation rates for end-to-end latency or TPOT related SLOs. This paper proposes SLICE, an innovative scheduling solution designed for edge computing scenarios with differentiated SLO requirements. By combining a utility-maximizing request scheduling algorithm with a dynamic iterative control mechanism for generation rates, SLICE significantly improves LLM inference service SLO attainment. Experimental results demonstrate that compared to state-of-the-art solutions Orca and FastServe, SLICE achieves up to 35x higher SLO attainment and 3.4x advantage in task completion time than the other two solutions. This version is temporarily hosted anonymously for double-blind review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14617v1">Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 16 pages, 12 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become critical for advancing modern Large Language Models (LLMs), yet existing synchronous RL systems face severe performance bottlenecks. The rollout phase, which dominates end-to-end iteration time, suffers from substantial long-tail latency and poor resource utilization due to inherent workload imbalance. We present Seer, a novel online context learning system that addresses these challenges by exploiting previously overlooked similarities in output lengths and generation patterns among requests sharing the same prompt. Seer introduces three key techniques: divided rollout for dynamic load balancing, context-aware scheduling, and adaptive grouped speculative decoding. Together, these mechanisms substantially reduce long-tail latency and improve resource efficiency during rollout. Evaluations on production-grade RL workloads demonstrate that Seer improves end-to-end rollout throughput by 74% to 97% and reduces long-tail latency by 75% to 93% compared to state-of-the-art synchronous RL systems, significantly accelerating RL training iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10046v2">GraphCodeAgent: Dual Graph-Guided LLM Agent for Retrieval-Augmented Repo-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Writing code requires significant time and effort in software development. To automate this process, researchers have made substantial progress for code generation. Recently, large language models (LLMs) have demonstrated remarkable proficiency in function-level code generation, yet their performance significantly degrades in the real-world software development process, where coding tasks are deeply embedded within specific repository contexts. Existing studies attempt to use retrieval-augmented code generation (RACG) approaches to mitigate this demand. However, there is a gap between natural language (NL) requirements and programming implementations. This results in the failure to retrieve the relevant code of these fine-grained subtasks. To address this challenge, we propose GraphCodeAgent, a dual graph-guided LLM agent for retrieval-augmented repo-level code generation, bridging the gap between NL requirements and programming implementations. Our approach constructs two interconnected graphs: a Requirement Graph (RG) to model requirement relations of code snippets within the repository, as well as the relations between the target requirement and the requirements of these code snippets, and a Structural-Semantic Code Graph (SSCG) to capture the repository's intricate code dependencies. Guided by this, an LLM-powered agent performs multi-hop reasoning to systematically retrieve all context code snippets, including implicit and explicit code snippets, even if they are not explicitly expressed in requirements. We evaluated GraphCodeAgent on three advanced LLMs with the two widely-used repo-level code generation benchmarks DevEval and CoderEval. Extensive experiment results show that GraphCodeAgent significantly outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14584v1">ReflexGrad: Three-Way Synergistic Architecture for Zero-Shot Generalization in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Enabling agents to learn from experience and generalize across diverse tasks without task-specific training remains a fundamental challenge in reinforcement learning and decision-making. While recent approaches have explored episodic memory (Reflexion), gradient-based prompt optimization (TextGrad),and hierarchical task decomposition independently, their potential for synergistic integration remains unexplored. We introduce ReflexGrad, a novel architecture that tightly couples three complementary mechanisms: (1) LLM-based hierarchical TODO decomposition for strategic planning, (2) history-aware causal reflection that analyzes recent action patterns to identify failure root causes and enable within-trial learning, and (3) gradient-based optimization for systematic improvement. Unlike prior work relying on few-shot demonstrations, our system achieves true zero-shot generalization through pure LLM semantic reasoning,requiring no task-specific examples, fine-tuning, or hardcoded similarity metrics. Evaluated on ALFWorld benchmark tasks, ReflexGrad demonstrates 67% zero-shot success rate on Trial 0 without any prior task experience or demonstrations, establishing effective performance on first exposure. Through empirical analysis, we identify the architectural mechanisms underlying stable convergence (zero action loops) and effective cross-task transfer (67% to 78% improvement).Our work demonstrates that synergistic integration of complementary learning mechanisms enables robust zero-shot generalization that approaches few-shot baselines from prior work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14565v1">Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Robots can adapt to user preferences by learning reward functions from demonstrations, but with limited data, reward models often overfit to spurious correlations and fail to generalize. This happens because demonstrations show robots how to do a task but not what matters for that task, causing the model to focus on irrelevant state details. Natural language can more directly specify what the robot should focus on, and, in principle, disambiguate between many reward functions consistent with the demonstrations. However, existing language-conditioned reward learning methods typically treat instructions as simple conditioning signals, without fully exploiting their potential to resolve ambiguity. Moreover, real instructions are often ambiguous themselves, so naive conditioning is unreliable. Our key insight is that these two input types carry complementary information: demonstrations show how to act, while language specifies what is important. We propose Masked Inverse Reinforcement Learning (Masked IRL), a framework that uses large language models (LLMs) to combine the strengths of both input types. Masked IRL infers state-relevance masks from language instructions and enforces invariance to irrelevant state components. When instructions are ambiguous, it uses LLM reasoning to clarify them in the context of the demonstrations. In simulation and on a real robot, Masked IRL outperforms prior language-conditioned IRL methods by up to 15% while using up to 4.7 times less data, demonstrating improved sample-efficiency, generalization, and robustness to ambiguous language. Project page: https://MIT-CLEAR-Lab.github.io/Masked-IRL and Code: https://github.com/MIT-CLEAR-Lab/Masked-IRL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14528v1">LLM-Assisted Thematic Analysis: Opportunities, Limitations, and Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      [Context] Large Language Models (LLMs) are increasingly used to assist qualitative research in Software Engineering (SE), yet the methodological implications of this usage remain underexplored. Their integration into interpretive processes such as thematic analysis raises fundamental questions about rigor, transparency, and researcher agency. [Objective] This study investigates how experienced SE researchers conceptualize the opportunities, risks, and methodological implications of integrating LLMs into thematic analysis. [Method] A reflective workshop with 25 ISERN researchers guided participants through structured discussions of LLM-assisted open coding, theme generation, and theme reviewing, using color-coded canvases to document perceived opportunities, limitations, and recommendations. [Results] Participants recognized potential efficiency and scalability gains, but highlighted risks related to bias, contextual loss, reproducibility, and the rapid evolution of LLMs. They also emphasized the need for prompting literacy and continuous human oversight. [Conclusion] Findings portray LLMs as tools that can support, but not substitute, interpretive analysis. The study contributes to ongoing community reflections on how LLMs can responsibly enhance qualitative research in SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14510v1">CLO: Efficient LLM Inference System with CPU-Light KVCache Offloading via Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      The growth of million-token LLMs exposes the scalability limits of inference systems, where the KVCache dominates memory usage and data transfer overhead. Recent offloading systems migrate the KVCache to CPU memory and incorporate top-k attention to reduce the volume of data transferred from the CPU, while further applying system-level optimizations such as on-GPU caching and prefetching to lower transfer overhead. However, they overlook the CPU bottleneck in three aspects: (1) substantial overhead of fine-grained dynamic cache management performed on the CPU side, (2) significant transfer overhead from poor PCIe bandwidth utilization caused by heavy gathering operations at the CPU side, and (3) GPU runtime bubbles introduced by coarse-grained CPU-centric synchronization. To address these challenges, we propose CLO, a CPU-light KVCache offloading system via algorithm-system co-design. CLO features: (1) a coarse-grained head-wise approximate on-GPU caching strategy with negligible cache management cost, (2) seamless combination of data prefetching and on-GPU persistent caching for lower transfer overhead, (3) a zero-copy transfer engine to fully exploit PCIe bandwidth, and a GPU-centric synchronization method to eliminate GPU stalls. Evaluation on two widely-used LLMs demonstrates that CLO achieves comparable accuracy to state-of-the-art systems, while substantially minimizing CPU overhead, fully utilizing PCIe bandwidth, thus improving decoding throughput by 9.3%-66.6%. Our results highlight that algorithm-system co-design is essential for memory-constrained LLM inference on modern GPU platforms. We open source CLO at https://github.com/CommediaJW/CLO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14467v1">From Topology to Behavioral Semantics: Enhancing BGP Security by Understanding BGP's Language with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 18 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The trust-based nature of Border Gateway Protocol (BGP) makes it vulnerable to disruptions like prefix hijacking and misconfigurations, threatening routing stability. Traditional detection relies on manual inspection with limited scalability. Machine/Deep Learning (M/DL) approaches automate detection but suffer from suboptimal precision, limited generalizability, and high retraining costs. This is because existing methods focus on topological structures rather than comprehensive semantic characteristics of Autonomous Systems (ASes), often misinterpreting functionally similar but topologically distant ASes. To address this, we propose BGPShield, an anomaly detection framework built on LLM embeddings that captures the Behavior Portrait and Routing Policy Rationale of each AS beyond topology, such as operational scale and global role. We propose a segment-wise aggregation scheme to transform AS descriptions into LLM representations without information loss, and a lightweight contrastive reduction network to compress them into a semantic-consistent version. Using these representations, our AR-DTW algorithm aligns and accumulates semantic distances to reveal behavioral inconsistencies. Evaluated on 16 real-world datasets, BGPShield detects 100% of verified anomalies with a false discovery rate below 5%. Notably, the employed LLMs were released prior to evaluation events, verifying generalizability. Furthermore, BGPShield constructs representations for unseen ASes within one second, significantly outperforming BEAM which demands costly retraining (averaging 65 hours).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14460v1">Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-18
      | 💬 This paper serves as the technical report of the Agent-R1 project
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being explored for building Agents capable of active environmental interaction (e.g., via tool use) to solve complex problems. Reinforcement Learning (RL) is considered a key technology with significant potential for training such Agents; however, the effective application of RL to LLM Agents is still in its nascent stages and faces considerable challenges. Currently, this emerging field lacks in-depth exploration into RL approaches specifically tailored for the LLM Agent context, alongside a scarcity of flexible and easily extensible training frameworks designed for this purpose. To help advance this area, this paper first revisits and clarifies Reinforcement Learning methodologies for LLM Agents by systematically extending the Markov Decision Process (MDP) framework to comprehensively define the key components of an LLM Agent. Secondly, we introduce Agent-R1, a modular, flexible, and user-friendly training framework for RL-based LLM Agents, designed for straightforward adaptation across diverse task scenarios and interactive environments. We conducted experiments on Multihop QA benchmark tasks, providing initial validation for the effectiveness of our proposed methods and framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17052v2">SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) power many modern applications, but serving them at scale remains costly and resource-intensive. Current server-centric systems overlook consumer-grade GPUs at the edge. We introduce SpecEdge, an edge-assisted inference framework that splits LLM workloads between edge and server GPUs using a speculative decoding scheme, exchanging only token outputs over the network. SpecEdge employs proactive edge drafting to overlap edge token creation with server verification and pipeline-aware scheduling that interleaves multiple user requests to increase server-side throughput. Experiments show SpecEdge enhances overall cost efficiency by 1.91x through achieving 2.22x server throughput, and reduces inter token latency by 11.24% compared to a server-only baseline, introducing a scalable, cost-effective paradigm for LLM serving. The code is available at https://github.com/kaist-ina/specedge
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.21359v3">Can Machines Think Like Humans? A Behavioral Evaluation of LLM Agents in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM)-based agents increasingly engage with human society, how well do we understand their prosocial behaviors? We (1) investigate how LLM agents' prosocial behaviors can be induced by different personas and benchmarked against human behaviors; and (2) introduce a social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal that merely assigning a human-like identity to LLMs does not produce human-like behaviors. These findings suggest that LLM agents' reasoning does not consistently exhibit textual markers of human decision-making in dictator games and that their alignment with human behavior varies substantially across model architectures and prompt formulations; even worse, such dependence does not follow a clear pattern. As society increasingly integrates machine intelligence, "Prosocial AI" emerges as a promising and urgent research direction in philanthropic studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13900v1">What Works for 'Lost-in-the-Middle' in LLMs? A Study on GM-Extract and Mitigations</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 To be submitted for publication
    </div>
    <details class="paper-abstract">
      The diminishing ability of large language models (LLMs) to effectively utilize long-range context-the "lost-in-the-middle" phenomenon-poses a significant challenge in retrieval-based LLM applications. To study the impact of this phenomenon in a real-world application setting, we introduce GM-Extract, a novel benchmark dataset meticulously designed to evaluate LLM performance on retrieval of control variables. To accurately diagnose failure modes, we propose a simple yet elegant evaluation system using two distinct metrics: one for spatial retrieval capability (Document Metric) and the other for semantic retrieval capability (Variable Extraction Metric). We conduct a systematic evaluation of 7-8B parameter models on two multi-document tasks (key-value extraction and question-answering), demonstrating a significant change in retrieval performance simply by altering how the data is represented in the context window. While a distinct U-shaped curve was not consistently observed, our analysis reveals a clear pattern of performance across models, which we further correlate with perplexity scores. Furthermore, we perform a literature survey of mitigation methods, which we categorize into two distinct approaches: black-box and white-box methods. We then apply these techniques to our benchmark, finding that their efficacy is highly nuanced. Our evaluation highlights scenarios where these strategies successfully improve performance, as well as surprising cases where they lead to a negative impact, providing a comprehensive understanding of their utility in a practical context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13876v1">QwenCLIP: Boosting Medical Vision-Language Pretraining via LLM Embeddings and Prompt tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This work has been submitted to the IEEE ISBI for possible publication
    </div>
    <details class="paper-abstract">
      Contrastive Language-Image Pretraining (CLIP) has demonstrated strong generalization for vision-language tasks in computer vision and medical domains, yet its text encoder accepts only up to 77 tokens, which limits its ability to represent long and information-rich radiology reports. Recent adaptations using domain-specific encoders, such as PubMedBERT or ClinicalBERT, mitigate this issue by leveraging medical corpora, but remain constrained by their limited input length (typically 512 tokens) and relatively shallow semantic understanding. To address these limitations, we propose QwenCLIP, a vision-language framework that replaces CLIP's text encoder with a large language model (LLM)-based embedding module (e.g., Qwen3-Embedding) and introduces learnable prompts to enhance cross-modal alignment. By leveraging the extended context window and richer representations of LLMs, QwenCLIP captures comprehensive medical semantics from long-form clinical text, substantially improving medical image-text alignment and downstream performance on radiology benchmarks. Our code is publicly available at https://github.com/Wxy-24/QwenCLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10459v2">LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v2">Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Withdrawing temporarily to coordinate revisions with co-authors. A revised version will be resubmitted
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13717v1">TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed on mobile devices offer benefits like user privacy and reduced network latency, but introduce a significant security risk: the leakage of proprietary models to end users. To mitigate this risk, we propose a system design for protecting on-device LLMs using Arm Trusted Execution Environment (TEE), TrustZone. Our system addresses two primary challenges: (1) The dilemma between memory efficiency and fast inference (caching model parameters within TEE memory). (2) The lack of efficient and secure Neural Processing Unit (NPU) time-sharing between Rich Execution Environment (REE) and TEE. Our approach incorporates two key innovations. First, we employ pipelined restoration, leveraging the deterministic memory access patterns of LLM inference to prefetch parameters on demand, hiding memory allocation, I/O and decryption latency under computation time. Second, we introduce a co-driver design, creating a minimal data plane NPU driver in the TEE that collaborates with the full-fledged REE driver. This reduces the TEE TCB size and eliminates control plane reinitialization overhead during NPU world switches. We implemented our system on the emerging OpenHarmony OS and the llama.cpp inference framework, and evaluated it with various LLMs on an Arm Rockchip device. Compared to a strawman TEE baseline lacking our optimizations, our system reduces TTFT by up to 90.9% and increases decoding speed by up to 23.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13676v1">T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to DATE 2026
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have outpaced the computational and memory capacities of edge platforms that primarily employ CPUs, thereby challenging efficient and scalable deployment. While ternary quantization enables significant resource savings, existing CPU solutions rely heavily on memory-based lookup tables (LUTs) which limit scalability, and FPGA or GPU accelerators remain impractical for edge use. This paper presents T-SAR, the first framework to achieve scalable ternary LLM inference on CPUs by repurposing the SIMD register file for dynamic, in-register LUT generation with minimal hardware modifications. T-SAR eliminates memory bottlenecks and maximizes data-level parallelism, delivering 5.6-24.5x and 1.1-86.2x improvements in GEMM latency and GEMV throughput, respectively, with only 3.2% power and 1.4% area overheads in SIMD units. T-SAR achieves up to 2.5-4.9x the energy efficiency of an NVIDIA Jetson AGX Orin, establishing a practical approach for efficient LLM inference on edge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13658v1">Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13640v1">Data Value in the Age of Scaling: Understanding LLM Scaling Dynamics Under Real-Synthetic Data Mixtures</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) is fueled by the growing reliance on datasets that blend real and synthetic data. While synthetic data offers scalability and cost-efficiency, it often introduces systematic distributional discrepancies, particularly underrepresenting long-tail knowledge due to truncation effects from data generation mechanisms like top-p sampling, temperature scaling, and finite sampling. These discrepancies pose fundamental challenges in characterizing and evaluating the utility of mixed real-synthetic datasets. In this paper, we identify a three-phase scaling behavior characterized by two breakpoints that reflect transitions in model behavior across learning head and tail knowledge. We further derive an LLM generalization bound designed for real and synthetic mixtures, revealing several key factors that govern their generalization performance. Building on our theoretical findings, we propose an effective yet efficient data valuation method that scales to large-scale datasets. Comprehensive experiments across four tasks, including image classification, sentiment classification, instruction following, and complex reasoning, demonstrate that our method surpasses state-of-the-art baselines in data valuation with significantly low computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.21323v2">LLM-driven Provenance Forensics for Threat Investigation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We introduce PROVSEEK, an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction. PROVSEEK employs specialized toolchains to dynamically retrieve relevant context by generating precise, context-aware queries that fuse knowledge from threat reports with evidence from system provenance data. The framework resolves provenance queries, orchestrates multiple role-specific agents, and synthesizes structured, ground-truth verifiable forensic summaries. By combining agent orchestration with Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning, data-guided filtration using a behavioral model, PROVSEEK enables adaptive multi-step analysis that iteratively refines hypotheses, verifies supporting evidence, and produces scalable, interpretable forensic explanations of attack behaviors. PROVSEEK is designed for automated threat investigation without task-specific training data, enabling forensic-style investigation even when no prior knowledge of the environment. We conduct a comprehensive evaluation on publicly available DARPA datasets, demonstrating that PROVSEEK outperforms retrieval-based methods for the intelligence extraction task, achieving a 34% improvement in contextual precision/recall; and for threat detection task, PROVSEEK achieves 22%/29% higher precision/recall compared to both a baseline agent approach and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection System (PIDS). In our scalability study, we show that PROVSEEK increases token usage by 1.42x and latency by 1.63x as the database size increases 50x, making it optimal for large-scale deployment. We also conducted an ablation and error analysis study to show how different components of PROVSEEK affect the detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.19662v3">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Quantization is critical for efficiently deploying large language models (LLMs). Yet conventional methods remain hardware-agnostic, limited to bit-width constraints, and do not account for intrinsic circuit characteristics such as the timing behaviors and energy profiles of Multiply-Accumulate (MAC) units. This disconnect from circuit-level behavior limits the ability to exploit available timing margins and energy-saving opportunities, reducing the overall efficiency of deployment on modern accelerators. To address these limitations, we propose HALO, a versatile framework for Hardware-Aware Post-Training Quantization (PTQ). Unlike traditional methods, HALO explicitly incorporates detailed hardware characteristics, including critical-path timing and power consumption, into its quantization approach. HALO strategically selects weights with low critical-path-delays enabling higher operational frequencies and dynamic frequency scaling without disrupting the architecture's dataflow. Remarkably, HALO achieves these improvements with only a few dynamic voltage and frequency scaling (DVFS) adjustments, ensuring simplicity and practicality in deployment. Additionally, by reducing switching activity within the MAC units, HALO effectively lowers energy consumption. Evaluations on accelerators such as Tensor Processing Units (TPUs) and Graphics Processing Units (GPUs) demonstrate that HALO significantly enhances inference efficiency, achieving average performance improvements of 270% and energy savings of 51% over baseline quantization methods, all with minimal impact on accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06390v2">Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted at AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19100v2">Personalizing Prostate Cancer Education for Patients Using an EHR-Integrated LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Cancer patients often lack timely education and personalized support due to clinician workload. This quality improvement study develops and evaluates a Large Language Model (LLM) agent, MedEduChat, which is integrated with the clinic's electronic health records (EHR) and designed to enhance prostate cancer patient education. Fifteen non-metastatic prostate cancer patients and three clinicians recruited from the Mayo Clinic interacted with the agent between May 2024 and April 2025. Findings showed that MedEduChat has a high usability score (UMUX 83.7 out of 100) and improves patients' health confidence (Health Confidence Score rose from 9.9 to 13.9). Clinicians evaluated the patient-chat interaction history and rated MedEduChat as highly correct (2.9 out of 3), complete (2.7 out of 3), and safe (2.7 out of 3), with moderate personalization (2.3 out of 3). This study highlights the potential of LLM agents to improve patient engagement and health education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19838v3">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Paper accepted to TMLR 2025, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents. The collection of papers reviewed in this survey will be hosted and regularly updated on the GitHub repository: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04486v2">EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EDIT-Bench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EDIT-Bench comprises of 540 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EDIT-Bench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EDIT-Bench is a challenging set of problems where only 1 model scores over 60%. We find that model performance varies across different categories of user instructions. Further, we find that varying levels of contextual information greatly affect task success rate, with performance varying up to 11%, indicating the importance of evaluating with realistic context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16124v3">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10950v2">Unveiling Challenges for LLMs in Enterprise Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise to automate data engineering on tabular data, offering enterprises a valuable opportunity to cut the high costs of manual data handling. But the enterprise domain comes with unique challenges that existing LLM-based approaches for data engineering often overlook, such as large table sizes, more complex tasks, and the need for internal knowledge. To bridge these gaps, we identify key enterprise-specific challenges related to data, tasks, and background knowledge and extensively evaluate how they affect data engineering with LLMs. Our analysis reveals that LLMs face substantial limitations in real-world enterprise scenarios, with accuracy declining sharply. Our findings contribute to a systematic understanding of LLMs for enterprise data engineering to support their adoption in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13373v1">A Novel Hierarchical Integration Method for Efficient Model Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face significant challenges in distributed healthcare, including consolidating specialized domain knowledge across institutions while maintaining privacy, reducing computational overhead, and preventing catastrophic forgetting during model updates.This paper presents a systematic evaluation of six parameter-space merging techniques applied to two architecturally compatible medical LLMs derived from the Mistral-7B base model. We introduce a novel hierarchical method that combines selective Optimal Transport (OT) alignment for attention layers with cosine similarity-weighted interpolation, designed to address permutation variance while minimizing computational overhead for edge deployment scenarios. Our study evaluates Task Arithmetic, Linear Averaging, DARE-TIES, DELLA, Breadcrumbs, and our Hierarchical approach across five medical benchmarks. Results demonstrate that architecturally compatible models benefit significantly from simple averaging methods, with Task Arithmetic achieving 45.80% accuracy on MedQA, outperforming complex pruning-based approaches. These findings offer critical insights for the deployment of distributed medical AI in resource-constrained IoT environments, where computational efficiency and model compatibility are paramount. Our work establishes that for architecturally compatible models, simple averaging provides a robust and computationally efficient baseline for knowledge consolidation, offering a pragmatic path forward for scalable medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04108v2">Can Linear Probes Measure LLM Uncertainty?</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Effective Uncertainty Quantification (UQ) represents a key aspect for reliable deployment of Large Language Models (LLMs) in automated decision-making and beyond. Yet, for LLM generation with multiple choice structure, the state-of-the-art in UQ is still dominated by the naive baseline given by the maximum softmax score. To address this shortcoming, we demonstrate that taking a principled approach via Bayesian statistics leads to improved performance despite leveraging the simplest possible model, namely linear regression. More precisely, we propose to train multiple Bayesian linear models, each predicting the output of a layer given the output of the previous one. Based on the obtained layer-level posterior distributions, we infer the global uncertainty level of the LLM by identifying a sparse combination of distributional features, leading to an efficient UQ scheme. Numerical experiments on various LLMs show consistent improvement over state-of-the-art baselines.
    </details>
</div>
