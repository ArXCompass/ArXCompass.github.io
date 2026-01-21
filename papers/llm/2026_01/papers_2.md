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

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05484v2">Evaluating LLM Safety Across Child Development Stages: A Simulated Agent Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Current safety alignment for Large Language Models (LLMs) implicitly optimizes for a "modal adult user," leaving models vulnerable to distributional shifts in user cognition. We present ChildSafe, a benchmark that quantifies alignment robustness under cognitive shifts corresponding to four developmental stages. Unlike static persona-based evaluations, we introduce a parametric cognitive simulation approach, formalizing developmental stages as hyperparameter constraints (e.g., volatility, context horizon) to generate out-of-distribution interaction traces. We validate these agents against ground-truth human linguistic data (CHILDES) and deploy them across 1,200 multi-turn interactions. Our results reveal a systematic alignment generalization gap: state-of-the-art models exhibit up to 11.5% performance degradation when interacting with early-childhood agents compared to standard baselines. We provide the research community with the validated agent artifacts and evaluation protocols to facilitate robust alignment testing against non-adversarial, cognitively diverse populations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23783v2">Boosting In-Context Learning in LLMs Through the Lens of Classical Supervised Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 We are withdrawing this submission due to an issue discovered in our analysis/evaluation pipeline that impacts the reported experimental findings. Until the results have been fully revalidated, we do not believe the current version provides a reliable basis for the conclusions. We intend to release an updated manuscript after re-running and cross-checking the experiments
    </div>
    <details class="paper-abstract">
      In-Context Learning (ICL) allows Large Language Models (LLMs) to adapt to new tasks with just a few examples, but their predictions often suffer from systematic biases, leading to unstable performances in classification. While calibration techniques are proposed to mitigate these biases, we show that, in the logit space, many of these methods are equivalent to merely shifting the LLM's decision boundary without having the ability to alter its orientation. This proves inadequate when biases cause the LLM to be severely misdirected. To address these limitations and provide a unifying framework, we propose Supervised Calibration (SC), a loss-minimization based framework which learns an optimal, per-class affine transformation of the LLM's predictive probabilities in the logit space without requiring external data beyond the context. By using a more expressive functional class, SC not only subsumes many existing calibration methods in ICL as special cases, but also enables the ability to alter and even completely reverse the orientation of the LLM's decision boundary. Furthermore, SC's loss-based nature facilitates the seamless integration of two purpose-built regularization techniques: context-invariance and directional trust-region. The former is designed to tackle the instability issue in ICL, while the latter controls the degree of calibration. Finally, SC delivers state-of-the-art performance over calibration baselines in the 4-shot, 8-shot, and 16-shot settings across all nine datasets for Mistral-7B-Instruct-v0.3, LLaMA-2-7B-chat, and Qwen2-7B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01412v2">Bias Association Discovery Framework for Open-Ended LLM Generations</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Social biases embedded in Large Language Models (LLMs) raise critical concerns, resulting in representational harms -- unfair or distorted portrayals of demographic groups -- that may be expressed in subtle ways through generated language. Existing evaluation methods often depend on predefined identity-concept associations, limiting their ability to surface new or unexpected forms of bias. In this work, we present the Bias Association Discovery Framework (BADF), a systematic approach for extracting both known and previously unrecognized associations between demographic identities and descriptive concepts from open-ended LLM outputs. Through comprehensive experiments spanning multiple models and diverse real-world contexts, BADF enables robust mapping and analysis of the varied concepts that characterize demographic identities. Our findings advance the understanding of biases in open-ended generation and provide a scalable tool for identifying and analyzing bias associations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12091v1">Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit systematic Western-centric bias, yet whether prompting in non-Western languages (e.g., Chinese) can mitigate this remains understudied. Answering this question requires rigorous evaluation and effective mitigation, but existing approaches fall short on both fronts: evaluation methods force outputs into predefined cultural categories without a neutral option, while mitigation relies on expensive multi-cultural corpora or agent frameworks that use functional roles (e.g., Planner--Critique) lacking explicit cultural representation. To address these gaps, we introduce CEBiasBench, a Chinese--English bilingual benchmark, and Multi-Agent Vote (MAV), which enables explicit ``no bias'' judgments. Using this framework, we find that Chinese prompting merely shifts bias toward East Asian perspectives rather than eliminating it. To mitigate such persistent bias, we propose Multi-Agent Cultural Debate (MACD), a training-free framework that assigns agents distinct cultural personas and orchestrates deliberation via a "Seeking Common Ground while Reserving Differences" strategy. Experiments demonstrate that MACD achieves 57.6% average No Bias Rate evaluated by LLM-as-judge and 86.0% evaluated by MAV (vs. 47.6% and 69.0% baseline using GPT-4o as backbone) on CEBiasBench and generalizes to the Arabic CAMeL benchmark, confirming that explicit cultural representation in agent frameworks is essential for cross-cultural fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12078v1">Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at general-purpose tasks, yet adapting their responses to individual users remains challenging. Retrieval augmentation provides a lightweight alternative to fine-tuning by conditioning LLMs on user history records, and existing approaches typically select these records based on semantic relevance. We argue that relevance serves as an unreliable proxy for utility: a record may be semantically similar to a query yet fail to improve generation quality or even degrade it due to redundancy or conflicting information. To bridge this gap, we propose PURPLE, a contextual bandit framework that oPtimizes UseR Profiles for Llm pErsonalization. In contrast to a greedy selection of the most relevant records, PURPLE treats profile construction as a set generation process and utilizes a Plackett-Luce ranking model to capture complex inter-record dependencies. By training with dense feedback provided by the likelihood of the reference response, our method aligns retrieval directly with generation quality. Extensive experiments on nine personalization tasks demonstrate that PURPLE consistently outperforms strong heuristic and retrieval-augmented baselines in both effectiveness and efficiency, establishing a principled and scalable solution for optimizing user profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12061v1">Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 Under Review for ACL 2026
    </div>
    <details class="paper-abstract">
      Dialogue Act (DA) annotation typically treats communicative or pedagogical intent as localized to individual utterances or turns. This leads annotators to agree on the underlying action while disagreeing on segment boundaries, reducing apparent reliability. We propose codebook-injected segmentation, which conditions boundary decisions on downstream annotation criteria, and evaluate LLM-based segmenters against standard and retrieval-augmented baselines. To assess these without gold labels, we introduce evaluation metrics for span consistency, distinctiveness, and human-AI distributional agreement. We found DA-awareness produces segments that are internally more consistent than text-only baselines. While LLMs excel at creating construct-consistent spans, coherence-based baselines remain superior at detecting global shifts in dialogue flow. Across two datasets, no single segmenter dominates. Improvements in within-segment coherence frequently trade off against boundary distinctiveness and human-AI distributional agreement. These results highlight segmentation as a consequential design choice that should be optimized for downstream objectives rather than a single performance score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08187v2">Improving LLM Reasoning with Homophily-aware Structural and Semantic Text-Attributed Graph Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising capabilities in Text-Attributed Graph (TAG) understanding. Recent studies typically focus on verbalizing the graph structures via handcrafted prompts, feeding the target node and its neighborhood context into LLMs. However, constrained by the context window, existing methods mainly resort to random sampling, often implemented via dropping node/edge randomly, which inevitably introduces noise and cause reasoning instability. We argue that graphs inherently contain rich structural and semantic information, and that their effective exploitation can unlock potential gains in LLMs reasoning performance. To this end, we propose Homophily-aware Structural and Semantic Compression for LLMs (HS2C), a framework centered on exploiting graph homophily. Structurally, guided by the principle of Structural Entropy minimization, we perform a global hierarchical partition that decodes the graph's essential topology. This partition identifies naturally cohesive, homophilic communities, while discarding stochastic connectivity noise. Semantically, we deliver the detected structural homophily to the LLM, empowering it to perform differentiated semantic aggregation based on predefined community type. This process compresses redundant background contexts into concise community-level consensus, selectively preserving semantically homophilic information aligned with the target nodes. Extensive experiments on 10 node-level benchmarks across LLMs of varying sizes and families demonstrate that, by feeding LLMs with structurally and semantically compressed inputs, HS2C simultaneously enhances the compression rate and downstream inference accuracy, validating its superiority and scalability. Extensions to 7 diverse graph-level benchmarks further consolidate HS2C's task generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12034v1">Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 Accepted to AAAI 2026 (Oral). 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Personalization in Large Language Models (LLMs) often relies on user-specific soft prompts. However, these prompts become obsolete when the foundation model is upgraded, necessitating costly, full-scale retraining. To overcome this limitation, we propose the Prompt-level User Migration Adapter (PUMA), a lightweight framework to efficiently migrate personalized prompts across incompatible models. PUMA utilizes a parameter-efficient adapter to bridge the semantic gap, combined with a group-based user selection strategy to significantly reduce training costs. Experiments on three large-scale datasets show our method matches or even surpasses the performance of retraining from scratch, reducing computational cost by up to 98%. The framework demonstrates strong generalization across diverse model architectures and robustness in advanced scenarios like chained and aggregated migrations, offering a practical path for the sustainable evolution of personalized AI by decoupling user assets from the underlying models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12033v1">Preserving Fairness and Safety in Quantized LLMs Through Critical Weight Protection</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Quantization is widely adopted to reduce the computational cost of large language models (LLMs); however, its implications for fairness and safety, particularly in dynamic quantization and multilingual contexts, remain underexplored. In this work, we conduct a systematic study of how static and dynamic quantization methods impact fairness and safety across benchmarks measuring intrinsic and extrinsic bias and safety alignment. For fairness, we evaluate English, French, Dutch, Spanish, and Turkish; for safety, we focus on English, Korean, and Arabic. Our findings reveal that quantization consistently degrades fairness and safety, with dynamic methods demonstrating greater stability than static ones. Moreover, fairness degradation varies across languages, while safety deterioration is especially pronounced in non-English settings. To address these risks, we introduce Critical Weight Protection, a novel technique that identifies and preserves fairness- and safety-critical weights during quantization. This approach effectively mitigates bias and safety deterioration without costly retraining or alignment, maintaining trustworthiness while retaining efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12019v1">Acting Flatterers via LLMs Sycophancy: Combating Clickbait with LLMs Opposing-Stance Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      The widespread proliferation of online content has intensified concerns about clickbait, deceptive or exaggerated headlines designed to attract attention. While Large Language Models (LLMs) offer a promising avenue for addressing this issue, their effectiveness is often hindered by Sycophancy, a tendency to produce reasoning that matches users' beliefs over truthful ones, which deviates from instruction-following principles. Rather than treating sycophancy as a flaw to be eliminated, this work proposes a novel approach that initially harnesses this behavior to generate contrastive reasoning from opposing perspectives. Specifically, we design a Self-renewal Opposing-stance Reasoning Generation (SORG) framework that prompts LLMs to produce high-quality agree and disagree reasoning pairs for a given news title without requiring ground-truth labels. To utilize the generated reasoning, we develop a local Opposing Reasoning-based Clickbait Detection (ORCD) model that integrates three BERT encoders to represent the title and its associated reasoning. The model leverages contrastive learning, guided by soft labels derived from LLM-generated credibility scores, to enhance detection robustness. Experimental evaluations on three benchmark datasets demonstrate that our method consistently outperforms LLM prompting, fine-tuned smaller language models, and state-of-the-art clickbait detection baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12014v1">Are LLMs Ready for TOON? Benchmarking Structural Correctness-Sustainability Trade-offs in Novel Structured Output Formats</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly required to generate structured, machine-readable outputs for downstream systems. While recent benchmarks have focused on evaluating the structural correctness of such outputs, the environmental impact of inference for different output formats has largely been overlooked. In this paper, we argue that structured output formats should be assessed not only in terms of correctness, but also with respect to their environmental efficiency. To this end, we introduce a sustainability-aware evaluation framework for structured generation that measures token usage, generation time, and estimated carbon emissions. Within this framework, we propose the Environment-Aware Generation Correctness Score (GCS_env), a unified metric that integrates structural correctness with carbon-aware efficiency. Using this framework, we systematically benchmark the novel TOON format against established representations (JSON, XML, YAML) across multiple LLMs spanning different architectures and parameter scales. Our results reveal a consistent trade-off: TOON yields markedly more compact outputs and lower emissions, but lower structural correctness when models lack native support. We show that increased model capacity reduces this gap and that environment-aware scoring can shift format rankings depending on deployment priorities. highlighting the need for sustainability-inclusive benchmarking and provides empirical evidence that compact representations such as TOON can offer practical advantages in large-scale, carbon-conscious LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06092v2">Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08626v2">How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at semantic understanding, yet their ability to reconstruct internal structure from scrambled inputs remains underexplored. Sentence-level restoration is ill-posed for automated evaluation because multiple valid word orders often exist. We introduce OrderProbe, a deterministic benchmark for structural reconstruction using fixed four-character expressions in Chinese, Japanese, and Korean, which have a unique canonical order and thus support exact-match scoring. We further propose a diagnostic framework that evaluates models beyond recovery accuracy, including semantic fidelity, logical validity, consistency, robustness sensitivity, and information density. Experiments on twelve widely used LLMs show that structural reconstruction remains difficult even for frontier systems: zero-shot recovery frequently falls below 35%. We also observe a consistent dissociation between semantic recall and structural planning, suggesting that structural robustness is not an automatic byproduct of semantic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11914v3">Forgetting-MarI: LLM Unlearning via Marginal Information Regularization</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11960v1">R$^2$PO: Decoupling Training Trajectories from Inference Responses for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning has become a central paradigm for improving LLM reasoning. However, existing methods use a single policy to produce both inference responses and training optimization trajectories. The objective conflict between generating stable inference responses and diverse training trajectories leads to insufficient exploration, which harms reasoning capability. In this paper, to address the problem, we propose R$^2$PO (Residual Rollout Policy Optimization), which introduces a lightweight Residual Rollout-Head atop the policy to decouple training trajectories from inference responses, enabling controlled trajectory diversification during training while keeping inference generation stable. Experiments across multiple benchmarks show that our method consistently outperforms baselines, achieving average accuracy gains of 3.1% on MATH-500 and 2.4% on APPS, while also reducing formatting errors and mitigating length bias for stable optimization. Our code is publicly available at https://github.com/RRPO-ARR/Code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11956v1">Double-Calibration: Towards Trustworthy LLMs via Calibrating Knowledge and Reasoning Confidence</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Trustworthy reasoning in Large Language Models (LLMs) is challenged by their propensity for hallucination. While augmenting LLMs with Knowledge Graphs (KGs) improves factual accuracy, existing KG-augmented methods fail to quantify epistemic uncertainty in both the retrieved evidence and LLMs' reasoning. To bridge this gap, we introduce DoublyCal, a framework built on a novel double-calibration principle. DoublyCal employs a lightweight proxy model to first generate KG evidence alongside a calibrated evidence confidence. This calibrated supporting evidence then guides a black-box LLM, yielding final predictions that are not only more accurate but also well-calibrated, with confidence scores traceable to the uncertainty of the supporting evidence. Experiments on knowledge-intensive benchmarks show that DoublyCal significantly improves both the accuracy and confidence calibration of black-box LLMs with low token cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17642v4">May the Feedback Be with You! Unlocking the Power of Feedback-Driven Deep Learning Framework Fuzzing via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Deep Learning (DL) frameworks have served as fundamental components in DL systems over the last decade. However, bugs in DL frameworks could lead to catastrophic consequences in critical scenarios. A simple yet effective way to find bugs in DL frameworks is fuzz testing (Fuzzing). Existing approaches focus on test generation, leaving execution results with high semantic value (e.g., coverage information, bug reports, and exception logs) in the wild, which can serve as multiple types of feedback. To fill this gap, we propose FUEL to effectively utilize the feedback information, which comprises two Large Language Models (LLMs): analysis LLM and generation LLM. Specifically, analysis LLM infers analysis summaries from feedback information, while the generation LLM creates tests guided by these summaries. Furthermore, based on multiple feedback guidance, we design two additional components: (i) a feedback-aware simulated annealing algorithm to select operators for test generation, enriching test diversity. (ii) a program self-repair strategy to automatically repair invalid tests, enhancing test validity. We evaluate FUEL on the two most popular DL frameworks, and experiment results show that FUEL can improve line code coverage of PyTorch and TensorFlow by 4.48% and 9.14% over four state-of-the-art baselines. By the time of submission, FUEL has detected 104 previously unknown bugs for PyTorch and TensorFlow, with 93 confirmed as new bugs, 53 already fixed. 14 vulnerabilities have been assigned CVE IDs, among which 7 are rated as high-severity with a CVSS score of "7.5 HIGH". Our artifact is available at https://github.com/NJU-iSE/FUEL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.15015v2">EduThink4AI: Bridging Educational Critical Thinking and Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential as educational tutoring agents, capable of tailoring hints, orchestrating lessons, and grading with near-human finesse across various academic domains. However, current LLM-based educational systems exhibit critical limitations in promoting genuine critical thinking, failing on over one-third of multi-hop questions with counterfactual premises, and remaining vulnerable to adversarial prompts that trigger biased or factually incorrect responses. To address these gaps, we propose \textbf{EDU-Prompting}, a novel multi-agent framework that bridges established educational critical thinking theories with LLM agent design to generate critical, bias-aware explanations while fostering diverse perspectives. Our systematic evaluation across theoretical benchmarks and practical college-level critical writing scenarios demonstrates that EDU-Prompting significantly enhances both content truthfulness and logical soundness in AI-generated educational responses. The framework's modular design enables seamless integration into existing prompting frameworks and educational applications, allowing practitioners to directly incorporate critical thinking catalysts that promote analytical reasoning and introduce multiple perspectives without requiring extensive system modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11920v1">Enhancing LLM-Based Data Annotation with Error Decomposition</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models offer a scalable alternative to human coding for data annotation tasks, enabling the scale-up of research across data-intensive domains. While LLMs are already achieving near-human accuracy on objective annotation tasks, their performance on subjective annotation tasks, such as those involving psychological constructs, is less consistent and more prone to errors. Standard evaluation practices typically collapse all annotation errors into a single alignment metric, but this simplified approach may obscure different kinds of errors that affect final analytical conclusions in different ways. Here, we propose a diagnostic evaluation paradigm that incorporates a human-in-the-loop step to separate task-inherent ambiguity from model-driven inaccuracies and assess annotation quality in terms of their potential downstream impacts. We refine this paradigm on ordinal annotation tasks, which are common in subjective annotation. The refined paradigm includes: (1) a diagnostic taxonomy that categorizes LLM annotation errors along two dimensions: source (model-specific vs. task-inherent) and type (boundary ambiguity vs. conceptual misidentification); (2) a lightweight human annotation test to estimate task-inherent ambiguity from LLM annotations; and (3) a computational method to decompose observed LLM annotation errors following our taxonomy. We validate this paradigm on four educational annotation tasks, demonstrating both its conceptual validity and practical utility. Theoretically, our work provides empirical evidence for why excessively high alignment is unrealistic in specific annotation tasks and why single alignment metrics inadequately reflect the quality of LLM annotations. In practice, our paradigm can be a low-cost diagnostic tool that assesses the suitability of a given task for LLM annotation and provides actionable insights for further technical optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11908v1">PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 23 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with reasoning over long contexts where relevant information is sparsely distributed. Although plan-and-execute frameworks mitigate this by decomposing tasks into planning and execution, their effectiveness is often limited by unreliable plan generation due to dependence on surface-level cues. Consequently, plans may be based on incorrect assumptions, and once a plan is formed, identifying what went wrong and revising it reliably becomes difficult, limiting the effectiveness of reactive refinement. To address this limitation, we propose PPA-Plan, a proactive planning strategy for long-context reasoning that focuses on preventing such failures before plan generation. PPA-Plan identifies potential logical pitfalls and false assumptions, formulates them as negative constraints, and conditions plan generation on explicitly avoiding these constraints. Experiments on long-context QA benchmarks show that executing plans generated by PPA-Plan consistently outperforms existing plan-and-execute methods and direct prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11903v1">AEMA: Verifiable Evaluation Framework for Trustworthy and Controlled Agentic LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 Workshop on W51: How Can We Trust and Control Agentic AI? Toward Alignment, Robustness, and Verifiability in Autonomous LLM Agents at AAAI 2026
    </div>
    <details class="paper-abstract">
      Evaluating large language model (LLM)-based multi-agent systems remains a critical challenge, as these systems must exhibit reliable coordination, transparent decision-making, and verifiable performance across evolving tasks. Existing evaluation approaches often limit themselves to single-response scoring or narrow benchmarks, which lack stability, extensibility, and automation when deployed in enterprise settings at multi-agent scale. We present AEMA (Adaptive Evaluation Multi-Agent), a process-aware and auditable framework that plans, executes, and aggregates multi-step evaluations across heterogeneous agentic workflows under human oversight. Compared to a single LLM-as-a-Judge, AEMA achieves greater stability, human alignment, and traceable records that support accountable automation. Our results on enterprise-style agent workflows simulated using realistic business scenarios demonstrate that AEMA provides a transparent and reproducible pathway toward responsible evaluation of LLM-based multi-agent systems. Keywords Agentic AI, Multi-Agent Systems, Trustworthy AI, Verifiable Evaluation, Human Oversight
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11893v1">Taming Various Privilege Escalation in LLM-Based Agent Systems: A Mandatory Access Control Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agent systems are increasingly deployed for complex real-world tasks but remain vulnerable to natural language-based attacks that exploit over-privileged tool use. This paper aims to understand and mitigate such attacks through the lens of privilege escalation, defined as agent actions exceeding the least privilege required for a user's intended task. Based on a formal model of LLM agent systems, we identify novel privilege escalation scenarios, particularly in multi-agent systems, including a variant akin to the classic confused deputy problem. To defend against both known and newly demonstrated privilege escalation, we propose SEAgent, a mandatory access control (MAC) framework built upon attribute-based access control (ABAC). SEAgent monitors agent-tool interactions via an information flow graph and enforces customizable security policies based on entity attributes. Our evaluations show that SEAgent effectively blocks various privilege escalation while maintaining a low false positive rate and negligible system overhead. This demonstrates its robustness and adaptability in securing LLM-based agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11886v1">Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      In high-stakes domains like medicine, it may be generally desirable for models to faithfully adhere to the context provided. But what happens if the context does not align with model priors or safety protocols? In this paper, we investigate how LLMs behave and reason when presented with counterfactual or even adversarial medical evidence. We first construct MedCounterFact, a counterfactual medical QA dataset that requires the models to answer clinical comparison questions (i.e., judge the efficacy of certain treatments, with evidence consisting of randomized controlled trials provided as context). In MedCounterFact, real-world medical interventions within the questions and evidence are systematically replaced with four types of counterfactual stimuli, ranging from unknown words to toxic substances. Our evaluation across multiple frontier LLMs on MedCounterFact reveals that in the presence of counterfactual evidence, existing models overwhelmingly accept such "evidence" at face value even when it is dangerous or implausible, and provide confident and uncaveated answers. While it may be prudent to draw a boundary between faithfulness and safety, our findings reveal that there exists no such boundary yet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05080v3">An Architectural Advantage of The Instruction-Tuned LLM in Containing The Readability-Accuracy Tension in Text Simplification</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models (LLMs), however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses two major classes of general-purpose LLMs, demonstrating how they navigate the readability-accuracy tension compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral-Small 3 24B and the reasoning-augmented QWen2.5 32B, we identify an architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance and a reasonable BERTScore of 0.89, but its operational strategy shows a disconnect in balancing between readability and accuracy. Additionally, a comprehensive correlation analysis of a suite of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies, and informs metric selection and domain adaptation for text simplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09159v2">LLMs Meet Isolation Kernel: Lightweight, Learning-free Binary Embeddings for Fast Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently enabled remarkable progress in text representation. However, their embeddings are typically high-dimensional, leading to substantial storage and retrieval overhead. Although recent approaches such as Matryoshka Representation Learning (MRL) and Contrastive Sparse Representation (CSR) alleviate these issues to some extent, they still suffer from retrieval accuracy degradation. This paper proposes \emph{Isolation Kernel Embedding} or IKE, a learning-free method that transforms an LLM embedding into a binary embedding using Isolation Kernel (IK). IKE is an ensemble of diverse (random) partitions, enabling robust estimation of ideal kernel in the LLM embedding space, thus reducing retrieval accuracy loss as the ensemble grows. Lightweight and based on binary encoding, it offers low memory footprint and fast bitwise computation, lowering retrieval latency. Experiments on multiple text retrieval datasets demonstrate that IKE offers up to 16.7x faster retrieval and 16x lower memory usage than LLM embeddings, while maintaining comparable or better accuracy. Compared to CSR and other compression methods, IKE consistently achieves the best balance between retrieval efficiency and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09212v2">Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-17
    </div>
    <details class="paper-abstract">
      Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11866v1">Advances in LLM Reasoning Enable Flexibility in Clinical Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2026-01-17
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved high accuracy on medical question-answer (QA) benchmarks, yet their capacity for flexible clinical reasoning has been debated. Here, we asked whether advances in reasoning LLMs improve their cognitive flexibility in clinical reasoning. We assessed reasoning models from the OpenAI, Grok, Gemini, Claude, and DeepSeek families on the medicine abstraction and reasoning corpus (mARC), an adversarial medical QA benchmark which utilizes the Einstellung effect to induce inflexible overreliance on learned heuristic patterns in contexts where they become suboptimal. We found that strong reasoning models avoided Einstellung-based traps more often than weaker reasoning models, achieving human-level performance on mARC. On questions most commonly missed by physicians, the top 5 performing models answered 55% to 70% correctly with high confidence, indicating that these models may be less susceptible than humans to Einstellung effects. Our results indicate that strong reasoning models demonstrate improved flexibility in medical reasoning, achieving performance on par with humans on mARC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v2">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11479v1">Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder preferences. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we propose a hybrid framework that systematically integrates expert knowledge with optimization techniques. Classical optimization methods provide theoretical guarantees but require explicit, quantitative objectives, whereas stakeholder criteria are often articulated in natural language and difficult to formalize. To bridge these domains, we develop the Large language model and Extended Greedy (LEG) framework. Our framework combines a provable approximation algorithm for population coverage optimization with LLM-driven iterative refinement that incorporates human-AI alignment to ensure solutions reflect expert qualitative guidance while preserving coverage guarantees. Experiments on real-world data from three Ethiopian regions demonstrate the framework's effectiveness and its potential to inform equitable, data-driven health system planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12537v3">What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11468v1">Exploring LLM Features in Predictive Process Monitoring for Small-Scale Event-Logs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 19 pages, 4 figure, TMIS journal submission
    </div>
    <details class="paper-abstract">
      Predictive Process Monitoring is a branch of process mining that aims to predict the outcome of an ongoing process. Recently, it leveraged machine-and-deep learning architectures. In this paper, we extend our prior LLM-based Predictive Process Monitoring framework, which was initially focused on total time prediction via prompting. The extension consists of comprehensively evaluating its generality, semantic leverage, and reasoning mechanisms, also across multiple Key Performance Indicators. Empirical evaluations conducted on three distinct event logs and across the Key Performance Indicators of Total Time and Activity Occurrence prediction indicate that, in data-scarce settings with only 100 traces, the LLM surpasses the benchmark methods. Furthermore, the experiments also show that the LLM exploits both its embodied prior knowledge and the internal correlations among training traces. Finally, we examine the reasoning strategies employed by the model, demonstrating that the LLM does not merely replicate existing predictive methods but performs higher-order reasoning to generate the predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.03585v2">Causal-SAM-LLM: Large Language Models as Causal Reasoners for Robust Medical Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted by IEEE ICASSP 2026
    </div>
    <details class="paper-abstract">
      The clinical utility of deep learning models for medical image segmentation is severely constrained by their inability to generalize to unseen domains. This failure is often rooted in the models learning spurious correlations between anatomical content and domain-specific imaging styles. To overcome this fundamental challenge, we introduce Causal-SAM-LLM, a novel framework that elevates Large Language Models (LLMs) to the role of causal reasoners. Our framework, built upon a frozen Segment Anything Model (SAM) encoder, incorporates two synergistic innovations. First, Linguistic Adversarial Disentanglement (LAD) employs a Vision-Language Model to generate rich, textual descriptions of confounding image styles. By training the segmentation model's features to be contrastively dissimilar to these style descriptions, it learns a representation robustly purged of non-causal information. Second, Test-Time Causal Intervention (TCI) provides an interactive mechanism where an LLM interprets a clinician's natural language command to modulate the segmentation decoder's features in real-time, enabling targeted error correction. We conduct an extensive empirical evaluation on a composite benchmark from four public datasets (BTCV, CHAOS, AMOS, BraTS), assessing generalization under cross-scanner, cross-modality, and cross-anatomy settings. Causal-SAM-LLM establishes a new state of the art in out-of-distribution (OOD) robustness, improving the average Dice score by up to 6.2 points and reducing the Hausdorff Distance by 15.8 mm over the strongest baseline, all while using less than 9% of the full model's trainable parameters. Our work charts a new course for building robust, efficient, and interactively controllable medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02370v2">LLM-as-evaluator in Strategy Research: A Normative, Variance-Aware Protocol</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 41 pages for the main paper 53 pages for appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer strategy researchers powerful tools for annotating text at scale, but treating LLM-generated labels as deterministic overlooks substantial instability. Grounded in content analysis and generalizability theory, we diagnose five variance sources: construct specification, interface effects, model preferences, output extraction, and system-level aggregation. Empirical demonstrations show that minor design choices-prompt phrasing, model selection-can shift outcomes by 12-85 percentage points. Such variance threatens not only reproducibility but econometric identification: annotation errors correlated with covariates bias parameter estimates regardless of average accuracy. We develop a variance-aware protocol specifying sampling budgets, aggregation rules, and reporting standards, and delineate scope conditions where LLM annotation should not be used. These contributions transform LLM-based annotation from ad hoc practice into auditable measurement infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11379v1">Evaluating LLM Behavior in Hiring: Implicit Weights, Fairness Across Groups, and Alignment with Human Preferences</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      General-purpose Large Language Models (LLMs) show significant potential in recruitment applications, where decisions require reasoning over unstructured text, balancing multiple criteria, and inferring fit and competence from indirect productivity signals. Yet, it is still uncertain how LLMs assign importance to each attribute and whether such assignments are in line with economic principles, recruiter preferences or broader societal norms. We propose a framework to evaluate an LLM's decision logic in recruitment, by drawing on established economic methodologies for analyzing human hiring behavior. We build synthetic datasets from real freelancer profiles and project descriptions from a major European online freelance marketplace and apply a full factorial design to estimate how a LLM weighs different match-relevant criteria when evaluating freelancer-project fit. We identify which attributes the LLM prioritizes and analyze how these weights vary across project contexts and demographic subgroups. Finally, we explain how a comparable experimental setup could be implemented with human recruiters to assess alignment between model and human decisions. Our findings reveal that the LLM weighs core productivity signals, such as skills and experience, but interprets certain features beyond their explicit matching value. While showing minimal average discrimination against minority groups, intersectional effects reveal that productivity signals carry different weights between demographic groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11578v3">Efficient LLM Collaboration via Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated strong performance, ranging from simple to complex tasks. However, while large proprietary models (e.g., models with over 100B parameters) achieve remarkable results across diverse tasks, they are often accessible through costly APIs, making frequent use too costly for many applications. In contrast, small open-source models (e.g., models with fewer than 3B parameters) are freely available and easy to deploy locally, but their performance on complex tasks remains limited. This trade-off raises a natural question: how can small and large models efficiently collaborate to combine their complementary strengths? To bridge this trade-off, we propose COPE, a test-time collaboration framework. A planner model first generates a plan that serves as a lightweight intermediate that guides a downstream executor model. Small and large models take turns acting as planner and executor, exchanging plans in a multi-stage cascade to collaboratively solve tasks. Through comprehensive experiments on benchmarks spanning mathematical reasoning, code generation, open-ended tasks, and agent tasks, we demonstrate that COPE achieves performance comparable to large proprietary models, while drastically reducing the inference API cost. These results highlight planning as an effective prior for cost-efficient inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11369v1">Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Multi-agent LLM ensembles can converge on coordinated, socially harmful equilibria. This paper advances an experimental framework for evaluating Institutional AI, our system-level approach to AI alignment that reframes alignment from preference engineering in agent-space to mechanism design in institution-space. Central to this approach is the governance graph, a public, immutable manifest that declares legal states, transitions, sanctions, and restorative paths; an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance. We apply the Institutional AI framework to govern the Cournot collusion case documented by prior work and compare three regimes: Ungoverned (baseline incentives from the structure of the Cournot market), Constitutional (a prompt-only policy-as-prompt prohibition implemented as a fixed written anti-collusion constitution, and Institutional (governance-graph-based). Across six model configurations including cross-provider pairs (N=90 runs/condition), the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure. These results suggest that multi-agent alignment may benefit from being framed as an institutional design problem, where governance graphs can provide a tractable abstraction for alignment-relevant collective behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10321v2">An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09122v2">Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 ICIT2026
    </div>
    <details class="paper-abstract">
      Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11344v1">How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in drafting responses to patient portal messages, yet their integration into clinical workflows raises various concerns, including whether they would actually save clinicians time and effort in their portal workload. We investigate LLM alignment with individual clinicians through a comprehensive evaluation of the patient message response drafting task. We develop a novel taxonomy of thematic elements in clinician responses and propose a novel evaluation framework for assessing clinician editing load of LLM-drafted responses at both content and theme levels. We release an expert-annotated dataset and conduct large-scale evaluations of local and commercial LLMs using various adaptation techniques including thematic prompting, retrieval-augmented generation, supervised fine-tuning, and direct preference optimization. Our results reveal substantial epistemic uncertainty in aligning LLM drafts with clinician responses. While LLMs demonstrate capability in drafting certain thematic elements, they struggle with clinician-aligned generation in other themes, particularly question asking to elicit further information from patients. Theme-driven adaptation strategies yield improvements across most themes. Our findings underscore the necessity of adapting LLMs to individual clinician preferences to enable reliable and responsible use in patient-clinician communication workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22753v2">Opportunities and Challenges of LLMs in Education: An NLP Perspective</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v5">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11332v1">Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly succeed on competitive programming problems, yet existing evaluations conflate algorithmic reasoning with code-level implementation. We argue that competitive programming is fundamentally a problem-solving task and propose centering natural-language editorials in both solution generation and evaluation. Generating an editorial prior to code improves solve rates for some LLMs, with substantially larger gains when using expertly written gold editorials. However, even with gold editorials, models continue to struggle with implementation, while the gap between generated and gold editorials reveals a persistent problem-solving bottleneck in specifying correct and complete algorithms. Beyond pass/fail metrics, we diagnose reasoning errors by comparing model-generated editorials to gold standards using expert annotations and validate an LLM-as-a-judge protocol for scalable evaluation. We introduce a dataset of 83 ICPC-style problems with gold editorials and full test suites, and evaluate 19 LLMs, arguing that future benchmarks should explicitly separate problem solving from implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11314v1">Membership Inference on LLMs in the Wild</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Membership Inference Attacks (MIAs) act as a crucial auditing tool for the opaque training data of Large Language Models (LLMs). However, existing techniques predominantly rely on inaccessible model internals (e.g., logits) or suffer from poor generalization across domains in strict black-box settings where only generated text is available. In this work, we propose SimMIA, a robust MIA framework tailored for this text-only regime by leveraging an advanced sampling strategy and scoring mechanism. Furthermore, we present WikiMIA-25, a new benchmark curated to evaluate MIA performance on modern proprietary LLMs. Experiments demonstrate that SimMIA achieves state-of-the-art results in the black-box setting, rivaling baselines that exploit internal model information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20075v6">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 21 pages, main paper 9 pages. v5 contains an Italian translation of this paper by the author
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11293v1">One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted version in ECIR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping automated fact-checking (AFC) by enabling unified, end-to-end verification pipelines rather than isolated components. While large proprietary models achieve strong performance, their closed weights, complexity, and high costs limit sustainability. Fine-tuning smaller open weight models for individual AFC tasks can help but requires multiple specialized models resulting in high costs. We propose \textbf{multi-task learning (MTL)} as a more efficient alternative that fine-tunes a single model to perform claim detection, evidence ranking, and stance detection jointly. Using small decoder-only LLMs (e.g., Qwen3-4b), we explore three MTL strategies: classification heads, causal language modeling heads, and instruction-tuning, and evaluate them across model sizes, task orders, and standard non-LLM baselines. While multitask models do not universally surpass single-task baselines, they yield substantial improvements, achieving up to \textbf{44\%}, \textbf{54\%}, and \textbf{31\%} relative gains for claim detection, evidence re-ranking, and stance detection, respectively, over zero-/few-shot settings. Finally, we also provide practical, empirically grounded guidelines to help practitioners apply MTL with LLMs for automated fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11286v1">XChoice: Explainable Evaluation of AI-Human Alignment in LLM-based Constrained Choice Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      We present XChoice, an explainable framework for evaluating AI-human alignment in constrained decision making. Moving beyond outcome agreement such as accuracy and F1 score, XChoice fits a mechanism-based decision model to human data and LLM-generated decisions, recovering interpretable parameters that capture the relative importance of decision factors, constraint sensitivity, and implied trade-offs. Alignment is assessed by comparing these parameter vectors across models, options, and subgroups. We demonstrate XChoice on Americans' daily time allocation using the American Time Use Survey (ATUS) as human ground truth, revealing heterogeneous alignment across models and activities and salient misalignment concentrated in Black and married groups. We further validate robustness of XChoice via an invariance analysis and evaluate targeted mitigation with a retrieval augmented generation (RAG) intervention. Overall, XChoice provides mechanism-based metrics that diagnose misalignment and support informed improvements beyond surface outcome matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14161v3">MIST: Towards Multi-dimensional Implicit BiaS Evaluation of LLMs for Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) in Large Language Models (LLMs) refers to the model's ability to infer the mental states of others, with failures in this ability often manifesting as systemic implicit biases. Assessing this challenge is difficult, as traditional direct inquiry methods are often met with refusal to answer and fail to capture its subtle and multidimensional nature. Therefore, we propose MIST, which reconceptualizes the content model of stereotypes into multidimensional failures of ToM, specifically in the domains of competence, sociability, and morality. The framework introduces two indirect tasks. The Word Association Bias Test (WABT) assesses implicit lexical associations, while the Affective Attribution Test (AAT) measures implicit emotional tendencies, aiming to uncover latent stereotypes without triggering model avoidance. Through extensive experimentation on eight state-of-the-art LLMs, our framework demonstrates the ability to reveal complex bias structures and improved robustness. All data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11238v1">LLM-Assisted Pseudo-Relevance Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Accepted ECIR 2026
    </div>
    <details class="paper-abstract">
      Query expansion is a long-standing technique to mitigate vocabulary mismatch in ad hoc Information Retrieval. Pseudo-relevance feedback methods, such as RM3, estimate an expanded query model from the top-ranked documents, but remain vulnerable to topic drift when early results include noisy or tangential content. Recent approaches instead prompt Large Language Models to generate synthetic expansions or query variants. While effective, these methods risk hallucinations and misalignment with collection-specific terminology. We propose a hybrid alternative that preserves the robustness and interpretability of classical PRF while leveraging LLM semantic judgement. Our method inserts an LLM-based filtering stage prior to RM3 estimation: the LLM judges the documents in the initial top-$k$ ranking, and RM3 is computed only over those accepted as relevant. This simple intervention improves over blind PRF and a strong baseline across several datasets and metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04359v3">Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03687v2">MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Medical problem-solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction. With only a minimal subset of randomly sampled training examples and lightweight fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while significantly cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improvement, reducing reliance on external supervision and extensive task-specific fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05748v2">Communication Enables Cooperation in LLM Agents: A Comparison with Curriculum-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Eliciting cooperation in multi-agent LLM systems is critical for AI alignment. We investigate two approaches: direct communication and curriculum learning. In a 4-player Stag Hunt, a one-word "cheap talk" channel increases cooperation from 0% to 48.3%, demonstrating communication as a robust coordination mechanism. In contrast, we find that curriculum learning is highly sensitive to design choices: our pedagogical curriculum through progressively complex games reduced agent payoffs by 27.4% in an Iterated Public Goods Game with Punishment, demonstrating that optimizing for short-term rationality can actively undermine alignment goals. Qualitative analysis reveals that curricula emphasizing defection-equilibrium games can induce "learned pessimism" in agents. These findings suggest that for coordination problems, simple communication protocols may be more reliable than experience-based training, and that curriculum design for social dilemmas requires careful attention to the strategic lessons embedded in game sequences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11288v2">Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      Recent work has shown that narrow finetuning can produce broadly misaligned LLMs, a phenomenon termed emergent misalignment (EM). While concerning, these findings were limited to finetuning and activation steering, leaving out in-context learning (ICL). We therefore ask: does EM emerge in ICL? We find that it does: across four model families (Gemini, Kimi-K2, Grok, and Qwen), narrow in-context examples cause models to produce misaligned responses to benign, unrelated queries. With 16 in-context examples, EM rates range from 1\% to 24\% depending on model and domain, appearing with as few as 2 examples. Neither larger model scale nor explicit reasoning provides reliable protection. We formulate and test a hypothesis, which explains in-context EM as conflict between safety objectives and context-following behavior. Consistent with this, instructing models to prioritize safety reduces EM while prioritizing context-following increases it. These findings establish ICL as a previously underappreciated vector for emergent misalignment that operates without parameter modification and resists simple scaling-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11124v1">Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) adapted via contrastive learning excel in general representation learning but struggle in vertical domains like chemistry and law, primarily due to a lack of domain-specific knowledge. This work identifies a core bottleneck: the prevailing ``LLM+CL'' paradigm focuses on semantic alignment but cannot perform knowledge acquisition, leading to failures on specialized terminology. To bridge this gap, we propose Learn Before Represent (LBR), a novel two-stage framework. LBR first injects domain knowledge via an Information Bottleneck-Constrained Generative Learning stage, preserving the LLM's causal attention to maximize knowledge acquisition while compressing semantics. It then performs Generative-Refined Contrastive Learning on the compressed representations for alignment. This approach maintains architectural consistency and resolves the objective conflict between generative and contrastive learning. Extensive experiments on medical, chemistry, and code retrieval tasks show that LBR significantly outperforms strong baselines. Our work establishes a new paradigm for building accurate and robust representations in vertical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11118v1">Optimized Algorithms for Text Clustering with LLM-Generated Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 AAAI-26
    </div>
    <details class="paper-abstract">
      Clustering is a fundamental tool that has garnered significant interest across a wide range of applications including text analysis. To improve clustering accuracy, many researchers have incorporated background knowledge, typically in the form of must-link and cannot-link constraints, to guide the clustering process. With the recent advent of large language models (LLMs), there is growing interest in improving clustering quality through LLM-based automatic constraint generation. In this paper, we propose a novel constraint-generation approach that reduces resource consumption by generating constraint sets rather than using traditional pairwise constraints. This approach improves both query efficiency and constraint accuracy compared to state-of-the-art methods. We further introduce a constrained clustering algorithm tailored to the characteristics of LLM-generated constraints. Our method incorporates a confidence threshold and a penalty mechanism to address potentially inaccurate constraints. We evaluate our approach on five text datasets, considering both the cost of constraint generation and the overall clustering performance. The results show that our method achieves clustering accuracy comparable to the state-of-the-art algorithms while reducing the number of LLM queries by more than 20 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11063v1">H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11061v1">Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-16
      | 💬 Work in process
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is highly effective for enhancing LLM reasoning, yet recent evidence shows models like Qwen 2.5 achieve significant gains even with spurious or incorrect rewards. We investigate this phenomenon and identify a "Perplexity Paradox": spurious RLVR triggers a divergence where answer-token perplexity drops while prompt-side coherence degrades, suggesting the model is bypassing reasoning in favor of memorization. Using Path Patching, Logit Lens, JSD analysis, and Neural Differential Equations, we uncover a hidden Anchor-Adapter circuit that facilitates this shortcut. We localize a Functional Anchor in the middle layers (L18-20) that triggers the retrieval of memorized solutions, followed by Structural Adapters in later layers (L21+) that transform representations to accommodate the shortcut signal. Finally, we demonstrate that scaling specific MLP keys within this circuit allows for bidirectional causal steering-artificially amplifying or suppressing contamination-driven performance. Our results provide a mechanistic roadmap for identifying and mitigating data contamination in RLVR-tuned models. Code is available at https://github.com/idwts/How-RLVR-Activates-Memorization-Shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11060v1">Children's Expectations, Engagement, and Evaluation of an LLM-enabled Spherical Visualization Platform in the Classroom</a></div>
    <div class="paper-meta">
      📅 2026-01-16
    </div>
    <details class="paper-abstract">
      We present our first stage results from deploying an LLM-augmented visualization software in a classroom setting to engage primary school children with earth-related datasets. Motivated by the growing interest in conversational AI as a means to support inquiry-based learning, we investigate children's expectations, engagement, and evaluation of a spoken LLM interface with a shared, immersive visualization system in a formal educational context. Our system integrates a speech-capable large language model with an interactive spherical display. It enables children to ask natural-language questions and receive coordinated verbal explanations and visual responses through the LLM-augmented visualization updating in real time based on spoken queries. We report on a classroom study with Swedish children aged 9-10, combining structured observation and small-group discussions to capture expectations prior to interaction, interaction patterns during facilitated sessions, and children's reflections on their encounter afterward. Our results provide empirical insights into children's initial encounters with an LLM-enabled visualization platform within a classroom setting and their expectations, interactions, and evaluations of the system. These findings inform the technology's potential for educational use and highlight important directions for future research.
    </details>
</div>
