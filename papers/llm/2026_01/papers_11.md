# llm - 2026_01

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06827v1">PDR: A Plug-and-Play Positional Decay Framework for LLM Pre-training Data Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Detecting pre-training data in Large Language Models (LLMs) is crucial for auditing data privacy and copyright compliance, yet it remains challenging in black-box, zero-shot settings where computational resources and training data are scarce. While existing likelihood-based methods have shown promise, they typically aggregate token-level scores using uniform weights, thereby neglecting the inherent information-theoretic dynamics of autoregressive generation. In this paper, we hypothesize and empirically validate that memorization signals are heavily skewed towards the high-entropy initial tokens, where model uncertainty is highest, and decay as context accumulates. To leverage this linguistic property, we introduce Positional Decay Reweighting (PDR), a training-free and plug-and-play framework. PDR explicitly reweights token-level scores to amplify distinct signals from early positions while suppressing noise from later ones. Extensive experiments show that PDR acts as a robust prior and can usually enhance a wide range of advanced methods across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06818v1">AgentHallu: Benchmarking Automated Hallucination Attribution of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Project page: https://liuxuannan.github.io/AgentHallu.github.io/
    </div>
    <details class="paper-abstract">
      As LLM-based agents operate over sequential multi-step reasoning, hallucinations arising at intermediate steps risk propagating along the trajectory, thus degrading overall reliability. Unlike hallucination detection in single-turn responses, diagnosing hallucinations in multi-step workflows requires identifying which step causes the initial divergence. To fill this gap, we propose a new research task, automated hallucination attribution of LLM-based agents, aiming to identify the step responsible for the hallucination and explain why. To support this task, we introduce AgentHallu, a comprehensive benchmark with: (1) 693 high-quality trajectories spanning 7 agent frameworks and 5 domains, (2) a hallucination taxonomy organized into 5 categories (Planning, Retrieval, Reasoning, Human-Interaction, and Tool-Use) and 14 sub-categories, and (3) multi-level annotations curated by humans, covering binary labels, hallucination-responsible steps, and causal explanations. We evaluate 13 leading models, and results show the task is challenging even for top-tier models (like GPT-5, Gemini-2.5-Pro). The best-performing model achieves only 41.1\% step localization accuracy, where tool-use hallucinations are the most challenging at just 11.6\%. We believe AgentHallu will catalyze future research into developing robust, transparent, and reliable agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04566v2">BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13940v3">Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Code: https://github.com/EnVision-Research/MTI
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +9.28% average improvement on six benchmarks for DeepSeek-R1-7B and +11.25% on AIME2024 using Ling-mini-2.0-while remaining highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06798v1">Unleashing the Native Recommendation Potential: LLM-Based Generative Recommendation via Structured Term Identifiers</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Leveraging the vast open-world knowledge and understanding capabilities of Large Language Models (LLMs) to develop general-purpose, semantically-aware recommender systems has emerged as a pivotal research direction in generative recommendation. However, existing methods face bottlenecks in constructing item identifiers. Text-based methods introduce LLMs' vast output space, leading to hallucination, while methods based on Semantic IDs (SIDs) encounter a semantic gap between SIDs and LLMs' native vocabulary, requiring costly vocabulary expansion and alignment training. To address this, this paper introduces Term IDs (TIDs), defined as a set of semantically rich and standardized textual keywords, to serve as robust item identifiers. We propose GRLM, a novel framework centered on TIDs, employs Context-aware Term Generation to convert item's metadata into standardized TIDs and utilizes Integrative Instruction Fine-tuning to collaboratively optimize term internalization and sequential recommendation. Additionally, Elastic Identifier Grounding is designed for robust item mapping. Extensive experiments on real-world datasets demonstrate that GRLM significantly outperforms baselines across multiple scenarios, pointing a promising direction for generalizable and high-performance generative recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06786v1">EpiCaR: Knowing What You Don't Know Matters for Better Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Improving the reasoning abilities of large language models (LLMs) has largely relied on iterative self-training with model-generated data. While effective at boosting accuracy, existing approaches primarily reinforce successful reasoning paths, incurring a substantial calibration cost: models become overconfident and lose the ability to represent uncertainty. This failure has been characterized as a form of model collapse in alignment, where predictive distributions degenerate toward low-variance point estimates. We address this issue by reframing reasoning training as an epistemic learning problem, in which models must learn not only how to reason, but also when their reasoning should be trusted. We propose epistemically-calibrated reasoning (EpiCaR) as a training objective that jointly optimizes reasoning performance and calibration, and instantiate it within an iterative supervised fine-tuning framework using explicit self-evaluation signals. Experiments on Llama-3 and Qwen-3 families demonstrate that our approach achieves Pareto-superiority over standard baselines in both accuracy and calibration, particularly in models with sufficient reasoning capacity (e.g., 3B+). This framework generalizes effectively to OOD mathematical reasoning (GSM8K) and code generation (MBPP). Ultimately, our approach enables a 3X reduction in inference compute, matching the K=30 performance of STaR with only K=10 samples in capable models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11210v3">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06779v1">CyberLLM-FINDS 2025: Instruction-Tuned Fine-tuning of Domain-Specific LLMs with Retrieval-Augmented Generation and Graph Integration for MITRE Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as Gemma-2B have shown strong performance in various natural language processing tasks. However, general-purpose models often lack the domain expertise required for cybersecurity applications. This work presents a methodology to fine-tune the Gemma-2B model into a domain-specific cybersecurity LLM. We detail the processes of dataset preparation, fine-tuning, and synthetic data generation, along with implications for real-world applications in threat detection, forensic investigation, and attack analysis. Experiments highlight challenges in prompt length distribution during domain-specific fine-tuning. Uneven prompt lengths limit the model's effective use of the context window, constraining local inference to 200-400 tokens despite hardware support for longer sequences. Chain-of-thought styled prompts, paired with quantized weights, yielded the best performance under these constraints. To address context limitations, we employed a hybrid strategy using cloud LLMs for synthetic data generation and local fine-tuning for deployment efficiency. To extend the evaluation, we introduce a Retrieval-Augmented Generation (RAG) pipeline and graph-based reasoning framework. This approach enables structured alignment with MITRE ATT&CK techniques through STIX-based threat intelligence, enhancing recall in multi-hop and long-context scenarios. Graph modules encode entity-neighborhood context and tactic chains, helping mitigate the constraints of short prompt windows. Results demonstrate improved model alignment with tactic, technique, and procedure (TTP) coverage, validating the utility of graph-augmented LLMs in cybersecurity threat intelligence applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06776v1">From Text to Simulation: A Multi-Agent LLM Workflow for Automated Chemical Process Design</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Process simulation is a critical cornerstone of chemical engineering design. Current automated chemical design methodologies focus mainly on various representations of process flow diagrams. However, transforming these diagrams into executable simulation flowsheets remains a time-consuming and labor-intensive endeavor, requiring extensive manual parameter configuration within simulation software. In this work, we propose a novel multi-agent workflow that leverages the semantic understanding capabilities of large language models(LLMs) and enables iterative interactions with chemical process simulation software, achieving end-to-end automated simulation from textual process specifications to computationally validated software configurations for design enhancement. Our approach integrates four specialized agents responsible for task understanding, topology generation, parameter configuration, and evaluation analysis, respectively, coupled with Enhanced Monte Carlo Tree Search to accurately interpret semantics and robustly generate configurations. Evaluated on Simona, a large-scale process description dataset, our method achieves a 31.1% improvement in the simulation convergence rate compared to state-of-the-art baselines and reduces the design time by 89. 0% compared to the expert manual design. This work demonstrates the potential of AI-assisted chemical process design, which bridges the gap between conceptual design and practical implementation. Our workflow is applicable to diverse process-oriented industries, including pharmaceuticals, petrochemicals, food processing, and manufacturing, offering a generalizable solution for automated process design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.06472v2">KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 24 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Maintaining comprehensive and up-to-date knowledge graphs (KGs) is critical for modern AI systems, but manual curation struggles to scale with the rapid growth of scientific literature. This paper presents KARMA, a novel framework employing multi-agent large language models (LLMs) to automate KG enrichment through structured analysis of unstructured text. Our approach employs nine collaborative agents, spanning entity discovery, relation extraction, schema alignment, and conflict resolution that iteratively parse documents, verify extracted knowledge, and integrate it into existing graph structures while adhering to domain-specific schema. Experiments on 1,200 PubMed articles from three different domains demonstrate the effectiveness of KARMA in knowledge graph enrichment, with the identification of up to 38,230 new entities while achieving 83.1\% LLM-verified correctness and reducing conflict edges by 18.6\% through multi-layer assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25602v2">TRUE: A Reproducible Framework for LLM-Driven Relevance Judgment in Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      LLM-based relevance judgment generation has become a crucial approach in advancing evaluation methodologies in Information Retrieval (IR). It has progressed significantly, often showing high correlation with human judgments as reflected in LLMJudge leaderboards \cite{rahmani2025judging}. However, existing methods for relevance judgments, rely heavily on sensitive prompting strategies, lacking standardized workflows for generating reliable labels. To fill this gap, we reintroduce our method, \textit{Task-aware Rubric-based Evaluation} (TRUE), for relevance judgment generation. Originally developed for usefulness evaluation in search sessions, we extend TRUE to mitigate the gap in relevance judgment due to its demonstrated effectiveness and reproducible workflow. This framework leverages iterative data sampling and reasoning to evaluate relevance judgments across multiple factors including intent, coverage, specificity, accuracy and usefulness. In this paper, we evaluate TRUE on the TREC DL 2019, 2020 and LLMJudge datasets and our results show that TRUE achieves strong performance on the system-ranking LLM leaderboards. The primary focus of this work is to provide a reproducible framework for LLM-based relevance judgments, and we further analyze the effectiveness of TRUE across multiple dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07877v1">E^2-LLM: Bridging Neural Signals and Interpretable Affective Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Emotion recognition from electroencephalography (EEG) signals remains challenging due to high inter-subject variability, limited labeled data, and the lack of interpretable reasoning in existing approaches. While recent multimodal large language models (MLLMs) have advanced emotion analysis, they have not been adapted to handle the unique spatiotemporal characteristics of neural signals. We present E^2-LLM (EEG-to-Emotion Large Language Model), the first MLLM framework for interpretable emotion analysis from EEG. E^2-LLM integrates a pretrained EEG encoder with Qwen-based LLMs through learnable projection layers, employing a multi-stage training pipeline that encompasses emotion-discriminative pretraining, cross-modal alignment, and instruction tuning with chain-of-thought reasoning. We design a comprehensive evaluation protocol covering basic emotion prediction, multi-task reasoning, and zero-shot scenario understanding. Experiments on the dataset across seven emotion categories demonstrate that E^2-LLM achieves excellent performance on emotion classification, with larger variants showing enhanced reliability and superior zero-shot generalization to complex reasoning scenarios. Our work establishes a new paradigm combining physiological signals with LLM reasoning capabilities, showing that model scaling improves both recognition accuracy and interpretable emotional understanding in affective computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05623v3">Deployability-Centric Infrastructure-as-Code Generation: Fail, Learn, Refine, and Succeed through LLM-Empowered DevOps Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted by FSE 2026
    </div>
    <details class="paper-abstract">
      Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions. However, current evaluation focuses on syntactic correctness while ignoring deployability, the critical measure of the utility of IaC configuration files. Six state-of-the-art LLMs performed poorly on deployability, achieving only 20.8$\sim$30.2% deployment success rate on the first attempt. In this paper, we construct DPIaC-Eval, the first deployability-centric IaC template benchmark consisting of 153 real-world scenarios cross 58 unique services. Also, we propose an LLM-based deployability-centric framework, dubbed IaCGen, that uses iterative feedback mechanism encompassing format verification, syntax checking, and live deployment stages, thereby closely mirroring the real DevOps workflows. Results show that IaCGen can make 54.6$\sim$91.6% generated IaC templates from all evaluated models deployable in the first 10 iterations. Additionally, human-in-the-loop feedback that provide direct guidance for the deployability errors, can further boost the performance to over 90% passItr@25 on all evaluated LLMs. Furthermore, we explore the trustworthiness of the generated IaC templates on user intent alignment and security compliance. The poor performance (25.2% user requirement coverage and 8.4% security compliance rate) indicates a critical need for continued research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07084v1">How Secure is Secure Code Generation? Adversarial Prompts Put LLM Defenses to the Test</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Recent secure code generation methods, using vulnerability-aware fine-tuning, prefix-tuning, and prompt optimization, claim to prevent LLMs from producing insecure code. However, their robustness under adversarial conditions remains untested, and current evaluations decouple security from functionality, potentially inflating reported gains. We present the first systematic adversarial audit of state-of-the-art secure code generation methods (SVEN, SafeCoder, PromSec). We subject them to realistic prompt perturbations such as paraphrasing, cue inversion, and context manipulation that developers might inadvertently introduce or adversaries deliberately exploit. To enable fair comparison, we evaluate all methods under consistent conditions, jointly assessing security and functionality using multiple analyzers and executable tests. Our findings reveal critical robustness gaps: static analyzers overestimate security by 7 to 21 times, with 37 to 60% of ``secure'' outputs being non-functional. Under adversarial conditions, true secure-and-functional rates collapse to 3 to 17%. Based on these findings, we propose best practices for building and evaluating robust secure code generation methods. Our code is available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02598v2">LongDA: Benchmarking LLM Agents for Long-Document Data Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21144v2">NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07072v1">Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on retrieving information from external corpora. This creates a new attack surface: indirect prompt injection (IPI), where hidden instructions are planted in the corpora and hijack model behavior once retrieved. Previous studies have highlighted this risk but often avoid the hardest step: ensuring that malicious content is actually retrieved. In practice, unoptimized IPI is rarely retrieved under natural queries, which leaves its real-world impact unclear. We address this challenge by decomposing the malicious content into a trigger fragment that guarantees retrieval and an attack fragment that encodes arbitrary attack objectives. Based on this idea, we design an efficient and effective black-box attack algorithm that constructs a compact trigger fragment to guarantee retrieval for any attack fragment. Our attack requires only API access to embedding models, is cost-efficient (as little as $0.21 per target user query on OpenAI's embedding models), and achieves near-100% retrieval across 11 benchmarks and 8 embedding models (including both open-source models and proprietary services). Based on this attack, we present the first end-to-end IPI exploits under natural queries and realistic external corpora, spanning both RAG and agentic systems with diverse attack objectives. These results establish IPI as a practical and severe threat: when a user issued a natural query to summarize emails on frequently asked topics, a single poisoned email was sufficient to coerce GPT-4o into exfiltrating SSH keys with over 80% success in a multi-agent workflow. We further evaluate several defenses and find that they are insufficient to prevent the retrieval of malicious text, highlighting retrieval as a critical open vulnerability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20573v2">Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 2.0$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17331v2">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation combines eye-tracking data analysis, including exploratory explainable machine learning analysis with SHAP, and standardized questionnaires (SUS, IPQ, CSQ-VR, NASA-TLX) to examine user experience through both objective gaze-based measures and subjective self-reports of usability, presence, cybersickness, and cognitive load. Our findings show no statistically significant differences in usability, presence, or cybersickness between LLM-driven locomotion and established methods such as teleportation, suggesting its potential as a viable, natural language-based, hands-free alternative. In addition, eye-tracking analysis revealed patterns suggesting tendency toward increased user attention and engagement in the LLM-driven condition. Complementary to these findings, exploratory SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18882v4">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07006v1">LLM Performance Predictors: Learning When to Escalate in Hybrid Human-AI Moderation Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted as a full paper at the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into human-in-the-loop content moderation systems, a central challenge is deciding when their outputs can be trusted versus when escalation for human review is preferable. We propose a novel framework for supervised LLM uncertainty quantification, learning a dedicated meta-model based on LLM Performance Predictors (LPPs) derived from LLM outputs: log-probabilities, entropy, and novel uncertainty attribution indicators. We demonstrate that our method enables cost-aware selective classification in real-world human-AI workflows: escalating high-risk cases while automating the rest. Experiments across state-of-the-art LLMs, including both off-the-shelf (Gemini, GPT) and open-source (Llama, Qwen), on multimodal and multilingual moderation tasks, show significant improvements over existing uncertainty estimators in accuracy-cost trade-offs. Beyond uncertainty estimation, the LPPs enhance explainability by providing new insights into failure conditions (e.g., ambiguous content vs. under-specified policy). This work establishes a principled framework for uncertainty-aware, scalable, and responsible human-AI moderation workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07005v1">MicLog: Towards Accurate and Efficient LLM-based Log Parsing via Progressive Meta In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Log parsing converts semi-structured logs into structured templates, forming a critical foundation for downstream analysis. Traditional syntax and semantic-based parsers often struggle with semantic variations in evolving logs and data scarcity stemming from their limited domain coverage. Recent large language model (LLM)-based parsers leverage in-context learning (ICL) to extract semantics from examples, demonstrating superior accuracy. However, LLM-based parsers face two main challenges: 1) underutilization of ICL capabilities, particularly in dynamic example selection and cross-domain generalization, leading to inconsistent performance; 2) time-consuming and costly LLM querying. To address these challenges, we present MicLog, the first progressive meta in-context learning (ProgMeta-ICL) log parsing framework that combines meta-learning with ICL on small open-source LLMs (i.e., Qwen-2.5-3B). Specifically, MicLog: i) enhances LLMs' ICL capability through a zero-shot to k-shot ProgMeta-ICL paradigm, employing weighted DBSCAN candidate sampling and enhanced BM25 demonstration selection; ii) accelerates parsing via a multi-level pre-query cache that dynamically matches and refines recently parsed templates. Evaluated on Loghub-2.0, MicLog achieves 10.3% higher parsing accuracy than the state-of-the-art parser while reducing parsing time by 42.4%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06979v1">MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 Accepted to EMNLP 2025 (System Demonstrations)
    </div>
    <details class="paper-abstract">
      The learning process for medical residents presents significant challenges, demanding both the ability to interpret complex case reports and the rapid acquisition of accurate medical knowledge from reliable sources. Residents typically study case reports and engage in discussions with peers and mentors, but finding relevant educational materials and evidence to support their learning from these cases is often time-consuming and challenging. To address this, we introduce MedTutor, a novel system designed to augment resident training by automatically generating evidence-based educational content and multiple-choice questions from clinical case reports. MedTutor leverages a Retrieval-Augmented Generation (RAG) pipeline that takes clinical case reports as input and produces targeted educational materials. The system's architecture features a hybrid retrieval mechanism that synergistically queries a local knowledge base of medical textbooks and academic literature (using PubMed, Semantic Scholar APIs) for the latest related research, ensuring the generated content is both foundationally sound and current. The retrieved evidence is filtered and ordered using a state-of-the-art reranking model and then an LLM generates the final long-form output describing the main educational content regarding the case-report. We conduct a rigorous evaluation of the system. First, three radiologists assessed the quality of outputs, finding them to be of high clinical and educational value. Second, we perform a large scale evaluation using an LLM-as-a Judge to understand if LLMs can be used to evaluate the output of the system. Our analysis using correlation between LLMs outputs and human expert judgments reveals a moderate alignment and highlights the continued necessity of expert oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06973v1">LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing consistent public responses. We show theoretically that any agent restricted to the public conversation history cannot simultaneously preserve secrecy and consistency in PSITs, yielding an impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fail this test regardless of scale, demonstrating that semantic retrieval does not enable true state maintenance. To address this, we propose a novel architecture incorporating an explicit private working memory; we demonstrate that this mechanism restores consistency, establishing private state as a necessary component for interactive language agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17673v3">Bridging Symbolic Control and Neural Reasoning in LLM Agents: The Structured Cognitive Loop</a></div>
    <div class="paper-meta">
      📅 2026-01-11
      | 💬 The reference list has been updated to reflect recent work
    </div>
    <details class="paper-abstract">
      Large language model agents suffer from fundamental architectural problems: entangled reasoning and execution, memory volatility, and uncontrolled action sequences. We introduce Structured Cognitive Loop (SCL), a modular architecture that explicitly separates agent cognition into five phases: Retrieval, Cognition, Control, Action, and Memory (R-CCAM). At the core of SCL is Soft Symbolic Control, an adaptive governance mechanism that applies symbolic constraints to probabilistic inference, preserving neural flexibility while restoring the explainability and controllability of classical symbolic systems. Through empirical validation on multi-step conditional reasoning tasks, we demonstrate that SCL achieves zero policy violations, eliminates redundant tool calls, and maintains complete decision traceability. These results address critical gaps in existing frameworks such as ReAct, AutoGPT, and memory-augmented approaches. Our contributions are threefold: (1) we situate SCL within the taxonomy of hybrid intelligence, differentiating it from prompt-centric and memory-only approaches; (2) we formally define Soft Symbolic Control and contrast it with neuro-symbolic AI; and (3) we derive three design principles for trustworthy agents: modular decomposition, adaptive symbolic governance, and transparent state management. We provide a complete open-source implementation demonstrating the R-CCAM loop architecture, alongside a live GPT-4o-powered travel planning agent. By connecting expert system principles with modern LLM capabilities, this work offers a practical and theoretically grounded path toward reliable, explainable, and governable AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06966v1">RealMem: Benchmarking LLMs in Real-World Memory-Driven Interaction</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) evolve from static dialogue interfaces to autonomous general agents, effective memory is paramount to ensuring long-term consistency. However, existing benchmarks primarily focus on casual conversation or task-oriented dialogue, failing to capture **"long-term project-oriented"** interactions where agents must track evolving goals. To bridge this gap, we introduce **RealMem**, the first benchmark grounded in realistic project scenarios. RealMem comprises over 2,000 cross-session dialogues across eleven scenarios, utilizing natural user queries for evaluation. We propose a synthesis pipeline that integrates Project Foundation Construction, Multi-Agent Dialogue Generation, and Memory and Schedule Management to simulate the dynamic evolution of memory. Experiments reveal that current memory systems face significant challenges in managing the long-term project states and dynamic context dependencies inherent in real-world projects. Our code and datasets are available at [https://github.com/AvatarMemory/RealMemBench](https://github.com/AvatarMemory/RealMemBench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06959v1">HAS-VQ: Hessian-Adaptive Sparse Vector Quantization for High-Fidelity LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Post-training quantization is essential for deploying Large Language Models (LLMs) on resource- constrained devices. However, standard integer quantization (e.g., INT4) fundamentally degrades per- formance by imposing a uniform grid on the heavy-tailed distribution of weight parameters, particularly in smaller-scale models (e.g., <2B parameters). We introduce HAS-VQ (Hessian-Adaptive Sparse Vec- tor Quantization), a compression framework that strictly decouples high-sensitivity outliers from the bulk weight distribution using second-order sensitivity analysis. HAS-VQ employs a Hessian-Masked Decoupling strategy to isolate sensitive parameters, followed by robust Vector Quantization (VQ) of the remaining dense body. Crucially, we introduce a residual sparse feedback mechanism that corrects quan- tization errors in the most sensitive dimensions, ensuring exact reconstruction of outliers. We evaluate HAS-VQ on SmolLM2-1.7B, demonstrating two distinct regimes of superiority: (1) Pareto Dominance over Integer Baselines: At 4.23 effective bits-per-parameter (BPP), we achieve a perplexity of 14.23, significantly outperforming the standard INT4 baseline (20.03 PPL at 4.71 BPP). (2) High-Fidelity Compression: Relative to the FP16 baseline, HAS-VQ achieves a 2.3x reduction in model size (7.03 BPP) while maintaining statistically indistinguishable perplexity (10.12 vs. 10.04), effectively offering a lossless compression alternative for bandwidth-constrained environments. The code is available at https://github.com/VladimerKhasia/HASVQ
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06914v1">Towards Compositional Generalization in LLMs for Smart Contract Security: A Case Study on Reentrancy Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities in natural language understanding and generation. Despite being trained on large-scale, high-quality data, LLMs still fail to outperform traditional static analysis tools in specialized domains like smart contract vulnerability detection. To address this issue, this paper proposes a post-training algorithm based on atomic task decomposition and fusion. This algorithm aims to achieve combinatorial generalization under limited data by decomposing complex reasoning tasks. Specifically, we decompose the reentrancy vulnerability detection task into four linearly independent atomic tasks: identifying external calls, identifying state updates, identifying data dependencies between external calls and state updates, and determining their data flow order. These tasks form the core components of our approach. By training on synthetic datasets, we generate three compiler-verified datasets. We then employ the Slither tool to extract structural information from the control flow graph and data flow graph, which is used to fine-tune the LLM's adapter. Experimental results demonstrate that low-rank normalization fusion with the LoRA adapter improves the LLM's reentrancy vulnerability detection accuracy to 98.2%, surpassing state-of-the-art methods. On 31 real-world contracts, the algorithm achieves a 20% higher recall than traditional analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08870v2">Fed-SE: Federated Self-Evolution for Privacy-Constrained Multi-Environment LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-11
    </div>
    <details class="paper-abstract">
      LLM agents are widely deployed in complex interactive tasks, yet privacy constraints often preclude centralized optimization and co-evolution across dynamic environments. Despite the demonstrated success of Federated Learning (FL) on static datasets, its effectiveness in open-ended, self-evolving agent systems remains largely unexplored. In such settings, the direct application of standard FL is particularly challenging, as heterogeneous tasks and sparse, trajectory-level reward signals give rise to severe gradient instability, which undermines the global optimization process. To bridge this gap, we propose Fed-SE, a Federated Self-Evolution framework for LLM agents that establishes a local evolution-global aggregation paradigm. Locally, agents employ parameter-efficient fine-tuning on filtered, high-return trajectories to achieve stable gradient updates. Globally, Fed-SE aggregates updates within a low-rank subspace, reducing communication cost across clients. Experiments across five heterogeneous environments demonstrate that Fed-SE improves average task success rates by 10\% over the state-of-the-art FedIT, validating its effectiveness in cross-environment knowledge transfer under privacy constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06596v1">Are LLMs Vulnerable to Preference-Undermining Attacks (PUA)? A Factorial Analysis Methodology for Diagnosing the Trade-off between Preference Alignment and Real-World Validity</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) training often optimizes for preference alignment, rewarding outputs that are perceived as helpful and interaction-friendly. However, this preference-oriented objective can be exploited: manipulative prompts can steer responses toward user-appeasing agreement and away from truth-oriented correction. In this work, we investigate whether aligned models are vulnerable to Preference-Undermining Attacks (PUA), a class of manipulative prompting strategies designed to exploit the model's desire to please user preferences at the expense of truthfulness. We propose a diagnostic methodology that provides a finer-grained and more directive analysis than aggregate benchmark scores, using a factorial evaluation framework to decompose prompt-induced shifts into interpretable effects of system objectives (truth- vs. preference-oriented) and PUA-style dialogue factors (directive control, personal derogation, conditional approval, reality denial) within a controlled $2 \times 2^4$ design. Surprisingly, more advanced models are sometimes more susceptible to manipulative prompts. Beyond the dominant reality-denial factor, we observe model-specific sign reversals and interactions with PUA-style factors, suggesting tailored defenses rather than uniform robustness. These findings offer a novel, reproducible factorial evaluation methodology that provides finer-grained diagnostics for post-training processes like RLHF, enabling better trade-offs in the product iteration of LLMs by offering a more nuanced understanding of preference alignment risks and the impact of manipulative prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06586v1">Detecting LLM-Generated Text with Performance Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as GPT, Claude, Gemini, and Grok have been deeply integrated into our daily life. They now support a wide range of tasks -- from dialogue and email drafting to assisting with teaching and coding, serving as search engines, and much more. However, their ability to produce highly human-like text raises serious concerns, including the spread of fake news, the generation of misleading governmental reports, and academic misconduct. To address this practical problem, we train a classifier to determine whether a piece of text is authored by an LLM or a human. Our detector is deployed on an online CPU-based platform https://huggingface.co/spaces/stats-powered-ai/StatDetectLLM, and contains three novelties over existing detectors: (i) it does not rely on auxiliary information, such as watermarks or knowledge of the specific LLM used to generate the text; (ii) it more effectively distinguishes between human- and LLM-authored text; and (iii) it enables statistical inference, which is largely absent in the current literature. Empirically, our classifier achieves higher classification accuracy compared to existing detectors, while maintaining type-I error control, high statistical power, and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06580v1">Stylistic Evolution and LLM Neutrality in Singlish Language</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Singlish is a creole rooted in Singapore's multilingual environment and continues to evolve alongside social and technological change. This study investigates the evolution of Singlish over a decade of informal digital text messages. We propose a stylistic similarity framework that compares lexico-structural, pragmatic, psycholinguistic, and encoder-derived features across years to quantify temporal variation. Our analysis reveals notable diachronic changes in tone, expressivity and sentence construction over the years. Conversely, while some LLMs were able to generate superficially realistic Singlish messages, they do not produce temporally neutral outputs, and residual temporal signals remain detectable despite prompting and fine-tuning. Our findings highlight the dynamic evolution of Singlish, as well as the capabilities and limitations of current LLMs in modeling sociolectal and temporal variations in the colloquial language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12313v2">Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 21 pages, 1 figure, 5 tables, 2 algorithms
    </div>
    <details class="paper-abstract">
      Software supply chain attacks targeting the npm ecosystem have become increasingly sophisticated, leveraging obfuscation and complex logic to evade traditional detection mechanisms. Recently, large language models (LLMs) have attracted significant attention for malicious code detection due to their strong capabilities in semantic code understanding. However, the practical deployment of LLMs in this domain is severely constrained by limited context windows and high computational costs. Naive approaches, such as token-based code splitting, often fragment semantic context, leading to degraded detection performance. To overcome these challenges, this paper introduces a novel LLM-based framework for malicious npm package detection that leverages code slicing techniques. A specialized taint-based slicing method tailored to the JavaScript ecosystem is proposed to recover malicious data flows. By isolating security-relevant logic from benign boilerplate code, the approach reduces the input code volume by over 99\% while preserving critical malicious behaviors. The framework is evaluated on a curated dataset comprising over \num{7000} malicious and benign npm packages. Experimental results using the DeepSeek-Coder-6.7B model demonstrate that the proposed approach achieves a detection accuracy of \num{87.04}\%, significantly outperforming a full-package baseline based on naive token splitting (\num{75.41}\%). These results indicate that semantically optimized input representations via code slicing not only mitigate the LLM context window bottleneck but also enhance reasoning precision for security analysis, providing an effective defense against evolving open-source software supply chain threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02620v3">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 11 pages, and the last one is the appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accepted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.37$\times$-1.73$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06543v1">SimLLM: Fine-Tuning Code LLMs for SimPy-Based Queueing System Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The Python package SimPy is widely used for modeling queueing systems due to its flexibility, simplicity, and smooth integration with modern data analysis and optimization frameworks. Recent advances in large language models (LLMs) have shown strong ability in generating clear and executable code, making them powerful and suitable tools for writing SimPy queueing simulation code. However, directly employing closed-source models like GPT-4o to generate such code may lead to high computational costs and raise data privacy concerns. To address this, we fine-tune two open-source LLMs, Qwen-Coder-7B and DeepSeek-Coder-6.7B, on curated SimPy queueing data, which enhances their code-generating performance in executability, output-format compliance, and instruction-code consistency. Particularly, we proposed a multi-stage fine-tuning framework comprising two stages of supervised fine-tuning (SFT) and one stage of direct preference optimization (DPO), progressively enhancing the model's ability in SimPy-based queueing simulation code generation. Extensive evaluations demonstrate that both fine-tuned models achieve substantial improvements in executability, output-format compliance, and instruct consistency. These results confirm that domain-specific fine-tuning can effectively transform compact open-source code models into reliable SimPy simulation generators which provide a practical alternative to closed-source LLMs for education, research, and operational decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06502v1">DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 This paper has been accepted for presentation and publication at the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026), source code will be available soon
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown promise in addressing combinatorial optimization problems (COPs) through prompt-based strategies. However, their scalability and generalization remain limited, and their effectiveness diminishes as problem size increases, particularly in routing problems involving more than 30 nodes. We propose DRAGON, which stands for Decomposition and Reconstruction Agents Guided OptimizatioN, a novel framework that combines the strengths of metaheuristic design and LLM reasoning. Starting from an initial global solution, DRAGON autonomously identifies regions with high optimization potential and strategically decompose large-scale COPs into manageable subproblems. Each subproblem is then reformulated as a concise, localized optimization task and solved through targeted LLM prompting guided by accumulated experiences. Finally, the locally optimized solutions are systematically reintegrated into the original global context to yield a significantly improved overall outcome. By continuously interacting with the optimization environment and leveraging an adaptive experience memory, the agents iteratively learn from feedback, effectively coupling symbolic reasoning with heuristic search. Empirical results show that, unlike existing LLM-based solvers limited to small-scale instances, DRAGON consistently produces feasible solutions on TSPLIB, CVRPLIB, and Weibull-5k bin packing benchmarks, and achieves near-optimal results (0.16% gap) on knapsack problems with over 3M variables. This work shows the potential of feedback-driven language agents as a new paradigm for generalizable and interpretable large-scale optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06497v1">Coding in a Bubble? Evaluating LLMs in Resolving Context Adaptation Bugs During Code Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 24 pages, 11 figures, accepted by FSE 2026
    </div>
    <details class="paper-abstract">
      Code adaptation is a fundamental but challenging task in software development, requiring developers to modify existing code for new contexts. A key challenge is to resolve Context Adaptation Bugs (CtxBugs), which occurs when code correct in its original context violates constraints in the target environment. Unlike isolated bugs, CtxBugs cannot be resolved through local fixes and require cross-context reasoning to identify semantic mismatches. Overlooking them may lead to critical failures in adaptation. Although Large Language Models (LLMs) show great potential in automating code-related tasks, their ability to resolve CtxBugs remains a significant and unexplored obstacle to their practical use in code adaptation. To bridge this gap, we propose CtxBugGen, a novel framework for generating CtxBugs to evaluate LLMs. Its core idea is to leverage LLMs' tendency to generate plausible but context-free code when contextual constraints are absent. The framework generates CtxBugs through a four-step process to ensure their relevance and validity: (1) Adaptation Task Selection, (2) Task-specific Perturbation,(3) LLM-based Variant Generation and (4) CtxBugs Identification. Based on the benchmark constructed by CtxBugGen, we conduct an empirical study with four state-of-the-art LLMs. Our results reveal their unsatisfactory performance in CtxBug resolution. The best performing LLM, Kimi-K2, achieves 55.93% on Pass@1 and resolves just 52.47% of CtxBugs. The presence of CtxBugs degrades LLMs' adaptation performance by up to 30%. Failure analysis indicates that LLMs often overlook CtxBugs and replicate them in their outputs. Our study highlights a critical weakness in LLMs' cross-context reasoning and emphasize the need for new methods to enhance their context awareness for reliable code adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06490v1">Bi-Mem: Bidirectional Construction of Hierarchical Memory for Personalized LLMs via Inductive-Reflective Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Constructing memory from users' long-term conversations overcomes LLMs' contextual limitations and enables personalized interactions. Recent studies focus on hierarchical memory to model users' multi-granular behavioral patterns via clustering and aggregating historical conversations. However, conversational noise and memory hallucinations can be amplified during clustering, causing locally aggregated memories to misalign with the user's global persona. To mitigate this issue, we propose Bi-Mem, an agentic framework ensuring hierarchical memory fidelity through bidirectional construction. Specifically, we deploy an inductive agent to form the hierarchical memory: it extracts factual information from raw conversations to form fact-level memory, aggregates them into thematic scenes (i.e., local scene-level memory) using graph clustering, and infers users' profiles as global persona-level memory. Simultaneously, a reflective agent is designed to calibrate local scene-level memories using global constraints derived from the persona-level memory, thereby enforcing global-local alignment. For coherent memory recall, we propose an associative retrieval mechanism: beyond initial hierarchical search, a spreading activation process allows facts to evoke contextual scenes, while scene-level matches retrieve salient supporting factual information. Empirical evaluations demonstrate that Bi-Mem achieves significant improvements in question answering performance on long-term personalized conversational tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14268v2">Think-J: Learning to Think for Generative LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge refers to the automatic modeling of preferences for responses generated by Large Language Models (LLMs), which is of significant importance for both LLM evaluation and reward modeling. Although generative LLMs have made substantial progress in various tasks, their performance as LLM-Judge still falls short of expectations. In this work, we propose Think-J, which improves generative LLM-as-a-Judge by learning how to think. We first utilized a small amount of curated data to develop the model with initial judgment thinking capabilities. Subsequently, we optimize the judgment thinking traces based on reinforcement learning (RL). We propose two methods for judgment thinking optimization, based on offline and online RL, respectively. The offline method requires training a critic model to construct positive and negative examples for learning. The online method defines rule-based reward as feedback for optimization. Experimental results showed that our approach can significantly enhance the evaluation capability of generative LLM-Judge, surpassing both generative and classifier-based LLM-Judge without requiring extra human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.12195v3">Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      With the bloom of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) that incorporate LLMs with pre-trained vision models have recently demonstrated impressive performance across diverse vision-language tasks. However, they fall short to comprehend context involving multiple images. A primary reason for this shortcoming is that the visual features for each images are encoded individually by frozen encoders before feeding into the LLM backbone, lacking awareness of other images and the multimodal instructions. We term this issue as prior-LLM modality isolation and propose a two phase paradigm, browse-and-concentrate, to enable in-depth multimodal context fusion prior to feeding the features into LLMs. This paradigm initially "browses" through the inputs for essential insights, and then revisits the inputs to "concentrate" on crucial details, guided by these insights, to achieve a more comprehensive understanding of the multimodal inputs. Additionally, we develop training strategies specifically to enhance the understanding of multi-image inputs. Our method markedly boosts the performance on 7 multi-image scenarios, contributing to increments on average accuracy by 2.13% and 7.60% against strong MLLMs baselines with 3B and 11B LLMs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06426v1">NC-Bench: An LLM Benchmark for Evaluating Conversational Competence</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 9 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      The Natural Conversation Benchmark (NC-Bench) introduce a new approach to evaluating the general conversational competence of large language models (LLMs). Unlike prior benchmarks that focus on the content of model behavior, NC-Bench focuses on the form and structure of natural conversation. Grounded in the IBM Natural Conversation Framework (NCF), NC-Bench comprises three distinct sets. The Basic Conversation Competence set evaluates fundamental sequence management practices, such as answering inquiries, repairing responses, and closing conversational pairs. The RAG set applies the same sequence management patterns as the first set but incorporates retrieval-augmented generation (RAG). The Complex Request set extends the evaluation to complex requests involving more intricate sequence management patterns. Each benchmark tests a model's ability to produce contextually appropriate conversational actions in response to characteristic interaction patterns. Initial evaluations across 6 open-source models and 14 interaction patterns show that models perform well on basic answering tasks, struggle more with repair tasks (especially repeat), have mixed performance on closing sequences, and find complex multi-turn requests most challenging, with Qwen models excelling on the Basic set and Granite models on the RAG set and the Complex Request set. By operationalizing fundamental principles of human conversation, NC-Bench provides a lightweight, extensible, and theory-grounded framework for assessing and improving the conversational abilities of LLMs beyond topical or task-specific benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01191v4">Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 Accepted by the Foundations of Reasoning in Language Models (FoRLM) at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has been shown to be effective in eliciting structured reasoning (i.e., CoT reasoning) from large language models (LLMs). Regardless of its popularity, recent studies expose its failures in some reasoning tasks, raising fundamental questions about the nature of CoT reasoning. In this work, we propose a data distribution lens to understand when and why CoT reasoning succeeds or fails. We hypothesize that CoT reasoning reflects a structured inductive bias learned from in-distribution data, enabling models to conditionally generate reasoning trajectories that approximate those observed during training. As such, the effectiveness of CoT reasoning is fundamentally governed by the nature and degree of distribution discrepancy between training data and test queries. Guided by this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To test the hypothesis, we introduce DataAlchemy, an abstract and fully controllable environment that trains LLMs from scratch and systematically probes them under various distribution conditions. Through rigorous controlled experiments, we reveal that CoT reasoning is a brittle mirage when it is pushed beyond training distributions, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23853v2">Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly used to interact with and execute tasks in dynamic environments. However, a critical yet overlooked limitation of these agents is that they, by default, assume a stationary context, failing to account for the real-world time elapsed between messages. We refer to this as "temporal blindness". This limitation hinders decisions about when to invoke tools, leading agents to either over-rely on stale context and skip needed tool calls, or under-rely on it and redundantly repeat tool calls. To study this challenge, we constructed TicToc, a diverse dataset of multi-turn user-agent message trajectories across 76 scenarios, spanning dynamic environments with high, medium, and low time sensitivity. We collected human preferences between "calling a tool" and "directly answering" on each sample, and evaluated how well LLM tool-calling decisions align with human preferences under varying amounts of elapsed time. Our analysis reveals that existing models display poor alignment with human temporal perception, with no model achieving a normalized alignment rate better than 65% when given time stamp information. We also show that naive, prompt-based alignment techniques have limited effectiveness for most models, but specific post-training alignment can be a viable way to align multi-turn LLM tool use with human temporal perception. Our data and findings provide a first step toward understanding and mitigating temporal blindness, offering insights to foster the development of more time-aware and human-aligned agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06419v1">Lightweight Yet Secure: Secure Scripting Language Generation via Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 19 pages,8 figures,conference
    </div>
    <details class="paper-abstract">
      The security of scripting languages such as PowerShell is critical given their powerful automation and administration capabilities, often exercised with elevated privileges. Today, securing these languages still demands substantial human effort to craft and enforce rules, imposing heavy burdens on typical administrators and creating critical production risks (e.g., misoperations that shut down servers).Large language models (LLMs) have demonstrated strong capabilities in code generation, vulnerability detection, and automated repair for languages like Python and JavaScript. However, their ability to assist with generating secure scripting-language code remains largely underexplored. In this paper, we present SecGenEval-PS, a benchmark designed to systematically evaluate LLMs on secure scripting generation, security analysis, and automated repair. Our results show that both proprietary and open-source models fall short in these areas. For instance, over 60% of PowerShell scripts produced by GPT-4o and o3-mini are insecure without structured guidance.To bridge this gap, we propose PSSec, a framework that combines data synthesis with fine-tuning to enhance model security capabilities. We develop a self-debugging agent that integrates static analyzers with the reasoning abilities of advanced LLMs to synthesize large-scale structured triplets of insecure scripts, violation analyses, and corresponding repairs. We then fine-tune lightweight LLMs (as small as 1.7B parameters) using supervised fine-tuning (SFT) and reinforcement learning (RL), enabling security-aware reasoning and the generation of secure PowerShell code.Across multiple LLM families, including GPT and Qwen, \textit{PSSec}-trained models match or surpass general-purpose large models on PowerShell security tasks while reducing inference cost by more than an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06377v1">HiMem: Hierarchical Long-Term Memory for LLM Long-Horizon Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Although long-term memory systems have made substantial progress in recent years, they still exhibit clear limitations in adaptability, scalability, and self-evolution under continuous interaction settings. Inspired by cognitive theories, we propose HiMem, a hierarchical long-term memory framework for long-horizon dialogues, designed to support memory construction, retrieval, and dynamic updating during sustained interactions. HiMem constructs cognitively consistent Episode Memory via a Topic-Aware Event--Surprise Dual-Channel Segmentation strategy, and builds Note Memory that captures stable knowledge through a multi-stage information extraction pipeline. These two memory types are semantically linked to form a hierarchical structure that bridges concrete interaction events and abstract knowledge, enabling efficient retrieval without sacrificing information fidelity. HiMem supports both hybrid and best-effort retrieval strategies to balance accuracy and efficiency, and incorporates conflict-aware Memory Reconsolidation to revise and supplement stored knowledge based on retrieval feedback. This design enables continual memory self-evolution over long-term use. Experimental results on long-horizon dialogue benchmarks demonstrate that HiMem consistently outperforms representative baselines in accuracy, consistency, and long-term reasoning, while maintaining favorable efficiency. Overall, HiMem provides a principled and scalable design paradigm for building adaptive and self-evolving LLM-based conversational agents. The code is available at https://github.com/jojopdq/HiMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24189v3">SPECTRA: Revealing the Full Spectrum of User Preferences via Distributional LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to understand user preferences, typically via the direct generation of ranked item lists. However, this end-to-end generative paradigm inherits the bias and opacity of autoregressive decoding, over-emphasizing frequent (head) preferences and obscure long-tail ones, thereby biasing personalization toward head preferences. To address this, we propose SPECTRA (Semantic Preference Extraction and Clustered TRAcking), which treats the LLM as an implicit probabilistic model by probing it to infer a probability distribution over interpretable preference clusters. In doing so, SPECTRA reframes user modeling from sequence generation with decoding heuristics to distributional inference, yielding explicit, cluster-level user preference representations. We evaluate SPECTRA on MovieLens, Yelp, and a large-scale short-video platform, demonstrating significant gains across three dimensions: SPECTRA achieves (i) distributional alignment, reducing Jensen-Shannon divergence to empirical distributions by 25% against strong baselines; (ii) long-tail exposure, reducing decoding-induced head concentration and increasing global exposure entropy by 30%; and (iii) downstream applications such as personalized ranking, translating these gains into a 40% NDCG boost on public datasets and a 7x improvement on ranking long-tail preferences against an industry-leading Transformer-based production baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06366v1">SafeGPT: Preventing Data Leakage and Unethical Outputs in Enterprise LLM Use</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming enterprise workflows but introduce security and ethics challenges when employees inadvertently share confidential data or generate policy-violating content. This paper proposes SafeGPT, a two-sided guardrail system preventing sensitive data leakage and unethical outputs. SafeGPT integrates input-side detection/redaction, output-side moderation/reframing, and human-in-the-loop feedback. Experiments demonstrate SafeGPT effectively reduces data leakage risk and biased outputs while maintaining satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00140v2">LLM-based Zero-shot Triple Extraction for Automated Ontology Generation from Software Engineering Standards</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 Semantic Data Integration Workshop, held in conjunction with IEEE International Conference on Semantic Computing (IEEE ICSC 2026), accepted, 2026
    </div>
    <details class="paper-abstract">
      Ontologies have supported knowledge representation and white-box reasoning for decades; thus, the automated ontology generation (AOG) plays a crucial role in scaling their use. Software engineering standards (SES) consist of long, unstructured text (with high noise) and paragraphs with domain-specific terms. In this setting, relation triple extraction (RTE), together with term extraction, constitutes the first stage toward AOG. This work proposes an open-source large language model (LLM)-assisted approach to RTE for SES. Instead of solely relying on prompt-engineering-based methods, this study promotes the use of LLMs as an aid in constructing ontologies and explores an effective AOG workflow that includes document segmentation, candidate term mining, LLM-based relation inference, term normalization, and cross-section alignment. Expert-annotated reference sets at three granularities are constructed and used to evaluate the ontology generated from the study. The results show that it is comparable and potentially superior to the OpenIE method of triple extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11618v3">Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06362v1">Styles + Persona-plug = Customized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      We discover a previously overlooked challenge in personalized text generation: personalization methods are increasingly applied under explicit style instructions, yet their behavior under such constraints remains poorly understood. To balance implicit personalization and explicit style, we formulate personalization as a distributional residual and propose PsPLUG, a lightweight soft-prompt plug-in trained with style-conditioned preference contrasts. Across LaMP benchmark, our framework improves persona alignment, maintains stylistic fidelity, and outperforms retrieval-based and soft-prompt baselines with minimal computation. These results show that residual modeling provides a simple and principled foundation for controllable, style-aware LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06666v1">InFi-Check: Interpretable and Fine-Grained Fact-Checking of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often hallucinate, yet most existing fact-checking methods treat factuality evaluation as a binary classification problem, offering limited interpretability and failing to capture fine-grained error types. In this paper, we introduce InFi-Check, a framework for interpretable and fine-grained fact-checking of LLM outputs. Specifically, we first propose a controlled data synthesis pipeline that generates high-quality data featuring explicit evidence, fine-grained error type labels, justifications, and corrections. Based on this, we further construct large-scale training data and a manually verified benchmark InFi-Check-FG for fine-grained fact-checking of LLM outputs. Building on these high-quality training data, we further propose InFi-Checker, which can jointly provide supporting evidence, classify fine-grained error types, and produce justifications along with corrections. Experiments show that InFi-Checker achieves state-of-the-art performance on InFi-Check-FG and strong generalization across various downstream tasks, significantly improving the utility and trustworthiness of factuality evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06652v1">Follow the Signs: Using Textual Cues and LLMs to Guide Efficient Robot Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Autonomous navigation in unfamiliar environments often relies on geometric mapping and planning strategies that overlook rich semantic cues such as signs, room numbers, and textual labels. We propose a novel semantic navigation framework that leverages large language models (LLMs) to infer patterns from partial observations and predict regions where the goal is most likely located. Our method combines local perceptual inputs with frontier-based exploration and periodic LLM queries, which extract symbolic patterns (e.g., room numbering schemes and building layout structures) and update a confidence grid used to guide exploration. This enables robots to move efficiently toward goal locations labeled with textual identifiers (e.g., "room 8") even before direct observation. We demonstrate that this approach enables more efficient navigation in sparse, partially observable grid environments by exploiting symbolic patterns. Experiments across environments modeled after real floor plans show that our approach consistently achieves near-optimal paths and outperforms baselines by over 25% in Success weighted by Path Length.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06636v1">MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through Counterfactual Differential Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Despite achieving high accuracy on medical benchmarks, LLMs exhibit the Einstellung Effect in clinical diagnosis--relying on statistical shortcuts rather than patient-specific evidence, causing misdiagnosis in atypical cases. Existing benchmarks fail to detect this critical failure mode. We introduce MedEinst, a counterfactual benchmark with 5,383 paired clinical cases across 49 diseases. Each pair contains a control case and a "trap" case with altered discriminative evidence that flips the diagnosis. We measure susceptibility via Bias Trap Rate--probability of misdiagnosing traps despite correctly diagnosing controls. Extensive Evaluation of 17 LLMs shows frontier models achieve high baseline accuracy but severe bias trap rates. Thus, we propose ECR-Agent, aligning LLM reasoning with Evidence-Based Medicine standard via two components: (1) Dynamic Causal Inference (DCI) performs structured reasoning through dual-pathway perception, dynamic causal graph reasoning across three levels (association, intervention, counterfactual), and evidence audit for final diagnosis; (2) Critic-Driven Graph and Memory Evolution (CGME) iteratively refines the system by storing validated reasoning paths in an exemplar base and consolidating disease-specific knowledge into evolving illness graphs. Source code is to be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06627v1">Burn-After-Use for Preventing Data Leakage through a Secure Multi-Tenant Architecture in Enterprise LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-10
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This study presents a Secure Multi-Tenant Architecture (SMTA) combined with a novel concept Burn-After-Use (BAU) mechanism for enterprise LLM environments to effectively prevent data leakage. As institutions increasingly adopt LLMs across departments, the risks of data leakage have become a critical security and compliance concern. The proposed SMTA isolates LLM instances across departments and enforces rigorous context ownership boundaries within an internally deployed infrastructure. The BAU mechanism introduces data confidentiality by enforcing ephemeral conversational contexts that are automatically destroyed after use, preventing cross-session or cross-user inference. The evaluation to SMTA and BAU is through two sets of realistic and reproducible experiments comprising of 127 test iterations. One aspect of this experiment is to assess prompt-based and semantic leakage attacks in a multi-tenant architecture (Appendix A) across 55 infrastructure-level attack tests, including vector-database credential compromise and shared logging pipeline exposure. SMTA achieves 92% defense success rate, demonstrating strong semantic isolation while highlighting residual risks from credential misconfiguration and observability pipelines. Another aspect is to evaluate the robustness of BAU under realistic failure scenarios (Appendix B) using four empirical metrics: Local Residual Persistence Rate (LRPR), Remote Residual Persistence Rate (RRPR), Image Frame Exposure Rate (IFER), and Burn Timer Persistence Rate (BTPR). Across 72 test iterations, BAU achieves a 76.75% success rate in mitigating post-session leakage threats across the client, server, application, infrastructure, and cache layers. These results show that SMTA and BAU together enforce strict isolation, complete session ephemerality, strong confidentiality guarantees, non-persistence, and policy-aligned behavior for enterprise LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06616v1">LLM-Driven Accessible Interface: A Model-Based Approach</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into interactive systems opens new opportunities for adaptive user experiences, yet it also raises challenges regarding accessibility, explainability, and normative compliance. This paper presents an implemented model-driven architecture for generating personalised, multimodal, and accessibility-aligned user interfaces. The approach combines structured user profiles, declarative adaptation rules, and validated prompt templates to refine baseline accessible UI templates that conform to WCAG 2.2 and EN 301 549, tailored to cognitive and sensory support needs. LLMs dynamically transform language complexity, modality, and visual structure, producing outputs such as Plain-Language text, pictograms, and high-contrast layouts aligned with ISO 24495-1 and W3C COGA guidance. A healthcare use case demonstrates how the system generates accessible post-consultation medication instructions tailored to a user profile comprising cognitive disability and hearing impairment. SysML v2 models provide explicit traceability between user needs, adaptation rules, and normative requirements, ensuring explainable and auditable transformations. Grounded in Human-Centered AI (HCAI), the framework incorporates co-design processes and structured feedback mechanisms to guide iterative refinement and support trustworthy generative behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06599v1">How Context Shapes Truth: Geometric Transformations of Statement-level Truth Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often encode whether a statement is true as a vector in their residual stream activations. These vectors, also known as truth vectors, have been studied in prior work, however how they change when context is introduced remains unexplored. We study this question by measuring (1) the directional change ($θ$) between the truth vectors with and without context and (2) the relative magnitude of the truth vectors upon adding context. Across four LLMs and four datasets, we find that (1) truth vectors are roughly orthogonal in early layers, converge in middle layers, and may stabilize or continue increasing in later layers; (2) adding context generally increases the truth vector magnitude, i.e., the separation between true and false representations in the activation space is amplified; (3) larger models distinguish relevant from irrelevant context mainly through directional change ($θ$), while smaller models show this distinction through magnitude differences. We also find that context conflicting with parametric knowledge produces larger geometric changes than parametrically aligned context. To the best of our knowledge, this is the first work that provides a geometric characterization of how context transforms the truth vector in the activation space of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05835v1">Left, Right, or Center? Evaluating LLM Framing in News Classification and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based summarization and text generation are increasingly used for producing and rewriting text, raising concerns about political framing in journalism where subtle wording choices can shape interpretation. Across nine state-of-the-art LLMs, we study political framing by testing whether LLMs' classification-based bias signals align with framing behavior in their generated summaries. We first compare few-shot ideology predictions against LEFT/CENTER/RIGHT labels. We then generate "steered" summaries under FAITHFUL, CENTRIST, LEFT, and RIGHT prompts, and score all outputs using a single fixed ideology evaluator. We find pervasive ideological center-collapse in both article-level ratings and generated text, indicating a systematic tendency toward centrist framing. Among evaluated models, Grok 4 is by far the most ideologically expressive generator, while Claude Sonnet 4.5 and Llama 3.1 achieve the strongest bias-rating performance among commercial and open-weight models, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19153v2">LLMs as verification oracles for Solidity</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs aid in assessing the validity of arbitrary contract-specific properties? In this paper, we provide the first systematic empirical evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs - although lacking soundness guarantees - can be surprisingly effective at predicting the (in)validity of complex properties, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05821v1">LLMs as Science Journalists: Supporting Early-stage Researchers in Communicating Their Science to the Public</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      The scientific community needs tools that help early-stage researchers effectively communicate their findings and innovations to the public. Although existing general-purpose Large Language Models (LLMs) can assist in this endeavor, they are not optimally aligned for it. To address this, we propose a framework for training LLMs to emulate the role of a science journalist that can be used by early-stage researchers to learn how to properly communicate their papers to the general public. We evaluate the usefulness of our trained LLM Journalists in leading conversations with both simulated and human researchers. %compared to the general-purpose ones. Our experiments indicate that LLMs trained using our framework ask more relevant questions that address the societal impact of research, prompting researchers to clarify and elaborate on their findings. In the user study, the majority of participants who interacted with our trained LLM Journalist appreciated it more than interacting with general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05808v1">EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at https://github.com/RUC-NLPIR/EnvScaler.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05806v1">Modular Autonomy with Conversational Interaction: An LLM-driven Framework for Decision Making in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Submitted to the IEEE Intelligent Vehicles Symposium (IV 2026), Detroit, MI, United States
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) offer new opportunities to create natural language interfaces for Autonomous Driving Systems (ADSs), moving beyond rigid inputs. This paper addresses the challenge of mapping the complexity of human language to the structured action space of modular ADS software. We propose a framework that integrates an LLM-based interaction layer with Autoware, a widely used open-source software. This system enables passengers to issue high-level commands, from querying status information to modifying driving behavior. Our methodology is grounded in three key components: a taxonomization of interaction categories, an application-centric Domain Specific Language (DSL) for command translation, and a safety-preserving validation layer. A two-stage LLM architecture ensures high transparency by providing feedback based on the definitive execution status. Evaluation confirms the system's timing efficiency and translation robustness. Simulation successfully validated command execution across all five interaction categories. This work provides a foundation for extensible, DSL-assisted interaction in modular and safety-conscious autonomy stacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05794v1">Simplify-This: A Comparative Analysis of Prompt-Based and Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable strong text generation, and in general there is a practical tradeoff between fine-tuning and prompt engineering. We introduce Simplify-This, a comparative study evaluating both paradigms for text simplification with encoder-decoder LLMs across multiple benchmarks, using a range of evaluation metrics. Fine-tuned models consistently deliver stronger structural simplification, whereas prompting often attains higher semantic similarity scores yet tends to copy inputs. A human evaluation favors fine-tuned outputs overall. We release code, a cleaned derivative dataset used in our study, checkpoints of fine-tuned models, and prompt templates to facilitate reproducibility and future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17749v3">Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 In Engineering Applications of Artificial Intelligence, Vol. 162, December 2025
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs. We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_paraphrased_code_via_coding_style_features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05755v1">VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility. Code is available at https://anonymous.4open.science/r/VIGIL-378B/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05752v1">AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05751v1">Analysing Differences in Persuasive Language in LLM-Generated Text: Uncovering Stereotypical Gender Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for everyday communication tasks, including drafting interpersonal messages intended to influence and persuade. Prior work has shown that LLMs can successfully persuade humans and amplify persuasive language. It is therefore essential to understand how user instructions affect the generation of persuasive language, and to understand whether the generated persuasive language differs, for example, when targeting different groups. In this work, we propose a framework for evaluating how persuasive language generation is affected by recipient gender, sender intent, or output language. We evaluate 13 LLMs and 16 languages using pairwise prompt instructions. We evaluate model responses on 19 categories of persuasive language using an LLM-as-judge setup grounded in social psychology and communication science. Our results reveal significant gender differences in the persuasive language generated across all models. These patterns reflect biases consistent with gender-stereotypical linguistic tendencies documented in social psychology and sociolinguistics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05742v1">The Echo Chamber Multi-Turn LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      The availability of Large Language Models (LLMs) has led to a new generation of powerful chatbots that can be developed at relatively low cost. As companies deploy these tools, security challenges need to be addressed to prevent financial loss and reputational damage. A key security challenge is jailbreaking, the malicious manipulation of prompts and inputs to bypass a chatbot's safety guardrails. Multi-turn attacks are a relatively new form of jailbreaking involving a carefully crafted chain of interactions with a chatbot. We introduce Echo Chamber, a new multi-turn attack using a gradual escalation method. We describe this attack in detail, compare it to other multi-turn attacks, and demonstrate its performance against multiple state-of-the-art models through extensive evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05705v1">Logic-Parametric Neuro-Symbolic NLI: Controlling Logical Formalisms for Verifiable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and theorem provers (TPs) can be effectively combined for verifiable natural language inference (NLI). However, existing approaches rely on a fixed logical formalism, a feature that limits robustness and adaptability. We propose a logic-parametric framework for neuro-symbolic NLI that treats the underlying logic not as a static background, but as a controllable component. Using the LogiKEy methodology, we embed a range of classical and non-classical formalisms into higher-order logic (HOL), enabling a systematic comparison of inference quality, explanation refinement, and proof behavior. We focus on normative reasoning, where the choice of logic has significant implications. In particular, we compare logic-external approaches, where normative requirements are encoded via axioms, with logic-internal approaches, where normative patterns emerge from the logic's built-in structure. Extensive experiments demonstrate that logic-internal strategies can consistently improve performance and produce more efficient hybrid proofs for NLI. In addition, we show that the effectiveness of a logic is domain-dependent, with first-order logic favouring commonsense reasoning, while deontic and modal logics excel in ethical domains. Our results highlight the value of making logic a first-class, parametric element in neuro-symbolic architectures for more robust, modular, and adaptable reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05684v1">FLRQ: Faster LLM Quantization with Flexible Low-Rank Matrix Sketching</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Traditional post-training quantization (PTQ) is considered an effective approach to reduce model size and accelerate inference of large-scale language models (LLMs). However, existing low-rank PTQ methods require costly fine-tuning to determine a compromise rank for diverse data and layers in large models, failing to exploit their full potential. Additionally, the current SVD-based low-rank approximation compounds the computational overhead. In this work, we thoroughly analyze the varying effectiveness of low-rank approximation across different layers in representative models. Accordingly, we introduce \underline{F}lexible \underline{L}ow-\underline{R}ank \underline{Q}uantization (FLRQ), a novel solution designed to quickly identify the accuracy-optimal ranks and aggregate them to achieve minimal storage combinations. FLRQ comprises two powerful components, Rank1-Sketch-based Flexible Rank Selection (R1-FLR) and Best Low-rank Approximation under Clipping (BLC). R1-FLR applies the R1-Sketch with Gaussian projection for the fast low-rank approximation, enabling outlier-aware rank extraction for each layer. Meanwhile, BLC aims at minimizing the low-rank quantization error under the scaling and clipping strategy through an iterative method. FLRQ demonstrates strong effectiveness and robustness in comprehensive experiments, achieving state-of-the-art performance in both quantization quality and algorithm efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v2">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05635v1">Continual Pretraining on Encrypted Synthetic Data for Privacy-Preserving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Preserving privacy in sensitive data while pretraining large language models on small, domain-specific corpora presents a significant challenge. In this work, we take an exploratory step toward privacy-preserving continual pretraining by proposing an entity-based framework that synthesizes encrypted training data to protect personally identifiable information (PII). Our approach constructs a weighted entity graph to guide data synthesis and applies deterministic encryption to PII entities, enabling LLMs to encode new knowledge through continual pretraining while granting authorized access to sensitive data through decryption keys. Our results on limited-scale datasets demonstrate that our pretrained models outperform base models and ensure PII security, while exhibiting a modest performance gap compared to models trained on unencrypted synthetic data. We further show that increasing the number of entities and leveraging graph-based synthesis improves model performance, and that encrypted models retain instruction-following capabilities with long retrieved contexts. We discuss the security implications and limitations of deterministic encryption, positioning this work as an initial investigation into the design space of encrypted data pretraining for privacy-preserving LLMs. Our code is available at https://github.com/DataArcTech/SoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05633v1">GIFT: Games as Informal Training for Generalizable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable success in formal learning tasks such as mathematics and code generation, they still struggle with the "practical wisdom" and generalizable intelligence, such as strategic creativity and social reasoning, that characterize human cognition. This gap arises from a lack of informal learning, which thrives on interactive feedback rather than goal-oriented instruction. In this paper, we propose treating Games as a primary environment for LLM informal learning, leveraging their intrinsic reward signals and abstracted complexity to cultivate diverse competencies. To address the performance degradation observed in multi-task learning, we introduce a Nested Training Framework. Unlike naive task mixing optimizing an implicit "OR" objective, our framework employs sequential task composition to enforce an explicit "AND" objective, compelling the model to master multiple abilities simultaneously to achieve maximal rewards. Using GRPO-based reinforcement learning across Matrix Games, TicTacToe, and Who's the Spy games, we demonstrate that integrating game-based informal learning not only prevents task interference but also significantly bolsters the model's generalization across broad ability-oriented benchmarks. The framework and implementation are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05616v1">Dual-Phase LLM Reasoning: Self-Evolved Mathematical Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have demonstrated significant potential in complex reasoning tasks like mathematical problem-solving. However, existing research predominantly relies on reinforcement learning (RL) frameworks while overlooking supervised fine-tuning (SFT) methods. This paper proposes a new two-stage training framework that enhances models' self-correction capabilities through self-generated long chain-of-thought (CoT) data. During the first stage, a multi-turn dialogue strategy guides the model to generate CoT data incorporating verification, backtracking, subgoal decomposition, and backward reasoning, with predefined rules filtering high-quality samples for supervised fine-tuning. The second stage employs a difficulty-aware rejection sampling mechanism to dynamically optimize data distribution, strengthening the model's ability to handle complex problems. The approach generates reasoning chains extended over 4 times longer while maintaining strong scalability, proving that SFT effectively activates models' intrinsic reasoning capabilities and provides a resource-efficient pathway for complex task optimization. Experimental results demonstrate performance improvements on mathematical benchmarks including GSM8K and MATH500, with the fine-tuned model achieving a substantial improvement on competition-level problems like AIME24. Code will be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05606v1">Conformity Dynamics in LLM Multi-Agent Systems: The Roles of Topology and Self-Social Weighting</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly instantiated as interacting agents in multi-agent systems (MAS), where collective decisions emerge through social interaction rather than independent reasoning. A fundamental yet underexplored mechanism in this process is conformity, the tendency of agents to align their judgments with prevailing group opinions. This paper presents a systematic study of how network topology shapes conformity dynamics in LLM-based MAS through a misinformation detection task. We introduce a confidence-normalized pooling rule that controls the trade-off between self-reliance and social influence, enabling comparisons between two canonical decision paradigms: Centralized Aggregation and Distributed Consensus. Experimental results demonstrate that network topology critically governs both the efficiency and robustness of collective judgments. Centralized structures enable immediate decisions but are sensitive to hub competence and exhibit same-model alignment biases. In contrast, distributed structures promote more robust consensus, while increased network connectivity speeds up convergence but also heightens the risk of wrong-but-sure cascades, in which agents converge on incorrect decisions with high confidence. These findings characterize the conformity dynamics in LLM-based MAS, clarifying how network topology and self-social weighting jointly shape the efficiency, robustness, and failure modes of collective decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05603v1">Revisiting Human-vs-LLM judgments using the TREC Podcast Track</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 The paper has been accepted to appear at ECIR 2026
    </div>
    <details class="paper-abstract">
      Using large language models (LLMs) to annotate relevance is an increasingly important technique in the information retrieval community. While some studies demonstrate that LLMs can achieve high user agreement with ground truth (human) judgments, other studies have argued for the opposite conclusion. To the best of our knowledge, these studies have primarily focused on classic ad-hoc text search scenarios. In this paper, we conduct an analysis on user agreement between LLM and human experts, and explore the impact disagreement has on system rankings. In contrast to prior studies, we focus on a collection composed of audio files that are transcribed into two-minute segments -- the TREC 2020 and 2021 podcast track. We employ five different LLM models to re-assess all of the query-segment pairs, which were originally annotated by TREC assessors. Furthermore, we re-assess a small subset of pairs where LLM and TREC assessors have the highest disagreement, and found that the human experts tend to agree with LLMs more than with the TREC assessors. Our results reinforce the previous insights of Sormunen in 2002 -- that relying on a single assessor leads to lower user agreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09713v2">A First Look at Bugs in LLM Inference Engines</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Accepted by ACM Transactions on Software Engineering and Methodology
    </div>
    <details class="paper-abstract">
      Large language model-specific inference engines (in short as \emph{LLM inference engines}) have become a fundamental component of modern AI infrastructure, enabling the deployment of LLM-powered applications (LLM apps) across cloud and local devices. Despite their critical role, LLM inference engines are prone to bugs due to the immense resource demands of LLMs and the complexities of cross-platform compatibility. However, a systematic understanding of these bugs remains lacking. To bridge this gap, we present the first empirical study on bugs in LLM inference engines. We mine official repositories of 5 widely adopted LLM inference engines, constructing a comprehensive dataset of 929 real-world bugs. Through a rigorous open coding process, we analyze these bugs to uncover their symptoms, root causes, commonality, fix effort, fix strategies, and temporal evolution. Our findings reveal six bug symptom types and a taxonomy of 28 root causes, shedding light on the key challenges in bug detection and location within LLM inference engines. Based on these insights, we propose a series of actionable implications for researchers, inference engine vendors, and LLM app developers, along with general guidelines for developing LLM inference engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06931v6">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      The ability to wield tools was once considered exclusive to human intelligence, but it's now known that many other animals, like crows, possess this capability. Yet, robotic systems still fall short of matching biological dexterity. In this paper, we investigate the use of Large Language Models (LLMs), tool affordances, and object manoeuvrability for non-prehensile tool-based manipulation tasks. Our novel method leverages LLMs based on scene information and natural language instructions to enable symbolic task planning for tool-object manipulation. This approach allows the system to convert a human language sentence into a sequence of feasible motion functions. We have developed a novel manoeuvrability-driven controller using a new tool affordance model derived from visual feedback. This controller helps guide the robot's tool utilization and manipulation actions, even within confined areas, using a stepping incremental approach. The proposed methodology is evaluated with experiments to prove its effectiveness under various manipulation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05564v1">The ICASSP 2026 HumDial Challenge: Benchmarking Human-like Spoken Dialogue Systems in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Official summary paper for the ICASSP 2026 HumDial Challenge
    </div>
    <details class="paper-abstract">
      Driven by the rapid advancement of Large Language Models (LLMs), particularly Audio-LLMs and Omni-models, spoken dialogue systems have evolved significantly, progressively narrowing the gap between human-machine and human-human interactions. Achieving truly ``human-like'' communication necessitates a dual capability: emotional intelligence to perceive and resonate with users' emotional states, and robust interaction mechanisms to navigate the dynamic, natural flow of conversation, such as real-time turn-taking. Therefore, we launched the first Human-like Spoken Dialogue Systems Challenge (HumDial) at ICASSP 2026 to benchmark these dual capabilities. Anchored by a sizable dataset derived from authentic human conversations, this initiative establishes a fair evaluation platform across two tracks: (1) Emotional Intelligence, targeting long-term emotion understanding and empathetic generation; and (2) Full-Duplex Interaction, systematically evaluating real-time decision-making under `` listening-while-speaking'' conditions. This paper summarizes the dataset, track configurations, and the final results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05542v1">Understanding LLM-Driven Test Oracle Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Accepted for presentation at the 2nd ACM/IEEE International Conference on AI-powered Software (AIware 2025)
    </div>
    <details class="paper-abstract">
      Automated unit test generation aims to improve software quality while reducing the time and effort required for creating tests manually. However, existing techniques primarily generate regression oracles that predicate on the implemented behavior of the class under test. They do not address the oracle problem: the challenge of distinguishing correct from incorrect program behavior. With the rise of Foundation Models (FMs), particularly Large Language Models (LLMs), there is a new opportunity to generate test oracles that reflect intended behavior. This positions LLMs as enablers of Promptware, where software creation and testing are driven by natural-language prompts. This paper presents an empirical study on the effectiveness of LLMs in generating test oracles that expose software failures. We investigate how different prompting strategies and levels of contextual input impact the quality of LLM-generated oracles. Our findings offer insights into the strengths and limitations of LLM-based oracle generation in the FM era, improving our understanding of their capabilities and fostering future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05539v1">LIDL: LLM Integration Defect Localization via Knowledge Graph-Enhanced Multi-Agent Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      LLM-integrated software, which embeds or interacts with large language models (LLMs) as functional components, exhibits probabilistic and context-dependent behaviors that fundamentally differ from those of traditional software. This shift introduces a new category of integration defects that arise not only from code errors but also from misaligned interactions among LLM-specific artifacts, including prompts, API calls, configurations, and model outputs. However, existing defect localization techniques are ineffective at identifying these LLM-specific integration defects because they fail to capture cross-layer dependencies across heterogeneous artifacts, cannot exploit incomplete or misleading error traces, and lack semantic reasoning capabilities for identifying root causes. To address these challenges, we propose LIDL, a multi-agent framework for defect localization in LLM-integrated software. LIDL (1) constructs a code knowledge graph enriched with LLM-aware annotations that represent interaction boundaries across source code, prompts, and configuration files, (2) fuses three complementary sources of error evidence inferred by LLMs to surface candidate defect locations, and (3) applies context-aware validation that uses counterfactual reasoning to distinguish true root causes from propagated symptoms. We evaluate LIDL on 146 real-world defect instances collected from 105 GitHub repositories and 16 agent-based systems. The results show that LIDL significantly outperforms five state-of-the-art baselines across all metrics, achieving a Top-3 accuracy of 0.64 and a MAP of 0.48, which represents a 64.1% improvement over the best-performing baseline. Notably, LIDL achieves these gains while reducing cost by 92.5%, demonstrating both high accuracy and cost efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05529v1">Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05513v1">LEAPS: An LLM-Empowered Adaptive Plugin for Taobao AI Search</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models has reshaped user search cognition, driving a paradigm shift from discrete keyword-based search to high-dimensional conversational interaction. However, existing e-commerce search architectures face a critical capability deficit in adapting to this change. Users are often caught in a dilemma: precise natural language descriptions frequently trigger zero-result scenarios, while the forced simplification of queries leads to decision overload from noisy, generic results. To tackle this challenge, we propose LEAPS (LLM-Empowered Adaptive Plugin for Taobao AI Search), which seamlessly upgrades traditional search systems via a "Broaden-and-Refine" paradigm. Specifically, it attaches plugins to both ends of the search pipeline: (1) Upstream, a Query Expander acts as an intent translator. It employs a novel three-stage training strategy--inverse data augmentation, posterior-knowledge supervised fine-tuning, and diversity-aware reinforcement learning--to generate adaptive and complementary query combinations that maximize the candidate product set. (2) Downstream, a Relevance Verifier serves as a semantic gatekeeper. By synthesizing multi-source data (e.g., OCR text, reviews) and leveraging chain-of-thought reasoning, it precisely filters noise to resolve selection overload. Extensive offline experiments and online A/B testing demonstrate that LEAPS significantly enhances conversational search experiences. Crucially, its non-invasive architecture preserves established retrieval performance optimized for short-text queries, while simultaneously allowing for low-cost integration into diverse back-ends. Fully deployed on Taobao AI Search since August 2025, LEAPS currently serves hundreds of millions of users monthly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01804v2">Causality-Aware Temporal Projection for Video Understanding in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent Video Large Language Models (Video-LLMs) have shown strong multimodal reasoning capabilities, yet remain challenged by video understanding tasks that require consistent temporal ordering and causal coherence. Many parameter-efficient Video-LLMs rely on unconstrained bidirectional projectors to model inter-frame interactions, which can blur temporal ordering by allowing later frames to influence earlier representations, without explicit architectural mechanisms to respect the directional nature of video reasoning. To address this limitation, we propose V-CORE, a parameter-efficient framework that introduces explicit temporal ordering constraints for video understanding. V-CORE consists of two key components: (1) Learnable Spatial Aggregation (LSA), which adaptively selects salient spatial tokens to reduce redundancy, and (2) a Causality-Aware Temporal Projector (CATP), which enforces structured unidirectional information flow via block-causal attention and a terminal dynamic summary token acting as a causal sink. This design preserves intra-frame spatial interactions while ensuring that temporal information is aggregated in a strictly ordered manner. With 4-bit QLoRA and a frozen LLM backbone, V-CORE can be trained efficiently on a single consumer GPU. Experiments show that V-CORE achieves strong performance on the challenging NExT-QA benchmark, reaching 61.2% accuracy, and remains competitive across MSVD-QA, MSRVTT-QA, and TGIF-QA, with gains concentrated in temporal and causal reasoning subcategories (+3.5% and +5.2% respectively), directly validating the importance of explicit temporal ordering constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05504v1">Memory Poisoning Attack and Defense on Memory Based LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Large language model agents equipped with persistent memory are vulnerable to memory poisoning attacks, where adversaries inject malicious instructions through query only interactions that corrupt the agents long term memory and influence future responses. Recent work demonstrated that the MINJA (Memory Injection Attack) achieves over 95 % injection success rate and 70 % attack success rate under idealized conditions. However, the robustness of these attacks in realistic deployments and effective defensive mechanisms remain understudied. This work addresses these gaps through systematic empirical evaluation of memory poisoning attacks and defenses in Electronic Health Record (EHR) agents. We investigate attack robustness by varying three critical dimensions: initial memory state, number of indication prompts, and retrieval parameters. Our experiments on GPT-4o-mini, Gemini-2.0-Flash and Llama-3.1-8B-Instruct models using MIMIC-III clinical data reveal that realistic conditions with pre-existing legitimate memories dramatically reduce attack effectiveness. We then propose and evaluate two novel defense mechanisms: (1) Input/Output Moderation using composite trust scoring across multiple orthogonal signals, and (2) Memory Sanitization with trust-aware retrieval employing temporal decay and pattern-based filtering. Our defense evaluation reveals that effective memory sanitization requires careful trust threshold calibration to prevent both overly conservative rejection (blocking all entries) and insufficient filtering (missing subtle attacks), establishing important baselines for future adaptive defense mechanisms. These findings provide crucial insights for securing memory-augmented LLM agents in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05502v1">Evaluating the Use of LLMs for Automated DOM-Level Resolution of Web Performance Issues</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Accepted to the The ACM International Conference on Mining Software Repositories (MSR) (MSR 2026)
    </div>
    <details class="paper-abstract">
      Users demand fast, seamless webpage experiences, yet developers often struggle to meet these expectations within tight constraints. Performance optimization, while critical, is a time-consuming and often manual process. One of the most complex tasks in this domain is modifying the Document Object Model (DOM), which is why this study focuses on it. Recent advances in Large Language Models (LLMs) offer a promising avenue to automate this complex task, potentially transforming how developers address web performance issues. This study evaluates the effectiveness of nine state-of-the-art LLMs for automated web performance issue resolution. For this purpose, we first extracted the DOM trees of 15 popular webpages (e.g., Facebook), and then we used Lighthouse to retrieve their performance audit reports. Subsequently, we passed the extracted DOM trees and corresponding audits to each model for resolution. Our study considers 7 unique audit categories, revealing that LLMs universally excel at SEO & Accessibility issues. However, their efficacy in performance-critical DOM manipulations is mixed. While high-performing models like GPT-4.1 delivered significant reductions in areas like Initial Load, Interactivity, and Network Optimization (e.g., 46.52% to 48.68% audit incidence reductions), others, such as GPT-4o-mini, notably underperformed, consistently. A further analysis of these modifications showed a predominant additive strategy and frequent positional changes, alongside regressions particularly impacting Visual Stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05501v1">Hi-ZFO: Hierarchical Zeroth- and First-Order LLM Fine-Tuning via Importance-Guided Tensor Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using standard first-order (FO) optimization often drives training toward sharp, poorly generalizing minima. Conversely, zeroth-order (ZO) methods offer stronger exploratory behavior without relying on explicit gradients, yet suffer from slow convergence. More critically, our analysis reveals that in generative tasks, the vast output and search space significantly amplify estimation variance, rendering ZO methods both noisy and inefficient. To address these challenges, we propose \textbf{Hi-ZFO} (\textbf{Hi}erarchical \textbf{Z}eroth- and \textbf{F}irst-\textbf{O}rder optimization), a hybrid framework designed to synergize the precision of FO gradients with the exploratory capability of ZO estimation. Hi-ZFO adaptively partitions the model through layer-wise importance profiling, applying precise FO updates to critical layers while leveraging ZO optimization for less sensitive ones. Notably, ZO in Hi-ZFO is not merely a memory-saving surrogate; it is intentionally introduced as a source of "beneficial stochasticity" to help the model escape the local minima where pure FO optimization tends to stagnate. Validated across diverse generative, mathematical, and code reasoning tasks, Hi-ZFO consistently achieves superior performance while significantly reducing the training time. These results demonstrate the effectiveness of hierarchical hybrid optimization for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05500v1">The Evaluation Gap in Medicine, AI and LLMs: Navigating Elusive Ground Truth & Uncertainty via a Probabilistic Paradigm</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Benchmarking the relative capabilities of AI systems, including Large Language Models (LLMs) and Vision Models, typically ignores the impact of uncertainty in the underlying ground truth answers from experts. This ambiguity is particularly consequential in medicine where uncertainty is pervasive. In this paper, we introduce a probabilistic paradigm to theoretically explain how high certainty in ground truth answers is almost always necessary for even an expert to achieve high scores, whereas in datasets with high variation in ground truth answers there may be little difference between a random labeller and an expert. Therefore, ignoring uncertainty in ground truth evaluation data can result in the misleading conclusion that a non-expert has similar performance to that of an expert. Using the probabilistic paradigm, we thus bring forth the concepts of expected accuracy and expected F1 to estimate the score an expert human or system can achieve given ground truth answer variability. Our work leads to the recommendation that when establishing the capability of a system, results should be stratified by probability of the ground truth answer, typically measured by the agreement rate of ground truth experts. Stratification becomes critical when the overall performance drops below a threshold of 80%. Under stratified evaluation, performance comparison becomes more reliable in high certainty bins, mitigating the effect of the key confounding factor -- uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15835v4">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Pragmatic reasoning is pervasive in human-human communication - it allows us to leverage shared knowledge and counterfactual reasoning in order to infer the intention of a conversational partner given their ambiguous or underspecified message. In human-computer communication, underspecified messages often represent a major challenge: for instance, translating natural language instructions into code is difficult when user instructions contain inherent ambiguities. In the present paper, we aim to scale up the pragmatic "Rational Speech Act" framework to naturalistic language-to-code problems, and propose a way of dealing with multiple meaning-equivalent instruction alternatives, an issue that does not arise in previous toy-scale problems. We evaluate our method, CodeRSA, with two recent LLMs (Llama-3-8B-Instruct and Qwen-2.5-7B-Instruct) on two widely used code generation benchmarks (HumanEval and MBPP). Our experimental results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. Qualitative analyses demonstrate that it exhibits the desired behavior for the right reasons. These findings underscore the effectiveness of integrating pragmatic reasoning into a naturalistic complex communication task, language-to-code generation, offering a promising direction for enhancing code generation quality in LLMs and emphasizing the importance of pragmatic reasoning in complex communication settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05488v1">MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 19 pages (9 main + 10 appendix), 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05478v1">The Facade of Truth: Uncovering and Mitigating LLM Susceptibility to Deceptive Evidence</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      To reliably assist human decision-making, LLMs must maintain factual internal beliefs against misleading injections. While current models resist explicit misinformation, we uncover a fundamental vulnerability to sophisticated, hard-to-falsify evidence. To systematically probe this weakness, we introduce MisBelief, a framework that generates misleading evidence via collaborative, multi-round interactions among multi-role LLMs. This process mimics subtle, defeasible reasoning and progressive refinement to create logically persuasive yet factually deceptive claims. Using MisBelief, we generate 4,800 instances across three difficulty levels to evaluate 7 representative LLMs. Results indicate that while models are robust to direct misinformation, they are highly sensitive to this refined evidence: belief scores in falsehoods increase by an average of 93.0\%, fundamentally compromising downstream recommendations. To address this, we propose Deceptive Intent Shielding (DIS), a governance mechanism that provides an early warning signal by inferring the deceptive intent behind evidence. Empirical results demonstrate that DIS consistently mitigates belief shifts and promotes more cautious evidence evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13201v2">Benchmarking LLM-based Agents for Single-cell Omics Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 6 main figures; 13 supplementary figures
    </div>
    <details class="paper-abstract">
      The surge in multimodal single-cell omics data exposes limitations in traditional, manually defined analysis workflows. AI agents offer a paradigm shift, enabling adaptive planning, executable code generation, traceable decisions, and real-time knowledge fusion. However, the lack of a comprehensive benchmark critically hinders progress. We introduce a novel benchmarking evaluation system to rigorously assess agent capabilities in single-cell omics analysis. This system comprises: a unified platform compatible with diverse agent frameworks and LLMs; multidimensional metrics assessing cognitive program synthesis, collaboration, execution efficiency, bioinformatics knowledge integration, and task completion quality; and 50 diverse real-world single-cell omics analysis tasks spanning multi-omics, species, and sequencing technologies. Our evaluation reveals that Grok-3-beta achieves state-of-the-art performance among tested agent frameworks. Multi-agent frameworks significantly enhance collaboration and execution efficiency over single-agent approaches through specialized role division. Attribution analyses of agent capabilities identify that high-quality code generation is crucial for task success, and self-reflection has the most significant overall impact, followed by retrieval-augmented generation (RAG) and planning. This work highlights persistent challenges in code generation, long-context handling, and context-aware knowledge retrieval, providing a critical empirical foundation and best practices for developing robust AI agents in computational biology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05467v1">STELP: Secure Transpilation and Execution of LLM-Generated Programs</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Rapid evolution of Large Language Models (LLMs) has achieved major advances in reasoning, planning, and function-calling capabilities. Multi-agentic collaborative frameworks using such LLMs place them at the center of solving software development-related tasks such as code generation. However, direct use of LLM generated code in production software development systems is problematic. The code could be unstable or erroneous and contain vulnerabilities such as data poisoning, malicious attacks, and hallucinations that could lead to widespread system malfunctions. This prohibits the adoption of LLM generated code in production AI systems where human code reviews and traditional secure testing tools are impractical or untrustworthy. In this paper, we discuss safety and reliability problems with the execution of LLM generated code and propose a Secure Transpiler and Executor of LLM-Generated Program (STELP), capable of executing LLM-generated code in a controlled and safe manner. STELP secures autonomous production AI systems involving code generation, filling the critical void left by the impracticality or limitations of traditional secure testing methodologies and human oversight. This includes applications such as headless code generation-execution and LLMs that produce executable code snippets as an action plan to be executed in real time. We contribute a human-validated dataset of insecure code snippets and benchmark our approach on publicly available datasets for correctness, safety, and latency. Our results demonstrate that our approach outperforms an existing method by a significant margin, particularly in its ability to safely execute risky code snippets. Warning: This paper contains malicious code snippets that should be run with caution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.05459v1">Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 IJCNLP-AACL 2025 (Main), Outstanding Paper Award
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06357v1">Smart Privacy Policy Assistant: An LLM-Powered System for Transparent and Actionable Privacy Notices</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Most users agree to online privacy policies without reading or understanding them, even though these documents govern how personal data is collected, shared, and monetized. Privacy policies are typically long, legally complex, and difficult for non-experts to interpret. This paper presents the Smart Privacy Policy Assistant, an LLM-powered system that automatically ingests privacy policies, extracts and categorizes key clauses, assigns human-interpretable risk levels, and generates clear, concise explanations. The system is designed for real-time use through browser extensions or mobile interfaces, surfacing contextual warnings before users disclose sensitive information or grant risky permissions. We describe the end-to-end pipeline, including policy ingestion, clause categorization, risk scoring, and explanation generation, and propose an evaluation framework based on clause-level accuracy, policy-level risk agreement, and user comprehension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15862v2">The Subtle Art of Defection: Understanding Uncooperative Behaviors in LLM based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework for simulating and analyzing how uncooperative behaviors can destabilize or collapse LLM-based multi-agent systems. Our framework includes two key components: (1) a game theory-based taxonomy of uncooperative agent behaviors, addressing a notable gap in the existing literature; and (2) a structured, multi-stage simulation pipeline that dynamically generates and refines uncooperative behaviors as agents' states evolve. We evaluate the framework via a collaborative resource management setting, measuring system stability using metrics such as survival time and resource overuse rate. Empirically, our framework achieves 96.7% accuracy in generating realistic uncooperative behaviors, validated by human evaluations. Our results reveal a striking contrast: cooperative agents maintain perfect system stability (100% survival over 12 rounds with 0% resource overuse), while any uncooperative behavior can trigger rapid system collapse within 1 to 7 rounds. We also evaluate LLM-based defense methods, finding they detect some uncooperative behaviors, but some behaviors remain largely undetectable. These gaps highlight how uncooperative agents degrade collective outcomes and underscore the need for more resilient multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05004v3">R-Zero: Self-Evolving Reasoning LLM from Zero Data</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06301v1">Beyond BeautifulSoup: Benchmarking LLM-Powered Web Scraping for Everyday Users</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Web scraping has historically required technical expertise in HTML parsing, session management, and authentication circumvention, which limited large-scale data extraction to skilled developers. We argue that large language models (LLMs) have democratized web scraping, enabling low-skill users to execute sophisticated operations through simple natural language prompts. While extensive benchmarks evaluate these tools under optimal expert conditions, we show that without extensive manual effort, current LLM-based workflows allow novice users to scrape complex websites that would otherwise be inaccessible. We systematically benchmark what everyday users can do with off-the-shelf LLM tools across 35 sites spanning five security tiers, including authentication, anti-bot, and CAPTCHA controls. We devise and evaluate two distinct workflows: (a) LLM-assisted scripting, where users prompt LLMs to generate traditional scraping code but maintain manual execution control, and (b) end-to-end LLM agents, which autonomously navigate and extract data through integrated tool use. Our results demonstrate that end-to-end agents have made complex scraping accessible - requiring as little as a single prompt with minimal refinement (less than 5 changes) to complete workflows. We also highlight scenarios where LLM-assisted scripting may be simpler and faster for static sites. In light of these findings, we provide simple procedures for novices to use these workflows and gauge what adversaries could achieve using these.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06289v1">How well can off-the-shelf LLMs elucidate molecular structures from mass spectra using chain-of-thought reasoning?</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Mass spectrometry (MS) is a powerful analytical technique for identifying small molecules, yet determining complete molecular structures directly from tandem mass spectra (MS/MS) remains a long-standing challenge due to complex fragmentation patterns and the vast diversity of chemical space. Recent progress in large language models (LLMs) has shown promise for reasoning-intensive scientific tasks, but their capability for chemical interpretation is still unclear. In this work, we introduce a Chain-of-Thought (CoT) prompting framework and benchmark that evaluate how LLMs reason about mass spectral data to predict molecular structures. We formalize expert chemists' reasoning steps-such as double bond equivalent (DBE) analysis, neutral loss identification, and fragment assembly-into structured prompts and assess multiple state-of-the-art LLMs (Claude-3.5-Sonnet, GPT-4o-mini, and Llama-3 series) in a zero-shot setting using the MassSpecGym dataset. Our evaluation across metrics of SMILES validity, formula consistency, and structural similarity reveals that while LLMs can produce syntactically valid and partially plausible structures, they fail to achieve chemical accuracy or link reasoning to correct molecular predictions. These findings highlight both the interpretive potential and the current limitations of LLM-based reasoning for molecular elucidation, providing a foundation for future work that combines domain knowledge and reinforcement learning to achieve chemically grounded AI reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06288v1">AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-09
    </div>
    <details class="paper-abstract">
      Optimizing Large Language Model (LLM) inference in production systems is increasingly difficult due to dynamic workloads, stringent latency/throughput targets, and a rapidly expanding configuration space. This complexity spans not only distributed parallelism strategies (tensor/pipeline/expert) but also intricate framework-specific runtime parameters such as those concerning the enablement of CUDA graphs, available KV-cache memory fractions, and maximum token capacity, which drastically impact performance. The diversity of modern inference frameworks (e.g., TRT-LLM, vLLM, SGLang), each employing distinct kernels and execution policies, makes manual tuning both framework-specific and computationally prohibitive. We present AIConfigurator, a unified performance-modeling system that enables rapid, framework-agnostic inference configuration search without requiring GPU-based profiling. AIConfigurator combines (1) a methodology that decomposes inference into analytically modelable primitives - GEMM, attention, communication, and memory operations while capturing framework-specific scheduling dynamics; (2) a calibrated kernel-level performance database for these primitives across a wide range of hardware platforms and popular open-weights models (GPT-OSS, Qwen, DeepSeek, LLama, Mistral); and (3) an abstraction layer that automatically resolves optimal launch parameters for the target backend, seamlessly integrating into production-grade orchestration systems. Evaluation on production LLM serving workloads demonstrates that AIConfigurator identifies superior serving configurations that improve performance by up to 40% for dense models (e.g., Qwen3-32B) and 50% for MoE architectures (e.g., DeepSeek-V3), while completing searches within 30 seconds on average. Enabling the rapid exploration of vast design spaces - from cluster topology down to engine specific flags.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06266v1">Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software</a></div>
    <div class="paper-meta">
      📅 2026-01-09
      | 💬 Accepted to SANER 2026 (IEEE International Conference on Software Analysis, Evolution and Reengineering)
    </div>
    <details class="paper-abstract">
      Self-admitted technical debt (SATD), referring to comments flagged by developers that explicitly acknowledge suboptimal code or incomplete functionality, has received extensive attention in machine learning (ML) and traditional (Non-ML) software. However, little is known about how SATD manifests and evolves in contemporary Large Language Model (LLM)-based systems, whose architectures, workflows, and dependencies differ fundamentally from both traditional and pre-LLM ML software. In this paper, we conduct the first empirical study of SATD in the LLM era, replicating and extending prior work on ML technical debt to modern LLM-based systems. We compare SATD prevalence across LLM, ML, and non-ML repositories across a total of 477 repositories (159 per category). We perform survival analysis of SATD introduction and removal to understand the dynamics of technical debt across different development paradigms. Surprisingly, despite their architectural complexity, our results reveal that LLM repositories accumulate SATD at similar rates to ML systems (3.95% vs. 4.10%). However, we observe that LLM repositories remain debt-free 2.4x longer than ML repositories (a median of 492 days vs. 204 days), and then start to accumulate technical debt rapidly. Moreover, our qualitative analysis of 377 SATD instances reveals three new forms of technical debt unique to LLM-based development that have not been reported in prior research: Model-Stack Workaround Debt, Model Dependency Debt, and Performance Optimization Debt. Finally, by mapping SATD to stages of the LLM development pipeline, we observe that debt concentrates
    </details>
</div>
