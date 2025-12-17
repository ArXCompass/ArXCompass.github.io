# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11811v2">Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback
    </div>
    <details class="paper-abstract">
      Crowdsourced street-view imagery from social media provides real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing image geo-localization approaches, also known as Visual Place Recognition (VPR) models, exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geo-knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17399v2">DIWALI: Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment \citep{ryan-etal-2024-unintended, alkhamissi-etal-2024-investigating} and produce biased generations \cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises ~8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: https://huggingface.co/datasets/nlip/DIWALI, project webpage https://nlip-lab.github.io/nlip/publications/diwali/, and our codebase with model outputs can be found here: https://github.com/pramitsahoo/culture-evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14142v1">Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed as intelligent agents. Their multi-stage workflows, which alternate between local computation and calls to external network services like Web APIs, introduce a mismatch in their execution pattern and the scheduling granularity of existing inference systems such as vLLM. Existing systems typically focus on per-segment optimization which prevents them from minimizing the end-to-end latency of the complete agentic workflow, i.e., the global Job Completion Time (JCT) over the entire request lifecycle. To address this limitation, we propose Astraea, a service engine designed to shift the optimization from local segments to the global request lifecycle. Astraea employs a state-aware, hierarchical scheduling algorithm that integrates a request's historical state with future predictions. It dynamically classifies requests by their I/O and compute intensive nature and uses an enhanced HRRN policy to balance efficiency and fairness. Astraea also implements an adaptive KV cache manager that intelligently handles the agent state during I/O waits based on the system memory pressure. Extensive experiments show that Astraea reduces average JCT by up to 25.5\% compared to baseline methods. Moreover, our approach demonstrates strong robustness and stability under high load across various model scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14138v1">LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Many real-world tasks, such as trip planning or meal planning, can be formulated as combinatorial optimization problems. However, using optimization solvers is difficult for end users because it requires problem instantiation: defining candidate items, assigning preference scores, and specifying constraints. We introduce LAPPI (LLM-Assisted Preference-based Problem Instantiation), an interactive approach that uses large language models (LLMs) to support users in this instantiation process. Through natural language conversations, the system helps users transform vague preferences into well-defined optimization problems. These instantiated problems are then passed to existing optimization solvers to generate solutions. In a user study on trip planning, our method successfully captured user preferences and generated feasible plans that outperformed both conventional and prompt-engineering approaches. We further demonstrate LAPPI's versatility by adapting it to an additional use case.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14121v1">SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances of Large Language Models (LMMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by constrasting the keyframes with the targe models. Finally, we propose SportsRAG, a RAG-based training guidance model based on Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22333v2">PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by ASPLOS'26, code available at https://github.com/flashserve/PAT
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: one-query-per-CTA execution repeatedly loads shared prefix KV cache, while one-size-fits-all tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 53.5% on average and TPOT by 17.0-93.1% under the same configurations against state-of-the-art attention kernels. PAT's source code is publicly available at https://github.com/flashserve/PAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13109v3">FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14522v3">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. An evaluation on 100+ real-world packages, including high-profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<30m on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when given the source code of the component and prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17874v2">Beyond Jailbreak: Unveiling Risks in LLM Applications Arising from Blurred Capability Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      LLM applications (i.e., LLM apps) leverage the powerful capabilities of LLMs to provide users with customized services, revolutionizing traditional application development. While the increasing prevalence of LLM-powered applications provides users with unprecedented convenience, it also brings forth new security challenges. For such an emerging ecosystem, the security community lacks sufficient understanding of the LLM application ecosystem, especially regarding the capability boundaries of the applications themselves. In this paper, we systematically analyzed the new development paradigm and defined the concept of the LLM app capability space. We also uncovered potential new risks beyond jailbreak that arise from ambiguous capability boundaries in real-world scenarios, namely, capability downgrade and upgrade. To evaluate the impact of these risks, we designed and implemented an LLM app capability evaluation framework, LLMApp-Eval. First, we collected application metadata across 4 platforms and conducted a cross-platform ecosystem analysis. Then, we evaluated the risks for 199 popular applications among 4 platforms and 6 open-source LLMs. We identified that 178 (89.45%) potentially affected applications, which can perform tasks from more than 15 scenarios or be malicious. We even found 17 applications in our study that executed malicious tasks directly, without applying any adversarial rewriting. Furthermore, our experiments also reveal a positive correlation between the quality of prompt design and application robustness. We found that well-designed prompts enhance security, while poorly designed ones can facilitate abuse. We hope our work inspires the community to focus on the real-world risks of LLM applications and foster the development of a more robust LLM application ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v2">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13979v1">ReflCtrl: Controlling LLM Reflection via Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Spotlight in NeurIPS 25 MI workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with Chain-of-Thought (CoT) reasoning have achieved strong performance across diverse tasks, including mathematics, coding, and general reasoning. A distinctive ability of these reasoning models is self-reflection: the ability to review and revise previous reasoning steps. While self-reflection enhances reasoning performance, it also increases inference cost. In this work, we study self-reflection through the lens of representation engineering. We segment the model's reasoning into steps, identify the steps corresponding to reflection, and extract a reflection direction in the latent space that governs this behavior. Using this direction, we propose a stepwise steering method that can control reflection frequency. We call our framework ReflCtrl. Our experiments show that (1) in many cases reflections are redundant, especially in stronger models (in our experiments, we can save up to 33.6 percent of reasoning tokens while preserving performance), and (2) the model's reflection behavior is highly correlated with an internal uncertainty signal, implying self-reflection may be controlled by the model's uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13978v1">Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark on a Textbook in Theoretical Computer Science about Randomized Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has led to significant breakthroughs in automated mathematical reasoning and scientific discovery. Georgiev, G${ó}$mez-Serrano, Tao, and Wagner [GGSTW+25] demonstrate that AI systems can explore new constructions and improve existing bounds, illustrating the growing potential of LLMs to accelerate mathematical discovery. Similarly, Bubeck et al. [BCE+25] show that GPT-5 can meaningfully contribute to scientific workflows, from proposing hypotheses to generating proofs and analyses. Despite these advances, a rigorous evaluation of these models on canonical, graduate-level mathematical theory remains necessary to understand their baseline reasoning capabilities. In this paper, we present a comprehensive benchmark of four frontier models: GPT-5-Thinking, Gemini-3-Pro, Claude-Sonnet-4.5-Thinking, and Grok-4 against the classic curriculum of Randomized Algorithms by Motwani and Raghavan [MR95]. We tasked each model with generating formal LaTeX proofs for a series of lemmas and exercises spanning the textbook. We find that while the top-tier models (Gemini, and Claude) achieve a high accuracy rate (approx. 66%), demonstrating a robust grasp of probabilistic method and formal logic, other models lag significantly in consistency (approx. 40%). We provide a qualitative analysis of the generated proofs, highlighting differences in conciseness, hallucination rates, and logical structure. Our results suggest that while frontier models have reached a threshold of proficiency suitable for graduate-level pedagogical assistance and formalization, significant variance exists in their reliability for rigorous mathematical derivation. The code and the full set of LLM-generated responses are open-sourced and publicly available at https://github.com/magiclinux/math_benchmark_probability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13974v1">Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining. Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive. This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report. To this end, we proposed a multi-layer framework with two main modules: robotics and AI. On the robotics side, SLAM and autonomous navigation provide repeatable coverage and targeted revisits via waypoints. On AI side, a Vision Language Model (VLM)-based layer produces scene descriptions; a retrieval component powered grounds those descriptions in OSHA and site policies; Another VLM-based layer assesses the safety situation based on rules; and finally Large Language Model (LLM) layer generates safety reports based on previous outputs. The framework is validated with a proof-of-concept implementation and evaluated in a lab environment that simulates common hazards across three scenarios. Results show high recall with competitive precision compared to state-of-the-art closed-source models. This paper contributes a transparent, generalizable pipeline that moves beyond black-box models by exposing intermediate artifacts from each layer and keeping the human in the loop. This work provides a foundation for future extensions to additional tasks and settings within and beyond construction context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13676v1">Towards Effective Model Editing for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 15 pages (including appendix), 7 figures. Code, data, results, and additional resources are available at: https://model-editing.github.io
    </div>
    <details class="paper-abstract">
      Personalization is becoming indispensable for LLMs to align with individual user preferences and needs. Yet current approaches are often computationally expensive, data-intensive, susceptible to catastrophic forgetting, and prone to performance degradation in multi-turn interactions or when handling implicit queries. To address these challenges, we conceptualize personalization as a model editing task and introduce Personalization Editing, a framework that applies localized edits guided by clustered preference representations. This design enables precise preference-aligned updates while preserving overall model capabilities. In addition, existing personalization benchmarks frequently rely on persona-based dialogs between LLMs rather than user-LLM interactions, or focus primarily on stylistic imitation while neglecting information-seeking tasks that require accurate recall of user-specific preferences. We introduce User Preference Question Answering (UPQA), a short-answer QA dataset constructed from in-situ user queries with varying levels of difficulty. Unlike prior benchmarks, UPQA directly evaluates a model's ability to recall and apply specific user preferences. Across experimental settings, Personalization Editing achieves higher editing accuracy and greater computational efficiency than fine-tuning, while outperforming prompting-based baselines in multi-turn conversations and implicit preference questions settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.03604v3">Bilevel ZOFO: Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning (PEFT) methods address these by freezing most model parameters and training only a small subset. However, PEFT often underperforms compared to full fine-tuning when high task-specific accuracy is required. Zeroth-Order (ZO) methods fine-tune the entire pre-trained model without back-propagation, estimating gradients through forward passes only. While memory-efficient, ZO methods suffer from slow convergence and high sensitivity to prompt selection. We bridge these two worlds with Bilevel-ZOFO, a bilevel optimization method that couples fast, local FO-PEFT adaptation at the inner level with stable, memory-efficient ZO updates of the full backbone at the outer level. The FO-PEFT inner loop performs fast, low-memory local adaptation that reduces the variance of ZO estimates and stabilizes the search, guiding the outer ZO updates of the full backbone and reducing prompt sensitivity. In the mean time, the outer ZO provides better generalization ability for PEFT. We provide theoretical convergence guarantees and empirically demonstrate that Bilevel-ZOFO significantly outperforms existing ZO and FO-PEFT methods, achieving 2-4 times faster training while maintaining similar memory efficiency. Additionally, we show by updating the backbone with ZO and adapting only a tiny FO-PEFT block per task, Bilevel-ZOFO combines full-model capacity with few-shot efficiency, making it a very efficient meta-learning algorithm that quickly adapts to new tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v2">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13526v1">Async Control: Stress-testing Asynchronous Control Measures for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      LLM-based software engineering agents are increasingly used in real-world development tasks, often with access to sensitive data or security-critical codebases. Such agents could intentionally sabotage these codebases if they were misaligned. We investigate asynchronous monitoring, in which a monitoring system reviews agent actions after the fact. Unlike synchronous monitoring, this approach does not impose runtime latency, while still attempting to disrupt attacks before irreversible harm occurs. We treat monitor development as an adversarial game between a blue team (who design monitors) and a red team (who create sabotaging agents). We attempt to set the game rules such that they upper bound the sabotage potential of an agent based on Claude 4.1 Opus. To ground this game in a realistic, high-stakes deployment scenario, we develop a suite of 5 diverse software engineering environments that simulate tasks that an agent might perform within an AI developer's internal infrastructure. Over the course of the game, we develop an ensemble monitor that achieves a 6% false negative rate at 1% false positive rate on a held out test environment. Then, we estimate risk of sabotage at deployment time by extrapolating from our monitor's false negative rate. We describe one simple model for this extrapolation, present a sensitivity analysis, and describe situations in which the model would be invalid. Code is available at: https://github.com/UKGovernmentBEIS/async-control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08403v2">DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Audio deepfake detection has recently garnered public concern due to its implications for security and reliability. Traditional deep learning methods have been widely applied to this task but often lack generalisability when confronted with newly emerging spoofing techniques and more tasks such as spoof attribution recognition rather than simple binary classification. In principle, Large Language Models (LLMs) are considered to possess the needed generalisation capabilities. However, previous research on Audio LLMs (ALLMs) indicates a generalization bottleneck in audio deepfake detection performance, even when sufficient data is available. Consequently, this study investigates the model architecture and examines the effects of the primary components of ALLMs, namely the audio encoder and the text-based LLM. Our experiments demonstrate that the careful selection and combination of audio encoders and text-based LLMs are crucial for unlocking the deepfake detection potential of ALLMs. We further propose an ALLM structure capable of generalizing deepfake detection abilities to out-of-domain spoofing tests and other deepfake tasks, such as spoof positioning and spoof attribution recognition. Our proposed model architecture achieves state-of-the-art (SOTA) performance across multiple datasets, including ASVSpoof2019, InTheWild, and Demopage, with accuracy reaching up to 95.76% on average, and exhibits competitive capabilities in other deepfake detection tasks such as attribution, and localisation compared to SOTA audio understanding models. Data and codes are provided in supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13481v1">neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Envy is a common human behavior that shapes competitiveness and can alter outcomes in team settings. As large language models (LLMs) increasingly act on behalf of humans in collaborative and competitive workflows, there is a pressing need to evaluate whether and under what conditions they exhibit envy-like preferences. In this paper, we test whether LLMs show envy-like behavior toward each other. We considered two scenarios: (1) A point allocation game that tests whether a model tries to win over its peer. (2) A workplace setting observing behaviour when recognition is unfair. Our findings reveal consistent evidence of envy-like patterns in certain LLMs, with large variation across models and contexts. For instance, GPT-5-mini and Claude-3.7-Sonnet show a clear tendency to pull down the peer model to equalize outcomes, whereas Mistral-Small-3.2-24B instead focuses on maximizing its own individual gains. These results highlight the need to consider competitive dispositions as a safety and design factor in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13438v1">From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents show great potential for automated UI navigation such as automated UI testing and AI assistants, their efficiency has been largely overlooked. Our motivating study reveals that inefficient UI representation creates a critical performance bottleneck. However, UI representation optimization, formulated as the task of automatically generating programs that transform UI representations, faces two unique challenges. First, the lack of Boolean oracles, which traditional program synthesis uses to decisively validate semantic correctness, poses a fundamental challenge to co-optimization of token efficiency and completeness. Second, the need to process large, complex UI trees as input while generating long, compositional transformation programs, making the search space vast and error-prone. Toward addressing the preceding limitations, we present UIFormer, the first automated optimization framework that synthesizes UI transformation programs by conducting constraint-based optimization with structured decomposition of the complex synthesis task. First, UIFormer restricts the program space using a domain-specific language (DSL) that captures UI-specific operations. Second, UIFormer conducts LLM-based iterative refinement with correctness and efficiency rewards, providing guidance for achieving the efficiency-completeness co-optimization. UIFormer operates as a lightweight plugin that applies transformation programs for seamless integration with existing LLM agents, requiring minimal modifications to their core logic. Evaluations across three UI navigation benchmarks spanning Android and Web platforms with five LLMs demonstrate that UIFormer achieves 48.7% to 55.8% token reduction with minimal runtime overhead while maintaining or improving agent performance. Real-world industry deployment at WeChat further validates the practical impact of UIFormer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00319v2">RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 13 pages, 9 figures. Model is available at https://huggingface.co/Freakz3z/Qwen-JSON
    </div>
    <details class="paper-abstract">
      The Structure Gap between probabilistic LLM generation and deterministic schema requirements hinders automated workflows. We propose RL-Struct, a lightweight framework using Gradient Regularized Policy Optimization (GRPO) with a hierarchical reward function to align LLMs with structural constraints. This approach eliminates the critic network, reducing peak VRAM by 38% compared to PPO. On complex JSON tasks, RL-Struct achieves 89.7% structural accuracy and 92.1% validity, significantly outperforming SFT and zero-shot baselines. We also report an emergent curriculum--a self-organized learning process where the model prioritizes syntax before semantics. Our model is publicly available at https://huggingface.co/Freakz3z/Qwen-JSON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08768v4">AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://github.com/gufranSabri/deepseek-evals
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13360v1">UCRBench: Benchmarking LLMs on Use Case Recovery</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Use cases are widely employed to specify functional requirements, yet existing benchmarks are scarce and face the risk of being misaligned with actual system behavior, similarly limiting the rigorous evaluation of large language models (LLMs) in generating use cases from source code. We address this gap by introducing code-aligned use case benchmarks, constructed through manual validation of both user-goal and subfunction use cases across nine real-world software projects. Using this benchmark, we conduct the first systematic study of LLMs and propose a hierarchical evaluation protocol that assesses actor correctness, name accuracy, path fidelity, and behavioral coverage. The results show that while LLMs can partially reconstruct system functionality, their performance varies significantly across projects, with particularly noticeable shortcomings in domain-specific and multi-module systems. The models also exhibit high omission rates and struggle to maintain consistent abstraction when aggregating subfunctions into user-goal use cases, highlighting both the potential and current limitations of LLM-based use case reverse engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13337v1">FROC: A Unified Framework with Risk-Optimized Control for Machine Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Machine unlearning (MU) seeks to eliminate the influence of specific training examples from deployed models. As large language models (LLMs) become widely used, managing risks arising from insufficient forgetting or utility loss is increasingly crucial. Current MU techniques lack effective mechanisms for evaluating and controlling these risks, hindering the selection of strategies that appropriately balance safety and utility, and raising trust concerns surrounding the "right to be forgotten." To address these issues, we propose FROC, a unified framework with Risk-Optimized Control for machine unlearning in LLMs. FROC is built around a conformal-style risk-control formulation that expresses a user-specified risk budget on unlearning behavior. This probability-based constraint enables FROC to compare MU strategies, identify feasible operating regions, and guide hyperparameter selection according to desired trade-offs between forgetting sufficiency and utility preservation. To operationalize this constraint, FROC introduces a smoothly varying continuous risk model that aggregates forgetting deficiency and utility degradation into a single configuration-level score. Building on conformal risk analysis, FROC computes (1) the Conformal Unlearning Risk (CUR), a data-driven estimated value on the probability that forgotten samples continue to influence model predictions, and (2) risk-controlled configuration sets, which identify unlearning hyperparameters that are valid under the specified risk budget. Experiments across multiple LLM MU methods demonstrate that FROC produces stable, interpretable risk landscapes and reveals consistent relationships between unlearning configurations, semantic shift, and utility impact. FROC reframes MU as a controllable, risk-aware process and offers a practical foundation for managing unlearning behavior in large-scale LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10611v2">Phythesis: Physics-Guided Evolutionary Scene Synthesis for Energy-Efficient Data Center Design via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Data center (DC) infrastructure serves as the backbone to support the escalating demand for computing capacity. Traditional design methodologies that blend human expertise with specialized simulation tools scale poorly with the increasing system complexity. Recent studies adopt generative artificial intelligence to design plausible human-centric indoor layouts. However, they do not consider the underlying physics, making them unsuitable for the DC design that sets quantifiable operational objectives and strict physical constraints. To bridge the gap, we propose Phythesis, a novel framework that synergizes large language models (LLMs) and physics-guided evolutionary optimization to automate simulation-ready (SimReady) scene synthesis for energy-efficient DC design. Phythesis employs an iterative bi-level optimization architecture, where (i) the LLM-driven optimization level generates physically plausible three-dimensional layouts and self-criticizes them to refine the scene topology, and (ii) the physics-informed optimization level identifies the optimal asset parameters and selects the best asset combination. Experiments on three generation scales show that Phythesis achieves 57.3% generation success rate increase and 11.5% power usage effectiveness (PUE) improvement, compared with the vanilla LLM-based solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13298v1">MiniLingua: A Small Open-Source LLM for European Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 9+6 pages, 6 figures and 3 tables in the main text. Code at https://github.com/MiniLingua-ai/training_artifacts
    </div>
    <details class="paper-abstract">
      Large language models are powerful but often limited by high computational cost, privacy concerns, and English-centric training. Recent progress demonstrates that small, efficient models with around one billion parameters can deliver strong results and enable on-device use. This paper introduces MiniLingua, a multilingual open-source LLM of one billion parameters trained from scratch for 13 European languages, designed to balance coverage and instruction-following capabilities. Based on evaluation results, the instruction-tuned version of MiniLingua outperforms EuroLLM, a model with a similar training approach but a larger training budget, on summarization, classification and both open- and closed-book question answering. Moreover, it remains competitive with more advanced state-of-the-art models on open-ended generation tasks. We release model weights, tokenizer and source code used for data processing and model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18375v3">Progressive Localisation in Localist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      This paper demonstrates that progressive localization, the gradual increase of attention locality from early distributed layers to late localized layers, represents the optimal architecture for creating interpretable large language models (LLMs) while preserving performance. Through systematic experimentation with GPT-2 fine-tuned on The Psychology of Artificial Superintelligence, we evaluate five locality configurations: two uniform baselines (fully distributed and fully localist) and three progressive polynomial schedules. We investigate whether interpretability constraints can be aligned with natural semantic structure while being applied strategically across network depth. We demonstrate that progressive semantic localization, combining adaptive semantic block partitioning with steep polynomial locality schedules, achieves near-baseline language modeling performance while providing interpretable attention patterns. Multiple independent training runs with different random seeds establish that results are statistically robust and highly reproducible. The approach dramatically outperforms both fixed-window localization and naive uniform locality constraints. Analysis reveals that maintaining flexibility through low-fidelity constraints preserves model capacity while providing interpretability benefits, and that steep schedules concentrating locality in decision-critical final layers while preserving distributed learning in early layers achieve near-baseline attention distribution characteristics. These findings demonstrate that interpretability mechanisms should align with semantic structure to achieve practical performance-interpretability tradeoffs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06522v3">Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09549v2">Chasing Shadows: Pitfalls in LLM Security Research</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 About to appear at NDSS'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in security research. Their unique characteristics, however, introduce challenges that undermine established paradigms of reproducibility, rigor, and evaluation. Prior work has identified common pitfalls in traditional machine learning research, but these studies predate the advent of LLMs. In this paper, we identify nine common pitfalls that have become (more) relevant with the emergence of LLMs and that can compromise the validity of research involving them. These pitfalls span the entire computation process, from data collection, pre-training, and fine-tuning to prompting and evaluation. We assess the prevalence of these pitfalls across all 72 peer-reviewed papers published at leading Security and Software Engineering venues between 2023 and 2024. We find that every paper contains at least one pitfall, and each pitfall appears in multiple papers. Yet only 15.7% of the present pitfalls were explicitly discussed, suggesting that the majority remain unrecognized. To understand their practical impact, we conduct four empirical case studies showing how individual pitfalls can mislead evaluation, inflate performance, or impair reproducibility. Based on our findings, we offer actionable guidelines to support the community in future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15101v2">Cost-aware LLM-based Online Dataset Annotation</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled automated dataset labeling with minimal human supervision. While majority voting across multiple LLMs can improve label reliability by mitigating individual model biases, it incurs high computational costs due to repeated querying. In this work, we propose a novel online framework, Cost-aware Majority Voting (CaMVo), for efficient and accurate LLM-based dataset annotation. CaMVo adaptively selects a subset of LLMs for each data instance based on contextual embeddings, balancing confidence and cost without requiring pre-training or ground-truth labels. Leveraging a LinUCB-based selection mechanism and a Bayesian estimator over confidence scores, CaMVo estimates a lower bound on labeling accuracy for each LLM and aggregates responses through weighted majority voting. Our empirical evaluation on the MMLU and IMDB Movie Review datasets demonstrates that CaMVo achieves comparable or superior accuracy to full majority voting while significantly reducing labeling costs. This establishes CaMVo as a practical and robust solution for cost-efficient annotation in dynamic labeling environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13077v1">LikeBench: Evaluating Subjective Likability in LLMs for Personalization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      A personalized LLM should remember user facts, apply them correctly, and adapt over time to provide responses that the user prefers. Existing LLM personalization benchmarks are largely centered on two axes: accurately recalling user information and accurately applying remembered information in downstream tasks. We argue that a third axis, likability, is both subjective and central to user experience, yet under-measured by current benchmarks. To measure likability holistically, we introduce LikeBench, a multi-session, dynamic evaluation framework that measures likability across multiple dimensions by how much an LLM can adapt over time to a user's preferences to provide more likable responses. In LikeBench, the LLMs engage in conversation with a simulated user and learn preferences only from the ongoing dialogue. As the interaction unfolds, models try to adapt to responses, and after each turn, they are evaluated for likability across seven dimensions by the same simulated user. To the best of our knowledge, we are the first to decompose likability into multiple diagnostic metrics: emotional adaptation, formality matching, knowledge adaptation, reference understanding, conversation length fit, humor fit, and callback, which makes it easier to pinpoint where a model falls short. To make the simulated user more realistic and discriminative, LikeBench uses fine-grained, psychologically grounded descriptive personas rather than the coarse high/low trait rating based personas used in prior work. Our benchmark shows that strong memory performance does not guarantee high likability: DeepSeek R1, with lower memory accuracy (86%, 17 facts/profile), outperformed Qwen3 by 28% on likability score despite Qwen3's higher memory accuracy (93%, 43 facts/profile). Even SOTA models like GPT-5 adapt well in short exchanges but show only limited robustness in longer, noisier interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00507v2">Towards Resource-Efficient Serverless LLM Inference with SLINFER</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 To appear at HPCA 2026
    </div>
    <details class="paper-abstract">
      The rise of LLMs has driven demand for private serverless deployments, characterized by moderate-sized models and infrequent requests. While existing serverless solutions follow exclusive GPU allocation, we take a step back to explore modern platforms and find that: Emerging CPU architectures with built-in accelerators are capable of serving LLMs but remain underutilized, and both CPUs and GPUs can accommodate multiple LLMs simultaneously. We propose SLINFER, a resource-efficient serverless inference scheme tailored for small- to mid-sized LLMs that enables elastic and on-demand sharing across heterogeneous hardware. SLINFER tackles three fundamental challenges: (1) precise, fine-grained compute resource allocation at token-level to handle fluctuating computational demands; (2) a coordinated and forward-looking memory scaling mechanism to detect out-of-memory hazards and reduce operational overhead; and (3) a dual approach that consolidates fragmented instances through proactive preemption and reactive bin-packing. Experimental results on 4 32-core CPUs and 4 A100 GPUs show that SLINFER improves serving capacity by 47% - 62% through sharing, while further leveraging CPUs boosts this to 86% - 154%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13063v1">LLM Rationalis? Measuring Bargaining Capabilities of AI Negotiators</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Published in the First Workshop on Multi-Turn Interactions in Large Language Models at Neurips 2025
    </div>
    <details class="paper-abstract">
      Bilateral negotiation is a complex, context-sensitive task in which human negotiators dynamically adjust anchors, pacing, and flexibility to exploit power asymmetries and informal cues. We introduce a unified mathematical framework for modeling concession dynamics based on a hyperbolic tangent curve, and propose two metrics burstiness tau and the Concession-Rigidity Index (CRI) to quantify the timing and rigidity of offer trajectories. We conduct a large-scale empirical comparison between human negotiators and four state-of-the-art large language models (LLMs) across natural-language and numeric-offers settings, with and without rich market context, as well as six controlled power-asymmetry scenarios. Our results reveal that, unlike humans who smoothly adapt to situations and infer the opponents position and strategies, LLMs systematically anchor at extremes of the possible agreement zone for negotiations and optimize for fixed points irrespective of leverage or context. Qualitative analysis further shows limited strategy diversity and occasional deceptive tactics used by LLMs. Moreover the ability of LLMs to negotiate does not improve with better models. These findings highlight fundamental limitations in current LLM negotiation capabilities and point to the need for models that better internalize opponent reasoning and context-dependent strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13040v1">Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10449v2">When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09321v3">ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 To appear in NDSS 2026. For slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf
    </div>
    <details class="paper-abstract">
      Prompt injection attacks aim to contaminate the input data of an LLM to mislead it into completing an attacker-chosen task instead of the intended task. In many applications and agents, the input data originates from multiple sources, with each source contributing a segment of the overall input. In these multi-source scenarios, an attacker may control only a subset of the sources and contaminate the corresponding segments, but typically does not know the order in which the segments are arranged within the input. Existing prompt injection attacks either assume that the entire input data comes from a single source under the attacker's control or ignore the uncertainty in the ordering of segments from different sources. As a result, their success is limited in domains involving multi-source data. In this work, we propose ObliInjection, the first prompt injection attack targeting LLM applications and agents with multi-source input data. ObliInjection introduces two key technical innovations: the order-oblivious loss, which quantifies the likelihood that the LLM will complete the attacker-chosen task regardless of how the clean and contaminated segments are ordered; and the orderGCG algorithm, which is tailored to minimize the order-oblivious loss and optimize the contaminated segments. Comprehensive experiments across three datasets spanning diverse application domains and twelve LLMs demonstrate that ObliInjection is highly effective, even when only one out of 6-100 segments in the input data is contaminated. Our code and data are available at: https://github.com/ReachalWang/ObliInjection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23188v3">Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Upon further review, we realized that the version submitted to arXiv was not the final draft and omits crucial results and discussion. To avoid confusion and ensure the integrity of the record, we request withdrawal and will resubmit once the complete work is ready
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12928v1">PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) for interactive applications necessitates serving systems that can handle thousands of concurrent requests with diverse Service Level Objective (SLO) requirements. A critical yet often overlooked dimension in this context is the inherent priority difference among clients; for instance, business-critical functions demand higher performance guarantees, as fulfilling such requests yields significantly greater business value. However, existing LLM serving schedulers fail to jointly optimize for both SLO attainment and client-level priorities. To bridge this gap, we first \textit{formalize multi-priority request scheduling as a service gain maximization problem}, where satisfying latency requirements for requests of different priorities contributes varying levels of gain. We then propose PROSERVE, a unified two-tier scheduling framework designed to maximize overall service gain. At the engine level, SlideBatching dynamically adapts batch formation and request ordering under varying load conditions, employing a sliding boundary mechanism to balance deadline-first and density-first strategies. At the service level, GoRouting performs gain-oriented and capability-aware dispatching across distributed instances, proactively reserving capacity for future high-priority or long requests. Extensive evaluation across four open-source datasets and a real-world industrial trace demonstrates that \systemname{} consistently outperforms state-of-the-art baselines, improving system gain by up to 35% and boosting SLO attainment by up to 52%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12922v1">LLM-based Personalized Portfolio Recommender: Integrating Large Language Models and Reinforcement Learning for Intelligent Investment Strategy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      In modern financial markets, investors increasingly seek personalized and adaptive portfolio strategies that reflect their individual risk preferences and respond to dynamic market conditions. Traditional rule-based or static optimization approaches often fail to capture the nonlinear interactions among investor behavior, market volatility, and evolving financial objectives. To address these limitations, this paper introduces the LLM-based Personalized Portfolio Recommender , an integrated framework that combines Large Language Models, reinforcement learning, and individualized risk preference modeling to support intelligent investment decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12914v1">CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01198v2">Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 This is a duplicate of arXiv:2511.23059
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop (HITL) framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis (IPA). Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient, and replicable HITL methodological pathway for the translation of ancient, concept-dense texts such as TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13914v1">Context Branching for LLM Conversations: A Version Control Approach to Exploratory Programming</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 11 pages, 4 figures, 2 tables, 1 code snippet, 4 algorithms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to software engineering workflows, yet their effectiveness degrades significantly in multi-turn conversations. Recent studies demonstrate an average 39% performance drop when instructions are delivered across multiple turns, with models making premature assumptions and failing to course correct (Laban et al., 2025). This degradation is particularly problematic in exploratory programming tasks where developers need to investigate alternative approaches without committing to a single path. Current solutions force users into a false dichotomy: continue in a context-polluted conversation where the LLM becomes increasingly confused, or start fresh and lose all accumulated context. We present ContextBranch, a conversation management system that applies version control semantics to LLM interactions. ContextBranch provides four core primitives--checkpoint, branch, switch, and inject--enabling users to capture conversation state, explore alternatives in isolation, and selectively merge insights. We evaluate ContextBranch through a controlled experiment with 30 software engineering scenarios featuring intentionally polluting explorations. Branched conversations achieved higher response quality compared to linear conversations, with large improvements in focus and context awareness. Benefits were concentrated in complex scenarios involving conceptually distant explorations. Branching reduced context size by 58.1% (31.0 to 13.0 messages), eliminating irrelevant exploratory content. Our work establishes conversation branching as a fundamental primitive for AI-assisted exploratory work, demonstrating that isolation prevents context pollution when exploring alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17985v3">How K-12 Educators Use AI: LLM-Assisted Qualitative Analysis at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      This study investigates how K-12 educators use generative AI tools in real-world instructional contexts and how large language models (LLMs) can support scalable qualitative analysis of these interactions. Drawing on over 13,000 unscripted educator-AI conversations from an open-access platform, we examine educators' use of AI for lesson planning, differentiation, assessment, and pedagogical reflection. Methodologically, we introduce a replicable, LLM-assisted qualitative analysis pipeline that supports inductive theme discovery, codebook development, and large-scale annotation while preserving researcher control over conceptual synthesis. Empirically, the findings surface concrete patterns in how educators prompt, adapt, and evaluate AI-generated suggestions as part of their instructional reasoning. This work demonstrates the feasibility of combining LLM support with qualitative rigor to analyze complex educator behaviors at scale and inform the design of AI-powered educational tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13898v1">Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Progress on training and architecture strategies has enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapidly diminishing returns and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose a simple method that, through targeted gradient updates on the given context, provably overcomes limitations of static self-attention. We find that this shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. Our method leads to large 12.6 and 14.1 percentage point improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20768v2">RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Presented as a poster at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13886v1">OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Post-training model pruning is a promising solution, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT offer a practical trade-off in optimality by applying efficient, approximate heuristic weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. Solving these QPs yields the per-row globally optimal update with respect to the reconstruction objective given the estimated Hessian. The shared-Hessian structure makes the problem highly amenable to batching on accelerators. We implement an accelerator-friendly QP solver that accumulates one Hessian per layer and solves many small QPs in parallel, enabling one-shot post-training pruning at scale on a single accelerator without fine-tuning. OPTIMA integrates with existing mask selectors and consistently improves zero-shot performance across multiple LLM families and sparsity regimes, yielding up to 3.97% absolute accuracy improvement. On an NVIDIA H100, OPTIMA prunes a 8B-parameter transformer end-to-end in 40 hours with 60GB peak memory. Together, these results set a new state-of-the-art accuracy-efficiency trade-offs for one-shot post-training pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v3">LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 12 Figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Since all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9x-3.6x lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72x reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13860v1">Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-Editors</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Accepted by AAAI 2026 Workshop on Agentic AI Benchmarks and Applications for Enterprise Tasks
    </div>
    <details class="paper-abstract">
      Tool calling enables large language models (LLMs) to interact with external environments through tool invocation, providing a practical way to overcome the limitations of pretraining. However, the effectiveness of tool use depends heavily on the quality of the associated documentation and knowledge base context. These materials are usually written for human users and are often misaligned with how LLMs interpret information. This problem is even more pronounced in industrial settings, where hundreds of tools with overlapping functionality create challenges in scalability, variability, and ambiguity. We propose Verification-Guided Context Optimization (VGCO), a framework that uses LLMs as editors to automatically refine tool-related documentation and knowledge base context. VGCO works in two stages. First, Evaluation collects real-world failure cases and identifies mismatches between tools and their context. Second, Optimization performs hierarchical editing through offline learning with structure-aware, in-context optimization. The novelty of our LLM editors has three main aspects. First, they use a hierarchical structure that naturally integrates into the tool-calling workflow. Second, they are state-aware, action-specific, and verification-guided, which constrains the search space and enables efficient, targeted improvements. Third, they enable cost-efficient sub-task specialization, either by prompt engineering large editor models or by post-training smaller editor models. Unlike prior work that emphasizes multi-turn reasoning, VGCO focuses on the single-turn, large-scale tool-calling problem and achieves significant improvements in accuracy, robustness, and generalization across LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13857v1">EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08786v2">A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 This paper is accepted at the NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of aligning large language models (LLMs) with diverse human preferences within federated learning (FL) environments, where standard methods often fail to adequately represent diverse viewpoints. We introduce a comprehensive evaluation framework that systematically assesses the trade-off between alignment quality and fairness when using different aggregation strategies for human preferences. In our federated setting, each group locally evaluates rollouts and produces reward signals, and the server aggregates these group-level rewards without accessing any raw data. Specifically, we evaluate standard reward aggregation techniques (min, max, and average) and introduce a novel adaptive scheme that dynamically adjusts preference weights based on a group's historical alignment performance. Our experiments on question-answering (Q/A) tasks using a PPO-based RLHF pipeline demonstrate that our adaptive approach consistently achieves superior fairness while maintaining competitive alignment scores. This work offers a robust methodology for evaluating LLM behavior across diverse populations and provides a practical solution for developing truly pluralistic and fairly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12868v1">Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel on multiple-choice clinical diagnosis benchmarks, yet it is unclear how much of this performance reflects underlying probabilistic reasoning. We study this through questions from MedQA, where the task is to select the most likely diagnosis. We introduce the Frequency-Based Probabilistic Ranker (FBPR), a lightweight method that scores options with a smoothed Naive Bayes over concept-diagnosis co-occurrence statistics from a large corpus. When co-occurrence statistics were sourced from the pretraining corpora for OLMo and Llama, FBPR achieves comparable performance to the corresponding LLMs pretrained on that same corpus. Direct LLM inference and FBPR largely get different questions correct, with an overlap only slightly above random chance, indicating complementary strengths of each method. These findings highlight the continued value of explicit probabilistic baselines: they provide a meaningful performance reference point and a complementary signal for potential hybridization. While the performance of LLMs seems to be driven by a mechanism other than simple frequency aggregation, we show that an approach similar to the historically grounded, low-complexity expert systems still accounts for a substantial portion of benchmark performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.23260v2">From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 The paper is published in ICT Express (Elsevier)
    </div>
    <details class="paper-abstract">
      Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12812v1">Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, LLaMA</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Prompt engineering has emerged as a critical factor influencing large language model (LLM) performance, yet the impact of pragmatic elements such as linguistic tone and politeness remains underexplored, particularly across different model families. In this work, we propose a systematic evaluation framework to examine how interaction tone affects model accuracy and apply it to three recently released and widely available LLMs: GPT-4o mini (OpenAI), Gemini 2.0 Flash (Google DeepMind), and Llama 4 Scout (Meta). Using the MMMLU benchmark, we evaluate model performance under Very Friendly, Neutral, and Very Rude prompt variants across six tasks spanning STEM and Humanities domains, and analyze pairwise accuracy differences with statistical significance testing. Our results show that tone sensitivity is both model-dependent and domain-specific. Neutral or Very Friendly prompts generally yield higher accuracy than Very Rude prompts, but statistically significant effects appear only in a subset of Humanities tasks, where rude tone reduces accuracy for GPT and Llama, while Gemini remains comparatively tone-insensitive. When performance is aggregated across tasks within each domain, tone effects diminish and largely lose statistical significance. Compared with earlier researches, these findings suggest that dataset scale and coverage materially influence the detection of tone effects. Overall, our study indicates that while interaction tone can matter in specific interpretive settings, modern LLMs are broadly robust to tonal variation in typical mixed-domain use, providing practical guidance for prompt design and model selection in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12801v1">Fine-Grained Energy Prediction For Parallellized LLM Inference With PIE-P</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), energy costs of running LLMs is quickly becoming a critical concern. However, precisely measuring the energy consumption of LLMs is often infeasible because hardware-based power monitors are not always accessible and software-based energy measurement tools are not accurate. While various prediction techniques have been developed to estimate LLM energy consumption, these approaches are limited to single-GPU environments and thus are not applicable to modern LLM inference which is typically parallelized across multiple GPUs. In this work, we remedy this gap and introduce PIE-P, a fine-grained energy prediction framework for multi-GPU inference, including tensor, pipeline, and data parallelism. Predicting the energy under parallelized inference is complicated by the non-determinism in inter-GPU communication, additional communication overheads, and difficulties in isolating energy during the communication/synchronization phase. We develop a scalable prediction framework that addresses these issues via precise sampling, fine-grained modeling of inter-GPU communication, and careful accounting of parallelization overhead. Our evaluation results show that PIE-P yields accurate and fine-grained energy predictions across parallelism strategies, significantly outperforming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24850v2">Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 22 pages, 10 figures. Code available at https://github.com/Tim-Siu/reinforcement-distillation
    </div>
    <details class="paper-abstract">
      Recent advances in model distillation show that data from advanced reasoning models can effectively train smaller student models. However, standard practices discard incorrect reasoning traces -- valuable, yet underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? We employ a two-stage training recipe: first, Supervised Fine-Tuning (SFT) on positive traces, followed by a refinement stage using both positive and negative traces. We find that a simple REINFORCE-style objective, which we term the Reinforcement Distillation (REDI) objective, outperforms established preference optimization methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate the effectiveness of this approach. Notably, our Qwen-REDI-1.5B model, trained on just 131k traces from the open Open-R1 dataset, achieves an 83.1% score on MATH-500. Its performance matches that of DeepSeek-R1-Distill-Qwen-1.5B, a model trained on 800k proprietary data. This result showcases the remarkable data efficiency of utilizing previously discarded negative traces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06029v3">Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 aaai26 camera-ready version, 10 pages
    </div>
    <details class="paper-abstract">
      Generative reasoning with large language models (LLMs) often involves long decoding sequences, leading to substantial memory and latency overheads from accumulating key-value (KV) caches. While existing KV compression methods primarily focus on reducing prefill memory from long input sequences, they fall short in addressing the dynamic and layer-sensitive nature of long-form generation, which is central to reasoning tasks. We propose Lethe, a dynamic KV cache management framework that introduces adaptivity along both the spatial and temporal dimensions of decoding. Along the spatial dimension, Lethe performs layerwise sparsity-aware allocation, assigning token pruning budgets to each transformer layer based on estimated attention redundancy. Along the temporal dimension, Lethe conducts multi-round token pruning during generation, driven by a Recency-Aware Selective Retention} (RASR) mechanism. RASR extends traditional recency-based heuristics by also considering token relevance derived from evolving attention patterns, enabling informed decisions about which tokens to retain or evict. Empirical results demonstrate that Lethe achieves a favorable balance between efficiency and generation quality across diverse models and tasks, increases throughput by up to 2.56x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12706v1">Synergizing Code Coverage and Gameplay Intent: Coverage-Aware Game Playtesting with LLM-Guided Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      The widespread adoption of the "Games as a Service" model necessitates frequent content updates, placing immense pressure on quality assurance. In response, automated game testing has been viewed as a promising solution to cope with this demanding release cadence. However, existing automated testing approaches typically create a dichotomy: code-centric methods focus on structural coverage without understanding gameplay context, while player-centric agents validate high-level intent but often fail to cover specific underlying code changes. To bridge this gap, we propose SMART (Structural Mapping for Augmented Reinforcement Testing), a novel framework that synergizes structural verification and functional validation for game update testing. SMART leverages large language models (LLMs) to interpret abstract syntax tree (AST) differences and extract functional intent, constructing a context-aware hybrid reward mechanism. This mechanism guides reinforcement learning agents to sequentially fulfill gameplay goals while adaptively exploring modified code branches. We evaluate SMART on two environments, Overcooked and Minecraft. The results demonstrate that SMART significantly outperforms state-of-the-art baselines; it achieves over 94% branch coverage of modified code, nearly double that of traditional reinforcement learning methods, while maintaining a 98% task completion rate, effectively balancing structural comprehensiveness with functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12677v1">Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      We explore efficient strategies to fine-tune decoder-only Large Language Models (LLMs) for downstream text classification under resource constraints. Two approaches are investigated: (1) attaching a classification head to a pre-trained causal LLM and fine-tuning on the task (using the LLM's final token embedding as a sequence representation), and (2) instruction-tuning the LLM in a prompt->response format for classification. To enable single-GPU fine-tuning of models up to 8B parameters, we combine 4-bit model quantization with Low-Rank Adaptation (LoRA) for parameter-efficient training. Experiments on two datasets - a proprietary single-label dataset and the public WIPO-Alpha patent dataset (extreme multi-label classification) - show that the embedding-based method significantly outperforms the instruction-tuned method in F1-score, and is very competitive with - even surpassing - fine-tuned domain-specific models (e.g. BERT) on the same tasks. These results demonstrate that directly leveraging the internal representations of causal LLMs, along with efficient fine-tuning techniques, yields impressive classification performance under limited computational resources. We discuss the advantages of each approach while outlining practical guidelines and future directions for optimizing LLM fine-tuning in classification scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11878v4">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13405v2">RealHiTBench: A Comprehensive Realistic Hierarchical Table Benchmark for Evaluating LLM-Based Table Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), there is an increasing need for challenging benchmarks to evaluate their capabilities in handling complex tabular data. However, existing benchmarks are either based on outdated data setups or focus solely on simple, flat table structures. In this paper, we introduce RealHiTBench, a comprehensive benchmark designed to evaluate the performance of both LLMs and Multimodal LLMs (MLLMs) across a variety of input formats for complex tabular data, including LaTeX, HTML, and PNG. RealHiTBench also includes a diverse collection of tables with intricate structures, spanning a wide range of task types. Our experimental results, using 25 state-of-the-art LLMs, demonstrate that RealHiTBench is indeed a challenging benchmark. Moreover, we also develop TreeThinker, a tree-based pipeline that organizes hierarchical headers into a tree structure for enhanced tabular reasoning, validating the importance of improving LLMs' perception of table hierarchies. We hope that our work will inspire further research on tabular data reasoning and the development of more robust models. The code and data are available at https://github.com/cspzyy/RealHiTBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12630v1">ORIBA: Exploring LLM-Driven Role-Play Chatbot as a Creativity Support Tool for Original Character Artists</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Recent advances in Generative AI (GAI) have led to new opportunities for creativity support. However, this technology has raised ethical concerns in the visual artists community. This paper explores how GAI can assist visual artists in developing original characters (OCs) while respecting their creative agency. We present ORIBA, an AI chatbot leveraging large language models (LLMs) to enable artists to role-play with their OCs, focusing on conceptualization (e.g., backstories) while leaving exposition (visual creation) to creators. Through a study with 14 artists, we found ORIBA motivated artists' imaginative engagement, developing multidimensional attributes and stronger bonds with OCs that inspire their creative process. Our contributions include design insights for AI systems that develop from artists' perspectives, demonstrating how LLMs can support cross-modal creativity while preserving creative agency in OC art. This paper highlights the potential of GAI as a neutral, non-visual support that strengthens existing creative practice, without infringing artistic exposition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12620v1">Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-12-14
      | 💬 9 pages, 4 figures, 5 tables. Submitted to AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
    </div>
    <details class="paper-abstract">
      We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12597v1">AgentSHAP: Interpreting LLM Agent Tool Importance with Monte Carlo Shapley Value Estimation</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      LLM agents that use external tools can solve complex tasks, but understanding which tools actually contributed to a response remains a blind spot. No existing XAI methods address tool-level explanations. We introduce AgentSHAP, the first framework for explaining tool importance in LLM agents. AgentSHAP is model-agnostic: it treats the agent as a black box and works with any LLM (GPT, Claude, Llama, etc.) without needing access to internal weights or gradients. Using Monte Carlo Shapley values, AgentSHAP tests how an agent responds with different tool subsets and computes fair importance scores based on game theory. Our contributions are: (1) the first explainability method for agent tool attribution, grounded in Shapley values from game theory; (2) Monte Carlo sampling that reduces cost from O(2n) to practical levels; and (3) comprehensive experiments on API-Bank showing that AgentSHAP produces consistent scores across runs, correctly identifies which tools matter, and distinguishes relevant from irrelevant tools. AgentSHAP joins TokenSHAP (for tokens) and PixelSHAP (for image regions) to complete a family of Shapley-based XAI tools for modern generative AI. Code: https://github.com/GenAISHAP/TokenSHAP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12544v1">HyperEdit: Unlocking Instruction-based Text Editing in LLMs via Hypernetworks</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Instruction-based text editing is increasingly critical for real-world applications such as code editors (e.g., Cursor), but Large Language Models (LLMs) continue to struggle with this task. Unlike free-form generation, editing requires faithfully implementing user instructions while preserving unchanged content, as even minor unintended modifications can break functionality. Existing approaches treat editing as generic text generation, leading to two key failures: they struggle to faithfully align edits with diverse user intents, and they often over-edit unchanged regions. We propose HyperEdit to address both issues. First, we introduce hypernetwork-based dynamic adaptation that generates request-specific parameters, enabling the model to tailor its editing strategy to each instruction. Second, we develop difference-aware regularization that focuses supervision on modified spans, preventing over-editing while ensuring precise, minimal changes. HyperEdit achieves a 9%--30% relative improvement in BLEU on modified regions over state-of-the-art baselines, despite utilizing only 3B parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12536v1">Diverse LLMs vs. Vulnerabilities: Who Detects and Fixes Them Better?</a></div>
    <div class="paper-meta">
      📅 2025-12-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being studied for Software Vulnerability Detection (SVD) and Repair (SVR). Individual LLMs have demonstrated code understanding abilities, but they frequently struggle when identifying complex vulnerabilities and generating fixes. This study presents DVDR-LLM, an ensemble framework that combines outputs from diverse LLMs to determine whether aggregating multiple models reduces error rates. Our evaluation reveals that DVDR-LLM achieves 10-12% higher detection accuracy compared to the average performance of individual models, with benefits increasing as code complexity grows. For multi-file vulnerabilities, the ensemble approach demonstrates significant improvements in recall (+18%) and F1 score (+11.8%) over individual models. However, the approach raises measurable trade-offs: reducing false positives in verification tasks while simultaneously increasing false negatives in detection tasks, requiring careful decision on the required level of agreement among the LLMs (threshold) for increased performance across different security contexts. Artifact: https://github.com/Erroristotle/DVDR_LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12476v1">HetRL: Efficient Reinforcement Learning for LLMs in Heterogeneous Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale and new GPUs are released even more frequently, there is an increasing demand for LLM post-training in heterogeneous environments to fully leverage underutilized mid-range or previous-generation GPUs across regions and alleviate the shortage of homogeneous high-end GPUs within a single region. However, achieving high-performance reinforcement learning (RL) training for LLMs on such computing resources remains challenging because the workflow involves multiple models and tasks with complex computation and data dependencies. In this paper, we present HetRL, a distributed system for efficient RL training in infrastructures with heterogeneous GPUs and networks. HetRL formulates the scheduling of RL training in heterogeneous environments as a constrained joint optimization problem and introduces a novel scheduling algorithm that (1) decomposes the complex search space with a multi-level search framework; and (2) allocates the search budget via successive halving. Our extensive evaluation, consuming 20,000 GPU-hours, shows that HetRL delivers up to 9.17x the throughput of state-of-the-art systems, and 3.17x on average, under various workloads and settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14285v3">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted at the 11th IEEE WIECON-ECE 2025
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00057v2">Incoherence as Oracle-less Measure of Error in LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted at AAAI'26 (extended version). 8 pages + refs and appendix
    </div>
    <details class="paper-abstract">
      Generating code from a natural language programming task is one of the most successful applications of Large Language Models (LLMs). Yet, the generated program may be buggy. Without an oracle, such as an existing, correct implementation or a formal specification, can we somehow estimate how likely the generated program is correct? In this paper, we propose a measure of incorrectness, called *incoherence*, that can be estimated efficiently in the absence of an oracle and allows us to establish a lower bound on the error, i.e., the probability that the LLM-generated program for that specification is incorrect. In our experiments, our incoherence-based methodology can automatically identify about two-thirds of incorrect programs without reports of false positives for the average task. In fact, *an oracle-based evaluation of LLMs can be reliably replaced by an incoherence-based evaluation*. In particular, we find a very strong agreement between the ranking of LLMs by the number of programs deemed correct via an oracle (pass@1) and the ranking of LLMs by the number of programs deemed correct via incoherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17904v2">BreakFun: Jailbreaking LLMs via Schema Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06380v2">Protecting Bystander Privacy via Selective Hearing in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Dataset: https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench
    </div>
    <details class="paper-abstract">
      Audio Large language models (LLMs) are increasingly deployed in the real world, where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences did not consider. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures, including both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. In addition, we propose Selective Efficacy (SE), a novel metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LLMs reveals substantial bystander privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we also present Bystander Privacy Fine-Tuning (BPFT), a novel training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. We show that BPFT yields substantial gains, achieving an absolute 47% higher bystander accuracy under selective mode and an absolute 16% higher SE compared to Gemini 2.5 Pro, which is the best audio LLM without BPFT. Together, SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12313v1">Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 17 pages, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.08697v3">LLMs as Span Annotators: A Comparative Study of LLMs and Humans</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Span annotation - annotating specific text features at the span level - can be used to evaluate texts where single-score metrics fail to provide actionable feedback. Until recently, span annotation was done by human annotators or fine-tuned models. In this paper, we study whether large language models (LLMs) can serve as an alternative to human annotators. We compare the abilities of LLMs to skilled human annotators on three span annotation tasks: evaluating data-to-text generation, identifying translation errors, and detecting propaganda techniques. We show that overall, LLMs have only moderate inter-annotator agreement (IAA) with human annotators. However, we demonstrate that LLMs make errors at a similar rate as skilled crowdworkers. LLMs also produce annotations at a fraction of the cost per output annotation. We release the dataset of over 40k model and human span annotations for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v1">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 14 pages, 20 figures, conference
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12281v1">Cognitive-YOLO: LLM-Driven Architecture Synthesis from First Principles of Data for Object Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 12 pages, 4 figures, 3 ttables
    </div>
    <details class="paper-abstract">
      Designing high-performance object detection architectures is a complex task, where traditional manual design is time-consuming and labor-intensive, and Neural Architecture Search (NAS) is computationally prohibitive. While recent approaches using Large Language Models (LLMs) show promise, they often function as iterative optimizers within a search loop, rather than generating architectures directly from a holistic understanding of the data. To address this gap, we propose Cognitive-YOLO, a novel framework for LLM-driven architecture synthesis that generates network configurations directly from the intrinsic characteristics of the dataset. Our method consists of three stages: first, an analysis module extracts key meta-features (e.g., object scale distribution and scene density) from the target dataset; second, the LLM reasons upon these features, augmented with state-of-the-art components retrieved via Retrieval-Augmented Generation (RAG), to synthesize the architecture into a structured Neural Architecture Description Language (NADL); finally, a compiler instantiates this description into a deployable model. Extensive experiments on five diverse object detection datasets demonstrate that our proposed Cognitive-YOLO consistently generates superior architectures, achieving highly competitive performance and demonstrating a superior performance-per-parameter trade-off compared to strong baseline models across multiple benchmarks. Crucially, our ablation studies prove that the LLM's data-driven reasoning is the primary driver of performance, demonstrating that a deep understanding of data "first principles" is more critical for achieving a superior architecture than simply retrieving SOTA components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14479v5">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12224v1">Cluster-guided LLM-Based Anonymization of Software Analytics Data: Studying Privacy-Utility Trade-offs in JIT Defect Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      The increasing use of machine learning (ML) for Just-In-Time (JIT) defect prediction raises concerns about privacy leakage from software analytics data. Existing anonymization methods, such as tabular transformations and graph perturbations, often overlook contextual dependencies among software metrics, leading to suboptimal privacy-utility tradeoffs. Leveraging the contextual reasoning of Large Language Models (LLMs), we propose a cluster-guided anonymization technique that preserves contextual and statistical relationships within JIT datasets. Our method groups commits into feature-based clusters and employs an LLM to generate context-aware parameter configurations for each commit cluster, defining alpha-beta ratios and churn mixture distributions used for anonymization. Our evaluation on six projects (Cassandra, Flink, Groovy, Ignite, OpenStack, and Qt) shows that our LLM-based approach achieves privacy level 2 (IPR >= 80 percent), improving privacy by 18 to 25 percent over four state-of-the-art graph-based anonymization baselines while maintaining comparable F1 scores. Our results demonstrate that LLMs can act as adaptive anonymization engines when provided with cluster-specific statistical information about similar data points, enabling context-sensitive and privacy-preserving software analytics without compromising predictive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12177v1">Floorplan2Guide: LLM-Guided Floorplan Parsing for BLV Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-12-13
      | 💬 Accepted for publication in the proceedings of the IEEE International Conference on Big Data (IEEE BigData 2025)
    </div>
    <details class="paper-abstract">
      Indoor navigation remains a critical challenge for people with visual impairments. The current solutions mainly rely on infrastructure-based systems, which limit their ability to navigate safely in dynamic environments. We propose a novel navigation approach that utilizes a foundation model to transform floor plans into navigable knowledge graphs and generate human-readable navigation instructions. Floorplan2Guide integrates a large language model (LLM) to extract spatial information from architectural layouts, reducing the manual preprocessing required by earlier floorplan parsing methods. Experimental results indicate that few-shot learning improves navigation accuracy in comparison to zero-shot learning on simulated and real-world evaluations. Claude 3.7 Sonnet achieves the highest accuracy among the evaluated models, with 92.31%, 76.92%, and 61.54% on the short, medium, and long routes, respectively, under 5-shot prompting of the MP-1 floor plan. The success rate of graph-based spatial structure is 15.4% higher than that of direct visual reasoning among all models, which confirms that graphical representation and in-context learning enhance navigation performance and make our solution more precise for indoor navigation of Blind and Low Vision (BLV) users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12154v1">Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12117v1">Citation-Grounded Code Comprehension: Preventing LLM Hallucination Through Hybrid Retrieval and Graph-Augmented Context</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Large language models have become essential tools for code comprehension, enabling developers to query unfamiliar codebases through natural language interfaces. However, LLM hallucination, generating plausible but factually incorrect citations to source code, remains a critical barrier to reliable developer assistance. This paper addresses the challenges of achieving verifiable, citation grounded code comprehension through hybrid retrieval and lightweight structural reasoning. Our work is grounded in systematic evaluation across 30 Python repositories with 180 developer queries, comparing retrieval modalities, graph expansion strategies, and citation verification mechanisms. We find that challenges of citation accuracy arise from the interplay between sparse lexical matching, dense semantic similarity, and cross file architectural dependencies. Among these, cross file evidence discovery is the largest contributor to citation completeness, but it is largely overlooked because existing systems rely on pure textual similarity without leveraging code structure. We advocate for citation grounded generation as an architectural principle for code comprehension systems and demonstrate this need by achieving 92 percent citation accuracy with zero hallucinations. Specifically, we develop a hybrid retrieval system combining BM25 sparse matching, BGE dense embeddings, and Neo4j graph expansion via import relationships, which outperforms single mode baselines by 14 to 18 percentage points while discovering cross file evidence missed by pure text similarity in 62 percent of architectural queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17833v2">Learning to Debug: LLM-Organized Knowledge Trees for Solving RTL Assertion Failures</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      Debugging is the dominant cost in modern hardware verification, where assertion failures are among the most frequent and expensive to resolve. While Large Language Models (LLMs) show promise, they often fail to capture the precise, reusable expertise that engineers apply, leading to inaccurate responses. We propose GROVE, a hierarchical knowledge management framework that learns and organizes reusable debugging expertise into an LLM-organized knowledge tree for solving assertion failures. GROVE distills debugging knowledge from prior cases and organizes it into a vertical tree of configurable depth, with each node encoding a concise knowledge item and explicit applicability conditions. During training, GROVE uses a parallel, gradient-free loop where an LLM proposes tree modifications as structured JSON edits by learning from the cases. At test time, a budget-aware iterative zoom is performed to navigate the tree, retrieving a small set of applicable knowledge items that guide a base LLM's hypothesis generation and fix proposals. Evaluated on a suite of assertion-failure cases, GROVE delivers consistent gains in pass@1 and pass@5, demonstrating the value of structured knowledge evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.11641v4">Revolutionizing Finance with LLMs: An Overview of Applications and Insights</a></div>
    <div class="paper-meta">
      📅 2025-12-13
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) like ChatGPT have seen considerable advancements and have been applied in diverse fields. Built on the Transformer architecture, these models are trained on extensive datasets, enabling them to understand and generate human language effectively. In the financial domain, the deployment of LLMs is gaining momentum. These models are being utilized for automating financial report generation, forecasting market trends, analyzing investor sentiment, and offering personalized financial advice. Leveraging their natural language processing capabilities, LLMs can distill key insights from vast financial data, aiding institutions in making informed investment choices and enhancing both operational efficiency and customer satisfaction. In this study, we provide a comprehensive overview of the emerging integration of LLMs into various financial tasks. Additionally, we conducted holistic tests on multiple financial tasks through the combination of natural language instructions. Our findings show that GPT-4 effectively follow prompt instructions across various financial tasks. This survey and evaluation of LLMs in the financial domain aim to deepen the understanding of LLMs' current role in finance for both financial practitioners and LLM researchers, identify new research and application prospects, and highlight how these technologies can be leveraged to solve practical challenges in the finance industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11755v1">SUMFORU: An LLM-Based Review Summarization Framework for Personalized Purchase Decision Support</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Code available at https://github.com/Harry20030331/SumForU
    </div>
    <details class="paper-abstract">
      Online product reviews contain rich but noisy signals that overwhelm users and hinder effective decision-making. Existing LLM-based summarizers remain generic and fail to account for individual preferences, limiting their practical utility. We propose SUMFORU, a steerable review summarization framework that aligns outputs with explicit user personas to support personalized purchase decisions. Our approach integrates a high-quality data pipeline built from the Amazon 2023 Review Dataset with a two-stage alignment procedure: (1) persona-aware Supervised Fine-Tuning (SFT) via asymmetric knowledge distillation, and (2) Reinforcement Learning with AI Feedback (RLAIF) using a preference estimator to capture fine-grained, persona-relevant signals. We evaluate the model across rule-based, LLM-based, and human-centered metrics, demonstrating consistent improvements in consistency, grounding, and preference alignment. Our framework achieves the highest performance across all evaluation settings and generalizes effectively to unseen product categories. Our results highlight the promise of steerable pluralistic alignment for building next-generation personalized decision-support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01374v4">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed Reasoning Compiler) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
