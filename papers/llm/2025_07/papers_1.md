# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08450v2">IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v2">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanations</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22050v2">DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 22 pages, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at https://github.com/MinghoKwok/DeepSieve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22758v1">MASCA: LLM based-Multi Agents System for Credit Assessment</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Accepted at ACL REALM Workshop. Work in Progress
    </div>
    <details class="paper-abstract">
      Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22753v1">Opportunities and Challenges of LLMs in Education: An NLP Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22716v1">From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22619v1">Enhancing Manufacturing Knowledge Access with LLMs and Context-aware Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 European Conference on Artificial Intelligence (ECAI) 2024
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs) have transformed data management within the manufacturing industry, offering effective means for integrating disparate data sources through shared and structured conceptual schemas. However, harnessing the power of KGs can be daunting for non-experts, as it often requires formulating complex SPARQL queries to retrieve specific information. With the advent of Large Language Models (LLMs), there is a growing potential to automatically translate natural language queries into the SPARQL format, thus bridging the gap between user-friendly interfaces and the sophisticated architecture of KGs. The challenge remains in adequately informing LLMs about the relevant context and structure of domain-specific KGs, e.g., in manufacturing, to improve the accuracy of generated queries. In this paper, we evaluate multiple strategies that use LLMs as mediators to facilitate information retrieval from KGs. We focus on the manufacturing domain, particularly on the Bosch Line Information System KG and the I40 Core Information Model. In our evaluation, we compare various approaches for feeding relevant context from the KG to the LLM and analyze their proficiency in transforming real-world questions into SPARQL queries. Our findings show that LLMs can significantly improve their performance on generating correct and complete queries when provided only the adequate context of the KG schema. Such context-aware prompting techniques help LLMs to focus on the relevant parts of the ontology and reduce the risk of hallucination. We anticipate that the proposed techniques help LLMs to democratize access to complex data repositories and empower informed decision-making in manufacturing settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22565v1">Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22564v1">Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22467v1">Towards Simulating Social Influence Dynamics with LLM-based Multi-agents</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models offer promising capabilities to simulate complex human social interactions. We investigate whether LLM-based multi-agent simulations can reproduce core human social dynamics observed in online forums. We evaluate conformity dynamics, group polarization, and fragmentation across different model scales and reasoning capabilities using a structured simulation framework. Our findings indicate that smaller models exhibit higher conformity rates, whereas models optimized for reasoning are more resistant to social influence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09213v3">FineMedLM-o1: Enhancing Medical Knowledge Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the deep reasoning required for complex medical problems, such as differential diagnosis and medication recommendations. We propose FineMedLM-o1, which leverages high-quality medical synthetic data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduce Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also propose a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22447v1">Breaking Obfuscation: Cluster-Aware Graph with LLM-Aided Recovery for Malicious JavaScript Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      With the rapid expansion of web-based applications and cloud services, malicious JavaScript code continues to pose significant threats to user privacy, system integrity, and enterprise security. But, detecting such threats remains challenging due to sophisticated code obfuscation techniques and JavaScript's inherent language characteristics, particularly its nested closure structures and syntactic flexibility. In this work, we propose DeCoda, a hybrid defense framework that combines large language model (LLM)-based deobfuscation with code graph learning: (1) We first construct a sophisticated prompt-learning pipeline with multi-stage refinement, where the LLM progressively reconstructs the original code structure from obfuscated inputs and then generates normalized Abstract Syntax Tree (AST) representations; (2) In JavaScript ASTs, dynamic typing scatters semantically similar nodes while deeply nested functions fracture scope capturing, introducing structural noise and semantic ambiguity. To address these challenges, we then propose to learn hierarchical code graph representations via a Cluster-wise Graph that synergistically integrates graph transformer network, node clustering, and node-to-cluster attention to simultaneously capture both local node-level semantics and global cluster-induced structural relationships from AST graph. Experimental results demonstrate that our method achieves F1-scores of 94.64% and 97.71% on two benchmark datasets, demonstrating absolute improvements of 10.74% and 13.85% over state-of-the-art baselines. In false-positive control evaluation at fixed FPR levels (0.0001, 0.001, 0.01), our approach delivers 4.82, 5.91, and 2.53 higher TPR respectively compared to the best-performing baseline. These results highlight the effectiveness of LLM-based deobfuscation and underscore the importance of modeling cluster-level relationships in detecting malicious code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15790v3">ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 4 pages, 1 figure. To appear in Proceedings of the 16th Asia-Pacific Symposium on Internetware (Internetware 2025), ACM ICPS. DOI: https://doi.org/10.1145/3755881.3755934
    </div>
    <details class="paper-abstract">
      With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21990v2">ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 13 figures, 4 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoner LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized knowledge points to enhance the model's understanding of the fundamental principles and logical structure of chemistry. Then, we propose a mix-sourced distillation strategy that integrates expert-curated knowledge with general-domain reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the reliability, transparency, and practical utility of the model in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22387v1">PATENTWRITER: A Benchmarking Study for Patent Drafting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as transformative approaches in several important fields. This paper aims for a paradigm shift for patent writing by leveraging LLMs to overcome the tedious patent-filing process. In this work, we present PATENTWRITER, the first unified benchmarking framework for evaluating LLMs in patent abstract generation. Given the first claim of a patent, we evaluate six leading LLMs -- including GPT-4 and LLaMA-3 -- under a consistent setup spanning zero-shot, few-shot, and chain-of-thought prompting strategies to generate the abstract of the patent. Our benchmark PATENTWRITER goes beyond surface-level evaluation: we systematically assess the output quality using a comprehensive suite of metrics -- standard NLP measures (e.g., BLEU, ROUGE, BERTScore), robustness under three types of input perturbations, and applicability in two downstream patent classification and retrieval tasks. We also conduct stylistic analysis to assess length, readability, and tone. Experimental results show that modern LLMs can generate high-fidelity and stylistically appropriate patent abstracts, often surpassing domain-specific baselines. Our code and dataset are open-sourced to support reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22367v1">Traits Run Deep: Enhancing Personality Assessment via Psychology-Guided LLM Representations and Multimodal Apparent Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 8 pages, 3 figures, ACM MM 2025
    </div>
    <details class="paper-abstract">
      Accurate and reliable personality assessment plays a vital role in many fields, such as emotional intelligence, mental health diagnostics, and personalized education. Unlike fleeting emotions, personality traits are stable, often subconsciously leaked through language, facial expressions, and body behaviors, with asynchronous patterns across modalities. It was hard to model personality semantics with traditional superficial features and seemed impossible to achieve effective cross-modal understanding. To address these challenges, we propose a novel personality assessment framework called \textit{\textbf{Traits Run Deep}}. It employs \textit{\textbf{psychology-informed prompts}} to elicit high-level personality-relevant semantic representations. Besides, it devises a \textit{\textbf{Text-Centric Trait Fusion Network}} that anchors rich text semantics to align and integrate asynchronous signals from other modalities. To be specific, such fusion module includes a Chunk-Wise Projector to decrease dimensionality, a Cross-Modal Connector and a Text Feature Enhancer for effective modality fusion and an ensemble regression head to improve generalization in data-scarce situations. To our knowledge, we are the first to apply personality-specific prompts to guide large language models (LLMs) in extracting personality-aware semantics for improved representation quality. Furthermore, extracting and fusing audio-visual apparent behavior features further improves the accuracy. Experimental results on the AVI validation set have demonstrated the effectiveness of the proposed components, i.e., approximately a 45\% reduction in mean squared error (MSE). Final evaluations on the test set of the AVI Challenge 2025 confirm our method's superiority, ranking first in the Personality Assessment track. The source code will be made available at https://github.com/MSA-LMC/TraitsRunDeep.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20300v2">Talking-to-Build: How LLM-Assisted Interface Shapes Player Performance and Experience in Minecraft</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      With large language models (LLMs) on the rise, in-game interactions are shifting from rigid commands to natural conversations. However, the impacts of LLMs on player performance and game experience remain underexplored. This work explores LLM's role as a co-builder during gameplay, examining its impact on task performance, usability, and player experience. Using Minecraft as a sandbox, we present an LLM-assisted interface that engages players through natural language, aiming to facilitate creativity and simplify complex gaming commands. We conducted a mixed-methods study with 30 participants, comparing LLM-assisted and command-based interfaces across simple and complex game tasks. Quantitative and qualitative analyses reveal that the LLM-assisted interface significantly improves player performance, engagement, and overall game experience. Additionally, task complexity has a notable effect on player performance and experience across both interfaces. Our findings highlight the potential of LLM-assisted interfaces to revolutionize virtual experiences, emphasizing the importance of balancing intuitiveness with predictability, transparency, and user agency in AI-driven, multimodal gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22352v1">Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 15 pages, 8 figures. Published at the 7th ACM Conference on Conversational User Interfaces (CUI '25), July 8-10, 2025, Waterloo, Canada. Open-source code available at https://github.com/ISUE/iva-cui
    </div>
    <details class="paper-abstract">
      We investigated the challenges of mitigating response delays in free-form conversations with virtual agents powered by Large Language Models (LLMs) within Virtual Reality (VR). For this, we used conversational fillers, such as gestures and verbal cues, to bridge delays between user input and system responses and evaluate their effectiveness across various latency levels and interaction scenarios. We found that latency above 4 seconds degrades quality of experience, while natural conversational fillers improve perceived response time, especially in high-delay conditions. Our findings provide insights for practitioners and researchers to optimize user engagement whenever conversational systems' responses are delayed by network limitations or slow hardware. We also contribute an open-source pipeline that streamlines deploying conversational agents in virtual environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01282v2">Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-07-30
      | 💬 accepted in COLM2025, 9 pages
    </div>
    <details class="paper-abstract">
      While the inconsistency of LLMs is not a novel topic, prior research has predominantly addressed two types of generative inconsistencies: i) Randomness Inconsistency: running the same LLM multiple trials, yielding varying responses; ii) Paraphrase Inconsistency: paraphrased prompts result in different responses from the same LLM. Randomness Inconsistency arises from the inherent randomness due to stochastic sampling in generative models, while Paraphrase Inconsistency is a consequence of the language modeling objectives, where paraphrased prompts alter the distribution of vocabulary logits. This research discovers Prompt-Reverse Inconsistency (PRIN), a new form of LLM self-inconsistency: given a question and a couple of LLM-generated answer candidates, the LLM often has conflicting responses when prompted "Which are correct answers?" and "Which are incorrect answers?". PRIN poses a big concern as it undermines the credibility of LLM-as-a-judge, and suggests a challenge for LLMs to adhere to basic logical rules. We conduct a series of experiments to investigate PRIN, examining the extent of PRIN across different LLMs, methods to mitigate it, potential applications, and its relationship with Randomness Inconsistency and Paraphrase Inconsistency. As the first study to explore PRIN, our findings offer valuable insights into the inner workings of LLMs and contribute to advancing trustworthy AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22326v1">An Explainable Emotion Alignment Framework for LLM-Empowered Agent in Metaverse Service Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-30
    </div>
    <details class="paper-abstract">
      Metaverse service is a product of the convergence between Metaverse and service systems, designed to address service-related challenges concerning digital avatars, digital twins, and digital natives within Metaverse. With the rise of large language models (LLMs), agents now play a pivotal role in Metaverse service ecosystem, serving dual functions: as digital avatars representing users in the virtual realm and as service assistants (or NPCs) providing personalized support. However, during the modeling of Metaverse service ecosystems, existing LLM-based agents face significant challenges in bridging virtual-world services with real-world services, particularly regarding issues such as character data fusion, character knowledge association, and ethical safety concerns. This paper proposes an explainable emotion alignment framework for LLM-based agents in Metaverse Service Ecosystem. It aims to integrate factual factors into the decision-making loop of LLM-based agents, systematically demonstrating how to achieve more relational fact alignment for these agents. Finally, a simulation experiment in the Offline-to-Offline food delivery scenario is conducted to evaluate the effectiveness of this framework, obtaining more realistic social emergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22050v1">DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 22 pages, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22048v1">Composable Effect Handling for Programming LLM-integrated Scripts</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Implementing LLM-integrated scripts introduces challenges in modularity and performance, as scripts are often coupled to specific LLM implementations and fail to exploit parallelization opportunities. This paper proposes using composable effect handling to separate workflow logic from effectful operations, such as LLM calls, I/O, and concurrency, enabling modularity without sacrificing the opportunity for performance optimization. By treating these operations as abstract interfaces and discharging them via effect handlers, this paper shows that scripts can achieve significant speedups (e.g., 10$\times$ in a Tree-of-Thoughts case study) without compromising modularity. This paper aims to promote composable effect handling as a programming style for LLM scripting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20527v2">SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21990v1">ChemDFM-R: An Chemical Reasoner LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 13 figures, 4 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoner LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized knowledge points to enhance the model's understanding of the fundamental principles and logical structure of chemistry. Then, we propose a mix-sourced distillation strategy that integrates expert-curated knowledge with general-domain reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves state-of-the-art performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the reliability, transparency, and practical utility of the model in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21969v1">Towards Cognitive Synergy in LLM-Based Multi-Agent Systems: Integrating Theory of Mind and Critical Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted at CogSci 2025
    </div>
    <details class="paper-abstract">
      Recently, the field of Multi-Agent Systems (MAS) has gained popularity as researchers are trying to develop artificial intelligence capable of efficient collective reasoning. Agents based on Large Language Models (LLMs) perform well in isolated tasks, yet struggle with higher-order cognition required for adaptive collaboration. Human teams achieve synergy not only through knowledge sharing, but also through recursive reasoning, structured critique, and the ability to infer others' mental states. Current artificial systems lack these essential mechanisms, limiting their ability to engage in sophisticated collective reasoning. This work explores cognitive processes that enable effective collaboration, focusing on adaptive theory of mind (ToM) and systematic critical evaluation. We investigate three key questions. First, how does the ability to model others' perspectives enhance coordination and reduce redundant reasoning? Second, to what extent does structured critique improve reasoning quality by identifying logical gaps and mitigating biases? Third, the interplay of these mechanisms can lead to emergent cognitive synergy, where the collective intelligence of the system exceeds the sum of its parts. Through an empirical case study on complex decision making, we show that the integration of these cognitive mechanisms leads to more coherent, adaptive, and rigorous agent interactions. This article contributes to the field of cognitive science and AI research by presenting a structured framework that emulates human-like collaborative reasoning MAS. It highlights the significance of dynamic ToM and critical evaluation in advancing multi-agent systems' ability to tackle complex, real-world challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19980v2">Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21914v1">Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Rote learning is a memorization technique based on repetition. It is commonly believed to hinder generalization by encouraging verbatim memorization rather than deeper understanding. This insight holds for even learning factual knowledge that inevitably requires a certain degree of memorization. In this work, we demonstrate that LLMs can be trained to generalize from rote memorized data. We introduce a two-phase memorize-then-generalize framework, where the model first rote memorizes factual subject-object associations using a semantically meaningless token and then learns to generalize by fine-tuning on a small set of semantically meaningful prompts. Extensive experiments over 8 LLMs show that the models can reinterpret rote memorized data through the semantically meaningful prompts, as evidenced by the emergence of structured, semantically aligned latent representations between the two. This surprising finding opens the door to both effective and efficient knowledge injection and possible risks of repurposing the memorized data for malicious usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05413v2">SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 6 pages, 3 figures, 5 tables. Accepted to IEEE SMC 2025 conference proceedings
    </div>
    <details class="paper-abstract">
      We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22296v5">Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Supercedes arXiv:2407.00236v1. arXiv admin note: text overlap with arXiv:2407.00236
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown promise in biomolecule optimization problems, they incur heavy computational costs and struggle to satisfy precise constraints. On the other hand, specialized solvers like LaMBO-2 offer efficiency and fine-grained control but require more domain expertise. Comparing these approaches is challenging due to expensive laboratory validation and inadequate synthetic benchmarks. We address this by introducing Ehrlich functions, a synthetic test suite that captures the geometric structure of biophysical sequence optimization problems. With prompting alone, off-the-shelf LLMs struggle to optimize Ehrlich functions. In response, we propose LLOME (Language Model Optimization with Margin Expectation), a bilevel optimization routine for online black-box optimization. When combined with a novel preference learning loss, we find LLOME can not only learn to solve some Ehrlich functions, but can even perform as well as or better than LaMBO-2 on moderately difficult Ehrlich variants. However, LLMs also exhibit some likelihood-reward miscalibration and struggle without explicit rewards. Our results indicate LLMs can occasionally provide significant benefits, but specialized solvers are still competitive and incur less overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21900v1">Leveraging LLMs for Persona-Based Visualization of Election Data</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Visualizations are essential tools for disseminating information regarding elections and their outcomes, potentially influencing public perceptions. Personas, delineating distinctive segments within the populace, furnish a valuable framework for comprehending the nuanced perspectives, requisites, and behaviors of diverse voter demographics. In this work, we propose making visualizations tailored to these personas to make election information easier to understand and more relevant. Using data from UK parliamentary elections and new developments in Large Language Models (LLMs), we create personas that encompass the diverse demographics, technological preferences, voting tendencies, and information consumption patterns observed among voters.Subsequently, we elucidate how these personas can inform the design of visualizations through specific design criteria. We then provide illustrative examples of visualization prototypes based on these criteria and evaluate these prototypes using these personas and LLMs. We finally propose some actionable insights based upon the framework and the different design artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21899v1">LLM-based Content Classification Approach for GitHub Repositories by the README Files</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 8 pages, 4 Figures
    </div>
    <details class="paper-abstract">
      GitHub is the world's most popular platform for storing, sharing, and managing code. Every GitHub repository has a README file associated with it. The README files should contain project-related information as per the recommendations of GitHub to support the usage and improvement of repositories. However, GitHub repository owners sometimes neglected these recommendations. This prevents a GitHub repository from reaching its full potential. This research posits that the comprehensiveness of a GitHub repository's README file significantly influences its adoption and utilization, with a lack of detail potentially hindering its full potential for widespread engagement and impact within the research community. Large Language Models (LLMs) have shown great performance in many text-based tasks including text classification, text generation, text summarization and text translation. In this study, an approach is developed to fine-tune LLMs for automatically classifying different sections of GitHub README files. Three encoder-only LLMs are utilized, including BERT, DistilBERT and RoBERTa. These pre-trained models are then fine-tuned based on a gold-standard dataset consisting of 4226 README file sections. This approach outperforms current state-of-the-art methods and has achieved an overall F1 score of 0.98. Moreover, we have also investigated the use of Parameter-Efficient Fine-Tuning (PEFT) techniques like Low-Rank Adaptation (LoRA) and shown an economical alternative to full fine-tuning without compromising much performance. The results demonstrate the potential of using LLMs in designing an automatic classifier for categorizing the content of GitHub README files. Consequently, this study contributes to the development of automated tools for GitHub repositories to improve their identifications and potential usages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10774v2">Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Time series forecasting is important for applications spanning energy markets, climate analysis, and traffic management. However, existing methods struggle to effectively integrate exogenous texts and align them with the probabilistic nature of large language models (LLMs). Current approaches either employ shallow text-time series fusion via basic prompts or rely on deterministic numerical decoding that conflict with LLMs' token-generation paradigm, which limits contextual awareness and distribution modeling. To address these limitations, we propose CAPTime, a context-aware probabilistic multimodal time series forecasting method that leverages text-informed abstraction and autoregressive LLM decoding. Our method first encodes temporal patterns using a pretrained time series encoder, then aligns them with textual contexts via learnable interactions to produce joint multimodal representations. By combining a mixture of distribution experts with frozen LLMs, we enable context-aware probabilistic forecasting while preserving LLMs' inherent distribution modeling capabilities. Experiments on diverse time series forecasting tasks demonstrate the superior accuracy and generalization of CAPTime, particularly in multimodal scenarios. Additional analysis highlights its robustness in data-scarce scenarios through hybrid probabilistic decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v3">WakenLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label Unknown in reasoning tasks, where two scenarios may appear: (i) an input sample is genuinely unverifiable, but the model cannot understand why; and (ii) a verifiable problem that the model fails to solve, thus outputs Unknown. We refer to these cases collectively as the Vague Perception phenomenon. Current evaluations focus on whether such answers are honest, rather than analyzing the limits of LLM reasoning. To address this, we introduce WakenLLM, a framework that quantifies the portion of Unknown output attributable to model incapacity and evaluates whether stimulation can convert them into either correct answers (verifiable) or justified (unverifiable) responses with valid reasoning. Our method offers a clearer picture of the limits of LLM reasoning and the potential for corrections across various datasets. Comprehensive experiments on six LLMs suggest that, without any training or parameter revision, LLMs can achieve up to a 68.53% accuracy improvement on Vague Perception samples through guided understanding. Our work reveals that current baseline methods only activate a small portion of LLMs' reasoning potential, indicating considerable unexplored capacity. This extends the theoretical upper bounds of reasoning accuracy in LLMs. Consequently, this study deepens our understanding of the latent reasoning capacity of LLMs and offers a new perspective on addressing the Vague Perception phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05629v2">Enhancing Student Learning with LLM-Generated Retrieval Practice Questions: An Empirical Study in Data Science Courses</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Retrieval practice is a well-established pedagogical technique known to significantly enhance student learning and knowledge retention. However, generating high-quality retrieval practice questions is often time-consuming and labor intensive for instructors, especially in rapidly evolving technical subjects. Large Language Models (LLMs) offer the potential to automate this process by generating questions in response to prompts, yet the effectiveness of LLM-generated retrieval practice on student learning remains to be established. In this study, we conducted an empirical study involving two college-level data science courses, with approximately 60 students. We compared learning outcomes during one week in which students received LLM-generated multiple-choice retrieval practice questions to those from a week in which no such questions were provided. Results indicate that students exposed to LLM-generated retrieval practice achieved significantly higher knowledge retention, with an average accuracy of 89%, compared to 73% in the week without such practice. These findings suggest that LLM-generated retrieval questions can effectively support student learning and may provide a scalable solution for integrating retrieval practice into real-time teaching. However, despite these encouraging outcomes and the potential time-saving benefits, cautions must be taken, as the quality of LLM-generated questions can vary. Instructors must still manually verify and revise the generated questions before releasing them to students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21831v1">Introducing HALC: A general pipeline for finding optimal prompting strategies for automated coding with LLMs in the computational social sciences</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 48 pages, 9 figures and 8 tables
    </div>
    <details class="paper-abstract">
      LLMs are seeing widespread use for task automation, including automated coding in the social sciences. However, even though researchers have proposed different prompting strategies, their effectiveness varies across LLMs and tasks. Often trial and error practices are still widespread. We propose HALC$-$a general pipeline that allows for the systematic and reliable construction of optimal prompts for any given coding task and model, permitting the integration of any prompting strategy deemed relevant. To investigate LLM coding and validate our pipeline, we sent a total of 1,512 individual prompts to our local LLMs in over two million requests. We test prompting strategies and LLM task performance based on few expert codings (ground truth). When compared to these expert codings, we find prompts that code reliably for single variables (${\alpha}$climate = .76; ${\alpha}$movement = .78) and across two variables (${\alpha}$climate = .71; ${\alpha}$movement = .74) using the LLM Mistral NeMo. Our prompting strategies are set up in a way that aligns the LLM to our codebook$-$we are not optimizing our codebook for LLM friendliness. Our paper provides insights into the effectiveness of different prompting strategies, crucial influencing factors, and the identification of reliable prompts for each coding task and model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21820v1">Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21817v1">Out of Distribution, Out of Luck: How Well Can LLMs Trained on Vulnerability Datasets Detect Top 25 CWE Weaknesses?</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Automated vulnerability detection research has made substantial progress, yet its real-world impact remains limited. Current vulnerability datasets suffer from issues including label inaccuracy rates of 20-71%, extensive duplication, and poor coverage of critical CWE types. These issues create a significant "generalization gap" where models achieve misleading self-testing performance (measured on held-out data from same dataset for training) by exploiting spurious correlations rather than learning true vulnerability patterns. Our analysis reveals that many models experience substantial performance drops of up to 40.6% when evaluated on independent data, sometimes underperforming random guessing. To address these limitations, we present a three-part solution. First, we introduce a manually curated test dataset, BenchVul, covering the MITRE Top 25 Most Dangerous CWEs. Second, we construct a high-quality training dataset, TitanVul, comprising 35,045 functions by aggregating seven public sources and applying deduplication and validation using a novel multi-agent LLM framework. Third, we propose a Realistic Vulnerability Generation (RVG) framework, which synthesizes context-aware vulnerability examples for underrepresented but critical CWE types through simulated development workflows. Our evaluation shows the strengths of each component in closing the generalization gap. First, BenchVul shows the limitations of self-testing: models trained on existing datasets, such as BigVul and PrimeVul, experience performance drops on BenchVul (from 0.776 to 0.519 and from 0.567 to 0.337). Second, training models on TitanVul demonstrates improved generalization, with model performance increasing from 0.584 when evaluated on the same dataset to 0.767 when tested on BenchVul. Third, supplementing TitanVul with RVG-generated data yields further gains, increasing model performance by 14.0% to 0.874.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21815v1">HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 15 pages, 5 figures, 12 tables, a dataset
    </div>
    <details class="paper-abstract">
      Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21778v1">AU-LLM: Micro-Expression Action Unit Detection via Enhanced LLM-Based Feature Fusion</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The detection of micro-expression Action Units (AUs) is a formidable challenge in affective computing, pivotal for decoding subtle, involuntary human emotions. While Large Language Models (LLMs) demonstrate profound reasoning abilities, their application to the fine-grained, low-intensity domain of micro-expression AU detection remains unexplored. This paper pioneers this direction by introducing \textbf{AU-LLM}, a novel framework that for the first time uses LLM to detect AUs in micro-expression datasets with subtle intensities and the scarcity of data. We specifically address the critical vision-language semantic gap, the \textbf{Enhanced Fusion Projector (EFP)}. The EFP employs a Multi-Layer Perceptron (MLP) to intelligently fuse mid-level (local texture) and high-level (global semantics) visual features from a specialized 3D-CNN backbone into a single, information-dense token. This compact representation effectively empowers the LLM to perform nuanced reasoning over subtle facial muscle movements.Through extensive evaluations on the benchmark CASME II and SAMM datasets, including stringent Leave-One-Subject-Out (LOSO) and cross-domain protocols, AU-LLM establishes a new state-of-the-art, validating the significant potential and robustness of LLM-based reasoning for micro-expression analysis. The codes are available at https://github.com/ZS-liu-JLU/AU-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10541v2">Multi-Modal Hypergraph Enhanced LLM Learning for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages, 4 figures, submitted to IEEE Transactions on Knowledge and Data Engineering
    </div>
    <details class="paper-abstract">
      The burgeoning presence of Large Language Models (LLM) is propelling the development of personalized recommender systems. Most existing LLM-based methods fail to sufficiently explore the multi-view graph structure correlations inherent in recommendation scenarios. To this end, we propose a novel framework, Hypergraph Enhanced LLM Learning for multimodal Recommendation (HeLLM), designed to equip LLMs with the capability to capture intricate higher-order semantic correlations by fusing graph-level contextual signals with sequence-level behavioral patterns. In the recommender pre-training phase, we design a user hypergraph to uncover shared interest preferences among users and an item hypergraph to capture correlations within multimodal similarities among items. The hypergraph convolution and synergistic contrastive learning mechanism are introduced to enhance the distinguishability of learned representations. In the LLM fine-tuning phase, we inject the learned graph-structured embeddings directly into the LLM's architecture and integrate sequential features capturing each user's chronological behavior. This process enables hypergraphs to leverage graph-structured information as global context, enhancing the LLM's ability to perceive complex relational patterns and integrate multimodal information, while also modeling local temporal dynamics. Extensive experiments demonstrate the superiority of our proposed method over state-of-the-art baselines, confirming the advantages of fusing hypergraph-based context with sequential user behavior in LLMs for recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00378v2">iPanda: An LLM-based Agent for Automated Conformance Testing of Communication Protocols</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Conformance testing is essential for ensuring that protocol implementations comply with their specifications. However, traditional testing approaches involve manually creating numerous test cases and scripts, making the process labor-intensive and inefficient. Recently, Large Language Models (LLMs) have demonstrated impressive text comprehension and code generation abilities, providing promising opportunities for automation. In this paper, we propose iPanda, the first framework that leverages LLMs to automate protocol conformance testing. Given a protocol specification document and its implementation, iPanda first employs a keyword-based method to automatically generate comprehensive test cases. Then, it utilizes retrieval-augmented generation and customized CoT strategy to effectively interpret the implementation and produce executable test programs. To further enhance programs' quality, iPanda incorporates an iterative optimization mechanism to refine generated test scripts interactively. Finally, by executing and analyzing the generated tests, iPanda systematically verifies compliance between implementations and protocol specifications. Comprehensive experiments on various protocols show that iPanda significantly outperforms pure LLM-based approaches, improving the success rate (Pass@1) of test-program generation by factors ranging from 4.675 times to 10.751 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21653v1">DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Real-world fraud detection applications benefit from graph learning techniques that jointly exploit node features, often rich in textual data, and graph structural information. Recently, Graph-Enhanced LLMs emerge as a promising graph learning approach that converts graph information into prompts, exploiting LLMs' ability to reason over both textual and structural information. Among them, text-only prompting, which converts graph information to prompts consisting solely of text tokens, offers a solution that relies only on LLM tuning without requiring additional graph-specific encoders. However, text-only prompting struggles on heterogeneous fraud-detection graphs: multi-hop relations expand exponentially with each additional hop, leading to rapidly growing neighborhoods associated with dense textual information. These neighborhoods may overwhelm the model with long, irrelevant content in the prompt and suppress key signals from the target node, thereby degrading performance. To address this challenge, we propose Dual Granularity Prompting (DGP), which mitigates information overload by preserving fine-grained textual details for the target node while summarizing neighbor information into coarse-grained text prompts. DGP introduces tailored summarization strategies for different data modalities, bi-level semantic abstraction for textual fields and statistical aggregation for numerical features, enabling effective compression of verbose neighbor content into concise, informative prompts. Experiments across public and industrial datasets demonstrate that DGP operates within a manageable token budget while improving fraud detection performance by up to 6.8% (AUPRC) over state-of-the-art methods, showing the potential of Graph-Enhanced LLMs for fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21636v1">StaffPro: an LLM Agent for Joint Staffing and Profiling</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents integrate pre-trained LLMs with modular algorithmic components and have shown remarkable reasoning and decision-making abilities. In this work, we investigate their use for two tightly intertwined challenges in workforce management: staffing, i.e., the assignment and scheduling of tasks to workers, which may require team formation; and profiling, i.e., the continuous estimation of workers' skills, preferences, and other latent attributes from unstructured data. We cast these problems in a formal mathematical framework that links scheduling decisions to latent feature estimation, and we introduce StaffPro, an LLM agent that addresses staffing and profiling jointly. Differently from existing staffing solutions, StaffPro allows expressing optimization objectives using natural language, accepts textual task descriptions and provides high flexibility. StaffPro interacts directly with humans by establishing a continuous human-agent feedback loop, ensuring natural and intuitive use. By analyzing human feedback, our agent continuously estimates the latent features of workers, realizing life-long worker profiling and ensuring optimal staffing performance over time. A consulting firm simulation example demonstrates that StaffPro successfully estimates workers' attributes and generates high quality schedules. With its innovative design, StaffPro offers a robust, interpretable, and human-centric solution for automated personnel management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10635v3">Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 website: https://llm-strategist.github.io
    </div>
    <details class="paper-abstract">
      Traditional reinforcement learning and planning typically requires vast amounts of data and training to develop effective policies. In contrast, large language models (LLMs) exhibit strong generalization and zero-shot capabilities, but struggle with tasks that require detailed planning and decision-making in complex action spaces. We introduce STRATEGIST, a novel approach that integrates the strengths of both methods. Our approach leverages LLMs to search and update high-level strategies (as text), which are then refined and executed by low-level Monte Carlo Tree Search (MCTS). STRATEGIST is a generalizable framework to optimize the strategy through population-based self-play simulations without the need for any training data. We demonstrate the effectiveness of STRATEGIST in learning optimal strategies for competitive, multi-turn games with partial information, including Game of Pure Strategy (GOPS) and multi-agent, hidden-identity discussion games like The Resistance: Avalon. Our results show that agents equipped with STRATEGIST outperform those trained with traditional RL methods, other LLM-based skill acquisition techniques, pre-existing LLM agents across both game environments and achieves comparable performance against human players.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15549v3">Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Code at https://github.com/aengusl/latent-adversarial-training. Models at https://huggingface.co/LLM-LAT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07237v2">Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance across a range of natural language processing tasks. However, their vast number of parameters introduces significant memory challenges during training, particularly when using memory-intensive optimizers like Adam. Existing memory-efficient algorithms often rely on techniques such as singular value decomposition projection or weight freezing. While these approaches help alleviate memory constraints, they generally produce suboptimal results compared to full-rank updates. In this paper, we investigate the memory-efficient method beyond low-rank training, proposing a novel solution called Gradient Wavelet Transform (GWT), which applies wavelet transforms to gradients in order to significantly reduce the memory requirements for maintaining optimizer states. We demonstrate that GWT can be seamlessly integrated with memory-intensive optimizers, enabling efficient training without sacrificing performance. Through extensive experiments on both pre-training and fine-tuning tasks, we show that GWT achieves state-of-the-art performance compared with advanced memory-efficient optimizers and full-rank approaches in terms of both memory usage and training performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03248v2">AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21563v1">Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v2">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19537v2">Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21538v1">Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21507v1">VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 21 pages, 19 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Video Anomaly Detection (VAD) aims to identify anomalous events in videos and accurately determine their time intervals. Current VAD methods mainly fall into two categories: traditional DNN-based approaches that focus on temporal localization, and LLM-based approaches that emphasize semantic understanding. Both anomaly understanding and grounding are essential for comprehensive video anomaly detection and can complement each other. However, no existing model or dataset supports both tasks simultaneously. To address this, we introduce VAGU (Video Anomaly Grounding and Understanding), the first benchmark to integrate both tasks. Each VAGU instance includes annotations for anomaly category, semantic explanation, precise temporal grounding and Video QA. We also provide multiple-choice Video QA for objective evaluation. Based on this dataset, we propose Glance then Scrutinize (GtS), a training-free framework guided by textual prompts. The framework first enables coarse localization of high-probability anomalous regions, followed by detailed anomaly interpretation and temporal boundary refinement. Additionally, we propose the JeAUG metric, which jointly evaluates semantic interpretability and temporal precision, overcoming the limitations of traditional metrics. Extensive experiments verify the effectiveness of our benchmark, framework, and evaluation metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14479v4">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21504v1">Evaluation and Benchmarking of LLM Agents: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The rise of LLM-based agents has opened new frontiers in AI applications, yet evaluating these agents remains a complex and underdeveloped area. This survey provides an in-depth overview of the emerging field of LLM agent evaluation, introducing a two-dimensional taxonomy that organizes existing work along (1) evaluation objectives -- what to evaluate, such as agent behavior, capabilities, reliability, and safety -- and (2) evaluation process -- how to evaluate, including interaction modes, datasets and benchmarks, metric computation methods, and tooling. In addition to taxonomy, we highlight enterprise-specific challenges, such as role-based access to data, the need for reliability guarantees, dynamic and long-horizon interactions, and compliance, which are often overlooked in current research. We also identify future research directions, including holistic, more realistic, and scalable evaluation. This work aims to bring clarity to the fragmented landscape of agent evaluation and provide a framework for systematic assessment, enabling researchers and practitioners to evaluate LLM agents for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21485v1">HLSDebugger: Identification and Correction of Logic Bugs in HLS Code with LLM Solutions</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 This work has been accepted at ICCAD 2025 (International Conference on Computer-Aided Design)
    </div>
    <details class="paper-abstract">
      High-level synthesis (HLS) accelerates hardware design by enabling the automatic translation of high-level descriptions into efficient hardware implementations. However, debugging HLS code is a challenging and labor-intensive task, especially for novice circuit designers or software engineers without sufficient hardware domain knowledge. The recent emergence of Large Language Models (LLMs) is promising in automating the HLS debugging process. Despite the great potential, three key challenges persist when applying LLMs to HLS logic debugging: 1) High-quality circuit data for training LLMs is scarce, posing a significant challenge. 2) Debugging logic bugs in hardware is inherently more complex than identifying software bugs with existing golden test cases. 3) The absence of reliable test cases requires multi-tasking solutions, performing both bug identification and correction. complicates the multi-tasking required for effective HLS debugging. In this work, we propose a customized solution named HLSDebugger to address the challenges. HLSDebugger first generates and releases a large labeled dataset with 300K data samples, targeting HLS logic bugs. The HLSDebugger model adopts an encoder-decoder structure, performing bug location identification, bug type prediction, and bug correction with the same model. HLSDebugger significantly outperforms advanced LLMs like GPT-4 in bug identification and by more than 3x in bug correction. It makes a substantial advancement in the exploration of automated debugging of HLS code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21482v1">Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21476v1">Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and Cartoonstock.com, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21471v1">An LLM Driven Agent Framework for Automated Infrared Spectral Multi Task Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Infrared spectroscopy offers rapid, non destructive measurement of chemical and material properties but suffers from high dimensional, overlapping spectral bands that challenge conventional chemometric approaches. Emerging large language models (LLMs), with their capacity for generalization and reasoning, offer promising potential for automating complex scientific workflows. Despite this promise, their application in IR spectral analysis remains largely unexplored. This study addresses the critical challenge of achieving accurate, automated infrared spectral interpretation under low-data conditions using an LLM-driven framework. We introduce an end-to-end, large language model driven agent framework that integrates a structured literature knowledge base, automated spectral preprocessing, feature extraction, and multi task reasoning in a unified pipeline. By querying a curated corpus of peer reviewed IR publications, the agent selects scientifically validated routines. The selected methods transform each spectrum into low dimensional feature sets, which are fed into few shot prompt templates for classification, regression, and anomaly detection. A closed loop, multi turn protocol iteratively appends mispredicted samples to the prompt, enabling dynamic refinement of predictions. Across diverse materials: stamp pad ink, Chinese medicine, Pu'er tea, Citri Reticulatae Pericarpium and waste water COD datasets, the multi turn LLM consistently outperforms single turn inference, rivaling or exceeding machine learning and deep learning models under low data regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v3">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21447v1">LLM4VV: Evaluating Cutting-Edge LLMs for Generation and Evaluation of Directive-Based Parallel Programming Model Compiler Tests</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The usage of Large Language Models (LLMs) for software and test development has continued to increase since LLMs were first introduced, but only recently have the expectations of LLMs become more realistic. Verifying the correctness of code generated by LLMs is key to improving their usefulness, but there have been no comprehensive and fully autonomous solutions developed yet. Hallucinations are a major concern when LLMs are applied blindly to problems without taking the time and effort to verify their outputs, and an inability to explain the logical reasoning of LLMs leads to issues with trusting their results. To address these challenges while also aiming to effectively apply LLMs, this paper proposes a dual-LLM system (i.e. a generative LLM and a discriminative LLM) and experiments with the usage of LLMs for the generation of a large volume of compiler tests. We experimented with a number of LLMs possessing varying parameter counts and presented results using ten carefully-chosen metrics that we describe in detail in our narrative. Through our findings, it is evident that LLMs possess the promising potential to generate quality compiler tests and verify them automatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v2">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v5">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 take a middle-ground approach, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfiniteHBD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfiniteHBD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt to variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh)-based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfiniteHBD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios are under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12225v2">Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Enhancing Large Language Model (LLM)'s performance with best-of-N sampling is effective and has attracted significant attention. However, it is computationally prohibitive due to massive, data-hungry text-based reward models. By changing the data source from text to hidden states, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel, lightweight technique that leverages the rich information embedded in LLM hidden states to address these issues, which operates on token-level and consists of only linear layers. Extensive experiments show that SWIFT outperforms baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training, demonstrating significant efficiency improvement. SWIFT's robust scalability, applicability to some closed-source models via logits, and ability to be combined with traditional reward models to yield further performance gains underscore its practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21428v1">MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agent Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 23 Pages, 20 Figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have shown significant autonomous capabilities in dynamically searching and incorporating relevant tools or Model Context Protocol (MCP) servers for individual queries. However, fixed context windows limit effectiveness in multi-turn interactions requiring repeated, independent tool usage. We introduce MemTool, a short-term memory framework enabling LLM agents to dynamically manage tools or MCP server contexts across multi-turn conversations. MemTool offers three agentic architectures: 1) Autonomous Agent Mode, granting full tool management autonomy, 2) Workflow Mode, providing deterministic control without autonomy, and 3) Hybrid Mode, combining autonomous and deterministic control. Evaluating each MemTool mode across 13+ LLMs on the ScaleMCP benchmark, we conducted experiments over 100 consecutive user interactions, measuring tool removal ratios (short-term memory efficiency) and task completion accuracy. In Autonomous Agent Mode, reasoning LLMs achieve high tool-removal efficiency (90-94% over a 3-window average), while medium-sized models exhibit significantly lower efficiency (0-60%). Workflow and Hybrid modes consistently manage tool removal effectively, whereas Autonomous and Hybrid modes excel at task completion. We present trade-offs and recommendations for each MemTool mode based on task accuracy, agency, and model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22289v1">Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted for publication at SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22286v1">Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 5 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
    </div>
    <details class="paper-abstract">
      The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22267v1">Promoting Online Safety by Simulating Unsafe Conversations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14907v2">GneissWeb: Preparing High Quality Data for LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09147v2">LLM-as-a-qualitative-judge: automating error analysis in natural language generation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22223v1">Secure coding for web applications: Frameworks, challenges, and the role of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 11 pages, 5 figures, 3 tables, 6 listings
    </div>
    <details class="paper-abstract">
      Secure coding is a critical yet often overlooked practice in software development. Despite extensive awareness efforts, real-world adoption remains inconsistent due to organizational, educational, and technical barriers. This paper provides a comprehensive review of secure coding practices across major frameworks and domains, including web development, DevSecOps, and cloud security. It introduces a structured framework comparison and categorizes threats aligned with the OWASP Top 10. Additionally, we explore the rising role of Large Language Models (LLMs) in evaluating and recommending secure code, presenting a reproducible case study across four major vulnerability types. This paper offers practical insights for researchers, developers, and educators on integrating secure coding into real-world development processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22205v1">CTG-Insight: A Multi-Agent Interpretable LLM Framework for Cardiotocography Analysis and Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Remote fetal monitoring technologies are becoming increasingly common. Yet, most current systems offer limited interpretability, leaving expectant parents with raw cardiotocography (CTG) data that is difficult to understand. In this work, we present CTG-Insight, a multi-agent LLM system that provides structured interpretations of fetal heart rate (FHR) and uterine contraction (UC) signals. Drawing from established medical guidelines, CTG-Insight decomposes each CTG trace into five medically defined features: baseline, variability, accelerations, decelerations, and sinusoidal pattern, each analyzed by a dedicated agent. A final aggregation agent synthesizes the outputs to deliver a holistic classification of fetal health, accompanied by a natural language explanation. We evaluate CTG-Insight on the NeuroFetalNet Dataset and compare it against deep learning models and the single-agent LLM baseline. Results show that CTG-Insight achieves state-of-the-art accuracy (96.4%) and F1-score (97.8%) while producing transparent and interpretable outputs. This work contributes an interpretable and extensible CTG analysis framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22168v1">Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22160v1">Strategic Deflection: Defending LLMs from Logit Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22134v1">IntentFlow: Interactive Support for Communicating Intent with LLMs in Writing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are widely used for writing, users often struggle to express their nuanced and evolving intents through prompt-based interfaces. Intents -- low-level strategies or preferences for achieving a writing goal -- are often vague, fluid, or even subconscious, making it difficult for users to articulate and adjust them. To address this, we present IntentFlow, which supports the communication of dynamically evolving intents throughout LLM-assisted writing. IntentFlow extracts goals and intents from user prompts and presents them as editable interface components, which users can revise, remove, or refine via direct manipulation or follow-up prompts. Visual links connect each component to the output segments it influences, helping users understand model behavior. In a within-subjects study (N=12), participants using IntentFlow, compared to a chat-based baseline, expressed their intents more easily and in detail, engaged in more meaningful actions to communicate intents, such as adjusting and deleting, and produced outputs that better aligned with their evolving intents. We found that editable intent representations help users refine and consolidate a final set of intents, which can be reused across similar tasks to support consistent and transferable LLM-assisted writing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22133v1">Prompt Optimization and Evaluation for LLM Automated Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 9 pages, 5 Figures, and 1 Appendix item
    </div>
    <details class="paper-abstract">
      Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21028v1">Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Nearly all human work is collaborative; thus, the evaluation of real-world NLP applications often requires multiple dimensions that align with diverse human perspectives. As real human evaluator resources are often scarce and costly, the emerging "LLM-as-a-judge" paradigm sheds light on a promising approach to leverage LLM agents to believably simulate human evaluators. Yet, to date, existing LLM-as-a-judge approaches face two limitations: persona descriptions of agents are often arbitrarily designed, and the frameworks are not generalizable to other tasks. To address these challenges, we propose MAJ-EVAL, a Multi-Agent-as-Judge evaluation framework that can automatically construct multiple evaluator personas with distinct dimensions from relevant text documents (e.g., research papers), instantiate LLM agents with the personas, and engage in-group debates with multi-agents to Generate multi-dimensional feedback. Our evaluation experiments in both the educational and medical domains demonstrate that MAJ-EVAL can generate evaluation results that better align with human experts' ratings compared with conventional automated evaluation metrics and existing LLM-as-a-judge methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12854v3">Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21017v1">MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Code and data: https://github.com/sunblaze-ucb/mirage-bench.git
    </div>
    <details class="paper-abstract">
      Hallucinations pose critical risks for large language model (LLM)-based agents, often manifesting as hallucinative actions resulting from fabricated or misinterpreted information within the cognitive context. While recent studies have exposed such failures, existing evaluations remain fragmented and lack a principled testbed. In this paper, we present MIRAGE-Bench--Measuring Illusions in Risky AGEnt settings--the first unified benchmark for eliciting and evaluating hallucinations in interactive LLM-agent scenarios. We begin by introducing a three-part taxonomy to address agentic hallucinations: actions that are unfaithful to (i) task instructions, (ii) execution history, or (iii) environment observations. To analyze, we first elicit such failures by performing a systematic audit of existing agent benchmarks, then synthesize test cases using a snapshot strategy that isolates decision points in deterministic and reproducible manners. To evaluate hallucination behaviors, we adopt a fine-grained-level LLM-as-a-Judge paradigm with tailored risk-aware prompts, enabling scalable, high-fidelity assessment of agent actions without enumerating full action spaces. MIRAGE-Bench provides actionable insights on failure modes of LLM agents and lays the groundwork for principled progress in mitigating hallucinations in interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02087v2">Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v1">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20977v1">Repairing vulnerabilities without invisible hands. A differentiated replication study on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Background: Automated Vulnerability Repair (AVR) is a fast-growing branch of program repair. Recent studies show that large language models (LLMs) outperform traditional techniques, extending their success beyond code generation and fault detection. Hypothesis: These gains may be driven by hidden factors -- "invisible hands" such as training-data leakage or perfect fault localization -- that let an LLM reproduce human-authored fixes for the same code. Objective: We replicate prior AVR studies under controlled conditions by deliberately adding errors to the reported vulnerability location in the prompt. If LLMs merely regurgitate memorized fixes, both small and large localization errors should yield the same number of correct patches, because any offset should divert the model from the original fix. Method: Our pipeline repairs vulnerabilities from the Vul4J and VJTrans benchmarks after shifting the fault location by n lines from the ground truth. A first LLM generates a patch, a second LLM reviews it, and we validate the result with regression and proof-of-vulnerability tests. Finally, we manually audit a sample of patches and estimate the error rate with the Agresti-Coull-Wilson method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v2">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v1">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16117v2">BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 9 figures. Accepted to IEEE VIS 2025 (Full Papers Track, submission ID 1204)
    </div>
    <details class="paper-abstract">
      Biomedical data harmonization is essential for enabling exploratory analyses and meta-studies, but the process of schema matching - identifying semantic correspondences between elements of disparate datasets (schemas) - remains a labor-intensive and error-prone task. Even state-of-the-art automated methods often yield low accuracy when applied to biomedical schemas due to the large number of attributes and nuanced semantic differences between them. We present BDIViz, a novel visual analytics system designed to streamline the schema matching process for biomedical data. Through formative studies with domain experts, we identified key requirements for an effective solution and developed interactive visualization techniques that address both scalability challenges and semantic ambiguity. BDIViz employs an ensemble approach that combines multiple matching methods with LLM-based validation, summarizes matches through interactive heatmaps, and provides coordinated views that enable users to quickly compare attributes and their values. Our method-agnostic design allows the system to integrate various schema matching algorithms and adapt to application-specific needs. Through two biomedical case studies and a within-subject user study with domain experts, we demonstrate that BDIViz significantly improves matching accuracy while reducing cognitive load and curation time compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20870v1">A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20849v1">Latent Inter-User Difference Modeling for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00582v2">Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by ACL 2025 Findings, 20 pages
    </div>
    <details class="paper-abstract">
      Psychology research has shown that humans are poor at estimating their performance on tasks, tending towards underconfidence on easy tasks and overconfidence on difficult tasks. We examine three LLMs, Llama-3-70B-instruct, Claude-3-Sonnet, and GPT-4o, on a range of QA tasks of varying difficulty, and show that models exhibit subtle differences from human patterns of overconfidence: less sensitive to task difficulty, and when prompted to answer based on different personas -- e.g., expert vs layman, or different race, gender, and ages -- the models will respond with stereotypically biased confidence estimations even though their underlying answer accuracy remains the same. Based on these observations, we propose Answer-Free Confidence Estimation (AFCE) to improve confidence calibration and LLM interpretability in these settings. AFCE is a self-assessment method that employs two stages of prompting, first eliciting only confidence scores on questions, then asking separately for the answer. Experiments on the MMLU and GPQA datasets spanning subjects and difficulty show that this separation of tasks significantly reduces overconfidence and delivers more human-like sensitivity to task difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20774v1">evalSmarT: An LLM-Based Framework for Evaluating Smart Contract Generated Comments</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 4 pages, 4 tables
    </div>
    <details class="paper-abstract">
      Smart contract comment generation has gained traction as a means to improve code comprehension and maintainability in blockchain systems. However, evaluating the quality of generated comments remains a challenge. Traditional metrics such as BLEU and ROUGE fail to capture domain-specific nuances, while human evaluation is costly and unscalable. In this paper, we present \texttt{evalSmarT}, a modular and extensible framework that leverages large language models (LLMs) as evaluators. The system supports over 400 evaluator configurations by combining approximately 40 LLMs with 10 prompting strategies. We demonstrate its application in benchmarking comment generation tools and selecting the most informative outputs. Our results show that prompt design significantly impacts alignment with human judgment, and that LLM-based evaluation offers a scalable and semantically rich alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19299v3">The Impact of LoRA Adapters on LLMs for Clinical Text Classification Under Computational and Data Constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted for publication in the IEEE Access
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to domain gap, limited data, and stringent hardware constraints. In this study, we evaluate four adapter techniques-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN) - equivalent to Low-Rank Adaptation (LoRA), for clinical note classification under real-world, resource-constrained conditions. All experiments were conducted on a single NVIDIA Quadro P620 GPU (2 GB VRAM, 512 CUDA cores, 1.386 TFLOPS FP32), limiting batch sizes to <8 sequences and maximum sequence length to 256 tokens. Our clinical corpus comprises only 580 000 tokens, several orders of magnitude smaller than standard LLM pre-training datasets. We fine-tuned three biomedical pre-trained LLMs (CamemBERT-bio, AliBERT, DrBERT) and two lightweight Transformer models trained from scratch. Results show that 1) adapter structures provide no consistent gains when fine-tuning biomedical LLMs under these constraints, and 2) simpler Transformers, with minimal parameter counts and training times under six hours, outperform adapter-augmented LLMs, which required over 1000 GPU-hours. Among adapters, GRN achieved the best metrics (accuracy, precision, recall, F1 = 0.88). These findings demonstrate that, in low-resource clinical settings with limited data and compute, lightweight Transformers trained from scratch offer a more practical and efficient solution than large LLMs, while GRN remains a viable adapter choice when minimal adaptation is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06003v3">LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Conference Version (ISCA'25). Fixed a typo
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference becomes resource-intensive, prompting a shift toward low-bit model weights to reduce the memory footprint and improve efficiency. Such low-bit LLMs necessitate the mixed-precision matrix multiplication (mpGEMM), an important yet underexplored operation involving the multiplication of lower-precision weights with higher-precision activations. Off-the-shelf hardware does not support this operation natively, leading to indirect, thus inefficient, dequantization-based implementations. In this paper, we study the lookup table (LUT)-based approach for mpGEMM and find that a conventional LUT implementation fails to achieve the promised gains. To unlock the full potential of LUT-based mpGEMM, we propose LUT Tensor Core, a software-hardware co-design for low-bit LLM inference. LUT Tensor Core differentiates itself from conventional LUT designs through: 1) software-based optimizations to minimize table precompute overhead and weight reinterpretation to reduce table storage; 2) a LUT-based Tensor Core hardware design with an elongated tiling shape to maximize table reuse and a bit-serial design to support diverse precision combinations in mpGEMM; 3) a new instruction set and compilation optimizations for LUT-based mpGEMM. LUT Tensor Core significantly outperforms existing pure software LUT implementations and achieves a 1.44$\times$ improvement in compute density and energy efficiency compared to previous state-of-the-art LUT-based accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20674v1">LLM-Based Repair of Static Nullability Errors</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Modern Java projects increasingly adopt static analysis tools that prevent null-pointer exceptions by treating nullness as a type property. However, integrating such tools into large, existing codebases remains a significant challenge. While annotation inference can eliminate many errors automatically, a subset of residual errors -- typically a mix of real bugs and false positives -- often persist and can only be resolved via code changes. Manually addressing these errors is tedious and error-prone. Large language models (LLMs) offer a promising path toward automating these repairs, but naively-prompted LLMs often generate incorrect, contextually-inappropriate edits. Resolving a nullability error demands a deep understanding of how a symbol is used across the codebase, often spanning methods, classes, and packages. We present NullRepair, a system that integrates LLMs into a structured workflow for resolving the errors from a nullability checker. NullRepair's decision process follows a flowchart derived from manual analysis of 200 real-world errors. It leverages static analysis to identify safe and unsafe usage regions of symbols, using error-free usage examples to contextualize model prompts. Patches are generated through an iterative interaction with the LLM that incorporates project-wide context and decision logic. Our evaluation on 12 real-world Java projects shows that NullRepair resolves an average of 72% of the errors that remain after applying a state-of-the-art annotation inference technique. Unlike a naively-prompted LLM, NullRepair also largely preserves program semantics, with all unit tests passing in 10/12 projects after applying every edit proposed by NullRepair, and 98% or more tests passing in the remaining two projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20666v1">MIMII-Agent: Leveraging LLMs with Function Calling for Relative Evaluation of Anomalous Sound Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper proposes a method for generating machine-type-specific anomalies to evaluate the relative performance of unsupervised anomalous sound detection (UASD) systems across different machine types, even in the absence of real anomaly sound data. Conventional keyword-based data augmentation methods often produce unrealistic sounds due to their reliance on manually defined labels, limiting scalability as machine types and anomaly patterns diversify. Advanced audio generative models, such as MIMII-Gen, show promise but typically depend on anomalous training data, making them less effective when diverse anomalous examples are unavailable. To address these limitations, we propose a novel synthesis approach leveraging large language models (LLMs) to interpret textual descriptions of faults and automatically select audio transformation functions, converting normal machine sounds into diverse and plausible anomalous sounds. We validate this approach by evaluating a UASD system trained only on normal sounds from five machine types, using both real and synthetic anomaly data. Experimental results reveal consistent trends in relative detection difficulty across machine types between synthetic and real anomalies. This finding supports our hypothesis and highlights the effectiveness of the proposed LLM-based synthesis approach for relative evaluation of UASD systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20655v1">CoGrader: Transforming Instructors' Assessment of Project Reports through Collaborative LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Grading project reports are increasingly significant in today's educational landscape, where they serve as key assessments of students' comprehensive problem-solving abilities. However, it remains challenging due to the multifaceted evaluation criteria involved, such as creativity and peer-comparative achievement. Meanwhile, instructors often struggle to maintain fairness throughout the time-consuming grading process. Recent advances in AI, particularly large language models, have demonstrated potential for automating simpler grading tasks, such as assessing quizzes or basic writing quality. However, these tools often fall short when it comes to complex metrics, like design innovation and the practical application of knowledge, that require an instructor's educational insights into the class situation. To address this challenge, we conducted a formative study with six instructors and developed CoGrader, which introduces a novel grading workflow combining human-LLM collaborative metrics design, benchmarking, and AI-assisted feedback. CoGrader was found effective in improving grading efficiency and consistency while providing reliable peer-comparative feedback to students. We also discuss design insights and ethical considerations for the development of human-AI collaborative grading systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18181v2">SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by Design Automation Conference (DAC) 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13940v2">DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 29 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) integrated with autonomous agents hold significant potential for advancing scientific discovery through automated reasoning and task execution. However, applying LLM agents to drug discovery is still constrained by challenges such as large-scale multimodal data processing, limited task automation, and poor support for domain-specific tools. To overcome these limitations, we introduce DrugPilot, a LLM-based agent system with a parameterized reasoning architecture designed for end-to-end scientific workflows in drug discovery. DrugPilot enables multi-stage research processes by integrating structured tool use with a novel parameterized memory pool. The memory pool converts heterogeneous data from both public sources and user-defined inputs into standardized representations. This design supports efficient multi-turn dialogue, reduces information loss during data exchange, and enhances complex scientific decision-making. To support training and benchmarking, we construct a drug instruction dataset covering eight core drug discovery tasks. Under the Berkeley function-calling benchmark, DrugPilot significantly outperforms state-of-the-art agents such as ReAct and LoT, achieving task completion rates of 98.0%, 93.5%, and 64.0% for simple, multi-tool, and multi-turn scenarios, respectively. These results highlight DrugPilot's potential as a versatile agent framework for computational science domains requiring automated, interactive, and data-integrated reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06102v3">SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs) in Business Intelligence (BI), existing solutions face critical challenges in industrial deployments: functionality deficiencies from legacy systems failing to meet evolving LLM-era user demands, interaction limitations from single-round SQL generation paradigms inadequate for multi-round clarification, and cost for domain adaptation arising from cross-domain methods migration. We present SiriusBI, a practical LLM-powered BI system addressing the challenges of industrial deployments through three key innovations: (a) An end-to-end architecture integrating multi-module coordination to overcome functionality gaps in legacy systems; (b) A multi-round dialogue with querying mechanism, consisting of semantic completion, knowledge-guided clarification, and proactive querying processes, to resolve interaction constraints in SQL generation; (c) A data-conditioned SQL generation method selection strategy that supports both an efficient one-step Fine-Tuning approach and a two-step method leveraging Semantic Intermediate Representation for low-cost cross-domain applications. Experiments on both real-world datasets and public benchmarks demonstrate the effectiveness of SiriusBI. User studies further confirm that SiriusBI enhances both productivity and user experience. As an independent service on Tencent's data platform, SiriusBI is deployed across finance, advertising, and cloud sectors, serving dozens of enterprise clients. It achieves over 93% accuracy in SQL generation and reduces data analysts' query time from minutes to seconds in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v1">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20527v1">SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v1">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20509v1">LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models.
    </details>
</div>
