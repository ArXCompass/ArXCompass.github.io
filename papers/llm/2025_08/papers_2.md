# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23486v2">A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q\&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2\%, safety 54.7\%, effectiveness 62.3\%), with a significant 13.3\% performance drop in high-risk scenarios (p $<$ 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08795v1">A Dual-Axis Taxonomy of Knowledge Editing for LLMs: From Mechanisms to Functions</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 13 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) acquire vast knowledge from large text corpora, but this information can become outdated or inaccurate. Since retraining is computationally expensive, knowledge editing offers an efficient alternative -- modifying internal knowledge without full retraining. These methods aim to update facts precisely while preserving the model's overall capabilities. While existing surveys focus on the mechanism of editing (e.g., parameter changes vs. external memory), they often overlook the function of the knowledge being edited. This survey introduces a novel, complementary function-based taxonomy to provide a more holistic view. We examine how different mechanisms apply to various knowledge types -- factual, temporal, conceptual, commonsense, and social -- highlighting how editing effectiveness depends on the nature of the target knowledge. By organizing our review along these two axes, we map the current landscape, outline the strengths and limitations of existing methods, define the problem formally, survey evaluation tasks and datasets, and conclude with open challenges and future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08777v1">Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Accepted at RecSys '25
    </div>
    <details class="paper-abstract">
      Evaluating personalized recommendations remains a central challenge, especially in long-form audio domains like podcasts, where traditional offline metrics suffer from exposure bias and online methods such as A/B testing are costly and operationally constrained. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) as offline judges to assess the quality of podcast recommendations in a scalable and interpretable manner. Our two-stage profile-aware approach first constructs natural-language user profiles distilled from 90 days of listening history. These profiles summarize both topical interests and behavioral patterns, serving as compact, interpretable representations of user preferences. Rather than prompting the LLM with raw data, we use these profiles to provide high-level, semantically rich context-enabling the LLM to reason more effectively about alignment between a user's interests and recommended episodes. This reduces input complexity and improves interpretability. The LLM is then prompted to deliver fine-grained pointwise and pairwise judgments based on the profile-episode match. In a controlled study with 47 participants, our profile-aware judge matched human judgments with high fidelity and outperformed or matched a variant using raw listening histories. The framework enables efficient, profile-aware evaluation for iterative testing and model selection in recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08761v1">DevNous: An LLM-Based Multi-Agent System for Grounding IT Project Management in Unstructured Conversation</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      The manual translation of unstructured team dialogue into the structured artifacts required for Information Technology (IT) project governance is a critical bottleneck in modern information systems management. We introduce DevNous, a Large Language Model-based (LLM) multi-agent expert system, to automate this unstructured-to-structured translation process. DevNous integrates directly into team chat environments, identifying actionable intents from informal dialogue and managing stateful, multi-turn workflows for core administrative tasks like automated task formalization and progress summary synthesis. To quantitatively evaluate the system, we introduce a new benchmark of 160 realistic, interactive conversational turns. The dataset was manually annotated with a multi-label ground truth and is publicly available. On this benchmark, DevNous achieves an exact match turn accuracy of 81.3\% and a multiset F1-Score of 0.845, providing strong evidence for its viability. The primary contributions of this work are twofold: (1) a validated architectural pattern for developing ambient administrative agents, and (2) the introduction of the first robust empirical baseline and public benchmark dataset for this challenging problem domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08742v1">SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05496v2">InfiAlign: A Scalable and Sample-Efficient Framework for Aligning LLMs to Enhance Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive reasoning abilities on a wide range of complex tasks. However, enhancing these capabilities through post-training remains resource intensive, particularly in terms of data and computational cost. Although recent efforts have sought to improve sample efficiency through selective data curation, existing methods often rely on heuristic or task-specific strategies that hinder scalability. In this work, we introduce InfiAlign, a scalable and sample-efficient post-training framework that integrates supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to align LLMs for enhanced reasoning. At the core of InfiAlign is a robust data selection pipeline that automatically curates high-quality alignment data from open-source reasoning datasets using multidimensional quality metrics. This pipeline enables significant performance gains while drastically reducing data requirements and remains extensible to new data sources. When applied to the Qwen2.5-Math-7B-Base model, our SFT model achieves performance on par with DeepSeek-R1-Distill-Qwen-7B, while using only approximately 12% of the training data, and demonstrates strong generalization across diverse reasoning tasks. Additional improvements are obtained through the application of DPO, with particularly notable gains in mathematical reasoning tasks. The model achieves an average improvement of 3.89% on AIME 24/25 benchmarks. Our results highlight the effectiveness of combining principled data selection with full-stage post-training, offering a practical solution for aligning large reasoning models in a scalable and data-efficient manner. The model checkpoints are available at https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08715v1">MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 5 figures
    </div>
    <details class="paper-abstract">
      Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07745v2">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Insider threats, which can lead to severe losses, remain a major security concern. While machine learning-based insider threat detection (ITD) methods have shown promising results, their progress is hindered by the scarcity of high-quality data. Enterprise data is sensitive and rarely accessible, while publicly available datasets, when limited in scale due to cost, lack sufficient real-world coverage; and when purely synthetic, they fail to capture rich semantics and realistic user behavior. To address this, we propose Chimera, the first large language model (LLM)-based multi-agent framework that automatically simulates both benign and malicious insider activities and collects diverse logs across diverse enterprise environments. Chimera models each employee with agents that have role-specific behavior and integrates modules for group meetings, pairwise interactions, and autonomous scheduling, capturing realistic organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP theft, system sabotage) and has been deployed to simulate activities in three sensitive domains: technology company, finance corporation, and medical institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via human studies and quantitative analysis, confirming its diversity, realism, and presence of explainable threat patterns. Evaluations of existing ITD methods show an average F1-score of 0.83, which is significantly lower than 0.99 on the CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for advancing ITD research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08709v1">CRADLE: Conversational RTL Design Space Exploration with LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Accepted for presentation at the 22nd International SoC Conference (ISOCC 2025). Proceedings to be included in IEEE Xplore
    </div>
    <details class="paper-abstract">
      This paper presents CRADLE, a conversational framework for design space exploration of RTL designs using LLM-based multi-agent systems. Unlike existing rigid approaches, CRADLE enables user-guided flows with internal self-verification, correction, and optimization. We demonstrate the framework with a generator-critic agent system targeting FPGA resource minimization using state-of-the-art LLMs. Experimental results on the RTLLM benchmark show that CRADLE achieves significant reductions in resource usage with averages of 48% and 40% in LUTs and FFs across all benchmark designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08657v1">$\text{M}^{2}$LLM: Multi-view Molecular Representation Learning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 IJCAI 2025
    </div>
    <details class="paper-abstract">
      Accurate molecular property prediction is a critical challenge with wide-ranging applications in chemistry, materials science, and drug discovery. Molecular representation methods, including fingerprints and graph neural networks (GNNs), achieve state-of-the-art results by effectively deriving features from molecular structures. However, these methods often overlook decades of accumulated semantic and contextual knowledge. Recent advancements in large language models (LLMs) demonstrate remarkable reasoning abilities and prior knowledge across scientific domains, leading us to hypothesize that LLMs can generate rich molecular representations when guided to reason in multiple perspectives. To address these gaps, we propose $\text{M}^{2}$LLM, a multi-view framework that integrates three perspectives: the molecular structure view, the molecular task view, and the molecular rules view. These views are fused dynamically to adapt to task requirements, and experiments demonstrate that $\text{M}^{2}$LLM achieves state-of-the-art performance on multiple benchmarks across classification and regression tasks. Moreover, we demonstrate that representation derived from LLM achieves exceptional performance by leveraging two core functionalities: the generation of molecular embeddings through their encoding capabilities and the curation of molecular features through advanced reasoning processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08653v1">LLM driven Text-to-Table Generation through Sub-Tasks Guidance and Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Transforming unstructured text into structured data is a complex task, requiring semantic understanding, reasoning, and structural comprehension. While Large Language Models (LLMs) offer potential, they often struggle with handling ambiguous or domain-specific data, maintaining table structure, managing long inputs, and addressing numerical reasoning. This paper proposes an efficient system for LLM-driven text-to-table generation that leverages novel prompting techniques. Specifically, the system incorporates two key strategies: breaking down the text-to-table task into manageable, guided sub-tasks and refining the generated tables through iterative self-feedback. We show that this custom task decomposition allows the model to address the problem in a stepwise manner and improves the quality of the generated table. Furthermore, we discuss the benefits and potential risks associated with iterative self-feedback on the generated tables while highlighting the trade-offs between enhanced performance and computational cost. Our methods achieve strong results compared to baselines on two complex text-to-table generation datasets available in the public domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08636v1">InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 InternBootcamp Tech Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04209v3">To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08001v2">Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      "Fedspeak", the stylized and often nuanced language used by the U.S. Federal Reserve, encodes implicit policy signals and strategic stances. The Federal Open Market Committee strategically employs Fedspeak as a communication tool to shape market expectations and influence both domestic and global economic conditions. As such, automatically parsing and interpreting Fedspeak presents a high-impact challenge, with significant implications for financial forecasting, algorithmic trading, and data-driven policy analysis. In this paper, we propose an LLM-based, uncertainty-aware framework for deciphering Fedspeak and classifying its underlying monetary policy stance. Technically, to enrich the semantic and contextual representation of Fedspeak texts, we incorporate domain-specific reasoning grounded in the monetary policy transmission mechanism. We further introduce a dynamic uncertainty decoding module to assess the confidence of model predictions, thereby enhancing both classification accuracy and model reliability. Experimental results demonstrate that our framework achieves state-of-the-art performance on the policy stance analysis task. Moreover, statistical analysis reveals a significant positive correlation between perceptual uncertainty and model error rates, validating the effectiveness of perceptual uncertainty as a diagnostic signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08629v1">Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06799v2">LSDTs: LLM-Augmented Semantic Digital Twins for Adaptive Knowledge-Intensive Infrastructure Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Digital Twins (DTs) offer powerful tools for managing complex infrastructure systems, but their effectiveness is often limited by challenges in integrating unstructured knowledge. Recent advances in Large Language Models (LLMs) bring new potential to address this gap, with strong abilities in extracting and organizing diverse textual information. We therefore propose LSDTs (LLM-Augmented Semantic Digital Twins), a framework that helps LLMs extract planning knowledge from unstructured documents like environmental regulations and technical guidelines, and organize it into a formal ontology. This ontology forms a semantic layer that powers a digital twin-a virtual model of the physical system-allowing it to simulate realistic, regulation-aware planning scenarios. We evaluate LSDTs through a case study of offshore wind farm planning in Maryland, including its application during Hurricane Sandy. Results demonstrate that LSDTs support interpretable, regulation-aware layout optimization, enable high-fidelity simulation, and enhance adaptability in infrastructure planning. This work shows the potential of combining generative AI with digital twins to support complex, knowledge-driven planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06944v2">AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 https://github.com/hlxtsyj/AMFT
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of \textbf{implicit rewards}, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce \textbf{Adaptive Meta Fine-Tuning (AMFT)}, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a \textbf{meta-gradient adaptive weight controller} that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. This forward-looking approach, regularized by policy entropy for stability, autonomously discovers an effective training curriculum. We conduct a comprehensive evaluation on challenging benchmarks spanning mathematical reasoning, abstract visual reasoning (General Points), and vision-language navigation (V-IRL). AMFT consistently establishes a new state-of-the-art and demonstrats superior generalization on out-of-distribution (OOD) tasks. Ablation studies and training dynamic analysis confirm that the meta-learning controller is crucial for AMFT's stability, sample efficiency, and performance, offering a more principled and effective paradigm for LLM alignment. Our codes are open-sourced via https://github.com/hlxtsyj/AMFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14758v3">Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing large language model (LLM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LLMs. Through empirical analysis, we uncover positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LLMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LLM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07675v2">Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing how users interact with information systems, yet their high inference cost poses serious scalability and sustainability challenges. Caching inference responses, allowing them to be retrieved without another forward pass through the LLM, has emerged as one possible solution. Traditional exact-match caching, however, overlooks the semantic similarity between queries, leading to unnecessary recomputation. Semantic caching addresses this by retrieving responses based on semantic similarity, but introduces a fundamentally different cache eviction problem: one must account for mismatch costs between incoming queries and cached responses. Moreover, key system parameters, such as query arrival probabilities and serving costs, are often unknown and must be learned over time. Existing semantic caching methods are largely ad-hoc, lacking theoretical foundations and unable to adapt to real-world uncertainty. In this paper, we present a principled, learning-based framework for semantic cache eviction under unknown query and cost distributions. We formulate both offline optimization and online learning variants of the problem, and develop provably efficient algorithms with state-of-the-art guarantees. We also evaluate our framework on a synthetic dataset, showing that our proposed algorithms perform matching or superior performance compared with baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23261v2">DynaSwarm: Dynamically Graph Structure Selection for LLM-based Multi-agent System</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 content error
    </div>
    <details class="paper-abstract">
      Current multi-agent systems (MAS) frameworks often rely on manually designed and static collaboration graph structures, limiting adaptability and performance. To address these limitations, we propose DynaSwarm, a dynamic framework that enhances LLM-based MAS through two key innovations: (1) an actor-critic reinforcement learning (A2C) mechanism to optimize graph structures with improved stability over prior RL methods, and (2) a dynamic graph selector that adaptively chooses the optimal graph structure for each input sample via parameter-efficient LLM fine-tuning. DynaSwarm eliminates the need for rigid, one-fits-all graph architectures, instead leveraging sample-specific idiosyncrasies to dynamically route queries through specialized agent networks. (c) We propose to fine-tune the demonstration retriever to fully exploit the power of in-context learning (ICL). Extensive experiments on question answering, mathematical reasoning, and coding tasks demonstrate that DynaSwarm consistently outperforms state-of-the-art single-agent and MAS baselines across multiple LLM backbones. Our findings highlight the importance of sample-aware structural flexibility in LLM MAS designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08545v1">OmniLLP: Enhancing LLM-based Log Level Prediction with Context-Aware Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Developers insert logging statements in source code to capture relevant runtime information essential for maintenance and debugging activities. Log level choice is an integral, yet tricky part of the logging activity as it controls log verbosity and therefore influences systems' observability and performance. Recent advances in ML-based log level prediction have leveraged large language models (LLMs) to propose log level predictors (LLPs) that demonstrated promising performance improvements (AUC between 0.64 and 0.8). Nevertheless, current LLM-based LLPs rely on randomly selected in-context examples, overlooking the structure and the diverse logging practices within modern software projects. In this paper, we propose OmniLLP, a novel LLP enhancement framework that clusters source files based on (1) semantic similarity reflecting the code's functional purpose, and (2) developer ownership cohesion. By retrieving in-context learning examples exclusively from these semantic and ownership aware clusters, we aim to provide more coherent prompts to LLPs leveraging LLMs, thereby improving their predictive accuracy. Our results show that both semantic and ownership-aware clusterings statistically significantly improve the accuracy (by up to 8\% AUC) of the evaluated LLM-based LLPs compared to random predictors (i.e., leveraging randomly selected in-context examples from the whole project). Additionally, our approach that combines the semantic and ownership signal for in-context prediction achieves an impressive 0.88 to 0.96 AUC across our evaluated projects. Our findings highlight the value of integrating software engineering-specific context, such as code semantic and developer ownership signals into LLM-LLPs, offering developers a more accurate, contextually-aware approach to logging and therefore, enhancing system maintainability and observability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v2">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18862v2">One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Published at COLM 2025. 30 pages, 7 figures. Code is available at https://github.com/jacobdunefsky/one-shot-steering-repro and https://github.com/jacobdunefsky/one-shot-steering-misalignment
    </div>
    <details class="paper-abstract">
      Steering vectors (SVs) have emerged as a promising approach for interpreting and controlling LLMs, but current methods typically require large contrastive datasets that are often impractical to construct and may capture spurious correlations. We propose directly optimizing SVs through gradient descent on a single training example, and systematically investigate how these SVs generalize. We consider several SV optimization techniques and find that the resulting SVs effectively mediate safety-relevant behaviors in multiple models. Indeed, in experiments on an alignment-faking model, we are able to optimize one-shot SVs that induce harmful behavior on benign examples and whose negations suppress harmful behavior on malign examples. And in experiments on refusal suppression, we demonstrate that one-shot optimized SVs can transfer across inputs, yielding a Harmbench attack success rate of 96.9%. Furthermore, we extend work on "emergent misalignment" and show that SVs optimized to induce a model to write vulnerable code cause the model to respond harmfully on unrelated open-ended prompts. Finally, we use one-shot SV optimization to investigate how an instruction-tuned LLM recovers from outputting false information, and find that this ability is independent of the model's explicit verbalization that the information was false. Overall, our findings suggest that optimizing SVs on a single example can mediate a wide array of misaligned behaviors in LLMs. Code can be found at https://github.com/jacobdunefsky/one-shot-steering-repro and https://github.com/jacobdunefsky/one-shot-steering-misalignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20179v4">Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Code LLMs have shown promising results with converting tasks in natural language to programs that can be executed by service robots. We are interested in finetuning small, specialized LLMs for this purpose, but collecting datasets of task-program pairs specific to each robot is time-consuming and expensive. While approaches such as SELF-INSTRUCT and EVOL-INSTRUCT are capable of generating novel tasks given a few examples, they are unable to provide the corresponding programs that correctly abide by physical-world and robot-constraints using the provided programming interface. Using a simulator is a natural potential solution to checking for such constraints, but building simulation environments that can handle arbitrary tasks and their necessary objects and locations, is challenging. To address these challenges, we introduce ROBO-INSTRUCT, which synthesizes task-specific simulation environments on the fly during program execution, by opportunistically inferring entity properties and enforcing corresponding constraints based on how the entities are used in the task program. Additionally, ROBO-INSTRUCT integrates an LLM-aided post-processing procedure to refine instructions for better alignment with robot programs. We demonstrate the effectiveness of ROBO-INSTRUCT across multiple LLMs, showing that our fine-tuned models outperform all baseline methods and even match or surpass the performance of several larger and proprietary models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09332v1">Teaching Code Refactoring Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Accepted for presentation at the Frontiers in Education Conference, Nashville, Tennessee, USA, 2-5 November 2025
    </div>
    <details class="paper-abstract">
      This Innovative Practice full paper explores how Large Language Models (LLMs) can enhance the teaching of code refactoring in software engineering courses through real-time, context-aware feedback. Refactoring improves code quality but is difficult to teach, especially with complex, real-world codebases. Traditional methods like code reviews and static analysis tools offer limited, inconsistent feedback. Our approach integrates LLM-assisted refactoring into a course project using structured prompts to help students identify and address code smells such as long methods and low cohesion. Implemented in Spring 2025 in a long-lived OSS project, the intervention is evaluated through student feedback and planned analysis of code quality improvements. Findings suggest that LLMs can bridge theoretical and practical learning, supporting a deeper understanding of maintainability and refactoring principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06323v2">Mosaic: Composite Projection Pruning for Resource-efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models. Mosaic is available for public use from https://github.com/blessonvar/Mosaic
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02009v3">Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 10 pages, 5 figures. Accepted at the International Joint Conferences on Artificial Intelligence IJCAI 2025 (main track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for harmful content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. We share TTP, TTP-Eval, HAVOC and a sample of C4 inferenced on HarmFormer. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09303v1">ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Reasoning-augmented search agents such as Search-R1, trained via reinforcement learning with verifiable rewards (RLVR), demonstrate remarkable capabilities in multi-step information retrieval from external knowledge sources. These agents address the limitations of their parametric memory by dynamically gathering relevant facts to address complex reasoning tasks. However, existing approaches suffer from a fundamental architectural limitation: they process search queries strictly sequentially, even when handling inherently parallelizable and logically independent comparisons. This sequential bottleneck significantly constrains computational efficiency, particularly for queries that require multiple entity comparisons. To address this critical limitation, we propose ParallelSearch, a novel reinforcement learning framework that empowers large language models (LLMs) to recognize parallelizable query structures and execute multiple search operations concurrently. Our approach introduces dedicated reward functions that incentivize the identification of independent query components while preserving answer accuracy through jointly considering correctness, query decomposition quality, and parallel execution benefits. Comprehensive experiments demonstrate that ParallelSearch outperforms state-of-the-art baselines by an average performance gain of 2.9% across seven question-answering benchmarks. Notably, on parallelizable questions, our method achieves a 12.7% performance improvement while requiring only 69.6% of the LLM calls compared to sequential approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09288v1">Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 2 figures, 3 tables; code and certification harness: https://github.com/ayushgupta4897/Contextual-Integrity-Verification ; Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04125v4">VulScribeR: Exploring RAG-based Vulnerability Augmentation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 Accepted by TOSEM; 26 pages, 6 figures, 8 tables, 3 prompt templates, 1 algorithm
    </div>
    <details class="paper-abstract">
      Detecting vulnerabilities is vital for software security, yet deep learning-based vulnerability detectors (DLVD) face a data shortage, which limits their effectiveness. Data augmentation can potentially alleviate the data shortage, but augmenting vulnerable code is challenging and requires a generative solution that maintains vulnerability. Previous works have only focused on generating samples that contain single statements or specific types of vulnerabilities. Recently, large language models (LLMs) have been used to solve various code generation and comprehension tasks with inspiring results, especially when fused with retrieval augmented generation (RAG). Therefore, we propose VulScribeR, a novel LLM-based solution that leverages carefully curated prompt templates to augment vulnerable datasets. More specifically, we explore three strategies to augment both single and multi-statement vulnerabilities, with LLMs, namely Mutation, Injection, and Extension. Our extensive evaluation across four vulnerability datasets and DLVD models, using three LLMs, show that our approach beats two SOTA methods Vulgen and VGX, and Random Oversampling (ROS) by 27.48%, 27.93%, and 15.41% in f1-score with 5K generated vulnerable samples on average, and 53.84%, 54.10%, 69.90%, and 40.93% with 15K generated vulnerable samples. Our approach demonstrates its feasibility for large-scale data augmentation by generating 1K samples at as cheap as US$ 1.88.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09263v1">LLM Empowered Prototype Learning for Zero and Few-Shot Tasks on Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in large language models (LLMs) have opened the door to in-depth investigation of their potential in tabular data modeling. However, effectively utilizing advanced LLMs in few-shot and even zero-shot scenarios is still challenging. To this end, we propose a novel LLM-based prototype estimation framework for tabular learning. Our key idea is to query the LLM to generate feature values based example-free prompt, which solely relies on task and feature descriptions. With the feature values generated by LLM, we can build a zero-shot prototype in a training-free manner, which can be further enhanced by fusing few-shot samples, avoiding training a classifier or finetuning the LLMs. Thanks to the example-free prompt and prototype estimation, ours bypasses the constraints brought by the example-based prompt, providing a scalable and robust framework. Extensive experiments demonstrate the effectiveness of ours in zero and few-shot tabular learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09240v1">NEFMind: Parameter-Efficient Fine-Tuning of Open-Source LLMs for Telecom APIs Automation</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The use of Service-Based Architecture in modern telecommunications has exponentially increased Network Functions (NFs) and Application Programming Interfaces (APIs), creating substantial operational complexities in service discovery and management. We introduce \textit{NEFMind}, a framework leveraging parameter-efficient fine-tuning of open-source Large Language Models (LLMs) to address these challenges. It integrates three core components: synthetic dataset generation from Network Exposure Function (NEF) API specifications, model optimization through Quantized-Low-Rank Adaptation, and performance evaluation via GPT-4 Ref Score and BertScore metrics. Targeting 5G Service-Based Architecture APIs, our approach achieves 85% reduction in communication overhead compared to manual discovery methods. Experimental validation using the open-source Phi-2 model demonstrates exceptional API call identification performance at 98-100% accuracy. The fine-tuned Phi-2 model delivers performance comparable to significantly larger models like GPT-4 while maintaining computational efficiency for telecommunications infrastructure deployment. These findings validate domain-specific, parameter-efficient LLM strategies for managing complex API ecosystems in next-generation telecommunications networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10059v1">FormalGrad: Integrating Formal Methods with Gradient-Based LLM Refinement</a></div>
    <div class="paper-meta">
      📅 2025-08-12
      | 💬 6 Pages
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, they often produce solutions that lack guarantees of correctness, robustness, and efficiency. The limitation is acute in domains requiring strict constraints. FormalGrad introduces a principled framework that integrates formal methods directly into an iterative LLM-based generation loop. It uniquely treats code as a differentiable variable, converting structured feedback and formal constraints into a textual pseudo-gradient. This gradient guides the model to iteratively refine solutions, ensuring they are not only functional but also robust and formally justified. We evaluate FormalGrad on the HumanEval, HumanEval+, and LiveCodeBench benchmarks. Our implementation outperforms strong baselines, achieving an absolute improvement of up to 27% on HumanEval and a 41% relative improvement on the challenging LiveCodeBench V6. FormalGrad generates formally justified code that is robust and efficient, paving the way for reliable AI-assisted software development in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10047v1">A Survey of Optimization Modeling Meets LLMs: Progress and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-08-12
    </div>
    <details class="paper-abstract">
      By virtue of its great utility in solving real-world problems, optimization modeling has been widely employed for optimal decision-making across various sectors, but it requires substantial expertise from operations research professionals. With the advent of large language models (LLMs), new opportunities have emerged to automate the procedure of mathematical modeling. This survey presents a comprehensive and timely review of recent advancements that cover the entire technical stack, including data synthesis and fine-tuning for the base model, inference frameworks, benchmark datasets, and performance evaluation. In addition, we conducted an in-depth analysis on the quality of benchmark datasets, which was found to have a surprisingly high error rate. We cleaned the datasets and constructed a new leaderboard with fair performance evaluation in terms of base LLM model and datasets. We also build an online portal that integrates resources of cleaned datasets, code and paper repository to benefit the community. Finally, we identify limitations in current methodologies and outline future research opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08243v1">Jinx: Unlimited LLMs for Probing Alignment Failures</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 https://huggingface.co/Jinx-org
    </div>
    <details class="paper-abstract">
      Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08242v1">Bringing Everyone to the Table: An Experimental Study of LLM-Facilitated Group Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Group decision-making often suffers from uneven information sharing, hindering decision quality. While large language models (LLMs) have been widely studied as aids for individuals, their potential to support groups of users, potentially as facilitators, is relatively underexplored. We present a pre-registered randomized experiment with 1,475 participants assigned to 281 five-person groups completing a hidden profile task--selecting an optimal city for a hypothetical sporting event--under one of four facilitation conditions: no facilitation, a one-time message prompting information sharing, a human facilitator, or an LLM (GPT-4o) facilitator. We find that LLM facilitation increases information shared within a discussion by raising the minimum level of engagement with the task among group members, and that these gains come at limited cost in terms of participants' attitudes towards the task, their group, or their facilitator. Whether by human or AI, there is no significant effect of facilitation on the final decision outcome, suggesting that even substantial but partial increases in information sharing are insufficient to overcome the hidden profile effect studied. To support further research into how LLM-based interfaces can support the future of collaborative decision making, we release our experimental platform, the Group-AI Interaction Laboratory (GRAIL), as an open-source tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08236v1">Exploring Safety Alignment Evaluation of LLMs in Chinese Mental Health Dialogues via LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Evaluating the safety alignment of LLM responses in high-risk mental health dialogues is particularly difficult due to missing gold-standard answers and the ethically sensitive nature of these interactions. To address this challenge, we propose PsyCrisis-Bench, a reference-free evaluation benchmark based on real-world Chinese mental health dialogues. It evaluates whether the model responses align with the safety principles defined by experts. Specifically designed for settings without standard references, our method adopts a prompt-based LLM-as-Judge approach that conducts in-context evaluation using expert-defined reasoning chains grounded in psychological intervention principles. We employ binary point-wise scoring across multiple safety dimensions to enhance the explainability and traceability of the evaluation. Additionally, we present a manually curated, high-quality Chinese-language dataset covering self-harm, suicidal ideation, and existential distress, derived from real-world online discourse. Experiments on 3600 judgments show that our method achieves the highest agreement with expert assessments and produces more interpretable evaluation rationales compared to existing approaches. Our dataset and evaluation tool are publicly available to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08221v1">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 26 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09373v2">QUDsim: Quantifying Discourse Similarities in LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 COLM 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      As large language models become increasingly capable at various writing tasks, their weakness at generating unique and creative content becomes a major liability. Although LLMs have the ability to generate text covering diverse topics, there is an overall sense of repetitiveness across texts that we aim to formalize and quantify via a similarity metric. The familiarity between documents arises from the persistence of underlying discourse structures. However, existing similarity metrics dependent on lexical overlap and syntactic patterns largely capture $\textit{content}$ overlap, thus making them unsuitable for detecting $\textit{structural}$ similarities. We introduce an abstraction based on linguistic theories in Questions Under Discussion (QUD) and question semantics to help quantify differences in discourse progression. We then use this framework to build $\textbf{QUDsim}$, a similarity metric that can detect discursive parallels between documents. Using QUDsim, we find that LLMs often reuse discourse structures (more so than humans) across samples, even when content differs. Furthermore, LLMs are not only repetitive and structurally uniform, but are also divergent from human authors in the types of structures they use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08211v1">SAEMark: Multi-bit LLM Watermarking with Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 24 pages, 12 figures, code available: https://zhuohaoyu.github.io/SAEMark
    </div>
    <details class="paper-abstract">
      Watermarking LLM-generated text is critical for content attribution and misinformation prevention. However, existing methods compromise text quality, require white-box model access and logit manipulation. These limitations exclude API-based models and multilingual scenarios. We propose SAEMark, a general framework for post-hoc multi-bit watermarking that embeds personalized messages solely via inference-time, feature-based rejection sampling without altering model logits or requiring training. Our approach operates on deterministic features extracted from generated text, selecting outputs whose feature statistics align with key-derived targets. This framework naturally generalizes across languages and domains while preserving text quality through sampling LLM outputs instead of modifying. We provide theoretical guarantees relating watermark success probability and compute budget that hold for any suitable feature extractor. Empirically, we demonstrate the framework's effectiveness using Sparse Autoencoders (SAEs), achieving superior detection accuracy and text quality. Experiments across 4 datasets show SAEMark's consistent performance, with 99.7% F1 on English and strong multi-bit detection accuracy. SAEMark establishes a new paradigm for scalable watermarking that works out-of-the-box with closed-source LLMs while enabling content attribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17130v3">Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector. Our code is publicly available at: https://github.com/hannahxchen/llm-censorship-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02904v2">How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. Our code is publicly available at https://github.com/HZD01/post-training-mechanistic-analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08171v1">PyVeritas: On Verifying Python via LLM-Based Transpilation and Bounded Model Checking for C</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 14 pages, 6 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Python has become the dominant language for general-purpose programming, yet it lacks robust tools for formal verification. In contrast, programmers working in languages such as C benefit from mature model checkers, for example CBMC, which enable exhaustive symbolic reasoning and fault localisation. The inherent complexity of Python, coupled with the verbosity and low-level nature of existing transpilers (e.g., Cython), have historically limited the applicability of formal verification to Python programs. In this paper, we propose PyVeritas, a novel framework that leverages Large Language Models (LLMs) for high-level transpilation from Python to C, followed by bounded model checking and MaxSAT-based fault localisation in the generated C code. PyVeritas enables verification and bug localisation for Python code using existing model checking tools for C. Our empirical evaluation on two Python benchmarks demonstrates that LLM-based transpilation can achieve a high degree of accuracy, up to 80--90% for some LLMs, enabling effective development environment that supports assertion-based verification and interpretable fault diagnosis for small yet non-trivial Python programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23701v2">TextQuests: How Good are LLMs at Text-Based Video Games?</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at https://textquests.ai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08147v1">From Natural Language to Solver-Ready Power System Optimization: An LLM-Assisted, Validation-in-the-Loop Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Large Language Models (LLMs)-assisted agent that automatically converts natural-language descriptions of power system optimization scenarios into compact, solver-ready formulations and generates corresponding solutions. In contrast to approaches that rely solely on LLM to produce solutions directly, the proposed method focuses on discovering a mathematically compatible formulation that can be efficiently solved by off-the-shelf optimization solvers. Directly using LLMs to produce solutions often leads to infeasible or suboptimal results, as these models lack the numerical precision and constraint-handling capabilities of established optimization solvers. The pipeline integrates a domain-aware prompt and schema with an LLM, enforces feasibility through systematic validation and iterative repair, and returns both solver-ready models and user-facing results. Using the unit commitment problem as a representative case study, the agent produces optimal or near-optimal schedules along with the associated objective costs. Results demonstrate that coupling the solver with task-specific validation significantly enhances solution reliability. This work shows that combining AI with established optimization frameworks bridges high-level problem descriptions and executable mathematical models, enabling more efficient decision-making in energy systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08139v1">Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08127v1">BlindGuard: Safeguarding LLM-based Multi-Agent Systems under Unknown Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The security of LLM-based multi-agent systems (MAS) is critically threatened by propagation vulnerability, where malicious agents can distort collective decision-making through inter-agent message interactions. While existing supervised defense methods demonstrate promising performance, they may be impractical in real-world scenarios due to their heavy reliance on labeled malicious agents to train a supervised malicious detection model. To enable practical and generalizable MAS defenses, in this paper, we propose BlindGuard, an unsupervised defense method that learns without requiring any attack-specific labels or prior knowledge of malicious behaviors. To this end, we establish a hierarchical agent encoder to capture individual, neighborhood, and global interaction patterns of each agent, providing a comprehensive understanding for malicious agent detection. Meanwhile, we design a corruption-guided detector that consists of directional noise injection and contrastive learning, allowing effective detection model training solely on normal agent behaviors. Extensive experiments show that BlindGuard effectively detects diverse attack types (i.e., prompt injection, memory poisoning, and tool attack) across MAS with various communication patterns while maintaining superior generalizability compared to supervised baselines. The code is available at: https://github.com/MR9812/BlindGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08120v1">Vision-Based Localization and LLM-based Navigation for Indoor Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 20 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Indoor navigation remains a complex challenge due to the absence of reliable GPS signals and the architectural intricacies of large enclosed environments. This study presents an indoor localization and navigation approach that integrates vision-based localization with large language model (LLM)-based navigation. The localization system utilizes a ResNet-50 convolutional neural network fine-tuned through a two-stage process to identify the user's position using smartphone camera input. To complement localization, the navigation module employs an LLM, guided by a carefully crafted system prompt, to interpret preprocessed floor plan images and generate step-by-step directions. Experimental evaluation was conducted in a realistic office corridor with repetitive features and limited visibility to test localization robustness. The model achieved high confidence and an accuracy of 96% across all tested waypoints, even under constrained viewing conditions and short-duration queries. Navigation tests using ChatGPT on real building floor maps yielded an average instruction accuracy of 75%, with observed limitations in zero-shot reasoning and inference time. This research demonstrates the potential for scalable, infrastructure-free indoor navigation using off-the-shelf cameras and publicly available floor plans, particularly in resource-constrained settings like hospitals, airports, and educational institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08115v1">TeamMedAgents: Enhancing Medical Decision-Making of LLMs Through Structured Teamwork</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 10 pages, 1 figure, 6 tables(2 in main, 4 in appendix)
    </div>
    <details class="paper-abstract">
      We present TeamMedAgents, a novel multi-agent approach that systematically integrates evidence-based teamwork components from human-human collaboration into medical decision-making with large language models (LLMs). Our approach validates an organizational psychology teamwork model from human collaboration to computational multi-agent medical systems by operationalizing six core teamwork components derived from Salas et al.'s "Big Five" model: team leadership, mutual performance monitoring, team orientation, shared mental models, closed-loop communication, and mutual trust. We implement and evaluate these components as modular, configurable mechanisms within an adaptive collaboration architecture while assessing the effect of the number of agents involved based on the task's requirements and domain. Systematic evaluation of computational implementations of teamwork behaviors across eight medical benchmarks (MedQA, MedMCQA, MMLU-Pro Medical, PubMedQA, DDXPlus, MedBullets, Path-VQA, and PMC-VQA) demonstrates consistent improvements across 7 out of 8 evaluated datasets. Controlled ablation studies conducted on 50 questions per configuration across 3 independent runs provide mechanistic insights into individual component contributions, revealing optimal teamwork configurations that vary by reasoning task complexity and domain-specific requirements. Our ablation analyses reveal dataset-specific optimal teamwork configurations, indicating that different medical reasoning modalities benefit from distinct collaborative patterns. TeamMedAgents represents an advancement in collaborative AI by providing a systematic translation of established teamwork theories from human collaboration into agentic collaboration, establishing a foundation for evidence-based multi-agent system design in critical decision-making domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08096v1">Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Preprint as provided by the authors (19 pages, 12 figures, 9 tables)
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) and their increased accessibility have made it easier than ever for students to automatically generate texts, posing new challenges for educational institutions. To enforce norms of academic integrity and ensure students' learning, learning analytics methods to automatically detect LLM-generated text appear increasingly appealing. This paper benchmarks the performance of different state-of-the-art detectors in educational contexts, introducing a novel dataset, called Generative Essay Detection in Education (GEDE), containing over 900 student-written essays and over 12,500 LLM-generated essays from various domains. To capture the diversity of LLM usage practices in generating text, we propose the concept of contribution levels, representing students' contribution to a given assignment. These levels range from purely human-written texts, to slightly LLM-improved versions, to fully LLM-generated texts, and finally to active attacks on the detector by "humanizing" generated texts. We show that most detectors struggle to accurately classify texts of intermediate student contribution levels, like LLM-improved human-written texts. Detectors are particularly likely to produce false positives, which is problematic in educational settings where false suspicions can severely impact students' lives. Our dataset, code, and additional supplementary materials are publicly available at https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08029v1">Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08027v1">Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Speech Recognition (ASR) due to phoneme distortions and high variability. While self-supervised ASR models like Wav2Vec, HuBERT, and Whisper have shown promise, their effectiveness in dysarthric speech remains unclear. This study systematically benchmarks these models with different decoding strategies, including CTC, seq2seq, and LLM-enhanced decoding (BART,GPT-2, Vicuna). Our contributions include (1) benchmarking ASR architectures for dysarthric speech, (2) introducing LLM-based decoding to improve intelligibility, (3) analyzing generalization across datasets, and (4) providing insights into recognition errors across severity levels. Findings highlight that LLM-enhanced decoding improves dysarthric ASR by leveraging linguistic constraints for phoneme restoration and grammatical correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08001v1">Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Rui Yao, Qi Chai, and Jinhai Yao contributed equally to this work. Corresponding authors: Qi Zhang (zhang.qi@sjtu.edu.cn) and Hao Wang (haowang@hkust-gz.edu.cn)
    </div>
    <details class="paper-abstract">
      "Fedspeak", the stylized and often nuanced language used by the U.S. Federal Reserve, encodes implicit policy signals and strategic stances. The Federal Open Market Committee strategically employs Fedspeak as a communication tool to shape market expectations and influence both domestic and global economic conditions. As such, automatically parsing and interpreting Fedspeak presents a high-impact challenge, with significant implications for financial forecasting, algorithmic trading, and data-driven policy analysis. In this paper, we propose an LLM-based, uncertainty-aware framework for deciphering Fedspeak and classifying its underlying monetary policy stance. Technically, to enrich the semantic and contextual representation of Fedspeak texts, we incorporate domain-specific reasoning grounded in the monetary policy transmission mechanism. We further introduce a dynamic uncertainty decoding module to assess the confidence of model predictions, thereby enhancing both classification accuracy and model reliability. Experimental results demonstrate that our framework achieves state-of-the-art performance on the policy stance analysis task. Moreover, statistical analysis reveals a significant positive correlation between perceptual uncertainty and model error rates, validating the effectiveness of perceptual uncertainty as a diagnostic signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17001v2">PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Personalizing language models by effectively incorporating user interaction history remains a central challenge in the development of adaptive AI systems. While large language models (LLMs) combined with Retrieval-Augmented Generation (RAG) have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graphs, automatically constructed and updated by the LLM itself, and capable of encoding information in multiple formats-including nodes, triplets, higher-order propositions, and episodic traces. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyperedges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle propagation, beam search, and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks-TriviaQA, HotpotQA, and DiaASQ-demonstrating that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10434v3">Spotter+GPT: Turning Sign Spottings into Sentences with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted at the 9th Workshop on Sign Language Translation and Avatar Technologies (SLTAT) in ACM International Conference on Intelligent Virtual Agents (IVA`25)
    </div>
    <details class="paper-abstract">
      Sign Language Translation (SLT) is a challenging task that aims to generate spoken language sentences from sign language videos. In this paper, we introduce a lightweight, modular SLT framework, Spotter+GPT, that leverages the power of Large Language Models (LLMs) and avoids heavy end-to-end training. Spotter+GPT breaks down the SLT task into two distinct stages. First, a sign spotter identifies individual signs within the input video. The spotted signs are then passed to an LLM, which transforms them into meaningful spoken language sentences. Spotter+GPT eliminates the requirement for SLT-specific training. This significantly reduces computational costs and time requirements. The source code and pretrained weights of the Spotter are available at https://gitlab.surrey.ac.uk/cogvispublic/sign-spotter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07935v1">SHIELDA: Structured Handling of Exceptions in LLM-Driven Agentic Workflows</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agentic systems are software systems powered by LLMs that autonomously reason, plan, and execute multi-step workflows to achieve human goals, rather than merely executing predefined steps. During execution, these workflows frequently encounter exceptions. Existing exception handling solutions often treat exceptions superficially, failing to trace execution-phase exceptions to their reasoning-phase root causes. Furthermore, their recovery logic is brittle, lacking structured escalation pathways when initial attempts fail. To tackle these challenges, we first present a comprehensive taxonomy of 36 exception types across 12 agent artifacts. Building on this, we propose SHIELDA (Structured Handling of Exceptions in LLM-Driven Agentic Workflows), a modular runtime exception handling framework for LLM agentic workflows. SHIELDA uses an exception classifier to select a predefined exception handling pattern from a handling pattern registry. These patterns are then executed via a structured handling executor, comprising local handling, flow control, and state recovery, to enable phase-aware recovery by linking exceptions to their root causes and facilitating composable strategies. We validate SHIELDA's effectiveness through a case study on the AutoPR agent, demonstrating effective, cross-phase recovery from a reasoning-induced exception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17066v3">Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02233v2">A Methodological Framework for LLM-Based Mining of Software Repositories</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software engineering research, offering new opportunities for automating repository mining tasks. However, despite their growing popularity, the methodological integration of LLMs into Mining Software Repositories (MSR) remains poorly understood. Existing studies tend to focus on specific capabilities or performance benchmarks, providing limited insight into how researchers utilize LLMs across the full research pipeline. To address this gap, we conduct a mixed-method study that combines a rapid review and questionnaire survey in the field of LLM4MSR. We investigate (1) the approaches and (2) the threats that affect the empirical rigor of researchers involved in this field. Our findings reveal 15 methodological approaches, nine main threats, and 25 mitigation strategies. Building on these findings, we present PRIMES 2.0, a refined empirical framework organized into six stages, comprising 23 methodological substeps, each mapped to specific threats and corresponding mitigation strategies, providing prescriptive and adaptive support throughout the lifecycle of LLM-based MSR studies. Our work contributes to establishing a more transparent and reproducible foundation for LLM-based MSR research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07885v1">Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06225v2">Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the Overconfidence Phenomenon in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce TH-Score, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose LLM-as-a-Fuser, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07849v1">LLMs for Law: Evaluating Legal-Specific LLMs on Contract Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Under review. 4 pages + references
    </div>
    <details class="paper-abstract">
      Despite advances in legal NLP, no comprehensive evaluation covering multiple legal-specific LLMs currently exists for contract classification tasks in contract understanding. To address this gap, we present an evaluation of 10 legal-specific LLMs on three English language contract understanding tasks and compare them with 7 general-purpose LLMs. The results show that legal-specific LLMs consistently outperform general-purpose models, especially on tasks requiring nuanced legal understanding. Legal-BERT and Contracts-BERT establish new SOTAs on two of the three tasks, despite having 69% fewer parameters than the best-performing general-purpose LLM. We also identify CaseLaw-BERT and LexLM as strong additional baselines for contract understanding. Our results provide a holistic evaluation of legal-specific LLMs and will facilitate the development of more accurate contract understanding systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13380v3">DAGR: Decomposition Augmented Graph Retrieval with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many Natural Language Processing (NLP) tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To address this challenge, we introduce DAGR, a retrieval method that leverages both complex questions and their decomposition in subquestions to extract relevant, linked textual subgraphs. DAGR first breaks down complex queries, retrieves subgraphs guided by a weighted similarity function over both the original and decomposed queries, and creates a question-specific knowledge graph to guide answer generation. The resulting Graph-RAG pipeline is suited to handle complex multi-hop questions and effectively reason over graph-structured data. We evaluate DAGR on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07583v3">Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at https://github.com/deep-spin/treqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07805v1">Can You Trick the Grader? Adversarial Persuasion of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      As large language models take on growing roles as automated evaluators in practical settings, a critical question arises: Can individuals persuade an LLM judge to assign unfairly high scores? This study is the first to reveal that strategically embedded persuasive language can bias LLM judges when scoring mathematical reasoning tasks, where correctness should be independent of stylistic variation. Grounded in Aristotle's rhetorical principles, we formalize seven persuasion techniques (Majority, Consistency, Flattery, Reciprocity, Pity, Authority, Identity) and embed them into otherwise identical responses. Across six math benchmarks, we find that persuasive language leads LLM judges to assign inflated scores to incorrect solutions, by up to 8% on average, with Consistency causing the most severe distortion. Notably, increasing model size does not substantially mitigate this vulnerability. Further analysis demonstrates that combining multiple persuasion techniques amplifies the bias, and pairwise evaluation is likewise susceptible. Moreover, the persuasive effect persists under counter prompting strategies, highlighting a critical vulnerability in LLM-as-a-Judge pipelines and underscoring the need for robust defenses against persuasion-based attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07785v1">Grove MoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      The Mixture of Experts (MoE) architecture is a cornerstone of modern state-of-the-art (SOTA) large language models (LLMs). MoE models facilitate scalability by enabling sparse parameter activation. However, traditional MoE architecture uses homogeneous experts of a uniform size, activating a fixed number of parameters irrespective of input complexity and thus limiting computational efficiency. To overcome this limitation, we introduce Grove MoE, a novel architecture incorporating experts of varying sizes, inspired by the heterogeneous big.LITTLE CPU architecture. This architecture features novel adjugate experts with a dynamic activation mechanism, enabling model capacity expansion while maintaining manageable computational overhead. Building on this architecture, we present GroveMoE-Base and GroveMoE-Inst, 33B-parameter LLMs developed by applying an upcycling strategy to the Qwen3-30B-A3B-Base model during mid-training and post-training. GroveMoE models dynamically activate 3.14-3.28B parameters based on token complexity and achieve performance comparable to SOTA open-source models of similar or even larger size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07781v1">SASST: Leveraging Syntax-Aware Chunking and LLMs for Simultaneous Speech Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      This work proposes a grammar-based chunking strategy that segments input streams into semantically complete units by parsing dependency relations (e.g., noun phrase boundaries, verb-object structures) and punctuation features. The method ensures chunk coherence and minimizes semantic fragmentation. Building on this mechanism, we present SASST (Syntax-Aware Simultaneous Speech Translation), an end-to-end framework integrating frozen Whisper encoder and decoder-only LLM. The unified architecture dynamically outputs translation tokens or <WAIT> symbols to jointly optimize translation timing and content, with target-side reordering addressing word-order divergence. Experiments on CoVoST2 multilingual corpus En-{De, Zh, Ja} demonstrate significant translation quality improvements across languages and validate the effectiveness of syntactic structures in LLM-driven SimulST systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16633v3">POEX: Towards Policy Executable Jailbreak Attacks Against the LLM-based Robots</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Homepage: https://poex-jailbreak.github.io/
    </div>
    <details class="paper-abstract">
      The integration of LLMs into robots has witnessed significant growth, where LLMs can convert instructions into executable robot policies. However, the inherent vulnerability of LLMs to jailbreak attacks brings critical security risks from the digital domain to the physical world. An attacked LLM-based robot could execute harmful policies and cause physical harm. In this paper, we investigate the feasibility and rationale of jailbreak attacks against LLM-based robots and answer three research questions: (1) How applicable are existing LLM jailbreak attacks against LLM-based robots? (2) What unique challenges arise if they are not directly applicable? (3) How to defend against such jailbreak attacks? To this end, we first construct a "human-object-environment" robot risks-oriented Harmful-RLbench and then conduct a measurement study on LLM-based robot systems. Our findings conclude that traditional LLM jailbreak attacks are inapplicable in robot scenarios, and we identify two unique challenges: determining policy-executable optimization directions and accurately evaluating robot-jailbroken policies. To enable a more thorough security analysis, we introduce POEX (POlicy EXecutable) jailbreak, a red-teaming framework that induces harmful yet executable policy to jailbreak LLM-based robots. POEX incorporates hidden layer gradient optimization to guarantee jailbreak success and policy execution as well as a multi-agent evaluator to accurately assess the practical executability of policies. Experiments conducted on the real-world robotic systems and in simulation demonstrate the efficacy of POEX, highlighting critical security vulnerabilities and its transferability across LLMs. Finally, we propose prompt-based and model-based defenses to mitigate attacks. Our findings underscore the urgent need for security measures to ensure the safe deployment of LLM-based robots in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07745v1">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Insider threats, which can lead to severe losses, remain a major security concern. While machine learning-based insider threat detection (ITD) methods have shown promising results, their progress is hindered by the scarcity of high-quality data. Enterprise data is sensitive and rarely accessible, while publicly available datasets, when limited in scale due to cost, lack sufficient real-world coverage; and when purely synthetic, they fail to capture rich semantics and realistic user behavior. To address this, we propose Chimera, the first large language model (LLM)-based multi-agent framework that automatically simulates both benign and malicious insider activities and collects diverse logs across diverse enterprise environments. Chimera models each employee with agents that have role-specific behavior and integrates modules for group meetings, pairwise interactions, and autonomous scheduling, capturing realistic organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP theft, system sabotage) and has been deployed to simulate activities in three sensitive domains: technology company, finance corporation, and medical institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via human studies and quantitative analysis, confirming its diversity, realism, and presence of explainable threat patterns. Evaluations of existing ITD methods show an average F1-score of 0.83, which is significantly lower than 0.99 on the CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for advancing ITD research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16155v2">CITYWALK: Enhancing LLM-Based C++ Unit Test Generation via Project-Dependency Awareness and Language-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Preprint, to appear in the ACM Transactions on Software Engineering and Methodology (TOSEM)
    </div>
    <details class="paper-abstract">
      Unit testing plays a pivotal role in the software development lifecycle, as it ensures code quality. However, writing high-quality unit tests remains a time-consuming task for developers in practice. More recently, the application of large language models (LLMs) in automated unit test generation has demonstrated promising results. Existing approaches primarily focus on interpreted programming languages (e.g., Java), while mature solutions tailored to compiled programming languages like C++ are yet to be explored. The intricate language features of C++, such as pointers, templates, and virtual functions, pose particular challenges for LLMs in generating both executable and high-coverage unit tests. To tackle the aforementioned problems, this paper introduces CITYWALK, a novel LLM-based framework for C++ unit test generation. CITYWALK enhances LLMs by providing a comprehensive understanding of the dependency relationships within the project under test via program analysis. Furthermore, CITYWALK incorporates language-specific knowledge about C++ derived from project documentation and empirical observations, significantly improving the correctness of the LLM-generated unit tests. We implement CITYWALK by employing the widely popular LLM GPT-4o. The experimental results show that CITYWALK outperforms current state-of-the-art approaches on a collection of ten popular C++ projects. Our findings demonstrate the effectiveness of CITYWALK in generating high-quality C++ unit tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03733v2">WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      LLM-based agents have demonstrated great potential in generating and managing code within complex codebases. In this paper, we introduce WebGen-Bench, a novel benchmark designed to measure an LLM-based agent's ability to create multi-file website codebases from scratch. It contains diverse instructions for website generation, created through the combined efforts of human annotators and GPT-4o. These instructions span three major categories and thirteen minor categories, encompassing nearly all important types of web applications. To assess the quality of the generated websites, we use GPT-4o to generate test cases targeting each functionality described in the instructions, and then manually filter, adjust, and organize them to ensure accuracy, resulting in 647 test cases. Each test case specifies an operation to be performed on the website and the expected result after the operation. To automate testing and improve reproducibility, we employ a powerful web-navigation agent to execute tests on the generated websites and determine whether the observed responses align with the expected results. We evaluate three high-performance code-agent frameworks, Bolt.diy, OpenHands, and Aider, using multiple proprietary and open-source LLMs as engines. The best-performing combination, Bolt.diy powered by DeepSeek-R1, achieves only 27.8\% accuracy on the test cases, highlighting the challenging nature of our benchmark. Additionally, we construct WebGen-Instruct, a training set consisting of 6,667 website-generation instructions. Training Qwen2.5-Coder-32B-Instruct on Bolt.diy trajectories generated from a subset of this training set achieves an accuracy of 38.2\%, surpassing the performance of the best proprietary model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07675v1">Semantic Caching for Low-Cost LLM Serving: From Offline Learning to Online Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing how users interact with information systems, yet their high inference cost poses serious scalability and sustainability challenges. Caching inference responses, allowing them to be retrieved without another forward pass through the LLM, has emerged as one possible solution. Traditional exact-match caching, however, overlooks the semantic similarity between queries, leading to unnecessary recomputation. Semantic caching addresses this by retrieving responses based on semantic similarity, but introduces a fundamentally different cache eviction problem: one must account for mismatch costs between incoming queries and cached responses. Moreover, key system parameters, such as query arrival probabilities and serving costs, are often unknown and must be learned over time. Existing semantic caching methods are largely ad-hoc, lacking theoretical foundations and unable to adapt to real-world uncertainty. In this paper, we present a principled, learning-based framework for semantic cache eviction under unknown query and cost distributions. We formulate both offline optimization and online learning variants of the problem, and develop provably efficient algorithms with state-of-the-art guarantees. We also evaluate our framework on a synthetic dataset, showing that our proposed algorithms perform matching or superior performance compared with baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07667v1">1-2-3 Check: Enhancing Contextual Privacy in LLM via Multi-Agent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Addressing contextual privacy concerns remains challenging in interactive settings where large language models (LLMs) process information from multiple sources (e.g., summarizing meetings with private and public information). We introduce a multi-agent framework that decomposes privacy reasoning into specialized subtasks (extraction, classification), reducing the information load on any single agent while enabling iterative validation and more reliable adherence to contextual privacy norms. To understand how privacy errors emerge and propagate, we conduct a systematic ablation over information-flow topologies, revealing when and why upstream detection mistakes cascade into downstream leakage. Experiments on the ConfAIde and PrivacyLens benchmark with several open-source and closed-sourced LLMs demonstrate that our best multi-agent configuration substantially reduces private information leakage (\textbf{18\%} on ConfAIde and \textbf{19\%} on PrivacyLens with GPT-4o) while preserving the fidelity of public content, outperforming single-agent baselines. These results highlight the promise of principled information-flow design in multi-agent systems for contextual privacy with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07664v1">Understanding Users' Privacy Perceptions Towards LLM's RAG-based Memory</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrating memory functionalities to provide personalized and context-aware interactions. However, user understanding, practices and expectations regarding these memory systems are not yet well understood. This paper presents a thematic analysis of semi-structured interviews with 18 users to explore their mental models of LLM's Retrieval Augmented Generation (RAG)-based memory, current usage practices, perceived benefits and drawbacks, privacy concerns and expectations for future memory systems. Our findings reveal diverse and often incomplete mental models of how memory operates. While users appreciate the potential for enhanced personalization and efficiency, significant concerns exist regarding privacy, control and the accuracy of remembered information. Users express a desire for granular control over memory generation, management, usage and updating, including clear mechanisms for reviewing, editing, deleting and categorizing memories, as well as transparent insight into how memories and inferred information are used. We discuss design implications for creating more user-centric, transparent, and trustworthy LLM memory systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07638v1">Beyond Single: A Data Selection Principle for LLM Alignment via Fine-Grained Preference Signals</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Aligning Large Language Models (LLMs) with diverse human values requires moving beyond a single holistic "better-than" preference criterion. While collecting fine-grained, aspect-specific preference data is more reliable and scalable, existing methods like Direct Preference Optimization (DPO) struggle with the severe noise and conflicts inherent in such aggregated datasets. In this paper, we tackle this challenge from a data-centric perspective. We first derive the Direct Multi-Preference Optimization (DMPO) objective, and uncover a key Preference Divergence (PD) term that quantifies inter-aspect preference conflicts. Instead of using this term for direct optimization, we leverage it to formulate a novel, theoretically-grounded data selection principle. Our principle advocates for selecting a subset of high-consensus data-identified by the most negative PD values-for efficient DPO training. We prove the optimality of this strategy by analyzing the loss bounds of the DMPO objective in the selection problem. To operationalize our approach, we introduce practical methods of PD term estimation and length bias mitigation, thereby proposing our PD selection method. Evaluation on the UltraFeedback dataset with three varying conflict levels shows that our simple yet effective strategy achieves over 10% relative improvement against both the standard holistic preference and a stronger oracle using aggregated preference signals, all while boosting training efficiency and obviating the need for intractable holistic preference annotating, unlocking the potential of robust LLM alignment via fine-grained preference signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06387v2">End-to-End Text-to-SQL with Dataset Selection: Leveraging LLMs for Adaptive Query Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted in IJCNN25
    </div>
    <details class="paper-abstract">
      Text-to-SQL bridges the gap between natural language and structured database language, thus allowing non-technical users to easily query databases. Traditional approaches model text-to-SQL as a direct translation task, where a given Natural Language Query (NLQ) is mapped to an SQL command. Recent advances in large language models (LLMs) have significantly improved translation accuracy, however, these methods all require that the target database is pre-specified. This becomes problematic in scenarios with multiple extensive databases, where identifying the correct database becomes a crucial yet overlooked step. In this paper, we propose a three-stage end-to-end text-to-SQL framework to identify the user's intended database before generating SQL queries. Our approach leverages LLMs and prompt engineering to extract implicit information from natural language queries (NLQs) in the form of a ruleset. We then train a large db\_id prediction model, which includes a RoBERTa-based finetuned encoder, to predict the correct Database identifier (db\_id) based on both the NLQ and the LLM-generated rules. Finally, we refine the generated SQL by using critic agents to correct errors. Experimental results demonstrate that our framework outperforms the current state-of-the-art models in both database intent prediction and SQL generation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08500v3">Exploring Spatial Representation to Enhance LLM Reasoning in Aerial Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. However, it remains challenging due to the complex spatial relationships in aerial scenes.In this paper, we propose a training-free, zero-shot framework for aerial VLN tasks, where the large language model (LLM) is leveraged as the agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning capabilities of LLMs. This is achieved by extracting and projecting instruction-related semantic masks onto a top-down map, which presents spatial and topological information about surrounding landmarks and grows during the navigation process. At each step, a local map centered at the UAV is extracted from the growing top-down map, and transformed into a ma trix representation with distance metrics, serving as the text prompt to LLM for action prediction in response to the given instruction. Experiments conducted in real and simulation environments have proved the effectiveness and robustness of our method, achieving absolute success rate improvements of 26.8% and 5.8% over current state-of-the-art methods on simple and complex navigation tasks, respectively. The dataset and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14860v2">ALFA: Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 29 pages, 8 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decision-making. We present ALignment via Fine-grained Attributes, (ALFA) a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SoTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16646v3">SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine reasoning or superficial pattern recognition. Common evaluation methods, which focus on the either the final answer or the reasoning process, fail to assess the entire problem-solving procedure. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework, together with its corresponding benchmark, SMART-Bench. SMART decomposes the entire problem solving process into four distinct cognitive dimensions: Understanding, Reasoning, Arithmetic, and Reflection \& Refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings reveal genuine weaknesses in current LLMs and motivate a new metric, the All-Pass Score, to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07534v1">From Trial-and-Error to Improvement: A Systematic Analysis of LLM Exploration Mechanisms in RLVR</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 27pages,25figures. arXiv admin note: text overlap with arXiv:2508.02260
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs). Unlike traditional RL approaches, RLVR leverages rule-based feedback to guide LLMs in generating and refining complex reasoning chains -- a process critically dependent on effective exploration strategies. While prior work has demonstrated RLVR's empirical success, the fundamental mechanisms governing LLMs' exploration behaviors remain underexplored. This technical report presents a systematic investigation of exploration capacities in RLVR, covering four main aspects: (1) exploration space shaping, where we develop quantitative metrics to characterize LLMs' capability boundaries; (2) entropy-performance exchange, analyzed across training stages, individual instances, and token-level patterns; and (3) RL performance optimization, examining methods to effectively translate exploration gains into measurable improvements. By unifying previously identified insights with new empirical evidence, this work aims to provide a foundational framework for advancing RLVR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07517v1">Word Clouds as Common Voices: LLM-Assisted Visualization of Participant-Weighted Themes in Qualitative Interviews</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Word clouds are a common way to summarize qualitative interviews, yet traditional frequency-based methods often fail in conversational contexts: they surface filler words, ignore paraphrase, and fragment semantically related ideas. This limits their usefulness in early-stage analysis, when researchers need fast, interpretable overviews of what participant actually said. We introduce ThemeClouds, an open-source visualization tool that uses large language models (LLMs) to generate thematic, participant-weighted word clouds from dialogue transcripts. The system prompts an LLM to identify concept-level themes across a corpus and then counts how many unique participants mention each topic, yielding a visualization grounded in breadth of mention rather than raw term frequency. Researchers can customize prompts and visualization parameters, providing transparency and control. Using interviews from a user study comparing five recording-device configurations (31 participants; 155 transcripts, Whisper ASR), our approach surfaces more actionable device concerns than frequency clouds and topic-modeling baselines (e.g., LDA, BERTopic). We discuss design trade-offs for integrating LLM assistance into qualitative workflows, implications for interpretability and researcher agency, and opportunities for interactive analyses such as per-condition contrasts (``diff clouds'').
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08512v1">Using LLMs to Capture Users' Temporal Context for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Effective recommender systems demand dynamic user understanding, especially in complex, evolving environments. Traditional user profiling often fails to capture the nuanced, temporal contextual factors of user preferences, such as transient short-term interests and enduring long-term tastes. This paper presents an assessment of Large Language Models (LLMs) for generating semantically rich, time-aware user profiles. We do not propose a novel end-to-end recommendation architecture; instead, the core contribution is a systematic investigation into the degree of LLM effectiveness in capturing the dynamics of user context by disentangling short-term and long-term preferences. This approach, framing temporal preferences as dynamic user contexts for recommendations, adaptively fuses these distinct contextual components into comprehensive user embeddings. The evaluation across Movies&TV and Video Games domains suggests that while LLM-generated profiles offer semantic depth and temporal structure, their effectiveness for context-aware recommendations is notably contingent on the richness of user interaction histories. Significant gains are observed in dense domains (e.g., Movies&TV), whereas improvements are less pronounced in sparse environments (e.g., Video Games). This work highlights LLMs' nuanced potential in enhancing user profiling for adaptive, context-aware recommendations, emphasizing the critical role of dataset characteristics for practical applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08504v1">When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World Lessons from LLM Co-Design in a Safety-Net Hospital</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have the potential to address social and behavioral determinants of health by transforming labor intensive workflows in resource-constrained settings. Creating LLM-based applications that serve the needs of underserved communities requires a deep understanding of their local context, but it is often the case that neither LLMs nor their developers possess this local expertise, and the experts in these communities often face severe time/resource constraints. This creates a disconnect: how can one engage in meaningful co-design of an LLM-based application for an under-resourced community when the communication channel between the LLM developer and domain expert is constrained? We explored this question through a real-world case study, in which our data science team sought to partner with social workers at a safety net hospital to build an LLM application that summarizes patients' social needs. Whereas prior works focus on the challenge of prompt tuning, we found that the most critical challenge in this setting is the careful and precise specification of \what information to surface to providers so that the LLM application is accurate, comprehensive, and verifiable. Here we present a novel co-design framework for settings with limited access to domain experts, in which the summary generation task is first decomposed into individually-optimizable attributes and then each attribute is efficiently refined and validated through a multi-tier cascading approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08466v1">Enhancing Small LLM Alignment through Margin-Based Objective Modifications under Resource Constraints</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Small large language models (LLMs) often face difficulties in aligning output to human preferences, particularly when operating under severe performance gaps. In this work, we propose two lightweight DPO-based variants -- Adaptive Margin-Sigmoid Loss and APO-hinge-zero -- to better address underperformance scenarios by introducing margin-based objectives and selective update mechanisms. Our APO-hinge-zero method, which combines hinge-induced hard-example mining with the chosen-focused optimization of APO-zero, achieves strong results. In AlpacaEval, APO-hinge-zero improves the win rate by +2.0 points and the length-controlled win rate by +1.4 points compared to the APO-zero baseline. In MT-Bench, our methods maintain competitive performance in diverse categories, particularly excelling in STEM and Humanities tasks. These results demonstrate that simple modifications to preference-based objectives can significantly enhance small LLM alignment under resource constraints, offering a practical path toward more efficient deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08457v1">Architecting Long-Context LLM Acceleration with Packing-Prefetch Scheduler and Ultra-Large Capacity On-Chip Memories</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 7 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Long-context Large Language Model (LLM) inference faces increasing compute bottlenecks as attention calculations scale with context length, primarily due to the growing KV-cache transfer overhead that saturates High Bandwidth Memory (HBM). While prefetching techniques mitigate cache misses by fetching KV data in advance, their spatial and temporal benefits present new opportunities to exploit. This work proposes a packing-prefetch scheduling architecture with monolithic 3D (M3D) back-end-of-line (BEOL) compatible embedded memories with ultra-large on-chip capacity to accelerate long-context LLM inference. Our optimizations demonstrate 8.06x decode speedup and 1.83x overall latency reduction on Llama3.1-8B using TPUv6e-like hardware with additional 512MB BEOL memories over the serial execution. Evaluations of multi-request workloads on TPU-like architectures show 1.7x-2.4x throughput improvement and 1.5x-2.4x HBM bandwidth reduction compared to packing-only methods on Llama3.1-8B and Llama3.1-70B models. With the co-design of packing, prefetching, and BEOL memories, our approach alleviates HBM constraints and enables efficient long-context LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08454v1">Temporal User Profiling with LLMs: Balancing Short-Term and Long-Term Preferences for Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Accurately modeling user preferences is crucial for improving the performance of content-based recommender systems. Existing approaches often rely on simplistic user profiling methods, such as averaging or concatenating item embeddings, which fail to capture the nuanced nature of user preference dynamics, particularly the interactions between long-term and short-term preferences. In this work, we propose LLM-driven Temporal User Profiling (LLM-TUP), a novel method for user profiling that explicitly models short-term and long-term preferences by leveraging interaction timestamps and generating natural language representations of user histories using a large language model (LLM). These representations are encoded into high-dimensional embeddings using a pre-trained BERT model, and an attention mechanism is applied to dynamically fuse the short-term and long-term embeddings into a comprehensive user profile. Experimental results on real-world datasets demonstrate that LLM-TUP achieves substantial improvements over several baselines, underscoring the effectiveness of our temporally aware user-profiling approach and the use of semantically rich user profiles, generated by LLMs, for personalized content-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00043v2">CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-11
    </div>
    <details class="paper-abstract">
      Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly assess either text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles -- a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in two formats (text and image), supports adjustable difficulty through prefill ratio control, and offers different evaluation strategies, ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs substantially outperform non-reasoning models by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings highlight limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08438v1">Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 17 pages,17 figures
    </div>
    <details class="paper-abstract">
      Global KV-cache sharing has emerged as a key optimization for accelerating large language model (LLM) inference. However, it exposes a new class of timing side-channel attacks, enabling adversaries to infer sensitive user inputs via shared cache entries. Existing defenses, such as per-user isolation, eliminate leakage but degrade performance by up to 38.9% in time-to-first-token (TTFT), making them impractical for high-throughput deployment. To address this gap, we introduce SafeKV (Secure and Flexible KV Cache Sharing), a privacy-aware KV-cache management framework that selectively shares non-sensitive entries while confining sensitive content to private caches. SafeKV comprises three components: (i) a hybrid, multi-tier detection pipeline that integrates rule-based pattern matching, a general-purpose privacy detector, and context-aware validation; (ii) a unified radix-tree index that manages public and private entries across heterogeneous memory tiers (HBM, DRAM, SSD); and (iii) entropy-based access monitoring to detect and mitigate residual information leakage. Our evaluation shows that SafeKV mitigates 94% - 97% of timing-based side-channel attacks. Compared to per-user isolation method, SafeKV improves TTFT by up to 40.58% and throughput by up to 2.66X across diverse LLMs and workloads. SafeKV reduces cache-induced TTFT overhead from 50.41% to 11.74% on Qwen3-235B. By combining fine-grained privacy control with high cache reuse efficiency, SafeKV reclaims the performance advantages of global sharing while providing robust runtime privacy guarantees for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12899v3">DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Accepted by TPAMI
    </div>
    <details class="paper-abstract">
      Story visualization aims to create visually compelling images or videos corresponding to textual narratives. Despite recent advances in diffusion models yielding promising results, existing methods still struggle to create a coherent sequence of subject-consistent frames based solely on a story. To this end, we propose DreamStory, an automatic open-domain story visualization framework by leveraging the LLMs and a novel multi-subject consistent diffusion model. DreamStory consists of (1) an LLM acting as a story director and (2) an innovative Multi-Subject consistent Diffusion model (MSD) for generating consistent multi-subject across the images. First, DreamStory employs the LLM to generate descriptive prompts for subjects and scenes aligned with the story, annotating each scene's subjects for subsequent subject-consistent generation. Second, DreamStory utilizes these detailed subject descriptions to create portraits of the subjects, with these portraits and their corresponding textual information serving as multimodal anchors (guidance). Finally, the MSD uses these multimodal anchors to generate story scenes with consistent multi-subject. Specifically, the MSD includes Masked Mutual Self-Attention (MMSA) and Masked Mutual Cross-Attention (MMCA) modules. MMSA and MMCA modules ensure appearance and semantic consistency with reference images and text, respectively. Both modules employ masking mechanisms to prevent subject blending. To validate our approach and promote progress in story visualization, we established a benchmark, DS-500, which can assess the overall performance of the story visualization framework, subject-identification accuracy, and the consistency of the generation model. Extensive experiments validate the effectiveness of DreamStory in both subjective and objective evaluations. Please visit our project homepage at https://dream-xyz.github.io/dreamstory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08343v1">Maximizing GPU Efficiency via Optimal Adapter Caching: An Analytical Approach for Multi-Tenant LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-08-11
      | 💬 Under review for a computer science conference
    </div>
    <details class="paper-abstract">
      Serving LLM adapters has gained significant attention as an effective approach to adapt general-purpose language models to diverse, task-specific use cases. However, serving a wide range of adapters introduces several and substantial overheads, leading to performance degradation and challenges in optimal placement. To address these challenges, we present an analytical, AI-driven pipeline that accurately determines the optimal allocation of adapters in single-node setups. This allocation maximizes performance, effectively using GPU resources, while preventing request starvation. Crucially, the proposed allocation is given based on current workload patterns. These insights in single-node setups can be leveraged in multi-replica deployments for overall placement, load balancing and server configuration, ultimately enhancing overall performance and improving resource efficiency. Our approach builds on an in-depth analysis of LLM adapter serving, accounting for overheads and performance variability, and includes the development of the first Digital Twin capable of replicating online LLM-adapter serving systems with matching key performance metrics. The experimental results demonstrate that the Digital Twin achieves a SMAPE difference of no more than 5.5% in throughput compared to real results, and the proposed pipeline accurately predicts the optimal placement with minimal latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09652v2">How Chinese are Chinese Language Models? The Puzzling Lack of Language Policy in China's LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 We have reworked the paper substantially. Please refer to the new, updated article: arXiv:2504.00289
    </div>
    <details class="paper-abstract">
      Contemporary language models are increasingly multilingual, but Chinese LLM developers must navigate complex political and business considerations of language diversity. Language policy in China aims at influencing the public discourse and governing a multi-ethnic society, and has gradually transitioned from a pluralist to a more assimilationist approach since 1949. We explore the impact of these influences on current language technology. We evaluate six open-source multilingual LLMs pre-trained by Chinese companies on 18 languages, spanning a wide range of Chinese, Asian, and Anglo-European languages. Our experiments show Chinese LLMs performance on diverse languages is indistinguishable from international LLMs. Similarly, the models' technical reports also show lack of consideration for pretraining data language coverage except for English and Mandarin Chinese. Examining Chinese AI policy, model experiments, and technical reports, we find no sign of any consistent policy, either for or against, language diversity in China's LLM development. This leaves a puzzling fact that while China regulates both the languages people use daily as well as language model development, they do not seem to have any policy on the languages in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07466v1">Grounding Natural Language for Multi-agent Decision-Making with Multi-agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Language is a ubiquitous tool that is foundational to reasoning and collaboration, ranging from everyday interactions to sophisticated problem-solving tasks. The establishment of a common language can serve as a powerful asset in ensuring clear communication and understanding amongst agents, facilitating desired coordination and strategies. In this work, we extend the capabilities of large language models (LLMs) by integrating them with advancements in multi-agent decision-making algorithms. We propose a systematic framework for the design of multi-agentic large language models (LLMs), focusing on key integration practices. These include advanced prompt engineering techniques, the development of effective memory architectures, multi-modal information processing, and alignment strategies through fine-tuning algorithms. We evaluate these design choices through extensive ablation studies on classic game settings with significant underlying social dilemmas and game-theoretic considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07434v1">Let's Revise Step-by-Step: A Unified Local Search Framework for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with inference-time scaling techniques show promise for code generation, yet face notable efficiency and scalability challenges. Construction-based tree-search methods suffer from rapid growth in tree size, high token consumption, and lack of anytime property. In contrast, improvement-based methods offer better performance but often struggle with uninformative reward signals and inefficient search strategies. In this work, we propose \textbf{ReLoc}, a unified local search framework which effectively performs step-by-step code revision. Specifically, ReLoc explores a series of local revisions through four key algorithmic components: initial code drafting, neighborhood code generation, candidate evaluation, and incumbent code updating, each of which can be instantiated with specific decision rules to realize different local search algorithms such as Hill Climbing (HC) or Genetic Algorithm (GA). Furthermore, we develop a specialized revision reward model that evaluates code quality based on revision distance to produce fine-grained preferences that guide the local search toward more promising candidates. Finally, our extensive experimental results demonstrate that our approach achieves superior performance across diverse code generation tasks, significantly outperforming both construction-based tree search as well as the state-of-the-art improvement-based code generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07078v3">Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07421v1">Triple-S: A Collaborative Multi-LLM Framework for Solving Long-Horizon Implicative Tasks in Robotics</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 Accepted to IROS 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) to write policy code for controlling robots has gained significant attention. However, in long-horizon implicative tasks, this approach often results in API parameter, comments and sequencing errors, leading to task failure. To address this problem, we propose a collaborative Triple-S framework that involves multiple LLMs. Through In-Context Learning, different LLMs assume specific roles in a closed-loop Simplification-Solution-Summary process, effectively improving success rates and robustness in long-horizon implicative tasks. Additionally, a novel demonstration library update mechanism which learned from success allows it to generalize to previously failed tasks. We validate the framework in the Long-horizon Desktop Implicative Placement (LDIP) dataset across various baseline models, where Triple-S successfully executes 89% of tasks in both observable and partially observable scenarios. Experiments in both simulation and real-world robot settings further validated the effectiveness of Triple-S. Our code and dataset is available at: https://github.com/Ghbbbbb/Triple-S.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06327v3">A Taxonomy of Inefficiencies in LLM-Generated Python Code</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for automated code generation with promising results. Although prior research has assessed LLM-generated code and identified various quality issues -- such as redundancy, poor maintainability, and sub-optimal performance a systematic understanding and categorization of these inefficiencies remain unexplored. Without such knowledge, practitioners struggle to optimize LLM-generated code for real-world applications, limiting its adoption. This study can also guide improving code LLMs, enhancing the quality and efficiency of code generation. Therefore, in this study, we empirically investigate inefficiencies in LLM-generated code by state-of-the-art models, i.e., CodeLlama, DeepSeek-Coder, and CodeGemma. To do so, we analyze 492 generated code snippets in the HumanEval++ dataset. We then construct a taxonomy of inefficiencies in LLM-generated code that includes 5 categories General Logic, Performance, Readability, Maintainability, and Errors) and 19 subcategories of inefficiencies. We then validate the proposed taxonomy through an online survey with 58 LLM practitioners and researchers. Our study indicates that logic and performance-related inefficiencies are the most popular, relevant, and frequently co-occur and impact overall code quality inefficiency. Our taxonomy provides a structured basis for evaluating the quality LLM-generated code and guiding future research to improve code generation efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07408v1">Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 16 pages, 12 figures, accepted at ICML 2025 New in ML Workshop
    </div>
    <details class="paper-abstract">
      In this study, we wish to showcase the unique utility of large language models (LLMs) in financial semantic annotation and alpha signal discovery. Leveraging a corpus of company-related tweets, we use an LLM to automatically assign multi-label event categories to high-sentiment-intensity tweets. We align these labeled sentiment signals with forward returns over 1-to-7-day horizons to evaluate their statistical efficacy and market tradability. Our experiments reveal that certain event labels consistently yield negative alpha, with Sharpe ratios as low as -0.38 and information coefficients exceeding 0.05, all statistically significant at the 95\% confidence level. This study establishes the feasibility of transforming unstructured social media text into structured, multi-label event variables. A key contribution of this work is its commitment to transparency and reproducibility; all code and methodologies are made publicly available. Our results provide compelling evidence that social media sentiment is a valuable, albeit noisy, signal in financial forecasting and underscore the potential of open-source frameworks to democratize algorithmic trading research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07371v1">AutoAssert 1: A LoRA Fine-Tuned LLM Model for Efficient Automated Assertion Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 16pages,6figures
    </div>
    <details class="paper-abstract">
      As the complexity of software systems continues to increase, the demand for automated testing and maintenance tools is growing exponentially. To meet this urgent need, we propose a new assertion generation method based on Hardware Description Language (HDL). This method combines a lightweight, parameter-adjustable large language model (LLM) with the Unsloth platform to automatically generate test cases, thereby significantly reducing training costs without sacrificing accuracy or generalization performance. Empirical evaluation shows that our method can efficiently generate assertions that strictly conform to the hardware logic. This framework provides a robust and flexible solution to modern software testing and maintenance challenges. https://github.com/liusu-orange/AutoAssert-1 and https://gitee.com/OpenBPU/auto-assert1 are the locations of the source code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v3">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09426v4">FlatQuant: Flatness Matters for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-10
      | 💬 27 pages, accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Recently, quantization has been widely used for the compression and acceleration of large language models (LLMs). Due to the outliers in LLMs, it is crucial to flatten weights and activations to minimize quantization error with equally spaced quantization points. Prior research explores various pre-quantization transformations to suppress outliers, such as per-channel scaling and Hadamard transformation. However, we observe that these transformed weights and activations can still exhibit steep and dispersed distributions. In this paper, we propose FlatQuant (Fast and Learnable Affine Transformation), a new post-training quantization approach that enhances the flatness of weights and activations. Our approach identifies optimal affine transformations for each linear layer, calibrated in hours via a lightweight objective. To reduce runtime overhead of affine transformation, we apply Kronecker product with two lightweight matrices, and fuse all operations in FlatQuant into a single kernel. Extensive experiments demonstrate that FlatQuant establishes a new state-of-the-art benchmark for quantization. For example, it achieves less than 1\% accuracy drop for W4A4 quantization on the LLaMA-3-70B model, surpassing SpinQuant by 7.5\%. Additionally, it provides up to 2.3x prefill speedup and 1.7x decoding speedup compared to the FP16 model. Code is available at: https://github.com/ruikangliu/FlatQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07353v1">Rethinking Domain-Specific LLM Benchmark Construction: A Comprehensiveness-Compactness Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-10
    </div>
    <details class="paper-abstract">
      Numerous benchmarks have been built to evaluate the domain-specific abilities of large language models (LLMs), highlighting the need for effective and efficient benchmark construction. Existing domain-specific benchmarks primarily focus on the scaling law, relying on massive corpora for supervised fine-tuning or generating extensive question sets for broad coverage. However, the impact of corpus and question-answer (QA) set design on the precision and recall of domain-specific LLMs remains unexplored. In this paper, we address this gap and demonstrate that the scaling law is not always the optimal principle for benchmark construction in specific domains. Instead, we propose Comp-Comp, an iterative benchmarking framework based on a comprehensiveness-compactness principle. Here, comprehensiveness ensures semantic recall of the domain, while compactness enhances precision, guiding both corpus and QA set construction. To validate our framework, we conducted a case study in a well-renowned university, resulting in the creation of XUBench, a large-scale and comprehensive closed-domain benchmark. Although we use the academic domain as the case in this work, our Comp-Comp framework is designed to be extensible beyond academia, providing valuable insights for benchmark construction across various domains.
    </details>
</div>
