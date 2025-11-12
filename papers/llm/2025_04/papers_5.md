# llm - 2025_04

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14401v1">LLM-Driven Usefulness Judgment for Web Search Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v2">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that leverages self-supervised speech representations in combination with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English-language data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising solution for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02051v2">Self-Resource Allocation in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      With the development of LLMs as agents, there is a growing interest in connecting multiple agents into multi-agent systems to solve tasks concurrently, focusing on their role in task assignment and coordination. This paper explores how LLMs can effectively allocate computational tasks among multiple agents, considering factors such as cost, efficiency, and performance. In this work, we address key questions, including the effectiveness of LLMs as orchestrators and planners, comparing their effectiveness in task assignment and coordination. Our experiments demonstrate that LLMs can achieve high validity and accuracy in resource allocation tasks. We find that the planner method outperforms the orchestrator method in handling concurrent actions, resulting in improved efficiency and better utilization of agents. Additionally, we show that providing explicit information about worker capabilities enhances the allocation strategies of planners, particularly when dealing with suboptimal workers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14365v1">Accelerating LLM Inference with Flexible N:M Sparsity via A Fully Digital Compute-in-Memory Accelerator</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) pruning with fixed N:M structured sparsity significantly limits the expressivity of the sparse model, yielding sub-optimal performance. In contrast, supporting multiple N:M patterns to provide sparse representational freedom introduces costly overhead in hardware. To address these challenges for LLMs, we first present a flexible layer-wise outlier-density-aware N:M sparsity (FLOW) selection method. FLOW enables the identification of optimal layer-wise N and M values (from a given range) by simultaneously accounting for the presence and distribution of outliers, allowing a higher degree of representational freedom. To deploy sparse models with such N:M flexibility, we then introduce a flexible, low-overhead digital compute-in-memory architecture (FlexCiM). FlexCiM supports diverse sparsity patterns by partitioning a digital CiM (DCiM) macro into smaller sub-macros, which are adaptively aggregated and disaggregated through distribution and merging mechanisms for different N and M values. Extensive experiments on both transformer-based and recurrence-based state space foundation models (SSMs) demonstrate that FLOW outperforms existing alternatives with an accuracy improvement of up to 36%, while FlexCiM achieves up to 1.75x lower inference latency and 1.5x lower energy consumption compared to existing sparse accelerators. Code is available at: https://github.com/FLOW-open-project/FLOW
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14363v1">Improving RL Exploration for LLM Reasoning through Retrospective Replay</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19456v2">SSFF: Investigating LLM Predictive Capabilities for Startup Success through a Multi-Agent Framework with Enhanced Explainability and Performance</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 For relevant code: https://github.com/Xisen-Wang/Startup-Success-Forecasting-Framework
    </div>
    <details class="paper-abstract">
      LLM based agents have recently demonstrated strong potential in automating complex tasks, yet accurately predicting startup success remains an open challenge with few benchmarks and tailored frameworks. To address these limitations, we propose the Startup Success Forecasting Framework, an autonomous system that emulates the reasoning of venture capital analysts through a multi agent collaboration model. Our framework integrates traditional machine learning methods such as random forests and neural networks within a retrieval augmented generation framework composed of three interconnected modules: a prediction block, an analysis block, and an external knowledge block. We evaluate our framework and identify three main findings. First, by leveraging founder segmentation, startups led by L5 founders are 3.79 times more likely to succeed than those led by L1 founders. Second, baseline large language models consistently overpredict startup success and struggle under realistic class imbalances largely due to overreliance on founder claims. Third, our framework significantly enhances prediction accuracy, yielding a 108.3 percent relative improvement over GPT 4o mini and a 30.8 percent relative improvement over GPT 4o. These results demonstrate the value of a multi agent approach combined with discriminative machine learning in mitigating the limitations of standard large language model based prediction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14350v1">Time Up! An Empirical Study of LLM Reasoning Ability Under Output Length Constraint</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making the models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given to the user within a certain output length. It is unclear whether and how the reasoning abilities of LLMs remain effective under such constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test more than 25 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between the token budgets and the actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning that differ from the unconstrained situation, e.g. the optimal choices of model sizes and prompts change under different budgets. These findings offer practical guidance for users to deploy LLMs under real-world latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14345v1">Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 Presented at the ICLR 2025 Workshop on Financial AI (https://sites.google.com/view/financialaiiclr25/home)
    </div>
    <details class="paper-abstract">
      Portfolio optimization faces challenges due to the sensitivity in traditional mean-variance models. The Black-Litterman model mitigates this by integrating investor views, but defining these views remains difficult. This study explores the integration of large language models (LLMs) generated views into portfolio optimization using the Black-Litterman framework. Our method leverages LLMs to estimate expected stock returns from historical prices and company metadata, incorporating uncertainty through the variance in predictions. We conduct a backtest of the LLM-optimized portfolios from June 2024 to February 2025, rebalancing biweekly using the previous two weeks of price data. As baselines, we compare against the S&P 500, an equal-weighted portfolio, and a traditional mean-variance optimized portfolio constructed using the same set of stocks. Empirical results suggest that different LLMs exhibit varying levels of predictive optimism and confidence stability, which impact portfolio performance. The source code and data are available at https://github.com/youngandbin/LLM-MVO-BLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05216v2">Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14286v1">SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement Learning on LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Recent advances of reasoning models, exemplified by OpenAI's o1 and DeepSeek's R1, highlight the significant potential of Reinforcement Learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs). However, replicating these advancements across diverse domains remains challenging due to limited methodological transparency. In this work, we present two-Staged history-Resampling Policy Optimization (SRPO), which successfully surpasses the performance of DeepSeek-R1-Zero-32B on the AIME24 and LiveCodeBench benchmarks. SRPO achieves this using the same base model as DeepSeek (i.e. Qwen2.5-32B) and relies solely on RL, without prior Supervised Fine-Tuning (SFT). Building upon Group Relative Policy Optimization (GRPO), we introduce two key methodological innovations: (1) a two-stage cross-domain training paradigm designed to balance the development of mathematical reasoning and coding proficiency, and (2) History Resampling (HR), a technique to address ineffective samples. Our comprehensive experiments validate the effectiveness of our approach, dedicating to offer valuable insights into scaling LLM reasoning capabilities across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14225v1">Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios. In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at github.com/bowen-upenn/PersonaMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14223v1">SimplifyMyText: An LLM-Based System for Inclusive Plain Language Text Simplification</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 accepted at ECIR 2025
    </div>
    <details class="paper-abstract">
      Text simplification is essential for making complex content accessible to diverse audiences who face comprehension challenges. Yet, the limited availability of simplified materials creates significant barriers to personal and professional growth and hinders social inclusion. Although researchers have explored various methods for automatic text simplification, none fully leverage large language models (LLMs) to offer tailored customization for different target groups and varying levels of simplicity. Moreover, despite its proven benefits for both consumers and organizations, the well-established practice of plain language remains underutilized. In this paper, we https://simplifymytext.org, the first system designed to produce plain language content from multiple input formats, including typed text and file uploads, with flexible customization options for diverse audiences. We employ GPT-4 and Llama-3 and evaluate outputs across multiple metrics. Overall, our work contributes to research on automatic text simplification and highlights the importance of tailored communication in promoting inclusivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17987v2">Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 This paper includes model-generated content that may contain offensive or distressing material
    </div>
    <details class="paper-abstract">
      Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design prompts for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15545v5">SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Scientific literature understanding is crucial for extracting targeted information and garnering insights, thereby significantly advancing scientific discovery. Despite the remarkable success of Large Language Models (LLMs), they face challenges in scientific literature understanding, primarily due to (1) a lack of scientific knowledge and (2) unfamiliarity with specialized scientific tasks. To develop an LLM specialized in scientific literature understanding, we propose a hybrid strategy that integrates continual pre-training (CPT) and supervised fine-tuning (SFT), to simultaneously infuse scientific domain knowledge and enhance instruction-following capabilities for domain-specific tasks.cIn this process, we identify two key challenges: (1) constructing high-quality CPT corpora, and (2) generating diverse SFT instructions. We address these challenges through a meticulous pipeline, including PDF text extraction, parsing content error correction, quality filtering, and synthetic instruction creation. Applying this strategy, we present a suite of LLMs: SciLitLLM, specialized in scientific literature understanding. These models demonstrate promising performance on scientific literature understanding benchmarks. Our contributions are threefold: (1) We present an effective framework that integrates CPT and SFT to adapt LLMs to scientific literature understanding, which can also be easily adapted to other domains. (2) We propose an LLM-based synthesis method to generate diverse and high-quality scientific instructions, resulting in a new instruction set -- SciLitIns -- for supervised fine-tuning in less-represented scientific domains. (3) SciLitLLM achieves promising performance improvements on scientific literature understanding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11056v2">Large Language Models are Good Multi-lingual Learners : When LLMs Meet Cross-lingual Prompts</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      With the advent of Large Language Models (LLMs), generating rule-based data for real-world applications has become more accessible. Due to the inherent ambiguity of natural language and the complexity of rule sets, especially in long contexts, LLMs often struggle to follow all specified rules, frequently omitting at least one. To enhance the reasoning and understanding of LLMs on long and complex contexts, we propose a novel prompting strategy Multi-Lingual Prompt, namely MLPrompt, which automatically translates the error-prone rule that an LLM struggles to follow into another language, thus drawing greater attention to it. Experimental results on public datasets across various tasks have shown MLPrompt can outperform state-of-the-art prompting methods such as Chain of Thought, Tree of Thought, and Self-Consistency. Additionally, we introduce a framework integrating MLPrompt with an auto-checking mechanism for structured data generation, with a specific case study in text-to-MIP instances. Further, we extend the proposed framework for text-to-SQL to demonstrate its generation ability towards structured data synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13843v2">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted by SIGIR 2025, 6 pages
    </div>
    <details class="paper-abstract">
      LLM-based user agents, which simulate user interaction behavior, are emerging as a promising approach to enhancing recommender systems. In real-world scenarios, users' interactions often exhibit cross-domain characteristics and are influenced by others. However, the memory design in current methods causes user agents to introduce significant irrelevant information during decision-making in cross-domain scenarios and makes them unable to recognize the influence of other users' interactions, such as popularity factors. To tackle this issue, we propose a dual-layer memory architecture combined with a two-step fusion mechanism. This design avoids irrelevant information during decision-making while ensuring effective integration of cross-domain preferences. We also introduce the concepts of interest groups and group-shared memory to better capture the influence of popularity factors on users with similar interests. Comprehensive experiments validate the effectiveness of AgentCF++. Our code is available at https://github.com/jhliu0807/AgentCF-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02497v2">PennyLang: Pioneering LLM-Based Quantum Code Generation with a Novel PennyLane-Centric Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 10 pages, 7 figures, 7 tables, submitted for review under QCE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer remarkable capabilities in code generation, natural language processing, and domain-specific reasoning. However, their application in quantum software development remains underexplored, particularly for PennyLane-a leading framework for hybrid quantum-classical computing. To address this gap, we introduce a novel, high-quality dataset comprising 3,347 PennyLane-specific quantum code samples and contextual descriptions, specifically curated to support LLM training and fine-tuning for quantum code assistance. Our contributions are threefold: (1) the automatic construction and open-source release of a comprehensive PennyLane dataset derived from textbooks, official documentation, and open-source repositories; (2) a structured methodology for data curation, annotation, and formatting to enhance LLM usability and relevance; and (3) a rigorous evaluation of code generation capabilities using both baseline Retrieval-Augmented Generation (RAG) and a GraphRAG-enhanced pipeline. Using the PennyLang framework, we demonstrate that GraphRAG, when applied to a GPT-4o Mini model, substantially outperforms standard prompting and baseline RAG. Accuracy improves from 20.5% (without RAG) to 58.2% with GraphRAG, showcasing its effectiveness in reducing hallucinations and improving code correctness in quantum programming tasks. Compared to prior efforts focused largely on Qiskit, our work expands LLM-based assistance to the PennyLane ecosystem, contributing practical tools and reproducible methodologies for advancing AI-assisted quantum software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13845v2">Improving LLM-powered Recommendations with Personalized Information</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted by SIGIR 2025, 7 pages
    </div>
    <details class="paper-abstract">
      Due to the lack of explicit reasoning modeling, existing LLM-powered recommendations fail to leverage LLMs' reasoning capabilities effectively. In this paper, we propose a pipeline called CoT-Rec, which integrates two key Chain-of-Thought (CoT) processes -- user preference analysis and item perception analysis -- into LLM-powered recommendations, thereby enhancing the utilization of LLMs' reasoning abilities. CoT-Rec consists of two stages: (1) personalized information extraction, where user preferences and item perception are extracted, and (2) personalized information utilization, where this information is incorporated into the LLM-powered recommendation process. Experimental results demonstrate that CoT-Rec shows potential for improving LLM-powered recommendations. The implementation is publicly available at https://github.com/jhliu0807/CoT-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00383v2">Unified Parameter-Efficient Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized natural language processing, enabling advanced understanding and reasoning capabilities across a variety of tasks. Fine-tuning these models for specific domains, particularly through Parameter-Efficient Fine-Tuning (PEFT) strategies like LoRA, has become a prevalent practice due to its efficiency. However, this raises significant privacy and security concerns, as models may inadvertently retain and disseminate sensitive or undesirable information. To address these issues, we introduce a novel instance-wise unlearning framework, LLMEraser, which systematically categorizes unlearning tasks and applies precise parameter adjustments using influence functions. Unlike traditional unlearning techniques that are often limited in scope and require extensive retraining, LLMEraser is designed to handle a broad spectrum of unlearning tasks without compromising model performance. Extensive experiments on benchmark datasets demonstrate that LLMEraser excels in efficiently managing various unlearning scenarios while maintaining the overall integrity and efficacy of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13475v1">LLM Sensitivity Evaluation Framework for Clinical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance across various domains. However, for clinical diagnosis, higher expectations are required for LLM's reliability and sensitivity: thinking like physicians and remaining sensitive to key medical information that affects diagnostic reasoning, as subtle variations can lead to different diagnosis results. Yet, existing works focus mainly on investigating the sensitivity of LLMs to irrelevant context and overlook the importance of key information. In this paper, we investigate the sensitivity of LLMs, i.e. GPT-3.5, GPT-4, Gemini, Claude3 and LLaMA2-7b, to key medical information by introducing different perturbation strategies. The evaluation results highlight the limitations of current LLMs in remaining sensitive to key medical information for diagnostic decision-making. The evolution of LLMs must focus on improving their reliability, enhancing their ability to be sensitive to key information, and effectively utilizing this information. These improvements will enhance human trust in LLMs and facilitate their practical application in real-world scenarios. Our code and dataset are available at https://github.com/chenwei23333/DiagnosisQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13474v1">Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models are a promising tool for automated vulnerability detection, thanks to their success in code generation and repair. However, despite widespread adoption, a critical question remains: Are LLMs truly effective at detecting real-world vulnerabilities? Current evaluations, which often assess models on isolated functions or files, ignore the broader execution and data-flow context essential for understanding vulnerabilities. This oversight leads to two types of misleading outcomes: incorrect conclusions and flawed rationales, collectively undermining the reliability of prior assessments. Therefore, in this paper, we challenge three widely held community beliefs: that LLMs are (i) unreliable, (ii) insensitive to code patches, and (iii) performance-plateaued across model scales. We argue that these beliefs are artifacts of context-deprived evaluations. To address this, we propose CORRECT (Context-Rich Reasoning Evaluation of Code with Trust), a new evaluation framework that systematically incorporates contextual information into LLM-based vulnerability detection. We construct a context-rich dataset of 2,000 vulnerable-patched program pairs spanning 99 CWEs and evaluate 13 LLMs across four model families. Our framework elicits both binary predictions and natural-language rationales, which are further validated using LLM-as-a-judge techniques. Our findings overturn existing misconceptions. When provided with sufficient context, SOTA LLMs achieve significantly improved performance (e.g., 0.7 F1-score on key CWEs), with 0.8 precision. We show that most false positives stem from reasoning errors rather than misclassification, and that while model and test-time scaling improve performance, they introduce diminishing returns and trade-offs in recall. Finally, we uncover new flaws in current LLM-based detection systems, such as limited generalization and overthinking biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v1">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) pipelines, improving performance and generalization. This has spurred their integration into various systems. Many NLP systems, including ours, employ a "one-stage" pipeline directly incorporating LLMs. While effective, this approach incurs substantial costs and latency due to the need for large model parameters to achieve satisfactory outcomes. This paper introduces a three-stage cost-efficient end-to-end LLM deployment pipeline-including prototyping, knowledge transfer, and model compression-to tackle the cost-performance dilemma in LLM-based frameworks. Our approach yields a super tiny model optimized for cost and performance in online systems, simplifying the system architecture. Initially, by transforming complex tasks into a function call-based LLM-driven pipeline, an optimal performance prototype system is constructed to produce high-quality data as a teacher model. The second stage combine techniques like rejection fine-tuning, reinforcement learning and knowledge distillation to transfer knowledge to a smaller 0.5B student model, delivering effective performance at minimal cost. The final stage applies quantization and pruning to extremely compress model to 0.4B, achieving ultra-low latency and cost. The framework's modular design and cross-domain capabilities suggest potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15288v2">If LLMs Would Just Look: Simple Line-by-line Checking Improves Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      The rapid expansion of software systems and the growing number of reported vulnerabilities have emphasized the importance of accurately identifying vulnerable code segments. Traditional methods for vulnerability localization, such as manual code audits or rule-based tools, are often time-consuming and limited in scope, typically focusing on specific programming languages or types of vulnerabilities. In recent years, the introduction of large language models (LLMs) such as GPT and LLaMA has opened new possibilities for automating vulnerability detection. However, while LLMs show promise in this area, they face challenges, particularly in maintaining accuracy over longer code contexts. This paper introduces LOVA, a novel framework leveraging the self-attention mechanisms inherent in LLMs to enhance vulnerability localization. Our key insight is that self-attention mechanisms assign varying importance to different parts of the input, making it possible to track how much attention the model focuses on specific lines of code. In the context of vulnerability localization, the hypothesis is that vulnerable lines of code will naturally attract higher attention weights because they have a greater influence on the model's output. By systematically tracking changes in attention weights and focusing on specific lines of code, LOVA improves the precision of identifying vulnerable lines across various programming languages. Through rigorous experimentation and evaluation, we demonstrate that LOVA significantly outperforms existing LLM-based approaches, achieving up to a 5.3x improvement in F1-scores. LOVA also demonstrated strong scalability, with up to a 14.6x improvement in smart contract vulnerability localization across languages like C, Python, Java, and Solidity. Its robustness was proven through consistent performance across different LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12943v2">Customizing Emotional Support: How Do Individuals Construct and Interact With LLM-Powered Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 20 pages, 3 figures, 3 tables. Accepted to CHI 2025, ACM Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Personalized support is essential to fulfill individuals' emotional needs and sustain their mental well-being. Large language models (LLMs), with great customization flexibility, hold promises to enable individuals to create their own emotional support agents. In this work, we developed ChatLab, where users could construct LLM-powered chatbots with additional interaction features including voices and avatars. Using a Research through Design approach, we conducted a week-long field study followed by interviews and design activities (N = 22), which uncovered how participants created diverse chatbot personas for emotional reliance, confronting stressors, connecting to intellectual discourse, reflecting mirrored selves, etc. We found that participants actively enriched the personas they constructed, shaping the dynamics between themselves and the chatbot to foster open and honest conversations. They also suggested other customizable features, such as integrating online activities and adjustable memory settings. Based on these findings, we discuss opportunities for enhancing personalized emotional support through emerging AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v4">DNR Bench: Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Reason Bench (DNR Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNR Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNR Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14096v1">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully generated adversarial examples that deliberately violate spatial, temporal, or cross-frame relations. By applying Direct Preference Optimization to just 7,020 preference pairs, VideoPASTA learns robust representations that capture fine-grained spatial relationships and long-range temporal dynamics. Experiments on standard video benchmarks show significant relative performance gains of 3.05% on VideoMME, 1.97% on NeXTQA, and 1.31% on LongVideoBench, over the baseline Qwen2.5-VL model. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without human annotation or captioning, relying on just 32-frame sampling, compared to the 96-frame, multi-GPU setups of prior work. This efficiency makes our approach a scalable, plug-and-play solution that seamlessly integrates with existing models while preserving their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09002v2">KNighter: Transforming Static Analysis with LLM-Synthesized Checkers</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Static analysis is a powerful technique for bug detection in critical systems like operating system kernels. However, designing and implementing static analyzers is challenging, time-consuming, and typically limited to predefined bug patterns. While large language models (LLMs) have shown promise for static analysis, directly applying them to scan large systems remains impractical due to computational constraints and contextual limitations. We present KNighter, the first approach that unlocks scalable LLM-based static analysis by automatically synthesizing static analyzers from historical bug patterns. Rather than using LLMs to directly analyze massive systems, our key insight is leveraging LLMs to generate specialized static analyzers guided by historical patch knowledge. KNighter implements this vision through a multi-stage synthesis pipeline that validates checker correctness against original patches and employs an automated refinement process to iteratively reduce false positives. Our evaluation on the Linux kernel demonstrates that KNighter generates high-precision checkers capable of detecting diverse bug patterns overlooked by existing human-written analyzers. To date, KNighter-synthesized checkers have discovered 92 new, critical, long-latent bugs (average 4.3 years) in the Linux kernel; 77 are confirmed, 57 fixed, and 16 have been assigned CVE numbers. This work establishes an entirely new paradigm for scalable, reliable, and traceable LLM-based static analysis for real-world systems via checker synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13145v2">Exploring Expert Failures Improves LLM Agent Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19314v2">LiveBench: A Challenging, Contamination-Limited LLM Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 ICLR 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be resistant to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, LiveBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-limited versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5B to 405B in size. LiveBench is difficult, with top models achieving below 70% accuracy. We release all questions, code, and model answers. Questions are added and updated on a monthly basis, and we release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14039v1">MEQA: A Meta-Evaluation Framework for Question & Answer LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance, their potential for widespread societal impact grows simultaneously. Hence, rigorous LLM evaluations are both a technical necessity and social imperative. While numerous evaluation benchmarks have been developed, there remains a critical gap in meta-evaluation: effectively assessing benchmarks' quality. We propose MEQA, a framework for the meta-evaluation of question and answer (QA) benchmarks, to provide standardized assessments, quantifiable scores, and enable meaningful intra-benchmark comparisons. We demonstrate this approach on cybersecurity benchmarks, using human and LLM evaluators, highlighting the benchmarks' strengths and weaknesses. We motivate our choice of test domain by AI models' dual nature as powerful defensive tools and security threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14038v1">Flowco: Rethinking Data Analysis in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Conducting data analysis typically involves authoring code to transform, visualize, analyze, and interpret data. Large language models (LLMs) are now capable of generating such code for simple, routine analyses. LLMs promise to democratize data science by enabling those with limited programming expertise to conduct data analyses, including in scientific research, business, and policymaking. However, analysts in many real-world settings must often exercise fine-grained control over specific analysis steps, verify intermediate results explicitly, and iteratively refine their analytical approaches. Such tasks present barriers to building robust and reproducible analyses using LLMs alone or even in conjunction with existing authoring tools (e.g., computational notebooks). This paper introduces Flowco, a new mixed-initiative system to address these challenges. Flowco leverages a visual dataflow programming model and integrates LLMs into every phase of the authoring process. A user study suggests that Flowco supports analysts, particularly those with less programming experience, in quickly authoring, debugging, and refining data analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14024v1">Simplicity by Obfuscation: Evaluating LLM-Driven Code Transformation with Semantic Elasticity</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 The paper has been peer-reviewed and accepted for publication in the proceedings of the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025)
    </div>
    <details class="paper-abstract">
      Code obfuscation is the conversion of original source code into a functionally equivalent but less readable form, aiming to prevent reverse engineering and intellectual property theft. This is a challenging task since it is crucial to maintain functional correctness of the code while substantially disguising the input code. The recent development of large language models (LLMs) paves the way for practical applications in different domains, including software engineering. This work performs an empirical study on the ability of LLMs to obfuscate Python source code and introduces a metric (i.e., semantic elasticity) to measure the quality degree of obfuscated code. We experimented with 3 leading LLMs, i.e., Claude-3.5-Sonnet, Gemini-1.5, GPT-4-Turbo across 30 Python functions from diverse computational domains. Our findings reveal GPT-4-Turbo's remarkable effectiveness with few-shot prompting (81% pass rate versus 29% standard prompting), significantly outperforming both Gemini-1.5 (39%) and Claude-3.5-Sonnet (30%). Notably, we discovered a counter-intuitive "obfuscation by simplification" phenomenon where models consistently reduce rather than increase cyclomatic complexity. This study provides a methodological framework for evaluating AI-driven obfuscation while highlighting promising directions for leveraging LLMs in software security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13993v1">CPR: Leveraging LLMs for Topic and Phrase Suggestion to Facilitate Comprehensive Product Reviews</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Consumers often heavily rely on online product reviews, analyzing both quantitative ratings and textual descriptions to assess product quality. However, existing research hasn't adequately addressed how to systematically encourage the creation of comprehensive reviews that capture both customers sentiment and detailed product feature analysis. This paper presents CPR, a novel methodology that leverages the power of Large Language Models (LLMs) and Topic Modeling to guide users in crafting insightful and well-rounded reviews. Our approach employs a three-stage process: first, we present users with product-specific terms for rating; second, we generate targeted phrase suggestions based on these ratings; and third, we integrate user-written text through topic modeling, ensuring all key aspects are addressed. We evaluate CPR using text-to-text LLMs, comparing its performance against real-world customer reviews from Walmart. Our results demonstrate that CPR effectively identifies relevant product terms, even for new products lacking prior reviews, and provides sentiment-aligned phrase suggestions, saving users time and enhancing reviews quality. Quantitative analysis reveals a 12.3% improvement in BLEU score over baseline methods, further supported by manual evaluation of generated phrases. We conclude by discussing potential extensions and future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13989v1">Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing outliers.We achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15303v1">High-Throughput LLM inference on Heterogeneous Clusters</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Nowadays, many companies possess various types of AI accelerators, forming heterogeneous clusters. Efficiently leveraging these clusters for high-throughput large language model (LLM) inference services can significantly reduce costs and expedite task processing. However, LLM inference on heterogeneous clusters presents two main challenges. Firstly, different deployment configurations can result in vastly different performance. The number of possible configurations is large, and evaluating the effectiveness of a specific setup is complex. Thus, finding an optimal configuration is not an easy task. Secondly, LLM inference instances within a heterogeneous cluster possess varying processing capacities, leading to different processing speeds for handling inference requests. Evaluating these capacities and designing a request scheduling algorithm that fully maximizes the potential of each instance is challenging. In this paper, we propose a high-throughput inference service system on heterogeneous clusters. First, the deployment configuration is optimized by modeling the resource amount and expected throughput and using the exhaustive search method. Second, a novel mechanism is proposed to schedule requests among instances, which fully considers the different processing capabilities of various instances. Extensive experiments show that the proposed scheduler improves throughput by 122.5% and 33.6% on two heterogeneous clusters, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16116v1">DMind Benchmark: The First Comprehensive Benchmark for LLM Evaluation in the Web3 Domain</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have led to significant progress on a wide range of natural language processing tasks. However, their effectiveness in specialized and rapidly evolving domains such as Web3 remains underexplored. In this paper, we introduce DMind Benchmark, a novel framework that systematically tests LLMs across nine key categories encompassing blockchain fundamentals, infrastructure, smart contract analysis, decentralized finance (DeFi), decentralized autonomous organizations (DAOs), non-fungible tokens (NFTs), token economics, meme concepts, and security vulnerabilities. DMind Benchmark goes beyond conventional multiple-choice questions by incorporating domain-specific subjective tasks (e.g., smart contract code auditing and repair, numeric reasoning on on-chain data, and fill-in assessments), thereby capturing real-world complexities and stress-testing model adaptability. We evaluate fifteen popular LLMs (from ChatGPT, DeepSeek, Claude, and Gemini series) on DMind Benchmark, uncovering performance gaps in Web3-specific reasoning and application, particularly in emerging areas like token economics and meme concepts. Even the strongest models face significant challenges in identifying subtle security vulnerabilities and analyzing complex DeFi mechanisms. To foster progress in this area, we publicly release our benchmark dataset, evaluation pipeline, and annotated results at http://www.dmind.ai, offering a valuable resource for advancing specialized domain adaptation and the development of more robust Web3-enabled LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16112v1">HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v1">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 24 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning capabilities of LLMs, particularly in mathematics and programming tasks. It is widely believed that RLVR enables LLMs to continuously self-improve, thus acquiring novel reasoning abilities that exceed corresponding base models' capacity. In this study, however, we critically re-examines this assumption by measuring the pass@\textit{k} metric with large values of \textit{k} to explore the reasoning capability boundary of the models across a wide range of model families and benchmarks. Surprisingly, the RL does \emph{not}, in fact, elicit fundamentally new reasoning patterns. While RL-trained models outperform their base models at smaller values of $k$ (\eg, $k$=1), base models can achieve a comparable or even higher pass@$k$ score compared to their RL counterparts at large $k$ values. The reasoning paths generated by RL-trained models are already included in the base models' sampling distribution, suggesting that most reasoning abilities manifested in RL-trained models are already obtained by base models. Further analysis shows that RL training boosts the performance by biasing the model's output distribution toward paths that are more likely to yield rewards, therefore sampling correct responses more efficiently. But this also results in a narrower reasoning capability boundary compared to base models. Similar results are observed in visual reasoning tasks trained with RLVR. Moreover, we find that distillation can genuinely introduce new knowledge into the model, different from RLVR. These findings underscore a critical limitation of RLVR in advancing LLM reasoning abilities which requires us to fundamentally rethink the impact of RL training in reasoning LLMs and the need of a better paradigm. Project Page: https://limit-of-RLVR.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13816v1">Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v5">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13774v1">DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 49 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently revolutionized language processing tasks but have also brought ethical and legal issues. LLMs have a tendency to memorize potentially private or copyrighted information present in the training data, which might then be delivered to end users at inference time. When this happens, a naive solution is to retrain the model from scratch after excluding the undesired data. Although this guarantees that the target data have been forgotten, it is also prohibitively expensive for LLMs. Approximate unlearning offers a more efficient alternative, as it consists of ex post modifications of the trained model itself to prevent undesirable results, but it lacks forgetting guarantees because it relies solely on empirical evidence. In this work, we present DP2Unlearning, a novel LLM unlearning framework that offers formal forgetting guarantees at a significantly lower cost than retraining from scratch on the data to be retained. DP2Unlearning involves training LLMs on textual data protected using {\epsilon}-differential privacy (DP), which later enables efficient unlearning with the guarantees against disclosure associated with the chosen {\epsilon}. Our experiments demonstrate that DP2Unlearning achieves similar model performance post-unlearning, compared to an LLM retraining from scratch on retained data -- the gold standard exact unlearning -- but at approximately half the unlearning cost. In addition, with a reasonable computational cost, it outperforms approximate unlearning methods at both preserving the utility of the model post-unlearning and effectively forgetting the targeted information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13769v1">Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 The paper has been peer-reviewed and accepted for publication to the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025)
    </div>
    <details class="paper-abstract">
      Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19413v2">Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Paywalls, licenses and copyright rules often restrict the broad dissemination and reuse of scientific knowledge. We take the position that it is both legally and technically feasible to extract the scientific knowledge in scholarly texts. Current methods, like text embeddings, fail to reliably preserve factual content, and simple paraphrasing may not be legally sound. We propose a new idea for the community to adopt: convert scholarly documents into knowledge preserving, but style agnostic representations we term Knowledge Units using LLMs. These units use structured data capturing entities, attributes and relationships without stylistic content. We provide evidence that Knowledge Units (1) form a legally defensible framework for sharing knowledge from copyrighted research texts, based on legal analyses of German copyright law and U.S. Fair Use doctrine, and (2) preserve most (~95\%) factual knowledge from original text, measured by MCQ performance on facts from the original copyrighted text across four research domains. Freeing scientific knowledge from copyright promises transformative benefits for scientific research and education by allowing language models to reuse important facts from copyrighted text. To support this, we share open-source tools for converting research documents into Knowledge Units. Overall, our work posits the feasibility of democratizing access to scientific knowledge while respecting copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09024v3">AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      The robustness of LLMs to jailbreak attacks, where users design prompts to circumvent safety measures and misuse model capabilities, has been studied primarily for LLMs acting as simple chatbots. Meanwhile, LLM agents -- which use external tools and can execute multi-stage tasks -- may pose a greater risk if misused, but their robustness remains underexplored. To facilitate research on LLM agent misuse, we propose a new benchmark called AgentHarm. The benchmark includes a diverse set of 110 explicitly malicious agent tasks (440 with augmentations), covering 11 harm categories including fraud, cybercrime, and harassment. In addition to measuring whether models refuse harmful agentic requests, scoring well on AgentHarm requires jailbroken agents to maintain their capabilities following an attack to complete a multi-step task. We evaluate a range of leading LLMs, and find (1) leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking, (2) simple universal jailbreak templates can be adapted to effectively jailbreak agents, and (3) these jailbreaks enable coherent and malicious multi-step agent behavior and retain model capabilities. To enable simple and reliable evaluation of attacks and defenses for LLM-based agents, we publicly release AgentHarm at https://huggingface.co/datasets/ai-safety-institute/AgentHarm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11005v3">A Theory of LLM Sampling: Part Descriptive and Part Prescriptive</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in autonomous decision-making, where they sample options from vast action spaces. However, the heuristics that guide this sampling process remain under-explored. We study this sampling behavior and show that this underlying heuristics resembles that of human decision-making: comprising a descriptive component (reflecting statistical norm) and a prescriptive component (implicit ideal encoded in the LLM) of a concept. We show that this deviation of a sample from the statistical norm towards a prescriptive component consistently appears in concepts across diverse real-world domains like public health, and economic trends. To further illustrate the theory, we demonstrate that concept prototypes in LLMs are affected by prescriptive norms, similar to the concept of normality in humans. Through case studies and comparison with human studies, we illustrate that in real-world applications, the shift of samples toward an ideal value in LLMs' outputs can result in significantly biased decision-making, raising ethical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12319v4">The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used as evaluators for natural language generation tasks, ensuring unbiased assessments is essential. However, LLM evaluators often display biased preferences, such as favoring verbosity and authoritative tones. Our empirical analysis reveals that these biases are exacerbated in pairwise evaluation, where LLMs directly compare two outputs and easily prioritize superficial attributes. In contrast, pointwise evaluation, which assesses outputs independently, is less susceptible to such bias because each output is judged in isolation. To address the limitations of the pairwise evaluation, we introduce a novel evaluation method, PRePair, which integrates pointwise reasoning within a pairwise framework. PRePair effectively alleviates biased preference, improving performance on the adversarial benchmark (LLMBar) while outperforming pointwise evaluation on the standard benchmark (MT-Bench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19874v3">Is In-Context Learning Sufficient for Instruction Following in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted at ICLR 2025. This camera-ready version v3 adds multi-turn alignment via ICL, revisiting main results on instruct models, and simple mechanistic study. Updates in the v2: experiment with decoding schemes, scaling in-context alignment, ICL vs IFT for instruction following. Code at https://github.com/tml-epfl/icl-alignment
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) allows LLMs to learn from examples without changing their weights: this is a particularly promising capability for long-context LLMs that can potentially learn from many examples. Recently, Lin et al. (2024) proposed URIAL, a method using only three in-context examples to align base LLMs, achieving non-trivial instruction following performance. In this work, we show that, while effective, ICL alignment with URIAL still underperforms compared to instruction fine-tuning on the established benchmark MT-Bench, especially with more capable base LLMs. We then uncover the most relevant elements for successful in-context alignment, finding the crucial role of the decoding parameters. Based on these insights, we show that the approach of URIAL can indeed be improved by adding high-quality, potentially carefully selected via greedy search, demonstrations in context, getting closer to the performance of instruct models. Finally, we provide the first, to our knowledge, systematic comparison of ICL and instruction fine-tuning (IFT) for instruction following in the low data regime, where ICL can be a viable alternative to IFT. Overall, our work advances the understanding of ICL as an alignment technique and its relationship to IFT. We provide our code at https://github.com/tml-epfl/icl-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13629v1">Divergent LLM Adoption and Heterogeneous Convergence Paths in Research Writing</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as ChatGPT, are reshaping content creation and academic writing. This study investigates the impact of AI-assisted generative revisions on research manuscripts, focusing on heterogeneous adoption patterns and their influence on writing convergence. Leveraging a dataset of over 627,000 academic papers from arXiv, we develop a novel classification framework by fine-tuning prompt- and discipline-specific large language models to detect the style of ChatGPT-revised texts. Our findings reveal substantial disparities in LLM adoption across academic disciplines, gender, native language status, and career stage, alongside a rapid evolution in scholarly writing styles. Moreover, LLM usage enhances clarity, conciseness, and adherence to formal writing conventions, with improvements varying by revision type. Finally, a difference-in-differences analysis shows that while LLMs drive convergence in academic writing, early adopters, male researchers, non-native speakers, and junior scholars exhibit the most pronounced stylistic shifts, aligning their writing more closely with that of established researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v2">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20999v3">MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Typically, LLMs are first pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage, leading to a decline in general capabilities. Existing approaches to mitigate forgetting often rely on access to pre-training data, which may be unavailable in many real-world scenarios--such as fine-tuning checkpoint-only open-source LLMs. To address this challenge, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). MoFO is an extension of greedy block coordinate descent (BCD) methods: in each iteration, MoFO only updates the model parameters with the largest momentum magnitudes, while keeping all other parameters fixed. MoFO achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting. We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its effectiveness in mitigating forgetting without pre-training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13562v1">DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13557v1">Integrating LLMs for Grading and Appeal Resolution in Computer Science Education</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This study explores the integration of Large Language Models (LLMs) into the grading and appeal resolution process in computer science education. We introduce AI-PAT, an AI-powered assessment tool that leverages LLMs to evaluate computer science exams, generate feedback, and address student appeals. AI-PAT was used to assess over 850 exam submissions and handle 185 appeal cases. Our multi-model comparison (ChatGPT, Gemini) reveals strong correlations between model outputs, though significant variability persists depending on configuration and prompt design. Human graders, while internally consistent, showed notable inter-rater disagreement, further highlighting subjectivity in manual evaluation. The appeal process led to grade changes in 74% of cases, indicating the need for continued refinement of AI evaluation strategies. While students appreciated the speed and detail of AI feedback, survey responses revealed trust and fairness concerns. We conclude that AI-PAT offers scalable benefits for formative assessment and feedback, but must be accompanied by transparent grading rubrics, human oversight, and appeal mechanisms to ensure equitable outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v2">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and chain-of-modality (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://anonymous.4open.science/r/EmoVoice-DF55. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11969v4">Does Refusal Training in LLMs Generalize to the Past Tense?</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5 Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts: https://github.com/tml-epfl/llm-past-tense
    </div>
    <details class="paper-abstract">
      Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13272v1">Using LLMs for Library Migration</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Library migration is the process of replacing a used software library with another library that provides similar functionality. Manual library migration is time-consuming and error prone, as it requires developers to understand the APIs of both libraries, map them, and perform the necessary code transformations. Due to its difficulty, most of the existing automated techniques and tooling stop at the API mapping stage or support a limited set of code transformations. On the other hand, Large Language Models (LLMs) are good at generating and transforming code and finding similar code, which are necessary upstream tasks for library migration. Such capabilities suggest that LLMs may be suitable for library migration. Therefore, in this paper, we investigate the effectiveness of LLMs for migration between Python libraries. We evaluate three LLMs, LLama 3.1, GPT-4o mini, and GPT-4o on PyMigBench, where we migrate 321 real-world library migrations that include 2,989 migration-related code changes. We measure the correctness of the migration results in two ways. We first compare the LLM's migrated code with the developers' migrated code in the benchmark and then run the unit tests available in the client repositories. We find that LLama 3.1, GPT-4o mini, and GPT-4o correctly migrate 89%, 89%, and 94% of the migration-related code changes. respectively. We also find that 36%, 52% and 64% of the LLama 3.1, GPT-4o mini, and GPT-4o migrations pass the same tests that passed in the developer's migration. Overall, our results suggest that LLMs can be effective in migrating code between libraries, but we also identify cases that pose difficulties for the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13217v1">Sustainability via LLM Right-sizing</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 17 pages, 2 Figures, 6 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model "good enough"? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups -- premium all-rounders, competent generalists, and limited but safe performers -- highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15299v1">D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted by MobiCom 2025
    </div>
    <details class="paper-abstract">
      The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefits, MoE is still too expensive to deploy on resource-constrained edge devices, especially with the demands of on-device inference services. Recent research efforts often apply model compression techniques, such as quantization, pruning and merging, to restrict MoE complexity. Unfortunately, due to their predefined static model optimization strategies, they cannot always achieve the desired quality-overhead trade-off when handling multiple requests, finally degrading the on-device quality of service. These limitations motivate us to propose the D$^2$MoE, an algorithm-system co-design framework that matches diverse task requirements by dynamically allocating the most proper bit-width to each expert. Specifically, inspired by the nested structure of matryoshka dolls, we propose the matryoshka weight quantization (MWQ) to progressively compress expert weights in a bit-nested manner and reduce the required runtime memory. On top of it, we further optimize the I/O-computation pipeline and design a heuristic scheduling algorithm following our hottest-expert-bit-first (HEBF) principle, which maximizes the expert parallelism between I/O and computation queue under constrained memory budgets, thus significantly reducing the idle temporal bubbles waiting for the experts to load. Evaluations on real edge devices show that D$^2$MoE improves the overall inference throughput by up to 1.39$\times$ and reduces the peak memory footprint by up to 53% over the latest on-device inference frameworks, while still preserving comparable serving accuracy as its INT8 counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13145v1">Exploring Expert Failures Improves LLM Agent Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13125v1">LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16063v4">Citation-Enhanced Generation for LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc Citation-Enhanced Generation (CEG) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-based citation generation module. Once the statements in the generated content lack of reference, our model can regenerate responses until all statements are supported by citations. Note that our method is a training-free plug-and-play plugin that is capable of various LLMs. Experiments on various hallucination-related datasets show our framework outperforms state-of-the-art methods in both hallucination detection and response regeneration on three benchmarks. Our codes and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11514v3">Towards Translating Real-World Code with LLMs: A Study of Translating to Rust</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 12 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in code translation - the task of translating code written in one programming language to another language - due to their ability to write code in most programming languages. However, LLM's effectiveness on translating real-world code remains largely unstudied. In this work, we perform the first substantial study on LLM-based translation to Rust by assessing the ability of five state-of-the-art LLMs, GPT4, Claude 3, Claude 2.1, Gemini Pro, and Mixtral. We conduct our study on code extracted from real-world open source projects. To enable our study, we develop FLOURINE, an end-to-end code translation tool that uses differential fuzzing to check if a Rust translation is I/O equivalent to the original source program, eliminating the need for pre-existing test cases. As part of our investigation, we assess both the LLM's ability to produce an initially successful translation, as well as their capacity to fix a previously generated buggy one. If the original and the translated programs are not I/O equivalent, we apply a set of automated feedback strategies, including feedback to the LLM with counterexamples. Our results show that the most successful LLM can translate 47% of our benchmarks, and also provides insights into next steps for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09012v2">Multimodal LLMs Can Reason about Aesthetics in Zero-Shot</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 WIP, Homepage https://github.com/songrise/MLLM4Art
    </div>
    <details class="paper-abstract">
      The rapid progress of generative art has democratized the creation of visually pleasing imagery. However, achieving genuine artistic impact - the kind that resonates with viewers on a deeper, more meaningful level - requires a sophisticated aesthetic sensibility. This sensibility involves a multi-faceted reasoning process extending beyond mere visual appeal, which is often overlooked by current computational models. This paper pioneers an approach to capture this complex process by investigating how the reasoning capabilities of Multimodal LLMs (MLLMs) can be effectively elicited for aesthetic judgment. Our analysis reveals a critical challenge: MLLMs exhibit a tendency towards hallucinations during aesthetic reasoning, characterized by subjective opinions and unsubstantiated artistic interpretations. We further demonstrate that these limitations can be overcome by employing an evidence-based, objective reasoning process, as substantiated by our proposed baseline, ArtCoT. MLLMs prompted by this principle produce multi-faceted and in-depth aesthetic reasoning that aligns significantly better with human judgment. These findings have direct applications in areas such as AI art tutoring and as reward models for generative art. Ultimately, our work paves the way for AI systems that can truly understand, appreciate, and generate artworks that align with the sensible human aesthetic standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v3">Unlocking LLM Repair Capabilities in Low-Resource Programming Languages Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v2">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13052v1">GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20973v2">Arabizi vs LLMs: Can the Genie Understand the Language of Aladdin?</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted to MT Summit 2025 (Track: Implementation and Case Studies) https://mtsummit2025.unige.ch/
    </div>
    <details class="paper-abstract">
      In this era of rapid technological advancements, communication continues to evolve as new linguistic phenomena emerge. Among these is Arabizi, a hybrid form of Arabic that incorporates Latin characters and numbers to represent the spoken dialects of Arab communities. Arabizi is widely used on social media and allows people to communicate in an informal and dynamic way, but it poses significant challenges for machine translation due to its lack of formal structure and deeply embedded cultural nuances. This case study arises from a growing need to translate Arabizi for gisting purposes. It evaluates the capacity of different LLMs to decode and translate Arabizi, focusing on multiple Arabic dialects that have rarely been studied up until now. Using a combination of human evaluators and automatic metrics, this research project investigates the model's performance in translating Arabizi into both Modern Standard Arabic and English. Key questions explored include which dialects are translated most effectively and whether translations into English surpass those into Arabic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13038v1">How Large Language Models Are Changing MOOC Essay Answers: A Comparison of Pre- and Post-LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The release of ChatGPT in late 2022 caused a flurry of activity and concern in the academic and educational communities. Some see the tool's ability to generate human-like text that passes at least cursory inspections for factual accuracy ``often enough'' a golden age of information retrieval and computer-assisted learning. Some, on the other hand, worry the tool may lead to unprecedented levels of academic dishonesty and cheating. In this work, we quantify some of the effects of the emergence of Large Language Models (LLMs) on online education by analyzing a multi-year dataset of student essay responses from a free university-level MOOC on AI ethics. Our dataset includes essays submitted both before and after ChatGPT's release. We find that the launch of ChatGPT coincided with significant changes in both the length and style of student essays, mirroring observations in other contexts such as academic publishing. We also observe -- as expected based on related public discourse -- changes in prevalence of key content words related to AI and LLMs, but not necessarily the general themes or topics discussed in the student essays as identified through (dynamic) topic modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13032v1">InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 This paper has been accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11672v3">MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      While current large language models (LLMs) perform well on many knowledge-related tasks, they are limited by relying on their parameters as an implicit storage mechanism. As a result, they struggle with memorizing rare events and with updating their memory as facts change over time. In addition, the uninterpretable nature of parametric memory makes it challenging to prevent hallucination. Model editing and augmenting LLMs with parameters specialized for memory are only partial solutions. In this paper, we introduce MemLLM, a novel method of enhancing LLMs by integrating a structured and explicit read-and-write memory module. MemLLM tackles the aforementioned challenges by enabling dynamic interaction with the memory and improving the LLM's capabilities in using stored knowledge. Our experiments indicate that MemLLM enhances the LLM's performance and interpretability, in language modeling in general and knowledge-intensive tasks in particular. We see MemLLM as an important step towards making LLMs more grounded and factual through memory augmentation. The project repository is publicly available at https://github.com/amodaresi/MemLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12984v1">A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) is critical for AI-powered applications but demands substantial computational resources, particularly in memory bandwidth and computational throughput. Low-precision computation has emerged as a key technique to improve efficiency while reducing resource consumption. Existing approaches for generating low-precision kernels are limited to weight bit widths that are powers of two and suffer from suboptimal performance due to high-level GPU programming abstractions. These abstractions restrict critical optimizations, such as fine-grained register management and optimized memory access patterns, which are essential for efficient low-precision computations. In this paper, we introduce a virtual machine (VM) designed for General-Purpose GPU (GPGPU) computing, enabling support for low-precision data types with arbitrary bit widths while maintaining GPU programmability. The proposed VM features a thread-block-level programming model, a hierarchical memory space, a novel algebraic layout system, and extensive support for diverse low-precision data types. VM programs are compiled into highly efficient GPU programs with automatic vectorization and instruction selection. Extensive experiments demonstrate that our VM efficiently supports a full spectrum of low-precision data types, and outperforms state-of-the-art low-precision kernels on their supported types. Compared to existing compilers like Triton and Ladder, as well as hand-optimized kernels such as QuantLLM and Marlin, our VM achieves performance improvements of 1.75x, 2.61x, 1.29x and 1.03x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12982v1">Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has significantly advanced information retrieval systems, particularly in response generation (RG). Unfortunately, LLMs often face knowledge conflicts between internal memory and retrievaled external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences. However, when the distinction is ambiguous, LLMs experience heightened uncertainty. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models into adaptive augmentation of retrieved information and guiding LLM preference in response generation. Extensive experiments on single-choice, open-ended question-answering (QA), and retrieval augmented generation (RAG) validate our theoretical findings and demonstrate the efficacy of Swin-VIB. Notably, our method improves single-choice task accuracy by at least 7.54\% over competitive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12943v1">Customizing Emotional Support: How Do Individuals Construct and Interact With LLM-Powered Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 20 pages, 3 figures, 3 tables. Accepted to CHI 2025, ACM Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Personalized support is essential to fulfill individuals' emotional needs and sustain their mental well-being. Large language models (LLMs), with great customization flexibility, hold promises to enable individuals to create their own emotional support agents. In this work, we developed ChatLab, where users could construct LLM-powered chatbots with additional interaction features including voices and avatars. Using a Research through Design approach, we conducted a week-long field study followed by interviews and design activities (N = 22), which uncovered how participants created diverse chatbot personas for emotional reliance, confronting stressors, connecting to intellectual discourse, reflecting mirrored selves, etc. We found that participants actively enriched the personas they constructed, shaping the dynamics between themselves and the chatbot to foster open and honest conversations. They also suggested other customizable features, such as integrating online activities and adjustable memory settings. Based on these findings, we discuss opportunities for enhancing personalized emotional support through emerging AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v1">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and modality-of-thought (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://anonymous.4open.science/r/EmoVoice-DF55. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12865v1">DashChat: Interactive Authoring of Industrial Dashboard Design Prototypes through Conversation with LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Industrial dashboards, commonly deployed by organizations such as enterprises and governments, are increasingly crucial in data communication and decision-making support across various domains. Designing an industrial dashboard prototype is particularly challenging due to its visual complexity, which can include data visualization, layout configuration, embellishments, and animations. Additionally, in real-world industrial settings, designers often encounter numerous constraints. For instance, when companies negotiate collaborations with clients and determine design plans, they typically need to demo design prototypes and iterate on them based on mock data quickly. Such a task is very common and crucial during the ideation stage, as it not only helps save developmental costs but also avoids data-related issues such as lengthy data handover periods. However, existing authoring tools of dashboards are mostly not tailored to such prototyping needs, and motivated by these gaps, we propose DashChat, an interactive system that leverages large language models (LLMs) to generate industrial dashboard design prototypes from natural language. We collaborated closely with designers from the industry and derived the requirements based on their practical experience. First, by analyzing 114 high-quality industrial dashboards, we summarized their common design patterns and inject the identified ones into LLMs as reference. Next, we built a multi-agent pipeline powered by LLMs to understand textual requirements from users and generate practical, aesthetic prototypes. Besides, functionally distinct, parallel-operating agents are created to enable efficient generation. Then, we developed a user-friendly interface that supports text-based interaction for generating and modifying prototypes. Two user studies demonstrated that our system is both effective and efficient in supporting design prototyping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12845v1">Can LLMs reason over extended multilingual contexts? Towards long-context evaluation beyond retrieval and haystacks</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 33 Pages in Total - 23 (Main Manuscript) + 10 (Appendix)
    </div>
    <details class="paper-abstract">
      Existing multilingual long-context benchmarks, often based on the popular needle-in-a-haystack test, primarily evaluate a model's ability to locate specific information buried within irrelevant texts. However, such a retrieval-centric approach is myopic and inherently limited, as successful recall alone does not indicate a model's capacity to reason over extended contexts. Moreover, these benchmarks are susceptible to data leakage, short-circuiting, and risk making the evaluation a priori identifiable. To address these limitations, we introduce MLRBench, a new synthetic benchmark for multilingual long-context reasoning. Unlike existing benchmarks, MLRBench goes beyond surface-level retrieval by including tasks that assess multi-hop inference, aggregation, and epistemic reasoning. Spanning seven languages, MLRBench is designed to be parallel, resistant to leakage, and scalable to arbitrary context lengths. Our extensive experiments with an open-weight large language model (LLM) reveal a pronounced gap between high- and low-resource languages, particularly for tasks requiring the model to aggregate multiple facts or predict the absence of information. We also find that, in multilingual settings, LLMs effectively utilize less than 30% of their claimed context length. Although off-the-shelf Retrieval Augmented Generation helps alleviate this to a certain extent, it does not solve the long-context problem. We open-source MLRBench to enable future research in improved evaluation and training of multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12805v1">Assesing LLMs in Art Contexts: Critique Generation and Theory of Mind Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 30 pages, 13 figures, 1 table
    </div>
    <details class="paper-abstract">
      This study explored how large language models (LLMs) perform in two areas related to art: writing critiques of artworks and reasoning about mental states (Theory of Mind, or ToM) in art-related situations. For the critique generation part, we built a system that combines Noel Carroll's evaluative framework with a broad selection of art criticism theories. The model was prompted to first write a full-length critique and then shorter, more coherent versions using a step-by-step prompting process. These AI-generated critiques were then compared with those written by human experts in a Turing test-style evaluation. In many cases, human subjects had difficulty telling which was which, and the results suggest that LLMs can produce critiques that are not only plausible in style but also rich in interpretation, as long as they are carefully guided. In the second part, we introduced new simple ToM tasks based on situations involving interpretation, emotion, and moral tension, which can appear in the context of art. These go beyond standard false-belief tests and allow for more complex, socially embedded forms of reasoning. We tested 41 recent LLMs and found that their performance varied across tasks and models. In particular, tasks that involved affective or ambiguous situations tended to reveal clearer differences. Taken together, these results help clarify how LLMs respond to complex interpretative challenges, revealing both their cognitive limitations and potential. While our findings do not directly contradict the so-called Generative AI Paradox--the idea that LLMs can produce expert-like output without genuine understanding--they suggest that, depending on how LLMs are instructed, such as through carefully designed prompts, these models may begin to show behaviors that resemble understanding more closely than we might assume.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12773v1">Enhancing the Geometric Problem-Solving Ability of Multimodal LLMs via Symbolic-Neural Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Multimodal Large Language Models (MLLMs) have achieved remarkable progress in general domains and demonstrated promise in multimodal mathematical reasoning. However, applying MLLMs to geometry problem solving (GPS) remains challenging due to lack of accurate step-by-step solution data and severe hallucinations during reasoning. In this paper, we propose GeoGen, a pipeline that can automatically generates step-wise reasoning paths for geometry diagrams. By leveraging the precise symbolic reasoning, \textbf{GeoGen} produces large-scale, high-quality question-answer pairs. To further enhance the logical reasoning ability of MLLMs, we train \textbf{GeoLogic}, a Large Language Model (LLM) using synthetic data generated by GeoGen. Serving as a bridge between natural language and symbolic systems, GeoLogic enables symbolic tools to help verifying MLLM outputs, making the reasoning process more rigorous and alleviating hallucinations. Experimental results show that our approach consistently improves the performance of MLLMs, achieving remarkable results on benchmarks for geometric reasoning tasks. This improvement stems from our integration of the strengths of LLMs and symbolic systems, which enables a more reliable and interpretable approach for the GPS task. Codes are available at https://github.com/ycpNotFound/GeoGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12732v1">Validating LLM-Generated Relevance Labels for Educational Resource Search</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Presented in the LLM4Eval Workshop Co-located with WSDM '25 in Hannover, Germany
    </div>
    <details class="paper-abstract">
      Manual relevance judgements in Information Retrieval are costly and require expertise, driving interest in using Large Language Models (LLMs) for automatic assessment. While LLMs have shown promise in general web search scenarios, their effectiveness for evaluating domain-specific search results, such as educational resources, remains unexplored. To investigate different ways of including domain-specific criteria in LLM prompts for relevance judgement, we collected and released a dataset of 401 human relevance judgements from a user study involving teaching professionals performing search tasks related to lesson planning. We compared three approaches to structuring these prompts: a simple two-aspect evaluation baseline from prior work on using LLMs as relevance judges, a comprehensive 12-dimensional rubric derived from educational literature, and criteria directly informed by the study participants. Using domain-specific frameworks, LLMs achieved strong agreement with human judgements (Cohen's $\kappa$ up to 0.650), significantly outperforming the baseline approach. The participant-derived framework proved particularly robust, with GPT-3.5 achieving $\kappa$ scores of 0.639 and 0.613 for 10-dimension and 5-dimension versions respectively. System-level evaluation showed that LLM judgements reliably identified top-performing retrieval approaches (RBO scores 0.71-0.76) while maintaining reasonable discrimination between systems (RBO 0.52-0.56). These findings suggest that LLMs can effectively evaluate educational resources when prompted with domain-specific criteria, though performance varies with framework complexity and input structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24047v2">Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 34 pages, 10 figures
    </div>
    <details class="paper-abstract">
      As scientific research becomes increasingly complex, innovative tools are needed to manage vast data, facilitate interdisciplinary collaboration, and accelerate discovery. Large language models (LLMs) are now evolving into LLM-based scientific agents that automate critical tasks, ranging from hypothesis generation and experiment design to data analysis and simulation. Unlike general-purpose LLMs, these specialized agents integrate domain-specific knowledge, advanced tool sets, and robust validation mechanisms, enabling them to handle complex data types, ensure reproducibility, and drive scientific breakthroughs. This survey provides a focused review of the architectures, design, benchmarks, applications, and ethical considerations surrounding LLM-based scientific agents. We highlight why they differ from general agents and the ways in which they advance research across various scientific fields. By examining their development and challenges, this survey offers a comprehensive roadmap for researchers and practitioners to harness these agents for more efficient, reliable, and ethically sound scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12691v1">Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate hallucinations-content that deviates from factual accuracy or provided context-posing challenges for diagnosis due to the complex interplay of underlying causes. This paper introduces a subsequence association framework to systematically trace and understand hallucinations. Our key insight is that hallucinations arise when dominant hallucinatory associations outweigh faithful ones. Through theoretical and empirical analyses, we demonstrate that decoder-only transformers effectively function as subsequence embedding models, with linear layers encoding input-output associations. We propose a tracing algorithm that identifies causal subsequences by analyzing hallucination probabilities across randomized input contexts. Experiments show our method outperforms standard attribution techniques in identifying hallucination causes and aligns with evidence from the model's training corpus. This work provides a unified perspective on hallucinations and a robust framework for their tracing and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12687v1">Data-efficient LLM Fine-tuning for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 arXiv admin note: text overlap with arXiv:2408.02193
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in code generation tasks. However, there remains a performance gap between open-source and closed-source models. To address this gap, existing approaches typically generate large amounts of synthetic data for fine-tuning, which often leads to inefficient training. In this work, we propose a data selection strategy in order to improve the effectiveness and efficiency of training for code-based LLMs. By prioritizing data complexity and ensuring that the sampled subset aligns with the distribution of the original dataset, our sampling strategy effectively selects high-quality data. Additionally, we optimize the tokenization process through a "dynamic pack" technique, which minimizes padding tokens and reduces computational resource consumption. Experimental results show that when training on 40% of the OSS-Instruct dataset, the DeepSeek-Coder-Base-6.7B model achieves an average performance of 66.9%, surpassing the 66.1% performance with the full dataset. Moreover, training time is reduced from 47 minutes to 34 minutes, and the peak GPU memory decreases from 61.47 GB to 42.72 GB during a single epoch. Similar improvements are observed with the CodeLlama-Python-7B model on the Evol-Instruct dataset. By optimizing both data selection and tokenization, our approach not only improves model performance but also improves training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12682v1">WebLists: Extracting Structured Information From Complex Interactive Websites Using Executable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Most recent web agent research has focused on navigation and transaction tasks, with little emphasis on extracting structured data at scale. We present WebLists, a benchmark of 200 data-extraction tasks across four common business and enterprise use-cases. Each task requires an agent to navigate to a webpage, configure it appropriately, and extract complete datasets with well-defined schemas. We show that both LLMs with search capabilities and SOTA web agents struggle with these tasks, with a recall of 3% and 31%, respectively, despite higher performance on question-answering tasks. To address this challenge, we propose BardeenAgent, a novel framework that enables web agents to convert their execution into repeatable programs, and replay them at scale across pages with similar structure. BardeenAgent is also the first LLM agent to take advantage of the regular structure of HTML. In particular BardeenAgent constructs a generalizable CSS selector to capture all relevant items on the page, then fits the operations to extract the data. On the WebLists benchmark, BardeenAgent achieves 66% recall overall, more than doubling the performance of SOTA web agents, and reducing cost per output row by 3x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12681v1">GRAIL: Gradient-Based Adaptive Unlearning for Privacy and Copyright in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on extensive datasets often learn sensitive information, which raises significant social and legal concerns under principles such as the "Right to be forgotten." Retraining entire models from scratch to remove undesired information is both costly and impractical. Furthermore, existing single-domain unlearning methods fail to address multi-domain scenarios, where knowledge is interwoven across domains such as privacy and copyright, creating overlapping representations that lead to excessive knowledge removal or degraded performance. To tackle these issues, we propose GRAIL (GRadient-based AdaptIve unLearning), a novel multi-domain unlearning framework. GRAIL leverages gradient information from multiple domains to precisely distinguish the unlearning scope from the retention scope, and applies an adaptive parameter-wise localization strategy to selectively remove targeted knowledge while preserving critical parameters for each domain. Experimental results on unlearning benchmarks show that GRAIL achieves unlearning success on par with the existing approaches, while also demonstrating up to 17% stronger knowledge retention success compared to the previous state-of-art method. Our findings establish a new paradigm for effectively managing and regulating sensitive information in large-scale pre-trained language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12637v1">Scaling Instruction-Tuned LLMs to Million-Token Contexts via Hierarchical Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 26 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) struggle with long-context reasoning, not only due to the quadratic scaling of computational complexity with sequence length but also because of the scarcity and expense of annotating long-context data. There has been barely any open-source work that systematically ablates long-context data, nor is there any openly available instruction tuning dataset with contexts surpassing 100K tokens. To bridge this gap, we introduce a novel post-training synthetic data generation strategy designed to efficiently extend the context window of LLMs while preserving their general task performance. Our approach scalably extends to arbitrarily long context lengths, unconstrained by the length of available real-world data, which effectively addresses the scarcity of raw long-context data. Through a step-by-step rotary position embedding (RoPE) scaling training strategy, we demonstrate that our model, with a context length of up to 1M tokens, performs well on the RULER benchmark and InfiniteBench and maintains robust performance on general language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12876v3">In-context KV-Cache Eviction for LLMs via Attention-Gate</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      The KV-Cache technique has become the standard for the inference of large language models (LLMs). Yet, it is widely criticized that KV-Cache can become a bottleneck of the LLM inference system. This paper enables a novel dynamic KV-Cache eviction policy by injecting a lightweight module called Attention-Gate to the model. It accepts the global context as input and yields eviction flags for each token. The self-attention modules in the model proceed according to the flags and cache only a subset of the KV states for next token prediction. The Attention-Gates can yield various flags for different heads and layers and be easily tuned on top of a pre-trained LLM via continual pre-training or supervised fine-tuning. The computational and memory overhead introduced by Attention-Gates can be minimal. We empirically evaluate the proposed approach across multiple scenarios, showing that effective eviction of redundant tokens can not only improve efficiency but also enhance performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09593v2">ControlNET: A Firewall for RAG-based LLM System</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 Project Page: https://ai.zjuicsr.cn/firewall
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15395v2">Beyond Tools: Understanding How Heavy Users Integrate LLMs into Everyday Tasks and Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for both everyday and specialized tasks. While HCI research focuses on domain-specific applications, little is known about how heavy users integrate LLMs into everyday decision-making. Through qualitative interviews with heavy LLM users (n=7) who employ these systems for both intuitive and analytical thinking tasks, our findings show that participants use LLMs for social validation, self-regulation, and interpersonal guidance, seeking to build self-confidence and optimize cognitive resources. These users viewed LLMs either as rational, consistent entities or average human decision-makers. Our findings suggest that heavy LLM users develop nuanced interaction patterns beyond simple delegation, highlighting the need to reconsider how we study LLM integration in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v3">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12608v1">Code Copycat Conundrum: Demystifying Repetition in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Despite recent advances in Large Language Models (LLMs) for code generation, the quality of LLM-generated code still faces significant challenges. One significant issue is code repetition, which refers to the model's tendency to generate structurally redundant code, resulting in inefficiencies and reduced readability. To address this, we conduct the first empirical study to investigate the prevalence and nature of repetition across 19 state-of-the-art code LLMs using three widely-used benchmarks. Our study includes both quantitative and qualitative analyses, revealing that repetition is pervasive and manifests at various granularities and extents, including character, statement, and block levels. We further summarize a taxonomy of 20 repetition patterns. Building on our findings, we propose DeRep, a rule-based technique designed to detect and mitigate repetition in generated code. We evaluate DeRep using both open-source benchmarks and in an industrial setting. Our results demonstrate that DeRep significantly outperforms baselines in reducing repetition (with an average improvements of 91.3%, 93.5%, and 79.9% in rep-3, rep-line, and sim-line metrics) and enhancing code quality (with a Pass@1 increase of 208.3% over greedy search). Furthermore, integrating DeRep improves the performance of existing repetition mitigation methods, with Pass@1 improvements ranging from 53.7% to 215.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07791v8">Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has emerged as a promising alternative to human evaluators across various tasks, yet inherent biases - particularly position bias, the tendency to favor solutions based on their position within the prompt - compromise its reliability. This exploratory study evaluates position bias in LLM judges across pairwise and list-wise comparison settings, introducing three metrics: repetition stability, position consistency, and preference fairness. Our experiments, involving 15 LLM judges across MTBench and DevBench with 22 tasks and approximately 40 solution-generating models, result in over 150,000 evaluation instances. We identify Judge-Level, Candidate-Level, and Task-Level factors contributing to bias. The findings confirm that position bias is not due to random chance and varies significantly across judges and tasks. While position bias is weakly influenced by the length of prompt components, it is strongly affected by the quality gap between solutions. Our agreement and disagreement analysis among judges further provides insights into the distribution of judging difficulty across the dataset, and highlights the potential for dataset modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11711v2">The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10673v2">ZeroSumEval: An Extensible Framework For Scaling LLM Evaluation with Inter-Model Competition</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      We introduce ZeroSumEval, a dynamic, competition-based, and evolving evaluation framework for Large Language Models (LLMs) that leverages competitive games. ZeroSumEval encompasses a diverse suite of games, including security challenges (Capture the Flag), classic board games (chess), and knowledge tests (MathQuiz). These games are designed to evaluate a range of capabilities such as strategic reasoning, planning, knowledge application, safety, and adaptability. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework for easily implementing games and leverages DSPy to provide a better abstraction for LLM player strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12589v1">Efficient MAP Estimation of LLM Judgment Performance with Prior Transfer</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      LLM ensembles are widely used for LLM judges. However, how to estimate their accuracy, especially in an efficient way, is unknown. In this paper, we present a principled maximum a posteriori (MAP) framework for an economical and precise estimation of the performance of LLM ensemble judgment. We first propose a mixture of Beta-Binomial distributions to model the judgment distribution, revising from the vanilla Binomial distribution. Next, we introduce a conformal prediction-driven approach that enables adaptive stopping during iterative sampling to balance accuracy with efficiency. Furthermore, we design a prior transfer mechanism that utilizes learned distributions on open-source datasets to improve estimation on a target dataset when only scarce annotations are available. Finally, we present BetaConform, a framework that integrates our distribution assumption, adaptive stopping, and the prior transfer mechanism to deliver a theoretically guaranteed distribution estimation of LLM ensemble judgment with minimum labeled samples. BetaConform is also validated empirically. For instance, with only 10 samples from the TruthfulQA dataset, for a Llama ensembled judge, BetaConform gauges its performance with error margin as small as 3.37%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08260v2">Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v3">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-04-17
      | 💬 45 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have gained significant attention for their exceptional capabilities in natural language processing and multimodal data understanding. Meanwhile, the rapid expansion of information services has driven the growing need for intelligence, efficient, and adaptable wireless networks. Wireless networks require the empowerment of RL-based LLMs while these models also benefit from wireless networks to broaden their application scenarios. Specifically, RL-based LLMs can enhance wireless communication systems through intelligent resource allocation, adaptive network optimization, and real-time decision-making. Conversely, wireless networks provide a vital infrastructure for the efficient training, deployment, and distributed inference of RL-based LLMs, especially in decentralized and edge computing environments. This mutual empowerment highlights the need for a deeper exploration of the interplay between these two domains. We first review recent advancements in wireless communications, highlighting the associated challenges and potential solutions. We then discuss the progress of RL-based LLMs, focusing on key technologies for LLM training, challenges, and potential solutions. Subsequently, we explore the mutual empowerment between these two fields, highlighting key motivations, open challenges, and potential solutions. Finally, we provide insights into future directions, applications, and their societal impact to further explore this intersection, paving the way for next-generation intelligent communication systems. Overall, this survey provides a comprehensive overview of the relationship between RL-based LLMs and wireless networks, offering a vision where these domains empower each other to drive innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12562v1">ZeroSumEval: Scaling LLM Evaluation with Inter-Model Competition</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities of Large Language Models (LLMs) has traditionally relied on static benchmark datasets, human assessments, or model-based evaluations - methods that often suffer from overfitting, high costs, and biases. ZeroSumEval is a novel competition-based evaluation protocol that leverages zero-sum games to assess LLMs with dynamic benchmarks that resist saturation. ZeroSumEval encompasses a diverse suite of games, including security challenges (PyJail), classic games (Chess, Liar's Dice, Poker), knowledge tests (MathQuiz), and persuasion challenges (Gandalf, Debate). These games are designed to evaluate a range of AI capabilities such as strategic reasoning, planning, knowledge application, and creativity. Building upon recent studies that highlight the effectiveness of game-based evaluations for LLMs, ZeroSumEval enhances these approaches by providing a standardized and extensible framework. To demonstrate this, we conduct extensive experiments with >7000 simulations across 7 games and 13 models. Our results show that while frontier models from the GPT and Claude families can play common games and answer questions, they struggle to play games that require creating novel and challenging questions. We also observe that models cannot reliably jailbreak each other and fail generally at tasks requiring creativity. We release our code at https://github.com/facebookresearch/ZeroSumEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12558v1">Benchmarking LLM-based Relevance Judgment Methods</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in both academic and industry settings to automate the evaluation of information seeking systems, particularly by generating graded relevance judgments. Previous work on LLM-based relevance assessment has primarily focused on replicating graded human relevance judgments through various prompting strategies. However, there has been limited exploration of alternative assessment methods or comprehensive comparative studies. In this paper, we systematically compare multiple LLM-based relevance assessment methods, including binary relevance judgments, graded relevance assessments, pairwise preference-based methods, and two nugget-based evaluation methods~--~document-agnostic and document-dependent. In addition to a traditional comparison based on system rankings using Kendall correlations, we also examine how well LLM judgments align with human preferences, as inferred from relevance grades. We conduct extensive experiments on datasets from three TREC Deep Learning tracks 2019, 2020 and 2021 as well as the ANTIQUE dataset, which focuses on non-factoid open-domain question answering. As part of our data release, we include relevance judgments generated by both an open-source (Llama3.2b) and a commercial (gpt-4o) model. Our goal is to \textit{reproduce} various LLM-based relevance judgment methods to provide a comprehensive comparison. All code, data, and resources are publicly available in our GitHub Repository at https://github.com/Narabzad/llm-relevance-judgement-comparison.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12553v1">ELAB: Extensive LLM Alignment Benchmark in Persian Language</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive evaluation framework for aligning Persian Large Language Models (LLMs) with critical ethical dimensions, including safety, fairness, and social norms. It addresses the gaps in existing LLM evaluation frameworks by adapting them to Persian linguistic and cultural contexts. This benchmark creates three types of Persian-language benchmarks: (i) translated data, (ii) new data generated synthetically, and (iii) new naturally collected data. We translate Anthropic Red Teaming data, AdvBench, HarmBench, and DecodingTrust into Persian. Furthermore, we create ProhibiBench-fa, SafeBench-fa, FairBench-fa, and SocialBench-fa as new datasets to address harmful and prohibited content in indigenous culture. Moreover, we collect extensive dataset as GuardBench-fa to consider Persian cultural norms. By combining these datasets, our work establishes a unified framework for evaluating Persian LLMs, offering a new approach to culturally grounded alignment evaluation. A systematic evaluation of Persian LLMs is performed across the three alignment aspects: safety (avoiding harmful content), fairness (mitigating biases), and social norms (adhering to culturally accepted behaviors). We present a publicly available leaderboard that benchmarks Persian LLMs with respect to safety, fairness, and social norms at: https://huggingface.co/spaces/MCILAB/LLM_Alignment_Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12545v1">Knowledge Acquisition on Mass-shooting Events via LLMs for AI-Driven Justice</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Mass-shooting events pose a significant challenge to public safety, generating large volumes of unstructured textual data that hinder effective investigations and the formulation of public policy. Despite the urgency, few prior studies have effectively automated the extraction of key information from these events to support legal and investigative efforts. This paper presented the first dataset designed for knowledge acquisition on mass-shooting events through the application of named entity recognition (NER) techniques. It focuses on identifying key entities such as offenders, victims, locations, and criminal instruments, that are vital for legal and investigative purposes. The NER process is powered by Large Language Models (LLMs) using few-shot prompting, facilitating the efficient extraction and organization of critical information from diverse sources, including news articles, police reports, and social media. Experimental results on real-world mass-shooting corpora demonstrate that GPT-4o is the most effective model for mass-shooting NER, achieving the highest Micro Precision, Micro Recall, and Micro F1-scores. Meanwhile, o1-mini delivers competitive performance, making it a resource-efficient alternative for less complex NER tasks. It is also observed that increasing the shot count enhances the performance of all models, but the gains are more substantial for GPT-4o and o1-mini, highlighting their superior adaptability to few-shot learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23798v2">Adaptive Layer-skipping in Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      Various layer-skipping methods have been proposed to accelerate token generation in large language models (LLMs). However, they have overlooked a fundamental question: How do computational demands vary across the generation of different tokens? In this work, we introduce FlexiDepth, a method that dynamically adjusts the number of Transformer layers used in text generation. By incorporating a plug-in router and adapter, FlexiDepth enables adaptive layer-skipping in LLMs without modifying their original parameters. Introducing FlexiDepth to Llama-3-8B model achieves layer skipping of 8 layers out of 32, and meanwhile maintains the full 100\% benchmark performance. Experimental results with FlexiDepth demonstrate that computational demands in LLMs significantly vary based on token type. Specifically, generating repetitive tokens or fixed phrases requires fewer layers, whereas producing tokens involving computation or high uncertainty requires more layers. Interestingly, this adaptive allocation pattern aligns with human intuition. To advance research in this area, we open sourced FlexiDepth and a dataset documenting FlexiDepth's layer allocation patterns for future exploration.
    </details>
</div>
