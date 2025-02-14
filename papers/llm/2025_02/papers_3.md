# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17040v2">Arabic Dataset for LLM Safeguard Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 Accepted at NAACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The growing use of large language models (LLMs) has raised concerns regarding their safety. While many studies have focused on English, the safety of LLMs in Arabic, with its linguistic and cultural complexities, remains under-explored. Here, we aim to bridge this gap. In particular, we present an Arab-region-specific safety evaluation dataset consisting of 5,799 questions, including direct attacks, indirect attacks, and harmless requests with sensitive words, adapted to reflect the socio-cultural context of the Arab world. To uncover the impact of different stances in handling sensitive and controversial topics, we propose a dual-perspective evaluation framework. It assesses the LLM responses from both governmental and opposition viewpoints. Experiments over five leading Arabic-centric and multilingual LLMs reveal substantial disparities in their safety performance. This reinforces the need for culturally specific datasets to ensure the responsible deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05790v1">I3S: Importance Sampling Subspace Selection for Low-Rank Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      Low-rank optimization has emerged as a promising approach to enabling memory-efficient training of large language models (LLMs). Existing low-rank optimization methods typically project gradients onto a low-rank subspace, reducing the memory cost of storing optimizer states. A key challenge in these methods is identifying suitable subspaces to ensure an effective optimization trajectory. Most existing approaches select the dominant subspace to preserve gradient information, as this intuitively provides the best approximation. However, we find that in practice, the dominant subspace stops changing during pretraining, thereby constraining weight updates to similar subspaces. In this paper, we propose importance sampling subspace selection (I3S) for low-rank optimization, which theoretically offers a comparable convergence rate to the dominant subspace approach. Empirically, we demonstrate that I3S significantly outperforms previous methods in LLM pretraining tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05782v1">Quality Assurance for LLM-RAG Systems: Empirical Insights from Tourism Application Testing</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive framework for testing and evaluating quality characteristics of Large Language Model (LLM) systems enhanced with Retrieval-Augmented Generation (RAG) in tourism applications. Through systematic empirical evaluation of three different LLM variants across multiple parameter configurations, we demonstrate the effectiveness of our testing methodology in assessing both functional correctness and extra-functional properties. Our framework implements 17 distinct metrics that encompass syntactic analysis, semantic evaluation, and behavioral evaluation through LLM judges. The study reveals significant information about how different architectural choices and parameter configurations affect system performance, particularly highlighting the impact of temperature and top-p parameters on response quality. The tests were carried out on a tourism recommendation system for the V\"armland region, utilizing standard and RAG-enhanced configurations. The results indicate that the newer LLM versions show modest improvements in performance metrics, though the differences are more pronounced in response length and complexity rather than in semantic quality. The research contributes practical insights for implementing robust testing practices in LLM-RAG systems, providing valuable guidance to organizations deploying these architectures in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14012v2">LLMs are Biased Teachers: Evaluating LLM Bias in Personalized Education</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 49 Pages, 55 Figures, NAACL Findings 2025
    </div>
    <details class="paper-abstract">
      With the increasing adoption of large language models (LLMs) in education, concerns about inherent biases in these models have gained prominence. We evaluate LLMs for bias in the personalized educational setting, specifically focusing on the models' roles as "teachers." We reveal significant biases in how models generate and select educational content tailored to different demographic groups, including race, ethnicity, sex, gender, disability status, income, and national origin. We introduce and apply two bias score metrics--Mean Absolute Bias (MAB) and Maximum Difference Bias (MDB)--to analyze 9 open and closed state-of-the-art LLMs. Our experiments, which utilize over 17,000 educational explanations across multiple difficulty levels and topics, uncover that models potentially harm student learning by both perpetuating harmful stereotypes and reversing them. We find that bias is similar for all frontier models, with the highest MAB along income levels while MDB is highest relative to both income and disability status. For both metrics, we find the lowest bias exists for sex/gender and race/ethnicity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05049v3">ProverbEval: Exploring LLM Evaluation Challenges for Low-resource Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      With the rapid development of evaluation datasets to assess LLMs understanding across a wide range of subjects and domains, identifying a suitable language understanding benchmark has become increasingly challenging. In this work, we explore LLM evaluation challenges for low-resource language understanding and introduce \proverbeval, LLM evaluation benchmark for low-resource languages, focusing on low-resource language understanding in culture-specific scenarios. We benchmark various LLMs and explore factors that create variability in the benchmarking process. We observed performance variances of up to 50\%, depending on the order in which answer choices were presented in multiple-choice tasks. Native language proverb descriptions significantly improve tasks such as proverb generation, contributing to improved outcomes. Additionally, monolingual evaluations consistently outperformed their cross-lingual counterparts in generation tasks. We argue that special attention must be given to the order of choices, the choice of prompt language, task variability, and generation tasks when creating LLM evaluation benchmarks. Evaluation data available at https://huggingface.co/datasets/israel/ProverbEval, evaluation code https://github.com/EthioNLP/EthioProverbEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06890v1">LLMs for Drug-Drug Interaction Prediction: A Comprehensive Comparison</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      The increasing volume of drug combinations in modern therapeutic regimens needs reliable methods for predicting drug-drug interactions (DDIs). While Large Language Models (LLMs) have revolutionized various domains, their potential in pharmaceutical research, particularly in DDI prediction, remains largely unexplored. This study thoroughly investigates LLMs' capabilities in predicting DDIs by uniquely processing molecular structures (SMILES), target organisms, and gene interaction data as raw text input from the latest DrugBank dataset. We evaluated 18 different LLMs, including proprietary models (GPT-4, Claude, Gemini) and open-source variants (from 1.5B to 72B parameters), first assessing their zero-shot capabilities in DDI prediction. We then fine-tuned selected models (GPT-4, Phi-3.5 2.7B, Qwen-2.5 3B, Gemma-2 9B, and Deepseek R1 distilled Qwen 1.5B) to optimize their performance. Our comprehensive evaluation framework included validation across 13 external DDI datasets, comparing against traditional approaches such as l2-regularized logistic regression. Fine-tuned LLMs demonstrated superior performance, with Phi-3.5 2.7B achieving a sensitivity of 0.978 in DDI prediction, with an accuracy of 0.919 on balanced datasets (50% positive, 50% negative cases). This result represents an improvement over both zero-shot predictions and state-of-the-art machine-learning methods used for DDI prediction. Our analysis reveals that LLMs can effectively capture complex molecular interaction patterns and cases where drug pairs target common genes, making them valuable tools for practical applications in pharmaceutical research and clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08417v2">AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 In Proceedings of the Conference on Applied Machine Learning in Information Security, 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly capable of completing knowledge intensive tasks by recalling information from a static pretraining corpus. Here we are concerned with LLMs in the context of evolving data requirements. For instance: batches of new data that are introduced periodically; subsets of data with user-based access controls; or requirements on dynamic removal of documents with guarantees that associated knowledge cannot be recalled. We wish to satisfy these requirements while at the same time ensuring a model does not forget old information when new data becomes available. To address these issues, we introduce AdapterSwap, a training and inference scheme that organizes knowledge from a data collection into a set of low-rank adapters, which are dynamically composed during inference. Our experiments demonstrate AdapterSwap's ability to support efficient continual learning, while also enabling organizations to have fine-grained control over data access and deletion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06004v1">Analysis of LLM as a grammatical feature tagger for African American English</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 13 pages, Accepted to "Findings of the Association for Computational Linguistics: NAACL 2025"
    </div>
    <details class="paper-abstract">
      African American English (AAE) presents unique challenges in natural language processing (NLP). This research systematically compares the performance of available NLP models--rule-based, transformer-based, and large language models (LLMs)--capable of identifying key grammatical features of AAE, namely Habitual Be and Multiple Negation. These features were selected for their distinct grammatical complexity and frequency of occurrence. The evaluation involved sentence-level binary classification tasks, using both zero-shot and few-shot strategies. The analysis reveals that while LLMs show promise compared to the baseline, they are influenced by biases such as recency and unrelated features in the text such as formality. This study highlights the necessity for improved model training and architectural adjustments to better accommodate AAE's unique linguistic characteristics. Data and code are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04801v3">Alpaca against Vicuna: Using LLMs to Uncover Memorization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a black-box prompt optimization method that uses an attacker LLM agent to uncover higher levels of memorization in a victim agent, compared to what is revealed by prompting the target model with the training data directly, which is the dominant approach of quantifying memorization in LLMs. We use an iterative rejection-sampling optimization process to find instruction-based prompts with two main characteristics: (1) minimal overlap with the training data to avoid presenting the solution directly to the model, and (2) maximal overlap between the victim model's output and the training data, aiming to induce the victim to spit out training data. We observe that our instruction-based prompts generate outputs with 23.7% higher overlap with training data compared to the baseline prefix-suffix measurements. Our findings show that (1) instruction-tuned models can expose pre-training data as much as their base-models, if not more so, (2) contexts other than the original training data can lead to leakage, and (3) using instructions proposed by other LLMs can open a new avenue of automated attacks that we should further study and explore. The code can be found at https://github.com/Alymostafa/Instruction_based_attack .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05982v1">HamRaz: A Culture-Based Persian Conversation Dataset for Person-Centered Therapy Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      This paper presents HamRaz, a novel Persian-language mental health dataset designed for Person-Centered Therapy (PCT) using Large Language Models (LLMs). Despite the growing application of LLMs in AI-driven psychological counseling, existing datasets predominantly focus on Western and East Asian contexts, overlooking cultural and linguistic nuances essential for effective Persian-language therapy. To address this gap, HamRaz combines script-based dialogues with adaptive LLM role-playing, ensuring coherent and dynamic therapy interactions. We also introduce HamRazEval, a dual evaluation framework that measures conversational quality and therapeutic effectiveness using General Dialogue Metrics and the Barrett-Lennard Relationship Inventory (BLRI). Experimental results show HamRaz outperforms conventional Script Mode and Two-Agent Mode, producing more empathetic, context-aware, and realistic therapy sessions. By releasing HamRaz, we contribute a culturally adapted, LLM-driven resource to advance AI-powered psychotherapy research in diverse communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15999v3">Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 Accepted at NAACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can store a significant amount of factual knowledge in their parameters. However, their parametric knowledge may conflict with the information provided in the context -- this phenomenon, known as \emph{context-memory knowledge conflicts}, can lead to undesirable model behaviour, such as reliance on outdated or incorrect information. Analysing the internal activations of LLMs, we find that they can internally register the signals of knowledge conflict at mid-layers. Such signals allow us to detect whether a knowledge conflict occurs and use \emph{inference-time} intervention strategies to resolve it. In this work, we propose \textsc{SpARE}, a \emph{training-free} representation engineering method that uses pre-trained sparse auto-encoders (SAEs) to control the knowledge selection behaviour of LLMs. \textsc{SpARE} identifies the functional features that control the knowledge selection behaviours and applies them to edit the internal activations of LLMs at inference time. Our experimental results show that \textsc{SpARE} can effectively control the usage of either knowledge source to resolve knowledge conflict in open-domain question-answering tasks, surpassing existing representation engineering methods ($+10\%$) as well as contrastive decoding methods ($+15\%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05957v1">MetaChain: A Fully-Automated and Zero-Code Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 Code: https://github.com/HKUDS/MetaChain
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce MetaChain-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, MetaChain comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, MetaChain also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate MetaChain's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, MetaChain's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14635v2">GenEOL: Harnessing the Generative Power of LLMs for Training-Free Sentence Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 NAACL Findings 2025, 9 pages, 4 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Training-free embedding methods directly leverage pretrained large language models (LLMs) to embed text, bypassing the costly and complex procedure of contrastive learning. Previous training-free embedding methods have mainly focused on optimizing embedding prompts and have overlooked the benefits of utilizing the generative abilities of LLMs. We propose a novel method, GenEOL, which uses LLMs to generate diverse transformations of a sentence that preserve its meaning, and aggregates the resulting embeddings of these transformations to enhance the overall sentence embedding. GenEOL significantly outperforms the existing training-free embedding methods by an average of 2.85 points across several LLMs on the sentence semantic text similarity (STS) benchmark. GenEOL also achieves notable gains in clustering, reranking, and pair-classification tasks from the MTEB benchmark. Additionally, GenEOL stabilizes representation quality across LLM layers and remains robust to perturbations of embedding prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17874v2">Evaluating LLM Reasoning in the Operations Research Domain with ORQA</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 12 pages, 10 figures. Accepted and to be published in AAAI25
    </div>
    <details class="paper-abstract">
      In this paper, we introduce and apply Operations Research Question Answering (ORQA), a new benchmark designed to assess the generalization capabilities of Large Language Models (LLMs) in the specialized technical domain of Operations Research (OR). This benchmark evaluates whether LLMs can emulate the knowledge and reasoning skills of OR experts when confronted with diverse and complex optimization problems. The dataset, developed by OR experts, features real-world optimization problems that demand multistep reasoning to construct their mathematical models. Our evaluations of various open source LLMs, such as LLaMA 3.1, DeepSeek, and Mixtral, reveal their modest performance, highlighting a gap in their ability to generalize to specialized technical domains. This work contributes to the ongoing discourse on LLMs generalization capabilities, offering valuable insights for future research in this area. The dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05947v1">Acceleration Multiple Heads Decoding for LLM via Dynamic Tree Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      Multiple heads decoding accelerates the inference of Large Language Models (LLMs) by predicting next several tokens simultaneously. It generates and verifies multiple candidate sequences in parallel via tree attention with a fixed structure. In this paper, we replace the fixed tree attention with dynamic tree attention on multiple head decoding, specifically in the context of MEDUSA. We propose a simple and low complexity strategy to generate candidates and construct the dynamic tree structure. Preliminary experiments show that the proposed method improves the decoding efficiency of multiple head decoding for LLMs while maintaining the generation quality. This result demonstrates the potential for improvement of multiple head decoding in candidate generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10181v2">Practical offloading for fine-tuning LLM on commodity GPU via learned sparse projectors</a></div>
    <div class="paper-meta">
      📅 2025-02-09
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) requires significant memory, often exceeding the capacity of a single GPU. A common solution to this memory challenge is offloading compute and data from the GPU to the CPU. However, this approach is hampered by the limited bandwidth of commodity hardware, which constrains communication between the CPU and GPU, and by slower matrix multiplications on the CPU. In this paper, we present an offloading framework, LSP-Offload, that enables near-native speed LLM fine-tuning on commodity hardware through learned sparse projectors. Our data-driven approach involves learning efficient sparse compressors that minimize communication with minimal precision loss. Additionally, we introduce a novel layer-wise communication schedule to maximize parallelism between communication and computation. As a result, our framework can fine-tune a 1.3 billion parameter model on a 4GB laptop GPU and a 6.7 billion parameter model on a 24GB NVIDIA RTX 4090 GPU. Compared to state-of-the-art offloading frameworks, our approach reduces end-to-end fine-tuning time by 33.1%-62.5% when converging to the same accuracy. We open source our framework at https://github.com/gulang2019/LSP-Offload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23252v2">Evaluating Cultural and Social Awareness of LLM Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) expand into performing as agents for real-world applications beyond traditional NLP tasks, evaluating their robustness becomes increasingly important. However, existing benchmarks often overlook critical dimensions like cultural and social awareness. To address these, we introduce CASA, a benchmark designed to assess LLM agents' sensitivity to cultural and social norms across two web-based tasks: online shopping and social discussion forums. Our approach evaluates LLM agents' ability to detect and appropriately respond to norm-violating user queries and observations. Furthermore, we propose a comprehensive evaluation framework that measures awareness coverage, helpfulness in managing user queries, and the violation rate when facing misleading web content. Experiments show that current LLMs perform significantly better in non-agent than in web-based agent environments, with agents achieving less than 10% awareness coverage and over 40% violation rates. To improve performance, we explore two methods: prompting and fine-tuning, and find that combining both methods can offer complementary advantages -- fine-tuning on culture-specific datasets significantly enhances the agents' ability to generalize across different regions, while prompting boosts the agents' ability to navigate complex tasks. These findings highlight the importance of constantly benchmarking LLM agents' cultural and social awareness during the development cycle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07162v3">How Effectively Do LLMs Extract Feature-Sentiment Pairs from App Reviews?</a></div>
    <div class="paper-meta">
      📅 2025-02-09
      | 💬 The summary of the project is available at https://bit.ly/3XGcRM1
    </div>
    <details class="paper-abstract">
      Automatic analysis of user reviews to understand user sentiments toward app functionality (i.e. app features) helps align development efforts with user expectations and needs. Recent advances in Large Language Models (LLMs) such as ChatGPT have shown impressive performance on several new tasks without updating the model's parameters i.e. using zero or a few labeled examples, but the capabilities of LLMs are yet unexplored for feature-specific sentiment analysis. The goal of our study is to explore the capabilities of LLMs to perform feature-specific sentiment analysis of user reviews. This study compares the performance of state-of-the-art LLMs, including GPT-4, ChatGPT, and different variants of Llama-2 chat, against previous approaches for extracting app features and associated sentiments in zero-shot, 1-shot, and 5-shot scenarios. The results indicate that GPT-4 outperforms the rule-based SAFE by 17% in f1-score for extracting app features in the zero-shot scenario, with 5-shot further improving it by 6%. However, the fine-tuned RE-BERT exceeds GPT-4 by 6% in f1-score. For predicting positive and neutral sentiments, GPT-4 achieves f1-scores of 76% and 45% in the zero-shot setting, which improve by 7% and 23% in the 5-shot setting, respectively. Our study conducts a thorough evaluation of both proprietary and open-source LLMs to provide an objective assessment of their performance in extracting feature-sentiment pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09565v2">Obfuscated Activations Bypass LLM Latent-Space Defenses</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Project page: https://obfuscated-activations.github.io/ Code: https://github.com/LukeBailey181/obfuscated-activations
    </div>
    <details class="paper-abstract">
      Recent latent-space monitoring techniques have shown promise as defenses against LLM attacks. These defenses act as scanners that seek to detect harmful activations before they lead to undesirable actions. This prompts the question: Can models execute harmful behavior via inconspicuous latent states? Here, we study such obfuscated activations. We show that state-of-the-art latent-space defenses -- including sparse autoencoders, representation probing, and latent OOD detection -- are all vulnerable to obfuscated activations. For example, against probes trained to classify harmfulness, our attacks can often reduce recall from 100% to 0% while retaining a 90% jailbreaking rate. However, obfuscation has limits: we find that on a complex task (writing SQL code), obfuscation reduces model performance. Together, our results demonstrate that neural activations are highly malleable: we can reshape activation patterns in a variety of ways, often while preserving a network's behavior. This poses a fundamental challenge to latent-space defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13588v2">ChainBuddy: An AI Agent System for Generating LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 21 pages, 12 figures, pre-print
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance, their potential applications have grown significantly. However, it remains difficult to evaluate LLM behavior on user-defined tasks and craft effective pipelines to do so. Many users struggle with where to start, often referred to as the "blank page problem." ChainBuddy, an AI workflow generation assistant built into the ChainForge platform, aims to tackle this issue. From a single prompt or chat, ChainBuddy generates a starter evaluative LLM pipeline in ChainForge aligned to the user's requirements. ChainBuddy offers a straightforward and user-friendly way to plan and evaluate LLM behavior and make the process less daunting and more accessible across a wide range of possible tasks and use cases. We report a within-subjects user study comparing ChainBuddy to the baseline interface. We find that when using AI assistance, participants reported a less demanding workload, felt more confident, and produced higher quality pipelines evaluating LLM behavior. However, we also uncover a mismatch between subjective and objective ratings of performance: participants rated their successfulness similarly across conditions, while independent experts rated participant workflows significantly higher with AI assistance. Drawing connections to the Dunning-Kruger effect, we draw design implications for the future of workflow generation assistants to mitigate the risk of over-reliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15352v2">CompAct: Compressed Activations for Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      We introduce CompAct, a technique that reduces peak memory utilization on GPU by 25-30% for pretraining and 50% for fine-tuning of LLMs. Peak device memory is a major limiting factor in training LLMs, with various recent works aiming to reduce model memory. However most works don't target the largest component of allocated memory during training: the model's compute graph, which is stored for the backward pass. By storing low-rank, compressed activations to be used in the backward pass we greatly reduce the required memory, unlike previous methods which only reduce optimizer overheads or the number of trained parameters. Our compression uses random projection matrices, thus avoiding additional memory overheads. Comparisons with previous techniques for either pretraining or fine-tuning show that CompAct substantially improves existing compute-performance tradeoffs. We expect CompAct's savings to scale even higher for larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15589v2">LLMs as Meta-Reviewers' Assistants: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Accepted to NAACL 2025, 41 pages
    </div>
    <details class="paper-abstract">
      One of the most important yet onerous tasks in the academic peer-reviewing process is composing meta-reviews, which involves assimilating diverse opinions from multiple expert peers, formulating one's self-judgment as a senior expert, and then summarizing all these perspectives into a concise holistic overview to make an overall recommendation. This process is time-consuming and can be compromised by human factors like fatigue, inconsistency, missing tiny details, etc. Given the latest major developments in Large Language Models (LLMs), it is very compelling to rigorously study whether LLMs can help metareviewers perform this important task better. In this paper, we perform a case study with three popular LLMs, i.e., GPT-3.5, LLaMA2, and PaLM2, to assist meta-reviewers in better comprehending multiple experts perspectives by generating a controlled multi-perspective summary (MPS) of their opinions. To achieve this, we prompt three LLMs with different types/levels of prompts based on the recently proposed TELeR taxonomy. Finally, we perform a detailed qualitative study of the MPSs generated by the LLMs and report our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13299v2">Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Accepted in NAACL 2025
    </div>
    <details class="paper-abstract">
      Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05675v1">Investigating the Shortcomings of LLMs in Step-by-Step Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Accepted to NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Reasoning abilities of LLMs have been a key focus in recent years. One challenging reasoning domain with interesting nuances is legal reasoning, which requires careful application of rules, and precedents while balancing deductive and analogical reasoning, and conflicts between rules. Although there have been a few works on using LLMs for legal reasoning, their focus has been on overall accuracy. In this paper, we dig deeper to do a step-by-step analysis and figure out where they commit errors. We use the college-level Multiple Choice Question-Answering (MCQA) task from the \textit{Civil Procedure} dataset and propose a new error taxonomy derived from initial manual analysis of reasoning chains with respect to several LLMs, including two objective measures: soundness and correctness scores. We then develop an LLM-based automated evaluation framework to identify reasoning errors and evaluate the performance of LLMs. The computation of soundness and correctness on the dataset using the auto-evaluator framework reveals several interesting insights. Furthermore, we show that incorporating the error taxonomy as feedback in popular prompting techniques marginally increases LLM performance. Our work will also serve as an evaluation framework that can be used in detailed error analysis of reasoning chains for logic-intensive complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05612v1">Rambler in the Wild: A Diary Study of LLM-Assisted Writing With Speech</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Speech-to-text technologies have been shown to improve text input efficiency and potentially lower the barriers to writing. Recent LLM-assisted dictation tools aim to support writing with speech by bridging the gaps between speaking and traditional writing. This case study reports on the real-world writing experiences of twelve academic or creative writers using one such tool, Rambler, to write various pieces such as blog posts, diaries, screenplays, notes, or fictional stories, etc. Through a ten-day diary study, we identified the participants' in-context writing strategies using Rambler, such as how they expanded from an outline or organized their loose thoughts for different writing goals. The interviews uncovered the psychological and productivity affordances of writing with speech, pointing to future directions of designing for this writing modality and the utilization of AI support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15204v2">Can Unconfident LLM Annotations Be Used for Confident Conclusions?</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Please cite as: Can Unconfident LLM Annotations Be Used for Confident Conclusions? Kristina Gligori\'c, Tijana Zrnic, Cinoo Lee, Emmanuel Cand\`es, and Dan Jurafsky. NAACL, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown high agreement with human raters across a variety of tasks, demonstrating potential to ease the challenges of human data collection. In computational social science (CSS), researchers are increasingly leveraging LLM annotations to complement slow and expensive human annotations. Still, guidelines for collecting and using LLM annotations, without compromising the validity of downstream conclusions, remain limited. We introduce Confidence-Driven Inference: a method that combines LLM annotations and LLM confidence indicators to strategically select which human annotations should be collected, with the goal of producing accurate statistical estimates and provably valid confidence intervals while reducing the number of human annotations needed. Our approach comes with safeguards against LLM annotations of poor quality, guaranteeing that the conclusions will be both valid and no less accurate than if we only relied on human annotations. We demonstrate the effectiveness of Confidence-Driven Inference over baselines in statistical estimation tasks across three CSS settings--text politeness, stance, and bias--reducing the needed number of human annotations by over 25% in each. Although we use CSS settings for demonstration, Confidence-Driven Inference can be used to estimate most standard quantities across a broad range of NLP problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05551v1">FRAMES: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced human language understanding and generation, with pretraining data quality and organization being crucial to their performance. Multi-stage pretraining is a promising approach, but existing methods often lack quantitative criteria for data partitioning and instead rely on intuitive heuristics. In this paper, we propose the novel Four-quadRAnt Multi-stage prEtraining Strategy (FRAMES), guided by the established principle of organizing the pretraining process into four stages to achieve significant loss reductions four times. This principle is grounded in two key findings: first, training on high Perplexity (PPL) data followed by low PPL data, and second, training on low PPL difference (PD) data followed by high PD data, both causing the loss to drop significantly twice and performance enhancements. By partitioning data into four quadrants and strategically organizing them, FRAMES achieves a remarkable 16.8% average improvement over random sampling across MMLU and CMMLU, effectively boosting LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00352v2">Does Alignment Tuning Really Break LLMs' Internal Confidence?</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Presented at the BlackboxNLP Workshop at EMNLP 2024 (Poster)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable progress, but their real-world application necessitates reliable calibration. This study conducts a comprehensive analysis of calibration degradation of LLMs across four dimensions: models, calibration metrics, tasks, and confidence extraction methods. Initial analysis showed that the relationship between alignment and calibration is not always a trade-off, but under stricter analysis conditions, we found the alignment process consistently harms calibration. This highlights the need for (1) a careful approach when measuring model confidences and calibration errors and (2) future research into algorithms that can help LLMs to achieve both instruction-following and calibration without sacrificing either.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09416v2">Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      In this paper, we investigate whether current state-of-the-art large language models (LLMs) are effective as AI tutors and whether they demonstrate pedagogical abilities necessary for good AI tutoring in educational dialogues. Previous efforts towards evaluation have been limited to subjective protocols and benchmarks. To bridge this gap, we propose a unified evaluation taxonomy with eight pedagogical dimensions based on key learning sciences principles, which is designed to assess the pedagogical value of LLM-powered AI tutor responses grounded in student mistakes or confusions in the mathematical domain. We release MRBench - a new evaluation benchmark containing 192 conversations and 1,596 responses from seven state-of-the-art LLM-based and human tutors, providing gold annotations for eight pedagogical dimensions. We assess reliability of the popular Prometheus2 and Llama-3.1-8B LLMs as evaluators and analyze each tutor's pedagogical abilities, highlighting which LLMs are good tutors and which ones are more suitable as question-answering systems. We believe that the presented taxonomy, benchmark, and human-annotated labels will streamline the evaluation process and help track the progress in AI tutors' development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05534v1">Fg-T2M++: LLMs-Augmented Fine-Grained Text Driven Human Motion Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      We address the challenging problem of fine-grained text-driven human motion generation. Existing works generate imprecise motions that fail to accurately capture relationships specified in text due to: (1) lack of effective text parsing for detailed semantic cues regarding body parts, (2) not fully modeling linguistic structures between words to comprehend text comprehensively. To tackle these limitations, we propose a novel fine-grained framework Fg-T2M++ that consists of: (1) an LLMs semantic parsing module to extract body part descriptions and semantics from text, (2) a hyperbolic text representation module to encode relational information between text units by embedding the syntactic dependency graph into hyperbolic space, and (3) a multi-modal fusion module to hierarchically fuse text and motion features. Extensive experiments on HumanML3D and KIT-ML datasets demonstrate that Fg-T2M++ outperforms SOTA methods, validating its ability to accurately generate motions adhering to comprehensive text semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12007v2">People will agree what I think: Investigating LLM's False Consensus Effect</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been recently adopted in interactive systems requiring communication. As the false belief in a model can harm the usability of such systems, LLMs should not have cognitive biases that humans have. Psychologists especially focus on the False Consensus Effect (FCE), a cognitive bias where individuals overestimate the extent to which others share their beliefs or behaviors, because FCE can distract smooth communication by posing false beliefs. However, previous studies have less examined FCE in LLMs thoroughly, which needs more consideration of confounding biases, general situations, and prompt changes. Therefore, in this paper, we conduct two studies to examine the FCE phenomenon in LLMs. In Study 1, we investigate whether LLMs have FCE. In Study 2, we explore how various prompting styles affect the demonstration of FCE. As a result of these studies, we identified that popular LLMs have FCE. Also, the result specifies the conditions when FCE becomes more or less prevalent compared to normal usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.05138v2">Are LLMs Correctly Integrated into Software Systems?</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 accepted by ICSE 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide effective solutions in various application scenarios, with the support of retrieval-augmented generation (RAG). However, developers face challenges in integrating LLM and RAG into software systems, due to lacking interface specifications, various requirements from software context, and complicated system management. In this paper, we have conducted a comprehensive study of 100 open-source applications that incorporate LLMs with RAG support, and identified 18 defect patterns. Our study reveals that 77% of these applications contain more than three types of integration defects that degrade software functionality, efficiency, and security. Guided by our study, we propose systematic guidelines for resolving these defects in software life cycle. We also construct an open-source defect library Hydrangea.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01387v2">TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates an Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. Our experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. These findings suggest that TeLL-Drive significantly enhances both the adaptability and safety of autonomous driving systems, while offering a more efficient and scalable approach for policy learning. Full validation results are available on our website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05467v1">Position: LLMs Can be Good Tutors in Foreign Language Education</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While recent efforts have begun integrating large language models (LLMs) into foreign language education (FLE), they often rely on traditional approaches to learning tasks without fully embracing educational methodologies, thus lacking adaptability to language learning. To address this gap, we argue that LLMs have the potential to serve as effective tutors in FLE. Specifically, LLMs can play three critical roles: (1) as data enhancers, improving the creation of learning materials or serving as student simulations; (2) as task predictors, serving as learner assessment or optimizing learning pathway; and (3) as agents, enabling personalized and inclusive education. We encourage interdisciplinary research to explore these roles, fostering innovation while addressing challenges and risks, ultimately advancing FLE through the thoughtful integration of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05453v1">LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical Knowledge Graph for Cooperative Planning</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Developing intelligent agents for long-term cooperation in dynamic open-world scenarios is a major challenge in multi-agent systems. Traditional Multi-agent Reinforcement Learning (MARL) frameworks like centralized training decentralized execution (CTDE) struggle with scalability and flexibility. They require centralized long-term planning, which is difficult without custom reward functions, and face challenges in processing multi-modal data. CTDE approaches also assume fixed cooperation strategies, making them impractical in dynamic environments where agents need to adapt and plan independently. To address decentralized multi-agent cooperation, we propose Decentralized Adaptive Knowledge Graph Memory and Structured Communication System (DAMCS) in a novel Multi-agent Crafter environment. Our generative agents, powered by Large Language Models (LLMs), are more scalable than traditional MARL agents by leveraging external knowledge and language for long-term planning and reasoning. Instead of fully sharing information from all past experiences, DAMCS introduces a multi-modal memory system organized as a hierarchical knowledge graph and a structured communication protocol to optimize agent cooperation. This allows agents to reason from past interactions and share relevant information efficiently. Experiments on novel multi-agent open-world tasks show that DAMCS outperforms both MARL and LLM baselines in task efficiency and collaboration. Compared to single-agent scenarios, the two-agent scenario achieves the same goal with 63% fewer steps, and the six-agent scenario with 74% fewer steps, highlighting the importance of adaptive memory and structured communication in achieving long-term goals. We publicly release our project at: https://happyeureka.github.io/damcs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13677v2">HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13918v2">FTSmartAudit: A Knowledge Distillation-Enhanced Framework for Automated Smart Contract Auditing Using Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 26 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The rise of blockchain technologies has greatly accelerated the development and deployment of smart contracts. However, their inherent vulnerabilities and susceptibility to bugs have led to significant financial losses, underscoring the challenges in securing smart contracts. While traditional auditing methods are crucial, they often fall short in addressing the increasing complexity and volume of smart contracts. Recent advancements in Large Language Models (LLMs) offer promising solutions for enhancing software auditing by automatically identifying security vulnerabilities. Despite their potential, the practical application of these models is hindered by substantial computational demands. This paper investigates the feasibility of using smaller, fine-tuned models to achieve comparable or even superior results in smart contract auditing. We introduce the FTSmartAudit framework, which is designed to develop cost-effective, specialized models for smart contract auditing through the fine-tuning of LLMs. Our contributions include: (1) a single-task learning framework that streamlines data preparation, training, evaluation, and continuous learning; (2) a robust dataset generation method utilizing domain-special knowledge distillation to produce high-quality datasets from advanced models like GPT-4o; (3) an adaptive learning strategy to maintain model accuracy and robustness; (4) the proven effectiveness of fine-tuned models in detecting specific vulnerabilities and complex logical errors; and (5) a framework that can be extended to other domains requiring LLM solutions. Our experimental results demonstrate that smaller models can surpass state-of-the-art commercial models and tools in detecting vulnerabilities in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13045v2">Explainable LLM-driven Multi-dimensional Distillation for E-Commerce Relevance Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 Accepted by WWW 2025 oral
    </div>
    <details class="paper-abstract">
      Effective query-item relevance modeling is pivotal for enhancing user experience and safeguarding user satisfaction in e-commerce search systems. Recently, benefiting from the vast inherent knowledge, Large Language Model (LLM) approach demonstrates strong performance and long-tail generalization ability compared with previous neural-based specialized relevance learning methods. Though promising, current LLM-based methods encounter the following inadequacies in practice: First, the massive parameters and computational demands make it difficult to be deployed online. Second, distilling LLM models to online models is a feasible direction, but the LLM relevance modeling is a black box, and its rich intrinsic knowledge is difficult to extract and apply online. To improve the interpretability of LLM and boost the performance of online relevance models via LLM, we propose an Explainable LLM-driven Multi-dimensional Distillation framework for e-commerce relevance learning, which comprises two core components: (1) An Explainable LLM for relevance modeling (ELLM-rele), which decomposes the relevance learning into intermediate steps and models relevance learning as a Chain-of-Thought (CoT) reasoning, thereby enhancing both interpretability and performance of LLM. (2) A Multi-dimensional Knowledge Distillation (MKD) architecture that transfers the knowledge of ELLM-rele to current deployable interaction-based and representation-based student models from both the relevance score distribution and CoT reasoning aspects. Through distilling the probabilistic and CoT reasoning knowledge, MKD improves both the semantic interaction and long-tail generalization abilities of student models. Extensive offline evaluations and online experiments on Taobao search ad scene demonstrate that our proposed framework significantly enhances e-commerce relevance learning performance and user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05413v1">XPUTimer: Anomaly Diagnostics for Divergent LLM Training in GPU Clusters of Thousand-Plus Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      The rapid proliferation of large language models has driven the need for efficient GPU training clusters. However, ensuring high-performance training in these clusters is challenging due to the complexity of software-hardware interactions and the frequent occurrence of training anomalies. Since existing diagnostic tools are narrowly tailored to specific issues, there are gaps in their ability to address anomalies spanning the entire training stack. In response, we introduce XPUTimer, a real-time diagnostic framework designed for distributed LLM training at scale. XPUTimer first integrates a lightweight tracing daemon to monitor key code segments with minimal overhead. Additionally, it features a diagnostic engine that employs novel intra-kernel tracing and holistic aggregated metrics to efficiently identify and resolve anomalies. Deployment of XPUTimer across 6,000 GPUs over eight months demonstrated significant improvements across the training stack, validating its effectiveness in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01705v2">Progressive Binarization with Semi-Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, but their high computational and memory demands pose challenges for deployment on resource-constrained devices. Binarization, as an efficient compression method that reduces model weights to just 1 bit, significantly lowers both computational and memory requirements. Despite this, the binarized LLM still contains redundancy, which can be further compressed. Semi-structured pruning provides a promising approach to achieve this, which offers a better trade-off between model performance and hardware efficiency. However, simply combining binarization with semi-structured pruning can lead to a significant performance drop. To address this issue, we propose a Progressive Binarization with Semi-Structured Pruning (PBS$^2$P) method for LLM compression. We first propose a Stepwise semi-structured Pruning with Binarization Optimization (SPBO). Our optimization strategy significantly reduces the total error caused by pruning and binarization, even below that of the no-pruning scenario. Furthermore, we design a Coarse-to-Fine Search (CFS) method to select pruning elements more effectively. Extensive experiments demonstrate that PBS$^2$P achieves superior accuracy across various LLM families and evaluation metrics, noticeably outperforming state-of-the-art (SOTA) binary PTQ methods. The code and models will be available at https://github.com/XIANGLONGYAN/PBS2P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05400v1">Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-02-08
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved significant success, their reliance on large volumes of human-annotated data has limited their potential for further scaling. In this situation, utilizing self-generated synthetic data has become crucial for fine-tuning LLMs without extensive human annotation. However, current methods often fail to ensure consistent improvements across iterations, with performance stagnating after only minimal updates. To overcome these challenges, we introduce Dynamic Noise Preference Optimization (DNPO). DNPO employs a dynamic sample labeling mechanism to construct preference pairs for training and introduces controlled, trainable noise into the preference optimization process. Our approach effectively prevents stagnation and enables continuous improvement. In experiments with Zephyr-7B, DNPO consistently outperforms existing methods, showing an average performance boost of 2.6% across multiple benchmarks. Additionally, DNPO shows a significant improvement in model-generated data quality, with a 29.4% win-loss rate gap compared to the baseline in GPT-4 evaluations. This highlights its effectiveness in enhancing model performance through iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06950v2">A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation</a></div>
    <div class="paper-meta">
      📅 2025-02-08
      | 💬 NAACL 2025 (main conference)
    </div>
    <details class="paper-abstract">
      This paper focuses on the task of hallucination detection, which aims to determine the truthfulness of LLM-generated statements. To address this problem, a popular class of methods utilize the LLM's self-consistencies in its beliefs in a set of logically related augmented statements generated by the LLM, which does not require external knowledge databases and can work with both white-box and black-box LLMs. However, in many existing approaches, the augmented statements tend to be very monotone and unstructured, which makes it difficult to integrate meaningful information from the LLM beliefs in these statements. Also, many methods work with the binarized version of the LLM's belief, instead of the continuous version, which significantly loses information. To overcome these limitations, in this paper, we propose Belief Tree Propagation (BTProp), a probabilistic framework for LLM hallucination detection. BTProp introduces a belief tree of logically related statements by recursively decomposing a parent statement into child statements with three decomposition strategies, and builds a hidden Markov tree model to integrate the LLM's belief scores in these statements in a principled way. Experiment results show that our method improves baselines by 3%-9% (evaluated by AUROC and AUC-PR) on multiple hallucination detection benchmarks. Code is available at https://github.com/UCSB-NLP-Chang/BTProp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04677v1">LLM Query Scheduling with Prefix Reuse and Latency Constraints</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) in online settings requires optimizing inference performance under stringent latency constraints, particularly the time-to-first-token (TTFT) and time-per-output-token (TPOT). This paper focuses on the query scheduling problem for LLM inference with prefix reuse, a technique that leverages shared prefixes across queries to reduce computational overhead. Our work reveals previously unknown limitations of the existing first-come-first-serve (FCFS) and longest-prefix-match (LPM) scheduling strategies with respect to satisfying latency constraints. We present a formal theoretical framework for LLM query scheduling under RadixAttention, a prefix reuse mechanism that stores and reuses intermediate representations in a radix tree structure. Our analysis establishes the NP-hardness of the scheduling problem with prefix reuse under TTFT constraints and proposes a novel scheduling algorithm, $k$-LPM, which generalizes existing methods by balancing prefix reuse and fairness in query processing. Theoretical guarantees demonstrate that $k$-LPM achieves improved TTFT performance under realistic traffic patterns captured by a data generative model. Empirical evaluations in a realistic serving setting validates our findings, showing significant reductions in P99 TTFT compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10198v3">ClashEval: Quantifying the tug-of-war between an LLM's internal prior and external evidence</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Revised June 9 2024
    </div>
    <details class="paper-abstract">
      Retrieval augmented generation (RAG) is frequently used to mitigate hallucinations and provide up-to-date knowledge for large language models (LLMs). However, given that document retrieval is an imprecise task and sometimes results in erroneous or even harmful content being presented in context, this raises the question of how LLMs handle retrieved information: If the provided content is incorrect, does the model know to ignore it, or does it recapitulate the error? Conversely, when the model's initial response is incorrect, does it always know to use the retrieved information to correct itself, or does it insist on its wrong prior response? To answer this, we curate a dataset of over 1200 questions across six domains (e.g., drug dosages, Olympic records, locations) along with content relevant to answering each question. We further apply precise perturbations to the answers in the content that range from subtle to blatant errors. We benchmark six top-performing LLMs, including GPT-4o, on this dataset and find that LLMs are susceptible to adopting incorrect retrieved content, overriding their own correct prior knowledge over 60% of the time. However, the more unrealistic the retrieved content is (i.e. more deviated from truth), the less likely the model is to adopt it. Also, the less confident a model is in its initial response (via measuring token probabilities), the more likely it is to adopt the information in the retrieved content. We exploit this finding and demonstrate simple methods for improving model accuracy where there is conflicting retrieved content. Our results highlight a difficult task and benchmark for LLMs -- namely, their ability to correctly discern when it is wrong in light of correct retrieved content and to reject cases when the provided content is incorrect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04644v1">Agentic Reasoning: Reasoning LLMs with Tools for the Deep Research</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      We introduce Agentic Reasoning, a framework that enhances large language model (LLM) reasoning by integrating external tool-using agents. Unlike conventional LLM-based reasoning approaches, which rely solely on internal inference, Agentic Reasoning dynamically engages web search, code execution, and structured reasoning-context memory to solve complex problems requiring deep research and multi-step logical deduction. Our framework introduces the Mind Map agent, which constructs a structured knowledge graph to track logical relationships, improving deductive reasoning. Additionally, the integration of web-search and coding agents enables real-time retrieval and computational analysis, enhancing reasoning accuracy and decision-making. Evaluations on PhD-level scientific reasoning (GPQA) and domain-specific deep research tasks demonstrate that our approach significantly outperforms existing models, including leading retrieval-augmented generation (RAG) systems and closed-source LLMs. Moreover, our results indicate that agentic reasoning improves expert-level knowledge synthesis, test-time scalability, and structured problem-solving. The code is at: https://github.com/theworldofagents/Agentic-Reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10136v2">Can LLMs Convert Graphs to Text-Attributed Graphs?</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Accepted by NAACL 25 Main Conference
    </div>
    <details class="paper-abstract">
      Graphs are ubiquitous structures found in numerous real-world applications, such as drug discovery, recommender systems, and social network analysis. To model graph-structured data, graph neural networks (GNNs) have become a popular tool. However, existing GNN architectures encounter challenges in cross-graph learning where multiple graphs have different feature spaces. To address this, recent approaches introduce text-attributed graphs (TAGs), where each node is associated with a textual description, which can be projected into a unified feature space using textual encoders. While promising, this method relies heavily on the availability of text-attributed graph data, which is difficult to obtain in practice. To bridge this gap, we propose a novel method named Topology-Aware Node description Synthesis (TANS), leveraging large language models (LLMs) to convert existing graphs into text-attributed graphs. The key idea is to integrate topological information into LLMs to explain how graph topology influences node semantics. We evaluate our TANS on text-rich, text-limited, and text-free graphs, demonstrating its applicability. Notably, on text-free graphs, our method significantly outperforms existing approaches that manually design node features, showcasing the potential of LLMs for preprocessing graph-structured data in the absence of textual information. The code and data are available at https://github.com/Zehong-Wang/TANS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05376v1">BCQ: Block Clustered Quantization for 4-bit (W4A4) LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) is a promising approach to reducing the storage and computational requirements of large language models (LLMs) without additional training cost. Recent PTQ studies have primarily focused on quantizing only weights to sub-8-bits while maintaining activations at 8-bits or higher. Accurate sub-8-bit quantization for both weights and activations without relying on quantization-aware training remains a significant challenge. We propose a novel quantization method called block clustered quantization (BCQ) wherein each operand tensor is decomposed into blocks (a block is a group of contiguous scalars), blocks are clustered based on their statistics, and a dedicated optimal quantization codebook is designed for each cluster. As a specific embodiment of this approach, we propose a PTQ algorithm called Locally-Optimal BCQ (LO-BCQ) that iterates between the steps of block clustering and codebook design to greedily minimize the quantization mean squared error. When weight and activation scalars are encoded to W4A4 format (with 0.5-bits of overhead for storing scaling factors and codebook selectors), we advance the current state-of-the-art by demonstrating <1% loss in inference accuracy across several LLMs and downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05374v1">Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23605v2">Dynamic Uncertainty Ranking: Enhancing Retrieval-Augmented In-Context Learning for Long-Tail Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can learn vast amounts of knowledge from diverse domains during pre-training. However, long-tail knowledge from specialized domains is often scarce and underrepresented, rarely appearing in the models' memorization. Prior work has shown that in-context learning (ICL) with retriever augmentation can help LLMs better capture long-tail knowledge, reducing their reliance on pre-trained data. Despite these advances, we observe that LLM predictions for long-tail questions remain uncertain to variations in retrieved samples. To take advantage of the uncertainty in ICL for guiding LLM predictions toward correct answers on long-tail samples, we propose a reinforcement learning-based dynamic uncertainty ranking method for ICL that accounts for the varying impact of each retrieved sample on LLM predictions. Our approach prioritizes more informative and stable samples while demoting misleading ones, updating rankings based on the feedback from the LLM w.r.t. each retrieved sample. To enhance training efficiency and reduce query costs, we introduce a learnable dynamic ranking threshold, adjusted when the model encounters negative prediction shifts. Experimental results on various question-answering datasets from different domains show that our method outperforms the best baseline by $2.76\%$, with a notable $5.96\%$ boost in accuracy on long-tail questions that elude zero-shot inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05331v1">Fine-Tuned LLMs are "Time Capsules" for Tracking Societal Bias Through Books</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 9 pages (excluding references), accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Books, while often rich in cultural insights, can also mirror societal biases of their eras - biases that Large Language Models (LLMs) may learn and perpetuate during training. We introduce a novel method to trace and quantify these biases using fine-tuned LLMs. We develop BookPAGE, a corpus comprising 593 fictional books across seven decades (1950-2019), to track bias evolution. By fine-tuning LLMs on books from each decade and using targeted prompts, we examine shifts in biases related to gender, sexual orientation, race, and religion. Our findings indicate that LLMs trained on decade-specific books manifest biases reflective of their times, with both gradual trends and notable shifts. For example, model responses showed a progressive increase in the portrayal of women in leadership roles (from 8% to 22%) from the 1950s to 2010s, with a significant uptick in the 1990s (from 4% to 12%), possibly aligning with third-wave feminism. Same-sex relationship references increased markedly from the 1980s to 2000s (from 0% to 10%), mirroring growing LGBTQ+ visibility. Concerningly, negative portrayals of Islam rose sharply in the 2000s (26% to 38%), likely reflecting post-9/11 sentiments. Importantly, we demonstrate that these biases stem mainly from the books' content and not the models' architecture or initial training. Our study offers a new perspective on societal bias trends by bridging AI, literary studies, and social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05310v1">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Large Language Models have proved surprisingly effective at solving a wide range of tasks from just a handful of examples. However, their lack of reliability and modularity limits their capacity to tackle large problems that require many steps of reasoning. In response, researchers have proposed advanced pipelines that leverage domain-specific knowledge to chain smaller prompts, provide intermediate feedback and improve performance through search. However, the current complexity of writing, tuning, maintaining and improving such pipelines has limited their sophistication. We propose oracular programming, a foundational paradigm for building LLM-enabled applications that lets domain experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful search tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05291v1">Can LLMs Rank the Harmfulness of Smaller LLMs? We are Not There Yet</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become ubiquitous, thus it is important to understand their risks and limitations. Smaller LLMs can be deployed where compute resources are constrained, such as edge devices, but with different propensity to generate harmful output. Mitigation of LLM harm typically depends on annotating the harmfulness of LLM output, which is expensive to collect from humans. This work studies two questions: How do smaller LLMs rank regarding generation of harmful content? How well can larger LLMs annotate harmfulness? We prompt three small LLMs to elicit harmful content of various types, such as discriminatory language, offensive content, privacy invasion, or negative influence, and collect human rankings of their outputs. Then, we evaluate three state-of-the-art large LLMs on their ability to annotate the harmfulness of these responses. We find that the smaller models differ with respect to harmfulness. We also find that large LLMs show low to moderate agreement with humans. These findings underline the need for further work on harm mitigation in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05253v1">LLMs Can Teach Themselves to Better Predict the Future</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      We present an outcome-driven fine-tuning framework that enhances the forecasting capabilities of large language models (LLMs) without relying on human-curated reasoning samples. Our method leverages model self-play to generate pairs of diverse reasoning trajectories and probabilistic forecasts for a set of diverse questions that resolve after the models' knowledge cutoff date. We then rank pairs of these reasoning traces by their distance to the actual outcomes before fine-tuning the model via Direct Preference Optimization (DPO). On a separate test set, our approach increases prediction accuracy of Phi-4 14B and DeepSeek-R1 14B by between 7--10\% over a base model and a DPO fine-tuned control model with randomized labels, bringing them on par with forecasting capabilities of much larger frontier models like GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05252v1">GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Long-context large language models (LLMs) have recently shown strong performance in information retrieval and long-document QA. However, to tackle the most challenging intellectual problems, LLMs must reason effectively in long and complex contexts (e.g., frontier mathematical research). Studying how LLMs handle increasing reasoning complexity and context length is essential, yet existing benchmarks lack a solid basis for quantitative evaluation. Inspired by the abstraction of GSM-8K problems as computational graphs, and the ability to introduce noise by adding unnecessary nodes and edges, we develop a grade school math problem generator capable of producing arithmetic problems with infinite difficulty and context length under fine-grained control. Using our newly synthesized GSM-Infinite benchmark, we comprehensively evaluate existing LLMs. We find a consistent sigmoid decline in reasoning performance as complexity increases, along with a systematic inference scaling trend: exponentially increasing inference computation yields only linear performance gains. These findings underscore the fundamental limitations of current long-context LLMs and the key challenges in scaling reasoning capabilities. Our GSM-Infinite benchmark provides a scalable and controllable testbed for systematically studying and advancing LLM reasoning in long and complex contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05163v1">DuoGuard: A Two-Player RL-Driven Framework for Multilingual LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 24 pages, 9 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has increased the need for guardrail models to ensure responsible use, particularly in detecting unsafe and illegal content. While substantial safety data exist in English, multilingual guardrail modeling remains underexplored due to the scarcity of open-source safety data in other languages. To address this gap, we propose a novel two-player Reinforcement Learning (RL) framework, where a generator and a guardrail model co-evolve adversarially to produce high-quality synthetic data for multilingual guardrail training. We theoretically formalize this interaction as a two-player game, proving convergence to a Nash equilibrium. Empirical evaluations show that our model \ours outperforms state-of-the-art models, achieving nearly 10% improvement over LlamaGuard3 (8B) on English benchmarks while being 4.5x faster at inference with a significantly smaller model (0.5B). We achieve substantial advancements in multilingual safety tasks, particularly in addressing the imbalance for lower-resource languages in a collected real dataset. Ablation studies emphasize the critical role of synthetic data generation in bridging the imbalance in open-source data between English and other languages. These findings establish a scalable and efficient approach to synthetic data generation, paving the way for improved multilingual guardrail models to enhance LLM safety. Code, model, and data will be open-sourced at https://github.com/yihedeng9/DuoGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05159v1">A Lightweight Method to Disrupt Memorized Sequences in LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 20 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive capabilities across many tasks yet risk reproducing copyrighted content verbatim, raising legal and ethical concerns. Although methods like differential privacy or neuron editing can reduce memorization, they typically require costly retraining or direct access to model weights and may degrade performance. To address these challenges, we propose TokenSwap, a lightweight, post-hoc approach that replaces the probabilities of grammar-related tokens with those from a small auxiliary model (e.g., DistilGPT-2). We run extensive experiments on commercial grade models such as Pythia-6.9b and LLaMA-3-8b and demonstrate that our method effectively reduces well-known cases of memorized generation by upto 10x with little to no impact on downstream tasks. Our approach offers a uniquely accessible and effective solution to users of real-world systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07163v3">Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      This work studies the problem of large language model (LLM) unlearning, aiming to remove unwanted data influences (e.g., copyrighted or harmful content) while preserving model utility. Despite the increasing demand for unlearning, a technically-grounded optimization framework is lacking. Gradient ascent (GA)-type methods, though widely used, are suboptimal as they reverse the learning process without controlling optimization divergence (i.e., deviation from the pre-trained state), leading to risks of over-forgetting and potential model collapse. Negative preference optimization (NPO) has been proposed to address this issue and is considered one of the state-of-the-art LLM unlearning approaches. In this work, we revisit NPO and identify another critical issue: reference model bias. This bias arises from using the reference model (i.e., the model prior to unlearning) to evaluate the unlearning success, which can compromise NPO's effectiveness. Specifically, it leads to (a) uneven allocation of optimization power across forget data with varying difficulty levels and (b) ineffective gradient weight smoothing during the early stages of unlearning optimization. To overcome these challenges, we propose a simple yet effective unlearning optimization framework, called SimNPO, showing that `simplicity' in removing the reliance on a reference model (through the lens of simple preference optimization) benefits unlearning. We provide deeper insights into SimNPO's advantages through an analysis based on mixtures of Markov chains. Extensive experiments further validate SimNPO's efficacy on benchmarks like TOFU and MUSE, as well as its robustness against relearning attacks. Codes are available at https://github.com/OPTML-Group/Unlearn-Simple.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05148v1">An Annotated Reading of 'The Singer of Tales' in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      The Parry-Lord oral-formulaic theory was a breakthrough in understanding how oral narrative poetry is learned, composed, and transmitted by illiterate bards. In this paper, we provide an annotated reading of the mechanism underlying this theory from the lens of large language models (LLMs) and generative artificial intelligence (AI). We point out the the similarities and differences between oral composition and LLM generation, and comment on the implications to society and AI policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12924v2">Interpreting token compositionality in LLMs: A robustness analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 19 pages, 2 Figures, 12 tables
    </div>
    <details class="paper-abstract">
      Understanding the internal mechanisms of large language models (LLMs) is integral to enhancing their reliability, interpretability, and inference processes. We present Constituent-Aware Pooling (CAP), a methodology designed to analyse how LLMs process compositional linguistic structures. Grounded in principles of compositionality, mechanistic interpretability, and information theory, CAP systematically intervenes in model activations through constituent-based pooling at various model levels. Our experiments on inverse definition modelling, hypernym and synonym prediction reveal critical insights into transformers' limitations in handling compositional abstractions. No specific layer integrates tokens into unified semantic representations based on their constituent parts. We observe fragmented information processing, which intensifies with model size, suggesting that larger models struggle more with these interventions and exhibit greater information dispersion. This fragmentation likely stems from transformers' training objectives and architectural design, preventing systematic and cohesive representations. Our findings highlight fundamental limitations in current transformer architectures regarding compositional semantics processing and model interpretability, underscoring the critical need for novel approaches in LLM design to address these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13757v2">GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05003v1">QuEST: Stable Training of LLMs with 1-Bit Weights and Activations</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      One approach to reducing the massive costs of large language models (LLMs) is the use of quantized or sparse representations for training or deployment. While post-training compression methods are very popular, the question of obtaining even more accurate compressed models by directly training over such representations, i.e., Quantization-Aware Training (QAT), is still open: for example, a recent study (arXiv:2411.04330v2) put the "optimal" bit-width at which models can be trained using QAT, while staying accuracy-competitive with standard FP16/BF16 precision, at 8-bits weights and activations. We advance this state-of-the-art via a new method called QuEST, which is Pareto-competitive with FP16, i.e., it provides better accuracy at lower model size, while training models with weights and activations in 4-bits or less. Moreover, QuEST allows stable training with 1-bit weights and activations. QuEST achieves this by improving two key aspects of QAT methods: (1) accurate and fast quantization of the (continuous) distributions of weights and activations via Hadamard normalization and MSE-optimal fitting; (2) a new trust gradient estimator based on the idea of explicitly minimizing the error between the noisy gradient computed over quantized states and the "true" (but unknown) full-precision gradient. Experiments on Llama-type architectures show that QuEST induces stable scaling laws across the entire range of hardware-supported precisions, and can be extended to sparse representations. We provide GPU kernel support showing that models produced by QuEST can be executed efficiently. Our code is available at https://github.com/IST-DASLab/QuEST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04983v1">MoGraphGPT: Creating Interactive Scenes Using Modular LLM and Graphical Control</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Creating interactive scenes often involves complex programming tasks. Although large language models (LLMs) like ChatGPT can generate code from natural language, their output is often error-prone, particularly when scripting interactions among multiple elements. The linear conversational structure limits the editing of individual elements, and lacking graphical and precise control complicates visual integration. To address these issues, we integrate an element-level modularization technique that processes textual descriptions for individual elements through separate LLM modules, with a central module managing interactions among elements. This modular approach allows for refining each element independently. We design a graphical user interface, MoGraphGPT , which combines modular LLMs with enhanced graphical control to generate codes for 2D interactive scenes. It enables direct integration of graphical information and offers quick, precise control through automatically generated sliders. Our comparative evaluation against an AI coding tool, Cursor Composer, as the baseline system and a usability study show MoGraphGPT significantly improves easiness, controllability, and refinement in creating complex 2D interactive scenes with multiple visual elements in a coding-free manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v1">CoCoA: A Generalized Approach to Uncertainty Quantification by Integrating Confidence and Consistency of LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompasses a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches and shown impressive performance in various applications. However, they sometimes fail to outperform much simpler baseline methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency that leads to a family of efficient and robust UQ methods. We evaluate our approach across a variety of tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18712v4">Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Fingerprinting refers to the process of identifying underlying Machine Learning (ML) models of AI Systemts, such as Large Language Models (LLMs), by analyzing their unique characteristics or patterns, much like a human fingerprint. The fingerprinting of Large Language Models (LLMs) has become essential for ensuring the security and transparency of AI-integrated applications. While existing methods primarily rely on access to direct interactions with the application to infer model identity, they often fail in real-world scenarios involving multi-agent systems, frequent model updates, and restricted access to model internals. In this paper, we introduce a novel fingerprinting framework designed to address these challenges by integrating static and dynamic fingerprinting techniques. Our approach identifies architectural features and behavioral traits, enabling accurate and robust fingerprinting of LLMs in dynamic environments. We also highlight new threat scenarios where traditional fingerprinting methods are ineffective, bridging the gap between theoretical techniques and practical application. To validate our framework, we present an extensive evaluation setup that simulates real-world conditions and demonstrate the effectiveness of our methods in identifying and monitoring LLMs in Gen-AI applications. Our results highlight the framework's adaptability to diverse and evolving deployment contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04931v1">Breaking the News: A LLM-based Game where Players Act as Influencer or Debunker for Raising Awareness About Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Game-based interventions are widely used to combat misinformation online by employing the "inoculation approach". However, most current interventions are designed as single-player games, presenting players with limited predefined choices. Such restrictions reduce replayability and may lead to an overly simplistic understanding of the processes of misinformation phenomenon and the debunking. This study seeks to address these issues, and empower people to better understand the opinion influencing and misinformation debunking processes. We did this by creating a Player versus Player (PvP) game where participants attempt to either generate or debunk misinformation to convince LLM-represented public opinion. Using a within-subjects mixed-methods study design (N=47), we found that this game significantly raised participants' media literacy and improved their ability to identify misinformation. Our qualitative exploration revealed how participants' use of debunking and content creation strategies deepened their understanding of the nature of disinformation. We demonstrate how LLMs can be integrated into PvP games to foster greater understanding of contrasting viewpoints and highlight social challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10198v2">From Allies to Adversaries: Manipulating LLM Tool-Calling through Adversarial Injection</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Tool-calling has changed Large Language Model (LLM) applications by integrating external tools, significantly enhancing their functionality across diverse tasks. However, this integration also introduces new security vulnerabilities, particularly in the tool scheduling mechanisms of LLM, which have not been extensively studied. To fill this gap, we present ToolCommander, a novel framework designed to exploit vulnerabilities in LLM tool-calling systems through adversarial tool injection. Our framework employs a well-designed two-stage attack strategy. Firstly, it injects malicious tools to collect user queries, then dynamically updates the injected tools based on the stolen information to enhance subsequent attacks. These stages enable ToolCommander to execute privacy theft, launch denial-of-service attacks, and even manipulate business competition by triggering unscheduled tool-calling. Notably, the ASR reaches 91.67% for privacy theft and hits 100% for denial-of-service and unscheduled tool calling in certain cases. Our work demonstrates that these vulnerabilities can lead to severe consequences beyond simple misuse of tool-calling systems, underscoring the urgent need for robust defensive strategies to secure LLM Tool-calling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11633v2">Self-seeding and Multi-intent Self-instructing LLMs for Generating Intent-aware Information-Seeking dialogs</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 Accepted at NAACL 2025
    </div>
    <details class="paper-abstract">
      Identifying user intents in information-seeking dialogs is crucial for a system to meet user's information needs. Intent prediction (IP) is challenging and demands sufficient dialogs with human-labeled intents for training. However, manually annotating intents is resource-intensive. While large language models (LLMs) have been shown to be effective in generating synthetic data, there is no study on using LLMs to generate intent-aware information-seeking dialogs. In this paper, we focus on leveraging LLMs for zero-shot generation of large-scale, open-domain, and intent-aware information-seeking dialogs. We propose SOLID, which has novel self-seeding and multi-intent self-instructing schemes. The former improves the generation quality by using the LLM's own knowledge scope to initiate dialog generation; the latter prompts the LLM to generate utterances sequentially, and mitigates the need for manual prompt design by asking the LLM to autonomously adapt its prompt instruction when generating complex multi-intent utterances. Furthermore, we propose SOLID-RL, which is further trained to generate a dialog in one step on the data generated by SOLID. We propose a length-based quality estimation mechanism to assign varying weights to SOLID-generated dialogs based on their quality during the training process of SOLID-RL. We use SOLID and SOLID-RL to generate more than 300k intent-aware dialogs, surpassing the size of existing datasets. Experiments show that IP methods trained on dialogs generated by SOLID and SOLID-RL achieve better IP quality than ones trained on human-generated dialogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00026v2">Pushing the Limits of BFP on Narrow Precision LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      The substantial computational and memory demands of Large Language Models (LLMs) hinder their deployment. Block Floating Point (BFP) has proven effective in accelerating linear operations, a cornerstone of LLM workloads. However, as sequence lengths grow, nonlinear operations, such as Attention, increasingly become performance bottlenecks due to their quadratic computational complexity. These nonlinear operations are predominantly executed using inefficient floating-point formats, which renders the system challenging to optimize software efficiency and hardware overhead. In this paper, we delve into the limitations and potential of applying BFP to nonlinear operations. Given our findings, we introduce a hardware-software co-design framework (DB-Attn), including: (i) DBFP, an advanced BFP version, overcomes nonlinear operation challenges with a pivot-focus strategy for diverse data and an adaptive grouping strategy for flexible exponent sharing. (ii) DH-LUT, a novel lookup table algorithm dedicated to accelerating nonlinear operations with DBFP format. (iii) An RTL-level DBFP-based engine is implemented to support DB-Attn, applicable to FPGA and ASIC. Results show that DB-Attn provides significant performance improvements with negligible accuracy loss, achieving 74% GPU speedup on Softmax of LLaMA and 10x low overhead performance improvement over SOTA designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09639v2">How to Make the Most of LLMs' Grammatical Knowledge for Acceptability Judgments</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 NAACL 2025 main
    </div>
    <details class="paper-abstract">
      The grammatical knowledge of language models (LMs) is often measured using a benchmark of linguistic minimal pairs, where the LMs are presented with a pair of acceptable and unacceptable sentences and required to judge which is more acceptable. Conventional approaches directly compare sentence probabilities assigned by LMs, but recent large language models (LLMs) are trained to perform tasks via prompting, and thus, the raw probabilities they assign may not fully reflect their grammatical knowledge. In this study, we attempt to derive more accurate acceptability judgments from LLMs using prompts and templates. Through extensive experiments in English and Chinese, we compare nine judgment methods and find two of them, a probability readout method -- in-template LP and a prompt-based method -- Yes/No probability computing, achieve higher accuracy than the conventional ones. Our analysis reveals that these methods excel in different linguistic phenomena, suggesting they access different aspects of LLMs' knowledge. We also find that ensembling the two methods outperforms single methods. Consequently, we recommend these techniques, either individually or ensembled, as more effective alternatives to conventional approaches for assessing grammatical knowledge in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v2">InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-02-07
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 takes a middle-ground approach by leveraging Optical Circuit Switches, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfinitePOD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfinitePOD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt into variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh) based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfinitePOD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06858v1">LLM-Supported Natural Language to Bash Translation</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 13 pages, NAACL 2025
    </div>
    <details class="paper-abstract">
      The Bourne-Again Shell (Bash) command-line interface for Linux systems has complex syntax and requires extensive specialized knowledge. Using the natural language to Bash command (NL2SH) translation capabilities of large language models (LLMs) for command composition circumvents these issues. However, the NL2SH performance of LLMs is difficult to assess due to inaccurate test data and unreliable heuristics for determining the functional equivalence of Bash commands. We present a manually verified test dataset of 600 instruction-command pairs and a training dataset of 40,939 pairs, increasing the size of previous datasets by 441% and 135%, respectively. Further, we present a novel functional equivalence heuristic that combines command execution with LLM evaluation of command outputs. Our heuristic can determine the functional equivalence of two Bash commands with 95% confidence, a 16% increase over previous heuristics. Evaluation of popular LLMs using our test dataset and heuristic demonstrates that parsing, in-context learning, in-weight learning, and constrained decoding can improve NL2SH accuracy by up to 32%. Our findings emphasize the importance of dataset quality, execution-based evaluation and translation method for advancing NL2SH translation. Our code is available at https://github.com/westenfelder/NL2SH
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06846v1">Prot2Chat: Protein LLM with Early Fusion of Sequence and Structure</a></div>
    <div class="paper-meta">
      📅 2025-02-07
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Proteins play a pivotal role in living organisms, yet understanding their functions presents significant challenges, including the limited flexibility of classification-based methods, the inability to effectively leverage spatial structural information, and the lack of systematic evaluation metrics for protein Q&A systems. To address these limitations, we propose Prot2Chat, a novel framework that integrates multimodal protein representations with natural language through a unified module, enabling large language model (LLM)-driven answer generation. Our model incorporates a modified ProteinMPNN encoder, which encodes protein sequence and structural information in a unified manner, a protein-text adapter with cross-attention mechanisms, and a LLaMA3 decoder. To optimize training efficiency, we freeze the encoder and employ LoRA techniques for the decoder. We conducted experiments on two datasets, both automated metrics and expert evaluations demonstrate the superior performance of our model. Furthermore, zero-shot prediction results highlight its strong generalization capabilities. This framework offers a promising solution for bridging protein domain knowledge with natural language understanding, paving the way for transformative advancements in protein-related research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04322v1">Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04295v1">Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant capability across various tasks, with their real-world effectiveness often driven by prompt design. While recent research has focused on optimizing prompt content, the role of prompt formatting, a critical but often overlooked dimension, has received limited systematic investigation. In this paper, we introduce Content-Format Integrated Prompt Optimization (CFPO), an innovative methodology that jointly optimizes both prompt content and formatting through an iterative refinement process. CFPO leverages natural language mutations to explore content variations and employs a dynamic format exploration strategy that systematically evaluates diverse format options. Our extensive evaluations across multiple tasks and open-source LLMs demonstrate that CFPO demonstrates measurable performance improvements compared to content-only optimization methods. This highlights the importance of integrated content-format optimization and offers a practical, model-agnostic approach to enhancing LLM performance. Code will be available at https://github.com/HenryLau7/CFPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04227v1">Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      We explore the feasibility and effectiveness of using LLM-driven autonomous systems for Assumed Breach penetration testing in enterprise networks. We introduce a novel prototype that, driven by Large Language Models (LLMs), can compromise accounts within a real-life Active Directory testbed. Our research provides a comprehensive evaluation of the prototype's capabilities, and highlights both strengths and limitations while executing attack. The evaluation uses a realistic simulation environment (Game of Active Directory, GOAD) to capture intricate interactions, stochastic outcomes, and timing dependencies that characterize live network scenarios. The study concludes that autonomous LLMs are able to conduct Assumed Breach simulations, potentially democratizing access to penetration testing for organizations facing budgetary constraints. The prototype's source code, traces, and analyzed logs are released as open-source to enhance collective cybersecurity and facilitate future research in LLM-driven cybersecurity automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04204v1">"Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the number of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix during jailbreaking to the length during AT. Our findings show that it is practical to defend "long-length" jailbreak attacks via efficient "short-length" AT. The code is available at https://github.com/fshp971/adv-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06833v3">Does Mapo Tofu Contain Coffee? Probing LLMs for Food-related Cultural Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 cultural bias analysis, cultural knowledge probing, large language models, cultural NLP; Accepted by NAACL2025
    </div>
    <details class="paper-abstract">
      Recent studies have highlighted the presence of cultural biases in Large Language Models (LLMs), yet often lack a robust methodology to dissect these phenomena comprehensively. Our work aims to bridge this gap by delving into the Food domain, a universally relevant yet culturally diverse aspect of human life. We introduce FmLAMA, a multilingual dataset centered on food-related cultural facts and variations in food practices. We analyze LLMs across various architectures and configurations, evaluating their performance in both monolingual and multilingual settings. By leveraging templates in six different languages, we investigate how LLMs interact with language-specific and cultural knowledge. Our findings reveal that (1) LLMs demonstrate a pronounced bias towards food knowledge prevalent in the United States; (2) Incorporating relevant cultural context significantly improves LLMs' ability to access cultural knowledge; (3) The efficacy of LLMs in capturing cultural nuances is highly dependent on the interplay between the probing language, the specific model architecture, and the cultural context in question. This research underscores the complexity of integrating cultural understanding into LLMs and emphasizes the importance of culturally diverse datasets to mitigate biases and enhance model performance across different cultural domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04134v1">The Order Effect: Investigating Prompt Sensitivity in Closed-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 The first 3 authors have contributed equally
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integral to diverse applications, ensuring their reliability under varying input conditions is crucial. One key issue affecting this reliability is order sensitivity, wherein slight variations in input arrangement can lead to inconsistent or biased outputs. Although recent advances have reduced this sensitivity, the problem remains unresolved. This paper investigates the extent of order sensitivity in closed-source LLMs by conducting experiments across multiple tasks, including paraphrasing, relevance judgment, and multiple-choice questions. Our results show that input order significantly affects performance across tasks, with shuffled inputs leading to measurable declines in output accuracy. Few-shot prompting demonstrates mixed effectiveness and offers partial mitigation, however, fails to fully resolve the problem. These findings highlight persistent risks, particularly in high-stakes applications, and point to the need for more robust LLMs or improved input-handling techniques in future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04095v1">LLMs to Support a Domain Specific Knowledge Assistant</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      This work presents a custom approach to developing a domain specific knowledge assistant for sustainability reporting using the International Financial Reporting Standards (IFRS). In this domain, there is no publicly available question-answer dataset, which has impeded the development of a high-quality chatbot to support companies with IFRS reporting. The two key contributions of this project therefore are: (1) A high-quality synthetic question-answer (QA) dataset based on IFRS sustainability standards, created using a novel generation and evaluation pipeline leveraging Large Language Models (LLMs). This comprises 1,063 diverse QA pairs that address a wide spectrum of potential user queries in sustainability reporting. Various LLM-based techniques are employed to create the dataset, including chain-of-thought reasoning and few-shot prompting. A custom evaluation framework is developed to assess question and answer quality across multiple dimensions, including faithfulness, relevance, and domain specificity. The dataset averages a score range of 8.16 out of 10 on these metrics. (2) Two architectures for question-answering in the sustainability reporting domain - a RAG pipeline and a fully LLM-based pipeline. The architectures are developed by experimenting, fine-tuning, and training on the QA dataset. The final pipelines feature an LLM fine-tuned on domain specific data and an industry classification component to improve the handling of complex queries. The RAG architecture achieves an accuracy of 85.32% on single-industry and 72.15% on cross-industry multiple-choice questions, outperforming the baseline approach by 4.67 and 19.21 percentage points, respectively. The LLM-based pipeline achieves an accuracy of 93.45% on single-industry and 80.30% on cross-industry multiple-choice questions, an improvement of 12.80 and 27.36 percentage points over the baseline, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04077v1">AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      With the development of large language models (LLMs), efficient inference through Key-Value (KV) cache compression has attracted considerable attention, especially for long-context generation. To compress the KV cache, recent methods identify critical KV tokens through heuristic ranking with attention scores. However, these methods often struggle to accurately determine critical tokens as they neglect the \textit{temporal patterns} in attention scores, resulting in a noticeable degradation in LLM performance. To address this challenge, we propose AttentionPredictor, which is the first learning-based critical token identification approach. Specifically, AttentionPredictor learns a lightweight convolution model to capture spatiotemporal patterns and predict the next-token attention score. An appealing feature of AttentionPredictor is that it accurately predicts the attention score while consuming negligible memory. Moreover, we propose a cross-token critical cache prefetching framework that hides the token estimation time overhead to accelerate the decoding stage. By retaining most of the attention information, AttentionPredictor achieves 16$\times$ KV cache compression with comparable LLM performance, significantly outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06809v2">Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated immense utility across various industries. However, as LLMs advance, the risk of harmful outputs increases due to incorrect or malicious instruction prompts. While current methods effectively address jailbreak risks, they share common limitations: 1) Judging harmful responses from the prefill-level lacks utilization of the model's decoding outputs, leading to relatively lower effectiveness and robustness. 2) Rejecting potentially harmful responses based on a single evaluation can significantly impair the model's helpfulness.This paper examines the LLMs' capability to recognize harmful outputs, revealing and quantifying their proficiency in assessing the danger of previous tokens. Motivated by pilot experiment results, we design a robust defense mechanism at the decoding level. Our novel decoder-oriented, step-by-step defense architecture corrects harmful queries directly rather than rejecting them outright. We introduce speculative decoding to enhance usability and facilitate deployment to boost secure decoding speed. Extensive experiments demonstrate that our approach improves model security without compromising reasoning speed. Notably, our method leverages the model's ability to discern hazardous information, maintaining its helpfulness compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04022v1">Quantification of Biodiversity from Historical Survey Text with LLM-based Best-Worst Scaling</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 NoDaLiDa 2025, EcoNLP Workshop
    </div>
    <details class="paper-abstract">
      In this study, we evaluate methods to determine the frequency of species via quantity estimation from historical survey text. To that end, we formulate classification tasks and finally show that this problem can be adequately framed as a regression task using Best-Worst Scaling (BWS) with Large Language Models (LLMs). We test Ministral-8B, DeepSeek-V3, and GPT-4, finding that the latter two have reasonable agreement with humans and each other. We conclude that this approach is more cost-effective and similarly robust compared to a fine-grained multi-class approach, allowing automated quantity estimation across species.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04008v1">Automating a Complete Software Test Process Using LLMs: An Automotive Case Study</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by International Conference on Software Engineering (ICSE) 2025
    </div>
    <details class="paper-abstract">
      Vehicle API testing verifies whether the interactions between a vehicle's internal systems and external applications meet expectations, ensuring that users can access and control various vehicle functions and data. However, this task is inherently complex, requiring the alignment and coordination of API systems, communication protocols, and even vehicle simulation systems to develop valid test cases. In practical industrial scenarios, inconsistencies, ambiguities, and interdependencies across various documents and system specifications pose significant challenges. This paper presents a system designed for the automated testing of in-vehicle APIs. By clearly defining and segmenting the testing process, we enable Large Language Models (LLMs) to focus on specific tasks, ensuring a stable and controlled testing workflow. Experiments conducted on over 100 APIs demonstrate that our system effectively automates vehicle API testing. The results also confirm that LLMs can efficiently handle mundane tasks requiring human judgment, making them suitable for complete automation in similar industrial contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09243v3">SPRec: Self-Play to Debias LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by WWW 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have attracted significant attention in recommendation systems. Current work primarily applies supervised fine-tuning (SFT) to adapt the model for recommendation tasks. However, SFT on positive examples only limits the model's ability to align with user preference. To address this, researchers recently introduced Direct Preference Optimization (DPO), which explicitly aligns LLMs with user preferences using offline preference ranking data. However, we found that DPO inherently biases the model towards a few items, exacerbating the filter bubble issue and ultimately degrading user experience. In this paper, we propose SPRec, a novel self-play framework designed to mitigate over-recommendation and improve fairness without requiring additional data or manual intervention. In each self-play iteration, the model undergoes an SFT step followed by a DPO step, treating offline interaction data as positive samples and the predicted outputs from the previous iteration as negative samples. This effectively re-weights the DPO loss function using the model's logits, adaptively suppressing biased items. Extensive experiments on multiple real-world datasets demonstrate SPRec's effectiveness in enhancing recommendation accuracy and fairness. The implementation is available via https://github.com/RegionCh/SPRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01866v2">House of Cards: Massive Weights in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Massive activations, which manifest in specific feature dimensions of hidden states, introduce a significant bias in large language models (LLMs), leading to an overemphasis on the corresponding token. In this paper, we identify that massive activations originate not from the hidden state but from the intermediate state of a feed-forward network module in an early layer. Expanding on the previous observation that massive activations occur only in specific feature dimensions, we dive deep into the weights that cause massive activations. Specifically, we define top-$k$ massive weights as the weights that contribute to the dimensions with the top-$k$ magnitudes in the intermediate state. When these massive weights are set to zero, the functionality of LLMs is entirely disrupted. However, when all weights except for massive weights are set to zero, it results in a relatively minor performance drop, even though a much larger number of weights are set to zero. This implies that during the pre-training process, learning is dominantly focused on massive weights. Building on this observation, we propose a simple plug-and-play method called MacDrop (massive weights curriculum dropout), to rely less on massive weights during parameter-efficient fine-tuning. This method applies dropout to the pre-trained massive weights, starting with a high dropout probability and gradually decreasing it as fine-tuning progresses. Through various experiments, including zero-shot downstream tasks, long-context tasks, and ablation studies, we demonstrate that \texttt{MacDrop} generally improves performance and strengthens robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01269v4">CPRM: A LLM-based Continual Pre-training Framework for Relevance Modeling in Commercial Search</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Relevance modeling between queries and items stands as a pivotal component in commercial search engines, directly affecting the user experience. Given the remarkable achievements of large language models (LLMs) in various natural language processing (NLP) tasks, LLM-based relevance modeling is gradually being adopted within industrial search systems. Nevertheless, foundational LLMs lack domain-specific knowledge and do not fully exploit the potential of in-context learning. Furthermore, structured item text remains underutilized, and there is a shortage in the supply of corresponding queries and background knowledge. We thereby propose CPRM (Continual Pre-training for Relevance Modeling), a framework designed for the continual pre-training of LLMs to address these issues. Our CPRM framework includes three modules: 1) employing both queries and multi-field item to jointly pre-train for enhancing domain knowledge, 2) applying in-context pre-training, a novel approach where LLMs are pre-trained on a sequence of related queries or items, and 3) conducting reading comprehension on items to produce associated domain knowledge and background information (e.g., generating summaries and corresponding queries) to further strengthen LLMs. Results on offline experiments and online A/B testing demonstrate that our model achieves convincing performance compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03964v1">"It Warned Me Just at the Right Moment": Exploring LLM-based Real-time Detection of Phone Scams</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Despite living in the era of the internet, phone-based scams remain one of the most prevalent forms of scams. These scams aim to exploit victims for financial gain, causing both monetary losses and psychological distress. While governments, industries, and academia have actively introduced various countermeasures, scammers also continue to evolve their tactics, making phone scams a persistent threat. To combat these increasingly sophisticated scams, detection technologies must also advance. In this work, we propose a framework for modeling scam calls and introduce an LLM-based real-time detection approach, which assesses fraudulent intent in conversations, further providing immediate warnings to users to mitigate harm. Through experiments, we evaluate the method's performance and analyze key factors influencing its effectiveness. This analysis enables us to refine the method to improve precision while exploring the trade-off between recall and timeliness, paving the way for future directions in this critical area of research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18652v6">$C^2$: Scalable Auto-Feedback for LLM-based Chart Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 NAACL 2025 Main (Long)
    </div>
    <details class="paper-abstract">
      Generating high-quality charts with Large Language Models (LLMs) presents significant challenges due to limited data and the high cost of scaling through human curation. $\langle \text{instruction}, \text{data}, \text{code} \rangle$ triplets are scarce and expensive to manually curate as their creation demands technical expertise. To address this scalability challenge, we introduce a reference-free automatic feedback generator, which eliminates the need for costly human intervention. Our novel framework, C$^2$, consists of (1) an automatic feedback provider (ChartAF) and (2) a diverse, reference-free dataset (ChartUIE-8K). The results are compelling: in our first experiment, 74% of respondents strongly preferred, and 10% preferred, the results after feedback. The second post-feedback experiment demonstrates that ChartAF outperform nine baselines. Moreover, ChartUIE-8K significantly improves data diversity by increasing queries, datasets, and chart types by 5982%, 1936%, and 91%, respectively, over benchmarks. Finally, a study of LLM users revealed that 94% of participants preferred ChartUIE-8K's queries, with 93% deeming them aligned with real-world use cases. Core contributions are available as open-source at chartsquared.github.io, with ample qualitative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v1">InfinitePOD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 takes a middle-ground approach by leveraging Optical Circuit Switches, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfinitePOD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfinitePOD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt into variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh) based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfinitePOD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03884v1">Rank Also Matters: Hierarchical Configuration for Mixture of Adapter Experts in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across various tasks, accompanied by a continuous increase in their parameter size. Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), address the challenges of fine-tuning LLMs by significantly reducing the number of trainable parameters. Recent studies have integrated LoRA with Mixture of Experts (MoE) architectures, leveraging multiple adapter experts and gating mechanisms to further improve fine-tuning performance. However, existing approaches primarily focus on adjusting the allocations of adapter experts per layer to optimize the introduced trainable parameter size, while neglecting a critical factor of adapters' rank. To this end, we propose a hierarchical scheme for expert allocation and rank configuration, HILO, which dynamically adjusts the number and rank of adapter experts across layers, matching the varying representational complexity of model layers in adapter-granularity. Extensive experiments on multiple benchmark tasks demonstrate that HILO outperforms existing methods in accuracy while introducing fewer trainable parameters, providing an efficient and practical solution for fine-tuning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03843v1">Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      High-quality, large-scale instructions are crucial for aligning large language models (LLMs), however, there is a severe shortage of instruction in the field of natural language understanding (NLU). Previous works on constructing NLU instructions mainly focus on information extraction (IE), neglecting tasks such as machine reading comprehension, question answering, and text classification. Furthermore, the lack of diversity in the data has led to a decreased generalization ability of trained LLMs in other NLU tasks and a noticeable decline in the fundamental model's general capabilities. To address this issue, we propose Hum, a large-scale, high-quality synthetic instruction corpus for NLU tasks, designed to enhance the NLU capabilities of LLMs. Specifically, Hum includes IE (either close IE or open IE), machine reading comprehension, text classification, and instruction generalist tasks, thereby enriching task diversity. Additionally, we introduce a human-LLMs collaborative mechanism to synthesize instructions, which enriches instruction diversity by incorporating guidelines, preference rules, and format variants. We conduct extensive experiments on 5 NLU tasks and 28 general capability evaluation datasets for LLMs. Experimental results show that Hum enhances the NLU capabilities of six LLMs by an average of 3.1\%, with no significant decline observed in other general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v1">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06387v4">Self-Training Meets Consistency: Improving LLMs' Reasoning with Consistency-Driven Rationale Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Self-training approach for large language models (LLMs) improves reasoning abilities by training the models on their self-generated rationales. Previous approaches have labeled rationales that produce correct answers for a given question as appropriate for training. However, a single measure risks misjudging rationale quality, leading the models to learn flawed reasoning patterns. To address this issue, we propose CREST (Consistency-driven Rationale Evaluation for Self-Training), a self-training framework that further evaluates each rationale through follow-up questions and leverages this evaluation to guide its training. Specifically, we introduce two methods: (1) filtering out rationales that frequently result in incorrect answers on follow-up questions and (2) preference learning based on mixed preferences from rationale evaluation results of both original and follow-up questions. Experiments on three question-answering datasets using open LLMs show that CREST not only improves the logical robustness and correctness of rationales but also improves reasoning abilities compared to previous self-training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01720v3">Towards a Theoretical Understanding of Synthetic Data in LLM Post-Training: A Reverse-Bottleneck Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      Synthetic data has become a pivotal resource in post-training tasks for large language models (LLMs) due to the scarcity of high-quality, specific data. While various methods have been developed to generate synthetic data, there remains a discernible gap between the practical effects of synthetic data and our theoretical comprehension. To address this challenge, we commence by presenting a detailed modeling of the prevalent synthetic data generation process. Building upon this modeling, we demonstrate that the generalization capability of the post-trained model is critically determined by the information gain derived from the generative model, as analyzed from a novel reverse-bottleneck perspective. Moreover, we introduce the concept of Generalization Gain via Mutual Information (GGMI) and elucidate the relationship between generalization gain and information gain. This analysis serves as a theoretical foundation for synthetic data generation and further highlights its connection with the generalization capability of post-trained models, offering an understanding about the design of synthetic data generation techniques and the optimization of the post-training process. We open-source our code at https://github.com/ZyGan1999/Towards-a-Theoretical-Understanding-of-Synthetic-Data-in-LLM-Post-Training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14259v2">Beyond Binary: Towards Fine-Grained LLM-Generated Text Detection via Role Recognition and Involvement Measurement</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 Social Media, Large Language Models, LLM-generated Text Detection, AI-assisted News Detection; Accepted by WWW2025
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs), like ChatGPT, has resulted in the widespread presence of LLM-generated content on social media platforms, raising concerns about misinformation, data biases, and privacy violations, which can undermine trust in online discourse. While detecting LLM-generated content is crucial for mitigating these risks, current methods often focus on binary classification, failing to address the complexities of real-world scenarios like human-LLM collaboration. To move beyond binary classification and address these challenges, we propose a new paradigm for detecting LLM-generated content. This approach introduces two novel tasks: LLM Role Recognition (LLM-RR), a multi-class classification task that identifies specific roles of LLM in content generation, and LLM Influence Measurement (LLM-IM), a regression task that quantifies the extent of LLM involvement in content creation. To support these tasks, we propose LLMDetect, a benchmark designed to evaluate detectors' performance on these new tasks. LLMDetect includes the Hybrid News Detection Corpus (HNDC) for training detectors, as well as DetectEval, a comprehensive evaluation suite that considers five distinct cross-context variations and two multi-intensity variations within the same LLM role. This allows for a thorough assessment of detectors' generalization and robustness across diverse contexts. Our empirical validation of 10 baseline detection methods demonstrates that fine-tuned PLM-based models consistently outperform others on both tasks, while advanced LLMs face challenges in accurately detecting their own generated content. Our experimental results and analysis offer insights for developing more effective detection models for LLM-generated content. This research enhances the understanding of LLM-generated content and establishes a foundation for more nuanced detection methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v3">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 32 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but, even with domain-specific fine-tuning, often produce hallucinations for complex ones. While integrating LLMs with tools can mitigate this reliability issue, models finetuned on tool usage only often over-rely on them, incurring unnecessary costs from resource-intensive scientific tools even for simpler problems. Inspired by how human experts assess the complexity of the problem before choosing the solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tools-generated solutions. In the second component, Tool Usage Adaptation (TUA), we classify questions as easy or hard based on the WKL-trained model's accuracy, and train it to maintain direct reasoning for simple problems while switching to tools for challenging ones. We validate our method on 6 scientific benchmark datasets in climate science, epidemiology, and mathematics. Compared to the base 8B model, our trained models achieve 28.27% higher answer accuracy and 13.76% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4 and Claude-3.5 on 4 custom-created datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v2">An Investigation of FP8 Across Accelerators for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      The introduction of 8-bit floating-point (FP8) computation units in modern AI accelerators has generated significant interest in FP8-based large language model (LLM) inference. Unlike 16-bit floating-point formats, FP8 in deep learning requires a shared scaling factor. Additionally, while E4M3 and E5M2 are well-defined at the individual value level, their scaling and accumulation methods remain unspecified and vary across hardware and software implementations. As a result, FP8 behaves more like a quantization format than a standard numeric representation. In this work, we provide the first comprehensive analysis of FP8 computation and acceleration on two AI accelerators: the NVIDIA H100 and Intel Gaudi 2. Our findings highlight that the Gaudi 2, by leveraging FP8, achieves higher throughput-to-power efficiency during LLM inference, offering valuable insights into the practical implications of FP8 adoption for datacenter-scale LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03723v1">Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Credit assignment, the process of attributing credit or blame to individual agents for their contributions to a team's success or failure, remains a fundamental challenge in multi-agent reinforcement learning (MARL), particularly in environments with sparse rewards. Commonly-used approaches such as value decomposition often lead to suboptimal policies in these settings, and designing dense reward functions that align with human intuition can be complex and labor-intensive. In this work, we propose a novel framework where a large language model (LLM) generates dense, agent-specific rewards based on a natural language description of the task and the overall team goal. By learning a potential-based reward function over multiple queries, our method reduces the impact of ranking errors while allowing the LLM to evaluate each agent's contribution to the overall task. Through extensive experiments, we demonstrate that our approach achieves faster convergence and higher policy returns compared to state-of-the-art MARL baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03708v1">Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers</a></div>
    <div class="paper-meta">
      📅 2025-02-06
    </div>
    <details class="paper-abstract">
      A trained Large Language Model (LLM) contains much of human knowledge. Yet, it is difficult to gauge the extent or accuracy of that knowledge, as LLMs do not always ``know what they know'' and may even be actively misleading. In this work, we give a general method for detecting semantic concepts in the internal activations of LLMs. Furthermore, we show that our methodology can be easily adapted to steer LLMs toward desirable outputs. Our innovations are the following: (1) we use a nonlinear feature learning method to identify important linear directions for predicting concepts from each layer; (2) we aggregate features across layers to build powerful concept detectors and steering mechanisms. We showcase the power of our approach by attaining state-of-the-art results for detecting hallucinations, harmfulness, toxicity, and untruthful content on seven benchmarks. We highlight the generality of our approach by steering LLMs towards new concepts that, to the best of our knowledge, have not been previously considered in the literature, including: semantic disambiguation, human languages, programming languages, hallucinated responses, science subjects, poetic/Shakespearean English, and even multiple concepts simultaneously. Moreover, our method can steer concepts with numerical attributes such as product reviews. We provide our code (including a simple API for our methods) at https://github.com/dmbeaglehole/neural_controllers .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03699v1">LLM Alignment as Retriever Optimization: An Information Retrieval Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-06
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
    </details>
</div>
