# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06753v1">Pushing the Envelope of LLM Inference on AI-PC</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The advent of ultra-low-bit LLM models (1/1.58/2-bit), which match the perplexity and end-task performance of their full-precision counterparts using the same model size, is ushering in a new era of LLM inference for resource-constrained environments such as edge devices and AI PCs. While these quantization advances promise models that are more cost-effective in terms of latency, memory, throughput, and energy consumption, the computational efficiency of state-of-the-art (SOTA) inference runtimes (e.g., bitnet.cpp) used to deploy them remains underexplored. In this work, we take a bottom-up approach: we first design and implement 1-bit and 2-bit microkernels optimized for modern CPUs, achieving peak computational efficiency across a variety of CPU platforms. We integrate these microkernels into a state-of-the-art LLM inference framework, namely PyTorch-TPP, and present end-to-end inference results with 2-bit models that outperform the current SOTA runtime bitnet.cpp by up to 2.2x, and deliver up to 7x speedup compared to the 16-bit model inference. Our optimized runtime advances the state of LLM inference on AI PCs and edge devices, paving the way for efficient deployment of ultra-low-bit LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03628v2">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Sellers at eBay are recommended keyphrases to bid on to enhance the performance of their advertising campaigns. The relevance of these keyphrases is crucial in avoiding the overcrowding of search systems with irrelevant items and maintaining a positive seller perception. It is essential that keyphrase recommendations align with both seller and Search judgments regarding auctions. Due to the difficulty in procuring negative human judgment at scale, employing LLM-as-a-judge to mimic seller judgment has been established as the norm in several studies. This study introduces a novel two-step LLM distillation process from a LLM-judge used to debias our Embedding Based Retrieval (EBR) model from the various biases that exist in click-data. We distill from an LLM teacher via a cross-encoder assistant into a bi-encoder student using a multi-task training approach, ultimately employing the student bi-encoder to retrieve relevant advertiser keyphrases. We show that integrating a knowledge distillation process from LLMs in a multi-task training setup enhances bi-encoder performance in retrieving relevant advertiser keyphrases at eBay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06709v1">Play Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can serve as judges that offer rapid and reliable assessments of other LLM outputs. However, models may systematically assign overly favorable ratings to their own outputs, a phenomenon known as self-bias, which can distort evaluations of true model performance. Previous studies often conflate genuine differences in model quality with bias or incorrectly assume that evaluations from LLMs and humans follow the same rating distributions. In this work, we present a statistical framework that explicitly formalizes assumptions under which self-bias can be identified and estimated. Our method models the difference in the scoring distribution that LLM-as-a-judge assigns to its own completions compared to other models, while accounting for the underlying quality of the completions provided by an independent, third-party judge (e.g., humans). Our method reliably isolates and quantifies self-bias, even when models vary in ability, ensuring that genuine performance differences are not mistaken for self-bias. We conduct an empirical analysis of self-bias on a large dataset (>5000 prompt-completion pairs) consisting of expert human annotations and judgments from nine different LLM judges. We find that some models, such as GPT-4o and Claude 3.5 Sonnet, systematically assign higher scores to their own outputs. These models also display family-bias; systematically assigning higher ratings to outputs produced by other models of the same family. Our findings highlight potential pitfalls of using LLM judges and offer practical guidance to mitigate biases when interpreting automated evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08954v3">Should you use LLMs to simulate opinions? Quality checks for early-stage deliberation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The emergent capabilities of large language models (LLMs) have prompted interest in using them as surrogates for human subjects in opinion surveys. However, prior evaluations of LLM-based opinion simulation have relied heavily on costly, domain-specific survey data, and mixed empirical results leave their reliability in question. To enable cost-effective, early-stage evaluation, we introduce a quality control assessment designed to test the viability of LLM-simulated opinions on Likert-scale tasks without requiring large-scale human data for validation. This assessment comprises two key tests: \emph{logical consistency} and \emph{alignment with stakeholder expectations}, offering a low-cost, domain-adaptable validation tool. We apply our quality control assessment to an opinion simulation task relevant to AI-assisted content moderation and fact-checking workflows -- a socially impactful use case -- and evaluate seven LLMs using a baseline prompt engineering method (backstory prompting), as well as fine-tuning and in-context learning variants. None of the models or methods pass the full assessment, revealing several failure modes. We conclude with a discussion of the risk management implications and release \texttt{TopicMisinfo}, a benchmark dataset with paired human and LLM annotations simulated by various models and approaches, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06601v1">Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 https://deepignorance.ai/
    </div>
    <details class="paper-abstract">
      Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text -- outperforming existing post-training baselines by over an order of magnitude -- with no observed degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06583v1">Discerning minds or generic tutors? Evaluating instructional guidance capabilities in Socratic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The conversational capabilities of large language models hold significant promise for enabling scalable and interactive tutoring. While prior research has primarily examined their capacity for Socratic questioning, it often overlooks a critical dimension: adaptively guiding learners based on their cognitive states. This study shifts focus from mere question generation to the broader instructional guidance capability. We ask: Can LLMs emulate expert tutors who dynamically adjust strategies in response to learners' understanding? To investigate this, we propose GuideEval, a benchmark grounded in authentic educational dialogues that evaluates pedagogical guidance through a three-phase behavioral framework: (1) Perception, inferring learner states; (2) Orchestration, adapting instructional strategies; and (3) Elicitation, stimulating proper reflections. Empirical findings reveal that existing LLMs frequently fail to provide effective adaptive scaffolding when learners exhibit confusion or require redirection. Furthermore, we introduce a behavior-guided finetuning strategy that leverages behavior-prompted instructional dialogues, significantly enhancing guidance performance. By shifting the focus from isolated content evaluation to learner-centered interaction, our work advocates a more dialogic paradigm for evaluating Socratic LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06479v1">The Problem of Atypicality in LLM-Powered Psychiatry</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Preprint of 8/8/2025 -- please cite published version. This article has been published in the Journal of Medical Ethics (2025) following peer review and can also be viewed on the journal's website at 10.1136/jme-2025-110972
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed as scalable solutions to the global mental health crisis. But their deployment in psychiatric contexts raises a distinctive ethical concern: the problem of atypicality. Because LLMs generate outputs based on population-level statistical regularities, their responses -- while typically appropriate for general users -- may be dangerously inappropriate when interpreted by psychiatric patients, who often exhibit atypical cognitive or interpretive patterns. We argue that standard mitigation strategies, such as prompt engineering or fine-tuning, are insufficient to resolve this structural risk. Instead, we propose dynamic contextual certification (DCC): a staged, reversible and context-sensitive framework for deploying LLMs in psychiatry, inspired by clinical translation and dynamic safety models from artificial intelligence governance. DCC reframes chatbot deployment as an ongoing epistemic and ethical process that prioritises interpretive safety over static performance benchmarks. Atypicality, we argue, cannot be eliminated -- but it can, and must, be proactively managed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06467v1">LLM Unlearning using Gradient Ratio-Based Influence Estimation and Noise Injection</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 14 Pages, 3 Figures, 11 Tables
    </div>
    <details class="paper-abstract">
      The growing legal and ethical scrutiny of large language models (LLMs) necessitates effective machine unlearning, particularly for sensitive or unauthorized data. Existing empirical methods often yield incomplete forgetting or unintended degradation of unrelated knowledge due to poor localization. In this work, we propose GRIN: a modular and targeted framework for LLM unlearning. GRIN introduces a novel gradient-ratio-based metric to identify parameters most responsible for memorizing forget data. We then perform selective noise injection into these parameters prior to fine-tuning, which improves unlearning performance while maintaining model utility. Finally, we propose new evaluation metrics tailored to the LLM setting and validate our approach on standard benchmarks such as TOFU, WMDP, and SafePKU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06447v1">SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Long-context inference for Large Language Models (LLMs) is heavily limited by high computational demands. While several existing methods optimize attention computation, they still process the full set of hidden states at each layer, limiting overall efficiency. In this work, we propose SlimInfer, an innovative framework that aims to accelerate inference by directly pruning less critical prompt tokens during the forward pass. Our key insight is an information diffusion phenomenon: As information from critical tokens propagates through layers, it becomes distributed across the entire sequence. This diffusion process suggests that LLMs can maintain their semantic integrity when excessive tokens, even including these critical ones, are pruned in hidden states. Motivated by this, SlimInfer introduces a dynamic fine-grained pruning mechanism that accurately removes redundant tokens of hidden state at intermediate layers. This layer-wise pruning naturally enables an asynchronous KV cache manager that prefetches required token blocks without complex predictors, reducing both memory usage and I/O costs. Extensive experiments show that SlimInfer can achieve up to $\mathbf{2.53\times}$ time-to-first-token (TTFT) speedup and $\mathbf{1.88\times}$ end-to-end latency reduction for LLaMA3.1-8B-Instruct on a single RTX 4090, without sacrificing performance on LongBench. Our code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06445v1">Echoes of Automation: The Increasing Use of LLMs in Newsmaking</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 To appear in 18th International Conference on Social Computing, Behavioral-Cultural Modeling, & Prediction and Behavior Representation in Modeling and Simulation, and to be published in the Springer LNCS series
    </div>
    <details class="paper-abstract">
      The rapid rise of Generative AI (GenAI), particularly LLMs, poses concerns for journalistic integrity and authorship. This study examines AI-generated content across over 40,000 news articles from major, local, and college news media, in various media formats. Using three advanced AI-text detectors (e.g., Binoculars, Fast-Detect GPT, and GPTZero), we find substantial increase of GenAI use in recent years, especially in local and college news. Sentence-level analysis reveals LLMs are often used in the introduction of news, while conclusions usually written manually. Linguistic analysis shows GenAI boosts word richness and readability but lowers formality, leading to more uniform writing styles, particularly in local media.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06435v1">Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming social-science research by enabling scalable, precise analysis. Their adaptability raises the question of whether knowledge acquired through fine-tuning in a few languages can transfer to unseen languages that only appeared during pre-training. To examine this, we fine-tune lightweight LLaMA 3.2-3B models on monolingual, bilingual, or multilingual data sets to classify immigration-related tweets from X/Twitter across 13 languages, a domain characterised by polarised, culturally specific discourse. We evaluate whether minimal language-specific fine-tuning enables cross-lingual topic detection and whether adding targeted languages corrects pre-training biases. Results show that LLMs fine-tuned in one or two languages can reliably classify immigration-related content in unseen languages. However, identifying whether a tweet expresses a pro- or anti-immigration stance benefits from multilingual fine-tuning. Pre-training bias favours dominant languages, but even minimal exposure to under-represented languages during fine-tuning (as little as $9.62\times10^{-11}$ of the original pre-training token volume) yields significant gains. These findings challenge the assumption that cross-lingual mastery requires extensive multilingual training: limited language coverage suffices for topic-level generalisation, and structural biases can be corrected with lightweight interventions. By releasing 4-bit-quantised, LoRA fine-tuned models, we provide an open-source, reproducible alternative to proprietary LLMs that delivers 35 times faster inference at just 0.00000989% of the dollar cost of the OpenAI GPT-4o model, enabling scalable, inclusive research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06412v1">Sample-efficient LLM Optimization with Reset Replay</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training Large Language Models (LLMs), particularly through Reinforcement Learning (RL) and preference optimization methods, are key drivers for enhancing their reasoning capabilities. However, these methods are often plagued by low sample efficiency and a susceptibility to primacy bias, where overfitting to initial experiences degrades policy quality and damages the learning process. To address these challenges, we introduce LLM optimization with Reset Replay (LoRR), a general and powerful plugin designed to enhance sample efficiency in any preference-based optimization framework. LoRR core mechanism enables training at a high replay number, maximizing the utility of each collected data batch. To counteract the risk of overfitting inherent in high-replay training, LoRR incorporates a periodic reset strategy with reusing initial data, which preserves network plasticity. Furthermore, it leverages a hybrid optimization objective, combining supervised fine-tuning (SFT) and preference-based losses to further bolster data exploitation. Our extensive experiments demonstrate that LoRR significantly boosts the performance of various preference optimization methods on both mathematical and general reasoning benchmarks. Notably, an iterative DPO approach augmented with LoRR achieves comparable performance on challenging math tasks, outperforming some complex and computationally intensive RL-based algorithms. These findings highlight that LoRR offers a practical, sample-efficient, and highly effective paradigm for LLM finetuning, unlocking greater performance from limited data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00207v2">Can LLM "Self-report"?: Evaluating the Validity of Self-report Scales in Measuring Personality Design in LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted by COLM 2025
    </div>
    <details class="paper-abstract">
      A chatbot's personality design is key to interaction quality. As chatbots evolved from rule-based systems to those powered by large language models (LLMs), evaluating the effectiveness of their personality design has become increasingly complex, particularly due to the open-ended nature of interactions. A recent and widely adopted method for assessing the personality design of LLM-based chatbots is the use of self-report questionnaires. These questionnaires, often borrowed from established human personality inventories, ask the chatbot to rate itself on various personality traits. Can LLM-based chatbots meaningfully "self-report" their personality? We created 500 chatbots with distinct personality designs and evaluated the validity of their self-report personality scores by examining human perceptions formed during interactions with these chatbots. Our findings indicate that the chatbot's answers on human personality scales exhibit weak correlations with both human-perceived personality traits and the overall interaction quality. These findings raise concerns about both the criterion validity and the predictive validity of self-report methods in this context. Further analysis revealed the role of task context and interaction in the chatbot's personality design assessment. We further discuss design implications for creating more contextualized and interactive evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v5">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 ACL 2025 Camera, Benchmark: https://huggingface.co/datasets/opencompass/LiveMathBench, Code: https://github.com/open-compass/GPassK
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has shown remarkable progress in complex reasoning tasks. However, a significant disparity exists between benchmark performances and real-world applications. We attribute this gap primarily to current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, especially in complex reasoning tasks where both accuracy and consistency are essential. In this paper, we introduce G-Pass@$k$, a novel evaluation metric that continuously assesses model performance across multiple sampling attempts, quantifying both the model's performance potential and its stability. Through extensive experiments on various public and newly constructed benchmarks, we employ G-Pass@$k$ in conjunction with state-of-the-art large language models to provide comprehensive insights into their potential capabilities and operational consistency. Our findings reveal a significant opportunity to enhance the realistic reasoning abilities of LLMs, underscoring the necessity for more robust evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06394v1">When AIOps Become "AI Oops": Subverting LLM-driven IT Operations via Telemetry Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 v0.1
    </div>
    <details class="paper-abstract">
      AI for IT Operations (AIOps) is transforming how organizations manage complex software systems by automating anomaly detection, incident diagnosis, and remediation. Modern AIOps solutions increasingly rely on autonomous LLM-based agents to interpret telemetry data and take corrective actions with minimal human intervention, promising faster response times and operational cost savings. In this work, we perform the first security analysis of AIOps solutions, showing that, once again, AI-driven automation comes with a profound security cost. We demonstrate that adversaries can manipulate system telemetry to mislead AIOps agents into taking actions that compromise the integrity of the infrastructure they manage. We introduce techniques to reliably inject telemetry data using error-inducing requests that influence agent behavior through a form of adversarial reward-hacking; plausible but incorrect system error interpretations that steer the agent's decision-making. Our attack methodology, AIOpsDoom, is fully automated--combining reconnaissance, fuzzing, and LLM-driven adversarial input generation--and operates without any prior knowledge of the target system. To counter this threat, we propose AIOpsShield, a defense mechanism that sanitizes telemetry data by exploiting its structured nature and the minimal role of user-generated content. Our experiments show that AIOpsShield reliably blocks telemetry-based attacks without affecting normal agent performance. Ultimately, this work exposes AIOps as an emerging attack vector for system compromise and underscores the urgent need for security-aware AIOps design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06388v1">LLMs vs. Chinese Anime Enthusiasts: A Comparative Study on Emotionally Supportive Role-Playing</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 21 pages, 17 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing conversations and providing emotional support as separate research directions. However, there remains a significant research gap in combining these capabilities to enable emotionally supportive interactions with virtual characters. To address this research gap, we focus on anime characters as a case study because of their well-defined personalities and large fan bases. This choice enables us to effectively evaluate how well LLMs can provide emotional support while maintaining specific character traits. We introduce ChatAnime, the first Emotionally Supportive Role-Playing (ESRP) dataset. We first thoughtfully select 20 top-tier characters from popular anime communities and design 60 emotion-centric real-world scenario questions. Then, we execute a nationwide selection process to identify 40 Chinese anime enthusiasts with profound knowledge of specific characters and extensive experience in role-playing. Next, we systematically collect two rounds of dialogue data from 10 LLMs and these 40 Chinese anime enthusiasts. To evaluate the ESRP performance of LLMs, we design a user experience-oriented evaluation system featuring 9 fine-grained metrics across three dimensions: basic dialogue, role-playing and emotional support, along with an overall metric for response diversity. In total, the dataset comprises 2,400 human-written and 24,000 LLM-generated answers, supported by over 132,000 human annotations. Experimental results show that top-performing LLMs surpass human fans in role-playing and emotional support, while humans still lead in response diversity. We hope this work can provide valuable resources and insights for future research on optimizing LLMs in ESRP. Our datasets are available at https://github.com/LanlanQiu/ChatAnime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06387v1">End-to-End Text-to-SQL with Dataset Selection: Leveraging LLMs for Adaptive Query Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted in IJCNN25
    </div>
    <details class="paper-abstract">
      Text-to-SQL bridges the gap between natural language and structured database language, thus allowing non-technical users to easily query databases. Traditional approaches model text-to-SQL as a direct translation task, where a given Natural Language Query (NLQ) is mapped to an SQL command. Recent advances in large language models (LLMs) have significantly improved translation accuracy, however, these methods all require that the target database is pre-specified. This becomes problematic in scenarios with multiple extensive databases, where identifying the correct database becomes a crucial yet overlooked step. In this paper, we propose a three-stage end-to-end text-to-SQL framework to identify the user's intended database before generating SQL queries. Our approach leverages LLMs and prompt engineering to extract implicit information from natural language queries (NLQs) in the form of a ruleset. We then train a large db\_id prediction model, which includes a RoBERTa-based finetuned encoder, to predict the correct Database identifier (db\_id) based on both the NLQ and the LLM-generated rules. Finally, we refine the generated SQL by using critic agents to correct errors. Experimental results demonstrate that our framework outperforms the current state-of-the-art models in both database intent prediction and SQL generation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06361v1">Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been widely deployed in reasoning, planning, and decision-making tasks, making their trustworthiness a critical concern. The potential for intentional deception, where an LLM deliberately fabricates or conceals information to serve a hidden objective, remains a significant and underexplored threat. Existing studies typically induce such deception by explicitly setting a "hidden" objective through prompting or fine-tuning, which may not fully reflect real-world human-LLM interactions. Moving beyond this human-induced deception, we investigate LLMs' self-initiated deception on benign prompts. To address the absence of ground truth in this evaluation, we propose a novel framework using "contact searching questions." This framework introduces two statistical metrics derived from psychological principles to quantify the likelihood of deception. The first, the Deceptive Intention Score, measures the model's bias towards a hidden objective. The second, Deceptive Behavior Score, measures the inconsistency between the LLM's internal belief and its expressed output. Upon evaluating 14 leading LLMs, we find that both metrics escalate as task difficulty increases, rising in parallel for most models. Building on these findings, we formulate a mathematical model to explain this behavior. These results reveal that even the most advanced LLMs exhibit an increasing tendency toward deception when handling complex problems, raising critical concerns for the deployment of LLM agents in complex and crucial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06309v1">Matrix-Driven Instant Review: Confident Detection and Reconstruction of LLM Plagiarism on PC</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      In recent years, concerns about intellectual property (IP) in large language models (LLMs) have grown significantly. Plagiarizing other LLMs (through direct weight copying, upcycling, pruning, or continual pretraining) and claiming authorship without properly attributing to the original license, is a serious misconduct that can lead to significant financial and reputational harm to the original developers. However, existing methods for detecting LLM plagiarism fall short in key areas. They fail to accurately reconstruct weight correspondences, lack the ability to compute statistical significance measures such as $p$-values, and may mistakenly flag models trained on similar data as being related. To address these limitations, we propose Matrix-Driven Instant Review (MDIR), a novel method that leverages matrix analysis and Large Deviation Theory. MDIR achieves accurate reconstruction of weight relationships, provides rigorous $p$-value estimation, and focuses exclusively on weight similarity without requiring full model inference. Experimental results demonstrate that MDIR reliably detects plagiarism even after extensive transformations, such as random permutations and continual pretraining with trillions of tokens. Moreover, all detections can be performed on a single PC within an hour, making MDIR both efficient and accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14810v2">DONOD: Efficient and Generalizable Instruction Fine-Tuning for LLMs via Model-Intrinsic Dataset Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Ad-hoc instruction fine-tuning of large language models (LLMs) is widely adopted for domain-specific adaptation. While domain-specific supervised fine-tuning (SFT) is effective and efficient, it often weakens cross-domain generalization and struggles with noisy training data. To address these challenges, we propose DONOD, a lightweight model-intrinsic data pruning method. Our approach evaluates data using two model-parameter-based metrics: Delta of Norm (DON), which captures the cumulative influence on model weights, and Norm of Delta (NOD), which quantifies weight instability. Moreover, by employing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) algorithm, we effectively filter noisy, unlearnable, and generalization-harming samples without relying on auxiliary models during the SFT process. Experiments on mathematical tasks demonstrate that data selected by DONOD achieves superior fine-tuning efficiency and improved robustness against noisy data. By filtering out 70% of the whole dataset, we improve target-domain accuracy by 14.90% and cross-domain accuracy by 5.67%. Meanwhile, our selected data present superior cross-architecture generalization. Data pruned by smaller models (e.g., Llama 3.1-8B) generalize effectively on larger models (e.g., Llama 2-13B). Compared to existing related methodologies, DONOD demonstrates comparable or superior performance while remaining dataset-agnostic, enabling broader applicability. Code will be made publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06297v1">KV Cache Compression for Inference Efficiency in LLMs: A Review</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Withtherapid advancement of large language models (LLMs), the context length for inference has been continuously increasing, leading to an exponential growth in the demand for Key-Value (KV) caching. This has resulted in a significant memory bottleneck, limiting the inference efficiency and scalability of the models. Therefore, optimizing the KV cache during inference is crucial for enhancing performance and efficiency. This review systematically examines current KV cache optimization techniques, including compression strategies such as selective token strategies, quantization, and attention compression. We evaluate the effectiveness, trade-offs, and application scenarios of these methods, providing a comprehensive analysis of their impact on memory usage and inference speed. We focus on identifying the limitations and challenges of existing methods, such as compatibility issues with different models and tasks. Additionally, this review highlights future research directions, including hybrid optimization techniques, adaptive dynamic strategies, and software-hardware co-design. These approaches aim to improve inference efficiency and promote the practical application of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06296v1">LLM Robustness Leaderboard v1 --Technical report</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02679v2">LLM Agent-Based Simulation of Student Activities and Mental Health Using Smartphone Sensing Data</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Students' mental well-being is vital for academic success, with activities such as studying, socializing, and sleeping playing a role. Current mobile sensing data highlight this intricate link using statistical and machine learning analyses. We propose a novel LLM agent-based simulation framework to model student activities and mental health using the StudentLife Dataset. Each LLM agent was initialized with personality questionnaires and guided by smartphone sensing data throughout the simulated semester. These agents predict individual behaviors, provide self-reported mental health data via ecological momentary assessments (EMAs), and complete follow-up personality questionnaires. To ensure accuracy, we investigated various prompting techniques, memory systems, and activity-based mental state management strategies that dynamically update an agent's mental state based on their daily activities. This simulation goes beyond simply replicating existing data. This allows us to explore new scenarios that are not present in the original dataset, such as peer influence through agent-to-agent interactions and the impact of social media. Furthermore, we can conduct intervention studies by manipulating activity patterns via sensing signals and personality traits using questionnaire responses. This provides valuable insights into the behavioral changes that could enhance student well-being. The framework also facilitates hypothetical interviews with LLM agents, offering deeper insights into their mental health. This study showcases the power of LLM-driven behavioral modeling with sensing data, opening new avenues for understanding and supporting student mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14110v2">Feedback-Guided Extraction of Knowledge Base from Retrieval-Augmented LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) by integrating external knowledge bases, whose construction is often time-consuming and laborious. If an adversary extracts the knowledge base verbatim, it not only severely infringes the owner's intellectual property but also enables the adversary to replicate the application's functionality for unfair competition. Previous works on knowledge base extraction are limited either by low extraction coverage (usually less than 4%) in query-based attacks or by impractical assumptions of white-box access in embedding-based optimization methods. In this work, we propose CopyBreakRAG, an agent-based black-box attack that reasons from feedback and adaptively generates new adversarial queries for progressive extraction. By balancing exploration and exploitation through curiosity-driven queries and feedback-guided query refinement, our method overcomes the limitations of prior approaches and achieves significantly higher extraction coverage in realistic black-box settings. Experimental results show that CopyBreakRAG outperforms the state-of-the-art black-box approach by 45% on average in terms of chunk extraction ratio from applications built with mainstream RAG frameworks, and extracts over 70% of the data from the knowledge base in applications on commercial platforms including OpenAI's GPTs and ByteDance's Coze when essential protection is in place.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06225v1">Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as automated judges, where practical value depends on both accuracy and trustworthy, risk-aware judgments. Existing approaches predominantly focus on accuracy, overlooking the necessity of well-calibrated confidence, which is vital for adaptive and reliable evaluation pipelines. In this work, we advocate a shift from accuracy-centric evaluation to confidence-driven, risk-aware LLM-as-a-Judge systems, emphasizing the necessity of well-calibrated confidence for trustworthy and adaptive evaluation. We systematically identify the **Overconfidence Phenomenon** in current LLM-as-a-Judges, where predicted confidence significantly overstates actual correctness, undermining reliability in practical deployment. To quantify this phenomenon, we introduce **TH-Score**, a novel metric measuring confidence-accuracy alignment. Furthermore, we propose **LLM-as-a-Fuser**, an ensemble framework that transforms LLMs into reliable, risk-aware evaluators. Extensive experiments demonstrate that our approach substantially improves calibration and enables adaptive, confidence-driven evaluation pipelines, achieving superior reliability and accuracy compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06186v1">DKG-LLM : A Framework for Medical Diagnosis and Personalized Treatment Recommendations via Dynamic Knowledge Graph and Large Language Model Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have grown exponentially since the release of ChatGPT. These models have gained attention due to their robust performance on various tasks, including language processing tasks. These models achieve understanding and comprehension of tasks by training billions of parameters. The development of these models is a transformative force in enhancing natural language understanding and has taken a significant step towards artificial general intelligence (AGI). In this study, we aim to present the DKG-LLM framework. The DKG-LLM framework introduces a groundbreaking approach to medical diagnosis and personalized treatment recommendations by integrating a dynamic knowledge graph (DKG) with the Grok 3 large language model. Using the Adaptive Semantic Fusion Algorithm (ASFA), heterogeneous medical data (including clinical reports and PubMed articles) and patient records dynamically generate a knowledge graph consisting of 15,964 nodes in 13 distinct types (e.g., diseases, symptoms, treatments, patient profiles) and 127,392 edges in 26 relationship types (e.g., causal, therapeutic, association). ASFA utilizes advanced probabilistic models, Bayesian inference, and graph optimization to extract semantic information, dynamically updating the graph with approximately 150 new nodes and edges in each data category while maintaining scalability with up to 987,654 edges. Real-world datasets, including MIMIC-III and PubMed, were utilized to evaluate the proposed architecture. The evaluation results show that DKG-LLM achieves a diagnostic accuracy of 84.19%. The model also has a treatment recommendation accuracy of 89.63% and a semantic coverage of 93.48%. DKG-LLM is a reliable and transformative tool that handles noisy data and complex multi-symptom diseases, along with feedback-based learning from physician input.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19349v2">DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      To alleviate the memory bandwidth bottleneck in Large Language Model (LLM) inference workloads, weight matrices are stored in memory in quantized and sparsified formats. Hence, before tiles of these matrices can be processed by in-core generalized matrix multiplication (GeMM) hardware engines, they need to be dequantized and de-sparsified. This is currently performed in software with vector operations. Unfortunately, this approach delivers only modest performance. Moreover, it is hard to understand how to improve the system, as the overall GeMM performance depends on the interaction between memory resources, vector units, and hardware matrix engines. To improve the performance of LLM inference in advanced platforms equipped with in-core GeMM engines and HBM, this paper makes three main contributions. First, it develops an analytical performance model with a 3D visual representation that provides insights into how memory resources, vector units, and hardware matrix engines interact to deliver compressed GeMM performance. Second, it proposes DECA, a new near-core ML-model decompression accelerator. DECA offloads tile de-sparsification and dequantization from the CPU, producing ready-to-use tiles for in-core GeMM engines. Third, it introduces a new ISA extension that enables out-of-order invocation of the near-core accelerator. With this extension, accelerator and core computations can interleave and overlap with high-performance. Our evaluation shows that, in a simulated 56-core Xeon 4 server with HBM, DECA accelerates the execution of compressed GeMMs by up to 4x over the use of optimized Intel software kernels. Further, DECA reduces the next-token generation time of Llama2-70B and OPT-66B by 1.6x-2.6x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00806v2">Adacc: An Adaptive Framework Unifying Compression and Activation Recomputation for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) is often constrained by GPU memory limitations. To alleviate memory pressure, activation recomputation and data compression have been proposed as two major strategies. However, both approaches have limitations: recomputation introduces significant training overhead, while compression can lead to accuracy degradation and computational inefficiency when applied naively. In this paper, we propose Adacc, the first adaptive memory optimization framework that unifies activation recomputation and data compression to improve training efficiency for LLMs while preserving model accuracy. Unlike existing methods that apply static, rule-based strategies or rely solely on one technique, Adacc makes fine-grained, tensor-level decisions, dynamically selecting between recomputation, retention, and compression based on tensor characteristics and runtime hardware constraints. Adacc tackles three key challenges: (1) it introduces layer-specific compression algorithms that mitigate accuracy loss by accounting for outliers in LLM activations; (2) it employs a MILP-based scheduling policy to globally optimize memory strategies across layers; and (3) it integrates an adaptive policy evolution mechanism to update strategies during training in response to changing data distributions. Experimental results show that Adacc improves training throughput by 1.01x to 1.37x compared to state-of-the-art frameworks, while maintaining accuracy comparable to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06178v1">Comparing Knowledge Injection Methods for LLMs in a Low-Resource Regime</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often require vast amounts of text to effectively acquire new knowledge. While continuing pre-training on large corpora or employing retrieval-augmented generation (RAG) has proven successful, updating an LLM with only a few thousand or million tokens remains challenging. In this work, we investigate the task of injecting small, unstructured information into LLMs and its relation to the catastrophic forgetting phenomenon. We use a dataset of recent news -- ensuring no overlap with the model's pre-training data -- to evaluate the knowledge acquisition by probing the model with question-answer pairs related the learned information. Starting from a continued pre-training baseline, we explored different augmentation algorithms to generate synthetic data to improve the knowledge acquisition capabilities. Our experiments show that simply continuing pre-training on limited data yields modest improvements, whereas exposing the model to diverse textual variations significantly improves the learning of new facts -- particularly with methods that induce greater variability through diverse prompting. Furthermore, we shed light on the forgetting phenomenon in small-data regimes, illustrating the delicate balance between learning new content and retaining existing capabilities. We also confirm the sensitivity of RAG-based approaches for knowledge injection, which often lead to greater degradation on control datasets compared to parametric methods. Finally, we demonstrate that models can generate effective synthetic training data themselves, suggesting a pathway toward self-improving model updates. All code and generated data used in our experiments are publicly available, providing a resource for studying efficient knowledge injection in LLMs with limited data at https://github.com/hugoabonizio/knowledge-injection-methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06167v1">Pragmatics beyond humans: meaning, communication, and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The paper reconceptualizes pragmatics not as a subordinate, third dimension of meaning, but as a dynamic interface through which language operates as a socially embedded tool for action. With the emergence of large language models (LLMs) in communicative contexts, this understanding needs to be further refined and methodologically reconsidered. The first section challenges the traditional semiotic trichotomy, arguing that connectionist LLM architectures destabilize established hierarchies of meaning, and proposes the Human-Machine Communication (HMC) framework as a more suitable alternative. The second section examines the tension between human-centred pragmatic theories and the machine-centred nature of LLMs. While traditional, Gricean-inspired pragmatics continue to dominate, it relies on human-specific assumptions ill-suited to predictive systems like LLMs. Probabilistic pragmatics, particularly the Rational Speech Act framework, offers a more compatible teleology by focusing on optimization rather than truth-evaluation. The third section addresses the issue of substitutionalism in three forms - generalizing, linguistic, and communicative - highlighting the anthropomorphic biases that distort LLM evaluation and obscure the role of human communicative subjects. Finally, the paper introduces the concept of context frustration to describe the paradox of increased contextual input paired with a collapse in contextual understanding, emphasizing how users are compelled to co-construct pragmatic conditions both for the model and themselves. These arguments suggest that pragmatic theory may need to be adjusted or expanded to better account for communication involving generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05734v2">LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted by COLM 2025
    </div>
    <details class="paper-abstract">
      Recent studies have discovered that large language models (LLM) may be ``fooled'' to output private information, including training data, system prompts, and personally identifiable information, under carefully crafted adversarial prompts. Existing red-teaming approaches for privacy leakage either rely on manual efforts or focus solely on system prompt extraction, making them ineffective for severe risks of training data leakage. We propose LeakAgent, a novel black-box red-teaming framework for LLM privacy leakage. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for both training data extraction and system prompt extraction. To achieve this, we propose a novel reward function to provide effective and fine-grained rewards and design novel mechanisms to balance exploration and exploitation during learning and enhance the diversity of adversarial prompts. Through extensive evaluations, we first show that LeakAgent significantly outperforms existing rule-based approaches in training data extraction and automated methods in system prompt leakage. We also demonstrate the effectiveness of LeakAgent in extracting system prompts from real-world applications in OpenAI's GPT Store. We further demonstrate LeakAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/LeakAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06149v1">Scaling Personality Control in LLMs with Big Five Scaler Prompts</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      We present Big5-Scaler, a prompt-based framework for conditioning large language models (LLMs) with controllable Big Five personality traits. By embedding numeric trait values into natural language prompts, our method enables fine-grained personality control without additional training. We evaluate Big5-Scaler across trait expression, dialogue generation, and human trait imitation tasks. Results show that it induces consistent and distinguishable personality traits across models, with performance varying by prompt type and scale. Our analysis highlights the effectiveness of concise prompts and lower trait intensities, providing a efficient approach for building personality-aware dialogue agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05616v1">TrajEvo: Trajectory Prediction Heuristics Design via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 arXiv admin note: substantial text overlap with arXiv:2505.04480
    </div>
    <details class="paper-abstract">
      Trajectory prediction is a critical task in modeling human behavior, especially in safety-critical domains such as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy and generalizability. Although deep learning approaches offer improved performance, they typically suffer from high computational cost, limited explainability, and, importantly, poor generalization to out-of-distribution (OOD) scenarios. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We propose two key innovations: Cross-Generation Elite Sampling to encourage population diversity, and a Statistics Feedback Loop that enables the LLM to analyze and improve alternative predictions. Our evaluations demonstrate that TrajEvo outperforms existing heuristic methods across multiple real-world datasets, and notably surpasses both heuristic and deep learning methods in generalizing to an unseen OOD real-world dataset. TrajEvo marks a promising step toward the automated design of fast, explainable, and generalizable trajectory prediction heuristics. We release our source code to facilitate future research at https://github.com/ai4co/trajevo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05571v1">Fairy$\pm i$: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v3">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00255v2">SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      This study evaluates large language models (LLMs) in generating code from algorithm descriptions in recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a dual-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implements solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful non-reasoning and reasoning LLMs as foundational models. The best-performing LLM using \ModelName~achieves only 39% execution accuracy, highlighting the benchmark's difficulty. Our analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We make available our benchmark and code at https://github.com/xyzCS/SciReplicate-Bench and project homepage at https://xyzcs.github.io/scireplicate.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09032v2">Teaching LLMs How to Learn with Contextual Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Prompting Large Language Models (LLMs), or providing context on the expected model of operation, is an effective way to steer the outputs of such models to satisfy human desiderata after they have been trained. But in rapidly evolving domains, there is often need to fine-tune LLMs to improve either the kind of knowledge in their memory or their abilities to perform open ended reasoning in new domains. When human's learn new concepts, we often do so by linking the new material that we are studying to concepts we have already learned before. To that end, we ask, "can prompting help us teach LLMs how to learn". In this work, we study a novel generalization of instruction tuning, called contextual fine-tuning, to fine-tune LLMs. Our method leverages instructional prompts designed to mimic human cognitive strategies in learning and problem-solving to guide the learning process during training, aiming to improve the model's interpretation and understanding of domain-specific knowledge. We empirically demonstrate that this simple yet effective modification improves the ability of LLMs to be fine-tuned rapidly on new datasets both within the medical and financial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05525v1">The World According to LLMs: How Geographic Origin Influences LLMs' Entity Deduction Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Conference on Language Modeling 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been extensively tuned to mitigate explicit biases, yet they often exhibit subtle implicit biases rooted in their pre-training data. Rather than directly probing LLMs with human-crafted questions that may trigger guardrails, we propose studying how models behave when they proactively ask questions themselves. The 20 Questions game, a multi-turn deduction task, serves as an ideal testbed for this purpose. We systematically evaluate geographic performance disparities in entity deduction using a new dataset, Geo20Q+, consisting of both notable people and culturally significant objects (e.g., foods, landmarks, animals) from diverse regions. We test popular LLMs across two gameplay configurations (canonical 20-question and unlimited turns) and in seven languages (English, Hindi, Mandarin, Japanese, French, Spanish, and Turkish). Our results reveal geographic disparities: LLMs are substantially more successful at deducing entities from the Global North than the Global South, and the Global West than the Global East. While Wikipedia pageviews and pre-training corpus frequency correlate mildly with performance, they fail to fully explain these disparities. Notably, the language in which the game is played has minimal impact on performance gaps. These findings demonstrate the value of creative, free-form evaluation frameworks for uncovering subtle biases in LLMs that remain hidden in standard prompting setups. By analyzing how models initiate and pursue reasoning goals over multiple turns, we find geographic and cultural disparities embedded in their reasoning processes. We release the dataset (Geo20Q+) and code at https://sites.google.com/view/llmbias20q/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05512v1">RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accept at CIKM 2025
    </div>
    <details class="paper-abstract">
      Evaluating the quality of retrieval-augmented generation (RAG) and document reranking systems remains challenging due to the lack of scalable, user-centric, and multi-perspective evaluation tools. We introduce RankArena, a unified platform for comparing and analysing the performance of retrieval pipelines, rerankers, and RAG systems using structured human and LLM-based feedback as well as for collecting such feedback. RankArena supports multiple evaluation modes: direct reranking visualisation, blind pairwise comparisons with human or LLM voting, supervised manual document annotation, and end-to-end RAG answer quality assessment. It captures fine-grained relevance feedback through both pairwise preferences and full-list annotations, along with auxiliary metadata such as movement metrics, annotation time, and quality ratings. The platform also integrates LLM-as-a-judge evaluation, enabling comparison between model-generated rankings and human ground truth annotations. All interactions are stored as structured evaluation datasets that can be used to train rerankers, reward models, judgment agents, or retrieval strategy selectors. Our platform is publicly available at https://rankarena.ngrok.io/, and the Demo video is provided https://youtu.be/jIYAP4PaSSI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05496v1">InfiAlign: A Scalable and Sample-Efficient Framework for Aligning LLMs to Enhance Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive reasoning abilities on a wide range of complex tasks. However, enhancing these capabilities through post-training remains resource intensive, particularly in terms of data and computational cost. Although recent efforts have sought to improve sample efficiency through selective data curation, existing methods often rely on heuristic or task-specific strategies that hinder scalability. In this work, we introduce InfiAlign, a scalable and sample-efficient post-training framework that integrates supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to align LLMs for enhanced reasoning. At the core of InfiAlign is a robust data selection pipeline that automatically curates high-quality alignment data from open-source reasoning datasets using multidimensional quality metrics. This pipeline enables significant performance gains while drastically reducing data requirements and remains extensible to new data sources. When applied to the Qwen2.5-Math-7B-Base model, our SFT model achieves performance on par with DeepSeek-R1-Distill-Qwen-7B, while using only approximately 12% of the training data, and demonstrates strong generalization across diverse reasoning tasks. Additional improvements are obtained through the application of DPO, with particularly notable gains in mathematical reasoning tasks. The model achieves an average improvement of 3.89% on AIME 24/25 benchmarks. Our results highlight the effectiveness of combining principled data selection with full-stage post-training, offering a practical solution for aligning large reasoning models in a scalable and data-efficient manner. The model checkpoints are available at https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05469v1">Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      We develop mechanisms for evaluating AI systems without ground truth by exploiting a connection between gaming resistance and output quality. The data processing inequality ensures post-hoc attempts to game a metric degrades both information content and task performance. We prove that f-mutual information measures are the unique gaming resistant mechanisms under natural conditions, with the overseer acting as an agent. While Shannon mutual information faces exponential sample complexity, bounded measures like total variation distance remain tractable. Empirically, across ten domains from translation to peer review, all information-theoretic mechanisms achieve perfect discrimination (d > 0.5) between faithful and strategic agents. In contrast, LLM judges exhibit systematic evaluation inversion, preferring fabricated content over accurate summaries. Our mechanisms show 10-100x better robustness to adversarial manipulation than current practices. We also find performance follows an inverted-U curve with compression ratio, peaking at 10:1 where agent responses exhibit optimal information diversity (3 effective dimensions), giving a bias-variance perspective on when our approach is expected to be most effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01545v2">Getting out of the Big-Muddy: Escalation of Commitment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in autonomous decision-making roles across high-stakes domains. However, since models are trained on human-generated data, they may inherit cognitive biases that systematically distort human judgment, including escalation of commitment, where decision-makers continue investing in failing courses of action due to prior investment. Understanding when LLMs exhibit such biases presents a unique challenge. While these biases are well-documented in humans, it remains unclear whether they manifest consistently in LLMs or require specific triggering conditions. This paper investigates this question using a two-stage investment task across four experimental conditions: model as investor, model as advisor, multi-agent deliberation, and compound pressure scenario. Across N = 6,500 trials, we find that bias manifestation in LLMs is highly context-dependent. In individual decision-making contexts (Studies 1-2, N = 4,000), LLMs demonstrate strong rational cost-benefit logic with minimal escalation of commitment. However, multi-agent deliberation reveals a striking hierarchy effect (Study 3, N = 500): while asymmetrical hierarchies show moderate escalation rates (46.2%), symmetrical peer-based decision-making produces near-universal escalation (99.2%). Similarly, when subjected to compound organizational and personal pressures (Study 4, N = 2,000), models exhibit high degrees of escalation of commitment (68.95% average allocation to failing divisions). These findings reveal that LLM bias manifestation depends critically on social and organizational context rather than being inherent, with significant implications for the deployment of multi-agent systems and unsupervised operations where such conditions may emerge naturally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09994v2">TIME: Temporal-Sensitive Multi-Dimensional Instruction Tuning and Robust Benchmarking for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Video large language models have achieved remarkable performance in tasks such as video question answering, however, their temporal understanding remains suboptimal. To address this limitation, we curate a dedicated instruction fine-tuning dataset that focuses on enhancing temporal comprehension across five key dimensions. In order to reduce reliance on costly temporal annotations, we introduce a multi-task prompt fine-tuning approach that seamlessly integrates temporal-sensitive tasks into existing instruction datasets without requiring additional annotations. Furthermore, we develop a novel benchmark for temporal-sensitive video understanding that not only fills the gaps in dimension coverage left by existing benchmarks but also rigorously filters out potential shortcuts, ensuring a more accurate evaluation. Extensive experimental results demonstrate that our approach significantly enhances the temporal understanding of video-LLMs while avoiding reliance on shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05421v1">LLM-based Multi-Agent Copilot for Quantum Sensor</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages,4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLM) exhibit broad utility but face limitations in quantum sensor development, stemming from interdisciplinary knowledge barriers and involving complex optimization processes. Here we present QCopilot, an LLM-based multi-agent framework integrating external knowledge access, active learning, and uncertainty quantification for quantum sensor design and diagnosis. Comprising commercial LLMs with few-shot prompt engineering and vector knowledge base, QCopilot employs specialized agents to adaptively select optimization methods, automate modeling analysis, and independently perform problem diagnosis. Applying QCopilot to atom cooling experiments, we generated 10${}^{\rm{8}}$ sub-$\rm{\mu}$K atoms without any human intervention within a few hours, representing $\sim$100$\times$ speedup over manual experimentation. Notably, by continuously accumulating prior knowledge and enabling dynamic modeling, QCopilot can autonomously identify anomalous parameters in multi-parameter experimental settings. Our work reduces barriers to large-scale quantum sensor deployment and readily extends to other quantum information systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05370v1">Simulating LLM training workloads for heterogeneous compute and network infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The growing demand for large-scale GPU clusters in distributed model training presents a significant barrier to innovation, particularly in model optimization, performance tuning, and system-level enhancements. To address this challenge, LLM training simulators are employed to estimate training time and guide design decisions. However, the state-of-the-art LLM training simulators assume homogeneous compute and network infrastructure. In practice, device heterogeneity is inevitable due to resource sharing in cloud environments, frequent shifts in device generations, and inherent intra-chip interconnect heterogeneity. To address the gap between state-of-the-art and practical requirements, we propose the design of a heterogeneity-aware distributed LLM simulator capable of predicting training time while enabling abstractions to specify custom configurations for device groups and device-to-parallelism mapping. We present the design requirements and challenges in building a heterogeneity-aware distributed ML training simulator, and design components such as non-uniform workload partitioning. Our initial simulation results demonstrate the impact of heterogeneity on the model computation and communication time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05344v1">NomicLaw: Emergent Trust and Strategic Argumentation in LLMs During Collaborative Law-Making</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have extended their capabilities from basic text processing to complex reasoning tasks, including legal interpretation, argumentation, and strategic interaction. However, empirical understanding of LLM behavior in open-ended, multi-agent settings especially those involving deliberation over legal and ethical dilemmas remains limited. We introduce NomicLaw, a structured multi-agent simulation where LLMs engage in collaborative law-making, responding to complex legal vignettes by proposing rules, justifying them, and voting on peer proposals. We quantitatively measure trust and reciprocity via voting patterns and qualitatively assess how agents use strategic language to justify proposals and influence outcomes. Experiments involving homogeneous and heterogeneous LLM groups demonstrate how agents spontaneously form alliances, betray trust, and adapt their rhetoric to shape collective decisions. Our results highlight the latent social reasoning and persuasive capabilities of ten open-source LLMs and provide insights into the design of future AI systems capable of autonomous negotiation, coordination and drafting legislation in legal settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v3">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 26 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Effective agent performance relies on the ability to compose tools and agents into effective workflows. However, progress in Large Language Model (LLM) planning and reasoning is limited by the scarcity of scalable, reliable evaluation data. This study addresses this limitation by identifying a suitable workflow domain for LLM application. I introduce NL2Flow, a fully automated system for parametrically generating planning problems, which are expressed in natural language, a structured intermediate representation, and formal PDDL, and rigorously evaluating the quality of generated plans. NL2Flow generates a dataset of 2296 low-difficulty problems in automated workflow generation and evaluates multiple open-sourced, instruct-tuned LLMs without task-specific optimization or architectural modifications. Results reveal that the highest performing model achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible plans. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. To investigate the potential of LLMs as natural language-to-JSON translators for workflow definition, and to facilitate integration with downstream symbolic computation tools and a symbolic planner, I evaluated the LLM's translation performance on natural language workflow descriptions. I observed that translating natural language into a JSON representation of a workflow problem yielded a lower success rate than generating a plan directly, suggesting that unnecessary decomposition of the reasoning task may degrade performance and highlighting the benefit of models capable of reasoning directly from natural language to action. As LLM reasoning scales to increasingly complex problems, understanding the shifting bottlenecks and sources of error within these systems will be crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v5">Nexus:Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05311v1">A Novel Architecture for Symbolic Reasoning with Decision Trees and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      We propose a hybrid architecture that integrates decision tree-based symbolic reasoning with the generative capabilities of large language models (LLMs) within a coordinated multi-agent framework. Unlike prior approaches that loosely couple symbolic and neural modules, our design embeds decision trees and random forests as callable oracles within a unified reasoning system. Tree-based modules enable interpretable rule inference and causal logic, while LLM agents handle abductive reasoning, generalization, and interactive planning. A central orchestrator maintains belief state consistency and mediates communication across agents and external tools, enabling reasoning over both structured and unstructured inputs. The system achieves strong performance on reasoning benchmarks. On \textit{ProofWriter}, it improves entailment consistency by +7.2\% through logic-grounded tree validation. On GSM8k, it achieves +5.3\% accuracy gains in multistep mathematical problems via symbolic augmentation. On \textit{ARC}, it boosts abstraction accuracy by +6.0\% through integration of symbolic oracles. Applications in clinical decision support and scientific discovery show how the system encodes domain rules symbolically while leveraging LLMs for contextual inference and hypothesis generation. This architecture offers a robust, interpretable, and extensible solution for general-purpose neuro-symbolic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05299v1">VS-LLM: Visual-Semantic Depression Assessment based on LLM for Drawing Projection Test</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The Drawing Projection Test (DPT) is an essential tool in art therapy, allowing psychologists to assess participants' mental states through their sketches. Specifically, through sketches with the theme of "a person picking an apple from a tree (PPAT)", it can be revealed whether the participants are in mental states such as depression. Compared with scales, the DPT can enrich psychologists' understanding of an individual's mental state. However, the interpretation of the PPAT is laborious and depends on the experience of the psychologists. To address this issue, we propose an effective identification method to support psychologists in conducting a large-scale automatic DPT. Unlike traditional sketch recognition, DPT more focus on the overall evaluation of the sketches, such as color usage and space utilization. Moreover, PPAT imposes a time limit and prohibits verbal reminders, resulting in low drawing accuracy and a lack of detailed depiction. To address these challenges, we propose the following efforts: (1) Providing an experimental environment for automated analysis of PPAT sketches for depression assessment; (2) Offering a Visual-Semantic depression assessment based on LLM (VS-LLM) method; (3) Experimental results demonstrate that our method improves by 17.6% compared to the psychologist assessment method. We anticipate that this work will contribute to the research in mental state assessment based on PPAT sketches' elements recognition. Our datasets and codes are available at https://github.com/wmeiqi/VS-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05298v1">GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 17 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      We present GhostShell, a novel approach that leverages Large Language Models (LLMs) to enable streaming and concurrent behavioral programming for embodied systems. In contrast to conventional methods that rely on pre-scheduled action sequences or behavior trees, GhostShell drives embodied systems to act on-the-fly by issuing function calls incrementally as tokens are streamed from the LLM. GhostShell features a streaming XML function token parser, a dynamic function interface mapper, and a multi-channel scheduler that orchestrates intra-channel synchronous and inter-channel asynchronous function calls, thereby coordinating serial-parallel embodied actions across multiple robotic components as directed by the LLM. We evaluate GhostShell on our robot prototype COCO through comprehensive grounded experiments across 34 real-world interaction tasks and multiple LLMs. The results demonstrate that our approach achieves state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4 Sonnet and up to 66X faster response times compared to LLM native function calling APIs. GhostShell also proves effective in long-horizon multimodal tasks, demonstrating strong robustness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05289v1">RLHF Fine-Tuning of LLMs for Alignment with Implicit User Feedback in Conversational Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Conversational recommender systems (CRS) based on Large Language Models (LLMs) need to constantly be aligned to the user preferences to provide satisfying and context-relevant item recommendations. The traditional supervised fine-tuning cannot capture the implicit feedback signal, e.g., dwell time, sentiment polarity, or engagement patterns. In this paper, we share a fine-tuning solution using human feedback reinforcement learning (RLHF) to maximize implied user feedback (IUF) in a multi-turn recommendation context. We specify a reward model $R_{\phi}$ learnt on weakly-labelled engagement information and maximize user-centric utility by optimizing the foundational LLM M_{\theta} through a proximal policy optimization (PPO) approach. The architecture models conversational state transitions $s_t \to a_t \to s_{t +1}$, where the action $a_t$ is associated with LLM-generated item suggestions only on condition of conversation history in the past. The evaluation across synthetic and real-world datasets (e.g.REDIAL, OpenDialKG) demonstrates that our RLHF-fine-tuned models can perform better in terms of top-$k$ recommendation accuracy, coherence, and user satisfaction compared to (arrow-zero-cmwrquca-teja-falset ensuite 2Round group-deca States penalty give up This paper shows that implicit signal alignment can be efficient in achieving scalable and user-adaptive design of CRS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05282v1">ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05266v1">Understanding and Mitigating Errors of LLM-Generated RTL Code</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 14 pages, 26 figures
    </div>
    <details class="paper-abstract">
      Despite the promising potential of large language model (LLM) based register-transfer-level (RTL) code generation, the overall success rate remains unsatisfactory. Errors arise from various factors, with limited understanding of specific failure causes hindering improvement. To address this, we conduct a comprehensive error analysis and manual categorization. Our findings reveal that most errors stem not from LLM reasoning limitations, but from insufficient RTL programming knowledge, poor understanding of circuit concepts, ambiguous design descriptions, or misinterpretation of complex multimodal inputs. Leveraging in-context learning, we propose targeted error correction techniques. Specifically, we construct a domain-specific knowledge base and employ retrieval-augmented generation (RAG) to supply necessary RTL knowledge. To mitigate ambiguity errors, we introduce design description rules and implement a rule-checking mechanism. For multimodal misinterpretation, we integrate external tools to convert inputs into LLM-compatible meta-formats. For remaining errors, we adopt an iterative debugging loop (simulation-error localization-correction). Integrating these techniques into an LLM-based framework significantly improves performance. We incorporate these error correction techniques into a foundational LLM-based RTL code generation framework, resulting in significantly improved performance. Experimental results show that our enhanced framework achieves 91.0\% accuracy on the VerilogEval benchmark, surpassing the baseline code generation approach by 32.7\%, demonstrating the effectiveness of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05257v1">MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The Mixture-of-Experts (MoE) architecture has become a predominant paradigm for scaling large language models (LLMs). Despite offering strong performance and computational efficiency, large MoE-based LLMs like DeepSeek-V3-0324 and Kimi-K2-Instruct present serious challenges due to substantial memory requirements in deployment. While recent works have explored MoE compression to address this issue, existing methods often suffer from considerable accuracy drops (e.g., 7-14% relatively) even at modest compression rates. This paper introduces a novel Mixture-of-Basis-Experts (MoBE) method that achieves model compression while incurring minimal accuracy drops. Specifically, each up/gate matrix in an expert is decomposed via a rank decomposition as W = AB, where matrix A is unique to each expert. The relatively larger matrix B is further re-parameterized as a linear combination of basis matrices {Bi} shared across all experts within a given MoE layer. The factorization is learned by minimizing the reconstruction error relative to the original weight matrices. Experiments demonstrate that MoBE achieves notably lower accuracy drops compared to prior works. For instance, MoBE can reduce the parameter counts of Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B) and Kimi-K2-Instruct (1T) by 24%-30% with only 1%-2% accuracy drop (about 2% drops when measured relatively).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05242v1">CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Technical report. Project page: https://github.com/sijieaaa/CodeBoost
    </div>
    <details class="paper-abstract">
      Code large language models (LLMs) have become indispensable tools for building efficient and automated coding pipelines. Existing models are typically post-trained using reinforcement learning (RL) from general-purpose LLMs using "human instruction-final answer" pairs, where the instructions are usually from manual annotations. However, collecting high-quality coding instructions is both labor-intensive and difficult to scale. On the other hand, code snippets are abundantly available from various sources. This imbalance presents a major bottleneck in instruction-based post-training. We propose CodeBoost, a post-training framework that enhances code LLMs purely from code snippets, without relying on human-annotated instructions. CodeBoost introduces the following key components: (1) maximum-clique curation, which selects a representative and diverse training corpus from code; (2) bi-directional prediction, which enables the model to learn from both forward and backward prediction objectives; (3) error-aware prediction, which incorporates learning signals from both correct and incorrect outputs; (4) heterogeneous augmentation, which diversifies the training distribution to enrich code semantics; and (5) heterogeneous rewarding, which guides model learning through multiple reward types including format correctness and execution feedback from both successes and failures. Extensive experiments across several code LLMs and benchmarks verify that CodeBoost consistently improves performance, demonstrating its effectiveness as a scalable and effective training pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05232v1">Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Traditional parameter-efficient fine-tuning (PEFT) methods such as LoRA are tightly coupled with the base model architecture, which constrains their applicability across heterogeneous pretrained large language models (LLMs). To address this limitation, we introduce Cross-LoRA, a data-free framework for transferring LoRA modules between diverse base models without requiring additional training data. Cross-LoRA consists of two key components: (a) LoRA-Align, which performs subspace alignment between source and target base models through rank-truncated singular value decomposition (SVD) and Frobenius-optimal linear transformation, ensuring compatibility under dimension mismatch; and (b) LoRA-Shift, which applies the aligned subspaces to project source LoRA weight updates into the target model parameter space. Both components are data-free, training-free, and enable lightweight adaptation on a commodity GPU in 20 minutes. Experiments on ARCs, OBOA and HellaSwag show that Cross-LoRA achieves relative gains of up to 5.26% over base models. Across other commonsense reasoning benchmarks, Cross-LoRA maintains performance comparable to that of directly trained LoRA adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14964v2">Efficient Knowledge Injection in LLMs via Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      In many practical applications, large language models (LLMs) need to acquire new knowledge not present in their pre-training data. Efficiently leveraging this knowledge usually relies on supervised fine-tuning or retrieval-augmented generation (RAG). Although RAG has emerged as the industry standard for knowledge injection, fine-tuning has not yet achieved comparable success. This paper proposes utilizing prompt distillation, a self-distillation-based method previously explored primarily for style alignment and instruction tuning, to internalize new factual knowledge from free-form documents. Unlike prior methods, our approach requires neither larger teacher models nor structured knowledge formats. Across multiple LLM sizes and model families, we show that prompt distillation outperforms standard supervised fine-tuning and can even surpass RAG. We analyze the key factors contributing to prompt distillation's effectiveness and examine how it scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05165v1">Aligning LLMs on a Budget: Inference-Time Alignment with Heuristic Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Aligning LLMs with user preferences is crucial for real-world use but often requires costly fine-tuning or expensive inference, forcing trade-offs between alignment quality and computational cost. Existing inference-time methods typically ignore this balance, focusing solely on the optimized policy's performance. We propose HIA (Heuristic-Guided Inference-time Alignment), a tuning-free, black-box-compatible approach that uses a lightweight prompt optimizer, heuristic reward models, and two-stage filtering to reduce inference calls while preserving alignment quality. On real-world prompt datasets, HelpSteer and ComPRed, HIA outperforms best-of-N sampling, beam search, and greedy search baselines in multi-objective, goal-conditioned tasks under the same inference budget. We also find that HIA is effective under low-inference budgets with as little as one or two response queries, offering a practical solution for scalable, personalized LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20367v5">Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLM), reinforcement learning (RL) has emerged as a pivotal technique for code generation and optimization in various domains. This paper presents a systematic survey of the application of RL in code optimization and generation, highlighting its role in enhancing compiler optimization, resource allocation, and the development of frameworks and tools. Subsequent sections first delve into the intricate processes of compiler optimization, where RL algorithms are leveraged to improve efficiency and resource utilization. The discussion then progresses to the function of RL in resource allocation, emphasizing register allocation and system optimization. We also explore the burgeoning role of frameworks and tools in code generation, examining how RL can be integrated to bolster their capabilities. This survey aims to serve as a comprehensive resource for researchers and practitioners interested in harnessing the power of RL to advance code generation and optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05149v1">Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted at Interspeech 2025. 5 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in handling spoken inputs for high-resource languages, reaching state-of-the-art performance in various tasks. However, their applicability is still less explored in low-resource settings. This work investigates the use of Speech LLMs for low-resource Automatic Speech Recognition using the SLAM-ASR framework, where a trainable lightweight projector connects a speech encoder and a LLM. Firstly, we assess training data volume requirements to match Whisper-only performance, re-emphasizing the challenges of limited data. Secondly, we show that leveraging mono- or multilingual projectors pretrained on high-resource languages reduces the impact of data scarcity, especially with small training sets. Using multilingual LLMs (EuroLLM, Salamandra) with whisper-large-v3-turbo, we evaluate performance on several public benchmarks, providing insights for future research on optimizing Speech LLMs for low-resource languages and multilinguality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05129v1">Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      With the rapid and continuous increase in academic publications, identifying high-quality research has become an increasingly pressing challenge. While recent methods leveraging Large Language Models (LLMs) for automated paper evaluation have shown great promise, they are often constrained by outdated domain knowledge and limited reasoning capabilities. In this work, we present PaperEval, a novel LLM-based framework for automated paper evaluation that addresses these limitations through two key components: 1) a domain-aware paper retrieval module that retrieves relevant concurrent work to support contextualized assessments of novelty and contributions, and 2) a latent reasoning mechanism that enables deep understanding of complex motivations and methodologies, along with comprehensive comparison against concurrently related work, to support more accurate and reliable evaluation. To guide the reasoning process, we introduce a progressive ranking optimization strategy that encourages the LLM to iteratively refine its predictions with an emphasis on relative comparison. Experiments on two datasets demonstrate that PaperEval consistently outperforms existing methods in both academic impact and paper quality evaluation. In addition, we deploy PaperEval in a real-world paper recommendation system for filtering high-quality papers, which has gained strong engagement on social media -- amassing over 8,000 subscribers and attracting over 10,000 views for many filtered high-quality papers -- demonstrating the practical effectiveness of PaperEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05113v1">EasySize: Elastic Analog Circuit Sizing via LLM-Guided Heuristic Search</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Analog circuit design is a time-consuming, experience-driven task in chip development. Despite advances in AI, developing universal, fast, and stable gate sizing methods for analog circuits remains a significant challenge. Recent approaches combine Large Language Models (LLMs) with heuristic search techniques to enhance generalizability, but they often depend on large model sizes and lack portability across different technology nodes. To overcome these limitations, we propose EasySize, the first lightweight gate sizing framework based on a finetuned Qwen3-8B model, designed for universal applicability across process nodes, design specifications, and circuit topologies. EasySize exploits the varying Ease of Attainability (EOA) of performance metrics to dynamically construct task-specific loss functions, enabling efficient heuristic search through global Differential Evolution (DE) and local Particle Swarm Optimization (PSO) within a feedback-enhanced flow. Although finetuned solely on 350nm node data, EasySize achieves strong performance on 5 operational amplifier (Op-Amp) netlists across 180nm, 45nm, and 22nm technology nodes without additional targeted training, and outperforms AutoCkt, a widely-used Reinforcement Learning based sizing framework, on 86.67\% of tasks with more than 96.67\% of simulation resources reduction. We argue that EasySize can significantly reduce the reliance on human expertise and computational resources in gate sizing, thereby accelerating and simplifying the analog circuit design process. EasySize will be open-sourced at a later date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v3">LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 11 pages, 7 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03864v2">DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01674v2">CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to COLM 2025. Project Website: https://cupid.kixlab.org/
    </div>
    <details class="paper-abstract">
      Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06518v2">Medal Matters: Probing LLMs' Failure Cases Through Olympic Rankings</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 COLM 2025 ORIGen Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, yet their internal knowledge structures remain poorly understood. This study examines these structures through the lens of historical Olympic medal tallies, evaluating LLMs on two tasks: (1) retrieving medal counts for specific teams and (2) identifying rankings of each team. While state-of-the-art LLMs excel in recalling medal counts, they struggle with providing rankings, highlighting a key difference between their knowledge organization and human reasoning. These findings shed light on the limitations of LLMs' internal knowledge integration and suggest directions for improvement. To facilitate further research, we release our code, dataset, and model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05028v1">Evaluation of LLMs in AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 27 pages, 32 figures
    </div>
    <details class="paper-abstract">
      Meaning Representation (AMR) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17963v3">M$^{2}$Chat: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      While current LLM chatbots like GPT-4V bridge the gap between human instructions and visual representations to enable text-image generations, they still lack efficient alignment methods for high-fidelity performance on multiple downstream tasks. In this paper, we propose \textbf{$M^{2}Chat$}, a novel unified multimodal LLM framework for generating interleaved text-image conversation across various scenarios. Specifically, we propose an $M^{3}Adapter$ that efficiently integrates granular low-level visual information and high-level semantic features from multi-modality prompts. Upon the well-aligned fused feature, $M^{3}Adapter$ tailors a learnable gating strategy to balance the model creativity and consistency across various tasks adaptively. Moreover, to further enhance the effectiveness of $M^{3}Adapter$ while preserving the coherence of semantic context comprehension, we introduce a two-stage $M^{3}FT$ fine-tuning strategy. This strategy optimizes disjoint groups of parameters for image-text alignment and visual-instruction respectively. Extensive experiments demonstrate our $M^{2}Chat$ surpasses state-of-the-art counterparts across diverse benchmarks, showcasing its prowess in interleaving generation, storytelling, and multimodal dialogue systems. The demo and code are available at \red{https://mattie-e.github.io/M2Chat.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05012v1">Making Prompts First-Class Citizens for Adaptive LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Modern LLM pipelines increasingly resemble data-centric systems: they retrieve external context, compose intermediate outputs, validate results, and adapt based on runtime feedback. Yet, the central element guiding this process -- the prompt -- remains a brittle, opaque string, disconnected from the surrounding dataflow. This disconnect limits reuse, optimization, and runtime control. In this paper, we describe our vision and an initial design for SPEAR, a language and runtime that fills this prompt management gap by making prompts structured, adaptive, and first-class components of the execution model. SPEAR enables (1) runtime prompt refinement -- modifying prompts dynamically in response to execution-time signals such as confidence, latency, or missing context; and (2) structured prompt management -- organizing prompt fragments into versioned views with support for introspection and logging. SPEAR defines a prompt algebra that governs how prompts are constructed and adapted within a pipeline. It supports multiple refinement modes (manual, assisted, and automatic), giving developers a balance between control and automation. By treating prompt logic as structured data, SPEAR enables optimizations such as operator fusion, prefix caching, and view reuse. Preliminary experiments quantify the behavior of different refinement modes compared to static prompts and agentic retries, as well as the impact of prompt-level optimizations such as operator fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05004v1">R-Zero: Self-Evolving Reasoning LLM from Zero Data</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05853v2">"Mango Mango, How to Let The Lettuce Dry Without A Spinner?": Exploring User Perceptions of Using An LLM-Based Conversational Assistant Toward Cooking Partner</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 To appear at CSCW 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has created numerous potentials for integration with conversational assistants (CAs) assisting people in their daily tasks, particularly due to their extensive flexibility. However, users' real-world experiences interacting with these assistants remain unexplored. In this research, we chose cooking, a complex daily task, as a scenario to explore people's successful and unsatisfactory experiences while receiving assistance from an LLM-based CA, Mango Mango. We discovered that participants value the system's ability to offer customized instructions based on context, provide extensive information beyond the recipe, and assist them in dynamic task planning. However, users expect the system to be more adaptive to oral conversation and provide more suggestive responses to keep them actively involved. Recognizing that users began treating our LLM-CA as a personal assistant or even a partner rather than just a recipe-reading tool, we propose five design considerations for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04975v1">Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Traditionally, traders and quantitative analysts address alpha decay by manually crafting formulaic alphas, mathematical expressions that identify patterns or signals in financial data, through domain expertise and trial-and-error. This process is often time-consuming and difficult to scale. With recent advances in large language models (LLMs), it is now possible to automate the generation of such alphas by leveraging the reasoning capabilities of LLMs. This paper introduces a novel framework that integrates a prompt-based LLM with a Transformer model for stock price prediction. The LLM first generates diverse and adaptive alphas using structured inputs such as historical stock features (Close, Open, High, Low, Volume), technical indicators, sentiment scores of both target and related companies. These alphas, instead of being used directly for trading, are treated as high-level features that capture complex dependencies within the financial data. To evaluate the effectiveness of these LLM-generated formulaic alphas, the alpha features are then fed into prediction models such as Transformer, LSTM, TCN, SVR, and Random Forest to forecast future stock prices. Experimental results demonstrate that the LLM-generated alphas significantly improve predictive accuracy. Moreover, the accompanying natural language reasoning provided by the LLM enhances the interpretability and transparency of the predictions, supporting more informed financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13213v4">Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      We study 15 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigorous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07132v3">Interactive Data Harmonization with LLM Agents: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Data harmonization is an essential task that entails integrating datasets from diverse sources. Despite years of research in this area, it remains a time-consuming and challenging task due to schema mismatches, varying terminologies, and differences in data collection methodologies. This paper presents the case for agentic data harmonization as a means to both empower experts to harmonize their data and to streamline the process. We introduce Harmonia, a system that combines LLM-based reasoning, an interactive user interface, and a library of data harmonization primitives to automate the synthesis of data harmonization pipelines. We demonstrate Harmonia in a clinical data harmonization scenario, where it helps to interactively create reusable pipelines that map datasets to a standard format. Finally, we discuss challenges and open problems, and suggest research directions for advancing our vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v3">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04030v2">OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05835v1">NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to Interspeech 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced audio processing by leveraging audio codecs to discretize audio into tokens, enabling the application of language modeling techniques to speech data. However, existing audio codecs often operate at high frame rates, leading to slow training and inference, particularly for autoregressive models. To address this, there is growing interest in low frame-rate audio codecs, which reduce the number of autoregressive steps required to generate one second of audio. In this paper, we conduct ablation studies to examine the impact of frame rate, bitrate, and causality on codec reconstruction quality. Based on our findings, we introduce NanoCodec, a state-of-the-art audio codec that achieves high-quality compression at just 12.5 frames per second (FPS). NanoCodec outperforms related works across various bitrate ranges, establishing a new benchmark for low-latency and efficient Speech LLM training and inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17008v2">Benchmarking LLMs on the Semantic Overlap Summarization Task</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Semantic Overlap Summarization (SOS) is a constrained multi-document summarization task, where the constraint is to capture the common/overlapping information between two alternative narratives. In this work, we perform a benchmarking study of popular Large Language Models (LLMs) exclusively on the SOS task. Additionally, we introduce the PrivacyPolicyPairs (3P) dataset to expand the space of SOS benchmarks in terms of quantity and variety. This dataset provides 135 high-quality SOS data samples sourced from privacy policy documents. We then use a standard prompting taxonomy called TELeR to create and evaluate 905,216 distinct LLM-generated summaries over two SOS datasets from different domains, and we further conduct human evaluation on a subset of 540 samples. We conclude the paper by analyzing models' performances and the reliability of automatic evaluation. The code and datasets used to conduct this study are available at https://anonymous.4open.science/r/llm_eval-E16D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05728v1">CLAPP: The CLASS LLM Agent for Pair Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Code: https://github.com/santiagocasas/clapp, Streamlit app: https://classclapp.streamlit.app
    </div>
    <details class="paper-abstract">
      We introduce CLAPP (CLASS LLM Agent for Pair Programming), an interactive AI assistant designed to support researchers working with the Einstein-Boltzmann solver CLASS. CLAPP leverages large language models (LLMs) and domain-specific retrieval to provide conversational coding support for CLASS-answering questions, generating code, debugging errors, and producing plots. Its architecture combines multi-agent LLM orchestration, semantic search across CLASS documentation, and a live Python execution environment. Deployed as a user-friendly web application, CLAPP lowers the entry barrier for scientists unfamiliar with AI tools and enables more productive human-AI collaboration in computational and numerical cosmology. The app is available at https://classclapp.streamlit.app
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05702v1">Semantic Reasoning Meets Numerical Precision: An LLM-Powered Multi-Agent System for Power Grid Control</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06577v1">Leveraging LLMs for Privacy-Aware Predictions in Participatory Budgeting</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Participatory Budgeting (PB) empowers citizens to propose and vote on public investment projects. Yet, despite its democratic potential, PB initiatives often suffer from low participation rates, limiting their visibility and perceived legitimacy. In this work, we aim to strengthen PB elections in two key ways: by supporting project proposers in crafting better proposals, and by helping PB organizers manage large volumes of submissions in a transparent manner. We propose a privacy-preserving approach to predict which PB proposals are likely to be funded, using only their textual descriptions and anonymous historical voting records -- without relying on voter demographics or personally identifiable information. We evaluate the performance of GPT 4 Turbo in forecasting proposal outcomes across varying contextual scenarios, observing that the LLM's prior knowledge needs to be complemented by past voting data to obtain predictions reflecting real-world PB voting behavior. Our findings highlight the potential of AI-driven tools to support PB processes by improving transparency, planning efficiency, and civic engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05625v1">How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have started to demonstrate the ability to persuade humans, yet our understanding of how this dynamic transpires is limited. Recent work has used linear probes, lightweight tools for analyzing model representations, to study various LLM skills such as the ability to model user sentiment and political perspective. Motivated by this, we apply probes to study persuasion dynamics in natural, multi-turn conversations. We leverage insights from cognitive science to train probes on distinct aspects of persuasion: persuasion success, persuadee personality, and persuasion strategy. Despite their simplicity, we show that they capture various aspects of persuasion at both the sample and dataset levels. For instance, probes can identify the point in a conversation where the persuadee was persuaded or where persuasive success generally occurs across the entire dataset. We also show that in addition to being faster than expensive prompting-based approaches, probes can do just as well and even outperform prompting in some settings, such as when uncovering persuasion strategy. This suggests probes as a plausible avenue for studying other complex behaviours such as deception and manipulation, especially in multi-turn settings and large-scale dataset analysis where prompting-based methods would be computationally inefficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05622v1">Simulating Human-Like Learning Dynamics with LLM-Empowered Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Capturing human learning behavior based on deep learning methods has become a major research focus in both psychology and intelligent systems. Recent approaches rely on controlled experiments or rule-based models to explore cognitive processes. However, they struggle to capture learning dynamics, track progress over time, or provide explainability. To address these challenges, we introduce LearnerAgent, a novel multi-agent framework based on Large Language Models (LLMs) to simulate a realistic teaching environment. To explore human-like learning dynamics, we construct learners with psychologically grounded profiles-such as Deep, Surface, and Lazy-as well as a persona-free General Learner to inspect the base LLM's default behavior. Through weekly knowledge acquisition, monthly strategic choices, periodic tests, and peer interaction, we can track the dynamic learning progress of individual learners over a full-year journey. Our findings are fourfold: 1) Longitudinal analysis reveals that only Deep Learner achieves sustained cognitive growth. Our specially designed "trap questions" effectively diagnose Surface Learner's shallow knowledge. 2) The behavioral and cognitive patterns of distinct learners align closely with their psychological profiles. 3) Learners' self-concept scores evolve realistically, with the General Learner developing surprisingly high self-efficacy despite its cognitive limitations. 4) Critically, the default profile of base LLM is a "diligent but brittle Surface Learner"-an agent that mimics the behaviors of a good student but lacks true, generalizable understanding. Extensive simulation experiments demonstrate that LearnerAgent aligns well with real scenarios, yielding more insightful findings about LLMs' behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16086v2">Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04903v1">RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Multi-agent large language model (LLM) systems have shown strong potential in complex reasoning and collaborative decision-making tasks. However, most existing coordination schemes rely on static or full-context routing strategies, which lead to excessive token consumption, redundant memory exposure, and limited adaptability across interaction rounds. We introduce RCR-Router, a modular and role-aware context routing framework designed to enable efficient, adaptive collaboration in multi-agent LLMs. To our knowledge, this is the first routing approach that dynamically selects semantically relevant memory subsets for each agent based on its role and task stage, while adhering to a strict token budget. A lightweight scoring policy guides memory selection, and agent outputs are iteratively integrated into a shared memory store to facilitate progressive context refinement. To better evaluate model behavior, we further propose an Answer Quality Score metric that captures LLM-generated explanations beyond standard QA accuracy. Experiments on three multi-hop QA benchmarks -- HotPotQA, MuSiQue, and 2WikiMultihop -- demonstrate that RCR-Router reduces token usage (up to 30%) while improving or maintaining answer quality. These results highlight the importance of structured memory routing and output-aware evaluation in advancing scalable multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13417v3">RLTHF: Targeted Human Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Presented at ICML 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04894v1">Adversarial Attacks and Defenses on Graph-aware Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated with graph-structured data for tasks like node classification, a domain traditionally dominated by Graph Neural Networks (GNNs). While this integration leverages rich relational information to improve task performance, their robustness against adversarial attacks remains unexplored. We take the first step to explore the vulnerabilities of graph-aware LLMs by leveraging existing adversarial attack methods tailored for graph-based models, including those for poisoning (training-time attacks) and evasion (test-time attacks), on two representative models, LLAGA (Chen et al. 2024) and GRAPHPROMPTER (Liu et al. 2024). Additionally, we discover a new attack surface for LLAGA where an attacker can inject malicious nodes as placeholders into the node sequence template to severely degrade its performance. Our systematic analysis reveals that certain design choices in graph encoding can enhance attack success, with specific findings that: (1) the node sequence template in LLAGA increases its vulnerability; (2) the GNN encoder used in GRAPHPROMPTER demonstrates greater robustness; and (3) both approaches remain susceptible to imperceptible feature perturbation attacks. Finally, we propose an end-to-end defense framework GALGUARD, that combines an LLM-based feature correction module to mitigate feature-level perturbations and adapted GNN defenses to protect against structural attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v2">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 18 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04842v1">Charts-of-Thought: Enhancing LLM Visualization Literacy Through Structured Data Extraction</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 11 pages, 8 figures. Accepted at IEEE VIS: Visualization & Visual Analytics 2025 conference, November 2-7, 2025, Vienna, Austria
    </div>
    <details class="paper-abstract">
      This paper evaluates the visualization literacy of modern Large Language Models (LLMs) and introduces a novel prompting technique called Charts-of-Thought. We tested three state-of-the-art LLMs (Claude-3.7-sonnet, GPT-4.5 preview, and Gemini-2.0-pro) on the Visualization Literacy Assessment Test (VLAT) using standard prompts and our structured approach. The Charts-of-Thought method guides LLMs through a systematic data extraction, verification, and analysis process before answering visualization questions. Our results show Claude-3.7-sonnet achieved a score of 50.17 using this method, far exceeding the human baseline of 28.82. This approach improved performance across all models, with score increases of 21.8% for GPT-4.5, 9.4% for Gemini-2.0, and 13.5% for Claude-3.7 compared to standard prompting. The performance gains were consistent across original and modified VLAT charts, with Claude correctly answering 100% of questions for several chart types that previously challenged LLMs. Our study reveals that modern multimodal LLMs can surpass human performance on visualization literacy tasks when given the proper analytical framework. These findings establish a new benchmark for LLM visualization literacy and demonstrate the importance of structured prompting strategies for complex visual interpretation tasks. Beyond improving LLM visualization literacy, Charts-of-Thought could also enhance the accessibility of visualizations, potentially benefiting individuals with visual impairments or lower visualization literacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04826v1">Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large language models require consistent behavioral patterns for safe deployment, yet their personality-like traits remain poorly understood. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25+ open-source models (1B-671B parameters) across 500,000+ responses. Using traditional (BFI-44, SD3) and novel LLM-adapted personality instruments, we systematically vary question order, paraphrasing, personas, and reasoning modes. Our findings challenge fundamental deployment assumptions: (1) Even 400B+ models exhibit substantial response variability (SD > 0.4); (2) Minor prompt reordering alone shifts personality measurements by up to 20%; (3) Interventions expected to stabilize behavior, such as chain-of-thought reasoning, detailed personas instruction, inclusion of conversation history, can paradoxically increase variability; (4) LLM-adapted instruments show equal instability to human-centric versions, confirming architectural rather than translational limitations. This persistent instability across scales and mitigation strategies suggests current LLMs lack the foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that personality-based alignment strategies may be fundamentally inadequate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04820v1">Automated File-Level Logging Generation for Machine Learning Applications using LLMs: A Case Study using GPT-4o Mini</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Logging is essential in software development, helping developers monitor system behavior and aiding in debugging applications. Given the ability of large language models (LLMs) to generate natural language and code, researchers are exploring their potential to generate log statements. However, prior work focuses on evaluating logs introduced in code functions, leaving file-level log generation underexplored -- especially in machine learning (ML) applications, where comprehensive logging can enhance reliability. In this study, we evaluate the capacity of GPT-4o mini as a case study to generate log statements for ML projects at file level. We gathered a set of 171 ML repositories containing 4,073 Python files with at least one log statement. We identified and removed the original logs from the files, prompted the LLM to generate logs for them, and evaluated both the position of the logs and log level, variables, and text quality of the generated logs compared to human-written logs. In addition, we manually analyzed a representative sample of generated logs to identify common patterns and challenges. We find that the LLM introduces logs in the same place as humans in 63.91% of cases, but at the cost of a high overlogging rate of 82.66%. Furthermore, our manual analysis reveals challenges for file-level logging, which shows overlogging at the beginning or end of a function, difficulty logging within large code blocks, and misalignment with project-specific logging conventions. While the LLM shows promise for generating logs for complete files, these limitations remain to be addressed for practical implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04795v1">Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
    </div>
    <details class="paper-abstract">
      In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04787v1">Evaluating the Impact of LLM-guided Reflection on Learning Outcomes with Interactive AI-Generated Educational Podcasts</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to NCME Special Interest Group on AI in Measurement: AIME-CON 2025 conference
    </div>
    <details class="paper-abstract">
      This study examined whether embedding LLM-guided reflection prompts in an interactive AI-generated podcast improved learning and user experience compared to a version without prompts. Thirty-six undergraduates participated, and while learning outcomes were similar across conditions, reflection prompts reduced perceived attractiveness, highlighting a call for more research on reflective interactivity design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05694v1">DMFI: Dual-Modality Fine-Tuning and Inference Framework for LLM-Based Insider Threat Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Submitted to the 2025 IEEE International Conference on Data Mining (ICDM)
    </div>
    <details class="paper-abstract">
      Insider threat detection (ITD) poses a persistent and high-impact challenge in cybersecurity due to the subtle, long-term, and context-dependent nature of malicious insider behaviors. Traditional models often struggle to capture semantic intent and complex behavior dynamics, while existing LLM-based solutions face limitations in prompt adaptability and modality coverage. To bridge this gap, we propose DMFI, a dual-modality framework that integrates semantic inference with behavior-aware fine-tuning. DMFI converts raw logs into two structured views: (1) a semantic view that processes content-rich artifacts (e.g., emails, https) using instruction-formatted prompts; and (2) a behavioral abstraction, constructed via a 4W-guided (When-Where-What-Which) transformation to encode contextual action sequences. Two LoRA-enhanced LLMs are fine-tuned independently, and their outputs are fused via a lightweight MLP-based decision module. We further introduce DMFI-B, a discriminative adaptation strategy that separates normal and abnormal behavior representations, improving robustness under severe class imbalance. Experiments on CERT r4.2 and r5.2 datasets demonstrate that DMFI outperforms state-of-the-art methods in detection accuracy. Our approach combines the semantic reasoning power of LLMs with structured behavior modeling, offering a scalable and effective solution for real-world insider threat detection. Our work demonstrates the effectiveness of combining LLM reasoning with structured behavioral modeling, offering a scalable and deployable solution for modern insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05687v1">Risk Analysis Techniques for Governed LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Organisations are starting to adopt LLM-based AI agents, with their deployments naturally evolving from single agents towards interconnected, multi-agent networks. Yet a collection of safe agents does not guarantee a safe collection of agents, as interactions between agents over time create emergent behaviours and induce novel failure modes. This means multi-agent systems require a fundamentally different risk analysis approach than that used for a single agent. This report addresses the early stages of risk identification and analysis for multi-agent AI systems operating within governed environments where organisations control their agent configurations and deployment. In this setting, we examine six critical failure modes: cascading reliability failures, inter-agent communication failures, monoculture collapse, conformity bias, deficient theory of mind, and mixed motive dynamics. For each, we provide a toolkit for practitioners to extend or integrate into their existing frameworks to assess these failure modes within their organisational contexts. Given fundamental limitations in current LLM behavioural understanding, our approach centres on analysis validity, and advocates for progressively increasing validity through staged testing across stages of abstraction and deployment that gradually increases exposure to potential negative impacts, while collecting convergent evidence through simulation, observational analysis, benchmarking, and red teaming. This methodology establishes the groundwork for robust organisational risk management as these LLM-based multi-agent systems are deployed and operated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00873v2">Aligning Human and LLM Judgments: Insights from EvalAssist on Task-Specific Evaluations and AI-assisted Assessment Strategy Preferences</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Evaluation of large language model (LLM) outputs requires users to make critical judgments about the best outputs across various configurations. This process is costly and takes time given the large amounts of data. LLMs are increasingly used as evaluators to filter training data, evaluate model performance or assist human evaluators with detailed assessments. To support this process, effective front-end tools are critical for evaluation. Two common approaches for using LLMs as evaluators are direct assessment and pairwise comparison. In our study with machine learning practitioners (n=15), each completing 6 tasks yielding 131 evaluations, we explore how task-related factors and assessment strategies influence criteria refinement and user perceptions. Findings show that users performed more evaluations with direct assessment by making criteria task-specific, modifying judgments, and changing the evaluator model. We conclude with recommendations for how systems can better support interactions in LLM-assisted evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04676v1">GeRe: Towards Efficient Anti-Forgetting in Continual Learning of LLM via General Samples Replay</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      The continual learning capability of large language models (LLMs) is crucial for advancing artificial general intelligence. However, continual fine-tuning LLMs across various domains often suffers from catastrophic forgetting, characterized by: 1) significant forgetting of their general capabilities, and 2) sharp performance declines in previously learned tasks. To simultaneously address both issues in a simple yet stable manner, we propose General Sample Replay (GeRe), a framework that use usual pretraining texts for efficient anti-forgetting. Beyond revisiting the most prevalent replay-based practices under GeRe, we further leverage neural states to introduce a enhanced activation states constrained optimization method using threshold-based margin (TM) loss, which maintains activation state consistency during replay learning. We are the first to validate that a small, fixed set of pre-collected general replay samples is sufficient to resolve both concerns--retaining general capabilities while promoting overall performance across sequential tasks. Indeed, the former can inherently facilitate the latter. Through controlled experiments, we systematically compare TM with different replay strategies under the GeRe framework, including vanilla label fitting, logit imitation via KL divergence and feature imitation via L1/L2 losses. Results demonstrate that TM consistently improves performance and exhibits better robustness. Our work paves the way for efficient replay of LLMs for the future. Our code and data are available at https://github.com/Qznan/GeRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06362v2">Adaptive Audio-Visual Speech Recognition via Matryoshka-Based Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to IEEE ASRU 2025
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) leverages audio and visual modalities to improve robustness in noisy environments. Recent advances in Large Language Models (LLMs) show strong performance in speech recognition, including AVSR. However, the long speech representations lead to high computational costs for LLMs. Prior methods compress inputs before feeding them to LLMs, but high compression often harms accuracy. To address this, we propose Llama-MTSK, the first Matryoshka-based Multimodal LLM for AVSR, which flexibly adapts audio-visual token allocation under varying compute constraints. Inspired by Matryoshka Representation Learning, our model encodes representations at multiple granularities with a single architecture, avoiding the need for separate models. For efficient fine-tuning, we introduce three LoRA-based strategies using global and scale-specific modules. Evaluations on major AVSR datasets show Llama-MTSK matches or outperforms models trained at fixed compression levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04664v1">Sculptor: Empowering LLMs with Cognitive Agency via Active Context Management</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Preprint. Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from significant performance degradation when processing long contexts due to proactive interference, where irrelevant information in earlier parts of the context disrupts reasoning and memory recall. While most research focuses on external memory systems to augment LLMs' capabilities, we propose a complementary approach: empowering LLMs with Active Context Management (ACM) tools to actively sculpt their internal working memory. We introduce Sculptor, a framework that equips LLMs with three categories of tools: (1) context fragmentation, (2) summary, hide, and restore, and (3) intelligent search. Our approach enables LLMs to proactively manage their attention and working memory, analogous to how humans selectively focus on relevant information while filtering out distractions. Experimental evaluation on information-sparse benchmarks-PI-LLM (proactive interference) and NeedleBench Multi-Needle Reasoning-demonstrates that Sculptor significantly improves performance even without specific training, leveraging LLMs' inherent tool calling generalization capabilities. By enabling Active Context Management, Sculptor not only mitigates proactive interference but also provides a cognitive foundation for more reliable reasoning across diverse long-context tasks-highlighting that explicit context-control strategies, rather than merely larger token windows, are key to robustness at scale.
    </details>
</div>
