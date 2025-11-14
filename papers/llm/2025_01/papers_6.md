# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11876v2">Rescriber: Smaller-LLM-Powered User-Led Data Minimization for Navigating Privacy Trade-offs in LLM-Based Conversational Agent</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      The proliferation of LLM-based conversational agents has resulted in excessive disclosure of identifiable or sensitive information. However, existing technologies fail to offer perceptible control or account for users' personal preferences about privacy-utility tradeoffs due to the lack of user involvement. To bridge this gap, we designed, built, and evaluated Rescriber, a browser extension that supports user-led data minimization in LLM-based conversational agents by helping users detect and sanitize personal information in their prompts. Our studies (N=12) showed that Rescriber helped users reduce unnecessary disclosure and addressed their privacy concerns. Users' subjective perceptions of the system powered by Llama3-8B were on par with that by GPT-4o. The comprehensiveness and consistency of the detection and sanitization emerge as essential factors that affect users' trust and perceived protection. Our findings confirm the viability of smaller-LLM-powered, user-facing, on-device privacy controls, presenting a promising approach to address the privacy and trust challenges of AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12935v2">ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 This paper is accepted to AAAI 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale. In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17740v2">Bridging the Language Gap: Dynamic Learning Strategies for Improving Multilingual Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized various domains but still struggle with non-Latin scripts and low-resource languages. This paper addresses the critical challenge of improving multilingual performance without extensive fine-tuning. We introduce a novel dynamic learning approach that optimizes prompt strategy, embedding model, and LLM per query at runtime. By adapting configurations dynamically, our method achieves significant improvements over static, best and random baselines. It operates efficiently in both offline and online settings, generalizing seamlessly across new languages and datasets. Leveraging Retrieval-Augmented Generation (RAG) with state-of-the-art multilingual embeddings, we achieve superior task performance across diverse linguistic contexts. Through systematic investigation and evaluation across 18 diverse languages using popular question-answering (QA) datasets we show our approach results in 10-15% improvements in multilingual performance over pre-trained models and 4x gains compared to fine-tuned, language-specific models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00166v2">EEG Emotion Copilot: Optimizing Lightweight LLMs for Emotional EEG Interpretation with Assisted Medical Record Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 10 pages, 12 figures, 2 tables
    </div>
    <details class="paper-abstract">
      In the fields of affective computing (AC) and brain-machine interface (BMI), the analysis of physiological and behavioral signals to discern individual emotional states has emerged as a critical research frontier. While deep learning-based approaches have made notable strides in EEG emotion recognition, particularly in feature extraction and pattern recognition, significant challenges persist in achieving end-to-end emotion computation, including real-time processing, individual adaptation, and seamless user interaction. This paper presents the EEG Emotion Copilot, a system optimizing a lightweight large language model (LLM) with 0.5B parameters operating in a local setting, which first recognizes emotional states directly from EEG signals, subsequently generates personalized diagnostic and treatment suggestions, and finally supports the automation of assisted electronic medical records. Specifically, we demonstrate the critical techniques in the novel data structure of prompt, model pruning and fine-tuning training, and deployment strategies aiming at improving real-time performance and computational efficiency. Extensive experiments show that our optimized lightweight LLM-based copilot achieves an enhanced intuitive interface for participant interaction, superior accuracy of emotion recognition and assisted electronic medical records generation, in comparison to such models with similar scale parameters or large-scale parameters such as 1.5B, 1.8B, 3B and 7B. In summary, through these efforts, the proposed copilot is expected to advance the application of AC in the medical domain, offering innovative solution to mental health monitoring. The codes will be released at https://github.com/NZWANG/EEG_Emotion_Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03491v1">Can LLMs Design Good Questions Based on Context?</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      This paper evaluates questions generated by LLMs from context, comparing them to human-generated questions across six dimensions. We introduce an automated LLM-based evaluation method, focusing on aspects like question length, type, context coverage, and answerability. Our findings highlight unique characteristics of LLM-generated questions, contributing insights that can support further research in question quality and downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03486v1">Align-Pro: A Principled Approach to Prompt Optimization for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 27 pages, Accepted in AAAI 2025
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values is critical as these models become increasingly integrated into various societal and decision-making processes. Traditional methods, such as reinforcement learning from human feedback (RLHF), achieve alignment by fine-tuning model parameters, but these approaches are often computationally expensive and impractical when models are frozen or inaccessible for parameter modification. In contrast, prompt optimization is a viable alternative to RLHF for LLM alignment. While the existing literature has shown empirical promise of prompt optimization, its theoretical underpinning remains under-explored. We address this gap by formulating prompt optimization as an optimization problem and try to provide theoretical insights into the optimality of such a framework. To analyze the performance of the prompt optimization, we study theoretical suboptimality bounds and provide insights in terms of how prompt optimization depends upon the given prompter and target model. We also provide empirical validation through experiments on various datasets, demonstrating that prompt optimization can effectively align LLMs, even when parameter fine-tuning is not feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22376v2">Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00826v2">LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Cryptocurrency investment is inherently difficult due to its shorter history compared to traditional assets, the need to integrate vast amounts of data from various modalities, and the requirement for complex reasoning. While deep learning approaches have been applied to address these challenges, their black-box nature raises concerns about trust and explainability. Recently, large language models (LLMs) have shown promise in financial applications due to their ability to understand multi-modal data and generate explainable decisions. However, single LLM faces limitations in complex, comprehensive tasks such as asset investment. These limitations are even more pronounced in cryptocurrency investment, where LLMs have less domain-specific knowledge in their training corpora. To overcome these challenges, we propose an explainable, multi-modal, multi-agent framework for cryptocurrency investment. Our framework uses specialized agents that collaborate within and across teams to handle subtasks such as data analysis, literature integration, and investment decision-making for the top 30 cryptocurrencies by market capitalization. The expert training module fine-tunes agents using multi-modal historical data and professional investment literature, while the multi-agent investment module employs real-time data to make informed cryptocurrency investment decisions. Unique intrateam and interteam collaboration mechanisms enhance prediction accuracy by adjusting final predictions based on confidence levels within agent teams and facilitating information sharing between teams. Empirical evaluation using data from November 2023 to September 2024 demonstrates that our framework outperforms single-agent models and market benchmarks in classification, asset pricing, portfolio, and explainability performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05482v1">HP-BERT: A framework for longitudinal study of Hinduphobia on social media via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      During the COVID-19 pandemic, community tensions intensified, fuelling Hinduphobic sentiments and discrimination against individuals of Hindu descent within India and worldwide. Large language models (LLMs) have become prominent in natural language processing (NLP) tasks and social media analysis, enabling longitudinal studies of platforms like X (formerly Twitter) for specific issues during COVID-19. We present an abuse detection and sentiment analysis framework that offers a longitudinal analysis of Hinduphobia on X (Twitter) during and after the COVID-19 pandemic. This framework assesses the prevalence and intensity of Hinduphobic discourse, capturing elements such as derogatory jokes and racist remarks through sentiment analysis and abuse detection from pre-trained and fine-tuned LLMs. Additionally, we curate and publish a "Hinduphobic COVID-19 X (Twitter) Dataset" of 8,000 tweets annotated for Hinduphobic abuse detection, which is used to fine-tune a BERT model, resulting in the development of the Hinduphobic BERT (HP-BERT) model. We then further fine-tune HP-BERT using the SenWave dataset for multi-label sentiment analysis. Our study encompasses approximately 27.4 million tweets from six countries, including Australia, Brazil, India, Indonesia, Japan, and the United Kingdom. Our findings reveal a strong correlation between spikes in COVID-19 cases and surges in Hinduphobic rhetoric, highlighting how political narratives, misinformation, and targeted jokes contributed to communal polarisation. These insights provide valuable guidance for developing strategies to mitigate communal tensions in future crises, both locally and globally. We advocate implementing automated monitoring and removal of such content on social media to curb divisive discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04156v1">AdaptiveCoPilot: Design and Testing of a NeuroAdaptive LLM Cockpit Guidance System in both Novice and Expert Pilots</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Pilots operating modern cockpits often face high cognitive demands due to complex interfaces and multitasking requirements, which can lead to overload and decreased performance. This study introduces AdaptiveCoPilot, a neuroadaptive guidance system that adapts visual, auditory, and textual cues in real time based on the pilot's cognitive workload, measured via functional Near-Infrared Spectroscopy (fNIRS). A formative study with expert pilots (N=3) identified adaptive rules for modality switching and information load adjustments during preflight tasks. These insights informed the design of AdaptiveCoPilot, which integrates cognitive state assessments, behavioral data, and adaptive strategies within a context-aware Large Language Model (LLM). The system was evaluated in a virtual reality (VR) simulated cockpit with licensed pilots (N=8), comparing its performance against baseline and random feedback conditions. The results indicate that the pilots using AdaptiveCoPilot exhibited higher rates of optimal cognitive load states on the facets of working memory and perception, along with reduced task completion times. Based on the formative study, experimental findings, qualitative interviews, we propose a set of strategies for future development of neuroadaptive pilot guidance systems and highlight the potential of neuroadaptive systems to enhance pilot performance and safety in aviation environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17218v2">To Err is Machine: Vulnerability Detection Challenges LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      In this paper, we present a challenging code reasoning task: vulnerability detection. Large Language Models (LLMs) have shown promising results in natural-language and math reasoning, but state-of-the-art (SOTA) models reported only 54.5% Balanced Accuracy in our vulnerability detection evaluation, even those models pre-trained on large amounts of source code. Our error analysis on LLM responses shows that the models struggle to reason about the code semantics relevant to identifying vulnerabilities, especially subtle semantic differences caused by small textual changes. We explored prominent models and training settings to understand their effects on vulnerability detection performance -- including better prompts, larger models, more pre-training data, and fine-tuning -- but none led to significant improvements. This raises the question of whether simply scaling training data and model size will allow us to "solve" complex code reasoning tasks like vulnerability detection, or if a fundamental shift in modeling and training techniques is required. We also explored adding domain knowledge to prompts; although it helped certain models understand some code semantics, vulnerability detection requires multi-step reasoning, and these models still failed in steps, such as reasoning about variable relations. Our results suggest that new models, new training methods, or more execution-specific pretraining data may be needed to conquer vulnerability detection. We speculate that auto-regressive pre-training on source code may not effectively extract code semantics, especially on the current pretraining mixtures, in which execution data is scarce. Success on vulnerability detection as a code reasoning task can benefit many areas of software engineering such as debugging, test input generation, and program repair. Our code and data are available at https://doi.org/10.6084/m9.figshare.27368025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04138v1">"Yeah Right!" -- Do LLMs Exhibit Multimodal Feature Transfer?</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Human communication is a multifaceted and multimodal skill. Communication requires an understanding of both the surface-level textual content and the connotative intent of a piece of communication. In humans, learning to go beyond the surface level starts by learning communicative intent in speech. Once humans acquire these skills in spoken communication, they transfer those skills to written communication. In this paper, we assess the ability of speech+text models and text models trained with special emphasis on human-to-human conversations to make this multimodal transfer of skill. We specifically test these models on their ability to detect covert deceptive communication. We find that with no special prompting speech+text LLMs have an advantage over unimodal LLMs in performing this task. Likewise, we find that human-to-human conversation-trained LLMs are also advantaged in this skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03991v1">Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 24 pages, 11 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Calibration, the alignment between model confidence and prediction accuracy, is critical for the reliable deployment of large language models (LLMs). Existing works neglect to measure the generalization of their methods to other prompt styles and different sizes of LLMs. To address this, we define a controlled experimental setting covering 12 LLMs and four prompt styles. We additionally investigate if incorporating the response agreement of multiple LLMs and an appropriate loss function can improve calibration performance. Concretely, we build Calib-n, a novel framework that trains an auxiliary model for confidence estimation that aggregates responses from multiple LLMs to capture inter-model agreement. To optimize calibration, we integrate focal and AUC surrogate losses alongside binary cross-entropy. Experiments across four datasets demonstrate that both response agreement and focal loss improve calibration from baselines. We find that few-shot prompts are the most effective for auxiliary model-based methods, and auxiliary models demonstrate robust calibration performance across accuracy variations, outperforming LLMs' internal probabilities and verbalized confidences. These insights deepen the understanding of influence factors in LLM calibration, supporting their reliable deployment in diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13510v2">Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) workloads have distinct prefill and decode phases with different compute and memory requirements which should ideally be accounted for when scheduling input queries across different LLM instances in a cluster. However existing scheduling algorithms treat LLM workloads as monolithic jobs without considering the distinct characteristics of the two phases in each workload. This leads to sub-optimal scheduling and increased response latency. In this work, we start by characterizing factors affecting the response latency during LLM inference serving. We establish that better load balancing of inference requests across the available LLM instances can improve the end-to-end latency to a larger extent than merely focusing on optimizing the instance-level scheduler. Motivated by our findings, we propose a heuristic-guided reinforcement learning-based intelligent router for data-driven and workload-aware scheduling. Our router schedules queries across LLM instances by leveraging a trainable response-length predictor, and a novel formulation for estimating the impact of mixing different workloads and achieves over 11% lower end-to-end latency than existing approaches on a mix of public datasets and 7.8% lower end-to-end latency on real workload data with diverse input and output trends from Cloud Provider X. Additionally, the proposed framework can also serve as a standard for benchmarking different LLM inference schedulers since it provides the best latency for a given model, hardware, and instance-level scheduler combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14746v3">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale more than four fold (2) for efficient LLMs, the number of parameters N and the size D of a natural training corpus scale as $N \propto D^{0.44}$; (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v1">AlphaPO -- Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Preprint. Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Examples include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce AlphaPO, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14841v2">Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are one of the most promising developments in the field of artificial intelligence, and the software engineering community has readily noticed their potential role in the software development life-cycle. Developers routinely ask LLMs to generate code snippets, increasing productivity but also potentially introducing ownership, privacy, correctness, and security issues. Previous work highlighted how code generated by mainstream commercial LLMs is often not safe, containing vulnerabilities, bugs, and code smells. In this paper, we present a framework that leverages testing and static analysis to assess the quality, and guide the self-improvement, of code generated by general-purpose, open-source LLMs. First, we ask LLMs to generate C code to solve a number of programming tasks. Then we employ ground-truth tests to assess the (in)correctness of the generated code, and a static analysis tool to detect potential safety vulnerabilities. Next, we assess the models ability to evaluate the generated code, by asking them to detect errors and vulnerabilities. Finally, we test the models ability to fix the generated code, providing the reports produced during the static analysis and incorrectness evaluation phases as feedback. Our results show that models often produce incorrect code, and that the generated code can include safety issues. Moreover, they perform very poorly at detecting either issue. On the positive side, we observe a substantial ability to fix flawed code when provided with information about failed tests or potential vulnerabilities, indicating a promising avenue for improving the safety of LLM-based code generation tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19155v3">Lived Experience Not Found: LLMs Struggle to Align with Experts on Addressing Adverse Drug Reactions from Psychiatric Medication Use</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 30 pages, 8 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Adverse Drug Reactions (ADRs) from psychiatric medications are the leading cause of hospitalizations among mental health patients. With healthcare systems and online communities facing limitations in resolving ADR-related issues, Large Language Models (LLMs) have the potential to fill this gap. Despite the increasing capabilities of LLMs, past research has not explored their capabilities in detecting ADRs related to psychiatric medications or in providing effective harm reduction strategies. To address this, we introduce the Psych-ADR benchmark and the Adverse Drug Reaction Response Assessment (ADRA) framework to systematically evaluate LLM performance in detecting ADR expressions and delivering expert-aligned mitigation strategies. Our analyses show that LLMs struggle with understanding the nuances of ADRs and differentiating between types of ADRs. While LLMs align with experts in terms of expressed emotions and tone of the text, their responses are more complex, harder to read, and only 70.86% aligned with expert strategies. Furthermore, they provide less actionable advice by a margin of 12.32% on average. Our work provides a comprehensive benchmark and evaluation framework for assessing LLMs in strategy-driven tasks within high-risk domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17481v2">A Survey on LLM-based Multi-Agent System: Recent Advances and New Frontiers in Application</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 13 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      LLM-based Multi-Agent Systems ( LLM-MAS ) have become a research hotspot since the rise of large language models (LLMs). However, with the continuous influx of new related works, the existing reviews struggle to capture them comprehensively. This paper presents a comprehensive survey of these studies. We first discuss the definition of LLM-MAS, a framework encompassing much of previous work. We provide an overview of the various applications of LLM-MAS in (i) solving complex tasks, (ii) simulating specific scenarios, and (iii) evaluating generative agents. Building on previous studies, we also highlight several challenges and propose future directions for research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.02266v1">LLMzSzŁ: a comprehensive LLM benchmark for Polish</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      This article introduces the first comprehensive benchmark for the Polish language at this scale: LLMzSzŁ (LLMs Behind the School Desk). It is based on a coherent collection of Polish national exams, including both academic and professional tests extracted from the archives of the Polish Central Examination Board. It covers 4 types of exams, coming from 154 domains. Altogether, it consists of almost 19k closed-ended questions. We investigate the performance of open-source multilingual, English, and Polish LLMs to verify LLMs' abilities to transfer knowledge between languages. Also, the correlation between LLMs and humans at model accuracy and exam pass rate levels is examined. We show that multilingual LLMs can obtain superior results over monolingual ones; however, monolingual models may be beneficial when model size matters. Our analysis highlights the potential of LLMs in assisting with exam validation, particularly in identifying anomalies or errors in examination tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03747v1">Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 no comment
    </div>
    <details class="paper-abstract">
      Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment but overlook LLMs' inherent strength on natural language processing -- their deep understanding of linguistic logic and structure rather than superficial embedding processing. We propose Context-Alignment, a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by a Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Demonstration examples prompt are employed to construct Demonstration Examples based Context-Alignment (DECA) following DSCA-GNNs framework. DECA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of DECA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provide powerful prior knowledge on context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12761v2">BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to generating behavior trees for robots using lightweight large language models (LLMs) with a maximum of 7 billion parameters. The study demonstrates that it is possible to achieve satisfying results with compact LLMs when fine-tuned on a specific dataset. The key contributions of this research include the creation of a fine-tuning dataset based on existing behavior trees using GPT-3.5 and a comprehensive comparison of multiple LLMs (namely llama2, llama-chat, and code-llama) across nine distinct tasks. To be thorough, we evaluated the generated behavior trees using static syntactical analysis, a validation system, a simulated environment, and a real robot. Furthermore, this work opens the possibility of deploying such solutions directly on the robot, enhancing its practical applicability. Findings from this study demonstrate the potential of LLMs with a limited number of parameters in generating effective and efficient robot behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17624v2">Forecasting Credit Ratings: A Case Study where Traditional Methods Outperform Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to perform well for many downstream tasks. Transfer learning can enable LLMs to acquire skills that were not targeted during pre-training. In financial contexts, LLMs can sometimes beat well-established benchmarks. This paper investigates how well LLMs perform in the task of forecasting corporate credit ratings. We show that while LLMs are very good at encoding textual information, traditional methods are still very competitive when it comes to encoding numeric and multimodal data. For our task, current LLMs perform worse than a more traditional XGBoost architecture that combines fundamental and macroeconomic data with high-density text-based embedding features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05475v1">Retrieval-Augmented Generation by Evidence Retroactivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation has gained significant attention due to its ability to integrate relevant external knowledge, enhancing the accuracy and reliability of the LLMs' responses. Most of the existing methods apply a dynamic multiple retrieval-generating process, to address multi-hop complex questions by decomposing them into sub-problems. However, these methods rely on an unidirectional forward reasoning paradigm, where errors from insufficient reasoning steps or inherent flaws in current retrieval systems are irreversible, potentially derailing the entire reasoning chain. For the first time, this work introduces Retroactive Retrieval-Augmented Generation (RetroRAG), a novel framework to build a retroactive reasoning paradigm. RetroRAG revises and updates the evidence, redirecting the reasoning chain to the correct direction. RetroRAG constructs an evidence-collation-discovery framework to search, generate, and refine credible evidence. It synthesizes inferential evidence related to the key entities in the question from the existing source knowledge and formulates search queries to uncover additional information. As new evidence is found, RetroRAG continually updates and organizes this information, enhancing its ability to locate further necessary evidence. Paired with an Answerer to generate and evaluate outputs, RetroRAG is capable of refining its reasoning process iteratively until a reliable answer is obtained. Empirical evaluations show that RetroRAG significantly outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03569v1">What Does a Software Engineer Look Like? Exploring Societal Stereotypes in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Accepted for publication in Software Engineering in Society (SEIS) in ICSE 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly gained popularity and are being embedded into professional applications due to their capabilities in generating human-like content. However, unquestioned reliance on their outputs and recommendations can be problematic as LLMs can reinforce societal biases and stereotypes. This study investigates how LLMs, specifically OpenAI's GPT-4 and Microsoft Copilot, can reinforce gender and racial stereotypes within the software engineering (SE) profession through both textual and graphical outputs. We used each LLM to generate 300 profiles, consisting of 100 gender-based and 50 gender-neutral profiles, for a recruitment scenario in SE roles. Recommendations were generated for each profile and evaluated against the job requirements for four distinct SE positions. Each LLM was asked to select the top 5 candidates and subsequently the best candidate for each role. Each LLM was also asked to generate images for the top 5 candidates, providing a dataset for analysing potential biases in both text-based selections and visual representations. Our analysis reveals that both models preferred male and Caucasian profiles, particularly for senior roles, and favoured images featuring traits such as lighter skin tones, slimmer body types, and younger appearances. These findings highlight underlying societal biases influence the outputs of LLMs, contributing to narrow, exclusionary stereotypes that can further limit diversity and perpetuate inequities in the SE field. As LLMs are increasingly adopted within SE research and professional practices, awareness of these biases is crucial to prevent the reinforcement of discriminatory norms and to ensure that AI tools are leveraged to promote an inclusive and equitable engineering culture rather than hinder it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02155v2">CausalMob: Causal Human Mobility Prediction with LLMs-derived Human Intentions toward Public Events</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Large-scale human mobility exhibits spatial and temporal patterns that can assist policymakers in decision making. Although traditional prediction models attempt to capture these patterns, they often interfered by non-periodic public events, such as disasters and occasional celebrations. Since regular human mobility patterns are heavily affected by these events, estimating their causal effects is critical to accurate mobility predictions. Although news articles provide unique perspectives on these events in an unstructured format, processing is a challenge. In this study, we propose a causality-augmented prediction model, called CausalMob, to analyze the causal effects of public events. We first utilize large language models (LLMs) to extract human intentions from news articles and transform them into features that act as causal treatments. Next, the model learns representations of spatio-temporal regional covariates from multiple data sources to serve as confounders for causal inference. Finally, we present a causal effect estimation framework to ensure event features remain independent of confounders during prediction. Based on large-scale real-world data, the experimental results show that the proposed model excels in human mobility prediction, outperforming state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13226v2">Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly trained on datasets consisting of fixed-length token sequences. These datasets are created by randomly concatenating documents of various lengths and then chunking them into sequences of a predetermined target length (concat-and-chunk). Recent attention implementations mask cross-document attention, reducing the effective length of a chunk of tokens. Additionally, training on long sequences becomes computationally prohibitive due to the quadratic cost of attention. In this study, we introduce dataset decomposition, a novel variable sequence length training technique, to tackle these challenges. We decompose a dataset into a union of buckets, each containing sequences of the same size extracted from a unique document. During training, we use variable sequence length and batch-size, sampling simultaneously from all buckets with a curriculum. In contrast to the concat-and-chunk baseline, which incurs a fixed attention cost at every step of training, our proposed method incurs a computational cost proportional to the actual document lengths at each step, resulting in significant savings in training time. We train an 8k context-length 1B model at the same cost as a 2k context-length model trained with the baseline approach. Experiments on a web-scale corpus demonstrate that our approach significantly enhances performance on standard language evaluations and long-context benchmarks, reaching target accuracy with up to 6x faster training compared to the baseline. Our method not only enables efficient pretraining on long sequences but also scales effectively with dataset size. Lastly, we shed light on a critical yet less studied aspect of training large language models: the distribution and curriculum of sequence lengths, which results in a non-negligible difference in performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00168v2">Takeaways from Applying LLM Capabilities to Multiple Conversational Avatars in a VR Pilot Study</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      We present a virtual reality (VR) environment featuring conversational avatars powered by a locally-deployed LLM, integrated with automatic speech recognition (ASR), text-to-speech (TTS), and lip-syncing. Through a pilot study, we explored the effects of three types of avatar status indicators during response generation. Our findings reveal design considerations for improving responsiveness and realism in LLM-driven conversational systems. We also detail two system architectures: one using an LLM-based state machine to control avatar behavior and another integrating retrieval-augmented generation (RAG) for context-grounded responses. Together, these contributions offer practical insights to guide future work in developing task-oriented conversational AI in VR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03218v1">Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      Active Real-time interaction with video LLMs introduces a new paradigm for human-computer interaction, where the model not only understands user intent but also responds while continuously processing streaming video on the fly. Unlike offline video LLMs, which analyze the entire video before answering questions, active real-time interaction requires three capabilities: 1) Perception: real-time video monitoring and interaction capturing. 2) Decision: raising proactive interaction in proper situations, 3) Reaction: continuous interaction with users. However, inherent conflicts exist among the desired capabilities. The Decision and Reaction require a contrary Perception scale and grain, and the autoregressive decoding blocks the real-time Perception and Decision during the Reaction. To unify the conflicted capabilities within a harmonious system, we present Dispider, a system that disentangles Perception, Decision, and Reaction. Dispider features a lightweight proactive streaming video processing module that tracks the video stream and identifies optimal moments for interaction. Once the interaction is triggered, an asynchronous interaction module provides detailed responses, while the processing module continues to monitor the video in the meantime. Our disentangled and asynchronous design ensures timely, contextually accurate, and computationally efficient responses, making Dispider ideal for active real-time interaction for long-duration video streams. Experiments show that Dispider not only maintains strong performance in conventional video QA tasks, but also significantly surpasses previous online models in streaming scenario responses, thereby validating the effectiveness of our architecture. The code and model are released at \url{https://github.com/Mark12Ding/Dispider}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03212v1">Leveraging Explainable AI for LLM Text Attribution: Differentiating Human-Written and Multiple LLMs-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The development of Generative AI Large Language Models (LLMs) raised the alarm regarding identifying content produced through generative AI or humans. In one case, issues arise when students heavily rely on such tools in a manner that can affect the development of their writing or coding skills. Other issues of plagiarism also apply. This study aims to support efforts to detect and identify textual content generated using LLM tools. We hypothesize that LLMs-generated text is detectable by machine learning (ML), and investigate ML models that can recognize and differentiate texts generated by multiple LLMs tools. We leverage several ML and Deep Learning (DL) algorithms such as Random Forest (RF), and Recurrent Neural Networks (RNN), and utilized Explainable Artificial Intelligence (XAI) to understand the important features in attribution. Our method is divided into 1) binary classification to differentiate between human-written and AI-text, and 2) multi classification, to differentiate between human-written text and the text generated by the five different LLM tools (ChatGPT, LLaMA, Google Bard, Claude, and Perplexity). Results show high accuracy in the multi and binary classification. Our model outperformed GPTZero with 98.5\% accuracy to 78.3\%. Notably, GPTZero was unable to recognize about 4.2\% of the observations, but our model was able to recognize the complete test dataset. XAI results showed that understanding feature importance across different classes enables detailed author/source profiles. Further, aiding in attribution and supporting plagiarism detection by highlighting unique stylistic and structural elements ensuring robust content originality verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16353v2">The Two-Hop Curse: LLMs trained on A$\rightarrow$B, B$\rightarrow$C fail to learn A$\rightarrow$C</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      [Notice: This version is outdated. Recent research contradicts some key claims; we are working on a major revision with more nuanced analysis. Please wait for the updated version.] While LLMs excel at multi-hop questions (e.g. "Who is the spouse of the performer of Imagine?") when using chain-of-thought reasoning (CoT), they struggle when forced to reason internally (without CoT). Previous work on the size and nature of this gap produced mixed evidence with inconclusive results. In this paper, we introduce a controlled setting for investigating two-hop reasoning in LLMs, where the above-chance performance constitutes undeniable evidence for latent reasoning. We fine-tune LLMs (including Llama 3 8B Instruct and GPT-4o) on fictional facts and confirm that they generalize to answering two-hop questions about them using CoT. We find that models can perform latent reasoning when facts appear together during training or in the prompt. However, to our surprise, models completely fail at two-hop reasoning without CoT when learned facts only appear in different documents, achieving chance-level accuracy and chance-level test loss. We call this complete failure to compose separately learned facts the Two-Hop Curse. Moreover, we evaluate 9 frontier LLMs on real-world facts, finding that models completely fail at two-hop no-CoT reasoning for over half of question categories while maintaining partial success with CoT across most categories. These results suggest that LLMs lack a general capability for latent multi-hop reasoning independent of the question type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v3">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has demonstrated remarkable progress in complex reasoning tasks. However, a significant discrepancy persists between benchmark performances and real-world applications. We identify this gap as primarily stemming from current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, particularly in complex reasoning tasks where both accuracy and consistency are crucial. This work makes two key contributions. First, we introduce G-Pass@k, a novel evaluation metric that provides a continuous assessment of model performance across multiple sampling attempts, quantifying both the model's peak performance potential and its stability. Second, we present LiveMathBench, a dynamic benchmark comprising challenging, contemporary mathematical problems designed to minimize data leakage risks during evaluation. Through extensive experiments using G-Pass@k on state-of-the-art LLMs with LiveMathBench, we provide comprehensive insights into both their maximum capabilities and operational consistency. Our findings reveal substantial room for improvement in LLMs' "realistic" reasoning capabilities, highlighting the need for more robust evaluation methods. The benchmark and detailed results are available at: https://github.com/open-compass/GPassK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03628v2">Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 82 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Imbalanced classification and spurious correlation are common challenges in data science and machine learning. Both issues are linked to data imbalance, with certain groups of data samples significantly underrepresented, which in turn would compromise the accuracy, robustness and generalizability of the learned models. Recent advances have proposed leveraging the flexibility and generative capabilities of large language models (LLMs), typically built on transformer architectures, to generate synthetic samples and to augment the observed data. In the context of imbalanced data, LLMs are used to oversample underrepresented groups and have shown promising improvements. However, there is a clear lack of theoretical understanding of such synthetic data approaches. In this article, we develop novel theoretical foundations to systematically study the roles of synthetic samples in addressing imbalanced classification and spurious correlation. Specifically, we first explicitly quantify the benefits of synthetic oversampling. Next, we analyze the scaling dynamics in synthetic data augmentation, and derive the corresponding scaling law. Finally, we demonstrate the capacity of transformer models to generate high-quality synthetic samples. We further conduct extensive numerical experiments to validate the efficacy of the LLM-based synthetic oversampling and augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00722v2">Pathway to Secure and Trustworthy ZSM for LLMs: Attacks, Defense, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that ZSM networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in ZSM networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of ZSM networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09830v3">Automating the Generation of Prompts for LLM-based Action Choice in PDDL Planning</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Latest Version of the paper previously called "AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL"; Added extended experiments; newer gpt4 model
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized a large variety of NLP tasks. An active debate is to what extent they can do reasoning and planning. Prior work has assessed the latter in the specific context of PDDL planning, based on manually converting three PDDL domains into natural language (NL) prompts. Here we automate this conversion step, showing how to leverage an LLM to automatically generate NL prompts from PDDL input. Our automatically generated NL prompts result in similar LLM-planning performance as the previous manually generated ones. Beyond this, the automation enables us to run much larger experiments, providing for the first time a broad evaluation of LLM planning performance in PDDL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03012v1">Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 The first three authors contributed equally
    </div>
    <details class="paper-abstract">
      Multimodal LLMs have reached remarkable levels of proficiency in understanding multimodal inputs, driving extensive research to develop increasingly powerful models. However, much less attention has been paid to understanding and explaining the underlying mechanisms of these models. Most existing explainability research examines these models only in their final states, overlooking the dynamic representational shifts that occur during training. In this work, we systematically analyze the evolution of hidden state representations to reveal how fine-tuning alters the internal structure of a model to specialize in new multimodal tasks. Using a concept-based approach, we map hidden states to interpretable visual and textual concepts, enabling us to trace changes in encoded concepts across modalities as training progresses. We also demonstrate the use of shift vectors to capture these concepts changes. These shift vectors allow us to recover fine-tuned concepts by shifting those in the original model. Finally, we explore the practical impact of our findings on model steering, showing that we can adjust multimodal LLMs behaviors without any training, such as modifying answer types, captions style, or biasing the model toward specific responses. Our work sheds light on how multimodal representations evolve through fine-tuning and offers a new perspective for interpreting model adaptation in multimodal tasks. The code for this project is publicly available at https://github.com/mshukor/xl-vlms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02885v1">MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Video large language models (Video-LLMs) have made significant progress in understanding videos. However, processing multiple frames leads to lengthy visual token sequences, presenting challenges such as the limited context length cannot accommodate the entire video, and the inclusion of irrelevant frames hinders visual perception. Hence, effective frame selection is crucial. This paper emphasizes that frame selection should follow three key principles: query relevance, list-wise diversity, and sequentiality. Existing methods, such as uniform frame sampling and query-frame matching, do not capture all of these principles. Thus, we propose Markov decision determinantal point process with dynamic programming (MDP3) for frame selection, a training-free and model-agnostic method that can be seamlessly integrated into existing Video-LLMs. Our method first estimates frame similarities conditioned on the query using a conditional Gaussian kernel within the reproducing kernel Hilbert space~(RKHS). We then apply the determinantal point process~(DPP) to the similarity matrix to capture both query relevance and list-wise diversity. To incorporate sequentiality, we segment the video and apply DPP within each segment, conditioned on the preceding segment selection, modeled as a Markov decision process~(MDP) for allocating selection sizes across segments. Theoretically, MDP3 provides a \((1 - 1/e)\)-approximate solution to the NP-hard list-wise frame selection problem with pseudo-polynomial time complexity, demonstrating its efficiency. Empirically, MDP3 significantly outperforms existing methods, verifying its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20787v3">SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) is crucial for understanding their capabilities and limitations across various applications, including natural language processing and code generation. Existing benchmarks like MMLU, C-Eval, and HumanEval assess general LLM performance but lack focus on specific expert domains such as cybersecurity. Previous attempts to create cybersecurity datasets have faced limitations, including insufficient data volume and a reliance on multiple-choice questions (MCQs). To address these gaps, we propose SecBench, a multi-dimensional benchmarking dataset designed to evaluate LLMs in the cybersecurity domain. SecBench includes questions in various formats (MCQs and short-answer questions (SAQs)), at different capability levels (Knowledge Retention and Logical Reasoning), in multiple languages (Chinese and English), and across various sub-domains. The dataset was constructed by collecting high-quality data from open sources and organizing a Cybersecurity Question Design Contest, resulting in 44,823 MCQs and 3,087 SAQs. Particularly, we used the powerful while cost-effective LLMs to (1). label the data and (2). constructing a grading agent for automatic evaluation of SAQs. Benchmarking results on 16 SOTA LLMs demonstrate the usability of SecBench, which is arguably the largest and most comprehensive benchmark dataset for LLMs in cybersecurity. More information about SecBench can be found at our website, and the dataset can be accessed via the artifact link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03288v1">CodeVision: Detecting LLM-Generated Code Using 2D Token Probability Maps and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) like ChatGPT has significantly improved automated code generation, enhancing software development efficiency. However, this introduces challenges in academia, particularly in distinguishing between human-written and LLM-generated code, which complicates issues of academic integrity. Existing detection methods, such as pre-trained models and watermarking, face limitations in adaptability and computational efficiency. In this paper, we propose a novel detection method using 2D token probability maps combined with vision models, preserving spatial code structures such as indentation and brackets. By transforming code into log probability matrices and applying vision models like Vision Transformers (ViT) and ResNet, we capture both content and structure for more accurate detection. Our method shows robustness across multiple programming languages and improves upon traditional detectors, offering a scalable and computationally efficient solution for identifying LLM-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16594v5">From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 v5: add new citations; 36 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Assessment and evaluation have long been critical challenges in artificial intelligence (AI) and natural language processing (NLP). However, traditional methods, whether matching-based or embedding-based, often fall short of judging subtle attributes and delivering satisfactory results. Recent advancements in Large Language Models (LLMs) inspire the "LLM-as-a-judge" paradigm, where LLMs are leveraged to perform scoring, ranking, or selection across various tasks and applications. This paper provides a comprehensive survey of LLM-based judgment and assessment, offering an in-depth overview to advance this emerging field. We begin by giving detailed definitions from both input and output perspectives. Then we introduce a comprehensive taxonomy to explore LLM-as-a-judge from three dimensions: what to judge, how to judge and where to judge. Finally, we compile benchmarks for evaluating LLM-as-a-judge and highlight key challenges and promising directions, aiming to provide valuable insights and inspire future research in this promising research area. Paper list and more resources about LLM-as-a-judge can be found at \url{https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge} and \url{https://llm-as-a-judge.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12470v3">DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Accepted by WSDM 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into recommender systems has led to substantial performance improvements. However, this often comes at the cost of diminished recommendation diversity, which can negatively impact user satisfaction. To address this issue, controllable recommendation has emerged as a promising approach, allowing users to specify their preferences and receive recommendations that meet their diverse needs. Despite its potential, existing controllable recommender systems frequently rely on simplistic mechanisms, such as a single prompt, to regulate diversity-an approach that falls short of capturing the full complexity of user preferences. In response to these limitations, we propose DLCRec, a novel framework designed to enable fine-grained control over diversity in LLM-based recommendations. Unlike traditional methods, DLCRec adopts a fine-grained task decomposition strategy, breaking down the recommendation process into three sequential sub-tasks: genre prediction, genre filling, and item prediction. These sub-tasks are trained independently and inferred sequentially according to user-defined control numbers, ensuring more precise control over diversity. Furthermore, the scarcity and uneven distribution of diversity-related user behavior data pose significant challenges for fine-tuning. To overcome these obstacles, we introduce two data augmentation techniques that enhance the model's robustness to noisy and out-of-distribution data. These techniques expose the model to a broader range of patterns, improving its adaptability in generating recommendations with varying levels of diversity. Our extensive empirical evaluation demonstrates that DLCRec not only provides precise control over diversity but also outperforms state-of-the-art baselines across multiple recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14743v2">LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 5 pages, ICASSP 2025
    </div>
    <details class="paper-abstract">
      Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset that contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining valuable information for both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that the current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.49% equal error rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09824v6">LLM-Based Multi-Agent Systems are Scalable Graph Generative Models</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The structural properties of naturally arising social graphs are extensively studied to understand their evolution. Prior approaches for modeling network dynamics typically rely on rule-based models, which lack realism and generalizability, or deep learning-based models, which require large-scale training datasets. Social graphs, as abstract graph representations of entity-wise interactions, present an opportunity to explore network evolution mechanisms through realistic simulations of human-item interactions. Leveraging the pre-trained social consensus knowledge embedded in large language models (LLMs), we present GraphAgent-Generator (GAG), a novel simulation-based framework for dynamic, text-attributed social graph generation. GAG simulates the temporal node and edge generation processes for zero-shot social graph generation. The resulting graphs exhibit adherence to seven key macroscopic network properties, achieving an 11% improvement in microscopic graph structure metrics. Through the node classification benchmarking task, we validate GAG effectively captures the intricate text-structure correlations in graph generation. Furthermore, GAG supports generating graphs with up to nearly 100,000 nodes or 10 million edges through large-scale LLM-based agent simulation with parallel acceleration, achieving a minimum speed-up of 90.4%. The source code is available at https://github.com/Ji-Cather/GraphAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.01818v1">Rerouting LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      LLM routers aim to balance quality and cost of generation by classifying queries and routing them to a cheaper or more expensive LLM depending on their complexity. Routers represent one type of what we call LLM control planes: systems that orchestrate use of one or more LLMs. In this paper, we investigate routers' adversarial robustness. We first define LLM control plane integrity, i.e., robustness of LLM orchestration to adversarial inputs, as a distinct problem in AI safety. Next, we demonstrate that an adversary can generate query-independent token sequences we call ``confounder gadgets'' that, when added to any query, cause LLM routers to send the query to a strong LLM. Our quantitative evaluation shows that this attack is successful both in white-box and black-box settings against a variety of open-source and commercial routers, and that confounding queries do not affect the quality of LLM responses. Finally, we demonstrate that gadgets can be effective while maintaining low perplexity, thus perplexity-based filtering is not an effective defense. We finish by investigating alternative defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02688v1">Decoding specialised feature neurons in LLMs with the final projection layer</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) typically have billions of parameters and are thus often difficult to interpret in their operation. Such black-box models can pose a significant risk to safety when trusted to make important decisions. The lack of interpretability of LLMs is more related to their sheer size, rather than the complexity of their individual components. The TARS method for knowledge removal (Davies et al 2024) provides strong evidence for the hypothesis that that linear layer weights which act directly on the residual stream may have high correlation with different concepts encoded in the residual stream. Building upon this, we attempt to decode neuron weights directly into token probabilities through the final projection layer of the model (the LM-head). Firstly, we show that with Llama 3.1 8B we can utilise the LM-head to decode specialised feature neurons that respond strongly to certain concepts, with examples such as "dog" and "California". This is then confirmed by demonstrating that these neurons can be clamped to affect the probability of the concept in the output. This extends to the fine-tuned assistant Llama 3.1 8B instruct model, where we find that over 75% of neurons in the up-projection layers have the same top associated token compared to the pretrained model. Finally, we demonstrate that clamping the "dog" neuron leads the instruct model to always discuss dogs when asked about its favourite animal. Through our method, it is possible to map the entirety of Llama 3.1 8B's up-projection neurons in less than 15 minutes with no parallelization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03516v3">Query3D: LLM-Powered Open-Vocabulary Scene Segmentation with Language Embedded 3D Gaussian</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      This paper introduces a novel method for open-vocabulary 3D scene querying in autonomous driving by combining Language Embedded 3D Gaussians with Large Language Models (LLMs). We propose utilizing LLMs to generate both contextually canonical phrases and helping positive words for enhanced segmentation and scene interpretation. Our method leverages GPT-3.5 Turbo as an expert model to create a high-quality text dataset, which we then use to fine-tune smaller, more efficient LLMs for on-device deployment. Our comprehensive evaluation on the WayveScenes101 dataset demonstrates that LLM-guided segmentation significantly outperforms traditional approaches based on predefined canonical phrases. Notably, our fine-tuned smaller models achieve performance comparable to larger expert models while maintaining faster inference times. Through ablation studies, we discover that the effectiveness of helping positive words correlates with model scale, with larger models better equipped to leverage additional semantic information. This work represents a significant advancement towards more efficient, context-aware autonomous driving systems, effectively bridging 3D scene representation with high-level semantic querying while maintaining practical deployment considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02628v1">Cracks in The Stack: Hidden Vulnerabilities and Licensing Risks in LLM Pre-Training Datasets</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted in the Second International Workshop on Large Language Models for Code (LLM4Code 2025)
    </div>
    <details class="paper-abstract">
      A critical part of creating code suggestion systems is the pre-training of Large Language Models on vast amounts of source code and natural language text, often of questionable origin or quality. This may contribute to the presence of bugs and vulnerabilities in code generated by LLMs. While efforts to identify bugs at or after code generation exist, it is preferable to pre-train or fine-tune LLMs on curated, high-quality, and compliant datasets. The need for vast amounts of training data necessitates that such curation be automated, minimizing human intervention. We propose an automated source code autocuration technique that leverages the complete version history of open-source software projects to improve the quality of training data. This approach leverages the version history of all OSS projects to identify training data samples that have been modified or have undergone changes in at least one OSS project, and pinpoint a subset of samples that include fixes for bugs or vulnerabilities. We evaluate this method using The Stack v2 dataset, and find that 17% of the code versions in the dataset have newer versions, with 17% of those representing bug fixes, including 2.36% addressing known CVEs. The deduplicated version of Stack v2 still includes blobs vulnerable to 6,947 known CVEs. Furthermore, 58% of the blobs in the dataset were never modified after creation, suggesting they likely represent software with minimal or no use. Misidentified blob origins present an additional challenge, as they lead to the inclusion of non-permissively licensed code, raising serious compliance concerns. By addressing these issues, the training of new models can avoid perpetuating buggy code patterns or license violations. We expect our results to inspire process improvements for automated data curation, with the potential to enhance the reliability of outputs generated by AI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v1">HALO: Hadamard-Assisted Lossless Optimization for Efficient Low-Precision LLM Training and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which often already have large weight and activation outlier values that render quantized optimization difficult. We present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, to mitigate outliers during the low-precision computation, 2) FSDP integration for low-precision communication, and 3) high-performance kernel support. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.31x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. Our method supports both standard and parameter-efficient fine-tuning (PEFT) methods, both backed by efficient kernel implementations. Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in FP8 precision, while delivering performance benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02621v1">LLMs Help Alleviate the Cross-Subject Variability in Brain Signal and Language Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      Decoding human activity from EEG signals has long been a popular research topic. While recent studies have increasingly shifted focus from single-subject to cross-subject analysis, few have explored the model's ability to perform zero-shot predictions on EEG signals from previously unseen subjects. This research aims to investigate whether deep learning methods can capture subject-independent semantic information inherent in human EEG signals. Such insights are crucial for Brain-Computer Interfaces (BCI) because, on one hand, they demonstrate the model's robustness against subject-specific temporal biases, and on the other, they significantly enhance the generalizability of downstream tasks. We employ Large Language Models (LLMs) as denoising agents to extract subject-independent semantic features from noisy EEG signals. Experimental results, including ablation studies, highlight the pivotal role of LLMs in decoding subject-independent semantic information from noisy EEG data. We hope our findings will contribute to advancing BCI research and assist both academia and industry in applying EEG signals to a broader range of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10351v3">Bias Unveiled: Investigating Social Bias in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 9pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the field of automated code generation. However, a notable research gap exists in the evaluation of social biases that may be present in the code produced by LLMs. To solve this issue, we propose a novel fairness framework, i.e., Solar, to assess and mitigate the social biases of LLM-generated code. Specifically, Solar can automatically generate test cases for quantitatively uncovering social biases of the auto-generated code by LLMs. To quantify the severity of social biases in generated code, we develop a dataset that covers a diverse set of social problems. We applied Solar and the crafted dataset to four state-of-the-art LLMs for code generation. Our evaluation reveals severe bias in the LLM-generated code from all the subject LLMs. Furthermore, we explore several strategies for bias mitigation, including Chain-of-Thought (CoT) prompting, combining positive role-playing with CoT prompting and iterative prompting. Our experiments show that iterative prompting can effectively reduce social bias in LLM-generated code by up to 90%. Solar is highly extensible to evaluate new social problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02600v1">TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      The rising demand for generative large language models (LLMs) poses challenges for thermal and power management in cloud datacenters. Traditional techniques often are inadequate for LLM inference due to the fine-grained, millisecond-scale execution phases, each with distinct performance, thermal, and power profiles. Additionally, LLM inference workloads are sensitive to various configuration parameters (e.g., model parallelism, size, and quantization) that involve trade-offs between performance, temperature, power, and output quality. Moreover, clouds often co-locate SaaS and IaaS workloads, each with different levels of visibility and flexibility. We propose TAPAS, a thermal- and power-aware framework designed for LLM inference clusters in the cloud. TAPAS enhances cooling and power oversubscription capabilities, reducing the total cost of ownership (TCO) while effectively handling emergencies (e.g., cooling and power failures). The system leverages historical temperature and power data, along with the adaptability of SaaS workloads, to: (1) efficiently place new GPU workload VMs within cooling and power constraints, (2) route LLM inference requests across SaaS VMs, and (3) reconfigure SaaS VMs to manage load spikes and emergency situations. Our evaluation on a large GPU cluster demonstrates significant reductions in thermal and power throttling events, boosting system efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02552v1">Multi-LLM Collaborative Caption Generation in Scientific Documents</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted to AAAI 2025 AI4Research Workshop
    </div>
    <details class="paper-abstract">
      Scientific figure captioning is a complex task that requires generating contextually appropriate descriptions of visual content. However, existing methods often fall short by utilizing incomplete information, treating the task solely as either an image-to-text or text summarization problem. This limitation hinders the generation of high-quality captions that fully capture the necessary details. Moreover, existing data sourced from arXiv papers contain low-quality captions, posing significant challenges for training large language models (LLMs). In this paper, we introduce a framework called Multi-LLM Collaborative Figure Caption Generation (MLBCAP) to address these challenges by leveraging specialized LLMs for distinct sub-tasks. Our approach unfolds in three key modules: (Quality Assessment) We utilize multimodal LLMs to assess the quality of training data, enabling the filtration of low-quality captions. (Diverse Caption Generation) We then employ a strategy of fine-tuning/prompting multiple LLMs on the captioning task to generate candidate captions. (Judgment) Lastly, we prompt a prominent LLM to select the highest quality caption from the candidates, followed by refining any remaining inaccuracies. Human evaluations demonstrate that informative captions produced by our approach rank better than human-written captions, highlighting its effectiveness. Our code is available at https://github.com/teamreboott/MLBCAP
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02531v1">Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 20 pages, 1 figure
    </div>
    <details class="paper-abstract">
      With the expansion of neural networks, such as large language models, humanity is exponentially heading towards superintelligence. As various AI systems are increasingly integrated into the fabric of societies-through recommending values, devising creative solutions, and making decisions-it becomes critical to assess how these AI systems impact humans in the long run. This research aims to contribute towards establishing a benchmark for evaluating the sentiment of various Large Language Models in socially importan issues. The methodology adopted was a Likert scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared against sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results highlighted a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment score towards AGI, whereas Bard was leaning towards the neutral sentiment. The human samples, contrastingly, showed a lower average sentiment of 2.97. The temporal comparison revealed differences in sentiment evolution between LLMs in three days, ranging from 1.03% to 8.21%. The study's analysis outlines the prospect of potential conflicts of interest and bias possibilities in LLMs' sentiment formation. Results indicate that LLMs, akin to human cognitive processes, could potentially develop unique sentiments and subtly influence societies' perceptions towards various opinions formed within the LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19616v2">Gradient Weight-normalized Low-rank Projection for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted by the 39th AAAI Conference on Artificial Intelligence (AAAI-25) [Main Technical Track]
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across various tasks, but the escalating demands on computational resources pose significant challenges, particularly in the extensive utilization of full fine-tuning for downstream tasks. To address this, parameter-efficient fine-tuning (PEFT) methods have been developed, but they often underperform compared to full fine-tuning and struggle with memory efficiency. In this work, we introduce Gradient Weight-Normalized Low-Rank Projection (GradNormLoRP), a novel approach that enhances both parameter and memory efficiency while maintaining comparable performance to full fine-tuning. GradNormLoRP normalizes the weight matrix to improve gradient conditioning, facilitating better convergence during optimization. Additionally, it applies low-rank approximations to the weight and gradient matrices, significantly reducing memory usage during training. Extensive experiments demonstrate that our 8-bit GradNormLoRP reduces optimizer memory usage by up to 89.5% and enables the pre-training of large LLMs, such as LLaMA 7B, on consumer-level GPUs like the NVIDIA RTX 4090, without additional inference costs. Moreover, GradNormLoRP outperforms existing low-rank methods in fine-tuning tasks. For instance, when fine-tuning the RoBERTa model on all GLUE tasks with a rank of 8, GradNormLoRP achieves an average score of 80.65, surpassing LoRA's score of 79.23. These results underscore GradNormLoRP as a promising alternative for efficient LLM pre-training and fine-tuning. Source code: https://github.com/Jhhuangkay/Gradient-Weight-normalized-Low-rank-Projection-for-Efficient-LLM-Training
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17522v2">DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21520v3">LLM-Forest: Ensemble Learning of LLMs with Graph-Augmented Prompts for Data Imputation</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      Missing data imputation is a critical challenge in various domains, such as healthcare and finance, where data completeness is vital for accurate analysis. Large language models (LLMs), trained on vast corpora, have shown strong potential in data generation, making them a promising tool for data imputation. However, challenges persist in designing effective prompts for a finetuning-free process and in mitigating the risk of LLM hallucinations. To address these issues, we propose a novel framework, LLM-Forest, which introduces a "forest" of few-shot learning LLM "trees" with confidence-based weighted voting, inspired by ensemble learning (Random Forest). This framework is established on a new concept of bipartite information graphs to identify high-quality relevant neighboring entries with both feature and value granularity. Extensive experiments on 9 real-world datasets demonstrate the effectiveness and efficiency of LLM-Forest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00697v2">PANDA -- Paired Anti-hate Narratives Dataset from Asia: Using an LLM-as-a-Judge to Create the First Chinese Counterspeech Dataset</a></div>
    <div class="paper-meta">
      📅 2025-01-04
      | 💬 to be published in MCG-COLING 2025's conference proceedings
    </div>
    <details class="paper-abstract">
      Despite the global prevalence of Modern Standard Chinese language, counterspeech (CS) resources for Chinese remain virtually nonexistent. To address this gap in East Asian counterspeech research we introduce the a corpus of Modern Standard Mandarin counterspeech that focuses on combating hate speech in Mainland China. This paper proposes a novel approach of generating CS by using an LLM-as-a-Judge, simulated annealing, LLMs zero-shot CN generation and a round-robin algorithm. This is followed by manual verification for quality and contextual relevance. This paper details the methodology for creating effective counterspeech in Chinese and other non-Eurocentric languages, including unique cultural patterns of which groups are maligned and linguistic patterns in what kinds of discourse markers are programmatically marked as hate speech (HS). Analysis of the generated corpora, we provide strong evidence for the lack of open-source, properly labeled Chinese hate speech data and the limitations of using an LLM-as-Judge to score possible answers in Chinese. Moreover, the present corpus serves as the first East Asian language based CS corpus and provides an essential resource for future research on counterspeech generation and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02341v1">UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      Low-altitude mobility, exemplified by unmanned aerial vehicles (UAVs), has introduced transformative advancements across various domains, like transportation, logistics, and agriculture. Leveraging flexible perspectives and rapid maneuverability, UAVs extend traditional systems' perception and action capabilities, garnering widespread attention from academia and industry. However, current UAV operations primarily depend on human control, with only limited autonomy in simple scenarios, and lack the intelligence and adaptability needed for more complex environments and tasks. The emergence of large language models (LLMs) demonstrates remarkable problem-solving and generalization capabilities, offering a promising pathway for advancing UAV intelligence. This paper explores the integration of LLMs and UAVs, beginning with an overview of UAV systems' fundamental components and functionalities, followed by an overview of the state-of-the-art in LLM technology. Subsequently, it systematically highlights the multimodal data resources available for UAVs, which provide critical support for training and evaluation. Furthermore, it categorizes and analyzes key tasks and application scenarios where UAVs and LLMs converge. Finally, a reference roadmap towards agentic UAVs is proposed, aiming to enable UAVs to achieve agentic intelligence through autonomous perception, memory, reasoning, and tool utilization. Related resources are available at https://github.com/Hub-Tian/UAVs_Meet_LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02336v1">AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-04
      | 💬 9 pages,10 figures, AAAI
    </div>
    <details class="paper-abstract">
      Long-context large language models (LLMs) inference is increasingly critical, motivating a number of studies devoted to alleviating the substantial storage and computational costs in such scenarios. Layer-wise skipping methods are promising optimizations but rarely explored in long-context inference. We observe that existing layer-wise skipping strategies have several limitations when applied in long-context inference, including the inability to adapt to model and context variability, disregard for sublayer significance, and inapplicability for the prefilling phase. This paper proposes \sysname, an adaptive sublayer skipping method specifically designed for long-context inference. \sysname adaptively identifies less important layers by leveraging on-the-fly similarity information, enables sublayer-wise skipping, and accelerates both the prefilling and decoding phases. The effectiveness of \sysname is demonstrated through extensive experiments on various long-context benchmarks and models, showcasing its superior inference performance over existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02266v1">LLMzSzŁ: a comprehensive LLM benchmark for Polish</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      This article introduces the first comprehensive benchmark for the Polish language at this scale: LLMzSz{\L} (LLMs Behind the School Desk). It is based on a coherent collection of Polish national exams, including both academic and professional tests extracted from the archives of the Polish Central Examination Board. It covers 4 types of exams, coming from 154 domains. Altogether, it consists of almost 19k closed-ended questions. We investigate the performance of open-source multilingual, English, and Polish LLMs to verify LLMs' abilities to transfer knowledge between languages. Also, the correlation between LLMs and humans at model accuracy and exam pass rate levels is examined. We show that multilingual LLMs can obtain superior results over monolingual ones; however, monolingual models may be beneficial when model size matters. Our analysis highlights the potential of LLMs in assisting with exam validation, particularly in identifying anomalies or errors in examination tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15173v3">Second-Order Fine-Tuning without Pain for LLMs:A Hessian Informed Zeroth-Order Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with classic first-order optimizers entails prohibitive GPU memory due to the backpropagation process. Recent works have turned to zeroth-order optimizers for fine-tuning, which save substantial memory by using two forward passes. However, these optimizers are plagued by the heterogeneity of parameter curvatures across different dimensions. In this work, we propose HiZOO, a diagonal Hessian informed zeroth-order optimizer which is the first work to leverage the diagonal Hessian to enhance zeroth-order optimizer for fine-tuning LLMs. What's more, HiZOO avoids the expensive memory cost and only increases one forward pass per step. Extensive experiments on various models (350M~66B parameters) indicate that HiZOO improves model convergence, significantly reducing training steps and effectively enhancing model accuracy. Moreover, we visualize the optimization trajectories of HiZOO on test functions, illustrating its effectiveness in handling heterogeneous curvatures. Lastly, we provide theoretical proofs of convergence for HiZOO. Code is publicly available at https://anonymous.4open.science/r/HiZOO27F8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02237v1">Financial Named Entity Recognition: How Far Can LLM Go?</a></div>
    <div class="paper-meta">
      📅 2025-01-04
      | 💬 Accepted at The Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), in conjunction with COLING 2025
    </div>
    <details class="paper-abstract">
      The surge of large language models (LLMs) has revolutionized the extraction and analysis of crucial information from a growing volume of financial statements, announcements, and business news. Recognition for named entities to construct structured data poses a significant challenge in analyzing financial documents and is a foundational task for intelligent financial analytics. However, how effective are these generic LLMs and their performance under various prompts are yet need a better understanding. To fill in the blank, we present a systematic evaluation of state-of-the-art LLMs and prompting methods in the financial Named Entity Recognition (NER) problem. Specifically, our experimental results highlight their strengths and limitations, identify five representative failure types, and provide insights into their potential and challenges for domain-specific tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02226v1">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-01-04
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Recommender systems have become increasingly vital in our daily lives, helping to alleviate the problem of information overload across various user-oriented online services. The emergence of Large Language Models (LLMs) has yielded remarkable achievements, demonstrating their potential for the development of next-generation recommender systems. Despite these advancements, LLM-based recommender systems face inherent limitations stemming from their LLM backbones, particularly issues of hallucinations and the lack of up-to-date and domain-specific knowledge. Recently, Retrieval-Augmented Generation (RAG) has garnered significant attention for addressing these limitations by leveraging external knowledge sources to enhance the understanding and generation of LLMs. However, vanilla RAG methods often introduce noise and neglect structural relationships in knowledge, limiting their effectiveness in LLM-based recommendations. To address these limitations, we propose to retrieve high-quality and up-to-date structure information from the knowledge graph (KG) to augment recommendations. Specifically, our approach develops a retrieval-augmented framework, termed K-RagRec, that facilitates the recommendation generation process by incorporating structure information from the external KG. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03266v1">LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      LLM safety and ethical alignment are widely discussed, but the impact of content moderation on user satisfaction remains underexplored. To address this, we analyze nearly 50,000 Chatbot Arena response-pairs using a novel fine-tuned RoBERTa model, that we trained on hand-labeled data to disentangle refusals due to ethical concerns from other refusals due to technical disabilities or lack of information. Our findings reveal a significant refusal penalty on content moderation, with users choosing ethical-based refusals roughly one-fourth as often as their preferred LLM response compared to standard responses. However, the context and phrasing play critical roles: refusals on highly sensitive prompts, such as illegal content, achieve higher win rates than less sensitive ethical concerns, and longer responses closely aligned with the prompt perform better. These results emphasize the need for nuanced moderation strategies that balance ethical safeguards with user satisfaction. Moreover, we find that the refusal penalty is notably lower in evaluations using the LLM-as-a-Judge method, highlighting discrepancies between user and automated assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02191v1">On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      Missing data imputation, which aims to impute the missing values in the raw datasets to achieve the completeness of datasets, is crucial for modern data-driven models like large language models (LLMs) and has attracted increasing interest over the past decades. Despite its importance, existing solutions for missing data imputation either 1) only support numerical and categorical data or 2) show an unsatisfactory performance due to their design prioritizing text data and the lack of key properties for tabular data imputation. In this paper, we propose UnIMP, a Unified IMPutation framework that leverages LLM and high-order message passing to enhance the imputation of mixed-type data including numerical, categorical, and text data. Specifically, we first introduce a cell-oriented hypergraph to model the table. We then propose BiHMP, an efficient Bidirectional High-order Message-Passing network to aggregate global-local information and high-order relationships on the constructed hypergraph while capturing the inter-column heterogeneity and intra-column homogeneity. To effectively and efficiently align the capacity of the LLM with the information aggregated by BiHMP, we introduce Xfusion, which, together with BiHMP, acts as adapters for the LLM. We follow a pre-training and fine-tuning pipeline to train UnIMP, integrating two optimizations: chunking technique, which divides tables into smaller chunks to enhance efficiency; and progressive masking technique, which gradually adapts the model to learn more complex data patterns. Both theoretical proofs and empirical experiments on 10 real world datasets highlight the superiority of UnIMP over existing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02173v1">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in recommender systems for predicting Click-Through Rates (CTR) necessitates a delicate balance between computational efficiency and predictive accuracy. This paper presents an optimization framework that combines Retrieval-Augmented Generation (RAG) with an innovative multi-head early exit architecture to concurrently enhance both aspects. By integrating Graph Convolutional Networks (GCNs) as efficient retrieval mechanisms, we are able to significantly reduce data retrieval times while maintaining high model performance. The early exit strategy employed allows for dynamic termination of model inference, utilizing real-time predictive confidence assessments across multiple heads. This not only quickens the responsiveness of LLMs but also upholds or improves their accuracy, making it ideal for real-time application scenarios. Our experiments demonstrate how this architecture effectively decreases computation time without sacrificing the accuracy needed for reliable recommendation delivery, establishing a new standard for efficient, real-time LLM deployment in commercial systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14784v1">DeServe: Towards Affordable Offline LLM Inference via Decentralization</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      The rapid growth of generative AI and its integration into everyday workflows have significantly increased the demand for large language model (LLM) inference services. While proprietary models remain popular, recent advancements in open-source LLMs have positioned them as strong contenders. However, deploying these models is often constrained by the high costs and limited availability of GPU resources. In response, this paper presents the design of a decentralized offline serving system for LLM inference. Utilizing idle GPU resources, our proposed system, DeServe, decentralizes access to LLMs at a lower cost. DeServe specifically addresses key challenges in optimizing serving throughput in high-latency network environments. Experiments demonstrate that DeServe achieves a 6.7x-12.6x improvement in throughput over existing serving system baselines in such conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02152v1">Table as Thought: Exploring Structured Thoughts in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      Large language models' reasoning abilities benefit from methods that organize their thought processes, such as chain-of-thought prompting, which employs a sequential structure to guide the reasoning process step-by-step. However, existing approaches focus primarily on organizing the sequence of thoughts, leaving structure in individual thought steps underexplored. To address this gap, we propose Table as Thought, a framework inspired by cognitive neuroscience theories on human thought. Table as Thought organizes reasoning within a tabular schema, where rows represent sequential thought steps and columns capture critical constraints and contextual information to enhance reasoning. The reasoning process iteratively populates the table until self-verification ensures completeness and correctness. Our experiments show that Table as Thought excels in planning tasks and demonstrates a strong potential for enhancing LLM performance in mathematical reasoning compared to unstructured thought baselines. This work provides a novel exploration of refining thought representation within LLMs, paving the way for advancements in reasoning and AI cognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.02266v1">LLMzSzŁ: a comprehensive LLM benchmark for Polish</a></div>
    <div class="paper-meta">
      📅 2025-01-04
    </div>
    <details class="paper-abstract">
      This article introduces the first comprehensive benchmark for the Polish language at this scale: LLMzSzŁ (LLMs Behind the School Desk). It is based on a coherent collection of Polish national exams, including both academic and professional tests extracted from the archives of the Polish Central Examination Board. It covers 4 types of exams, coming from 154 domains. Altogether, it consists of almost 19k closed-ended questions. We investigate the performance of open-source multilingual, English, and Polish LLMs to verify LLMs' abilities to transfer knowledge between languages. Also, the correlation between LLMs and humans at model accuracy and exam pass rate levels is examined. We show that multilingual LLMs can obtain superior results over monolingual ones; however, monolingual models may be beneficial when model size matters. Our analysis highlights the potential of LLMs in assisting with exam validation, particularly in identifying anomalies or errors in examination tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02138v1">Effective LLM-Driven Code Generation with Pythoness</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has paved the way for a new era of programming tools with both significant capabilities and risks, as the generated code lacks guarantees of correctness and reliability. Developers using LLMs currently face the difficult task of optimizing, integrating, and maintaining code generated by AI. We propose an embedded domain-specific language (DSL), Pythoness, to address those challenges. In Pythoness, developers program with LLMs at a higher level of abstraction. Rather than interacting directly with generated code, developers using Pythoness operate at the level of behavioral specifications when writing functions, classes, or an entire program. These specifications can take the form of unit tests and property-based tests, which may be expressed formally or in natural language. Guided by these specifications, Pythoness generates code that both passes the tests and can be continuously checked during execution. We posit that the Pythoness approach lets developers harness the full potential of LLMs for code generation while substantially mitigating their inherent risks. We describe our current prototype implementation of Pythoness and demonstrate that it can successfully leverage a combination of tests and code generation to yield higher quality code than specifications alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04193v2">AL-QASIDA: Analyzing LLM Quality and Accuracy Systematically in Dialectal Arabic</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Dialectal Arabic (DA) varieties are under-served by language technologies, particularly large language models (LLMs). This trend threatens to exacerbate existing social inequalities and limits LLM applications, yet the research community lacks operationalized performance measurements in DA. We present a framework that comprehensively assesses LLMs' DA modeling capabilities across four dimensions: fidelity, understanding, quality, and diglossia. We evaluate nine LLMs in eight DA varieties and provide practical recommendations. Our evaluation suggests that LLMs do not produce DA as well as they understand it, not because their DA fluency is poor, but because they are reluctant to generate DA. Further analysis suggests that current post-training can contribute to bias against DA, that few-shot examples can overcome this deficiency, and that otherwise no measurable features of input text correlate well with LLM DA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01257v2">CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      With the increasing code reasoning capabilities of existing large language models (LLMs) and breakthroughs in reasoning models like OpenAI o1 and o3, there is a growing need to develop more challenging and comprehensive benchmarks that effectively test their sophisticated competition-level coding abilities. Existing benchmarks, like LiveCodeBench and USACO, fall short due to the unavailability of private test cases, lack of support for special judges, and misaligned execution environments. To bridge this gap, we introduce CodeElo, a standardized competition-level code generation benchmark that effectively addresses all these challenges for the first time. CodeElo benchmark is mainly based on the official CodeForces platform and tries to align with the platform as much as possible. We compile the recent six months of contest problems on CodeForces with detailed information such as contest divisions, problem difficulty ratings, and problem algorithm tags. We introduce a unique judging method in which problems are submitted directly to the platform and develop a reliable Elo rating calculation system that aligns with the platform and is comparable with human participants but has lower variance. By testing on our CodeElo, we provide the Elo ratings of 30 existing popular open-source and 3 proprietary LLMs for the first time. The results show that o1-mini and QwQ-32B-Preview stand out significantly, achieving Elo ratings of 1578 and 1261, respectively, while other models struggle even with the easiest problems, placing in the lowest 25 percent among all human participants. Detailed analysis experiments are also conducted to provide insights into performance across algorithms and comparisons between using C++ and Python, which can suggest directions for future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01849v1">Multi-Agent Conversational Online Learning for Adaptive LLM Response Identification</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      The remarkable generative capability of large language models (LLMs) has sparked a growing interest in automatically generating responses for different applications. Given the dynamic nature of user preferences and the uncertainty of LLM response performance, it is crucial to design efficient online learning algorithms to identify optimal LLM responses (i.e., high-quality responses that also meet user preferences). Most existing online algorithms adopt a centralized approach and fail to leverage explicit user preferences for more efficient and personalized LLM response identification. In contrast, this paper introduces \textit{MACO} (\underline{M}ulti-\underline{A}gent \underline{C}onversational \underline{O}nline Learning for Adaptive LLM Response Identification): 1) The online LLM response identification process is accelerated by multiple local agents (such as smartphones), while enhancing data privacy; 2) A novel conversational mechanism is proposed to adaptively conduct conversations for soliciting user preferences (e.g., a preference for a humorous tone over a serious one in generated responses), so to minimize uncertainty in preference estimation. Our theoretical analysis demonstrates that \cadi\ is near-optimal regarding cumulative regret. Additionally, \cadi\ offers reduced communication costs and computational complexity by eliminating the traditional, computing-intensive ``G-optimal design" found in previous works. Extensive experiments with the open LLM \textit{Llama}, coupled with two different embedding models from Google and OpenAI for text vector representation, demonstrate that \cadi\ significantly outperforms the current state-of-the-art in online LLM response identification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02039v1">An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01818v1">Rerouting LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      LLM routers aim to balance quality and cost of generation by classifying queries and routing them to a cheaper or more expensive LLM depending on their complexity. Routers represent one type of what we call LLM control planes: systems that orchestrate use of one or more LLMs. In this paper, we investigate routers' adversarial robustness. We first define LLM control plane integrity, i.e., robustness of LLM orchestration to adversarial inputs, as a distinct problem in AI safety. Next, we demonstrate that an adversary can generate query-independent token sequences we call ``confounder gadgets'' that, when added to any query, cause LLM routers to send the query to a strong LLM. Our quantitative evaluation shows that this attack is successful both in white-box and black-box settings against a variety of open-source and commercial routers, and that confounding queries do not affect the quality of LLM responses. Finally, we demonstrate that gadgets can be effective while maintaining low perplexity, thus perplexity-based filtering is not an effective defense. We finish by investigating alternative defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01792v1">Efficient LLM Inference with Activation Checkpointing and Hybrid Caching</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 14 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) with enormous model sizes use many GPUs to meet memory capacity requirements incurring substantial costs for token generation. To provide cost-effective LLM inference with relaxed latency constraints, extensive research has focused on expanding GPU memory by leveraging the host memory. However, LLM inference engines that utilize the host memory often face underutilization of GPU compute units, as a considerable portion of inference time is spent in loading the model onto the GPU via host-GPU interconnect. To tackle these challenges of the host memory offloading for LLM, we introduce HybridServe, an LLM inference system with activation checkpointing based on activation caching. The activation cache stores activation checkpoints generated during intermediate inference stages, allowing the fast recomputation of KV cache while model parameters are transferred to GPU from host memory. Unlike conventional methods that recompute the KV cache from scratch using token IDs, the activation cache allows bypassing projection and FFN operations. To balance between the activation recomputation and parameter loading overhead, this study proposes a KV-activation hybrid caching scheme which finds the best ratio of the key-value and activation caches to adjust the recomputation time. Our system achieves 2.19x throughput improvement over the state-of-the-art prior work for offloading both model weights and KV cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23242v2">A little less conversation, a little more action, please: Investigating the physical common-sense of LLMs in a 3D embodied environment</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 25 pages, 4 figures; v2: Added AFMR Acknowledgment
    </div>
    <details class="paper-abstract">
      As general-purpose tools, Large Language Models (LLMs) must often reason about everyday physical environments. In a question-and-answer capacity, understanding the interactions of physical objects may be necessary to give appropriate responses. Moreover, LLMs are increasingly used as reasoning engines in agentic systems, designing and controlling their action sequences. The vast majority of research has tackled this issue using static benchmarks, comprised of text or image-based questions about the physical world. However, these benchmarks do not capture the complexity and nuance of real-life physical processes. Here we advocate for a second, relatively unexplored, approach: 'embodying' the LLMs by granting them control of an agent within a 3D environment. We present the first embodied and cognitively meaningful evaluation of physical common-sense reasoning in LLMs. Our framework allows direct comparison of LLMs with other embodied agents, such as those based on Deep Reinforcement Learning, and human and non-human animals. We employ the Animal-AI (AAI) environment, a simulated 3D virtual laboratory, to study physical common-sense reasoning in LLMs. For this, we use the AAI Testbed, a suite of experiments that replicate laboratory studies with non-human animals, to study physical reasoning capabilities including distance estimation, tracking out-of-sight objects, and tool use. We demonstrate that state-of-the-art multi-modal models with no finetuning can complete this style of task, allowing meaningful comparison to the entrants of the 2019 Animal-AI Olympics competition and to human children. Our results show that LLMs are currently outperformed by human children on these tasks. We argue that this approach allows the study of physical reasoning using ecologically valid experiments drawn directly from cognitive science, improving the predictability and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01743v1">Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      Legal articles often include vague concepts to adapt to the ever-changing society. Providing detailed interpretations of these concepts is a critical task for legal practitioners, which requires meticulous and professional annotations by legal experts, admittedly time-consuming and expensive to collect at scale. In this paper, we introduce a novel retrieval-augmented generation framework, ATRI, for AuTomatically Retrieving relevant information from past judicial precedents and Interpreting vague legal concepts. We further propose a new benchmark, Legal Concept Entailment, to automate the evaluation of generated concept interpretations without expert involvement. Automatic evaluations indicate that our generated interpretations can effectively assist large language models (LLMs) in understanding vague legal concepts. Multi-faceted evaluations by legal experts indicate that the quality of our concept interpretations is comparable to those written by human experts. Our work has strong implications for leveraging LLMs to support legal practitioners in interpreting vague legal concepts and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01711v1">LLMs & Legal Aid: Understanding Legal Needs Exhibited Through User Queries</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Accepted at AI for Access to Justice Workshop at Jurix 2024, Brno, Czechia
    </div>
    <details class="paper-abstract">
      The paper presents a preliminary analysis of an experiment conducted by Frank Bold, a Czech expert group, to explore user interactions with GPT-4 for addressing legal queries. Between May 3, 2023, and July 25, 2023, 1,252 users submitted 3,847 queries. Unlike studies that primarily focus on the accuracy, factuality, or hallucination tendencies of large language models (LLMs), our analysis focuses on the user query dimension of the interaction. Using GPT-4o for zero-shot classification, we categorized queries on (1) whether users provided factual information about their issue (29.95%) or not (70.05%), (2) whether they sought legal information (64.93%) or advice on the course of action (35.07\%), and (3) whether they imposed requirements to shape or control the model's answer (28.57%) or not (71.43%). We provide both quantitative and qualitative insight into user needs and contribute to a better understanding of user engagement with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19878v3">HDMoLE: Mixture of LoRA Experts with Hierarchical Routing and Dynamic Thresholds for Fine-Tuning LLM-based ASR Models</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in integrating Large Language Models (LLM) with automatic speech recognition (ASR) have performed remarkably in general domains. While supervised fine-tuning (SFT) of all model parameters is often employed to adapt pre-trained LLM-based ASR models to specific domains, it imposes high computational costs and notably reduces their performance in general domains. In this paper, we propose a novel parameter-efficient multi-domain fine-tuning method for adapting pre-trained LLM-based ASR models to multi-accent domains without catastrophic forgetting named \textit{HDMoLE}, which leverages hierarchical routing and dynamic thresholds based on combining low-rank adaptation (LoRA) with the mixer of experts (MoE) and can be generalized to any linear layer. Hierarchical routing establishes a clear correspondence between LoRA experts and accent domains, improving cross-domain collaboration among the LoRA experts. Unlike the static Top-K strategy for activating LoRA experts, dynamic thresholds can adaptively activate varying numbers of LoRA experts at each MoE layer. Experiments on the multi-accent and standard Mandarin datasets demonstrate the efficacy of HDMoLE. Applying HDMoLE to an LLM-based ASR model projector module achieves similar performance to full fine-tuning in the target multi-accent domains while using only 9.6% of the trainable parameters required for full fine-tuning and minimal degradation in the source general domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01668v1">CoT-based Synthesizer: Enhancing LLM Performance through Answer Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      Current inference scaling methods, such as Self-consistency and Best-of-N, have proven effective in improving the accuracy of LLMs on complex reasoning tasks. However, these methods rely heavily on the quality of candidate responses and are unable to produce correct answers when all candidates are incorrect. In this paper, we propose a novel inference scaling strategy, CoT-based Synthesizer, which leverages CoT reasoning to synthesize superior answers by analyzing complementary information from multiple candidate responses, even when all candidate responses are flawed. To enable a lightweight and cost-effective implementation, we introduce an automated data generation pipeline that creates diverse training data. This allows smaller LLMs trained on this data to improve the inference accuracy of larger models, including API-based LLMs. Experimental results across four benchmark datasets with seven policy models demonstrate that our method significantly enhances performance, with gains of 11.8% for Llama3-8B and 10.3% for GPT-4o on the MATH dataset. The corresponding training data and code are publicly available on https://github.com/RUCKBReasoning/CoT-based-Synthesizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01664v1">BARTPredict: Empowering IoT Security with LLM-Driven Cyber Threat Prediction</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      The integration of Internet of Things (IoT) technology in various domains has led to operational advancements, but it has also introduced new vulnerabilities to cybersecurity threats, as evidenced by recent widespread cyberattacks on IoT devices. Intrusion detection systems are often reactive, triggered by specific patterns or anomalies observed within the network. To address this challenge, this work proposes a proactive approach to anticipate and preemptively mitigate malicious activities, aiming to prevent potential damage before it occurs. This paper proposes an innovative intrusion prediction framework empowered by Pre-trained Large Language Models (LLMs). The framework incorporates two LLMs: a fine-tuned Bidirectional and AutoRegressive Transformers (BART) model for predicting network traffic and a fine-tuned Bidirectional Encoder Representations from Transformers (BERT) model for evaluating the predicted traffic. By harnessing the bidirectional capabilities of BART the framework then identifies malicious packets among these predictions. Evaluated using the CICIoT2023 IoT attack dataset, our framework showcases a notable enhancement in predictive performance, attaining an impressive 98% overall accuracy, providing a powerful response to the cybersecurity challenges that confront IoT networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00430v2">Enhancing LLM Reasoning with Multi-Path Collaborative Reactive and Reflection agents</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      Agents have demonstrated their potential in scientific reasoning tasks through large language models. However, they often face challenges such as insufficient accuracy and degeneration of thought when handling complex reasoning tasks, which impede their performance. To overcome these issues, we propose the Reactive and Reflection agents with Multi-Path Reasoning (RR-MP) Framework, aimed at enhancing the reasoning capabilities of LLMs. Our approach improves scientific reasoning accuracy by employing a multi-path reasoning mechanism where each path consists of a reactive agent and a reflection agent that collaborate to prevent degeneration of thought inherent in single-agent reliance. Additionally, the RR-MP framework does not require additional training; it utilizes multiple dialogue instances for each reasoning path and a separate summarizer to consolidate insights from all paths. This design integrates diverse perspectives and strengthens reasoning across each path. We conducted zero-shot and few-shot evaluations on tasks involving moral scenarios, college-level physics, and mathematics. Experimental results demonstrate that our method outperforms baseline approaches, highlighting the effectiveness and advantages of the RR-MP framework in managing complex scientific reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15221v2">Reflections from the 2024 Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 Updating author information, the submission remains largely unchanged. 98 pages total
    </div>
    <details class="paper-abstract">
      Here, we present the outcomes from the second Large Language Model (LLM) Hackathon for Applications in Materials Science and Chemistry, which engaged participants across global hybrid locations, resulting in 34 team submissions. The submissions spanned seven key application areas and demonstrated the diverse utility of LLMs for applications in (1) molecular and material property prediction; (2) molecular and material design; (3) automation and novel interfaces; (4) scientific communication and education; (5) research data management and automation; (6) hypothesis generation and evaluation; and (7) knowledge extraction and reasoning from scientific literature. Each team submission is presented in a summary table with links to the code and as brief papers in the appendix. Beyond team results, we discuss the hackathon event and its hybrid format, which included physical hubs in Toronto, Montreal, San Francisco, Berlin, Lausanne, and Tokyo, alongside a global online hub to enable local and virtual collaboration. Overall, the event highlighted significant improvements in LLM capabilities since the previous year's hackathon, suggesting continued expansion of LLMs for applications in materials science and chemistry research. These outcomes demonstrate the dual utility of LLMs as both multipurpose models for diverse machine learning tasks and platforms for rapid prototyping custom applications in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16833v2">KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-01-03
      | 💬 10 pages,5 figures,published to AAAI-25 Bridge Program
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) in healthcare diagnosis demands systematic frameworks that can handle complex medical scenarios while maintaining specialized expertise. We present KG4Diagnosis, a novel hierarchical multi-agent framework that combines LLMs with automated knowledge graph construction, encompassing 362 common diseases across medical specialties. Our framework mirrors real-world medical systems through a two-tier architecture: a general practitioner (GP) agent for initial assessment and triage, coordinating with specialized agents for in-depth diagnosis in specific domains. The core innovation lies in our end-to-end knowledge graph generation methodology, incorporating: (1) semantic-driven entity and relation extraction optimized for medical terminology, (2) multi-dimensional decision relationship reconstruction from unstructured medical texts, and (3) human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an extensible foundation for specialized medical diagnosis systems, with capabilities to incorporate new diseases and medical knowledge. The framework's modular design enables seamless integration of domain-specific enhancements, making it valuable for developing targeted medical diagnosis systems. We provide architectural guidelines and protocols to facilitate adoption across medical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01818v1">Rerouting LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-01-03
    </div>
    <details class="paper-abstract">
      LLM routers aim to balance quality and cost of generation by classifying queries and routing them to a cheaper or more expensive LLM depending on their complexity. Routers represent one type of what we call LLM control planes: systems that orchestrate use of one or more LLMs. In this paper, we investigate routers' adversarial robustness. We first define LLM control plane integrity, i.e., robustness of LLM orchestration to adversarial inputs, as a distinct problem in AI safety. Next, we demonstrate that an adversary can generate query-independent token sequences we call ``confounder gadgets'' that, when added to any query, cause LLM routers to send the query to a strong LLM. Our quantitative evaluation shows that this attack is successful both in white-box and black-box settings against a variety of open-source and commercial routers, and that confounding queries do not affect the quality of LLM responses. Finally, we demonstrate that gadgets can be effective while maintaining low perplexity, thus perplexity-based filtering is not an effective defense. We finish by investigating alternative defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01558v1">Predicting the Performance of Black-box LLMs through Self-Queries</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01545v1">Enhancing User Engagement in Large-Scale Social Annotation Platforms: Community-Based Design Interventions and Implications for Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Social annotation platforms enable student engagement by integrating discussions directly into course materials. However, in large online courses, the sheer volume of comments can overwhelm students and impede learning. This paper investigates community-based design interventions on a social annotation platform (NB) to address this challenge and foster more meaningful online educational discussions. By examining student preferences and reactions to different curation strategies, this research aims to optimize the utility of social annotations in educational contexts. A key emphasis is placed on how the visibility of comments shapes group interactions, guides conversational flows, and enriches learning experiences. The study combined iterative design and development with two large-scale experiments to create and refine comment curation strategies, involving thousands of students. The study introduced specific features of the platform, such as targeted comment visibility controls, which demonstrably improved peer interactions and reduced discussion overload. These findings inform the design of next-generation social annotation systems and highlight opportunities to integrate Large Language Models (LLMs) for key activities like summarizing annotations, improving clarity in student writing, and assisting instructors with efficient comment curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01329v1">The Prompt Alchemist: Automated LLM-Tailored Prompt Optimization for Test Case Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Test cases are essential for validating the reliability and quality of software applications. Recent studies have demonstrated the capability of Large Language Models (LLMs) to generate useful test cases for given source code. However, the existing work primarily relies on human-written plain prompts, which often leads to suboptimal results since the performance of LLMs can be highly influenced by the prompts. Moreover, these approaches use the same prompt for all LLMs, overlooking the fact that different LLMs might be best suited to different prompts. Given the wide variety of possible prompt formulations, automatically discovering the optimal prompt for each LLM presents a significant challenge. Although there are methods on automated prompt optimization in the natural language processing field, they are hard to produce effective prompts for the test case generation task. First, the methods iteratively optimize prompts by simply combining and mutating existing ones without proper guidance, resulting in prompts that lack diversity and tend to repeat the same errors in the generated test cases. Second, the prompts are generally lack of domain contextual knowledge, limiting LLMs' performance in the task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01887v3">Beyond Numeric Awards: In-Context Dueling Bandits with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      In-context reinforcement learning (ICRL) is a frontier paradigm for solving reinforcement learning problems in the foundation model era. While ICRL capabilities have been demonstrated in transformers through task-specific training, the potential of Large Language Models (LLMs) out-of-the-box remains largely unexplored. Recent findings highlight that LLMs often face challenges when dealing with numerical contexts, and limited attention has been paid to evaluating their performance through preference feedback generated by the environment. This paper is the first to investigate LLMs as in-context decision-makers under the problem of Dueling Bandits (DB), a stateless preference-based reinforcement learning setting that extends the classic Multi-Armed Bandit (MAB) model by querying for preference feedback. We compare GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, Llama 3.1, and o1-Preview against nine well-established DB algorithms. Our results reveal that our top-performing LLM, GPT-4 Turbo, has the zero-shot relative decision-making ability to achieve surprisingly low weak regret across all the DB environment instances by quickly including the best arm in duels. However, an optimality gap exists between LLMs and classic DB algorithms in terms of strong regret. LLMs struggle to converge and consistently exploit even when explicitly prompted to do so, and are sensitive to prompt variations. To bridge this gap, we propose an agentic flow framework: LLM with Enhanced Algorithmic Dueling (LEAD), which integrates off-the-shelf DB algorithms with LLM agents through fine-grained adaptive interplay. We show that LEAD has theoretical guarantees inherited from classic DB algorithms on both weak and strong regret. We validate its efficacy and robustness even with noisy and adversarial prompts. The design of our framework sheds light on how to enhance the trustworthiness of LLMs used for in-context decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11006v4">Security Attacks on LLM-based Code Completion Tools</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 Paper accepted at AAAI 2025
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has significantly advanced code completion capabilities, giving rise to a new generation of LLM-based Code Completion Tools (LCCTs). Unlike general-purpose LLMs, these tools possess unique workflows, integrating multiple information sources as input and prioritizing code suggestions over natural language interaction, which introduces distinct security challenges. Additionally, LCCTs often rely on proprietary code datasets for training, raising concerns about the potential exposure of sensitive data. This paper exploits these distinct characteristics of LCCTs to develop targeted attack methodologies on two critical security risks: jailbreaking and training data extraction attacks. Our experimental results expose significant vulnerabilities within LCCTs, including a 99.4% success rate in jailbreaking attacks on GitHub Copilot and a 46.3% success rate on Amazon Q. Furthermore, We successfully extracted sensitive user data from GitHub Copilot, including 54 real email addresses and 314 physical addresses associated with GitHub usernames. Our study also demonstrates that these code-based attack methods are effective against general-purpose LLMs, such as the GPT series, highlighting a broader security misalignment in the handling of code by modern LLMs. These findings underscore critical security challenges associated with LCCTs and suggest essential directions for strengthening their security frameworks. The example code and attack samples from our research are provided at https://github.com/Sensente/Security-Attacks-on-LCCTs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01237v1">Automated Self-Refinement and Self-Correction for LLM-based Product Attribute Value Extraction</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Structured product data, in the form of attribute-value pairs, is essential for e-commerce platforms to support features such as faceted product search and attribute-based product comparison. However, vendors often provide unstructured product descriptions, making attribute value extraction necessary to ensure data consistency and usability. Large language models (LLMs) have demonstrated their potential for product attribute value extraction in few-shot scenarios. Recent research has shown that self-refinement techniques can improve the performance of LLMs on tasks such as code generation and text-to-SQL translation. For other tasks, the application of these techniques has resulted in increased costs due to processing additional tokens, without achieving any improvement in performance. This paper investigates applying two self-refinement techniques, error-based prompt rewriting and self-correction, to the product attribute value extraction task. The self-refinement techniques are evaluated across zero-shot, few-shot in-context learning, and fine-tuning scenarios using GPT-4o. The experiments show that both self-refinement techniques have only a marginal impact on the model's performance across the different scenarios, while significantly increasing processing costs. For scenarios with training data, fine-tuning yields the highest performance, while the ramp-up costs of fine-tuning are balanced out as the amount of product descriptions increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03259v1">Toward Inclusive Educational AI: Auditing Frontier LLMs through a Multiplexity Lens</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) like GPT-4 and Llama 3 become integral to educational contexts, concerns are mounting over the cultural biases, power imbalances, and ethical limitations embedded within these technologies. Though generative AI tools aim to enhance learning experiences, they often reflect values rooted in Western, Educated, Industrialized, Rich, and Democratic (WEIRD) cultural paradigms, potentially sidelining diverse global perspectives. This paper proposes a framework to assess and mitigate cultural bias within LLMs through the lens of applied multiplexity. Multiplexity, inspired by Senturk et al. and rooted in Islamic and other wisdom traditions, emphasizes the coexistence of diverse cultural viewpoints, supporting a multi-layered epistemology that integrates both empirical sciences and normative values. Our analysis reveals that LLMs frequently exhibit cultural polarization, with biases appearing in both overt responses and subtle contextual cues. To address inherent biases and incorporate multiplexity in LLMs, we propose two strategies: \textit{Contextually-Implemented Multiplex LLMs}, which embed multiplex principles directly into the system prompt, influencing LLM outputs at a foundational level and independent of individual prompts, and \textit{Multi-Agent System (MAS)-Implemented Multiplex LLMs}, where multiple LLM agents, each representing distinct cultural viewpoints, collaboratively generate a balanced, synthesized response. Our findings demonstrate that as mitigation strategies evolve from contextual prompting to MAS-implementation, cultural inclusivity markedly improves, evidenced by a significant rise in the Perspectives Distribution Score (PDS) and a PDS Entropy increase from 3.25\% at baseline to 98\% with the MAS-Implemented Multiplex LLMs. Sentiment analysis further shows a shift towards positive sentiment across cultures,...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01205v1">Harnessing Multi-Agent LLMs for Complex Engineering Problem-Solving: A Framework for Senior Design Projects</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Multi-Agent Large Language Models (LLMs) are gaining significant attention for their ability to harness collective intelligence in complex problem-solving, decision-making, and planning tasks. This aligns with the concept of the wisdom of crowds, where diverse agents contribute collectively to generating effective solutions, making it particularly suitable for educational settings. Senior design projects, also known as capstone or final year projects, are pivotal in engineering education as they integrate theoretical knowledge with practical application, fostering critical thinking, teamwork, and real-world problem-solving skills. In this paper, we explore the use of Multi-Agent LLMs in supporting these senior design projects undertaken by engineering students, which often involve multidisciplinary considerations and conflicting objectives, such as optimizing technical performance while addressing ethical, social, and environmental concerns. We propose a framework where distinct LLM agents represent different expert perspectives, such as problem formulation agents, system complexity agents, societal and ethical agents, or project managers, thus facilitating a holistic problem-solving approach. This implementation leverages standard multi-agent system (MAS) concepts such as coordination, cooperation, and negotiation, incorporating prompt engineering to develop diverse personas for each agent. These agents engage in rich, collaborative dialogues to simulate human engineering teams, guided by principles from swarm AI to efficiently balance individual contributions towards a unified solution. We adapt these techniques to create a collaboration structure for LLM agents, encouraging interdisciplinary reasoning and negotiation similar to real-world senior design projects. To assess the efficacy of this framework, we collected six proposals of engineering and computer science of...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01203v1">HetGCoT-Rec: Heterogeneous Graph-Enhanced Chain-of-Thought LLM Reasoning for Journal Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Academic journal recommendation requires effectively combining structural understanding of scholarly networks with interpretable recommendations. While graph neural networks (GNNs) and large language models (LLMs) excel in their respective domains, current approaches often fail to achieve true integration at the reasoning level. We propose HetGCoT-Rec, a framework that deeply integrates heterogeneous graph transformer with LLMs through chain-of-thought reasoning. Our framework features two key technical innovations: (1) a structure-aware mechanism that transforms heterogeneous graph neural network learned subgraph information into natural language contexts, utilizing predefined metapaths to capture academic relationships, and (2) a multi-step reasoning strategy that systematically embeds graph-derived contexts into the LLM's stage-wise reasoning process. Experiments on a dataset collected from OpenAlex demonstrate that our approach significantly outperforms baseline methods, achieving 96.48% Hit rate and 92.21% H@1 accuracy. Furthermore, we validate the framework's adaptability across different LLM architectures, showing consistent improvements in both recommendation accuracy and explanation quality. Our work demonstrates an effective approach for combining graph-structured reasoning with language models for interpretable academic venue recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20367v2">Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLM), reinforcement learning (RL) has emerged as a pivotal technique for code generation and optimization in various domains. This paper presents a systematic survey of the application of RL in code optimization and generation, highlighting its role in enhancing compiler optimization, resource allocation, and the development of frameworks and tools. Subsequent sections first delve into the intricate processes of compiler optimization, where RL algorithms are leveraged to improve efficiency and resource utilization. The discussion then progresses to the function of RL in resource allocation, emphasizing register allocation and system optimization. We also explore the burgeoning role of frameworks and tools in code generation, examining how RL can be integrated to bolster their capabilities. This survey aims to serve as a comprehensive resource for researchers and practitioners interested in harnessing the power of RL to advance code generation and optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00927v2">Text Clustering as Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Text clustering remains valuable in real-world applications where manual labeling is cost-prohibitive. It facilitates efficient organization and analysis of information by grouping similar texts based on their representations. However, implementing this approach necessitates fine-tuned embedders for downstream data and sophisticated similarity metrics. To address this issue, this study presents a novel framework for text clustering that effectively leverages the in-context learning capacity of Large Language Models (LLMs). Instead of fine-tuning embedders, we propose to transform the text clustering into a classification task via LLM. First, we prompt LLM to generate potential labels for a given dataset. Second, after integrating similar labels generated by the LLM, we prompt the LLM to assign the most appropriate label to each sample in the dataset. Our framework has been experimentally proven to achieve comparable or superior performance to state-of-the-art clustering methods that employ embeddings, without requiring complex fine-tuning or clustering algorithms. We make our code available to the public for utilization at https://github.com/ECNU-Text-Computing/Text-Clustering-via-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01985v2">Lost-in-Distance: Impact of Contextual Proximity on LLM Performance in Graph Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-02
    </div>
    <details class="paper-abstract">
      Despite significant advancements, Large Language Models (LLMs) exhibit blind spots that impair their ability to retrieve and process relevant contextual data effectively. We demonstrate that LLM performance in graph tasks with complexities beyond the "needle-in-a-haystack" scenario-where solving the problem requires cross-referencing and reasoning across multiple subproblems jointly-is influenced by the proximity of relevant information within the context, a phenomenon we term "lost-in-distance". We examine two fundamental graph tasks: identifying common connections between two nodes and assessing similarity among three nodes, and show that the model's performance in these tasks significantly depends on the relative positioning of common edges. We evaluate three publicly available LLMs using various graph encoding techniques that represent graph structures for LLM input. We propose a formulation for the lost-in-distance phenomenon and demonstrate that lost-in-distance and lost-in-the middle phenomenas occur independently. Results indicate that model accuracy can decline by up to 6x as the distance between node connections increases, independent of graph encoding and model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08877v4">Aligning the Objective of LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 Accepted by ICSE'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved decent results on automated program repair (APR). However, the next token prediction training objective of decoder-only LLMs (e.g., GPT-4) is misaligned with the masked span prediction objective of current infilling-style methods, which impedes LLMs from fully leveraging pre-trained knowledge for program repair. In addition, while some LLMs can locate and repair bugs in certain functions using the related artifacts (e.g., test cases), existing methods still depend on statement-level fault localization methods to provide a list of buggy hunks for repair. This restriction hinders LLMs from exploring potential patches beyond the given locations. In this paper, we investigate a new approach to adapt LLMs to program repair. Our core insight is that LLM's APR capability can be greatly improved by simply aligning the output to their training objective and allowing them to refine the whole program without first identifying faulty statements. Based on this insight, we designed D4C, a straightforward prompting framework for APR. D4C can repair 180 bugs correctly in Defects4J, with each patch being sampled only 10 times. This surpasses the SOTA APR methods with perfect fault localization by 10% and reduces the patch sampling number by 90%. Our findings reveal that (1) objective alignment is crucial for fully exploiting LLM's pre-trained capability, and (2) replacing the traditional localize-buggy-hunks-then-repair workflow with direct debugging is more effective for LLM-based APR methods. Thus, we believe this paper introduces a new mindset for harnessing LLMs in APR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01005v1">FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-01-02
      | 💬 code available at http://github.com/flashinfer-ai/flashinfer
    </div>
    <details class="paper-abstract">
      Transformers, driven by attention mechanisms, form the foundation of large language models (LLMs). As these models scale up, efficient GPU attention kernels become essential for high-throughput and low-latency inference. Diverse LLM applications demand flexible and high-performance attention solutions. We present FlashInfer: a customizable and efficient attention engine for LLM serving. FlashInfer tackles KV-cache storage heterogeneity using block-sparse format and composable formats to optimize memory access and reduce redundancy. It also offers a customizable attention template, enabling adaptation to various settings through Just-In-Time (JIT) compilation. Additionally, FlashInfer's load-balanced scheduling algorithm adjusts to dynamism of user requests while maintaining compatibility with CUDAGraph which requires static configuration. FlashInfer have been integrated into leading LLM serving frameworks like SGLang, vLLM and MLC-Engine. Comprehensive kernel-level and end-to-end evaluations demonstrate FlashInfer's ability to significantly boost kernel performance across diverse inference scenarios: compared to state-of-the-art LLM serving solutions, FlashInfer achieve 29-69% inter-token-latency reduction compared to compiler backends for LLM serving benchmark, 28-30% latency reduction for long-context inference, and 13-17% speedup for LLM serving with parallel generation.
    </details>
</div>
