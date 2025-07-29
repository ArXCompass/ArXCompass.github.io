# llm - 2025_07

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
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18417v1">FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Opinions expressed in online finance-related textual data are having an increasingly profound impact on trading decisions and market movements. This trend highlights the vital role of sentiment analysis as a tool for quantifying the nature and strength of such opinions. With the rapid development of Generative AI (GenAI), supervised fine-tuned (SFT) large language models (LLMs) have become the de facto standard for financial sentiment analysis. However, the SFT paradigm can lead to memorization of the training data and often fails to generalize to unseen samples. This is a critical limitation in financial domains, where models must adapt to previously unobserved events and the nuanced, domain-specific language of finance. To this end, we introduce FinDPO, the first finance-specific LLM framework based on post-training human preference alignment via Direct Preference Optimization (DPO). The proposed FinDPO achieves state-of-the-art performance on standard sentiment classification benchmarks, outperforming existing supervised fine-tuned models by 11% on the average. Uniquely, the FinDPO framework enables the integration of a fine-tuned causal LLM into realistic portfolio strategies through a novel 'logit-to-score' conversion, which transforms discrete sentiment predictions into continuous, rankable sentiment scores (probabilities). In this way, simulations demonstrate that FinDPO is the first sentiment-based approach to maintain substantial positive returns of 67% annually and strong risk-adjusted performance, as indicated by a Sharpe ratio of 2.0, even under realistic transaction costs of 5 basis points (bps).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18392v1">CLEAR: Error Analysis via LLM-as-a-Judge Made Easy</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18391v1">Revisiting LLM Reasoning via Information Bottleneck</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable progress in reasoning capabilities through reinforcement learning with verifiable rewards (RLVR). By leveraging simple rule-based rewards, RL effectively incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning trajectories, progressively guiding them toward correct answers. However, existing approaches remain largely heuristic and intuition-driven, limiting the development of principled methodologies. In this paper, we present a theoretical characterization of LLM reasoning grounded in information bottleneck (IB) principle, introducing IB-aware reasoning optimization (IBRO), a framework that encourages reasoning trajectories to be both informative about the final correct answer and generalizable across diverse prompts. We derive a practical token-level surrogate objective and propose an efficient approximation, resulting in the lightweight IB regularization method. This technique integrates seamlessly into existing RL-based post-training frameworks without additional computational overhead, requiring only a one-line code modification. Empirically, we validate IB regularization across multiple mathematical reasoning benchmarks and RL algorithms, demonstrating consistent improvements in LLM reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18368v1">Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by Agentic & GenAI Evaluation KDD2025: KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/
    </div>
    <details class="paper-abstract">
      Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step logic. In finance, however, professionals must not only converge on optimal decisions but also generate creative, plausible futures under uncertainty. We introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent thinking in LLMs for financial tasks. ConDiFi features 607 macro-financial prompts for divergent reasoning and 990 multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we evaluated 14 leading models and uncovered striking differences. Despite high fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models like DeepSeek-R1 and Cohere Command R+ rank among the top for generating actionable, insights suitable for investment decisions. ConDiFi provides a new perspective to assess reasoning capabilities essential to safe and strategic deployment of LLMs in finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18366v1">Efficient Uncertainty in LLMs through Evidential Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification remains a key challenge for standard LLMs, prompting the adoption of Bayesian and ensemble-based methods. However, such methods typically necessitate computationally expensive sampling, involving multiple forward passes to effectively estimate predictive uncertainty. In this paper, we introduce a novel approach enabling efficient and effective uncertainty estimation in LLMs without sacrificing performance. Specifically, we distill uncertainty-aware teacher models - originally requiring multiple forward passes - into compact student models sharing the same architecture but fine-tuned using Low-Rank Adaptation (LoRA). We compare two distinct distillation strategies: one in which the student employs traditional softmax-based outputs, and another in which the student leverages Dirichlet-distributed outputs to explicitly model epistemic uncertainty via evidential learning. Empirical evaluations on classification datasets demonstrate that such students can achieve comparable or superior predictive and uncertainty quantification performance relative to their teacher models, while critically requiring only a single forward pass. To our knowledge, this is the first demonstration that immediate and robust uncertainty quantification can be achieved in LLMs through evidential distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12988v3">Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought patterns as manifested in the textual works of a character. Using Lu Xun, a renowned Chinese writer as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope this work inspires future research on deep character persona simulation LLMs while considering the importance of ethical standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18343v1">Hybrid Annotation for Propaganda Detection: Integrating LLM Pre-Annotations with Human Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 NLP4PI at ACL
    </div>
    <details class="paper-abstract">
      Propaganda detection on social media remains challenging due to task complexity and limited high-quality labeled data. This paper introduces a novel framework that combines human expertise with Large Language Model (LLM) assistance to improve both annotation consistency and scalability. We propose a hierarchical taxonomy that organizes 14 fine-grained propaganda techniques into three broader categories, conduct a human annotation study on the HQP dataset that reveals low inter-annotator agreement for fine-grained labels, and implement an LLM-assisted pre-annotation pipeline that extracts propagandistic spans, generates concise explanations, and assigns local labels as well as a global label. A secondary human verification study shows significant improvements in both agreement and time-efficiency. Building on this, we fine-tune smaller language models (SLMs) to perform structured annotation. Instead of fine-tuning on human annotations, we train on high-quality LLM-generated data, allowing a large model to produce these annotations and a smaller model to learn to generate them via knowledge distillation. Our work contributes towards the development of scalable and robust propaganda detection systems, supporting the idea of transparent and accountable media ecosystems in line with SDG 16. The code is publicly available at our GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18340v1">TDR: Task-Decoupled Retrieval with Fine-Grained LLM Feedback for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) has become a classic approach for enabling LLMs to handle various tasks based on a few input-output examples. The effectiveness of ICL heavily relies on the quality of these examples, and previous works which focused on enhancing example retrieval capabilities have achieved impressive performances. However, two challenges remain in retrieving high-quality examples: (1) Difficulty in distinguishing cross-task data distributions, (2) Difficulty in making the fine-grained connection between retriever output and feedback from LLMs. In this paper, we propose a novel framework called TDR. TDR decouples the ICL examples from different tasks, which enables the retrieval module to retrieve examples specific to the target task within a multi-task dataset. Furthermore, TDR models fine-grained feedback from LLMs to supervise and guide the training of the retrieval module, which helps to retrieve high-quality examples. We conducted extensive experiments on a suite of 30 NLP tasks, the results demonstrate that TDR consistently improved results across all datasets and achieves state-of-the-art performance. Meanwhile, our approach is a plug-and-play method, which can be easily combined with various LLMs to improve example retrieval abilities for ICL. The code is available at https://github.com/Nnn-s/TDR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08621v2">A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18316v1">YATE: The Role of Test Repair in LLM-Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advances in automated test generation utilises language models to produce unit tests. While effective, language models tend to generate many incorrect tests with respect to both syntax and semantics. Although such incorrect tests can be easily detected and discarded, they constitute a "missed opportunity" -- if fixed, they are often valuable as they directly add testing value (they effectively target the underlying program logic to be tested) and indirectly form good seeds for generating additional tests. To this end, we propose a simple technique for repairing some of these incorrect tests through a combination of rule-based static analysis and re-prompting. We evaluate this simple approach, named YATE, on a set of 6 open-source projects and show that it can effectively produce tests that cover on average 32.06% more lines and kill 21.77% more mutants than a plain LLM-based method. We also compare YATE with four other LLM-based methods, namely HITS, SYMPROMPT, TESTSPARK and COVERUP and show that it produces tests that cover substantially more code. YATE achieves 22% higher line coverage, 20% higher branch coverage and kill 20% more mutants at a comparable cost (number of calls to LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12972v2">Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 14 pages, 7 figures, 6 tables; Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10009v2">OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problems with Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 8 pages, 12 figures
    </div>
    <details class="paper-abstract">
      With the rise of artificial intelligence (AI), applying large language models (LLMs) to Operations Research (OR) problem-solving has attracted increasing attention. Most existing approaches attempt to improve OR problem-solving through prompt engineering or fine-tuning strategies for LLMs. However, these methods are fundamentally constrained by the limited capabilities of non-reasoning LLMs. To overcome these limitations, we propose OR-LLM-Agent, an AI agent built on reasoning LLMs for automated OR problem solving. The agent decomposes the task into three sequential stages: mathematical modeling, code generation, and debugging. Each task is handled by a dedicated sub-agent, which enables more targeted reasoning. We also construct BWOR, a high-quality dataset for evaluating LLM performance on OR tasks. Our analysis shows that existing benchmarks such as NL4OPT, MAMO, and IndustryOR suffer from certain issues, making them less suitable for reliably evaluating LLM performance. In contrast, BWOR provides a more consistent and discriminative assessment of model capabilities. Experimental results demonstrate that OR-LLM-Agent outperforms advanced methods, including GPT-o3, Gemini 2.5 Pro, and ORLM, by at least 7% in accuracy. These results demonstrate the effectiveness of task decomposition for OR problem solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19795v2">VolDoGer: LLM-assisted Datasets for Domain Generalization in Vision-Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL)
    </div>
    <details class="paper-abstract">
      Domain generalizability is a crucial aspect of a deep learning model since it determines the capability of the model to perform well on data from unseen domains. However, research on the domain generalizability of deep learning models for vision-language tasks remains limited, primarily because of the lack of required datasets. To address these challenges, we propose VolDoGer: Vision-Language Dataset for Domain Generalization, a dedicated dataset designed for domain generalization that addresses three vision-language tasks: image captioning, visual question answering, and visual entailment. We constructed VolDoGer by extending LLM-based data annotation techniques to vision-language tasks, thereby alleviating the burden of recruiting human annotators. We evaluated the domain generalizability of various models, ranging from fine-tuned models to a recent multimodal large language model, through VolDoGer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18252v1">Multimodal Behavioral Patterns Analysis with Eye-Tracking and LLM-Based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Eye-tracking data reveals valuable insights into users' cognitive states but is difficult to analyze due to its structured, non-linguistic nature. While large language models (LLMs) excel at reasoning over text, they struggle with temporal and numerical data. This paper presents a multimodal human-AI collaborative framework designed to enhance cognitive pattern extraction from eye-tracking signals. The framework includes: (1) a multi-stage pipeline using horizontal and vertical segmentation alongside LLM reasoning to uncover latent gaze patterns; (2) an Expert-Model Co-Scoring Module that integrates expert judgment with LLM output to generate trust scores for behavioral interpretations; and (3) a hybrid anomaly detection module combining LSTM-based temporal modeling with LLM-driven semantic analysis. Our results across several LLMs and prompt strategies show improvements in consistency, interpretability, and performance, with up to 50% accuracy in difficulty prediction tasks. This approach offers a scalable, interpretable solution for cognitive modeling and has broad potential in adaptive learning, human-computer interaction, and educational analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18215v1">Information Security Based on LLM Approaches: A Review</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Information security is facing increasingly severe challenges, and traditional protection means are difficult to cope with complex and changing threats. In recent years, as an emerging intelligent technology, large language models (LLMs) have shown a broad application prospect in the field of information security. In this paper, we focus on the key role of LLM in information security, systematically review its application progress in malicious behavior prediction, network threat analysis, system vulnerability detection, malicious code identification, and cryptographic algorithm optimization, and explore its potential in enhancing security protection performance. Based on neural networks and Transformer architecture, this paper analyzes the technical basis of large language models and their advantages in natural language processing tasks. It is shown that the introduction of large language modeling helps to improve the detection accuracy and reduce the false alarm rate of security systems. Finally, this paper summarizes the current application results and points out that it still faces challenges in model transparency, interpretability, and scene adaptability, among other issues. It is necessary to explore further the optimization of the model structure and the improvement of the generalization ability to realize a more intelligent and accurate information security protection system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18212v1">Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Layer pruning has emerged as a promising technique for compressing large language models (LLMs) while achieving acceleration proportional to the pruning ratio. In this work, we identify that removing any layer induces a significant magnitude gap in hidden states, resulting in substantial performance degradation. To address this issue, we propose Prune&Comp, a novel plug-and-play layer pruning scheme that leverages magnitude compensation to mitigate such gaps in a training-free manner. Specifically, we first estimate the magnitude gap caused by layer removal and then eliminate this gap by rescaling the remaining weights offline, with zero runtime overhead incurred. We further demonstrate the advantages of Prune&Comp through an iterative pruning strategy. When integrated with an iterative prune-and-compensate loop, Prune&Comp consistently enhances existing layer pruning metrics. For instance, when 5 layers of LLaMA-3-8B are pruned using the prevalent block influence metric, Prune&Comp nearly halves the perplexity and retains 93.19\% of the original model's question-answering performance, outperforming the baseline by 4.01%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20658v2">Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 11 pages, 5 figures, published to ACL 2025
    </div>
    <details class="paper-abstract">
      Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18203v1">Exploring the Impact of Instruction-Tuning on LLM's Susceptibility to Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ACL 2025 Main Accepted
    </div>
    <details class="paper-abstract">
      Instruction-tuning enhances the ability of large language models (LLMs) to follow user instructions more accurately, improving usability while reducing harmful outputs. However, this process may increase the model's dependence on user input, potentially leading to the unfiltered acceptance of misinformation and the generation of hallucinations. Existing studies primarily highlight that LLMs are receptive to external information that contradict their parametric knowledge, but little research has been conducted on the direct impact of instruction-tuning on this phenomenon. In our study, we investigate the impact of instruction-tuning on LLM's susceptibility to misinformation. Our analysis reveals that instruction-tuned LLMs are significantly more likely to accept misinformation when it is presented by the user. A comparison with base models shows that instruction-tuning increases reliance on user-provided information, shifting susceptibility from the assistant role to the user role. Furthermore, we explore additional factors influencing misinformation susceptibility, such as the role of the user in prompt structure, misinformation length, and the presence of warnings in the system prompt. Our findings underscore the need for systematic approaches to mitigate unintended consequences of instruction-tuning and enhance the reliability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18181v1">SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18178v1">Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) leverage both knowledge and reasoning during inference, the capacity to distinguish between them plays a pivotal role in model analysis, interpretability, and development. Inspired by dual-system cognitive theory, we propose a cognition attribution framework to decouple the contribution of knowledge and reasoning. In particular, the cognition of LLMs is decomposed into two distinct yet complementary phases: knowledge retrieval (Phase 1) and reasoning adjustment (Phase 2). To separate these phases, LLMs are prompted to generate answers under two different cognitive modes, fast thinking and slow thinking, respectively. The performance under different cognitive modes is analyzed to quantify the contribution of knowledge and reasoning. This architecture is employed to 15 LLMs across 3 datasets. Results reveal: (1) reasoning adjustment is domain-specific, benefiting reasoning-intensive domains (e.g., mathematics, physics, and chemistry) and potentially imparing knowledge-intensive domains. (2) Parameter scaling improves both knowledge and reasoning, with knowledge improvements being more pronounced. Additionally, parameter scaling make LLMs reasoning significantly more prudent, while moderately more intelligent. (3) Knowledge primarily resides in lower network layers, while reasoning operates in higher layers. Our framework not only helps understand LLMs from a "decoupling" perspective, but also provides new insights into existing research, including scaling laws, hierarchical knowledge editing, and limitations of small-model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17723v2">Statistical Runtime Verification for LLMs via Robustness Estimation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Adversarial robustness verification is essential for ensuring the safe deployment of Large Language Models (LLMs) in runtime-critical applications. However, formal verification techniques remain computationally infeasible for modern LLMs due to their exponential runtime and white-box access requirements. This paper presents a case study adapting and extending the RoMA statistical verification framework to assess its feasibility as an online runtime robustness monitor for LLMs in black-box deployment settings. Our adaptation of RoMA analyzes confidence score distributions under semantic perturbations to provide quantitative robustness assessments with statistically validated bounds. Our empirical validation against formal verification baselines demonstrates that RoMA achieves comparable accuracy (within 1\% deviation), and reduces verification times from hours to minutes. We evaluate this framework across semantic, categorial, and orthographic perturbation domains. Our results demonstrate RoMA's effectiveness for robustness monitoring in operational LLM deployments. These findings point to RoMA as a potentially scalable alternative when formal methods are infeasible, with promising implications for runtime verification in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18165v1">ProactiveVA: Proactive Visual Analytics with LLM-Based UI Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Visual analytics (VA) is typically applied to complex data, thus requiring complex tools. While visual analytics empowers analysts in data analysis, analysts may get lost in the complexity occasionally. This highlights the need for intelligent assistance mechanisms. However, even the latest LLM-assisted VA systems only provide help when explicitly requested by the user, making them insufficiently intelligent to offer suggestions when analysts need them the most. We propose a ProactiveVA framework in which LLM-powered UI agent monitors user interactions and delivers context-aware assistance proactively. To design effective proactive assistance, we first conducted a formative study analyzing help-seeking behaviors in user interaction logs, identifying when users need proactive help, what assistance they require, and how the agent should intervene. Based on this analysis, we distilled key design requirements in terms of intent recognition, solution generation, interpretability and controllability. Guided by these requirements, we develop a three-stage UI agent pipeline including perception, reasoning, and acting. The agent autonomously perceives users' needs from VA interaction logs, providing tailored suggestions and intuitive guidance through interactive exploration of the system. We implemented the framework in two representative types of VA systems, demonstrating its generalizability, and evaluated the effectiveness through an algorithm evaluation, case and expert study and a user study. We also discuss current design trade-offs of proactive VA and areas for further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18153v1">When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v4">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15690v3">LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The increasing use of synthetic data from the public Internet has enhanced data usage efficiency in large language model (LLM) training. However, the potential threat of model collapse remains insufficiently explored. Existing studies primarily examine model collapse in a single model setting or rely solely on statistical surrogates. In this work, we introduce LLM Web Dynamics (LWD), an efficient framework for investigating model collapse at the network level. By simulating the Internet with a retrieval-augmented generation (RAG) database, we analyze the convergence pattern of model outputs. Furthermore, we provide theoretical guarantees for this convergence by drawing an analogy to interacting Gaussian Mixture Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14928v2">EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Paper URL: https://aclanthology.org/2025.acl-long.1576/; Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18073v1">Squeeze10-LLM: Squeezing LLMs' Weights by 10 Times via a Staged Mixed-Precision Quantization Method</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) is challenging due to their massive parameters and high computational costs. Ultra low-bit quantization can significantly reduce storage and accelerate inference, but extreme compression (i.e., mean bit-width <= 2) often leads to severe performance degradation. To address this, we propose Squeeze10-LLM, effectively "squeezing" 16-bit LLMs' weights by 10 times. Specifically, Squeeze10-LLM is a staged mixed-precision post-training quantization (PTQ) framework and achieves an average of 1.6 bits per weight by quantizing 80% of the weights to 1 bit and 20% to 4 bits. We introduce Squeeze10LLM with two key innovations: Post-Binarization Activation Robustness (PBAR) and Full Information Activation Supervision (FIAS). PBAR is a refined weight significance metric that accounts for the impact of quantization on activations, improving accuracy in low-bit settings. FIAS is a strategy that preserves full activation information during quantization to mitigate cumulative error propagation across layers. Experiments on LLaMA and LLaMA2 show that Squeeze10-LLM achieves state-of-the-art performance for sub-2bit weight-only quantization, improving average accuracy from 43% to 56% on six zero-shot classification tasks--a significant boost over existing PTQ methods. Our code will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01144v5">BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.10678v4">Causally Testing Gender Bias in LLMs: A Case Study on Occupational Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Generated texts from large language models (LLMs) have been shown to exhibit a variety of harmful, human-like biases against various demographics. These findings motivate research efforts aiming to understand and measure such effects. This paper introduces a causal formulation for bias measurement in generative language models. Based on this theoretical foundation, we outline a list of desiderata for designing robust bias benchmarks. We then propose a benchmark called OccuGender, with a bias-measuring procedure to investigate occupational gender bias. We test several state-of-the-art open-source LLMs on OccuGender, including Llama, Mistral, and their instruction-tuned versions. The results show that these models exhibit substantial occupational gender bias. Lastly, we discuss prompting strategies for bias mitigation and an extension of our causal formulation to illustrate the generalizability of our framework. Our code and data https://github.com/chenyuen0103/gender-bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18055v1">Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The increasing use of synthetic data generated by Large Language Models (LLMs) presents both opportunities and challenges in data-driven applications. While synthetic data provides a cost-effective, scalable alternative to real-world data to facilitate model training, its diversity and privacy risks remain underexplored. Focusing on text-based synthetic data, we propose a comprehensive set of metrics to quantitatively assess the diversity (i.e., linguistic expression, sentiment, and user perspective), and privacy (i.e., re-identification risk and stylistic outliers) of synthetic datasets generated by several state-of-the-art LLMs. Experiment results reveal significant limitations in LLMs' capabilities in generating diverse and privacy-preserving synthetic data. Guided by the evaluation results, a prompt-based approach is proposed to enhance the diversity of synthetic reviews while preserving reviewer privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18043v1">GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 21 pages. Code: https://github.com/duykhuongnguyen/GrAInS
    </div>
    <details class="paper-abstract">
      Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18028v1">NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18009v1">GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18007v1">Cloud Native System for LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing numerous industries, but their substantial computational demands create challenges for efficient deployment, particularly in cloud environments. Traditional approaches to inference serving often struggle with resource inefficiencies, leading to high operational costs, latency issues, and limited scalability. This article explores how Cloud Native technologies, such as containerization, microservices, and dynamic scheduling, can fundamentally improve LLM inference serving. By leveraging these technologies, we demonstrate how a Cloud Native system enables more efficient resource allocation, reduces latency, and enhances throughput in high-demand scenarios. Through real-world evaluations using Kubernetes-based autoscaling, we show that Cloud Native architectures can dynamically adapt to workload fluctuations, mitigating performance bottlenecks while optimizing LLM inference serving performance. This discussion provides a broader perspective on how Cloud Native frameworks could reshape the future of scalable LLM inference serving, offering key insights for researchers, practitioners, and industry leaders in cloud computing and artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18006v1">Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Scaling</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created new opportunities across various fields but has also introduced significant challenges in resource management. Current LLM serving systems face a fundamental tension: balancing serving demands with limited resources while adapting to unpredictable traffic patterns. Static deployments lead to suboptimal resource utilization and performance degradation under dynamic workloads. Furthermore, the high cost of adjusting instances hinders dynamic scaling, limiting the true potential of efficient LLM serving. To address this, we propose CoCoServe, an elastic system that facilitates dynamic and fine-grained scaling. Its key innovation lies in the module-level operations for the replication and migration of LLM modules, such as decoder layers and projections. Through a comprehensive analysis of the trade-offs associated with these operations, we develop an auto-scaling mechanism that dynamically regulates module-level resource allocation and performance optimization, enabling a more cost-effective deployment of LLMs. Our evaluation demonstrates that the scaling operations employed by CoCoServe exhibit excellent scalability and can reduce costs by 46% while maintaining availability. Compared to state-of-the-art LLM serving systems (e.g., Hugging Face Transformers and vLLM), our approach reduces latency by 14%-75% and achieves 1.16x-4x throughput on average across different model sizes and workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18273v2">Analyzing Islamophobic Discourse Using Semi-Coded Terms and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      In recent years, Islamophobia has gained significant traction across Western societies, fueled by the rise of digital communication networks. This paper performs a large-scale analysis of specialized, semi-coded Islamophobic terms such as (muzrat, pislam, mudslime, mohammedan, muzzies) floated on extremist social platforms, i.e., 4Chan, Gab, Telegram, etc. Many of these terms appear lexically neutral or ambiguous outside of specific contexts, making them difficult for both human moderators and automated systems to reliably identify as hate speech. First, we use Large Language Models (LLMs) to show their ability to understand these terms. Second, Google Perspective API suggests that Islamophobic posts tend to receive higher toxicity scores than other categories of hate speech like Antisemitism. Finally, we use BERT topic modeling approach to extract different topics and Islamophobic discourse on these social platforms. Our findings indicate that LLMs understand these Out-Of-Vocabulary (OOV) slurs; however, further improvements in moderation strategies and algorithmic detection are necessary to address such discourse effectively. Our topic modeling also indicates that Islamophobic text is found across various political, conspiratorial, and far-right movements and is particularly directed against Muslim immigrants. Taken altogether, we performed one of the first studies on Islamophobic semi-coded terms and shed a global light on Islamophobia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07623v7">The Curious Case of Class Accuracy Imbalance in LLMs: Post-hoc Debiasing via Nonlinear Integer Programming</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are good knowledge bases but struggle to perform equally well for all classes in text classification. This paper investigates the case of class accuracy imbalance in LLMs, where deeply entangled pretraining biases and prompt-specific cues contribute to the imbalance. To overcome the difficulty in bias identification and inaccessibility of retraining, we post-hoc balance class accuracy using only output probabilities. This is enabled by reformulating debiasing as a combinatorial optimization problem. In details, we first motivate a post-hoc bias metric, the Contextual Oddity Bias (COBias), to quantify the over-/under-prediction (a tendency to over-predict some classes while under-predicting others) in LLMs. We then propose the Debiasing as Nonlinear Integer Programming (DNIP) method to reweight LLM output class probabilities towards minimizing COBias and maximizing overall accuracy, without being constrained by bias sources or updating LLM parameters. Since the DNIP model contains non-differentiable elements, we use simulated annealing to efficiently solve it. Evaluations on five LLMs across NLP classification benchmarks show that DNIP simultaneously achieves significant COBias reduction (61% relative reduction) and accuracy improvement (18% relative increase) under different LLM prompting setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18380v2">RedactOR: An LLM-Powered Framework for Automatic Clinical Data De-Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted to ACL 2025 Industry Track. To appear
    </div>
    <details class="paper-abstract">
      Ensuring clinical data privacy while preserving utility is critical for AI-driven healthcare and data analytics. Existing de-identification (De-ID) methods, including rule-based techniques, deep learning models, and large language models (LLMs), often suffer from recall errors, limited generalization, and inefficiencies, limiting their real-world applicability. We propose a fully automated, multi-modal framework, RedactOR for de-identifying structured and unstructured electronic health records, including clinical audio records. Our framework employs cost-efficient De-ID strategies, including intelligent routing, hybrid rule and LLM based approaches, and a two-step audio redaction approach. We present a retrieval-based entity relexicalization approach to ensure consistent substitutions of protected entities, thereby enhancing data coherence for downstream applications. We discuss key design desiderata, de-identification and relexicalization methodology, and modular architecture of RedactOR and its integration with the Oracle Health Clinical AI system. Evaluated on the i2b2 2014 De-ID dataset using standard metrics with strict recall, our approach achieves competitive performance while optimizing token usage to reduce LLM costs. Finally, we discuss key lessons and insights from deployment in real-world AI- driven healthcare data pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v4">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18812v1">MemoCoder: Automated Function Synthesis using LLM-Supported Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs) such as GitHub Copilot and ChatGPT, developers increasingly rely on AI-assisted tools to support code generation. While LLMs can generate syntactically correct solutions for well-structured programming tasks, they often struggle with challenges that require iterative debugging, error handling, or adaptation to diverse problem structures. Existing approaches such as fine-tuning or self-repair strategies either require costly retraining or lack mechanisms to accumulate and reuse knowledge from previous attempts. To address these limitations, we propose MemoCoder, a multi-agent framework that enables collaborative problem solving and persistent learning from past fixes. At the core of MemoCoder is a Fixing Knowledge Set, which stores successful repairs and supports retrieval for future tasks. A central Mentor Agent supervises the repair process by identifying recurring error patterns and refining high-level fixing strategies, providing a novel supervisory role that guides the self-repair loop. We evaluate MemoCoder across three public benchmarks -- MBPP, HumanEval, and LiveCodeBench -- spanning a range of problem complexities. Experimental results show that MemoCoder consistently outperforms both zero-shot prompting and a Self-Repair strategy, with improvements ranging from 3.1% to 12.1% in Pass@10 and from 1.4% to 14.5% in Pass@50, demonstrating its effectiveness in iterative refinement and knowledge-guided code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00151v2">Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 More information about our dataset is available at our project page: https://github.com/UBC-NLP/palm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18802v1">DxHF: Providing High-Quality Human Feedback for LLM Alignment via Interactive Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Human preferences are widely used to align large language models (LLMs) through methods such as reinforcement learning from human feedback (RLHF). However, the current user interfaces require annotators to compare text paragraphs, which is cognitively challenging when the texts are long or unfamiliar. This paper contributes by studying the decomposition principle as an approach to improving the quality of human feedback for LLM alignment. This approach breaks down the text into individual claims instead of directly comparing two long-form text responses. Based on the principle, we build a novel user interface DxHF. It enhances the comparison process by showing decomposed claims, visually encoding the relevance of claims to the conversation and linking similar claims. This allows users to skim through key information and identify differences for better and quicker judgment. Our technical evaluation shows evidence that decomposition generally improves feedback accuracy regarding the ground truth, particularly for users with uncertainty. A crowdsourcing study with 160 participants indicates that using DxHF improves feedback accuracy by an average of 5%, although it increases the average feedback time by 18 seconds. Notably, accuracy is significantly higher in situations where users have less certainty. The finding of the study highlights the potential of HCI as an effective method for improving human-AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18791v1">Evaluating Code-Mixing in LLMs Across 18 Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Code-mixing, the practice of switching between languages within a conversation, presents unique challenges for traditional natural language processing. Existing benchmarks, such as LinCE and GLUECoS, are limited by narrow language pairings and tasks, failing to adequately evaluate the code-mixing capabilities of large language models (LLMs). Despite the significance of code-mixing for multilingual users, research on LLMs in this context remains limited. Additionally, current methods for generating code-mixed data are underdeveloped. In this paper, we conduct a comprehensive evaluation of LLMs' performance on code-mixed data across 18 languages from seven language families. We also propose a novel approach for generating synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our analysis reveals consistent underperformance of LLMs on code-mixed datasets involving multiple language families. We suggest that improvements in training data size, model scale, and few-shot learning could enhance their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14641v3">HLSTester: Efficient Testing of Behavioral Discrepancies with LLMs for High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 arXiv admin note: text overlap with arXiv:2407.03889
    </div>
    <details class="paper-abstract">
      In high-level synthesis (HLS), C/C++ programs with synthesis directives are used to generate circuits for FPGA implementations. However, hardware-specific and platform-dependent characteristics in these implementations can introduce behavioral discrepancies between the original C/C++ programs and the circuits after high-level synthesis. Existing methods for testing behavioral discrepancies in HLS are still immature, and the testing workflow requires significant human efforts. To address this challenge, we propose HLSTester, a large language model (LLM) aided testing framework that efficiently detects behavioral discrepancies in HLS. To mitigate hallucinations in LLMs and enhance prompt quality, the testbenches for original C/C++ programs are leveraged to guide LLMs in generating HLS-compatible testbenches, effectively eliminating certain traditional C/C++ constructs that are incompatible with HLS tools. Key variables are pinpointed through a backward slicing technique in both C/C++ and HLS programs to monitor their runtime spectra, enabling an in-depth analysis of the discrepancy symptoms. To reduce test time, a testing input generation mechanism is introduced to integrate dynamic mutation with insights from an LLM-based progressive reasoning chain. In addition, repetitive hardware testing is skipped by a redundancy-aware filtering technique for the generated test inputs. Experimental results demonstrate that the proposed LLM-aided testing framework significantly accelerates the testing workflow while achieving higher testbench simulation pass rates compared with the traditional method and the direct use of LLMs on the same HLS programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19549v1">AccessGuru: Leveraging LLMs to Detect and Correct Web Accessibility Violations in HTML Code</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The vast majority of Web pages fail to comply with established Web accessibility guidelines, excluding a range of users with diverse abilities from interacting with their content. Making Web pages accessible to all users requires dedicated expertise and additional manual efforts from Web page providers. To lower their efforts and promote inclusiveness, we aim to automatically detect and correct Web accessibility violations in HTML code. While previous work has made progress in detecting certain types of accessibility violations, the problem of automatically detecting and correcting accessibility violations remains an open challenge that we address. We introduce a novel taxonomy classifying Web accessibility violations into three key categories - Syntactic, Semantic, and Layout. This taxonomy provides a structured foundation for developing our detection and correction method and redefining evaluation metrics. We propose a novel method, AccessGuru, which combines existing accessibility testing tools and Large Language Models (LLMs) to detect violations and applies taxonomy-driven prompting strategies to correct all three categories. To evaluate these capabilities, we develop a benchmark of real-world Web accessibility violations. Our benchmark quantifies syntactic and layout compliance and judges semantic accuracy through comparative analysis with human expert corrections. Evaluation against our benchmark shows that AccessGuru achieves up to 84% average violation score decrease, significantly outperforming prior methods that achieve at most 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15606v2">LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16044v2">Making REST APIs Agent-Ready: From OpenAPI to Model Context Protocol Servers for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are evolving from passive text generators into active agents that invoke external tools. To support this shift, scalable protocols for tool integration are essential. The Model Context Protocol (MCP), introduced by Anthropic in 2024, offers a schema-driven standard for dynamic tool discovery and invocation. Yet, building MCP servers remains manual and repetitive, requiring developers to write glue code, handle authentication, and configure schemas by hand-replicating much of the integration effort MCP aims to eliminate. This paper investigates whether MCP server construction can be meaningfully automated. We begin by analyzing adoption trends: among 22,000+ MCP-tagged GitHub repositories created within six months of release, fewer than 5% include servers, typically small, single-maintainer projects dominated by repetitive scaffolding. To address this gap, we present AutoMCP, a compiler that generates MCP servers from OpenAPI 2.0/3.0 specifications. AutoMCP parses REST API definitions and produces complete server implementations, including schema registration and authentication handling. We evaluate AutoMCP on 50 real-world APIs spanning 5,066 endpoints across over 10 domains. From a stratified sample of 1,023 tool calls, 76.5% succeeded out of the box. Manual failure analysis revealed five recurring issues, all attributable to inconsistencies or omissions in the OpenAPI contracts. After minor fixes, averaging 19 lines of spec changes per API, AutoMCP achieved 99.9% success. Our findings (i) analyze MCP adoption and quantify the cost of manual server development, (ii) demonstrate that OpenAPI specifications, despite quality issues, enable near-complete MCP server automation, and (iii) contribute a corpus of 5,066 callable tools along with insights on repairing common specification flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17636v1">Who Attacks, and Why? Using LLMs to Identify Negative Campaigning in 18M Tweets across 19 Countries</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Negative campaigning is a central feature of political competition, yet empirical research has been limited by the high cost and limited scalability of existing classification methods. This study makes two key contributions. First, it introduces zero-shot Large Language Models (LLMs) as a novel approach for cross-lingual classification of negative campaigning. Using benchmark datasets in ten languages, we demonstrate that LLMs achieve performance on par with native-speaking human coders and outperform conventional supervised machine learning approaches. Second, we leverage this novel method to conduct the largest cross-national study of negative campaigning to date, analyzing 18 million tweets posted by parliamentarians in 19 European countries between 2017 and 2022. The results reveal consistent cross-national patterns: governing parties are less likely to use negative messaging, while ideologically extreme and populist parties -- particularly those on the radical right -- engage in significantly higher levels of negativity. These findings advance our understanding of how party-level characteristics shape strategic communication in multiparty systems. More broadly, the study demonstrates the potential of LLMs to enable scalable, transparent, and replicable research in political communication across linguistic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17542v1">AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Bug reproduction is critical in the software debugging and repair process, yet the majority of bugs in open-source and industrial settings lack executable tests to reproduce them at the time they are reported, making diagnosis and resolution more difficult and time-consuming. To address this challenge, we introduce AssertFlip, a novel technique for automatically generating Bug Reproducible Tests (BRTs) using large language models (LLMs). Unlike existing methods that attempt direct generation of failing tests, AssertFlip first generates passing tests on the buggy behaviour and then inverts these tests to fail when the bug is present. We hypothesize that LLMs are better at writing passing tests than ones that crash or fail on purpose. Our results show that AssertFlip outperforms all known techniques in the leaderboard of SWT-Bench, a benchmark curated for BRTs. Specifically, AssertFlip achieves a fail-to-pass success rate of 43.6% on the SWT-Bench-Verified subset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17476v1">MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Although recent Large Language Models (LLMs) have shown rapid improvement on reasoning benchmarks in English, the evaluation of such LLMs' multilingual reasoning capability across diverse languages and cultural contexts remains limited. Existing multilingual reasoning benchmarks are typically constructed by translating existing English reasoning benchmarks, biasing these benchmarks towards reasoning problems with context in English language/cultures. In this work, we introduce the Multilingual Native Reasoning Challenge (MultiNRC), a benchmark designed to assess LLMs on more than 1,000 native, linguistic and culturally grounded reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. For cultural/tradition reasoning and math reasoning with cultural relevance, we also provide English equivalent translations of the multilingual questions by manual translation from native speakers fluent in English. This set of English equivalents can provide a direct comparison of LLM reasoning capacity in other languages vs. English on the same reasoning questions. We systematically evaluate current 14 leading LLMs covering most LLM families on MultiNRC and its English equivalent set. The results show that (1) current LLMs are still not good at native multilingual reasoning, with none scoring above 50% on MultiNRC; (2) LLMs exhibit distinct strengths and weaknesses in handling linguistic, cultural, and logical reasoning tasks; (3) Most models perform substantially better in math reasoning in English compared to in original languages (+10%), indicating persistent challenges with culturally grounded knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v2">WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label Unknown, yet current evaluations focus almost exclusively on whether such answers are honest rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon Vague Perception. And thus we introduce a framework that quantifies the proportion of Unknown responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct Known or correct Unknown with valid reasoning. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the potential reasoning ability of LLMs and providing a new perspective on solving the Vague Perception phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10054v2">Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted to ICSME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10382v2">Leveraging RAG-LLMs for Urban Mobility Simulation and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      With the rise of smart mobility and shared e-mobility services, numerous advanced technologies have been applied to this field. Cloud-based traffic simulation solutions have flourished, offering increasingly realistic representations of the evolving mobility landscape. LLMs have emerged as pioneering tools, providing robust support for various applications, including intelligent decision-making, user interaction, and real-time traffic analysis. As user demand for e-mobility continues to grow, delivering comprehensive end-to-end solutions has become crucial. In this paper, we present a cloud-based, LLM-powered shared e-mobility platform, integrated with a mobile application for personalized route recommendations. The optimization module is evaluated based on travel time and cost across different traffic scenarios. Additionally, the LLM-powered RAG framework is evaluated at the schema level for different users, using various evaluation methods. Schema-level RAG with XiYanSQL achieves an average execution accuracy of 0.81 on system operator queries and 0.98 on user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17290v1">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 RecSys2025
    </div>
    <details class="paper-abstract">
      Serendipity plays a pivotal role in enhancing user satisfaction within recommender systems, yet its evaluation poses significant challenges due to its inherently subjective nature and conceptual ambiguity. Current algorithmic approaches predominantly rely on proxy metrics for indirect assessment, often failing to align with real user perceptions, thus creating a gap. With large language models (LLMs) increasingly revolutionizing evaluation methodologies across various human annotation tasks, we are inspired to explore a core research proposition: Can LLMs effectively simulate human users for serendipity evaluation? To address this question, we conduct a meta-evaluation on two datasets derived from real user studies in the e-commerce and movie domains, focusing on three key aspects: the accuracy of LLMs compared to conventional proxy metrics, the influence of auxiliary data on LLM comprehension, and the efficacy of recently popular multi-LLM techniques. Our findings indicate that even the simplest zero-shot LLMs achieve parity with, or surpass, the performance of conventional metrics. Furthermore, multi-LLM techniques and the incorporation of auxiliary data further enhance alignment with human perspectives. Based on our findings, the optimal evaluation by LLMs yields a Pearson correlation coefficient of 21.5\% when compared to the results of the user study. This research implies that LLMs may serve as potentially accurate and cost-effective evaluators, introducing a new paradigm for serendipity evaluation in recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17288v1">Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17273v1">Leveraging Knowledge Graphs and LLM Reasoning to Identify Operational Bottlenecks for Warehouse Planning Assistance</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Analyzing large, complex output datasets from Discrete Event Simulations (DES) of warehouse operations to identify bottlenecks and inefficiencies is a critical yet challenging task, often demanding significant manual effort or specialized analytical tools. Our framework integrates Knowledge Graphs (KGs) and Large Language Model (LLM)-based agents to analyze complex Discrete Event Simulation (DES) output data from warehouse operations. It transforms raw DES data into a semantically rich KG, capturing relationships between simulation events and entities. An LLM-based agent uses iterative reasoning, generating interdependent sub-questions. For each sub-question, it creates Cypher queries for KG interaction, extracts information, and self-reflects to correct errors. This adaptive, iterative, and self-correcting process identifies operational issues mimicking human analysis. Our DES approach for warehouse bottleneck identification, tested with equipment breakdowns and process irregularities, outperforms baseline methods. For operational questions, it achieves near-perfect pass rates in pinpointing inefficiencies. For complex investigative questions, we demonstrate its superior diagnostic ability to uncover subtle, interconnected issues. This work bridges simulation modeling and AI (KG+LLM), offering a more intuitive method for actionable insights, reducing time-to-insight, and enabling automated warehouse inefficiency evaluation and diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17259v1">Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13343v2">TwiUSD: A Benchmark Dataset and Structure-Aware LLM Framework for User Stance Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      User-level stance detection (UserSD) remains challenging due to the lack of high-quality benchmarks that jointly capture linguistic and social structure. In this paper, we introduce TwiUSD, the first large-scale, manually annotated UserSD benchmark with explicit followee relationships, containing 16,211 users and 47,757 tweets. TwiUSD enables rigorous evaluation of stance models by integrating tweet content and social links, with superior scale and annotation quality. Building on this resource, we propose MRFG: a structure-aware framework that uses LLM-based relevance filtering and feature routing to address noise and context heterogeneity. MRFG employs multi-scale filtering and adaptively routes features through graph neural networks or multi-layer perceptrons based on topological informativeness. Experiments show MRFG consistently outperforms strong baselines (including PLMs, graph-based models, and LLM prompting) in both in-target and cross-target evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16799v2">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17209v1">HypoChainer: A Collaborative System Combining LLMs and Knowledge Graphs for Hypothesis-Driven Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Modern scientific discovery faces growing challenges in integrating vast and heterogeneous knowledge critical to breakthroughs in biomedicine and drug development. Traditional hypothesis-driven research, though effective, is constrained by human cognitive limits, the complexity of biological systems, and the high cost of trial-and-error experimentation. Deep learning models, especially graph neural networks (GNNs), have accelerated prediction generation, but the sheer volume of outputs makes manual selection for validation unscalable. Large language models (LLMs) offer promise in filtering and hypothesis generation, yet suffer from hallucinations and lack grounding in structured knowledge, limiting their reliability. To address these issues, we propose HypoChainer, a collaborative visualization framework that integrates human expertise, LLM-driven reasoning, and knowledge graphs (KGs) to enhance hypothesis generation and validation. HypoChainer operates in three stages: First, exploration and contextualization -- experts use retrieval-augmented LLMs (RAGs) and dimensionality reduction to navigate large-scale GNN predictions, assisted by interactive explanations. Second, hypothesis chain formation -- experts iteratively examine KG relationships around predictions and semantically linked entities, refining hypotheses with LLM and KG suggestions. Third, validation prioritization -- refined hypotheses are filtered based on KG-supported evidence to identify high-priority candidates for experimentation, with visual analytics further strengthening weak links in reasoning. We demonstrate HypoChainer's effectiveness through case studies in two domains and expert interviews, highlighting its potential to support interpretable, scalable, and knowledge-grounded scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12988v2">Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 19 pages, 3 figures, ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought processes of a character. Using Lu Xun, a renowned Chinese writer, as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope that this work inspires future research on deep character persona simulation LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17188v1">LLM Meets the Sky: Heuristic Multi-Agent Reinforcement Learning for Secure Heterogeneous UAV Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Submitted to IEEE Transactions on Mobile Computing
    </div>
    <details class="paper-abstract">
      This work tackles the physical layer security (PLS) problem of maximizing the secrecy rate in heterogeneous UAV networks (HetUAVNs) under propulsion energy constraints. Unlike prior studies that assume uniform UAV capabilities or overlook energy-security trade-offs, we consider a realistic scenario where UAVs with diverse payloads and computation resources collaborate to serve ground terminals in the presence of eavesdroppers. To manage the complex coupling between UAV motion and communication, we propose a hierarchical optimization framework. The inner layer uses a semidefinite relaxation (SDR)-based S2DC algorithm combining penalty functions and difference-of-convex (d.c.) programming to solve the secrecy precoding problem with fixed UAV positions. The outer layer introduces a Large Language Model (LLM)-guided heuristic multi-agent reinforcement learning approach (LLM-HeMARL) for trajectory optimization. LLM-HeMARL efficiently incorporates expert heuristics policy generated by the LLM, enabling UAVs to learn energy-aware, security-driven trajectories without the inference overhead of real-time LLM calls. The simulation results show that our method outperforms existing baselines in secrecy rate and energy efficiency, with consistent robustness across varying UAV swarm sizes and random seeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17178v1">SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/Lza12a/SKA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16727v2">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 The inconsistency of base models undermines the fairness of evaluation comparisons and affects the validity of the paper's conclusions
    </div>
    <details class="paper-abstract">
      Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17168v1">Improving LLMs' Generalized Reasoning Abilities by Graph Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 COLM2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made remarkable strides in reasoning tasks, yet their performance often falters on novel and complex problems. Domain-specific continued pretraining (CPT) methods, such as those tailored for mathematical reasoning, have shown promise but lack transferability to broader reasoning tasks. In this work, we pioneer the use of Graph Problem Reasoning (GPR) to enhance the general reasoning capabilities of LLMs. GPR tasks, spanning pathfinding, network analysis, numerical computation, and topological reasoning, require sophisticated logical and relational reasoning, making them ideal for teaching diverse reasoning patterns. To achieve this, we introduce GraphPile, the first large-scale corpus specifically designed for CPT using GPR data. Spanning 10.9 billion tokens across 23 graph tasks, the dataset includes chain-of-thought, program-of-thought, trace of execution, and real-world graph data. Using GraphPile, we train GraphMind on popular base models Llama 3 and 3.1, as well as Gemma 2, achieving up to 4.9 percent higher accuracy in mathematical reasoning and up to 21.2 percent improvement in non-mathematical reasoning tasks such as logical and commonsense reasoning. By being the first to harness GPR for enhancing reasoning patterns and introducing the first dataset of its kind, our work bridges the gap between domain-specific pretraining and universal reasoning capabilities, advancing the adaptability and robustness of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17165v1">Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted at the 41st IEEE International Conference on Software Maintenance and Evolution 2025 (ICSME'25)
    </div>
    <details class="paper-abstract">
      Continuous Integration (CI) services, such as GitHub Actions, require developers to write YAML-based configurations, which can be tedious and error-prone. Despite the increasing use of Large Language Models (LLMs) to automate software engineering tasks, their ability to generate CI configurations remains underexplored. This paper presents a preliminary study evaluating six LLMs for generating GitHub Actions configurations from natural language descriptions. We assess three general-purpose foundation models (GPT-4o, Llama, and Gemma) and three code-pretrained models (GPT-4.1, Code Llama, and CodeGemma). We also introduce the first labeled dataset of its kind, constructed from GitHub Actions documentation, pairing descriptions with corresponding best-practice YAML configurations. Zero-shot prompting achieves up to 69% similarity with the ground truth, with only 3% perfect matches. Code-pretrained models slightly underperform compared to general-purpose ones in YAML-based CI tasks, revealing LLM limitations for CI configuration generation. Analyzing GPT-4o outputs reveals issues like missing or renamed steps, misinterpreted descriptions, and unnecessary additions that may affect structural and contextual correctness, indicating a gap between generation quality and the precision required for executable CI configurations. Our research offers insights for improving LLM alignment with configuration languages and guiding future efforts on CI automation and tooling support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17147v1">CogDual: Enhancing Dual Cognition of LLMs via Reinforcement Learning with Implicit Rule-Based Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Role-Playing Language Agents (RPLAs) have emerged as a significant application direction for Large Language Models (LLMs). Existing approaches typically rely on prompt engineering or supervised fine-tuning to enable models to imitate character behaviors in specific scenarios, but often neglect the underlying \emph{cognitive} mechanisms driving these behaviors. Inspired by cognitive psychology, we introduce \textbf{CogDual}, a novel RPLA adopting a \textit{cognize-then-respond } reasoning paradigm. By jointly modeling external situational awareness and internal self-awareness, CogDual generates responses with improved character consistency and contextual alignment. To further optimize the performance, we employ reinforcement learning with two general-purpose reward schemes designed for open-domain text generation. Extensive experiments on the CoSER benchmark, as well as Cross-MR and LifeChoice, demonstrate that CogDual consistently outperforms existing baselines and generalizes effectively across diverse role-playing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17134v1">Resilient Multi-Agent Negotiation for Medical Supply Chains:Integrating LLMs and Blockchain for Transparent Coordination</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 11 pages, 6 figure
    </div>
    <details class="paper-abstract">
      Global health emergencies, such as the COVID-19 pandemic, have exposed critical weaknesses in traditional medical supply chains, including inefficiencies in resource allocation, lack of transparency, and poor adaptability to dynamic disruptions. This paper presents a novel hybrid framework that integrates blockchain technology with a decentralized, large language model (LLM) powered multi-agent negotiation system to enhance the resilience and accountability of medical supply chains during crises. In this system, autonomous agents-representing manufacturers, distributors, and healthcare institutions-engage in structured, context-aware negotiation and decision-making processes facilitated by LLMs, enabling rapid and ethical allocation of scarce medical resources. The off-chain agent layer supports adaptive reasoning and local decision-making, while the on-chain blockchain layer ensures immutable, transparent, and auditable enforcement of decisions via smart contracts. The framework also incorporates a formal cross-layer communication protocol to bridge decentralized negotiation with institutional enforcement. A simulation environment emulating pandemic scenarios evaluates the system's performance, demonstrating improvements in negotiation efficiency, fairness of allocation, supply chain responsiveness, and auditability. This research contributes an innovative approach that synergizes blockchain trust guarantees with the adaptive intelligence of LLM-driven agents, providing a robust and scalable solution for critical supply chain coordination under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17133v1">BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      In recent years, the Mixture-of-Experts (MoE) architecture has been widely applied to large language models (LLMs), providing a promising solution that activates only a subset of the model's parameters during computation, thereby reducing overall memory requirements and allowing for faster inference compared to dense models. Despite these advantages, existing systems still face issues of low efficiency due to static model placement and lack of dynamic workloads adaptation. This leads to suboptimal resource utilization and increased latency, especially during bursty requests periods. To address these challenges, this paper introduces BrownoutServe, a novel serving framework designed to optimize inference efficiency and maintain service reliability for MoE-based LLMs under dynamic computational demands and traffic conditions. BrownoutServe introduces "united experts" that integrate knowledge from multiple experts, reducing the times of expert access and inference latency. Additionally, it proposes a dynamic brownout mechanism to adaptively adjust the processing of certain tokens, optimizing inference performance while guaranteeing service level objectives (SLOs) are met. Our evaluations show the effectiveness of BrownoutServe under various workloads: it achieves up to 2.07x throughput improvement compared to vLLM and reduces SLO violations by 90.28%, showcasing its robustness under bursty traffic while maintaining acceptable inference accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v2">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model. Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to thousands of GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17120v1">BucketServe: Bucket-Based Dynamic Batching for Smart and Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly popular in various areas, traditional business gradually shifting from rule-based systems to LLM-based solutions. However, the inference of LLMs is resource-intensive or latency-sensitive, posing significant challenges for serving systems. Existing LLM serving systems often use static or continuous batching strategies, which can lead to inefficient GPU memory utilization and increased latency, especially under heterogeneous workloads. These methods may also struggle to adapt to dynamic workload fluctuations, resulting in suboptimal throughput and potential service level objective (SLO) violations. In this paper, we introduce BucketServe, a bucket-based dynamic batching framework designed to optimize LLM inference performance. By grouping requests into size-homogeneous buckets based on sequence length, BucketServe minimizes padding overhead and optimizes GPU memory usage through real-time batch size adjustments preventing out-of-memory (OOM) errors. It introduces adaptive bucket splitting/merging and priority-aware scheduling to mitigate resource fragmentation and ensure SLO compliance. Experiment shows that BucketServe significantly outperforms UELLM in throughput, achieving up to 3.58x improvement. It can also handle 1.93x more request load under the SLO attainment of 80% compared with DistServe and demonstrates 1.975x higher system load capacity compared to the UELLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08513v2">Advancing Multimodal LLMs by Large-Scale 3D Visual Instruction Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) struggle with accurately capturing camera-object relations, especially for object orientation, camera viewpoint, and camera shots. This stems from the fact that existing MLLMs are trained on images with limited diverse camera-object relations and corresponding textual descriptions. To address this, we propose a synthetic generation pipeline to create large-scale 3D visual instruction datasets. Our framework takes 3D assets as input and uses rendering and diffusion-based image generation models to create photorealistic images preserving precise camera-object relations. Additionally, large language models (LLMs) are used to generate text prompts for guiding visual instruction tuning and controlling image generation. We create Ultimate3D, a dataset of 240K VQAs with precise camera-object annotations, and corresponding benchmark. MLLMs fine-tuned on our proposed dataset outperform commercial models by a large margin, achieving an average accuracy improvement of 33.4% on camera-object relation recognition tasks. Our code, dataset, and benchmark will contribute to broad MLLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17962v1">TimelyHLS: LLM-Based Timing-Aware and Architecture-Specific FPGA HLS Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Achieving timing closure and design-specific optimizations in FPGA-targeted High-Level Synthesis (HLS) remains a significant challenge due to the complex interaction between architectural constraints, resource utilization, and the absence of automated support for platform-specific pragmas. In this work, we propose TimelyHLS, a novel framework integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to automatically generate and iteratively refine HLS code optimized for FPGA-specific timing and performance requirements. TimelyHLS is driven by a structured architectural knowledge base containing FPGA-specific features, synthesis directives, and pragma templates. Given a kernel, TimelyHLS generates HLS code annotated with both timing-critical and design-specific pragmas. The synthesized RTL is then evaluated using commercial toolchains, and simulation correctness is verified against reference outputs via custom testbenches. TimelyHLS iteratively incorporates synthesis logs and performance reports into the LLM engine for refinement in the presence of functional discrepancies. Experimental results across 10 FPGA architectures and diverse benchmarks show that TimelyHLS reduces the need for manual tuning by up to 70%, while achieving up to 4x latency speedup (e.g., 3.85x for Matrix Multiplication, 3.7x for Bitonic Sort) and over 50% area savings in certain cases (e.g., 57% FF reduction in Viterbi). TimelyHLS consistently achieves timing closure and functional correctness across platforms, highlighting the effectiveness of LLM-driven, architecture-aware synthesis in automating FPGA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13238v2">Multilingual LLMs Are Not Multilingual Thinkers: Evidence from Hindi Analogy Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Analogies test a model's ability to infer implicit relationships between concepts, making them a key benchmark for evaluating reasoning capabilities. While large language models (LLMs) are widely evaluated for reasoning in English, their abilities in Indic languages remain understudied, limiting our understanding of whether these models generalize across languages. To address this gap, we introduce a new Hindi Analogy Test Set (HATS), comprising 405 multiple-choice questions sourced from Indian government exams. We benchmark state-of-the-art multilingual LLMs using various prompting strategies and introduce a grounded Chain of Thought approach that leverages cognitive theories of analogical reasoning. This approach improves model performance on Hindi analogy questions. Our experiments show that models perform best with English prompts, irrespective of the prompting strategy. Our test set addresses the lack of a critical resource to evaluate LLM reasoning capabilities in Hindi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17951v1">Are LLM Belief Updates Consistent with Bayes' Theorem?</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted at the ICML 2025 Workshop on Assessing World Models
    </div>
    <details class="paper-abstract">
      Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03699v3">LLM Alignment as Retriever Optimization: An Information Retrieval Perspective</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17927v1">SMARTAPS: Tool-augmented LLMs for Operations Management</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 https://aaai.org/conference/aaai/aaai-25/bridge-ai-orms/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) present intriguing opportunities to enhance user interaction with traditional algorithms and tools in real-world applications. An advanced planning system (APS) is a sophisticated software that leverages optimization to help operations planners create, interpret, and modify an operational plan. While highly beneficial, many customers are priced out of using an APS due to the ongoing costs of consultants responsible for customization and maintenance. To address the need for a more accessible APS expressed by supply chain planners, we present SmartAPS, a conversational system built on a tool-augmented LLM. Our system provides operations planners with an intuitive natural language chat interface, allowing them to query information, perform counterfactual reasoning, receive recommendations, and execute scenario analysis to better manage their operation. A short video demonstrating the system has been released: https://youtu.be/KtIrJjlDbyw
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08537v2">Chemical reasoning in LLMs unlocks strategy-aware synthesis planning and reaction mechanism elucidation</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      While automated chemical tools excel at specific tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning. Here we demonstrate that large language models (LLMs) can serve as powerful tools enabling chemical analysis. When integrated with traditional search algorithms, they enable a new approach to computer-aided synthesis that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our system allows chemists to specify desired synthetic strategies in natural language -- from protecting group strategies to global feasibility assessment -- and uses traditional or LLM-guided Monte Carlo Tree Search to find routes that satisfy these constraints. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. This approach shows strong performance across diverse chemical tasks, with newer and larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical automation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17999v2">Streaming, Fast and Slow: Cognitive Load-Aware Streaming for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Generative conversational interfaces powered by large language models (LLMs) typically stream output token-by-token at a rate determined by computational budget, often neglecting actual human reading speeds and the cognitive load associated with the content. This mismatch frequently leads to inefficient use of computational resources. For example, in cloud-based services, streaming content faster than users can read appears unnecessary, resulting in wasted computational resources and potential delays for other users, particularly during peak usage periods. To address this issue, we propose an adaptive streaming method that dynamically adjusts the pacing of LLM streaming output in real-time based on inferred cognitive load. Our approach estimates the cognitive load associated with streaming content and strategically slows down the stream during complex or information-rich segments, thereby freeing computational resources for other users. We conducted a statistical analysis and simulation based on a statistical model derived from data collected in a crowdsourced user study across various types of LLM-generated content. Our results show that this adaptive method can effectively reduce computational consumption while largely maintaining streaming speed above user's normal reading speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17865v1">Talk with the Things: Integrating LLMs into IoT Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 arXiv admin note: text overlap with arXiv:2407.20970
    </div>
    <details class="paper-abstract">
      The convergence of Large Language Models (LLMs) and Internet of Things (IoT) networks open new opportunities for building intelligent, responsive, and user-friendly systems. This work presents an edge-centric framework that integrates LLMs into IoT architectures to enable natural language-based control, context-aware decision-making, and enhanced automation. The proposed modular and lightweight Retrieval Augmented Generation (RAG)-based LLMs are deployed on edge computing devices connected to IoT gateways, enabling local processing of user commands and sensor data for reduced latency, improved privacy, and enhanced inference quality. We validate the framework through a smart home prototype using LLaMA 3 and Gemma 2B models for controlling smart devices. Experimental results highlight the trade-offs between model accuracy and inference time with respect to models size. At last, we also discuss the potential applications that can use LLM-based IoT systems, and a few key challenges associated with such systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17842v1">Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong potential in generating 'believable human-like' behavior in web environments. Prior work has explored augmenting training data with LLM-synthesized rationales and applying supervised fine-tuning (SFT) to enhance reasoning ability, which in turn can improve downstream action prediction. However, the performance of such approaches remains inherently bounded by the reasoning capabilities of the model used to generate the rationales. In this paper, we introduce Shop-R1, a novel reinforcement learning (RL) framework aimed at enhancing the reasoning ability of LLMs for simulation of real human behavior in online shopping environments Specifically, Shop-R1 decomposes the human behavior simulation task into two stages: rationale generation and action prediction, each guided by distinct reward signals. For rationale generation, we leverage internal model signals (e.g., logit distributions) to guide the reasoning process in a self-supervised manner. For action prediction, we propose a hierarchical reward structure with difficulty-aware scaling to prevent reward hacking and enable fine-grained reward assignment. This design evaluates both high-level action types and the correctness of fine-grained sub-action details (attributes and values), rewarding outputs proportionally to their difficulty. Experimental results show that our method achieves a relative improvement of over 65% compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17795v1">LSDM: LLM-Enhanced Spatio-temporal Diffusion Model for Service-Level Mobile Traffic Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Service-level mobile traffic prediction for individual users is essential for network efficiency and quality of service enhancement. However, current prediction methods are limited in their adaptability across different urban environments and produce inaccurate results due to the high uncertainty in personal traffic patterns, the lack of detailed environmental context, and the complex dependencies among different network services. These challenges demand advanced modeling techniques that can capture dynamic traffic distributions and rich environmental features. Inspired by the recent success of diffusion models in distribution modeling and Large Language Models (LLMs) in contextual understanding, we propose an LLM-Enhanced Spatio-temporal Diffusion Model (LSDM). LSDM integrates the generative power of diffusion models with the adaptive learning capabilities of transformers, augmented by the ability to capture multimodal environmental information for modeling service-level patterns and dynamics. Extensive evaluations on real-world service-level datasets demonstrate that the model excels in traffic usage predictions, showing outstanding generalization and adaptability. After incorporating contextual information via LLM, the performance improves by at least 2.83% in terms of the coefficient of determination. Compared to models of a similar type, such as CSDI, the root mean squared error can be reduced by at least 8.29%. The code and dataset will be available at: https://github.com/SoftYuaneR/LSDM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17788v1">Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      When using LLMs to rank items based on given criteria, or evaluate answers, the order of candidate items can influence the model's final decision. This sensitivity to item positioning in a LLM's prompt is known as position bias. Prior research shows that this bias exists even in large models, though its severity varies across models and tasks. In addition to position bias, LLMs also exhibit varying degrees of low repetition consistency, where repeating the LLM call with the same candidate ordering can lead to different rankings. To address both inconsistencies, a common approach is to prompt the model multiple times with different candidate orderings and aggregate the results via majority voting. However, this repetition strategy, significantly increases computational costs. Extending prior findings, we observe that both the direction -- favoring either the earlier or later candidate in the prompt -- and magnitude of position bias across instances vary substantially, even within a single dataset. This observation highlights the need for a per-instance mitigation strategy. To this end, we introduce a dynamic early-stopping method that adaptively determines the number of repetitions required for each instance. Evaluating our approach across three LLMs of varying sizes and on two tasks, namely re-ranking and alignment, we demonstrate that transitioning to a dynamic repetition strategy reduces the number of LLM calls by an average of 81%, while preserving the accuracy. Furthermore, we propose a confidence-based adaptation to our early-stopping method, reducing LLM calls by an average of 87% compared to static repetition, with only a slight accuracy trade-off relative to our original early-stopping method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16809v1">LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 41 pages, 17 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16808v1">Rethinking LLM-Based RTL Code Optimization Via Timing Logic Metamorphosis</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 13pages with 9 pictures and 2 tables
    </div>
    <details class="paper-abstract">
      Register Transfer Level(RTL) code optimization is crucial for achieving high performance and low power consumption in digital circuit design. However, traditional optimization methods often rely on manual tuning and heuristics, which can be time-consuming and error-prone. Recent studies proposed to leverage Large Language Models(LLMs) to assist in RTL code optimization. LLMs can generate optimized code snippets based on natural language descriptions, potentially speeding up the optimization process. However, existing approaches have not thoroughly evaluated the effectiveness of LLM-Based code optimization methods for RTL code with complex timing logic. To address this gap, we conducted a comprehensive empirical investigation to assess the capability of LLM-Based RTL code optimization methods in handling RTL code with complex timing logic. In this study, we first propose a new benchmark for RTL optimization evaluation. It comprises four subsets, each corresponding to a specific area of RTL code optimization. Then we introduce a method based on metamorphosis to systematically evaluate the effectiveness of LLM-Based RTL code optimization methods.Our key insight is that the optimization effectiveness should remain consistent for semantically equivalent but more complex code. After intensive experiments, we revealed several key findings. (1) LLM-Based RTL optimization methods can effectively optimize logic operations and outperform existing compiler-based methods. (2) LLM-Based RTL optimization methods do not perform better than existing compiler-based methods on RTL code with complex timing logic, particularly in timing control flow optimization and clock domain optimization. This is primarily attributed to the challenges LLMs face in understanding timing logic in RTL code. Based on these findings, we provide insights for further research in leveraging LLMs for RTL code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16799v1">Test-Time-Matching: Decouple Personality, Memory, and Linguistic Style in LLM-based Role-Playing Language Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled role-playing language agents to demonstrate significant potential in various applications. However, relying solely on prompts and contextual inputs often proves insufficient for achieving deep immersion in specific roles, particularly well-known fictional or public figures. On the other hand, fine-tuning-based approaches face limitations due to the challenges associated with data collection and the computational resources required for training, thereby restricting their broader applicability. To address these issues, we propose Test-Time-Matching (TTM), a training-free role-playing framework through test-time scaling and context engineering. TTM uses LLM agents to automatically decouple a character's features into personality, memory, and linguistic style. Our framework involves a structured, three-stage generation pipeline that utilizes these features for controlled role-playing. It achieves high-fidelity role-playing performance, also enables seamless combinations across diverse linguistic styles and even variations in personality and memory. We evaluate our framework through human assessment, and the results demonstrate that our method achieves the outstanding performance in generating expressive and stylistically consistent character dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16773v1">When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16754v1">Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However, designing effective pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches, and (2) address new questions without any close prior matches by automatically lowering the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage for unseen cases. We find that implementing a RAG pipeline combining hypothetical-documentation-embedding (HyDE) with the full-answer context performs best in retrieving and answering similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG pipeline to 4 open-source LLMs and compare the results to their zero-shot performance. Our findings show that RAG with our optimal RAG pipeline consistently outperforms zero-shot baselines across models, achieving higher scores for helpfulness, correctness, and detail with LLM-as-a-judge. These findings demonstrate that our optimal RAG pipelines robustly enhance answer quality for a wide range of developer queries including both previously seen and novel questions across different LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16731v1">Collaborative Inference and Learning between Edge SLMs and Cloud LLMs: A Survey of Algorithms, Execution, and Open Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 35 pages, 9 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, deploying them solely in the cloud or compressing them for edge devices has become inadequate due to concerns about latency, privacy, cost, and personalization. This survey explores a collaborative paradigm in which cloud-based LLMs and edge-deployed small language models (SLMs) cooperate across both inference and training. We present a unified taxonomy of edge-cloud collaboration strategies. For inference, we categorize approaches into task assignment, task division, and mixture-based collaboration at both task and token granularity, encompassing adaptive scheduling, resource-aware offloading, speculative decoding, and modular routing. For training, we review distributed adaptation techniques, including parameter alignment, pruning, bidirectional distillation, and small-model-guided optimization. We further summarize datasets, benchmarks, and deployment cases, and highlight privacy-preserving methods and vertical applications. This survey provides the first systematic foundation for LLM-SLM collaboration, bridging system and algorithm co-design to enable efficient, scalable, and trustworthy edge-cloud intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16727v1">Deliberative Searcher: Improving LLM Reliability via Reinforcement Learning with constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Improving the reliability of large language models (LLMs) is critical for deploying them in real-world scenarios. In this paper, we propose \textbf{Deliberative Searcher}, the first framework to integrate certainty calibration with retrieval-based search for open-domain question answering. The agent performs multi-step reflection and verification over Wikipedia data and is trained with a reinforcement learning algorithm that optimizes for accuracy under a soft reliability constraint. Empirical results show that proposed method improves alignment between model confidence and correctness, leading to more trustworthy outputs. This paper will be continuously updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16716v1">Enhancing Remote Sensing Vision-Language Models Through MLLM and LLM-Based High-Quality Image-Text Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 SUBMIT TO IEEE TRANSACTIONS
    </div>
    <details class="paper-abstract">
      The application of Vision-language foundation models (VLFMs) to remote sensing (RS) imagery has garnered significant attention due to their superior capability in various downstream tasks. A key challenge lies in the scarcity of high-quality, large-scale, image-text paired training data. Recently, several works introduced extensive image-text datasets for RS and trained their VLFMs. However, due to the rudimentary methods used for generating captions, the quality of datasets is suboptimal, requiring larger volumes of training data, while only yielding modest performance improvements. In this paper, we propose a two-stage method named MpGI(Multi-Perspective Generation and Integration) for generating high-quality text captions for RS images. Firstly, we generate distinct and detailed descriptions from different perspectives using Rule-MLLM(Multimodal Large Language Model) Relay Generation and MLLMs generation methods. Next, we utilize Large Language Models (LLMs) to integrate these diverse descriptions into comprehensive captions, capturing details from multiple perspectives. Finally, we have created the HQRS-IT-210K dataset, including about 210,000 RS images and 1.3 million captions. We fine-tuned two VLFMs using our dataset: CLIP, a discriminative model, and CoCa, an image-to-text generative model. This process resulted in our proposed HQRS-CLIP and RS-CoCa models. Experimental results demonstrate that HQRS-CLIP surpassed the previous SOTA RS CLIP model in various downstream tasks while using only 4.2\% of the training data. RS-CoCa outperforms other advanced approaches across benchmark datasets and can generate captions for RS images that rival or even exceed manual annotations. Dataset, pre-trained models, and codes will be released at https://github.com/YiguoHe/HQRS-210K-and-HQRS-CLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16708v1">Biases in LLM-Generated Musical Taste Profiles for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      One particularly promising use case of Large Language Models (LLMs) for recommendation is the automatic generation of Natural Language (NL) user taste profiles from consumption data. These profiles offer interpretable and editable alternatives to opaque collaborative filtering representations, enabling greater transparency and user control. However, it remains unclear whether users consider these profiles to be an accurate representation of their taste, which is crucial for trust and usability. Moreover, because LLMs inherit societal and data-driven biases, profile quality may systematically vary across user and item characteristics. In this paper, we study this issue in the context of music streaming, where personalization is challenged by a large and culturally diverse catalog. We conduct a user study in which participants rate NL profiles generated from their own listening histories. We analyze whether identification with the profiles is biased by user attributes (e.g., mainstreamness, taste diversity) and item features (e.g., genre, country of origin). We also compare these patterns to those observed when using the profiles in a downstream recommendation task. Our findings highlight both the potential and limitations of scrutable, LLM-based profiling in personalized systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08773v2">Universal Model Routing for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Model routing is a simple technique for reducing the inference cost of large language models (LLMs), wherein one maintains a pool of candidate LLMs, and learns to route each prompt to the smallest feasible LLM. Existing works focus on learning a router for a fixed pool of LLMs. In this paper, we consider the problem of dynamic routing, where new, previously unobserved LLMs are available at test time. We propose UniRoute, a new approach to this problem that relies on representing each LLM as a feature vector, derived based on predictions on a set of representative prompts. Based on this, we detail two effective instantiations of UniRoute, relying on cluster-based routing and a learned cluster map respectively. We show that these are estimates of a theoretically optimal routing rule, and quantify their errors via an excess risk bound. Experiments on a range of public benchmarks show the effectiveness of UniRoute in routing amongst more than 30 unseen LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16679v1">PICACO: Pluralistic In-Context Value Alignment of LLMs via Total Correlation Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      In-Context Learning has shown great potential for aligning Large Language Models (LLMs) with human values, helping reduce harmful outputs and accommodate diverse preferences without costly post-training, known as In-Context Alignment (ICA). However, LLMs' comprehension of input prompts remains agnostic, limiting ICA's ability to address value tensions--human values are inherently pluralistic, often imposing conflicting demands, e.g., stimulation vs. tradition. Current ICA methods therefore face the Instruction Bottleneck challenge, where LLMs struggle to reconcile multiple intended values within a single prompt, leading to incomplete or biased alignment. To address this, we propose PICACO, a novel pluralistic ICA method. Without fine-tuning, PICACO optimizes a meta-instruction that navigates multiple values to better elicit LLMs' understanding of them and improve their alignment. This is achieved by maximizing the total correlation between specified values and LLM responses, theoretically reinforcing value correlation while reducing distractive noise, resulting in effective value instructions. Extensive experiments on five value sets show that PICACO works well with both black-box and open-source LLMs, outperforms several recent strong baselines, and achieves a better balance across up to 8 distinct values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16672v1">Meta-Learning for Cold-Start Personalization in Prompt-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Generative, explainable, and flexible recommender systems, derived using Large Language Models (LLM) are promising and poorly adapted to the cold-start user situation, where there is little to no history of interaction. The current solutions i.e. supervised fine-tuning and collaborative filtering are dense-user-item focused and would be expensive to maintain and update. This paper introduces a meta-learning framework, that can be used to perform parameter-efficient prompt-tuning, to effectively personalize LLM-based recommender systems quickly at cold-start. The model learns soft prompt embeddings with first-order (Reptile) and second-order (MAML) optimization by treating each of the users as the tasks. As augmentations to the input tokens, these learnable vectors are the differentiable control variables that represent user behavioral priors. The prompts are meta-optimized through episodic sampling, inner-loop adaptation, and outer-loop generalization. On MovieLens-1M, Amazon Reviews, and Recbole, we can see that our adaptive model outperforms strong baselines in NDCG@10, HR@10, and MRR, and it runs in real-time (i.e., below 300 ms) on consumer GPUs. Zero-history personalization is also supported by this scalable solution, and its 275 ms rate of adaptation allows successful real-time risk profiling of financial systems by shortening detection latency and improving payment network stability. Crucially, the 275 ms adaptation capability can enable real-time risk profiling for financial institutions, reducing systemic vulnerability detection latency significantly versus traditional compliance checks. By preventing contagion in payment networks (e.g., Fedwire), the framework strengthens national financial infrastructure resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16656v1">P-CoT: A Pedagogically-motivated Participatory Chain-of-Thought Prompting for Phonological Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      This study explores the potential of phonological reasoning within text-based large language models (LLMs). Utilizing the PhonologyBench benchmark, we assess tasks like rhyme word generation, g2p conversion, and syllable counting. Our evaluations across 12 LLMs reveal that while few-shot learning offers inconsistent gains, the introduction of a novel Pedagogically-motivated Participatory Chain-of-Thought (P-CoT) prompt, which is anchored in educational theories like scaffolding and discovery learning, consistently enhances performance. This method leverages structured guidance to activate latent phonological abilities, achieving up to 52% improvement and even surpassing human baselines in certain tasks. Future work could aim to optimize P-CoT prompts for specific models or explore their application across different linguistic domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16587v1">On the Effectiveness of LLM-as-a-judge for Code Generation and Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Accepted at TSE. IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Large Language Models have been recently exploited as judges for complex natural language processing tasks, such as Q&A. The basic idea is to delegate to an LLM the assessment of the "quality" of the output provided by an automated technique for tasks for which: (i) quantitative metrics would only tell part of the story, and; (ii) a large-scale human-based evaluation would be too expensive. LLMs-as-a-judge, if proven effective for a specific task, can also unlock new possibilities for automation, with several LLMs proposing a solution for a given instance of the task and others judging and deciding what is the best output to show the user. We study the effectiveness of LLMs-as-a-judge for two code-related tasks, namely code generation and code summarization. The rationale for choosing these tasks is two-fold. First, quantitative metrics are usually not enough for the assessment of code summarizers/generators. For example, it is well documented that metrics such as BLEU are quite weak proxies for the quality of the generated summaries. Second, even state-of-the-art techniques still struggle with handling complex instances of these tasks, making them good candidates for benefiting from more advanced solutions envisioning collaboration among LLMs. For code generation, we check whether eight LLMs are able to judge the correctness of 1,405 Java methods and 1,281 Python functions generated by the same LLMs or implemented by humans. For code summarization, we compare the judgment of five LLMs to those provided by nine humans for ~1.2k summaries, related to both Java and Python functions. Our findings show that GPT-4-turbo is the best LLM in terms of judging capabilities for both tasks, with "smaller" LLMs featuring tens of billions parameters not being able to cope with judging tasks. However, even the best-performing LLM frequently misjudges the correctness of the code and summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06821v3">Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 37 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v2">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-22
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v5">Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-22
      | 💬 Project page: https://github.com/VITA-MLLM/Sparrow
    </div>
    <details class="paper-abstract">
      Recent years have seen the success of Multimodal Large Language Models (MLLMs) in the domain of vision understanding. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has been primarily driven by automatic data pipelines, which focus on the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our primary study approach involves fine-tuning pre-trained image-LLMs with video data and examining learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to that of baselines trained with significantly more samples. Meanwhile, we find that incorporating these synthetic samples can enhance the performance of long video understanding without requiring training on long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
    </details>
</div>
